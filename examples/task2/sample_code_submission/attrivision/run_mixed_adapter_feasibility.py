"""Feasibility test for a frozen CLIP plus a trainable image adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.datasets.attribute_prompts import MixedCategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import AttriVisionDataset  # noqa: E402
from attrivision.engine.evaluator_abpr import evaluate_native52_category_nll  # noqa: E402
from attrivision.engine.evaluator_mixed import evaluate_mixed_state_nll  # noqa: E402
from attrivision.losses.mixed_state_hybrid_loss import MixedStateHybridLoss  # noqa: E402
from attrivision.models.attrivision import AttriVision  # noqa: E402
from attrivision.models.clip_image_adapter import FrozenCLIPImageAdapter  # noqa: E402
from attrivision.transforms import build_eval_transform, build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import autocast  # noqa: E402


def _scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feasibility test: frozen CLIP plus trainable mixed-state image adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("smoke", "train_eval"), default="smoke")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--checkpoint", help="existing AttriVision checkpoint to freeze")
    parser.add_argument("--clip-model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained-tag", default="openai")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/attrivision_adapter_feasibility")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--augmentation", choices=("current", "resize_pad_crop"), default="resize_pad_crop")
    parser.add_argument("--rotation", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=16)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.image_size != 224:
        raise ValueError("Vanilla CLIP ViT-B/32 requires --image-size 224")
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.num_workers < 0 or args.smoke_batch_size <= 0:
        raise ValueError("num-workers cannot be negative and smoke-batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.bottleneck_dim <= 0 or args.max_train_samples <= 0 or args.max_val_samples <= 0:
        raise ValueError("bottleneck and feasibility sample limits must be positive")


def _build_model(args: argparse.Namespace, device: torch.device) -> tuple[FrozenCLIPImageAdapter, list[str]]:
    if args.checkpoint:
        base_model, payload = load_model(args.checkpoint, device)
        attribute_names = list(payload["attribute_names"])
    else:
        base_model = AttriVision(
            pretrained=None if args.no_pretrained else args.pretrained_tag,
            model_name=args.clip_model,
        ).to(device)
        train_gt = find_annotation_file(Path(args.data_root).resolve(), "train")
        attribute_names = list(read_gt_csv(train_gt).attribute_names)
    model = FrozenCLIPImageAdapter(base_model, args.bottleneck_dim).to(device)
    return model, attribute_names


def _build_criterion(
    mapper: MixedCategoryPromptMapper,
    semantic_labels: torch.Tensor,
    device: torch.device,
) -> MixedStateHybridLoss:
    return MixedStateHybridLoss(
        semantic_labels.float().mean(dim=0),
        [(kind, indices) for _, kind, indices in mapper.category_specs()],
        focal_alpha=1.0,
        focal_gamma=2.0,
        balance_max_weight=10.0,
        prototype_weight=1.0,
        set_weight=0.0,
        consistency_weight=0.1,
    ).to(device)


def _prototype_step(
    model: FrozenCLIPImageAdapter,
    criterion: MixedStateHybridLoss,
    images: torch.Tensor,
    labels: torch.Tensor,
    mapper: MixedCategoryPromptMapper,
    prototype_features: torch.Tensor,
    amp: bool,
) -> Any:
    semantic_labels = mapper.encode(labels)
    with autocast(images.device, amp):
        image_features = model.encode_image(images)
        output = criterion(
            image_features, prototype_features, semantic_labels,
            model.logit_scale, labels.float(), include_set=False,
        )
    return output


def _run_smoke_direct(args: argparse.Namespace) -> None:
    """Run the adapter smoke test without requiring a real 40-label mapper."""
    device = choose_device(args.device)
    model, _ = _build_model(args, device)
    model.image_adapter.train()
    images = torch.randn(
        args.smoke_batch_size, 3, args.image_size, args.image_size, device=device,
    )
    state_count = 5
    semantic_labels = torch.zeros(
        args.smoke_batch_size, state_count, dtype=torch.bool, device=device,
    )
    semantic_labels[:, 0] = True
    semantic_labels[:, 2] = True
    semantic_labels[0, 1] = True
    semantic_labels[1:, 3] = True
    tokens = model.tokenize([f"a photo of state {index}" for index in range(state_count)]).to(device)
    criterion = MixedStateHybridLoss(
        semantic_labels.float().mean(dim=0).clamp_min(1 / args.smoke_batch_size),
        [("single", [0, 1]), ("multi", [2, 3, 4])],
        prototype_weight=1.0, set_weight=0.0, consistency_weight=0.1,
        min_shared_categories=1,
    ).to(device)
    images_labels = torch.randint(0, 2, (args.smoke_batch_size, 40), device=device).float()
    with torch.no_grad():
        prototype_features = model.encode_text(tokens)
    with autocast(device, args.amp):
        image_features = model.encode_image(images)
        output = criterion(
            image_features, prototype_features, semantic_labels,
            model.logit_scale, images_labels, include_set=False,
        )
    output.loss.backward()
    adapter_grads = [parameter.grad for parameter in model.image_adapter.parameters()]
    base_grads = [parameter.grad for parameter in model.base.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in adapter_grads)
    assert all(gradient is None for gradient in base_grads)
    assert torch.isfinite(output.loss)
    assert float(output.set_contrastive) == 0.0
    print(
        "Frozen CLIP adapter feasibility smoke passed: "
        f"base_trainable={sum(p.requires_grad for p in model.base.parameters())}, "
        f"adapter_params={sum(p.numel() for p in model.image_adapter.parameters()):,}, "
        f"loss={float(output.loss.detach()):.6f}, set=0, adapter_backward=finite"
    )


def _mean_prototype_loss(
    model: FrozenCLIPImageAdapter,
    loader: DataLoader,
    mapper: MixedCategoryPromptMapper,
    criterion: MixedStateHybridLoss,
    prototype_features: torch.Tensor,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> dict[str, float]:
    training = optimizer is not None and scaler is not None
    model.image_adapter.train(training)
    model.base.eval()
    totals = {"loss": 0.0, "single_ce": 0.0, "multi_bce": 0.0, "consistency": 0.0, "samples": 0}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad() if training else torch.no_grad():
            output = _prototype_step(
                model, criterion, images, labels, mapper,
                prototype_features, amp,
            )
            if training:
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        count = len(images)
        totals["loss"] += float(output.loss.detach()) * count
        totals["single_ce"] += float(output.single_ce.detach()) * count
        totals["multi_bce"] += float(output.multi_bce.detach()) * count
        totals["consistency"] += float(output.consistency.detach()) * count
        totals["samples"] += count
    denominator = max(totals.pop("samples"), 1)
    return {key: value / denominator for key, value in totals.items()}


def _save_adapter(path: Path, model: FrozenCLIPImageAdapter, args: argparse.Namespace,
                  attribute_names: list[str], epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": "frozen_clip_image_adapter",
        "model_name": model.model_name,
        "base_checkpoint": args.checkpoint,
        "attribute_names": attribute_names,
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in model.image_adapter.state_dict().items()
        },
        "adapter_config": {"bottleneck_dim": args.bottleneck_dim},
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def _run_train_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=True)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    train_gt = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_gt)
    model, attribute_names = _build_model(args, device)
    if attribute_names != list(train_table.attribute_names):
        raise ValueError("checkpoint attribute order differs from train annotations")
    mapper = MixedCategoryPromptMapper(attribute_names)
    train_dataset = AttriVisionDataset(
        train_table, [data_root, train_gt.parent, REPOSITORY_ROOT],
        build_train_transform(args.image_size, args.rotation, args.augmentation),
        args.max_train_samples,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    semantic_labels = mapper.encode(torch.from_numpy(train_dataset.labels.copy()))
    criterion = _build_criterion(mapper, semantic_labels, device)
    prototype_tokens = model.tokenize(mapper.prompts).to(device)
    with torch.no_grad():
        prototype_features = model.encode_text(prototype_tokens)
    optimizer = torch.optim.AdamW(
        model.image_adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = _scaler(args.amp and device.type == "cuda")

    val_gt = find_annotation_file(data_root, "val")
    val_table = read_gt_csv(val_gt)
    val_dataset = AttriVisionDataset(
        val_table, [data_root, val_gt.parent, REPOSITORY_ROOT],
        build_eval_transform(args.image_size, args.augmentation), args.max_val_samples,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    best_loss = float("inf")
    best_metrics: dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = _mean_prototype_loss(
            model, train_loader, mapper, criterion, prototype_features,
            device, args.amp, optimizer, scaler,
        )
        val_metrics = _mean_prototype_loss(
            model, val_loader, mapper, criterion, prototype_features,
            device, args.amp,
        )
        metrics = {"train": train_metrics, "val": val_metrics}
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train={train_metrics['loss']:.6f}, val={val_metrics['loss']:.6f}, "
            f"single_ce={val_metrics['single_ce']:.6f}, "
            f"multi_bce={val_metrics['multi_bce']:.6f}, "
            f"adapter_params={sum(p.numel() for p in model.image_adapter.parameters()):,}",
            flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = metrics
            _save_adapter(output_dir / "checkpoint_best.pth", model, args, attribute_names, epoch, metrics)
        _save_adapter(output_dir / "checkpoint_last.pth", model, args, attribute_names, epoch, metrics)

    native_metrics = evaluate_native52_category_nll(
        model, attribute_names, data_root,
        build_eval_transform(args.image_size, args.augmentation), device,
        args.eval_batch_size, args.num_workers, args.amp, args.max_val_samples,
        args.category_temperature,
    )
    mixed_metrics = evaluate_mixed_state_nll(
        model, attribute_names, data_root,
        build_eval_transform(args.image_size, args.augmentation), device,
        args.eval_batch_size, args.num_workers, args.amp, args.max_val_samples,
        args.category_temperature,
    )
    report = {
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "native52": native_metrics,
        "mixed_state": mixed_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feasibility_metrics.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8",
    )
    print(
        "Native52 feasibility evaluation: "
        f"Rank-1={100 * native_metrics['rank1']:.2f}%, "
        f"mAP={100 * native_metrics['map']:.2f}%, "
        f"mADM={100 * native_metrics['mADM']:.2f}%",
        flush=True,
    )
    print(
        "Mixed-state feasibility evaluation: "
        f"Rank-1={100 * mixed_metrics['rank1']:.2f}%, "
        f"mAP={100 * mixed_metrics['map']:.2f}%, "
        f"mADM={100 * mixed_metrics['mADM']:.2f}%",
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.mode == "smoke":
        _run_smoke_direct(args)
    else:
        _run_train_eval(args)


if __name__ == "__main__":
    main()
