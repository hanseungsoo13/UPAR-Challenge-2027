"""Train A7 FCE + full 40-bit joint-query contrastive ablations.

This runner intentionally does not modify the existing A7 CLI or trainer.  It
loads an A7 checkpoint, keeps the original 52-state prompt/FCE path, and adds
one complete-query multi-positive InfoNCE term:

    L = L_FCE + lambda_joint * L_joint

The joint positive mask is based on exact equality of the original 40-bit UPAR
annotation row.  Consequently, duplicate attribute combinations in a batch are
all positives in both directions and are never treated as false negatives.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model, save_best, save_last  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import AttriVisionDataset, PromptBatch, PromptCollator  # noqa: E402
from attrivision.engine.evaluator_abpr import evaluate_native52_category_nll  # noqa: E402
from attrivision.engine.trainer_attrivision import (  # noqa: E402
    _grad_scaler, _optimizer_groups, _optimizer_step, build_scheduler,
)
from attrivision.losses.focal_clip_loss import FocalCLIPLoss  # noqa: E402
from attrivision.transforms import build_eval_transform, build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import autocast  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "a7_fce_joint.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "epoch", "lambda_joint", "total_loss", "fce_loss", "joint_loss",
        "fce_i2t", "fce_t2i", "joint_i2t", "joint_t2i",
        "rank1", "rank5", "rank10", "map", "mADM", "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _multi_positive_infonce(
    logits: torch.Tensor, positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Standard row-normalized multi-positive InfoNCE."""
    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have the same [B,B] shape")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every contrastive anchor needs at least one positive")
    log_probability = F.log_softmax(logits.float(), dim=1)
    weights = positive_mask.to(log_probability.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return -(weights * log_probability).sum(dim=1).mean()


def _exact_40bit_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """Mark every image pair with identical original UPAR attributes positive."""
    if labels.ndim != 2 or labels.shape[1] != 40:
        raise ValueError(f"Expected binary labels [B,40], got {tuple(labels.shape)}")
    binary = labels > 0.5
    # [B,1,40] == [1,B,40] produces the complete pairwise row comparison.
    return (binary[:, None, :] == binary[None, :, :]).all(dim=2)


def _joint_query_embeddings(
    text_features: torch.Tensor, semantic_labels: torch.Tensor,
) -> torch.Tensor:
    """Compose one query from the complete 40-bit row via Native52 states.

    ``semantic_labels`` is a deterministic 40-bit -> 52-state encoding.  The
    text encoder therefore creates one feature for every active state in the
    complete attribute combination, and the mean is normalized into one query
    embedding per image.
    """
    if text_features.ndim != 2 or semantic_labels.ndim != 2:
        raise ValueError("text_features and semantic_labels must be matrices")
    if semantic_labels.shape[1] != text_features.shape[0]:
        raise ValueError("semantic label width must equal the number of text states")
    weights = semantic_labels.to(text_features.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    if (counts <= 0).any():
        raise ValueError("Every 40-bit query must map to at least one Native52 state")
    return F.normalize(weights @ text_features / counts, dim=-1)


def _joint_loss(
    image_features: torch.Tensor,
    state_features: torch.Tensor,
    semantic_labels: torch.Tensor,
    binary_labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return symmetric joint InfoNCE plus its two directional components."""
    query_features = _joint_query_embeddings(state_features, semantic_labels)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale.float() * image_features.float() @ query_features.T
    positive_mask = _exact_40bit_positive_mask(binary_labels)
    image_to_query = _multi_positive_infonce(logits, positive_mask)
    query_to_image = _multi_positive_infonce(logits.T, positive_mask.T)
    return 0.5 * (image_to_query + query_to_image), image_to_query, query_to_image


def _train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    fce_criterion: FocalCLIPLoss,
    state_tokens: torch.Tensor,
    lambda_joint: float,
    scaler: Any,
    device: torch.device,
    amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "total_loss": 0.0, "fce_loss": 0.0, "joint_loss": 0.0,
        "fce_i2t": 0.0, "fce_t2i": 0.0,
        "joint_i2t": 0.0, "joint_t2i": 0.0,
        "samples": 0.0, "batches": 0.0,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in loader:
        if not isinstance(batch, PromptBatch):
            raise TypeError("A7 PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic_labels = batch.semantic_labels.to(device, non_blocking=True)
        sampled_tokens = batch.tokens.to(device, non_blocking=True)
        selected_semantics = batch.selected_semantics.to(device, non_blocking=True)
        text_owners = batch.text_owners.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device, amp):
            image_features, _, sampled_logits = model(images, sampled_tokens)
            fce_output = fce_criterion(
                sampled_logits, semantic_labels, selected_semantics, text_owners,
            )
            state_features = model.encode_text(state_tokens)
            joint, joint_i2t, joint_t2i = _joint_loss(
                image_features, state_features, semantic_labels,
                labels.float(), model.logit_scale,
            )
            total = fce_output.loss + lambda_joint * joint

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)

        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["fce_loss"] += float(fce_output.loss.detach()) * count
        totals["joint_loss"] += float(joint.detach()) * count
        totals["fce_i2t"] += float(fce_output.i2t.detach()) * count
        totals["fce_t2i"] += float(fce_output.t2i.detach()) * count
        totals["joint_i2t"] += float(joint_i2t.detach()) * count
        totals["joint_t2i"] += float(joint_t2i.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1

    denominator = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    result = {key: value / denominator for key, value in totals.items()}
    result["train_batches"] = batches
    result["average_batch_size"] = denominator / batches
    return result


def _evaluate(
    model: Any, attribute_names: list[str], config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return evaluate_native52_category_nll(
        model, attribute_names, Path(config["data_root"]).resolve(),
        build_eval_transform(config["image_size"], "resize_pad_crop"),
        device, config["eval_batch_size"], config["num_workers"], config["amp"],
        config.get("max_val_samples"), config["category_temperature"],
    )


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    overrides = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "output_root": args.output_root,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "seed": args.seed,
        "amp": args.amp,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.lambdas is not None:
        config["lambda_joint"] = args.lambdas
    if not config.get("lambda_joint"):
        raise ValueError("config.lambda_joint must contain at least one value")
    config["lambda_joint"] = [float(value) for value in config["lambda_joint"]]
    if any(value <= 0 for value in config["lambda_joint"]):
        raise ValueError("lambda_joint values must be positive")
    config["checkpoint"] = str(Path(config["checkpoint"]).resolve())
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output_root"] = str(Path(config["output_root"]).resolve())
    config["device"] = str(config.get("device", "auto"))
    config["amp"] = bool(config.get("amp", True))
    return config


def _build_loader(config: dict[str, Any], table: Any, dataset: AttriVisionDataset,
                  model: Any, stochastic: bool) -> DataLoader:
    collator = PromptCollator(
        table.attribute_names, model.tokenizer, text_sampling="single",
        multi_attributes=1, prompt_mode="category_complete",
        unique_prompts=False, stochastic=stochastic,
    )
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=stochastic,
        generator=torch.Generator().manual_seed(config["seed"]),
        drop_last=stochastic and len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0, collate_fn=collator,
    )


def _train_lambda(
    lambda_joint: float, config: dict[str, Any], train_table: Any,
    train_roots: list[Path], base_metrics: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    output_dir = Path(config["output_root"]) / f"lambda_{lambda_joint:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "lambda_joint": lambda_joint, "output_dir": str(output_dir)}
    _write_json(output_dir / "config.json", run_config)

    # Every lambda starts from the same A7 weights and the same random seed.
    set_seed(config["seed"], config.get("deterministic", False))
    model, payload = load_model(config["checkpoint"], device)
    dataset = AttriVisionDataset(
        train_table,
        train_roots,
        build_train_transform(config["image_size"], config["rotation"], "resize_pad_crop"),
        config.get("max_train_samples"),
    )
    loader = _build_loader(config, train_table, dataset, model, stochastic=True)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    semantic_labels = mapper.encode(torch.from_numpy(dataset.labels.copy()))
    state_tokens = model.tokenize(mapper.prompts).to(device)
    fce_criterion = FocalCLIPLoss(
        "focal_clip", "multi_positive", config["focal_alpha"], config["focal_gamma"],
    )
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, config["weight_decay"]), lr=config["learning_rate"],
    )
    scheduler = build_scheduler(
        optimizer, len(loader), config["epochs"], config["warmup_epochs"],
        config["learning_rate"], config["min_learning_rate"],
    )
    scaler = _grad_scaler(config["amp"] and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    best_score = float(base_metrics["mADM"])
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] = {
        **base_metrics, "epoch": 0, "lambda_joint": lambda_joint,
    }
    training_config = {
        "training_objective": "a7_fce_joint",
        "prompt_mode": "category_complete",
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
        "augmentation": "resize_pad_crop",
        "joint_positive": "exact_40bit_row",
        "joint_query_states": 52,
        "lambda_joint": lambda_joint,
        "validation_protocol": "native52_category_nll",
        "selection_metric": "mADM",
        "focal_alpha": config["focal_alpha"],
        "focal_gamma": config["focal_gamma"],
    }
    initial_metrics = {
        **base_metrics, "epoch": 0, "lambda_joint": lambda_joint,
        "checkpoint": str(config["checkpoint"]),
    }
    save_best(
        best_path, model, payload["attribute_names"], 0, initial_metrics,
        "category_complete", training_config,
    )

    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        losses = _train_one_epoch(
            model, loader, optimizer, scheduler, fce_criterion, state_tokens,
            lambda_joint, scaler, device, config["amp"], config["grad_clip"],
        )
        metrics = _evaluate(model, payload["attribute_names"], config, device)
        row = {
            "epoch": epoch, "lambda_joint": lambda_joint,
            **losses, **metrics, "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        _write_metrics_csv(output_dir / "metrics.csv", history)
        with (output_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=float) + "\n")

        current_score = float(metrics["mADM"])
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            stale = 0
            best_metrics = dict(row)
            save_best(
                best_path, model, payload["attribute_names"], epoch, row,
                "category_complete", training_config,
            )
        else:
            stale += 1
        save_last(
            last_path, model, payload["attribute_names"], epoch, row,
            optimizer, scheduler, scaler, best_score, best_epoch, stale,
            "category_complete", training_config,
        )
        print(
            f"lambda={lambda_joint:g} epoch={epoch:03d}/{config['epochs']}: "
            f"total={losses['total_loss']:.6f}, fce={losses['fce_loss']:.6f}, "
            f"joint={losses['joint_loss']:.6f}, "
            f"Rank-1={100 * metrics['rank1']:.2f}%, "
            f"mAP={100 * metrics['map']:.2f}%, "
            f"mADM={100 * metrics['mADM']:.2f}%",
            flush=True,
        )

    best_result = {
        "lambda_joint": lambda_joint,
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_mADM": best_score,
        "best_metrics": best_metrics,
        "baseline_mADM": base_metrics["mADM"],
        "delta_mADM": best_score - base_metrics["mADM"],
        "last_epoch": history[-1] if history else initial_metrics,
    }
    _write_json(output_dir / "validation_metrics.json", best_result)
    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_result


def _run_smoke() -> None:
    torch.manual_seed(7)
    image_features = F.normalize(torch.randn(4, 16), dim=-1).requires_grad_()
    query_features = F.normalize(torch.randn(4, 16), dim=-1).requires_grad_()
    labels = torch.zeros(4, 40)
    labels[0, 0] = labels[1, 0] = 1
    labels[2, 1] = labels[3, 1] = 1
    loss_i2q = _multi_positive_infonce(
        image_features @ query_features.T, _exact_40bit_positive_mask(labels),
    )
    loss_q2i = _multi_positive_infonce(
        (image_features @ query_features.T).T,
        _exact_40bit_positive_mask(labels).T,
    )
    loss = 0.5 * (loss_i2q + loss_q2i)
    loss.backward()
    assert torch.isfinite(loss)
    assert image_features.grad is not None and query_features.grad is not None
    print(
        "A7 FCE + joint-query smoke check passed: "
        f"joint={float(loss.detach()):.6f}, duplicate positives preserved"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone A7 FCE + exact 40-bit joint-query ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=("train", "smoke"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lambdas", type=float, nargs="+")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "smoke":
        _run_smoke()
        return
    config = _resolved_config(args)
    device = choose_device(config["device"])
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"A7 checkpoint not found: {checkpoint}. Pass --checkpoint explicitly."
        )
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "config.json", config)

    train_gt = find_annotation_file(Path(config["data_root"]), "train")
    train_table = read_gt_csv(train_gt)
    train_roots = [Path(config["data_root"]), train_gt.parent, REPOSITORY_ROOT]
    print(f"A7 base checkpoint: {checkpoint}", flush=True)
    print(f"Device: {device}; lambdas: {config['lambda_joint']}", flush=True)

    set_seed(config["seed"], config.get("deterministic", False))
    baseline_model, baseline_payload = load_model(checkpoint, device)
    if baseline_payload.get("prompt_mode") != "category_complete":
        raise ValueError(
            "This runner requires an original A7 category_complete checkpoint; "
            f"got prompt_mode={baseline_payload.get('prompt_mode')!r}"
        )
    if len(CategoryPromptMapper(baseline_payload["attribute_names"]).prompts) != 52:
        raise ValueError("This runner requires the original 52-state prompt vocabulary")
    baseline_metrics = _evaluate(
        baseline_model, baseline_payload["attribute_names"], config, device,
    )
    baseline_record = {
        **baseline_metrics, "lambda_joint": 0.0,
        "checkpoint": str(checkpoint), "label": "A7-FCE-only baseline",
    }
    _write_json(output_root / "baseline_metrics.json", baseline_record)
    print(
        f"Baseline A7: Rank-1={100 * baseline_metrics['rank1']:.2f}%, "
        f"mAP={100 * baseline_metrics['map']:.2f}%, "
        f"mADM={100 * baseline_metrics['mADM']:.2f}%",
        flush=True,
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = [baseline_record]
    for lambda_joint in config["lambda_joint"]:
        result = _train_lambda(
            lambda_joint, config, train_table, train_roots, baseline_metrics, device,
        )
        results.append(result)

    comparison_rows = []
    for result in results:
        comparison_rows.append({
            "lambda_joint": result["lambda_joint"],
            "label": result.get("label", "joint-query"),
            "checkpoint": result["checkpoint"],
            "best_epoch": result.get("best_epoch", 0),
            "rank1": result.get("rank1", result.get("best_metrics", {}).get("rank1")),
            "rank5": result.get("rank5", result.get("best_metrics", {}).get("rank5")),
            "rank10": result.get("rank10", result.get("best_metrics", {}).get("rank10")),
            "map": result.get("map", result.get("best_metrics", {}).get("map")),
            "mADM": result.get("mADM", result.get("best_mADM")),
            "delta_mADM": result.get("delta_mADM", 0.0),
        })
    _write_json(output_root / "comparison.json", comparison_rows)
    with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(comparison_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print("\nA7 FCE + joint-query comparison", flush=True)
    for row in comparison_rows:
        print(
            f"lambda={row['lambda_joint']:g}: "
            f"Rank-1={100 * float(row['rank1']):.2f}%, "
            f"mAP={100 * float(row['map']):.2f}%, "
            f"mADM={100 * float(row['mADM']):.2f}%, "
            f"delta_mADM={100 * float(row['delta_mADM']):+.2f} points",
            flush=True,
        )


if __name__ == "__main__":
    main()
