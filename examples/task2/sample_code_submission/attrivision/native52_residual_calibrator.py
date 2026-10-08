"""Head-only feasibility study for a structured Native52 residual calibrator.

The source A1 model is frozen bit-for-bit. Its image and text embeddings are
cached, converted to the 52 temperature-scaled Native52 logits, and only a
small category-local residual MLP is trained with equally weighted category CE.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.compare_inference import prediction_metrics  # noqa: E402
from attrivision.conditioned_head_feasibility import (  # noqa: E402
    feature_cache_path, load_or_encode_images, load_or_encode_texts,
    subset, tensor_state_sha256,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.engine.paired_attribute import learned_inverse_temperature  # noqa: E402
from attrivision.evaluate_native52_hard import (  # noqa: E402
    hard_prediction_metrics, retrieval_result, semantic_query_metrics,
)
from attrivision.native52_retrieval_ablation import category_indices, category_nll  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import load_retrieval_annotations, reorder_columns  # noqa: E402


class StructuredResidualCalibrator(nn.Module):
    """Apply an independent tiny residual MLP inside each semantic category."""

    def __init__(self, groups: Sequence[Sequence[int]], hidden_dim: int,
                 residual_scale: float, learnable_alpha: bool = False) -> None:
        super().__init__()
        if hidden_dim <= 0 or residual_scale <= 0:
            raise ValueError("hidden_dim and residual_scale must be positive")
        self.groups = [tuple(int(index) for index in group) for group in groups]
        covered = [index for group in self.groups for index in group]
        if len(self.groups) != 12 or sorted(covered) != list(range(52)):
            raise ValueError("Calibrator requires a 12-category partition of states 0..51")
        self.refiners = nn.ModuleList([
            nn.Sequential(nn.Linear(len(group), hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, len(group)))
            for group in self.groups
        ])
        # Exact identity at initialization preserves the A1 Native52 baseline.
        for refiner in self.refiners:
            nn.init.zeros_(refiner[-1].weight)
            nn.init.zeros_(refiner[-1].bias)
        alpha = torch.full((len(self.groups),), float(residual_scale))
        if learnable_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha, persistent=True)

    def forward(self, logits52: torch.Tensor) -> torch.Tensor:
        if logits52.ndim != 2 or logits52.shape[1] != 52:
            raise ValueError(f"Expected logits [B,52], got {tuple(logits52.shape)}")
        output = logits52.clone()
        for category, (indices, refiner) in enumerate(zip(self.groups, self.refiners)):
            block = logits52[:, indices]
            output[:, indices] = block + self.alpha[category] * refiner(block)
        return output


def category_ce(logits52: torch.Tensor, targets52: torch.Tensor,
                groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """Mean soft-target CE, weighting every category and sample equally."""
    losses = []
    for indices in groups:
        targets = targets52[:, indices].float()
        counts = targets.sum(dim=1, keepdim=True)
        if (counts <= 0).any():
            raise ValueError("Every sample must have an active state in every category")
        targets = targets / counts
        losses.append(-(targets * F.log_softmax(logits52[:, indices], dim=1)).sum(dim=1))
    return torch.stack(losses, dim=1).mean()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tiny category-local residual calibrator on frozen A1 Native52 logits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="outputs/attrivision_ablation/A1/checkpoint_best.pth")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs/native52_residual_calibrator")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--learnable-alpha", action="store_true")
    parser.add_argument("--attribute-temperature", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--train-features", help="optional cached finite [N,512] features")
    parser.add_argument("--val-features", help="optional cached finite [G,512] features")
    parser.add_argument("--native-text-features", help="optional cached finite [52,512] features")
    parser.add_argument("--max-train-samples", type=int, help="smoke testing only")
    parser.add_argument("--max-val-samples", type=int, help="smoke testing only")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "eval_batch_size", "encode_batch_size",
                 "image_size", "distance_chunk_size", "hidden_dim"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.residual_scale <= 0:
        raise ValueError("learning-rate/residual-scale must be positive and weight-decay non-negative")
    if args.attribute_temperature is not None and args.attribute_temperature <= 0:
        raise ValueError("--attribute-temperature must be positive")
    if args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("--num-workers and --progress-every cannot be negative")
    if args.image_size != 224:
        raise ValueError("AttriVision ViT-B/32 requires --image-size 224")


def positive_state_indices(mapper: CategoryPromptMapper) -> list[int]:
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    positive: dict[str, str] = {}
    for _, columns, state_keys, _ in mapper._MULTI_GROUPS:
        positive.update(zip(columns, state_keys))
    for column, positive_key, _ in mapper._BINARY_GROUPS:
        positive[column] = positive_key
    return [state_index[positive[name]] for name in mapper.attribute_names]


@torch.inference_mode()
def infer_logits(calibrator: nn.Module, logits: torch.Tensor, targets: torch.Tensor,
                 groups: Sequence[Sequence[int]], device: torch.device,
                 batch_size: int) -> tuple[np.ndarray, float]:
    calibrator.eval()
    chunks: list[torch.Tensor] = []
    total_loss = 0.0
    samples = 0
    loader = DataLoader(TensorDataset(logits, targets), batch_size=batch_size, shuffle=False)
    for batch_logits, batch_targets in loader:
        refined = calibrator(batch_logits.to(device, non_blocking=True))
        target = batch_targets.to(device, non_blocking=True)
        loss = category_ce(refined, target, groups)
        total_loss += float(loss) * len(batch_logits)
        samples += len(batch_logits)
        chunks.append(refined.float().cpu())
    return torch.cat(chunks).numpy(), total_loss / max(samples, 1)


def probabilities52(logits: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    values = torch.from_numpy(logits.astype(np.float32, copy=False))
    result = torch.empty_like(values)
    for indices in groups:
        result[:, indices] = values[:, indices].softmax(dim=1)
    return result.numpy()


def evaluate(logits52: np.ndarray, loss: float, labels40: np.ndarray,
             queries52: np.ndarray, ids: np.ndarray, groups: Sequence[Sequence[int]],
             output_indices: Sequence[int], chunk_size: int) -> dict[str, float]:
    probs52 = probabilities52(logits52, groups)
    probs40 = probs52[:, output_indices]
    prediction, _ = prediction_metrics(labels40, probs40)
    hard52 = np.zeros_like(probs52, dtype=bool)
    for indices in groups:
        winners = np.argmax(logits52[:, indices], axis=1)
        hard52[np.arange(len(logits52)), np.asarray(indices)[winners]] = True
    hard = hard_prediction_metrics(labels40, hard52[:, output_indices])
    distances = category_nll(queries52, probs52, groups)
    retrieval = retrieval_result(distances, ids)
    semantic = semantic_query_metrics(distances, ids, chunk_size)
    return {
        "category_ce": float(loss),
        "macro_auroc": prediction["macro_auroc"],
        "macro_ap": prediction["macro_ap"],
        "instance_f1": hard["instance_f1"],
        "mean_hamming_error": hard["mean_hamming_error"],
        "exact_match": hard["exact_match"],
        **retrieval, **semantic,
    }


def save_checkpoint(path: Path, calibrator: nn.Module, epoch: int,
                    metrics: dict[str, float], config: dict[str, Any],
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "native52_category_residual_calibrator",
        "calibrator_state_dict": {
            key: value.detach().cpu() for key, value in calibrator.state_dict().items()
        },
        "epoch": epoch, "metrics": metrics, "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_model(checkpoint, device)
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Source checkpoint must use category_complete Native52 prompts")
    attribute_names = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attribute_names)
    groups = category_indices(mapper)
    output_indices = positive_state_indices(mapper)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    frozen_hash_before = tensor_state_sha256(model.state_dict())

    train_gt = find_annotation_file(data_root, "train")
    val_gt = find_annotation_file(data_root, "val")
    train_table = subset(read_gt_csv(train_gt), args.max_train_samples)
    val_table = subset(read_gt_csv(val_gt), args.max_val_samples)
    train_labels = reorder_columns(train_table.labels, train_table.attribute_names, attribute_names)
    val_labels = reorder_columns(val_table.labels, val_table.attribute_names, attribute_names)
    if args.max_val_samples is None:
        queries40_raw, ids, query_names = load_retrieval_annotations(val_gt.parent, val_table)
        queries40 = reorder_columns(queries40_raw, query_names, attribute_names)
    else:
        queries40, ids = np.unique(val_labels, axis=0, return_inverse=True)
    if not np.array_equal(val_labels, queries40[ids]):
        raise RuntimeError("Validation labels do not match semantic-query IDs")

    train_features = load_or_encode_images(
        args.train_features, feature_cache_path(cache_dir, "train", args.max_train_samples),
        model, train_table, [data_root, train_gt.parent, REPOSITORY_ROOT], device, args,
        "[train image]",
    )
    default_val = checkpoint.parent / "native52_retrieval_ablation" / "gallery_features.npy"
    if args.max_val_samples is not None or not default_val.is_file():
        default_val = feature_cache_path(cache_dir, "val", args.max_val_samples)
    val_features = load_or_encode_images(
        args.val_features, default_val, model, val_table,
        [data_root, val_gt.parent, REPOSITORY_ROOT], device, args, "[val image]",
    )
    default_text = checkpoint.parent / "native52_retrieval_ablation" / "native52_text_features.npy"
    if not default_text.is_file():
        default_text = cache_dir / "native52_text_features.npy"
    text_features = load_or_encode_texts(
        args.native_text_features, default_text, model, mapper.prompts,
        device, args.amp, "[Native52 text]",
    )
    inverse_temperature = (1.0 / args.attribute_temperature
                           if args.attribute_temperature is not None
                           else learned_inverse_temperature(model))
    train_logits = (train_features.float() @ text_features.float().T) * inverse_temperature
    val_logits = (val_features.float() @ text_features.float().T) * inverse_temperature
    train_targets = mapper.encode(torch.from_numpy(train_labels)).float()
    val_targets = mapper.encode(torch.from_numpy(val_labels)).float()
    queries52 = mapper.encode(torch.from_numpy(queries40)).numpy().astype(np.float32)

    calibrator = StructuredResidualCalibrator(
        groups, args.hidden_dim, args.residual_scale, args.learnable_alpha,
    ).to(device)
    trainable = sum(parameter.numel() for parameter in calibrator.parameters()
                    if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in calibrator.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in calibrator.parameters()
                         if parameter.requires_grad}:
        raise RuntimeError("Optimizer must contain exactly the calibrator parameters")

    config = {
        "source_checkpoint": str(checkpoint), "encoder_frozen": True,
        "frozen_model_sha256": frozen_hash_before, "native_states": 52,
        "categories": 12, "category_sizes": [len(group) for group in groups],
        "hidden_dim": args.hidden_dim, "residual_scale": args.residual_scale,
        "learnable_alpha": args.learnable_alpha, "zero_initialized_residual": True,
        "trainable_parameters": trainable, "optimizer": "AdamW",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "temperature": 1.0 / inverse_temperature, "loss": "mean category soft-target CE",
        "selection_metric": "validation Category-NLL mAP",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    baseline_logits, baseline_loss = infer_logits(
        calibrator, val_logits, val_targets, groups, device, args.eval_batch_size,
    )
    if not np.array_equal(baseline_logits, val_logits.numpy()):
        raise RuntimeError("Zero-initialized calibrator is not an exact identity")
    baseline = evaluate(
        baseline_logits, baseline_loss, val_labels, queries52, ids, groups,
        output_indices, args.distance_chunk_size,
    )
    print(f"Frozen model params optimized = 0")
    print(f"Calibrator trainable params = {trainable:,}")
    print(f"Identity baseline: CE={baseline_loss:.6f}, R1={100*baseline['rank1']:.2f}%, "
          f"mAP={100*baseline['map']:.2f}%", flush=True)

    loader = DataLoader(
        TensorDataset(train_logits, train_targets), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda",
    )
    best_map = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        calibrator.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_logits, batch_targets in loader:
            batch_logits = batch_logits.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = category_ce(calibrator(batch_logits), batch_targets, groups)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_logits)
            sample_count += len(batch_logits)
        refined_logits, val_loss = infer_logits(
            calibrator, val_logits, val_targets, groups, device, args.eval_batch_size,
        )
        metrics = evaluate(
            refined_logits, val_loss, val_labels, queries52, ids, groups,
            output_indices, args.distance_chunk_size,
        )
        row = {"epoch": epoch, "train_category_ce": loss_sum / sample_count,
               **metrics, "elapsed_seconds": time.time() - started}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: trainCE={row['train_category_ce']:.6f}, "
              f"valCE={val_loss:.6f}, R1={100*metrics['rank1']:.2f}%, "
              f"mAP={100*metrics['map']:.2f}%, AUROC={metrics['macro_auroc']:.4f}", flush=True)
        if metrics["map"] > best_map:
            best_map, best_epoch = metrics["map"], epoch
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(calibrator.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training produced no checkpoint")
    last_metrics = {key: value for key, value in history[-1].items()
                    if key not in {"epoch", "elapsed_seconds", "train_category_ce"}}
    save_checkpoint(output_dir / "checkpoint_last.pth", calibrator, args.epochs,
                    last_metrics, config, optimizer)
    calibrator.load_state_dict(best_state)
    save_checkpoint(output_dir / "checkpoint_best.pth", calibrator, best_epoch,
                    best_metrics, config)
    with (output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    frozen_hash_after = tensor_state_sha256(model.state_dict())
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("Frozen A1 model changed during calibrator training")
    summary = {
        "baseline": baseline, "best_epoch": best_epoch, "best": best_metrics,
        "last": last_metrics,
        "delta_best_minus_baseline": {
            key: best_metrics[key] - baseline[key]
            for key in ("category_ce", "macro_auroc", "macro_ap", "instance_f1",
                        "mean_hamming_error", "exact_match", "rank1", "rank5", "rank10", "map",
                        "semantic_query_top1", "semantic_query_mean_rank",
                        "semantic_query_median_rank")
        },
        "frozen_model_sha256_before": frozen_hash_before,
        "frozen_model_sha256_after": frozen_hash_after,
        "frozen_model_exact_match": True,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Best epoch {best_epoch}: R1={100*best_metrics['rank1']:.2f}%, "
          f"mAP={100*best_metrics['map']:.2f}% "
          f"(delta={100*(best_metrics['map']-baseline['map']):+.2f} pp)")
    print(f"Frozen A1 integrity: exact SHA256 match\nArtifacts: {output_dir}")


if __name__ == "__main__":
    main()
