"""Linear-probe a frozen AttriVision checkpoint with plain 40-D BCE.

The CLIP image/text encoders and ``logit_scale`` are immutable.  Images are
encoded once, and only ``Linear(512, 40)`` is optimized.  Final validation
reports both the original native 52-state path and the learned binary head.
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
from torch.utils.data import DataLoader, TensorDataset

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.compare_inference import (  # noqa: E402
    encode_texts, native_state_outputs, prediction_metrics, ranks_at_k,
    semantic_query_metrics,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.engine.evaluator_abpr import encode_gallery  # noqa: E402
from attrivision.engine.paired_attribute import learned_inverse_temperature  # noqa: E402
from attrivision.evaluate_native52_hard import (  # noqa: E402
    hard_category_projection, hard_prediction_metrics,
)
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    l1_attribute_distances, load_retrieval_annotations, reorder_columns,
    retrieval_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train only Linear(512,40) on a frozen AttriVision checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="outputs/attrivision_category/checkpoint_best.pth")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs/attrivision_frozen_bce40")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024, help="cached-feature training batch")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--head-eval-batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--train-features", help="reuse finite [N,512] train features")
    parser.add_argument("--val-features", help="reuse finite [N,512] validation features")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "encode_batch_size", "head_eval_batch_size",
                 "image_size", "distance_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("--num-workers and --progress-every cannot be negative")
    if args.image_size != 224:
        raise ValueError("AttriVision ViT-B/32 requires --image-size 224")


def subset(table: AnnotationTable, maximum: int | None) -> AnnotationTable:
    if maximum is None:
        return table
    count = min(maximum, len(table.image_paths))
    return AnnotationTable(table.image_paths[:count], table.labels[:count], table.attribute_names)


def load_or_encode_features(
    path: str | None, save_path: Path, model: nn.Module, table: AnnotationTable,
    roots: Sequence[Path], device: torch.device, args: argparse.Namespace, prefix: str,
) -> torch.Tensor:
    source = Path(path).resolve() if path else None
    if source is not None:
        values = np.load(source, allow_pickle=False)
        features = torch.from_numpy(values.astype(np.float32, copy=False))
        print(f"{prefix} reusing features: {source}", flush=True)
    else:
        features = encode_gallery(
            model, table.image_paths, roots, build_eval_transform(args.image_size),
            device, args.encode_batch_size, args.num_workers, args.amp,
            args.progress_every, prefix,
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features.numpy())
        print(f"{prefix} saved features: {save_path}", flush=True)
    if features.shape != (len(table.image_paths), 512) or not torch.isfinite(features).all():
        raise ValueError(f"{prefix} features must be finite [{len(table.image_paths)},512], got {tuple(features.shape)}")
    return features.contiguous()


@torch.inference_mode()
def head_probabilities(head: nn.Linear, features: torch.Tensor, device: torch.device,
                       batch_size: int) -> np.ndarray:
    head.eval()
    chunks = []
    for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False):
        chunks.append(torch.sigmoid(head(batch.to(device))).float().cpu())
    return torch.cat(chunks).numpy() if chunks else np.empty((0, 40), dtype=np.float32)


def retrieval_and_prediction_metrics(
    probabilities: np.ndarray, labels: np.ndarray, queries: np.ndarray,
    ids: np.ndarray, chunk_size: int,
) -> dict[str, float]:
    distances = l1_attribute_distances(queries, probabilities, chunk_size)
    rank1, mean_ap = retrieval_metrics(distances, ids)
    rank_k = ranks_at_k(distances, ids, (5, 10))
    semantic = semantic_query_metrics(distances, ids, chunk_size)
    prediction, _ = prediction_metrics(labels, probabilities)
    hard = hard_prediction_metrics(labels, probabilities >= 0.5)
    return {
        "rank1": rank1, "rank5": rank_k[5], "rank10": rank_k[10], "map": mean_ap,
        "macro_auroc": prediction["macro_auroc"], "macro_ap": prediction["macro_ap"],
        "instance_f1": hard["instance_f1"],
        "mean_hamming_error": hard["mean_hamming_error"],
        "exact_match": hard["exact_match"],
        "semantic_query_top1": semantic["top1"],
        "semantic_query_mean_rank": semantic["mean_rank"],
    }


def native52_metrics(
    model: nn.Module, features: torch.Tensor, labels: np.ndarray, queries: np.ndarray,
    ids: np.ndarray, attribute_names: Sequence[str], device: torch.device,
    amp: bool, chunk_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    mapper = CategoryPromptMapper(attribute_names)
    if len(mapper.keys) != 52:
        raise RuntimeError(f"Expected exactly 52 native states, got {len(mapper.keys)}")
    state_features = encode_texts(model, mapper.prompts, device, amp)
    probabilities, _ = native_state_outputs(
        features, state_features, mapper, learned_inverse_temperature(model),
    )
    result = retrieval_and_prediction_metrics(
        probabilities, labels, queries, ids, chunk_size,
    )
    # Native52 retrieval uses the requested soft probabilities, while its PAR
    # Hamming/F1 metrics use the original within-category argmax projection.
    hard = hard_prediction_metrics(
        labels, hard_category_projection(features, state_features, mapper),
    )
    result.update({
        "instance_f1": hard["instance_f1"],
        "mean_hamming_error": hard["mean_hamming_error"],
        "exact_match": hard["exact_match"],
    })
    return result, probabilities


def print_full_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"  AUROC / AP          : {metrics['macro_auroc']:.4f} / {metrics['macro_ap']:.4f}")
    print(f"  Instance F1         : {metrics['instance_f1']:.6f}")
    print(f"  Mean Hamming error  : {metrics['mean_hamming_error']:.4f} / 40")
    print(f"  Exact 40-bit match  : {100*metrics['exact_match']:.2f}%")
    print(f"  R1 / R5 / R10       : {100*metrics['rank1']:.2f}% / {100*metrics['rank5']:.2f}% / {100*metrics['rank10']:.2f}%")
    print(f"  mAP                 : {100*metrics['map']:.2f}%")
    print(f"  Semantic-query Top-1: {100*metrics['semantic_query_top1']:.2f}%")


def save_checkpoint(path: Path, source_payload: dict[str, Any], model: nn.Module,
                    epoch: int, metrics: dict[str, float], args: argparse.Namespace) -> None:
    payload = {key: value for key, value in source_payload.items() if key != "training_state"}
    payload["model_state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    payload.update({
        "format_version": max(int(payload.get("format_version", 1)), 3),
        "epoch": epoch,
        "metrics": dict(metrics),
        "binary_head_training": {
            "objective": "BCEWithLogitsLoss", "learning_rate": args.learning_rate,
            "epochs": args.epochs, "encoder_frozen": True, "fce": False,
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
        },
    })
    path.parent.mkdir(parents=True, exist_ok=True)
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
    output_dir.mkdir(parents=True, exist_ok=True)

    model, source_payload = load_model(checkpoint, device)
    if source_payload.get("prompt_mode") != "category_complete":
        raise ValueError("The source checkpoint must be category_complete for Native52 validation")
    attribute_names = list(source_payload["attribute_names"])
    if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
        raise ValueError("Checkpoint must contain exactly 40 unique attributes")

    # The source may contain an unused/random head.  A linear probe always starts fresh.
    model.binary_head.reset_parameters()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.binary_head.parameters():
        parameter.requires_grad_(True)
    model.eval()

    frozen_clip = sum(parameter.numel() for parameter in model.clip.parameters() if not parameter.requires_grad)
    trainable_head = sum(parameter.numel() for parameter in model.binary_head.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(model.binary_head.parameters(), lr=args.learning_rate)
    optimizer_count = sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"])
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    head_ids = {id(parameter) for parameter in model.binary_head.parameters()}
    if trainable_head != 20_520 or optimizer_count != 20_520 or optimizer_ids != head_ids:
        raise RuntimeError("The trainable/optimizer parameter set is not exactly Linear(512,40)")
    if any(parameter.requires_grad for parameter in model.clip.parameters()):
        raise RuntimeError("CLIP image/text/logit_scale parameters must all be frozen")
    print("frozen CLIP params")
    print(f"trainable binary head params = {trainable_head:,}")
    print(f"optimizer param count = {optimizer_count:,}")
    print(f"frozen CLIP param count = {frozen_clip:,}")

    # Preserve every CLIP tensor for an exact post-training immutability assertion.
    frozen_reference = {key: value.detach().cpu().clone() for key, value in model.clip.state_dict().items()}

    train_gt = find_annotation_file(data_root, "train")
    val_gt = find_annotation_file(data_root, "val")
    train_table = subset(read_gt_csv(train_gt), args.max_train_samples)
    val_table = subset(read_gt_csv(val_gt), args.max_val_samples)
    if train_table.attribute_names != attribute_names:
        train_labels = reorder_columns(train_table.labels, train_table.attribute_names, attribute_names)
    else:
        train_labels = train_table.labels
    val_labels = reorder_columns(val_table.labels, val_table.attribute_names, attribute_names)

    if args.max_val_samples is None:
        queries_raw, ids, query_names = load_retrieval_annotations(val_gt.parent, val_table)
        queries = reorder_columns(queries_raw, query_names, attribute_names)
    else:
        queries, ids = np.unique(val_labels, axis=0, return_inverse=True)
    if not np.array_equal(val_labels, queries[ids]):
        raise RuntimeError("Validation labels do not match official semantic-query IDs")

    roots_train = [data_root, train_gt.parent, REPOSITORY_ROOT]
    roots_val = [data_root, val_gt.parent, REPOSITORY_ROOT]
    train_features = load_or_encode_features(
        args.train_features, output_dir / "train_features.npy", model, train_table,
        roots_train, device, args, "[train encoder]",
    )
    val_features = load_or_encode_features(
        args.val_features, output_dir / "val_features.npy", model, val_table,
        roots_val, device, args, "[val encoder]",
    )

    native_metrics, native_predictions = native52_metrics(
        model, val_features, val_labels, queries, ids, attribute_names, device,
        args.amp, args.distance_chunk_size,
    )
    np.save(output_dir / "native52_probabilities.npy", native_predictions)

    labels_tensor = torch.from_numpy(train_labels.astype(np.float32, copy=False))
    loader = DataLoader(
        TensorDataset(train_features, labels_tensor), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    criterion = nn.BCEWithLogitsLoss()
    best_map = -float("inf")
    best_epoch = 0
    best_head: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.eval()
        model.binary_head.train()
        total_loss = 0.0
        sample_count = 0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.binary_head(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(feature_batch)
            sample_count += len(feature_batch)
        probabilities = head_probabilities(
            model.binary_head, val_features, device, args.head_eval_batch_size,
        )
        metrics = retrieval_and_prediction_metrics(
            probabilities, val_labels, queries, ids, args.distance_chunk_size,
        )
        loss_value = total_loss / max(sample_count, 1)
        history.append({"epoch": epoch, "loss": loss_value, **metrics})
        print(
            f"Epoch {epoch:02d}/{args.epochs}: BCE={loss_value:.6f}, "
            f"AUROC={metrics['macro_auroc']:.4f}, InstF1={metrics['instance_f1']:.4f}, "
            f"BitErr={metrics['mean_hamming_error']:.3f}, R1={100*metrics['rank1']:.2f}%, "
            f"mAP={100*metrics['map']:.2f}% ({time.time()-started:.1f}s)", flush=True,
        )
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            best_epoch = epoch
            best_head = copy.deepcopy(model.binary_head.state_dict())

    if best_head is None:
        raise RuntimeError("Training produced no binary-head checkpoint")
    model.binary_head.load_state_dict(best_head)
    binary_probabilities = head_probabilities(
        model.binary_head, val_features, device, args.head_eval_batch_size,
    )
    binary_metrics = retrieval_and_prediction_metrics(
        binary_probabilities, val_labels, queries, ids, args.distance_chunk_size,
    )

    for key, reference in frozen_reference.items():
        current = model.clip.state_dict()[key].detach().cpu()
        if not torch.equal(current, reference):
            raise RuntimeError(f"Frozen CLIP tensor changed during training: {key}")
    print("frozen CLIP integrity check = exact match")

    save_checkpoint(output_dir / "checkpoint_best.pth", source_payload, model,
                    best_epoch, binary_metrics, args)
    np.save(output_dir / "binary_head_probabilities.npy", binary_probabilities)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "source_checkpoint": str(checkpoint), "best_epoch": best_epoch,
        "settings": {**vars(args), "device": str(device), "loss": "BCEWithLogitsLoss",
                     "fce": False, "trainable_parameters": trainable_head},
        "methods": {"Original Native52": native_metrics,
                    "Frozen AttriVision + BCE40": binary_metrics},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    print_full_metrics("A. Original Native52 (52-state -> 40-D soft -> paired L1)", native_metrics)
    print_full_metrics("B. Frozen AttriVision + BCE40 (sigmoid -> official paired L1)", binary_metrics)
    print("\n" + "=" * 88)
    print(f"{'Method':32s} {'R1':>8s} {'mAP':>8s} {'AUROC':>8s} {'InstF1':>9s} {'BitErr':>8s}")
    print("-" * 88)
    for name, row in (("Original Native52", native_metrics),
                      ("Frozen AttriVision + BCE40", binary_metrics)):
        print(f"{name:32s} {100*row['rank1']:7.2f}% {100*row['map']:7.2f}% "
              f"{row['macro_auroc']:8.4f} {row['instance_f1']:9.4f} "
              f"{row['mean_hamming_error']:8.3f}")
    print("=" * 88)
    print(f"Best epoch: {best_epoch}; artifacts: {output_dir}")


if __name__ == "__main__":
    main()
