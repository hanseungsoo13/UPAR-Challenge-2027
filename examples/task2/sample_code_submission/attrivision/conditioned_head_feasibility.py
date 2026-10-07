"""Frozen-feature feasibility comparison of Linear40 and a conditioned head.

The A1 AttriVision checkpoint is loaded read-only and never optimized.  Its
normalized image features and the exact 40 official positive-prompt text
features are cached once.  H0 and H1 then train only their newly initialized
heads with identical BCE, optimizer, seed, split, and cached image features.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from attrivision.compare_inference import (  # noqa: E402
    encode_texts, native_state_outputs, prediction_metrics, ranks_at_k,
)
from attrivision.datasets.attribute_prompts import (  # noqa: E402
    CategoryPromptMapper, prompts_for_attributes,
)
from attrivision.engine.evaluator_abpr import encode_gallery  # noqa: E402
from attrivision.engine.paired_attribute import learned_inverse_temperature  # noqa: E402
from attrivision.evaluate_native52_hard import (  # noqa: E402
    hard_category_projection, hard_prediction_metrics, hamming_distances,
    retrieval_result, semantic_query_metrics,
)
from attrivision.native52_retrieval_ablation import (  # noqa: E402
    category_indices, category_nll, category_softmax,
)
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    l1_attribute_distances, load_retrieval_annotations, reorder_columns,
)


METHODS = {
    "H0": "Frozen Linear40",
    "H1": "Text-Conditioned SharedMLP",
}


class Linear40Head(nn.Module):
    """H0: a fresh Linear(512, 40) probe."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(512, 40)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features.float())


class TextConditionedSharedHead(nn.Module):
    """H1: one shared MLP applied to all 40 image/text interactions."""

    def __init__(self, text_features: torch.Tensor) -> None:
        super().__init__()
        if text_features.shape != (40, 512):
            raise ValueError(f"Expected official text features [40,512], got {tuple(text_features.shape)}")
        if not torch.isfinite(text_features).all():
            raise ValueError("Official text features contain NaN or Inf")
        self.register_buffer("text_features", text_features.detach().float().clone(), persistent=True)
        self.shared_mlp = nn.Sequential(
            nn.Linear(2049, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image = image_features.float()
        text = self.text_features
        batch_size = image.shape[0]
        image_expanded = image[:, None, :].expand(batch_size, 40, 512)
        text_expanded = text[None, :, :].expand(batch_size, 40, 512)
        cosine = (image_expanded * text_expanded).sum(dim=-1, keepdim=True)
        interaction = torch.cat(
            (image_expanded, text_expanded, image_expanded * text_expanded,
             (image_expanded - text_expanded).abs(), cosine),
            dim=-1,
        )
        if interaction.shape != (batch_size, 40, 2049):
            raise RuntimeError(f"Invalid conditioned interaction shape: {tuple(interaction.shape)}")
        return self.shared_mlp(interaction).squeeze(-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen Linear40 and an official-text-conditioned shared head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="outputs/attrivision_ablation/A1/checkpoint_best.pth")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs/attrivision_conditioned_head")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512, help="cached-feature head-training batch")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--head-eval-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--train-features", help="optional cached finite [N,512] training features")
    parser.add_argument("--val-features", help="optional cached finite [G,512] validation features")
    parser.add_argument("--official-text-features", help="optional cached normalized [40,512] features")
    parser.add_argument("--native-text-features", help="optional cached normalized [52,512] features")
    parser.add_argument("--max-train-samples", type=int, help="smoke testing only")
    parser.add_argument("--max-val-samples", type=int, help="smoke testing only")
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


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash names, metadata, and every tensor byte for an exact integrity check."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_finite_cache(path: Path, shape: tuple[int, int], label: str) -> torch.Tensor:
    values = np.load(path, allow_pickle=False)
    tensor = torch.from_numpy(values.astype(np.float32, copy=False))
    if tensor.shape != shape or not torch.isfinite(tensor).all():
        raise ValueError(f"{label} cache must be finite {list(shape)}, got {tuple(tensor.shape)}: {path}")
    print(f"{label} cache reused: {path}", flush=True)
    return tensor.contiguous()


def feature_cache_path(root: Path, split: str, maximum: int | None) -> Path:
    suffix = "" if maximum is None else f"_smoke_{maximum}"
    return root / f"{split}_features{suffix}.npy"


def load_or_encode_images(
    requested: str | None, default_path: Path, model: nn.Module, table: AnnotationTable,
    roots: Sequence[Path], device: torch.device, args: argparse.Namespace, label: str,
) -> torch.Tensor:
    path = Path(requested).resolve() if requested else default_path
    if path.is_file():
        return load_finite_cache(path, (len(table.image_paths), 512), label)
    features = encode_gallery(
        model, table.image_paths, roots, build_eval_transform(args.image_size), device,
        args.encode_batch_size, args.num_workers, args.amp, args.progress_every, label,
    )
    if features.shape != (len(table.image_paths), 512) or not torch.isfinite(features).all():
        raise RuntimeError(f"{label} encoder produced invalid features {tuple(features.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features.numpy())
    print(f"{label} cache saved: {path}", flush=True)
    return features.contiguous()


def load_or_encode_texts(
    requested: str | None, default_path: Path, model: nn.Module, prompts: Sequence[str],
    device: torch.device, amp: bool, label: str,
) -> torch.Tensor:
    path = Path(requested).resolve() if requested else default_path
    if path.is_file():
        return load_finite_cache(path, (len(prompts), 512), label)
    features = encode_texts(model, prompts, device, amp)
    if features.shape != (len(prompts), 512) or not torch.isfinite(features).all():
        raise RuntimeError(f"{label} encoder produced invalid features {tuple(features.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features.numpy())
    print(f"{label} cache saved: {path}", flush=True)
    return features.contiguous()


@torch.inference_mode()
def infer_head(head: nn.Module, features: torch.Tensor, labels: np.ndarray,
               device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    head.eval()
    labels_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    loader = DataLoader(TensorDataset(features, labels_tensor), batch_size=batch_size, shuffle=False)
    chunks: list[torch.Tensor] = []
    total_bce = 0.0
    element_count = 0
    for feature_batch, label_batch in loader:
        logits = head(feature_batch.to(device, non_blocking=True))
        targets = label_batch.to(device, non_blocking=True)
        total_bce += float(F.binary_cross_entropy_with_logits(logits, targets, reduction="sum"))
        element_count += targets.numel()
        chunks.append(torch.sigmoid(logits).float().cpu())
    probabilities = torch.cat(chunks).numpy() if chunks else np.empty((0, 40), dtype=np.float32)
    return probabilities, total_bce / max(element_count, 1)


def evaluate_probabilities(
    probabilities: np.ndarray, validation_bce: float, labels: np.ndarray,
    queries: np.ndarray, ids: np.ndarray, chunk_size: int,
) -> dict[str, float]:
    prediction, _ = prediction_metrics(labels, probabilities)
    hard = hard_prediction_metrics(labels, probabilities >= 0.5)

    soft_distances = l1_attribute_distances(queries, probabilities, chunk_size)
    soft = retrieval_result(soft_distances, ids)
    soft_semantic = semantic_query_metrics(soft_distances, ids, chunk_size)
    del soft_distances

    hard_distances = hamming_distances(queries, probabilities >= 0.5, chunk_size)
    hard_retrieval = retrieval_result(hard_distances, ids)
    hard_semantic = semantic_query_metrics(hard_distances, ids, chunk_size)
    del hard_distances

    result: dict[str, float] = {
        "bce": float(validation_bce),
        "macro_auroc": prediction["macro_auroc"],
        "macro_ap": prediction["macro_ap"],
        "instance_f1": hard["instance_f1"],
        "mean_hamming_error": hard["mean_hamming_error"],
        "mean_hamming_error_over_40": hard["mean_hamming_error_over_40"],
        "exact_match": hard["exact_match"],
        "le1_bit_error": hard["le1_bit_error"],
        "le2_bit_error": hard["le2_bit_error"],
        **soft,
        **soft_semantic,
    }
    result.update({f"hard_hamming_{key}": value for key, value in hard_retrieval.items()})
    result.update({f"hard_hamming_{key}": value for key, value in hard_semantic.items()})
    return result


def native52_b2_metrics(
    model: nn.Module, gallery_features: torch.Tensor, state_features: torch.Tensor,
    labels: np.ndarray, queries: np.ndarray, ids: np.ndarray,
    attribute_names: Sequence[str], chunk_size: int,
) -> dict[str, float]:
    """Reproduce fixed-checkpoint B2 Category-NLL and Native52 PAR metrics."""
    mapper = CategoryPromptMapper(attribute_names)
    groups = category_indices(mapper)
    inverse_temperature = learned_inverse_temperature(model)
    logits52 = (gallery_features.float() @ state_features.float().T).numpy()
    probabilities52 = category_softmax(logits52 * inverse_temperature, groups)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    distances = category_nll(queries52, probabilities52, groups)
    retrieval = retrieval_result(distances, ids)
    semantic = semantic_query_metrics(distances, ids, chunk_size)
    del distances

    probabilities40, _ = native_state_outputs(
        gallery_features, state_features, mapper, inverse_temperature,
    )
    prediction, _ = prediction_metrics(labels, probabilities40)
    hard = hard_prediction_metrics(
        labels, hard_category_projection(gallery_features, state_features, mapper),
    )
    return {
        "temperature": float(1.0 / inverse_temperature),
        "macro_auroc": prediction["macro_auroc"],
        "macro_ap": prediction["macro_ap"],
        "instance_f1": hard["instance_f1"],
        "mean_hamming_error": hard["mean_hamming_error"],
        "mean_hamming_error_over_40": hard["mean_hamming_error_over_40"],
        "exact_match": hard["exact_match"],
        "le1_bit_error": hard["le1_bit_error"],
        "le2_bit_error": hard["le2_bit_error"],
        **retrieval,
        **semantic,
    }


def save_head_checkpoint(
    path: Path, method: str, head_state: dict[str, torch.Tensor], epoch: int,
    metrics: dict[str, float], optimizer_state: dict[str, Any] | None,
    config: dict[str, Any], source_checkpoint: Path, frozen_sha256: str,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": method,
        "head_state_dict": {key: value.detach().cpu() for key, value in head_state.items()},
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "source_checkpoint": str(source_checkpoint),
        "frozen_encoder_sha256": frozen_sha256,
        "config": config,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_head(
    key: str, head: nn.Module, train_features: torch.Tensor, train_labels: np.ndarray,
    val_features: torch.Tensor, val_labels: np.ndarray, queries: np.ndarray,
    ids: np.ndarray, device: torch.device, args: argparse.Namespace,
    output_dir: Path, common_config: dict[str, Any], source_checkpoint: Path,
    frozen_sha256: str,
) -> dict[str, Any]:
    set_seed(args.seed, args.deterministic)
    head = head.to(device)
    trainable_count = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    optimizer_count = sum(parameter.numel() for parameter in optimizer_parameters)
    if optimizer_count != trainable_count or {id(item) for item in optimizer_parameters} != {
        id(item) for item in head.parameters()
    }:
        raise RuntimeError(f"{key} optimizer does not contain exactly the head parameters")

    print(f"\n{key} {METHODS[key]}")
    print(f"Trainable {key} params = {trainable_count:,}")
    print(f"Optimizer param count = {optimizer_count:,}")
    print("Encoder optimizer param count = 0", flush=True)

    config = {
        **common_config,
        "method": key,
        "method_name": METHODS[key],
        "head_architecture": str(head),
        "trainable_parameters": trainable_count,
        "optimizer_parameter_count": optimizer_count,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    labels_tensor = torch.from_numpy(train_labels.astype(np.float32, copy=False))
    loader = DataLoader(
        TensorDataset(train_features, labels_tensor), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    criterion = nn.BCEWithLogitsLoss()
    best_map = -float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        head.train()
        total_loss = 0.0
        sample_count = 0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(feature_batch)
            sample_count += len(feature_batch)

        probabilities, validation_bce = infer_head(
            head, val_features, val_labels, device, args.head_eval_batch_size,
        )
        metrics = evaluate_probabilities(
            probabilities, validation_bce, val_labels, queries, ids,
            args.distance_chunk_size,
        )
        row = {
            "epoch": epoch,
            "train_bce": total_loss / max(sample_count, 1),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: trainBCE={row['train_bce']:.6f}, "
            f"valBCE={metrics['bce']:.6f}, AUROC={metrics['macro_auroc']:.4f}, "
            f"InstF1={metrics['instance_f1']:.4f}, BitErr={metrics['mean_hamming_error']:.3f}, "
            f"Exact={100*metrics['exact_match']:.2f}%, R1={100*metrics['rank1']:.2f}%, "
            f"mAP={100*metrics['map']:.2f}%", flush=True,
        )
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(head.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"{key} training produced no best checkpoint")
    last_state = copy.deepcopy(head.state_dict())
    last_metrics = {key: value for key, value in history[-1].items() if key not in {"epoch", "elapsed_seconds"}}
    save_head_checkpoint(
        output_dir / "checkpoint_best.pth", key, best_state, best_epoch, best_metrics,
        None, config, source_checkpoint, frozen_sha256,
    )
    save_head_checkpoint(
        output_dir / "checkpoint_last.pth", key, last_state, args.epochs, last_metrics,
        optimizer.state_dict(), config, source_checkpoint, frozen_sha256,
    )
    write_csv(output_dir / "training_log.csv", history)
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "best": best_metrics, "last": last_metrics},
                  handle, indent=2, ensure_ascii=False)
    return {"best_epoch": best_epoch, "metrics": best_metrics, "trainable_parameters": trainable_count}


def print_final(results: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 103)
    print(f"{'Method':34s} {'AUROC':>8s} {'InstF1':>8s} {'BitErr':>8s} "
          f"{'Exact':>8s} {'R1':>8s} {'mAP':>8s}")
    print("-" * 103)
    for name, row in results.items():
        print(f"{name:34s} {row['macro_auroc']:8.4f} {row['instance_f1']:8.4f} "
              f"{row['mean_hamming_error']:8.3f} {100*row['exact_match']:7.2f}% "
              f"{100*row['rank1']:7.2f}% {100*row['map']:7.2f}%")
    print("=" * 103)
    for baseline in ("H0 Frozen Linear40", "Native52 Category-NLL"):
        print(f"\nH1 - {baseline}:")
        for metric in ("macro_auroc", "instance_f1", "mean_hamming_error", "exact_match", "rank1", "map"):
            delta = results["H1 Text-Conditioned SharedMLP"][metric] - results[baseline][metric]
            print(f"  Δ{metric} = {delta:+.6f}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_dir).resolve()
    cache_root = output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    model, source_payload = load_model(checkpoint, device)
    if source_payload.get("prompt_mode") != "category_complete":
        raise ValueError("The source checkpoint must use category_complete Native52 prompts")
    attribute_names = list(source_payload["attribute_names"])
    if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
        raise ValueError("Checkpoint must contain exactly 40 unique official attributes")
    official_prompts = prompts_for_attributes(attribute_names)
    mapper = CategoryPromptMapper(attribute_names)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The complete source AttriVision model must be frozen")
    frozen_params = sum(parameter.numel() for parameter in model.clip.parameters())
    frozen_sha256_before = tensor_state_sha256(model.clip.state_dict())
    print(f"Frozen encoder params = {frozen_params:,}")
    print("Frozen encoder trainable params = 0")
    print(f"Frozen encoder SHA256 before = {frozen_sha256_before}", flush=True)

    train_gt = find_annotation_file(data_root, "train")
    val_gt = find_annotation_file(data_root, "val")
    train_table = subset(read_gt_csv(train_gt), args.max_train_samples)
    val_table = subset(read_gt_csv(val_gt), args.max_val_samples)
    train_labels = reorder_columns(train_table.labels, train_table.attribute_names, attribute_names)
    val_labels = reorder_columns(val_table.labels, val_table.attribute_names, attribute_names)
    if args.max_val_samples is None:
        queries_raw, ids, query_names = load_retrieval_annotations(val_gt.parent, val_table)
        queries = reorder_columns(queries_raw, query_names, attribute_names)
    else:
        queries, ids = np.unique(val_labels, axis=0, return_inverse=True)
    if not np.array_equal(val_labels, queries[ids]):
        raise RuntimeError("Validation labels do not match official semantic-query IDs")

    default_val_cache = feature_cache_path(cache_root, "val", args.max_val_samples)
    existing_a1_gallery = checkpoint.parent / "native52_retrieval_ablation" / "gallery_features.npy"
    if args.val_features is None and args.max_val_samples is None and existing_a1_gallery.is_file():
        default_val_cache = existing_a1_gallery
    train_features = load_or_encode_images(
        args.train_features, feature_cache_path(cache_root, "train", args.max_train_samples),
        model, train_table, [data_root, train_gt.parent, REPOSITORY_ROOT], device, args,
        "[train image]",
    )
    val_features = load_or_encode_images(
        args.val_features, default_val_cache, model, val_table,
        [data_root, val_gt.parent, REPOSITORY_ROOT], device, args, "[val image]",
    )
    official_text_features = load_or_encode_texts(
        args.official_text_features, cache_root / "official40_text_features.npy", model,
        official_prompts, device, args.amp, "[official40 text]",
    )
    default_native_cache = checkpoint.parent / "native52_retrieval_ablation" / "native52_text_features.npy"
    native_text_features = load_or_encode_texts(
        args.native_text_features, default_native_cache, model, mapper.prompts,
        device, args.amp, "[native52 text]",
    )

    native_metrics = native52_b2_metrics(
        model, val_features, native_text_features, val_labels, queries, ids,
        attribute_names, args.distance_chunk_size,
    )
    print(
        f"Native52 Category-NLL reproduced: T={native_metrics['temperature']:.8f}, "
        f"R1={100*native_metrics['rank1']:.2f}%, mAP={100*native_metrics['map']:.2f}%",
        flush=True,
    )

    common_config: dict[str, Any] = {
        "source_checkpoint": str(checkpoint),
        "frozen_encoder_sha256": frozen_sha256_before,
        "encoder_frozen": True,
        "encoder_optimizer_parameter_count": 0,
        "official_attribute_names": attribute_names,
        "official_positive_prompts": official_prompts,
        "text_embeddings_frozen": True,
        "text_embedding_shape": [40, 512],
        "image_feature_shape": [512],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "class_balancing": "none (matches train_frozen_binary_head.py)",
        "fce": False,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "train_samples": len(train_features),
        "validation_samples": len(val_features),
        "semantic_queries": len(queries),
        "retrieval_soft": "official paired 40-D L1 on sigmoid probabilities",
        "retrieval_hard_diagnostic": "Hamming on sigmoid(logit)>=0.5",
    }

    h0 = train_head(
        "H0", Linear40Head(), train_features, train_labels, val_features, val_labels,
        queries, ids, device, args, output_root / "H0_linear40", common_config,
        checkpoint, frozen_sha256_before,
    )
    h1 = train_head(
        "H1", TextConditionedSharedHead(official_text_features), train_features,
        train_labels, val_features, val_labels, queries, ids, device, args,
        output_root / "H1_conditioned", common_config, checkpoint, frozen_sha256_before,
    )

    frozen_sha256_after = tensor_state_sha256(model.clip.state_dict())
    exact_match = frozen_sha256_after == frozen_sha256_before
    print(f"\nFrozen encoder SHA256 after  = {frozen_sha256_after}")
    print(f"Frozen encoder integrity check = {'bitwise/exact match' if exact_match else 'FAILED'}")
    if not exact_match:
        raise RuntimeError("Frozen image/text encoder, projection, or logit_scale changed")

    results = {
        "Native52 Category-NLL": native_metrics,
        "H0 Frozen Linear40": h0["metrics"],
        "H1 Text-Conditioned SharedMLP": h1["metrics"],
    }
    summary = {
        "source_checkpoint": str(checkpoint),
        "frozen_encoder_sha256_before": frozen_sha256_before,
        "frozen_encoder_sha256_after": frozen_sha256_after,
        "frozen_encoder_exact_match": exact_match,
        "methods": results,
        "deltas": {
            "H1_minus_H0": {metric: h1["metrics"][metric] - h0["metrics"][metric]
                            for metric in ("macro_auroc", "instance_f1", "mean_hamming_error",
                                           "exact_match", "rank1", "map")},
            "H1_minus_Native52_B2": {
                metric: h1["metrics"][metric] - native_metrics[metric]
                for metric in ("macro_auroc", "instance_f1", "mean_hamming_error",
                               "exact_match", "rank1", "map")
            },
        },
    }
    with (output_root / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print_final(results)
    print(f"\nArtifacts: {output_root}")


if __name__ == "__main__":
    main()
