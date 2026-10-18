#!/usr/bin/env python3
"""Audit per-value retrieval performance for the shared CLIP ViT-L/14 model.

Validation queries are grouped by each active state in the canonical 12-category,
52-state mapper. Every query keeps the official exact-label gallery positives and
Native52 Category-NLL ranking. Train annotations are read only to count state
frequency; no train images or labels enter inference or retrieval scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

TOOLS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = TOOLS_DIR.parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_fromscratch_jointfce import (  # noqa: E402
    category_nll,
    fused_category_probabilities,
    retrieval_views,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.shared_clip_l14.experiment import _load_shared_model  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import (  # noqa: E402
    ImagePathDataset,
    find_annotation_file,
    read_gt_csv,
)
from upar.retrieval import autocast, reorder_columns  # noqa: E402


METHODS = ("global", "local", "fusion")
RETRIEVAL_METRICS = ("rank1", "rank5", "rank10", "mAP", "mADM", "mINP", "semantic_top1")
BOOTSTRAP_METRICS = ("rank1", "mAP", "mADM")
BOOTSTRAP_REPLICATES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.progress_every < 0:
        parser.error("batch size must be positive; workers and progress interval cannot be negative")
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def state_names(mapper: CategoryPromptMapper) -> tuple[list[str], list[str], list[int]]:
    """Return user-facing value labels, category labels, and category indices."""
    fallback_names = {
        "age_unknown": "Age-Unknown",
        "hair_other": "Hair-Length-Other",
        "upper_color_unspecified": "UpperBody-Color-Unspecified",
        "lower_color_unspecified": "LowerBody-Color-Unspecified",
        "lower_type_other": "LowerBody-Type-Other",
        "glasses_none": "Accessory-Glasses-None",
    }
    binary_negative_names = {
        "gender_man": "Gender-Male",
        "upper_sleeves_long": "UpperBody-Length-Long",
        "lower_length_long": "LowerBody-Length-Long",
        "backpack_no": "Accessory-Backpack-Absent",
        "bag_no": "Accessory-Bag-Absent",
        "hat_no": "Accessory-Hat-Absent",
    }
    labels: dict[str, str] = {}
    categories: dict[str, str] = {}
    category_indices: dict[str, int] = {}
    category_names = [entry[0] for entry in mapper._MULTI_GROUPS]
    category_names.extend(entry[0] for entry in mapper._BINARY_GROUPS)
    index = 0
    for category, columns, keys, fallback in mapper._MULTI_GROUPS:
        for column, key in zip(columns, keys):
            labels[key] = column
            categories[key] = category
            category_indices[key] = index
        labels[fallback] = fallback_names[fallback]
        categories[fallback] = category
        category_indices[fallback] = index
        index += 1
    for column, positive_key, negative_key in mapper._BINARY_GROUPS:
        labels[positive_key] = column
        labels[negative_key] = binary_negative_names[negative_key]
        categories[positive_key] = categories[negative_key] = column
        category_indices[positive_key] = category_indices[negative_key] = index
        index += 1
    if set(labels) != set(mapper.keys) or index != 12:
        raise RuntimeError("State labels do not cover the canonical 52 states and 12 categories")
    return (
        [labels[key] for key in mapper.keys],
        [categories[key] for key in mapper.keys],
        [category_indices[key] for key in mapper.keys],
    )


def _score_cache_identity(
    checkpoint: Path,
    validation_annotation: Path,
    image_count: int,
    attribute_names: Sequence[str],
) -> dict[str, Any]:
    checkpoint_stat = checkpoint.stat()
    annotation_stat = validation_annotation.stat()
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "validation_annotation": str(validation_annotation),
        "validation_annotation_size_bytes": annotation_stat.st_size,
        "validation_annotation_mtime_ns": annotation_stat.st_mtime_ns,
        "image_count": image_count,
        "attribute_names": list(attribute_names),
        "transform": "resize_pad_crop_224_eval",
        "model_forward": "joint_features_with_tokens_return_attention",
        "split": "val",
    }


def load_score_cache(
    path: Path,
    identity: dict[str, Any],
    expected_images: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            global_scores = archive["global_scores"]
            local_scores = archive["local_scores"]
        cache_identity = {key: metadata.get(key) for key in identity}
        if cache_identity != identity:
            return None
        if global_scores.shape != (expected_images, 52):
            return None
        if local_scores.shape != (expected_images, 12, 52):
            return None
        if not np.isfinite(global_scores).all() or not np.isfinite(local_scores).all():
            return None
        return global_scores.astype(np.float32), local_scores.astype(np.float32), metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_score_cache(
    path: Path,
    global_scores: np.ndarray,
    local_scores: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(
        temporary,
        global_scores=global_scores.astype(np.float32),
        local_scores=local_scores.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(path)


@torch.inference_mode()
def encode_validation_scores(
    checkpoint: Path,
    table: Any,
    gt_path: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = choose_device(args.device)
    use_amp = not args.disable_amp and device.type == "cuda"
    model, payload = _load_shared_model(checkpoint, device)
    model.eval()
    mapper = CategoryPromptMapper(payload["attribute_names"])
    category_temperature = float(
        payload.get("training_config", {}).get("category_temperature", 0.01)
    )
    state_tokens = model.base.tokenize(mapper.prompts).to(device)
    with autocast(device, use_amp):
        state_features = model.encode_text(state_tokens).float()

    loader = DataLoader(
        ImagePathDataset(
            table.image_paths,
            [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(224, "resize_pad_crop"),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    global_chunks: list[np.ndarray] = []
    local_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, use_amp):
            global_features, local_features, _, attention = model.joint_features_with_tokens(
                images.to(device, non_blocking=True),
            )
            # Requesting weights follows the preceding diagnostic's attention
            # forward path; discard them after obtaining numerically aligned features.
            del attention
            global_scores = global_features @ state_features.T
            local_scores = model.category_state_scores_from_features(
                local_features, state_features,
            )
        global_chunks.append(global_scores.float().cpu().numpy())
        local_chunks.append(local_scores.float().cpu().numpy())
        if args.progress_every and (
            batch_index % args.progress_every == 0 or batch_index == len(loader)
        ):
            processed = min(batch_index * args.batch_size, len(table.image_paths))
            print(
                f"[value-retrieval] batch {batch_index}/{len(loader)} "
                f"images {processed}/{len(table.image_paths)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    global_result = np.concatenate(global_chunks, axis=0).astype(np.float32)
    local_result = np.concatenate(local_chunks, axis=0).astype(np.float32)
    if global_result.shape != (len(table.image_paths), 52):
        raise RuntimeError(f"Unexpected global score shape {global_result.shape}")
    if local_result.shape != (len(table.image_paths), 12, 52):
        raise RuntimeError(f"Unexpected local score shape {local_result.shape}")
    if not np.isfinite(global_result).all() or not np.isfinite(local_result).all():
        raise RuntimeError("Validation scores contain NaN or Inf")
    return global_result, local_result, {
        "category_temperature": category_temperature,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amp": use_amp,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "inference_seconds": time.perf_counter() - started,
        "score_cache_reused": False,
        "experiment": payload.get("experiment"),
        "epoch": payload.get("epoch"),
        "model_stage": payload.get("model_stage"),
        "architecture": payload.get("architecture"),
        "model_name": payload.get("model_name"),
    }


def per_query_metrics(
    distance_rows: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute each official exact-match metric for each query row."""
    count = len(query_labels)
    result = {key: np.empty(count, dtype=np.float64) for key in RETRIEVAL_METRICS}
    for query_index, (query, distances) in enumerate(zip(query_labels, distance_rows)):
        agreement = 1.0 - np.abs(gallery_labels - query).sum(axis=1) / gallery_labels.shape[1]
        relevant = agreement == 1.0
        positive_count = int(relevant.sum())
        if positive_count == 0:
            raise ValueError(f"Validation query {query_index} has no exact-label gallery positive")
        order = np.argsort(distances, kind="stable")
        positions = np.flatnonzero(relevant[order])
        ranks = positions.astype(np.float64) + 1.0
        mean_agreement = float(agreement.mean())
        maximum_agreement = float(agreement.max())
        if maximum_agreement == mean_agreement:
            normalized_agreement = np.ones(len(agreement), dtype=np.float64)
        else:
            normalized_agreement = np.maximum(
                0.0,
                (agreement - mean_agreement) / (maximum_agreement - mean_agreement),
            )
        prefix = np.cumsum(normalized_agreement[order], dtype=np.float64)
        result["rank1"][query_index] = float(ranks[0] <= 1)
        result["rank5"][query_index] = float(ranks[0] <= 5)
        result["rank10"][query_index] = float(ranks[0] <= 10)
        result["mAP"][query_index] = float(np.mean(np.arange(1, positive_count + 1) / ranks))
        result["mADM"][query_index] = float(np.mean(prefix[positions] / ranks))
        result["mINP"][query_index] = float(positive_count / ranks[-1])
        result["semantic_top1"][query_index] = float(positions[0] == 0)
    return result


def per_query_metrics_chunked(
    probabilities: np.ndarray,
    queries52: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    groups: Sequence[np.ndarray],
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    metrics = {key: np.empty(len(queries52), dtype=np.float64) for key in RETRIEVAL_METRICS}
    for start in range(0, len(queries52), batch_size):
        stop = min(start + batch_size, len(queries52))
        distances = category_nll(queries52[start:stop], probabilities, groups)
        batch_metrics = per_query_metrics(
            distances, query_labels[start:stop], gallery_labels,
        )
        for key in RETRIEVAL_METRICS:
            metrics[key][start:stop] = batch_metrics[key]
    return metrics


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        result[order[cursor:stop]] = ((cursor + 1) + stop) / 2.0
        cursor = stop
    return result


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if len(left_array) < 3:
        return None
    left_ranks = average_rank(left_array)
    right_ranks = average_rank(right_array)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None, None
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[sampled].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def category_correlation_rows(
    category_names: Sequence[str],
    category_state_indices: Sequence[Sequence[int]],
    state_counts: np.ndarray,
    train_total: int,
    query_counts: np.ndarray,
    fused_metrics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for category_index, (category, indices) in enumerate(zip(category_names, category_state_indices)):
        selected_indices = np.asarray(indices, dtype=np.int64)
        training_counts = state_counts[selected_indices]
        query_support = query_counts[selected_indices]
        positive = training_counts > 0
        head_count = int(training_counts.max()) if len(training_counts) else 0
        nonzero_counts = training_counts[positive]
        tail_count = int(nonzero_counts.min()) if len(nonzero_counts) else 0
        category_sum = int(training_counts.sum())
        supported = query_support > 0
        stable = query_support >= 30
        row: dict[str, Any] = {
            "category_index": category_index,
            "category": category,
            "state_count": len(selected_indices),
            "supported_state_count": int(supported.sum()),
            "states_with_at_least_30_queries": int(stable.sum()),
            "train_image_count_total": train_total,
            "train_state_positive_count_sum": category_sum,
            "train_value_share_head": head_count / category_sum if category_sum else None,
            "train_value_share_tail_nonzero": tail_count / category_sum if category_sum else None,
            "head_to_tail_nonzero_count_ratio": (
                head_count / tail_count if tail_count else None
            ),
            "zero_train_support_state_count": int((training_counts == 0).sum()),
            "median_validation_query_support": (
                float(np.median(query_support[supported])) if supported.any() else None
            ),
        }
        for metric in BOOTSTRAP_METRICS:
            row[f"spearman_log_train_count_vs_fusion_{metric}_supported"] = (
                spearman_rho(
                    np.log1p(training_counts[supported]),
                    fused_metrics[metric][selected_indices][supported],
                ) if supported.sum() >= 3 else None
            )
            row[f"spearman_log_train_count_vs_fusion_{metric}_queries_ge_30"] = (
                spearman_rho(
                    np.log1p(training_counts[stable]),
                    fused_metrics[metric][selected_indices][stable],
                ) if stable.sum() >= 3 else None
            )
        rows.append(row)
    return rows


def black_red_confusion_rows(
    mapper: CategoryPromptMapper,
    labels52: np.ndarray,
    groups: Sequence[np.ndarray],
    methods: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    specs = (
        ("upper_color", "upper_color_black", "upper_color_red"),
        ("lower_color", "lower_color_black", "lower_color_red"),
    )
    rows = []
    for category, black_key, red_key in specs:
        category_index = next(
            index for index, entry in enumerate(mapper._MULTI_GROUPS)
            if entry[0] == category
        )
        category_indices = np.asarray(groups[category_index], dtype=np.int64)
        black_index = state_index[black_key]
        red_index = state_index[red_key]
        truth_black = labels52[:, black_index] & ~labels52[:, red_index]
        truth_red = labels52[:, red_index] & ~labels52[:, black_index]
        for method, probabilities in methods.items():
            predictions = category_indices[np.argmax(probabilities[:, category_indices], axis=1)]
            for truth_mask, truth_key, competing_key, competing_index in (
                (truth_black, black_key, red_key, red_index),
                (truth_red, red_key, black_key, black_index),
            ):
                support = int(truth_mask.sum())
                confusion_count = int(np.count_nonzero(truth_mask & (predictions == competing_index)))
                rows.append({
                    "category": category,
                    "method": method,
                    "unambiguous_true_state": truth_key,
                    "competing_predicted_state": competing_key,
                    "unambiguous_true_image_count": support,
                    "black_red_confusion_count": confusion_count,
                    "black_red_confusion_rate": confusion_count / support if support else None,
                })
    return rows


def load_prior_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path, table, queries_raw, ids, query_names = retrieval_views(data_root, "val", None)
    validation_annotation = gt_path.resolve()
    train_path = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_path)
    attribute_names = list(table.attribute_names)
    queries = reorder_columns(queries_raw, query_names, attribute_names)
    gallery_labels = reorder_columns(table.labels, table.attribute_names, attribute_names)
    train_labels = reorder_columns(train_table.labels, train_table.attribute_names, attribute_names)
    mapper = CategoryPromptMapper(attribute_names)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    category_names = [entry[0] for entry in mapper._MULTI_GROUPS]
    category_names.extend(entry[0] for entry in mapper._BINARY_GROUPS)
    state_labels, state_categories, state_category_indices = state_names(mapper)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy().astype(bool)
    gallery_labels52 = mapper.encode(torch.from_numpy(gallery_labels).float()).numpy().astype(bool)
    train_labels52 = mapper.encode(torch.from_numpy(train_labels).float()).numpy().astype(bool)
    if len(ids) != len(gallery_labels) or not np.array_equal(gallery_labels, queries[ids]):
        raise RuntimeError("Validation image/query IDs do not reproduce the official exact labels")
    if not np.array_equal(gallery_labels52, queries52[ids]):
        raise RuntimeError("Validation IDs do not reproduce all mapped 52-state labels")

    cache_path = output_dir / "validation_state_scores.npz"
    cache_identity = _score_cache_identity(
        checkpoint, validation_annotation, len(table.image_paths), attribute_names,
    )
    cached = load_score_cache(cache_path, cache_identity, len(table.image_paths))
    if cached is None:
        global_scores, local_scores, inference = encode_validation_scores(
            checkpoint, table, gt_path, data_root, args,
        )
        category_temperature = float(inference["category_temperature"])
        cache_metadata = {**cache_identity, **inference, "category_temperature": category_temperature}
        save_score_cache(cache_path, global_scores, local_scores, cache_metadata)
    else:
        global_scores, local_scores, cache_metadata = cached
        category_temperature = float(cache_metadata["category_temperature"])
        inference = {
            "device": cache_metadata.get("device", "cached"),
            "device_name": cache_metadata.get("device_name", "cached"),
            "cuda_visible_devices": cache_metadata.get("cuda_visible_devices"),
            "amp": cache_metadata.get("amp"),
            "inference_seconds": cache_metadata.get("inference_seconds"),
            "score_cache_reused": True,
        }
    if global_scores.shape[0] != len(gallery_labels):
        raise RuntimeError("Cached/inferred score row order does not match the validation gallery")

    global_probabilities, local_probabilities, fusion_probabilities = fused_category_probabilities(
        global_scores, local_scores, groups, category_temperature, 0.5,
    )
    method_probabilities = {
        "global": global_probabilities,
        "local": local_probabilities,
        "fusion": fusion_probabilities,
    }

    started = time.perf_counter()
    query_metric_values: dict[str, dict[str, np.ndarray]] = {}
    for method, probabilities in method_probabilities.items():
        query_metric_values[method] = per_query_metrics_chunked(
            probabilities, queries52, queries, gallery_labels, groups,
        )
        print(f"[value-retrieval] ranked {method} queries", flush=True)

    expected_positive_counts = np.bincount(ids, minlength=len(queries))
    exact_positive_counts = np.asarray([
        np.count_nonzero(np.all(gallery_labels == query[None, :], axis=1))
        for query in queries
    ])
    if not np.array_equal(expected_positive_counts, exact_positive_counts):
        raise RuntimeError("Exact-label gallery positives differ from the official query IDs")

    train_state_counts = train_labels52.sum(axis=0).astype(np.int64)
    val_image_state_counts = gallery_labels52.sum(axis=0).astype(np.int64)
    val_query_state_counts = queries52.sum(axis=0).astype(np.int64)
    rng = np.random.default_rng(20261018)
    per_state_rows: list[dict[str, Any]] = []
    for state_index, state_key in enumerate(mapper.keys):
        included_queries = queries52[:, state_index]
        query_count = int(included_queries.sum())
        category_index = state_category_indices[state_index]
        category_group = groups[category_index]
        category_train_total = int(train_state_counts[category_group].sum())
        row: dict[str, Any] = {
            "category": state_categories[state_index],
            "category_index": category_index,
            "state_key": state_key,
            "value": state_labels[state_index],
            "train_image_count": int(train_state_counts[state_index]),
            "train_image_prevalence": float(train_state_counts[state_index] / len(train_labels)),
            "train_within_category_share": (
                float(train_state_counts[state_index] / category_train_total)
                if category_train_total else None
            ),
            "val_image_count": int(val_image_state_counts[state_index]),
            "val_image_prevalence": float(val_image_state_counts[state_index] / len(gallery_labels)),
            "val_query_count": query_count,
            "query_support_label": (
                "no validation queries" if query_count == 0
                else "low" if query_count < 30
                else "moderate" if query_count < 100
                else "higher"
            ),
        }
        for method in METHODS:
            for metric in BOOTSTRAP_METRICS:
                values = query_metric_values[method][metric][included_queries]
                row[f"{method}_{metric}"] = float(values.mean()) if len(values) else None
                if method == "fusion":
                    low, high = bootstrap_interval(values, rng)
                    row[f"fusion_{metric}_ci95_low"] = low
                    row[f"fusion_{metric}_ci95_high"] = high
        per_state_rows.append(row)

    category_state_indices = [
        [index for index, category_index in enumerate(state_category_indices)
         if category_index == category]
        for category in range(12)
    ]
    category_rows = category_correlation_rows(
        category_names,
        category_state_indices,
        train_state_counts,
        len(train_labels),
        val_query_state_counts,
        query_metric_values["fusion"],
    )
    confusion_rows = black_red_confusion_rows(
        mapper, gallery_labels52, groups, method_probabilities,
    )

    overall_metrics = {
        method: {
            metric: float(values.mean())
            for metric, values in query_metric_values[method].items()
        }
        for method in METHODS
    }
    prior_summary = load_prior_summary(args.prior_summary)
    prior_comparison: dict[str, Any] | None = None
    if prior_summary:
        prior_rows = {
            row["score_type"]: row for row in prior_summary.get("retrieval_metrics", [])
        }
        prior_comparison = {}
        for method in METHODS:
            if method not in prior_rows:
                continue
            prior_comparison[method] = {
                metric: {
                    "prior_full_val": float(prior_rows[method][metric]),
                    "recomputed_full_val": overall_metrics[method][metric],
                    "absolute_difference": abs(
                        float(prior_rows[method][metric]) - overall_metrics[method][metric]
                    ),
                }
                for metric in ("rank1", "mAP", "mADM")
            }

    checkpoint_stat = checkpoint.stat()
    summary = {
        "diagnostic": "per-state conditional retrieval and training frequency audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": socket.gethostname(),
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint_stat.st_size,
            "mtime_utc": datetime.fromtimestamp(checkpoint_stat.st_mtime, timezone.utc).isoformat(),
            "experiment": cache_metadata.get("experiment"),
            "epoch": cache_metadata.get("epoch"),
            "model_stage": cache_metadata.get("model_stage"),
            "architecture": cache_metadata.get("architecture"),
            "model_name": cache_metadata.get("model_name"),
        },
        "split": "val",
        "train_annotation_frequency_only": str(train_path.resolve()),
        "validation_annotation": str(validation_annotation),
        "data_root": str(data_root),
        "train_images": len(train_labels),
        "validation_images": len(gallery_labels),
        "validation_queries": len(queries),
        "gallery_id_alignment_verified": True,
        "exact_label_positive_definition_verified": True,
        "retrieval_scoring": "Native52 Category-NLL with checkpoint category temperature",
        "retrieval_positive_definition": "exact match of all 40 attribute bits, verified against validation ids.csv",
        "conditional_query_definition": "query's mapped 52-state target contains the state; candidate gallery remains the full validation split",
        "state_count": len(mapper.keys),
        "category_count": len(groups),
        "category_temperature": category_temperature,
        "fusion_alpha_local_weight": 0.5,
        "score_cache": str(cache_path),
        "score_cache_reused": bool(inference.get("score_cache_reused", False)),
        "score_inference": inference,
        "retrieval_metric_seconds": time.perf_counter() - started,
        "overall_metrics_recomputed_from_per_query_values": overall_metrics,
        "prior_full_validation_metric_comparison": prior_comparison,
        "category_frequency_metric_correlations": category_rows,
        "correlation_caveat": "Exploratory Spearman rank correlations within each category; states are few, query groups overlap, and correlations do not show that frequency causes retrieval performance.",
        "confidence_intervals": {
            "method": "percentile bootstrap over validation queries within each state",
            "replicates": BOOTSTRAP_REPLICATES,
            "reported_for": "fusion Rank-1, mAP, and mADM only",
            "caveat": "Intervals reflect query sampling variation only; they do not include checkpoint, annotation, or multiple-comparison uncertainty. Estimates across state values are dependent because query groups overlap.",
        },
        "support_labels": {
            "low": "1-29 unique validation queries",
            "moderate": "30-99 unique validation queries",
            "higher": "100 or more unique validation queries",
            "no validation queries": "conditional retrieval metric is undefined",
        },
        "train_data_usage": "train gt.csv is used only to count mapped state occurrences; model inference and retrieval rankings use validation images only.",
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_state_retrieval.csv", per_state_rows)
    write_csv(output_dir / "category_frequency_correlation.csv", category_rows)
    write_csv(output_dir / "black_red_confusion.csv", confusion_rows)

    print(f"Checkpoint: {checkpoint}")
    print(
        f"Train images={len(train_labels)} val images={len(gallery_labels)} "
        f"queries={len(queries)} states={len(mapper.keys)} categories={len(groups)}"
    )
    print(f"Score cache reused={inference.get('score_cache_reused', False)} path={cache_path}")
    for method, values in overall_metrics.items():
        print(
            f"{method}: Rank1={values['rank1']:.6f} mAP={values['mAP']:.6f} "
            f"mADM={values['mADM']:.6f}"
        )
    if prior_comparison:
        max_difference = max(
            detail[metric]["absolute_difference"]
            for detail in prior_comparison.values()
            for metric in ("rank1", "mAP", "mADM")
        )
        print(f"Maximum difference from prior full-val summary: {max_difference:.8g}")
    print(f"Wrote per-state audit: {output_dir}")


if __name__ == "__main__":
    main()
