#!/usr/bin/env python3
"""Measure L/14 category-query attention and validation behavior.

This is an inference-only diagnostic over the official validation split. It
collects shared CLIP global/local scores and attention in one pass, then uses
the repository's Native52 Category-NLL evaluator for global, local, and fused
retrieval metrics. No thresholds or model parameters are fitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

TOOLS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = TOOLS_DIR.parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_fromscratch_jointfce import (  # noqa: E402
    evaluate_score_arrays,
    fused_category_probabilities,
    retrieval_views,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.shared_clip_l14.experiment import _load_shared_model  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import ImagePathDataset  # noqa: E402
from upar.retrieval import autocast  # noqa: E402


FREQUENCY_BINS = (
    ("very_rare_lt_1pct", 0.0, 0.01),
    ("rare_1_to_5pct", 0.01, 0.05),
    ("medium_5_to_20pct", 0.05, 0.20),
    ("common_ge_20pct", 0.20, 1.000001),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.progress_every < 0:
        parser.error("batch size must be positive; workers and progress interval cannot be negative")
    return args


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _frequency_bin(prevalence: float) -> str:
    for name, lower, upper in FREQUENCY_BINS:
        if lower <= prevalence < upper:
            return name
    raise ValueError(f"No frequency bin for prevalence {prevalence}")


def _binary_auc_ap(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")

    # Mann-Whitney AUC with average ranks for ties.
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    sorted_target = target[order]
    rank_sum = 0.0
    index = 0
    while index < len(order):
        stop = index + 1
        while stop < len(order) and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        rank_sum += average_rank * int(sorted_target[index:stop].sum())
        index = stop
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    # Non-interpolated average precision, accumulating each tied score as one
    # threshold so the result does not depend on gallery order within a tie.
    order = np.argsort(-score, kind="mergesort")
    sorted_scores = score[order]
    sorted_target = target[order]
    true_positives = 0
    seen = 0
    ap = 0.0
    index = 0
    while index < len(order):
        stop = index + 1
        while stop < len(order) and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        group_positives = int(sorted_target[index:stop].sum())
        seen += stop - index
        true_positives += group_positives
        if group_positives:
            ap += (group_positives / positives) * (true_positives / seen)
        index = stop
    return float(auc), float(ap)


def _attribute_state_map(mapper: CategoryPromptMapper) -> tuple[dict[str, int], dict[str, int]]:
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    positive_state: dict[str, int] = {}
    attribute_category: dict[str, int] = {}
    category_index = 0
    for _, columns, state_keys, _ in mapper._MULTI_GROUPS:
        for column, key in zip(columns, state_keys):
            positive_state[column] = state_index[key]
            attribute_category[column] = category_index
        category_index += 1
    for column, positive_key, _ in mapper._BINARY_GROUPS:
        positive_state[column] = state_index[positive_key]
        attribute_category[column] = category_index
        category_index += 1
    if set(positive_state) != set(mapper.attribute_names):
        raise RuntimeError("The 40 positive attribute states do not match the mapper vocabulary")
    return positive_state, attribute_category


def _attention_accumulators(category_count: int) -> dict[str, Any]:
    return {
        "images": 0,
        "entropy": np.zeros(category_count, dtype=np.float64),
        "head_entropy": np.zeros(category_count, dtype=np.float64),
        "effective_patches": np.zeros(category_count, dtype=np.float64),
        "max_mass": np.zeros(category_count, dtype=np.float64),
        "map_cosine": np.zeros((category_count, category_count), dtype=np.float64),
        "head_cosine": np.zeros((category_count, category_count), dtype=np.float64),
        "topk_jaccard": np.zeros((category_count, category_count), dtype=np.float64),
        "feature_cosine": np.zeros((category_count, category_count), dtype=np.float64),
    }


def _update_attention_stats(
    accum: dict[str, Any],
    attention: torch.Tensor,
    local_features: torch.Tensor,
) -> None:
    # attention [B,H,C,P], with every head's spatial weights summing to one.
    values = attention.float().clamp_min(0.0)
    batch, heads, categories, patches = values.shape
    map_values = values.mean(dim=1)
    tiny = torch.finfo(map_values.dtype).tiny
    map_entropy = -(map_values * map_values.clamp_min(tiny).log()).sum(dim=-1)
    head_entropy = -(values * values.clamp_min(tiny).log()).sum(dim=-1).mean(dim=1)
    effective = map_entropy.exp()
    max_mass = map_values.max(dim=-1).values

    map_normalized = F.normalize(map_values, p=2, dim=-1)
    map_cosine = map_normalized @ map_normalized.transpose(1, 2)
    head_normalized = F.normalize(values, p=2, dim=-1)
    head_cosine = torch.einsum("bhcp,bhdp->bhcd", head_normalized, head_normalized).mean(dim=1)

    top_count = max(1, round(patches * 0.10))
    top_indices = map_values.topk(top_count, dim=-1).indices
    top_mask = torch.zeros((batch, categories, patches), dtype=torch.bool, device=values.device)
    top_mask.scatter_(2, top_indices, True)
    intersection = (top_mask.unsqueeze(2) & top_mask.unsqueeze(1)).sum(dim=-1).float()
    topk_jaccard = intersection / (2 * top_count - intersection).clamp_min(1.0)

    feature_normalized = F.normalize(local_features.float(), p=2, dim=-1)
    feature_cosine = feature_normalized @ feature_normalized.transpose(1, 2)

    accum["images"] += batch
    accum["entropy"] += map_entropy.sum(dim=0).double().cpu().numpy()
    accum["head_entropy"] += head_entropy.sum(dim=0).double().cpu().numpy()
    accum["effective_patches"] += effective.sum(dim=0).double().cpu().numpy()
    accum["max_mass"] += max_mass.sum(dim=0).double().cpu().numpy()
    accum["map_cosine"] += map_cosine.sum(dim=0).double().cpu().numpy()
    accum["head_cosine"] += head_cosine.sum(dim=0).double().cpu().numpy()
    accum["topk_jaccard"] += topk_jaccard.sum(dim=0).double().cpu().numpy()
    accum["feature_cosine"] += feature_cosine.sum(dim=0).double().cpu().numpy()


def _attention_rows(
    accum: dict[str, Any], category_names: Sequence[str], patch_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    count = max(int(accum["images"]), 1)
    category_count = len(category_names)
    map_cosine = accum["map_cosine"] / count
    head_cosine = accum["head_cosine"] / count
    jaccard = accum["topk_jaccard"] / count
    feature_cosine = accum["feature_cosine"] / count
    rows = []
    for index, name in enumerate(category_names):
        others = [other for other in range(category_count) if other != index]
        mean_entropy = float(accum["entropy"][index] / count)
        rows.append({
            "category": name,
            "attention_entropy": mean_entropy,
            "normalized_attention_entropy": mean_entropy / float(np.log(patch_count)),
            "effective_patches": float(accum["effective_patches"][index] / count),
            "max_patch_mass": float(accum["max_mass"][index] / count),
            "mean_head_entropy": float(accum["head_entropy"][index] / count),
            "mean_pairwise_attention_cosine": float(map_cosine[index, others].mean()),
            "mean_pairwise_top10pct_jaccard": float(jaccard[index, others].mean()),
            "mean_pairwise_local_feature_cosine": float(feature_cosine[index, others].mean()),
        })
    pairs = []
    for left in range(category_count):
        for right in range(left + 1, category_count):
            pairs.append({
                "category_a": category_names[left],
                "category_b": category_names[right],
                "attention_map_cosine": float(map_cosine[left, right]),
                "attention_head_mean_cosine": float(head_cosine[left, right]),
                "attention_top10pct_jaccard": float(jaccard[left, right]),
                "local_feature_cosine": float(feature_cosine[left, right]),
            })
    upper = np.triu_indices(category_count, k=1)
    summary = {
        "images": int(accum["images"]),
        "patch_count": int(patch_count),
        "heads": None,
        "top_patch_fraction": 0.10,
        "uniform_attention_reference": {
            "entropy": float(np.log(patch_count)),
            "effective_patches": int(patch_count),
            "max_patch_mass": 1.0 / patch_count,
        },
        "mean_pairwise_attention_map_cosine": float(map_cosine[upper].mean()),
        "mean_pairwise_attention_head_cosine": float(head_cosine[upper].mean()),
        "mean_pairwise_attention_top10pct_jaccard": float(jaccard[upper].mean()),
        "mean_pairwise_local_feature_cosine": float(feature_cosine[upper].mean()),
    }
    return rows, pairs, summary


def _attribute_diagnostics(
    mapper: CategoryPromptMapper,
    labels: np.ndarray,
    global_probabilities: np.ndarray,
    local_probabilities: np.ndarray,
    fusion_probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive_state, attribute_category = _attribute_state_map(mapper)
    category_labels = [entry[0] for entry in mapper._MULTI_GROUPS]
    category_labels.extend(entry[0] for entry in mapper._BINARY_GROUPS)
    category_groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    method_probs = {
        "global": global_probabilities,
        "local": local_probabilities,
        "fusion": fusion_probabilities,
    }
    rows: list[dict[str, Any]] = []
    bin_scores: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for attribute_index, attribute in enumerate(mapper.attribute_names):
        target = labels[:, attribute_index] > 0.5
        support = int(target.sum())
        prevalence = float(target.mean())
        frequency_bin = _frequency_bin(prevalence)
        row: dict[str, Any] = {
            "attribute": attribute,
            "category": category_labels[attribute_category[attribute]],
            "positive_count": support,
            "negative_count": int(len(target) - support),
            "positive_prevalence": prevalence,
            "frequency_bin": frequency_bin,
        }
        for method, probabilities in method_probs.items():
            positive_scores = probabilities[:, positive_state[attribute]]
            auc, average_precision = _binary_auc_ap(target, positive_scores)
            indices = category_groups[attribute_category[attribute]]
            prediction = indices[np.argmax(probabilities[:, indices], axis=1)] == positive_state[attribute]
            accuracy = float(np.mean(prediction == target))
            row[f"{method}_positive_state_auc"] = auc
            row[f"{method}_positive_state_ap"] = average_precision
            row[f"{method}_state_accuracy"] = accuracy
            if np.isfinite(auc) and np.isfinite(average_precision):
                bin_scores[frequency_bin][method].append({
                    "auc": auc, "ap": average_precision, "accuracy": accuracy,
                })
        rows.append(row)

    bins = []
    for name, lower, upper in FREQUENCY_BINS:
        selected = [row for row in rows if row["frequency_bin"] == name]
        result: dict[str, Any] = {
            "frequency_bin": name,
            "lower_prevalence_inclusive": lower,
            "upper_prevalence_exclusive": upper,
            "attribute_count": len(selected),
            "attributes": [row["attribute"] for row in selected],
            "mean_positive_prevalence": (
                float(np.mean([row["positive_prevalence"] for row in selected]))
                if selected else None
            ),
        }
        for method in method_probs:
            metrics = bin_scores[name][method]
            for metric in ("auc", "ap", "accuracy"):
                result[f"{method}_macro_{metric}"] = (
                    float(np.mean([item[metric] for item in metrics])) if metrics else None
                )
        bins.append(result)
    return rows, bins


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

    device = choose_device(args.device)
    use_amp = not args.disable_amp and device.type == "cuda"
    gt_path, table, queries_raw, ids, query_names = retrieval_views(data_root, "val", None)
    model, payload = _load_shared_model(checkpoint, device)
    model.eval()
    attribute_names = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attribute_names)
    category_names = [entry[0] for entry in mapper._MULTI_GROUPS]
    category_names.extend(entry[0] for entry in mapper._BINARY_GROUPS)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    category_temperature = float(payload.get("training_config", {}).get("category_temperature", 0.01))

    state_tokens = model.base.tokenize(mapper.prompts).to(device)
    with torch.inference_mode(), autocast(device, use_amp):
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
    attention_stats = _attention_accumulators(len(category_names))
    started = time.perf_counter()
    for batch_index, images in enumerate(loader, start=1):
        with torch.inference_mode(), autocast(device, use_amp):
            global_features, local_features, _, attention = model.joint_features_with_tokens(
                images.to(device, non_blocking=True),
            )
            global_scores = global_features @ state_features.T
            local_scores = model.category_state_scores_from_features(local_features, state_features)
        global_chunks.append(global_scores.float().cpu().numpy())
        local_chunks.append(local_scores.float().cpu().numpy())
        _update_attention_stats(attention_stats, attention, local_features)
        if args.progress_every and (
            batch_index % args.progress_every == 0 or batch_index == len(loader)
        ):
            processed = min(batch_index * args.batch_size, len(table.image_paths))
            elapsed = time.perf_counter() - started
            print(
                f"[query-distribution] batch {batch_index}/{len(loader)} "
                f"images {processed}/{len(table.image_paths)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    global_scores = np.concatenate(global_chunks, axis=0)
    local_scores = np.concatenate(local_chunks, axis=0)
    if global_scores.shape != (len(table.image_paths), 52):
        raise RuntimeError(f"Unexpected global score shape {global_scores.shape}")
    if local_scores.shape != (len(table.image_paths), 12, 52):
        raise RuntimeError(f"Unexpected local score shape {local_scores.shape}")

    rows, details = evaluate_score_arrays(
        global_scores, local_scores, table, queries_raw, ids, query_names,
        attribute_names, category_temperature, alphas=(0.5,),
    )
    reordered_labels = details["labels"]
    reordered_queries = details["queries"]
    reordered_ids = details["ids"]
    if not np.array_equal(reordered_labels, reordered_queries[reordered_ids]):
        raise RuntimeError("Validation query IDs do not reproduce the aligned 40-D labels")
    _, _, fusion_probabilities = fused_category_probabilities(
        global_scores, local_scores, groups, category_temperature, 0.5,
    )

    attention_rows, attention_pairs, attention_summary = _attention_rows(
        attention_stats, category_names, int(model.num_patch_tokens),
    )
    attention_summary["heads"] = int(model.num_heads)
    attribute_rows, frequency_rows = _attribute_diagnostics(
        mapper,
        reordered_labels,
        details["global_probabilities"],
        details["local_probabilities"],
        fusion_probabilities,
    )
    query_attribute_counts = reordered_queries.sum(axis=1).astype(np.int32)
    query_support = np.bincount(reordered_ids, minlength=len(reordered_queries))
    query_histogram = Counter(int(count) for count in query_attribute_counts)
    query_rows = [
        {
            "query_index": index,
            "positive_attribute_count": int(query_attribute_counts[index]),
            "gallery_images": int(query_support[index]),
        }
        for index in range(len(reordered_queries))
    ]

    elapsed = time.perf_counter() - started
    checkpoint_stat = checkpoint.stat()
    metric_by_type = {
        (str(row["score_type"]), float(row["alpha"]) if row["alpha"] != "" else None): row
        for row in rows
    }
    global_row = metric_by_type[("global", None)]
    local_row = metric_by_type[("local", None)]
    fusion_row = metric_by_type[("fusion", 0.5)]
    benefit = {}
    for key in ("rank1", "rank5", "rank10", "mAP", "mADM", "mINP", "semantic_top1"):
        benefit[key] = {
            "local_minus_global": float(local_row[key] - global_row[key]),
            "fusion_minus_global": float(fusion_row[key] - global_row[key]),
            "fusion_minus_local": float(fusion_row[key] - local_row[key]),
        }

    summary = {
        "diagnostic": "shared CLIP ViT-L/14 attribute query distribution",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": socket.gethostname(),
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint_stat.st_size,
            "mtime_utc": datetime.fromtimestamp(checkpoint_stat.st_mtime, timezone.utc).isoformat(),
            "experiment": payload.get("experiment"),
            "epoch": payload.get("epoch"),
            "model_stage": payload.get("model_stage"),
            "architecture": payload.get("architecture"),
            "model_name": payload.get("model_name"),
        },
        "split": "val",
        "validation_annotation": str(gt_path.resolve()),
        "data_root": str(data_root),
        "images": len(table.image_paths),
        "queries": len(reordered_queries),
        "gallery_id_count": int(len(reordered_ids)),
        "query_id_alignment_verified": True,
        "evaluation_protocol": "native52_category_nll",
        "category_temperature": category_temperature,
        "fusion_alpha_local_weight": 0.5,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amp": use_amp,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "inference_seconds": elapsed,
        "no_training_samples_or_parameters_used": True,
        "retrieval_metrics": rows,
        "local_global_fusion_metric_deltas": benefit,
        "query_positive_attribute_count": {
            "min": int(query_attribute_counts.min()),
            "max": int(query_attribute_counts.max()),
            "mean": float(query_attribute_counts.mean()),
            "median": float(np.median(query_attribute_counts)),
            "histogram": {str(key): query_histogram[key] for key in sorted(query_histogram)},
        },
        "attention_distribution": attention_summary,
        "validation_frequency_bin_definition": [
            {"name": name, "lower_prevalence_inclusive": low, "upper_prevalence_exclusive": high}
            for name, low, high in FREQUENCY_BINS
        ],
        "frequency_bin_aggregates": frequency_rows,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "attention_by_category.csv", attention_rows)
    _write_csv(output_dir / "attention_pairwise.csv", attention_pairs)
    _write_csv(output_dir / "attribute_frequency_and_scores.csv", attribute_rows)
    _write_csv(output_dir / "attribute_frequency_bins.csv", frequency_rows)
    _write_csv(output_dir / "query_attribute_counts.csv", query_rows)

    print(f"Checkpoint: {checkpoint} ({payload.get('experiment')} epoch={payload.get('epoch')})")
    print(f"Validation: {gt_path} images={len(table.image_paths)} queries={len(reordered_queries)}")
    print(f"Runtime: {device} {summary['device_name']} elapsed={elapsed:.1f}s")
    for row in rows:
        print(
            f"{row['score_type']} alpha={row['alpha']}: "
            f"R1={row['rank1']:.6f} mAP={row['mAP']:.6f} mADM={row['mADM']:.6f} "
            f"Rank5={row['rank5']:.6f} Rank10={row['rank10']:.6f}",
        )
    print(
        "Attention overlap: map-cosine="
        f"{attention_summary['mean_pairwise_attention_map_cosine']:.6f}, "
        f"top10%-Jaccard={attention_summary['mean_pairwise_attention_top10pct_jaccard']:.6f}, "
        f"local-feature-cosine={attention_summary['mean_pairwise_local_feature_cosine']:.6f}"
    )
    print(f"Wrote diagnostics: {output_dir}")


if __name__ == "__main__":
    main()
