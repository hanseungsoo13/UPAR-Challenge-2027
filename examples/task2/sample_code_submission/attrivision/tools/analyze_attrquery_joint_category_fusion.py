"""Inference-only Global/Local and category-wise fusion diagnostics.

The script uses one AttrQuery-FromScratchJointFCE checkpoint for every view:
the global CLIP representation and the category-local cross-attention
representation are produced by the same model and the same image forward.
No optimizer, gradient, or model-weight update is performed.

Category-wise fusion is selected on a deterministic query split and reported
on a disjoint query split.  The gallery stays common, as required by the
UPAR retrieval protocol.  The full validation result is also written only as
a reference, never as the held-out selection result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

TOOLS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = TOOLS_DIR.parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.attrquery_fromscratch_jointfce import category_nll  # noqa: E402
from attrivision.models.attrquery_fromscratch_jointfce import (  # noqa: E402
    category_names,
    load_experiment_model,
    validate_mapper,
)
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import (  # noqa: E402
    AnnotationTable,
    ImagePathDataset,
    find_annotation_file,
    read_gt_csv,
)
from upar.retrieval import (  # noqa: E402
    autocast,
    load_retrieval_annotations,
    official_retrieval_metrics,
    reorder_columns,
)


DEFAULT_CHECKPOINT = (
    "outputs/AttrQuery-FromScratchJointFCE/JointFCE-Lambda050/checkpoint_best.pth"
)
DEFAULT_OUTPUT = "outputs/AttrQuery-JointCategoryFusion-Diagnostics"
DEFAULT_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_is_valid(
    cache_path: Path,
    manifest_path: Path,
    checkpoint: Path,
    image_paths: Sequence[str],
    global_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> bool:
    if not cache_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(cache_path, allow_pickle=False) as archive:
            global_scores = archive["global_scores"]
            local_scores = archive["local_scores"]
        return bool(
            manifest.get("checkpoint") == _checkpoint_fingerprint(checkpoint)
            and manifest.get("image_paths") == list(image_paths)
            and tuple(global_scores.shape) == global_shape
            and tuple(local_scores.shape) == local_shape
            and np.isfinite(global_scores).all()
            and np.isfinite(local_scores).all()
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _save_score_cache(
    path: Path,
    global_scores: np.ndarray,
    local_scores: np.ndarray,
    checkpoint: Path,
    image_paths: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            global_scores=global_scores.astype(np.float32, copy=False),
            local_scores=local_scores.astype(np.float32, copy=False),
        )
    temporary.replace(path)
    _write_json(path.with_suffix(path.suffix + ".json"), {
        "checkpoint": _checkpoint_fingerprint(checkpoint),
        "image_paths": list(image_paths),
        "global_shape": list(global_scores.shape),
        "local_shape": list(local_scores.shape),
        "score_definition": "shared joint_features; normalized image/text cosine scores",
    })


def _load_score_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["global_scores"].astype(np.float32, copy=False),
            archive["local_scores"].astype(np.float32, copy=False),
        )


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def _load_validation_views(
    data_root: Path,
    max_val_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if max_val_samples is not None:
        count = min(int(max_val_samples), len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    return gt_path, table, queries, ids, query_names


def _category_log_probabilities(
    scores: np.ndarray,
    groups: Sequence[np.ndarray],
    temperature: float,
    local: bool,
) -> np.ndarray:
    """Return [gallery, category, 52] category-normalized log probabilities."""
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    if local:
        if scores.ndim != 3:
            raise ValueError(f"Local scores must be [G,C,52], got {scores.shape}")
        category_scores = scores
    else:
        if scores.ndim != 2:
            raise ValueError(f"Global scores must be [G,52], got {scores.shape}")
        category_scores = np.broadcast_to(
            scores[:, None, :], (len(scores), len(groups), scores.shape[1]),
        )
    if category_scores.shape[1:] != (len(groups), 52):
        raise ValueError(
            f"Score shape {category_scores.shape} does not match {len(groups)} categories and 52 states"
        )
    result = np.full(category_scores.shape, -np.inf, dtype=np.float64)
    for category_index, indices in enumerate(groups):
        values = category_scores[:, category_index, indices].astype(np.float64) / temperature
        maximum = values.max(axis=1, keepdims=True)
        shifted = values - maximum
        result[:, category_index, indices] = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True),
        )
    # States outside a category are intentionally left at -inf.  Validate
    # only the state block that belongs to each category.
    for category_index, indices in enumerate(groups):
        if not np.isfinite(result[:, category_index, indices]).all():
            raise RuntimeError(
                f"Category {category_index} log probabilities contain NaN or Inf"
            )
    return result.astype(np.float32)


def _query_category_targets(
    queries52: np.ndarray,
    groups: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-query category targets and a valid-category mask."""
    target = np.zeros((len(queries52), len(groups), 52), dtype=np.float32)
    valid = np.zeros((len(queries52), len(groups)), dtype=bool)
    for category_index, indices in enumerate(groups):
        values = queries52[:, indices].astype(np.float32)
        counts = values.sum(axis=1, keepdims=True)
        valid[:, category_index] = counts[:, 0] > 0
        safe_counts = np.maximum(counts, 1.0)
        target[:, category_index, indices] = values / safe_counts
    if not valid.any(axis=1).all():
        raise ValueError("At least one query has no active category")
    return target, valid


def _full_category_probabilities(
    category_logs: np.ndarray,
    groups: Sequence[np.ndarray],
) -> np.ndarray:
    probabilities = np.zeros((category_logs.shape[0], 52), dtype=np.float32)
    for category_index, indices in enumerate(groups):
        probabilities[:, indices] = np.exp(category_logs[:, category_index, indices])
    return probabilities


def _category_contribution(
    global_logs: np.ndarray,
    local_logs: np.ndarray,
    query_targets: np.ndarray,
    category_index: int,
    indices: np.ndarray,
    alpha: float,
    query_indices: np.ndarray,
) -> np.ndarray:
    """Negative target log probability for one category, shape [Q,G]."""
    blended = (
        (1.0 - float(alpha)) * global_logs[:, category_index, indices]
        + float(alpha) * local_logs[:, category_index, indices]
    ).astype(np.float64)
    maximum = blended.max(axis=1, keepdims=True)
    blended -= maximum
    blended -= np.log(np.exp(blended).sum(axis=1, keepdims=True))
    targets = query_targets[query_indices, category_index, indices].astype(np.float64)
    return -(targets @ blended.T)


def _fused_distances(
    global_logs: np.ndarray,
    local_logs: np.ndarray,
    query_targets: np.ndarray,
    valid_categories: np.ndarray,
    groups: Sequence[np.ndarray],
    alphas: Sequence[float],
    query_indices: np.ndarray,
) -> np.ndarray:
    if len(alphas) != len(groups):
        raise ValueError("One fusion alpha is required per category")
    total = np.zeros((len(query_indices), global_logs.shape[0]), dtype=np.float64)
    for category_index, indices in enumerate(groups):
        contribution = _category_contribution(
            global_logs, local_logs, query_targets, category_index,
            indices, float(alphas[category_index]), query_indices,
        )
        total += contribution
    denominator = valid_categories[query_indices].sum(axis=1).astype(np.float64)
    return (total / denominator[:, None]).astype(np.float32)


def _metrics(
    distances: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
) -> dict[str, float]:
    result = official_retrieval_metrics(distances, queries, labels, ids)
    return {
        "rank1": float(result["Rank-1"]),
        "rank5": float(result["Rank-5"]),
        "rank10": float(result["Rank-10"]),
        "mAP": float(result["mAP"]),
        "mADM": float(result["mADM"]),
        "mINP": float(result["mINP"]),
        "semantic_top1": float(result["semantic_top1"]),
    }


def _evaluate_query_subset(
    distances: np.ndarray,
    query_indices: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
) -> dict[str, float]:
    return _metrics(distances, queries[query_indices], labels, ids)


def _best_row(rows: Sequence[dict[str, Any]], metric: str = "mADM") -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select a best result from an empty list")
    return max(rows, key=lambda row: (float(row[metric]), -float(row.get("alpha", 0.0))))


def _uniform_sweep(
    global_logs: np.ndarray,
    local_logs: np.ndarray,
    query_targets: np.ndarray,
    valid_categories: np.ndarray,
    groups: Sequence[np.ndarray],
    query_indices: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    alphas: Sequence[float],
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        distances = _fused_distances(
            global_logs, local_logs, query_targets, valid_categories, groups,
            [float(alpha)] * len(groups), query_indices,
        )
        metrics = _evaluate_query_subset(distances, query_indices, queries, labels, ids)
        row = {
            "scope": scope,
            "method": "uniform_fusion",
            "selection_source": "grid",
            "alpha": float(alpha),
            "category_alphas": "",
            "query_count": int(len(query_indices)),
            **metrics,
        }
        rows.append(row)
    return rows


def _coordinate_descent(
    global_logs: np.ndarray,
    local_logs: np.ndarray,
    query_targets: np.ndarray,
    valid_categories: np.ndarray,
    groups: Sequence[np.ndarray],
    query_indices: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    grid: Sequence[float],
    passes: int,
    initial_alpha: float,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Coordinate-wise grid optimization of the 12 independent alpha values."""
    if not grid or any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("Category alpha grid must be within [0,1]")
    weights = np.full(len(groups), float(initial_alpha), dtype=np.float64)
    current_sum = np.zeros((len(query_indices), global_logs.shape[0]), dtype=np.float64)
    current_contributions: list[np.ndarray] = []
    for category_index, indices in enumerate(groups):
        contribution = _category_contribution(
            global_logs, local_logs, query_targets, category_index,
            indices, weights[category_index], query_indices,
        )
        current_contributions.append(contribution)
        current_sum += contribution
    denominator = valid_categories[query_indices].sum(axis=1).astype(np.float64)
    current_distances = (current_sum / denominator[:, None]).astype(np.float32)
    current_score = _evaluate_query_subset(
        current_distances, query_indices, queries, labels, ids,
    )["mADM"]
    history: list[dict[str, Any]] = [{
        "pass": 0,
        "category_index": -1,
        "category": "all",
        "selected_alpha": float(initial_alpha),
        "mADM": float(current_score),
    }]
    for pass_index in range(1, passes + 1):
        changed = False
        for category_index, indices in enumerate(groups):
            base_sum = current_sum - current_contributions[category_index]
            candidates: list[tuple[float, float, np.ndarray]] = []
            for candidate in grid:
                contribution = _category_contribution(
                    global_logs, local_logs, query_targets, category_index,
                    indices, float(candidate), query_indices,
                )
                distances = ((base_sum + contribution) / denominator[:, None]).astype(np.float32)
                score = _evaluate_query_subset(
                    distances, query_indices, queries, labels, ids,
                )["mADM"]
                candidates.append((float(score), -float(candidate), contribution))
            selected_score, negative_alpha, selected_contribution = max(candidates, key=lambda item: (item[0], item[1]))
            selected_alpha = -negative_alpha
            if abs(selected_alpha - weights[category_index]) > 1e-12:
                changed = True
            weights[category_index] = selected_alpha
            current_contributions[category_index] = selected_contribution
            current_sum = base_sum + selected_contribution
            current_score = selected_score
            history.append({
                "pass": pass_index,
                "category_index": category_index,
                "category": f"category_{category_index:02d}",
                "selected_alpha": selected_alpha,
                "mADM": float(current_score),
            })
        if not changed:
            break
    return weights, float(current_score), history
def _query_stats(
    distances: np.ndarray,
    query_indices: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_query_index, query_index in enumerate(query_indices):
        query = queries[query_index]
        relevant = np.all(labels == query, axis=1)
        if not relevant.any():
            raise ValueError(f"Query {query_index} has no exact gallery match")
        order = np.argsort(distances[local_query_index], kind="stable")
        relevant_positions = np.flatnonzero(relevant[order])
        ranks = relevant_positions + 1
        top1_index = int(order[0])
        best_index = int(order[relevant_positions[0]])
        ap = float(np.mean(np.arange(1, len(ranks) + 1) / ranks))
        rows.append({
            "query_index": int(query_index),
            "top1_gallery_index": top1_index,
            "best_correct_gallery_index": best_index,
            "best_correct_rank": int(ranks[0]),
            "average_precision": ap,
            "top1_correct": bool(relevant[top1_index]),
            "relevant_gallery_count": int(relevant.sum()),
        })
    return rows


def _transition(old: bool, new: bool) -> str:
    if old and new:
        return "correct_to_correct"
    if not old and new:
        return "wrong_to_correct"
    if old and not new:
        return "correct_to_wrong"
    return "wrong_to_wrong"


def _ece(
    confidence: np.ndarray,
    correctness: np.ndarray,
    bin_count: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    total = max(len(confidence), 1)
    result = 0.0
    for index in range(bin_count):
        if index == bin_count - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correctness[mask].mean()),
            )
    return result


def _category_comparison_rows(
    global_logs: np.ndarray,
    local_logs: np.ndarray,
    target52: np.ndarray,
    groups: Sequence[np.ndarray],
    names: Sequence[str],
    ece_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category_index, (name, indices) in enumerate(zip(names, groups)):
        target = target52[:, indices].argmax(axis=1)
        global_probabilities = np.exp(global_logs[:, category_index, indices].astype(np.float64))
        local_probabilities = np.exp(local_logs[:, category_index, indices].astype(np.float64))
        global_prediction = global_probabilities.argmax(axis=1)
        local_prediction = local_probabilities.argmax(axis=1)
        global_correct = global_prediction == target
        local_correct = local_prediction == target
        global_confidence = global_probabilities.max(axis=1)
        local_confidence = local_probabilities.max(axis=1)
        rows.append({
            "category_index": category_index,
            "category": name,
            "state_count": len(indices),
            "gallery_images": len(target52),
            "global_nll": float(-np.mean(np.log(np.maximum(global_probabilities[np.arange(len(target)), target], 1e-12)))),
            "local_nll": float(-np.mean(np.log(np.maximum(local_probabilities[np.arange(len(target)), target], 1e-12)))),
            "nll_improvement_global_minus_local": float(
                -np.mean(np.log(np.maximum(global_probabilities[np.arange(len(target)), target], 1e-12)))
                + np.mean(np.log(np.maximum(local_probabilities[np.arange(len(target)), target], 1e-12)))
            ),
            "global_accuracy": float(global_correct.mean()),
            "local_accuracy": float(local_correct.mean()),
            "accuracy_improvement_local_minus_global": float(local_correct.mean() - global_correct.mean()),
            "global_mean_confidence": float(global_confidence.mean()),
            "local_mean_confidence": float(local_confidence.mean()),
            "global_ece": _ece(global_confidence, global_correct.astype(np.float32), ece_bins),
            "local_ece": _ece(local_confidence, local_correct.astype(np.float32), ece_bins),
            "global_correct_local_correct": int(np.sum(global_correct & local_correct)),
            "global_correct_local_wrong": int(np.sum(global_correct & ~local_correct)),
            "global_wrong_local_correct": int(np.sum(~global_correct & local_correct)),
            "global_wrong_local_wrong": int(np.sum(~global_correct & ~local_correct)),
            "net_local_correction": int(np.sum(~global_correct & local_correct) - np.sum(global_correct & ~local_correct)),
        })
    return rows


def _plot_outputs(
    output_dir: Path,
    category_rows: Sequence[dict[str, Any]],
    fusion_rows: Sequence[dict[str, Any]],
    weight_rows: Sequence[dict[str, Any]],
    rank_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("This diagnostic requires matplotlib for its graphs") from exc

    names = [row["category"] for row in category_rows]
    x = np.arange(len(names))
    weights = [float(row["categorywise_alpha"] ) for row in weight_rows]
    nll_delta = [float(row["nll_improvement_global_minus_local"]) for row in category_rows]
    uniform_eval = next(
        row for row in fusion_rows
        if row["scope"] == "evaluation" and row["method"] == "uniform_fusion_tuning_selected"
    )
    category_eval = next(
        row for row in fusion_rows
        if row["scope"] == "evaluation" and row["method"] == "categorywise_fusion"
    )
    heldout_rank_rows = [row for row in rank_rows if row["split"] == "evaluation"]
    ap_delta = np.asarray([
        float(row["categorywise_ap_minus_global"]) for row in heldout_rank_rows
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(x, weights, color="#4472c4")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Category-wise local fusion weights")
    axes[0, 0].set_ylabel("alpha_c")
    axes[0, 1].bar(x, nll_delta, color=["#70ad47" if value > 0 else "#c0504d" for value in nll_delta])
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("NLL improvement: Global − Local")
    axes[0, 1].set_ylabel("positive = Local better")
    axes[1, 0].bar(
        ["Global", "Local", "Uniform", "Category-wise"],
        [
            float(next(row for row in fusion_rows if row["scope"] == "evaluation" and row["method"] == "global_only")["mADM"]),
            float(next(row for row in fusion_rows if row["scope"] == "evaluation" and row["method"] == "local_only")["mADM"]),
            float(uniform_eval["mADM"]),
            float(category_eval["mADM"]),
        ],
        color=["#7f7f7f", "#ed7d31", "#5b9bd5", "#70ad47"],
    )
    axes[1, 0].set_title("Held-out mADM")
    axes[1, 0].set_ylabel("mADM")
    axes[1, 1].hist(ap_delta, bins=25, color="#a5a5a5", edgecolor="white")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("Category-wise AP change vs Global")
    axes[1, 1].set_xlabel("AP change per query")
    for axis in axes[0, :]:
        axis.set_xticks(x, names, rotation=50, ha="right")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "category_fusion_analysis.png", dpi=180)
    plt.close(fig)

    sweep = sorted(
        [row for row in fusion_rows
         if row["scope"] == "evaluation" and row["method"] == "uniform_fusion"],
        key=lambda row: float(row["alpha"]),
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, label in (("rank1", "Rank-1"), ("mAP", "mAP"), ("mADM", "mADM"), ("mINP", "mINP")):
        ax.plot(
            [float(row["alpha"]) for row in sweep],
            [100 * float(row[metric]) for row in sweep],
            marker="o", label=label,
        )
    ax.set_xlabel("Uniform local weight alpha")
    ax.set_ylabel("held-out metric (%)")
    ax.set_xticks([float(row["alpha"]) for row in sweep])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "uniform_fusion_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - 0.2, [row["global_accuracy"] * 100 for row in category_rows], 0.2, label="Global")
    axes[0].bar(x, [row["local_accuracy"] * 100 for row in category_rows], 0.2, label="Local")
    axes[0].bar(x + 0.2, [row["global_ece"] * 100 for row in category_rows], 0.2, label="Global ECE")
    axes[0].bar(x + 0.4, [row["local_ece"] * 100 for row in category_rows], 0.2, label="Local ECE")
    axes[0].set_xticks(x, names, rotation=50, ha="right")
    axes[0].set_title("Category accuracy and calibration")
    axes[0].set_ylabel("percent")
    axes[0].legend(fontsize=8)
    axes[1].scatter(nll_delta, weights, color="#4472c4")
    for row, delta, weight in zip(category_rows, nll_delta, weights):
        axes[1].annotate(row["category"], (delta, weight), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("NLL improvement: Global − Local")
    axes[1].set_ylabel("category-wise alpha")
    axes[1].set_title("Local advantage vs learned local weight")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "category_calibration_and_weight.png", dpi=180)
    plt.close(fig)


def _write_summary(
    path: Path,
    metadata: dict[str, Any],
    fusion_rows: Sequence[dict[str, Any]],
    weight_rows: Sequence[dict[str, Any]],
    transition_summary: dict[str, Any],
    correlations: dict[str, Any],
) -> None:
    evaluation_rows = [row for row in fusion_rows if row["scope"] == "evaluation"]
    global_row = next(row for row in evaluation_rows if row["method"] == "global_only")
    local_row = next(row for row in evaluation_rows if row["method"] == "local_only")
    category_row = next(row for row in evaluation_rows if row["method"] == "categorywise_fusion")
    uniform_row = next(
        row for row in evaluation_rows if row["method"] == "uniform_fusion_tuning_selected"
    )
    table = "\n".join(
        f"| {row['category']} | {float(row['categorywise_alpha']):.2f} | "
        f"{float(row['global_nll']):.4f} | {float(row['local_nll']):.4f} | "
        f"{float(row['nll_improvement_global_minus_local']):.4f} |"
        for row in weight_rows
    )
    lines = [
        "# AttrQuery-JointCategoryFusion-Diagnostics",
        "",
        "Inference-only diagnostics from one AttrQuery-FromScratchJointFCE checkpoint.",
        "",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Validation annotation: `{metadata['validation_annotation']}`",
        f"- Gallery images: `{metadata['gallery_images']}`",
        f"- Queries: tuning `{metadata['tuning_queries']}`, evaluation `{metadata['evaluation_queries']}`",
        f"- Queries used to optimize alpha: `{metadata['optimization_queries']}` (deterministic subset of tuning)",
        f"- Category temperature: `{metadata['category_temperature']}`",
        f"- Category alpha grid: `{metadata['category_alpha_grid']}`",
        f"- Selection metric: `{metadata['selection_metric']}`",
        "",
        "## Protocol and leakage control",
        "",
        "Global and Local scores are generated by the same checkpoint and one shared image forward. "
        "The gallery is common to both query splits. Category-wise alpha is selected only on the tuning query split; "
        "the evaluation rows below use the disjoint held-out query split. Full-validation selection is marked reference-only.",
        "",
        "## Held-out evaluation",
        "",
        "| Method | Rank-1 | mAP | mADM | mINP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (global_row, local_row, uniform_row, category_row):
        lines.append(
            f"| {row['method']} "
            f"(alpha={row.get('alpha', '')}) | {100*float(row['rank1']):.3f}% | "
            f"{100*float(row['mAP']):.3f}% | {100*float(row['mADM']):.3f}% | "
            f"{100*float(row['mINP']):.3f}% |"
        )
    lines.extend([
        "",
        f"- Category-wise minus tuning-selected Uniform mADM: `{100*(float(category_row['mADM']) - float(uniform_row['mADM'])):.3f}` percentage points.",
        "- Category-wise minus Global mADM: `" + f"{100*(float(category_row['mADM']) - float(global_row['mADM'])):.3f}` percentage points.",
        "",
        "## Category-wise alpha and Local advantage",
        "",
        "| Category | alpha_c | Global NLL | Local NLL | Global−Local NLL |",
        "|---|---:|---:|---:|---:|",
        table,
        "",
        f"- Pearson correlation between Global−Local NLL improvement and alpha_c: `{correlations['pearson_nll_improvement_vs_alpha']}`.",
        f"- Categories with alpha_c >= 0.75: `{', '.join(correlations['high_local_weight_categories']) or 'none'}`.",
        "",
        "## Rank correction",
        "",
        f"- Global → Local Wrong→Correct: `{transition_summary['global_to_local']['wrong_to_correct']}`",
        f"- Global → Local Correct→Wrong: `{transition_summary['global_to_local']['correct_to_wrong']}`",
        f"- Global → Category-wise Wrong→Correct: `{transition_summary['global_to_categorywise']['wrong_to_correct']}`",
        f"- Global → Category-wise Correct→Wrong: `{transition_summary['global_to_categorywise']['correct_to_wrong']}`",
        f"- Category-wise versus Uniform AP improved on `{transition_summary['categorywise_vs_uniform']['ap_improved_queries']}` / `{transition_summary['categorywise_vs_uniform']['query_count']}` queries.",
        "",
        "The category-wise model can improve the aggregate metric while hurting individual queries. "
        "Use `rank_correction.csv` to inspect those query-level transitions; no claim of correct spatial localization is made here.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_alpha_grid(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alpha grid must be comma-separated floats") from exc
    if not result or any(value < 0.0 or value > 1.0 for value in result):
        raise argparse.ArgumentTypeError("alpha grid values must be in [0,1]")
    return sorted(set(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Same-checkpoint Global/Local and category-wise fusion diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-fraction", type=float, default=0.5)
    parser.add_argument(
        "--max-tuning-optimization-queries", type=int, default=256,
        help="cap queries used for mADM alpha search; held-out evaluation always uses all evaluation queries",
    )
    parser.add_argument("--category-alpha-grid", type=_parse_alpha_grid, default=None)
    parser.add_argument("--category-opt-passes", type=int, default=2)
    parser.add_argument("--uniform-alphas", type=_parse_alpha_grid, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def _default_grid() -> list[float]:
    return [round(value, 2) for value in np.arange(0.0, 1.0001, 0.05)]


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("eval-batch-size must be positive and num-workers non-negative")
    if args.image_size != 224:
        raise ValueError("The existing ViT-B/32 evaluation protocol requires image-size=224")
    if args.category_temperature <= 0 or args.ece_bins <= 0:
        raise ValueError("category temperature and ece-bins must be positive")
    if not 0.0 < args.tuning_fraction < 1.0:
        raise ValueError("tuning-fraction must be strictly between 0 and 1")
    if args.category_opt_passes <= 0 or args.progress_every < 0 or args.max_tuning_optimization_queries <= 0:
        raise ValueError(
            "category-opt-passes/max-tuning-optimization-queries must be positive and progress-every non-negative"
        )

    set_seed(args.seed, deterministic=True)
    device = choose_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    grid = args.category_alpha_grid or _default_grid()

    payload = _load_checkpoint_payload(checkpoint)
    if payload.get("architecture") != "attrquery_fromscratch_jointfce":
        raise ValueError("This diagnostic accepts only AttrQuery-FromScratchJointFCE checkpoints")
    initialization = payload.get("initialization") or {}
    if initialization.get("source") != "original_pretrained_clip":
        raise ValueError("Checkpoint is not documented as original pretrained CLIP initialization")
    if initialization.get("a7_checkpoint_used") or initialization.get("attrquery_checkpoint_used"):
        raise ValueError("A7/AttrQuery initialization is forbidden for this diagnostic")
    training_config = payload.get("training_config") or {}
    if training_config.get("validation_protocol") not in {None, "native52_category_nll"}:
        raise ValueError("Checkpoint does not use the Native52 Category-NLL validation protocol")
    if training_config.get("category_temperature") is not None and not math.isclose(
        float(training_config["category_temperature"]), args.category_temperature,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("CLI category temperature differs from the checkpoint training protocol")

    gt_path, table, queries_raw, ids, query_names = _load_validation_views(
        data_root, args.max_val_samples,
    )
    attributes = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attributes)
    validate_mapper(mapper)
    if list(table.attribute_names) != attributes:
        labels = reorder_columns(table.labels, table.attribute_names, attributes)
    else:
        labels = table.labels.astype(np.float32, copy=False)
    queries = reorder_columns(queries_raw, query_names, attributes)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy().astype(np.float32)
    target52 = mapper.encode(torch.from_numpy(labels).float()).numpy().astype(np.float32)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    names = category_names(mapper)
    if list(payload.get("category_groups", [])) not in ([], [list(group) for group in groups]):
        raise ValueError("Checkpoint category partition differs from the canonical 52-state mapping")

    query_count = len(queries)
    if query_count < 2:
        raise ValueError("At least two unique queries are required for tuning/evaluation split")
    split_rng = np.random.default_rng(args.seed)
    shuffled = split_rng.permutation(query_count)
    tuning_count = min(max(int(round(query_count * args.tuning_fraction)), 1), query_count - 1)
    tuning_query_indices = np.sort(shuffled[:tuning_count]).astype(np.int64)
    evaluation_query_indices = np.sort(shuffled[tuning_count:]).astype(np.int64)
    if len(tuning_query_indices) > args.max_tuning_optimization_queries:
        optimization_query_indices = np.sort(
            split_rng.choice(
                tuning_query_indices,
                size=args.max_tuning_optimization_queries,
                replace=False,
            ),
        ).astype(np.int64)
    else:
        optimization_query_indices = tuning_query_indices.copy()
    tuning_query_set = set(int(index) for index in tuning_query_indices)
    split_rows = [
        {"query_index": int(index), "split": "tuning" if index in tuning_query_set else "evaluation",
         "query_attributes": "; ".join(name for value, name in zip(query, attributes) if value > 0.5)}
        for index, query in enumerate(queries)
    ]
    _write_csv(output_dir / "query_split.csv", split_rows)

    model, loaded_payload = load_experiment_model(checkpoint, device)
    model.eval()
    transform = build_eval_transform(args.image_size, "resize_pad_crop")
    roots = [data_root, gt_path.parent, REPOSITORY_ROOT]
    cache_path = output_dir / "gallery_scores.npz"
    global_shape = (len(table.image_paths), 52)
    local_shape = (len(table.image_paths), 12, 52)
    if not args.no_cache and _cache_is_valid(
        cache_path, cache_path.with_suffix(cache_path.suffix + ".json"),
        checkpoint, table.image_paths, global_shape, local_shape,
    ):
        global_scores, local_scores = _load_score_cache(cache_path)
        print(f"Reusing gallery score cache: {cache_path}", flush=True)
    else:
        tokens = model.base.tokenize(mapper.prompts).to(device)
        with torch.inference_mode(), autocast(device, args.amp):
            state_features = model.encode_text(tokens).float()
        loader = DataLoader(
            ImagePathDataset(table.image_paths, roots, transform),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        global_chunks: list[np.ndarray] = []
        local_chunks: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_index, images in enumerate(loader, start=1):
                with autocast(device, args.amp):
                    global_features, local_features = model.joint_features(
                        images.to(device, non_blocking=True),
                    )
                    batch_global = global_features @ state_features.T
                    batch_local = model.category_state_scores_from_features(
                        local_features, state_features,
                    )
                global_chunks.append(batch_global.float().cpu().numpy())
                local_chunks.append(batch_local.float().cpu().numpy())
                if args.progress_every and (
                    batch_index % args.progress_every == 0 or batch_index == len(loader)
                ):
                    print(
                        f"[shared Global+Local] batch {batch_index}/{len(loader)}; "
                        f"elapsed={time.perf_counter() - started:.1f}s", flush=True,
                    )
        global_scores = np.concatenate(global_chunks, axis=0).astype(np.float32, copy=False)
        local_scores = np.concatenate(local_chunks, axis=0).astype(np.float32, copy=False)
        if global_scores.shape != global_shape or local_scores.shape != local_shape:
            raise RuntimeError(
                f"Invalid score shapes: global={global_scores.shape}, local={local_scores.shape}"
            )
        _save_score_cache(cache_path, global_scores, local_scores, checkpoint, table.image_paths)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    global_logs = _category_log_probabilities(global_scores, groups, args.category_temperature, False)
    local_logs = _category_log_probabilities(local_scores, groups, args.category_temperature, True)
    query_targets, valid_categories = _query_category_targets(queries52, groups)

    category_rows = _category_comparison_rows(
        global_logs,
        local_logs,
        target52, groups, names, args.ece_bins,
    )
    _write_csv(output_dir / "category_global_local_comparison.csv", category_rows)

    all_query_indices = np.arange(query_count, dtype=np.int64)
    global_distances = _fused_distances(
        global_logs, local_logs, query_targets, valid_categories, groups,
        [0.0] * len(groups), all_query_indices,
    )
    global_stats = _query_stats(global_distances, all_query_indices, queries, labels)
    global_protocol_match = bool(np.allclose(
        global_distances,
        category_nll(
            queries52,
            _full_category_probabilities(global_logs, groups),
            groups,
        ),
        atol=1e-5,
        rtol=1e-5,
    ))
    if not global_protocol_match:
        raise RuntimeError("Global Category-NLL does not match the established implementation")
    global_scope_metrics = {
        "tuning": _metrics(global_distances[tuning_query_indices], queries[tuning_query_indices], labels, ids),
        "evaluation": _metrics(global_distances[evaluation_query_indices], queries[evaluation_query_indices], labels, ids),
        "full_validation_reference": _metrics(global_distances, queries, labels, ids),
    }
    del global_distances
    local_distances = _fused_distances(
        global_logs, local_logs, query_targets, valid_categories, groups,
        [1.0] * len(groups), all_query_indices,
    )
    local_stats = _query_stats(local_distances, all_query_indices, queries, labels)
    local_protocol_match = bool(np.allclose(
        local_distances,
        category_nll(
            queries52,
            _full_category_probabilities(local_logs, groups),
            groups,
        ),
        atol=1e-5,
        rtol=1e-5,
    ))
    if not local_protocol_match:
        raise RuntimeError("Local Category-NLL does not match the established implementation")
    local_scope_metrics = {
        "tuning": _metrics(local_distances[tuning_query_indices], queries[tuning_query_indices], labels, ids),
        "evaluation": _metrics(local_distances[evaluation_query_indices], queries[evaluation_query_indices], labels, ids),
        "full_validation_reference": _metrics(local_distances, queries, labels, ids),
    }
    del local_distances
    uniform_rows_tuning = _uniform_sweep(
        global_logs, local_logs, query_targets, valid_categories, groups,
        tuning_query_indices, queries, labels, ids, args.uniform_alphas, "tuning",
    )
    uniform_rows_tuning_selection = _uniform_sweep(
        global_logs, local_logs, query_targets, valid_categories, groups,
        optimization_query_indices, queries, labels, ids, args.uniform_alphas,
        "tuning_selection",
    )
    best_uniform_tuning = _best_row(uniform_rows_tuning_selection)
    tuning_uniform_alpha = float(best_uniform_tuning["alpha"])
    category_weights, tuning_category_madm, tuning_history = _coordinate_descent(
        global_logs, local_logs, query_targets, valid_categories, groups,
        optimization_query_indices, queries, labels, ids, grid, args.category_opt_passes,
        tuning_uniform_alpha,
    )
    for row in tuning_history:
        if int(row["category_index"]) >= 0:
            row["category"] = names[int(row["category_index"])]
    categorywise_distances_all = _fused_distances(
        global_logs, local_logs, query_targets, valid_categories, groups,
        category_weights, np.arange(query_count, dtype=np.int64),
    )

    uniform_rows_eval = _uniform_sweep(
        global_logs, local_logs, query_targets, valid_categories, groups,
        evaluation_query_indices, queries, labels, ids, args.uniform_alphas, "evaluation",
    )
    uniform_rows_full = _uniform_sweep(
        global_logs, local_logs, query_targets, valid_categories, groups,
        np.arange(query_count, dtype=np.int64), queries, labels, ids,
        args.uniform_alphas, "full_validation_reference",
    )
    best_uniform_full = _best_row(uniform_rows_full)
    categorywise_all_stats = _query_stats(
        categorywise_distances_all, all_query_indices, queries, labels,
    )
    tuned_category_eval_metrics = _metrics(
        categorywise_distances_all[evaluation_query_indices],
        queries[evaluation_query_indices], labels, ids,
    )
    tuned_category_tuning_metrics = _metrics(
        categorywise_distances_all[tuning_query_indices],
        queries[tuning_query_indices], labels, ids,
    )
    reference_tuned_category_metrics = _metrics(categorywise_distances_all, queries, labels, ids)
    del categorywise_distances_all
    # Apply the alpha selected on tuning to the held-out split and to the
    # full-validation reference view; do not select it from held-out metrics.
    all_query_indices = np.arange(query_count, dtype=np.int64)
    tuned_uniform_distances = _fused_distances(
        global_logs, local_logs, query_targets, valid_categories, groups,
        [tuning_uniform_alpha] * len(groups), all_query_indices,
    )
    uniform_stats = _query_stats(tuned_uniform_distances, all_query_indices, queries, labels)
    tuned_uniform_tuning_metrics = _metrics(
        tuned_uniform_distances[tuning_query_indices],
        queries[tuning_query_indices], labels, ids,
    )
    tuned_uniform_eval_metrics = _metrics(
        tuned_uniform_distances[evaluation_query_indices],
        queries[evaluation_query_indices], labels, ids,
    )
    tuned_uniform_full_metrics = _metrics(tuned_uniform_distances, queries, labels, ids)
    del tuned_uniform_distances

    comparison_rows: list[dict[str, Any]] = []
    for scope, scope_indices in (
        ("tuning", tuning_query_indices),
        ("evaluation", evaluation_query_indices),
        ("full_validation_reference", np.arange(query_count, dtype=np.int64)),
    ):
        scope_global = global_scope_metrics[scope]
        scope_local = local_scope_metrics[scope]
        for method, metrics, alpha, source in (
            ("global_only", scope_global, 0.0, "fixed"),
            ("local_only", scope_local, 1.0, "fixed"),
        ):
            comparison_rows.append({
                "scope": scope, "method": method, "selection_source": source,
                "alpha": alpha, "category_alphas": "", "query_count": int(len(scope_indices)),
                **metrics,
            })
    comparison_rows.extend(uniform_rows_tuning)
    comparison_rows.extend(uniform_rows_tuning_selection)
    comparison_rows.extend(uniform_rows_eval)
    comparison_rows.extend(uniform_rows_full)
    for scope, metrics, scope_indices in (
        ("tuning", tuned_uniform_tuning_metrics, tuning_query_indices),
        ("evaluation", tuned_uniform_eval_metrics, evaluation_query_indices),
        ("full_validation_reference", tuned_uniform_full_metrics, all_query_indices),
    ):
        comparison_rows.append({
            "scope": scope,
            "method": "uniform_fusion_tuning_selected",
            "selection_source": "tuning_query_split",
            "alpha": tuning_uniform_alpha,
            "category_alphas": "",
            "query_count": int(len(scope_indices)),
            **metrics,
        })
    evaluation_category_row = {
        "scope": "evaluation", "method": "categorywise_fusion",
        "selection_source": "tuning_query_split", "alpha": "",
        "category_alphas": ",".join(f"{value:.4f}" for value in category_weights),
        "query_count": int(len(evaluation_query_indices)), **tuned_category_eval_metrics,
    }
    tuning_category_row = {
        "scope": "tuning", "method": "categorywise_fusion",
        "selection_source": "tuning_query_split", "alpha": "",
        "category_alphas": ",".join(f"{value:.4f}" for value in category_weights),
        "query_count": int(len(tuning_query_indices)), **tuned_category_tuning_metrics,
    }
    reference_tuned_category_row = {
        "scope": "full_validation_reference", "method": "categorywise_fusion",
        "selection_source": "tuning_query_split", "alpha": "",
        "category_alphas": ",".join(f"{value:.4f}" for value in category_weights),
        "query_count": int(query_count), **reference_tuned_category_metrics,
    }
    comparison_rows.extend([
        tuning_category_row, evaluation_category_row,
        reference_tuned_category_row,
    ])
    _write_csv(output_dir / "fusion_comparison.csv", comparison_rows)

    weight_rows: list[dict[str, Any]] = []
    for category_index, (name, category_row) in enumerate(zip(names, category_rows)):
        weight_rows.append({
            "category_index": category_index,
            "category": name,
            "categorywise_alpha": float(category_weights[category_index]),
            "global_nll": category_row["global_nll"],
            "local_nll": category_row["local_nll"],
            "nll_improvement_global_minus_local": category_row["nll_improvement_global_minus_local"],
            "global_accuracy": category_row["global_accuracy"],
            "local_accuracy": category_row["local_accuracy"],
            "global_ece": category_row["global_ece"],
            "local_ece": category_row["local_ece"],
            "optimization_mADM_after_category_optimization": float(tuning_category_madm),
        })
    _write_csv(output_dir / "category_fusion_weights.csv", weight_rows)
    _write_csv(output_dir / "category_alpha_optimization.csv", tuning_history)

    tuned_category_stats = categorywise_all_stats
    rank_rows: list[dict[str, Any]] = []
    transition_counts = {
        "global_to_local": {key: 0 for key in ("correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")},
        "global_to_categorywise": {key: 0 for key in ("correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")},
        "categorywise_vs_uniform": {key: 0 for key in ("correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")},
    }
    for position, query_index in enumerate(all_query_indices):
        global_stat = global_stats[position]
        local_stat = local_stats[position]
        category_stat = tuned_category_stats[position]
        uniform_stat = uniform_stats[position]
        global_to_local = _transition(global_stat["top1_correct"], local_stat["top1_correct"])
        global_to_category = _transition(global_stat["top1_correct"], category_stat["top1_correct"])
        category_to_uniform = _transition(uniform_stat["top1_correct"], category_stat["top1_correct"])
        transition_counts["global_to_local"][global_to_local] += 1
        transition_counts["global_to_categorywise"][global_to_category] += 1
        transition_counts["categorywise_vs_uniform"][category_to_uniform] += 1
        rank_rows.append({
            "query_index": int(query_index),
            "query_attributes": "; ".join(
                name for value, name in zip(queries[query_index], attributes) if value > 0.5
            ),
            "split": "tuning" if int(query_index) in tuning_query_set else "evaluation",
            "global_top1_correct": global_stat["top1_correct"],
            "local_top1_correct": local_stat["top1_correct"],
            "uniform_top1_correct": uniform_stat["top1_correct"],
            "categorywise_top1_correct": category_stat["top1_correct"],
            "global_to_local_transition": global_to_local,
            "global_to_categorywise_transition": global_to_category,
            "categorywise_vs_uniform_transition": category_to_uniform,
            "global_best_rank": global_stat["best_correct_rank"],
            "local_best_rank": local_stat["best_correct_rank"],
            "uniform_best_rank": uniform_stat["best_correct_rank"],
            "categorywise_best_rank": category_stat["best_correct_rank"],
            "local_rank_change_vs_global": local_stat["best_correct_rank"] - global_stat["best_correct_rank"],
            "categorywise_rank_change_vs_global": category_stat["best_correct_rank"] - global_stat["best_correct_rank"],
            "categorywise_rank_change_vs_uniform": category_stat["best_correct_rank"] - uniform_stat["best_correct_rank"],
            "global_ap": global_stat["average_precision"],
            "local_ap": local_stat["average_precision"],
            "uniform_ap": uniform_stat["average_precision"],
            "categorywise_ap": category_stat["average_precision"],
            "local_ap_minus_global": local_stat["average_precision"] - global_stat["average_precision"],
            "categorywise_ap_minus_global": category_stat["average_precision"] - global_stat["average_precision"],
            "categorywise_ap_minus_uniform": category_stat["average_precision"] - uniform_stat["average_precision"],
            "global_top1_gallery_index": global_stat["top1_gallery_index"],
            "local_top1_gallery_index": local_stat["top1_gallery_index"],
            "uniform_top1_gallery_index": uniform_stat["top1_gallery_index"],
            "categorywise_top1_gallery_index": category_stat["top1_gallery_index"],
            "best_correct_gallery_index": category_stat["best_correct_gallery_index"],
            "global_top1_image": table.image_paths[global_stat["top1_gallery_index"]],
            "local_top1_image": table.image_paths[local_stat["top1_gallery_index"]],
            "categorywise_top1_image": table.image_paths[category_stat["top1_gallery_index"]],
            "best_correct_image": table.image_paths[category_stat["best_correct_gallery_index"]],
        })
    _write_csv(output_dir / "rank_correction.csv", rank_rows)
    evaluation_rank_rows = [row for row in rank_rows if row["split"] == "evaluation"]

    def _count_rank_transitions(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
        return {
            "global_to_local": {
                key: sum(row["global_to_local_transition"] == key for row in rows)
                for key in ("correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")
            },
            "global_to_categorywise": {
                key: sum(row["global_to_categorywise_transition"] == key for row in rows)
                for key in ("correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")
            },
            "categorywise_vs_uniform": {
                key: sum(row["categorywise_vs_uniform_transition"] == key for row in rows)
                for key in ("correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")
            },
        }

    heldout_transition_counts = _count_rank_transitions(evaluation_rank_rows)
    summary_rows = []
    for scope, counts_by_comparison, count in (
        ("all", transition_counts, len(rank_rows)),
        ("evaluation", heldout_transition_counts, len(evaluation_rank_rows)),
    ):
        for comparison, counts in counts_by_comparison.items():
            summary_rows.append({
                "scope": scope,
                "comparison": comparison,
                **counts,
                "net_correction": counts["wrong_to_correct"] - counts["correct_to_wrong"],
                "query_count": count,
            })
    _write_csv(output_dir / "rank_correction_summary.csv", summary_rows)

    categorywise_eval_rows = evaluation_rank_rows
    category_vs_uniform_eval = [
        row for row in categorywise_eval_rows if row["categorywise_ap_minus_uniform"] > 0
    ]
    nll_values = np.asarray([row["nll_improvement_global_minus_local"] for row in weight_rows], dtype=np.float64)
    alpha_values = np.asarray([row["categorywise_alpha"] for row in weight_rows], dtype=np.float64)
    if len(nll_values) > 1 and np.std(nll_values) > 0 and np.std(alpha_values) > 0:
        pearson = float(np.corrcoef(nll_values, alpha_values)[0, 1])
    else:
        pearson = float("nan")
    correlations = {
        "pearson_nll_improvement_vs_alpha": pearson,
        "high_local_weight_categories": [
            row["category"] for row in weight_rows if float(row["categorywise_alpha"]) >= 0.75
        ],
    }
    transition_summary = {
        **heldout_transition_counts,
        "categorywise_vs_uniform": {
            **heldout_transition_counts["categorywise_vs_uniform"],
            "ap_improved_queries": len(category_vs_uniform_eval),
            "query_count": len(categorywise_eval_rows),
        },
    }
    _plot_outputs(output_dir, category_rows, comparison_rows, weight_rows, rank_rows)

    metadata = {
        "experiment": "AttrQuery-JointCategoryFusion-Diagnostics",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": loaded_payload.get("epoch"),
        "data_root": str(data_root),
        "validation_annotation": str(gt_path),
        "gallery_images": len(table.image_paths),
        "total_queries": query_count,
        "tuning_queries": len(tuning_query_indices),
        "evaluation_queries": len(evaluation_query_indices),
        "optimization_queries": len(optimization_query_indices),
        "split_seed": args.seed,
        "tuning_fraction": args.tuning_fraction,
        "category_temperature": args.category_temperature,
        "category_alpha_grid": grid,
        "uniform_alphas": list(args.uniform_alphas),
        "category_opt_passes": args.category_opt_passes,
        "ece_bins": args.ece_bins,
        "selection_metric": "mADM",
        "same_checkpoint_global_local": True,
        "shared_joint_image_forward": True,
        "score_cache": str(cache_path),
        "protocol": {
            "states": 52,
            "categories": 12,
            "category_nll": True,
            "fusion_space": "category log-probability",
            "raw_logit_mixing": False,
            "gallery_common_between_splits": True,
            "unqueried_categories_excluded_from_distance": True,
            "global_category_nll_matches_existing": global_protocol_match,
            "local_category_nll_matches_existing": local_protocol_match,
        },
        "initialization": initialization,
    }
    _write_json(output_dir / "protocol_validation.json", metadata["protocol"] | {
        "checkpoint_initialization": initialization,
        "training_config": training_config,
    })
    _write_json(output_dir / "analysis_metadata.json", metadata | {
        "tuning_uniform_alpha": tuning_uniform_alpha,
        "tuning_selected_uniform_alpha": tuning_uniform_alpha,
        "evaluation_sweep_best_uniform_alpha_reference": float(_best_row(uniform_rows_eval)["alpha"]),
        "full_validation_reference_uniform_best_alpha": float(best_uniform_full["alpha"]),
        "categorywise_alpha": category_weights.tolist(),
        "correlations": correlations,
        "transitions": transition_summary,
        "transitions_all_queries": transition_counts,
    })
    _write_summary(
        output_dir / "analysis_summary.md", metadata, comparison_rows,
        weight_rows, transition_summary, correlations,
    )
    print(
        f"JointCategoryFusion diagnostics complete: output={output_dir}; "
        f"tuning_mADM={tuning_category_madm:.6f}; device={device}", flush=True,
    )


if __name__ == "__main__":
    main()
