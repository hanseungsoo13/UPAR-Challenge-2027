"""Diagnostics for A7 vs AttrQuery-FrozenCLIP-LocalFCE.

This script is inference-only.  It computes the A7 global score and the
AttrQuery category-local score once, reuses both score tensors for every alpha,
and writes fusion, rank-correction, and attention diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

TOOLS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = TOOLS_DIR.parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_frozenclip_localfce import (  # noqa: E402
    category_names, validate_mapper,
)
from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.models.attrquery_frozenclip_localfce import (  # noqa: E402
    AttrQueryFrozenCLIPLocalFCE, build_text_prototypes,
)
from attrivision.native52_retrieval_ablation import category_nll  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import (  # noqa: E402
    AnnotationTable, ImagePathDataset, find_annotation_file, read_gt_csv,
    resolve_image_path,
)
from upar.retrieval import (  # noqa: E402
    autocast, load_retrieval_annotations, official_retrieval_metrics,
    reorder_columns,
)


DEFAULT_A7_CHECKPOINT = "outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth"
DEFAULT_ATTRQUERY_CHECKPOINT = "outputs/AttrQuery-FrozenCLIP-LocalFCE/checkpoint_best.pth"
DEFAULT_OUTPUT_DIR = "outputs/AttrQuery-Fusion-Diagnostics"
DEFAULT_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _load_pyplot() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "This diagnostic requires matplotlib for curves and attention figures."
        ) from exc
    return plt


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def _checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _annotation_views(
    data_root: Path, max_val_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    return gt_path, table, queries, ids, query_names


def _cache_manifest(
    cache_path: Path, checkpoint: Path, table: AnnotationTable,
) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _cache_is_usable(
    cache_path: Path, checkpoint: Path, table: AnnotationTable,
    expected_shape: tuple[int, ...],
) -> bool:
    manifest_path = _cache_manifest(cache_path, checkpoint, table)
    if not cache_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint") != _checkpoint_fingerprint(checkpoint):
            return False
        if manifest.get("image_paths") != list(table.image_paths):
            return False
        values = np.load(cache_path, allow_pickle=False)
        return values.shape == expected_shape and np.isfinite(values).all()
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _save_score_cache(
    cache_path: Path, values: np.ndarray, checkpoint: Path, table: AnnotationTable,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, values.astype(np.float32, copy=False), allow_pickle=False)
    _write_json(_cache_manifest(cache_path, checkpoint, table), {
        "checkpoint": _checkpoint_fingerprint(checkpoint),
        "image_paths": list(table.image_paths),
        "shape": list(values.shape),
    })


@torch.inference_mode()
def _encode_global_scores(
    model: Any,
    text_features: np.ndarray,
    table: AnnotationTable,
    gt_path: Path,
    data_root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    progress_prefix: str,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(
            table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT], transform,
        ),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            features = model.encode_image(images.to(device, non_blocking=True))
        chunks.append(features.float().cpu().numpy())
        if batch_index == 1 or batch_index == len(loader) or batch_index % 25 == 0:
            print(f"[{progress_prefix}] batch {batch_index}/{len(loader)}", flush=True)
    scores = np.concatenate(chunks, axis=0) @ text_features.T
    scores = scores.astype(np.float32, copy=False)
    expected = (len(table.image_paths), 52)
    if scores.shape != expected or not np.isfinite(scores).all():
        raise RuntimeError(f"Invalid global score shape {scores.shape}; expected {expected}")
    return scores


@torch.inference_mode()
def _encode_local_scores(
    model: AttrQueryFrozenCLIPLocalFCE,
    table: AnnotationTable,
    gt_path: Path,
    data_root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(
            table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT], transform,
        ),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            scores = model.category_state_scores(images.to(device, non_blocking=True))
        chunks.append(scores.float().cpu().numpy())
        if batch_index == 1 or batch_index == len(loader) or batch_index % 25 == 0:
            print(f"[AttrQuery local] batch {batch_index}/{len(loader)}", flush=True)
    result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    expected = (len(table.image_paths), 12, 52)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid local score shape {result.shape}; expected {expected}")
    return result


def _category_log_probabilities(
    scores: np.ndarray, groups: Sequence[np.ndarray], temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    if scores.ndim == 2:
        category_scores = np.broadcast_to(
            scores[:, None, :], (len(scores), len(groups), scores.shape[1]),
        )
    elif scores.ndim == 3:
        category_scores = scores
    else:
        raise ValueError(f"Expected [G,52] or [G,C,52] scores, got {scores.shape}")
    if category_scores.shape[1] != len(groups) or category_scores.shape[2] != 52:
        raise ValueError(f"Score/category mismatch: {category_scores.shape}, {len(groups)} groups")
    result = np.full((len(scores), 52), -np.inf, dtype=np.float64)
    for category_index, indices in enumerate(groups):
        values = category_scores[:, category_index, indices].astype(np.float64) / temperature
        maximum = values.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(np.exp(values - maximum).sum(axis=1, keepdims=True))
        result[:, indices] = values - log_normalizer
    if not np.isfinite(result).all():
        raise RuntimeError("Category log-probabilities contain NaN or Inf")
    return result


def _renormalize_category_log_probs(
    values: np.ndarray, groups: Sequence[np.ndarray],
) -> np.ndarray:
    result = values.copy()
    for indices in groups:
        block = result[:, indices]
        maximum = block.max(axis=1, keepdims=True)
        result[:, indices] = block - maximum - np.log(
            np.exp(block - maximum).sum(axis=1, keepdims=True),
        )
    return result


def _category_nll_from_log_probs(
    query_states: np.ndarray, log_probabilities: np.ndarray,
    groups: Sequence[np.ndarray],
) -> np.ndarray:
    # Match native52_retrieval_ablation.category_nll, which clips probabilities
    # at epsilon before taking log.  This keeps alpha=0 and alpha=1 exactly on
    # the established evaluation definition even for saturated softmax blocks.
    log_epsilon = math.log(1e-12)
    log_probabilities = np.maximum(log_probabilities, log_epsilon)
    distances = np.zeros((len(query_states), len(log_probabilities)), dtype=np.float64)
    for indices in groups:
        targets = query_states[:, indices].astype(np.float64)
        counts = targets.sum(axis=1, keepdims=True)
        if np.any(counts <= 0):
            raise ValueError("Every query category must contain an active state")
        distances -= (targets / counts) @ log_probabilities[:, indices].T
    distances /= len(groups)
    if not np.isfinite(distances).all():
        raise RuntimeError("Category-NLL distances contain NaN or Inf")
    return distances.astype(np.float32, copy=False)


def _query_attributes(query: np.ndarray, names: Sequence[str]) -> str:
    active = [name for value, name in zip(query, names) if value > 0.5]
    return "; ".join(active) if active else "<none>"


def _per_query_stats(
    distances: np.ndarray, queries: np.ndarray, labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id, row in enumerate(distances):
        order = np.argsort(row, kind="stable")
        # Match official_retrieval_metrics: exact equality in the 40-bit
        # reordered annotation space defines a correct gallery item.
        relevant = np.all(labels == queries[query_id], axis=1)
        if not relevant.any():
            raise ValueError(f"Query {query_id} has no exact gallery match")
        relevant_positions = np.flatnonzero(relevant[order])
        ranks = relevant_positions + 1
        top1_index = int(order[0])
        best_index = int(order[relevant_positions[0]])
        ap = float(np.mean(np.arange(1, len(ranks) + 1) / ranks))
        rows.append({
            "query_id": query_id,
            "top1_gallery_index": top1_index,
            "best_correct_gallery_index": best_index,
            "best_correct_rank": int(ranks[0]),
            "average_precision": ap,
            "top1_correct": bool(relevant[top1_index]),
            "relevant_gallery_count": int(relevant.sum()),
        })
    return rows


def _evaluate_distances(
    distances: np.ndarray, queries: np.ndarray, labels: np.ndarray, ids: np.ndarray,
) -> dict[str, float]:
    official = official_retrieval_metrics(distances, queries, labels, ids)
    return {
        "rank1": official["Rank-1"],
        "rank5": official["Rank-5"],
        "rank10": official["Rank-10"],
        "mAP": official["mAP"],
        "mADM": official["mADM"],
        "mINP": official["mINP"],
        "semantic_top1": official["semantic_top1"],
    }


def _validate_protocol(
    a7_payload: dict[str, Any], attr_payload: dict[str, Any],
    a7_text: np.ndarray, attr_text: np.ndarray,
    mapper: CategoryPromptMapper, temperature: float,
) -> dict[str, Any]:
    validate_mapper(mapper)
    if a7_text.shape != attr_text.shape:
        text_diff = float("inf")
    else:
        text_diff = float(np.max(np.abs(a7_text - attr_text)))
    a7_training_config = a7_payload.get("training_config") or {}
    attr_training_config = attr_payload.get("training_config") or {}
    recorded_temperatures = [
        value for value in (
            a7_training_config.get("category_temperature"),
            attr_training_config.get("category_temperature"),
        ) if value is not None
    ]
    temperature_matches_checkpoints = all(
        math.isclose(float(value), temperature, rel_tol=0.0, abs_tol=1e-12)
        for value in recorded_temperatures
    )
    checks: dict[str, Any] = {
        "a7_architecture": a7_payload.get("architecture"),
        "a7_prompt_mode": a7_payload.get("prompt_mode"),
        "attrquery_architecture": attr_payload.get("architecture"),
        "attrquery_feature_mode": attr_payload.get("feature_mode"),
        "states": len(mapper.keys),
        "categories": len(mapper.category_indices()),
        "temperature": temperature,
        "recorded_checkpoint_temperatures": [float(value) for value in recorded_temperatures],
        "category_temperature_matches_checkpoints": temperature_matches_checkpoints,
        "attributes_match": list(a7_payload.get("attribute_names", [])) == mapper.attribute_names,
        "a7_category_complete": a7_payload.get("prompt_mode") == "category_complete",
        "attrquery_local": attr_payload.get("feature_mode") == "local",
        "text_prototype_shape_a7": list(a7_text.shape),
        "text_prototype_shape_attrquery": list(attr_text.shape),
        "text_prototype_max_abs_diff": text_diff,
    }
    checks["text_prototypes_match"] = (
        a7_text.shape == attr_text.shape
        and checks["text_prototype_max_abs_diff"] < 1e-4
    )
    checks["same_native52_partition"] = (
        list(attr_payload.get("category_groups", []))
        == [list(group) for group in mapper.category_indices()]
    )
    required = (
        checks["attributes_match"], checks["a7_category_complete"],
        checks["attrquery_local"], checks["text_prototypes_match"],
        checks["same_native52_partition"], checks["states"] == 52,
        checks["categories"] == 12,
        temperature_matches_checkpoints,
    )
    if not all(required):
        raise RuntimeError(f"A7/AttrQuery protocol validation failed: {checks}")
    return checks


def _save_alpha_curve(output_dir: Path, rows: Sequence[dict[str, Any]], selection_metric: str) -> None:
    plt = _load_pyplot()
    alpha = [row["alpha"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in (("rank1", "Rank-1"), ("mAP", "mAP"), ("mADM", "mADM")):
        ax.plot(alpha, [row[key] for row in rows], marker="o", label=label)
    best = max(rows, key=lambda row: (row[selection_metric], -row["alpha"]))
    ax.axvline(best["alpha"], color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fusion alpha")
    ax.set_ylabel("Metric")
    ax.set_title("A7 global + AttrQuery local Category-NLL fusion")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fusion_alpha_curve.png", dpi=180)
    plt.close(fig)


def _save_rank_plots(
    output_dir: Path, transitions: dict[str, int], rank_delta: np.ndarray,
) -> None:
    plt = _load_pyplot()
    matrix = np.array([
        [transitions["wrong_to_wrong"], transitions["wrong_to_correct"]],
        [transitions["correct_to_wrong"], transitions["correct_to_correct"]],
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Fusion Wrong", "Fusion Correct"])
    ax.set_yticks([0, 1], ["A7 Wrong", "A7 Correct"])
    for y in range(2):
        for x in range(2):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center")
    ax.set_title("Top-1 rank correction matrix")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_correction_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    finite = rank_delta[np.isfinite(rank_delta)]
    if len(finite):
        bins = np.arange(finite.min() - 0.5, finite.max() + 1.5, 1.0)
        ax.hist(finite, bins=bins, color="#4472c4", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Fusion best-correct rank − A7 best-correct rank")
    ax.set_ylabel("Queries")
    ax.set_title("Best-correct rank change distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "rank_change_distribution.png", dpi=180)
    plt.close(fig)


def _open_image(path: Path, size: tuple[int, int] = (320, 420)) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (FileNotFoundError, OSError):
        image = Image.new("RGB", size, "#dddddd")
        ImageDraw.Draw(image).text((10, 10), f"Missing image\n{path}", fill="black")
        return image


def _save_case_figure(
    output_path: Path,
    case: dict[str, Any],
    table: AnnotationTable,
    query: np.ndarray,
    query_names: Sequence[str],
    roots: Sequence[Path],
) -> None:
    plt = _load_pyplot()
    query_text = _query_attributes(query, query_names)
    target_index = int(case["best_correct_gallery_index"])
    a7_index = int(case["a7_top1_gallery_index"])
    fusion_index = int(case["fusion_top1_gallery_index"])
    images = []
    titles = ["Ground-truth gallery", "A7 Top-1", "Fusion Top-1"]
    for index in (target_index, a7_index, fusion_index):
        path = resolve_image_path(table.image_paths[index], roots)
        images.append(_open_image(path))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"Query {case['query_id']} | {case['transition']} | "
        f"A7 rank={case['a7_best_rank']}, Fusion rank={case['fusion_best_rank']}\n{query_text}",
        fontsize=9,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def _collect_attention(
    model: AttrQueryFrozenCLIPLocalFCE,
    indices: Sequence[int],
    table: AnnotationTable,
    gt_path: Path,
    data_root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    if model.feature_mode != "local":
        raise ValueError("Attention diagnostics require a local AttrQuery checkpoint")
    selected_paths = [table.image_paths[index] for index in indices]
    loader = DataLoader(
        ImagePathDataset(
            selected_paths, [data_root, gt_path.parent, REPOSITORY_ROOT], transform,
        ),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    model.eval()
    result: dict[int, np.ndarray] = {}
    grid: tuple[int, int] | None = None
    offset = 0
    for images in loader:
        with autocast(device, amp):
            _, attention, current_grid = model.category_features_with_attention(
                images.to(device, non_blocking=True),
            )
        current = attention.float().cpu().numpy()
        if grid is None:
            grid = current_grid
        elif grid != current_grid:
            raise RuntimeError(f"Patch grid changed from {grid} to {current_grid}")
        for local_index in range(len(current)):
            # [heads, categories, patches]
            result[int(indices[offset + local_index])] = current[local_index]
        offset += len(current)
    if grid is None:
        raise RuntimeError("No attention images were selected")
    return result, grid


def _save_attention_outputs(
    output_dir: Path,
    attention_by_image: dict[int, np.ndarray],
    grid: tuple[int, int],
    categories: Sequence[str],
    table: AnnotationTable,
    gt_path: Path,
    data_root: Path,
    attention_indices: Sequence[int],
) -> dict[str, Any]:
    plt = _load_pyplot()
    attention_dir = output_dir / "attention_heatmaps"
    attention_dir.mkdir(parents=True, exist_ok=True)
    roots = [data_root, gt_path.parent, REPOSITORY_ROOT]
    all_categories = category_names(CategoryPromptMapper(table.attribute_names))
    category_positions = {category: all_categories.index(category) for category in categories}
    similarities: list[np.ndarray] = []
    summary: dict[str, Any] = {"images": [], "grid": list(grid), "heads": 0}
    for image_index in attention_indices:
        weights = attention_by_image[image_index]  # [H,C,N]
        head_count, category_count, patch_count = weights.shape
        summary["heads"] = head_count
        mean_weights = weights.mean(axis=0)
        norms = np.linalg.norm(mean_weights, axis=1, keepdims=True).clip(min=1e-12)
        normalized = mean_weights / norms
        similarity = normalized @ normalized.T
        similarities.append(similarity)
        stem = f"image_{image_index:06d}_{_safe_name(Path(table.image_paths[image_index]).stem)}"
        np.save(attention_dir / f"{stem}_head_category_patch.npy", weights)
        np.save(attention_dir / f"{stem}_mean_category_patch.npy", mean_weights)
        np.save(attention_dir / f"{stem}_category_similarity.npy", similarity)

        image_path = resolve_image_path(table.image_paths[image_index], roots)
        original = _open_image(image_path)
        for category in categories:
            category_position = category_positions[category]
            mean_map = mean_weights[category_position].reshape(grid)
            head_maps = weights[:, category_position].reshape(head_count, *grid)
            image_file = attention_dir / f"{stem}_{_safe_name(category)}.png"
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(original)
            axes[0].set_title("Original")
            axes[1].imshow(mean_map, cmap="magma")
            axes[1].set_title(f"{category} mean attention")
            axes[2].imshow(original)
            axes[2].imshow(mean_map, cmap="jet", alpha=0.48, interpolation="bilinear")
            axes[2].set_title("Overlay")
            for axis in axes:
                axis.axis("off")
            fig.suptitle(
                f"{stem} | {category} | heads={head_count}, grid={grid[0]}x{grid[1]}",
                fontsize=9,
            )
            fig.tight_layout()
            fig.savefig(image_file, dpi=160, bbox_inches="tight")
            plt.close(fig)

            head_file = attention_dir / f"{stem}_{_safe_name(category)}_heads.png"
            columns = min(4, head_count)
            rows = math.ceil(head_count / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
            for head_index, axis in enumerate(axes.flat):
                if head_index < head_count:
                    axis.imshow(head_maps[head_index], cmap="magma")
                    axis.set_title(f"Head {head_index}")
                axis.axis("off")
            fig.suptitle(f"{stem} | {category} | per-head attention", fontsize=10)
            fig.tight_layout()
            fig.savefig(head_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
        summary["images"].append({
            "image_index": image_index,
            "path": table.image_paths[image_index],
            "attention_shape": list(weights.shape),
            "mean_head_cosine_similarity": float(
                (similarity.sum() - np.trace(similarity))
                / max(category_count * (category_count - 1), 1)
            ),
        })

    aggregate = np.mean(similarities, axis=0)
    np.save(output_dir / "attention_similarity_matrix.npy", aggregate)
    _write_csv(
        output_dir / "attention_similarity_matrix.csv",
        [
            {"category": all_categories[row], **{
                all_categories[column]: float(aggregate[row, column])
                for column in range(len(all_categories))
            }}
            for row in range(len(all_categories))
        ],
        ["category", *all_categories],
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(aggregate, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(all_categories)), all_categories, rotation=75, ha="right")
    ax.set_yticks(range(len(all_categories)), all_categories)
    ax.set_title("Mean category attention cosine similarity")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "attention_similarity_matrix.png", dpi=180)
    plt.close(fig)
    _write_json(output_dir / "attention_summary.json", summary)
    return summary


def _parse_alphas(value: str) -> list[float]:
    try:
        alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated floats") from exc
    if not alphas or any(alpha < 0 or alpha > 1 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must be in [0,1]")
    return alphas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference-only diagnostics for A7 and AttrQuery-FrozenCLIP-LocalFCE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a7-checkpoint", default=DEFAULT_A7_CHECKPOINT)
    parser.add_argument("--attrquery-checkpoint", default=DEFAULT_ATTRQUERY_CHECKPOINT)
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--alphas", type=_parse_alphas, default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--selection-metric", choices=("rank1", "mAP", "mADM"), default="mADM",
        help="metric used to select the reported fusion alpha",
    )
    parser.add_argument("--attention-image-count", type=int, default=12)
    parser.add_argument("--attention-indices", type=int, nargs="*")
    parser.add_argument("--attention-categories", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.image_size != 224:
        raise ValueError("eval-batch-size must be positive, num-workers non-negative, image-size=224")
    if args.category_temperature <= 0 or args.attention_image_count <= 0:
        raise ValueError("category temperature and attention image count must be positive")
    set_seed(args.seed, deterministic=True)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    a7_checkpoint = Path(args.a7_checkpoint).resolve()
    attr_checkpoint = Path(args.attrquery_checkpoint).resolve()
    if not a7_checkpoint.is_file():
        raise FileNotFoundError(f"A7 checkpoint not found: {a7_checkpoint}")
    if not attr_checkpoint.is_file():
        raise FileNotFoundError(f"AttrQuery checkpoint not found: {attr_checkpoint}")
    if args.alphas != sorted(set(args.alphas)):
        raise ValueError("alphas must be unique and sorted in ascending order")

    gt_path, table, queries_raw, ids, query_names = _annotation_views(
        Path(args.data_root).resolve(), args.max_val_samples,
    )
    data_root = Path(args.data_root).resolve()
    attr_model, attr_payload = __import__(
        "attrivision.attrquery_frozenclip_localfce", fromlist=["load_experiment_model"],
    ).load_experiment_model(attr_checkpoint, device)
    if attr_model.feature_mode != "local":
        raise ValueError("AttrQuery-Fusion-Diagnostics requires a local AttrQuery checkpoint")
    a7_model, a7_payload = load_model(a7_checkpoint, device)
    mapper = CategoryPromptMapper(list(attr_payload["attribute_names"]))
    validate_mapper(mapper)
    if list(a7_payload.get("attribute_names", [])) != mapper.attribute_names:
        raise ValueError("A7 and AttrQuery attribute order differs")
    a7_text_tensor = build_text_prototypes(a7_model, mapper, device, args.amp)
    a7_text = a7_text_tensor.cpu().numpy().astype(np.float32, copy=False)
    attr_text = attr_model.text_prototypes.detach().cpu().numpy().astype(np.float32, copy=False)
    protocol = _validate_protocol(a7_payload, attr_payload, a7_text, attr_text, mapper, args.category_temperature)
    mapper_groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    queries = reorder_columns(queries_raw, query_names, mapper.attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, mapper.attribute_names)
    query_states = mapper.encode(torch.from_numpy(queries).float()).numpy()
    eval_transform = build_eval_transform(args.image_size, "resize_pad_crop")

    global_cache = output_dir / "a7_global_scores.npy"
    local_cache = output_dir / "attrquery_local_scores.npy"
    expected_global = (len(table.image_paths), 52)
    expected_local = (len(table.image_paths), 12, 52)
    if _cache_is_usable(global_cache, a7_checkpoint, table, expected_global):
        global_scores = np.load(global_cache, allow_pickle=False).astype(np.float32, copy=False)
        print(f"Reusing A7 global score cache: {global_cache}", flush=True)
    else:
        global_scores = _encode_global_scores(
            a7_model, a7_text, table, gt_path, data_root, eval_transform, device,
            args.eval_batch_size, args.num_workers, args.amp, "A7 global",
        )
        _save_score_cache(global_cache, global_scores, a7_checkpoint, table)
    del a7_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if _cache_is_usable(local_cache, attr_checkpoint, table, expected_local):
        local_scores = np.load(local_cache, allow_pickle=False).astype(np.float32, copy=False)
        print(f"Reusing AttrQuery local score cache: {local_cache}", flush=True)
    else:
        local_scores = __import__(
            "attrivision.attrquery_frozenclip_localfce", fromlist=["_encode_gallery_scores"],
        )._encode_gallery_scores(
            attr_model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
            eval_transform, device, args.eval_batch_size, args.num_workers, args.amp, 25,
        )
        _save_score_cache(local_cache, local_scores, attr_checkpoint, table)

    global_log_probs = _category_log_probabilities(global_scores, mapper_groups, args.category_temperature)
    local_log_probs = _category_log_probabilities(local_scores, mapper_groups, args.category_temperature)
    protocol["global_log_probability_nll_matches_existing"] = bool(np.allclose(
        _category_nll_from_log_probs(query_states, global_log_probs, mapper_groups),
        category_nll(query_states, np.exp(global_log_probs).astype(np.float32), mapper_groups),
        atol=1e-5,
    ))
    if not protocol["global_log_probability_nll_matches_existing"]:
        raise RuntimeError("Global log-probability NLL does not match the existing Category-NLL implementation")
    protocol["local_log_probability_nll_matches_existing"] = bool(np.allclose(
        _category_nll_from_log_probs(query_states, local_log_probs, mapper_groups),
        category_nll(query_states, np.exp(local_log_probs).astype(np.float32), mapper_groups),
        atol=1e-5,
    ))
    if not protocol["local_log_probability_nll_matches_existing"]:
        raise RuntimeError("Local log-probability NLL does not match the existing Category-NLL implementation")

    global_distances = _category_nll_from_log_probs(query_states, global_log_probs, mapper_groups)
    local_distances = _category_nll_from_log_probs(query_states, local_log_probs, mapper_groups)
    global_metrics = _evaluate_distances(global_distances, queries, labels, ids)
    local_metrics = _evaluate_distances(local_distances, queries, labels, ids)
    alpha_rows: list[dict[str, Any]] = []
    distances_by_alpha: dict[float, np.ndarray] = {}
    for alpha in args.alphas:
        fused_log_probs = _renormalize_category_log_probs(
            (1.0 - alpha) * global_log_probs + alpha * local_log_probs,
            mapper_groups,
        )
        distances = _category_nll_from_log_probs(query_states, fused_log_probs, mapper_groups)
        metrics = _evaluate_distances(distances, queries, labels, ids)
        alpha_rows.append({
            "alpha": alpha,
            **metrics,
            "delta_rank1_vs_a7": metrics["rank1"] - global_metrics["rank1"],
            "delta_mAP_vs_a7": metrics["mAP"] - global_metrics["mAP"],
            "delta_mADM_vs_a7": metrics["mADM"] - global_metrics["mADM"],
            "delta_rank1_vs_local": metrics["rank1"] - local_metrics["rank1"],
            "delta_mAP_vs_local": metrics["mAP"] - local_metrics["mAP"],
            "delta_mADM_vs_local": metrics["mADM"] - local_metrics["mADM"],
        })
        distances_by_alpha[alpha] = distances
    _write_csv(output_dir / "fusion_alpha_sweep.csv", alpha_rows)
    _save_alpha_curve(output_dir, alpha_rows, args.selection_metric)
    best_row = max(alpha_rows, key=lambda row: (row[args.selection_metric], -row["alpha"]))
    best_alpha = float(best_row["alpha"])
    fusion_distances = distances_by_alpha[best_alpha]
    fusion_metrics = _evaluate_distances(fusion_distances, queries, labels, ids)
    global_query_stats = _per_query_stats(global_distances, queries, labels)
    fusion_query_stats = _per_query_stats(fusion_distances, queries, labels)

    transitions = {
        "correct_to_correct": 0, "wrong_to_correct": 0,
        "correct_to_wrong": 0, "wrong_to_wrong": 0,
    }
    correction_rows: list[dict[str, Any]] = []
    for query_id, (old, new) in enumerate(zip(global_query_stats, fusion_query_stats)):
        old_correct = bool(old["top1_correct"])
        new_correct = bool(new["top1_correct"])
        if old_correct and new_correct:
            transition = "correct_to_correct"
        elif not old_correct and new_correct:
            transition = "wrong_to_correct"
        elif old_correct and not new_correct:
            transition = "correct_to_wrong"
        else:
            transition = "wrong_to_wrong"
        transitions[transition] += 1
        correction_rows.append({
            "query_id": query_id,
            "query_attributes": _query_attributes(queries[query_id], mapper.attribute_names),
            "transition": transition,
            "a7_top1_correct": old_correct,
            "fusion_top1_correct": new_correct,
            "a7_best_rank": old["best_correct_rank"],
            "fusion_best_rank": new["best_correct_rank"],
            "rank_change": new["best_correct_rank"] - old["best_correct_rank"],
            "a7_ap": old["average_precision"],
            "fusion_ap": new["average_precision"],
            "ap_change": new["average_precision"] - old["average_precision"],
            "relevant_gallery_count": old["relevant_gallery_count"],
            "a7_top1_gallery_index": old["top1_gallery_index"],
            "fusion_top1_gallery_index": new["top1_gallery_index"],
            "best_correct_gallery_index": new["best_correct_gallery_index"],
            "a7_top1_image": table.image_paths[old["top1_gallery_index"]],
            "fusion_top1_image": table.image_paths[new["top1_gallery_index"]],
            "best_correct_image": table.image_paths[new["best_correct_gallery_index"]],
            "a7_best_correct_image": table.image_paths[old["best_correct_gallery_index"]],
        })
    _write_csv(output_dir / "rank_correction.csv", correction_rows)
    improved = [row for row in correction_rows if row["transition"] == "wrong_to_correct"]
    improved.sort(key=lambda row: (row["rank_change"], -row["ap_change"]))
    degraded = [row for row in correction_rows if row["transition"] == "correct_to_wrong"]
    degraded.sort(key=lambda row: (-row["rank_change"], row["ap_change"]))
    _write_csv(output_dir / "rank_correction_improved_top20.csv", improved[:20])
    _write_csv(output_dir / "rank_correction_degraded_top20.csv", degraded[:20])
    _save_rank_plots(
        output_dir, transitions,
        np.asarray([row["rank_change"] for row in correction_rows], dtype=np.float64),
    )

    roots = [data_root, gt_path.parent, REPOSITORY_ROOT]
    for label, rows in (("improved", improved[:20]), ("degraded", degraded[:20])):
        for rank, row in enumerate(rows, start=1):
            _save_case_figure(
                output_dir / "rank_correction_cases" / label / f"{rank:02d}_query_{row['query_id']:05d}.png",
                row, table, queries[row["query_id"]], mapper.attribute_names, roots,
            )

    mapper_category_names = category_names(mapper)
    if args.attention_categories:
        unknown = sorted(set(args.attention_categories) - set(mapper_category_names))
        if unknown:
            raise ValueError(f"Unknown attention categories {unknown}; available={mapper_category_names}")
        attention_categories = list(args.attention_categories)
    else:
        attention_categories = mapper_category_names
    if args.attention_indices is not None:
        attention_indices = list(dict.fromkeys(args.attention_indices))
    else:
        attention_indices = list(range(min(args.attention_image_count, len(table.image_paths))))
    if not attention_indices or any(index < 0 or index >= len(table.image_paths) for index in attention_indices):
        raise ValueError("attention indices must reference validation gallery images")
    attention_by_image, grid = _collect_attention(
        attr_model, attention_indices, table, gt_path, data_root, eval_transform,
        device, args.eval_batch_size, args.num_workers, args.amp,
    )
    attention_summary = _save_attention_outputs(
        output_dir, attention_by_image, grid, attention_categories,
        table, gt_path, data_root, attention_indices,
    )

    summary = {
        "experiment": "AttrQuery-Fusion-Diagnostics",
        "a7_checkpoint": str(a7_checkpoint),
        "attrquery_checkpoint": str(attr_checkpoint),
        "data_root": str(data_root),
        "validation_annotation": str(gt_path),
        "images": len(table.image_paths),
        "queries": len(queries),
        "device": str(device),
        "seed": args.seed,
        "category_temperature": args.category_temperature,
        "alphas": list(args.alphas),
        "selection_metric": args.selection_metric,
        "best_alpha": best_alpha,
        "protocol_validation": protocol,
        "metrics": {"a7_global": global_metrics, "attrquery_local": local_metrics, "fusion": fusion_metrics},
        "fusion_improvement_vs_a7": {
            key: fusion_metrics[key] - global_metrics[key]
            for key in ("rank1", "mAP", "mADM")
        },
        "rank_transitions": transitions,
        "net_correction": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
        "attention": {
            "indices": attention_indices,
            "categories": attention_categories,
            **attention_summary,
        },
        "note": (
            "Attention similarity and overlays are diagnostic correlations only; "
            "they do not establish correct attribute localization."
        ),
    }
    _write_json(output_dir / "analysis_summary.json", summary)
    alpha_table = "\n".join(
        f"| {row['alpha']:.2f} | {100*row['rank1']:.2f}% | {100*row['mAP']:.2f}% | {100*row['mADM']:.2f}% |"
        for row in alpha_rows
    )
    markdown = f"""# AttrQuery-Fusion-Diagnostics

Inference-only comparison of the A7 global score and AttrQuery local score.

- A7 checkpoint: `{a7_checkpoint}`
- AttrQuery checkpoint: `{attr_checkpoint}`
- Validation images / queries: `{len(table.image_paths)} / {len(queries)}`
- Category temperature: `{args.category_temperature}`
- Selected alpha by `{args.selection_metric}`: **`{best_alpha:.2f}`**

## Protocol validation

```json
{json.dumps(protocol, indent=2, ensure_ascii=False, default=str)}
```

## Alpha sweep

| Alpha | Rank-1 | mAP | mADM |
|---:|---:|---:|---:|
{alpha_table}

## Best fusion comparison

| Method | Rank-1 | mAP | mADM |
|---|---:|---:|---:|
| A7 global | {100*global_metrics['rank1']:.2f}% | {100*global_metrics['mAP']:.2f}% | {100*global_metrics['mADM']:.2f}% |
| AttrQuery local | {100*local_metrics['rank1']:.2f}% | {100*local_metrics['mAP']:.2f}% | {100*local_metrics['mADM']:.2f}% |
| Fusion alpha={best_alpha:.2f} | {100*fusion_metrics['rank1']:.2f}% | {100*fusion_metrics['mAP']:.2f}% | {100*fusion_metrics['mADM']:.2f}% |

## Rank correction

- Correct → Correct: `{transitions['correct_to_correct']}`
- Wrong → Correct: `{transitions['wrong_to_correct']}`
- Correct → Wrong: `{transitions['correct_to_wrong']}`
- Wrong → Wrong: `{transitions['wrong_to_wrong']}`
- Net correction: **`{transitions['wrong_to_correct'] - transitions['correct_to_wrong']}`**

The attention visualizations are correlation diagnostics.  Similarity or
overlap between category attention maps must not be interpreted as proof of
correct attribute localization.
"""
    (output_dir / "analysis_summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "best_alpha": best_alpha,
        "a7": global_metrics, "local": local_metrics, "fusion": fusion_metrics,
        "rank_transitions": transitions,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
