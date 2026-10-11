"""Shared training, validation, fusion, and attention utilities.

The functions in this module intentionally keep the A7 validation protocol in
one place.  Global, local, and fused scores all use the same validation table,
query IDs, 52-state mapper, category-NLL distance, and official metrics.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .datasets.attribute_prompts import CategoryPromptMapper
from .datasets.upar_abpr import AttriVisionDataset
from .models.attrquery_fromscratch_jointfce import (
    AttrQueryFromScratchJointFCE,
    category_names,
    validate_mapper,
)
from .native52_retrieval_ablation import category_nll
from .transforms import build_eval_transform
from upar.config import REPOSITORY_ROOT
from upar.data import (
    AnnotationTable,
    ImagePathDataset,
    find_annotation_file,
    read_gt_csv,
    resolve_image_path,
)
from upar.retrieval import (
    autocast,
    load_retrieval_annotations,
    official_retrieval_metrics,
    reorder_columns,
)


ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def protocol_validation(config: dict[str, Any], mapper: CategoryPromptMapper) -> dict[str, Any]:
    """Validate the A7-equivalent training and retrieval protocol explicitly."""
    validate_mapper(mapper)
    checks = {
        "vocabulary_is_category_complete_52_state": len(mapper.prompts) == 52,
        "category_count_is_12": len(mapper.category_indices()) == 12,
        "category_groups_partition_52_states": sorted(
            index for group in mapper.category_indices() for index in group
        ) == list(range(52)),
        "text_sampling_is_single": config.get("text_sampling") == "single",
        "contrastive_target_is_multi_positive": config.get("contrastive_target") == "multi_positive",
        "multi_label_is_off": not bool(config.get("multi_label", True)),
        "category_ce_is_off": not bool(config.get("category_ce", True)),
        "augmentation_is_a7_resize_pad_crop": config.get("augmentation") == "resize_pad_crop",
        "global_objective_is_paper_fce": config.get("global_objective") == "paper_fce",
        "validation_is_category_nll": config.get("validation_protocol") == "native52_category_nll",
        "selection_is_fusion_alpha_0_5": float(config.get("selection_alpha", -1.0)) == 0.5,
        "fusion_combines_category_log_probabilities": True,
        "fusion_does_not_mix_raw_logits": True,
    }
    failed = [name for name, value in checks.items() if not value]
    result = {
        "protocol": "A7-compatible Native52 Category-NLL",
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
        "category_names": category_names(mapper),
        "state_keys": list(mapper.keys),
    }
    if failed:
        raise ValueError(f"A7 protocol validation failed: {failed}")
    return result


def valid_category_mask(
    semantic_targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    masks = []
    for group in groups:
        indices = torch.as_tensor(group, device=semantic_targets.device, dtype=torch.long)
        masks.append(semantic_targets[:, indices].sum(dim=1) > 0)
    return torch.stack(masks, dim=1)


def retrieval_views(
    data_root: Path, split: str, max_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, split)
    table = read_gt_csv(gt_path)
    if max_samples is not None:
        count = min(max_samples, len(table.image_paths))
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
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    result = np.full(scores.shape, -np.inf, dtype=np.float64)
    for indices in groups:
        values = scores[:, indices].astype(np.float64) / temperature
        values -= values.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(values).sum(axis=1, keepdims=True))
        result[:, indices] = values - log_normalizer
    return result.astype(np.float32)


def _local_category_log_probabilities(
    scores: np.ndarray,
    groups: Sequence[np.ndarray],
    temperature: float,
) -> np.ndarray:
    if scores.ndim != 3:
        raise ValueError(f"Expected local scores [G,C,52], got {scores.shape}")
    result = np.full(scores.shape, -np.inf, dtype=np.float64)
    for category_index, indices in enumerate(groups):
        values = scores[:, category_index, indices].astype(np.float64) / temperature
        values -= values.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(values).sum(axis=1, keepdims=True))
        result[:, category_index, indices] = values - log_normalizer
    return result.astype(np.float32)


def fused_category_probabilities(
    global_scores: np.ndarray,
    local_scores: np.ndarray,
    groups: Sequence[np.ndarray],
    temperature: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global, local, and fused category probabilities.

    Fusion is performed in category log-probability space.  The returned
    probabilities are therefore valid per-category distributions and can be
    passed to the established ``category_nll`` implementation.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("fusion alpha must be in [0,1]")
    global_log = _category_log_probabilities(global_scores, groups, temperature)
    local_log_by_category = _local_category_log_probabilities(local_scores, groups, temperature)
    local_log = np.full(global_scores.shape, -np.inf, dtype=np.float32)
    for category_index, indices in enumerate(groups):
        local_log[:, indices] = local_log_by_category[:, category_index, indices]
    fused_log = (1.0 - alpha) * global_log + alpha * local_log
    return (
        np.exp(global_log).astype(np.float32),
        np.exp(local_log).astype(np.float32),
        np.exp(fused_log).astype(np.float32),
    )


def _rank_at_k(distances: np.ndarray, ids: np.ndarray, k: int) -> float:
    hits = 0
    for query_id, row in enumerate(distances):
        order = np.argsort(row, kind="stable")[:min(k, len(row))]
        hits += int(np.any(ids[order] == query_id))
    return hits / max(len(distances), 1)


def _category_accuracy(
    scores: np.ndarray,
    target52: np.ndarray,
    groups: Sequence[np.ndarray],
    category_labels: Sequence[str],
    local: bool,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for category_index, (name, indices) in enumerate(zip(category_labels, groups)):
        if local:
            prediction = scores[:, category_index, indices].argmax(axis=1)
        else:
            prediction = scores[:, indices].argmax(axis=1)
        target = target52[:, indices].argmax(axis=1)
        result[f"category_accuracy_{name}"] = float(np.mean(prediction == target))
    return result


def metrics_for_view(
    name: str,
    probabilities: np.ndarray,
    score_for_accuracy: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    queries52: np.ndarray,
    target52: np.ndarray,
    groups: Sequence[np.ndarray],
    category_labels: Sequence[str],
    local_accuracy: bool,
    alpha: float | None,
) -> dict[str, Any]:
    distances = category_nll(queries52, probabilities, groups)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    row: dict[str, Any] = {
        "score_type": name,
        "alpha": "" if alpha is None else float(alpha),
        "rank1": official["Rank-1"],
        "rank5": _rank_at_k(distances, ids, 5),
        "rank10": _rank_at_k(distances, ids, 10),
        "mAP": official["mAP"],
        "mADM": official["mADM"],
        "mINP": official["mINP"],
        "semantic_top1": official["semantic_top1"],
        "images": len(labels),
        "queries": len(queries),
        "retrieval_scoring": "native52_category_nll",
    }
    row.update(_category_accuracy(
        score_for_accuracy, target52, groups, category_labels, local_accuracy,
    ))
    return row


@torch.inference_mode()
def encode_gallery_scores(
    model: AttrQueryFromScratchJointFCE,
    paths: Sequence[str],
    roots: Sequence[Path],
    transform: Any,
    state_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ImagePathDataset(paths, roots, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    global_chunks: list[np.ndarray] = []
    local_chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            global_features, local_features = model.joint_features(
                images.to(device, non_blocking=True),
            )
            global_scores = global_features @ state_features.T
            local_scores = model.category_state_scores_from_features(local_features, state_features)
        global_chunks.append(global_scores.float().cpu().numpy())
        local_chunks.append(local_scores.float().cpu().numpy())
        if progress_every and (batch_index % progress_every == 0 or batch_index == len(loader)):
            processed = sum(len(item) for item in global_chunks)
            print(
                f"[AttrQuery-FromScratch gallery] batch {batch_index}/{len(loader)} "
                f"({processed}/{len(paths)} images)", flush=True,
            )
    global_result = (
        np.concatenate(global_chunks, axis=0)
        if global_chunks else np.empty((0, model.num_states), dtype=np.float32)
    )
    local_result = (
        np.concatenate(local_chunks, axis=0)
        if local_chunks else np.empty((0, model.num_categories, model.num_states), dtype=np.float32)
    )
    expected_global = (len(paths), model.num_states)
    expected_local = (len(paths), model.num_categories, model.num_states)
    if global_result.shape != expected_global or local_result.shape != expected_local:
        raise RuntimeError(
            f"Invalid score shapes: global={global_result.shape} expected={expected_global}; "
            f"local={local_result.shape} expected={expected_local}"
        )
    if not np.isfinite(global_result).all() or not np.isfinite(local_result).all():
        raise RuntimeError("Gallery score cache contains NaN or Inf")
    return global_result.astype(np.float32), local_result.astype(np.float32)


def evaluate_score_arrays(
    global_scores: np.ndarray,
    local_scores: np.ndarray,
    table: AnnotationTable,
    queries_raw: np.ndarray,
    ids: np.ndarray,
    query_names: Sequence[str],
    model_attribute_names: Sequence[str],
    category_temperature: float,
    alphas: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = reorder_columns(queries_raw, query_names, model_attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    mapper = CategoryPromptMapper(model_attribute_names)
    validate_mapper(mapper)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    target52 = mapper.encode(torch.from_numpy(labels).float()).numpy()
    if global_scores.shape[0] != len(labels) or local_scores.shape[0] != len(labels):
        raise ValueError("Gallery score count does not match validation table")

    global_probabilities, local_probabilities, _ = fused_category_probabilities(
        global_scores, local_scores, groups, category_temperature, 0.5,
    )
    rows = [
        metrics_for_view(
            "global", global_probabilities, global_scores, queries, labels, ids,
            queries52, target52, groups, category_names(mapper), False, None,
        ),
        metrics_for_view(
            "local", local_probabilities, local_scores, queries, labels, ids,
            queries52, target52, groups, category_names(mapper), True, None,
        ),
    ]
    fusion_rows: list[dict[str, Any]] = []
    fusion_scores: dict[str, np.ndarray] = {}
    for alpha in alphas:
        _, _, fused_probabilities = fused_category_probabilities(
            global_scores, local_scores, groups, category_temperature, float(alpha),
        )
        fused_log_score = np.zeros_like(global_scores, dtype=np.float32)
        global_log = np.log(np.clip(global_probabilities, 1e-12, 1.0))
        local_log = np.log(np.clip(local_probabilities, 1e-12, 1.0))
        fused_log_score[:] = (1.0 - float(alpha)) * global_log + float(alpha) * local_log
        fusion_scores[f"alpha_{float(alpha):.3f}"] = fused_log_score
        fusion_rows.append(metrics_for_view(
            "fusion", fused_probabilities, fused_log_score, queries, labels, ids,
            queries52, target52, groups, category_names(mapper), False, float(alpha),
        ))
    rows.extend(fusion_rows)
    return rows, {
        "queries": queries,
        "labels": labels,
        "ids": ids,
        "query_names": list(query_names),
        "model_attribute_names": list(model_attribute_names),
        "groups": groups,
        "category_names": category_names(mapper),
        "global_probabilities": global_probabilities,
        "local_probabilities": local_probabilities,
        "fusion_rows": fusion_rows,
        "fusion_scores": fusion_scores,
    }


def evaluate_model(
    model: AttrQueryFromScratchJointFCE,
    model_attribute_names: Sequence[str],
    data_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    category_temperature: float,
    max_val_samples: int | None = None,
    alphas: Sequence[float] = ALPHAS,
    progress_every: int = 0,
    gallery_scores: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    gt_path, table, queries, ids, query_names = retrieval_views(
        data_root, "val", max_val_samples,
    )
    mapper = CategoryPromptMapper(model_attribute_names)
    state_tokens = model.base.tokenize(mapper.prompts).to(device)
    with torch.inference_mode(), autocast(device, amp):
        state_features = model.encode_text(state_tokens).float()
    if gallery_scores is None:
        gallery_scores = encode_gallery_scores(
            model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(224, "resize_pad_crop"), state_features,
            device, batch_size, num_workers, amp, progress_every,
        )
    rows, details = evaluate_score_arrays(
        gallery_scores[0], gallery_scores[1], table, queries, ids, query_names,
        model_attribute_names, category_temperature, alphas,
    )
    details.update({
        "table": table,
        "gt_path": gt_path,
        "state_features": state_features,
        "global_scores": gallery_scores[0],
        "local_scores": gallery_scores[1],
    })
    return rows, details


def best_fusion_row(rows: Sequence[dict[str, Any]], alpha: float) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if row.get("score_type") == "fusion" and abs(float(row.get("alpha", -1.0)) - alpha) < 1e-9
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one fusion row for alpha={alpha}, found {len(candidates)}")
    return candidates[0]


def save_gallery_scores(path: Path, scores: tuple[np.ndarray, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, global_scores=scores[0], local_scores=scores[1])


def load_gallery_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "global_scores" not in archive or "local_scores" not in archive:
            raise ValueError(f"Gallery score cache lacks global_scores/local_scores: {path}")
        return archive["global_scores"], archive["local_scores"]


def plot_alpha_curve(rows: Sequence[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fusion = sorted(
        [row for row in rows if row.get("score_type") == "fusion"],
        key=lambda row: float(row["alpha"]),
    )
    if not fusion:
        raise ValueError("No fusion rows available for alpha curve")
    alpha = [float(row["alpha"]) for row in fusion]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for key, label in (("rank1", "Rank-1"), ("mAP", "mAP"), ("mADM", "mADM"), ("mINP", "mINP")):
        plt.plot(alpha, [float(row[key]) * 100.0 for row in fusion], marker="o", label=label)
    plt.xlabel("Fusion alpha (local weight)")
    plt.ylabel("Metric (%)")
    plt.xticks(alpha)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _save_attention_image(
    original: Image.Image,
    heatmap: np.ndarray,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.0, 4.5))
    plt.imshow(original)
    image = plt.imshow(heatmap, cmap="magma", alpha=0.52, vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _save_heatmap_only(heatmap: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.0, 4.5))
    image = plt.imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_attention_visualizations(
    model: AttrQueryFromScratchJointFCE,
    table: AnnotationTable,
    roots: Sequence[Path],
    transform: Any,
    indices: Sequence[int],
    output_dir: Path,
    device: torch.device,
    amp: bool,
    selected_categories: Sequence[str] | None = None,
    save_all_heads: bool = True,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    mapper = model.mapper
    name_to_index = {name: index for index, name in enumerate(category_names(mapper))}
    if selected_categories is None:
        category_indices = list(range(model.num_categories))
    else:
        missing = [name for name in selected_categories if name not in name_to_index]
        if missing:
            raise ValueError(f"Unknown attention categories: {missing}")
        category_indices = [name_to_index[name] for name in selected_categories]
    rows: list[dict[str, Any]] = []
    model.eval()
    for index in indices:
        if index < 0 or index >= len(table.image_paths):
            raise IndexError(f"Attention image index {index} is outside validation set")
        image_path = resolve_image_path(table.image_paths[index], roots)
        with Image.open(image_path) as image:
            original = image.convert("RGB").copy()
        tensor = transform(original).unsqueeze(0).to(device)
        with torch.inference_mode(), autocast(device, amp):
            _, _, attention, grid = model.joint_features_with_attention(tensor)
        attention_np = attention[0].float().cpu().numpy()  # [H,C,N]
        sample_dir = output_dir / f"sample_{index:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        original.save(sample_dir / "original.png")
        np.save(sample_dir / "attention_heads.npy", attention_np)
        np.save(sample_dir / "attention_mean.npy", attention_np.mean(axis=0))
        vectors: list[np.ndarray] = []
        for category_index in category_indices:
            category_name = category_names(mapper)[category_index]
            slug = _slug(category_name)
            head_values = attention_np[:, category_index]
            mean_values = head_values.mean(axis=0)
            mean_values = mean_values / max(float(mean_values.max()), 1e-12)
            heatmap = mean_values.reshape(grid)
            _save_attention_image(
                original, heatmap, sample_dir / f"{slug}_mean_overlay.png",
                f"{category_name} / mean attention",
            )
            _save_heatmap_only(
                heatmap, sample_dir / f"{slug}_mean_heatmap.png",
                f"{category_name} / mean attention",
            )
            np.save(sample_dir / f"{slug}_mean.npy", heatmap)
            vectors.append(mean_values / max(float(np.linalg.norm(mean_values)), 1e-12))
            if save_all_heads:
                for head_index, values in enumerate(head_values):
                    values = values / max(float(values.max()), 1e-12)
                    _save_attention_image(
                        original, values.reshape(grid),
                        sample_dir / f"{slug}_head{head_index:02d}_overlay.png",
                        f"{category_name} / head {head_index}",
                    )
        similarity = np.asarray(vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
        np.save(sample_dir / "category_attention_similarity.npy", similarity)
        plt.figure(figsize=(7, 6))
        plt.imshow(similarity, cmap="viridis", vmin=-1.0, vmax=1.0)
        plt.colorbar(label="cosine similarity")
        labels = [category_names(mapper)[index] for index in category_indices]
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
        plt.yticks(range(len(labels)), labels, fontsize=7)
        plt.tight_layout()
        plt.savefig(sample_dir / "category_attention_similarity.png", dpi=180)
        plt.close()
        rows.append({
            "image_index": index,
            "image_path": table.image_paths[index],
            "grid_h": grid[0],
            "grid_w": grid[1],
            "heads": attention_np.shape[0],
            "categories": len(category_indices),
        })
    return rows


def plot_attention_similarity_summary(
    output_dir: Path, attention_rows: Sequence[dict[str, Any]], category_labels: Sequence[str],
    summary_output_dir: Path | None = None,
) -> None:
    """Average per-image category attention similarity for a compact summary."""
    if not attention_rows:
        return
    matrices = []
    for row in attention_rows:
        path = output_dir / f"sample_{int(row['image_index']):05d}" / "category_attention_similarity.npy"
        if path.is_file():
            matrices.append(np.load(path))
    if not matrices:
        return
    matrix = np.mean(np.stack(matrices), axis=0)
    target_dir = output_dir if summary_output_dir is None else summary_output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "attention_similarity_matrix.npy", matrix)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(len(category_labels)), category_labels, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(category_labels)), category_labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(target_dir / "attention_similarity_matrix.png", dpi=180)
    plt.close()
