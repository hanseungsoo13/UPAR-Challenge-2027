"""Mixed categorical/multi-label retrieval evaluation for A7-mixed."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from upar.config import REPOSITORY_ROOT
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv
from upar.retrieval import (
    official_retrieval_metrics, load_retrieval_annotations, reorder_columns,
)

from ..datasets.attribute_prompts import MixedCategoryPromptMapper
from .evaluator_abpr import _binary_metric, _rank_at_k, encode_gallery
from ..compare_inference import encode_texts


def mixed_state_outputs(
    gallery_features: torch.Tensor,
    state_features: torch.Tensor,
    mapper: MixedCategoryPromptMapper,
    temperature: float,
) -> np.ndarray:
    """Return categorical softmax and multi-label sigmoid state probabilities."""
    if temperature <= 0:
        raise ValueError("mixed state temperature must be positive")
    scores = (gallery_features.float() @ state_features.float().T).numpy()
    probabilities = np.zeros_like(scores, dtype=np.float32)
    for _, kind, indices in mapper.category_specs():
        values = scores[:, indices].astype(np.float64) / temperature
        if kind == "single":
            values -= values.max(axis=1, keepdims=True)
            exp_values = np.exp(values)
            probabilities[:, indices] = (
                exp_values / exp_values.sum(axis=1, keepdims=True)
            ).astype(np.float32)
        else:
            values = np.clip(values, -80.0, 80.0)
            probabilities[:, indices] = (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Mixed state probabilities contain non-finite values")
    return probabilities


def mixed_state_distances(
    queries: np.ndarray,
    probabilities: np.ndarray,
    mapper: MixedCategoryPromptMapper,
    chunk_size: int = 256,
) -> np.ndarray:
    """Compute category-balanced NLL for mixed single/multi-label groups."""
    targets = mapper.encode(torch.as_tensor(queries, dtype=torch.float32)).numpy().astype(np.float32)
    distances = np.zeros((len(targets), len(probabilities)), dtype=np.float32)
    epsilon = 1e-6
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        query_block = targets[start:stop]
        result = np.zeros((stop - start, len(probabilities)), dtype=np.float32)
        for _, kind, indices in mapper.category_specs():
            block_targets = query_block[:, indices]
            counts = block_targets.sum(axis=1, keepdims=True)
            valid = counts[:, 0] > 0
            normalized = np.divide(
                block_targets, np.maximum(counts, 1.0),
                out=np.zeros_like(block_targets), where=counts > 0,
            )
            log_prob = np.log(np.clip(probabilities[:, indices], epsilon, 1.0))
            if kind == "single":
                contribution = -(normalized @ log_prob.T)
            else:
                log_not = np.log(np.clip(1.0 - probabilities[:, indices], epsilon, 1.0))
                contribution = -(
                    block_targets @ log_prob.T
                    + (1.0 - block_targets) @ log_not.T
                )
                # A colour group has no fallback in A7-mixed.  If a future
                # annotation has no colour, omit only that missing category
                # rather than creating an artificial target.
            contribution[~valid] = 0.0
            result += contribution
        # The previous line used a per-query denominator only implicitly for
        # valid category rows. Recompute it explicitly for mixed missing data.
        valid_group_count = np.zeros((stop - start, 1), dtype=np.float32)
        for _, _, indices in mapper.category_specs():
            valid_group_count += (query_block[:, indices].sum(axis=1, keepdims=True) > 0)
        distances[start:stop] = result / np.maximum(valid_group_count, 1.0)
    return distances


@torch.inference_mode()
def evaluate_mixed_state_nll(
    model: Any,
    model_attribute_names: Sequence[str],
    data_root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    max_val_samples: int | None = None,
    temperature: float = 0.01,
    distance_chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate A7-mixed with categorical NLL plus multi-label BCE-NLL."""
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
    queries = reorder_columns(queries, query_names, model_attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    gallery = encode_gallery(
        model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
        transform, device, batch_size, num_workers, amp,
    )
    mapper = MixedCategoryPromptMapper(model_attribute_names)
    state_features = encode_texts(model, mapper.prompts, device, amp)
    probabilities = mixed_state_outputs(gallery, state_features, mapper, temperature)
    distances = mixed_state_distances(queries, probabilities, mapper, distance_chunk_size)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    probabilities40 = mapper.project_40(torch.from_numpy(probabilities)).numpy()
    per_attribute = [
        _binary_metric(labels[:, index], probabilities40[:, index])
        for index in range(labels.shape[1])
    ]
    semantic_ranks = []
    for column, query_id in enumerate(ids):
        row = distances[:, column]
        semantic_ranks.append(1 + int(np.count_nonzero(row < row[query_id])))
    result: dict[str, Any] = {
        "images": float(len(table.image_paths)),
        "queries": float(len(queries)),
        "rank1": official["Rank-1"],
        "rank5": _rank_at_k(distances, ids, 5),
        "rank10": _rank_at_k(distances, ids, 10),
        "map": official["mAP"],
        "mADM": official["mADM"],
        "mINP": official["mINP"],
        "macro_auroc": float(np.nanmean([item["auroc"] for item in per_attribute])),
        "macro_ap": float(np.nanmean([item["ap"] for item in per_attribute])),
        "macro_f1": float(np.nanmean([item["f1"] for item in per_attribute])),
        "semantic_top1": float(np.mean(np.asarray(semantic_ranks) == 1)),
        "semantic_mean_rank": float(np.mean(semantic_ranks)),
        "retrieval_scoring": "mixed_state_nll",
        "state_count": float(len(mapper.keys)),
        "temperature": float(temperature),
    }
    return result
