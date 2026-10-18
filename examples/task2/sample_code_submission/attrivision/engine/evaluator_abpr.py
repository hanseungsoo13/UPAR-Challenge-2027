"""CLIP feature extraction and official-protocol local ABPR evaluation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from upar.config import REPOSITORY_ROOT
from upar.data import AnnotationTable, ImagePathDataset, find_annotation_file, read_gt_csv
from upar.retrieval import (
    autocast, l1_attribute_distances, load_retrieval_annotations,
    official_retrieval_metrics, reorder_columns,
)

from ..datasets.attribute_prompts import CategoryPromptMapper, prompts_for_attributes
from .paired_attribute import (
    encode_prompt_pairs, learned_inverse_temperature, paired_attribute_probabilities,
)


def _binary_metric(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return {"auroc": float("nan"), "ap": float("nan"), "f1": float("nan")}
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    for start, stop in zip(starts, stops):
        ranks[start:stop] = 0.5 * (start + 1 + stop)
    auroc = (ranks[y_true[order].astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    descending = np.argsort(-scores, kind="stable")
    sorted_y = y_true[descending]
    sorted_scores = scores[descending]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    tp = np.cumsum(sorted_y)[ends]
    precision = tp / (ends + 1)
    recall = tp / positives
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    predicted = scores >= 0.5
    truth = y_true.astype(bool)
    true_pos = int(np.count_nonzero(predicted & truth))
    false_pos = int(np.count_nonzero(predicted & ~truth))
    false_neg = int(np.count_nonzero(~predicted & truth))
    f1 = 2.0 * true_pos / max(2 * true_pos + false_pos + false_neg, 1)
    return {"auroc": float(auroc), "ap": ap, "f1": f1}


@torch.inference_mode()
def evaluate_binary_head(model: Any, model_attribute_names: Sequence[str], data_root: Path,
                         transform: Any, device: torch.device, batch_size: int,
                         num_workers: int, amp: bool, max_val_samples: int | None = None,
                         distance_chunk_size: int = 256) -> dict[str, Any]:
    """Evaluate sigmoid(Linear(image_feature,40)) with official Task-2 L1."""
    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = AnnotationTable(table.image_paths[:count], table.labels[:count], table.attribute_names)
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    queries = reorder_columns(queries, query_names, model_attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    features = encode_gallery(
        model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
        transform, device, batch_size, num_workers, amp,
    )
    logits = model.binary_logits_from_features(features.to(device)).float()
    probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
    distances = l1_attribute_distances(queries, probabilities, distance_chunk_size)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    rank1, mean_ap = official["Rank-1"], official["mAP"]
    rank_k = {}
    for k in (5, 10):
        order = np.argsort(distances, axis=1, kind="stable")[:, :min(k, distances.shape[1])]
        rank_k[k] = float(np.mean(np.any(ids[order] == np.arange(len(queries))[:, None], axis=1)))
    semantic_ranks = []
    for column, query_id in enumerate(ids):
        row = distances[:, column]
        semantic_ranks.append(1 + int(np.count_nonzero(row < row[query_id])))
    per_attribute = [_binary_metric(labels[:, i], probabilities[:, i]) for i in range(labels.shape[1])]
    return {
        "images": float(len(table.image_paths)), "queries": float(len(queries)),
        "rank1": rank1, "rank5": rank_k[5], "rank10": rank_k[10], "map": mean_ap,
        "mADM": official["mADM"], "mINP": official["mINP"],
        "macro_auroc": float(np.nanmean([m["auroc"] for m in per_attribute])),
        "macro_ap": float(np.nanmean([m["ap"] for m in per_attribute])),
        "macro_f1": float(np.nanmean([m["f1"] for m in per_attribute])),
        "semantic_top1": float(np.mean(np.asarray(semantic_ranks) == 1)),
        "semantic_mean_rank": float(np.mean(semantic_ranks)),
        "prediction_mean": float(probabilities.mean()),
        "prediction_std": float(probabilities.std()),
    }


@torch.inference_mode()
def evaluate_native52(
    model: Any, model_attribute_names: Sequence[str], data_root: Path,
    transform: Any, device: torch.device, batch_size: int,
    num_workers: int, amp: bool, max_val_samples: int | None = None,
    attribute_temperature: float | None = None, distance_chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate the fixed Native52 category-softmax -> 40-D -> L1 protocol."""
    # These established audit helpers import ``encode_gallery`` from this
    # module, so defer their import until module initialization is complete.
    from ..compare_inference import encode_texts, native_state_outputs
    from ..evaluate_native52_hard import (
        hard_category_projection, hard_prediction_metrics, semantic_query_metrics,
    )
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
    mapper = CategoryPromptMapper(model_attribute_names)
    if len(mapper.keys) != 52:
        raise RuntimeError(f"Expected exactly 52 native states, got {len(mapper.keys)}")
    state_features = encode_texts(model, mapper.prompts, device, amp)
    inverse_temperature = (
        1.0 / attribute_temperature
        if attribute_temperature is not None
        else learned_inverse_temperature(model)
    )
    probabilities, _ = native_state_outputs(
        gallery, state_features, mapper, inverse_temperature,
    )
    hard = hard_category_projection(gallery, state_features, mapper)
    distances = l1_attribute_distances(queries, probabilities, distance_chunk_size)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    rank1, mean_ap = official["Rank-1"], official["mAP"]
    binary = [_binary_metric(labels[:, i], probabilities[:, i])
              for i in range(labels.shape[1])]
    result: dict[str, Any] = {
        "images": float(len(table.image_paths)), "queries": float(len(queries)),
        "rank1": rank1, "rank5": _rank_at_k(distances, ids, 5),
        "rank10": _rank_at_k(distances, ids, 10), "map": mean_ap,
        "mADM": official["mADM"], "mINP": official["mINP"],
        "macro_auroc": float(np.nanmean([item["auroc"] for item in binary])),
        "macro_ap": float(np.nanmean([item["ap"] for item in binary])),
        "retrieval_scoring": "native52_soft_l1",
        "temperature": float(1.0 / inverse_temperature),
    }
    result.update(hard_prediction_metrics(labels, hard))
    semantic = semantic_query_metrics(distances, ids, distance_chunk_size)
    result["semantic_top1"] = semantic["semantic_query_top1"]
    result["semantic_mean_rank"] = semantic["semantic_query_mean_rank"]
    return result


@torch.inference_mode()
def evaluate_native52_category_nll(
    model: Any, model_attribute_names: Sequence[str], data_root: Path,
    transform: Any, device: torch.device, batch_size: int,
    num_workers: int, amp: bool, max_val_samples: int | None = None,
    attribute_temperature: float = 0.01, distance_chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate the fixed Native52 category-local NLL retrieval protocol."""
    if attribute_temperature <= 0:
        raise ValueError("Category-NLL temperature must be positive")
    # Deferred imports avoid the audit helpers' reverse dependency on this module.
    from ..compare_inference import encode_texts, native_state_outputs
    from ..evaluate_native52_hard import (
        hard_category_projection, hard_prediction_metrics, semantic_query_metrics,
    )
    from ..native52_retrieval_ablation import category_nll, category_softmax

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
    mapper = CategoryPromptMapper(model_attribute_names)
    state_features = encode_texts(model, mapper.prompts, device, amp)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    raw_logits = (gallery.float() @ state_features.float().T).numpy()
    probabilities52 = category_softmax(raw_logits / attribute_temperature, groups)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    distances = category_nll(queries52, probabilities52, groups)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    rank1, mean_ap = official["Rank-1"], official["mAP"]

    probabilities40, _ = native_state_outputs(
        gallery, state_features, mapper, 1.0 / attribute_temperature,
    )
    binary = [_binary_metric(labels[:, index], probabilities40[:, index])
              for index in range(labels.shape[1])]
    hard = hard_category_projection(gallery, state_features, mapper)
    result: dict[str, Any] = {
        "images": float(len(table.image_paths)), "queries": float(len(queries)),
        "rank1": rank1, "rank5": _rank_at_k(distances, ids, 5),
        "rank10": _rank_at_k(distances, ids, 10), "map": mean_ap,
        "mADM": official["mADM"], "mINP": official["mINP"],
        "macro_auroc": float(np.nanmean([item["auroc"] for item in binary])),
        "macro_ap": float(np.nanmean([item["ap"] for item in binary])),
        "retrieval_scoring": "native52_category_nll",
        "temperature": float(attribute_temperature),
    }
    result.update(hard_prediction_metrics(labels, hard))
    semantic = semantic_query_metrics(distances, ids, distance_chunk_size)
    result["semantic_top1"] = semantic["semantic_query_top1"]
    result["semantic_mean_rank"] = semantic["semantic_query_mean_rank"]
    return result


@torch.inference_mode()
def evaluate_native52_category_nll_domainwise(
    model: Any, model_attribute_names: Sequence[str], data_root: Path,
    transform: Any, device: torch.device, batch_size: int,
    num_workers: int, amp: bool, max_val_samples: int | None = None,
    attribute_temperature: float = 0.01, distance_chunk_size: int = 256,
    unseen_domain: str | None = None,
    exclude_domain: str | None = None,
) -> dict[str, Any]:
    """Evaluate overall and query-domain metrics.

    ``ids.csv`` maps every validation image to a retrieval query.  A query is
    included in a domain when that query has at least one positive image in the
    domain.  With ``unseen_domain`` set, each domain is evaluated against its
    own gallery, so the output contains both seen-source and unseen-target
    results.  The top-level metrics always use the complete validation gallery
    and are used for Overall checkpoint selection; ``unseen_*`` stores the
    target-only summary.
    """
    if attribute_temperature <= 0:
        raise ValueError("Category-NLL temperature must be positive")
    from ..compare_inference import encode_texts, native_state_outputs
    from ..evaluate_native52_hard import (
        hard_category_projection, hard_prediction_metrics, semantic_query_metrics,
    )
    from ..native52_retrieval_ablation import category_nll, category_softmax

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries: np.ndarray | None = None
    ids: np.ndarray | None = None
    query_names: list[str] | None = None
    if exclude_domain is not None:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
        domains = np.asarray(
            [str(path).split("/", 1)[0] for path in table.image_paths], dtype=object,
        )
        keep = domains != exclude_domain
        if not keep.any():
            raise ValueError(f"exclude-domain={exclude_domain!r} removed every validation image")
        table = AnnotationTable(
            [path for path, selected in zip(table.image_paths, keep) if selected],
            table.labels[keep], table.attribute_names,
        )
        ids = ids[keep]
        present_queries = np.unique(ids)
        remap = {int(old): new for new, old in enumerate(present_queries.tolist())}
        queries = queries[present_queries]
        ids = np.asarray([remap[int(query_id)] for query_id in ids], dtype=np.int64)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    elif exclude_domain is None:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    assert queries is not None and ids is not None and query_names is not None
    queries = reorder_columns(queries, query_names, model_attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    domain_names_in_table = np.asarray(
        [str(path).split("/", 1)[0] for path in table.image_paths], dtype=object,
    )
    if unseen_domain is not None and unseen_domain not in set(domain_names_in_table.tolist()):
        raise ValueError(f"unseen-domain={unseen_domain!r} was not found in validation annotations")
    all_gallery_indices = np.arange(len(table.image_paths), dtype=np.int64)
    gallery_indices = all_gallery_indices
    gallery_paths = [table.image_paths[index] for index in gallery_indices]
    gallery = encode_gallery(
        model, gallery_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
        transform, device, batch_size, num_workers, amp,
    )
    mapper = CategoryPromptMapper(model_attribute_names)
    state_features = encode_texts(model, mapper.prompts, device, amp)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    raw_logits = (gallery.float() @ state_features.float().T).numpy()
    probabilities52 = category_softmax(raw_logits / attribute_temperature, groups)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    distances = category_nll(queries52, probabilities52, groups)

    probabilities40, _ = native_state_outputs(
        gallery, state_features, mapper, 1.0 / attribute_temperature,
    )
    hard = hard_category_projection(gallery, state_features, mapper)

    def summarize(
        query_indices: np.ndarray, selected_gallery_indices: np.ndarray,
    ) -> dict[str, float]:
        query_indices = np.asarray(query_indices, dtype=np.int64)
        selected_gallery_indices = np.asarray(selected_gallery_indices, dtype=np.int64)
        if query_indices.size == 0:
            raise ValueError("Domain query group is empty")
        selected_labels = labels[selected_gallery_indices]
        selected_global_ids = ids[selected_gallery_indices]
        selected_distances = distances[np.ix_(query_indices, selected_gallery_indices)]
        local_ids = {int(global_id): local_id for local_id, global_id in enumerate(query_indices)}
        local_gallery_ids = np.asarray(
            [local_ids.get(int(global_id), -1) for global_id in selected_global_ids],
            dtype=np.int64,
        )
        official = official_retrieval_metrics(
            selected_distances, queries[query_indices], selected_labels,
            local_gallery_ids,
        )
        return {
            "queries": float(len(query_indices)),
            "gallery_images": float(len(selected_gallery_indices)),
            "rank1": official["Rank-1"],
            "rank5": _rank_at_k(selected_distances, local_gallery_ids, 5),
            "rank10": _rank_at_k(selected_distances, local_gallery_ids, 10),
            "map": official["mAP"],
            "mADM": official["mADM"],
            "mINP": official["mINP"],
        }

    domain_query_indices_by_name = {
        domain: np.unique(ids[domain_names_in_table == domain]).astype(np.int64)
        for domain in sorted(set(domain_names_in_table.tolist()))
    }
    all_query_indices = np.arange(len(queries), dtype=np.int64)
    all_summary = summarize(all_query_indices, all_gallery_indices)
    evaluation_labels = labels[all_gallery_indices]
    evaluation_probabilities = probabilities40[all_gallery_indices]
    evaluation_hard = hard[all_gallery_indices]
    binary = [_binary_metric(evaluation_labels[:, index], evaluation_probabilities[:, index])
              for index in range(evaluation_labels.shape[1])]
    result: dict[str, Any] = {
        **all_summary,
        "images": float(len(all_gallery_indices)),
        "retrieval_scoring": "native52_category_nll",
        "temperature": float(attribute_temperature),
        "macro_auroc": float(np.nanmean([item["auroc"] for item in binary])),
        "macro_ap": float(np.nanmean([item["ap"] for item in binary])),
    }
    result.update(hard_prediction_metrics(evaluation_labels, evaluation_hard))
    semantic = semantic_query_metrics(distances, ids, distance_chunk_size)
    result["semantic_top1"] = semantic["semantic_query_top1"]
    result["semantic_mean_rank"] = semantic["semantic_query_mean_rank"]

    domain_names = tuple(domain_query_indices_by_name)
    domain_summaries: dict[str, dict[str, float]] = {}
    for domain in domain_names:
        domain_gallery_indices = (
            np.flatnonzero(domain_names_in_table == domain).astype(np.int64)
            if unseen_domain is not None
            else all_gallery_indices
        )
        summary = summarize(domain_query_indices_by_name[domain], domain_gallery_indices)
        domain_summaries[domain] = summary
        safe_name = domain.replace("-", "_")
        for key, value in summary.items():
            result[f"domain_{safe_name}_{key}"] = value

    result["domain_macro_rank1"] = float(
        np.mean([summary["rank1"] for summary in domain_summaries.values()])
    )
    result["domain_macro_mADM"] = float(
        np.mean([summary["mADM"] for summary in domain_summaries.values()])
    )
    if unseen_domain is not None:
        source_domains = [domain for domain in domain_names if domain != unseen_domain]
        unseen_summary = summarize(
            domain_query_indices_by_name[unseen_domain],
            np.flatnonzero(domain_names_in_table == unseen_domain).astype(np.int64),
        )
        result["unseen_domain"] = unseen_domain
        result["unseen_rank1"] = unseen_summary["rank1"]
        result["unseen_mADM"] = unseen_summary["mADM"]
        result["unseen_queries"] = unseen_summary["queries"]
        result["unseen_gallery_images"] = unseen_summary["gallery_images"]
        result["seen_macro_rank1"] = float(
            np.mean([domain_summaries[domain]["rank1"] for domain in source_domains])
        )
        result["seen_macro_mADM"] = float(
            np.mean([domain_summaries[domain]["mADM"] for domain in source_domains])
        )
    elif exclude_domain is not None:
        result["seen_macro_mADM"] = result["domain_macro_mADM"]
    worst_domain = min(domain_summaries, key=lambda name: domain_summaries[name]["mADM"])
    result["worst_domain_mADM"] = float(domain_summaries[worst_domain]["mADM"])
    result["worst_domain"] = worst_domain
    return result


@torch.inference_mode()
def encode_gallery(model: Any, paths: Sequence[str], roots: Sequence[Path], transform: Any,
                   device: torch.device, batch_size: int, num_workers: int,
                   amp: bool, progress_every: int = 0,
                   progress_prefix: str = "[AttriVision inference]") -> torch.Tensor:
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    loader = DataLoader(
        ImagePathDataset(paths, roots, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks = []
    processed = 0
    started = time.perf_counter()
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            features = model.encode_image(images.to(device, non_blocking=True))
        chunks.append(features.float().cpu())
        processed += len(images)
        if progress_every and (batch_index % progress_every == 0 or batch_index == len(loader)):
            elapsed = time.perf_counter() - started
            print(
                f"{progress_prefix} batch {batch_index}/{len(loader)}: "
                f"{processed}/{len(paths)} images, {elapsed:.1f}s, "
                f"{processed / max(elapsed, 1e-9):.1f} images/s",
                flush=True,
            )
    return torch.cat(chunks) if chunks else torch.empty(0, 512)


@torch.inference_mode()
def encode_queries(model: Any, queries: np.ndarray, attribute_names: Sequence[str],
                   device: torch.device, amp: bool,
                   prompt_mode: str = "category_complete") -> torch.Tensor:
    if prompt_mode == "category_complete":
        mapper = CategoryPromptMapper(attribute_names)
        prompts = mapper.prompts
        query_weights = mapper.encode(torch.as_tensor(queries, dtype=torch.float32, device=device))
    elif prompt_mode == "binary_positive":
        prompts = prompts_for_attributes(attribute_names)
        query_weights = torch.as_tensor(queries, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    tokens = model.tokenize(prompts).to(device)
    model.eval()
    with autocast(device, amp):
        attribute_features = model.encode_text(tokens)
    query_weights = query_weights.to(attribute_features.dtype)
    counts = query_weights.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Every retrieval query needs at least one semantic state")
    query_features = F.normalize(query_weights @ attribute_features / counts, dim=-1)
    return query_features.float().cpu()


def _rank_at_k(distances: np.ndarray, gallery_ids: np.ndarray, k: int) -> float:
    order = np.argsort(distances, axis=1, kind="stable")[:, :min(k, distances.shape[1])]
    query_ids = np.arange(distances.shape[0], dtype=np.int64)[:, None]
    return float(np.mean(np.any(gallery_ids[order] == query_ids, axis=1)))


def evaluate_abpr(model: Any, model_attribute_names: Sequence[str], data_root: Path,
                  transform: Any, device: torch.device, batch_size: int,
                  num_workers: int, amp: bool,
                  max_val_samples: int | None = None,
                  prompt_mode: str = "category_complete",
                  retrieval_scoring: str = "cosine_set",
                  attribute_temperature: float | None = None) -> dict[str, Any]:
    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = AnnotationTable(table.image_paths[:count], table.labels[:count], table.attribute_names)
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
    if retrieval_scoring == "paired_l1":
        prompt_features = encode_prompt_pairs(model, model_attribute_names, device, amp)
        inverse_temperature = (
            1.0 / attribute_temperature
            if attribute_temperature is not None
            else learned_inverse_temperature(model)
        )
        probabilities = paired_attribute_probabilities(
            gallery, prompt_features, inverse_temperature,
        )
        distances = l1_attribute_distances(queries, probabilities)
    elif retrieval_scoring == "cosine_set":
        query_features = encode_queries(
            model, queries, model_attribute_names, device, amp, prompt_mode,
        )
        distances = -(query_features @ gallery.T).numpy().astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown retrieval scoring: {retrieval_scoring}")
    if distances.shape != (len(queries), len(table.image_paths)) or not np.isfinite(distances).all():
        raise RuntimeError(f"Invalid distance matrix: {distances.shape}")

    official = official_retrieval_metrics(distances, queries, labels, ids)
    result: dict[str, Any] = {
        "images": float(len(table.image_paths)),
        "queries": float(len(queries)),
        "rank1": official["Rank-1"],
        "rank5": _rank_at_k(distances, ids, 5),
        "rank10": _rank_at_k(distances, ids, 10),
        "map": official["mAP"],
        "mADM": official["mADM"],
        "mINP": official["mINP"],
        "semantic_top1": official["semantic_top1"],
        "retrieval_scoring": retrieval_scoring,
    }
    if retrieval_scoring == "paired_l1":
        per_attribute = [
            _binary_metric(labels[:, index], probabilities[:, index])
            for index in range(labels.shape[1])
        ]
        result.update({
            "macro_auroc": float(np.nanmean([item["auroc"] for item in per_attribute])),
            "macro_ap": float(np.nanmean([item["ap"] for item in per_attribute])),
            "macro_f1": float(np.nanmean([item["f1"] for item in per_attribute])),
        })
    return result


def print_evaluation(metrics: dict[str, Any], checkpoint: str) -> None:
    print("=" * 44)
    print("AttriVision UPAR Task2 Validation")
    print("=" * 44)
    print(f"Images       : {int(metrics['images'])}")
    print(f"Queries      : {int(metrics['queries'])}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Scoring      : {metrics.get('retrieval_scoring', 'cosine_set')}")
    print(f"Rank-1       : {100 * metrics['rank1']:.2f} %")
    print(f"Rank-5       : {100 * metrics['rank5']:.2f} %")
    print(f"Rank-10      : {100 * metrics['rank10']:.2f} %")
    print(f"mAP          : {100 * metrics['map']:.2f} %")
    if "mADM" in metrics:
        print(f"mADM         : {100 * metrics['mADM']:.2f} %")
    print("=" * 44)
