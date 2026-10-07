"""Inference-only Native52 retrieval scoring ablation for a fixed checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.compare_inference import encode_texts, native_state_outputs, ranks_at_k  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.engine.evaluator_abpr import encode_gallery  # noqa: E402
from attrivision.engine.paired_attribute import learned_inverse_temperature  # noqa: E402
from attrivision.evaluate_native52_hard import semantic_query_metrics  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    l1_attribute_distances, load_retrieval_annotations, reorder_columns,
    retrieval_metrics,
)


METHOD_NAMES = {
    "B0": "Native40-L1",
    "B1a": "Native52-L1",
    "B1b": "CategoryNorm52-L1",
    "B2": "Category-NLL",
    "B3": "Category-Margin",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fixed Native52 retrieval scoring functions without training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/attrivision_ablation/A1/checkpoint_best.pth",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument(
        "--output-dir",
        default="outputs/attrivision_ablation/A1/native52_retrieval_ablation",
    )
    parser.add_argument("--gallery-features")
    parser.add_argument("--native-text-features")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attribute-temperature", type=float)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-val-samples", type=int, help="smoke testing only")
    return parser


def category_indices(mapper: CategoryPromptMapper) -> list[np.ndarray]:
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    groups: list[np.ndarray] = []
    for _, _, state_keys, fallback_key in mapper._MULTI_GROUPS:
        groups.append(np.asarray([state_index[key] for key in (*state_keys, fallback_key)]))
    for _, positive_key, negative_key in mapper._BINARY_GROUPS:
        groups.append(np.asarray([state_index[positive_key], state_index[negative_key]]))
    covered = np.concatenate(groups)
    if len(groups) != 12 or len(covered) != 52 or len(np.unique(covered)) != 52:
        raise RuntimeError("Category groups must partition all 52 states into 12 categories")
    return groups


def category_softmax(logits: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float32)
    for indices in groups:
        values = logits[:, indices].astype(np.float64)
        values -= values.max(axis=1, keepdims=True)
        values = np.exp(values)
        probabilities[:, indices] = (values / values.sum(axis=1, keepdims=True)).astype(np.float32)
    return probabilities


def native52_l1(
    queries52: np.ndarray, probabilities52: np.ndarray,
    groups: Sequence[np.ndarray], normalized: bool,
) -> np.ndarray:
    result = np.zeros((len(queries52), len(probabilities52)), dtype=np.float32)
    for indices in groups:
        block = np.zeros_like(result)
        for index in indices:
            block += np.abs(
                queries52[:, index, None].astype(np.float32)
                - probabilities52[None, :, index]
            )
        result += block / len(indices) if normalized else block
    if normalized:
        result /= len(groups)
    return result


def category_nll(
    queries52: np.ndarray, probabilities52: np.ndarray,
    groups: Sequence[np.ndarray], epsilon: float = 1e-12,
) -> np.ndarray:
    result = np.zeros((len(queries52), len(probabilities52)), dtype=np.float32)
    log_probabilities = np.log(np.clip(probabilities52, epsilon, 1.0))
    for indices in groups:
        targets = queries52[:, indices].astype(np.float32)
        counts = targets.sum(axis=1, keepdims=True)
        if np.any(counts == 0):
            raise ValueError("Every query category must contain at least one active state")
        targets /= counts
        result -= targets @ log_probabilities[:, indices].T
    result /= len(groups)
    return result


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def category_margin_distances(
    queries52: np.ndarray, logits52: np.ndarray, groups: Sequence[np.ndarray],
) -> np.ndarray:
    scores = np.zeros((len(queries52), len(logits52)), dtype=np.float32)
    for indices in groups:
        masks = queries52[:, indices].astype(bool)
        if not masks.any(axis=1).all() or masks.all(axis=1).any():
            raise ValueError("Category margin requires non-empty positive and negative state sets")
        unique_masks, inverse = np.unique(masks, axis=0, return_inverse=True)
        for pattern_index, positive_mask in enumerate(unique_masks):
            positive = _logsumexp(logits52[:, indices[positive_mask]], axis=1)
            negative = _logsumexp(logits52[:, indices[~positive_mask]], axis=1)
            scores[inverse == pattern_index] += (positive - negative)[None, :]
    # The common evaluator expects smaller values to rank first.
    return -scores / len(groups)


def per_query_metrics(distances: np.ndarray, gallery_ids: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id, values in enumerate(distances):
        relevant = gallery_ids == query_id
        if not relevant.any() or relevant.all():
            raise ValueError(f"Query {query_id} lacks correct or wrong gallery examples")
        order = np.argsort(values, kind="stable")
        ordered_relevant = relevant[order]
        positions = np.flatnonzero(ordered_relevant) + 1
        average_precision = float(np.mean(np.arange(1, len(positions) + 1) / positions))
        best_correct = float(values[relevant].min())
        best_wrong = float(values[~relevant].min())
        rows.append({
            "query_id": query_id,
            "relevant_gallery_count": int(relevant.sum()),
            "rank1": int(ordered_relevant[:1].any()),
            "rank5": int(ordered_relevant[:5].any()),
            "rank10": int(ordered_relevant[:10].any()),
            "average_precision": average_precision,
            "best_correct_distance": best_correct,
            "best_wrong_distance": best_wrong,
            "correct_wrong_margin": best_wrong - best_correct,
        })
    return rows


def semantic_ranks_by_query(
    distances: np.ndarray, gallery_ids: np.ndarray, chunk_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    query_indices = np.arange(distances.shape[0], dtype=np.int64)[:, None]
    rank_chunks: list[np.ndarray] = []
    for start in range(0, distances.shape[1], chunk_size):
        stop = min(start + chunk_size, distances.shape[1])
        block = distances[:, start:stop]
        correct_ids = gallery_ids[start:stop]
        correct = block[correct_ids, np.arange(stop - start)]
        ranks = 1 + np.sum(
            (block < correct[None, :])
            | ((block == correct[None, :]) & (query_indices < correct_ids[None, :])),
            axis=0,
        )
        rank_chunks.append(ranks.astype(np.int64))
    all_ranks = np.concatenate(rank_chunks)
    rows = []
    for query_id in range(distances.shape[0]):
        values = all_ranks[gallery_ids == query_id]
        rows.append({
            "semantic_top1": float(np.mean(values == 1)),
            "correct_query_mean_rank": float(values.mean()),
            "correct_query_median_rank": float(np.median(values)),
        })
    return all_ranks, rows


def evaluate_method(
    key: str, distances: np.ndarray, gallery_ids: np.ndarray,
    output_dir: Path, chunk_size: int,
) -> dict[str, Any]:
    if distances.shape != (int(gallery_ids.max()) + 1, len(gallery_ids)):
        raise ValueError(f"Invalid {key} distance shape: {distances.shape}")
    if not np.isfinite(distances).all():
        raise RuntimeError(f"{key} produced NaN or Inf distances")
    rank1, mean_ap = retrieval_metrics(distances, gallery_ids)
    ranks = ranks_at_k(distances, gallery_ids, (5, 10))
    semantic = semantic_query_metrics(distances, gallery_ids, chunk_size)
    query_rows = per_query_metrics(distances, gallery_ids)
    _, semantic_rows = semantic_ranks_by_query(distances, gallery_ids, chunk_size)
    for row, semantic_row in zip(query_rows, semantic_rows):
        row.update(semantic_row)
        row["method"] = key
        row["method_name"] = METHOD_NAMES[key]
    fields = list(query_rows[0])
    with (output_dir / f"{key}_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_rows)
    return {
        "method": METHOD_NAMES[key], "rank1": rank1,
        "rank5": ranks[5], "rank10": ranks[10], "map": mean_ap,
        "semantic_top1": semantic["semantic_query_top1"],
        "mean_correct_query_rank": semantic["semantic_query_mean_rank"],
        "median_correct_query_rank": semantic["semantic_query_median_rank"],
        "mean_correct_wrong_margin": float(np.mean([
            row["correct_wrong_margin"] for row in query_rows
        ])),
    }


def print_table(results: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print(f"{'Method':30s} {'R1':>8s} {'R5':>8s} {'R10':>8s} {'mAP':>8s} "
          f"{'SemTop1':>9s} {'MeanRank':>11s} {'MedianRank':>12s} {'CWMargin':>11s}")
    print("-" * 112)
    for key, result in results.items():
        label = f"{key} {result['method']}"
        print(f"{label:30s} {100*result['rank1']:7.2f}% {100*result['rank5']:7.2f}% "
              f"{100*result['rank10']:7.2f}% {100*result['map']:7.2f}% "
              f"{100*result['semantic_top1']:8.2f}% "
              f"{result['mean_correct_query_rank']:11.3f} "
              f"{result['median_correct_query_rank']:12.3f} "
              f"{result['mean_correct_wrong_margin']:11.6f}")
    print("=" * 112)
    baseline = results["B0"]
    print("\nAbsolute deltas versus B0 (method minus B0):")
    for key, result in results.items():
        print(f"{key:3s}: ΔR1={result['rank1']-baseline['rank1']:+.6f}, "
              f"ΔmAP={result['map']-baseline['map']:+.6f}, "
              f"ΔMeanRank={result['mean_correct_query_rank']-baseline['mean_correct_query_rank']:+.6f}")


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.distance_chunk_size <= 0:
        raise ValueError("Batch/chunk sizes must be positive and num-workers non-negative")
    if args.attribute_temperature is not None and args.attribute_temperature <= 0:
        raise ValueError("attribute-temperature must be positive")

    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Checkpoint must use the exact category_complete training vocabulary")

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if args.max_val_samples is not None:
        count = min(args.max_val_samples, len(table.image_paths))
        table = type(table)(table.image_paths[:count], table.labels[:count], table.attribute_names)
        queries_raw, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    attribute_names = list(payload["attribute_names"])
    queries40 = reorder_columns(queries_raw, query_names, attribute_names)
    if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
        raise RuntimeError("Checkpoint must contain the exact 40 official attributes")

    mapper = CategoryPromptMapper(attribute_names)
    queries52 = mapper.encode(torch.as_tensor(queries40)).cpu().numpy().astype(np.float32)
    groups = category_indices(mapper)
    inverse_temperature = (
        1.0 / args.attribute_temperature
        if args.attribute_temperature is not None
        else learned_inverse_temperature(model)
    )

    gallery_path = (Path(args.gallery_features).resolve() if args.gallery_features
                    else output_dir / "gallery_features.npy")
    if gallery_path.is_file():
        gallery_array = np.load(gallery_path, allow_pickle=False)
        if gallery_array.shape[0] != len(table.image_paths):
            raise ValueError(f"Gallery cache/data mismatch: {gallery_array.shape}")
        gallery_features = torch.from_numpy(gallery_array.astype(np.float32, copy=False))
        print(f"Reusing gallery cache: {gallery_path}")
    else:
        gallery_features = encode_gallery(
            model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(args.image_size), device, args.eval_batch_size,
            args.num_workers, args.amp, args.progress_every, "[shared gallery]",
        )
        np.save(gallery_path, gallery_features.numpy())
        print(f"Saved gallery cache: {gallery_path}")

    text_path = (Path(args.native_text_features).resolve() if args.native_text_features
                 else output_dir / "native52_text_features.npy")
    if text_path.is_file():
        text_array = np.load(text_path, allow_pickle=False)
        if text_array.shape != (52, gallery_features.shape[1]):
            raise ValueError(f"Native text cache/model mismatch: {text_array.shape}")
        text_features = torch.from_numpy(text_array.astype(np.float32, copy=False))
        print(f"Reusing Native52 text cache: {text_path}")
    else:
        text_features = encode_texts(model, mapper.prompts, device, args.amp)
        np.save(text_path, text_features.numpy())
        print(f"Saved Native52 text cache: {text_path}")

    similarities = (gallery_features.float() @ text_features.float().T).numpy()
    logits52 = similarities.astype(np.float32, copy=False) * inverse_temperature
    probabilities52 = category_softmax(logits52, groups)
    native40, _ = native_state_outputs(
        gallery_features, text_features, mapper, inverse_temperature,
    )
    del model, gallery_features, text_features, similarities
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: dict[str, dict[str, Any]] = {}
    methods = (
        ("B0", lambda: l1_attribute_distances(queries40, native40, args.distance_chunk_size)),
        ("B1a", lambda: native52_l1(queries52, probabilities52, groups, False)),
        ("B1b", lambda: native52_l1(queries52, probabilities52, groups, True)),
        ("B2", lambda: category_nll(queries52, probabilities52, groups)),
        ("B3", lambda: category_margin_distances(queries52, logits52, groups)),
    )
    for key, scorer in methods:
        print(f"Evaluating {key} {METHOD_NAMES[key]}...", flush=True)
        distances = scorer()
        results[key] = evaluate_method(
            key, distances, ids, output_dir, args.distance_chunk_size,
        )
        del distances

    print_table(results)
    deltas = {
        key: {
            "delta_rank1": value["rank1"] - results["B0"]["rank1"],
            "delta_map": value["map"] - results["B0"]["map"],
            "delta_mean_rank": (
                value["mean_correct_query_rank"]
                - results["B0"]["mean_correct_query_rank"]
            ),
        }
        for key, value in results.items()
    }
    b0_reproduction = None
    reference_path = checkpoint.parent / "validation_metrics.json"
    if args.max_val_samples is None and reference_path.is_file():
        with reference_path.open(encoding="utf-8") as handle:
            reference = json.load(handle)
        keys = ("rank1", "rank5", "rank10", "map", "semantic_top1")
        if all(key in reference for key in keys):
            b0_reproduction = {
                "reference": str(reference_path),
                "absolute_error": {
                    key: abs(float(results["B0"][key]) - float(reference[key]))
                    for key in keys
                },
            }
            print("\nB0 reproduction absolute errors: " + ", ".join(
                f"{key}={value:.3e}"
                for key, value in b0_reproduction["absolute_error"].items()
            ))
    summary = {
        "checkpoint": str(checkpoint), "validation": str(gt_path),
        "gallery_features": str(gallery_path), "native_text_features": str(text_path),
        "images": len(table.image_paths), "queries": len(queries40),
        "attributes": len(attribute_names), "native_states": len(mapper.keys),
        "temperature": 1.0 / inverse_temperature,
        "category_count": len(groups), "epsilon": 1e-12,
        "correct_wrong_margin_definition": "best_wrong_distance - best_correct_distance",
        "native_state_keys": mapper.keys, "native_state_prompts": mapper.prompts,
        "results": results, "deltas_vs_B0": deltas,
        "b0_reproduction": b0_reproduction,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    fields = (
        "key", "method", "rank1", "rank5", "rank10", "map", "semantic_top1",
        "mean_correct_query_rank", "median_correct_query_rank",
        "mean_correct_wrong_margin", "delta_rank1", "delta_map", "delta_mean_rank",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            writer.writerow({"key": key, **result, **deltas[key]})
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
