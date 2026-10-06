"""Evaluate native 52-state AttriVision inference as hard 40-bit PAR.

This script is inference-only. It reuses the exact category prompts and mapping
used by ``category_complete`` training, and requires precomputed gallery image
features. It compares the existing category-softmax L1 score with Hamming
retrieval over category-argmax hard labels.
"""
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
from attrivision.compare_inference import (  # noqa: E402
    encode_texts, native_state_outputs, ranks_at_k,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.engine.paired_attribute import learned_inverse_temperature  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    l1_attribute_distances, load_retrieval_annotations, reorder_columns,
    retrieval_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate native 52-state soft L1 and hard-Hamming retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="outputs/attrivision_category/checkpoint_best.pth",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument(
        "--gallery-features",
        help=("cached normalized [gallery, feature_dim] .npy; default: "
              "<checkpoint-dir>/inference_comparison/gallery_features.npy"),
    )
    parser.add_argument(
        "--output-dir",
        help="default: <checkpoint-dir>/native52_hard_hamming",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument(
        "--attribute-temperature", type=float,
        help="softmax T for Soft-L1; default: checkpoint learned CLIP temperature",
    )
    return parser


def hard_category_projection(
    gallery_features: torch.Tensor,
    state_features: torch.Tensor,
    mapper: CategoryPromptMapper,
) -> np.ndarray:
    """Category argmax over all 52 states, projected to ordered 40 bits."""
    scores = (gallery_features.float() @ state_features.float().T).numpy()
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    attribute_index = {name: index for index, name in enumerate(mapper.attribute_names)}
    hard = np.zeros((len(scores), 40), dtype=np.uint8)

    covered: list[str] = []
    for _, columns, state_keys, fallback_key in mapper._MULTI_GROUPS:
        keys = [*state_keys, fallback_key]
        covered.extend(keys)
        indices = np.asarray([state_index[key] for key in keys])
        winners = scores[:, indices].argmax(axis=1)
        # A positive state sets exactly its corresponding official bit. The
        # final fallback state (unknown/other/none) leaves this group all-zero.
        for offset, column in enumerate(columns):
            hard[:, attribute_index[column]] = winners == offset

    for column, positive_key, negative_key in mapper._BINARY_GROUPS:
        keys = [positive_key, negative_key]
        covered.extend(keys)
        indices = np.asarray([state_index[key] for key in keys])
        hard[:, attribute_index[column]] = scores[:, indices].argmax(axis=1) == 0

    if len(covered) != 52 or len(set(covered)) != 52 or set(covered) != set(mapper.keys):
        raise RuntimeError("Hard projection does not cover the exact 52 training states")
    return hard


def hamming_distances(queries: np.ndarray, hard: np.ndarray,
                      chunk_size: int) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.uint8)
    hard = np.asarray(hard, dtype=np.uint8)
    distances = np.empty((len(queries), len(hard)), dtype=np.uint8)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        distances[start:stop] = np.count_nonzero(
            queries[start:stop, None, :] != hard[None, :, :], axis=2,
        )
    return distances


def semantic_query_metrics(distances: np.ndarray, ids: np.ndarray,
                           chunk_size: int) -> dict[str, float]:
    """Rank each gallery image's correct query with stable query-index ties."""
    query_indices = np.arange(distances.shape[0], dtype=np.int64)[:, None]
    chunks: list[np.ndarray] = []
    for start in range(0, distances.shape[1], chunk_size):
        stop = min(start + chunk_size, distances.shape[1])
        block = distances[:, start:stop]
        correct_ids = ids[start:stop]
        correct = block[correct_ids, np.arange(stop - start)]
        ranks = 1 + np.sum(
            (block < correct[None, :])
            | ((block == correct[None, :]) & (query_indices < correct_ids[None, :])),
            axis=0,
        )
        chunks.append(ranks.astype(np.int64))
    ranks = np.concatenate(chunks)
    return {
        "semantic_query_top1": float(np.mean(ranks == 1)),
        "semantic_query_mean_rank": float(ranks.mean()),
        "semantic_query_median_rank": float(np.median(ranks)),
    }


def hard_prediction_metrics(labels: np.ndarray, hard: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.uint8)
    predicted = np.asarray(hard, dtype=np.uint8)
    errors = np.count_nonzero(truth != predicted, axis=1)
    tp = np.count_nonzero((truth == 1) & (predicted == 1), axis=1)
    fp = np.count_nonzero((truth == 0) & (predicted == 1), axis=1)
    fn = np.count_nonzero((truth == 1) & (predicted == 0), axis=1)
    # Standard instance metrics: a zero denominator contributes 0. This is
    # explicit so results remain reproducible even for all-zero predictions.
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
                   where=(2 * tp + fp + fn) > 0)
    return {
        "mean_hamming_error": float(errors.mean()),
        "mean_hamming_error_over_40": float(errors.mean() / 40.0),
        "median_hamming_error": float(np.median(errors)),
        "exact_match": float(np.mean(errors == 0)),
        "le1_bit_error": float(np.mean(errors <= 1)),
        "le2_bit_error": float(np.mean(errors <= 2)),
        "le5_bit_error": float(np.mean(errors <= 5)),
        "instance_precision": float(precision.mean()),
        "instance_recall": float(recall.mean()),
        "instance_f1": float(f1.mean()),
    }


def retrieval_result(distances: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    rank1, mean_ap = retrieval_metrics(distances, ids)
    rank_k = ranks_at_k(distances, ids, (5, 10))
    return {"rank1": rank1, "rank5": rank_k[5], "rank10": rank_k[10], "map": mean_ap}


def print_metrics(soft: dict[str, float], hard: dict[str, float]) -> None:
    print("\nHard 40-bit PAR prediction")
    print(f"Mean Hamming error / 40 : {hard['mean_hamming_error']:.4f} / 40 "
          f"({hard['mean_hamming_error_over_40']:.6f})")
    print(f"Median Hamming error    : {hard['median_hamming_error']:.1f}")
    print(f"Exact 40-bit match rate : {100 * hard['exact_match']:.2f}%")
    print(f"<=1 bit error rate      : {100 * hard['le1_bit_error']:.2f}%")
    print(f"<=2 bit error rate      : {100 * hard['le2_bit_error']:.2f}%")
    print(f"<=5 bit error rate      : {100 * hard['le5_bit_error']:.2f}%")
    print(f"Instance Precision      : {hard['instance_precision']:.6f}")
    print(f"Instance Recall         : {hard['instance_recall']:.6f}")
    print(f"Instance F1             : {hard['instance_f1']:.6f}")
    print(f"Hard-Hamming Rank-1     : {100 * hard['rank1']:.2f}%")
    print(f"Rank-5                  : {100 * hard['rank5']:.2f}%")
    print(f"Rank-10                 : {100 * hard['rank10']:.2f}%")
    print(f"mAP                     : {100 * hard['map']:.2f}%")
    print(f"Semantic-query Top-1    : {100 * hard['semantic_query_top1']:.2f}%")
    print(f"Correct-query rank mean : {hard['semantic_query_mean_rank']:.2f}")
    print(f"Correct-query rank med. : {hard['semantic_query_median_rank']:.1f}")

    print("\n" + "=" * 94)
    print(f"{'Method':28s} {'R1':>8s} {'R5':>8s} {'R10':>8s} {'mAP':>8s} "
          f"{'ExactMatch':>12s} {'MeanBitErr':>12s} {'InstanceF1':>12s}")
    print("-" * 94)
    for name, row, include_hard in (
        ("Native52 Soft-L1", soft, False),
        ("Native52 Hard-Hamming", hard, True),
    ):
        exact = f"{100 * row['exact_match']:.2f}%" if include_hard else "-"
        bit_error = f"{row['mean_hamming_error']:.4f}" if include_hard else "-"
        instance_f1 = f"{row['instance_f1']:.6f}" if include_hard else "-"
        print(f"{name:28s} {100*row['rank1']:7.2f}% {100*row['rank5']:7.2f}% "
              f"{100*row['rank10']:7.2f}% {100*row['map']:7.2f}% "
              f"{exact:>12s} {bit_error:>12s} {instance_f1:>12s}")
    print("=" * 94)


def main() -> None:
    args = build_parser().parse_args()
    if args.distance_chunk_size <= 0:
        raise ValueError("--distance-chunk-size must be positive")
    if args.attribute_temperature is not None and args.attribute_temperature <= 0:
        raise ValueError("--attribute-temperature must be positive")

    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir else
                  checkpoint.parent / "native52_hard_hamming")
    feature_path = (Path(args.gallery_features).resolve() if args.gallery_features else
                    checkpoint.parent / "inference_comparison" / "gallery_features.npy")
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Cached gallery feature file not found: {feature_path}\n"
            "Pass the existing cache with --gallery-features; this script will not re-encode images."
        )

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Checkpoint must use the exact category_complete 52-state training mode; "
                         f"got {payload.get('prompt_mode')!r}")
    attribute_names = list(payload["attribute_names"])
    queries = reorder_columns(queries_raw, query_names, attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, attribute_names)
    if not np.array_equal(labels, queries[ids]):
        raise RuntimeError("Gallery IDs do not reproduce the ordered 40-bit ground truth")

    feature_array = np.load(feature_path, allow_pickle=False)
    if (feature_array.ndim != 2 or feature_array.shape[0] != len(table.image_paths)
            or not np.isfinite(feature_array).all()):
        raise ValueError(f"Invalid cached gallery features: {feature_array.shape}")
    gallery_features = torch.from_numpy(feature_array.astype(np.float32, copy=False))

    mapper = CategoryPromptMapper(attribute_names)
    if len(mapper.keys) != 52 or len(mapper.prompts) != 52:
        raise RuntimeError("Expected the exact 52-state category vocabulary")
    state_features = encode_texts(model, mapper.prompts, device, args.amp)
    inverse_temperature = (1.0 / args.attribute_temperature
                           if args.attribute_temperature is not None
                           else learned_inverse_temperature(model))
    soft_predictions, _ = native_state_outputs(
        gallery_features, state_features, mapper, inverse_temperature,
    )
    hard_predictions = hard_category_projection(gallery_features, state_features, mapper)

    soft_distances = l1_attribute_distances(
        queries, soft_predictions, args.distance_chunk_size,
    )
    hard_distances = hamming_distances(
        queries, hard_predictions, args.distance_chunk_size,
    )
    soft_result = retrieval_result(soft_distances, ids)
    hard_result = hard_prediction_metrics(labels, hard_predictions)
    hard_result.update(retrieval_result(hard_distances, ids))
    hard_result.update(semantic_query_metrics(
        hard_distances, ids, args.distance_chunk_size,
    ))

    print(f"Checkpoint       : {checkpoint}")
    print(f"Gallery cache    : {feature_path}")
    print(f"Validation       : {gt_path}")
    print(f"Images / queries : {len(labels)} / {len(queries)}")
    print(f"52-state prompts : exact CategoryPromptMapper training vocabulary")
    print(f"Softmax T        : {1.0 / inverse_temperature:.8f}")
    print_metrics(soft_result, hard_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "native52_soft_predictions.npy", soft_predictions)
    np.save(output_dir / "native52_hard_predictions.npy", hard_predictions)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint), "gallery_features": str(feature_path),
        "validation": str(gt_path), "images": len(labels), "queries": len(queries),
        "temperature": 1.0 / inverse_temperature,
        "native_state_count": 52, "native_state_keys": mapper.keys,
        "native_state_prompts": mapper.prompts,
        "methods": {"Native52 Soft-L1": soft_result,
                    "Native52 Hard-Hamming": hard_result},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    fields = ("method", "rank1", "rank5", "rank10", "map", "exact_match",
              "mean_hamming_error", "instance_f1")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in (
            ("Native52 Soft-L1", soft_result),
            ("Native52 Hard-Hamming", hard_result),
        ):
            writer.writerow({field: (method if field == "method" else result.get(field, ""))
                             for field in fields})
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
