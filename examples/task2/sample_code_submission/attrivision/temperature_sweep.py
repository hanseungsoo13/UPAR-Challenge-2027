"""Sweep native 52-state category-softmax temperature without image inference.

The script reuses a cached gallery feature matrix and (optionally cached) native
52-state text features.  Only the category softmax, 52-to-40 projection, L1
distances, and validation metrics are recomputed for each temperature.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    l1_attribute_distances, load_retrieval_annotations, reorder_columns,
    retrieval_metrics,
)
from attrivision.compare_inference import (  # noqa: E402
    encode_texts, prediction_metrics, ranks_at_k, semantic_query_metrics,
)


DEFAULT_TEMPERATURES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep native 52-state category-softmax temperature using cached features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="outputs/attrivision_category/checkpoint_best.pth")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--gallery-features", help="cached [gallery,dim] image features")
    parser.add_argument("--native-text-features", help="cached [52,dim] native text features")
    parser.add_argument("--output-dir", help="default: <checkpoint parent>/temperature_sweep")
    parser.add_argument("--device", default="auto", help="used only for loading/encoding text features")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--temperatures", nargs="+", type=float, default=list(DEFAULT_TEMPERATURES))
    return parser


def native_category_probabilities(
    gallery_features: np.ndarray,
    native_features: np.ndarray,
    mapper: CategoryPromptMapper,
    temperature: float,
) -> np.ndarray:
    """Apply exact training-state grouping and project positive states to 40-D."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = np.asarray(gallery_features, dtype=np.float32) @ np.asarray(native_features, dtype=np.float32).T
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    probabilities = np.zeros_like(scores, dtype=np.float32)
    positive_state: dict[str, str] = {}
    groups: list[list[str]] = []
    for _, columns, state_keys, fallback_key in mapper._MULTI_GROUPS:
        groups.append([*state_keys, fallback_key])
        positive_state.update(zip(columns, state_keys))
    for column, positive_key, negative_key in mapper._BINARY_GROUPS:
        groups.append([positive_key, negative_key])
        positive_state[column] = positive_key
    covered = [state for group in groups for state in group]
    if len(covered) != 52 or set(covered) != set(mapper.keys) or len(set(covered)) != 52:
        raise RuntimeError("Native category groups do not cover the exact 52 training states")
    for keys in groups:
        indices = [state_index[key] for key in keys]
        logits = scores[:, indices].astype(np.float64) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities[:, indices] = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)
    output_indices = [state_index[positive_state[name]] for name in mapper.attribute_names]
    result = probabilities[:, output_indices]
    if result.shape[1] != 40 or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid projected probabilities: {result.shape}")
    return result


def evaluate_temperature(
    temperature: float, gallery: np.ndarray, native: np.ndarray, mapper: CategoryPromptMapper,
    queries: np.ndarray, labels: np.ndarray, ids: np.ndarray, chunk_size: int,
) -> dict[str, Any]:
    predictions = native_category_probabilities(gallery, native, mapper, temperature)
    distances = l1_attribute_distances(queries, predictions, chunk_size)
    rank1, mean_ap = retrieval_metrics(distances, ids)
    rank_k = ranks_at_k(distances, ids, (5, 10))
    prediction_summary, _ = prediction_metrics(labels, predictions)
    semantic = semantic_query_metrics(distances, ids, chunk_size)
    result: dict[str, Any] = {
        "temperature": temperature, "rank1": rank1, "rank5": rank_k[5],
        "rank10": rank_k[10], "map": mean_ap, **prediction_summary, **semantic,
    }
    del predictions, distances
    return result


def load_feature_cache(path: Path, expected_rows: int, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} cache: {path}\n"
            "Run compare_inference.py once to create gallery_features.npy; "
            "native text features may be encoded with this script."
        )
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or values.shape[0] != expected_rows or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} cache {path}: shape={values.shape}")
    return np.asarray(values, dtype=np.float32)


def print_results(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 118)
    print("Native 52-state category-softmax temperature sweep")
    print("=" * 118)
    print(f"{'T':>7s} {'R1':>8s} {'R5':>8s} {'R10':>8s} {'mAP':>8s} {'AUROC':>8s} {'AP':>8s} {'F1':>8s} {'SemTop1':>10s} {'MeanRank':>10s} {'Pred mean/std':>17s}")
    for row in results:
        print(
            f"{row['temperature']:7.2f} {100*row['rank1']:7.2f}% {100*row['rank5']:7.2f}% "
            f"{100*row['rank10']:7.2f}% {100*row['map']:7.2f}% {row['macro_auroc']:8.4f} "
            f"{row['macro_ap']:8.4f} {row['macro_f1']:8.4f} {100*row['top1']:9.2f}% "
            f"{row['mean_rank']:10.1f} {row['prediction_mean']:.4f}/{row['prediction_std']:.4f}"
        )
    for metric, label in (("rank1", "R1"), ("map", "mAP"), ("top1", "Semantic-query Top-1")):
        best = max(results, key=lambda item: item[metric])
        print(f"Best T by {label}: {best['temperature']:.2f} ({best[metric]:.6f})")


def main() -> None:
    args = build_parser().parse_args()
    if args.distance_chunk_size <= 0 or not args.temperatures:
        raise ValueError("distance chunk size and temperatures must be non-empty/positive")
    if any(value <= 0 for value in args.temperatures):
        raise ValueError("all temperatures must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "temperature_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = Path(args.gallery_features).resolve() if args.gallery_features else checkpoint.parent / "inference_comparison" / "gallery_features.npy"
    text_path = Path(args.native_text_features).resolve() if args.native_text_features else output_dir / "native_state_features.npy"
    data_root = Path(args.data_root).resolve()

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    attribute_names = list(payload["attribute_names"])
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Temperature sweep requires a category_complete checkpoint")
    queries = reorder_columns(queries_raw, query_names, attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, attribute_names)
    if not np.all(labels == queries[ids]):
        raise RuntimeError("Validation gallery IDs do not reproduce the 40-D ground truth")
    mapper = CategoryPromptMapper(attribute_names)
    gallery = load_feature_cache(gallery_path, len(table.image_paths), "gallery feature")
    if text_path.exists():
        native = load_feature_cache(text_path, 52, "native text feature")
        print(f"Reusing native text features: {text_path}")
    else:
        print(f"Encoding 52 native text states once (no image forward): {text_path}")
        native = encode_texts(model, mapper.prompts, device, args.amp).numpy()
        np.save(text_path, native)
    del model
    print(f"Checkpoint: {checkpoint}")
    print(f"Gallery cache: {gallery_path} ({gallery.shape})")
    print(f"Temperatures: {args.temperatures}")

    results = []
    for temperature in args.temperatures:
        result = evaluate_temperature(temperature, gallery, native, mapper, queries, labels, ids, args.distance_chunk_size)
        results.append(result)
        print(
            f"T={temperature:.2f}: R1={100*result['rank1']:.2f}% "
            f"R5={100*result['rank5']:.2f}% R10={100*result['rank10']:.2f}% "
            f"mAP={100*result['map']:.2f}% AUROC={result['macro_auroc']:.4f} "
            f"AP={result['macro_ap']:.4f} F1={result['macro_f1']:.4f} "
            f"SemTop1={100*result['top1']:.2f}% MeanRank={result['mean_rank']:.1f} "
            f"mean/std={result['prediction_mean']:.4f}/{result['prediction_std']:.4f}",
            flush=True,
        )
    print_results(results)
    with (output_dir / "temperature_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(checkpoint), "gallery_features": str(gallery_path), "results": results}, handle, indent=2)
    with (output_dir / "temperature_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(results[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved sweep artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
