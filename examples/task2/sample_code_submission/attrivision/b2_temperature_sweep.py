"""Cache-only temperature sweep for the fixed B2 Category-NLL scorer."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import _load  # noqa: E402
from attrivision.compare_inference import ranks_at_k  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.evaluate_native52_hard import semantic_query_metrics  # noqa: E402
from attrivision.native52_retrieval_ablation import (  # noqa: E402
    category_indices, category_nll, category_softmax,
)
from upar.config import REPOSITORY_ROOT  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    load_retrieval_annotations, reorder_columns, retrieval_metrics,
)


TEMPERATURES = (0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100)
BASELINE_TEMPERATURE = 0.01019117


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep only B2 Category-NLL temperature using fixed cached features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/attrivision_ablation/A1/checkpoint_best.pth",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument(
        "--cache-dir",
        default="outputs/attrivision_ablation/A1/native52_retrieval_ablation",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/attrivision_ablation/A1/native52_retrieval_ablation/b2_temperature_sweep",
    )
    parser.add_argument("--gallery-features")
    parser.add_argument("--native-text-features")
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser


def evaluate(distances: np.ndarray, ids: np.ndarray, chunk_size: int) -> dict[str, float]:
    rank1, mean_ap = retrieval_metrics(distances, ids)
    ranks = ranks_at_k(distances, ids, (5, 10))
    semantic = semantic_query_metrics(distances, ids, chunk_size)
    return {
        "rank1": rank1, "rank5": ranks[5], "rank10": ranks[10], "map": mean_ap,
        "semantic_top1": semantic["semantic_query_top1"],
        "mean_correct_query_rank": semantic["semantic_query_mean_rank"],
        "median_correct_query_rank": semantic["semantic_query_median_rank"],
    }


def score_temperature(
    temperature: float, similarities: np.ndarray, queries52: np.ndarray,
    groups: list[np.ndarray], ids: np.ndarray, chunk_size: int,
) -> dict[str, float]:
    probabilities = category_softmax(similarities / temperature, groups)
    distances = category_nll(queries52, probabilities, groups)
    result = evaluate(distances, ids, chunk_size)
    result["temperature"] = temperature
    return result


def metric_deltas(result: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        f"delta_{key}": result[key] - baseline[key]
        for key in (
            "rank1", "rank5", "rank10", "map", "semantic_top1",
            "mean_correct_query_rank", "median_correct_query_rank",
        )
    }


def print_table(results: list[dict[str, float]]) -> None:
    print("T | R1 | R5 | R10 | mAP | SemTop1 | MeanRank | MedianRank")
    print("--- | --- | --- | --- | --- | --- | --- | ---")
    for row in results:
        print(
            f"{row['temperature']:.4f} | {row['rank1']:.6f} | {row['rank5']:.6f} | "
            f"{row['rank10']:.6f} | {row['map']:.6f} | "
            f"{row['semantic_top1']:.6f} | {row['mean_correct_query_rank']:.6f} | "
            f"{row['median_correct_query_rank']:.6f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.distance_chunk_size <= 0:
        raise ValueError("distance-chunk-size must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    gallery_path = (
        Path(args.gallery_features).resolve()
        if args.gallery_features else cache_dir / "gallery_features.npy"
    )
    text_path = (
        Path(args.native_text_features).resolve()
        if args.native_text_features else cache_dir / "native52_text_features.npy"
    )
    missing = [str(path) for path in (checkpoint, gallery_path, text_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cache-only sweep will not encode or train anything. Missing: " + ", ".join(missing)
        )

    payload = _load(checkpoint, map_location="cpu")
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Checkpoint must use category_complete Native52 prompts")
    attribute_names = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attribute_names)
    groups = category_indices(mapper)

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    queries40 = reorder_columns(queries_raw, query_names, attribute_names)
    queries52 = mapper.encode(torch.as_tensor(queries40)).cpu().numpy().astype(np.float32)

    gallery = np.load(gallery_path, allow_pickle=False).astype(np.float32, copy=False)
    text = np.load(text_path, allow_pickle=False).astype(np.float32, copy=False)
    if gallery.shape != (len(table.image_paths), 512):
        raise ValueError(f"Expected cached gallery [{len(table.image_paths)},512], got {gallery.shape}")
    if text.shape != (52, 512):
        raise ValueError(f"Expected cached Native52 text features [52,512], got {text.shape}")
    if not np.isfinite(gallery).all() or not np.isfinite(text).all():
        raise ValueError("Cached features contain NaN or Inf")
    similarities = gallery @ text.T
    del gallery, text

    baseline = score_temperature(
        BASELINE_TEMPERATURE, similarities, queries52, groups, ids,
        args.distance_chunk_size,
    )
    results = [
        score_temperature(
            temperature, similarities, queries52, groups, ids,
            args.distance_chunk_size,
        )
        for temperature in TEMPERATURES
    ]
    for result in results:
        result.update(metric_deltas(result, baseline))
    best_r1 = max(results, key=lambda row: (row["rank1"], row["map"], -row["temperature"]))
    best_map = max(results, key=lambda row: (row["map"], row["rank1"], -row["temperature"]))

    print_table(results)
    print(
        f"Best R1 temperature: T={best_r1['temperature']:.4f}, "
        f"R1={best_r1['rank1']:.6f}, "
        f"delta_vs_T={BASELINE_TEMPERATURE:.8f}: {best_r1['delta_rank1']:+.6f}"
    )
    print(
        f"Best mAP temperature: T={best_map['temperature']:.4f}, "
        f"mAP={best_map['map']:.6f}, "
        f"delta_vs_T={BASELINE_TEMPERATURE:.8f}: {best_map['delta_map']:+.6f}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint), "validation": str(gt_path),
        "gallery_features": str(gallery_path), "native_text_features": str(text_path),
        "images": len(table.image_paths), "queries": len(queries40),
        "temperatures": list(TEMPERATURES),
        "baseline_temperature": BASELINE_TEMPERATURE,
        "baseline": baseline, "results": results,
        "best_rank1": best_r1, "best_map": best_map,
        "category_weighting": "uniform mean over 12 categories",
        "learned_or_fitted_parameters": False,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    fields = list(results[0])
    with (output_dir / "temperature_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
