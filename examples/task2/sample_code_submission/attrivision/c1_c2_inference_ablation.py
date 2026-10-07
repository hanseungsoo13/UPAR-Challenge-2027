"""Inference-only C1 category weighting and C2 state-prior ablations."""
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

from attrivision.b2_temperature_sweep import evaluate  # noqa: E402
from attrivision.checkpoint import _load  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.native52_retrieval_ablation import category_indices, category_softmax  # noqa: E402
from upar.config import REPOSITORY_ROOT  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import load_retrieval_annotations, reorder_columns  # noqa: E402


TEMPERATURE = 0.01
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.00)
EPSILON = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-checkpoint C1 weighting and C2 prior-correction ablations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="outputs/attrivision_ablation/A1/checkpoint_best.pth",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument(
        "--cache-dir", default="outputs/attrivision_ablation/A1/native52_retrieval_ablation",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/attrivision_ablation/A1/native52_retrieval_ablation/c1_c2_ablation",
    )
    parser.add_argument("--gallery-features")
    parser.add_argument("--native-text-features")
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser


def category_names(mapper: CategoryPromptMapper) -> list[str]:
    names = [name for name, _, _, _ in mapper._MULTI_GROUPS]
    names.extend(column for column, _, _ in mapper._BINARY_GROUPS)
    if len(names) != 12 or len(set(names)) != 12:
        raise RuntimeError("Expected 12 uniquely named categories")
    return names


def normalized_targets(values52: np.ndarray, indices: np.ndarray) -> np.ndarray:
    targets = values52[:, indices].astype(np.float64)
    totals = targets.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Every sample must have an active state in every category")
    return targets / totals


def training_priors(
    training52: np.ndarray, groups: Sequence[np.ndarray], epsilon: float,
) -> np.ndarray:
    priors = np.zeros(training52.shape[1], dtype=np.float64)
    for indices in groups:
        values = normalized_targets(training52, indices).sum(axis=0) + epsilon
        priors[indices] = values / values.sum()
    return priors


def normalize_category_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Invalid category weights: {values}")
    return values / values.mean()


def category_weight_rules(
    priors: np.ndarray, groups: Sequence[np.ndarray], epsilon: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    entropies = np.asarray([
        -np.sum(priors[indices] * np.log(np.clip(priors[indices], epsilon, 1.0)))
        for indices in groups
    ])
    state_counts = np.asarray([len(indices) for indices in groups], dtype=np.float64)
    rules = {
        "Uniform": np.ones(len(groups), dtype=np.float64),
        "InverseEntropy": 1.0 / (entropies + epsilon),
        "Information": np.maximum(np.log(state_counts) - entropies, epsilon),
        "Rarity": np.asarray([
            np.mean(1.0 / np.clip(priors[indices], epsilon, 1.0))
            for indices in groups
        ]),
    }
    return {name: normalize_category_weights(values) for name, values in rules.items()}, entropies


def weighted_category_nll(
    queries52: np.ndarray, probabilities52: np.ndarray,
    groups: Sequence[np.ndarray], weights: np.ndarray,
    priors: np.ndarray, prior_lambda: float, epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.zeros((len(queries52), len(probabilities52)), dtype=np.float32)
    query_offsets = np.zeros(len(queries52), dtype=np.float64)
    log_probabilities = np.log(np.clip(probabilities52, epsilon, 1.0))
    log_priors = np.log(np.clip(priors, epsilon, 1.0))
    for weight, indices in zip(weights, groups):
        targets = normalized_targets(queries52, indices)
        distances -= weight * (targets @ log_probabilities[:, indices].T)
        query_offsets += weight * (targets @ log_priors[indices])
    scale = float(weights.sum())
    distances /= scale
    query_offsets *= prior_lambda / scale
    # C2 is a query-dependent constant across every gallery item. It is added
    # explicitly here while preserving the exact requested scoring equation.
    distances += query_offsets[:, None].astype(np.float32)
    return distances, query_offsets


def result_row(
    method: str, weighting: str, prior_lambda: float,
    metrics: dict[str, float], baseline: dict[str, float],
) -> dict[str, Any]:
    return {
        "method": method, "weighting": weighting, "lambda": prior_lambda,
        **metrics,
        "delta_rank1": metrics["rank1"] - baseline["rank1"],
        "delta_map": metrics["map"] - baseline["map"],
    }


def print_priors(
    names: Sequence[str], mapper: CategoryPromptMapper,
    groups: Sequence[np.ndarray], priors: np.ndarray,
) -> None:
    print("Training-set category state priors")
    for name, indices in zip(names, groups):
        states = ", ".join(
            f"{mapper.keys[index]}={priors[index]:.8f}" for index in indices
        )
        print(f"{name}: {states}")


def print_weights(names: Sequence[str], rules: dict[str, np.ndarray]) -> None:
    print("\nC1 category weights (each rule has mean 1)")
    for rule, weights in rules.items():
        values = ", ".join(f"{name}={weight:.8f}" for name, weight in zip(names, weights))
        print(f"{rule}: {values}")


def print_results(rows: Sequence[dict[str, Any]]) -> None:
    print("\nMethod | Weighting | Lambda | R1 | R5 | R10 | mAP | SemTop1 | MeanRank | MedianRank | ΔR1 | ΔmAP")
    print("--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        print(
            f"{row['method']} | {row['weighting']} | {row['lambda']:.2f} | "
            f"{row['rank1']:.6f} | {row['rank5']:.6f} | {row['rank10']:.6f} | "
            f"{row['map']:.6f} | {row['semantic_top1']:.6f} | "
            f"{row['mean_correct_query_rank']:.6f} | "
            f"{row['median_correct_query_rank']:.6f} | "
            f"{row['delta_rank1']:+.6f} | {row['delta_map']:+.6f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.distance_chunk_size <= 0:
        raise ValueError("distance-chunk-size must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    gallery_path = (Path(args.gallery_features).resolve() if args.gallery_features
                    else cache_dir / "gallery_features.npy")
    text_path = (Path(args.native_text_features).resolve() if args.native_text_features
                 else cache_dir / "native52_text_features.npy")
    missing = [str(path) for path in (checkpoint, gallery_path, text_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Inference-only ablation requires existing files: " + ", ".join(missing))

    payload = _load(checkpoint, map_location="cpu")
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Checkpoint must use category_complete Native52 prompts")
    attribute_names = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attribute_names)
    groups = category_indices(mapper)
    names = category_names(mapper)

    train_path = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_path)
    train40 = reorder_columns(train_table.labels, train_table.attribute_names, attribute_names)
    training52 = mapper.encode(torch.as_tensor(train40)).cpu().numpy().astype(np.float32)
    priors = training_priors(training52, groups, EPSILON)
    rules, entropies = category_weight_rules(priors, groups, EPSILON)
    del training52, train40

    val_path = find_annotation_file(data_root, "val")
    val_table = read_gt_csv(val_path)
    queries_raw, ids, query_names = load_retrieval_annotations(val_path.parent, val_table)
    queries40 = reorder_columns(queries_raw, query_names, attribute_names)
    queries52 = mapper.encode(torch.as_tensor(queries40)).cpu().numpy().astype(np.float32)

    probability_path = cache_dir / "category_probabilities_T0.01.npy"
    if probability_path.is_file():
        probabilities52 = np.load(probability_path, allow_pickle=False).astype(np.float32, copy=False)
        if probabilities52.shape != (len(val_table.image_paths), 52):
            raise ValueError(f"Cached probability shape mismatch: {probabilities52.shape}")
    else:
        gallery = np.load(gallery_path, allow_pickle=False).astype(np.float32, copy=False)
        text = np.load(text_path, allow_pickle=False).astype(np.float32, copy=False)
        if gallery.shape != (len(val_table.image_paths), 512) or text.shape != (52, 512):
            raise ValueError(f"Feature cache mismatch: gallery={gallery.shape}, text={text.shape}")
        probabilities52 = category_softmax((gallery @ text.T) / TEMPERATURE, groups)
        np.save(probability_path, probabilities52)
        del gallery, text
    if not np.isfinite(probabilities52).all():
        raise ValueError("Category probabilities contain NaN or Inf")

    evaluated: dict[str, tuple[dict[str, float], np.ndarray]] = {}
    for rule, weights in rules.items():
        distances, offsets = weighted_category_nll(
            queries52, probabilities52, groups, weights, priors, 0.0, EPSILON,
        )
        evaluated[rule] = (
            evaluate(distances, ids, args.distance_chunk_size), offsets,
        )
        del distances
    baseline = evaluated["Uniform"][0]
    best_rule = max(
        rules, key=lambda rule: (
            evaluated[rule][0]["map"], evaluated[rule][0]["rank1"],
            -list(rules).index(rule),
        ),
    )

    rows: list[dict[str, Any]] = []
    for rule in rules:
        rows.append(result_row("C1", rule, 0.0, evaluated[rule][0], baseline))
    # The correction term depends on the query but not the gallery image, so
    # every lambda preserves ranking exactly. Reuse the already evaluated
    # ranking metrics and retain the calculated offsets in summary metadata.
    for prior_lambda in LAMBDAS:
        rows.append(result_row("C2", "Uniform", prior_lambda, baseline, baseline))
    for prior_lambda in LAMBDAS:
        rows.append(result_row(
            "C1+C2", best_rule, prior_lambda, evaluated[best_rule][0], baseline,
        ))

    print_priors(names, mapper, groups, priors)
    print_weights(names, rules)
    print_results(rows)
    print(f"\nSelected C1 rule for C1+C2: {best_rule} (mAP first, R1 tie-break)")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint), "train_annotations": str(train_path),
        "validation": str(val_path), "gallery_features": str(gallery_path),
        "native_text_features": str(text_path),
        "category_probabilities": str(probability_path), "temperature": TEMPERATURE,
        "epsilon": EPSILON, "images": len(val_table.image_paths),
        "queries": len(queries52), "selected_c1_rule": best_rule,
        "selection_rule": "highest mAP among four prespecified C1 rules; R1 tie-break",
        "rarity_definition": "mean_s(1 / smoothed empirical state prior)",
        "prior_definition": "mean training target after within-category active-state normalization",
        "c2_ranking_invariance": (
            "lambda * sum_s(y_s log pi_s) is constant across gallery images for each query"
        ),
        "category_names": names, "category_entropies": entropies.tolist(),
        "state_keys": mapper.keys, "state_priors": priors.tolist(),
        "category_weights": {name: value.tolist() for name, value in rules.items()},
        "results": rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    fields = list(rows[0])
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
