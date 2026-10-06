"""Diagnose the AttriVision Task 2 validation pipeline without retraining.

The model prediction path intentionally reuses ``encode_gallery`` and
``paired_attribute_probabilities`` from the production evaluator.  This keeps
the diagnostic probabilities bit-for-bit equivalent to paired-L1 retrieval.
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
from attrivision.datasets.attribute_prompts import (  # noqa: E402
    ATTRIBUTE_PROMPTS,
    CATEGORY_PROMPTS,
    NEGATIVE_ATTRIBUTE_PROMPTS,
    CategoryPromptMapper,
)
from attrivision.engine.evaluator_abpr import encode_gallery  # noqa: E402
from attrivision.engine.paired_attribute import (  # noqa: E402
    encode_prompt_pairs,
    learned_inverse_temperature,
    paired_attribute_probabilities,
)
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import load_retrieval_annotations, reorder_columns  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AttriVision Task 2 validation diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="outputs/attrivision_category/checkpoint_best.pth",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", help="default: <checkpoint parent>/diagnostics")
    parser.add_argument(
        "--predictions", help="reuse a saved [G,40] pred_probs.npy instead of image inference",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attribute-temperature", type=float,
        help="paired softmax T; default uses exp(checkpoint logit_scale) as 1/T",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--semantic-chunk-size", type=int, default=512)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for field in ("eval_batch_size", "image_size", "semantic_chunk_size"):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("--num-workers and --progress-every cannot be negative")
    if args.attribute_temperature is not None and args.attribute_temperature <= 0:
        raise ValueError("--attribute-temperature must be positive")


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Binary metrics with tie-correct AUROC and threshold-grouped AP."""
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return {"auroc": np.nan, "ap": np.nan, "accuracy": np.nan, "f1": np.nan}

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    for start, stop in zip(starts, stops):
        ranks[start:stop] = 0.5 * (start + 1 + stop)
    positive_rank_sum = ranks[y_true[order].astype(bool)].sum()
    auroc = (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)

    descending = np.argsort(-scores, kind="stable")
    sorted_y = y_true[descending]
    descending_scores = scores[descending]
    threshold_ends = np.r_[
        np.flatnonzero(descending_scores[1:] != descending_scores[:-1]),
        len(scores) - 1,
    ]
    cumulative_tp = np.cumsum(sorted_y)[threshold_ends]
    retrieved = threshold_ends + 1
    recall = cumulative_tp / positives
    precision = cumulative_tp / retrieved
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    predicted = scores >= 0.5
    truth = y_true.astype(bool)
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    accuracy = float(np.mean(predicted == truth))
    f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
    return {"auroc": float(auroc), "ap": ap, "accuracy": accuracy, "f1": f1}


def auroc_against_columns(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute AUROC(scores, targets[:,j]) for every target using one sort."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if scores.ndim != 1 or targets.ndim != 2 or len(scores) != len(targets):
        raise ValueError("Expected scores [N] and targets [N,A]")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    for start, stop in zip(starts, stops):
        ranks[start:stop] = 0.5 * (start + 1 + stop)
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    rank_sums = targets[order].T @ ranks
    with np.errstate(divide="ignore", invalid="ignore"):
        aurocs = (
            rank_sums - positives * (positives + 1.0) / 2.0
        ) / (positives * negatives)
    aurocs[(positives == 0) | (negatives == 0)] = np.nan
    return aurocs.astype(np.float32)


def retrieval_metrics_memory_efficient(
    queries: np.ndarray, gallery: np.ndarray, gallery_ids: np.ndarray,
) -> dict[str, float]:
    """Official Rank-k/mAP without retaining a full QxG distance matrix."""
    queries = np.asarray(queries, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    gallery_ids = np.asarray(gallery_ids, dtype=np.int64)
    base = gallery.sum(axis=1, dtype=np.float32)
    projection = (1.0 - 2.0 * gallery).T
    hits = {1: 0, 5: 0, 10: 0}
    average_precisions: list[float] = []
    for query_id, query in enumerate(queries):
        distances = base + query @ projection
        order = np.argsort(distances, kind="stable")
        relevant = gallery_ids[order] == query_id
        positive_count = int(relevant.sum())
        if positive_count == 0:
            raise ValueError(f"Query {query_id} has no positive gallery image")
        ranks = np.flatnonzero(relevant) + 1
        average_precisions.append(
            float(np.mean(np.arange(1, positive_count + 1) / ranks))
        )
        for k in hits:
            hits[k] += int(np.any(relevant[:k]))
    count = len(queries)
    return {
        "rank1": hits[1] / count,
        "rank5": hits[5] / count,
        "rank10": hits[10] / count,
        "map": float(np.mean(average_precisions)),
    }


def semantic_query_metrics(
    probabilities: np.ndarray, queries: np.ndarray, ids: np.ndarray, chunk_size: int,
) -> dict[str, float]:
    """Rank each image's assigned semantic query among all query vectors."""
    query_projection = (1.0 - 2.0 * queries).T
    query_base = queries.sum(axis=1, dtype=np.float32)[None, :]
    ranks: list[np.ndarray] = []
    correct_distances: list[np.ndarray] = []
    nearest_wrong_distances: list[np.ndarray] = []
    query_indices = np.arange(len(queries), dtype=np.int64)[None, :]
    for start in range(0, len(probabilities), chunk_size):
        stop = min(start + chunk_size, len(probabilities))
        chunk = probabilities[start:stop]
        # d(p, q) = sum(q) + p @ (1 - 2q).  This is the transpose of
        # sum(p) + q @ (1 - 2p) used by query-to-gallery retrieval.
        distances = query_base + chunk @ query_projection
        correct_ids = ids[start:stop]
        rows = np.arange(stop - start)
        correct = distances[rows, correct_ids]
        # Stable tie handling matches np.argsort(..., kind="stable"): a tied
        # lower query index precedes the correct query.
        rank = 1 + np.sum(
            (distances < correct[:, None])
            | ((distances == correct[:, None]) & (query_indices < correct_ids[:, None])),
            axis=1,
        )
        distances[rows, correct_ids] = np.inf
        ranks.append(rank.astype(np.int64))
        correct_distances.append(correct)
        nearest_wrong_distances.append(distances.min(axis=1))
    all_ranks = np.concatenate(ranks)
    correct = np.concatenate(correct_distances)
    nearest_wrong = np.concatenate(nearest_wrong_distances)
    return {
        "top1": float(np.mean(all_ranks <= 1)),
        "top5": float(np.mean(all_ranks <= 5)),
        "top10": float(np.mean(all_ranks <= 10)),
        "mean_correct_distance": float(correct.mean()),
        "mean_nearest_wrong_distance": float(nearest_wrong.mean()),
        "mean_margin": float((nearest_wrong - correct).mean()),
        "mean_rank": float(all_ranks.mean()),
        "median_rank": float(np.median(all_ranks)),
        "p90_rank": float(np.percentile(all_ranks, 90)),
    }


def category_state_rows(attribute_names: Sequence[str]) -> list[dict[str, str | int]]:
    """Describe the 52 training states and their paired-inference 40-D mapping."""
    positive_states: dict[str, str] = {}
    negative_states: dict[str, str] = {}
    for _, columns, state_keys, fallback_key in CategoryPromptMapper._MULTI_GROUPS:
        all_states = [*state_keys, fallback_key]
        for column, state_key in zip(columns, state_keys):
            positive_states[column] = state_key
            negative_states[column] = " | ".join(
                key for key in all_states if key != state_key
            )
    for column, positive_key, negative_key in CategoryPromptMapper._BINARY_GROUPS:
        positive_states[column] = positive_key
        negative_states[column] = negative_key
    return [
        {
            "output_index": index,
            "attribute": name,
            "positive_state": positive_states[name],
            "negative_state": negative_states[name],
            "positive_prompt": ATTRIBUTE_PROMPTS[name],
            "negative_prompt": NEGATIVE_ATTRIBUTE_PROMPTS[name],
        }
        for index, name in enumerate(attribute_names)
    ]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_auroc_matrix(
    output_dir: Path, matrix: np.ndarray, attribute_names: Sequence[str],
) -> None:
    np.save(output_dir / "auroc_matrix.npy", matrix)
    with (output_dir / "auroc_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted\\gt", *attribute_names])
        for name, row in zip(attribute_names, matrix):
            writer.writerow([name, *[f"{value:.8f}" for value in row]])
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnostics] matplotlib unavailable; skipping auroc_matrix.png")
        return
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(attribute_names)), attribute_names, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(attribute_names)), attribute_names, fontsize=6)
    axis.set_xlabel("GT attribute")
    axis.set_ylabel("Predicted dimension")
    figure.colorbar(image, ax=axis, label="AUROC")
    figure.tight_layout()
    figure.savefig(output_dir / "auroc_matrix.png", dpi=180)
    plt.close(figure)


def print_retrieval(label: str, metrics: dict[str, float]) -> None:
    print(f"\n[{label}]")
    print(f"Rank-1 : {100 * metrics['rank1']:.2f} %")
    print(f"Rank-5 : {100 * metrics['rank5']:.2f} %")
    print(f"Rank-10: {100 * metrics['rank10']:.2f} %")
    print(f"mAP    : {100 * metrics['map']:.2f} %")


def diagnostic_messages(
    oracle: dict[str, float], match_ratio: float, macro: dict[str, float],
    diagonal_count: int, mean_diagonal: float, mean_best: float,
    semantic: dict[str, float], current: dict[str, float], collapsed: bool,
) -> list[str]:
    messages: list[str] = []
    if match_ratio < 0.99 or oracle["rank1"] < 0.99 or oracle["map"] < 0.99:
        messages.append(
            "Case A: GT oracle retrieval is broken. Likely query/gt/id mapping, "
            "attribute ordering, or evaluator failure."
        )
    if collapsed:
        messages.append(
            "Prediction collapse is present. Paired prompt logits or probability "
            "calibration are not separating gallery images."
        )
    if np.isfinite(macro["auroc"]) and macro["auroc"] < 0.65:
        messages.append(
            "Case B: GT oracle is usable but PAR AUROC is low. Likely model training "
            "failure or semantic-state probability conversion failure."
        )
    if mean_best >= 0.70 and mean_best - mean_diagonal >= 0.08 and diagonal_count < 30:
        messages.append(
            "Case C: Row-best AUROC is much higher than diagonal AUROC. Possible "
            "attribute ordering permutation or semantic-state mapping mismatch."
        )
    if macro["auroc"] >= 0.65 and semantic["top1"] < 0.50:
        messages.append(
            "Case D: PAR AUROC is reasonable but nearest-query accuracy is low. "
            "Likely calibration, L1 incompatibility, or insufficient joint semantic-ID modeling."
        )
    if (
        match_ratio >= 0.99 and macro["auroc"] >= 0.70 and semantic["top1"] >= 0.50
        and current["rank1"] < 0.20
    ):
        messages.append(
            "Case E: Core diagnostics are reasonable but final retrieval is low. "
            "Recheck evaluator orientation, ranking direction, and query/gallery alignment."
        )
    if not messages:
        messages.append("No single heuristic failure case dominates; inspect the saved tables.")
    return messages


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else checkpoint.parent / "diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    queries_gt_order = reorder_columns(queries_raw, query_names, table.attribute_names)

    print("=" * 72)
    print("AttriVision Task2 Diagnostics")
    print("=" * 72)
    print(f"Checkpoint : {checkpoint}")
    print(f"Validation : {gt_path}")
    print(f"Images     : {len(table.image_paths)}")
    print(f"Queries    : {len(queries_raw)}")
    print(f"Artifacts  : {output_dir}")

    exact = np.all(table.labels == queries_gt_order[ids], axis=1)
    matched = int(exact.sum())
    mismatch_indices = np.flatnonzero(~exact)
    match_ratio = float(exact.mean())
    print("\nGT ↔ assigned query exact-match:")
    print(f"matched    : {matched} / {len(exact)}")
    print(f"mismatched : {len(mismatch_indices)}")
    print(f"match ratio: {100 * match_ratio:.2f} %")
    for image_index in mismatch_indices[:20]:
        different = np.flatnonzero(table.labels[image_index] != queries_gt_order[ids[image_index]])
        print(f"\nMismatch gallery[{image_index}]")
        print(f"  image path                 : {table.image_paths[image_index]}")
        print(f"  id                         : {ids[image_index]}")
        print(f"  gt vector                  : {table.labels[image_index].astype(int).tolist()}")
        print(f"  assigned query vector      : {queries_gt_order[ids[image_index]].astype(int).tolist()}")
        print(f"  different attribute indices: {different.tolist()}")
        print(f"  different attribute names  : {[table.attribute_names[i] for i in different]}")

    oracle = retrieval_metrics_memory_efficient(queries_gt_order, table.labels, ids)
    print_retrieval("GT ORACLE RETRIEVAL", oracle)
    if match_ratio < 0.99 or oracle["rank1"] < 0.99 or oracle["map"] < 0.99:
        print("WARNING: GT oracle is not near-perfect; later model diagnostics may be misleading.")

    device = choose_device(args.device)
    print(f"\n[MODEL INFERENCE] device={device}, amp={args.amp}")
    model, payload = load_model(checkpoint, device)
    model_attribute_names = list(payload["attribute_names"])
    if len(model_attribute_names) != 40 or len(set(model_attribute_names)) != 40:
        raise ValueError("Checkpoint must contain 40 unique attribute names")
    gt_model_order = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    queries_model_order = reorder_columns(queries_raw, query_names, model_attribute_names)
    inverse_temperature = (
        1.0 / args.attribute_temperature
        if args.attribute_temperature is not None
        else learned_inverse_temperature(model)
    )
    if args.predictions:
        prediction_path = Path(args.predictions).resolve()
        probabilities = np.load(prediction_path).astype(np.float32, copy=False)
        if probabilities.shape != (len(table.image_paths), 40) or not np.isfinite(probabilities).all():
            raise ValueError(
                f"Saved predictions must be finite [{len(table.image_paths)},40], "
                f"got {probabilities.shape}"
            )
        print(f"Reusing paired-L1 predictions: {prediction_path}")
    else:
        gallery_features = encode_gallery(
            model,
            table.image_paths,
            [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(args.image_size),
            device,
            args.eval_batch_size,
            args.num_workers,
            args.amp,
            args.progress_every,
        )
        paired_features = encode_prompt_pairs(model, model_attribute_names, device, args.amp)
        probabilities = paired_attribute_probabilities(
            gallery_features, paired_features, inverse_temperature,
        )
        del gallery_features, paired_features
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.save(output_dir / "pred_probs.npy", probabilities)
    print(
        f"Prediction shape={probabilities.shape}; inverse_temperature={inverse_temperature:.6f}; "
        f"T={1.0 / inverse_temperature:.8f}"
    )

    print("\n[ATTRIVISION 40-D ATTRIBUTE PREDICTION QUALITY]")
    per_attribute: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, name in enumerate(model_attribute_names):
        metrics = binary_metrics(gt_model_order[:, index], probabilities[:, index])
        if not np.isfinite(metrics["auroc"]):
            skipped.append(name)
            reason = "no positive samples" if gt_model_order[:, index].sum() == 0 else "no negative samples"
            print(f"Skipping {name}: {reason}")
        per_attribute.append({
            "idx": index,
            "attribute_name": name,
            "gt_pos_rate": float(gt_model_order[:, index].mean()),
            "pred_mean": float(probabilities[:, index].mean()),
            "auroc": metrics["auroc"],
            "ap": metrics["ap"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
        })
    print("idx | attribute_name | gt_pos_rate | pred_mean | AUROC | AP | Acc | F1")
    for row in per_attribute:
        print(
            f"{row['idx']:02d} | {row['attribute_name']} | {row['gt_pos_rate']:.4f} | "
            f"{row['pred_mean']:.4f} | {row['auroc']:.4f} | {row['ap']:.4f} | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f}"
        )
    macro = {
        key: float(np.nanmean([row[key] for row in per_attribute]))
        for key in ("auroc", "ap", "accuracy", "f1")
    }
    print(f"Macro AUROC  : {macro['auroc']:.4f}")
    print(f"Macro AP     : {macro['ap']:.4f}")
    print(f"Macro Accuracy: {macro['accuracy']:.4f}")
    print(f"Macro F1     : {macro['f1']:.4f}")
    write_csv(
        output_dir / "per_attribute_metrics.csv",
        ("idx", "attribute_name", "gt_pos_rate", "pred_mean", "auroc", "ap", "accuracy", "f1"),
        per_attribute,
    )

    print("\n[40x40 PREDICTION ↔ GT AUROC MATRIX]")
    matrix = np.stack([
        auroc_against_columns(probabilities[:, predicted_index], gt_model_order)
        for predicted_index in range(40)
    ])
    save_auroc_matrix(output_dir, matrix, model_attribute_names)
    mapping_rows: list[dict[str, Any]] = []
    diagonal_best = 0
    for index, name in enumerate(model_attribute_names):
        if np.all(np.isnan(matrix[index])):
            best_index = -1
            best_value = np.nan
        else:
            best_index = int(np.nanargmax(matrix[index]))
            best_value = float(matrix[index, best_index])
        diagonal = float(matrix[index, index])
        correct = best_index == index
        diagonal_best += int(correct)
        best_name = model_attribute_names[best_index] if best_index >= 0 else "N/A"
        print(
            f"Pred[{index:02d}] expected={name}\n"
            f"    best GT = {best_name}\n"
            f"    AUROC   = {best_value:.4f}\n"
            f"    diagonal AUROC = {diagonal:.4f}\n"
            f"    correct mapping = {'YES' if correct else 'NO'}"
        )
        mapping_rows.append({
            "pred_idx": index,
            "pred_name": name,
            "best_gt_idx": best_index,
            "best_gt_name": best_name,
            "best_auroc": best_value,
            "diagonal_auroc": diagonal,
        })
    mean_diagonal = float(np.nanmean(np.diag(matrix)))
    mean_best = float(np.nanmean(np.nanmax(matrix, axis=1)))
    print(f"40 dimensions 중 argmax가 diagonal인 개수: {diagonal_best} / 40")
    print(f"mean diagonal AUROC: {mean_diagonal:.4f}")
    print(f"mean row-best AUROC: {mean_best:.4f}")
    if diagonal_best < 30 and mean_best - mean_diagonal >= 0.08:
        print("WARNING: possible attribute ordering / semantic-state mapping mismatch")
    write_csv(
        output_dir / "suspected_permutation.csv",
        ("pred_idx", "pred_name", "best_gt_idx", "best_gt_name", "best_auroc", "diagonal_auroc"),
        mapping_rows,
    )

    print("\n[NEAREST SEMANTIC-QUERY ACCURACY]")
    semantic = semantic_query_metrics(
        probabilities, queries_model_order, ids, args.semantic_chunk_size,
    )
    print(f"Top-1                         : {100 * semantic['top1']:.2f} %")
    print(f"Top-5                         : {100 * semantic['top5']:.2f} %")
    print(f"Top-10                        : {100 * semantic['top10']:.2f} %")
    print(f"mean distance to correct query: {semantic['mean_correct_distance']:.4f}")
    print(f"mean distance to nearest wrong: {semantic['mean_nearest_wrong_distance']:.4f}")
    print(f"margin (wrong - correct)       : {semantic['mean_margin']:.4f}")
    print(f"median correct-query rank      : {semantic['median_rank']:.1f}")
    print(f"mean correct-query rank        : {semantic['mean_rank']:.1f}")
    print(f"90th percentile rank           : {semantic['p90_rank']:.1f}")

    current = retrieval_metrics_memory_efficient(queries_model_order, probabilities, ids)
    print_retrieval("CURRENT PAIRED-L1 RETRIEVAL", current)

    print("\n[PREDICTION DISTRIBUTION]")
    distribution = {
        "min": float(probabilities.min()),
        "max": float(probabilities.max()),
        "mean": float(probabilities.mean()),
        "std": float(probabilities.std()),
        "saturated_fraction": float(np.mean((probabilities <= 0.01) | (probabilities >= 0.99))),
    }
    for key, value in distribution.items():
        print(f"global {key:18s}: {value:.6f}")
    distribution_rows = []
    for index, name in enumerate(model_attribute_names):
        column = probabilities[:, index]
        row = {
            "idx": index, "attribute_name": name, "mean": float(column.mean()),
            "std": float(column.std()), "min": float(column.min()), "max": float(column.max()),
        }
        distribution_rows.append(row)
        print(
            f"{index:02d} | {name} | mean={row['mean']:.4f} std={row['std']:.4f} "
            f"min={row['min']:.4f} max={row['max']:.4f}"
        )
    column_means = probabilities.mean(axis=0)
    column_stds = probabilities.std(axis=0)
    flat_columns = int(np.count_nonzero(column_stds < 0.02))
    near_half_flat_columns = int(np.count_nonzero(
        (np.abs(column_means - 0.5) < 0.05) & (column_stds < 0.02)
    ))
    distribution["flat_attribute_count"] = flat_columns
    distribution["near_half_flat_attribute_count"] = near_half_flat_columns
    print(f"flat attributes (std < .02)       : {flat_columns} / 40")
    print(f"near-0.5 flat attributes          : {near_half_flat_columns} / 40")
    collapsed = (
        distribution["std"] < 0.02
        or float(np.mean(column_stds)) < 0.01
        or near_half_flat_columns >= 8
    )
    if collapsed or (abs(distribution["mean"] - 0.5) < 0.05 and distribution["std"] < 0.05):
        print("WARNING: attribute probability collapse detected in multiple dimensions")
        collapsed = True
    if distribution["saturated_fraction"] >= 0.80:
        print("WARNING: attribute probabilities are strongly saturated near 0/1")
    write_csv(
        output_dir / "prediction_distribution.csv",
        ("idx", "attribute_name", "mean", "std", "min", "max"),
        distribution_rows,
    )

    print("\n[SEMANTIC-STATE → 40 BINARY OUTPUT MAPPING]")
    state_rows = category_state_rows(model_attribute_names)
    print(f"Training prompt mode            : {payload.get('prompt_mode', 'unknown')}")
    print(f"Category semantic-state count   : {len(CATEGORY_PROMPTS)}")
    print("Paired inference output count   : 40")
    print(
        "NOTE: paired_l1 does not aggregate the 52 category logits. It performs "
        "40 independent softmax operations over separately encoded negative/positive prompts."
    )
    for row in state_rows:
        print(
            f"idx={row['output_index']:02d} | UPAR attr={row['attribute']}\n"
            f"  positive state : {row['positive_state']}\n"
            f"  negative state : {row['negative_state']}\n"
            f"  positive prompt: {row['positive_prompt']}\n"
            f"  negative prompt: {row['negative_prompt']}"
        )
    write_csv(
        output_dir / "semantic_state_mapping.csv",
        ("output_index", "attribute", "positive_state", "negative_state", "positive_prompt", "negative_prompt"),
        state_rows,
    )
    duplicate_indices = len({row["output_index"] for row in state_rows}) != 40
    missing_attributes = sorted(set(model_attribute_names) - {str(row["attribute"]) for row in state_rows})
    print(f"duplicate output index          : {'YES' if duplicate_indices else 'NO'}")
    print(f"missing attribute               : {missing_attributes or 'NONE'}")
    print(f"GT order == query order         : {table.attribute_names == query_names}")
    print(f"GT order == checkpoint order    : {table.attribute_names == model_attribute_names}")
    print(f"query order == checkpoint order : {query_names == model_attribute_names}")

    messages = diagnostic_messages(
        oracle, match_ratio, macro, diagonal_best, mean_diagonal, mean_best,
        semantic, current, collapsed,
    )
    if (
        table.attribute_names == query_names == model_attribute_names
        and diagonal_best < 30 and mean_best - mean_diagonal >= 0.08
    ):
        messages.append(
            "GT, query, and checkpoint headers have exactly the same order, so a literal "
            "column permutation was not found; the off-diagonal AUROC pattern instead points "
            "to semantic prediction confusion and correlated attributes."
        )
    print("\n" + "=" * 44)
    print("AttriVision Task2 Diagnostic Summary")
    print("=" * 44)
    print("\n[1] GT Oracle Retrieval")
    print(f"GT-query exact match     : {100 * match_ratio:.2f} %")
    print(f"Rank-1                  : {100 * oracle['rank1']:.2f} %")
    print(f"mAP                     : {100 * oracle['map']:.2f} %")
    print("\n[2] PAR Prediction Quality")
    print(f"Macro AUROC             : {macro['auroc']:.4f}")
    print(f"Macro AP                : {macro['ap']:.4f}")
    print(f"Macro F1                : {macro['f1']:.4f}")
    print("\n[3] Attribute Mapping")
    print(f"Diagonal-best dimensions: {diagonal_best} / 40")
    print(f"Mean diagonal AUROC     : {mean_diagonal:.4f}")
    print(f"Mean row-best AUROC     : {mean_best:.4f}")
    print("\n[4] Semantic ID Recovery")
    print(f"Nearest-query Top-1     : {100 * semantic['top1']:.2f} %")
    print(f"Top-5                   : {100 * semantic['top5']:.2f} %")
    print(f"Top-10                  : {100 * semantic['top10']:.2f} %")
    print(f"Mean correct-query rank : {semantic['mean_rank']:.1f}")
    print("\n[5] Prediction Distribution")
    print(f"Mean                    : {distribution['mean']:.4f}")
    print(f"Std                     : {distribution['std']:.4f}")
    print(f"Min                     : {distribution['min']:.4f}")
    print(f"Max                     : {distribution['max']:.4f}")
    print("\n[Current paired-L1 retrieval]")
    print(f"Rank-1                  : {100 * current['rank1']:.2f} %")
    print(f"mAP                     : {100 * current['map']:.2f} %")
    print("\n[AUTOMATIC DIAGNOSIS]")
    for message in messages:
        print(f"- {message}")

    summary = {
        "checkpoint": str(checkpoint),
        "temperature": 1.0 / inverse_temperature,
        "oracle": oracle,
        "gt_query_match_ratio": match_ratio,
        "macro_prediction_metrics": macro,
        "skipped_attributes": skipped,
        "attribute_mapping": {
            "diagonal_best_dimensions": diagonal_best,
            "mean_diagonal_auroc": mean_diagonal,
            "mean_row_best_auroc": mean_best,
        },
        "semantic_query_recovery": semantic,
        "prediction_distribution": distribution,
        "current_paired_l1_retrieval": current,
        "automatic_diagnosis": messages,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved diagnostic artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
