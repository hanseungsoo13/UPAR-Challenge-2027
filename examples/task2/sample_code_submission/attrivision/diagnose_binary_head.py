"""Diagnose BCE40 semantic-ID retrieval without training or model mutation.

The script performs one validation image forward pass, then compares sigmoid-L1,
logits-L1, and hard-Hamming using the repository's Task-2 query/ID mapping.
Distance matrices are processed in chunks and are never retained in full.
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
from attrivision.engine.evaluator_abpr import encode_gallery  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import load_retrieval_annotations, reorder_columns  # noqa: E402


ArrayPair = tuple[np.ndarray, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a trained AttriVision BCE40 head on Task2 validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/attrivision_binary_head_paper_fce/checkpoint_best.pth",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", help="default: <checkpoint parent>/binary_head_diagnostic")
    parser.add_argument(
        "--logits",
        help="reuse a saved finite [G,40] .npy file and skip image/model inference",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--semantic-chunk-size", type=int, default=512)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--logit-epsilon", type=float, default=1e-4)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "eval_batch_size", "head_batch_size", "image_size", "query_chunk_size",
        "semantic_chunk_size", "ece_bins",
    )
    for field in positive:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("--num-workers and --progress-every cannot be negative")
    if not 0.0 < args.logit_epsilon < 0.5:
        raise ValueError("--logit-epsilon must be between 0 and 0.5")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    bin_ids = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    counts = np.bincount(bin_ids, minlength=bins).astype(np.float64)
    confidence = np.bincount(bin_ids, weights=probabilities, minlength=bins)
    accuracy = np.bincount(bin_ids, weights=labels, minlength=bins)
    populated = counts > 0
    confidence[populated] /= counts[populated]
    accuracy[populated] /= counts[populated]
    return float(np.sum(counts[populated] * np.abs(confidence[populated] - accuracy[populated])) / len(labels))


def instance_and_hamming_metrics(
    predicted: np.ndarray, labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    true_positive = np.count_nonzero(predicted & truth, axis=1)
    predicted_positive = predicted.sum(axis=1)
    actual_positive = truth.sum(axis=1)
    precision = np.divide(
        true_positive, predicted_positive, out=np.zeros(len(truth), dtype=np.float64),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive, actual_positive, out=np.zeros(len(truth), dtype=np.float64),
        where=actual_positive != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall, precision + recall,
        out=np.zeros(len(truth), dtype=np.float64), where=(precision + recall) != 0,
    )
    hamming = np.count_nonzero(predicted != truth, axis=1)
    metrics = {
        "instance_precision": float(precision.mean()),
        "instance_recall": float(recall.mean()),
        "instance_f1": float(f1.mean()),
        "mean_hamming": float(hamming.mean()),
        "median_hamming": float(np.median(hamming)),
        "p25_hamming": float(np.percentile(hamming, 25)),
        "p75_hamming": float(np.percentile(hamming, 75)),
        "p90_hamming": float(np.percentile(hamming, 90)),
        "exact_match_rate": float(np.mean(hamming == 0)),
        "le1_rate": float(np.mean(hamming <= 1)),
        "le2_rate": float(np.mean(hamming <= 2)),
        "le5_rate": float(np.mean(hamming <= 5)),
    }
    return metrics, hamming


def distance_components(values: np.ndarray, method: str, epsilon: float) -> ArrayPair:
    """Return base[G] and delta[G,40] for d(q,g)=base[g]+q@delta[g]."""
    values = np.asarray(values, dtype=np.float32)
    if method == "sigmoid_l1":
        cost_zero = values
        cost_one = 1.0 - values
    elif method == "hard_hamming":
        hard = values.astype(np.float32, copy=False)
        cost_zero = hard
        cost_one = 1.0 - hard
    elif method == "logits_l1":
        negative_logit = float(np.log(epsilon / (1.0 - epsilon)))
        positive_logit = -negative_logit
        cost_zero = np.abs(values - negative_logit)
        cost_one = np.abs(values - positive_logit)
    else:
        raise ValueError(f"Unknown method: {method}")
    base = cost_zero.sum(axis=1, dtype=np.float32)
    delta = (cost_one - cost_zero).astype(np.float32, copy=False)
    return base, delta


def retrieval_metrics_chunked(
    queries: np.ndarray, gallery_ids: np.ndarray, base: np.ndarray,
    delta: np.ndarray, chunk_size: int,
) -> dict[str, float]:
    """Official stable Task2 Rank-k/mAP without retaining QxG distances."""
    hits = {1: 0, 5: 0, 10: 0}
    average_precisions: list[float] = []
    query_count = len(queries)
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        distances = base[None, :] + queries[start:stop] @ delta.T
        for offset, row in enumerate(distances):
            query_id = start + offset
            order = np.argsort(row, kind="stable")
            relevant = gallery_ids[order] == query_id
            positive_count = int(relevant.sum())
            if positive_count == 0:
                raise ValueError(f"Query {query_id} has no gallery positives")
            positive_ranks = np.flatnonzero(relevant) + 1
            average_precisions.append(float(
                np.mean(np.arange(1, positive_count + 1) / positive_ranks)
            ))
            for k in hits:
                hits[k] += int(np.any(relevant[:k]))
        print(f"  retrieval queries {stop}/{query_count}", end="\r", flush=True)
    print(" " * 48, end="\r")
    return {
        "rank1": hits[1] / query_count,
        "rank5": hits[5] / query_count,
        "rank10": hits[10] / query_count,
        "map": float(np.mean(average_precisions)),
    }


def semantic_and_margin_metrics(
    queries: np.ndarray, ids: np.ndarray, base: np.ndarray, delta: np.ndarray,
    hamming: np.ndarray, chunk_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    ranks: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    correct_distances: list[np.ndarray] = []
    wrong_distances: list[np.ndarray] = []
    query_indices = np.arange(len(queries), dtype=np.int64)[None, :]
    for start in range(0, len(ids), chunk_size):
        stop = min(start + chunk_size, len(ids))
        distances = base[start:stop, None] + delta[start:stop] @ queries.T
        correct_ids = ids[start:stop]
        rows = np.arange(stop - start)
        correct = distances[rows, correct_ids].copy()
        rank = 1 + np.count_nonzero(
            (distances < correct[:, None])
            | ((distances == correct[:, None]) & (query_indices < correct_ids[:, None])),
            axis=1,
        )
        distances[rows, correct_ids] = np.inf
        wrong = distances.min(axis=1)
        ranks.append(rank.astype(np.int64))
        correct_distances.append(correct)
        wrong_distances.append(wrong)
        margins.append(wrong - correct)
    all_ranks = np.concatenate(ranks)
    all_margins = np.concatenate(margins)
    correct = np.concatenate(correct_distances)
    wrong = np.concatenate(wrong_distances)
    tolerance = 1e-6
    bins = (
        ("0 bit error", hamming == 0),
        ("1-2 bit errors", (hamming >= 1) & (hamming <= 2)),
        ("3-5 bit errors", (hamming >= 3) & (hamming <= 5)),
        ("6-10 bit errors", (hamming >= 6) & (hamming <= 10)),
        (">10 bit errors", hamming > 10),
    )
    by_hamming: dict[str, dict[str, float | int]] = {}
    for label, mask in bins:
        if np.any(mask):
            selected = all_margins[mask]
            by_hamming[label] = {
                "count": int(mask.sum()),
                "mean_margin": float(selected.mean()),
                "median_margin": float(np.median(selected)),
                "fraction_positive": float(np.mean(selected > tolerance)),
                "fraction_zero": float(np.mean(np.abs(selected) <= tolerance)),
                "fraction_negative": float(np.mean(selected < -tolerance)),
            }
        else:
            by_hamming[label] = {"count": 0}
    metrics: dict[str, Any] = {
        "semantic_top1": float(np.mean(all_ranks == 1)),
        "mean_correct_rank": float(all_ranks.mean()),
        "median_correct_rank": float(np.median(all_ranks)),
        "mean_d_correct": float(correct.mean()),
        "mean_d_wrong": float(wrong.mean()),
        "mean_margin": float(all_margins.mean()),
        "median_margin": float(np.median(all_margins)),
        "fraction_margin_positive": float(np.mean(all_margins > tolerance)),
        "fraction_margin_zero": float(np.mean(np.abs(all_margins) <= tolerance)),
        "fraction_margin_negative": float(np.mean(all_margins < -tolerance)),
        "margin_by_hamming": by_hamming,
    }
    return metrics, all_ranks, all_margins


def calibration_metrics(
    probabilities: np.ndarray, logits: np.ndarray, labels: np.ndarray,
    attribute_names: Sequence[str], bins: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    positive = labels.astype(bool)
    negative = ~positive

    def summarize(p: np.ndarray, z: np.ndarray, y: np.ndarray) -> dict[str, float]:
        pos = y.astype(bool)
        neg = ~pos
        return {
            "mean_p_y1": float(p[pos].mean()),
            "mean_p_y0": float(p[neg].mean()),
            "mean_logit_y1": float(z[pos].mean()),
            "std_logit_y1": float(z[pos].std()),
            "mean_logit_y0": float(z[neg].mean()),
            "std_logit_y0": float(z[neg].std()),
            "brier_score": float(np.mean((p - y) ** 2)),
            "ece": expected_calibration_error(p, y, bins),
        }

    overall = summarize(probabilities, logits, labels)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(attribute_names):
        row = {"index": index, "attribute": name}
        row.update(summarize(probabilities[:, index], logits[:, index], labels[:, index]))
        row["positive_count"] = int(positive[:, index].sum())
        row["negative_count"] = int(negative[:, index].sum())
        rows.append(row)
    return overall, rows


def infer_logits(
    args: argparse.Namespace, checkpoint: Path, data_root: Path, paths: Sequence[str],
    roots: Sequence[Path], expected_count: int,
) -> tuple[np.ndarray, list[str]]:
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    state = payload["model_state_dict"]
    required_head = {"binary_head.weight", "binary_head.bias"}
    missing_head = required_head - set(state)
    if missing_head:
        raise ValueError(
            "Checkpoint has no trained BCE40 binary head; refusing to evaluate a random "
            f"head. Missing keys: {sorted(missing_head)}"
        )
    attribute_names = list(payload["attribute_names"])
    if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
        raise ValueError("Checkpoint must contain 40 unique attribute names")
    print(f"[inference] device={device}, amp={args.amp}")
    features = encode_gallery(
        model, paths, roots, build_eval_transform(args.image_size), device,
        args.eval_batch_size, args.num_workers, args.amp, args.progress_every,
        "[binary-head diagnostic]",
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), args.head_batch_size):
            feature_chunk = features[start:start + args.head_batch_size].to(device)
            chunks.append(
                model.binary_logits_from_features(feature_chunk).float().cpu().numpy()
            )
    logits = np.concatenate(chunks).astype(np.float32, copy=False)
    if logits.shape != (expected_count, 40) or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid BCE40 logits: {logits.shape}")
    return logits, attribute_names


def print_method(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n[{label}]")
    print(f"Rank-1 / Rank-5 / Rank-10 : {100*metrics['rank1']:.2f}% / {100*metrics['rank5']:.2f}% / {100*metrics['rank10']:.2f}%")
    print(f"mAP                       : {100*metrics['map']:.2f}%")
    print(f"Semantic-query Top-1      : {100*metrics['semantic_top1']:.2f}%")
    print(f"Mean / median correct rank: {metrics['mean_correct_rank']:.1f} / {metrics['median_correct_rank']:.1f}")
    print(f"mean d_correct            : {metrics['mean_d_correct']:.4f}")
    print(f"mean d_wrong              : {metrics['mean_d_wrong']:.4f}")
    print(f"mean / median margin      : {metrics['mean_margin']:.4f} / {metrics['median_margin']:.4f}")
    print(
        "margin >0 / ==0 / <0       : "
        f"{100*metrics['fraction_margin_positive']:.2f}% / "
        f"{100*metrics['fraction_margin_zero']:.2f}% / "
        f"{100*metrics['fraction_margin_negative']:.2f}%"
    )
    print("Margin by Hamming error:")
    for name, row in metrics["margin_by_hamming"].items():
        if row["count"]:
            print(
                f"  {name:16s} n={row['count']:5d}, mean={row['mean_margin']:.4f}, "
                f"median={row['median_margin']:.4f}, >0={100*row['fraction_positive']:.2f}%"
            )
        else:
            print(f"  {name:16s} n=0")


def automatic_diagnosis(
    instance: dict[str, float], methods: dict[str, dict[str, Any]],
) -> list[str]:
    messages: list[str] = []
    mean_hamming = instance["mean_hamming"]
    exact = instance["exact_match_rate"]
    sigmoid = methods["sigmoid_l1"]
    hard = methods["hard_hamming"]
    if mean_hamming > 5.0:
        messages.append(
            "Hamming error가 크므로 independent BCE attribute prediction 자체가 "
            "semantic-ID retrieval에 부족합니다."
        )
    elif mean_hamming <= 2.0 and hard["rank1"] < 0.10:
        messages.append(
            "Hamming error는 작지만 retrieval이 낮아 query density, distance tie, "
            "또는 scoring geometry가 주요 병목입니다."
        )
    hard_gain = hard["rank1"] - sigmoid["rank1"]
    semantic_gain = hard["semantic_top1"] - sigmoid["semantic_top1"]
    if hard_gain >= 0.05 or semantic_gain >= 0.05:
        messages.append(
            "Sigmoid-L1보다 hard-Hamming이 뚜렷하게 높아 probability calibration "
            "문제가 강하게 의심됩니다."
        )
    if hard["rank1"] < 0.10 and hard["semantic_top1"] < 0.20:
        messages.append(
            "Hard-Hamming도 낮아 현재 40-bit prediction accuracy/조합 복원력이 "
            "semantic retrieval에 충분하지 않습니다."
        )
    if exact < 0.10:
        messages.append(
            "Exact 40-bit match rate가 매우 낮아 3,462-way semantic combination "
            "recovery가 사실상 실패한 상태입니다."
        )
    if not messages:
        messages.append("지정한 휴리스틱에서 단일 지배적 실패 원인은 발견되지 않았습니다.")
    return messages


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else checkpoint.parent / "binary_head_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)

    if args.logits:
        logits_path = Path(args.logits).resolve()
        logits = np.load(logits_path).astype(np.float32, copy=False)
        if logits.shape != (len(table.image_paths), 40) or not np.isfinite(logits).all():
            raise ValueError(f"--logits must be finite [{len(table.image_paths)},40], got {logits.shape}")
        # Ordering still comes from the checkpoint, while model/image inference is skipped.
        _, payload = load_model(checkpoint, torch.device("cpu"))
        required_head = {"binary_head.weight", "binary_head.bias"}
        if required_head - set(payload["model_state_dict"]):
            raise ValueError("Checkpoint does not contain a trained BCE40 head")
        attribute_names = list(payload["attribute_names"])
        print(f"[inference] reusing logits: {logits_path}")
    else:
        logits, attribute_names = infer_logits(
            args, checkpoint, data_root, table.image_paths,
            [data_root, gt_path.parent, REPOSITORY_ROOT], len(table.image_paths),
        )
    if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
        raise ValueError("Checkpoint must contain 40 unique attribute names")
    labels = reorder_columns(table.labels, table.attribute_names, attribute_names).astype(np.float32)
    queries = reorder_columns(queries_raw, query_names, attribute_names).astype(np.float32)
    if not np.all(labels == queries[ids]):
        raise RuntimeError("Validation labels do not exactly match their assigned semantic queries")

    np.save(output_dir / "binary_head_logits.npy", logits)
    probabilities = (1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))).astype(np.float32)
    predicted = probabilities > 0.5
    instance, hamming = instance_and_hamming_metrics(predicted, labels)
    calibration, calibration_rows = calibration_metrics(
        probabilities, logits, labels, attribute_names, args.ece_bins,
    )
    write_csv(
        output_dir / "attribute_calibration.csv", calibration_rows[0].keys(), calibration_rows,
    )

    methods: dict[str, dict[str, Any]] = {}
    per_image: dict[str, np.ndarray] = {"hamming_error": hamming}
    specifications: tuple[tuple[str, str, np.ndarray], ...] = (
        ("sigmoid_l1", "Sigmoid-L1", probabilities),
        ("logits_l1", "Logits-L1", logits),
        ("hard_hamming", "Hard-Hamming", predicted.astype(np.float32)),
    )
    for key, label, values in specifications:
        print(f"\n[computing {label}]")
        base, delta = distance_components(values, key, args.logit_epsilon)
        retrieval = retrieval_metrics_chunked(
            queries, ids, base, delta, args.query_chunk_size,
        )
        semantic, ranks, margins = semantic_and_margin_metrics(
            queries, ids, base, delta, hamming, args.semantic_chunk_size,
        )
        methods[key] = {**retrieval, **semantic}
        per_image[f"{key}_correct_rank"] = ranks
        per_image[f"{key}_margin"] = margins
        print_method(label, methods[key])

    per_image_rows = [
        {"image_index": index, **{name: values[index] for name, values in per_image.items()}}
        for index in range(len(table.image_paths))
    ]
    write_csv(output_dir / "per_image_diagnostic.csv", per_image_rows[0].keys(), per_image_rows)
    diagnosis = automatic_diagnosis(instance, methods)

    print("\n" + "=" * 44)
    print("BCE40 Retrieval Failure Diagnostic")
    print("=" * 44)
    print(f"Images / queries          : {len(labels)} / {len(queries)}")
    print(f"Instance Precision        : {100*instance['instance_precision']:.2f}%")
    print(f"Instance Recall           : {100*instance['instance_recall']:.2f}%")
    print(f"Instance F1               : {100*instance['instance_f1']:.2f}%")
    print(f"Mean Hamming error        : {instance['mean_hamming']:.2f} / 40")
    print(f"Median / P25 / P75 / P90 : {instance['median_hamming']:.1f} / {instance['p25_hamming']:.1f} / {instance['p75_hamming']:.1f} / {instance['p90_hamming']:.1f}")
    print(f"Exact 40-bit match        : {100*instance['exact_match_rate']:.2f}%")
    print(f"<=1 / <=2 / <=5 bit error: {100*instance['le1_rate']:.2f}% / {100*instance['le2_rate']:.2f}% / {100*instance['le5_rate']:.2f}%")
    for key, label, _ in specifications:
        metric = methods[key]
        print(f"\n[{label}]")
        print(f"R1 / mAP                 : {100*metric['rank1']:.2f} / {100*metric['map']:.2f}")
        print(f"Semantic Top1            : {100*metric['semantic_top1']:.2f}%")
        print(f"Mean correct rank        : {metric['mean_correct_rank']:.1f}")
        print(f"Mean margin              : {metric['mean_margin']:.4f}")
    print("\nCalibration:")
    print(f"mean p | y=1             : {calibration['mean_p_y1']:.4f}")
    print(f"mean p | y=0             : {calibration['mean_p_y0']:.4f}")
    print(f"mean/std logits | y=1    : {calibration['mean_logit_y1']:.4f} / {calibration['std_logit_y1']:.4f}")
    print(f"mean/std logits | y=0    : {calibration['mean_logit_y0']:.4f} / {calibration['std_logit_y0']:.4f}")
    print(f"Brier score              : {calibration['brier_score']:.4f}")
    print(f"ECE ({args.ece_bins} bins)            : {calibration['ece']:.4f}")
    print("\nAutomatic interpretation:")
    for message in diagnosis:
        print(f"- {message}")

    summary = {
        "checkpoint": str(checkpoint),
        "validation": str(gt_path),
        "images": len(labels),
        "queries": len(queries),
        "instance_and_hamming": instance,
        "methods": methods,
        "calibration": calibration,
        "automatic_diagnosis": diagnosis,
        "settings": {
            "threshold": 0.5, "logit_epsilon": args.logit_epsilon,
            "ece_bins": args.ece_bins,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved artifacts: {output_dir}")


if __name__ == "__main__":
    main()
