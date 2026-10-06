"""Compare four AttriVision inference formulations on Task 2 validation.

The checkpoint is never trained or modified. The gallery is encoded once and
shared by all methods.
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
import torch.nn.functional as F

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper, prompts_for_attributes  # noqa: E402
from attrivision.diagnose import binary_metrics  # noqa: E402
from attrivision.engine.evaluator_abpr import encode_gallery  # noqa: E402
from attrivision.engine.paired_attribute import (  # noqa: E402
    encode_prompt_pairs, learned_inverse_temperature, paired_attribute_probabilities,
)
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    autocast, l1_attribute_distances, load_retrieval_annotations,
    reorder_columns, retrieval_metrics,
)


METHOD_NAMES = {
    "A": "Positive-text aggregation",
    "B": "Paired pos/neg prompt",
    "C": "Native 52-state category",
    "D": "Native 52-state raw",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare AttriVision inference formulations without retraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="outputs/attrivision_category/checkpoint_best.pth")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", help="default: <checkpoint parent>/inference_comparison")
    parser.add_argument(
        "--gallery-features",
        help="reuse saved finite [G,D] features; otherwise encode once and save them",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument(
        "--attribute-temperature", type=float,
        help="B/C softmax T; default uses the checkpoint learned CLIP temperature",
    )
    parser.add_argument(
        "--raw-normalization", choices=("minmax", "sigmoid", "none"), default="minmax",
        help="normalization applied to D projected raw cosine scores before L1",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("eval_batch_size", "image_size", "distance_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("--num-workers and --progress-every cannot be negative")
    if args.attribute_temperature is not None and args.attribute_temperature <= 0:
        raise ValueError("--attribute-temperature must be positive")


def encode_texts(model: Any, prompts: Sequence[str], device: torch.device,
                 amp: bool) -> torch.Tensor:
    model.eval()
    with torch.inference_mode(), autocast(device, amp):
        features = model.encode_text(model.tokenize(list(prompts)).to(device))
    return features.float().cpu()


def minmax_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = values.min(axis=0, keepdims=True)
    span = values.max(axis=0, keepdims=True) - low
    return np.divide(
        values - low, span, out=np.full_like(values, 0.5), where=span > 1e-12,
    ).astype(np.float32, copy=False)


def normalize_raw(values: np.ndarray, mode: str, inverse_temperature: float) -> np.ndarray:
    if mode == "minmax":
        return minmax_columns(values)
    if mode == "sigmoid":
        scaled = np.clip(values.astype(np.float64) * inverse_temperature, -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)
    if mode == "none":
        return values.astype(np.float32, copy=False)
    raise ValueError(f"Unknown raw normalization: {mode}")


def native_state_outputs(
    gallery_features: torch.Tensor,
    state_features: torch.Tensor,
    mapper: CategoryPromptMapper,
    inverse_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return category-softmax and raw-positive projections in 40-D order."""
    scores = (gallery_features.float() @ state_features.float().T).numpy()
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    probabilities = np.zeros_like(scores, dtype=np.float32)
    positive_state: dict[str, str] = {}
    groups: list[list[str]] = []

    for _, columns, state_keys, fallback_key in mapper._MULTI_GROUPS:
        keys = [*state_keys, fallback_key]
        groups.append(keys)
        positive_state.update(zip(columns, state_keys))
    for column, positive_key, negative_key in mapper._BINARY_GROUPS:
        groups.append([positive_key, negative_key])
        positive_state[column] = positive_key

    covered = [key for group in groups for key in group]
    if len(covered) != 52 or len(set(covered)) != 52 or set(covered) != set(mapper.keys):
        raise RuntimeError("Native category groups do not cover the exact 52 training states")

    for keys in groups:
        indices = [state_index[key] for key in keys]
        logits = scores[:, indices].astype(np.float64) * inverse_temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities[:, indices] = (
            exp_logits / exp_logits.sum(axis=1, keepdims=True)
        ).astype(np.float32)

    output_indices = [state_index[positive_state[name]] for name in mapper.attribute_names]
    category_40 = probabilities[:, output_indices]
    raw_40 = scores[:, output_indices].astype(np.float32, copy=False)
    if category_40.shape[1] != 40 or not np.isfinite(category_40).all():
        raise RuntimeError("Invalid native-state 52-to-40 category projection")
    return category_40, raw_40


def positive_aggregation_distances(
    gallery_features: torch.Tensor, positive_features: torch.Tensor, queries: np.ndarray,
) -> np.ndarray:
    """Use the existing normalized positive-state query descriptor."""
    weights = torch.as_tensor(queries, dtype=positive_features.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Method A requires every query to have a positive attribute")
    query_features = F.normalize(weights @ positive_features / counts, dim=-1)
    return (-(query_features @ gallery_features.float().T)).numpy().astype(np.float32, copy=False)


def ranks_at_k(distances: np.ndarray, gallery_ids: np.ndarray,
               ks: Sequence[int]) -> dict[int, float]:
    hits = {k: 0 for k in ks}
    limit = min(max(ks), distances.shape[1])
    # Sort one row at a time: a full QxG int64 argsort would need ~0.9 GiB
    # for the complete validation split in addition to the distance matrix.
    for query_id, row in enumerate(distances):
        order = np.argsort(row, kind="stable")[:limit]
        relevant = gallery_ids[order] == query_id
        for k in ks:
            hits[k] += int(np.any(relevant[:k]))
    return {k: hits[k] / len(distances) for k in ks}


def semantic_query_metrics(distances: np.ndarray, ids: np.ndarray,
                           chunk_size: int) -> dict[str, float]:
    """Rank each image assigned query with the method own distance definition."""
    if distances.shape[1] != len(ids):
        raise ValueError("Distance columns and gallery IDs differ")
    ranks: list[np.ndarray] = []
    query_indices = np.arange(distances.shape[0], dtype=np.int64)[:, None]
    for start in range(0, distances.shape[1], chunk_size):
        stop = min(start + chunk_size, distances.shape[1])
        block = distances[:, start:stop]
        correct_ids = ids[start:stop]
        columns = np.arange(stop - start)
        correct = block[correct_ids, columns]
        rank = 1 + np.sum(
            (block < correct[None, :])
            | ((block == correct[None, :]) & (query_indices < correct_ids[None, :])),
            axis=0,
        )
        ranks.append(rank.astype(np.int64))
    all_ranks = np.concatenate(ranks)
    return {"top1": float(np.mean(all_ranks == 1)), "mean_rank": float(all_ranks.mean())}


def prediction_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index in range(labels.shape[1]):
        metrics = binary_metrics(labels[:, index], predictions[:, index])
        rows.append({
            "index": index, "auroc": metrics["auroc"], "ap": metrics["ap"],
            "f1": metrics["f1"], "mean": float(predictions[:, index].mean()),
            "std": float(predictions[:, index].std()),
        })
    summary = {
        "macro_auroc": float(np.nanmean([row["auroc"] for row in rows])),
        "macro_ap": float(np.nanmean([row["ap"] for row in rows])),
        "macro_f1": float(np.nanmean([row["f1"] for row in rows])),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
        "collapsed_dims": int(np.count_nonzero(predictions.std(axis=0) < 0.02)),
    }
    return summary, rows


def evaluate_method(
    key: str, distances: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
    ids: np.ndarray, attribute_names: Sequence[str], output_dir: Path, chunk_size: int,
) -> dict[str, Any]:
    query_count = int(ids.max()) + 1
    if distances.shape != (query_count, len(ids)) or not np.isfinite(distances).all():
        raise ValueError(f"Method {key} produced invalid distances {distances.shape}")
    if predictions.shape != labels.shape or not np.isfinite(predictions).all():
        raise ValueError(f"Method {key} produced invalid predictions {predictions.shape}")
    rank1, mean_ap = retrieval_metrics(distances, ids)
    rank_k = ranks_at_k(distances, ids, (5, 10))
    result: dict[str, Any] = {
        "method": METHOD_NAMES[key], "rank1": rank1,
        "rank5": rank_k[5], "rank10": rank_k[10],
        "map": mean_ap,
    }
    result.update(semantic_query_metrics(distances, ids, chunk_size))
    prediction_summary, rows = prediction_metrics(labels, predictions)
    result.update(prediction_summary)
    np.save(output_dir / f"{key.lower()}_predictions.npy", predictions)
    with (output_dir / f"{key.lower()}_per_attribute.csv").open(
        "w", encoding="utf-8", newline="",
    ) as handle:
        fields = ("index", "attribute", "auroc", "ap", "f1", "mean", "std")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, attribute in zip(rows, attribute_names):
            writer.writerow({**row, "attribute": attribute})
    print(
        f"{key} {METHOD_NAMES[key]}: R1={100*result['rank1']:.2f}% "
        f"R5={100*result['rank5']:.2f}% R10={100*result['rank10']:.2f}% "
        f"mAP={100*result['map']:.2f}% AUROC={result['macro_auroc']:.4f} "
        f"AP={result['macro_ap']:.4f} F1={result['macro_f1']:.4f} "
        f"SemanticTop1={100*result['top1']:.2f}% MeanRank={result['mean_rank']:.1f} "
        f"mean/std={result['prediction_mean']:.4f}/{result['prediction_std']:.4f} "
        f"collapsed={result['collapsed_dims']}/40", flush=True,
    )
    return result


def print_table(results: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 142)
    print("AttriVision inference comparison (full validation)")
    print("=" * 142)
    print(
        f"{'Method':34s} {'R1':>7s} {'R5':>7s} {'R10':>7s} {'mAP':>7s} "
        f"{'AUROC':>7s} {'AP':>7s} {'F1':>7s} {'SemTop1':>9s} {'MeanRank':>10s} "
        f"{'Pred mean/std':>17s} {'Flat':>6s}"
    )
    for key in METHOD_NAMES:
        row = results[key]
        print(
            f"{key + ' ' + row['method']:34s} {100*row['rank1']:6.2f}% "
            f"{100*row['rank5']:6.2f}% {100*row['rank10']:6.2f}% {100*row['map']:6.2f}% "
            f"{row['macro_auroc']:7.4f} {row['macro_ap']:7.4f} {row['macro_f1']:7.4f} "
            f"{100*row['top1']:8.2f}% {row['mean_rank']:10.1f} "
            f"{row['prediction_mean']:.4f}/{row['prediction_std']:.4f} "
            f"{row['collapsed_dims']:4d}/40"
        )


def interpretation(results: dict[str, dict[str, Any]]) -> list[str]:
    b, c = results["B"], results["C"]
    messages: list[str] = []
    if c["map"] >= b["map"] + 0.02 or c["rank1"] >= b["rank1"] + 0.02:
        messages.append("C가 B보다 크게 좋아서, 학습하지 않은 negative prompt가 핵심 문제로 보입니다.")
    if c["macro_auroc"] >= 0.70 and c["map"] >= 0.10:
        messages.append("C의 AUROC와 retrieval이 모두 양호해 현재 학습보다 inference formulation 문제가 큽니다.")
    elif c["macro_auroc"] >= 0.70:
        messages.append("C의 AUROC는 좋지만 retrieval이 낮아 52→40 calibration/category projection이 병목입니다.")
    else:
        messages.append("C의 AUROC도 낮아 inference만이 아니라 52-state contrastive training 수정이 필요합니다.")
    return messages


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "inference_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    queries_raw, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    attribute_names = list(payload["attribute_names"])
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError(
            "Native 52-state comparison requires a category_complete checkpoint; "
            f"got {payload.get('prompt_mode')!r}"
        )
    queries = reorder_columns(queries_raw, query_names, attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, attribute_names)
    if not np.all(labels == queries[ids]):
        raise RuntimeError("Validation gallery IDs do not reproduce the 40-D ground truth")
    inverse_temperature = (
        1.0 / args.attribute_temperature if args.attribute_temperature is not None
        else learned_inverse_temperature(model)
    )

    print("=" * 72)
    print("AttriVision inference-only comparison")
    print("=" * 72)
    print(f"Checkpoint       : {checkpoint}")
    print(f"Validation       : {gt_path}")
    print(f"Images / queries : {len(table.image_paths)} / {len(queries)}")
    print(f"Device / AMP     : {device} / {args.amp}")
    print(f"Temperature T    : {1.0 / inverse_temperature:.8f}")
    print(f"D normalization  : {args.raw_normalization}")
    print(f"Output           : {output_dir}")

    feature_path = Path(args.gallery_features).resolve() if args.gallery_features else output_dir / "gallery_features.npy"
    if args.gallery_features:
        feature_array = np.load(feature_path)
        if feature_array.ndim != 2 or feature_array.shape[0] != len(table.image_paths):
            raise ValueError(f"Invalid cached gallery features: {feature_array.shape}")
        gallery_features = torch.from_numpy(feature_array.astype(np.float32, copy=False))
        if not torch.isfinite(gallery_features).all():
            raise ValueError("Cached gallery features contain NaN or Inf")
        print(f"Reusing gallery features: {feature_path}")
    else:
        gallery_features = encode_gallery(
            model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(args.image_size), device, args.eval_batch_size,
            args.num_workers, args.amp, args.progress_every, "[shared gallery]",
        )
        np.save(feature_path, gallery_features.numpy())
        print(f"Saved shared gallery features: {feature_path}")

    positive_features = encode_texts(model, prompts_for_attributes(attribute_names), device, args.amp)
    paired_features = encode_prompt_pairs(model, attribute_names, device, args.amp)
    mapper = CategoryPromptMapper(attribute_names)
    if len(mapper.keys) != 52 or len(mapper.prompts) != 52:
        raise RuntimeError("Expected exactly 52 native training states")
    native_features = encode_texts(model, mapper.prompts, device, args.amp)
    category_predictions, raw_projection = native_state_outputs(
        gallery_features, native_features, mapper, inverse_temperature,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: dict[str, dict[str, Any]] = {}
    print("\n[A] positive-only text aggregation")
    distances = positive_aggregation_distances(gallery_features, positive_features, queries)
    # A has no probabilities. Min-max is used only for requested 40-D diagnostics.
    positive_scores = (gallery_features.float() @ positive_features.float().T).numpy()
    results["A"] = evaluate_method(
        "A", distances, minmax_columns(positive_scores), labels, ids,
        attribute_names, output_dir, args.distance_chunk_size,
    )
    del distances, positive_scores

    print("\n[B] independent paired positive/negative prompts")
    predictions = paired_attribute_probabilities(gallery_features, paired_features, inverse_temperature)
    distances = l1_attribute_distances(queries, predictions, args.distance_chunk_size)
    results["B"] = evaluate_method(
        "B", distances, predictions, labels, ids, attribute_names, output_dir,
        args.distance_chunk_size,
    )
    del distances, predictions, paired_features

    print("\n[C] exact 52-state within-category softmax → 40-D")
    distances = l1_attribute_distances(queries, category_predictions, args.distance_chunk_size)
    results["C"] = evaluate_method(
        "C", distances, category_predictions, labels, ids, attribute_names,
        output_dir, args.distance_chunk_size,
    )
    del distances, category_predictions

    print(f"\n[D] exact 52-state raw projection ({args.raw_normalization}) → 40-D")
    predictions = normalize_raw(raw_projection, args.raw_normalization, inverse_temperature)
    distances = l1_attribute_distances(queries, predictions, args.distance_chunk_size)
    results["D"] = evaluate_method(
        "D", distances, predictions, labels, ids, attribute_names, output_dir,
        args.distance_chunk_size,
    )
    del distances, predictions, raw_projection

    print_table(results)
    messages = interpretation(results)
    print("\nInterpretation")
    for message in messages:
        print(f"- {message}")

    summary = {
        "checkpoint": str(checkpoint), "validation": str(gt_path),
        "images": len(table.image_paths), "queries": len(queries),
        "temperature": 1.0 / inverse_temperature,
        "raw_normalization": args.raw_normalization,
        "method_a_diagnostic_normalization": "per-dimension validation minmax",
        "native_state_count": len(mapper.keys), "native_state_keys": mapper.keys,
        "native_state_prompts": mapper.prompts, "results": results,
        "interpretation": messages,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("key", "method", "rank1", "rank5", "rank10", "map", "macro_auroc",
                  "macro_ap", "macro_f1", "top1", "mean_rank", "prediction_mean",
                  "prediction_std", "collapsed_dims")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            writer.writerow({"key": key, **result})
    print(f"\nSaved comparison artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
