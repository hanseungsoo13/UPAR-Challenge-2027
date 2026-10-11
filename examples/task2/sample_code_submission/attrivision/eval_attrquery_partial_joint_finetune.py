"""Evaluate AttrQuery-PartialJointFinetune checkpoints.

This is inference-only.  It evaluates the original A7 checkpoint, optional
AttrQuery-FrozenCLIP-LocalFCE reference, and one or more newly trained partial
joint checkpoints under the same Native52 Category-NLL protocol.  Gallery
features are encoded once per checkpoint and reused for the full fusion alpha
sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_frozenclip_localfce import (  # noqa: E402
    _encode_gallery_scores as encode_attrquery_scores,
    load_experiment_model as load_attrquery_model,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import AttriVisionDataset  # noqa: E402
from attrivision.models.attrivision import AttriVision  # noqa: E402
from attrivision.models.attrquery_partial_joint_finetune import (  # noqa: E402
    AttrQueryPartialJointFinetune,
    load_experiment_model as load_partial_model,
)
from attrivision.native52_retrieval_ablation import category_nll  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import AnnotationTable, ImagePathDataset, find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import autocast, load_retrieval_annotations, official_retrieval_metrics, reorder_columns  # noqa: E402


ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


@dataclass
class RetrievalViews:
    gt_path: Path
    table: AnnotationTable
    queries: np.ndarray
    ids: np.ndarray
    query_names: list[str]


def _retrieval_views(data_root: Path, max_val_samples: int | None) -> RetrievalViews:
    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = list(table.attribute_names)
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
        query_names = list(query_names)
    return RetrievalViews(gt_path, table, queries, ids, query_names)


def _rank_at_k_local(distances: np.ndarray, ids: np.ndarray, k: int) -> float:
    hits = 0
    for query_id, row in enumerate(distances):
        order = np.argsort(row, kind="stable")[:min(k, len(row))]
        hits += int(np.any(ids[order] == query_id))
    return hits / max(len(distances), 1)


def _metric_row(
    distances: np.ndarray,
    views: RetrievalViews,
    model_name: str,
    score_type: str,
    alpha: float | None,
) -> dict[str, Any]:
    official = official_retrieval_metrics(
        distances, views.queries, views.table.labels, views.ids,
    )
    return {
        "model": model_name,
        "score_type": score_type,
        "alpha": "" if alpha is None else float(alpha),
        "rank1": float(official["Rank-1"]),
        "rank5": float(_rank_at_k_local(distances, views.ids, 5)),
        "rank10": float(_rank_at_k_local(distances, views.ids, 10)),
        "mAP": float(official["mAP"]),
        "mADM": float(official["mADM"]),
        "mINP": float(official["mINP"]),
        "images": int(len(views.table.image_paths)),
        "queries": int(len(views.queries)),
    }


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    maximum = values.max(axis=1, keepdims=True)
    return values - maximum - np.log(np.exp(values - maximum).sum(axis=1, keepdims=True))


def _category_log_probabilities(
    global_scores: np.ndarray | None,
    local_scores: np.ndarray | None,
    groups: Sequence[np.ndarray],
    temperature: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    global_log = None
    local_log = None
    if global_scores is not None:
        global_log = np.zeros((len(global_scores), 52), dtype=np.float64)
        for indices in groups:
            global_log[:, indices] = _log_softmax(global_scores[:, indices] / temperature)
    if local_scores is not None:
        local_log = np.zeros((len(local_scores), 12, 52), dtype=np.float64)
        for category_index, indices in enumerate(groups):
            local_log[:, category_index, indices] = _log_softmax(
                local_scores[:, category_index, indices] / temperature,
            )
    return global_log, local_log


def _category_nll_from_log_probabilities(
    queries52: np.ndarray,
    log_probabilities: np.ndarray,
    groups: Sequence[np.ndarray],
) -> np.ndarray:
    # This is algebraically the existing native52_retrieval_ablation.category_nll
    # implementation, but consumes already computed log-probabilities so the
    # fusion path never mixes raw logits.
    probabilities = np.exp(log_probabilities).astype(np.float32)
    return category_nll(queries52, probabilities, groups)


def _fuse_log_probabilities(
    global_log: np.ndarray,
    local_log: np.ndarray,
    groups: Sequence[np.ndarray],
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("fusion alpha must be in [0,1]")
    fused = np.zeros_like(global_log, dtype=np.float64)
    for category_index, indices in enumerate(groups):
        blended = (
            (1.0 - alpha) * global_log[:, indices]
            + alpha * local_log[:, category_index, indices]
        )
        # Convex log-probability interpolation is renormalized within each
        # category, so the result remains a valid category distribution.
        fused[:, indices] = blended - np.logaddexp.reduce(blended, axis=1, keepdims=True)
    return fused


def _save_or_load_array(
    path: Path,
    expected_shape: tuple[int, ...],
    producer: Any,
) -> np.ndarray:
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if cached.shape == expected_shape and np.isfinite(cached).all():
            print(f"Reusing gallery cache: {path}", flush=True)
            return cached.astype(np.float32, copy=False)
        print(f"Ignoring incompatible gallery cache: {path}", flush=True)
    value = np.asarray(producer(), dtype=np.float32)
    if value.shape != expected_shape or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid encoded gallery shape={value.shape}, expected={expected_shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    print(f"Saved gallery cache: {path}", flush=True)
    return value


def _encode_global_gallery(
    model: AttriVision | AttrQueryPartialJointFinetune,
    paths: Sequence[str],
    roots: Sequence[Path],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(paths, roots, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for images in loader:
        with autocast(device, amp):
            if isinstance(model, AttrQueryPartialJointFinetune):
                features = model.global_features(images.to(device, non_blocking=True))
            else:
                features = model.encode_image(images.to(device, non_blocking=True))
        chunks.append(features.float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 512), dtype=np.float32)


def _encode_joint_local_gallery(
    model: AttrQueryPartialJointFinetune,
    paths: Sequence[str],
    roots: Sequence[Path],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(paths, roots, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for images in loader:
        with autocast(device, amp):
            scores = model.category_state_scores(images.to(device, non_blocking=True))
        chunks.append(scores.float().cpu().numpy())
    return (
        np.concatenate(chunks, axis=0)
        if chunks else np.empty((0, 12, 52), dtype=np.float32)
    )


def _text_prototypes(
    model: AttriVision | AttrQueryPartialJointFinetune,
    mapper: CategoryPromptMapper,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    if isinstance(model, AttrQueryPartialJointFinetune):
        return model.text_prototypes.detach().float().cpu().numpy()
    tokens = model.tokenize(mapper.prompts).to(device)
    model.eval()
    with torch.inference_mode(), autocast(device, amp):
        features = model.encode_text(tokens)
    return features.float().cpu().numpy()


def _score_model(
    model_name: str,
    model: Any,
    views: RetrievalViews,
    mapper: CategoryPromptMapper,
    groups: Sequence[np.ndarray],
    data_root: Path,
    output_dir: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    temperature: float,
    alphas: Sequence[float],
    reference_kind: str,
) -> list[dict[str, Any]]:
    if len(mapper.keys) != 52 or len(groups) != 12:
        raise RuntimeError("Evaluation requires the exact 52-state, 12-category A7 mapping")
    roots = [data_root, views.gt_path.parent, REPOSITORY_ROOT]
    cache_prefix = model_name.replace("/", "_").replace(" ", "_")
    text_features = _text_prototypes(model, mapper, device, amp)
    global_scores: np.ndarray | None = None
    local_scores: np.ndarray | None = None

    if reference_kind in {"a7", "partial"}:
        global_features = _save_or_load_array(
            output_dir / f"{cache_prefix}_global_features.npy",
            (len(views.table.image_paths), 512),
            lambda: _encode_global_gallery(
                model, views.table.image_paths, roots, transform, device,
                batch_size, num_workers, amp,
            ),
        )
        global_scores = global_features @ text_features.T
    if reference_kind == "partial":
        local_scores = _save_or_load_array(
            output_dir / f"{cache_prefix}_local_scores.npy",
            (len(views.table.image_paths), 12, 52),
            lambda: _encode_joint_local_gallery(
                model, views.table.image_paths, roots, transform, device,
                batch_size, num_workers, amp,
            ),
        )
    elif reference_kind == "attrquery":
        local_scores = _save_or_load_array(
            output_dir / f"{cache_prefix}_local_scores.npy",
            (len(views.table.image_paths), 12, 52),
            lambda: encode_attrquery_scores(
                model, views.table.image_paths, roots, transform, device,
                batch_size, num_workers, amp, 0,
            ),
        )

    global_log, local_log = _category_log_probabilities(
        global_scores, local_scores, groups, temperature,
    )
    queries = reorder_columns(views.queries, views.query_names, mapper.attribute_names)
    labels = reorder_columns(views.table.labels, views.table.attribute_names, mapper.attribute_names)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    eval_table = AnnotationTable(
        views.table.image_paths, labels, list(mapper.attribute_names),
    )
    eval_views = RetrievalViews(views.gt_path, eval_table, queries, views.ids, mapper.attribute_names)
    rows: list[dict[str, Any]] = []

    if global_log is not None:
        distances = _category_nll_from_log_probabilities(queries52, global_log, groups)
        rows.append(_metric_row(distances, eval_views, model_name, "global-only", 0.0))
    if local_log is not None:
        local_full = np.zeros((len(local_log), 52), dtype=np.float64)
        for category_index, indices in enumerate(groups):
            local_full[:, indices] = local_log[:, category_index, indices]
        distances = _category_nll_from_log_probabilities(queries52, local_full, groups)
        rows.append(_metric_row(distances, eval_views, model_name, "local-only", 1.0))

    if global_log is not None and local_log is not None:
        for alpha in alphas:
            fused_log = _fuse_log_probabilities(global_log, local_log, groups, float(alpha))
            distances = _category_nll_from_log_probabilities(queries52, fused_log, groups)
            rows.append(_metric_row(distances, eval_views, model_name, "fusion", float(alpha)))
    return rows


def _parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        if not label or not path:
            raise ValueError(f"Checkpoint must use LABEL=PATH, got {spec!r}")
        return label, Path(path).resolve()
    path = Path(spec).resolve()
    return path.parent.name or path.stem, path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _feature_drift_rows(output_dir: Path, model_names: Sequence[str]) -> list[dict[str, Any]]:
    original_path = output_dir / "Original-A7_global_features.npy"
    if not original_path.is_file():
        return []
    original = np.load(original_path, allow_pickle=False).astype(np.float32, copy=False)
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        if model_name == "Original-A7":
            continue
        path = output_dir / f"{model_name.replace('/', '_').replace(' ', '_')}_global_features.npy"
        if not path.is_file():
            continue
        current = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if current.shape != original.shape:
            continue
        delta = current - original
        rows.append({
            "model": model_name,
            "images": int(len(current)),
            "mean_cosine_to_original_a7": float(np.mean(np.sum(current * original, axis=1))),
            "mean_l2_change": float(np.mean(np.linalg.norm(delta, axis=1))),
            "max_l2_change": float(np.max(np.linalg.norm(delta, axis=1))),
        })
    return rows


def _plot_curve(rows: Sequence[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Evaluation curves require matplotlib") from exc
    fusion_rows = [row for row in rows if row["score_type"] == "fusion"]
    if not fusion_rows:
        return
    models = list(dict.fromkeys(str(row["model"]) for row in fusion_rows))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("rank1", "mAP", "mADM"), ("Rank-1", "mAP", "mADM"),
    ):
        for model in models:
            selected = [row for row in fusion_rows if row["model"] == model]
            selected.sort(key=lambda row: float(row["alpha"]))
            axis.plot(
                [float(row["alpha"]) for row in selected],
                [100.0 * float(row[metric]) for row in selected],
                marker="o", label=model,
            )
        axis.set_xlabel("Fusion alpha")
        axis.set_ylabel(f"{title} (%)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AttrQuery-PartialJointFinetune with Native52 Category-NLL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="joint checkpoint as LABEL=PATH; repeat for Frozen/Last1/Last2",
    )
    parser.add_argument(
        "--original-a7-checkpoint",
        default="outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth",
    )
    parser.add_argument("--attrquery-checkpoint")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument(
        "--output-dir", default="outputs/AttrQuery-PartialJointFinetune/evaluation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=list(ALPHAS),
        help="fusion alpha values evaluated on category log-probabilities",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.image_size != 224:
        raise ValueError("ViT-B/32 evaluation requires image-size=224 and valid loader sizes")
    if args.category_temperature <= 0:
        raise ValueError("category-temperature must be positive")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("all fusion alphas must be in [0,1]")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("fusion alphas must be unique")

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    views = _retrieval_views(data_root, args.max_val_samples)
    attribute_names = list(views.table.attribute_names)
    mapper = CategoryPromptMapper(attribute_names)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]

    all_rows: list[dict[str, Any]] = []
    model_paths: dict[str, str] = {}

    original_path = Path(args.original_a7_checkpoint).resolve()
    if original_path.is_file():
        original_model = None
        from attrivision.checkpoint import load_model as load_a7_model
        original_model, original_payload = load_a7_model(original_path, device)
        if original_payload.get("prompt_mode") != "category_complete":
            raise ValueError("Original A7 checkpoint must use category_complete")
        if list(original_payload["attribute_names"]) != attribute_names:
            raise ValueError("Original A7 and validation attributes are not in the same order")
        all_rows.extend(_score_model(
            "Original-A7", original_model, views, mapper, groups, data_root, output_dir,
            build_eval_transform(args.image_size, "resize_pad_crop"), device,
            args.eval_batch_size, args.num_workers, args.amp,
            args.category_temperature, args.alphas, "a7",
        ))
        model_paths["Original-A7"] = str(original_path)
        del original_model
    else:
        print(f"Original A7 checkpoint not found; skipping reference: {original_path}", flush=True)

    if args.attrquery_checkpoint:
        attrquery_path = Path(args.attrquery_checkpoint).resolve()
        attrquery_model, attrquery_payload = load_attrquery_model(attrquery_path, device)
        if attrquery_payload.get("feature_mode") != "local":
            raise ValueError("The AttrQuery reference must use feature_mode=local")
        if list(attrquery_payload["attribute_names"]) != attribute_names:
            raise ValueError("AttrQuery and validation attributes are not in the same order")
        all_rows.extend(_score_model(
            "AttrQuery-FrozenCLIP-LocalFCE", attrquery_model, views, mapper, groups,
            data_root, output_dir, build_eval_transform(args.image_size, "resize_pad_crop"),
            device, args.eval_batch_size, args.num_workers, args.amp,
            args.category_temperature, args.alphas, "attrquery",
        ))
        model_paths["AttrQuery-FrozenCLIP-LocalFCE"] = str(attrquery_path)
        del attrquery_model

    for spec in args.checkpoint:
        label, checkpoint_path = _parse_checkpoint_spec(spec)
        model, payload = load_partial_model(checkpoint_path, device)
        if list(payload["attribute_names"]) != attribute_names:
            raise ValueError(f"{label}: checkpoint and validation attribute order differ")
        all_rows.extend(_score_model(
            label, model, views, mapper, groups, data_root, output_dir,
            build_eval_transform(args.image_size, "resize_pad_crop"), device,
            args.eval_batch_size, args.num_workers, args.amp,
            args.category_temperature, args.alphas, "partial",
        ))
        model_paths[label] = str(checkpoint_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(output_dir / "fusion_alpha_sweep.csv", all_rows)
    summary_rows = [
        row for row in all_rows
        if row["score_type"] in {"global-only", "local-only"}
        or (row["score_type"] == "fusion" and float(row["alpha"]) in {0.5})
    ]
    _write_csv(output_dir / "metrics_summary.csv", summary_rows)
    drift_rows = _feature_drift_rows(output_dir, list(model_paths))
    _write_csv(output_dir / "feature_drift.csv", drift_rows)
    original_rows = [
        row for row in all_rows
        if row["model"] == "Original-A7" and row["score_type"] == "global-only"
    ]
    baseline_comparison: list[dict[str, Any]] = []
    if original_rows:
        baseline = original_rows[0]
        for row in all_rows:
            if row["model"] == "Original-A7":
                continue
            baseline_comparison.append({
                "model": row["model"],
                "score_type": row["score_type"],
                "alpha": row["alpha"],
                "delta_rank1_vs_original_a7": float(row["rank1"]) - float(baseline["rank1"]),
                "delta_mAP_vs_original_a7": float(row["mAP"]) - float(baseline["mAP"]),
                "delta_mADM_vs_original_a7": float(row["mADM"]) - float(baseline["mADM"]),
                "delta_mINP_vs_original_a7": float(row["mINP"]) - float(baseline["mINP"]),
            })
    _write_csv(output_dir / "comparison_vs_original_a7.csv", baseline_comparison)

    best: dict[str, dict[str, Any]] = {}
    for model_name in dict.fromkeys(str(row["model"]) for row in all_rows):
        candidates = [row for row in all_rows if row["model"] == model_name and row["score_type"] == "fusion"]
        if candidates:
            selected = max(candidates, key=lambda row: (float(row["mADM"]), -float(row["alpha"])))
            best[model_name] = selected
    _plot_curve(all_rows, output_dir / "fusion_alpha_curve.png")

    summary = {
        "protocol": "native52_category_nll",
        "category_temperature": float(args.category_temperature),
        "alphas": [float(alpha) for alpha in args.alphas],
        "validation_annotation": str(views.gt_path),
        "images": len(views.table.image_paths),
        "queries": len(views.queries),
        "model_checkpoints": model_paths,
        "best_fusion_by_mADM": best,
        "rows": all_rows,
        "feature_drift_vs_original_a7": drift_rows,
        "comparison_vs_original_a7": baseline_comparison,
        "score_definition": {
            "global": "A7 cosine similarities -> per-category softmax -> Category-NLL",
            "local": "category-local cosine similarities -> per-category softmax -> Category-NLL",
            "fusion": "renormalized convex interpolation of global/local category log-probabilities",
        },
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    with (output_dir / "evaluation_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# AttrQuery-PartialJointFinetune Evaluation\n\n")
        handle.write("Protocol: Native52 `category_complete` Category-NLL.\n\n")
        handle.write("Gallery features are cached once per model and reused across alpha values.\n\n")
        handle.write("## Best Fusion by mADM\n\n")
        handle.write("| Model | Alpha | Rank-1 | mAP | mADM | mINP |\n|---|---:|---:|---:|---:|---:|\n")
        for model_name, row in best.items():
            handle.write(
                f"| {model_name} | {float(row['alpha']):.2f} | "
                f"{100 * float(row['rank1']):.2f}% | {100 * float(row['mAP']):.2f}% | "
                f"{100 * float(row['mADM']):.2f}% | {100 * float(row['mINP']):.2f}% |\n"
            )
        handle.write("\nFusion uses category-wise log-probability interpolation followed by "
                     "category renormalization; raw logits are never mixed.\n")
    print(f"Saved evaluation results to: {output_dir}")


if __name__ == "__main__":
    main()
