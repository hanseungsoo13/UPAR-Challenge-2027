"""Compare A7 global and AttrQuery local category predictions.

This is an inference-only diagnostic.  It uses the same validation gallery,
52-state CategoryPromptMapper, category temperature, and Category-NLL state
probabilities for both models.  The per-category NLL is the mean negative
log-probability assigned to the ground-truth state of each validation image.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

TOOLS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = TOOLS_DIR.parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.models.attrivision import AttriVision  # noqa: E402
from attrivision.models.attrquery_fromscratch_jointfce import (  # noqa: E402
    load_experiment_model as load_fromscratch_model,
    validate_mapper,
)
from attrivision.models.attrquery_frozenclip_localfce import (  # noqa: E402
    AttrQueryFrozenCLIPLocalFCE,
)
from attrivision.models.attrquery_partial_joint_finetune import (  # noqa: E402
    load_experiment_model as load_partial_model,
)
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import (  # noqa: E402
    AnnotationTable,
    ImagePathDataset,
    find_annotation_file,
    read_gt_csv,
)
from upar.retrieval import (  # noqa: E402
    autocast,
    load_retrieval_annotations,
    reorder_columns,
)


DEFAULT_OUTPUT = "outputs/AttrQuery-Categorywise-Diagnostics"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def _checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_valid(
    path: Path,
    manifest_path: Path,
    checkpoint: Path,
    image_paths: Sequence[str],
    shape: tuple[int, ...],
) -> bool:
    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = np.load(path, allow_pickle=False)
        return (
            manifest.get("checkpoint") == _checkpoint_fingerprint(checkpoint)
            and manifest.get("image_paths") == list(image_paths)
            and tuple(values.shape) == shape
            and np.isfinite(values).all()
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _save_cache(
    path: Path,
    values: np.ndarray,
    checkpoint: Path,
    image_paths: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float32, copy=False), allow_pickle=False)
    _write_json(path.with_suffix(path.suffix + ".json"), {
        "checkpoint": _checkpoint_fingerprint(checkpoint),
        "image_paths": list(image_paths),
        "shape": list(values.shape),
    })


def _validation_views(
    data_root: Path,
    max_val_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
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
    return gt_path, table, queries, ids, query_names


def _category_names(mapper: CategoryPromptMapper) -> list[str]:
    names = [spec[0] for spec in mapper._MULTI_GROUPS]
    names.extend(spec[0] for spec in mapper._BINARY_GROUPS)
    return names


def _category_log_probabilities(
    scores: np.ndarray,
    groups: Sequence[np.ndarray],
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    if scores.ndim == 2:
        category_scores = np.broadcast_to(
            scores[:, None, :], (len(scores), len(groups), scores.shape[1]),
        )
    elif scores.ndim == 3:
        category_scores = scores
    else:
        raise ValueError(f"Expected [G,52] or [G,12,52], got {scores.shape}")
    if category_scores.shape[1:] != (len(groups), 52):
        raise ValueError(
            f"Score shape {category_scores.shape} does not match {len(groups)} groups and 52 states"
        )
    result = np.full((len(scores), 52), -np.inf, dtype=np.float64)
    for category_index, indices in enumerate(groups):
        values = category_scores[:, category_index, indices].astype(np.float64)
        values = values / temperature
        maximum = values.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(
            np.exp(values - maximum).sum(axis=1, keepdims=True),
        )
        result[:, indices] = values - log_normalizer
    if not np.isfinite(result).all():
        raise RuntimeError("Category log-probabilities contain NaN or Inf")
    return result.astype(np.float32)


def _load_local_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], str]:
    payload = _load_payload(checkpoint)
    architecture = payload.get("architecture")
    if architecture == "attrquery_fromscratch_jointfce":
        model, payload = load_fromscratch_model(checkpoint, device)
        return model, payload, "fromscratch"
    if architecture == "attrquery_partial_joint_finetune":
        model, payload = load_partial_model(checkpoint, device)
        return model, payload, "partial"
    if architecture == "attrquery_frozenclip_localfce":
        attributes = list(payload["attribute_names"])
        state = payload["model_state_dict"]
        text_prototypes = state.get("text_prototypes", payload.get("text_prototypes"))
        if text_prototypes is None:
            raise ValueError("Frozen AttrQuery checkpoint has no text prototypes")
        base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
        model = AttrQueryFrozenCLIPLocalFCE(
            base,
            attributes,
            text_prototypes,
            feature_mode=payload.get("feature_mode", "local"),
            num_heads=int(payload.get("num_heads", 8)),
            dropout=float(payload.get("dropout", 0.1)),
            use_ffn=bool(payload.get("use_ffn", True)),
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Frozen AttrQuery checkpoint mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        model.to(device)
        model.eval()
        return model, payload, "frozen"
    raise ValueError(
        "Unsupported Local checkpoint architecture. Expected one of: "
        "attrquery_fromscratch_jointfce, attrquery_partial_joint_finetune, "
        f"attrquery_frozenclip_localfce; got {architecture!r}"
    )


@torch.inference_mode()
def _encode_a7_scores(
    model: AttriVision,
    table: AnnotationTable,
    roots: Sequence[Path],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    state_features: torch.Tensor,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(table.image_paths, roots, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            features = model.encode_image(images.to(device, non_blocking=True))
            scores = features @ state_features.T
        chunks.append(scores.float().cpu().numpy())
        if batch_index == 1 or batch_index == len(loader) or batch_index % 25 == 0:
            print(f"[A7 global] batch {batch_index}/{len(loader)}", flush=True)
    result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    expected = (len(table.image_paths), 52)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid A7 score shape {result.shape}; expected {expected}")
    return result


@torch.inference_mode()
def _encode_local_scores(
    model: Any,
    model_kind: str,
    table: AnnotationTable,
    roots: Sequence[Path],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    state_features: torch.Tensor | None,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(table.image_paths, roots, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with autocast(device, amp):
            if model_kind == "fromscratch":
                if state_features is None:
                    raise RuntimeError("From-scratch local model needs encoded state features")
                _, local_features = model.joint_features(images)
                scores = model.category_state_scores_from_features(local_features, state_features)
            else:
                scores = model.category_state_scores(images)
        chunks.append(scores.float().cpu().numpy())
        if batch_index == 1 or batch_index == len(loader) or batch_index % 25 == 0:
            print(f"[AttrQuery {model_kind} local] batch {batch_index}/{len(loader)}", flush=True)
    result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    expected = (len(table.image_paths), 12, 52)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid Local score shape {result.shape}; expected {expected}")
    return result


def _category_rows(
    global_scores: np.ndarray,
    local_scores: np.ndarray,
    target52: np.ndarray,
    groups: Sequence[np.ndarray],
    names: Sequence[str],
    temperature: float,
) -> list[dict[str, Any]]:
    global_log = _category_log_probabilities(global_scores, groups, temperature)
    local_log = _category_log_probabilities(local_scores, groups, temperature)
    rows: list[dict[str, Any]] = []
    gallery_indices = np.arange(len(target52))
    for category_index, (name, indices) in enumerate(zip(names, groups)):
        target_local = target52[:, indices].argmax(axis=1)
        target_state = indices[target_local]
        global_nll = -float(np.mean(global_log[gallery_indices, target_state]))
        local_nll = -float(np.mean(local_log[gallery_indices, target_state]))
        global_prediction = global_log[:, indices].argmax(axis=1)
        local_prediction = local_log[:, indices].argmax(axis=1)
        global_accuracy = float(np.mean(global_prediction == target_local))
        local_accuracy = float(np.mean(local_prediction == target_local))
        rows.append({
            "category_index": category_index,
            "category": name,
            "state_count": len(indices),
            "gallery_images": len(target52),
            "global_target_state_nll": global_nll,
            "local_target_state_nll": local_nll,
            "local_minus_global_nll": local_nll - global_nll,
            "global_minus_local_nll_improvement": global_nll - local_nll,
            "global_accuracy": global_accuracy,
            "local_accuracy": local_accuracy,
            "local_minus_global_accuracy": local_accuracy - global_accuracy,
            "local_better_nll": local_nll < global_nll,
            "local_better_accuracy": local_accuracy > global_accuracy,
        })
    return rows


def _plot_results(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["category"] for row in rows]
    x = np.arange(len(labels))
    width = 0.38
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [row["global_target_state_nll"] for row in rows], width, label="A7 Global")
    ax.bar(x + width / 2, [row["local_target_state_nll"] for row in rows], width, label="AttrQuery Local")
    ax.set_ylabel("Ground-truth state NLL (lower is better)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "category_nll_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [100 * row["global_accuracy"] for row in rows], width, label="A7 Global")
    ax.bar(x + width / 2, [100 * row["local_accuracy"] for row in rows], width, label="AttrQuery Local")
    ax.set_ylabel("Category accuracy (%)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "category_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, [row["global_minus_local_nll_improvement"] for row in rows], color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("NLL improvement: Global − Local")
    axes[0].set_ylabel("Positive = Local better")
    axes[1].bar(x, [100 * row["local_minus_global_accuracy"] for row in rows], color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Accuracy improvement: Local − Global")
    axes[1].set_ylabel("Percentage points")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "categorywise_improvement.png", dpi=180)
    plt.close(fig)


def _write_summary(
    path: Path,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    local_nll_wins = [row["category"] for row in rows if row["local_better_nll"]]
    global_nll_wins = [row["category"] for row in rows if not row["local_better_nll"]]
    local_acc_wins = [row["category"] for row in rows if row["local_better_accuracy"]]
    global_acc_wins = [row["category"] for row in rows if not row["local_better_accuracy"]]
    lines = [
        "# Category-wise A7 Global vs AttrQuery Local Diagnostics",
        "",
        "The comparison uses the same validation gallery, 52-state vocabulary, "
        "12-category partition, and category temperature for both models.",
        "",
        f"- A7 checkpoint: `{metadata['a7_checkpoint']}`",
        f"- Local checkpoint: `{metadata['local_checkpoint']}`",
        f"- Local architecture: `{metadata['local_architecture']}`",
        f"- Local text features: `{metadata['local_text_source']}`",
        f"- Gallery images: `{metadata['gallery_images']}`",
        f"- Category temperature: `{metadata['category_temperature']}`",
        "",
        "## Category summary",
        "",
        "| Category | A7 NLL | Local NLL | NLL Δ (A7−Local) | A7 Acc. | Local Acc. | Acc. Δ (Local−A7) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['global_target_state_nll']:.4f} | "
            f"{row['local_target_state_nll']:.4f} | "
            f"{row['global_minus_local_nll_improvement']:.4f} | "
            f"{100*row['global_accuracy']:.2f}% | {100*row['local_accuracy']:.2f}% | "
            f"{100*row['local_minus_global_accuracy']:.2f} pp |"
        )
    lines.extend([
        "",
        f"- Local lower-NLL categories: {', '.join(local_nll_wins) or 'none'}",
        f"- Global lower-NLL categories: {', '.join(global_nll_wins) or 'none'}",
        f"- Local higher-accuracy categories: {', '.join(local_acc_wins) or 'none'}",
        f"- Global higher/equal-accuracy categories: {', '.join(global_acc_wins) or 'none'}",
        "",
        "## Interpretation",
        "",
        "A Local advantage in a category means the category-specific score "
        "assigns more probability to the ground-truth state or predicts the "
        "state more often. Similar Global and Local values indicate that the "
        "local branch may not be adding much category-specific information. "
        "These metrics do not by themselves prove correct spatial localization.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare A7 Global and AttrQuery Local category NLL/accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a7-checkpoint", required=True)
    parser.add_argument("--local-checkpoint", required=True)
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid evaluation batch size or worker count")
    if args.image_size != 224:
        raise ValueError("CLIP ViT-B/32 and the existing evaluation protocol require image size 224")
    if args.category_temperature <= 0:
        raise ValueError("category temperature must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")

    set_seed(args.seed, deterministic=False)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    a7_checkpoint = Path(args.a7_checkpoint).resolve()
    local_checkpoint = Path(args.local_checkpoint).resolve()
    if not a7_checkpoint.is_file():
        raise FileNotFoundError(f"A7 checkpoint not found: {a7_checkpoint}")
    if not local_checkpoint.is_file():
        raise FileNotFoundError(f"Local checkpoint not found: {local_checkpoint}")

    gt_path, table, queries_raw, ids, query_names = _validation_views(
        data_root, args.max_val_samples,
    )
    a7_model, a7_payload = load_model(a7_checkpoint, device)
    if a7_payload.get("prompt_mode") != "category_complete":
        raise ValueError("A7 checkpoint must use prompt_mode=category_complete")
    mapper = CategoryPromptMapper(list(a7_payload["attribute_names"]))
    validate_mapper(mapper)
    if list(table.attribute_names) != mapper.attribute_names:
        raise ValueError("Validation annotation attributes do not match A7 checkpoint attributes")
    local_model, local_payload, local_kind = _load_local_model(local_checkpoint, device)
    local_model.eval()
    if list(local_payload.get("attribute_names", [])) != mapper.attribute_names:
        raise ValueError("Local checkpoint attributes do not match A7 checkpoint attributes")
    if getattr(local_model, "feature_mode", "local") != "local":
        raise ValueError("Local checkpoint must expose category-local features")

    queries = reorder_columns(queries_raw, query_names, mapper.attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, mapper.attribute_names)
    del queries, ids  # Categorywise metrics use every aligned validation gallery image.
    target52 = mapper.encode(torch.from_numpy(labels).float()).numpy()
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    transform = build_eval_transform(args.image_size, "resize_pad_crop")
    roots = [data_root, gt_path.parent, REPOSITORY_ROOT]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    a7_tokens = a7_model.tokenize(mapper.prompts).to(device)
    with torch.inference_mode(), autocast(device, args.amp):
        a7_text = a7_model.encode_text(a7_tokens).float()
    a7_cache = output_dir / "a7_global_scores.npy"
    a7_manifest = a7_cache.with_suffix(a7_cache.suffix + ".json")
    if not args.no_cache and _cache_valid(
        a7_cache, a7_manifest, a7_checkpoint, table.image_paths,
        (len(table.image_paths), 52),
    ):
        global_scores = np.load(a7_cache, allow_pickle=False).astype(np.float32, copy=False)
        print(f"Reusing A7 score cache: {a7_cache}", flush=True)
    else:
        global_scores = _encode_a7_scores(
            a7_model, table, roots, transform, device, args.eval_batch_size,
            args.num_workers, args.amp, a7_text,
        )
        _save_cache(a7_cache, global_scores, a7_checkpoint, table.image_paths)
    del a7_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    state_features: torch.Tensor | None = None
    if local_kind == "fromscratch":
        local_tokens = local_model.base.tokenize(mapper.prompts).to(device)
        with torch.inference_mode(), autocast(device, args.amp):
            state_features = local_model.encode_text(local_tokens).float()
    local_cache = output_dir / f"local_{_safe_name(local_kind)}_scores.npy"
    local_manifest = local_cache.with_suffix(local_cache.suffix + ".json")
    if not args.no_cache and _cache_valid(
        local_cache, local_manifest, local_checkpoint, table.image_paths,
        (len(table.image_paths), 12, 52),
    ):
        local_scores = np.load(local_cache, allow_pickle=False).astype(np.float32, copy=False)
        print(f"Reusing Local score cache: {local_cache}", flush=True)
    else:
        local_scores = _encode_local_scores(
            local_model, local_kind, table, roots, transform, device,
            args.eval_batch_size, args.num_workers, args.amp, state_features,
        )
        _save_cache(local_cache, local_scores, local_checkpoint, table.image_paths)

    category_rows = _category_rows(
        global_scores, local_scores, target52, groups, _category_names(mapper),
        args.category_temperature,
    )
    _write_csv(output_dir / "category_global_local_comparison.csv", category_rows)
    _plot_results(category_rows, output_dir)
    metadata = {
        "a7_checkpoint": str(a7_checkpoint),
        "local_checkpoint": str(local_checkpoint),
        "local_architecture": local_payload.get("architecture"),
        "local_model_kind": local_kind,
        "local_text_source": (
            "current_local_text_encoder"
            if local_kind == "fromscratch" else "checkpoint_cached_text_prototypes"
        ),
        "data_root": str(data_root),
        "validation_annotation": str(gt_path),
        "gallery_images": len(table.image_paths),
        "states": 52,
        "categories": 12,
        "category_temperature": args.category_temperature,
        "score_definition": "category-wise softmax over cosine state scores, then ground-truth state NLL",
        "same_validation_gallery": True,
        "same_attribute_order": True,
        "seed": args.seed,
    }
    _write_json(output_dir / "analysis_metadata.json", metadata)
    _write_summary(output_dir / "analysis_summary.md", category_rows, metadata)
    print(
        f"Categorywise analysis complete: local={local_kind}; "
        f"output={output_dir}; images={len(table.image_paths)}; device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
