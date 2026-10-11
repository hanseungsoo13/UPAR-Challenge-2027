"""Shared training/evaluation pipeline for AttrQuery-FrozenCLIP-LocalFCE."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import AttriVisionDataset  # noqa: E402
from attrivision.engine.trainer_attrivision import (  # noqa: E402
    _grad_scaler, _optimizer_step, build_scheduler,
)
from attrivision.losses.category_local_fce import CategoryLocalFCELoss  # noqa: E402
from attrivision.models.attrquery_frozenclip_localfce import (  # noqa: E402
    AttrQueryFrozenCLIPLocalFCE, build_text_prototypes,
)
from attrivision.models.attrivision import AttriVision  # noqa: E402
from attrivision.native52_retrieval_ablation import (  # noqa: E402
    category_nll, category_softmax,
)
from attrivision.transforms import build_eval_transform, build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import (  # noqa: E402
    AnnotationTable, ImagePathDataset, find_annotation_file, read_gt_csv,
)
from upar.retrieval import (  # noqa: E402
    autocast, load_retrieval_annotations, official_retrieval_metrics,
    reorder_columns,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "attrquery_frozenclip_localfce.json"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("architecture") != "attrquery_frozenclip_localfce":
        raise ValueError(f"Not an AttrQuery-FrozenCLIP-LocalFCE checkpoint: {path}")
    required = {"model_state_dict", "attribute_names", "base_model_name", "feature_mode"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"AttrQuery checkpoint is missing keys: {sorted(missing)}")
    return payload


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _rng_payload() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def category_names(mapper: CategoryPromptMapper) -> list[str]:
    names = [spec[0] for spec in mapper._MULTI_GROUPS]
    names.extend(spec[0] for spec in mapper._BINARY_GROUPS)
    if len(names) != 12:
        raise RuntimeError(f"Expected 12 category names, got {len(names)}")
    return names


def validate_mapper(mapper: CategoryPromptMapper) -> None:
    if len(mapper.keys) != 52 or len(mapper.prompts) != 52:
        raise RuntimeError("AttrQuery-FrozenCLIP-LocalFCE requires the exact 52-state vocabulary")
    groups = mapper.category_indices()
    covered = [index for group in groups for index in group]
    if len(groups) != 12 or sorted(covered) != list(range(52)):
        raise RuntimeError("The A7 category mapping is not a 12-category partition of 52 states")


def _valid_category_mask(semantic_targets: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    masks = []
    for group in groups:
        indices = torch.as_tensor(group, device=semantic_targets.device, dtype=torch.long)
        masks.append(semantic_targets[:, indices].sum(dim=1) > 0)
    return torch.stack(masks, dim=1)


def _build_model_from_a7(
    checkpoint: Path,
    attribute_names: Sequence[str],
    device: torch.device,
    amp: bool,
    feature_mode: str,
    num_heads: int,
    dropout: float,
    use_ffn: bool,
) -> tuple[AttrQueryFrozenCLIPLocalFCE, dict[str, Any]]:
    base, a7_payload = load_model(checkpoint, device)
    if a7_payload.get("prompt_mode") != "category_complete":
        raise ValueError(
            "The initialization checkpoint must be the original A7 category_complete checkpoint"
        )
    if list(a7_payload["attribute_names"]) != list(attribute_names):
        raise ValueError("A7 checkpoint attribute order differs from the training annotation order")
    mapper = CategoryPromptMapper(attribute_names)
    validate_mapper(mapper)
    text_prototypes = build_text_prototypes(base, mapper, device, amp)
    model = AttrQueryFrozenCLIPLocalFCE(
        base, attribute_names, text_prototypes, feature_mode,
        num_heads=num_heads, dropout=dropout, use_ffn=use_ffn,
    ).to(device)
    model.assert_frozen_backbone()
    return model, a7_payload


def load_experiment_model(
    checkpoint: str | Path, device: torch.device,
) -> tuple[AttrQueryFrozenCLIPLocalFCE, dict[str, Any]]:
    payload = _load_payload(checkpoint, map_location="cpu")
    attributes = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attributes)
    validate_mapper(mapper)
    state = payload["model_state_dict"]
    text_prototypes = state.get("text_prototypes", payload.get("text_prototypes"))
    if text_prototypes is None:
        raise ValueError("AttrQuery checkpoint does not contain cached text prototypes")
    base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
    model = AttrQueryFrozenCLIPLocalFCE(
        base, attributes, text_prototypes,
        feature_mode=payload["feature_mode"],
        num_heads=int(payload.get("num_heads", 8)),
        dropout=float(payload.get("dropout", 0.1)),
        use_ffn=bool(payload.get("use_ffn", True)),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"AttrQuery checkpoint/model mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.assert_frozen_backbone()
    return model, payload


def _checkpoint_payload(
    model: AttrQueryFrozenCLIPLocalFCE,
    epoch: int,
    metrics: dict[str, Any],
    training_config: dict[str, Any],
    base_checkpoint: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    best_score: float | None = None,
    best_epoch: int = 0,
    stale_evaluations: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "attrquery_frozenclip_localfce",
        "base_model_name": model.base_model_name,
        "base_checkpoint": str(base_checkpoint),
        "feature_mode": model.feature_mode,
        "num_heads": model.num_heads,
        "dropout": model.dropout_rate,
        "use_ffn": model.use_ffn,
        "visual_dim": model.visual_dim,
        "text_dim": model.text_dim,
        "num_categories": model.num_categories,
        "num_states": model.num_states,
        "model_state_dict": _cpu_state_dict(model),
        "text_prototypes": model.text_prototypes.detach().cpu(),
        "attribute_names": list(model.attribute_names),
        "category_groups": [list(group) for group in model.category_groups],
        "category_names": category_names(model.mapper),
        "prompt_mode": "category_complete",
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "training_config": dict(training_config),
    }
    if optimizer is not None:
        if scheduler is None or scaler is None or best_score is None:
            raise ValueError("A resumable checkpoint needs optimizer, scheduler, scaler, and best score")
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale_evaluations": int(stale_evaluations),
            "rng": _rng_payload(),
        }
    return payload


def save_experiment_checkpoint(
    path: Path, model: AttrQueryFrozenCLIPLocalFCE, epoch: int,
    metrics: dict[str, Any], training_config: dict[str, Any], base_checkpoint: str,
    **training_state: Any,
) -> None:
    _atomic_torch_save(
        path,
        _checkpoint_payload(
            model, epoch, metrics, training_config, base_checkpoint, **training_state,
        ),
    )


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


def _train_one_epoch(
    model: AttrQueryFrozenCLIPLocalFCE,
    loader: DataLoader,
    mapper: CategoryPromptMapper,
    criterion: CategoryLocalFCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    model.assert_frozen_backbone()
    category_labels = category_names(mapper)
    totals: dict[str, float] = {
        "total_loss": 0.0, "valid_pairs": 0.0, "samples": 0.0, "batches": 0.0,
    }
    for name in category_labels:
        totals[f"category_loss_{name}"] = 0.0
        totals[f"category_accuracy_{name}"] = 0.0
        totals[f"category_valid_{name}"] = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic_targets = mapper.encode(labels)
        valid_categories = _valid_category_mask(semantic_targets, mapper.category_indices())
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            category_features = model.category_features(images)
            output = criterion(
                category_features, model.text_prototypes,
                semantic_targets, valid_categories,
            )
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)

        batch_size = len(images)
        batch_valid_pairs = float(output.valid_counts.sum().detach())
        totals["total_loss"] += float(output.loss.detach()) * batch_valid_pairs
        totals["valid_pairs"] += float(output.valid_counts.sum().detach())
        totals["samples"] += batch_size
        totals["batches"] += 1.0
        for index, name in enumerate(category_labels):
            valid = float(output.valid_counts[index].detach())
            totals[f"category_loss_{name}"] += float(output.category_loss[index].detach()) * valid
            totals[f"category_accuracy_{name}"] += float(output.category_accuracy[index].detach()) * valid
            totals[f"category_valid_{name}"] += valid

    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    valid_pairs = max(totals["valid_pairs"], 1.0)
    result = {
        "total_loss": totals.pop("total_loss") / valid_pairs,
        "category_local_loss": sum(
            totals[f"category_loss_{name}"] for name in category_labels
        ) / valid_pairs,
        "valid_pairs": totals.pop("valid_pairs"),
        "train_batches": batches,
        "average_batch_size": samples / batches,
    }
    for name in category_labels:
        count = max(totals.pop(f"category_valid_{name}"), 1.0)
        result[f"category_loss_{name}"] = totals.pop(f"category_loss_{name}") / count
        result[f"category_accuracy_{name}"] = totals.pop(f"category_accuracy_{name}") / count
    return result


def _retrieval_views(
    data_root: Path, split: str, max_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, split)
    table = read_gt_csv(gt_path)
    if max_samples is not None:
        count = min(max_samples, len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    return gt_path, table, queries, ids, query_names


@torch.inference_mode()
def _encode_gallery_scores(
    model: AttrQueryFrozenCLIPLocalFCE,
    paths: Sequence[str],
    roots: Sequence[Path],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    progress_every: int,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(paths, roots, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            scores = model.category_state_scores(images.to(device, non_blocking=True))
        chunks.append(scores.float().cpu().numpy())
        if progress_every and (batch_index % progress_every == 0 or batch_index == len(loader)):
            print(
                f"[AttrQuery gallery] batch {batch_index}/{len(loader)} "
                f"({sum(len(item) for item in chunks)}/{len(paths)} images)", flush=True,
            )
    result = (
        np.concatenate(chunks, axis=0)
        if chunks else np.empty((0, model.num_categories, model.num_states), dtype=np.float32)
    )
    expected = (len(paths), model.num_categories, model.num_states)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid category score cache shape={result.shape}, expected={expected}")
    return result.astype(np.float32, copy=False)


def _rank_at_k(distances: np.ndarray, ids: np.ndarray, k: int) -> float:
    hits = 0
    for query_id, row in enumerate(distances):
        order = np.argsort(row, kind="stable")[:min(k, len(row))]
        hits += int(np.any(ids[order] == query_id))
    return hits / max(len(distances), 1)


def _category_predictions(
    scores: np.ndarray, groups: Sequence[np.ndarray], temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    probabilities = np.zeros((len(scores), 52), dtype=np.float32)
    # Each category slot supplies the logits for its own states.  Reuse the
    # established A7 category-softmax implementation, with a separate visual
    # feature per category instead of one shared image feature.
    for category_index, indices in enumerate(groups):
        category_logits = np.zeros((len(scores), 52), dtype=np.float32)
        category_logits[:, indices] = scores[:, category_index, indices] / temperature
        probabilities[:, indices] = category_softmax(
            category_logits, [indices],
        )[:, indices]
    return probabilities


def evaluate_model(
    model: AttrQueryFrozenCLIPLocalFCE,
    model_attribute_names: Sequence[str],
    data_root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    category_temperature: float,
    max_val_samples: int | None,
    gallery_scores_path: Path | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    gt_path, table, queries_raw, ids, query_names = _retrieval_views(
        data_root, "val", max_val_samples,
    )
    queries = reorder_columns(queries_raw, query_names, model_attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    mapper = CategoryPromptMapper(model_attribute_names)
    validate_mapper(mapper)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    if gallery_scores_path is not None and gallery_scores_path.is_file():
        scores = np.load(gallery_scores_path)
        expected = (len(table.image_paths), 12, 52)
        if scores.shape != expected or not np.isfinite(scores).all():
            raise ValueError(f"Invalid cached gallery category scores: {scores.shape}; expected {expected}")
    else:
        scores = _encode_gallery_scores(
            model, table.image_paths,
            [data_root, gt_path.parent, REPOSITORY_ROOT], transform, device,
            batch_size, num_workers, amp, progress_every,
        )
        if gallery_scores_path is not None:
            gallery_scores_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(gallery_scores_path, scores)

    probabilities = _category_predictions(scores, groups, category_temperature)
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    distances = category_nll(queries52, probabilities, groups)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    target52 = mapper.encode(torch.from_numpy(labels).float()).numpy()
    category_accuracy: dict[str, float] = {}
    for index, (name, group) in enumerate(zip(category_names(mapper), groups)):
        prediction = scores[:, index, group].argmax(axis=1)
        target = target52[:, group].argmax(axis=1)
        category_accuracy[f"category_accuracy_{name}"] = float(np.mean(prediction == target))
    result: dict[str, Any] = {
        "images": float(len(table.image_paths)),
        "queries": float(len(queries)),
        "rank1": official["Rank-1"],
        "rank5": _rank_at_k(distances, ids, 5),
        "rank10": _rank_at_k(distances, ids, 10),
        "map": official["mAP"],
        "mADM": official["mADM"],
        "mINP": official["mINP"],
        "semantic_top1": official["semantic_top1"],
        "retrieval_scoring": "native52_category_nll",
        "category_temperature": float(category_temperature),
        **category_accuracy,
    }
    return result


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    config["a7_checkpoint"] = str(Path(config["a7_checkpoint"]).resolve())
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    config["feature_mode"] = str(config.get("feature_mode", "local"))
    config["amp"] = bool(config.get("amp", True))
    if config["feature_mode"] not in {"local", "global"}:
        raise ValueError("feature_mode must be local or global")
    if config["category_temperature"] <= 0 or config["focal_gamma"] < 0:
        raise ValueError("category temperature must be positive and focal gamma non-negative")
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["eval_batch_size"] <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if config["num_workers"] < 0 or config["image_size"] != 224:
        raise ValueError("num_workers must be non-negative and ViT-B/32 image-size must be 224")
    if (
        config["learning_rate"] <= 0 or config["weight_decay"] < 0
        or config["warmup_epochs"] < 0 or config["min_learning_rate"] < 0
        or config["min_learning_rate"] > config["learning_rate"]
        or config["grad_clip"] < 0 or config["num_heads"] <= 0
        or not 0.0 <= config["dropout"] < 1.0
        or config["retrieval_interval"] <= 0
        or config["early_stopping_patience"] < 0
    ):
        raise ValueError("invalid optimizer, attention, dropout, or validation setting")
    if config["warmup_epochs"] >= config["epochs"]:
        raise ValueError("warmup_epochs must be smaller than epochs")
    return config


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AttrQuery-FrozenCLIP-LocalFCE from an A7 checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--a7-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--feature-mode", choices=("local", "global"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--rotation", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--category-temperature", type=float)
    parser.add_argument("--focal-gamma", type=float)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--use-ffn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--retrieval-interval", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AttrQuery-FrozenCLIP-LocalFCE with the A7 Native52 Category-NLL protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="outputs/AttrQuery-FrozenCLIP-LocalFCE/checkpoint_best.pth")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir")
    parser.add_argument("--gallery-scores", help="reuse/save [gallery,12,52] score cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def train_main(argv: Sequence[str] | None = None) -> None:
    args = build_train_parser().parse_args(argv)
    config = _resolve_config(args)
    set_seed(int(config["seed"]), bool(config.get("deterministic", False)))
    device = choose_device(config.get("device", "auto"))
    data_root = Path(config["data_root"])
    train_gt = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_gt)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    validate_mapper(mapper)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config)

    resume_payload: dict[str, Any] | None = None
    if config.get("resume"):
        resume_path = Path(config["resume"])
        if str(resume_path) == "__LAST__":
            resume_path = output_dir / "checkpoint_last.pth"
        model, resume_payload = load_experiment_model(resume_path, device)
        if resume_payload["attribute_names"] != train_table.attribute_names:
            raise ValueError("Resume checkpoint attribute order differs from training annotations")
        if resume_payload["feature_mode"] != config["feature_mode"]:
            raise ValueError("Resume checkpoint feature mode differs from the current config")
        base_checkpoint = str(resume_payload.get("base_checkpoint", config["a7_checkpoint"]))
    else:
        base_checkpoint = config["a7_checkpoint"]
        if not Path(base_checkpoint).is_file():
            raise FileNotFoundError(
                f"A7 checkpoint not found: {base_checkpoint}. Pass --a7-checkpoint explicitly."
            )
        model, _ = _build_model_from_a7(
            Path(base_checkpoint), train_table.attribute_names, device,
            bool(config["amp"]), config["feature_mode"], int(config["num_heads"]),
            float(config["dropout"]), bool(config["use_ffn"]),
        )
    model.assert_frozen_backbone()
    counts = model.parameter_counts()
    print(
        f"AttrQuery-FrozenCLIP-LocalFCE mode={model.feature_mode}; device={device}; "
        f"parameters total={counts['total']:,}, frozen={counts['frozen']:,}, "
        f"trainable={counts['trainable']:,}", flush=True,
    )

    dataset = AttriVisionDataset(
        train_table, [data_root, train_gt.parent, REPOSITORY_ROOT],
        build_train_transform(config["image_size"], config["rotation"], "resize_pad_crop"),
        config.get("max_train_samples"),
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        drop_last=len(dataset) >= config["batch_size"], num_workers=config["num_workers"],
        pin_memory=device.type == "cuda", persistent_workers=config["num_workers"] > 0,
    )
    criterion = CategoryLocalFCELoss(
        mapper.category_indices(), config["category_temperature"], config["focal_gamma"],
    )
    trainable = model.trainable_parameters()
    if not trainable:
        raise RuntimeError("No trainable AttrQuery parameters were created")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for parameter in trainable if parameter.ndim >= 2],
                "weight_decay": config["weight_decay"],
                "lr": config["learning_rate"],
            },
            {
                "params": [parameter for parameter in trainable if parameter.ndim < 2],
                "weight_decay": 0.0,
                "lr": config["learning_rate"],
            },
        ],
    )
    scheduler = build_scheduler(
        optimizer, max(len(loader), 1), config["epochs"], config["warmup_epochs"],
        config["learning_rate"], config["min_learning_rate"],
    )
    scaler = _grad_scaler(config["amp"] and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    training_config = dict(config)
    training_config.update({
        "architecture": "AttrQuery-FrozenCLIP-LocalFCE",
        "prompt_mode": "category_complete",
        "vocabulary_states": 52,
        "category_count": 12,
        "mixed_state": False,
        "multi_label": False,
        "text_sampling": "single",
        "contrastive_target": "multi-positive",
        "baseline_loss": "paper_fce",
        "category_ce": False,
        "retrieval_protocol": "native52_category_nll",
    })
    start_epoch = 1
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    if resume_payload is not None:
        state = resume_payload.get("training_state")
        if not isinstance(state, dict):
            raise ValueError("The resume checkpoint has no training state")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if state.get("rng"):
            _restore_rng(state["rng"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_score = float(state.get("best_score", -math.inf))
        best_epoch = int(state.get("best_epoch", 0))
        stale = int(state.get("stale_evaluations", 0))

    history: list[dict[str, Any]] = []
    eval_transform = build_eval_transform(config["image_size"], "resize_pad_crop")
    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        losses = _train_one_epoch(
            model, loader, mapper, criterion, optimizer, scheduler, scaler,
            device, config["amp"], config["grad_clip"],
        )
        metrics: dict[str, Any] = {}
        if epoch % config["retrieval_interval"] == 0 or epoch == config["epochs"]:
            metrics = evaluate_model(
                model, train_table.attribute_names, data_root, eval_transform, device,
                config["eval_batch_size"], config["num_workers"], config["amp"],
                config["category_temperature"], config.get("max_val_samples"),
            )
        row: dict[str, Any] = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
            **losses, **metrics,
        }
        history.append(row)
        _write_metrics_csv(output_dir / "metrics.csv", history)
        _atomic_json(output_dir / "metrics_latest.json", row)

        score = float(metrics["mADM"]) if metrics else -math.inf
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_experiment_checkpoint(
                best_path, model, epoch, row, training_config, base_checkpoint,
            )
        else:
            stale += 1
        save_experiment_checkpoint(
            last_path, model, epoch, row, training_config, base_checkpoint,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_score=best_score, best_epoch=best_epoch, stale_evaluations=stale,
        )
        print(
            f"epoch={epoch:03d}/{config['epochs']}: "
            f"local={losses['category_local_loss']:.6f}, "
            f"R1={100 * metrics.get('rank1', float('nan')):.2f}%, "
            f"mAP={100 * metrics.get('map', float('nan')):.2f}%, "
            f"mADM={100 * metrics.get('mADM', float('nan')):.2f}%",
            flush=True,
        )
        if (
            epoch >= config.get("minimum_training_epochs", 1)
            and stale >= config["early_stopping_patience"]
        ):
            break
    if not best_path.is_file():
        raise RuntimeError("Training finished without a validation-selected checkpoint")
    print(f"Best checkpoint: {best_path} (epoch={best_epoch}, mADM={best_score:.6f})")


def eval_main(argv: Sequence[str] | None = None) -> None:
    args = build_eval_parser().parse_args(argv)
    if args.image_size != 224 or args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("ViT-B/32 evaluation requires image-size=224; sizes must be valid")
    if args.category_temperature <= 0:
        raise ValueError("category-temperature must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    device = choose_device(args.device)
    model, payload = load_experiment_model(checkpoint, device)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.gallery_scores).resolve() if args.gallery_scores else output_dir / "gallery_category_scores.npy"
    metrics = evaluate_model(
        model, payload["attribute_names"], Path(args.data_root).resolve(),
        build_eval_transform(args.image_size, "resize_pad_crop"), device,
        args.eval_batch_size, args.num_workers, args.amp, args.category_temperature,
        args.max_val_samples, cache_path, args.progress_every,
    )
    result = {"checkpoint": str(checkpoint), "feature_mode": payload["feature_mode"], **metrics}
    _atomic_json(output_dir / "validation_metrics.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
