"""Training, validation, smoke-check and comparison CLIs for E1/E2/E3."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from attrivision.backbone_upgrade.model import (
    CLIP_MEAN, CLIP_STD, BackboneUpgradeModel,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper
from attrivision.datasets.upar_abpr import AttriVisionDataset
from attrivision.engine.trainer_attrivision import (
    _grad_scaler, _optimizer_step, build_scheduler,
)
from attrivision.losses.category_local_fce import CategoryLocalFCELoss
from attrivision.native52_retrieval_ablation import category_nll
from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import (
    AnnotationTable, ImagePathDataset, find_annotation_file, read_gt_csv,
)
from upar.retrieval import (
    autocast, load_retrieval_annotations, official_retrieval_metrics, reorder_columns,
)


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"
DEFAULT_CONFIGS = {
    "E1_CLIP_B16": CONFIG_DIR / "e1_clip_b16.json",
    "E2_DINOv3_B16": CONFIG_DIR / "e2_dinov3_b16.json",
    "E3A_SigLIP2_Frozen": CONFIG_DIR / "e3a_siglip2_frozen.json",
    "E3B_SigLIP2_Finetune": CONFIG_DIR / "e3b_siglip2_finetune.json",
}


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
    if not isinstance(payload, dict) or payload.get("architecture") != "backbone_upgrade_v1":
        raise ValueError(f"Not a BackboneUpgrade checkpoint: {path}")
    for required in ("experiment", "epoch", "attribute_names", "model_state_dict", "config"):
        if required not in payload:
            raise ValueError(f"BackboneUpgrade checkpoint lacks required field {required!r}")
    return payload


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


def _raw_train_transform(image_size: int, rotation: float) -> transforms.Compose:
    """Match A7 resize/pad/crop geometry and leave RGB values unnormalized."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.Pad(10),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def _raw_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
    ])


def _resolve_path(value: str | None, root: Path) -> str | None:
    if value is None or value == "":
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def _load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    expected = config.get("experiment")
    if expected not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown BackboneUpgrade experiment: {expected!r}")
    config["config_file"] = str(config_path.resolve())
    config["data_root"] = _resolve_path(config.get("data_root", "data"), REPOSITORY_ROOT)
    config["output_dir"] = _resolve_path(config.get("output_dir"), REPOSITORY_ROOT)
    config["a7_checkpoint"] = _resolve_path(config.get("a7_checkpoint"), REPOSITORY_ROOT)
    config["initialization_checkpoint"] = _resolve_path(
        config.get("initialization_checkpoint"), REPOSITORY_ROOT,
    )
    if not config.get("output_dir"):
        raise ValueError("Every experiment must have an isolated output_dir")
    if int(config.get("image_size", 224)) != 224:
        raise ValueError("The shared comparison protocol requires 224x224 input")
    if config.get("backbone") not in {"clip_b16", "dinov3_b16", "siglip2_b16"}:
        raise ValueError(f"Invalid backbone name: {config.get('backbone')}")
    if int(config.get("epochs", 0)) <= 0 or int(config.get("batch_size", 0)) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if int(config.get("eval_batch_size", 0)) <= 0:
        raise ValueError("eval_batch_size must be positive")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers cannot be negative")
    if not 0.0 <= float(config.get("selection_alpha", 0.5)) <= 1.0:
        raise ValueError("selection_alpha must be within [0,1]")
    if float(config.get("category_temperature", 0.01)) <= 0:
        raise ValueError("category_temperature must be positive")
    if expected.startswith(("E1_", "E2_")) and not config.get("a7_checkpoint"):
        raise ValueError("E1/E2 must point to the A7 best checkpoint")
    return config


def _build_model(
    config: dict[str, Any],
    attribute_names: Sequence[str],
    device: torch.device,
    *,
    initialization: str = "config",
) -> BackboneUpgradeModel:
    model = BackboneUpgradeModel(
        experiment=config["experiment"],
        backbone=config["backbone"],
        model_id=config["model_id"],
        attribute_names=attribute_names,
        a7_checkpoint=config.get("a7_checkpoint"),
        trainable_vision_last_blocks=int(config.get("trainable_vision_last_blocks", 0)),
        num_heads=int(config.get("num_heads", 8)),
        dropout=float(config.get("dropout", 0.1)),
        use_ffn=bool(config.get("use_ffn", True)),
    ).to(device)
    if list(model.attribute_names) != list(attribute_names):
        raise ValueError("Dataset attribute order differs from the model ontology")

    init_path = config.get("initialization_checkpoint")
    if config["experiment"] == "E3B_SigLIP2_Finetune" and initialization == "config":
        policy = config.get("initialization_policy", "require_e3a_best")
        if policy == "require_e3a_best":
            if not init_path or not Path(init_path).is_file():
                raise FileNotFoundError(
                    "E3-B requires the E3-A best checkpoint. Run E3-A first or "
                    "select --init-source pretrained."
                )
            _initialize_from_e3a(model, init_path, attribute_names)
        elif policy != "pretrained":
            raise ValueError(f"Unsupported E3-B initialization policy: {policy}")
    return model


def _initialize_from_e3a(
    model: BackboneUpgradeModel,
    checkpoint: str | Path,
    attribute_names: Sequence[str],
) -> None:
    payload = _load_payload(checkpoint)
    if payload["experiment"] != "E3A_SigLIP2_Frozen" or int(payload["epoch"]) < 1:
        raise ValueError("E3-B initialization must be a trained E3-A checkpoint")
    if payload.get("checkpoint_kind") != "best":
        raise ValueError("E3-B initialization requires E3-A's validation-selected best checkpoint")
    if list(payload["attribute_names"]) != list(attribute_names):
        raise ValueError("E3-A checkpoint attribute order differs from current training data")
    if payload["config"].get("model_id") != model.model_id:
        raise ValueError("E3-A and E3-B must use the same SigLIP2 pretrained model")
    target = model.state_dict()
    copied = []
    for name, value in payload["model_state_dict"].items():
        if name not in target:
            raise ValueError(f"E3-A state key is absent from E3-B model: {name}")
        if target[name].shape != value.shape:
            raise ValueError(f"E3-A/E3-B state shape mismatch for {name}")
        target[name] = value.to(dtype=target[name].dtype)
        copied.append(name)
    model.load_state_dict(target, strict=True)
    expected = {"local_projector.weight", "local_projector.bias"}
    if not expected.issubset(copied):
        raise ValueError("E3-A checkpoint is missing the trained local projector")


def _save_checkpoint(
    path: Path,
    model: BackboneUpgradeModel,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    *,
    checkpoint_kind: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    best_score: float | None = None,
    best_epoch: int = 0,
    stale_evaluations: int = 0,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "backbone_upgrade_v1",
        "experiment": config["experiment"],
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_id": model.model_id,
        "backbone": model.backbone_name,
        "global_mode": model.global_mode,
        "a7_checkpoint": model.a7_checkpoint,
        "attribute_names": list(model.attribute_names),
        "category_groups": [list(group) for group in model.category_groups],
        "visual_dim": model.visual_dim,
        "text_dim": model.text_dim,
        "patch_tokens_at_224": (224 // int(getattr(model.local_backbone.config, "patch_size", 16))) ** 2,
        "parameter_counts": model.parameter_counts(),
        "config": dict(config),
        "model_state_dict": model.saveable_state_dict(),
    }
    if optimizer is not None:
        if scheduler is None or scaler is None or best_score is None:
            raise ValueError("Resumable checkpoints need optimizer, scheduler, scaler and best score")
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale_evaluations": int(stale_evaluations),
            "rng": _rng_payload(),
        }
    _atomic_torch_save(path, payload)


def _category_groups(mapper: CategoryPromptMapper) -> list[np.ndarray]:
    return [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]


def _category_log_probs(
    scores: np.ndarray,
    groups: Sequence[np.ndarray],
    temperature: float,
    *,
    sigmoid_alignment: bool = False,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Scoring temperature must be positive")
    output = np.full(scores.shape, -np.inf, dtype=np.float32)
    for indices in groups:
        values = scores[:, indices].astype(np.float64) / temperature
        if sigmoid_alignment:
            # SigLIP2 is trained with independent sigmoid alignment. Normalize
            # those calibrated confidences only inside each semantic category
            # so they can be consumed by the established Native52 Category-NLL.
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
            values = np.log(np.clip(values, 1e-12, 1.0))
        else:
            values -= values.max(axis=1, keepdims=True)
        normalizer = np.log(np.exp(values).sum(axis=1, keepdims=True))
        output[:, indices] = (values - normalizer).astype(np.float32)
    return output


def _local_category_log_probs(
    local_scores: np.ndarray,
    groups: Sequence[np.ndarray],
    temperature: float,
) -> np.ndarray:
    if temperature <= 0 or local_scores.ndim != 3:
        raise ValueError("Expected local scores [N,12,52] and positive temperature")
    output = np.full(local_scores.shape, -np.inf, dtype=np.float32)
    for category_index, indices in enumerate(groups):
        values = local_scores[:, category_index, indices].astype(np.float64) / temperature
        values -= values.max(axis=1, keepdims=True)
        normalizer = np.log(np.exp(values).sum(axis=1, keepdims=True))
        output[:, category_index, indices] = (values - normalizer).astype(np.float32)
    return output


def _rank_at_k(distances: np.ndarray, ids: np.ndarray, k: int) -> float:
    hits = 0
    for query_id, row in enumerate(distances):
        order = np.argsort(row, kind="stable")[:min(k, len(row))]
        hits += int(np.any(ids[order] == query_id))
    return hits / max(len(distances), 1)


def _metric_row(
    distances: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
) -> dict[str, float]:
    official = official_retrieval_metrics(distances, queries, labels, ids)
    return {
        "rank1": float(official["Rank-1"]),
        "rank5": float(official["Rank-5"]),
        "rank10": float(official["Rank-10"]),
        "mAP": float(official["mAP"]),
        "mADM": float(official["mADM"]),
        "mINP": float(official["mINP"]),
        "semantic_top1": float(official["semantic_top1"]),
    }


def _score_views(
    global_scores: np.ndarray,
    local_scores: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    mapper: CategoryPromptMapper,
    config: dict[str, Any],
) -> dict[str, Any]:
    groups = _category_groups(mapper)
    target_queries = mapper.encode(torch.from_numpy(queries).float()).numpy()
    target_labels = mapper.encode(torch.from_numpy(labels).float()).numpy()
    global_log = _category_log_probs(
        global_scores, groups,
        float(config.get("siglip2_global_temperature", config["category_temperature"])),
        sigmoid_alignment=config.get("backbone") == "siglip2_b16",
    )
    local_by_category = _local_category_log_probs(
        local_scores, groups, float(config["category_temperature"]),
    )
    local_log = np.full(global_log.shape, -np.inf, dtype=np.float32)
    for category_index, indices in enumerate(groups):
        local_log[:, indices] = local_by_category[:, category_index, indices]
    alpha = float(config["selection_alpha"])
    fused_log = (1.0 - alpha) * global_log + alpha * local_log
    probabilities = {
        "global": np.exp(global_log).astype(np.float32),
        "local": np.exp(local_log).astype(np.float32),
        "fused": np.exp(fused_log).astype(np.float32),
    }
    category_labels = [spec[0] for spec in mapper._MULTI_GROUPS]
    category_labels.extend(spec[0] for spec in mapper._BINARY_GROUPS)
    results: dict[str, Any] = {"selection_alpha": alpha, "views": {}, "category_retrieval": {}}
    for view, probability in probabilities.items():
        distances = category_nll(target_queries, probability, groups)
        results["views"][view] = _metric_row(distances, queries, labels, ids)
        if view == "global":
            predicted_scores = global_scores
        elif view == "local":
            predicted_scores = local_scores
        else:
            predicted_scores = fused_log
        accuracy = {}
        for index, (name, group) in enumerate(zip(category_labels, groups)):
            values = (
                predicted_scores[:, index, group]
                if view == "local" else predicted_scores[:, group]
            )
            accuracy[name] = float(
                np.mean(values.argmax(axis=1) == target_labels[:, group].argmax(axis=1))
            )
        results["views"][view]["category_accuracy"] = accuracy

    # Each category's retrieval view ranks using only that category's NLL.
    fused_prob = probabilities["fused"]
    for index, (name, group) in enumerate(zip(category_labels, groups)):
        query_targets = target_queries[:, group]
        counts = query_targets.sum(axis=1, keepdims=True).clip(min=1.0)
        gallery_log = np.log(np.clip(fused_prob[:, group], 1e-12, 1.0))
        distances = -(query_targets / counts) @ gallery_log.T
        results["category_retrieval"][name] = _metric_row(distances, queries, labels, ids)
    results["images"] = int(len(labels))
    results["queries"] = int(len(queries))
    results["fusion_space"] = "category log-probabilities"
    results["siglip2_global_calibration"] = (
        "pretrained sigmoid alignment, then within-category normalization"
        if config.get("backbone") == "siglip2_b16" else None
    )
    return results


def _retrieval_data(
    data_root: Path,
    split: str,
    max_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, split)
    table = read_gt_csv(gt_path)
    if max_samples is not None:
        count = min(int(max_samples), len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    return gt_path, table, queries, ids, query_names


@torch.inference_mode()
def _encode_gallery(
    model: BackboneUpgradeModel,
    image_paths: Sequence[str],
    roots: Sequence[Path],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ImagePathDataset(image_paths, roots, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    global_chunks: list[np.ndarray] = []
    local_chunks: list[np.ndarray] = []
    for batch_index, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with autocast(device, amp):
            output = model(images)
        global_chunks.append(output["global_scores"].float().cpu().numpy())
        local_chunks.append(output["local_scores"].float().cpu().numpy())
        if progress_every and (batch_index % progress_every == 0 or batch_index == len(loader)):
            print(f"[BackboneUpgrade] gallery batch {batch_index}/{len(loader)}", flush=True)
    global_scores = (
        np.concatenate(global_chunks, axis=0)
        if global_chunks else np.empty((0, 52), dtype=np.float32)
    )
    local_scores = (
        np.concatenate(local_chunks, axis=0)
        if local_chunks else np.empty((0, 12, 52), dtype=np.float32)
    )
    if global_scores.shape != (len(image_paths), 52):
        raise RuntimeError(f"Unexpected global score shape {global_scores.shape}")
    if local_scores.shape != (len(image_paths), 12, 52):
        raise RuntimeError(f"Unexpected local score shape {local_scores.shape}")
    if not np.isfinite(global_scores).all() or not np.isfinite(local_scores).all():
        raise RuntimeError("Non-finite gallery scores")
    return global_scores.astype(np.float32), local_scores.astype(np.float32)


def evaluate_model(
    model: BackboneUpgradeModel,
    config: dict[str, Any],
    data_root: Path,
    *,
    split: str = "val",
    max_samples: int | None = None,
    save_gallery: bool = False,
    progress_every: int = 25,
) -> dict[str, Any]:
    gt_path, table, queries_raw, ids, query_names = _retrieval_data(data_root, split, max_samples)
    queries = reorder_columns(queries_raw, query_names, model.attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model.attribute_names)
    transform = _raw_eval_transform(int(config["image_size"]))
    global_scores, local_scores = _encode_gallery(
        model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
        transform, next(model.parameters()).device,
        int(config["eval_batch_size"]), int(config["num_workers"]),
        bool(config.get("amp", True)), progress_every,
    )
    mapper = CategoryPromptMapper(model.attribute_names)
    results = _score_views(global_scores, local_scores, queries, labels, ids, mapper, config)
    results.update({
        "experiment": config["experiment"],
        "split": split,
        "epoch": int(config.get("evaluated_epoch", 0)),
        "selection_metric": "fused.mADM",
        "best_checkpoint": config.get("evaluated_checkpoint"),
        "pretrained_model": model.model_id,
        "global_backbone": "A7" if model.global_mode == "a7" else "SigLIP2 ViT-B/16",
        "local_backbone": config["backbone"],
        "parameter_counts": model.parameter_counts(),
        "patch_token_count": int((int(config["image_size"]) // 16) ** 2),
        "visual_feature_dim": model.visual_dim,
        "text_embedding_dim": model.text_dim,
    })
    if save_gallery:
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / f"{split}_gallery_scores.npz",
            global_scores=global_scores, local_scores=local_scores,
        )
    return results


def _sigmoid_focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def _train_one_epoch(
    model: BackboneUpgradeModel,
    loader: DataLoader,
    mapper: CategoryPromptMapper,
    local_criterion: CategoryLocalFCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "local_fce_loss": 0.0, "global_sigmoid_focal_loss": 0.0, "batches": 0.0}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic_targets = mapper.encode(labels)
        valid_categories = torch.stack([
            semantic_targets.index_select(
                1, torch.as_tensor(group, device=semantic_targets.device, dtype=torch.long),
            ).sum(dim=1) > 0
            for group in model.category_groups
        ], dim=1)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, bool(config.get("amp", True))):
            output = model(images)
            local = local_criterion(
                output["local_features"], model.state_text_prototypes,
                semantic_targets, valid_categories,
            )
            loss = local.loss
            global_loss = output["global_scores"].new_zeros(())
            if config["experiment"] == "E3B_SigLIP2_Finetune":
                global_loss = _sigmoid_focal_binary_loss(
                    output["global_scores"], semantic_targets,
                    float(config.get("sigmoid_focal_alpha", 0.25)),
                    float(config.get("sigmoid_focal_gamma", 2.0)),
                )
                loss = (
                    float(config.get("global_loss_weight", 1.0)) * global_loss
                    + float(config.get("local_loss_weight", 0.25)) * local.loss
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip = float(config.get("grad_clip", 1.0))
        if clip > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), clip)
        _optimizer_step(scaler, optimizer, scheduler)
        totals["loss"] += float(loss.detach())
        totals["local_fce_loss"] += float(local.loss.detach())
        totals["global_sigmoid_focal_loss"] += float(global_loss.detach())
        totals["batches"] += 1
    batches = max(totals.pop("batches"), 1.0)
    return {key: value / batches for key, value in totals.items()}


def _build_optimizer(
    model: BackboneUpgradeModel,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    backbone_lr = float(config.get("backbone_learning_rate", config["learning_rate"]))
    head_lr = float(config["learning_rate"])
    decay = float(config["weight_decay"])
    grouped: dict[tuple[str, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("local_backbone.vision.vision_model.encoder.layers.")
        lr = backbone_lr if is_backbone else head_lr
        wd = decay if parameter.ndim >= 2 else 0.0
        grouped.setdefault(("backbone" if is_backbone else "head", wd), []).append(parameter)
    if not grouped:
        raise RuntimeError("BackboneUpgrade created no trainable parameters")
    param_groups = [
        {"params": params, "lr": backbone_lr if name == "backbone" else head_lr, "weight_decay": wd}
        for (name, wd), params in grouped.items()
    ]
    return torch.optim.AdamW(param_groups)


def _write_metrics_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_train_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "output_dir": args.output_dir,
        "data_root": args.data_root,
        "a7_checkpoint": args.a7_checkpoint,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "initialization_policy": (
            "pretrained" if args.init_source == "pretrained" else
            "require_e3a_best" if args.init_source == "e3a_best" else None
        ),
    }
    config = _load_config(args.config, overrides)
    if args.init_source == "pretrained":
        config["initialization_checkpoint"] = None
    return config


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one BackboneUpgrade experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--a7-checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--init-source", choices=("e3a_best", "pretrained"))
    return parser


def train_main(argv: Sequence[str] | None = None) -> None:
    args = build_train_parser().parse_args(argv)
    config = _resolve_train_args(args)
    set_seed(int(config["seed"]), bool(config.get("deterministic", False)))
    device = choose_device(str(config.get("device", "auto")))
    data_root = Path(config["data_root"])
    train_gt = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_gt)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    if len(mapper.keys) != 52:
        raise RuntimeError("Training ontology must contain 52 states from the 40 UPAR bits")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = _build_model(config, train_table.attribute_names, device)
    resume_payload: dict[str, Any] | None = None
    if config.get("resume"):
        resume_path = Path(config["resume"])
        if str(config["resume"]) == "__LAST__":
            resume_path = output_dir / "checkpoint_last.pth"
        resume_payload = _load_payload(resume_path)
        if resume_payload["experiment"] != config["experiment"]:
            raise ValueError("Resume checkpoint belongs to a different experiment")
        if list(resume_payload["attribute_names"]) != train_table.attribute_names:
            raise ValueError("Resume checkpoint attribute order differs from training annotations")
        model.load_saveable_state_dict(resume_payload["model_state_dict"])

    counts = model.parameter_counts()
    _atomic_json(output_dir / "parameter_counts.json", counts)
    _atomic_json(output_dir / "model_configuration.json", {
        "experiment": config["experiment"],
        "global_backbone": "A7" if model.global_mode == "a7" else config["model_id"],
        "local_backbone": config["model_id"],
        "global_mode": model.global_mode,
        "visual_hidden_dim": model.visual_dim,
        "text_embedding_dim": model.text_dim,
        "patch_tokens_at_224": (224 // 16) ** 2,
        "num_categories": model.num_categories,
        "num_states": model.num_states,
        "trainable_vision_last_blocks": model.trainable_vision_last_blocks,
        "parameter_counts": counts,
        "a7_checkpoint": config.get("a7_checkpoint"),
        "pretrained_model": config["model_id"],
    })
    print(
        f"{config['experiment']} backbone={model.model_id}; device={device}; "
        f"parameters total={counts['total']:,}, frozen={counts['frozen']:,}, "
        f"trainable={counts['trainable']:,}", flush=True,
    )

    dataset = AttriVisionDataset(
        train_table, [data_root, train_gt.parent, REPOSITORY_ROOT],
        _raw_train_transform(int(config["image_size"]), float(config["rotation"])),
        config.get("max_train_samples"),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=len(dataset) >= int(config["batch_size"]),
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
    )
    criterion = CategoryLocalFCELoss(
        mapper.category_indices(),
        float(config["category_temperature"]),
        float(config.get("focal_gamma", 2.0)),
    )
    optimizer = _build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer, max(len(loader), 1), int(config["epochs"]),
        int(config["warmup_epochs"]), float(config["learning_rate"]),
        float(config["min_learning_rate"]),
    )
    scaler = _grad_scaler(bool(config.get("amp", True)) and device.type == "cuda")
    start_epoch, best_score, best_epoch, stale = 1, -math.inf, 0, 0
    if resume_payload is not None:
        state = resume_payload.get("training_state")
        if not isinstance(state, dict):
            raise ValueError("The resume checkpoint has no optimizer/scheduler state")
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
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = _train_one_epoch(
            model, loader, mapper, criterion, optimizer, scheduler, scaler, device, config,
        )
        elapsed_seconds = time.perf_counter() - started
        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda" else 0.0
        )
        evaluation = evaluate_model(
            model, config, data_root, split="val",
            max_samples=config.get("max_val_samples"), progress_every=0,
        )
        fused_metrics = evaluation["views"]["fused"]
        score = float(fused_metrics["mADM"])
        evaluation["training_summary"] = {
            **train_metrics,
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_memory_mb,
        }
        epoch_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[-1]["lr"]),
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_memory_mb,
            **train_metrics,
            "global_R1": evaluation["views"]["global"]["rank1"],
            "global_R5": evaluation["views"]["global"]["rank5"],
            "global_R10": evaluation["views"]["global"]["rank10"],
            "global_mAP": evaluation["views"]["global"]["mAP"],
            "global_mADM": evaluation["views"]["global"]["mADM"],
            "global_mINP": evaluation["views"]["global"]["mINP"],
            "local_R1": evaluation["views"]["local"]["rank1"],
            "local_R5": evaluation["views"]["local"]["rank5"],
            "local_R10": evaluation["views"]["local"]["rank10"],
            "local_mAP": evaluation["views"]["local"]["mAP"],
            "local_mADM": evaluation["views"]["local"]["mADM"],
            "local_mINP": evaluation["views"]["local"]["mINP"],
            "fused_R1": fused_metrics["rank1"],
            "fused_R5": fused_metrics["rank5"],
            "fused_R10": fused_metrics["rank10"],
            "fused_mAP": fused_metrics["mAP"],
            "fused_mADM": score,
            "fused_mINP": fused_metrics["mINP"],
            "selection_alpha": config["selection_alpha"],
        }
        history.append(epoch_row)
        _write_metrics_csv(output_dir / "metrics.csv", history)
        _atomic_json(output_dir / "metrics_latest.json", {**epoch_row, **evaluation})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            _save_checkpoint(
                best_path, model, config, epoch, evaluation, checkpoint_kind="best",
            )
        else:
            stale += 1
        _save_checkpoint(
            last_path, model, config, epoch, evaluation, checkpoint_kind="last",
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_score=best_score, best_epoch=best_epoch, stale_evaluations=stale,
        )
        print(
            f"epoch={epoch:03d}/{config['epochs']} loss={train_metrics['loss']:.5f} "
            f"R1={100*fused_metrics['rank1']:.2f}% mAP={100*fused_metrics['mAP']:.2f}% "
            f"mADM={100*score:.2f}% (best epoch {best_epoch})", flush=True,
        )
        with (output_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")
        if stale >= int(config["early_stopping_patience"]):
            break
    if not best_path.is_file():
        raise RuntimeError("Training ended without a validation-selected best checkpoint")
    print(f"Best checkpoint: {best_path} (epoch={best_epoch}, fused mADM={best_score:.6f})")


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate/evaluate a BackboneUpgrade checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--save-gallery-scores", action="store_true")
    return parser


def eval_main(argv: Sequence[str] | None = None) -> None:
    args = build_eval_parser().parse_args(argv)
    config = _load_config(args.config, {
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "device": args.device,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "max_val_samples": args.max_val_samples,
    })
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(config["output_dir"]) / "checkpoint_best.pth"
    if not checkpoint.is_absolute():
        checkpoint = REPOSITORY_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Experiment checkpoint not found: {checkpoint}")
    payload = _load_payload(checkpoint)
    if int(payload["epoch"]) < 1 or payload.get("checkpoint_kind") not in {"best", "last"}:
        raise ValueError("Refusing to evaluate an untrained or smoke-only initialization")
    if payload["experiment"] != config["experiment"]:
        raise ValueError("Checkpoint experiment does not match the selected config")
    config.update({
        "evaluated_epoch": int(payload["epoch"]),
        "evaluated_checkpoint": str(checkpoint.resolve()),
    })
    device = choose_device(str(config.get("device", "auto")))
    model = _build_model(
        config, payload["attribute_names"], device, initialization="pretrained",
    )
    if payload.get("model_id") != model.model_id or payload.get("backbone") != model.backbone_name:
        raise ValueError("Checkpoint pretrained backbone does not match the current model config")
    if payload.get("a7_checkpoint") != model.a7_checkpoint:
        raise ValueError("Checkpoint A7 initialization does not match the current model config")
    model.load_saveable_state_dict(payload["model_state_dict"])
    model.eval()
    results = evaluate_model(
        model, config, Path(config["data_root"]), split="val",
        max_samples=config.get("max_val_samples"),
        save_gallery=bool(args.save_gallery_scores),
    )
    results["checkpoint"] = str(checkpoint.resolve())
    results["checkpoint_epoch"] = int(payload["epoch"])
    results["checkpoint_kind"] = payload["checkpoint_kind"]
    results["training_summary"] = payload.get("metrics", {}).get("training_summary")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "evaluation_val.json"
    _atomic_json(result_path, results)
    print(
        f"Evaluation saved to {result_path}; fused R1={results['views']['fused']['rank1']:.6f}, "
        f"mAP={results['views']['fused']['mAP']:.6f}, "
        f"mADM={results['views']['fused']['mADM']:.6f}", flush=True,
    )


def build_smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check BackboneUpgrade tensor shapes, trainable gradients and checkpoint IO"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--pretrained-init", action="store_true",
        help="For E3-B only, use the public SigLIP2 initialization instead of E3-A checkpoint weights.",
    )
    return parser


def smoke_main(argv: Sequence[str] | None = None) -> None:
    args = build_smoke_parser().parse_args(argv)
    config = _load_config(args.config, {"device": args.device})
    device = choose_device(str(config.get("device", "auto")))
    data_root = Path(config["data_root"])
    table = read_gt_csv(find_annotation_file(data_root, "train"))
    model = _build_model(
        config, table.attribute_names, device,
        initialization="pretrained" if args.pretrained_init else "config",
    )
    model.train()
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (batch_size, 40), dtype=torch.float32, device=device)
    mapper = CategoryPromptMapper(table.attribute_names)
    semantic = mapper.encode(labels)
    valid = torch.stack([
        semantic.index_select(
            1, torch.as_tensor(group, device=semantic.device, dtype=torch.long),
        ).sum(dim=1) > 0
        for group in model.category_groups
    ], dim=1)
    output = model(images)
    expected_patches = (224 // 16) ** 2
    expected_shapes = {
        "global_scores": (batch_size, 52),
        "local_scores": (batch_size, 12, 52),
        "local_features": (batch_size, 12, model.text_dim),
        "patch_tokens": (batch_size, expected_patches, model.visual_dim),
    }
    for name, shape in expected_shapes.items():
        if tuple(output[name].shape) != shape:
            raise RuntimeError(f"{name} shape {tuple(output[name].shape)} != {shape}")
    criterion = CategoryLocalFCELoss(
        mapper.category_indices(), float(config["category_temperature"]),
        float(config.get("focal_gamma", 2.0)),
    )
    loss = criterion(
        output["local_features"], model.state_text_prototypes, semantic, valid,
    ).loss
    if config["experiment"] == "E3B_SigLIP2_Finetune":
        loss = loss + _sigmoid_focal_binary_loss(
            output["global_scores"], semantic,
            float(config.get("sigmoid_focal_alpha", 0.25)),
            float(config.get("sigmoid_focal_gamma", 2.0)),
        )
    loss.backward()
    trainable = model.trainable_parameters()
    if not trainable or not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("Backward check did not reach any trainable parameter")
    for parameter in model.local_backbone.parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError("A frozen pretrained backbone parameter received a gradient")
    if model.global_a7 is not None:
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.global_a7.parameters()):
            raise RuntimeError("The frozen A7 global model must remain frozen")
    with tempfile.TemporaryDirectory(prefix="backbone-upgrade-smoke-") as temporary:
        smoke_path = Path(temporary) / "smoke_checkpoint.pth"
        _save_checkpoint(
            smoke_path, model, config, 0, {"smoke": True}, checkpoint_kind="smoke",
        )
        payload = _load_payload(smoke_path)
        if payload["checkpoint_kind"] != "smoke":
            raise RuntimeError("Checkpoint round-trip did not preserve the smoke marker")
        model.load_saveable_state_dict(payload["model_state_dict"])
    print(
        f"{config['experiment']} smoke checks passed: patches={expected_patches}, "
        f"local={tuple(output['local_scores'].shape)}, "
        f"trainable_parameters={sum(p.numel() for p in trainable):,}", flush=True,
    )


def _baseline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    a7_path = REPOSITORY_ROOT / "outputs/attrivision_ablation_mADM/A7/validation_metrics.json"
    if a7_path.is_file():
        with a7_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({
            "experiment": "A7 Native52",
            "global_backbone": "A7 ViT-B/32",
            "local_backbone": "A7 global descriptor",
            "rank1": metrics.get("rank1"), "mAP": metrics.get("map"),
            "mADM": metrics.get("mADM"), "rank5": metrics.get("rank5"),
            "rank10": metrics.get("rank10"), "mINP": metrics.get("mINP"),
            "best_epoch": metrics.get("checkpoint_epoch"),
            "global_only_mADM": metrics.get("mADM"), "local_only_mADM": None,
            "fused_mADM": metrics.get("mADM"), "baseline_delta_mADM": None,
            "fusion_delta_vs_global_mADM": None, "total_parameters": None,
            "trainable_parameters": None, "patch_tokens": None,
            "visual_feature_dim": 768, "text_embedding_dim": 512,
            "peak_gpu_memory_mb": None, "pretrained_checkpoint": "A7 best checkpoint",
            "category_mADM": None,
        })

    # The strongest existing AttriQuery reference is the frozen-backbone
    # joint-control run. Its verification file records the mADM-selected epoch.
    control_dir = REPOSITORY_ROOT / "outputs/AttrQuery-PartialJointFinetune/FrozenBackbone-Control"
    verification_path = control_dir / "initialization_verification.json"
    history_path = control_dir / "training_metrics.csv"
    if verification_path.is_file() and history_path.is_file():
        with verification_path.open(encoding="utf-8") as handle:
            verification = json.load(handle)
        best_epoch = int(verification["best_epoch"])
        with history_path.open(encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        best_row = next((row for row in history if int(row["epoch"]) == best_epoch), None)
        if best_row is None:
            raise ValueError("FrozenBackbone-Control best epoch is absent from its training history")
        rows.insert(0, {
            "experiment": "Baseline",
            "global_backbone": "A7 ViT-B/32",
            "local_backbone": "A7 ViT-B/32 patches",
            "rank1": float(best_row["rank1"]), "mAP": float(best_row["map"]),
            "mADM": float(best_row["mADM"]), "rank5": float(best_row["rank5"]),
            "rank10": float(best_row["rank10"]), "mINP": float(best_row["mINP"]),
            "best_epoch": best_epoch,
            "global_only_mADM": float(best_row["global_mADM"]),
            "local_only_mADM": float(best_row["local_mADM"]),
            "fused_mADM": float(best_row["mADM"]), "baseline_delta_mADM": 0.0,
            "fusion_delta_vs_global_mADM": float(best_row["mADM"]) - float(best_row["global_mADM"]),
            "total_parameters": verification.get("parameter_counts", {}).get("total"),
            "trainable_parameters": verification.get("parameter_counts", {}).get("trainable"),
            "patch_tokens": 49, "visual_feature_dim": 768,
            "text_embedding_dim": 512, "peak_gpu_memory_mb": None,
            "pretrained_checkpoint": "A7 best checkpoint; FrozenBackbone-Control",
            "category_mADM": None,
        })

    local_path = REPOSITORY_ROOT / "outputs/AttrQuery-FrozenCLIP-LocalFCE/metrics_latest.json"
    if local_path.is_file():
        with local_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({
            "experiment": "AttriQuery FrozenCLIP LocalFCE (latest saved epoch)",
            "global_backbone": "A7 ViT-B/32",
            "local_backbone": "CLIP ViT-B/32 patches",
            "rank1": metrics.get("rank1"), "mAP": metrics.get("map"),
            "mADM": metrics.get("mADM"), "rank5": metrics.get("rank5"),
            "rank10": metrics.get("rank10"), "mINP": metrics.get("mINP"),
            "best_epoch": metrics.get("epoch"), "global_only_mADM": None,
            "local_only_mADM": metrics.get("mADM"), "fused_mADM": None,
            "baseline_delta_mADM": None, "fusion_delta_vs_global_mADM": None,
            "total_parameters": None, "trainable_parameters": None,
            "patch_tokens": 49, "visual_feature_dim": 768,
            "text_embedding_dim": 512, "peak_gpu_memory_mb": None,
            "pretrained_checkpoint": "A7 best checkpoint", "category_mADM": None,
        })
    baseline = next((row for row in rows if row["experiment"] == "Baseline"), None)
    if baseline and baseline.get("mADM") is not None:
        for row in rows:
            if row["experiment"] != "Baseline" and row.get("mADM") is not None:
                row["baseline_delta_mADM"] = float(row["mADM"]) - float(baseline["mADM"])
    return rows


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the BackboneUpgrade result comparison")
    parser.add_argument(
        "--output-dir", default=str(REPOSITORY_ROOT / "outputs/BackboneUpgrade/Comparison"),
    )
    parser.add_argument("--require-all", action="store_true")
    return parser


def compare_main(argv: Sequence[str] | None = None) -> None:
    args = build_compare_parser().parse_args(argv)
    rows = _baseline_rows()
    for experiment, config_path in DEFAULT_CONFIGS.items():
        config = _load_config(config_path)
        result_path = Path(config["output_dir"]) / "evaluation_val.json"
        if not result_path.is_file():
            if args.require_all:
                raise FileNotFoundError(f"Missing completed evaluation result: {result_path}")
            rows.append({
                "experiment": experiment, "global_backbone": None, "local_backbone": None,
                "rank1": None, "mAP": None, "mADM": None, "rank5": None, "rank10": None,
                "mINP": None, "best_epoch": None, "global_only_mADM": None,
                "local_only_mADM": None, "fused_mADM": None, "baseline_delta_mADM": None,
                "fusion_delta_vs_global_mADM": None, "total_parameters": None,
                "trainable_parameters": None, "patch_tokens": None,
                "visual_feature_dim": None, "text_embedding_dim": None,
                "peak_gpu_memory_mb": None,
                "pretrained_checkpoint": config["model_id"], "category_mADM": None,
            })
            continue
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        views = result["views"]
        baseline = next((row for row in rows if row["experiment"] == "Baseline"), None)
        fused = views["fused"]
        counts = result.get("parameter_counts", {})
        category_summary = {name: metrics["mADM"] for name, metrics in result["category_retrieval"].items()}
        rows.append({
            "experiment": experiment,
            "global_backbone": result["global_backbone"],
            "local_backbone": result["local_backbone"],
            "rank1": fused["rank1"], "mAP": fused["mAP"], "mADM": fused["mADM"],
            "rank5": fused["rank5"], "rank10": fused["rank10"], "mINP": fused["mINP"],
            "best_epoch": result.get("checkpoint_epoch", result.get("epoch")),
            "global_only_mADM": views["global"]["mADM"],
            "local_only_mADM": views["local"]["mADM"],
            "fused_mADM": fused["mADM"],
            "baseline_delta_mADM": (
                float(fused["mADM"]) - float(baseline["mADM"])
                if baseline and baseline.get("mADM") is not None else None
            ),
            "fusion_delta_vs_global_mADM": float(fused["mADM"]) - float(views["global"]["mADM"]),
            "total_parameters": counts.get("total"),
            "trainable_parameters": counts.get("trainable"),
            "patch_tokens": result.get("patch_token_count"),
            "visual_feature_dim": result.get("visual_feature_dim"),
            "text_embedding_dim": result.get("text_embedding_dim"),
            "peak_gpu_memory_mb": (result.get("training_summary") or {}).get("peak_gpu_memory_mb"),
            "pretrained_checkpoint": config["model_id"],
            "category_mADM": json.dumps(category_summary, ensure_ascii=False, sort_keys=True),
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"
    _write_metrics_csv(csv_path, rows)
    table_rows = [
        "| Experiment | Global Backbone | Local Backbone | R1 | mAP | mADM |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        def percent(value: Any) -> str:
            return "" if value is None else f"{100.0 * float(value):.3f}%"
        table_rows.append(
            f"| {row['experiment']} | {row['global_backbone'] or ''} | "
            f"{row['local_backbone'] or ''} | {percent(row['rank1'])} | "
            f"{percent(row['mAP'])} | {percent(row['mADM'])} |"
        )
    table_rows.extend([
        "",
        "E3-A/E3-B replace both the global and local backbones, so these results are not a controlled local-only ablation against E1/E2.",
        "",
        "Metrics for global-only, local-only, fused, category retrieval, feature dimensions, parameter counts, and baseline deltas are in `comparison.csv`.",
    ])
    md_path.write_text("\n".join(table_rows) + "\n", encoding="utf-8")
    print(f"Comparison written to {csv_path} and {md_path}", flush=True)
