"""Training and evaluation pipeline for ``AttrQuery-AttributeAdapter``.

This is a new, standalone experiment.  Existing A7, AttrQuery-FrozenCLIP-
LocalFCE, and diagnostic files are not modified.  The pipeline deliberately
keeps the established 52-state/12-category/temperature=0.01 Category-NLL
protocol and adds a deterministic tuning/evaluation query split so that
checkpoint and fusion-alpha selection never use the held-out evaluation split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import load_model
from .datasets.attribute_prompts import CategoryPromptMapper
from .datasets.upar_abpr import AttriVisionDataset
from .engine.trainer_attrivision import _grad_scaler, _optimizer_step, build_scheduler
from .losses.category_local_fce import CategoryLocalFCELoss
from .models.attrivision import AttriVision
from .models.attrquery_attribute_adapter import (
    ADAPTER_ARCHITECTURES,
    AttrQueryAttributeAdapter,
    assert_identity_initialization,
)
from .models.attrquery_frozenclip_localfce import build_text_prototypes
from .transforms import build_eval_transform, build_train_transform
from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import AnnotationTable, ImagePathDataset, find_annotation_file, read_gt_csv
from upar.retrieval import autocast, load_retrieval_annotations, official_retrieval_metrics, reorder_columns


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "attrquery_attribute_adapter.json"
DEFAULT_ALPHAS = tuple(round(float(value), 2) for value in np.arange(0.0, 1.0001, 0.1))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _load_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
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


def _category_names(mapper: CategoryPromptMapper) -> list[str]:
    names = [spec[0] for spec in mapper._MULTI_GROUPS]
    names.extend(spec[0] for spec in mapper._BINARY_GROUPS)
    if len(names) != 12:
        raise RuntimeError(f"Expected 12 categories, got {len(names)}")
    return names


def _validate_mapper(mapper: CategoryPromptMapper) -> None:
    if len(mapper.keys) != 52 or len(mapper.prompts) != 52:
        raise RuntimeError("AttrQuery-AttributeAdapter requires the exact 52-state vocabulary")
    groups = mapper.category_indices()
    covered = [index for group in groups for index in group]
    if len(groups) != 12 or sorted(covered) != list(range(52)):
        raise RuntimeError("The category mapping must partition states 0..51 into 12 groups")


def _validate_backbone_match(
    a7_payload: dict[str, Any], local_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compare the actual frozen CLIP tensors in A7 and LocalFCE payloads."""
    if a7_payload.get("architecture") != "attrivision_openclip_vit_b_32":
        raise ValueError("Global initialization must be an original A7 AttriVision checkpoint")
    if local_payload.get("architecture") != "attrquery_frozenclip_localfce":
        raise ValueError(
            "Local initialization must be AttrQuery-FrozenCLIP-LocalFCE; "
            "FromScratchJointFCE is intentionally not accepted"
        )
    if a7_payload.get("model_name") != local_payload.get("base_model_name"):
        raise ValueError("A7 and LocalFCE checkpoints use different CLIP model names")
    a7_state = a7_payload.get("model_state_dict", {})
    local_state = local_payload.get("model_state_dict", {})
    compared: list[str] = []
    max_abs = 0.0
    for key, value in a7_state.items():
        if not key.startswith("clip."):
            continue
        local_key = f"base.{key}"
        other = local_state.get(local_key)
        if other is None:
            raise ValueError(f"LocalFCE checkpoint is missing backbone tensor {local_key}")
        if tuple(value.shape) != tuple(other.shape):
            raise ValueError(f"Backbone tensor shape mismatch for {key}: {value.shape} vs {other.shape}")
        difference = float((value.detach().float() - other.detach().float()).abs().max())
        max_abs = max(max_abs, difference)
        compared.append(key)
        if not torch.equal(value.detach().cpu(), other.detach().cpu()):
            raise ValueError(f"A7/LocalFCE backbone tensor differs: {key} (max_abs_diff={difference:.6g})")
    if not compared:
        raise ValueError("No CLIP backbone tensors were found for A7/LocalFCE verification")
    return {"matched": True, "tensor_count": len(compared), "max_abs_diff": max_abs}


def _validate_initial_payloads(
    a7_payload: dict[str, Any], local_payload: dict[str, Any],
    attributes: Sequence[str], mapper: CategoryPromptMapper,
) -> dict[str, Any]:
    if a7_payload.get("prompt_mode") != "category_complete":
        raise ValueError("A7 initialization must use category_complete")
    if local_payload.get("prompt_mode") != "category_complete":
        raise ValueError("AttrQuery LocalFCE initialization must use category_complete")
    if local_payload.get("feature_mode") != "local":
        raise ValueError("AttrQuery LocalFCE initialization must be the local head checkpoint")
    if list(a7_payload.get("attribute_names", [])) != list(attributes):
        raise ValueError("A7 attribute order differs from the training annotation order")
    if list(local_payload.get("attribute_names", [])) != list(attributes):
        raise ValueError("LocalFCE attribute order differs from the training annotation order")
    _validate_mapper(mapper)
    backbone = _validate_backbone_match(a7_payload, local_payload)
    return {
        "a7_architecture": a7_payload.get("architecture"),
        "local_architecture": local_payload.get("architecture"),
        "prompt_mode": "category_complete",
        "states": 52,
        "categories": 12,
        "backbone_verification": backbone,
    }


def _payload_text_prototypes(local_payload: dict[str, Any]) -> torch.Tensor:
    state = local_payload.get("model_state_dict", {})
    value = state.get("text_prototypes", local_payload.get("text_prototypes"))
    if value is None:
        raise ValueError("AttrQuery LocalFCE checkpoint does not contain text prototypes")
    return value.detach().float().cpu()


def build_initial_model(
    a7_checkpoint: Path,
    attrquery_checkpoint: Path,
    attributes: Sequence[str],
    architecture: str,
    device: torch.device,
    amp: bool,
    bottleneck_ratio: int,
    patchfilm_delta: float,
    patchfilm_category_chunk_size: int,
) -> tuple[AttrQueryAttributeAdapter, dict[str, Any]]:
    if architecture not in ADAPTER_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    base, a7_payload = load_model(a7_checkpoint, device)
    local_payload = _load_payload(attrquery_checkpoint)
    mapper = CategoryPromptMapper(attributes)
    protocol = _validate_initial_payloads(a7_payload, local_payload, attributes, mapper)
    local_text = _payload_text_prototypes(local_payload)
    a7_text = build_text_prototypes(base, mapper, device, amp).cpu()
    text_difference = (
        float((a7_text - local_text).abs().max())
        if a7_text.shape == local_text.shape else float("inf")
    )
    if a7_text.shape != local_text.shape or text_difference >= 1e-4:
        raise ValueError(
            "A7 and LocalFCE text prototypes differ; refusing to mix incompatible protocols "
            f"(shape={a7_text.shape}/{local_text.shape}, max_abs_diff={text_difference:.6g})"
        )
    model = AttrQueryAttributeAdapter(
        base, attributes, local_text, architecture=architecture,
        num_heads=int(local_payload.get("num_heads", 8)),
        dropout=float(local_payload.get("dropout", 0.1)),
        use_ffn=bool(local_payload.get("use_ffn", True)),
        bottleneck_ratio=bottleneck_ratio,
        patchfilm_delta=patchfilm_delta,
        patchfilm_category_chunk_size=patchfilm_category_chunk_size,
    ).to(device)
    local_state = local_payload["model_state_dict"]
    local_head_state = {
        key.removeprefix("local_head."): value
        for key, value in local_state.items()
        if key.startswith("local_head.")
    }
    if not local_head_state:
        raise ValueError("AttrQuery LocalFCE checkpoint has no local_head state")
    missing, unexpected = model.local_head.load_state_dict(local_head_state, strict=False)
    allowed_missing = {
        key for key in model.local_head.state_dict()
        if key.startswith(("feature_adapter.", "category_film.", "patch_film."))
    }
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"Local head checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    model.assert_frozen_backbone()
    identity_difference = assert_identity_initialization(model, device)
    protocol.update({
        "a7_checkpoint": str(a7_checkpoint.resolve()),
        "attrquery_checkpoint": str(attrquery_checkpoint.resolve()),
        "text_prototype_max_abs_diff": text_difference,
        "identity_forward_max_abs_diff": identity_difference,
        "architecture": architecture,
    })
    return model, protocol


def load_adapter_checkpoint(path: Path, device: torch.device) -> tuple[AttrQueryAttributeAdapter, dict[str, Any]]:
    payload = _load_payload(path)
    if payload.get("architecture") != "attrquery_attribute_adapter":
        raise ValueError(f"Not an AttrQuery-AttributeAdapter checkpoint: {path}")
    attributes = list(payload["attribute_names"])
    text = payload.get("text_prototypes")
    if text is None:
        text = payload["model_state_dict"].get("text_prototypes")
    if text is None:
        raise ValueError("AttributeAdapter checkpoint has no cached text prototypes")
    base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
    model = AttrQueryAttributeAdapter(
        base, attributes, text,
        architecture=payload["adapter_architecture"],
        num_heads=int(payload["num_heads"]),
        dropout=float(payload["dropout"]),
        use_ffn=bool(payload["use_ffn"]),
        bottleneck_ratio=int(payload.get("bottleneck_ratio", 4)),
        patchfilm_delta=float(payload.get("patchfilm_delta", 0.1)),
        patchfilm_category_chunk_size=int(payload.get("patchfilm_category_chunk_size", 4)),
    )
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(f"AttributeAdapter checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    model.assert_frozen_backbone()
    return model, payload


def _valid_category_mask(targets: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.stack([
        targets[:, torch.as_tensor(group, device=targets.device)].sum(dim=1) > 0
        for group in groups
    ], dim=1)


def _train_one_epoch(
    model: AttrQueryAttributeAdapter,
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
    model.assert_trainability(int(model._training_stage))
    totals = {"loss": 0.0, "valid_pairs": 0.0, "samples": 0.0, "batches": 0.0}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic_targets = mapper.encode(labels)
        valid_categories = _valid_category_mask(semantic_targets, mapper.category_indices())
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            features = model.category_features(images)
            output = criterion(features, model.text_prototypes, semantic_targets, valid_categories)
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)
        valid = float(output.valid_counts.sum().detach())
        totals["loss"] += float(output.loss.detach()) * valid
        totals["valid_pairs"] += valid
        totals["samples"] += len(images)
        totals["batches"] += 1
    denominator = max(totals["valid_pairs"], 1.0)
    return {
        "category_local_loss": totals["loss"] / denominator,
        "valid_pairs": totals["valid_pairs"],
        "train_batches": totals["batches"],
        "average_batch_size": totals["samples"] / max(totals["batches"], 1.0),
    }


def _retrieval_views(
    data_root: Path, split: str, max_samples: int | None,
) -> tuple[Path, AnnotationTable, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, split)
    table = read_gt_csv(gt_path)
    if max_samples is not None:
        count = min(max_samples, len(table.image_paths))
        table = AnnotationTable(table.image_paths[:count], table.labels[:count], table.attribute_names)
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    return gt_path, table, queries, ids, query_names


def _query_split(query_count: int, seed: int, tuning_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if query_count < 2:
        raise ValueError("At least two queries are required for a tuning/evaluation split")
    if not 0.0 < tuning_fraction < 1.0:
        raise ValueError("tuning_fraction must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(query_count)
    count = min(max(int(round(query_count * tuning_fraction)), 1), query_count - 1)
    return np.sort(order[:count]).astype(np.int64), np.sort(order[count:]).astype(np.int64)


def _category_logs(scores: np.ndarray, groups: Sequence[np.ndarray], temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("category temperature must be positive")
    if scores.ndim == 2:
        scores = np.broadcast_to(scores[:, None, :], (len(scores), len(groups), scores.shape[1]))
    if scores.ndim != 3 or scores.shape[1:] != (len(groups), 52):
        raise ValueError(f"Expected score shape [G,{len(groups)},52], got {scores.shape}")
    result = np.full(scores.shape, -np.inf, dtype=np.float64)
    for category_index, indices in enumerate(groups):
        values = scores[:, category_index, indices].astype(np.float64) / temperature
        maximum = values.max(axis=1, keepdims=True)
        result[:, category_index, indices] = values - maximum - np.log(
            np.exp(values - maximum).sum(axis=1, keepdims=True),
        )
    finite_values = np.concatenate([
        result[:, category_index, indices].reshape(-1)
        for category_index, indices in enumerate(groups)
    ])
    if not np.isfinite(finite_values).all():
        raise RuntimeError("Category log probabilities contain NaN or Inf")
    return result.astype(np.float32)


def _category_nll_distances(
    query_targets: np.ndarray, category_logs: np.ndarray, groups: Sequence[np.ndarray],
) -> np.ndarray:
    distances = np.zeros((len(query_targets), len(category_logs)), dtype=np.float64)
    for category_index, indices in enumerate(groups):
        target = query_targets[:, indices].astype(np.float64)
        counts = target.sum(axis=1, keepdims=True)
        if np.any(counts <= 0):
            raise ValueError("Every query must have an active state in every category")
        distances -= (target / counts) @ np.maximum(
            category_logs[:, category_index, indices].astype(np.float64), math.log(1e-12),
        ).T
    distances /= len(groups)
    return distances.astype(np.float32)


def _retrieval_metrics(
    distances: np.ndarray, queries: np.ndarray, labels: np.ndarray, ids: np.ndarray,
) -> dict[str, float]:
    official = official_retrieval_metrics(distances, queries, labels, ids)
    return {
        "Rank-1": float(official["Rank-1"]),
        "mAP": float(official["mAP"]),
        "mADM": float(official["mADM"]),
        "mINP": float(official["mINP"]),
        "Rank-5": float(official["Rank-5"]),
        "Rank-10": float(official["Rank-10"]),
        "semantic_top1": float(official["semantic_top1"]),
    }


def _subset_metrics(
    distances: np.ndarray, query_indices: np.ndarray, queries: np.ndarray,
    labels: np.ndarray, ids: np.ndarray,
) -> dict[str, float]:
    return _retrieval_metrics(distances[query_indices], queries[query_indices], labels, ids)


def _encode_local_scores(
    model: AttrQueryAttributeAdapter, table: AnnotationTable, gt_path: Path,
    data_root: Path, device: torch.device, batch_size: int, num_workers: int,
    amp: bool, progress_prefix: str,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
                          build_eval_transform(224, "resize_pad_crop")),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for index, images in enumerate(loader, start=1):
            with autocast(device, amp):
                scores = model.category_state_scores(images.to(device, non_blocking=True))
            chunks.append(scores.float().cpu().numpy())
            if index == 1 or index == len(loader) or index % 25 == 0:
                print(f"[{progress_prefix}] batch {index}/{len(loader)}", flush=True)
    result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    expected = (len(table.image_paths), 12, 52)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid local score shape {result.shape}; expected {expected}")
    return result


def _encode_global_scores(
    model: AttriVision, text_prototypes: torch.Tensor, table: AnnotationTable,
    gt_path: Path, data_root: Path, device: torch.device, batch_size: int,
    num_workers: int, amp: bool,
) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
                          build_eval_transform(224, "resize_pad_crop")),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for index, images in enumerate(loader, start=1):
            with autocast(device, amp):
                features = model.encode_image(images.to(device, non_blocking=True))
            chunks.append(features.float().cpu().numpy())
            if index == 1 or index == len(loader) or index % 25 == 0:
                print(f"[A7 global] batch {index}/{len(loader)}", flush=True)
    features = np.concatenate(chunks, axis=0)
    scores = features @ text_prototypes.float().cpu().numpy().T
    if scores.shape != (len(table.image_paths), 52) or not np.isfinite(scores).all():
        raise RuntimeError(f"Invalid global score shape {scores.shape}")
    return scores.astype(np.float32, copy=False)


def _save_score_cache(path: Path, values: np.ndarray, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float32, copy=False), allow_pickle=False)
    _atomic_json(path.with_suffix(path.suffix + ".json"), manifest)


def _load_score_cache(path: Path, expected: tuple[int, ...], manifest: dict[str, Any]) -> np.ndarray | None:
    manifest_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not manifest_path.is_file():
        return None
    try:
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = np.load(path, allow_pickle=False)
        if actual_manifest != manifest or tuple(values.shape) != expected or not np.isfinite(values).all():
            return None
        return values.astype(np.float32, copy=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _fuse_logs(global_logs: np.ndarray, local_logs: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("fusion alpha must be in [0,1]")
    return ((1.0 - alpha) * global_logs + alpha * local_logs).astype(np.float32)


def _ece(confidence: np.ndarray, correctness: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = ((confidence >= edges[index]) & (confidence <= edges[index + 1])) if index == bins - 1 else ((confidence >= edges[index]) & (confidence < edges[index + 1]))
        if mask.any():
            value += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correctness[mask].mean()))
    return value


def _category_rows(
    global_logs: np.ndarray, local_logs: np.ndarray, target52: np.ndarray,
    groups: Sequence[np.ndarray], names: Sequence[str], ece_bins: int,
    architecture: str, alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = {
        "Global": global_logs,
        "Local": local_logs,
        f"Global+Local(alpha={alpha:g})": _fuse_logs(global_logs, local_logs, alpha),
    }
    for method, logs in methods.items():
        for category_index, (name, indices) in enumerate(zip(names, groups)):
            probabilities = np.exp(logs[:, category_index, indices].astype(np.float64))
            target = target52[:, indices].argmax(axis=1)
            prediction = probabilities.argmax(axis=1)
            correct = prediction == target
            truth_probability = probabilities[np.arange(len(target)), target]
            confidence = probabilities.max(axis=1)
            rows.append({
                "architecture": architecture,
                "method": method,
                "category_index": category_index,
                "category": name,
                "state_count": len(indices),
                "accuracy": float(correct.mean()),
                "nll": float(-np.log(np.maximum(truth_probability, 1e-12)).mean()),
                "ece": _ece(confidence, correct.astype(np.float32), ece_bins),
                "mean_confidence": float(confidence.mean()),
                "gallery_images": len(target52),
            })
    return rows


def _transition_rows(
    global_distances: np.ndarray, candidate_distances: np.ndarray,
    query_indices: np.ndarray, queries: np.ndarray, labels: np.ndarray,
    architecture: str, method: str,
) -> dict[str, Any]:
    global_top = np.argmin(global_distances[query_indices], axis=1)
    candidate_top = np.argmin(candidate_distances[query_indices], axis=1)
    query_values = queries[query_indices]
    global_correct = np.asarray([np.array_equal(labels[index], query_values[row]) for row, index in enumerate(global_top)])
    candidate_correct = np.asarray([np.array_equal(labels[index], query_values[row]) for row, index in enumerate(candidate_top)])
    return {
        "architecture": architecture,
        "method": method,
        "query_count": int(len(query_indices)),
        "wrong_to_correct": int(np.sum(~global_correct & candidate_correct)),
        "correct_to_wrong": int(np.sum(global_correct & ~candidate_correct)),
        "wrong_to_wrong": int(np.sum(~global_correct & ~candidate_correct)),
        "correct_to_correct": int(np.sum(global_correct & candidate_correct)),
        "net_correction": int(np.sum(~global_correct & candidate_correct) - np.sum(global_correct & ~candidate_correct)),
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


@torch.inference_mode()
def _attention_diagnostics(
    output_dir: Path, architecture: str, model: AttrQueryAttributeAdapter,
    table: AnnotationTable, gt_path: Path, data_root: Path, device: torch.device,
    batch_size: int, num_workers: int, amp: bool, sample_count: int,
) -> dict[str, Any]:
    count = min(max(sample_count, 0), len(table.image_paths))
    if count == 0:
        return {"architecture": architecture, "images": 0, "status": "disabled"}
    indices = np.arange(count, dtype=np.int64)
    loader = DataLoader(
        ImagePathDataset(
            [table.image_paths[index] for index in indices],
            [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(224, "resize_pad_crop"),
        ),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    attention_chunks: list[np.ndarray] = []
    gain_chunks: list[np.ndarray] = []
    grid: tuple[int, int] | None = None
    model.eval()
    for images in loader:
        with autocast(device, amp):
            _, attention, current_grid, gain = model.category_features_with_attention(
                images.to(device, non_blocking=True), return_patch_gain=architecture == "PatchFiLM",
            )
        attention_chunks.append(attention.float().cpu().numpy())
        if gain is not None:
            gain_chunks.append(gain.float().cpu().numpy())
        grid = current_grid if grid is None else grid
        if grid != current_grid:
            raise RuntimeError("Attention patch grid changed within diagnostics")
    attention = np.concatenate(attention_chunks, axis=0)
    target_dir = output_dir / "attention_heatmaps" / architecture
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_dir / "attention.npz", image_indices=indices,
        attention=attention, grid=np.asarray(grid, dtype=np.int64),
    )
    summary: dict[str, Any] = {
        "architecture": architecture, "images": int(count),
        "attention_shape": list(attention.shape), "grid": list(grid or (0, 0)),
        "path": str((target_dir / "attention.npz").resolve()),
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mean_maps = attention.mean(axis=1).reshape(count, 12, *(grid or (1, 1)))
        names = _category_names(CategoryPromptMapper(table.attribute_names))
        for image_index, maps in zip(indices, mean_maps):
            fig, axes = plt.subplots(3, 4, figsize=(10, 7), squeeze=False)
            for axis, heatmap, name in zip(axes.flat, maps, names):
                axis.imshow(heatmap, cmap="magma")
                axis.set_title(name, fontsize=8)
                axis.axis("off")
            fig.suptitle(f"{architecture} attention | image {int(image_index)}")
            fig.tight_layout()
            fig.savefig(target_dir / f"image_{int(image_index):06d}_all_categories.png", dpi=150)
            plt.close(fig)
        summary["heatmap_figures"] = int(count)
    except ImportError:
        summary["heatmap_figures"] = 0
        summary["matplotlib"] = "unavailable; attention.npz was still written"

    if architecture == "PatchFiLM" and gain_chunks:
        gain = np.concatenate(gain_chunks, axis=0)
        effective = float(model.patchfilm_delta) * np.tanh(gain)
        np.savez_compressed(target_dir / "channel_modulation.npz", gain=gain, effective_modulation=effective)
        rows: list[dict[str, Any]] = []
        for category_index, name in enumerate(_category_names(CategoryPromptMapper(table.attribute_names))):
            for channel in range(effective.shape[-1]):
                channel_values = effective[:, :, category_index, :, channel]
                rows.append({
                    "architecture": architecture,
                    "category_index": category_index,
                    "category": name,
                    "head_channel": channel,
                    "mean_signed_modulation": float(channel_values.mean()),
                    "mean_abs_modulation": float(np.abs(channel_values).mean()),
                    "max_abs_modulation": float(np.abs(channel_values).max()),
                    "fraction_nonzero_modulation": float((np.abs(channel_values) > 1e-8).mean()),
                })
        _write_csv(target_dir / "patchfilm_channel_modulation.csv", rows)
        summary["channel_modulation_shape"] = list(effective.shape)
        summary["channel_modulation_csv"] = str((target_dir / "patchfilm_channel_modulation.csv").resolve())
    return summary


def _checkpoint_payload(
    model: AttrQueryAttributeAdapter, epoch: int, metrics: dict[str, Any],
    config: dict[str, Any], base_checkpoint: str, attrquery_checkpoint: str,
    optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None,
    scaler: Any = None, best_score: float | None = None, best_epoch: int = 0,
    stale_evaluations: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "attrquery_attribute_adapter",
        "experiment": "AttrQuery-AttributeAdapter",
        "adapter_architecture": model.architecture,
        "base_model_name": model.base_model_name,
        "base_checkpoint": str(base_checkpoint),
        "attrquery_checkpoint": str(attrquery_checkpoint),
        "feature_mode": "local",
        "num_heads": model.num_heads,
        "dropout": model.dropout_rate,
        "use_ffn": model.use_ffn,
        "visual_dim": model.visual_dim,
        "text_dim": model.text_dim,
        "num_categories": model.num_categories,
        "num_states": model.num_states,
        "bottleneck_ratio": model.bottleneck_ratio,
        "patchfilm_delta": model.patchfilm_delta,
        "patchfilm_category_chunk_size": model.patchfilm_category_chunk_size,
        "adapter_parameter_count": model.local_head.adapter_parameter_count(),
        "model_state_dict": _cpu_state_dict(model),
        "text_prototypes": model.text_prototypes.detach().cpu(),
        "attribute_names": list(model.attribute_names),
        "category_groups": [list(group) for group in model.category_groups],
        "category_names": _category_names(model.mapper) if hasattr(model, "mapper") else _category_names(CategoryPromptMapper(model.attribute_names)),
        "prompt_mode": "category_complete",
        "stage": int(config["stage"]),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "training_config": dict(config),
    }
    if optimizer is not None:
        if scheduler is None or scaler is None or best_score is None:
            raise ValueError("Resumable checkpoint requires optimizer, scheduler, scaler, and best score")
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_score": float(best_score),
            "best_epoch": int(best_epoch), "stale_evaluations": int(stale_evaluations),
            "rng": _rng_payload(),
        }
    return payload


def save_checkpoint(path: Path, model: AttrQueryAttributeAdapter, epoch: int,
                    metrics: dict[str, Any], config: dict[str, Any],
                    base_checkpoint: str, attrquery_checkpoint: str, **state: Any) -> None:
    _atomic_torch_save(path, _checkpoint_payload(
        model, epoch, metrics, config, base_checkpoint, attrquery_checkpoint, **state,
    ))


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    config["a7_checkpoint"] = str(Path(config["a7_checkpoint"]).resolve())
    config["attrquery_checkpoint"] = str(Path(config["attrquery_checkpoint"]).resolve())
    config["data_root"] = str(Path(config["data_root"]).resolve())
    if config.get("output_dir") is None:
        config["output_dir"] = str(Path(config["output_root"]) / f"Stage{config['stage']}" / config["architecture"])
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    if config["architecture"] not in ADAPTER_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ADAPTER_ARCHITECTURES}")
    if int(config["stage"]) not in {1, 2}:
        raise ValueError("stage must be 1 or 2")
    if config["stage"] == 1 and config["architecture"] == "NoAdapter":
        raise ValueError("NoAdapter Stage 1 is evaluation-only; use eval_attrquery_attribute_adapter.py")
    if int(config["epochs"]) <= 0 or int(config["batch_size"]) <= 0 or int(config["eval_batch_size"]) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if int(config["num_workers"]) < 0 or int(config["image_size"]) != 224:
        raise ValueError("num_workers must be non-negative and image_size must be 224")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if int(config["warmup_epochs"]) < 0 or int(config["warmup_epochs"]) >= int(config["epochs"]):
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if float(config["category_temperature"]) <= 0 or float(config["focal_gamma"]) < 0:
        raise ValueError("category_temperature must be positive and focal_gamma non-negative")
    if int(config["bottleneck_ratio"]) <= 0 or float(config["patchfilm_delta"]) < 0:
        raise ValueError("bottleneck_ratio must be positive and patchfilm_delta non-negative")
    return config


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AttrQuery-AttributeAdapter Stage 1 or Stage 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--a7-checkpoint")
    parser.add_argument("--attrquery-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--architecture", choices=ADAPTER_ARCHITECTURES)
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--device")
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
    parser.add_argument("--bottleneck-ratio", type=int)
    parser.add_argument("--patchfilm-delta", type=float)
    parser.add_argument("--patchfilm-category-chunk-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--tuning-fraction", type=float)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--retrieval-interval", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
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
    _validate_mapper(mapper)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config)

    resume_payload: dict[str, Any] | None = None
    if config.get("resume"):
        resume_path = output_dir / "checkpoint_last.pth" if config["resume"] == "__LAST__" else Path(config["resume"])
        resume_payload = _load_payload(resume_path)
        if resume_payload.get("architecture") != "attrquery_attribute_adapter":
            raise ValueError("Resume checkpoint is not an AttributeAdapter checkpoint")
        if resume_payload.get("adapter_architecture") != config["architecture"] or int(resume_payload.get("stage", -1)) != int(config["stage"]):
            raise ValueError("Resume checkpoint architecture/stage differs from the current run")
        model, protocol = load_adapter_checkpoint(resume_path, device)
        if list(resume_payload["attribute_names"]) != list(train_table.attribute_names):
            raise ValueError("Resume checkpoint attribute order differs from train annotations")
        base_checkpoint = str(resume_payload.get("base_checkpoint", config["a7_checkpoint"]))
        attrquery_checkpoint = str(resume_payload.get("attrquery_checkpoint", config["attrquery_checkpoint"]))
    else:
        base_checkpoint = config["a7_checkpoint"]
        attrquery_checkpoint = config["attrquery_checkpoint"]
        if not Path(base_checkpoint).is_file() or not Path(attrquery_checkpoint).is_file():
            raise FileNotFoundError("Both --a7-checkpoint and --attrquery-checkpoint must exist")
        model, protocol = build_initial_model(
            Path(base_checkpoint), Path(attrquery_checkpoint), train_table.attribute_names,
            config["architecture"], device, bool(config["amp"]),
            int(config["bottleneck_ratio"]), float(config["patchfilm_delta"]),
            int(config["patchfilm_category_chunk_size"]),
        )
    model._training_stage = int(config["stage"])
    model.set_trainability(int(config["stage"]))
    model.assert_trainability(int(config["stage"]))
    counts = model.parameter_counts()
    print(
        f"AttrQuery-AttributeAdapter architecture={config['architecture']} stage={config['stage']} "
        f"device={device} total={counts['total']:,} frozen={counts['frozen']:,} "
        f"trainable={counts['trainable']:,} adapter={counts['adapter']:,}", flush=True,
    )

    dataset = AttriVisionDataset(
        train_table, [data_root, train_gt.parent, REPOSITORY_ROOT],
        build_train_transform(int(config["image_size"]), float(config["rotation"]), "resize_pad_crop"),
        config.get("max_train_samples"),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=len(dataset) >= int(config["batch_size"]), num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda", persistent_workers=int(config["num_workers"]) > 0,
    )
    criterion = CategoryLocalFCELoss(mapper.category_indices(), float(config["category_temperature"]), float(config["focal_gamma"]))
    trainable = model.trainable_parameters()
    if not trainable:
        raise RuntimeError("No trainable parameters were created")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": float(config["weight_decay"]), "lr": float(config["learning_rate"])},
        {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0, "lr": float(config["learning_rate"])},
    ])
    scheduler = build_scheduler(
        optimizer, max(len(loader), 1), int(config["epochs"]), int(config["warmup_epochs"]),
        float(config["learning_rate"]), float(config["min_learning_rate"]),
    )
    scaler = _grad_scaler(bool(config["amp"]) and device.type == "cuda")

    _, val_table, val_queries_raw, val_ids, val_query_names = _retrieval_views(
        data_root, "val", config.get("max_val_samples"),
    )
    val_queries = reorder_columns(val_queries_raw, val_query_names, train_table.attribute_names)
    val_labels = reorder_columns(val_table.labels, val_table.attribute_names, train_table.attribute_names)
    tuning_indices, evaluation_indices = _query_split(len(val_queries), int(config["selection_seed"]), float(config["tuning_fraction"]))
    tuning_set = set(tuning_indices.tolist())
    _write_csv(output_dir / "query_split.csv", [
        {"query_index": int(index), "split": "tuning" if index in tuning_set else "evaluation"}
        for index in range(len(val_queries))
    ])
    del evaluation_indices  # checkpoint selection is intentionally tuning-only.
    query_targets = mapper.encode(torch.from_numpy(val_queries).float()).numpy().astype(np.float32)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    start_epoch, best_score, best_epoch, stale = 1, -math.inf, 0, 0
    if resume_payload is not None:
        state = resume_payload.get("training_state")
        if not isinstance(state, dict):
            raise ValueError("Resume checkpoint has no training state")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if state.get("rng"):
            _restore_rng(state["rng"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_score = float(state.get("best_score", -math.inf))
        best_epoch = int(state.get("best_epoch", 0))
        stale = int(state.get("stale_evaluations", 0))

    config.update({
        "experiment": "AttrQuery-AttributeAdapter",
        "architecture": config["architecture"], "stage": int(config["stage"]),
        "selection_metric": "mADM_on_tuning_query_split",
        "validation_protocol": "native52_category_nll",
        "vocabulary_states": 52, "category_count": 12,
        "evaluation_split_excluded_from_checkpoint_selection": True,
        "initialization": protocol,
        "parameter_counts": counts,
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
    })
    _atomic_json(output_dir / "config.json", config)
    _atomic_json(output_dir / "parameter_counts.json", {
        "experiment": "AttrQuery-AttributeAdapter",
        "architecture": config["architecture"],
        "stage": int(config["stage"]),
        **counts,
        "trainable_parameter_names": config["trainable_parameter_names"],
    })
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        started = time.perf_counter()
        losses = _train_one_epoch(model, loader, mapper, criterion, optimizer, scheduler, scaler, device, bool(config["amp"]), float(config["grad_clip"]))
        metrics: dict[str, Any] = {}
        if epoch % int(config["retrieval_interval"]) == 0 or epoch == int(config["epochs"]):
            local_scores = _encode_local_scores(model, val_table, find_annotation_file(data_root, "val"), data_root, device, int(config["eval_batch_size"]), int(config["num_workers"]), bool(config["amp"]), "selection")
            local_logs = _category_logs(local_scores, groups, float(config["category_temperature"]))
            distances = _category_nll_distances(query_targets, local_logs, groups)
            metrics = _subset_metrics(distances, tuning_indices, val_queries, val_labels, val_ids)
        row = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started, **losses, **metrics,
        }
        history.append(row)
        _write_csv(output_dir / "metrics.csv", history)
        _atomic_json(output_dir / "metrics_latest.json", row)
        score = float(metrics.get("mADM", -math.inf))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            save_checkpoint(best_path, model, epoch, row, config, base_checkpoint, attrquery_checkpoint)
        else:
            stale += 1
        save_checkpoint(
            last_path, model, epoch, row, config, base_checkpoint, attrquery_checkpoint,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_score=best_score, best_epoch=best_epoch, stale_evaluations=stale,
        )
        print(
            f"{config['architecture']} Stage{config['stage']} epoch={epoch:03d}/{config['epochs']} "
            f"loss={losses['category_local_loss']:.6f} tuning_mADM={metrics.get('mADM', float('nan')):.6f}", flush=True,
        )
        if epoch >= int(config.get("minimum_training_epochs", 1)) and stale >= int(config["early_stopping_patience"]):
            break
    if not best_path.is_file():
        raise RuntimeError("Training finished without a tuning-selected checkpoint")
    print(f"Best checkpoint: {best_path} (epoch={best_epoch}, tuning_mADM={best_score:.6f})", flush=True)


def _parse_architecture_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be ARCHITECTURE=PATH")
    architecture, path = value.split("=", 1)
    if architecture not in ADAPTER_ARCHITECTURES or not path:
        raise argparse.ArgumentTypeError("checkpoint must be one of NoAdapter/FeatureAdapter/CategoryFiLM/PatchFiLM=PATH")
    return architecture, Path(path)


def _parse_float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("alpha values must be in [0,1]")
    return values


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all AttrQuery-AttributeAdapter conditions with Category-NLL and fusion diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a7-checkpoint", default="outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth")
    parser.add_argument("--attrquery-checkpoint", default="outputs/AttrQuery-FrozenCLIP-LocalFCE/checkpoint_best.pth")
    parser.add_argument("--checkpoint", action="append", type=_parse_architecture_checkpoint, default=[], metavar="ARCH=PATH")
    parser.add_argument(
        "--stage2-noadapter-checkpoint", type=Path,
        help="Optional Stage 2 NoAdapter checkpoint; reported separately from the original NoAdapter baseline",
    )
    parser.add_argument(
        "--stage2-patchfilm-checkpoint", type=Path,
        help="Optional Stage 2 PatchFiLM checkpoint; reported separately from the other PatchFiLM condition",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs/AttrQuery-AttributeAdapter/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--fixed-alpha", type=float, default=0.5)
    parser.add_argument("--alpha-grid", type=_parse_float_list, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--tuning-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--bottleneck-ratio", type=int, default=4)
    parser.add_argument("--patchfilm-delta", type=float, default=0.1)
    parser.add_argument("--patchfilm-category-chunk-size", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--attention-samples", type=int, default=16)
    return parser


def eval_main(argv: Sequence[str] | None = None) -> None:
    args = build_eval_parser().parse_args(argv)
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.ece_bins <= 0:
        raise ValueError("eval-batch-size/ece-bins must be positive and num-workers non-negative")
    if args.category_temperature <= 0 or not 0.0 <= args.fixed_alpha <= 1.0:
        raise ValueError("category temperature must be positive and fixed-alpha must be in [0,1]")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    a7_checkpoint = Path(args.a7_checkpoint).resolve()
    attrquery_checkpoint = Path(args.attrquery_checkpoint).resolve()
    gt_path, table, queries_raw, ids, query_names = _retrieval_views(data_root, "val", args.max_val_samples)
    attributes = list(table.attribute_names)
    mapper = CategoryPromptMapper(attributes)
    _validate_mapper(mapper)
    queries = reorder_columns(queries_raw, query_names, attributes)
    labels = table.labels.astype(np.float32, copy=False)
    query_targets = mapper.encode(torch.from_numpy(queries).float()).numpy().astype(np.float32)
    target52 = mapper.encode(torch.from_numpy(labels).float()).numpy().astype(np.float32)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    names = _category_names(mapper)
    tuning_indices, evaluation_indices = _query_split(len(queries), args.split_seed, args.tuning_fraction)
    tuning_set = set(tuning_indices.tolist())
    _write_csv(output_dir / "query_split.csv", [
        {"query_index": int(index), "split": "tuning" if index in tuning_set else "evaluation"}
        for index in range(len(queries))
    ])

    protocol: dict[str, Any] = {
        "experiment": "AttrQuery-AttributeAdapter",
        "validation_protocol": "native52_category_nll",
        "states": 52, "categories": 12, "category_temperature": args.category_temperature,
        "fixed_fusion_alpha": args.fixed_alpha, "alpha_selection": "tuning_query_split",
        "checkpoint_selection": "training_tuning_query_split_only",
        "split_seed": args.split_seed, "tuning_fraction": args.tuning_fraction,
        "tuning_queries": int(len(tuning_indices)), "evaluation_queries": int(len(evaluation_indices)),
        "gallery_images": int(len(table.image_paths)), "a7_checkpoint": str(a7_checkpoint),
        "attrquery_checkpoint": str(attrquery_checkpoint),
        "stage2_noadapter_checkpoint": (
            str(args.stage2_noadapter_checkpoint.resolve())
            if args.stage2_noadapter_checkpoint is not None else None
        ),
        "stage2_patchfilm_checkpoint": (
            str(args.stage2_patchfilm_checkpoint.resolve())
            if args.stage2_patchfilm_checkpoint is not None else None
        ),
    }
    _atomic_json(output_dir / "protocol.json", protocol)

    global_model, a7_payload = load_model(a7_checkpoint, device)
    if a7_payload.get("prompt_mode") != "category_complete":
        raise ValueError("Global checkpoint must be A7 category_complete")
    global_text = build_text_prototypes(global_model, mapper, device, bool(args.amp)).cpu()
    global_cache = output_dir / "gallery_scores_global.npy"
    global_manifest = {"checkpoint": str(a7_checkpoint), "images": list(table.image_paths), "kind": "global"}
    global_scores = _load_score_cache(global_cache, (len(table.image_paths), 52), global_manifest)
    if global_scores is None:
        global_scores = _encode_global_scores(global_model, global_text, table, gt_path, data_root, device, args.eval_batch_size, args.num_workers, args.amp)
        _save_score_cache(global_cache, global_scores, global_manifest)
    del global_model
    global_logs = _category_logs(global_scores, groups, args.category_temperature)

    checkpoint_map: dict[str, Path] = {"NoAdapter": attrquery_checkpoint}
    for architecture, path in args.checkpoint:
        if architecture == "NoAdapter":
            raise ValueError("NoAdapter uses --attrquery-checkpoint and cannot be replaced")
        checkpoint_map[architecture] = path.resolve()
    missing = [architecture for architecture in ADAPTER_ARCHITECTURES if architecture not in checkpoint_map]
    if missing:
        raise ValueError(f"Missing checkpoints for architectures: {missing}; pass --checkpoint ARCH=PATH")
    conditions = [
        {
            "name": architecture,
            "model_architecture": architecture,
            "checkpoint": checkpoint_map[architecture],
            "kind": "initial" if architecture == "NoAdapter" else "adapter_checkpoint",
        }
        for architecture in ADAPTER_ARCHITECTURES
    ]
    for condition_name, model_architecture, checkpoint_path in (
        ("NoAdapter-Stage2", "NoAdapter", args.stage2_noadapter_checkpoint),
        ("PatchFiLM-Stage2", "PatchFiLM", args.stage2_patchfilm_checkpoint),
    ):
        if checkpoint_path is not None:
            conditions.append({
                "name": condition_name,
                "model_architecture": model_architecture,
                "checkpoint": checkpoint_path.resolve(),
                "kind": "stage2_checkpoint",
            })

    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    local_logs_by_arch: dict[str, np.ndarray] = {}
    for condition in conditions:
        condition_name = str(condition["name"])
        model_architecture = str(condition["model_architecture"])
        checkpoint = Path(condition["checkpoint"])
        if condition["kind"] == "initial":
            model, init_protocol = build_initial_model(
                a7_checkpoint, checkpoint, attributes, model_architecture, device, args.amp,
                args.bottleneck_ratio, args.patchfilm_delta, args.patchfilm_category_chunk_size,
            )
        else:
            model, checkpoint_payload = load_adapter_checkpoint(checkpoint, device)
            if checkpoint_payload.get("adapter_architecture") != model_architecture:
                raise ValueError(
                    f"{checkpoint} contains {checkpoint_payload.get('adapter_architecture')}, "
                    f"expected {model_architecture} for {condition_name}"
                )
            if condition["kind"] == "stage2_checkpoint" and int(checkpoint_payload.get("stage", -1)) != 2:
                raise ValueError(f"{checkpoint} is not a Stage 2 checkpoint")
            init_protocol = checkpoint_payload.get("training_config", {}).get("initialization", {})
            if list(model.attribute_names) != attributes:
                raise ValueError(f"{condition_name} checkpoint attribute order differs from validation annotations")
        cache = output_dir / f"gallery_scores_{_safe_name(condition_name)}.npy"
        manifest = {"checkpoint": str(checkpoint.resolve()), "images": list(table.image_paths), "kind": condition_name}
        local_scores = _load_score_cache(cache, (len(table.image_paths), 12, 52), manifest)
        if local_scores is None:
            local_scores = _encode_local_scores(model, table, gt_path, data_root, device, args.eval_batch_size, args.num_workers, args.amp, condition_name)
            _save_score_cache(cache, local_scores, manifest)
        local_logs = _category_logs(local_scores, groups, args.category_temperature)
        local_logs_by_arch[condition_name] = local_logs
        local_distances = _category_nll_distances(query_targets, local_logs, groups)
        fixed_logs = _fuse_logs(global_logs, local_logs, args.fixed_alpha)
        fixed_distances = _category_nll_distances(query_targets, fixed_logs, groups)
        for split_name, indices in (("tuning", tuning_indices), ("evaluation", evaluation_indices)):
            summary_rows.append({"architecture": condition_name, "method": "Local-only", "split": split_name, **_subset_metrics(local_distances, indices, queries, labels, ids)})
            summary_rows.append({"architecture": condition_name, "method": f"Global+Local(alpha={args.fixed_alpha:g})", "split": split_name, **_subset_metrics(fixed_distances, indices, queries, labels, ids)})
        category_rows.extend(_category_rows(global_logs, local_logs, target52, groups, names, args.ece_bins, condition_name, args.fixed_alpha))
        transition_rows.append(_transition_rows(
            _category_nll_distances(query_targets, global_logs, groups), local_distances,
            evaluation_indices, queries, labels, condition_name, "Local-only",
        ))
        transition_rows.append(_transition_rows(
            _category_nll_distances(query_targets, global_logs, groups), fixed_distances,
            evaluation_indices, queries, labels, condition_name, f"Global+Local(alpha={args.fixed_alpha:g})",
        ))
        alpha_tuning: list[dict[str, Any]] = []
        for alpha in args.alpha_grid:
            fused = _fuse_logs(global_logs, local_logs, float(alpha))
            distances = _category_nll_distances(query_targets, fused, groups)
            tuning_metrics = _subset_metrics(distances, tuning_indices, queries, labels, ids)
            evaluation_metrics = _subset_metrics(distances, evaluation_indices, queries, labels, ids)
            alpha_tuning.append({"architecture": condition_name, "alpha": float(alpha), "scope": "tuning", **tuning_metrics})
            sweep_rows.append({"architecture": condition_name, "alpha": float(alpha), "scope": "tuning", **tuning_metrics})
            sweep_rows.append({"architecture": condition_name, "alpha": float(alpha), "scope": "evaluation", **evaluation_metrics})
        best = max(alpha_tuning, key=lambda row: (row["mADM"], -row["alpha"]))
        best_alpha = float(best["alpha"])
        tuned_logs = _fuse_logs(global_logs, local_logs, best_alpha)
        tuned_distances = _category_nll_distances(query_targets, tuned_logs, groups)
        summary_rows.append({"architecture": condition_name, "method": f"Global+Local(tuning_alpha={best_alpha:g})", "split": "evaluation", "selected_alpha": best_alpha, **_subset_metrics(tuned_distances, evaluation_indices, queries, labels, ids)})
        attention_rows.append(_attention_diagnostics(
            output_dir, condition_name, model, table, gt_path, data_root, device,
            args.eval_batch_size, args.num_workers, args.amp, args.attention_samples,
        ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The global-only reference is independent of local architecture.
    global_distances = _category_nll_distances(query_targets, global_logs, groups)
    for split_name, indices in (("tuning", tuning_indices), ("evaluation", evaluation_indices)):
        summary_rows.append({"architecture": "GlobalReference", "method": "Global-only", "split": split_name, **_subset_metrics(global_distances, indices, queries, labels, ids)})
    _write_csv(output_dir / "metrics_summary.csv", summary_rows)
    _write_csv(output_dir / "category_metrics.csv", category_rows)
    _write_csv(output_dir / "fusion_alpha_sweep.csv", sweep_rows)
    _write_csv(output_dir / "retrieval_transitions.csv", transition_rows)
    _atomic_json(output_dir / "attention_summary.json", attention_rows)
    _atomic_json(output_dir / "metrics_summary.json", {
        "protocol": protocol, "metrics": summary_rows,
        "category_metrics": category_rows, "retrieval_transitions": transition_rows,
        "attention": attention_rows,
    })
    print(json.dumps({
        "output_dir": str(output_dir), "evaluation_metrics": [
            row for row in summary_rows if row["split"] == "evaluation"
        ], "selected_alphas": {
            condition["name"]: max(
                (
                    row for row in sweep_rows
                    if row["architecture"] == condition["name"] and row["scope"] == "tuning"
                ),
                key=lambda row: (row["mADM"], -row["alpha"]),
            )["alpha"]
            for condition in conditions
        },
    }, indent=2, ensure_ascii=False))
