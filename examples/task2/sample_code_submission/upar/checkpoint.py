"""Checkpoint serialization and validation-probability caching."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import NUM_ATTRIBUTES, PreprocessingConfig
from .modeling import UPARModel


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(path: Path, model: nn.Module, attribute_names: Sequence[str],
                    preprocessing: PreprocessingConfig, positive_ratios: np.ndarray,
                    epoch: int, metrics: dict[str, float], model_kind: str, dropout: float) -> None:
    payload = {
        "format_version": 1,
        "architecture": "torchvision_convnext_base",
        "model_state_dict": cpu_state_dict(model),
        "attribute_names": list(attribute_names),
        "preprocessing": asdict(preprocessing),
        "positive_ratios": np.asarray(positive_ratios, dtype=np.float32),
        "dropout": float(dropout),
        "epoch": int(epoch),
        "metrics": metrics,
        "model_kind": model_kind,
    }
    _atomic_torch_save(payload, path)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader_generator: torch.Generator,
    attribute_names: Sequence[str],
    preprocessing: PreprocessingConfig,
    positive_ratios: np.ndarray,
    epoch: int,
    latest_metrics: dict[str, float],
    best_map: float,
    best_epoch: int,
    evaluations_without_improvement: int,
    dropout: float,
) -> None:
    """Save the latest model and all state needed for training continuation."""
    payload = {
        "format_version": 2,
        "architecture": "torchvision_convnext_base",
        "model_state_dict": cpu_state_dict(model),
        "ema_state_dict": cpu_state_dict(ema_model) if ema_model is not None else None,
        "attribute_names": list(attribute_names),
        "preprocessing": asdict(preprocessing),
        "positive_ratios": np.asarray(positive_ratios, dtype=np.float32),
        "dropout": float(dropout),
        "epoch": int(epoch),
        "metrics": latest_metrics,
        "model_kind": "model",
        "training_state": {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_map": float(best_map),
            "best_epoch": int(best_epoch),
            "evaluations_without_improvement": int(evaluations_without_improvement),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if any(parameter.is_cuda for parameter in model.parameters())
                else None
            ),
            "loader_generator_state": loader_generator.get_state(),
        },
    }
    _atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    """Restore a ``last.pth`` checkpoint and return its progress metadata."""
    checkpoint = _load_payload(path)
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError(f"Checkpoint is not resumable (training state missing): {path}")
    checkpoint_ema = checkpoint.get("ema_state_dict")
    if (ema_model is None) != (checkpoint_ema is None):
        raise ValueError("EMA setting must match the checkpoint when resuming")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint_ema, strict=True)
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler.load_state_dict(training_state["scheduler_state_dict"])
    scaler.load_state_dict(training_state["scaler_state_dict"])
    loader_generator.set_state(training_state["loader_generator_state"])
    random.setstate(training_state["python_rng_state"])
    np.random.set_state(training_state["numpy_rng_state"])
    torch.set_rng_state(training_state["torch_rng_state"])
    cuda_state = training_state.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return checkpoint


def _load_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[UPARModel, dict[str, Any]]:
    path = Path(path)
    checkpoint = _load_payload(path)
    required = {"model_state_dict", "attribute_names", "preprocessing", "positive_ratios"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    attributes = checkpoint["attribute_names"]
    if len(attributes) != NUM_ATTRIBUTES or len(set(attributes)) != NUM_ATTRIBUTES:
        raise ValueError("Checkpoint has an invalid attribute vocabulary")
    model = UPARModel(dropout=float(checkpoint.get("dropout", 0.7)), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_probabilities(cache_path: Path, checkpoint_path: Path, compute: Any) -> np.ndarray:
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    fingerprint = _sha256(checkpoint_path)
    if cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("checkpoint_sha256") == fingerprint:
                return np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    probabilities = compute()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, probabilities, allow_pickle=False)
    metadata_path.write_text(json.dumps({
        "checkpoint_sha256": fingerprint,
        "shape": list(probabilities.shape),
    }, indent=2), encoding="utf-8")
    return probabilities
