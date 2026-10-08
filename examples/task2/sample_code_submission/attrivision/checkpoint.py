"""AttriVision best/last checkpoint persistence."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .models.attrivision import AttriVision


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_payload(model: AttriVision, attribute_names: Sequence[str], epoch: int,
                  metrics: dict[str, float], prompt_mode: str,
                  training_config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "format_version": 3,
        "architecture": "attrivision_openclip_vit_b_32",
        "model_name": model.model_name,
        "model_state_dict": _cpu_state_dict(model),
        "attribute_names": list(attribute_names),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "prompt_mode": prompt_mode,
    }
    if training_config is not None:
        payload["training_config"] = dict(training_config)
    return payload


def save_best(path: Path, model: AttriVision, attribute_names: Sequence[str],
              epoch: int, metrics: dict[str, float], prompt_mode: str,
              training_config: dict[str, Any]) -> None:
    _atomic_save(
        model_payload(
            model, attribute_names, epoch, metrics, prompt_mode, training_config,
        ),
        path,
    )


def save_last(path: Path, model: AttriVision, attribute_names: Sequence[str], epoch: int,
              metrics: dict[str, float], optimizer: torch.optim.Optimizer,
              scheduler: Any, scaler: Any, best_map: float, best_epoch: int,
              stale_evaluations: int, prompt_mode: str,
              training_config: dict[str, Any]) -> None:
    payload = model_payload(
        model, attribute_names, epoch, metrics, prompt_mode, training_config,
    )
    payload["training_state"] = {
        "scheduler_type": "warmup_cosine",
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_map": float(best_map),
        # Keep the historical key for resume compatibility; this value is the
        # configured selection score (mADM by default), not necessarily mAP.
        "best_score": float(best_map),
        "selection_metric": training_config.get("selection_metric", "mADM"),
        "best_epoch": int(best_epoch),
        "stale_evaluations": int(stale_evaluations),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    _atomic_save(payload, path)


def _load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    required = {"architecture", "model_name", "model_state_dict", "attribute_names"}
    missing = required - payload.keys()
    if missing or payload["architecture"] != "attrivision_openclip_vit_b_32":
        raise ValueError(f"Invalid AttriVision checkpoint; missing={sorted(missing)}")
    return payload


def load_model(path: str | Path, device: torch.device) -> tuple[AttriVision, dict[str, Any]]:
    payload = _load(path)
    model = AttriVision(pretrained=None, model_name=payload["model_name"])
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"binary_head.weight", "binary_head.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"Checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(device), payload


def resume_training(path: str | Path, model: AttriVision,
                    optimizer: torch.optim.Optimizer, scheduler: Any,
                    scaler: Any, device: torch.device) -> dict[str, Any]:
    payload = _load(path, map_location=device)
    state = payload.get("training_state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not resumable: {path}")
    if state.get("scheduler_type") != "warmup_cosine":
        raise ValueError(
            "This checkpoint uses the old validation-plateau scheduler and cannot be "
            "resumed with the new warmup-cosine schedule. Start a new output directory."
        )
    if int(payload.get("format_version", 1)) < 3:
        raise ValueError(
            "This checkpoint predates QuickGELU and standard CLIP AdamW parameter "
            "groups. Evaluate/package it as-is, but start a fresh training run."
        )
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if set(missing) - {"binary_head.weight", "binary_head.bias"} or unexpected:
        raise ValueError(f"Resume checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return payload
