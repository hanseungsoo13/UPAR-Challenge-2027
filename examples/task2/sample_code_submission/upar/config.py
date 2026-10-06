"""Shared constants, preprocessing configuration, and runtime utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = PACKAGE_DIR.parent


def _find_repository_root() -> Path:
    """Find the development checkout without assuming submission path depth."""
    for candidate in (SUBMISSION_DIR, *SUBMISSION_DIR.parents):
        if (candidate / "environment.yml").is_file() and (candidate / "examples").is_dir():
            return candidate
    return SUBMISSION_DIR


REPOSITORY_ROOT = _find_repository_root()
DEFAULT_CHECKPOINT = SUBMISSION_DIR / "assets" / "model_best.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_ATTRIBUTES = 40


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 224
    resize_size: int = 232
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    @classmethod
    def from_dict(cls, values: dict) -> "PreprocessingConfig":
        return cls(
            image_size=int(values["image_size"]),
            resize_size=int(values["resize_size"]),
            mean=tuple(values.get("mean", IMAGENET_MEAN)),
            std=tuple(values.get("std", IMAGENET_STD)),
        )


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    return device
