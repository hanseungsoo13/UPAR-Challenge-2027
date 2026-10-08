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
    image_width: int | None = None
    resize_width: int | None = None
    crop_policy: str = "current"
    augmix: bool = True
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    @property
    def output_size(self) -> int | tuple[int, int]:
        """Return the model input size, preserving legacy square configs."""
        return self.image_size if self.image_width is None else (self.image_size, self.image_width)

    @property
    def resize_output_size(self) -> int | tuple[int, int]:
        """Return the evaluation resize size, preserving legacy behavior."""
        if self.image_width is None and self.resize_width is None:
            return self.resize_size
        return (self.resize_size, self.resize_width or self.image_width)

    @classmethod
    def from_dict(cls, values: dict) -> "PreprocessingConfig":
        return cls(
            image_size=int(values["image_size"]),
            resize_size=int(values["resize_size"]),
            image_width=(int(values["image_width"]) if values.get("image_width") is not None else None),
            resize_width=(int(values["resize_width"]) if values.get("resize_width") is not None else None),
            crop_policy=str(values.get("crop_policy", "current")),
            augmix=bool(values.get("augmix", True)),
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
