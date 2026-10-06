"""Annotation parsing, image path resolution, datasets, and transforms."""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import NUM_ATTRIBUTES, PreprocessingConfig


@dataclass
class AnnotationTable:
    image_paths: list[str]
    labels: np.ndarray
    attribute_names: list[str]


def clean_header(value: str) -> str:
    value = value.strip().lstrip("\ufeff")
    return value[1:].strip() if value.startswith("#") else value


def read_gt_csv(path: str | Path) -> AnnotationTable:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Annotation file is empty: {path}") from exc
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"Annotation file has a header but no samples: {path}")

    header = [clean_header(item) for item in raw_header]
    if len(header) == NUM_ATTRIBUTES and len(rows[0]) == NUM_ATTRIBUTES + 1:
        warnings.warn(f"{path}: repairing legacy header missing '# image'", stacklevel=2)
        header = ["image", *header]
    if len(header) != NUM_ATTRIBUTES + 1:
        raise ValueError(f"Expected image + 40 attributes in {path}, got {len(header)} columns")
    if header[0].lower() not in {"image", "image_path", "path"}:
        raise ValueError(f"First column in {path} must be an image path, got {header[0]!r}")
    if len(set(header[1:])) != NUM_ATTRIBUTES:
        raise ValueError(f"Attribute names in {path} must be unique")
    for line_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"{path}:{line_number}: expected {len(header)} columns, got {len(row)}")

    image_paths = [row[0].strip() for row in rows]
    try:
        labels = np.asarray([row[1:] for row in rows], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Labels in {path} must be numeric") from exc
    valid = np.isin(labels, (0.0, 1.0))
    if not valid.all():
        bad = np.argwhere(~valid)[0]
        raise ValueError(f"{path}:{bad[0] + 2}: non-binary label {labels[tuple(bad)]}")
    return AnnotationTable(image_paths, labels, header[1:])


def find_annotation_file(data_root: Path, split: str, name: str = "gt.csv") -> Path:
    candidates = [
        data_root / "annotations" / "task2" / split / name,
        data_root / "annotations" / split / name,
    ]
    if name == "gt.csv":
        candidates.append(data_root / "annotations" / f"{split}.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Task 2 {split} {name}. Tried:\n  - {tried}")


def resolve_image_path(raw_path: str | Path, roots: Sequence[Path]) -> Path:
    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [root / raw for root in roots]
    unique = list(dict.fromkeys(candidate.resolve(strict=False) for candidate in candidates))
    for candidate in unique:
        if candidate.is_file():
            return candidate
    tried = "\n  - ".join(str(path) for path in unique)
    raise FileNotFoundError(f"Image {str(raw_path)!r} was not found. Tried:\n  - {tried}")


def build_train_transform(config: PreprocessingConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std),
    ])


def build_eval_transform(config: PreprocessingConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(config.resize_size, antialias=True),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std),
    ])


class AttributeDataset(Dataset):
    def __init__(self, table: AnnotationTable, roots: Sequence[Path], transform: Any,
                 max_samples: int | None = None) -> None:
        count = len(table.image_paths) if max_samples is None else min(max_samples, len(table.image_paths))
        self.image_paths = table.image_paths[:count]
        self.labels = table.labels[:count]
        self.roots = list(roots)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = resolve_image_path(self.image_paths[index], self.roots)
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to decode image: {path}") from exc
        return tensor, torch.from_numpy(self.labels[index].copy())


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str | Path], roots: Sequence[Path], transform: Any) -> None:
        self.paths = list(paths)
        self.roots = list(roots)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = resolve_image_path(self.paths[index], self.roots)
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to decode image: {path}") from exc
