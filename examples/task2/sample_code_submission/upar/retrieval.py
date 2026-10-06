"""Attribute inference, Task 2 annotation parsing, distances, and metrics."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import NUM_ATTRIBUTES
from .data import AnnotationTable, ImagePathDataset, clean_header


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


@torch.inference_mode()
def infer_probabilities(model: nn.Module, paths: Sequence[str | Path], roots: Sequence[Path],
                        transform: Any, device: torch.device, batch_size: int,
                        num_workers: int, amp: bool, progress_every: int = 0,
                        progress_prefix: str = "[inference]") -> np.ndarray:
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    loader = DataLoader(
        ImagePathDataset(paths, roots, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    processed = 0
    started = time.perf_counter()
    total_batches = len(loader)
    for batch_index, images in enumerate(loader, start=1):
        with autocast(device, amp):
            probabilities = torch.sigmoid(model(images.to(device, non_blocking=True)))
        chunks.append(probabilities.float().cpu().numpy())
        processed += len(images)
        if progress_every and (batch_index % progress_every == 0 or batch_index == total_batches):
            elapsed = time.perf_counter() - started
            rate = processed / max(elapsed, 1e-9)
            print(
                f"{progress_prefix} batch {batch_index}/{total_batches}: "
                f"{processed}/{len(paths)} images, {elapsed:.1f}s, {rate:.1f} images/s",
                flush=True,
            )
    result = (
        np.concatenate(chunks).astype(np.float32, copy=False)
        if chunks else np.empty((0, NUM_ATTRIBUTES), dtype=np.float32)
    )
    if result.shape != (len(paths), NUM_ATTRIBUTES) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid probability output: shape={result.shape}")
    return result


def l1_attribute_distances(queries: np.ndarray, probabilities: np.ndarray,
                           query_chunk_size: int = 256) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if queries.ndim != 2 or probabilities.ndim != 2 or queries.shape[1] != probabilities.shape[1]:
        raise ValueError(f"Expected matching [N,A] arrays, got {queries.shape} and {probabilities.shape}")
    if not np.isin(queries, (0.0, 1.0)).all():
        raise ValueError("queries must contain binary values")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or Inf")

    distances = np.empty((len(queries), len(probabilities)), dtype=np.float32)
    base = probabilities.sum(1, dtype=np.float32)[None, :]
    projection = (1.0 - 2.0 * probabilities).T
    for start in range(0, len(queries), query_chunk_size):
        stop = min(start + query_chunk_size, len(queries))
        distances[start:stop] = base + queries[start:stop] @ projection
    np.maximum(distances, 0.0, out=distances)
    if not np.isfinite(distances).all():
        raise RuntimeError("Distance matrix contains NaN or Inf")
    return distances


def retrieval_metrics(distances: np.ndarray, gallery_ids: np.ndarray) -> tuple[float, float]:
    gallery_ids = np.asarray(gallery_ids, dtype=np.int64).reshape(-1)
    if distances.ndim != 2 or distances.shape[1] != len(gallery_ids) or not len(gallery_ids):
        raise ValueError("distances must be non-empty [queries, gallery] matching gallery_ids")
    rank1_hits = 0
    average_precisions: list[float] = []
    for query_id, row in enumerate(distances):
        positive_count = int(np.count_nonzero(gallery_ids == query_id))
        if positive_count == 0:
            raise ValueError(f"Query {query_id} has no positive gallery image")
        relevant = gallery_ids[np.argsort(row, kind="stable")] == query_id
        rank1_hits += int(relevant[0])
        ranks = np.flatnonzero(relevant) + 1
        average_precisions.append(float((np.arange(1, positive_count + 1) / ranks).mean()))
    return rank1_hits / len(distances), float(np.mean(average_precisions))


def _read_numeric_csv(path: Path) -> tuple[np.ndarray, list[str] | None]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    header = None
    try:
        np.asarray(rows[0], dtype=np.float32)
    except ValueError:
        header = [clean_header(value) for value in rows.pop(0)]
    try:
        return np.asarray(rows, dtype=np.float32), header
    except ValueError as exc:
        raise ValueError(f"Expected numeric values in {path}") from exc


def _read_ids_csv(path: Path, expected_image_paths: Sequence[str]) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    try:
        float(rows[0][-1])
    except ValueError:
        rows.pop(0)
    if len(rows) != len(expected_image_paths):
        raise ValueError(f"{path} IDs and gt.csv image counts differ")

    values = np.empty(len(rows), dtype=np.int64)
    for index, (row, expected_path) in enumerate(zip(rows, expected_image_paths)):
        if len(row) not in (1, 2):
            raise ValueError(f"{path}:{index + 1}: expected id or image_path,id")
        if len(row) == 2 and Path(row[0].strip()).as_posix() != Path(expected_path).as_posix():
            raise ValueError(f"{path}:{index + 1}: image order does not match gt.csv")
        try:
            numeric = float(row[-1])
        except ValueError as exc:
            raise ValueError(f"{path}:{index + 1}: semantic ID must be numeric") from exc
        if not numeric.is_integer():
            raise ValueError(f"{path}:{index + 1}: semantic ID must be an integer")
        values[index] = int(numeric)
    return values


def load_retrieval_annotations(split_dir: Path, table: AnnotationTable) -> tuple[np.ndarray, np.ndarray, list[str]]:
    query_path = split_dir / "queries.csv"
    ids_path = split_dir / "ids.csv"
    if query_path.is_file() and ids_path.is_file():
        queries, header = _read_numeric_csv(query_path)
        ids = _read_ids_csv(ids_path, table.image_paths)
        attribute_names = header or table.attribute_names
        if header and len(header) == NUM_ATTRIBUTES + 1 and header[0].lower() in {"id", "query", "query_id"}:
            attribute_names = header[1:]
            queries = queries[:, 1:]
    else:
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        attribute_names = table.attribute_names
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != NUM_ATTRIBUTES or not np.isin(queries, (0, 1)).all():
        raise ValueError(f"queries must be binary [Q,40], got {queries.shape}")
    if len(ids) != len(table.image_paths) or (len(ids) and (ids.min() < 0 or ids.max() >= len(queries))):
        raise ValueError("ids must map every gallery image to a zero-based query row")
    if len(attribute_names) != NUM_ATTRIBUTES or len(set(attribute_names)) != NUM_ATTRIBUTES:
        raise ValueError("queries.csv must describe 40 unique attributes")
    return queries, ids, attribute_names


def reorder_columns(values: np.ndarray, source_names: Sequence[str], target_names: Sequence[str]) -> np.ndarray:
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        raise ValueError("Attribute names must be unique")
    missing = sorted(set(target_names) - set(source_names))
    extra = sorted(set(source_names) - set(target_names))
    if missing or extra:
        raise ValueError(f"Attribute vocabulary mismatch; missing={missing}, extra={extra}")
    index = {name: column for column, name in enumerate(source_names)}
    return values[:, [index[name] for name in target_names]]
