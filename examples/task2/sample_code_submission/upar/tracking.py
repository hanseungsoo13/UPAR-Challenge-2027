"""Persistent text and CSV logging for training runs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO


METRIC_FIELDS = (
    "epoch",
    "train_loss",
    "learning_rate",
    "model_rank1",
    "model_map",
    "model_madm",
    "ema_rank1",
    "ema_map",
    "ema_madm",
    "best_map",
    "best_score",
    "selection_metric",
    "best_epoch",
    "evaluations_without_improvement",
    "elapsed_seconds",
)


class TrainingLogger:
    """Mirror training messages to the terminal and ``train.log``."""

    def __init__(self, path: Path, resume: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = path.open("a" if resume else "w", encoding="utf-8")
        if resume:
            self._handle.write("\n" + "=" * 72 + "\nRESUMED TRAINING\n")
            self._handle.flush()

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class MetricsCSV:
    """Write one durable summary row per completed epoch."""

    def __init__(self, path: Path, resume: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        append = resume and path.is_file() and path.stat().st_size > 0
        self._handle: TextIO = path.open("a" if append else "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=METRIC_FIELDS)
        if not append:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, values: dict[str, Any]) -> None:
        self._writer.writerow({field: values.get(field, "") for field in METRIC_FIELDS})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
