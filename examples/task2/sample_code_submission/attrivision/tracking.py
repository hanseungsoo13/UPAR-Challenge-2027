"""Text and epoch-wise CSV logging for AttriVision."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FIELDS = (
    "epoch", "total_loss", "fce_loss", "category_ce_loss", "bce_loss", "prototype_loss", "set_loss", "i2t_loss", "t2i_loss", "learning_rate",
    "rank1", "rank5", "rank10", "map", "mADM", "mINP", "best_map", "best_score",
    "selection_metric", "best_epoch",
    "macro_auroc", "macro_ap", "instance_f1", "mean_hamming_error", "exact_match",
    "semantic_top1",
    "stale_evaluations", "elapsed_seconds",
    "train_batches", "average_batch_size", "cuda_peak_allocated_gib",
    "cuda_peak_reserved_gib",
)


class RunLogger:
    def __init__(self, output_dir: Path, resume: bool,
                 csv_filename: str = "metrics.csv") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log = (output_dir / "train.log").open("a" if resume else "w", encoding="utf-8")
        csv_path = output_dir / csv_filename
        append = resume and csv_path.is_file() and csv_path.stat().st_size > 0
        self._csv = csv_path.open("a" if append else "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=FIELDS)
        if not append:
            self._writer.writeheader()
        if resume:
            self.log("=" * 28 + " RESUMED TRAINING " + "=" * 28)

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._log.write(message + "\n")
        self._log.flush()

    def metrics(self, values: dict[str, Any]) -> None:
        self._writer.writerow({field: values.get(field, "") for field in FIELDS})
        self._csv.flush()

    def close(self) -> None:
        self._log.close()
        self._csv.close()
