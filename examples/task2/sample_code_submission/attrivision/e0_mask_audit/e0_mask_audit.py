"""Audit AttriVision's sampled FCE masks without loading a checkpoint.

The audit deliberately reuses the production PromptCollator and
FocalCLIPLoss.positive_mask implementations. It creates the same train batch
metadata as A1, but replaces images with tiny dummy tensors and replaces the
CLIP tokenizer with a shape-only tokenizer. No image/text encoder, checkpoint,
or gradient computation is needed because FCE mask construction depends only
on labels, sampled semantic states, and text owners.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SUBMISSION_DIR = Path(__file__).resolve().parents[2]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import (  # noqa: E402
    PromptBatch,
    PromptCollator,
    UniquePromptBatchSampler,
    semantic_label_matrix,
)
from attrivision.losses.focal_clip_loss import FocalCLIPLoss  # noqa: E402
from upar.config import REPOSITORY_ROOT, set_seed  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402


class _ShapeOnlyTokenizer:
    """Return CLIP-shaped token tensors without constructing a CLIP model."""

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.zeros((len(texts), 77), dtype=torch.long)


class _LabelOnlyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset preserving labels and index order while avoiding image I/O."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = np.asarray(labels, dtype=np.float32)
        self.dummy_image = torch.zeros(3, 1, 1, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dummy_image, torch.from_numpy(self.labels[index].copy())


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _mask_counts(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int | float | None]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Actual and expected masks must have the same shape, got "
            f"{tuple(actual.shape)} and {tuple(expected.shape)}"
        )
    expected = expected.bool()
    actual = actual.bool()
    gt_positive = int(expected.sum())
    gt_negative = int((~expected).sum())
    false_negative = int((expected & ~actual).sum())
    false_positive = int((~expected & actual).sum())
    return {
        "candidate_pairs": int(expected.numel()),
        "gt_positive_pairs": gt_positive,
        "gt_negative_pairs": gt_negative,
        "false_negative_pairs": false_negative,
        "false_positive_pairs": false_positive,
        "fnr": _ratio(false_negative, gt_positive),
        "fpr": _ratio(false_positive, gt_negative),
    }


def _new_mask_counter() -> dict[str, int]:
    return {
        "candidate_pairs": 0,
        "gt_positive_pairs": 0,
        "gt_negative_pairs": 0,
        "false_negative_pairs": 0,
        "false_positive_pairs": 0,
    }


def _add_counts(total: dict[str, int], counts: Mapping[str, int | float | None]) -> None:
    for key in total:
        total[key] += int(counts[key] or 0)


def _finalize_mask_counter(counter: Mapping[str, int]) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = dict(counter)
    result["fnr"] = _ratio(counter["false_negative_pairs"], counter["gt_positive_pairs"])
    result["fpr"] = _ratio(counter["false_positive_pairs"], counter["gt_negative_pairs"])
    return result


def _new_fallback_counter() -> dict[str, int]:
    return {
        "selected_texts": 0,
        "fallback_selected_texts": 0,
        "gt_positive_pairs": 0,
        "fallback_gt_positive_pairs": 0,
    }


def _finalize_fallback_counter(counter: Mapping[str, int]) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = dict(counter)
    result["selected_text_share"] = _ratio(
        counter["fallback_selected_texts"], counter["selected_texts"],
    )
    result["gt_positive_pair_share"] = _ratio(
        counter["fallback_gt_positive_pairs"], counter["gt_positive_pairs"],
    )
    return result


def _category_specs(mapper: CategoryPromptMapper) -> list[tuple[str, list[int]]]:
    names = [name for name, *_ in mapper._MULTI_GROUPS]
    names.extend(name for name, *_ in mapper._BINARY_GROUPS)
    return list(zip(names, mapper.category_indices()))


def _new_category_counter(
    specs: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, Any]:
    return {
        "images": 0,
        "category_slots": 0,
        "conflict_slots": 0,
        "images_with_conflict": 0,
        "by_category": {
            name: {"images": 0, "conflict_images": 0}
            for name, _ in specs
        },
    }


def _update_category_counter(
    counter: dict[str, Any],
    semantic_labels: torch.Tensor,
    specs: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, Any]:
    batch_size = int(semantic_labels.shape[0])
    any_conflict = torch.zeros(batch_size, dtype=torch.bool)
    counter["images"] += batch_size
    counter["category_slots"] += batch_size * len(specs)
    for name, indices in specs:
        active_count = semantic_labels[:, indices].sum(dim=1)
        conflict = active_count > 1
        conflict_count = int(conflict.sum())
        counter["conflict_slots"] += conflict_count
        any_conflict |= conflict
        counter["by_category"][name]["images"] += batch_size
        counter["by_category"][name]["conflict_images"] += conflict_count
    counter["images_with_conflict"] += int(any_conflict.sum())
    return counter


def _finalize_category_counter(counter: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "images": int(counter["images"]),
        "category_slots": int(counter["category_slots"]),
        "conflict_slots": int(counter["conflict_slots"]),
        "images_with_conflict": int(counter["images_with_conflict"]),
        "conflict_rate": _ratio(counter["conflict_slots"], counter["category_slots"]),
        "image_conflict_rate": _ratio(
            counter["images_with_conflict"], counter["images"],
        ),
        "by_category": {},
    }
    for name, values in counter["by_category"].items():
        result["by_category"][name] = {
            "images": int(values["images"]),
            "conflict_images": int(values["conflict_images"]),
            "conflict_rate": _ratio(values["conflict_images"], values["images"]),
        }
    return result


def _batch_row(
    seed: int,
    batch_index: int,
    batch: PromptBatch,
    image_to_text: Mapping[str, int | float | None],
    text_to_image: Mapping[str, int | float | None],
    fallback: Mapping[str, int],
    category: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "batch_index": batch_index,
        "images": len(batch.images),
        "texts": len(batch.tokens),
        "i2t_gt_positive_pairs": image_to_text["gt_positive_pairs"],
        "i2t_gt_negative_pairs": image_to_text["gt_negative_pairs"],
        "i2t_false_negative_pairs": image_to_text["false_negative_pairs"],
        "i2t_false_positive_pairs": image_to_text["false_positive_pairs"],
        "t2i_false_negative_pairs": text_to_image["false_negative_pairs"],
        "t2i_false_positive_pairs": text_to_image["false_positive_pairs"],
        "fallback_selected_texts": fallback["fallback_selected_texts"],
        "selected_texts": fallback["selected_texts"],
    }
    if category is not None:
        row.update({
            "category_conflict_slots": category["conflict_slots"],
            "category_slots": category["category_slots"],
            "images_with_category_conflict": category["images_with_conflict"],
        })
    return row


def _build_loader(
    labels: np.ndarray,
    attribute_names: Sequence[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[DataLoader, torch.Tensor, CategoryPromptMapper | None]:
    label_tensor = torch.from_numpy(labels.copy())
    semantic_labels = semantic_label_matrix(label_tensor, attribute_names, args.prompt_mode)
    mapper = CategoryPromptMapper(attribute_names) if args.prompt_mode == "category_complete" else None
    dataset = _LabelOnlyDataset(labels)
    tokenizer = _ShapeOnlyTokenizer()
    collator_unique = args.unique_prompts and args.prompt_mode != "paper_binary"
    collator = PromptCollator(
        attribute_names,
        tokenizer,
        args.text_sampling,
        args.multi_attributes,
        args.prompt_mode,
        collator_unique,
    )

    loader_options: dict[str, Any] = {}
    if args.unique_prompts and args.prompt_mode != "paper_binary":
        prompts_per_image = 1 if args.text_sampling == "single" else args.multi_attributes
        sampler = UniquePromptBatchSampler(
            semantic_labels,
            args.batch_size,
            prompts_per_image,
            seed,
        )
        collator.semantic_frequencies = sampler.frequencies
        loader_options["batch_sampler"] = sampler
    else:
        loader_options.update({
            "batch_size": args.batch_size,
            "shuffle": True,
            "generator": torch.Generator().manual_seed(seed),
            # Match trainer_attrivision.py: full-size datasets drop a short tail.
            "drop_last": len(dataset) >= args.batch_size,
        })

    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
        **loader_options,
    )
    return loader, semantic_labels, mapper


def audit_seed(
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recreate one train epoch and audit every sampled candidate pair."""
    set_seed(seed, deterministic=args.deterministic)
    data_root = Path(args.data_root).resolve()
    train_gt = find_annotation_file(data_root, "train")
    table = read_gt_csv(train_gt)
    count = len(table.image_paths) if args.max_train_samples is None else min(
        args.max_train_samples, len(table.image_paths),
    )
    labels = table.labels[:count]
    loader, _, mapper = _build_loader(labels, table.attribute_names, args, seed)
    criterion = FocalCLIPLoss("focal_clip", args.contrastive_target)
    category_specs = _category_specs(mapper) if mapper is not None else []
    category_counter = _new_category_counter(category_specs) if mapper is not None else None

    fallback_indices: list[int] = []
    if mapper is not None:
        fallback_indices = [
            mapper._semantic_index[fallback]
            for _, _, _, fallback in mapper._MULTI_GROUPS
        ]

    mask_counters = {
        "image_to_text": _new_mask_counter(),
        "text_to_image": _new_mask_counter(),
    }
    fallback_counter = _new_fallback_counter()
    batch_rows: list[dict[str, Any]] = []
    observed_images = 0
    observed_texts = 0
    observed_batches = 0

    try:
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            if not isinstance(batch, PromptBatch):
                raise TypeError("PromptCollator must return PromptBatch")
            semantic = batch.semantic_labels.bool()
            selected = batch.selected_semantics.bool()
            owners = batch.text_owners.long()
            logits = torch.zeros((len(batch.images), len(batch.tokens)), dtype=torch.float32)

            # This is the exact production FCE mask, including selected
            # candidate semantics and owner validation.
            actual_i2t = criterion.positive_mask(logits, semantic, selected, owners)
            expected_i2t = (semantic.float() @ selected.float().T) > 0
            actual_t2i = actual_i2t.T
            expected_t2i = expected_i2t.T
            i2t_counts = _mask_counts(actual_i2t, expected_i2t)
            t2i_counts = _mask_counts(actual_t2i, expected_t2i)
            _add_counts(mask_counters["image_to_text"], i2t_counts)
            _add_counts(mask_counters["text_to_image"], t2i_counts)

            fallback_text = (
                selected[:, fallback_indices].any(dim=1)
                if fallback_indices else torch.zeros(len(selected), dtype=torch.bool)
            )
            fallback_counter["selected_texts"] += int(selected.shape[0])
            fallback_counter["fallback_selected_texts"] += int(fallback_text.sum())
            fallback_counter["gt_positive_pairs"] += int(expected_i2t.sum())
            fallback_counter["fallback_gt_positive_pairs"] += int(
                expected_i2t[:, fallback_text].sum()
            )

            batch_category = None
            if mapper is not None and category_counter is not None:
                before = {
                    "images": category_counter["images"],
                    "category_slots": category_counter["category_slots"],
                    "conflict_slots": category_counter["conflict_slots"],
                    "images_with_conflict": category_counter["images_with_conflict"],
                }
                _update_category_counter(category_counter, semantic, category_specs)
                batch_category = {
                    key: category_counter[key] - before[key]
                    for key in before
                }

            batch_rows.append(_batch_row(
                seed, batch_index, batch, i2t_counts, t2i_counts,
                {
                    "fallback_selected_texts": int(fallback_text.sum()),
                    "selected_texts": int(selected.shape[0]),
                },
                batch_category,
            ))
            observed_batches += 1
            observed_images += len(batch.images)
            observed_texts += len(batch.tokens)
    finally:
        del loader

    if observed_batches == 0:
        raise RuntimeError(
            "The audit produced zero batches. Check --batch-size, --max-train-samples, "
            "and the trainer's drop_last behavior."
        )

    result = {
        "seed": seed,
        "batches": observed_batches,
        "images": observed_images,
        "texts": observed_texts,
        "mask": {
            "image_to_text": _finalize_mask_counter(mask_counters["image_to_text"]),
            "text_to_image": _finalize_mask_counter(mask_counters["text_to_image"]),
        },
        "fallback": (
            _finalize_fallback_counter(fallback_counter)
            if mapper is not None else None
        ),
        "category_conflict": (
            _finalize_category_counter(category_counter)
            if category_counter is not None else None
        ),
    }
    return result, batch_rows


def _collect_numeric_paths(
    value: Any,
    path: tuple[str, ...],
    output: dict[tuple[str, ...], list[float]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _collect_numeric_paths(child, (*path, str(key)), output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            output.setdefault(path, []).append(float(value))


def _set_nested(mapping: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for run in runs:
        _collect_numeric_paths(
            {key: run[key] for key in ("mask", "fallback", "category_conflict")},
            (),
            values,
        )
    aggregate: dict[str, Any] = {"seed_count": len(runs)}
    for path, numbers in sorted(values.items()):
        array = np.asarray(numbers, dtype=np.float64)
        _set_nested(aggregate, path, {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        })
    return aggregate


def _seed_summary_row(run: Mapping[str, Any]) -> dict[str, Any]:
    i2t = run["mask"]["image_to_text"]
    t2i = run["mask"]["text_to_image"]
    row = {
        "seed": run["seed"],
        "batches": run["batches"],
        "images": run["images"],
        "texts": run["texts"],
        "i2t_fnr": i2t["fnr"],
        "i2t_fpr": i2t["fpr"],
        "t2i_fnr": t2i["fnr"],
        "t2i_fpr": t2i["fpr"],
    }
    fallback = run["fallback"]
    if fallback is not None:
        row["fallback_selected_share"] = fallback["selected_text_share"]
        row["fallback_gt_positive_pair_share"] = fallback["gt_positive_pair_share"]
    category = run["category_conflict"]
    if category is not None:
        row["category_conflict_rate"] = category["conflict_rate"]
        row["category_image_conflict_rate"] = category["image_conflict_rate"]
    return row


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E0 audit of AttriVision FCE supervision masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs/attrivision_e0_mask_audit")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-batches", type=int, help="debug/preview limit; omit for one full epoch")
    parser.add_argument("--text-sampling", choices=("single", "multi"), default="multi")
    parser.add_argument("--multi-attributes", type=int, default=3)
    parser.add_argument(
        "--prompt-mode",
        choices=("category_complete", "binary_positive", "paper_binary"),
        default="category_complete",
    )
    parser.add_argument(
        "--contrastive-target",
        choices=("diagonal", "multi_positive"),
        default="multi_positive",
        help="FCE target whose production mask is audited",
    )
    parser.add_argument(
        "--unique-prompts", action=argparse.BooleanOptionalAction, default=False,
        help="match trainer_attrivision.py's unique prompt/batch sampler",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="run several independent seeds and report mean/std, e.g. --seeds 42 43 44",
    )
    parser.add_argument("--deterministic", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for field in ("batch_size", "multi_attributes"):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    for field in ("max_train_samples", "max_batches"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.seeds is not None and not args.seeds:
        raise ValueError("--seeds must contain at least one seed")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    seeds = list(dict.fromkeys(args.seeds if args.seeds is not None else [args.seed]))
    output_dir = Path(args.output_dir)
    runs: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[E0] auditing seed={seed}", flush=True)
        result, rows = audit_seed(args, seed)
        runs.append(result)
        batch_rows.extend(rows)
        _write_json(output_dir / f"seed_{seed}.json", result)

    config = {
        key: value for key, value in vars(args).items()
        if key != "seeds"
    }
    config["seeds"] = seeds
    summary = {
        "experiment": "E0_mask_audit",
        "checkpoint_loaded": False,
        "encoders_run": False,
        "candidate_scope": "only sampled text candidates in each recreated train batch",
        "config": config,
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "seed_summary.csv", [_seed_summary_row(run) for run in runs])
    _write_csv(output_dir / "batch_metrics.csv", batch_rows)

    aggregate = summary["aggregate"]
    i2t = aggregate.get("mask", {}).get("image_to_text", {})
    t2i = aggregate.get("mask", {}).get("text_to_image", {})
    fallback = aggregate.get("fallback", {})
    print(
        "[E0] complete: "
        f"i2t FNR={i2t.get('fnr', {}).get('mean', 'n/a')}, "
        f"i2t FPR={i2t.get('fpr', {}).get('mean', 'n/a')}, "
        f"t2i FNR={t2i.get('fnr', {}).get('mean', 'n/a')}, "
        f"t2i FPR={t2i.get('fpr', {}).get('mean', 'n/a')}, "
        f"fallback share={fallback.get('selected_text_share', {}).get('mean', 'n/a')}"
    )
    print(f"[E0] artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
