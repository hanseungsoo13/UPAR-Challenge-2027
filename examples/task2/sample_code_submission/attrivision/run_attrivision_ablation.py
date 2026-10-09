"""Run the controlled A0--A7 AttriVision training-formulation ablations."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.cli import build_parser, run, validate_args  # noqa: E402
from attrivision.engine.evaluator_abpr import (  # noqa: E402
    evaluate_native52, evaluate_native52_category_nll,
)
from attrivision.engine.trainer_attrivision import train  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import choose_device  # noqa: E402


EXPERIMENTS: dict[str, dict[str, str]] = {
    "A0": {"sampling": "single", "target": "multi_positive", "augmentation": "current"},
    "A1": {"sampling": "multi", "target": "multi_positive", "augmentation": "current"},
    "A2": {"sampling": "single", "target": "diagonal", "augmentation": "current"},
    "A3": {"sampling": "multi", "target": "diagonal", "augmentation": "current"},
    "A4": {"sampling": "multi", "target": "diagonal", "augmentation": "paper_like"},
    "A5": {"sampling": "multi", "target": "diagonal", "augmentation": "rrc_scale_050"},
    "A6": {"sampling": "single", "target": "multi_positive", "augmentation": "resize_pad_crop"},
    "A7": {"sampling": "single", "target": "multi_positive", "augmentation": "resize_pad_crop"},
}
ROOT_OUTPUT = Path("outputs/attrivision_ablation")
TABLE_METRICS = (
    ("AUROC", "macro_auroc"), ("InstF1", "instance_f1"),
    ("BitErr", "mean_hamming_error"), ("ExactMatch", "exact_match"),
    ("R1", "rank1"), ("mAP", "map"), ("mADM", "mADM"),
)
PAIRWISE = (
    ("A1", "A0", "multiple attribute effect"),
    ("A2", "A0", "diagonal target effect"),
    ("A3", "A2", "multiple attribute effect under diagonal"),
    ("A3", "A1", "diagonal effect under multi"),
    ("A4", "A3", "augmentation effect"),
    ("A5", "A3", "minimum RRC area 50% vs 8%"),
    ("A6", "A0", "full resize plus local translation vs A0 RRC"),
    ("A7", "A6", "Category-NLL vs Native52 soft-L1 under the same A6 crop"),
)


def _configure(args: Any) -> Path:
    spec = EXPERIMENTS[args.experiment]
    output_root = Path(getattr(args, "output_root", ROOT_OUTPUT))
    output_dir = output_root / args.experiment
    args.output_dir = str(output_dir)
    args.text_sampling = spec["sampling"]
    args.multi_attributes = 3
    args.contrastive_target = spec["target"]
    args.augmentation = spec["augmentation"]
    args.training_objective = "paper_fce"
    args.prompt_mode = "category_complete"
    args.loss = "focal_clip"
    args.lambda_attr = 0.0
    args.use_fce = True
    args.unique_prompts = False
    # A7 is the Category-NLL counterpart to A6. Other A-series runs retain
    # their original Native52 soft-L1 validation protocol.
    args.validation_protocol = (
        "native52_category_nll" if args.experiment == "A7" else "native52"
    )
    args.retrieval_scoring = "paired_l1"  # documented intent; native52 evaluator is authoritative
    args.selection_metric = "mADM"
    args.clip_model = "ViT-B-32-quickgelu"
    args.pretrained_tag = "openai"
    args.no_pretrained = False
    args.init_checkpoint = None
    args.resume = None
    args.batch_diagnostics = spec["sampling"] == "multi"
    args.training_log_filename = "training_log.csv"
    validate_args(args)
    return output_dir


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _collect_summary(output_root: Path = ROOT_OUTPUT) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, spec in EXPERIMENTS.items():
        path = output_root / name / "validation_metrics.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                rows[name] = {**spec, **json.load(handle)}
    return rows


def print_and_save_summary(output_root: Path = ROOT_OUTPUT) -> None:
    rows = _collect_summary(output_root)
    headers = ["Experiment", "Sampling", "Target", "Aug", *[name for name, _ in TABLE_METRICS]]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for experiment in EXPERIMENTS:
        row = rows.get(experiment)
        if row is None:
            print(" | ".join([experiment, EXPERIMENTS[experiment]["sampling"],
                              EXPERIMENTS[experiment]["target"],
                              EXPERIMENTS[experiment]["augmentation"], *(["-"] * len(TABLE_METRICS))]))
            continue
        values = [f"{float(row[key]):.6f}" if key in row else "-" for _, key in TABLE_METRICS]
        print(" | ".join([experiment, row["sampling"], row["target"], row["augmentation"], *values]))

    differences: dict[str, dict[str, float]] = {}
    print("\nPairwise absolute-point differences (first minus second):")
    for first, second, description in PAIRWISE:
        label = f"{first} - {second} : {description}"
        if first not in rows or second not in rows:
            print(f"{label}: unavailable")
            continue
        delta = {
            name: float(rows[first][key]) - float(rows[second][key])
            for name, key in TABLE_METRICS
            if key in rows[first] and key in rows[second]
        }
        differences[label] = delta
        print(f"{label}: " + ", ".join(f"{key}={value:+.6f}" for key, value in delta.items()))

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for experiment, row in rows.items():
            writer.writerow({
                "Experiment": experiment, "Sampling": row["sampling"],
                "Target": row["target"], "Aug": row["augmentation"],
                **{name: row.get(key, "") for name, key in TABLE_METRICS},
            })
    _write_json(output_root / "pairwise_differences.json", differences)


def main() -> None:
    parser = build_parser()
    parser.description = "Controlled AttriVision A0--A7 training ablations"
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument(
        "--output-root", default=str(ROOT_OUTPUT),
        help="root directory for per-experiment outputs; the experiment name is appended",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        print_and_save_summary(Path(args.output_root))
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --summary-only is used")
    output_dir = _configure(args)
    config = {
        **vars(args),
        "experiment_spec": EXPERIMENTS[args.experiment],
        "binary_head_loss": False,
        "initialization": "OpenAI CLIP ViT-B/32",
        "retrieval_protocol": (
            "Native52 Category-NLL" if args.validation_protocol == "native52_category_nll"
            else "Native52 category softmax -> 40-D -> L1"
        ),
    }
    _write_json(output_dir / "config.json", config)
    if args.mode == "smoke":
        run(args)
        print(f"{args.experiment} shared-path smoke test passed")
        return
    if args.mode not in {"train", "train_eval"}:
        raise ValueError("Ablations support --mode train, train_eval, or smoke")

    checkpoint = train(args)
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    eval_transform = build_eval_transform(args.image_size, args.augmentation)
    if args.validation_protocol == "native52_category_nll":
        metrics = evaluate_native52_category_nll(
            model, payload["attribute_names"], Path(args.data_root).resolve(),
            eval_transform, device, args.eval_batch_size, args.num_workers, args.amp,
            args.max_val_samples, args.category_temperature,
        )
    else:
        metrics = evaluate_native52(
            model, payload["attribute_names"], Path(args.data_root).resolve(),
            eval_transform, device, args.eval_batch_size, args.num_workers, args.amp,
            args.max_val_samples, args.attribute_temperature,
        )
    checkpoint_metrics = payload.get("metrics", {})
    metrics.update({
        "fce_loss": checkpoint_metrics.get("fce_loss"),
        "i2t_loss": checkpoint_metrics.get("i2t_loss"),
        "t2i_loss": checkpoint_metrics.get("t2i_loss"),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
    })
    _write_json(output_dir / "validation_metrics.json", metrics)
    print_and_save_summary(Path(args.output_root))


if __name__ == "__main__":
    main()
