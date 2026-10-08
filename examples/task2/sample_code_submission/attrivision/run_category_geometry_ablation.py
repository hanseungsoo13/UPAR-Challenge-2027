"""D0--D2 continuation fine-tuning for Native52 Category-NLL geometry."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import _load, load_model  # noqa: E402
from attrivision.cli import build_parser, validate_args  # noqa: E402
from attrivision.engine.evaluator_abpr import evaluate_native52_category_nll  # noqa: E402
from attrivision.engine.trainer_attrivision import train  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import choose_device  # noqa: E402


EXPERIMENTS = {
    "D0": {"category_ce_mode": "off", "objective": "L_FCE"},
    "D1": {"category_ce_mode": "only", "objective": "L_CatCE"},
    "D2": {"category_ce_mode": "combined", "objective": "L_FCE + 1.0 * L_CatCE"},
}
ROOT_OUTPUT = Path("outputs/attrivision_category_geometry_ablation")
SOURCE_CHECKPOINT = Path("outputs/attrivision_ablation/A1/checkpoint_best.pth")
METRICS = (
    ("AUROC", "macro_auroc"), ("InstF1", "instance_f1"),
    ("BitErr", "mean_hamming_error"), ("Exact", "exact_match"),
    ("R1", "rank1"), ("R5", "rank5"), ("R10", "rank10"), ("mAP", "map"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def configure(args: Any) -> Path:
    spec = EXPERIMENTS[args.experiment]
    output_dir = ROOT_OUTPUT / args.experiment
    args.output_dir = str(output_dir)
    args.checkpoint = None
    args.init_checkpoint = str(SOURCE_CHECKPOINT)
    args.resume = None
    args.mode = "train"
    args.epochs = 20
    args.retrieval_interval = 5
    args.minimum_training_epochs = 20
    args.early_stopping_patience = 100

    # Reproduce the complete A1 formulation except for the declared objective.
    args.clip_model = "ViT-B-32-quickgelu"
    args.no_pretrained = True
    args.training_objective = "paper_fce"
    args.prompt_mode = "category_complete"
    args.text_sampling = "multi"
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.loss = "focal_clip"
    args.focal_alpha = 1.0
    args.focal_gamma = 2.0
    args.augmentation = "current"
    args.lambda_attr = 0.0
    args.use_fce = True
    args.unique_prompts = False
    args.freeze_binary_head = True

    args.category_ce_mode = spec["category_ce_mode"]
    args.category_ce_weight = 1.0
    args.category_temperature = 0.01
    args.attribute_temperature = 0.01
    args.validation_protocol = "native52_category_nll"
    args.training_log_filename = "training_log.csv"
    args.batch_diagnostics = False
    validate_args(args)
    return output_dir


def evaluate_epoch_zero(args: Any, output_dir: Path) -> dict[str, Any]:
    device = choose_device(args.device)
    model, payload = load_model(SOURCE_CHECKPOINT, device)
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("A1 source checkpoint must use category_complete prompts")
    metrics = evaluate_native52_category_nll(
        model, payload["attribute_names"], Path(args.data_root).resolve(),
        build_eval_transform(args.image_size), device, args.eval_batch_size,
        args.num_workers, args.amp, args.max_val_samples,
        attribute_temperature=0.01,
    )
    result = {"epoch": 0, "checkpoint": str(SOURCE_CHECKPOINT), **metrics}
    write_json(output_dir / "epoch0_metrics.json", result)
    print(
        f"{args.experiment} epoch 0: R1={100*metrics['rank1']:.2f}%, "
        f"mAP={100*metrics['map']:.2f}% (Native52 Category-NLL, T=0.01)",
        flush=True,
    )
    if args.max_val_samples is None and (
        abs(metrics["rank1"] - 0.197) > 0.02 or abs(metrics["map"] - 0.177) > 0.02
    ):
        raise RuntimeError(
            "Epoch-0 A1 Category-NLL baseline did not reproduce the expected "
            "R1≈19.7% and mAP≈17.7% within two percentage points"
        )
    return result


def evaluated_rows(output_dir: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with (output_dir / "training_log.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if not raw.get("rank1"):
                continue
            rows.append({key: float(value) for key, value in raw.items() if value != ""})
    if not rows:
        raise RuntimeError(f"No retrieval evaluation rows found in {output_dir / 'training_log.csv'}")
    return rows


def experiment_result(name: str) -> dict[str, Any] | None:
    output_dir = ROOT_OUTPUT / name
    epoch0_path = output_dir / "epoch0_metrics.json"
    log_path = output_dir / "training_log.csv"
    if not epoch0_path.is_file() or not log_path.is_file():
        return None
    with epoch0_path.open(encoding="utf-8") as handle:
        epoch0 = json.load(handle)
    rows = evaluated_rows(output_dir)
    best_r1 = max(rows, key=lambda row: (row["rank1"], row["map"], -row["epoch"]))
    best_map = max(rows, key=lambda row: (row["map"], row["rank1"], -row["epoch"]))
    return {
        "method": name,
        "objective": EXPERIMENTS[name]["objective"],
        "epoch0": epoch0,
        "best_r1": best_r1,
        "best_map": best_map,
        "best_r1_delta_from_epoch0": best_r1["rank1"] - epoch0["rank1"],
        "best_map_delta_from_epoch0": best_map["map"] - epoch0["map"],
    }


def print_and_save_summary() -> None:
    results = {name: experiment_result(name) for name in EXPERIMENTS}
    print("\nMethod | Epoch | AUROC | InstF1 | BitErr | Exact | R1 | R5 | R10 | mAP")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    table_rows: list[dict[str, Any]] = []
    for name in EXPERIMENTS:
        result = results[name]
        if result is None:
            print(f"{name} | - | - | - | - | - | - | - | - | -")
            continue
        row = result["best_map"]
        print(
            f"{name} | {int(row['epoch'])} | {row['macro_auroc']:.6f} | "
            f"{row['instance_f1']:.6f} | {row['mean_hamming_error']:.6f} | "
            f"{row['exact_match']:.6f} | {row['rank1']:.6f} | {row['rank5']:.6f} | "
            f"{row['rank10']:.6f} | {row['map']:.6f}"
        )
        table_rows.append({"Method": name, "Epoch": int(row["epoch"]), **{
            label: row[key] for label, key in METRICS
        }})
        best_r1 = result["best_r1"]
        best_map = result["best_map"]
        print(
            f"{name} best R1={best_r1['rank1']:.6f} at epoch {int(best_r1['epoch'])}, "
            f"delta(epoch0)={result['best_r1_delta_from_epoch0']:+.6f}"
        )
        print(
            f"{name} best mAP={best_map['map']:.6f} at epoch {int(best_map['epoch'])}, "
            f"delta(epoch0)={result['best_map_delta_from_epoch0']:+.6f}"
        )

    ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    if table_rows:
        with (ROOT_OUTPUT / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
    write_json(ROOT_OUTPUT / "summary.json", {
        key: value for key, value in results.items() if value is not None
    })


def main() -> None:
    parser = build_parser()
    parser.description = "D0--D2 Native52 Category-NLL geometry fine-tuning ablation"
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        print_and_save_summary()
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --summary-only is used")
    output_dir = configure(args)
    config = {
        **vars(args),
        "experiment_spec": EXPERIMENTS[args.experiment],
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "continuation_optimizer_state": "fresh (best checkpoint has no optimizer state)",
        "retrieval_protocol": "Native52 Category-NLL",
        "retrieval_temperature": 0.01,
        "binary_head_used": False,
        "new_head": False,
        "category_weighting": "uniform",
        "prior_correction": False,
    }
    write_json(output_dir / "config.json", config)
    epoch0 = evaluate_epoch_zero(args, output_dir)
    checkpoint = train(args)
    payload = _load(checkpoint)
    final = {
        "experiment": args.experiment,
        "objective": EXPERIMENTS[args.experiment]["objective"],
        "epoch0": epoch0,
        "selected_checkpoint": str(checkpoint),
        "selected_epoch": payload.get("epoch"),
        "selected_metrics": payload.get("metrics", {}),
    }
    write_json(output_dir / "validation_metrics.json", final)
    print_and_save_summary()


if __name__ == "__main__":
    main()
