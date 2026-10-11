"""FCE-preserving feasibility studies initialized from the A7 best checkpoint."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser, run  # noqa: E402


DEFAULT_A7_CHECKPOINT = Path("outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth")
DEFAULT_OUTPUT_ROOT = Path("outputs/attrivision_fce_feasibility")


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "F0_low_lr": {
        "description": "A7 paper FCE continued from best with a conservative full-model LR",
        "training_objective": "paper_fce",
        "prompt_mode": "category_complete",
        "mixed_aux_prototype_weight": 0.0,
        "learning_rate": 2e-6,
        "freeze_text_encoder": False,
        "freeze_logit_scale": False,
        "trainable_vision_blocks": 0,
    },
    "F1_aux_010": {
        "description": "FCE plus the existing mixed CE/BCE auxiliary at weight 0.10",
        "training_objective": "a7_fce_mixed_aux",
        "prompt_mode": "mixed_category",
        "mixed_aux_prototype_weight": 0.10,
        "learning_rate": 2e-6,
        "freeze_text_encoder": False,
        "freeze_logit_scale": False,
        "trainable_vision_blocks": 0,
    },
    "F2_aux_025": {
        "description": "FCE plus the mixed CE/BCE auxiliary at weight 0.25",
        "training_objective": "a7_fce_mixed_aux",
        "prompt_mode": "mixed_category",
        "mixed_aux_prototype_weight": 0.25,
        "learning_rate": 2e-6,
        "freeze_text_encoder": False,
        "freeze_logit_scale": False,
        "trainable_vision_blocks": 0,
    },
    "F3_aux_text_frozen": {
        "description": "FCE plus auxiliary with frozen text tower and temperature",
        "training_objective": "a7_fce_mixed_aux",
        "prompt_mode": "mixed_category",
        "mixed_aux_prototype_weight": 0.10,
        "learning_rate": 2e-6,
        "freeze_text_encoder": True,
        "freeze_logit_scale": True,
        "trainable_vision_blocks": 0,
    },
    "F4_aux_last2": {
        "description": "FCE plus auxiliary with only the last two vision blocks trainable",
        "training_objective": "a7_fce_mixed_aux",
        "prompt_mode": "mixed_category",
        "mixed_aux_prototype_weight": 0.10,
        "learning_rate": 2e-6,
        "visual_learning_rate": 2e-6,
        "freeze_text_encoder": True,
        "freeze_logit_scale": True,
        "trainable_vision_blocks": 2,
    },
    "F5_fce_text_frozen": {
        "description": "Clean category-complete FCE with frozen text tower and temperature",
        "training_objective": "paper_fce",
        "prompt_mode": "category_complete",
        "mixed_aux_prototype_weight": 0.0,
        "learning_rate": 2e-6,
        "freeze_text_encoder": True,
        "freeze_logit_scale": True,
        "trainable_vision_blocks": 0,
    },
    "F6_fce_last2": {
        "description": "Clean category-complete FCE with only the last two vision blocks trainable",
        "training_objective": "paper_fce",
        "prompt_mode": "category_complete",
        "mixed_aux_prototype_weight": 0.0,
        "learning_rate": 2e-6,
        "visual_learning_rate": 2e-6,
        "freeze_text_encoder": True,
        "freeze_logit_scale": True,
        "trainable_vision_blocks": 2,
    },
    "F7_fce_ultralow_lr": {
        "description": "Clean category-complete FCE with a full-model ultra-low LR",
        "training_objective": "paper_fce",
        "prompt_mode": "category_complete",
        "mixed_aux_prototype_weight": 0.0,
        "learning_rate": 5e-7,
        "freeze_text_encoder": False,
        "freeze_logit_scale": False,
        "trainable_vision_blocks": 0,
    },
}

CLEAN_EXPERIMENTS = (
    "F5_fce_text_frozen",
    "F6_fce_last2",
    "F7_fce_ultralow_lr",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _best_row(output_dir: Path) -> dict[str, Any] | None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.is_file():
        return None
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("mADM", "") not in {"", "nan", "NaN"}]
    if not rows:
        return None
    row = max(rows, key=lambda value: float(value["mADM"]))
    return {
        "epoch": int(float(row["epoch"])),
        "mADM": float(row["mADM"]),
        "mAP": float(row.get("map", "nan")),
        "Rank-1": float(row.get("rank1", "nan")),
        "Rank-5": float(row.get("rank5", "nan")),
        "Rank-10": float(row.get("rank10", "nan")),
    }


def build_parser_with_suite_options():
    parser = build_parser()
    parser.description = "FCE-preserving AttriVision feasibility studies from an A7 checkpoint"
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--all", action="store_true", help="run every feasibility experiment")
    parser.add_argument("--clean", action="store_true", help="run only clean FCE/category-complete experiments")
    parser.add_argument(
        "--a7-checkpoint", type=Path, default=DEFAULT_A7_CHECKPOINT,
        help="A7 best checkpoint used as the common initialization",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="root directory for per-experiment outputs",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _configure(args: Any, name: str) -> Path:
    spec = EXPERIMENTS[name]
    checkpoint = args.a7_checkpoint.resolve()
    if args.mode != "smoke" and not checkpoint.is_file():
        raise FileNotFoundError(f"A7 checkpoint does not exist: {checkpoint}")
    output_dir = args.output_root / name
    args.output_dir = str(output_dir)
    args.init_checkpoint = str(checkpoint)
    args.training_objective = spec["training_objective"]
    args.prompt_mode = spec["prompt_mode"]
    args.text_sampling = "single"
    # Match the original A7 metadata.  With text_sampling=single this does
    # not change the sampled prompt count, but keeps the continuation config
    # faithful to the checkpoint's 3-state A7 setup.
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.unique_prompts = False
    args.mixed_aux_prototype_weight = spec["mixed_aux_prototype_weight"]
    # paper_fce never consumes set_loss_weight; retain A7's value in metadata
    # while keeping the mixed auxiliary variants' set loss explicitly disabled.
    args.set_loss_weight = 1.0 if spec["training_objective"] == "paper_fce" else 0.0
    args.validation_protocol = "native52_category_nll"
    args.category_temperature = 0.01
    args.selection_metric = "mADM"
    args.retrieval_scoring = "paired_l1"
    args.paper_faithful = False
    args.learning_rate = spec["learning_rate"]
    args.visual_learning_rate = spec.get("visual_learning_rate")
    args.freeze_text_encoder = spec["freeze_text_encoder"]
    args.freeze_logit_scale = spec["freeze_logit_scale"]
    args.trainable_vision_blocks = spec["trainable_vision_blocks"]
    _write_json(output_dir / "feasibility_config.json", {
        **vars(args), "experiment": name, "description": spec["description"],
        "a7_checkpoint": str(checkpoint),
    })
    return output_dir


def _write_summary(output_root: Path) -> None:
    summary = {}
    for name in EXPERIMENTS:
        row = _best_row(output_root / name)
        if row is not None:
            summary[name] = row
    _write_json(output_root / "summary.json", summary)
    if summary:
        print("\nFCE feasibility summary (best validation mADM)")
        for name, row in summary.items():
            print(
                f"{name}: epoch={row['epoch']} mADM={100 * row['mADM']:.2f}% "
                f"mAP={100 * row['mAP']:.2f}% Rank-1={100 * row['Rank-1']:.2f}%"
            )


def main() -> None:
    args = build_parser_with_suite_options().parse_args()
    if args.summary_only:
        _write_summary(args.output_root)
        return
    selectors = int(args.all) + int(args.clean) + int(args.experiment is not None)
    if selectors > 1:
        raise SystemExit("Choose only one of --experiment, --clean, or --all")
    if args.all:
        names = list(EXPERIMENTS)
    elif args.clean:
        names = list(CLEAN_EXPERIMENTS)
    else:
        names = [args.experiment] if args.experiment else []
    if not names:
        raise SystemExit("Specify --experiment NAME, --clean, --all, or --summary-only")
    for name in names:
        print(f"\n=== {name}: {EXPERIMENTS[name]['description']} ===", flush=True)
        _configure(args, name)
        run(args)
    _write_summary(args.output_root)


if __name__ == "__main__":
    main()
