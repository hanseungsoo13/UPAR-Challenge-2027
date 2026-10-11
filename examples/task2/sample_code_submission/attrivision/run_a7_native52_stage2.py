"""Native52 stage-2 objectives initialized from the A7 best checkpoint."""
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
DEFAULT_OUTPUT_ROOT = Path("outputs/attrivision_native52_stage2")


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "N1_native52_only": {
        "description": "Stage-2 Native52 category CE only after A7 FCE pretraining",
        "category_ce_mode": "only",
        "category_ce_weight": 1.0,
    },
    "N2_fce_native52": {
        "description": "Stage-2 A7 FCE plus Native52 category CE auxiliary",
        "category_ce_mode": "combined",
        "category_ce_weight": 0.05,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
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


def build_parser_with_stage2_options():
    parser = build_parser()
    parser.description = "A7-best Native52 stage-2 objective experiments"
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--all", action="store_true", help="run N1 and N2 sequentially")
    parser.add_argument(
        "--a7-checkpoint", type=Path, default=DEFAULT_A7_CHECKPOINT,
        help="A7 best checkpoint used as the common initialization",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="root directory for per-experiment outputs",
    )
    parser.add_argument("--summary-only", action="store_true")
    # The base CLI default is 1.0.  None lets the experiment preset provide
    # N2's small auxiliary weight while still allowing an explicit override.
    parser.set_defaults(category_ce_weight=None)
    return parser


def _configure(args: Any, name: str) -> Path:
    spec = EXPERIMENTS[name]
    checkpoint = args.a7_checkpoint.resolve()
    if args.mode != "smoke" and not checkpoint.is_file():
        raise FileNotFoundError(f"A7 checkpoint does not exist: {checkpoint}")

    output_dir = args.output_root / name
    args.output_dir = str(output_dir)
    args.init_checkpoint = str(checkpoint)

    # Keep the A7 representation and prompt geometry unchanged.  Only the
    # stage-2 objective is changed here.
    args.training_objective = "paper_fce"
    args.prompt_mode = "category_complete"
    args.text_sampling = "single"
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = spec["category_ce_mode"]
    args.category_ce_weight = (
        spec["category_ce_weight"]
        if args.category_ce_weight is None else args.category_ce_weight
    )
    args.category_temperature = 0.01
    args.mixed_aux_prototype_weight = 0.0
    args.set_loss_weight = 1.0
    args.validation_protocol = "native52_category_nll"
    args.selection_metric = "mADM"
    args.retrieval_scoring = "paired_l1"
    args.paper_faithful = False
    args.learning_rate = 2e-6
    args.visual_learning_rate = None
    args.text_learning_rate = None
    args.freeze_text_encoder = False
    args.freeze_logit_scale = False
    args.trainable_vision_blocks = 0
    _write_json(output_dir / "stage2_config.json", {
        **vars(args),
        "experiment": name,
        "description": spec["description"],
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
        print("\nNative52 stage-2 summary (best validation mADM)")
        for name, row in summary.items():
            print(
                f"{name}: epoch={row['epoch']} mADM={100 * row['mADM']:.2f}% "
                f"mAP={100 * row['mAP']:.2f}% Rank-1={100 * row['Rank-1']:.2f}%"
            )


def main() -> None:
    args = build_parser_with_stage2_options().parse_args()
    if args.summary_only:
        _write_summary(args.output_root)
        return
    selectors = int(args.all) + int(args.experiment is not None)
    if selectors > 1:
        raise SystemExit("Choose only one of --experiment or --all")
    names = list(EXPERIMENTS) if args.all else ([args.experiment] if args.experiment else [])
    if not names:
        raise SystemExit("Specify --experiment NAME, --all, or --summary-only")
    for name in names:
        print(f"\n=== {name}: {EXPERIMENTS[name]['description']} ===", flush=True)
        _configure(args, name)
        run(args)
    _write_summary(args.output_root)


if __name__ == "__main__":
    main()
