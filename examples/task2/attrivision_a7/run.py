"""A7-only train, evaluation, smoke-test, and submission entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SUBMISSION_SOURCE = THIS_DIR.parent / "sample_code_submission"
DEFAULT_OUTPUT_DIR = Path("outputs/attrivision_a7")
DEFAULT_SUBMISSION = Path("submissions/attrivision_a7.zip")

# These values reproduce the validated A7 experiment. They are intentionally
# not exposed as CLI flags: this entry point must not silently become another
# AttriVision ablation.
A7_LOCKED_CONFIG: dict[str, Any] = {
    "task": "task2",
    "model": "attrivision",
    "clip_model": "ViT-B-32-quickgelu",
    "pretrained_tag": "openai",
    "image_size": 224,
    "prompt_mode": "category_complete",
    "training_objective": "paper_fce",
    "text_sampling": "single",
    "multi_attributes": 3,
    "contrastive_target": "multi_positive",
    "augmentation": "resize_pad_crop",
    "loss": "focal_clip",
    "use_fce": True,
    "unique_prompts": False,
    "lambda_attr": 0.0,
    "category_ce_mode": "off",
    "validation_protocol": "native52_category_nll",
    "category_temperature": 0.01,
    "selection_metric": "mADM",
    "retrieval_scoring": "paired_l1",
    "paper_faithful": False,
    "init_checkpoint": None,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AttriVision A7-only workflow: train, evaluate, smoke-test, or build "
            "a Codabench submission"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_eval", "smoke", "package"),
        default="train_eval",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint", help="checkpoint for eval or package mode")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__LAST__",
        metavar="CHECKPOINT",
        help="resume training; without a path, use OUTPUT_DIR/checkpoint_last.pth",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--minimum-training-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--retrieval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="skip pretrained download (intended for smoke mode only)",
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=DEFAULT_SUBMISSION,
        help="output ZIP used by package mode",
    )
    return parser


def _ensure_import_path() -> None:
    source = str(SUBMISSION_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)


def resolved_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Merge the focused CLI with shared defaults, then lock the A7 recipe."""
    if args.no_pretrained and args.mode != "smoke":
        raise ValueError("--no-pretrained is only allowed in smoke mode for the A7 workflow")
    _ensure_import_path()
    from attrivision.cli import build_parser as build_shared_parser

    runtime = build_shared_parser().parse_args([])
    for key, value in vars(args).items():
        if key != "submission_output":
            setattr(runtime, key, value)
    for key, value in A7_LOCKED_CONFIG.items():
        setattr(runtime, key, value)
    return runtime


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _save_resolved_config(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    _write_json(
        output_dir / "config.json",
        {
            **vars(args),
            "pipeline": "attrivision_a7",
            "locked_config": A7_LOCKED_CONFIG,
            "retrieval_protocol": "Native52 Category-NLL",
        },
    )


def run_experiment(args: argparse.Namespace) -> None:
    _ensure_import_path()
    from attrivision.checkpoint import load_model
    from attrivision.cli import run_smoke, validate_args
    from attrivision.engine.evaluator_abpr import (
        evaluate_native52_category_nll,
        print_evaluation,
    )
    from attrivision.engine.trainer_attrivision import train
    from attrivision.transforms import build_eval_transform
    from upar.config import choose_device

    validate_args(args)
    _save_resolved_config(args)
    if args.mode == "smoke":
        run_smoke(args)
        print("AttriVision A7 locked-path smoke test passed")
        return

    checkpoint = train(args) if args.mode in {"train", "train_eval"} else None
    if args.mode == "train":
        return
    checkpoint = Path(args.checkpoint) if args.checkpoint else checkpoint
    if checkpoint is None:
        checkpoint = Path(args.output_dir) / "checkpoint_best.pth"

    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    metrics = evaluate_native52_category_nll(
        model,
        payload["attribute_names"],
        Path(args.data_root).resolve(),
        build_eval_transform(args.image_size, args.augmentation),
        device,
        args.eval_batch_size,
        args.num_workers,
        args.amp,
        args.max_val_samples,
        args.category_temperature,
    )
    checkpoint_metrics = payload.get("metrics", {})
    metrics.update(
        {
            "fce_loss": checkpoint_metrics.get("fce_loss"),
            "i2t_loss": checkpoint_metrics.get("i2t_loss"),
            "t2i_loss": checkpoint_metrics.get("t2i_loss"),
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
        }
    )
    _write_json(Path(args.output_dir) / "validation_metrics.json", metrics)
    print_evaluation(metrics, str(checkpoint))


def build_submission(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in package mode")
    if args.submission_output.suffix.lower() != ".zip":
        raise ValueError("--submission-output must end in .zip")
    _ensure_import_path()
    from package_submission import build_archive

    build_archive(
        Path(args.checkpoint).resolve(),
        args.submission_output.resolve(),
        model="attrivision_a7",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "package":
        build_submission(args)
        return
    run_experiment(resolved_runtime_args(args))


if __name__ == "__main__":
    main()
