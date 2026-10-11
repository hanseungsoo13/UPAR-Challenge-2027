"""Run the A7 domain-generalization screening experiments E0--E4.

E0 evaluates the preserved A7 best checkpoint.  E1--E4 start from the same
OpenAI CLIP initialization and keep A7's data, sampler, FCE objective, prompt
and Native52 Category-NLL scoring.  Training-time validation uses the complete
gallery and writes domain metrics to ``training_log.csv`` every epoch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.cli import build_parser, validate_args  # noqa: E402
from attrivision.engine.evaluator_abpr import (  # noqa: E402
    evaluate_native52_category_nll_domainwise,
)
from attrivision.engine.trainer_attrivision import train  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import choose_device  # noqa: E402


ROOT_OUTPUT = Path("outputs/dg_ablation_a7")
A7_CHECKPOINT = Path("outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth")
EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def configure(args: argparse.Namespace) -> None:
    experiment = args.experiment
    args.output_dir = str(Path(args.output_root) / experiment)
    args.mode = "eval" if experiment == "E0" else "train_eval"
    args.epochs = 10
    args.clip_model = "ViT-B-32-quickgelu"
    args.pretrained_tag = "openai"
    args.no_pretrained = False
    args.init_checkpoint = None
    args.resume = None
    args.image_size = 224
    args.prompt_mode = "category_complete"
    args.training_objective = "paper_fce"
    args.text_sampling = "single"
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.loss = "focal_clip"
    args.use_fce = True
    args.unique_prompts = False
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.category_temperature = 0.01
    args.validation_protocol = "native52_category_nll"
    args.retrieval_scoring = "paired_l1"
    args.selection_metric = "mADM"
    args.paper_faithful = False
    args.domain_wise_validation = True
    args.training_log_filename = "training_log.csv"
    args.distill_weight = 0.0
    args.rsc_drop_ratio = 0.0
    args.swad_start_epoch = 3

    if experiment == "E0":
        args.checkpoint = str(Path(args.checkpoint) if args.checkpoint else A7_CHECKPOINT)
        args.dg_method = "none"
    elif experiment == "E1":
        args.dg_method = "miro"
        args.distill_weight = 0.1
    elif experiment == "E2":
        args.dg_method = "partial"
        args.trainable_vision_blocks = 6
    elif experiment == "E3":
        args.dg_method = "swad"
        args.swad_start_epoch = 3
    elif experiment == "E4":
        args.dg_method = "rsc"
        args.rsc_drop_ratio = 0.25
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    validate_args(args)


def evaluate_and_save(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    metrics = evaluate_native52_category_nll_domainwise(
        model, payload["attribute_names"], Path(args.data_root).resolve(),
        build_eval_transform(args.image_size, args.augmentation), device,
        args.eval_batch_size, args.num_workers, args.amp, args.max_val_samples,
        args.category_temperature,
    )
    metrics.update({
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "experiment": args.experiment,
        "dg_method": args.dg_method,
        "a7_reference_checkpoint": str(A7_CHECKPOINT),
    })
    _write_json(Path(args.output_dir) / "validation_metrics.json", metrics)
    return metrics


def print_summary(metrics: dict[str, Any]) -> None:
    domains = ("Market1501", "PA100k", "PETA")
    print("\nDomain-wise validation (full gallery shared by every row)")
    print(f"{'domain':<12} {'queries':>8} {'R1':>10} {'mADM':>10}")
    print(f"{'Overall':<12} {int(metrics['queries']):>8} "
          f"{100 * metrics['rank1']:>9.4f}% {100 * metrics['mADM']:>9.4f}%")
    for domain in domains:
        prefix = f"domain_{domain}_"
        print(f"{domain:<12} {int(metrics[prefix + 'queries']):>8} "
              f"{100 * metrics[prefix + 'rank1']:>9.4f}% "
              f"{100 * metrics[prefix + 'mADM']:>9.4f}%")
    print(f"{'Macro':<12} {'-':>8} {100 * metrics['domain_macro_rank1']:>9.4f}% "
          f"{100 * metrics['domain_macro_mADM']:>9.4f}%")
    print(f"Worst-domain mADM: {metrics['worst_domain']} / "
          f"{100 * metrics['worst_domain_mADM']:.4f}%")


def main() -> None:
    parser = build_parser()
    parser.description = "A7 Domain Generalization screening experiments E0--E4"
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output-root", default=str(ROOT_OUTPUT))
    args = parser.parse_args()
    configure(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "config.json", vars(args))
    checkpoint = train(args) if args.mode == "train_eval" else Path(args.checkpoint)
    metrics = evaluate_and_save(args, Path(checkpoint))
    print_summary(metrics)


if __name__ == "__main__":
    main()
