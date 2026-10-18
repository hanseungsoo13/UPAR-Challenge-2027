"""Four controlled A7 augmentation runs with Market1501 held out."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser, validate_args  # noqa: E402
from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.engine.evaluator_abpr import (  # noqa: E402
    evaluate_native52_category_nll_domainwise,
)
from attrivision.engine.trainer_attrivision import train  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from run_dg_ablation import _write_json, print_summary  # noqa: E402
from upar.config import choose_device  # noqa: E402


EXPERIMENTS = {
    "Aug0": ("baseline", 0.0),
    "Aug1": ("sequential", 0.0),
    "Aug2": ("augmix", 0.0),
    "Aug3": ("augmix_js", 0.1),
}
DEFAULT_OUTPUT_ROOT = Path("outputs/augmix_a7_market1501")


def build_experiment_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = (
        "One A7 augmentation experiment with Market1501 excluded from training"
    )
    parser.set_defaults(epochs=10, unseen_domain="Market1501")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--augmix-alpha", type=float, default=1.0)
    return parser


def configure(args: argparse.Namespace) -> None:
    if args.augmix_alpha <= 0:
        raise ValueError("--augmix-alpha must be positive")
    policy, consistency_weight = EXPERIMENTS[args.experiment]
    args.output_root = str(Path(args.output_root))
    args.output_dir = str(Path(args.output_root) / args.experiment)
    args.mode = "train_eval"

    # Keep the validated B/32 A7 recipe fixed across all four runs.
    args.task = "task2"
    args.model = "attrivision"
    args.clip_model = "ViT-B-32-quickgelu"
    args.pretrained_tag = "openai"
    args.no_pretrained = False
    args.init_checkpoint = None
    args.resume = None
    args.image_size = 224
    args.augmentation = "resize_pad_crop"
    args.rotation = 10.0
    args.hard_augmentation = False
    args.augmentation_policy = policy
    args.augmix_consistency_weight = consistency_weight
    # Keep the B/32 batch size fixed while reducing saved transformer
    # activations, especially for Aug3's three differentiable image views.
    args.activation_checkpointing = True
    args.prompt_mode = "category_complete"
    args.training_objective = "paper_fce"
    args.text_sampling = "single"
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.loss = "focal_clip"
    args.use_fce = True
    args.unique_prompts = False
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.category_temperature = 0.01
    args.validation_protocol = "native52_category_nll"
    args.retrieval_scoring = "paired_l1"
    args.retrieval_interval = 1
    # Match the existing DG ablation protocol: validate every epoch and choose
    # the checkpoint using the full validation mADM. Report Market1501's
    # domain-specific score alongside it for a direct comparison.
    args.selection_metric = "mADM"
    args.paper_faithful = False
    args.domain_wise_validation = True
    args.dg_method = "none"
    args.distill_weight = 0.0
    args.rsc_drop_ratio = 0.0
    args.swad_start_epoch = 3
    validate_args(args)


def main() -> None:
    args = build_experiment_parser().parse_args()
    configure(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "config.json", vars(args))
    print(
        f"Starting {args.experiment}: policy={args.augmentation_policy}, "
        f"unseen={args.unseen_domain}, selection={args.selection_metric}",
        flush=True,
    )
    checkpoint = train(args)
    device = choose_device(args.device)
    model, payload = load_model(Path(checkpoint), device)
    metrics = evaluate_native52_category_nll_domainwise(
        model, payload["attribute_names"], Path(args.data_root).resolve(),
        build_eval_transform(args.image_size, args.augmentation), device,
        args.eval_batch_size, args.num_workers, args.amp, args.max_val_samples,
        args.category_temperature, unseen_domain=args.unseen_domain,
    )
    metrics.update({
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "experiment": args.experiment,
        "augmentation_policy": args.augmentation_policy,
        "unseen_domain": args.unseen_domain,
        "selection_metric": args.selection_metric,
    })
    _write_json(output_dir / "validation_metrics.json", metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()
