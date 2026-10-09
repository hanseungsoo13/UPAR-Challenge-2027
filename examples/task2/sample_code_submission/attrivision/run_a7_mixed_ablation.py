"""Run controlled A7-mixed loss ablations."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser, run  # noqa: E402


DEFAULT_OUTPUT_ROOT = "outputs/attrivision_ablation_mixed"


EXPERIMENTS = {
    # Reproduces the current A7-mixed weighting, with the category-balanced
    # mixed-state evaluator used by this ablation suite.
    "M0": {
        "prototype_weight": 0.25,
        "set_weight": 1.0,
        "set_positive": "exact",
        "min_shared_categories": 8,
    },
    # Direct category supervision only.  This isolates whether the exact-set
    # contrastive term is responsible for the early retrieval drop.
    "M1": {
        "prototype_weight": 1.0,
        "set_weight": 0.0,
        "set_positive": "exact",
        "min_shared_categories": 8,
    },
    # Keep set contrastive as a regularizer, but make CE/BCE the primary term.
    "M2": {
        "prototype_weight": 1.0,
        "set_weight": 0.25,
        "set_positive": "exact",
        "min_shared_categories": 8,
    },
    # Test whether a less sparse positive mask helps.  A pair is positive when
    # it shares at least eight of the twelve semantic categories.
    "M3": {
        "prototype_weight": 1.0,
        "set_weight": 0.25,
        "set_positive": "category_overlap",
        "min_shared_categories": 8,
    },
}


def main() -> None:
    parser = build_parser()
    parser.description = "AttriVision A7-mixed controlled loss ablations"
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="M0")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    spec = EXPERIMENTS[args.experiment]
    args.output_dir = str(Path(args.output_root) / args.experiment)
    args.prompt_mode = "mixed_category"
    args.training_objective = "a7_mixed"
    args.text_sampling = "single"
    args.multi_attributes = 1
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.validation_protocol = "mixed_state_nll"
    args.category_temperature = 0.01
    args.selection_metric = "mADM"
    args.retrieval_scoring = "cosine_set"
    args.paper_faithful = False
    args.prototype_loss_weight = spec["prototype_weight"]
    args.set_loss_weight = spec["set_weight"]
    args.mixed_set_positive = spec["set_positive"]
    args.mixed_min_shared_categories = spec["min_shared_categories"]
    args.mixed_ablation = args.experiment

    run(args)


if __name__ == "__main__":
    main()
