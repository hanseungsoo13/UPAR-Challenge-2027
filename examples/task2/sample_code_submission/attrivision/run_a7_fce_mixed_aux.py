"""Train/evaluate A7-FCE-50 with a mixed prototype CE/BCE auxiliary loss."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser, run  # noqa: E402


DEFAULT_OUTPUT = "outputs/attrivision_ablation_mixed/A7-FCE-50-aux"


def main() -> None:
    parser = build_parser()
    parser.description = (
        "AttriVision A7-FCE-50-aux: A7 FCE plus a 0.1 mixed prototype CE/BCE auxiliary loss"
    )
    args = parser.parse_args()

    if args.output_dir == "outputs/attrivision":
        args.output_dir = DEFAULT_OUTPUT

    # Keep A7-FCE-50 as the main objective and add only the mixed semantic
    # prototype supervision.  The mixed set-level loss is intentionally off.
    args.prompt_mode = "mixed_category"
    args.training_objective = "a7_fce_mixed_aux"
    args.text_sampling = "single"
    args.multi_attributes = 1
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.unique_prompts = False
    args.set_loss_weight = 0.0
    args.validation_protocol = "native52_category_nll"
    args.category_temperature = 0.01
    args.selection_metric = "mADM"
    args.retrieval_scoring = "paired_l1"
    args.paper_faithful = False

    run(args)


if __name__ == "__main__":
    main()
