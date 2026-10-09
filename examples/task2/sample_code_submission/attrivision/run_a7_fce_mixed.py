"""Train/evaluate A7-FCE-50: mixed prompts with the original A7 FCE signal."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser, run  # noqa: E402


DEFAULT_OUTPUT = "outputs/attrivision_ablation_mixed/A7-FCE-50"


def main() -> None:
    parser = build_parser()
    parser.description = (
        "AttriVision A7-FCE-50: mixed 50-state prompts with sampled multi-positive FCE"
    )
    args = parser.parse_args()

    if args.output_dir == "outputs/attrivision":
        args.output_dir = DEFAULT_OUTPUT

    # Keep the mixed 50-state ontology, but restore the original A7 training
    # signal.  No mixed prototype CE/BCE or set-level hybrid term is added in
    # this diagnostic experiment.
    args.prompt_mode = "mixed_category"
    args.training_objective = "paper_fce"
    args.text_sampling = "single"
    args.multi_attributes = 1
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.unique_prompts = False
    args.validation_protocol = "native52_category_nll"
    args.category_temperature = 0.01
    args.selection_metric = "mADM"
    args.retrieval_scoring = "paired_l1"
    args.paper_faithful = False

    run(args)


if __name__ == "__main__":
    main()
