"""Train/evaluate the mixed categorical/multi-label A7-mixed experiment."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser, run  # noqa: E402


DEFAULT_OUTPUT = "outputs/attrivision_ablation_mixed/A7-mixed"


def main() -> None:
    parser = build_parser()
    parser.description = "AttriVision A7-mixed: 50-state mixed categorical/multi-label training"
    args = parser.parse_args()

    # Preserve the old A7 implementation and checkpoint.  This wrapper fixes
    # the experiment definition so a command cannot accidentally train the
    # legacy 52-state objective under an A7-mixed output directory.
    if args.output_dir == "outputs/attrivision":
        args.output_dir = DEFAULT_OUTPUT
    args.prompt_mode = "mixed_category"
    args.training_objective = "a7_mixed"
    args.text_sampling = "single"
    args.multi_attributes = 1
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.mixed_set_positive = "exact"
    args.mixed_min_shared_categories = 8
    args.mixed_set_beta = 4.0
    args.validation_protocol = "mixed_state_nll"
    args.category_temperature = 0.01
    args.selection_metric = "mADM"
    args.retrieval_scoring = "cosine_set"
    args.paper_faithful = False

    # Keep output paths explicit in logs/checkpoints even when this wrapper is
    # invoked from a different working directory.
    args.output_dir = str(Path(args.output_dir))
    run(args)


if __name__ == "__main__":
    main()
