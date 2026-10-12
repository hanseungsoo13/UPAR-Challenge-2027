"""CLI wrapper for BackboneUpgrade validation/evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[2]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.backbone_upgrade.pipeline import eval_main  # noqa: E402


if __name__ == "__main__":
    eval_main()
