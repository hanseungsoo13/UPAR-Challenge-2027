"""CLI entry point for AttrQuery-AttributeAdapter training."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_attribute_adapter import train_main  # noqa: E402


if __name__ == "__main__":
    train_main()
