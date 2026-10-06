"""Executable entry point for the isolated AttriVision baseline."""
from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
