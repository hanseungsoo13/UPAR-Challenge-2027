"""Challenge API and local CLI entry point for the UPAR Task 2 baseline.

Implementation details live in the sibling ``upar`` package. The three names
re-exported here are the stable API imported by the challenge server.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Some ingestion programs load run.py by file path without adding its folder to
# sys.path. Ensure the packaged sibling modules remain importable in that case.
SUBMISSION_DIR = Path(__file__).resolve().parent
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from upar.challenge import load_model, predict_attributes, rank_gallery  # noqa: E402,F401
from upar.cli import main  # noqa: E402

__all__ = ["load_model", "predict_attributes", "rank_gallery"]


if __name__ == "__main__":
    main()
