#!/usr/bin/env python3
"""Clean no-AI entrypoint for Mansfield candidate-box generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "no_ai_candidate_boxes" / "pipeline.py"


def main() -> None:
    subprocess.run([sys.executable, str(PIPELINE), *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
