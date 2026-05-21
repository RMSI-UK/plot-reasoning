#!/usr/bin/env python3
"""Clean CLI wrapper for direct no-AI address/OCR point geocoding."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from no_ai_candidate_boxes.address_point_geocoder import main


if __name__ == "__main__":
    main()
