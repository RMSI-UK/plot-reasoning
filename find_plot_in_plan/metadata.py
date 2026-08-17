"""Join panel images to the council's case record.

A panel's filename encodes the case folder before the first underscore:
    93-00708-P_001-001_0001_p5.jpg  ->  93-00708-P

Getting this wrong is easy and quiet: matching on the first eight characters (93-00708) misses
every folder carrying a suffix like -P, -FHN or -P-90-00980-LB, which is most of them. That
mistake looked like "only 6 of 20 cases have records" when in fact all 20 did.
"""
from __future__ import annotations

import csv
from pathlib import Path

from locate import CaseRecord

# Braintree's wp7 join table. Columns used:
#   candidate_folder                  the scan folder, matching the panel stem's prefix
#   source_reference                  the planning reference, e.g. 93/00708/FUL
#   source_address                    the property address
#   source_supplementary_information  the proposal text, e.g. "Erection of extension to dwelling"
DEFAULT_CSV = Path("/data/braintree/file-matching/wp7_auto_address_date.csv")


def folder_of(stem: str) -> str:
    """The case folder a panel belongs to."""
    return stem.split("_")[0]


def load_records(csv_path: Path | None = None,
                 wanted: set[str] | None = None) -> dict[str, CaseRecord]:
    """Map case folder -> CaseRecord. Pass `wanted` to avoid holding all 20k rows.

    The first row for a folder wins; the table has one row per document, so a case appears
    many times with the same case-level fields.
    """
    path = Path(csv_path or DEFAULT_CSV)
    if not path.exists():
        return {}
    out: dict[str, CaseRecord] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            folder = (row.get("candidate_folder") or "").strip()
            if not folder or folder in out:
                continue
            if wanted is not None and folder not in wanted:
                continue
            out[folder] = CaseRecord(
                reference=(row.get("source_reference") or "").strip(),
                address=(row.get("source_address") or "").strip(),
                proposal=(row.get("source_supplementary_information") or "").strip(),
            )
    return out


def records_for(stems: list[str], csv_path: Path | None = None) -> dict[str, CaseRecord]:
    """Case record per panel stem, for the stems given. Missing stems are simply absent."""
    by_folder = load_records(csv_path, wanted={folder_of(s) for s in stems})
    return {s: by_folder[folder_of(s)] for s in stems if folder_of(s) in by_folder}
