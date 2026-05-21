#!/usr/bin/env python3
"""Post-fix Sheffield geocode rows whose current point is not on the polygon.

This is intentionally a zero-API pass.  It uses the existing Gemini/OS/Google
JSONL output and only revisits rows whose XLSX distance is non-zero or blank.
Rows with distance exactly zero are copied through unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


DEFAULT_QA_DIR = Path("/data/sheffield/spatial/QA")
DEFAULT_XLSX = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini.xlsx"
DEFAULT_JSONL = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini.jsonl"
DEFAULT_POLYGONS = DEFAULT_QA_DIR / "sheffieldwp6_27700.gpkg"
DEFAULT_LAYER = "sheffieldwp6"
DEFAULT_OUTPUT_XLSX = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini_distance_nonzero_optimized_v2.xlsx"
DEFAULT_OUTPUT_CSV = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini_distance_nonzero_optimized_v2.csv"
DEFAULT_CHANGED_CSV = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini_distance_nonzero_optimized_v2_changed.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini_distance_nonzero_optimized_v2.summary.json"

SHEFFIELD_WORK_BBOX = (423000.0, 379000.0, 445500.0, 400500.0)
DIRECTION_TOKENS = {"NORTH", "SOUTH", "EAST", "WEST"}

ROAD_SUFFIX_MAP = {
    "RD": "ROAD",
    "ROAD": "ROAD",
    "ST": "STREET",
    "STREET": "STREET",
    "AVE": "AVENUE",
    "AV": "AVENUE",
    "AVENUE": "AVENUE",
    "CL": "CLOSE",
    "CLOSE": "CLOSE",
    "LN": "LANE",
    "LANE": "LANE",
    "DR": "DRIVE",
    "DRIVE": "DRIVE",
    "PL": "PLACE",
    "PLACE": "PLACE",
    "CT": "COURT",
    "COURT": "COURT",
    "SQ": "SQUARE",
    "SQUARE": "SQUARE",
    "TER": "TERRACE",
    "TERRACE": "TERRACE",
    "CRES": "CRESCENT",
    "CRESCENT": "CRESCENT",
    "GDNS": "GARDENS",
    "GDN": "GARDENS",
    "GARDENS": "GARDENS",
    "GROVE": "GROVE",
    "WAY": "WAY",
    "GATE": "GATE",
    "HILL": "HILL",
    "VIEW": "VIEW",
    "ROW": "ROW",
    "RISE": "RISE",
    "MEWS": "MEWS",
    "WALK": "WALK",
    "PARK": "PARK",
    "CROFT": "CROFT",
    "GREEN": "GREEN",
    "FIELD": "FIELD",
    "FIELDS": "FIELDS",
}

ROAD_SUFFIX_RE = "|".join(re.escape(item) for item in sorted(ROAD_SUFFIX_MAP, key=len, reverse=True))
ROAD_RE = re.compile(
    r"\b([A-Z0-9][A-Z0-9'&.\-\s]+?\b(?:" + ROAD_SUFFIX_RE + r"))\b",
    re.IGNORECASE,
)
POSTCODE_DISTRICT_RE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\b", re.IGNORECASE)
LEADING_NUMBER_RE = re.compile(r"^\s*(\d+[A-Z]?)(?:\s*-\s*(\d+[A-Z]?))?\b", re.IGNORECASE)
COMPLEX_SITE_RE = re.compile(
    r"\b(site\s+location|land|plot|plots|rear|adjacent|adjoining|off|junction|corner|"
    r"former|forecourt|car\s+park|garage|premises)\b",
    re.IGNORECASE,
)


@dataclass
class ScoredCandidate:
    source: str
    address: str
    easting: float
    northing: float
    score: float
    hard_ok: bool
    reason: str
    road_similarity: float
    number_match: bool
    postcode_match: bool
    postcode_mismatch: bool


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" ,.")


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def tokens(value: Any) -> list[str]:
    text = re.sub(r"[^A-Z0-9]+", " ", clean_text(value).upper()).strip()
    return text.split() if text else []


def normalize_road_token(token: str) -> str:
    return ROAD_SUFFIX_MAP.get(token.upper(), token.upper())


def normalize_road(value: Any) -> str:
    return " ".join(normalize_road_token(token) for token in tokens(value))


def compact_road(value: Any) -> str:
    return "".join(normalize_road_token(token) for token in tokens(value))


def postcode_district(value: Any) -> str:
    match = POSTCODE_DISTRICT_RE.search(clean_text(value).upper())
    return match.group(1).upper().replace(" ", "") if match else ""


def leading_numbers(value: Any) -> list[str]:
    match = LEADING_NUMBER_RE.match(clean_text(value))
    if not match:
        return []
    first = match.group(1).upper()
    second = (match.group(2) or "").upper()
    if not second:
        return [first]
    try:
        start = int(re.match(r"\d+", first).group(0))  # type: ignore[union-attr]
        end = int(re.match(r"\d+", second).group(0))  # type: ignore[union-attr]
    except Exception:
        return [first, second]
    if 0 <= end - start <= 200:
        return [str(item) for item in range(start, end + 1)]
    return [first, second]


def is_complex_site(value: Any) -> bool:
    return bool(COMPLEX_SITE_RE.search(clean_text(value)))


def extract_roads(value: Any) -> list[str]:
    out: list[str] = []
    for match in ROAD_RE.finditer(clean_text(value).upper()):
        road = clean_text(match.group(1))
        road = re.sub(r"^\d+[A-Z]?(?:\s*-\s*\d+[A-Z]?)?\s+", "", road, flags=re.IGNORECASE)
        road = re.sub(
            r"^(?:SITE LOCATION|LAND|PLOT|PLOTS|REAR OF|ADJACENT TO|ADJ|FORMER|THE)\s+",
            "",
            road,
            flags=re.IGNORECASE,
        )
        if len(compact_road(road)) < 5:
            continue
        if compact_road(road) not in {compact_road(existing) for existing in out}:
            out.append(road)
    return out


def road_similarity(raw_address: str, candidate_road: str, candidate_address: str) -> float:
    best = 0.0
    candidate_roads = ([candidate_road] if candidate_road else []) + extract_roads(candidate_address)
    for raw_road in extract_roads(raw_address):
        raw_key = compact_road(raw_road)
        for cand_road in candidate_roads:
            cand_key = compact_road(cand_road)
            if not raw_key or not cand_key:
                continue
            if raw_key == cand_key:
                similarity = 1.0
            elif raw_key in cand_key or cand_key in raw_key:
                similarity = 0.94
            else:
                similarity = SequenceMatcher(None, normalize_road(raw_road), normalize_road(cand_road)).ratio()
            best = max(best, similarity)
    return best


def in_work_bbox(x: float | None, y: float | None, pad: float = 1500.0) -> bool:
    if x is None or y is None:
        return False
    minx, miny, maxx, maxy = SHEFFIELD_WORK_BBOX
    return minx - pad <= x <= maxx + pad and miny - pad <= y <= maxy + pad


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    try:
        data = json.loads(value or "[]")
    except Exception:
        return []
    return data if isinstance(data, list) else []


def address_cache_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", clean_text(value).upper()).strip()


def candidate_cache_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        re.sub(r"[^A-Z0-9]+", "", clean_text(candidate.get("address")).upper()),
        str(candidate.get("easting_27700") or ""),
        str(candidate.get("northing_27700") or ""),
    )


def has_added_direction_suffix(raw_address: str, candidate_address: str) -> bool:
    """Detect cases like raw Ecclesall Road vs candidate Ecclesall Road South."""

    candidate_norm = normalize_road(candidate_address)
    for raw_road in extract_roads(raw_address):
        raw_norm = normalize_road(raw_road)
        raw_tokens = raw_norm.split()
        if not raw_tokens or raw_tokens[-1] in DIRECTION_TOKENS:
            continue
        for direction in DIRECTION_TOKENS:
            if f"{raw_norm} {direction}" in candidate_norm:
                return True
    return False


def score_google_candidate(raw_address: str, candidate: dict[str, Any]) -> ScoredCandidate | None:
    address = clean_text(candidate.get("address"))
    road_name = clean_text(candidate.get("road_name"))
    place = clean_text(candidate.get("place"))
    easting = as_float(candidate.get("easting_27700"))
    northing = as_float(candidate.get("northing_27700"))
    if easting is None or northing is None:
        return None

    scope_text = " ".join([address, place]).upper()
    in_scope = in_work_bbox(easting, northing) and (
        "SHEFFIELD" in scope_text or bool(re.search(r"\bS\d{1,2}[A-Z]?\b", scope_text))
    )
    road_score = road_similarity(raw_address, road_name, address)
    raw_numbers = leading_numbers(raw_address)
    candidate_numbers = leading_numbers(address)
    number_match = bool(raw_numbers and candidate_numbers and set(raw_numbers).intersection(candidate_numbers))
    postcode_raw = postcode_district(raw_address)
    postcode_candidate = clean_text(candidate.get("postcode_district") or postcode_district(address)).upper()
    postcode_match = bool(postcode_raw and postcode_candidate and postcode_raw == postcode_candidate)
    postcode_mismatch = bool(postcode_raw and postcode_candidate and postcode_raw != postcode_candidate)
    complex_site = is_complex_site(raw_address)
    google_match = as_float(candidate.get("match")) or 0.0

    score = 0.0
    reasons: list[str] = []
    if in_scope:
        score += 25
        reasons.append("scope")
    else:
        score -= 80
        reasons.append("out_of_scope")

    if road_score >= 0.98:
        score += 35
        reasons.append("road_exact")
    elif road_score >= 0.88:
        score += 30
        reasons.append("road_strong")
    elif road_score >= 0.78:
        score += 18
        reasons.append("road_fuzzy")
    else:
        score -= 35
        reasons.append("road_weak")

    if raw_numbers:
        if number_match:
            score += 25
            reasons.append("number_match")
        elif complex_site:
            score -= 5
            reasons.append("site_number_not_required")
        else:
            score -= 45
            reasons.append("number_mismatch")

    if postcode_match:
        score += 12
        reasons.append("postcode_match")
    elif postcode_mismatch:
        score -= 8
        reasons.append("postcode_mismatch")

    if google_match >= 0.85:
        score += 8
        reasons.append("specific_google")
    elif google_match and google_match < 0.7:
        score -= 6
        reasons.append("road_level")

    hard_ok = in_scope and road_score >= 0.78
    if raw_numbers and not complex_site and not number_match:
        hard_ok = False
    if postcode_mismatch and road_score < 0.90:
        hard_ok = False

    return ScoredCandidate(
        source="gog",
        address=address,
        easting=easting,
        northing=northing,
        score=score,
        hard_ok=hard_ok,
        reason=",".join(reasons),
        road_similarity=road_score,
        number_match=number_match,
        postcode_match=postcode_match,
        postcode_mismatch=postcode_mismatch,
    )


def load_jsonl(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                clean_text(row.get("unique_key")),
                clean_text(row.get("variant_key")),
                clean_text(row.get("original_address")),
            )
            out[key] = row
    return out


def build_shared_google_candidates(rows: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_address: dict[str, list[dict[str, Any]]] = {}
    seen_by_address: dict[str, set[tuple[str, str, str]]] = {}
    for _, row in rows.items():
        key = address_cache_key(row.get("original_address"))
        if not key:
            continue
        seen = seen_by_address.setdefault(key, set())
        bucket = by_address.setdefault(key, [])
        for candidate in parse_json_list(row.get("gog_candidates_json")):
            candidate_key = candidate_cache_key(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            bucket.append(candidate)
    return by_address


def load_reference_geometries(path: Path, layer: str) -> dict[str, Any]:
    polygons = gpd.read_file(path, layer=layer).to_crs(27700)
    return {
        str(ref): unary_union(list(group.geometry))
        for ref, group in polygons.groupby(polygons["lafilerefe"].astype(str), sort=False)
    }


def distance_to_ref(ref_geoms: dict[str, Any], ref: str, x: float | None, y: float | None) -> float | None:
    if x is None or y is None:
        return None
    geom = ref_geoms.get(str(ref))
    if geom is None:
        return None
    return float(geom.distance(Point(float(x), float(y))))


def optimize_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sheet = pd.read_excel(args.input_xlsx, sheet_name="summary")
    json_rows = load_jsonl(args.input_jsonl)
    shared_google_candidates = build_shared_google_candidates(json_rows)
    ref_geoms = load_reference_geometries(args.polygons_gpkg, args.layer)

    distance = pd.to_numeric(sheet["best_distance_to_polygon_m_final"], errors="coerce")
    needs_fix = distance.isna() | (distance.abs() > float(args.zero_tolerance_m))
    rows_out: list[dict[str, Any]] = []

    for idx, row in sheet.iterrows():
        base = row.to_dict()
        key = (
            clean_text(row.get("unique_key")),
            clean_text(row.get("variant_key")),
            clean_text(row.get("original_address")),
        )
        raw = json_rows.get(key, {})
        current_x = as_float(raw.get("best_easting_27700_final"))
        current_y = as_float(raw.get("best_northing_27700_final"))
        current_source = clean_text(row.get("best_source_final")).lower()
        current_address = clean_text(row.get("best_address_final"))
        current_distance = distance.iloc[idx]

        chosen: ScoredCandidate | None = None
        current_has_added_direction = has_added_direction_suffix(
            clean_text(row.get("original_address")),
            current_address,
        )
        eligible_current_source = current_source in {"fallback", "none", ""}
        eligible_direction_fix = (
            bool(args.promote_os_direction_mismatch)
            and current_source == "os"
            and current_has_added_direction
        )
        if bool(needs_fix.iloc[idx]) and (eligible_current_source or eligible_direction_fix):
            raw_google_candidates = list(parse_json_list(raw.get("gog_candidates_json")))
            if args.reuse_shared_google_candidates:
                existing_keys = {candidate_cache_key(candidate) for candidate in raw_google_candidates}
                for candidate in shared_google_candidates.get(address_cache_key(row.get("original_address")), []):
                    if candidate_cache_key(candidate) in existing_keys:
                        continue
                    raw_google_candidates.append(candidate)
                    existing_keys.add(candidate_cache_key(candidate))
            candidates = [
                scored
                for candidate in raw_google_candidates
                if (scored := score_google_candidate(clean_text(row.get("original_address")), candidate))
                and scored.hard_ok
                and scored.score >= float(args.google_promote_score)
            ]
            candidates.sort(key=lambda item: item.score, reverse=True)
            chosen = candidates[0] if candidates else None
            if eligible_direction_fix and chosen is not None and chosen.score < float(args.os_google_promote_score):
                chosen = None

        optimized_source = current_source
        optimized_address = current_address
        optimized_x = current_x
        optimized_y = current_y
        optimized_score: float | None = None
        optimized_reason = "kept_current"
        changed = False

        if chosen is not None:
            optimized_source = chosen.source
            optimized_address = chosen.address
            optimized_x = chosen.easting
            optimized_y = chosen.northing
            optimized_score = chosen.score
            reason_prefix = "promoted_google"
            if eligible_direction_fix:
                reason_prefix = "promoted_google_over_os_direction_suffix"
            optimized_reason = f"{reason_prefix}:{chosen.reason}"
            changed = True

        optimized_distance = distance_to_ref(ref_geoms, clean_text(row.get("unique_key")), optimized_x, optimized_y)
        current_distance_float = as_float(current_distance)

        base.update(
            {
                "distance_nonzero_postfix_candidate_considered": bool(needs_fix.iloc[idx]),
                "optimized_changed": changed,
                "optimized_source_final": optimized_source,
                "optimized_address_final": optimized_address,
                "optimized_easting_27700": optimized_x,
                "optimized_northing_27700": optimized_y,
                "optimized_candidate_score": optimized_score,
                "optimized_reason": optimized_reason,
                "optimized_distance_to_polygon_m": optimized_distance,
                "optimized_distance_delta_m": (
                    optimized_distance - current_distance_float
                    if optimized_distance is not None and current_distance_float is not None
                    else None
                ),
            }
        )
        if chosen is not None:
            base.update(
                {
                    "optimized_road_similarity": chosen.road_similarity,
                    "optimized_number_match": chosen.number_match,
                    "optimized_postcode_match": chosen.postcode_match,
                "optimized_postcode_mismatch": chosen.postcode_mismatch,
                    "optimized_current_added_direction_suffix": current_has_added_direction,
                }
            )
        else:
            base.update(
                {
                    "optimized_road_similarity": None,
                    "optimized_number_match": None,
                    "optimized_postcode_match": None,
                    "optimized_postcode_mismatch": None,
                    "optimized_current_added_direction_suffix": current_has_added_direction,
                }
            )
        rows_out.append(base)

    out = pd.DataFrame(rows_out)
    changed = out[out["optimized_changed"]].copy()

    old_dist = pd.to_numeric(out["best_distance_to_polygon_m_final"], errors="coerce")
    new_dist = pd.to_numeric(out["optimized_distance_to_polygon_m"], errors="coerce")
    changed_old_dist = pd.to_numeric(changed["best_distance_to_polygon_m_final"], errors="coerce")
    changed_new_dist = pd.to_numeric(changed["optimized_distance_to_polygon_m"], errors="coerce")
    changed_delta = changed_new_dist - changed_old_dist
    summary = {
        "input_rows": int(len(out)),
        "distance_zero_rows_untouched": int((~needs_fix).sum()),
        "distance_nonzero_or_blank_rows_considered": int(needs_fix.sum()),
        "optimized_changed_rows": int(out["optimized_changed"].sum()),
        "changed_by_source": changed["best_source_final"].value_counts(dropna=False).to_dict(),
        "changed_distance_improved_rows": int((changed_delta < 0).sum()),
        "changed_distance_worsened_rows": int((changed_delta > 0).sum()),
        "changed_distance_old_blank_rows": int(changed_old_dist.isna().sum()),
        "changed_distance_new_blank_rows": int(changed_new_dist.isna().sum()),
        "old_distance_nonzero_count": int((old_dist.fillna(-1).abs() > float(args.zero_tolerance_m)).sum()),
        "new_distance_nonzero_count": int((new_dist.fillna(-1).abs() > float(args.zero_tolerance_m)).sum()),
        "old_distance_gt_25m": int((old_dist > 25).sum()),
        "new_distance_gt_25m": int((new_dist > 25).sum()),
        "old_distance_gt_100m": int((old_dist > 100).sum()),
        "new_distance_gt_100m": int((new_dist > 100).sum()),
        "old_distance_gt_500m": int((old_dist > 500).sum()),
        "new_distance_gt_500m": int((new_dist > 500).sum()),
        "old_distance_gt_5000m": int((old_dist > 5000).sum()),
        "new_distance_gt_5000m": int((new_dist > 5000).sum()),
    }
    return out, changed, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--polygons-gpkg", type=Path, default=DEFAULT_POLYGONS)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--changed-csv", type=Path, default=DEFAULT_CHANGED_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--zero-tolerance-m", type=float, default=0.01)
    parser.add_argument("--google-promote-score", type=float, default=85.0)
    parser.add_argument("--os-google-promote-score", type=float, default=95.0)
    parser.add_argument("--reuse-shared-google-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--promote-os-direction-mismatch", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out, changed, summary = optimize_rows(args)
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(args.output_xlsx, index=False)
    out.to_csv(args.output_csv, index=False)
    changed.to_csv(args.changed_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {args.output_xlsx}")
    print(f"written: {args.output_csv}")
    print(f"written: {args.changed_csv}")


if __name__ == "__main__":
    main()
