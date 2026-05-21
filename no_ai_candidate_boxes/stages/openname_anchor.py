#!/usr/bin/env python3
"""Experiment: use OCR text with OS Open Names as non-geocode anchors.

This runs after the OCR Geo Code experiment.  It treats OCR location/proposal
text as structured evidence and creates extra candidate points from OS Open
Names roads and populated places, then applies conservative gates to avoid
overriding good address-level geocodes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox, parse_names


ROOT = Path(__file__).resolve().parent
PREV_SCRIPT = ROOT / "ocr_candidate_rerank.py"
spec = importlib.util.spec_from_file_location("ocr_rerank_v2", PREV_SCRIPT)
ocr_v2 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["ocr_rerank_v2"] = ocr_v2
spec.loader.exec_module(ocr_v2)


DEFAULT_PREV_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_rerank_v2.csv"
)
DEFAULT_INPUT_JSON = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_gemini.json"
)
DEFAULT_INPUT_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42.gpkg"
)
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_openname_v3"
)

MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)
LOW_TRUST_SOURCES = {"lexicon", "fallback", "none", "parent_consensus"}
ROAD_SUFFIX = r"(?:ROAD|STREET|LANE|AVENUE|CLOSE|WAY|DRIVE|GROVE|CRESCENT|HILL|GATE|SIDE|WALK|PLACE|SQUARE|COURT|PARK)"
GENERIC_PLACES = {
    "MANSFIELD",
    "MANSFIELD WOODHOUSE",
    "WARSOP",
    "MARKET WARSOP",
    "FOREST TOWN",
    "NOTTINGHAM",
}
COUNTY_NAMES = {"NOTTINGHAMSHIRE"}
DISTRICT_NAMES = {"MANSFIELD"}


def norm(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def in_bbox(easting: float, northing: float) -> bool:
    minx, miny, maxx, maxy = MANSFIELD_BBOX
    return minx <= easting <= maxx and miny <= northing <= maxy


def distance_to_geom(easting: Any, northing: Any, geom: Any) -> float | None:
    e = parse_float(easting)
    n = parse_float(northing)
    if e is None or n is None:
        return None
    return Point(e, n).distance(geom)


@dataclass
class OpenNameCandidate:
    source: str
    name: str
    local_type: str
    easting: float
    northing: float
    score: float
    reason: str


def load_truth(gpkg_path: Path, layer: str) -> dict[str, Any]:
    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    return {str(row["unique_key"]): row.geometry for _, row in gdf.iterrows()}


def exact_openname_query(con: sqlite3.Connection, name: str) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT NAME1, TYPE, LOCAL_TYPE, GEOMETRY_X, GEOMETRY_Y, MBR_XMIN, MBR_YMIN, MBR_XMAX, MBR_YMAX,
               POPULATED_PLACE, DISTRICT_BOROUGH, COUNTY_UNITARY
        FROM open_names
        WHERE NAME1 = ? COLLATE NOCASE
          AND CAST(GEOMETRY_X AS REAL) BETWEEN ? AND ?
          AND CAST(GEOMETRY_Y AS REAL) BETWEEN ? AND ?
        """,
        (name, MANSFIELD_BBOX[0], MANSFIELD_BBOX[2], MANSFIELD_BBOX[1], MANSFIELD_BBOX[3]),
    ).fetchall()
    cols = [
        "name",
        "type",
        "local_type",
        "x",
        "y",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "populated_place",
        "district",
        "county",
    ]
    return [dict(zip(cols, row)) for row in rows]


def like_openname_query(con: sqlite3.Connection, phrase: str, limit: int = 20) -> list[dict[str, Any]]:
    # Only for substantial named-site phrases; exact road matching stays exact.
    words = [w for w in norm(phrase).split() if len(w) >= 4 and w not in {"THE", "LAND", "SITE"}]
    if not words:
        return []
    like = "%" + "%".join(words[:4]) + "%"
    rows = con.execute(
        """
        SELECT NAME1, TYPE, LOCAL_TYPE, GEOMETRY_X, GEOMETRY_Y, MBR_XMIN, MBR_YMIN, MBR_XMAX, MBR_YMAX,
               POPULATED_PLACE, DISTRICT_BOROUGH, COUNTY_UNITARY
        FROM open_names
        WHERE NAME1 LIKE ? COLLATE NOCASE
          AND CAST(GEOMETRY_X AS REAL) BETWEEN ? AND ?
          AND CAST(GEOMETRY_Y AS REAL) BETWEEN ? AND ?
        LIMIT ?
        """,
        (like, MANSFIELD_BBOX[0], MANSFIELD_BBOX[2], MANSFIELD_BBOX[1], MANSFIELD_BBOX[3], limit),
    ).fetchall()
    cols = [
        "name",
        "type",
        "local_type",
        "x",
        "y",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "populated_place",
        "district",
        "county",
    ]
    return [dict(zip(cols, row)) for row in rows]


def load_local_place_names(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT NAME1
        FROM open_names
        WHERE TYPE = 'populatedPlace'
          AND CAST(GEOMETRY_X AS REAL) BETWEEN ? AND ?
          AND CAST(GEOMETRY_Y AS REAL) BETWEEN ? AND ?
        """,
        (MANSFIELD_BBOX[0], MANSFIELD_BBOX[2], MANSFIELD_BBOX[1], MANSFIELD_BBOX[3]),
    ).fetchall()
    names = sorted({norm(r[0]) for r in rows if r and r[0]}, key=len, reverse=True)
    return names


def extract_road_phrases(text: str) -> list[str]:
    text = norm(text)
    pattern = re.compile(r"\b([A-Z0-9][A-Z0-9 '&.-]{1,45}?\s+" + ROAD_SUFFIX + r")\b")
    out = []
    for match in pattern.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if len(phrase) >= 6 and phrase not in out:
            out.append(phrase)
    return out[:20]


def extract_localities(text: str, known_places: list[str]) -> list[str]:
    ntext = norm(text)
    found = []
    for place in known_places:
        if len(place) >= 5 and re.search(r"\b" + re.escape(place) + r"\b", ntext):
            found.append(place)
    return found[:12]


def extract_named_site_phrases(original_address: str) -> list[str]:
    # Phrase before first comma is often a named premises: Granada Social Club,
    # Field Mill, Town Hall, The Business Park.
    first = norm(original_address.split(",", 1)[0])
    first = re.sub(r"^(LAND|SITE|PLOT|UNIT|UNITS)\s+(AT|ADJACENT TO|ADJACENT|OFF|REAR OF|OF)\s+", "", first)
    if len(first) >= 8 and not re.search(ROAD_SUFFIX + r"$", first):
        return [first]
    return []


def nearest_pool_distance(easting: float, northing: float, raw_row: dict[str, Any]) -> float | None:
    pool = ocr_v2.extract_candidate_pool(raw_row)
    distances = [math.hypot(easting - c.easting, northing - c.northing) for c in pool]
    return min(distances) if distances else None


def candidate_from_openname(
    rec: dict[str, Any],
    source: str,
    reason_bits: list[str],
    raw_row: dict[str, Any],
    localities: list[str],
) -> OpenNameCandidate | None:
    e = parse_float(rec.get("x"))
    n = parse_float(rec.get("y"))
    if e is None or n is None or not in_bbox(e, n):
        return None
    name = str(rec.get("name") or "")
    local_type = str(rec.get("local_type") or "")
    pop = norm(rec.get("populated_place"))
    district = norm(rec.get("district"))
    county = norm(rec.get("county"))

    nearest = nearest_pool_distance(e, n, raw_row)
    score = 10.0
    reasons = list(reason_bits)
    if county in COUNTY_NAMES:
        score += 10
        reasons.append("county_match")
    if district in DISTRICT_NAMES:
        score += 8
        reasons.append("district_match")
    if localities and any(loc in {pop, district, county} or loc == norm(name) for loc in localities):
        score += 20
        reasons.append("locality_match")
    if nearest is not None:
        if nearest <= 300:
            score += 10
            reasons.append("near_candidate_pool")
        elif nearest > 2000:
            score -= 20
            reasons.append("far_from_candidate_pool")
    if "ROAD" in local_type.upper():
        score += 5
    return OpenNameCandidate(source, name, local_type, e, n, score, ";".join(reasons))


def build_openname_candidates(
    con: sqlite3.Connection,
    prev_row: pd.Series,
    raw_row: dict[str, Any],
    known_places: list[str],
) -> list[OpenNameCandidate]:
    text = " | ".join(
        [
            str(prev_row.get("original_address") or ""),
            str(prev_row.get("ocr_roads") or ""),
            str(prev_row.get("ocr_location_lines") or ""),
        ]
    )
    road_phrases = extract_road_phrases(text)
    localities = extract_localities(text, known_places)
    candidates: list[OpenNameCandidate] = []

    for road in road_phrases:
        for rec in exact_openname_query(con, road):
            cand = candidate_from_openname(rec, "openname_road", [f"road={road}"], raw_row, localities)
            if cand:
                candidates.append(cand)

    for locality in localities:
        if locality in GENERIC_PLACES:
            continue
        for rec in exact_openname_query(con, locality):
            cand = candidate_from_openname(
                rec, "openname_place", [f"place={locality}"], raw_row, localities
            )
            if cand:
                # Populated place is less precise than road/site, but useful
                # when road geocoding chose the wrong same-name road.
                cand.score -= 5
                candidates.append(cand)

    for phrase in extract_named_site_phrases(str(prev_row.get("original_address") or "")):
        for rec in like_openname_query(con, phrase):
            cand = candidate_from_openname(rec, "openname_named", [f"site~={phrase}"], raw_row, localities)
            if cand:
                candidates.append(cand)

    # Relation: if OCR says "between A and B" and both roads resolve, add the
    # midpoint between the best two road candidates.
    relations = set(str(prev_row.get("ocr_relations") or "").split(","))
    if "between" in relations:
        road_cands = [c for c in candidates if c.source == "openname_road"]
        by_name: dict[str, OpenNameCandidate] = {}
        for cand in sorted(road_cands, key=lambda c: c.score, reverse=True):
            by_name.setdefault(norm(cand.name), cand)
        vals = list(by_name.values())
        if len(vals) >= 2:
            a, b = vals[:2]
            candidates.append(
                OpenNameCandidate(
                    "openname_between_midpoint",
                    f"between {a.name} / {b.name}",
                    "Relation Midpoint",
                    (a.easting + b.easting) / 2,
                    (a.northing + b.northing) / 2,
                    max(a.score, b.score) + 5,
                    f"between_midpoint;{a.reason};{b.reason}",
                )
            )

    # Deduplicate close candidates.
    dedup: dict[tuple[str, int, int], OpenNameCandidate] = {}
    for cand in candidates:
        key = (cand.source, round(cand.easting / 10), round(cand.northing / 10))
        old = dedup.get(key)
        if old is None or cand.score > old.score:
            dedup[key] = cand
    return sorted(dedup.values(), key=lambda c: c.score, reverse=True)


def should_accept(prev_row: pd.Series, cand: OpenNameCandidate) -> tuple[bool, str]:
    source = str(prev_row.get("baseline_source") or "").lower()
    selected_source = str(prev_row.get("selected_source") or "")
    selected_e = parse_float(prev_row.get("selected_easting"))
    selected_n = parse_float(prev_row.get("selected_northing"))
    cand_to_selected = (
        math.hypot(cand.easting - selected_e, cand.northing - selected_n)
        if selected_e is not None and selected_n is not None
        else None
    )
    selected_in_bbox = selected_e is not None and selected_n is not None and in_bbox(selected_e, selected_n)
    relations = set(str(prev_row.get("ocr_relations") or "").split(","))
    context_text = " | ".join(
        [
            str(prev_row.get("original_address") or ""),
            str(prev_row.get("ocr_roads") or ""),
            str(prev_row.get("ocr_location_lines") or ""),
        ]
    )
    road_count = len(extract_road_phrases(context_text))

    if selected_source == "ocr_geocode":
        return False, "keep_ocr_geocode"
    if cand.source == "openname_road" and road_count >= 3:
        return False, "reject_single_road_for_multi_road_context"
    if cand.source == "openname_place" and norm(cand.name) in GENERIC_PLACES:
        return False, "reject_generic_place_final"
    if not selected_in_bbox and cand.score >= 15:
        return True, "selected_outside_bbox"
    if source in LOW_TRUST_SOURCES and cand.score >= 25 and cand.source != "openname_place":
        return True, "low_trust_baseline_openname_anchor"
    if source in LOW_TRUST_SOURCES and cand.score >= 35 and cand.source == "openname_place":
        return True, "low_trust_baseline_specific_place_anchor"
    if cand_to_selected is not None and cand_to_selected <= 80 and cand.score >= 25:
        if source not in LOW_TRUST_SOURCES:
            return False, "near_existing_but_high_trust_baseline"
        return True, "near_existing_selected_refinement"
    return False, "gate_keep_baseline"


def run(args: argparse.Namespace) -> None:
    prev = pd.read_csv(args.prev_csv)
    raw_payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    raw_rows = {str(row["key"]): row for row in raw_payload["rows"]}
    truth = load_truth(Path(args.input_gpkg), args.layer)
    con = sqlite3.connect(args.open_names)
    known_places = load_local_place_names(con)

    output_rows = []
    for _, prev_row in prev.iterrows():
        key = str(prev_row["key"])
        raw_row = raw_rows.get(key) or raw_rows.get(key.split("_", 1)[0])
        geom = truth.get(key.split("_", 1)[0])
        if raw_row is None or geom is None:
            continue
        candidates = build_openname_candidates(con, prev_row, raw_row, known_places)
        best = candidates[0] if candidates else None
        accept, gate = should_accept(prev_row, best) if best else (False, "no_openname_candidate")

        current_e = parse_float(prev_row.get("selected_easting"))
        current_n = parse_float(prev_row.get("selected_northing"))
        new_e = best.easting if best and accept else current_e
        new_n = best.northing if best and accept else current_n
        current_dist = distance_to_geom(current_e, current_n, geom)
        new_dist = distance_to_geom(new_e, new_n, geom)
        delta = current_dist - new_dist if current_dist is not None and new_dist is not None else None

        output_rows.append(
            {
                **prev_row.to_dict(),
                "v3_selected_source": best.source if best and accept else prev_row.get("selected_source"),
                "v3_selected_address": best.name if best and accept else prev_row.get("selected_address"),
                "v3_selected_easting": new_e,
                "v3_selected_northing": new_n,
                "v3_selected_distance_m": new_dist,
                "v3_delta_vs_v2_m": delta,
                "v3_gate": gate,
                "v3_best_openname_source": best.source if best else None,
                "v3_best_openname_name": best.name if best else None,
                "v3_best_openname_type": best.local_type if best else None,
                "v3_best_openname_easting": best.easting if best else None,
                "v3_best_openname_northing": best.northing if best else None,
                "v3_best_openname_score": best.score if best else None,
                "v3_best_openname_reason": best.reason if best else None,
                "v3_openname_candidate_count": len(candidates),
            }
        )

    con.close()
    df = pd.DataFrame(output_rows)
    summary = summarize(df)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)
    df.to_excel(out_prefix.with_suffix(".xlsx"), index=False)
    out_prefix.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": output_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.xlsx')}")
    print(f"Wrote {out_prefix.with_suffix('.json')}")


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    comp = df[df["selected_distance_m"].notna() & df["v3_selected_distance_m"].notna()].copy()
    improved = comp[comp["v3_delta_vs_v2_m"] > 1]
    worse = comp[comp["v3_delta_vs_v2_m"] < -1]
    same = comp[comp["v3_delta_vs_v2_m"].abs() <= 1]

    def mean(series: pd.Series) -> float | None:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else None

    return {
        "rows": int(len(df)),
        "comparable_rows": int(len(comp)),
        "improved_vs_v2": int(len(improved)),
        "worse_vs_v2": int(len(worse)),
        "same_vs_v2": int(len(same)),
        "v2_mean_distance_m": mean(comp["selected_distance_m"]),
        "v3_mean_distance_m": mean(comp["v3_selected_distance_m"]),
        "mean_delta_vs_v2_m": mean(comp["v3_delta_vs_v2_m"]),
        "v2_over_100m": int((comp["selected_distance_m"] > 100).sum()),
        "v3_over_100m": int((comp["v3_selected_distance_m"] > 100).sum()),
        "v2_over_500m": int((comp["selected_distance_m"] > 500).sum()),
        "v3_over_500m": int((comp["v3_selected_distance_m"] > 500).sum()),
        "selected_source_counts": {
            str(k): int(v) for k, v in df["v3_selected_source"].value_counts(dropna=False).to_dict().items()
        },
        "top_improvements": [
            {
                "key": str(r["key"]),
                "delta_m": float(r["v3_delta_vs_v2_m"]),
                "v2_m": float(r["selected_distance_m"]),
                "v3_m": float(r["v3_selected_distance_m"]),
                "source": r["v3_selected_source"],
                "name": r["v3_selected_address"],
                "address": r["original_address"],
            }
            for _, r in improved.sort_values("v3_delta_vs_v2_m", ascending=False).head(15).iterrows()
        ],
        "top_regressions": [
            {
                "key": str(r["key"]),
                "delta_m": float(r["v3_delta_vs_v2_m"]),
                "v2_m": float(r["selected_distance_m"]),
                "v3_m": float(r["v3_selected_distance_m"]),
                "source": r["v3_selected_source"],
                "name": r["v3_selected_address"],
                "address": r["original_address"],
            }
            for _, r in worse.sort_values("v3_delta_vs_v2_m").head(15).iterrows()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prev-csv", default=str(DEFAULT_PREV_CSV))
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--input-gpkg", default=str(DEFAULT_INPUT_GPKG))
    parser.add_argument("--open-names", default=str(DEFAULT_OPEN_NAMES))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--layer", default="mansfield-manual-polygon-link-random200")
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--generic-place-names", default="")
    parser.add_argument("--county-names", default="")
    parser.add_argument("--district-names", default="")
    args = parser.parse_args()
    global MANSFIELD_BBOX, GENERIC_PLACES, COUNTY_NAMES, DISTRICT_NAMES
    MANSFIELD_BBOX = parse_bbox(args.local_bbox, MANSFIELD_BBOX)
    GENERIC_PLACES = set(parse_names(args.generic_place_names, GENERIC_PLACES))
    COUNTY_NAMES = set(parse_names(args.county_names, COUNTY_NAMES))
    DISTRICT_NAMES = set(parse_names(args.district_names, DISTRICT_NAMES))
    return args


if __name__ == "__main__":
    run(parse_args())
