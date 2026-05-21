#!/usr/bin/env python3
"""Audit whether selected plan-page OCR contains multiple usable place anchors.

The goal is diagnostic, not final geocoding: for each sampled Mansfield case,
join the selected location/plan pages to OCR boxes, then check whether those
boxes contain road or named-feature anchors that can be matched against local
OS Open Roads/Open Names data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from difflib import SequenceMatcher
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox, parse_names


DEFAULT_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42.gpkg"
)
DEFAULT_LAYER = "mansfield-manual-polygon-link-random200"
DEFAULT_PREV_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_openroads_v10.csv"
)
DEFAULT_FIND_PLAN = Path("/data/mansfield/skill/find-plan/full-scanraw/find_plan_results.jsonl")
DEFAULT_PLAN_IMAGES = Path("/data/mansfield/backup/plan_images.csv")
DEFAULT_OCR = Path("/data/mansfield/ocr/all_v5ocr.jsonl")
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_plan_ocr_anchor_audit"
)

MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)
ROAD_SUFFIXES = {
    "ROAD",
    "STREET",
    "LANE",
    "AVENUE",
    "CLOSE",
    "WAY",
    "DRIVE",
    "GROVE",
    "CRESCENT",
    "HILL",
    "GATE",
    "SIDE",
    "WALK",
    "PLACE",
    "SQUARE",
    "COURT",
    "PARK",
    "RISE",
    "VIEW",
    "TERRACE",
    "ROW",
    "YARD",
    "CROFT",
}
DIRECTION_SUFFIXES = {"EAST", "WEST", "NORTH", "SOUTH"}
ROAD_LEADING_STOP = {
    "LAND",
    "BETWEEN",
    "AND",
    "AT",
    "OF",
    "TO",
    "LOCATION",
    "OFF",
    "REAR",
    "ADJACENT",
    "ADJ",
    "SITE",
    "ON",
    "THE",
    "PROPOSED",
    "EXISTING",
    "PLAN",
    "FOR",
    "A",
    "AN",
}
GENERIC_PLACE_NAMES = {
    "MANSFIELD",
    "MANSFIELD WOODHOUSE",
    "SUTTON IN ASHFIELD",
    "KIRKBY IN ASHFIELD",
    "NOTTINGHAMSHIRE",
    "DERBYSHIRE",
    "BASSETLAW",
    "NEWARK AND SHERWOOD",
    "ASHFIELD",
}
FEATURE_TYPES = {
    "other",
    "landcover",
    "hydrography",
    "landform",
    "transportNetwork",
    "populatedPlace",
}


def norm(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).upper()
    text = text.translate(str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"}))
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def road_core_tokens(value: str) -> tuple[str, ...]:
    tokens = norm(value).split()
    if len(tokens) >= 3 and tokens[-1] in DIRECTION_SUFFIXES and tokens[-2] in ROAD_SUFFIXES:
        tokens = tokens[:-1]
    if len(tokens) >= 2 and tokens[-1] in ROAD_SUFFIXES:
        return tuple(tokens[:-1])
    return tuple(tokens)


def road_stem(value: str) -> str:
    tokens = norm(value).split()
    if len(tokens) >= 3 and tokens[-1] in DIRECTION_SUFFIXES and tokens[-2] in ROAD_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def resolve_roadlike_to_known(phrase: str, road_names: set[str], limit: int = 4) -> list[str]:
    phrase_n = norm(phrase)
    if not phrase_n:
        return []
    phrase_c = compact(phrase_n)
    phrase_stem = road_stem(phrase_n)
    phrase_core = road_core_tokens(phrase_n)
    exact = [road for road in road_names if compact(road) == phrase_c or road == phrase_n]
    if exact:
        return sorted(exact, key=lambda item: (len(item), item))[:limit]

    candidates: list[tuple[float, str]] = []
    for road in road_names:
        road_stem_n = road_stem(road)
        road_core = road_core_tokens(road)
        score = 0.0
        if road_stem_n == phrase_stem:
            score = 1.0
        elif phrase_core and road_core == phrase_core:
            score = 0.94
        elif len(phrase_c) >= 8 and compact(road_stem_n) == phrase_c:
            score = 0.9
        elif (len(phrase_core) >= 2 or (len(phrase_core) == 1 and len(phrase_core[0]) >= 8)) and road_core[
            : len(phrase_core)
        ] == phrase_core:
            score = 0.82
        elif road_core and phrase_core and road_core[0] == phrase_core[0]:
            ratio = SequenceMatcher(None, compact(road_stem_n), compact(phrase_stem)).ratio()
            if ratio >= 0.78:
                score = 0.72 + ratio / 10.0
        if score:
            candidates.append((score, road))
    candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [road for _, road in candidates[:limit]]


def basename_prefix(filepath: Any) -> str:
    if not filepath:
        return ""
    path = str(filepath)
    match = re.search(r"(MFD_[^/\\]+?)(?:_[0-9]{3}-[0-9]{3}_[0-9]{4}\.jpg)?$", path)
    if match:
        return match.group(1)
    parts = path.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return Path(path).stem


def rel_scan_path(path: str) -> str:
    marker = "/data/mansfield/mansfield_scan_raw/"
    if path.startswith(marker):
        return path[len(marker) :]
    if path.startswith("mansfield_scan_raw/"):
        return path[len("mansfield_scan_raw/") :]
    return path


def load_cases(gpkg: Path, layer: str) -> pd.DataFrame:
    con = sqlite3.connect(gpkg)
    rows = con.execute(
        f"""
        SELECT unique_key, FilePath, chargegeog, supplement
        FROM "{layer}"
        """
    ).fetchall()
    out = pd.DataFrame(rows, columns=["key", "FilePath", "chargegeog", "supplement"])
    out["key"] = out["key"].astype(str)
    out["prefix"] = out["FilePath"].map(basename_prefix)
    return out


def load_audit_rows(gpkg: Path, layer: str, prev_csv: Path) -> pd.DataFrame:
    """Return one row per current candidate row, including expanded sub-cases."""
    gpkg_cases = load_cases(gpkg, layer).rename(
        columns={
            "key": "base_key",
            "chargegeog": "source_chargegeog",
            "supplement": "source_supplement",
        }
    )
    prev = pd.read_csv(prev_csv, dtype={"key": str, "base_key": str})
    keep_cols = [
        "key",
        "base_key",
        "original_address",
        "selected_distance_m",
        "selected_source",
        "selected_reason",
    ]
    rows = prev[keep_cols].copy()
    rows["base_key"] = rows["base_key"].fillna(rows["key"]).astype(str)
    rows = rows.merge(
        gpkg_cases[["base_key", "FilePath", "prefix", "source_chargegeog", "source_supplement"]],
        on="base_key",
        how="left",
    )
    return rows


def load_plan_pages(find_plan: Path, plan_images: Path, prefixes: set[str]) -> dict[str, list[str]]:
    pages: dict[str, list[str]] = defaultdict(list)
    if find_plan.exists():
        with find_plan.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                prefix = rec.get("case_id")
                if prefix not in prefixes:
                    continue
                selected = rec.get("selected_plan_pages_final") or rec.get("selected_plan_pages") or []
                for page in selected:
                    rel = rel_scan_path(str(page))
                    if rel not in pages[prefix]:
                        pages[prefix].append(rel)

    missing = prefixes - set(pages)
    if missing and plan_images.exists():
        with plan_images.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("image_rel_path") or ""
                prefix = rel.split("/", 1)[0]
                if prefix not in missing:
                    continue
                try:
                    prob = float(row.get("plan_prob") or 0)
                except ValueError:
                    prob = 0.0
                if prob >= 0.98:
                    pages[prefix].append(rel)
    return {prefix: vals[:8] for prefix, vals in pages.items()}


def load_road_names(open_roads: Path) -> set[str]:
    gdf = gpd.read_file(
        open_roads,
        layer="road_link",
        bbox=MANSFIELD_BBOX,
        columns=["name_1", "name_2"],
    )
    names: set[str] = set()
    for _, row in gdf.iterrows():
        for col in ("name_1", "name_2"):
            value = norm(row.get(col))
            if len(value) >= 6 and any(value.endswith(" " + suffix) or value == suffix for suffix in ROAD_SUFFIXES):
                names.add(value)
    return names


def load_openname_names(open_names: Path) -> tuple[set[str], set[str]]:
    con = sqlite3.connect(open_names)
    rows = con.execute(
        """
        SELECT NAME1, TYPE, LOCAL_TYPE, GEOMETRY_X, GEOMETRY_Y
        FROM open_names
        WHERE CAST(GEOMETRY_X AS REAL) BETWEEN ? AND ?
          AND CAST(GEOMETRY_Y AS REAL) BETWEEN ? AND ?
        """,
        (MANSFIELD_BBOX[0], MANSFIELD_BBOX[2], MANSFIELD_BBOX[1], MANSFIELD_BBOX[3]),
    ).fetchall()
    locality: set[str] = set()
    features: set[str] = set()
    for name, typ, local_type, *_ in rows:
        n = norm(name)
        if len(n) < 5:
            continue
        if n in GENERIC_PLACE_NAMES:
            locality.add(n)
            continue
        if typ == "populatedPlace":
            locality.add(n)
        if typ in FEATURE_TYPES:
            features.add(n)
        if local_type and "Named Road" in str(local_type):
            features.discard(n)
    return locality, features


def phrase_match(text_compact: str, names: set[str], min_compact_len: int = 5) -> list[str]:
    out = []
    for name in sorted(names, key=lambda x: (-len(x), x)):
        c = compact(name)
        if len(c) >= min_compact_len and c in text_compact:
            out.append(name)
    return out


def extract_roadlike_phrases(text: str, road_names: set[str]) -> list[str]:
    words = norm(text).split()
    out: list[str] = []
    for idx, token in enumerate(words):
        if token not in ROAD_SUFFIXES:
            continue
        for width in range(2, 6):
            start = idx - width + 1
            if start < 0:
                continue
            cand = words[start : idx + 1]
            while cand and cand[0] in ROAD_LEADING_STOP:
                cand = cand[1:]
            if len(cand) < 2:
                continue
            phrase = " ".join(cand)
            if phrase in road_names or len(phrase) >= 8:
                if phrase not in out:
                    out.append(phrase)
    return out


@dataclass
class PageOcr:
    texts: list[str]
    high_texts: list[str]
    n_boxes: int
    n_high_boxes: int


def stream_selected_ocr(ocr_path: Path, wanted_pages: set[str], min_conf: float) -> dict[str, PageOcr]:
    found: dict[str, PageOcr] = {}
    with ocr_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            image_path = rec.get("image_path") or ""
            if image_path not in wanted_pages:
                continue
            texts = []
            high_texts = []
            for box in rec.get("results") or []:
                text = str(box.get("text") or "").strip()
                if not text:
                    continue
                texts.append(text)
                try:
                    conf = float(box.get("confidence") or 0)
                except ValueError:
                    conf = 0.0
                if conf >= min_conf:
                    high_texts.append(text)
            found[image_path] = PageOcr(
                texts=texts,
                high_texts=high_texts,
                n_boxes=len(texts),
                n_high_boxes=len(high_texts),
            )
            if len(found) == len(wanted_pages):
                break
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", type=Path, default=DEFAULT_GPKG)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--prev-csv", type=Path, default=DEFAULT_PREV_CSV)
    parser.add_argument("--find-plan", type=Path, default=DEFAULT_FIND_PLAN)
    parser.add_argument("--plan-images", type=Path, default=DEFAULT_PLAN_IMAGES)
    parser.add_argument("--ocr", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--open-names", type=Path, default=DEFAULT_OPEN_NAMES)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--generic-place-names", default="")
    args = parser.parse_args()
    global MANSFIELD_BBOX, GENERIC_PLACE_NAMES
    MANSFIELD_BBOX = parse_bbox(args.local_bbox, MANSFIELD_BBOX)
    GENERIC_PLACE_NAMES = set(parse_names(args.generic_place_names, GENERIC_PLACE_NAMES))

    cases = load_audit_rows(args.gpkg, args.layer, args.prev_csv)

    plan_pages = load_plan_pages(args.find_plan, args.plan_images, set(cases["prefix"]))
    wanted_pages = {page for pages in plan_pages.values() for page in pages}

    print(f"cases={len(cases)} prefixes={cases['prefix'].nunique()} selected_plan_pages={len(wanted_pages)}")
    print("loading OS local gazetteers...")
    road_names = load_road_names(args.open_roads)
    locality_names, feature_names = load_openname_names(args.open_names)
    feature_names -= road_names
    print(f"road_names={len(road_names)} locality_names={len(locality_names)} feature_names={len(feature_names)}")
    print("streaming selected OCR pages...")
    page_ocr = stream_selected_ocr(args.ocr, wanted_pages, args.min_conf)
    print(f"ocr_pages_found={len(page_ocr)}")

    rows = []
    for _, case in cases.iterrows():
        key = str(case["key"])
        prefix = case["prefix"]
        pages = plan_pages.get(prefix, [])
        all_texts: list[str] = []
        high_texts: list[str] = []
        n_boxes = n_high = 0
        for page in pages:
            ocr = page_ocr.get(page)
            if not ocr:
                continue
            all_texts.extend(ocr.texts)
            high_texts.extend(ocr.high_texts)
            n_boxes += ocr.n_boxes
            n_high += ocr.n_high_boxes

        text_blob = " | ".join(high_texts or all_texts)
        text_compact = compact(text_blob)
        matched_roads = phrase_match(text_compact, road_names, min_compact_len=5)
        matched_features = phrase_match(text_compact, feature_names, min_compact_len=8)
        matched_localities = phrase_match(text_compact, locality_names, min_compact_len=8)
        roadlike = extract_roadlike_phrases(text_blob, road_names)
        resolved_roadlike: list[str] = []
        resolved_phrases: set[str] = set()
        for phrase in roadlike:
            matches = resolve_roadlike_to_known(phrase, road_names)
            if matches:
                resolved_phrases.add(phrase)
            for road in matches:
                if road not in matched_roads and road not in resolved_roadlike:
                    resolved_roadlike.append(road)
        matched_roads = [*matched_roads, *resolved_roadlike]
        unmatched_roadlike = [x for x in roadlike if x not in matched_roads and x not in resolved_phrases]

        useful_anchor_count = len(set(matched_roads) | set(matched_features))
        all_anchor_count = len(set(matched_roads) | set(matched_features) | set(matched_localities))
        rows.append(
            {
                "key": key,
                "prefix": prefix,
                "base_key": case.get("base_key") or "",
                "original_address": case.get("original_address") or case.get("source_chargegeog") or "",
                "selected_distance_m": case.get("selected_distance_m"),
                "selected_source": case.get("selected_source"),
                "selected_reason": case.get("selected_reason"),
                "plan_page_count": len(pages),
                "ocr_plan_pages_found": sum(1 for p in pages if p in page_ocr),
                "ocr_box_count": n_boxes,
                "ocr_high_box_count": n_high,
                "matched_road_count": len(matched_roads),
                "matched_feature_count": len(matched_features),
                "matched_locality_count": len(matched_localities),
                "useful_anchor_count": useful_anchor_count,
                "all_anchor_count": all_anchor_count,
                "roadlike_unmatched_count": len(unmatched_roadlike),
                "matched_roads": " | ".join(matched_roads[:20]),
                "matched_features": " | ".join(matched_features[:20]),
                "matched_localities": " | ".join(matched_localities[:20]),
                "unmatched_roadlike_phrases": " | ".join(unmatched_roadlike[:20]),
                "sample_high_ocr": " | ".join(high_texts[:40]),
                "plan_pages": " | ".join(pages),
            }
        )

    out_df = pd.DataFrame(rows)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".summary.json")
    xlsx_path = args.output_prefix.with_suffix(".xlsx")
    out_df.to_csv(csv_path, index=False)
    out_df.to_excel(xlsx_path, index=False)

    def count(mask: pd.Series) -> int:
        return int(mask.fillna(False).sum())

    dist = pd.to_numeric(out_df["selected_distance_m"], errors="coerce")
    unresolved = dist > 100
    summary = {
        "case_count": int(len(out_df)),
        "cases_with_selected_plan_pages": count(out_df["plan_page_count"] > 0),
        "cases_with_plan_ocr": count(out_df["ocr_box_count"] > 0),
        "cases_with_any_useful_anchor": count(out_df["useful_anchor_count"] >= 1),
        "cases_with_2plus_useful_anchors": count(out_df["useful_anchor_count"] >= 2),
        "cases_with_3plus_useful_anchors": count(out_df["useful_anchor_count"] >= 3),
        "cases_with_2plus_road_anchors": count(out_df["matched_road_count"] >= 2),
        "cases_with_unmatched_roadlike_phrases": count(out_df["roadlike_unmatched_count"] > 0),
        "remaining_over_100m_count": count(unresolved),
        "remaining_over_100m_with_any_useful_anchor": count(unresolved & (out_df["useful_anchor_count"] >= 1)),
        "remaining_over_100m_with_2plus_useful_anchors": count(unresolved & (out_df["useful_anchor_count"] >= 2)),
        "remaining_over_100m_with_3plus_useful_anchors": count(unresolved & (out_df["useful_anchor_count"] >= 3)),
        "remaining_over_100m_with_unmatched_roadlike": count(unresolved & (out_df["roadlike_unmatched_count"] > 0)),
        "output_csv": str(csv_path),
        "output_xlsx": str(xlsx_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
