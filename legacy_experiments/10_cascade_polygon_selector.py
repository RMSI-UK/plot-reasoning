#!/usr/bin/env python3
"""Post-process Mansfield hybrid polygon candidates with a cascade selector.

This script does not create candidates from the manual truth layer. It reads the
production-visible top-N candidates emitted by ``9_hybrid_polygon_rerank.py`` and
adds one extra production-visible evidence source: address candidate points saved
in the geocoding JSON. For exact OS address cases, it prefers polygons that
contain the requested address candidate points while containing fewer unrelated
same-query address points.

The manual polygon layer is only represented by the diagnostic truth columns
already present in the top-N CSV.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parent
HYBRID_SCRIPT = ROOT / "9_hybrid_polygon_rerank.py"
spec = importlib.util.spec_from_file_location("hybrid_rerank", HYBRID_SCRIPT)
hybrid_rerank = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["hybrid_rerank"] = hybrid_rerank
spec.loader.exec_module(hybrid_rerank)
base_rerank = hybrid_rerank.base_rerank


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TAG = "mansfield-manual-polygon-link_random1200_seed42_43_combined"
DEFAULT_INPUT_TOP = TMP / f"{DEFAULT_TAG}_hybrid_polygon_rerank_v1_top5roi_top50.csv"
DEFAULT_INPUT_JSON = TMP / f"{DEFAULT_TAG}_gemini.json"
DEFAULT_OUTPUT_PREFIX = TMP / f"{DEFAULT_TAG}_cascade_polygon_selector_v1"

POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
LAND_STYLE_RE = re.compile(
    r"\b(?:land|rear|adjacent|adjoining|site|plot|plots|yard|field|former|part of|between|side of)\b",
    re.I,
)


def load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return {str(row.get("key")): row for row in rows if isinstance(row, dict)}


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def requested_numbers(text: Any) -> set[int]:
    value = POSTCODE_RE.sub(" ", str(text or "").upper())
    out: set[int] = set()
    for start_s, end_s in re.findall(r"\b(\d{1,4})\s*(?:-|TO|/)\s*(\d{1,4})\b", value):
        start = int(start_s)
        end = int(end_s)
        low, high = sorted((start, end))
        if 0 < low < 1000 and high - low <= 20:
            out.update(range(low, high + 1))
    for number_s in re.findall(r"\b(\d{1,4})[A-Z]?\b", value):
        number = int(number_s)
        if 0 < number < 1000:
            out.add(number)
    return out


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def candidate_address(candidate: dict[str, Any]) -> str:
    return str(candidate.get("address") or candidate.get("ADDRESS") or "")


def candidate_point(candidate: dict[str, Any]) -> Point | None:
    x = base_rerank.parse_float(candidate.get("easting_27700") or candidate.get("X_COORDINATE"))
    y = base_rerank.parse_float(candidate.get("northing_27700") or candidate.get("Y_COORDINATE"))
    if x is None or y is None:
        return None
    return Point(x, y)


def collect_address_points(raw_row: dict[str, Any], requested: set[int]) -> list[dict[str, Any]]:
    if not requested:
        return []

    points: list[dict[str, Any]] = []
    seen: set[tuple[float, float, tuple[int, ...]]] = set()
    for field in ("os_candidates_json", "gog_candidates_json", "os_fallback_pool_json"):
        for idx, candidate in enumerate(parse_json_list(raw_row.get(field))[:160]):
            point = candidate_point(candidate)
            numbers = requested_numbers(candidate_address(candidate))
            if point is None or not numbers:
                continue
            key = (round(point.x, 2), round(point.y, 2), tuple(sorted(numbers)))
            if key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "point": point,
                    "numbers": numbers,
                    "requested_hit": bool(numbers & requested),
                    "exact_numbers": numbers == requested,
                    "candidate_index": idx,
                }
            )

    best_x = base_rerank.parse_float(raw_row.get("best_easting_27700_final"))
    best_y = base_rerank.parse_float(raw_row.get("best_northing_27700_final"))
    if best_x is not None and best_y is not None:
        points.append(
            {
                "point": Point(best_x, best_y),
                "numbers": set(requested),
                "requested_hit": True,
                "exact_numbers": True,
                "candidate_index": -1,
            }
        )
    return points


def load_candidate_geometries(top: pd.DataFrame) -> dict[str, Any]:
    wanted_ids = set(top["candidate_id"].astype(str))
    sources = [
        hybrid_rerank.load_wfs_source(
            hybrid_rerank.DEFAULT_RAW_WFS_GPKG,
            hybrid_rerank.DEFAULT_RAW_WFS_LAYER,
            "wfs_raw",
            "Land|Building",
        )[["candidate_id", "geometry"]],
        hybrid_rerank.load_wfs_source(
            hybrid_rerank.DEFAULT_MERGED_WFS_GPKG,
            hybrid_rerank.DEFAULT_MERGED_WFS_LAYER,
            "wfs_merged",
            "Land|Building",
        )[["candidate_id", "geometry"]],
        hybrid_rerank.load_council_source(
            hybrid_rerank.DEFAULT_COUNCIL_GPKG,
            hybrid_rerank.DEFAULT_COUNCIL_LAYER,
            hybrid_rerank.MANSFIELD_BBOX,
        )[["candidate_id", "geometry"]],
    ]
    geometries: dict[str, Any] = {}
    for gdf in sources:
        sub = gdf[gdf["candidate_id"].astype(str).isin(wanted_ids)]
        for _, row in sub.iterrows():
            geometries[str(row["candidate_id"])] = row.geometry
    return geometries


def add_address_point_features(
    top: pd.DataFrame,
    raw_rows: dict[str, dict[str, Any]],
    geometries: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for case_key, group in top.groupby("case_key", sort=False):
        base_key = str(group["base_key"].iloc[0])
        raw_row = raw_rows.get(str(case_key)) or raw_rows.get(base_key) or {}
        requested = requested_numbers(raw_row.get("original_address") or group["original_address"].iloc[0])
        points = collect_address_points(raw_row, requested)

        for index, row in group.iterrows():
            geom = geometries.get(str(row["candidate_id"]))
            requested_inside = 0
            extra_inside = 0
            exact_inside = 0
            best_index_inside: int | None = None
            if geom is not None:
                for point_record in points:
                    if not geom.covers(point_record["point"]):
                        continue
                    if point_record["requested_hit"]:
                        requested_inside += 1
                        exact_inside += int(point_record["exact_numbers"])
                        candidate_index = int(point_record["candidate_index"])
                        if best_index_inside is None or candidate_index < best_index_inside:
                            best_index_inside = candidate_index
                    else:
                        extra_inside += 1
            records.append(
                {
                    "row_index": index,
                    "requested_number_count": max(1, len(requested)),
                    "address_candidate_point_count": len(points),
                    "address_requested_points_inside": requested_inside,
                    "address_extra_points_inside": extra_inside,
                    "address_exact_points_inside": exact_inside,
                    "address_best_candidate_index_inside": best_index_inside,
                }
            )

    feature_df = pd.DataFrame(records).set_index("row_index")
    return top.join(feature_df)


def is_exact_address_case(group: pd.DataFrame) -> bool:
    first = group.iloc[0]
    if bool(first["land_style"]):
        return False
    return (
        str(first.get("best_source_final") or "") == "os"
        and str(first.get("best_selection_category") or "") == "exact"
        and float(first.get("best_confidence") or 0.0) >= 85
    )


def select_case(group: pd.DataFrame) -> tuple[pd.Series, str]:
    current = group.sort_values("rule_rank").iloc[0]
    if not is_exact_address_case(group):
        return current, "current_rule_top1"

    candidates = group[pd.to_numeric(group["address_requested_points_inside"], errors="coerce").fillna(0) > 0].copy()
    if candidates.empty:
        return current, "current_rule_top1_no_address_point_inside"

    source_priority = {"council_cadastral": 3, "wfs_merged": 2, "wfs_raw": 1}
    requested_count = pd.to_numeric(candidates["requested_number_count"], errors="coerce").fillna(1).clip(lower=1)
    area_limit = 700.0 * requested_count
    candidates["source_priority"] = candidates["candidate_source"].map(source_priority).fillna(0)
    candidates["address_area_penalty"] = (
        np.log1p(pd.to_numeric(candidates["candidate_area_m2"], errors="coerce").fillna(0.0)) - np.log1p(area_limit)
    ).clip(lower=0)
    candidates = candidates.sort_values(
        [
            "address_exact_points_inside",
            "address_requested_points_inside",
            "address_extra_points_inside",
            "source_priority",
            "address_area_penalty",
            "rule_score",
        ],
        ascending=[False, False, True, False, True, False],
    )
    return candidates.iloc[0], "exact_address_point_selector"


def summarize(selected: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"cases": int(len(selected))}
    if "truth_iou" not in selected:
        return summary
    iou = pd.to_numeric(selected["truth_iou"], errors="coerce").fillna(0.0)
    summary.update(
        {
            "intersects": int((iou > 0).sum()),
            "intersects_rate": float((iou > 0).mean()),
            "iou50": int((iou >= 0.5).sum()),
            "iou50_rate": float((iou >= 0.5).mean()),
            "iou80": int((iou >= 0.8).sum()),
            "iou80_rate": float((iou >= 0.8).mean()),
            "median_iou": float(iou.median()),
        }
    )
    for split, split_df in selected.groupby("sample_split", dropna=False):
        split_iou = pd.to_numeric(split_df["truth_iou"], errors="coerce").fillna(0.0)
        summary[f"split_{split}"] = {
            "cases": int(len(split_df)),
            "iou50": int((split_iou >= 0.5).sum()),
            "iou50_rate": float((split_iou >= 0.5).mean()),
            "iou80": int((split_iou >= 0.8).sum()),
            "iou80_rate": float((split_iou >= 0.8).mean()),
            "median_iou": float(split_iou.median()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade selector for Mansfield hybrid polygon candidates.")
    parser.add_argument("--top-csv", type=Path, default=DEFAULT_INPUT_TOP)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-enhanced-top", action="store_true")
    args = parser.parse_args()

    print("loading top candidates and geocoding rows...")
    top = pd.read_csv(args.top_csv, dtype={"case_key": str, "base_key": str})
    top = top.sort_values(["case_key", "rule_rank"]).reset_index(drop=True)
    raw_rows = load_raw_rows(args.input_json)

    for column in (
        "truth_iou",
        "rule_rank",
        "candidate_area_m2",
        "best_confidence",
        "rule_score",
        "address_extra_points_inside",
    ):
        if column in top:
            top[column] = pd.to_numeric(top[column], errors="coerce")
    top["land_style"] = top["original_address"].fillna("").astype(str).str.contains(LAND_STYLE_RE)

    print("loading candidate geometries for top candidate IDs...")
    geometries = load_candidate_geometries(top)
    print(f"loaded candidate geometries={len(geometries)}")

    print("adding address-point support features...")
    top = add_address_point_features(top, raw_rows, geometries)

    selected_rows: list[dict[str, Any]] = []
    for case_key, group in top.groupby("case_key", sort=False):
        selected, reason = select_case(group)
        row = selected.to_dict()
        row["selected_reason"] = reason
        row["selected_rule_rank"] = row.get("rule_rank")
        selected_rows.append(row)

    selected_df = pd.DataFrame(selected_rows)
    summary = summarize(selected_df)
    summary["selected_reason_counts"] = selected_df["selected_reason"].value_counts(dropna=False).to_dict()
    summary["selected_source_counts"] = selected_df["candidate_source"].value_counts(dropna=False).to_dict()

    out_prefix = args.output_prefix
    selected_csv = out_prefix.with_name(out_prefix.name + "_selected.csv")
    summary_json = out_prefix.with_suffix(".summary.json")
    xlsx_path = out_prefix.with_suffix(".xlsx")
    selected_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(selected_csv, index=False)
    selected_df.to_excel(xlsx_path, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.write_enhanced_top:
        enhanced_csv = out_prefix.with_name(out_prefix.name + "_top50_enhanced.csv")
        top.to_csv(enhanced_csv, index=False)
        print(f"enhanced top candidates: {enhanced_csv}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"selected csv: {selected_csv}")
    print(f"xlsx: {xlsx_path}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
