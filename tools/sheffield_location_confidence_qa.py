#!/usr/bin/env python3
"""Create Sheffield polygon location QA confidence from address geocode evidence.

The output is one row per polygon.  The Gemini/OS address-point output may have
multiple rows per planning reference because multi-address cases are expanded.
For QA we compare each polygon against every valid geocode point for the same
planning reference and keep the nearest/strongest spatial evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


DEFAULT_QA_DIR = Path("/data/sheffield/spatial/QA")
DEFAULT_POLYGONS = DEFAULT_QA_DIR / "sheffieldwp6_27700.gpkg"
DEFAULT_LAYER = "sheffieldwp6"
DEFAULT_GEOCODE_JSONL = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini.jsonl"
DEFAULT_OUTSIDE_CSV = DEFAULT_QA_DIR / "sheffieldwp6_cases_not_fully_within_sheffield.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_QA_DIR / "sheffieldwp6_location_confidence_v1.csv"
DEFAULT_OUTPUT_GPKG = DEFAULT_QA_DIR / "sheffieldwp6_location_confidence_v1.gpkg"
SHEFFIELD_WORK_BBOX = (423000.0, 379000.0, 445500.0, 400500.0)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def in_work_bbox(x: float | None, y: float | None, pad: float = 1500.0) -> bool:
    if x is None or y is None:
        return False
    minx, miny, maxx, maxy = SHEFFIELD_WORK_BBOX
    return minx - pad <= x <= maxx + pad and miny - pad <= y <= maxy + pad


def load_geocode_rows(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
        row = json.loads(line)
        x = as_float(row.get("best_easting_27700_final"))
        y = as_float(row.get("best_northing_27700_final"))
        confidence = as_float(row.get("best_confidence"))
        rows.append(
            {
                "line_no": line_no,
                "unique_key": str(row.get("unique_key") or ""),
                "idx": row.get("idx"),
                "variant_key": row.get("variant_key"),
                "original_address": row.get("original_address"),
                "best_address_final": row.get("best_address_final"),
                "best_source_final": row.get("best_source_final"),
                "best_selection_category": row.get("best_selection_category"),
                "gemini_status": row.get("gemini_status"),
                "best_confidence": confidence,
                "x": x,
                "y": y,
                "has_xy": x is not None and y is not None,
                "point_in_work_bbox": in_work_bbox(x, y),
            }
        )
    return pd.DataFrame(rows)


def pairwise_spread(points: list[Point]) -> float | None:
    if len(points) < 2:
        return 0.0 if points else None
    max_distance = 0.0
    for left_idx, left in enumerate(points):
        for right in points[left_idx + 1 :]:
            max_distance = max(max_distance, float(left.distance(right)))
    return max_distance


def confidence_for_polygon(
    *,
    has_valid_points: bool,
    any_inside: bool,
    min_distance_m: float | None,
    nearest_confidence: float,
    nearest_source: str,
    nearest_status: str,
    nearest_in_bbox: bool,
    not_fully_within_boundary: bool,
) -> tuple[str, str]:
    """Return high/medium/low plus a compact reason string.

    This is intentionally conservative about weak or off-city geocoder results,
    but it treats points inside or near the polygon as strong evidence because
    location QA is about whether the captured polygon sits in the right place,
    not whether the geometry is parcel-perfect.
    """

    source = (nearest_source or "").lower()
    status = (nearest_status or "").lower()
    distance = float(min_distance_m) if min_distance_m is not None else math.inf
    reasons: list[str] = []

    if not has_valid_points:
        return "low", "no_valid_address_geocode_point"
    if not nearest_in_bbox:
        return "low", "nearest_geocode_point_outside_sheffield_work_bbox"
    if source == "none" or status == "parse_error":
        return "low", f"geocode_unusable:{source or status}"

    if any_inside and nearest_confidence >= 80 and source == "os":
        confidence = "high"
        reasons.append("os_point_inside_polygon_conf_ge80")
    elif any_inside and nearest_confidence >= 65:
        confidence = "medium"
        reasons.append("point_inside_polygon_conf_65_79_or_non_os")
    elif distance <= 25 and nearest_confidence >= 80 and source == "os":
        confidence = "high"
        reasons.append("os_point_within_25m_conf_ge80")
    elif distance <= 75 and nearest_confidence >= 75:
        confidence = "medium"
        reasons.append("point_within_75m_conf_ge75")
    elif distance <= 150 and nearest_confidence >= 80:
        confidence = "medium"
        reasons.append("point_within_150m_conf_ge80")
    else:
        confidence = "low"
        reasons.append("nearest_point_too_far_or_low_confidence")

    if not_fully_within_boundary and confidence == "high":
        confidence = "medium"
        reasons.append("demoted_boundary_not_fully_within_sheffield")

    return confidence, ";".join(reasons)


def build_location_qa(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    polygons = gpd.read_file(args.polygons_gpkg, layer=args.layer)
    polygons = polygons.set_crs(27700) if polygons.crs is None else polygons.to_crs(27700)
    polygons["lafilerefe"] = polygons["lafilerefe"].astype(str)
    polygons["_polygon_row_id"] = range(1, len(polygons) + 1)
    polygons["poly_area_m2"] = polygons.geometry.area

    geocode = load_geocode_rows(args.geocode_jsonl)
    geocode_by_ref = {key: group.copy() for key, group in geocode.groupby("unique_key", sort=False)}
    valid_by_ref = {
        key: group[group["has_xy"]].copy()
        for key, group in geocode.groupby("unique_key", sort=False)
    }

    outside_refs: set[str] = set()
    if args.outside_boundary_csv and args.outside_boundary_csv.exists():
        outside = pd.read_csv(args.outside_boundary_csv, dtype={"lafilerefe": str})
        if "lafilerefe" in outside:
            outside_refs = set(outside["lafilerefe"].astype(str))

    output_rows: list[dict[str, Any]] = []
    for _, poly in polygons.iterrows():
        ref = str(poly["lafilerefe"])
        geom = poly.geometry
        all_rows = geocode_by_ref.get(ref, pd.DataFrame())
        valid_rows = valid_by_ref.get(ref, pd.DataFrame())

        point_records: list[tuple[float, float, bool, pd.Series, Point]] = []
        for _, row in valid_rows.iterrows():
            point = Point(float(row["x"]), float(row["y"]))
            distance = float(geom.distance(point))
            centroid_distance = float(geom.centroid.distance(point))
            inside = bool(geom.covers(point))
            point_records.append((distance, centroid_distance, inside, row, point))

        point_records.sort(
            key=lambda item: (
                item[0],
                -float(item[3].get("best_confidence") or 0.0),
                str(item[3].get("best_source_final") or ""),
            )
        )
        nearest = point_records[0][3] if point_records else None
        min_distance = point_records[0][0] if point_records else None
        nearest_centroid_distance = point_records[0][1] if point_records else None
        any_inside = any(item[2] for item in point_records)
        inside_count = sum(1 for item in point_records if item[2])
        nearest_confidence = float(nearest.get("best_confidence") or 0.0) if nearest is not None else 0.0
        nearest_source = str(nearest.get("best_source_final") or "") if nearest is not None else ""
        nearest_status = str(nearest.get("gemini_status") or "") if nearest is not None else ""
        nearest_in_bbox = bool(nearest.get("point_in_work_bbox")) if nearest is not None else False
        confidence, reason = confidence_for_polygon(
            has_valid_points=bool(point_records),
            any_inside=any_inside,
            min_distance_m=min_distance,
            nearest_confidence=nearest_confidence,
            nearest_source=nearest_source,
            nearest_status=nearest_status,
            nearest_in_bbox=nearest_in_bbox,
            not_fully_within_boundary=ref in outside_refs,
        )

        valid_points = [item[4] for item in point_records]
        source_counts = Counter(valid_rows["best_source_final"].fillna("").astype(str)) if not valid_rows.empty else Counter()
        status_counts = Counter(all_rows["gemini_status"].fillna("").astype(str)) if not all_rows.empty else Counter()
        base = {
            "location_confidence": confidence,
            "location_confidence_reason": reason,
            "qa_geocode_rows": int(len(all_rows)),
            "qa_valid_geocode_rows": int(len(valid_rows)),
            "qa_missing_geocode_rows": int(len(all_rows) - len(valid_rows)),
            "qa_any_point_inside_polygon": any_inside,
            "qa_inside_point_count": int(inside_count),
            "qa_nearest_point_distance_m": min_distance,
            "qa_nearest_point_centroid_distance_m": nearest_centroid_distance,
            "qa_points_within_10m": sum(1 for item in point_records if item[0] <= 10),
            "qa_points_within_25m": sum(1 for item in point_records if item[0] <= 25),
            "qa_points_within_50m": sum(1 for item in point_records if item[0] <= 50),
            "qa_points_within_100m": sum(1 for item in point_records if item[0] <= 100),
            "qa_points_within_200m": sum(1 for item in point_records if item[0] <= 200),
            "qa_points_within_500m": sum(1 for item in point_records if item[0] <= 500),
            "qa_geocode_point_spread_m": pairwise_spread(valid_points),
            "qa_nearest_best_confidence": nearest_confidence if nearest is not None else None,
            "qa_nearest_best_source_final": nearest_source,
            "qa_nearest_gemini_status": nearest_status,
            "qa_nearest_best_selection_category": str(nearest.get("best_selection_category") or "") if nearest is not None else "",
            "qa_nearest_point_in_work_bbox": nearest_in_bbox,
            "qa_nearest_original_address": str(nearest.get("original_address") or "") if nearest is not None else "",
            "qa_nearest_best_address_final": str(nearest.get("best_address_final") or "") if nearest is not None else "",
            "qa_source_counts_json": json.dumps(source_counts, ensure_ascii=False),
            "qa_status_counts_json": json.dumps(status_counts, ensure_ascii=False),
            "qa_not_fully_within_sheffield_boundary": ref in outside_refs,
        }
        output_rows.append(base)

    qa = pd.DataFrame(output_rows)
    out = pd.concat([polygons.reset_index(drop=True), qa], axis=1)
    summary = {
        "polygons": int(len(out)),
        "geocode_rows": int(len(geocode)),
        "unique_polygon_refs": int(polygons["lafilerefe"].nunique()),
        "unique_geocode_refs": int(geocode["unique_key"].nunique()),
        "location_confidence_counts": out["location_confidence"].value_counts(dropna=False).to_dict(),
        "any_point_inside_polygon": int(out["qa_any_point_inside_polygon"].sum()),
        "points_within_25m": int((out["qa_points_within_25m"] > 0).sum()),
        "points_within_100m": int((out["qa_points_within_100m"] > 0).sum()),
        "no_valid_geocode_point": int((out["qa_valid_geocode_rows"] == 0).sum()),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sheffield location confidence QA field.")
    parser.add_argument("--polygons-gpkg", type=Path, default=DEFAULT_POLYGONS)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--geocode-jsonl", type=Path, default=DEFAULT_GEOCODE_JSONL)
    parser.add_argument("--outside-boundary-csv", type=Path, default=DEFAULT_OUTSIDE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-gpkg", type=Path, default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default="sheffieldwp6_location_confidence_v1")
    args = parser.parse_args()

    out, summary = build_location_qa(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="geometry").to_csv(args.output_csv, index=False)
    args.output_gpkg.unlink(missing_ok=True)
    out.to_file(args.output_gpkg, layer=args.output_layer, driver="GPKG")
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "output_csv": str(args.output_csv), "output_gpkg": str(args.output_gpkg)}, indent=2))


if __name__ == "__main__":
    main()
