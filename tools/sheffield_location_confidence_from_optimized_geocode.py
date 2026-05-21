#!/usr/bin/env python3
"""Build one Sheffield location-confidence row per actual polygon/case.

Input geocode table is the optimized v3 CSV produced by
`sheffield_geocode_distance_nonzero_postfix.py`.  The table has one row per
address variant; this script collapses those variants onto the actual Sheffield
polygon rows by `lafilerefe == unique_key`.
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
DEFAULT_GEOCODE_CSV = DEFAULT_QA_DIR / "sheffieldwp6_27700_gemini_distance_nonzero_optimized_v3.csv"
DEFAULT_OUTSIDE_CSV = DEFAULT_QA_DIR / "sheffieldwp6_cases_not_fully_within_sheffield.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_QA_DIR / "sheffieldwp6_location_confidence_v2.csv"
DEFAULT_OUTPUT_GPKG = DEFAULT_QA_DIR / "sheffieldwp6_location_confidence_v2.gpkg"
SHEFFIELD_WORK_BBOX = (423000.0, 379000.0, 445500.0, 400500.0)


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip()


def in_work_bbox(x: float | None, y: float | None, pad: float = 1500.0) -> bool:
    if x is None or y is None:
        return False
    minx, miny, maxx, maxy = SHEFFIELD_WORK_BBOX
    return minx - pad <= x <= maxx + pad and miny - pad <= y <= maxy + pad


def pairwise_spread(points: list[Point]) -> float | None:
    if len(points) < 2:
        return 0.0 if points else None
    max_distance = 0.0
    for left_idx, left in enumerate(points):
        for right in points[left_idx + 1 :]:
            max_distance = max(max_distance, float(left.distance(right)))
    return max_distance


def load_optimized_geocode(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for line_no, row in df.iterrows():
        x = as_float(row.get("optimized_easting_27700"))
        y = as_float(row.get("optimized_northing_27700"))
        confidence = as_float(row.get("best_confidence")) or 0.0
        changed = as_bool(row.get("optimized_changed"))
        source = clean_text(row.get("optimized_source_final") or row.get("best_source_final")).lower()
        address = clean_text(row.get("optimized_address_final") or row.get("best_address_final"))
        reason = clean_text(row.get("optimized_reason"))
        google_score = as_float(row.get("optimized_candidate_score"))

        rows.append(
            {
                "line_no": int(line_no) + 1,
                "unique_key": clean_text(row.get("unique_key")),
                "variant_key": clean_text(row.get("variant_key")),
                "original_address": clean_text(row.get("original_address")),
                "geocode_address": address,
                "geocode_source": source,
                "geocode_confidence": confidence,
                "geocode_selection_category": clean_text(row.get("best_selection_category")),
                "geocode_changed_by_postfix": changed,
                "geocode_postfix_reason": reason,
                "geocode_google_candidate_score": google_score,
                "x": x,
                "y": y,
                "has_xy": x is not None and y is not None,
                "point_in_work_bbox": in_work_bbox(x, y),
            }
        )
    return pd.DataFrame(rows)


def address_match_quality(row: pd.Series) -> str:
    source = str(row.get("geocode_source") or "").lower()
    confidence = float(row.get("geocode_confidence") or 0.0)
    changed = bool(row.get("geocode_changed_by_postfix"))
    reason = str(row.get("geocode_postfix_reason") or "")
    google_score = as_float(row.get("geocode_google_candidate_score"))
    category = str(row.get("geocode_selection_category") or "").lower()

    if not row.get("has_xy"):
        return "unresolved"
    if not row.get("point_in_work_bbox"):
        return "out_of_scope"
    if source == "gog" and changed and (google_score or 0.0) >= 85:
        return "google_promoted_strong"
    if source == "os" and confidence >= 90 and category in {"exact", "range"}:
        return "os_strong"
    if source == "os" and confidence >= 75:
        return "os_moderate"
    if source == "fallback" and "kept_current" in reason:
        return "fallback_weak"
    if source in {"none", ""}:
        return "unresolved"
    return "weak"


def confidence_for_polygon(
    *,
    has_valid_points: bool,
    any_inside: bool,
    min_distance_m: float | None,
    nearest_quality: str,
    nearest_source: str,
    nearest_confidence: float,
    nearest_in_bbox: bool,
    points_within_25m: int,
    points_within_100m: int,
    not_fully_within_boundary: bool,
) -> tuple[str, str]:
    distance = float(min_distance_m) if min_distance_m is not None else math.inf
    reasons: list[str] = []

    if not has_valid_points:
        return "low", "no_valid_geocode_point"
    if not nearest_in_bbox:
        return "low", "nearest_geocode_point_outside_sheffield_work_bbox"
    if nearest_quality in {"unresolved", "out_of_scope"}:
        return "low", f"nearest_address_match_{nearest_quality}"

    strong_address = nearest_quality in {"os_strong", "google_promoted_strong"}
    moderate_address = nearest_quality in {"os_moderate", "weak"}

    if any_inside and strong_address:
        confidence = "high"
        reasons.append("strong_address_point_inside_polygon")
    elif any_inside and nearest_confidence >= 65:
        confidence = "medium"
        reasons.append("point_inside_polygon_but_address_not_strong")
    elif distance <= 25 and strong_address:
        confidence = "high"
        reasons.append("strong_address_point_within_25m")
    elif distance <= 75 and (strong_address or moderate_address):
        confidence = "medium"
        reasons.append("address_point_within_75m")
    elif distance <= 150 and strong_address:
        confidence = "medium"
        reasons.append("strong_address_point_within_150m")
    elif points_within_100m >= 2 and strong_address:
        confidence = "medium"
        reasons.append("multiple_address_points_within_100m")
    else:
        confidence = "low"
        reasons.append("nearest_point_too_far_or_address_weak")

    if nearest_source == "fallback" and confidence == "high":
        confidence = "medium"
        reasons.append("demoted_fallback_source")
    if not_fully_within_boundary and confidence == "high":
        confidence = "medium"
        reasons.append("demoted_boundary_not_fully_within_sheffield")

    return confidence, ";".join(reasons)


def build_location_confidence(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    polygons = gpd.read_file(args.polygons_gpkg, layer=args.layer)
    polygons = polygons.set_crs(27700) if polygons.crs is None else polygons.to_crs(27700)
    polygons["lafilerefe"] = polygons["lafilerefe"].astype(str)
    polygons["_polygon_row_id"] = range(1, len(polygons) + 1)
    polygons["poly_area_m2"] = polygons.geometry.area

    geocode = load_optimized_geocode(args.geocode_csv)
    geocode["address_match_quality"] = geocode.apply(address_match_quality, axis=1)
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
                0 if item[3].get("address_match_quality") in {"os_strong", "google_promoted_strong"} else 1,
                -float(item[3].get("geocode_confidence") or 0.0),
            )
        )

        nearest = point_records[0][3] if point_records else None
        min_distance = point_records[0][0] if point_records else None
        nearest_centroid_distance = point_records[0][1] if point_records else None
        any_inside = any(item[2] for item in point_records)
        inside_count = sum(1 for item in point_records if item[2])
        points_within_25m = sum(1 for item in point_records if item[0] <= 25)
        points_within_100m = sum(1 for item in point_records if item[0] <= 100)
        nearest_quality = str(nearest.get("address_match_quality") or "") if nearest is not None else ""
        nearest_source = str(nearest.get("geocode_source") or "") if nearest is not None else ""
        nearest_confidence = float(nearest.get("geocode_confidence") or 0.0) if nearest is not None else 0.0
        nearest_in_bbox = bool(nearest.get("point_in_work_bbox")) if nearest is not None else False

        location_confidence, location_reason = confidence_for_polygon(
            has_valid_points=bool(point_records),
            any_inside=any_inside,
            min_distance_m=min_distance,
            nearest_quality=nearest_quality,
            nearest_source=nearest_source,
            nearest_confidence=nearest_confidence,
            nearest_in_bbox=nearest_in_bbox,
            points_within_25m=points_within_25m,
            points_within_100m=points_within_100m,
            not_fully_within_boundary=ref in outside_refs,
        )

        valid_points = [item[4] for item in point_records]
        source_counts = Counter(valid_rows["geocode_source"].fillna("").astype(str)) if not valid_rows.empty else Counter()
        quality_counts = Counter(valid_rows["address_match_quality"].fillna("").astype(str)) if not valid_rows.empty else Counter()
        base = {
            "location_confidence": location_confidence,
            "location_confidence_reason": location_reason,
            "address_match_quality_nearest": nearest_quality,
            "qa_geocode_rows": int(len(all_rows)),
            "qa_valid_geocode_rows": int(len(valid_rows)),
            "qa_missing_geocode_rows": int(len(all_rows) - len(valid_rows)),
            "qa_any_point_inside_polygon": any_inside,
            "qa_inside_point_count": int(inside_count),
            "qa_nearest_point_distance_m": min_distance,
            "qa_nearest_point_centroid_distance_m": nearest_centroid_distance,
            "qa_points_within_10m": sum(1 for item in point_records if item[0] <= 10),
            "qa_points_within_25m": points_within_25m,
            "qa_points_within_50m": sum(1 for item in point_records if item[0] <= 50),
            "qa_points_within_100m": points_within_100m,
            "qa_points_within_200m": sum(1 for item in point_records if item[0] <= 200),
            "qa_points_within_500m": sum(1 for item in point_records if item[0] <= 500),
            "qa_geocode_point_spread_m": pairwise_spread(valid_points),
            "qa_nearest_geocode_source": nearest_source,
            "qa_nearest_geocode_confidence": nearest_confidence if nearest is not None else None,
            "qa_nearest_geocode_changed_by_postfix": bool(nearest.get("geocode_changed_by_postfix")) if nearest is not None else False,
            "qa_nearest_original_address": str(nearest.get("original_address") or "") if nearest is not None else "",
            "qa_nearest_geocode_address": str(nearest.get("geocode_address") or "") if nearest is not None else "",
            "qa_nearest_geocode_x": float(nearest.get("x")) if nearest is not None and nearest.get("x") is not None else None,
            "qa_nearest_geocode_y": float(nearest.get("y")) if nearest is not None and nearest.get("y") is not None else None,
            "qa_source_counts_json": json.dumps(source_counts, ensure_ascii=False),
            "qa_address_quality_counts_json": json.dumps(quality_counts, ensure_ascii=False),
            "qa_not_fully_within_sheffield_boundary": ref in outside_refs,
        }
        output_rows.append(base)

    qa = pd.DataFrame(output_rows)
    out = pd.concat([polygons.reset_index(drop=True), qa], axis=1)
    summary = {
        "polygon_rows": int(len(out)),
        "unique_polygon_refs": int(polygons["lafilerefe"].nunique()),
        "geocode_variant_rows": int(len(geocode)),
        "unique_geocode_refs": int(geocode["unique_key"].nunique()),
        "location_confidence_counts": out["location_confidence"].value_counts(dropna=False).to_dict(),
        "address_match_quality_nearest_counts": out["address_match_quality_nearest"].value_counts(dropna=False).to_dict(),
        "any_point_inside_polygon": int(out["qa_any_point_inside_polygon"].sum()),
        "points_within_25m": int((out["qa_points_within_25m"] > 0).sum()),
        "points_within_100m": int((out["qa_points_within_100m"] > 0).sum()),
        "no_valid_geocode_point": int((out["qa_valid_geocode_rows"] == 0).sum()),
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons-gpkg", type=Path, default=DEFAULT_POLYGONS)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--geocode-csv", type=Path, default=DEFAULT_GEOCODE_CSV)
    parser.add_argument("--outside-boundary-csv", type=Path, default=DEFAULT_OUTSIDE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-gpkg", type=Path, default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default="sheffieldwp6_location_confidence_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out, summary = build_location_confidence(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="geometry").to_csv(args.output_csv, index=False)
    args.output_gpkg.unlink(missing_ok=True)
    out.to_file(args.output_gpkg, layer=args.output_layer, driver="GPKG")
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "output_csv": str(args.output_csv), "output_gpkg": str(args.output_gpkg)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
