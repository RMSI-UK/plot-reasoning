#!/usr/bin/env python3
"""Build multiple 150m candidate ROIs per case.

The previous experiment forced all evidence into one 150m square.  That is too
fragile when dense OS/GOG candidates form a wrong local cluster.  This variant
keeps the current/v10 point as a protected ROI and adds the strongest distinct
evidence ROIs, then evaluates the union as a candidate search area.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox


ROOT = Path(__file__).resolve().parent
ROI_SCRIPT = ROOT / "evidence_roi.py"
spec = importlib.util.spec_from_file_location("roi_v1", ROI_SCRIPT)
roi_v1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["roi_v1"] = roi_v1
spec.loader.exec_module(roi_v1)


DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_multi_roi_v2"
)


def roi_square(x: float, y: float, side: float) -> Any:
    half = side / 2.0
    return box(x - half, y - half, x + half, y + half)


def score_centers(points: list[Any], side: float) -> list[dict[str, Any]]:
    """Return ranked candidate ROI centers using the v1 scoring ingredients."""
    if not points:
        return []
    half = side / 2.0
    centers: list[tuple[float, float, str]] = [(p.x, p.y, p.source) for p in points]

    for p in points:
        local = [q for q in points if abs(q.x - p.x) <= 140 and abs(q.y - p.y) <= 140]
        if len({q.source for q in local}) >= 2:
            sw = sum(q.weight for q in local)
            centers.append(
                (
                    sum(q.x * q.weight for q in local) / sw,
                    sum(q.y * q.weight for q in local) / sw,
                    "local_centroid",
                )
            )

    ranked: list[dict[str, Any]] = []
    for cx, cy, center_source in centers:
        inside = [p for p in points if abs(p.x - cx) <= half and abs(p.y - cy) <= half]
        near = [p for p in points if math.hypot(p.x - cx, p.y - cy) <= 140]
        sources = {p.source for p in inside}
        source_bonus = len(sources) * 4.0
        agreement_bonus = 6.0 if len(sources & {"os", "gog", "fallback", "current", "ocr_grid", "range_parity"}) >= 2 else 0.0
        anchor_bonus = 5.0 if sources & {"road_zone", "road_midpoint", "range_parity"} and len(sources) >= 2 else 0.0
        score = sum(p.weight for p in inside) + 0.35 * sum(p.weight for p in near) + source_bonus + agreement_bonus + anchor_bonus
        ranked.append(
            {
                "cx": cx,
                "cy": cy,
                "score": score,
                "sources": "|".join(sorted(sources)),
                "center_source": center_source,
                "inside_count": len(inside),
                "inside_weight": sum(p.weight for p in inside),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    deduped: list[dict[str, Any]] = []
    for item in ranked:
        if all(math.hypot(item["cx"] - old["cx"], item["cy"] - old["cy"]) > side * 0.7 for old in deduped):
            deduped.append(item)
    return deduped


def current_rois(points: list[Any], side: float) -> list[dict[str, Any]]:
    out = []
    for p in points:
        if p.source != "current":
            continue
        out.append(
            {
                "cx": p.x,
                "cy": p.y,
                "score": p.weight + 25.0,
                "sources": "current",
                "center_source": "protected_current",
                "inside_count": 1,
                "inside_weight": p.weight,
                "roi_reason": "protected_current",
            }
        )
        break
    return out


def select_multi_rois(points: list[Any], side: float, evidence_count: int, include_current: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if include_current:
        selected.extend(current_rois(points, side))

    for item in score_centers(points, side):
        if item["center_source"] == "protected_current":
            continue
        if any(math.hypot(item["cx"] - old["cx"], item["cy"] - old["cy"]) <= side * 0.45 for old in selected):
            continue
        item = dict(item)
        item["roi_reason"] = "evidence"
        selected.append(item)
        if sum(1 for roi in selected if roi["roi_reason"] == "evidence") >= evidence_count:
            break
    return selected


def rank_union_polygons(polygons: gpd.GeoDataFrame, union_geom: Any, rois: list[dict[str, Any]], road_names: list[str], road_geoms: dict[str, Any]) -> pd.DataFrame:
    candidates = polygons[polygons.geometry.intersects(union_geom)].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["candidate_key"] = candidates["unique_key"].astype(str)
    candidates["roi_intersection_area"] = candidates.geometry.intersection(union_geom).area
    centers = [Point(item["cx"], item["cy"]) for item in rois]
    candidates["nearest_roi_center_m"] = candidates.geometry.centroid.map(lambda geom: min(geom.distance(center) for center in centers))
    candidates["polygon_area"] = candidates.geometry.area

    road_score = pd.Series(0.0, index=candidates.index)
    for road in road_names:
        geom = road_geoms.get(road)
        if geom is None:
            continue
        d = candidates.geometry.distance(geom)
        road_score += d.map(lambda x: 4.0 if x <= 50 else 3.0 if x <= 100 else 1.5 if x <= 250 else 0.0)
    candidates["road_score"] = road_score
    candidates["rank_score"] = (
        candidates["roi_intersection_area"].clip(upper=3000) / 80.0
        + candidates["road_score"]
        - candidates["nearest_roi_center_m"].clip(upper=250) * 0.012
    )
    return candidates.sort_values(["rank_score", "roi_intersection_area"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=roi_v1.DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=roi_v1.DEFAULT_V10_CSV)
    parser.add_argument("--audit-csv", type=Path, default=roi_v1.DEFAULT_AUDIT_CSV)
    parser.add_argument("--full-gpkg", type=Path, default=roi_v1.DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=roi_v1.DEFAULT_FULL_LAYER)
    parser.add_argument("--sample-gpkg", type=Path, default=roi_v1.DEFAULT_SAMPLE_GPKG)
    parser.add_argument("--sample-layer", default=roi_v1.DEFAULT_SAMPLE_LAYER)
    parser.add_argument("--open-roads", type=Path, default=roi_v1.DEFAULT_OPEN_ROADS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--roi-side", type=float, default=150.0)
    parser.add_argument("--evidence-roi-count", type=int, default=3)
    parser.add_argument("--no-current-roi", action="store_true")
    parser.add_argument("--local-bbox", default="")
    args = parser.parse_args()
    roi_v1.MANSFIELD_BBOX = parse_bbox(args.local_bbox, roi_v1.MANSFIELD_BBOX)

    raw_rows = {str(row["key"]): row for row in json.load(args.input_json.open())["rows"]}
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    audit = pd.read_csv(args.audit_csv, dtype={"key": str, "base_key": str})
    audit_by_key = {str(row["key"]): row for _, row in audit.iterrows()}
    for col in ["v7_selected_distance_m"]:
        v10[col] = pd.to_numeric(v10[col], errors="coerce")

    print("loading polygons and roads...")
    polygons = gpd.read_file(args.full_gpkg, layer=args.full_layer)
    if polygons.crs is not None:
        polygons = polygons.to_crs(27700)
    polygons = polygons[["unique_key", "geometry"]].copy()
    polygons["unique_key"] = polygons["unique_key"].astype(str)
    truth = roi_v1.load_truth(args.sample_gpkg, args.sample_layer)
    road_geoms = roi_v1.load_road_geoms(args.open_roads)

    rows = []
    roi_rows = []
    top_rows = []
    for _, row in v10.iterrows():
        key = str(row["key"])
        base_key = str(row.get("base_key") or key).split("_", 1)[0]
        raw = raw_rows.get(key)
        if raw is None:
            raw = raw_rows.get(base_key)
        audit_row = audit_by_key.get(key)
        if audit_row is None:
            audit_row = audit_by_key.get(base_key)
        target_geom = truth.get(base_key)
        if target_geom is None:
            continue

        points = roi_v1.evidence_points_for_case(row, raw, audit_row, road_geoms)
        selected_rois = select_multi_rois(
            points,
            side=args.roi_side,
            evidence_count=args.evidence_roi_count,
            include_current=not args.no_current_roi,
        )
        best_conf = roi_v1.parse_float(raw.get("best_confidence") if raw else None)
        plan_roads = roi_v1.split_anchors(audit_row.get("matched_roads") if audit_row is not None else "")
        address_roads = roi_v1.extract_address_roads(row.get("original_address"), set(road_geoms))
        all_roads = []
        for road in [*address_roads, *plan_roads]:
            if road not in all_roads:
                all_roads.append(road)

        if not selected_rois:
            rows.append({"key": key, "base_key": base_key, "status": "no_rois", "best_confidence": best_conf})
            continue

        roi_geoms = [roi_square(item["cx"], item["cy"], args.roi_side) for item in selected_rois]
        union_roi = unary_union(roi_geoms)
        ranked_polys = rank_union_polygons(polygons, union_roi, selected_rois, all_roads, road_geoms)
        target_hit = ranked_polys[ranked_polys["candidate_key"] == base_key] if not ranked_polys.empty else pd.DataFrame()
        target_rank = int(target_hit.index[0] + 1) if not target_hit.empty else None

        for idx, (item, geom) in enumerate(zip(selected_rois, roi_geoms), start=1):
            roi_rows.append(
                {
                    "case_key": key,
                    "base_key": base_key,
                    "roi_rank": idx,
                    "roi_reason": item["roi_reason"],
                    "roi_sources": item["sources"],
                    "roi_score": item["score"],
                    "roi_minx": geom.bounds[0],
                    "roi_miny": geom.bounds[1],
                    "roi_maxx": geom.bounds[2],
                    "roi_maxy": geom.bounds[3],
                    "roi_center_easting": item["cx"],
                    "roi_center_northing": item["cy"],
                    "roi_intersects_target_polygon": geom.intersects(target_geom),
                    "roi_contains_target_centroid": geom.contains(target_geom.centroid),
                }
            )

        for _, cand in ranked_polys.head(20).iterrows():
            top_rows.append(
                {
                    "case_key": key,
                    "base_key": base_key,
                    "candidate_key": cand["candidate_key"],
                    "rank": int(cand.name + 1),
                    "is_target": cand["candidate_key"] == base_key,
                    "rank_score": float(cand["rank_score"]),
                    "roi_intersection_area": float(cand["roi_intersection_area"]),
                    "nearest_roi_center_m": float(cand["nearest_roi_center_m"]),
                }
            )

        rows.append(
            {
                "key": key,
                "base_key": base_key,
                "original_address": row.get("original_address"),
                "best_confidence": best_conf,
                "current_distance_m": row.get("v7_selected_distance_m"),
                "status": "ok",
                "roi_side_m": args.roi_side,
                "roi_count": len(selected_rois),
                "protected_current_roi": any(item["roi_reason"] == "protected_current" for item in selected_rois),
                "evidence_roi_count": sum(1 for item in selected_rois if item["roi_reason"] == "evidence"),
                "roi_sources": " || ".join(f"{idx}:{item['roi_reason']}:{item['sources']}" for idx, item in enumerate(selected_rois, start=1)),
                "plan_roads": " | ".join(plan_roads),
                "address_roads": " | ".join(address_roads),
                "all_roads": " | ".join(all_roads),
                "union_contains_target_centroid": union_roi.contains(target_geom.centroid),
                "union_intersects_target_polygon": union_roi.intersects(target_geom),
                "target_area_inside_ratio": target_geom.intersection(union_roi).area / target_geom.area if target_geom.area else 0.0,
                "union_candidate_polygon_count": int(len(ranked_polys)),
                "target_rank_in_union_candidates": target_rank,
                "top1_candidate_key": ranked_polys.iloc[0]["candidate_key"] if not ranked_polys.empty else None,
                "top5_contains_target": target_rank is not None and target_rank <= 5,
                "top10_contains_target": target_rank is not None and target_rank <= 10,
                "top20_contains_target": target_rank is not None and target_rank <= 20,
            }
        )

    out = pd.DataFrame(rows)
    roi_df = pd.DataFrame(roi_rows)
    top = pd.DataFrame(top_rows)
    csv_path = args.output_prefix.with_suffix(".csv")
    xlsx_path = args.output_prefix.with_suffix(".xlsx")
    rois_path = args.output_prefix.with_name(args.output_prefix.name + "_rois.csv")
    top_path = args.output_prefix.with_name(args.output_prefix.name + "_top20.csv")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    out.to_csv(csv_path, index=False)
    roi_df.to_csv(rois_path, index=False)
    top.to_csv(top_path, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        out.to_excel(writer, sheet_name="multi_roi_summary", index=False)
        roi_df.to_excel(writer, sheet_name="rois", index=False)
        top.to_excel(writer, sheet_name="top20_polygons", index=False)

    ok = out[out["status"] == "ok"].copy()
    low = ok[pd.to_numeric(ok["best_confidence"], errors="coerce") < 75].copy()
    two_anchor = ok[ok["all_roads"].fillna("").map(lambda x: len([p for p in str(x).split("|") if p.strip()]) >= 2)].copy()

    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "cases": int(len(df)),
            "centroid_inside_union": int(df["union_contains_target_centroid"].fillna(False).sum()),
            "polygon_intersects_union": int(df["union_intersects_target_polygon"].fillna(False).sum()),
            "top1_polygon": int((pd.to_numeric(df["target_rank_in_union_candidates"], errors="coerce") <= 1).sum()),
            "top5_polygon": int(df["top5_contains_target"].fillna(False).sum()),
            "top10_polygon": int(df["top10_contains_target"].fillna(False).sum()),
            "top20_polygon": int(df["top20_contains_target"].fillna(False).sum()),
            "mean_candidate_count": float(pd.to_numeric(df["union_candidate_polygon_count"], errors="coerce").mean()),
            "median_candidate_count": float(pd.to_numeric(df["union_candidate_polygon_count"], errors="coerce").median()),
            "mean_current_distance_m": float(pd.to_numeric(df["current_distance_m"], errors="coerce").mean()),
        }

    summary = {
        "full_polygon_count": int(len(polygons)),
        "roi_side_m": args.roi_side,
        "protected_current": not args.no_current_roi,
        "evidence_roi_count": args.evidence_roi_count,
        "all_cases": summarize(ok),
        "low_confidence_cases": summarize(low),
        "cases_with_2plus_road_anchors": summarize(two_anchor),
        "output_csv": str(csv_path),
        "output_rois_csv": str(rois_path),
        "output_top20_csv": str(top_path),
        "output_xlsx": str(xlsx_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
