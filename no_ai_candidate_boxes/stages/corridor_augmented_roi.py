#!/usr/bin/env python3
"""Augment top2 multi-ROI with sampled road-corridor boxes."""

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
from shapely.ops import nearest_points, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox


ROOT = Path(__file__).resolve().parent
ROI_SCRIPT = ROOT / "evidence_roi.py"
spec = importlib.util.spec_from_file_location("roi_v1", ROI_SCRIPT)
roi_v1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["roi_v1"] = roi_v1
spec.loader.exec_module(roi_v1)


DEFAULT_BASE_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_multi_roi_top2_v2.csv"
)
DEFAULT_BASE_ROIS = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_multi_roi_top2_v2_rois.csv"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_corridor_augmented_roi_v1"
)


def roi_square(point: Point, side: float) -> Any:
    half = side / 2.0
    return box(point.x - half, point.y - half, point.x + half, point.y + half)


def compact(text: Any) -> str:
    return roi_v1.compact(text)


def road_corridor_samples(
    points: list[Any],
    roads: list[str],
    road_geoms: dict[str, Any],
    step_m: float,
    pad_m: float,
    max_per_road: int,
    max_total: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for road in roads:
        geom = road_geoms.get(road)
        if geom is None or geom.is_empty or getattr(geom, "length", 0.0) <= 0:
            continue
        croad = compact(road)
        support: list[tuple[float, float, bool]] = []
        for p in points:
            pt = Point(p.x, p.y)
            label_hit = len(croad) >= 7 and croad in compact(p.label)
            roadzone_hit = p.source == "road_zone" and croad in compact(p.label)
            near_line = p.source != "road_zone" and geom.distance(pt) <= 100
            if not (label_hit or roadzone_hit or near_line):
                continue
            try:
                projected = nearest_points(pt, geom)[1]
                measure = geom.project(projected)
            except Exception:
                continue
            support.append((measure, p.weight, label_hit or roadzone_hit))
        if not support:
            continue

        lo = max(0.0, min(m for m, _, _ in support) - pad_m)
        hi = min(float(geom.length), max(m for m, _, _ in support) + pad_m)
        road_samples: list[dict[str, Any]] = []
        measure = lo
        while measure <= hi + 1:
            try:
                point = geom.interpolate(measure)
            except Exception:
                break
            nearest_support = min(abs(measure - m) for m, _, _ in support)
            local_weight = max((w for m, w, _ in support if abs(measure - m) <= 90), default=0.0)
            strong_label = any(strong for m, _, strong in support if abs(measure - m) <= 130)
            score = local_weight + (3.0 if strong_label else 0.0) - nearest_support / 100.0
            road_samples.append(
                {
                    "point": point,
                    "score": score,
                    "road": road,
                    "measure": measure,
                    "support_count": len(support),
                }
            )
            measure += step_m

        road_samples.sort(key=lambda item: item["score"], reverse=True)
        chosen: list[dict[str, Any]] = []
        for item in road_samples:
            if all(item["point"].distance(old["point"]) > 70 for old in chosen):
                chosen.append(item)
            if len(chosen) >= max_per_road:
                break
        samples.extend(chosen)

    best_by_road: list[dict[str, Any]] = []
    remaining_by_road: list[dict[str, Any]] = []
    for road in roads:
        road_items = [item for item in samples if item["road"] == road]
        if not road_items:
            continue
        road_items.sort(key=lambda item: item["score"], reverse=True)
        best_by_road.append(road_items[0])
        remaining_by_road.extend(road_items[1:])

    samples = sorted(best_by_road, key=lambda item: item["score"], reverse=True) + sorted(
        remaining_by_road, key=lambda item: item["score"], reverse=True
    )
    deduped: list[dict[str, Any]] = []
    for item in samples:
        if all(item["point"].distance(old["point"]) > 70 for old in deduped):
            deduped.append(item)
        if len(deduped) >= max_total:
            break
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=roi_v1.DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=roi_v1.DEFAULT_V10_CSV)
    parser.add_argument("--audit-csv", type=Path, default=roi_v1.DEFAULT_AUDIT_CSV)
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--base-rois", type=Path, default=DEFAULT_BASE_ROIS)
    parser.add_argument("--full-gpkg", type=Path, default=roi_v1.DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=roi_v1.DEFAULT_FULL_LAYER)
    parser.add_argument("--sample-gpkg", type=Path, default=roi_v1.DEFAULT_SAMPLE_GPKG)
    parser.add_argument("--sample-layer", default=roi_v1.DEFAULT_SAMPLE_LAYER)
    parser.add_argument("--open-roads", type=Path, default=roi_v1.DEFAULT_OPEN_ROADS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--roi-side", type=float, default=150.0)
    parser.add_argument("--step-m", type=float, default=75.0)
    parser.add_argument("--pad-m", type=float, default=180.0)
    parser.add_argument("--max-per-road", type=int, default=2)
    parser.add_argument("--max-total-samples", type=int, default=8)
    parser.add_argument("--local-bbox", default="")
    args = parser.parse_args()
    roi_v1.MANSFIELD_BBOX = parse_bbox(args.local_bbox, roi_v1.MANSFIELD_BBOX)

    raw_rows = {str(row["key"]): row for row in json.load(args.input_json.open())["rows"]}
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    audit = pd.read_csv(args.audit_csv, dtype={"key": str, "base_key": str})
    audit_by_key = {str(row["key"]): row for _, row in audit.iterrows()}
    base = pd.read_csv(args.base_csv, dtype={"key": str, "base_key": str})
    base_rois = pd.read_csv(args.base_rois, dtype={"case_key": str, "base_key": str})

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
    base_by_key = {str(row["key"]): row for _, row in base.iterrows()}
    for _, row in v10.iterrows():
        key = str(row["key"])
        base_key = str(row.get("base_key") or key).split("_", 1)[0]
        raw = raw_rows.get(key) or raw_rows.get(base_key)
        audit_row = audit_by_key.get(key)
        if audit_row is None:
            audit_row = audit_by_key.get(base_key)
        target_geom = truth.get(base_key)
        if target_geom is None:
            continue

        existing = base_rois[base_rois["case_key"] == key].copy()
        roi_geoms = []
        case_roi_rank = 0
        for _, r in existing.sort_values("roi_rank").iterrows():
            geom = box(float(r["roi_minx"]), float(r["roi_miny"]), float(r["roi_maxx"]), float(r["roi_maxy"]))
            roi_geoms.append(geom)
            case_roi_rank += 1
            roi_rows.append(
                {
                    "case_key": key,
                    "base_key": base_key,
                    "roi_rank": case_roi_rank,
                    "roi_reason": r.get("roi_reason"),
                    "roi_sources": r.get("roi_sources"),
                    "roi_score": r.get("roi_score"),
                    "roi_minx": geom.bounds[0],
                    "roi_miny": geom.bounds[1],
                    "roi_maxx": geom.bounds[2],
                    "roi_maxy": geom.bounds[3],
                    "roi_center_easting": geom.centroid.x,
                    "roi_center_northing": geom.centroid.y,
                    "roi_intersects_target_polygon": geom.intersects(target_geom),
                    "roi_contains_target_centroid": geom.contains(target_geom.centroid),
                }
            )

        plan_roads = roi_v1.split_anchors(audit_row.get("matched_roads") if audit_row is not None else "")
        address_roads = roi_v1.extract_address_roads(row.get("original_address"), set(road_geoms))
        all_roads = []
        for road in [*address_roads, *plan_roads]:
            if road not in all_roads:
                all_roads.append(road)

        points = roi_v1.evidence_points_for_case(row, raw, audit_row, road_geoms)
        samples = road_corridor_samples(
            points,
            all_roads,
            road_geoms,
            step_m=args.step_m,
            pad_m=args.pad_m,
            max_per_road=args.max_per_road,
            max_total=args.max_total_samples,
        )
        for item in samples:
            geom = roi_square(item["point"], args.roi_side)
            if any(geom.centroid.distance(old.centroid) <= 70 for old in roi_geoms):
                continue
            roi_geoms.append(geom)
            case_roi_rank += 1
            roi_rows.append(
                {
                    "case_key": key,
                    "base_key": base_key,
                    "roi_rank": case_roi_rank,
                    "roi_reason": "corridor_sample",
                    "roi_sources": f"line:{item['road']}",
                    "roi_score": item["score"],
                    "roi_minx": geom.bounds[0],
                    "roi_miny": geom.bounds[1],
                    "roi_maxx": geom.bounds[2],
                    "roi_maxy": geom.bounds[3],
                    "roi_center_easting": item["point"].x,
                    "roi_center_northing": item["point"].y,
                    "roi_intersects_target_polygon": geom.intersects(target_geom),
                    "roi_contains_target_centroid": geom.contains(target_geom.centroid),
                }
            )

        if not roi_geoms:
            continue
        union_roi = unary_union(roi_geoms)
        candidates = polygons[polygons.geometry.intersects(union_roi)].copy()
        candidates["candidate_key"] = candidates["unique_key"].astype(str)
        target_hit = candidates[candidates["candidate_key"] == base_key] if not candidates.empty else pd.DataFrame()
        target_rank = None
        if not candidates.empty:
            centers = [geom.centroid for geom in roi_geoms]
            candidates["nearest_roi_center_m"] = candidates.geometry.centroid.map(lambda geom: min(geom.distance(center) for center in centers))
            candidates["roi_intersection_area"] = candidates.geometry.intersection(union_roi).area
            candidates = candidates.sort_values(["roi_intersection_area", "nearest_roi_center_m"], ascending=[False, True]).reset_index(drop=True)
            target_hit = candidates[candidates["candidate_key"] == base_key]
            target_rank = int(target_hit.index[0] + 1) if not target_hit.empty else None

        best_conf = roi_v1.parse_float(raw.get("best_confidence") if raw else None)
        rows.append(
            {
                "key": key,
                "base_key": base_key,
                "original_address": row.get("original_address"),
                "best_confidence": best_conf,
                "status": "ok",
                "roi_count": len(roi_geoms),
                "added_corridor_roi_count": len(roi_geoms) - len(existing),
                "all_roads": " | ".join(all_roads),
                "union_contains_target_centroid": union_roi.contains(target_geom.centroid),
                "union_intersects_target_polygon": union_roi.intersects(target_geom),
                "union_candidate_polygon_count": int(len(candidates)),
                "target_rank_in_union_candidates": target_rank,
                "base_union_hit": str(base_by_key.get(key, {}).get("union_intersects_target_polygon")),
            }
        )

    out = pd.DataFrame(rows)
    roi_df = pd.DataFrame(roi_rows)
    csv_path = args.output_prefix.with_suffix(".csv")
    rois_path = args.output_prefix.with_name(args.output_prefix.name + "_rois.csv")
    xlsx_path = args.output_prefix.with_suffix(".xlsx")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    out.to_csv(csv_path, index=False)
    roi_df.to_csv(rois_path, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        out.to_excel(writer, sheet_name="summary", index=False)
        roi_df.to_excel(writer, sheet_name="rois", index=False)

    ok = out[out["status"] == "ok"].copy()
    low = ok[pd.to_numeric(ok["best_confidence"], errors="coerce") < 75].copy()

    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "cases": int(len(df)),
            "centroid_inside_union": int(df["union_contains_target_centroid"].fillna(False).sum()),
            "polygon_intersects_union": int(df["union_intersects_target_polygon"].fillna(False).sum()),
            "mean_candidate_count": float(pd.to_numeric(df["union_candidate_polygon_count"], errors="coerce").mean()),
            "median_candidate_count": float(pd.to_numeric(df["union_candidate_polygon_count"], errors="coerce").median()),
            "mean_added_corridor_roi_count": float(pd.to_numeric(df["added_corridor_roi_count"], errors="coerce").mean()),
        }

    summary = {
        "full_polygon_count": int(len(polygons)),
        "roi_side_m": args.roi_side,
        "step_m": args.step_m,
        "pad_m": args.pad_m,
        "max_per_road": args.max_per_road,
        "max_total_samples": args.max_total_samples,
        "all_cases": summarize(ok),
        "low_confidence_cases": summarize(low),
        "output_csv": str(csv_path),
        "output_rois_csv": str(rois_path),
        "output_xlsx": str(xlsx_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
