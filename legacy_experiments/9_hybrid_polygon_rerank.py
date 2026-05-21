#!/usr/bin/env python3
"""Hybrid Mansfield polygon candidate reranker.

Production-visible candidate sources:
- raw Mansfield WFS Land/Building polygons;
- shared-edge merged Mansfield WFS Land/Building polygons;
- Mansfield council cadastral parcels clipped to the work bbox.

The manual polygon-link layer is used only for offline evaluation.
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


ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "8_candidate_base_layer_rerank_experiment.py"
spec = importlib.util.spec_from_file_location("base_rerank", BASE_SCRIPT)
base_rerank = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["base_rerank"] = base_rerank
spec.loader.exec_module(base_rerank)


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TAG = "mansfield-manual-polygon-link_random1200_seed42_43_combined"
DEFAULT_INPUT_JSON = TMP / f"{DEFAULT_TAG}_gemini.json"
DEFAULT_V10_CSV = TMP / f"{DEFAULT_TAG}_ocr_openroads_v10.csv"
DEFAULT_CASE_SUMMARY_CSV = TMP / f"{DEFAULT_TAG}_corridor_augmented_roi_step50_v1_all_expanded.csv"
DEFAULT_ROIS_CSV = TMP / f"{DEFAULT_TAG}_corridor_augmented_roi_step50_v1_rois.csv"
DEFAULT_TRUTH_GPKG = TMP / f"{DEFAULT_TAG}.gpkg"
DEFAULT_TRUTH_LAYER = "random1200_seed42_43_combined"
DEFAULT_RAW_WFS_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg")
DEFAULT_RAW_WFS_LAYER = "mansfield_polygons_in_buffers"
DEFAULT_MERGED_WFS_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg")
DEFAULT_MERGED_WFS_LAYER = "mansfield_polygons_in_buffers_merged"
DEFAULT_COUNCIL_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_councils_land.gpkg")
DEFAULT_COUNCIL_LAYER = "cadastral_parcels"
DEFAULT_UPRN_GPKG = Path("/data/base-data/osopenuprn_202602.gpkg")
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OUTPUT_PREFIX = TMP / f"{DEFAULT_TAG}_hybrid_polygon_rerank_v1_top5roi"
MANSFIELD_BBOX = base_rerank.MANSFIELD_BBOX


def parse_bbox_arg(value: str | None, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not value:
        return default
    parts = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"--local-bbox must contain four comma-separated numbers, got: {value}")
    return (parts[0], parts[1], parts[2], parts[3])


def load_27700(path: Path, layer: str, **kwargs: Any) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, **kwargs)
    if gdf.crs is None:
        gdf = gdf.set_crs(27700)
    else:
        gdf = gdf.to_crs(27700)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def load_wfs_source(path: Path, layer: str, source: str, theme_regex: str | None) -> gpd.GeoDataFrame:
    gdf = base_rerank.load_wfs_polygons(path, layer, theme_regex)
    gdf["candidate_source"] = source
    gdf["candidate_id"] = source + ":" + gdf["candidate_id"].astype(str)
    return gdf


def load_council_source(path: Path, layer: str, bbox_tuple: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    columns = ["gml_id", "INSPIREID", "LABEL", "NATIONALCADASTRALREFERENCE"]
    gdf = load_27700(path, layer, bbox=bbox_tuple, columns=columns)
    gdf = gdf.reset_index(drop=True)
    ids = gdf.get("INSPIREID", pd.Series([""] * len(gdf))).fillna("").astype(str)
    missing = ids.eq("") | ids.eq("nan")
    ids.loc[missing] = gdf.get("gml_id", pd.Series([""] * len(gdf))).fillna("").astype(str).loc[missing]
    missing = ids.eq("") | ids.eq("nan")
    ids.loc[missing] = [f"row{i}" for i in gdf.index[missing]]
    gdf["candidate_source"] = "council_cadastral"
    gdf["candidate_id"] = "council:" + ids.astype(str)
    gdf["candidate_toid"] = ""
    gdf["candidate_gmlid"] = gdf.get("gml_id")
    gdf["candidate_objectid"] = gdf.get("INSPIREID")
    gdf["candidate_theme"] = "Cadastral"
    gdf["candidate_descriptive_group"] = "Cadastral Parcel"
    gdf["candidate_descriptive_term"] = ""
    gdf["candidate_make"] = ""
    gdf["candidate_area_m2"] = gdf.geometry.area
    centroids = gdf.geometry.centroid
    gdf["candidate_centroid_easting"] = centroids.x
    gdf["candidate_centroid_northing"] = centroids.y
    return gdf


def load_truth(path: Path, layer: str) -> dict[str, Any]:
    gdf = load_27700(path, layer, columns=["unique_key"])
    return {str(row["unique_key"]): row.geometry for _, row in gdf.iterrows()}


def source_context_score(row: dict[str, Any]) -> tuple[float, str]:
    source = str(row.get("candidate_source") or "")
    address = str(row.get("original_address") or "").lower()
    group = str(row.get("candidate_descriptive_group") or "").upper()
    theme = str(row.get("candidate_theme") or "").upper()
    area = float(row.get("candidate_area_m2") or 0.0)
    relation_words = ("land", "rear", "adjacent", "adjoining", "between", "site", "plot", "plots", "yard", "field")
    is_land_style = any(word in address for word in relation_words)
    is_building = "BUILDING" in theme or "BUILDING" in group

    score = 0.0
    reasons: list[str] = []
    if source == "council_cadastral":
        score += 26.0
        reasons.append("source_council:+26")
        if 35 <= area <= 4500:
            score += 9.0
            reasons.append("council_area_plausible:+9")
        if is_land_style:
            score += 16.0
            reasons.append("land_style_council:+16")
    elif source == "wfs_merged":
        score += 14.0
        reasons.append("source_wfs_merged:+14")
        if is_land_style:
            score += 10.0
            reasons.append("land_style_wfs_merged:+10")
    elif source == "wfs_raw":
        if is_land_style and is_building:
            score -= 10.0
            reasons.append("land_style_raw_building:-10")

    if source == "council_cadastral" and row.get("uprn_count"):
        score += min(14.0, 4.0 + 2.0 * int(row["uprn_count"]))
        reasons.append("council_uprn_inside_bonus")
    return score, ";".join(reasons)


def score_candidate(row: dict[str, Any]) -> tuple[float, str]:
    score, reasons = base_rerank.score_candidate(row)
    source_score, source_reasons = source_context_score(row)
    score += source_score
    if source_reasons:
        reasons = f"{reasons};{source_reasons}" if reasons else source_reasons
    return score, reasons


def truth_metrics(geom: Any, target_geom: Any) -> dict[str, Any]:
    if target_geom is None or geom is None or geom.is_empty:
        return {
            "truth_intersects": False,
            "truth_centroid_inside": False,
            "truth_overlap_area_m2": 0.0,
            "truth_candidate_overlap_ratio": 0.0,
            "truth_cover_ratio": 0.0,
            "truth_iou": 0.0,
        }
    intersects = bool(geom.intersects(target_geom))
    centroid_inside = bool(target_geom.contains(geom.centroid))
    if not intersects:
        return {
            "truth_intersects": False,
            "truth_centroid_inside": centroid_inside,
            "truth_overlap_area_m2": 0.0,
            "truth_candidate_overlap_ratio": 0.0,
            "truth_cover_ratio": 0.0,
            "truth_iou": 0.0,
        }
    inter_area = float(geom.intersection(target_geom).area)
    candidate_area = float(geom.area)
    truth_area = float(target_geom.area)
    union_area = candidate_area + truth_area - inter_area
    return {
        "truth_intersects": inter_area > 0,
        "truth_centroid_inside": centroid_inside,
        "truth_overlap_area_m2": inter_area,
        "truth_candidate_overlap_ratio": inter_area / candidate_area if candidate_area else 0.0,
        "truth_cover_ratio": inter_area / truth_area if truth_area else 0.0,
        "truth_iou": inter_area / union_area if union_area else 0.0,
    }


def summarize(case_df: pd.DataFrame, label: str) -> dict[str, Any]:
    ok = case_df[case_df["status"] == "ok"].copy()
    if ok.empty:
        return {"cases": 0}
    rank_col = f"{label}_rank"
    present_col = f"{label}_present"
    rank = pd.to_numeric(ok[rank_col], errors="coerce")
    present = ok[present_col].fillna(False)
    return {
        "cases": int(len(ok)),
        "present": int(present.sum()),
        "present_rate": float(present.mean()),
        "top1": int((rank == 1).sum()),
        "top1_rate_all": float((rank == 1).sum() / len(ok)),
        "top3": int((rank <= 3).sum()),
        "top3_rate_all": float((rank <= 3).sum() / len(ok)),
        "top5": int((rank <= 5).sum()),
        "top5_rate_all": float((rank <= 5).sum() / len(ok)),
        "top10": int((rank <= 10).sum()),
        "top10_rate_all": float((rank <= 10).sum() / len(ok)),
        "top20": int((rank <= 20).sum()),
        "top20_rate_all": float((rank <= 20).sum() / len(ok)),
        "top50": int((rank <= 50).sum()),
        "top50_rate_all": float((rank <= 50).sum() / len(ok)),
        "mean_candidate_count": float(pd.to_numeric(ok["candidate_polygon_count"], errors="coerce").mean()),
        "median_candidate_count": float(pd.to_numeric(ok["candidate_polygon_count"], errors="coerce").median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Mansfield polygon candidate reranker.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=DEFAULT_V10_CSV)
    parser.add_argument("--case-summary-csv", type=Path, default=DEFAULT_CASE_SUMMARY_CSV)
    parser.add_argument("--rois-csv", type=Path, default=DEFAULT_ROIS_CSV)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKG)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--no-truth", action="store_true", help="Run production mode without offline truth polygons.")
    parser.add_argument("--raw-wfs-gpkg", type=Path, default=DEFAULT_RAW_WFS_GPKG)
    parser.add_argument("--raw-wfs-layer", default=DEFAULT_RAW_WFS_LAYER)
    parser.add_argument("--merged-wfs-gpkg", type=Path, default=DEFAULT_MERGED_WFS_GPKG)
    parser.add_argument("--merged-wfs-layer", default=DEFAULT_MERGED_WFS_LAYER)
    parser.add_argument("--council-gpkg", type=Path, default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--council-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--uprn-gpkg", type=Path, default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--theme-filter-regex", default="Land|Building")
    parser.add_argument("--max-roi-rank", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-output-rank", type=int, default=50)
    parser.add_argument("--disable-range-anchor", action="store_true")
    parser.add_argument("--local-bbox", default="", help="Optional EPSG:27700 bbox: minx,miny,maxx,maxy.")
    args = parser.parse_args()

    global MANSFIELD_BBOX
    MANSFIELD_BBOX = parse_bbox_arg(args.local_bbox, MANSFIELD_BBOX)
    base_rerank.MANSFIELD_BBOX = MANSFIELD_BBOX
    base_rerank.roi_v1.MANSFIELD_BBOX = MANSFIELD_BBOX

    print("loading tabular inputs...")
    raw_rows = base_rerank.load_raw_rows(args.input_json)
    raw_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows.values():
        source_key = base_rerank.clean_text(row.get("_source_unique_key") or row.get("key"))
        if source_key:
            raw_rows_by_source.setdefault(source_key, []).append(row)

    range_anchor_cache: dict[str, dict[str, Any] | None] = {}
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    v10_by_key = {str(row["key"]): row for _, row in v10.iterrows()}
    case_summary = pd.read_csv(args.case_summary_csv, dtype={"key": str, "base_key": str})
    rois = pd.read_csv(args.rois_csv, dtype={"case_key": str, "base_key": str})
    rois["roi_rank"] = pd.to_numeric(rois["roi_rank"], errors="coerce")
    rois["roi_score"] = pd.to_numeric(rois["roi_score"], errors="coerce").fillna(0.0)
    rois = rois[rois["roi_rank"] <= args.max_roi_rank].copy()

    print("loading production-visible candidate layers...")
    raw_wfs = load_wfs_source(args.raw_wfs_gpkg, args.raw_wfs_layer, "wfs_raw", args.theme_filter_regex)
    merged_wfs = load_wfs_source(args.merged_wfs_gpkg, args.merged_wfs_layer, "wfs_merged", args.theme_filter_regex)
    council = load_council_source(args.council_gpkg, args.council_layer, MANSFIELD_BBOX)
    uprn = base_rerank.load_uprn_points(args.uprn_gpkg, args.uprn_layer)
    road_geoms = base_rerank.roi_v1.load_road_geoms(args.open_roads)
    road_names = set(road_geoms)
    truth = {} if args.no_truth else load_truth(args.truth_gpkg, args.truth_layer)

    sources = [
        ("wfs_raw", raw_wfs, raw_wfs.sindex),
        ("wfs_merged", merged_wfs, merged_wfs.sindex),
        ("council_cadastral", council, council.sindex),
    ]
    uprn_sindex = uprn.sindex
    print(
        "loaded "
        f"raw_wfs={len(raw_wfs)} merged_wfs={len(merged_wfs)} "
        f"council={len(council)} uprn={len(uprn)} truth={len(truth)} rois={len(rois)}"
    )

    top_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    source_candidate_counts: dict[str, int] = {name: 0 for name, _, _ in sources}
    processed = 0

    for _, case in case_summary.iterrows():
        key = str(case["key"])
        base_key = str(case.get("base_key") or key).split("_", 1)[0]
        if str(case.get("status") or "") != "ok":
            case_rows.append({"key": key, "base_key": base_key, "status": case.get("status")})
            continue
        processed += 1
        if args.max_cases and processed > args.max_cases:
            break

        target_geom = truth.get(base_key)
        case_rois = rois[rois["case_key"] == key].sort_values("roi_rank").copy()
        if case_rois.empty:
            case_rows.append({"key": key, "base_key": base_key, "status": "no_rois"})
            continue

        roi_items: list[dict[str, Any]] = []
        roi_geoms = []
        for _, roi_row in case_rois.iterrows():
            geom = box(
                float(roi_row["roi_minx"]),
                float(roi_row["roi_miny"]),
                float(roi_row["roi_maxx"]),
                float(roi_row["roi_maxy"]),
            )
            roi_geoms.append(geom)
            roi_items.append(
                {
                    "rank": int(roi_row["roi_rank"]),
                    "reason": str(roi_row.get("roi_reason") or ""),
                    "sources": str(roi_row.get("roi_sources") or ""),
                    "score": float(roi_row.get("roi_score") or 0.0),
                    "geom": geom,
                    "center": geom.centroid,
                }
            )
        union_roi = unary_union(roi_geoms)

        raw = raw_rows.get(key) or raw_rows.get(base_key) or {}
        v10_row = v10_by_key.get(key)
        best_point = base_rerank.point_from_xy(raw.get("best_easting_27700_final"), raw.get("best_northing_27700_final"))
        if best_point is None and v10_row is not None:
            best_point = base_rerank.point_from_xy(v10_row.get("baseline_easting"), v10_row.get("baseline_northing"))
        v7_point = (
            base_rerank.point_from_xy(v10_row.get("v7_selected_easting"), v10_row.get("v7_selected_northing"))
            if v10_row is not None
            else None
        )
        roads = base_rerank.build_case_roads(case, v10_row, road_names)
        parent_raw = raw_rows.get(base_key) or raw
        range_anchor = None
        range_anchor_point = None
        if not args.disable_range_anchor and parent_raw:
            if base_key not in range_anchor_cache:
                range_anchor_cache[base_key] = base_rerank.build_range_anchor(
                    parent_raw, raw_rows_by_source.get(base_key, []), base_key
                )
            range_anchor = range_anchor_cache.get(base_key)
            if range_anchor:
                range_anchor_point = Point(float(range_anchor["x"]), float(range_anchor["y"]))

        per_case_rows: list[dict[str, Any]] = []
        for source_name, source_gdf, source_sindex in sources:
            candidate_idx = list(source_sindex.query(union_roi, predicate="intersects"))
            if not candidate_idx:
                continue
            source_candidates = source_gdf.iloc[candidate_idx].copy()
            source_candidate_counts[source_name] += int(len(source_candidates))
            for _, cand in source_candidates.iterrows():
                geom = cand.geometry
                centroid = geom.centroid
                roi_hits = [item for item in roi_items if geom.intersects(item["geom"])]
                if not roi_hits:
                    continue
                intersection_areas = [float(geom.intersection(item["geom"]).area) for item in roi_hits]
                center_distances = [float(centroid.distance(item["center"])) for item in roi_hits]
                road_distance, road_near_50, road_near_100 = base_rerank.road_features(geom, roads, road_geoms)
                uprn_data = base_rerank.uprn_features(geom, uprn, uprn_sindex, best_point, v7_point)
                truth_data = truth_metrics(geom, target_geom)
                truth_data["truth_available"] = target_geom is not None

                row = {
                    "case_key": key,
                    "base_key": base_key,
                    "sample_split": case.get("sample_split"),
                    "original_address": case.get("original_address"),
                    "best_confidence": base_rerank.parse_float(raw.get("best_confidence"))
                    or base_rerank.parse_float(case.get("best_confidence")),
                    "best_source_final": raw.get("best_source_final"),
                    "best_selection_category": raw.get("best_selection_category"),
                    "candidate_source": source_name,
                    "candidate_id": cand.get("candidate_id"),
                    "candidate_toid": cand.get("candidate_toid") or cand.get("TOID"),
                    "candidate_gmlid": cand.get("candidate_gmlid") or cand.get("GmlID") or cand.get("gml_id"),
                    "candidate_objectid": cand.get("candidate_objectid") or cand.get("OBJECTID") or cand.get("INSPIREID"),
                    "candidate_theme": cand.get("candidate_theme") or cand.get("Theme"),
                    "candidate_descriptive_group": cand.get("candidate_descriptive_group") or cand.get("DescriptiveGroup"),
                    "candidate_descriptive_term": cand.get("candidate_descriptive_term") or cand.get("DescriptiveTerm"),
                    "candidate_make": cand.get("candidate_make") or cand.get("Make"),
                    "candidate_area_m2": float(cand["candidate_area_m2"]),
                    "candidate_centroid_easting": float(cand["candidate_centroid_easting"]),
                    "candidate_centroid_northing": float(cand["candidate_centroid_northing"]),
                    "covers_best_point": bool(best_point is not None and geom.covers(best_point)),
                    "distance_to_best_point_m": float(geom.distance(best_point)) if best_point is not None else None,
                    "covers_v7_point": bool(v7_point is not None and geom.covers(v7_point)),
                    "distance_to_v7_point_m": float(geom.distance(v7_point)) if v7_point is not None else None,
                    "roi_intersects_count": len(roi_hits),
                    "roi_min_rank": min(item["rank"] for item in roi_hits),
                    "roi_hit_reasons": "|".join(sorted({item["reason"] for item in roi_hits if item["reason"]})),
                    "intersects_top1_roi": any(item["rank"] == 1 for item in roi_hits),
                    "intersects_protected_current_roi": any(item["reason"] == "protected_current" for item in roi_hits),
                    "intersects_evidence_roi": any(item["reason"] == "evidence" for item in roi_hits),
                    "intersects_corridor_roi": any(item["reason"] == "corridor_sample" for item in roi_hits),
                    "roi_total_intersection_area_m2": sum(intersection_areas),
                    "roi_rank_weighted_score": sum((item["score"] + 8.0) / max(1, item["rank"]) for item in roi_hits),
                    "distance_to_nearest_roi_center_m": min(center_distances),
                    "mentioned_roads": " | ".join(roads),
                    "distance_to_mentioned_road_m": road_distance,
                    "mentioned_road_near_count_50m": road_near_50,
                    "mentioned_road_near_count_100m": road_near_100,
                    "has_range_anchor": bool(range_anchor_point is not None),
                    "range_anchor_easting": range_anchor.get("x") if range_anchor else None,
                    "range_anchor_northing": range_anchor.get("y") if range_anchor else None,
                    "range_anchor_relation": range_anchor.get("relation") if range_anchor else "",
                    "range_anchor_number_kind": range_anchor.get("number_kind") if range_anchor else "",
                    "range_anchor_parity_corrected": bool(range_anchor.get("parity_corrected")) if range_anchor else False,
                    "range_anchor_requested_numbers": range_anchor.get("requested_numbers") if range_anchor else "",
                    "range_anchor_covered_numbers": range_anchor.get("covered_numbers") if range_anchor else "",
                    "range_anchor_endpoint_hits": range_anchor.get("endpoint_hits") if range_anchor else "",
                    "range_anchor_candidate_count": range_anchor.get("candidate_count") if range_anchor else 0,
                    "range_anchor_cluster_count": range_anchor.get("cluster_count") if range_anchor else 0,
                    "range_anchor_cluster_radius_m": range_anchor.get("cluster_radius_m") if range_anchor else None,
                    "range_anchor_cluster_score": range_anchor.get("cluster_score") if range_anchor else None,
                    "range_anchor_complete": bool(range_anchor.get("complete")) if range_anchor else False,
                    "range_anchor_top_addresses": range_anchor.get("top_addresses") if range_anchor else "",
                    "covers_range_anchor": bool(range_anchor_point is not None and geom.covers(range_anchor_point)),
                    "distance_to_range_anchor_m": float(geom.distance(range_anchor_point)) if range_anchor_point is not None else None,
                    **truth_data,
                    **uprn_data,
                }
                row["rule_score"], row["rule_score_reasons"] = score_candidate(row)
                per_case_rows.append(row)

        if not per_case_rows:
            case_rows.append({"key": key, "base_key": base_key, "status": "no_candidates"})
            continue

        per_case_rows.sort(
            key=lambda item: (
                -float(item["rule_score"]),
                float(item["distance_to_v7_point_m"] if item["distance_to_v7_point_m"] is not None else 1e9),
                float(item["distance_to_best_point_m"] if item["distance_to_best_point_m"] is not None else 1e9),
                float(item["candidate_area_m2"]),
                str(item["candidate_id"]),
            )
        )
        intersects_rank = None
        centroid_rank = None
        iou50_rank = None
        best_iou = 0.0
        best_iou_rank = None
        source_counts: dict[str, int] = {}
        for idx, row in enumerate(per_case_rows, start=1):
            row["rule_rank"] = idx
            source_counts[row["candidate_source"]] = source_counts.get(row["candidate_source"], 0) + 1
            if intersects_rank is None and row["truth_intersects"]:
                intersects_rank = idx
            if centroid_rank is None and row["truth_centroid_inside"]:
                centroid_rank = idx
            if iou50_rank is None and row["truth_iou"] >= 0.50:
                iou50_rank = idx
            if row["truth_iou"] > best_iou:
                best_iou = float(row["truth_iou"])
                best_iou_rank = idx
            if idx <= args.max_output_rank:
                top_rows.append(row.copy())

        top1 = per_case_rows[0]
        top2_score = per_case_rows[1]["rule_score"] if len(per_case_rows) > 1 else None
        case_rows.append(
            {
                "key": key,
                "base_key": base_key,
                "status": "ok",
                "sample_split": case.get("sample_split"),
                "original_address": case.get("original_address"),
                "best_confidence": base_rerank.parse_float(raw.get("best_confidence"))
                or base_rerank.parse_float(case.get("best_confidence")),
                "roi_union_intersects_truth": base_rerank.safe_bool(case.get("union_intersects_target_polygon")),
                "max_roi_rank": args.max_roi_rank,
                "candidate_polygon_count": len(per_case_rows),
                "raw_wfs_candidate_count": source_counts.get("wfs_raw", 0),
                "merged_wfs_candidate_count": source_counts.get("wfs_merged", 0),
                "council_candidate_count": source_counts.get("council_cadastral", 0),
                "intersects_present": intersects_rank is not None,
                "intersects_rank": intersects_rank,
                "centroid_present": centroid_rank is not None,
                "centroid_rank": centroid_rank,
                "iou50_present": iou50_rank is not None,
                "iou50_rank": iou50_rank,
                "best_iou": best_iou,
                "best_iou_rank": best_iou_rank,
                "top1_candidate_id": top1["candidate_id"],
                "top1_source": top1["candidate_source"],
                "top1_theme": top1["candidate_theme"],
                "top1_group": top1["candidate_descriptive_group"],
                "top1_intersects_truth": bool(top1["truth_intersects"]),
                "top1_centroid_inside_truth": bool(top1["truth_centroid_inside"]),
                "top1_iou": float(top1["truth_iou"]),
                "top1_truth_cover": float(top1["truth_cover_ratio"]),
                "top1_candidate_cover": float(top1["truth_candidate_overlap_ratio"]),
                "top1_score": top1["rule_score"],
                "top2_score": top2_score,
                "top1_margin": top1["rule_score"] - top2_score if top2_score is not None else None,
                "top1_uprn_count": top1["uprn_count"],
                "top1_covers_v7_point": top1["covers_v7_point"],
                "top1_covers_best_point": top1["covers_best_point"],
                "has_range_anchor": bool(range_anchor is not None),
                "range_anchor_relation": range_anchor.get("relation") if range_anchor else "",
                "range_anchor_easting": range_anchor.get("x") if range_anchor else None,
                "range_anchor_northing": range_anchor.get("y") if range_anchor else None,
                "range_anchor_parity_corrected": bool(range_anchor.get("parity_corrected")) if range_anchor else False,
                "range_anchor_covered_numbers": range_anchor.get("covered_numbers") if range_anchor else "",
                "range_anchor_endpoint_hits": range_anchor.get("endpoint_hits") if range_anchor else "",
                "range_anchor_cluster_radius_m": range_anchor.get("cluster_radius_m") if range_anchor else None,
                "range_anchor_complete": bool(range_anchor.get("complete")) if range_anchor else False,
                "range_anchor_top_addresses": range_anchor.get("top_addresses") if range_anchor else "",
                "top1_covers_range_anchor": top1["covers_range_anchor"],
                "top1_distance_to_range_anchor_m": top1["distance_to_range_anchor_m"],
            }
        )

    top_df = pd.DataFrame(top_rows)
    case_df = pd.DataFrame(case_rows)
    out_prefix = args.output_prefix
    top_csv = out_prefix.with_name(out_prefix.name + f"_top{args.max_output_rank}.csv")
    top20_csv = out_prefix.with_name(out_prefix.name + "_top20.csv")
    cases_csv = out_prefix.with_name(out_prefix.name + "_cases.csv")
    xlsx_path = out_prefix.with_suffix(".xlsx")
    summary_path = out_prefix.with_suffix(".summary.json")
    top_df.to_csv(top_csv, index=False)
    top_df[top_df["rule_rank"] <= 20].to_csv(top20_csv, index=False)
    case_df.to_csv(cases_csv, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        case_df.to_excel(writer, sheet_name="cases", index=False)
        top_df[top_df["rule_rank"] <= 20].to_excel(writer, sheet_name="top20", index=False)

    ok = case_df[case_df["status"] == "ok"].copy()
    low_conf = ok[pd.to_numeric(ok.get("best_confidence"), errors="coerce") < 75].copy()
    high_conf = ok[pd.to_numeric(ok.get("best_confidence"), errors="coerce") >= 80].copy()
    summary = {
        "candidate_sources": {
            "raw_wfs": str(args.raw_wfs_gpkg),
            "merged_wfs": str(args.merged_wfs_gpkg),
            "council_cadastral": str(args.council_gpkg),
        },
        "truth_source_for_evaluation_only": "" if args.no_truth else str(args.truth_gpkg),
        "truth_enabled": not bool(args.no_truth),
        "max_roi_rank": int(args.max_roi_rank),
        "raw_wfs_polygon_count_loaded": int(len(raw_wfs)),
        "merged_wfs_polygon_count_loaded": int(len(merged_wfs)),
        "council_polygon_count_loaded": int(len(council)),
        "uprn_point_count_loaded": int(len(uprn)),
        "source_candidate_counts_before_roi_hit_filter": source_candidate_counts,
        "all_cases_intersects_metric": summarize(case_df, "intersects"),
        "all_cases_centroid_metric": summarize(case_df, "centroid"),
        "all_cases_iou50_metric": summarize(case_df, "iou50"),
        "low_confidence_lt75_intersects_metric": summarize(low_conf, "intersects"),
        "high_confidence_ge80_intersects_metric": summarize(high_conf, "intersects"),
        "top1_source_counts": ok["top1_source"].value_counts(dropna=False).to_dict() if "top1_source" in ok else {},
        "output_cases_csv": str(cases_csv),
        "output_top20_csv": str(top20_csv),
        "output_topn_csv": str(top_csv),
        "output_xlsx": str(xlsx_path),
        "top_rows_written": int(len(top_df)),
        "range_anchor_cases": int(ok["has_range_anchor"].fillna(False).sum()) if "has_range_anchor" in ok else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
