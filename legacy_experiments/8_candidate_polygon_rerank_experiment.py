#!/usr/bin/env python3
"""Diagnostic-only manual-layer candidate reranker.

This script intentionally expands ROI boxes against the manual polygon-link
layer.  That leaks the hidden evaluation layer, so it is useful only as an
upper-bound/diagnostic experiment.  Production-visible candidate generation is
implemented in `8_candidate_base_layer_rerank_experiment.py`.
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
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ROI_SCRIPT = PROJECT_ROOT / "no_ai_candidate_boxes" / "stages" / "evidence_roi.py"
spec = importlib.util.spec_from_file_location("roi_v1", ROI_SCRIPT)
roi_v1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["roi_v1"] = roi_v1
spec.loader.exec_module(roi_v1)


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TAG = "mansfield-manual-polygon-link_random1200_seed42_43_combined"
DEFAULT_INPUT_JSON = TMP / f"{DEFAULT_TAG}_gemini.json"
DEFAULT_V10_CSV = TMP / f"{DEFAULT_TAG}_ocr_openroads_v10.csv"
DEFAULT_CASE_SUMMARY_CSV = TMP / f"{DEFAULT_TAG}_corridor_augmented_roi_step50_v1_all_expanded.csv"
DEFAULT_ROIS_CSV = TMP / f"{DEFAULT_TAG}_corridor_augmented_roi_step50_v1_rois.csv"
DEFAULT_FULL_GPKG = Path("/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg")
DEFAULT_FULL_LAYER = "mansfield-manual-polygon-link"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OUTPUT_PREFIX = TMP / f"{DEFAULT_TAG}_candidate_polygon_rerank_v2"


def parse_float(value: Any) -> float | None:
    return roi_v1.parse_float(value)


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def point_from_xy(x: Any, y: Any) -> Point | None:
    px = parse_float(x)
    py = parse_float(y)
    if px is None or py is None:
        return None
    return Point(px, py)


def norm(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", norm(text))


TEXT_STOP = {
    "THE",
    "AND",
    "FOR",
    "OF",
    "TO",
    "AT",
    "IN",
    "ON",
    "LAND",
    "SITE",
    "PLOT",
    "PLOTS",
    "UNIT",
    "UNITS",
    "ADJACENT",
    "ADJ",
    "REAR",
    "OFF",
    "MANSFIELD",
    "NOTTS",
    "NOTTINGHAMSHIRE",
    "FOREST",
    "TOWN",
    "WOODHOUSE",
}


def text_tokens(text: Any) -> set[str]:
    return {part for part in norm(text).split() if len(part) >= 3 and part not in TEXT_STOP}


def number_tokens(text: Any) -> set[str]:
    return set(re.findall(r"\b\d+[A-Z]?\b", norm(text)))


def text_match_features(original: Any, candidate: Any) -> dict[str, Any]:
    original_norm = norm(original)
    candidate_norm = norm(candidate)
    original_compact = compact(original)
    candidate_compact = compact(candidate)
    original_tokens = text_tokens(original)
    candidate_tokens = text_tokens(candidate)
    overlap = original_tokens & candidate_tokens
    union = original_tokens | candidate_tokens
    original_numbers = number_tokens(original)
    candidate_numbers = number_tokens(candidate)
    number_overlap = original_numbers & candidate_numbers
    compact_contains = bool(
        original_compact
        and candidate_compact
        and (original_compact in candidate_compact or candidate_compact in original_compact)
    )
    return {
        "candidate_text_token_overlap": len(overlap),
        "candidate_text_jaccard": len(overlap) / len(union) if union else 0.0,
        "candidate_text_number_overlap": len(number_overlap),
        "candidate_text_number_mismatch": bool(original_numbers and candidate_numbers and not number_overlap),
        "candidate_text_compact_contains": compact_contains,
        "candidate_text_exact_norm": bool(original_norm and original_norm == candidate_norm),
    }


def score_text(features: dict[str, Any]) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    if features["candidate_text_exact_norm"]:
        score += 55.0
        reasons.append("candidate_text_exact:+55")
    elif features["candidate_text_compact_contains"]:
        score += 38.0
        reasons.append("candidate_text_contains:+38")

    if features["candidate_text_number_overlap"]:
        gain = min(24.0, 14.0 + 5.0 * features["candidate_text_number_overlap"])
        score += gain
        reasons.append(f"number_overlap:+{gain:.0f}")
    elif features["candidate_text_number_mismatch"]:
        score -= 20.0
        reasons.append("number_mismatch:-20")

    token_gain = min(28.0, features["candidate_text_token_overlap"] * 4.0)
    jaccard_gain = min(18.0, features["candidate_text_jaccard"] * 35.0)
    score += token_gain + jaccard_gain
    if token_gain:
        reasons.append(f"token_overlap:+{token_gain:.0f}")
    if jaccard_gain:
        reasons.append(f"jaccard:+{jaccard_gain:.1f}")
    return score, ";".join(reasons)


def score_distance(distance: float | None, near_bonus: float, penalty_per_m: float, cap_m: float) -> float:
    if distance is None or not math.isfinite(distance):
        return 0.0
    if distance <= 0.001:
        return near_bonus
    return max(-penalty_per_m * cap_m, -penalty_per_m * min(distance, cap_m))


def area_prior(area: float | None) -> float:
    """Weak prior: most target parcels are not the huge union-wide polygons."""
    if area is None or not math.isfinite(area) or area <= 0:
        return 0.0
    if 20 <= area <= 2500:
        return 4.0
    if 2500 < area <= 10000:
        return 1.0
    if area > 40000:
        return -8.0
    if area > 15000:
        return -3.0
    return 0.0


def load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return {str(row.get("key")): row for row in rows if isinstance(row, dict)}


def load_polygons(path: Path, layer: str) -> gpd.GeoDataFrame:
    cols = ["unique_key", "chargegeog", "FilePath"]
    try:
        gdf = gpd.read_file(path, layer=layer, columns=cols)
    except Exception:
        gdf = gpd.read_file(path, layer=layer)
        keep = [col for col in [*cols, "geometry"] if col in gdf.columns]
        gdf = gdf[keep]
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["unique_key"] = gdf["unique_key"].astype(str)
    gdf["candidate_area_m2"] = gdf.geometry.area
    centroids = gdf.geometry.centroid
    gdf["candidate_centroid_easting"] = centroids.x
    gdf["candidate_centroid_northing"] = centroids.y
    return gdf.reset_index(drop=True)


def road_features(geom: Any, roads: list[str], road_geoms: dict[str, Any]) -> tuple[float | None, int, int]:
    distances = []
    for road in roads:
        road_geom = road_geoms.get(road)
        if road_geom is None or road_geom.is_empty:
            continue
        distances.append(float(geom.distance(road_geom)))
    if not distances:
        return None, 0, 0
    min_distance = min(distances)
    return min_distance, sum(d <= 50 for d in distances), sum(d <= 100 for d in distances)


def build_case_roads(case_row: pd.Series | None, v10_row: pd.Series | None, road_names: set[str]) -> list[str]:
    roads: list[str] = []
    sources = [
        ("anchor_list", case_row.get("all_roads") if case_row is not None else None),
        ("anchor_list", v10_row.get("ocr_roads") if v10_row is not None else None),
        ("address", v10_row.get("original_address") if v10_row is not None else None),
    ]
    for source_type, source in sources:
        if source is None or (isinstance(source, float) and math.isnan(source)):
            continue
        if source_type == "address":
            found = roi_v1.extract_address_roads(source, road_names)
        else:
            found = roi_v1.split_anchors(source)
        for road in found:
            if road in road_names and road not in roads:
                roads.append(road)
    return roads


def score_geometry(row: dict[str, Any]) -> tuple[float, str]:
    reasons: list[str] = []
    score = 0.0

    if row["covers_v7_point"]:
        score += 80.0
        reasons.append("covers_v7_point:+80")
    else:
        delta = score_distance(row["distance_to_v7_point_m"], 0.0, 0.045, 500)
        score += delta
        if delta:
            reasons.append(f"distance_to_v7:{delta:.1f}")

    if row["covers_best_point"]:
        score += 65.0
        reasons.append("covers_best_point:+65")
    else:
        delta = score_distance(row["distance_to_best_point_m"], 0.0, 0.035, 500)
        score += delta
        if delta:
            reasons.append(f"distance_to_best:{delta:.1f}")

    if row["intersects_top1_roi"]:
        score += 24.0
        reasons.append("top1_roi:+24")
    if row["intersects_protected_current_roi"]:
        score += 10.0
        reasons.append("protected_current_roi:+10")
    if row["intersects_evidence_roi"]:
        score += 12.0
        reasons.append("evidence_roi:+12")
    if row["intersects_corridor_roi"]:
        score += 10.0
        reasons.append("corridor_roi:+10")

    score += min(35.0, row["roi_total_intersection_area_m2"] / 100.0)
    score += min(18.0, row["roi_top1_intersection_area_m2"] / 90.0)
    score += min(14.0, row["roi_rank_weighted_score"] / 3.0)
    score += max(-12.0, -0.018 * min(row["distance_to_nearest_roi_center_m"], 650.0))

    road_distance = row["distance_to_mentioned_road_m"]
    if road_distance is not None and math.isfinite(road_distance):
        if road_distance <= 20:
            score += 14.0
            reasons.append("near_mentioned_road_20m:+14")
        elif road_distance <= 50:
            score += 10.0
            reasons.append("near_mentioned_road_50m:+10")
        elif road_distance <= 100:
            score += 5.0
            reasons.append("near_mentioned_road_100m:+5")

    area_score = area_prior(row["candidate_area_m2"])
    score += area_score
    if area_score:
        reasons.append(f"area_prior:{area_score:+.1f}")

    return score, ";".join(reasons)


def combine_scores(geometry_score: float, text_score: float, mode: str) -> float:
    if mode == "blended":
        return geometry_score + text_score
    if mode == "geometry":
        return geometry_score
    if mode == "text":
        return text_score
    if mode == "text_primary":
        # Candidate metadata is the strongest way to distinguish many
        # overlapping historical planning polygons.  Geometry should mainly
        # break ties between similarly-worded candidates inside the ROI union.
        return text_score * 1000.0 + geometry_score
    raise ValueError(f"unsupported score mode: {mode}")


def summarize(case_df: pd.DataFrame) -> dict[str, Any]:
    ok = case_df[case_df["status"] == "ok"].copy()
    if ok.empty:
        return {"cases": 0}
    rank = pd.to_numeric(ok["target_rank_rule"], errors="coerce")
    union_hit = ok["target_in_candidates"].fillna(False)
    return {
        "cases": int(len(ok)),
        "target_in_candidates": int(union_hit.sum()),
        "target_in_candidates_rate": float(union_hit.mean()),
        "top1": int((rank == 1).sum()),
        "top1_rate_all": float((rank == 1).sum() / len(ok)),
        "top1_rate_when_target_present": float((rank[union_hit] == 1).sum() / union_hit.sum()) if union_hit.sum() else 0.0,
        "top3": int((rank <= 3).sum()),
        "top3_rate_all": float((rank <= 3).sum() / len(ok)),
        "top5": int((rank <= 5).sum()),
        "top5_rate_all": float((rank <= 5).sum() / len(ok)),
        "top10": int((rank <= 10).sum()),
        "top10_rate_all": float((rank <= 10).sum() / len(ok)),
        "mean_candidate_count": float(pd.to_numeric(ok["candidate_polygon_count"], errors="coerce").mean()),
        "median_candidate_count": float(pd.to_numeric(ok["candidate_polygon_count"], errors="coerce").median()),
    }


def decision_band(row: pd.Series) -> str:
    margin = parse_float(row.get("top1_margin")) or 0.0
    text_score = parse_float(row.get("top1_text_score")) or 0.0
    if text_score >= 80 and margin >= 10000:
        return "auto_accept_high"
    if text_score >= 70 or margin >= 500:
        return "review_priority"
    return "review"


def summarize_bands(case_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if case_df.empty or "decision_band" not in case_df.columns:
        return out
    for band, group in case_df[case_df["status"] == "ok"].groupby("decision_band", dropna=False):
        top1 = group["top1_is_target"].fillna(False)
        out[str(band)] = {
            "cases": int(len(group)),
            "top1": int(top1.sum()),
            "top1_precision": float(top1.mean()) if len(group) else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=DEFAULT_V10_CSV)
    parser.add_argument("--case-summary-csv", type=Path, default=DEFAULT_CASE_SUMMARY_CSV)
    parser.add_argument("--rois-csv", type=Path, default=DEFAULT_ROIS_CSV)
    parser.add_argument("--full-gpkg", type=Path, default=DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=DEFAULT_FULL_LAYER)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--score-mode", choices=["text_primary", "blended", "geometry", "text"], default="text_primary")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    print("loading tabular inputs...")
    raw_rows = load_raw_rows(args.input_json)
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    v10_by_key = {str(row["key"]): row for _, row in v10.iterrows()}
    case_summary = pd.read_csv(args.case_summary_csv, dtype={"key": str, "base_key": str})
    rois = pd.read_csv(args.rois_csv, dtype={"case_key": str, "base_key": str})
    rois["roi_rank"] = pd.to_numeric(rois["roi_rank"], errors="coerce")
    rois["roi_score"] = pd.to_numeric(rois["roi_score"], errors="coerce").fillna(0.0)

    print("loading polygons and road anchors...")
    polygons = load_polygons(args.full_gpkg, args.full_layer)
    road_geoms = roi_v1.load_road_geoms(args.open_roads)
    road_names = set(road_geoms)
    sindex = polygons.sindex

    candidate_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    processed = 0

    for _, case in case_summary.iterrows():
        key = str(case["key"])
        base_key = str(case.get("base_key") or key).split("_", 1)[0]
        if str(case.get("status") or "") != "ok":
            case_rows.append(
                {
                    "key": key,
                    "base_key": base_key,
                    "status": case.get("status"),
                    "candidate_polygon_count": 0,
                    "target_in_candidates": False,
                }
            )
            continue
        processed += 1
        if args.max_cases and processed > args.max_cases:
            break

        case_rois = rois[rois["case_key"] == key].sort_values("roi_rank").copy()
        if case_rois.empty:
            case_rows.append(
                {
                    "key": key,
                    "base_key": base_key,
                    "status": "no_rois",
                    "candidate_polygon_count": 0,
                    "target_in_candidates": False,
                }
            )
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
        candidate_idx = list(sindex.query(union_roi, predicate="intersects"))
        candidates = polygons.iloc[candidate_idx].copy()
        if candidates.empty:
            case_rows.append(
                {
                    "key": key,
                    "base_key": base_key,
                    "status": "no_candidates",
                    "candidate_polygon_count": 0,
                    "target_in_candidates": False,
                }
            )
            continue

        raw = raw_rows.get(key) or raw_rows.get(base_key) or {}
        v10_row = v10_by_key.get(key)
        best_point = point_from_xy(raw.get("best_easting_27700_final"), raw.get("best_northing_27700_final"))
        if best_point is None and v10_row is not None:
            best_point = point_from_xy(v10_row.get("baseline_easting"), v10_row.get("baseline_northing"))
        v7_point = point_from_xy(v10_row.get("v7_selected_easting"), v10_row.get("v7_selected_northing")) if v10_row is not None else None
        roads = build_case_roads(case, v10_row, road_names)

        per_case_rows: list[dict[str, Any]] = []
        for _, cand in candidates.iterrows():
            geom = cand.geometry
            candidate_key = str(cand["unique_key"])
            centroid = geom.centroid
            roi_hits = [item for item in roi_items if geom.intersects(item["geom"])]
            if not roi_hits:
                continue
            intersection_areas = [float(geom.intersection(item["geom"]).area) for item in roi_hits]
            center_distances = [float(centroid.distance(item["center"])) for item in roi_hits]
            top1_area = sum(area for area, item in zip(intersection_areas, roi_hits) if item["rank"] == 1)
            weighted_score = sum((item["score"] + 8.0) / max(1, item["rank"]) for item in roi_hits)
            road_distance, road_near_50, road_near_100 = road_features(geom, roads, road_geoms)

            best_distance = float(geom.distance(best_point)) if best_point is not None else None
            v7_distance = float(geom.distance(v7_point)) if v7_point is not None else None
            row = {
                "case_key": key,
                "base_key": base_key,
                "sample_split": case.get("sample_split"),
                "original_address": case.get("original_address"),
                "best_confidence": parse_float(raw.get("best_confidence")) or parse_float(case.get("best_confidence")),
                "best_source_final": raw.get("best_source_final"),
                "best_selection_category": raw.get("best_selection_category"),
                "best_confidence_reason": raw.get("best_confidence_reason"),
                "candidate_key": candidate_key,
                "is_target": candidate_key == base_key,
                "candidate_chargegeog": cand.get("chargegeog"),
                "candidate_filepath": cand.get("FilePath"),
                "candidate_area_m2": float(cand["candidate_area_m2"]),
                "candidate_centroid_easting": float(cand["candidate_centroid_easting"]),
                "candidate_centroid_northing": float(cand["candidate_centroid_northing"]),
                "covers_best_point": bool(best_point is not None and geom.covers(best_point)),
                "distance_to_best_point_m": best_distance,
                "covers_v7_point": bool(v7_point is not None and geom.covers(v7_point)),
                "distance_to_v7_point_m": v7_distance,
                "roi_intersects_count": len(roi_hits),
                "roi_min_rank": min(item["rank"] for item in roi_hits),
                "roi_hit_reasons": "|".join(sorted({item["reason"] for item in roi_hits if item["reason"]})),
                "roi_hit_sources": "|".join(sorted({item["sources"] for item in roi_hits if item["sources"]})),
                "intersects_top1_roi": any(item["rank"] == 1 for item in roi_hits),
                "intersects_protected_current_roi": any(item["reason"] == "protected_current" for item in roi_hits),
                "intersects_evidence_roi": any(item["reason"] == "evidence" for item in roi_hits),
                "intersects_corridor_roi": any(item["reason"] == "corridor_sample" for item in roi_hits),
                "roi_total_intersection_area_m2": sum(intersection_areas),
                "roi_top1_intersection_area_m2": top1_area,
                "roi_rank_weighted_score": weighted_score,
                "distance_to_nearest_roi_center_m": min(center_distances),
                "mentioned_roads": " | ".join(roads),
                "distance_to_mentioned_road_m": road_distance,
                "mentioned_road_near_count_50m": road_near_50,
                "mentioned_road_near_count_100m": road_near_100,
            }
            row.update(text_match_features(case.get("original_address"), cand.get("chargegeog")))
            row["geometry_score"], row["geometry_score_reasons"] = score_geometry(row)
            row["text_score"], row["text_score_reasons"] = score_text(row)
            row["score_mode"] = args.score_mode
            row["rule_score"] = combine_scores(row["geometry_score"], row["text_score"], args.score_mode)
            row["rule_score_reasons"] = " || ".join(
                part for part in [row["geometry_score_reasons"], row["text_score_reasons"]] if part
            )
            per_case_rows.append(row)

        if not per_case_rows:
            case_rows.append(
                {
                    "key": key,
                    "base_key": base_key,
                    "status": "no_intersecting_candidates",
                    "candidate_polygon_count": 0,
                    "target_in_candidates": False,
                }
            )
            continue

        per_case_rows.sort(
            key=lambda item: (
                -float(item["rule_score"]),
                float(item["distance_to_v7_point_m"] if item["distance_to_v7_point_m"] is not None else 1e9),
                float(item["distance_to_best_point_m"] if item["distance_to_best_point_m"] is not None else 1e9),
                float(item["candidate_area_m2"]),
                item["candidate_key"],
            )
        )
        target_rank = None
        top1_key = per_case_rows[0]["candidate_key"]
        top1_score = per_case_rows[0]["rule_score"]
        top2_score = per_case_rows[1]["rule_score"] if len(per_case_rows) > 1 else None
        for idx, row in enumerate(per_case_rows, start=1):
            row["rule_rank"] = idx
            if row["is_target"]:
                target_rank = idx
            if idx <= 20:
                top_rows.append(row.copy())
            candidate_rows.append(row)

        target_in_candidates = target_rank is not None
        case_rows.append(
            {
                "key": key,
                "base_key": base_key,
                "status": "ok",
                "sample_split": case.get("sample_split"),
                "original_address": case.get("original_address"),
                "best_confidence": parse_float(raw.get("best_confidence")) or parse_float(case.get("best_confidence")),
                "union_intersects_target_polygon": safe_bool(case.get("union_intersects_target_polygon")),
                "candidate_polygon_count": len(per_case_rows),
                "target_in_candidates": target_in_candidates,
                "target_rank_rule": target_rank,
                "top1_candidate_key": top1_key,
                "top1_is_target": target_rank == 1,
                "top1_rule_score": top1_score,
                "top2_rule_score": top2_score,
                "top1_margin": top1_score - top2_score if top2_score is not None else None,
                "top3_contains_target": target_rank is not None and target_rank <= 3,
                "top5_contains_target": target_rank is not None and target_rank <= 5,
                "top10_contains_target": target_rank is not None and target_rank <= 10,
                "top20_contains_target": target_rank is not None and target_rank <= 20,
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    case_df = pd.DataFrame(case_rows)
    top_df = pd.DataFrame(top_rows)

    if not top_df.empty and not case_df.empty:
        top1 = top_df[pd.to_numeric(top_df["rule_rank"], errors="coerce") == 1].copy()
        top1 = top1[
            [
                "case_key",
                "text_score",
                "geometry_score",
                "candidate_text_jaccard",
                "candidate_text_token_overlap",
                "candidate_text_number_overlap",
                "covers_best_point",
                "covers_v7_point",
            ]
        ].rename(
            columns={
                "case_key": "key",
                "text_score": "top1_text_score",
                "geometry_score": "top1_geometry_score",
                "candidate_text_jaccard": "top1_candidate_text_jaccard",
                "candidate_text_token_overlap": "top1_candidate_text_token_overlap",
                "candidate_text_number_overlap": "top1_candidate_text_number_overlap",
                "covers_best_point": "top1_covers_best_point",
                "covers_v7_point": "top1_covers_v7_point",
            }
        )
        case_df = case_df.merge(top1, on="key", how="left")
        case_df["decision_band"] = case_df.apply(decision_band, axis=1)

    out_prefix = args.output_prefix
    candidates_csv = out_prefix.with_name(out_prefix.name + "_candidates.csv")
    top20_csv = out_prefix.with_name(out_prefix.name + "_top20.csv")
    cases_csv = out_prefix.with_name(out_prefix.name + "_cases.csv")
    xlsx_path = out_prefix.with_suffix(".xlsx")
    summary_path = out_prefix.with_suffix(".summary.json")

    candidate_df.to_csv(candidates_csv, index=False)
    top_df.to_csv(top20_csv, index=False)
    case_df.to_csv(cases_csv, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        case_df.to_excel(writer, sheet_name="cases", index=False)
        top_df.to_excel(writer, sheet_name="top20", index=False)

    low_conf = case_df[pd.to_numeric(case_df.get("best_confidence"), errors="coerce") < 75].copy()
    high_conf = case_df[pd.to_numeric(case_df.get("best_confidence"), errors="coerce") >= 80].copy()
    summary = {
        "all_cases": summarize(case_df),
        "low_confidence_lt75": summarize(low_conf),
        "high_confidence_ge80": summarize(high_conf),
        "decision_bands": summarize_bands(case_df),
        "output_candidates_csv": str(candidates_csv),
        "output_top20_csv": str(top20_csv),
        "output_cases_csv": str(cases_csv),
        "output_xlsx": str(xlsx_path),
        "candidate_rows": int(len(candidate_df)),
        "score_mode": args.score_mode,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
