#!/usr/bin/env python3
"""Composite and relation-aware Mansfield polygon selector.

This is an experiment layer on top of ``9_hybrid_polygon_rerank.py`` and
``10_cascade_polygon_selector.py``.  Candidate generation remains
production-visible:

- existing top-N raw WFS / WFS-merged / council candidates;
- case-local unions of top candidates that share/touch geometry;
- exact-address unions from polygons carrying requested address candidate points;
- relation-case unions from nearby non-anchor polygons.

The manual polygon layer is used only for offline evaluation metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent


def import_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


hybrid = import_script(ROOT / "9_hybrid_polygon_rerank.py", "hybrid_rerank_v11")
cascade = import_script(ROOT / "10_cascade_polygon_selector.py", "cascade_selector_v11")
base_rerank = hybrid.base_rerank


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TAG = "mansfield-manual-polygon-link_random1200_seed42_43_combined"
DEFAULT_TOP = TMP / f"{DEFAULT_TAG}_hybrid_polygon_rerank_v1_top5roi_top50.csv"
DEFAULT_INPUT_JSON = TMP / f"{DEFAULT_TAG}_gemini.json"
DEFAULT_V10_CSV = TMP / f"{DEFAULT_TAG}_ocr_openroads_v10.csv"
DEFAULT_TRUTH_GPKG = TMP / f"{DEFAULT_TAG}.gpkg"
DEFAULT_TRUTH_LAYER = "random1200_seed42_43_combined"
DEFAULT_OUTPUT_PREFIX = TMP / f"{DEFAULT_TAG}_composite_relation_selector_v1"


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def load_truth(path: Path, layer: str) -> dict[str, Any]:
    return hybrid.load_truth(path, layer)


def point_from_xy(x: Any, y: Any) -> Point | None:
    return base_rerank.point_from_xy(x, y)


def first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = base_rerank.parse_float(value)
        if parsed is not None and math.isfinite(parsed):
            return parsed
    return None


def coerce_top(top: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "truth_iou",
        "truth_overlap_area_m2",
        "truth_candidate_overlap_ratio",
        "truth_cover_ratio",
        "rule_score",
        "rule_rank",
        "candidate_area_m2",
        "candidate_centroid_easting",
        "candidate_centroid_northing",
        "best_confidence",
        "distance_to_best_point_m",
        "distance_to_v7_point_m",
        "distance_to_range_anchor_m",
        "range_anchor_easting",
        "range_anchor_northing",
        "roi_total_intersection_area_m2",
        "roi_rank_weighted_score",
        "distance_to_nearest_roi_center_m",
        "distance_to_mentioned_road_m",
        "uprn_count",
        "nearest_uprn_to_best_point_m",
        "nearest_uprn_to_v7_point_m",
    ]
    bool_columns = [
        "covers_best_point",
        "covers_v7_point",
        "covers_range_anchor",
        "intersects_top1_roi",
        "intersects_protected_current_roi",
        "intersects_evidence_roi",
        "intersects_corridor_roi",
        "range_anchor_complete",
        "truth_intersects",
        "truth_centroid_inside",
    ]
    for column in numeric_columns:
        if column in top:
            top[column] = pd.to_numeric(top[column], errors="coerce")
    for column in bool_columns:
        if column in top:
            top[column] = to_bool(top[column])
    top["land_style"] = top["original_address"].fillna("").astype(str).str.contains(cascade.LAND_STYLE_RE)
    top["candidate_kind"] = "single"
    top["source_ids"] = top["candidate_id"].astype(str)
    return top


def add_geometries(top: pd.DataFrame, geometries: dict[str, Any]) -> pd.DataFrame:
    top = top.copy()
    top["geometry_obj"] = top["candidate_id"].astype(str).map(geometries)
    return top[top["geometry_obj"].notna()].copy()


def truth_metrics(geom: Any, target_geom: Any) -> dict[str, Any]:
    return hybrid.truth_metrics(geom, target_geom)


def case_points(case_key: str, base_key: str, raw_rows: dict[str, dict[str, Any]], v10_by_key: dict[str, pd.Series]) -> dict[str, Point | None]:
    raw = raw_rows.get(case_key) or raw_rows.get(base_key) or {}
    v10 = v10_by_key.get(case_key)
    best_point = point_from_xy(raw.get("best_easting_27700_final"), raw.get("best_northing_27700_final"))
    if best_point is None and v10 is not None:
        best_point = point_from_xy(v10.get("baseline_easting"), v10.get("baseline_northing"))
    v7_point = point_from_xy(v10.get("v7_selected_easting"), v10.get("v7_selected_northing")) if v10 is not None else None
    return {"best": best_point, "v7": v7_point}


def address_features_for_geom(geom: Any, points: list[dict[str, Any]], requested_count: int) -> dict[str, Any]:
    requested_inside = 0
    extra_inside = 0
    exact_inside = 0
    best_index_inside: int | None = None
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
    return {
        "requested_number_count": max(1, requested_count),
        "address_candidate_point_count": len(points),
        "address_requested_points_inside": requested_inside,
        "address_extra_points_inside": extra_inside,
        "address_exact_points_inside": exact_inside,
        "address_best_candidate_index_inside": best_index_inside,
    }


def composite_row(
    *,
    group: pd.DataFrame,
    subset: pd.DataFrame,
    candidate_id: str,
    candidate_kind: str,
    candidate_source: str,
    geom: Any,
    target_geom: Any,
    best_point: Point | None,
    v7_point: Point | None,
    range_anchor_point: Point | None,
    address_points: list[dict[str, Any]],
    requested_count: int,
) -> dict[str, Any]:
    first = group.iloc[0]
    source_ids = sorted(set(subset["candidate_id"].astype(str)))
    centroid = geom.centroid
    row: dict[str, Any] = {
        "case_key": first["case_key"],
        "base_key": first["base_key"],
        "sample_split": first.get("sample_split"),
        "original_address": first.get("original_address"),
        "best_confidence": first.get("best_confidence"),
        "best_source_final": first.get("best_source_final"),
        "best_selection_category": first.get("best_selection_category"),
        "candidate_source": candidate_source,
        "candidate_kind": candidate_kind,
        "candidate_id": candidate_id,
        "candidate_toid": "",
        "candidate_gmlid": "",
        "candidate_objectid": "",
        "candidate_theme": "Composite",
        "candidate_descriptive_group": "Composite",
        "candidate_descriptive_term": candidate_kind,
        "candidate_make": "",
        "candidate_area_m2": float(geom.area),
        "candidate_centroid_easting": float(centroid.x),
        "candidate_centroid_northing": float(centroid.y),
        "covers_best_point": bool(best_point is not None and geom.covers(best_point)),
        "distance_to_best_point_m": float(geom.distance(best_point)) if best_point is not None else None,
        "covers_v7_point": bool(v7_point is not None and geom.covers(v7_point)),
        "distance_to_v7_point_m": float(geom.distance(v7_point)) if v7_point is not None else None,
        "has_range_anchor": bool(range_anchor_point is not None),
        "covers_range_anchor": bool(range_anchor_point is not None and geom.covers(range_anchor_point)),
        "distance_to_range_anchor_m": float(geom.distance(range_anchor_point)) if range_anchor_point is not None else None,
        "range_anchor_relation": first.get("range_anchor_relation"),
        "range_anchor_complete": first.get("range_anchor_complete"),
        "roi_intersects_count": int(subset["roi_intersects_count"].max()) if "roi_intersects_count" in subset else 0,
        "roi_min_rank": int(subset["roi_min_rank"].min()) if "roi_min_rank" in subset else None,
        "roi_hit_reasons": "|".join(sorted(set("|".join(subset["roi_hit_reasons"].fillna("").astype(str)).split("|")) - {""})),
        "intersects_top1_roi": bool(subset["intersects_top1_roi"].any()),
        "intersects_protected_current_roi": bool(subset["intersects_protected_current_roi"].any()),
        "intersects_evidence_roi": bool(subset["intersects_evidence_roi"].any()),
        "intersects_corridor_roi": bool(subset["intersects_corridor_roi"].any()),
        "roi_total_intersection_area_m2": float(subset["roi_total_intersection_area_m2"].sum()),
        "roi_rank_weighted_score": float(subset["roi_rank_weighted_score"].sum()),
        "distance_to_nearest_roi_center_m": float(subset["distance_to_nearest_roi_center_m"].min()),
        "mentioned_roads": first.get("mentioned_roads"),
        "distance_to_mentioned_road_m": float(subset["distance_to_mentioned_road_m"].min())
        if subset["distance_to_mentioned_road_m"].notna().any()
        else None,
        "mentioned_road_near_count_50m": int(subset["mentioned_road_near_count_50m"].max())
        if "mentioned_road_near_count_50m" in subset
        else 0,
        "mentioned_road_near_count_100m": int(subset["mentioned_road_near_count_100m"].max())
        if "mentioned_road_near_count_100m" in subset
        else 0,
        "uprn_count": int(subset["uprn_count"].sum()) if "uprn_count" in subset else 0,
        "nearest_uprn_to_best_point_m": float(subset["nearest_uprn_to_best_point_m"].min())
        if subset["nearest_uprn_to_best_point_m"].notna().any()
        else None,
        "nearest_uprn_to_v7_point_m": float(subset["nearest_uprn_to_v7_point_m"].min())
        if subset["nearest_uprn_to_v7_point_m"].notna().any()
        else None,
        "rule_score": float(subset["rule_score"].max()),
        "rule_rank": int(subset["rule_rank"].min()),
        "source_ids": "|".join(source_ids[:20]),
        "source_count": len(source_ids),
        "geometry_obj": geom,
        "land_style": bool(first.get("land_style")),
    }
    row.update(address_features_for_geom(geom, address_points, requested_count))
    row.update(truth_metrics(geom, target_geom))
    return row


def connected_components(indices: list[int], geoms: list[Any], max_gap: float) -> list[list[int]]:
    if not indices:
        return []
    parent = list(range(len(indices)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            if geoms[left].distance(geoms[right]) <= max_gap:
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for pos, original_idx in enumerate(indices):
        grouped.setdefault(find(pos), []).append(original_idx)
    return list(grouped.values())


def add_union_candidate(
    out: list[dict[str, Any]],
    *,
    group: pd.DataFrame,
    subset: pd.DataFrame,
    kind: str,
    source: str,
    target_geom: Any,
    best_point: Point | None,
    v7_point: Point | None,
    range_anchor_point: Point | None,
    address_points: list[dict[str, Any]],
    requested_count: int,
    max_area: float,
) -> None:
    subset = subset.drop_duplicates("candidate_id").copy()
    if len(subset) < 2 or len(subset) > 12:
        return
    geoms = [geom for geom in subset["geometry_obj"] if geom is not None and not geom.is_empty]
    if len(geoms) < 2:
        return
    geom = unary_union(geoms)
    if geom.is_empty or float(geom.area) <= 0 or float(geom.area) > max_area:
        return
    source_ids = sorted(set(subset["candidate_id"].astype(str)))
    candidate_id = f"{kind}:{base_rerank.norm_compact('|'.join(source_ids))[:140]}"
    out.append(
        composite_row(
            group=group,
            subset=subset,
            candidate_id=candidate_id,
            candidate_kind=kind,
            candidate_source=source,
            geom=geom,
            target_geom=target_geom,
            best_point=best_point,
            v7_point=v7_point,
            range_anchor_point=range_anchor_point,
            address_points=address_points,
            requested_count=requested_count,
        )
    )


def generate_composites(
    group: pd.DataFrame,
    target_geom: Any,
    raw_rows: dict[str, dict[str, Any]],
    v10_by_key: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    group = group[group["geometry_obj"].notna()].copy()
    if group.empty:
        return []
    case_key = str(group["case_key"].iloc[0])
    base_key = str(group["base_key"].iloc[0])
    raw_row = raw_rows.get(case_key) or raw_rows.get(base_key) or {}
    points = case_points(case_key, base_key, raw_rows, v10_by_key)
    best_point = points["best"]
    v7_point = points["v7"]
    range_x = first_finite(group["range_anchor_easting"].iloc[0])
    range_y = first_finite(group["range_anchor_northing"].iloc[0])
    range_anchor_point = Point(range_x, range_y) if range_x is not None and range_y is not None else None

    requested = cascade.requested_numbers(raw_row.get("original_address") or group["original_address"].iloc[0])
    address_points = cascade.collect_address_points(raw_row, requested)
    requested_count = max(1, len(requested))

    out: list[dict[str, Any]] = []

    # Exact-address unions: useful for number ranges and multi-unit sites.
    if cascade.is_exact_address_case(group):
        exact = group[pd.to_numeric(group["address_requested_points_inside"], errors="coerce").fillna(0) > 0].copy()
        if not exact.empty:
            for source_name, source_subset in exact.groupby("candidate_source"):
                source_subset = source_subset.sort_values(
                    [
                        "address_exact_points_inside",
                        "address_requested_points_inside",
                        "address_extra_points_inside",
                        "candidate_area_m2",
                    ],
                    ascending=[False, False, True, True],
                ).head(max(2, min(8, requested_count + 3)))
                add_union_candidate(
                    out,
                    group=group,
                    subset=source_subset,
                    kind=f"exact_address_union_{source_name}",
                    source="composite_exact_address",
                    target_geom=target_geom,
                    best_point=best_point,
                    v7_point=v7_point,
                    range_anchor_point=range_anchor_point,
                    address_points=address_points,
                    requested_count=requested_count,
                    max_area=max(2500.0, 1400.0 * requested_count),
                )
            exact_small = exact.sort_values(
                ["address_extra_points_inside", "candidate_area_m2", "rule_score"],
                ascending=[True, True, False],
            ).head(max(2, min(10, requested_count + 4)))
            add_union_candidate(
                out,
                group=group,
                subset=exact_small,
                kind="exact_address_union_mixed",
                source="composite_exact_address",
                target_geom=target_geom,
                best_point=best_point,
                v7_point=v7_point,
                range_anchor_point=range_anchor_point,
                address_points=address_points,
                requested_count=requested_count,
                max_area=max(3200.0, 1800.0 * requested_count),
            )

    # Generic touching components from the top of the pool.  Keep this tight to
    # avoid city-block unions.
    top_subset = group[pd.to_numeric(group["rule_rank"], errors="coerce").fillna(999) <= 18].copy()
    top_subset = top_subset[top_subset["candidate_area_m2"].fillna(0).between(20, 8000)]
    for source_name, source_subset in top_subset.groupby("candidate_source"):
        source_subset = source_subset.head(12).copy()
        idxs = list(source_subset.index)
        geoms = [source_subset.loc[idx, "geometry_obj"] for idx in idxs]
        for component in connected_components(idxs, geoms, 1.25):
            component_subset = source_subset.loc[component]
            add_union_candidate(
                out,
                group=group,
                subset=component_subset,
                kind=f"touching_union_{source_name}",
                source="composite_touching",
                target_geom=target_geom,
                best_point=best_point,
                v7_point=v7_point,
                range_anchor_point=range_anchor_point,
                address_points=address_points,
                requested_count=requested_count,
                max_area=12000.0,
            )

    # Relation cases: address point is an anchor.  Generate unions from nearby
    # non-anchor candidates, and keep relation-aware single candidates available
    # through scoring.
    if bool(group["land_style"].iloc[0]):
        anchor_cover = group["covers_best_point"] | group["covers_v7_point"] | group["covers_range_anchor"]
        near_distance = pd.concat(
            [
                group["distance_to_best_point_m"],
                group["distance_to_v7_point_m"],
                group["distance_to_range_anchor_m"],
            ],
            axis=1,
        ).min(axis=1, skipna=True)
        relation = group[~anchor_cover].copy()
        relation["relation_anchor_distance"] = near_distance.loc[relation.index]
        relation = relation[
            relation["relation_anchor_distance"].fillna(999).between(0, 120)
            & relation["candidate_area_m2"].fillna(0).between(20, 16000)
        ].copy()
        relation = relation.sort_values(["relation_anchor_distance", "rule_rank"]).head(16)
        if len(relation) >= 2:
            idxs = list(relation.index)
            geoms = [relation.loc[idx, "geometry_obj"] for idx in idxs]
            for component in connected_components(idxs, geoms, 2.5):
                component_subset = relation.loc[component]
                add_union_candidate(
                    out,
                    group=group,
                    subset=component_subset,
                    kind="relation_near_anchor_union",
                    source="composite_relation",
                    target_geom=target_geom,
                    best_point=best_point,
                    v7_point=v7_point,
                    range_anchor_point=range_anchor_point,
                    address_points=address_points,
                    requested_count=requested_count,
                    max_area=18000.0,
                )
            add_union_candidate(
                out,
                group=group,
                subset=relation.head(4),
                kind="relation_nearest_non_anchor_union",
                source="composite_relation",
                target_geom=target_geom,
                best_point=best_point,
                v7_point=v7_point,
                range_anchor_point=range_anchor_point,
                address_points=address_points,
                requested_count=requested_count,
                max_area=18000.0,
            )

    return out


def exact_score(group: pd.DataFrame) -> pd.Series:
    source_priority = {"council_cadastral": 3, "wfs_merged": 2, "wfs_raw": 1, "composite_exact_address": 1.5}
    requested_count = pd.to_numeric(group["requested_number_count"], errors="coerce").fillna(1).clip(lower=1)
    area_limit = 700.0 * requested_count
    area_penalty = (
        np.log1p(pd.to_numeric(group["candidate_area_m2"], errors="coerce").fillna(0.0)) - np.log1p(area_limit)
    ).clip(lower=0)
    return (
        1000.0 * pd.to_numeric(group["address_exact_points_inside"], errors="coerce").fillna(0)
        + 180.0 * pd.to_numeric(group["address_requested_points_inside"], errors="coerce").fillna(0)
        - 80.0 * pd.to_numeric(group["address_extra_points_inside"], errors="coerce").fillna(0)
        + 12.0 * group["candidate_source"].map(source_priority).fillna(0)
        - 18.0 * area_penalty
        + 0.01 * pd.to_numeric(group["rule_score"], errors="coerce").fillna(0)
    )


def relation_score(group: pd.DataFrame) -> pd.Series:
    anchor_cover = group["covers_best_point"] | group["covers_v7_point"] | group["covers_range_anchor"]
    anchor_distance = pd.concat(
        [group["distance_to_best_point_m"], group["distance_to_v7_point_m"], group["distance_to_range_anchor_m"]],
        axis=1,
    ).min(axis=1, skipna=True).fillna(999.0)
    source_bonus = group["candidate_source"].map(
        {
            "composite_relation": 42.0,
            "council_cadastral": 28.0,
            "wfs_merged": 18.0,
            "wfs_raw": 5.0,
            "composite_touching": 10.0,
        }
    ).fillna(0.0)
    area = pd.to_numeric(group["candidate_area_m2"], errors="coerce").fillna(0.0)
    area_bonus = pd.Series(0.0, index=group.index)
    area_bonus += area.between(60, 4500).astype(float) * 18.0
    area_bonus += area.between(4500, 16000).astype(float) * 4.0
    area_bonus -= (area > 24000).astype(float) * 35.0
    distance_bonus = pd.Series(0.0, index=group.index)
    distance_bonus += anchor_distance.between(0, 12).astype(float) * 18.0
    distance_bonus += anchor_distance.between(12, 45).astype(float) * 42.0
    distance_bonus += anchor_distance.between(45, 95).astype(float) * 18.0
    distance_bonus -= (anchor_distance > 180).astype(float) * 24.0
    roi_bonus = (
        group["intersects_evidence_roi"].astype(float) * 16.0
        + group["intersects_corridor_roi"].astype(float) * 8.0
        + group["intersects_top1_roi"].astype(float) * 4.0
    )
    return (
        source_bonus
        + area_bonus
        + distance_bonus
        + roi_bonus
        - anchor_cover.astype(float) * 85.0
        + 0.025 * pd.to_numeric(group["rule_score"], errors="coerce").fillna(0.0)
    )


def select_case(group: pd.DataFrame, *, promote_composites: bool = False) -> tuple[pd.Series, str]:
    singles = group[group["candidate_kind"].fillna("single") == "single"].copy()
    current = singles.sort_values("rule_rank").iloc[0] if not singles.empty else group.sort_values("rule_rank").iloc[0]
    baseline, baseline_reason = cascade.select_case(singles) if not singles.empty else (current, "current_rule_top1")

    # Be conservative: previous cascade is better than the first relation
    # selector.  Only promote a composite when it carries stronger, explicit
    # evidence than the baseline.  Generated composites still improve the
    # oracle metrics even when not auto-selected.
    if not promote_composites:
        return baseline, baseline_reason

    if cascade.is_exact_address_case(singles if not singles.empty else group):
        composites = group[group["candidate_kind"].fillna("single").astype(str).str.startswith("exact_address_union")].copy()
        if not composites.empty:
            requested_count = pd.to_numeric(composites["requested_number_count"], errors="coerce").fillna(1).clip(lower=1)
            composites = composites[
                (pd.to_numeric(composites["address_requested_points_inside"], errors="coerce").fillna(0) >= np.minimum(requested_count, 2))
                & (pd.to_numeric(composites["address_extra_points_inside"], errors="coerce").fillna(0) <= requested_count + 2)
                & (pd.to_numeric(composites["candidate_area_m2"], errors="coerce").fillna(1e9) <= 1800.0 * requested_count)
            ].copy()
            if not composites.empty:
                composites["selector_score"] = exact_score(composites)
                baseline_score = float(exact_score(pd.DataFrame([baseline])).iloc[0])
                selected = composites.sort_values(["selector_score", "rule_score"], ascending=[False, False]).iloc[0]
                if float(selected["selector_score"]) >= baseline_score + 120.0:
                    return selected, "exact_address_composite_promoted"
        return baseline, baseline_reason

    if bool(group["land_style"].iloc[0]):
        relation_composites = group[group["candidate_source"].eq("composite_relation")].copy()
        if not relation_composites.empty:
            relation_composites["selector_score"] = relation_score(relation_composites)
            baseline_score = float(relation_score(pd.DataFrame([baseline])).iloc[0])
            anchor_distance = pd.concat(
                [
                    relation_composites["distance_to_best_point_m"],
                    relation_composites["distance_to_v7_point_m"],
                    relation_composites["distance_to_range_anchor_m"],
                ],
                axis=1,
            ).min(axis=1, skipna=True)
            relation_composites = relation_composites[
                (anchor_distance.fillna(999).between(4, 65))
                & (pd.to_numeric(relation_composites["candidate_area_m2"], errors="coerce").fillna(1e9) <= 9000)
                & (~relation_composites["covers_best_point"].fillna(False).astype(bool))
                & (~relation_composites["covers_v7_point"].fillna(False).astype(bool))
            ].copy()
            if not relation_composites.empty:
                selected = relation_composites.sort_values(["selector_score", "rule_score"], ascending=[False, False]).iloc[0]
                if float(selected["selector_score"]) >= baseline_score + 95.0:
                    return selected, "relation_composite_promoted"
        return baseline, baseline_reason

    return baseline, baseline_reason


def summarize(selected: pd.DataFrame, all_candidates: pd.DataFrame) -> dict[str, Any]:
    iou = pd.to_numeric(selected["truth_iou"], errors="coerce").fillna(0.0)
    oracle = all_candidates.groupby("case_key")["truth_iou"].max()
    summary: dict[str, Any] = {
        "cases": int(len(selected)),
        "selected_iou50": int((iou >= 0.5).sum()),
        "selected_iou50_rate": float((iou >= 0.5).mean()),
        "selected_iou80": int((iou >= 0.8).sum()),
        "selected_iou80_rate": float((iou >= 0.8).mean()),
        "selected_intersects": int((iou > 0).sum()),
        "selected_intersects_rate": float((iou > 0).mean()),
        "selected_median_iou": float(iou.median()),
        "oracle_iou50": int((oracle >= 0.5).sum()),
        "oracle_iou50_rate": float((oracle >= 0.5).mean()),
        "oracle_iou80": int((oracle >= 0.8).sum()),
        "oracle_iou80_rate": float((oracle >= 0.8).mean()),
        "oracle_median_iou": float(oracle.median()),
        "candidate_rows": int(len(all_candidates)),
        "composite_rows": int((all_candidates["candidate_kind"] != "single").sum()),
        "selected_reason_counts": selected["selected_reason"].value_counts(dropna=False).to_dict(),
        "selected_source_counts": selected["candidate_source"].value_counts(dropna=False).to_dict(),
        "selected_kind_counts": selected["candidate_kind"].value_counts(dropna=False).to_dict(),
    }
    for split, split_df in selected.groupby("sample_split", dropna=False):
        split_iou = pd.to_numeric(split_df["truth_iou"], errors="coerce").fillna(0.0)
        summary[f"split_{split}"] = {
            "cases": int(len(split_df)),
            "selected_iou50": int((split_iou >= 0.5).sum()),
            "selected_iou50_rate": float((split_iou >= 0.5).mean()),
            "selected_iou80": int((split_iou >= 0.8).sum()),
            "selected_iou80_rate": float((split_iou >= 0.8).mean()),
            "selected_median_iou": float(split_iou.median()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite and relation-aware Mansfield selector.")
    parser.add_argument("--top-csv", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=DEFAULT_V10_CSV)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKG)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument(
        "--promote-composites",
        action="store_true",
        help="Allow composite/relation candidates to replace the cascade baseline. Disabled by default because v1 promotion is experimental.",
    )
    args = parser.parse_args()

    print("loading inputs...")
    top = pd.read_csv(args.top_csv, dtype={"case_key": str, "base_key": str})
    top = coerce_top(top.sort_values(["case_key", "rule_rank"]).reset_index(drop=True))
    if args.max_cases:
        keep_keys = list(dict.fromkeys(top["case_key"].astype(str)))[: args.max_cases]
        top = top[top["case_key"].astype(str).isin(keep_keys)].copy()
    raw_rows = cascade.load_raw_rows(args.input_json)
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    v10_by_key = {str(row["key"]): row for _, row in v10.iterrows()}
    truth = load_truth(args.truth_gpkg, args.truth_layer)

    print("loading candidate geometries...")
    geometries = cascade.load_candidate_geometries(top)
    top = add_geometries(top, geometries)
    print(f"top rows with geometry={len(top)} geometries={len(geometries)}")

    print("adding address-point features to single candidates...")
    top = cascade.add_address_point_features(top, raw_rows, geometries)

    print("generating composite/relation candidates...")
    composite_rows: list[dict[str, Any]] = []
    for _, group in top.groupby("case_key", sort=False):
        base_key = str(group["base_key"].iloc[0])
        composite_rows.extend(generate_composites(group, truth.get(base_key), raw_rows, v10_by_key))
    composites = pd.DataFrame(composite_rows)
    print(f"composite rows={len(composites)}")

    all_candidates = pd.concat([top, composites], ignore_index=True, sort=False) if not composites.empty else top.copy()
    for column in ["candidate_area_m2", "rule_score", "rule_rank", "truth_iou"]:
        all_candidates[column] = pd.to_numeric(all_candidates[column], errors="coerce")
    for column in [
        "covers_best_point",
        "covers_v7_point",
        "covers_range_anchor",
        "intersects_top1_roi",
        "intersects_evidence_roi",
        "intersects_corridor_roi",
    ]:
        all_candidates[column] = all_candidates[column].fillna(False).astype(bool)

    print("selecting per case...")
    selected_rows: list[dict[str, Any]] = []
    for case_key, group in all_candidates.groupby("case_key", sort=False):
        selected, reason = select_case(group, promote_composites=args.promote_composites)
        row = selected.drop(labels=["geometry_obj"], errors="ignore").to_dict()
        row["selected_reason"] = reason
        row["selected_rule_rank"] = row.get("rule_rank")
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    summary = summarize(selected, all_candidates)

    out_prefix = args.output_prefix
    selected_csv = out_prefix.with_name(out_prefix.name + "_selected.csv")
    summary_json = out_prefix.with_suffix(".summary.json")
    xlsx_path = out_prefix.with_suffix(".xlsx")
    selected_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_csv, index=False)
    selected.to_excel(xlsx_path, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_candidates:
        candidate_csv = out_prefix.with_name(out_prefix.name + "_candidates.csv")
        all_candidates.drop(columns=["geometry_obj"], errors="ignore").to_csv(candidate_csv, index=False)
        print(f"candidates: {candidate_csv}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"selected: {selected_csv}")
    print(f"xlsx: {xlsx_path}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
