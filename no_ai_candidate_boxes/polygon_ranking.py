#!/usr/bin/env python3
"""Rank in-box polygon candidates without AI/API calls.

Input rows are production-visible polygon candidates emitted by the hybrid
polygon candidate generator.  Truth columns, when present, are used only for
offline evaluation and are never used to select or rank candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/"
    "mansfield_random1200_hybrid_top200_v1_top200.csv"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/"
    "mansfield_polygon_ranking_v1"
)

NUMERIC_COLUMNS = [
    "rule_rank",
    "rule_score",
    "best_confidence",
    "candidate_area_m2",
    "candidate_centroid_easting",
    "candidate_centroid_northing",
    "distance_to_best_point_m",
    "distance_to_v7_point_m",
    "distance_to_range_anchor_m",
    "roi_min_rank",
    "roi_total_intersection_area_m2",
    "roi_rank_weighted_score",
    "distance_to_nearest_roi_center_m",
    "distance_to_mentioned_road_m",
    "uprn_count",
    "truth_iou",
    "truth_overlap_area_m2",
    "truth_candidate_overlap_ratio",
    "truth_cover_ratio",
]

BOOL_COLUMNS = [
    "covers_best_point",
    "covers_v7_point",
    "covers_range_anchor",
    "intersects_top1_roi",
    "intersects_protected_current_roi",
    "intersects_evidence_roi",
    "intersects_corridor_roi",
    "has_range_anchor",
    "range_anchor_complete",
    "truth_available",
    "truth_intersects",
    "truth_centroid_inside",
]

SOURCE_PRIORITY = {
    "council_cadastral": 3,
    "wfs_merged": 2,
    "wfs_raw": 1,
}

RELATION_ADDRESS_PATTERN = re.compile(
    r"\b("
    r"land|rear|adjacent|adjoining|behind|off|former|site|plot|plots|"
    r"school|college|sports|ground|field|plantation|farm|garage|station|"
    r"clinic|depot|yard|corner|between|north|south|east|west|side|"
    r"opposite|fronting"
    r")\b",
    re.IGNORECASE,
)
ROAD_SUFFIX_CONTEXT_RE = re.compile(
    r"\b(road|street|lane|avenue|close|way|drive|grove|crescent|hill|gate|walk|place|court|park|terrace|row|yard)\b",
    re.IGNORECASE,
)
LEADING_STREET_NUMBER_RE = re.compile(r"^\s*\d{1,4}[A-Za-z]?\b")
RELATION_CONTEXT_EXTRA_RE = re.compile(r"[/]|\b(back garden|front garden|garden land|site at)\b", re.IGNORECASE)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def coerce_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in BOOL_COLUMNS:
        if column in df:
            df[column] = df[column].map(to_bool)
    for column in ["case_key", "base_key", "candidate_id", "candidate_source"]:
        if column in df:
            df[column] = df[column].fillna("").astype(str)

    if "candidate_source" in df:
        df["_source_priority"] = df["candidate_source"].map(SOURCE_PRIORITY).fillna(0).astype(float)
    else:
        df["_source_priority"] = 0.0

    point_distance_columns = [
        column
        for column in ["distance_to_best_point_m", "distance_to_v7_point_m", "distance_to_range_anchor_m"]
        if column in df
    ]
    if point_distance_columns:
        df["_min_point_distance_m"] = df[point_distance_columns].min(axis=1, skipna=True)
    else:
        df["_min_point_distance_m"] = pd.NA

    covers_columns = [column for column in ["covers_best_point", "covers_v7_point", "covers_range_anchor"] if column in df]
    if covers_columns:
        df["_covers_any_anchor_point"] = df[covers_columns].any(axis=1)
    else:
        df["_covers_any_anchor_point"] = False

    df["_relation_alternate_score"] = relation_alternate_score(df)
    return df


def series_or_default(df: pd.DataFrame, column: str, default: float | bool) -> pd.Series:
    if column in df:
        return df[column]
    return pd.Series(default, index=df.index)


def relation_alternate_score(df: pd.DataFrame) -> pd.Series:
    """Score offset parcels for rear/adjacent/land-off style descriptions.

    This is intentionally not a replacement for the main rule score. It is only
    used to choose one bounded alternate after preserving the first rule-ranked
    candidates. Truth/evaluation fields are not referenced.
    """

    score = pd.Series(0.0, index=df.index)
    covers_any = series_or_default(df, "_covers_any_anchor_point", False).fillna(False).astype(bool)
    area = series_or_default(df, "candidate_area_m2", 1e12).fillna(1e12)
    min_point_distance = series_or_default(df, "_min_point_distance_m", 9999.0).fillna(9999.0)
    road_distance = series_or_default(df, "distance_to_mentioned_road_m", 9999.0).fillna(9999.0)
    roi_score = series_or_default(df, "roi_rank_weighted_score", 0.0).fillna(0.0)
    roi_center_distance = series_or_default(df, "distance_to_nearest_roi_center_m", 999.0).fillna(999.0)
    rule_rank = series_or_default(df, "rule_rank", 999.0).fillna(999.0)
    source_priority = series_or_default(df, "_source_priority", 0.0).fillna(0.0)

    score += (~covers_any).astype(float) * 50.0
    score += series_or_default(df, "intersects_evidence_roi", False).fillna(False).astype(bool).astype(float) * 32.0
    score += series_or_default(df, "intersects_corridor_roi", False).fillna(False).astype(bool).astype(float) * 26.0
    score += series_or_default(df, "intersects_top1_roi", False).fillna(False).astype(bool).astype(float) * 12.0
    score += (
        series_or_default(df, "intersects_protected_current_roi", False)
        .fillna(False)
        .astype(bool)
        .astype(float)
        * 8.0
    )
    score += (road_distance <= 12.0).astype(float) * 24.0
    score += ((road_distance > 12.0) & (road_distance <= 50.0)).astype(float) * 7.0
    score += ((min_point_distance >= 5.0) & (min_point_distance <= 220.0)).astype(float) * 24.0
    score += (min_point_distance <= 80.0).astype(float) * 7.0
    score += ((area >= 40.0) & (area <= 8000.0)).astype(float) * 13.0
    score += ((area > 8000.0) & (area <= 60000.0)).astype(float) * 5.0
    score -= (area > 120000.0).astype(float) * 20.0
    score += source_priority * 3.0
    score += roi_score * 0.8
    score -= roi_center_distance * 0.04
    score -= rule_rank * 0.12
    return score


def relation_case(group: pd.DataFrame, *, low_confidence_threshold: float) -> bool:
    confidence = group["best_confidence"].iloc[0] if "best_confidence" in group else None
    low_confidence = pd.isna(confidence) or float(confidence) < low_confidence_threshold
    address = str(group["original_address"].iloc[0]) if "original_address" in group else ""
    return low_confidence or bool(RELATION_ADDRESS_PATTERN.search(address))


def relation_context_case(group: pd.DataFrame, *, low_confidence_threshold: float) -> bool:
    if relation_case(group, low_confidence_threshold=low_confidence_threshold):
        return True
    address = str(group["original_address"].iloc[0]) if "original_address" in group else ""
    if RELATION_CONTEXT_EXTRA_RE.search(address):
        return True
    road_context = bool(ROAD_SUFFIX_CONTEXT_RE.search(address))
    has_number = bool(LEADING_STREET_NUMBER_RE.search(address))
    return road_context and not has_number


def candidate_sort(group: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["rule_rank", "rule_score", "candidate_id"]
    ascending = [True, False, True]
    existing_columns = [column for column in sort_columns if column in group]
    existing_ascending = [ascending[sort_columns.index(column)] for column in existing_columns]
    return group.sort_values(existing_columns, ascending=existing_ascending)


def append_unique(selected: list[int], candidates: pd.DataFrame, limit: int, max_size: int) -> None:
    for index in candidates.index:
        if index in selected:
            continue
        selected.append(index)
        limit -= 1
        if len(selected) >= max_size or limit <= 0:
            break


def similar_footprint(
    row: pd.Series,
    selected_rows: pd.DataFrame,
    *,
    max_centroid_distance_m: float,
    max_area_ratio: float,
) -> bool:
    if selected_rows.empty:
        return False
    required = ["candidate_centroid_easting", "candidate_centroid_northing", "candidate_area_m2"]
    if any(column not in row.index for column in required) or any(column not in selected_rows for column in required):
        return False
    row_x = row.get("candidate_centroid_easting")
    row_y = row.get("candidate_centroid_northing")
    row_area = row.get("candidate_area_m2")
    if pd.isna(row_x) or pd.isna(row_y):
        return False
    dx = selected_rows["candidate_centroid_easting"] - float(row_x)
    dy = selected_rows["candidate_centroid_northing"] - float(row_y)
    distance = (dx.pow(2) + dy.pow(2)).pow(0.5)
    close = distance <= max_centroid_distance_m
    if not close.any():
        return False

    selected_area = pd.to_numeric(selected_rows.loc[close, "candidate_area_m2"], errors="coerce")
    if pd.isna(row_area) or float(row_area) <= 0:
        return True
    row_area_float = float(row_area)
    area_min = selected_area.clip(lower=1e-6).combine(pd.Series(row_area_float, index=selected_area.index), min)
    area_max = selected_area.combine(pd.Series(row_area_float, index=selected_area.index), max)
    area_ratio = area_max / area_min
    return bool((area_ratio <= max_area_ratio).any())


def diverse_footprint_alternates(
    ordered: pd.DataFrame,
    selected: list[int],
    *,
    max_size: int,
    max_rule_rank: int,
    max_centroid_distance_m: float,
    max_area_ratio: float,
) -> pd.DataFrame:
    selected_rows = ordered.loc[selected] if selected else ordered.iloc[[]]
    accepted: list[int] = []
    for index, row in ordered.iterrows():
        if index in selected or index in accepted:
            continue
        if "rule_rank" in row and pd.notna(row["rule_rank"]) and float(row["rule_rank"]) > max_rule_rank:
            break
        current_selected = pd.concat([selected_rows, ordered.loc[accepted]]) if accepted else selected_rows
        if similar_footprint(
            row,
            current_selected,
            max_centroid_distance_m=max_centroid_distance_m,
            max_area_ratio=max_area_ratio,
        ):
            continue
        accepted.append(index)
        if len(selected) + len(accepted) >= max_size:
            break
    return ordered.loc[accepted] if accepted else ordered.iloc[[]]


def centroid_distance(row_a: pd.Series, row_b: pd.Series) -> float | None:
    ax = row_a.get("candidate_centroid_easting")
    ay = row_a.get("candidate_centroid_northing")
    bx = row_b.get("candidate_centroid_easting")
    by = row_b.get("candidate_centroid_northing")
    if pd.isna(ax) or pd.isna(ay) or pd.isna(bx) or pd.isna(by):
        return None
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def polygon_centroid_clusters(
    ordered: pd.DataFrame,
    *,
    max_rule_rank: int,
    centroid_distance_m: float,
) -> list[list[int]]:
    candidates = ordered[ordered["rule_rank"] <= max_rule_rank].copy() if "rule_rank" in ordered else ordered.copy()
    indices = list(candidates.index)
    if not indices:
        return []

    parent = list(range(len(indices)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    coords = candidates[["candidate_centroid_easting", "candidate_centroid_northing"]].to_numpy(dtype=float)
    valid = np.isfinite(coords).all(axis=1)
    max_distance_sq = centroid_distance_m * centroid_distance_m
    valid_positions = np.flatnonzero(valid)
    for offset, left in enumerate(valid_positions):
        deltas = coords[valid_positions[offset + 1 :]] - coords[left]
        if len(deltas) == 0:
            continue
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        for right in valid_positions[offset + 1 :][distances_sq <= max_distance_sq]:
            union(int(left), int(right))

    grouped: dict[int, list[int]] = {}
    for position, index in enumerate(indices):
        grouped.setdefault(find(position), []).append(index)
    return list(grouped.values())


def cluster_sort_key(group: pd.DataFrame, member_indices: list[int], order: str) -> tuple[float, ...]:
    members = group.loc[member_indices]
    min_rank = float(pd.to_numeric(members.get("rule_rank"), errors="coerce").min())
    max_score = float(pd.to_numeric(members.get("rule_score"), errors="coerce").max())
    sum_score = float(pd.to_numeric(members.get("rule_score"), errors="coerce").sum())
    min_center = float(pd.to_numeric(members.get("distance_to_nearest_roi_center_m"), errors="coerce").min())
    if math.isnan(min_center):
        min_center = 999999.0
    if order == "sum_score":
        return (-sum_score, min_rank, min_center)
    if order == "max_score":
        return (-max_score, min_rank, min_center)
    if order == "center":
        return (min_center, min_rank, -max_score)
    return (min_rank, -max_score, min_center)


def rank_centroid_clusters(
    group: pd.DataFrame,
    *,
    cluster_centroid_distance_m: float,
    max_cluster_rule_rank: int,
    cluster_order: str,
) -> pd.DataFrame:
    ordered = candidate_sort(group)
    clusters = polygon_centroid_clusters(
        ordered,
        max_rule_rank=max_cluster_rule_rank,
        centroid_distance_m=cluster_centroid_distance_m,
    )
    clusters = sorted(clusters, key=lambda member_indices: cluster_sort_key(ordered, member_indices, cluster_order))

    rows: list[pd.Series] = []
    used_indices: set[int] = set()
    for cluster_rank, member_indices in enumerate(clusters, start=1):
        members = ordered.loc[member_indices]
        representative_index = candidate_sort(members).index[0]
        row = ordered.loc[representative_index].copy()
        used_indices.update(member_indices)
        member_ids = members.get("candidate_id", pd.Series(member_indices, index=members.index)).fillna("").astype(str)
        row["polygon_rank"] = cluster_rank
        row["polygon_selection_bucket"] = "centroid_cluster"
        row["polygon_cluster_member_count"] = int(len(member_indices))
        row["polygon_cluster_member_ids"] = "|".join(member_ids.tolist())
        row["polygon_cluster_min_rule_rank"] = float(pd.to_numeric(members.get("rule_rank"), errors="coerce").min())
        row["polygon_cluster_max_rule_score"] = float(pd.to_numeric(members.get("rule_score"), errors="coerce").max())
        row["polygon_cluster_sum_rule_score"] = float(pd.to_numeric(members.get("rule_score"), errors="coerce").sum())
        if "truth_intersects" in members:
            row["truth_intersects"] = bool(members["truth_intersects"].any())
            row["polygon_cluster_truth_member_count"] = int(members["truth_intersects"].sum())
        rows.append(row)

    for index in ordered.index:
        if index in used_indices:
            continue
        row = ordered.loc[index].copy()
        row["polygon_rank"] = len(rows) + 1
        row["polygon_selection_bucket"] = "rule_remainder"
        row["polygon_cluster_member_count"] = 1
        row["polygon_cluster_member_ids"] = str(row.get("candidate_id", index))
        row["polygon_cluster_min_rule_rank"] = row.get("rule_rank")
        row["polygon_cluster_max_rule_score"] = row.get("rule_score")
        row["polygon_cluster_sum_rule_score"] = row.get("rule_score")
        if "truth_intersects" in row:
            row["polygon_cluster_truth_member_count"] = int(bool(row["truth_intersects"]))
        rows.append(row)

    return pd.DataFrame(rows)


def road_center_alternate(group: pd.DataFrame, *, max_road_distance_m: float, max_area_m2: float) -> pd.DataFrame:
    required = ["distance_to_mentioned_road_m", "candidate_area_m2", "distance_to_nearest_roi_center_m"]
    if any(column not in group for column in required):
        return group.iloc[[]]
    candidates = group[
        (group["distance_to_mentioned_road_m"] <= max_road_distance_m)
        & (group["candidate_area_m2"].between(20, max_area_m2))
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        [
            "distance_to_nearest_roi_center_m",
            "distance_to_mentioned_road_m",
            "_source_priority",
            "roi_min_rank",
            "rule_rank",
            "candidate_id",
        ],
        ascending=[True, True, False, True, True, True],
    )


def nearest_noncover_alternate(group: pd.DataFrame, *, max_point_distance_m: float, max_area_m2: float) -> pd.DataFrame:
    required = ["_covers_any_anchor_point", "_min_point_distance_m", "candidate_area_m2"]
    if any(column not in group for column in required):
        return group.iloc[[]]
    candidates = group[
        (~group["_covers_any_anchor_point"])
        & (group["_min_point_distance_m"] <= max_point_distance_m)
        & (group["candidate_area_m2"].between(20, max_area_m2))
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        [
            "_min_point_distance_m",
            "_source_priority",
            "distance_to_nearest_roi_center_m",
            "distance_to_mentioned_road_m",
            "rule_rank",
            "candidate_id",
        ],
        ascending=[True, False, True, True, True, True],
    )


def bounded_noncover_alternate(
    group: pd.DataFrame,
    *,
    max_point_distance_m: float,
    max_area_m2: float,
    max_rule_rank: int,
) -> pd.DataFrame:
    required = ["_covers_any_anchor_point", "_min_point_distance_m", "candidate_area_m2", "rule_rank"]
    if any(column not in group for column in required):
        return group.iloc[[]]
    candidates = group[
        (group["rule_rank"] <= max_rule_rank)
        & (~group["_covers_any_anchor_point"])
        & (group["_min_point_distance_m"] <= max_point_distance_m)
        & (group["candidate_area_m2"].between(20, max_area_m2))
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        [
            "_relation_alternate_score",
            "distance_to_nearest_roi_center_m",
            "rule_rank",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
    )


def evidence_spread_score(df: pd.DataFrame) -> pd.Series:
    covers_any = series_or_default(df, "_covers_any_anchor_point", False).fillna(False).astype(bool)
    area = series_or_default(df, "candidate_area_m2", 1e12).fillna(1e12)
    road_distance = series_or_default(df, "distance_to_mentioned_road_m", 9999.0).fillna(9999.0)
    roi_min_rank = series_or_default(df, "roi_min_rank", 999.0).fillna(999.0)
    roi_area = series_or_default(df, "roi_total_intersection_area_m2", 0.0).fillna(0.0)
    roi_center_distance = series_or_default(df, "distance_to_nearest_roi_center_m", 999.0).fillna(999.0)
    uprn_count = series_or_default(df, "uprn_count", 0.0).fillna(0.0)
    rule_rank = series_or_default(df, "rule_rank", 999.0).fillna(999.0)

    score = pd.Series(0.0, index=df.index)
    score += (~covers_any).astype(float) * 30.0
    score += (roi_min_rank > 1).astype(float) * 25.0
    score += ((roi_min_rank >= 2) & (roi_min_rank <= 5)).astype(float) * 12.0
    score += (roi_center_distance <= 80.0).astype(float) * 20.0
    score += (road_distance <= 15.0).astype(float) * 18.0
    score += ((area >= 40.0) & (area <= 8000.0)).astype(float) * 10.0
    score += roi_area.clip(upper=3000.0) / 150.0
    score += uprn_count.clip(upper=3.0) * 2.0
    score -= rule_rank * 0.03
    return score


def evidence_spread_alternates(group: pd.DataFrame, selected: list[int], *, max_rule_rank: int) -> pd.DataFrame:
    if "rule_rank" not in group:
        return group.iloc[[]]
    candidates = group[(group["rule_rank"] <= max_rule_rank) & (~group.index.isin(selected))].copy()
    if candidates.empty:
        return candidates
    candidates["_evidence_spread_score"] = evidence_spread_score(candidates)
    return candidates.sort_values(
        ["_evidence_spread_score", "rule_rank", "candidate_id"],
        ascending=[False, True, True],
    )


def select_case_rows(
    group: pd.DataFrame,
    *,
    top_k: int,
    strategy: str,
    low_confidence_threshold: float,
    preserve_rule_top_n: int,
    max_road_distance_m: float,
    max_point_distance_m: float,
    max_area_m2: float,
    alternate_confidence_threshold: float,
    max_alternate_rule_rank: int,
    diversity_centroid_distance_m: float,
    diversity_area_ratio: float,
    max_diversity_rule_rank: int,
) -> tuple[list[int], dict[int, str]]:
    ordered = candidate_sort(group)
    selected: list[int] = []
    buckets: dict[int, str] = {}

    def add(candidates: pd.DataFrame, count: int, bucket: str) -> None:
        before = set(selected)
        append_unique(selected, candidates, count, top_k)
        for index in selected:
            if index not in before and index not in buckets:
                buckets[index] = bucket

    if strategy == "rule":
        add(ordered, top_k, "rule_order")
        return selected[:top_k], buckets

    confidence = group["best_confidence"].iloc[0] if "best_confidence" in group else None
    low_confidence = pd.isna(confidence) or float(confidence) < low_confidence_threshold

    if strategy == "conservative_road":
        if not low_confidence:
            add(ordered, top_k, "rule_order_high_conf")
            return selected[:top_k], buckets
        add(ordered.head(max(0, min(top_k, preserve_rule_top_n))), preserve_rule_top_n, "rule_preserved")
        add(
            road_center_alternate(group, max_road_distance_m=max_road_distance_m, max_area_m2=max_area_m2),
            top_k - len(selected),
            "road_roi_center_alternate",
        )
        add(ordered, top_k - len(selected), "rule_fill")
        return selected[:top_k], buckets

    if strategy == "portfolio":
        add(ordered.head(min(top_k, preserve_rule_top_n)), preserve_rule_top_n, "rule_preserved")
        add(
            nearest_noncover_alternate(group, max_point_distance_m=max_point_distance_m, max_area_m2=max_area_m2),
            1,
            "nearest_noncover_alternate",
        )
        add(
            road_center_alternate(group, max_road_distance_m=max_road_distance_m, max_area_m2=max_area_m2),
            top_k - len(selected),
            "road_roi_center_alternate",
        )
        add(ordered, top_k - len(selected), "rule_fill")
        return selected[:top_k], buckets

    if strategy == "bounded_noncover":
        confidence = group["best_confidence"].iloc[0] if "best_confidence" in group else None
        alternate_confidence = pd.isna(confidence) or float(confidence) < alternate_confidence_threshold
        if not alternate_confidence or not relation_case(group, low_confidence_threshold=low_confidence_threshold):
            add(ordered, top_k, "rule_order_non_relation")
            return selected[:top_k], buckets
        add(ordered.head(max(0, min(top_k, preserve_rule_top_n))), preserve_rule_top_n, "rule_preserved")
        add(
            bounded_noncover_alternate(
                group,
                max_point_distance_m=max_point_distance_m,
                max_area_m2=max_area_m2,
                max_rule_rank=max_alternate_rule_rank,
            ),
            1,
            "bounded_noncover_alternate",
        )
        add(ordered, top_k - len(selected), "rule_fill")
        return selected[:top_k], buckets

    if strategy == "relation_evidence_mix":
        if not relation_context_case(group, low_confidence_threshold=low_confidence_threshold):
            add(ordered, top_k, "rule_order_non_relation")
            return selected[:top_k], buckets
        preserved = min(max(0, preserve_rule_top_n), top_k)
        add(ordered.head(preserved), preserved, "rule_preserved")
        add(
            evidence_spread_alternates(group, selected, max_rule_rank=max_alternate_rule_rank),
            top_k - len(selected),
            "evidence_spread_alternate",
        )
        add(ordered, top_k - len(selected), "rule_fill")
        return selected[:top_k], buckets

    if strategy == "diverse_footprint":
        add(ordered.head(1), 1, "rule_top1_preserved")
        add(
            diverse_footprint_alternates(
                ordered,
                selected,
                max_size=top_k,
                max_rule_rank=max_diversity_rule_rank,
                max_centroid_distance_m=diversity_centroid_distance_m,
                max_area_ratio=diversity_area_ratio,
            ),
            top_k - len(selected),
            "diverse_footprint",
        )
        add(ordered, top_k - len(selected), "rule_fill")
        return selected[:top_k], buckets

    raise ValueError(f"Unknown strategy: {strategy}")


def rank_case(
    group: pd.DataFrame,
    *,
    top_k: int,
    strategy: str,
    low_confidence_threshold: float,
    preserve_rule_top_n: int,
    max_road_distance_m: float,
    max_point_distance_m: float,
    max_area_m2: float,
    alternate_confidence_threshold: float,
    max_alternate_rule_rank: int,
    diversity_centroid_distance_m: float,
    diversity_area_ratio: float,
    max_diversity_rule_rank: int,
    cluster_centroid_distance_m: float,
    max_cluster_rule_rank: int,
    cluster_order: str,
) -> pd.DataFrame:
    if strategy == "centroid_cluster":
        return rank_centroid_clusters(
            group,
            cluster_centroid_distance_m=cluster_centroid_distance_m,
            max_cluster_rule_rank=max_cluster_rule_rank,
            cluster_order=cluster_order,
        )

    selected, buckets = select_case_rows(
        group,
        top_k=top_k,
        strategy=strategy,
        low_confidence_threshold=low_confidence_threshold,
        preserve_rule_top_n=preserve_rule_top_n,
        max_road_distance_m=max_road_distance_m,
        max_point_distance_m=max_point_distance_m,
        max_area_m2=max_area_m2,
        alternate_confidence_threshold=alternate_confidence_threshold,
        max_alternate_rule_rank=max_alternate_rule_rank,
        diversity_centroid_distance_m=diversity_centroid_distance_m,
        diversity_area_ratio=diversity_area_ratio,
        max_diversity_rule_rank=max_diversity_rule_rank,
    )
    ordered = candidate_sort(group)
    final_indices = list(selected)
    for index in ordered.index:
        if index not in final_indices:
            final_indices.append(index)
    ranked = group.loc[final_indices].copy()
    ranked["polygon_rank"] = range(1, len(ranked) + 1)
    ranked["polygon_selection_bucket"] = [buckets.get(index, "rule_remainder") for index in ranked.index]
    return ranked


def summarize_hits(df: pd.DataFrame, rank_column: str, all_cases: pd.Index, label: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"label": label, "cases": int(len(all_cases))}
    if "truth_intersects" not in df:
        return summary
    if "truth_available" in df and not bool(df["truth_available"].fillna(False).any()):
        return summary
    for k in [1, 3, 5, 8, 10, 20, 50, 100, 200]:
        hits = (
            df[(df[rank_column] <= k) & df["truth_intersects"]]
            .groupby("case_key")
            .size()
            .reindex(all_cases, fill_value=0)
            > 0
        )
        summary[f"top{k}"] = int(hits.sum())
        summary[f"top{k}_rate"] = float(hits.mean())
    present = df[df["truth_intersects"]].groupby("case_key").size().reindex(all_cases, fill_value=0) > 0
    summary["present"] = int(present.sum())
    summary["present_rate"] = float(present.mean())
    return summary


def confidence_bucket_summary(selected: pd.DataFrame, all_cases: pd.Index, top_k: int) -> dict[str, Any]:
    if "truth_intersects" not in selected or "best_confidence" not in selected:
        return {}
    if "truth_available" in selected and not bool(selected["truth_available"].fillna(False).any()):
        return {}
    case_meta = selected.groupby("case_key").first(numeric_only=False).reindex(all_cases)
    confidence = pd.to_numeric(case_meta["best_confidence"], errors="coerce")
    hits = (
        selected[(selected["polygon_rank"] <= top_k) & selected["truth_intersects"]]
        .groupby("case_key")
        .size()
        .reindex(all_cases, fill_value=0)
        > 0
    )
    out: dict[str, Any] = {}
    for name, mask in {
        "low_conf_lt75": confidence < 75,
        "mid_conf_75_80": (confidence >= 75) & (confidence < 80),
        "high_conf_ge80": confidence >= 80,
    }.items():
        bucket_hits = hits[mask.fillna(False)]
        out[name] = {
            "cases": int(mask.fillna(False).sum()),
            "topk": int(bucket_hits.sum()),
            "topk_rate": float(bucket_hits.mean()) if len(bucket_hits) else 0.0,
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(args.input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    df = coerce_input(df)
    if args.max_input_rank > 0 and "rule_rank" in df:
        df = df[df["rule_rank"] <= args.max_input_rank].copy()
    if args.candidate_source_filter:
        allowed_sources = {
            source.strip()
            for source in args.candidate_source_filter.split("|")
            if source.strip()
        }
        df = df[df["candidate_source"].astype(str).isin(allowed_sources)].copy()

    ranked_groups = [
        rank_case(
            group,
            top_k=args.top_k,
            strategy=args.strategy,
            low_confidence_threshold=args.low_confidence_threshold,
            preserve_rule_top_n=args.preserve_rule_top_n,
            max_road_distance_m=args.max_road_distance_m,
            max_point_distance_m=args.max_point_distance_m,
            max_area_m2=args.max_area_m2,
            alternate_confidence_threshold=args.alternate_confidence_threshold,
            max_alternate_rule_rank=args.max_alternate_rule_rank,
            diversity_centroid_distance_m=args.diversity_centroid_distance_m,
            diversity_area_ratio=args.diversity_area_ratio,
            max_diversity_rule_rank=args.max_diversity_rule_rank,
            cluster_centroid_distance_m=args.cluster_centroid_distance_m,
            max_cluster_rule_rank=args.max_cluster_rule_rank,
            cluster_order=args.cluster_order,
        )
        for _, group in df.groupby("case_key", sort=False)
    ]
    ranked = pd.concat(ranked_groups, ignore_index=True) if ranked_groups else df.iloc[[]].copy()
    all_cases = pd.Index(
        sorted(ranked["case_key"].unique(), key=lambda value: (len(str(value)), str(value))),
        name="case_key",
    )

    topn = ranked[ranked["polygon_rank"] <= args.output_top_n].copy()
    selected = ranked[ranked["polygon_rank"] <= args.top_k].copy()

    summary: dict[str, Any] = {
        "input_csv": str(args.input_csv),
        "output_prefix": str(args.output_prefix),
        "strategy": args.strategy,
        "candidate_source_filter": args.candidate_source_filter,
        "top_k": int(args.top_k),
        "output_top_n": int(args.output_top_n),
        "max_input_rank": int(args.max_input_rank),
        "low_confidence_threshold": float(args.low_confidence_threshold),
        "preserve_rule_top_n": int(args.preserve_rule_top_n),
        "max_road_distance_m": float(args.max_road_distance_m),
        "max_point_distance_m": float(args.max_point_distance_m),
        "max_area_m2": float(args.max_area_m2),
        "alternate_confidence_threshold": float(args.alternate_confidence_threshold),
        "max_alternate_rule_rank": int(args.max_alternate_rule_rank),
        "diversity_centroid_distance_m": float(args.diversity_centroid_distance_m),
        "diversity_area_ratio": float(args.diversity_area_ratio),
        "max_diversity_rule_rank": int(args.max_diversity_rule_rank),
        "cluster_centroid_distance_m": float(args.cluster_centroid_distance_m),
        "max_cluster_rule_rank": int(args.max_cluster_rule_rank),
        "cluster_order": args.cluster_order,
        "cases": int(len(all_cases)),
        "candidate_rows": int(len(df)),
        "ranked_rows": int(len(ranked)),
        "selection_bucket_counts": selected["polygon_selection_bucket"].value_counts(dropna=False).to_dict(),
        "baseline_rule": summarize_hits(ranked, "rule_rank", all_cases, "baseline_rule"),
        "polygon_rank": summarize_hits(ranked, "polygon_rank", all_cases, "polygon_rank"),
        "confidence_buckets": confidence_bucket_summary(ranked, all_cases, args.top_k),
    }

    truth_enabled = "truth_intersects" in ranked and (
        "truth_available" not in ranked or bool(ranked["truth_available"].fillna(False).any())
    )
    if truth_enabled:
        baseline_hits = (
            ranked[(ranked["rule_rank"] <= args.top_k) & ranked["truth_intersects"]]
            .groupby("case_key")
            .size()
            .reindex(all_cases, fill_value=0)
            > 0
        )
        polygon_hits = (
            ranked[(ranked["polygon_rank"] <= args.top_k) & ranked["truth_intersects"]]
            .groupby("case_key")
            .size()
            .reindex(all_cases, fill_value=0)
            > 0
        )
        recovered = (polygon_hits & ~baseline_hits)
        lost = (baseline_hits & ~polygon_hits)
        summary["recovered_cases"] = int(recovered.sum())
        summary["lost_cases"] = int(lost.sum())
        summary["net_gain_cases"] = int(polygon_hits.sum() - baseline_hits.sum())
        summary["recovered_case_keys"] = recovered[recovered].index.astype(str).tolist()
        summary["lost_case_keys"] = lost[lost].index.astype(str).tolist()

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ranked_csv = output_prefix.with_name(output_prefix.name + f"_ranked_top{args.output_top_n}.csv")
    selected_csv = output_prefix.with_name(output_prefix.name + f"_selected_top{args.top_k}.csv")
    case_csv = output_prefix.with_name(output_prefix.name + "_case_summary.csv")
    summary_json = output_prefix.with_suffix(".summary.json")

    topn.to_csv(ranked_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    case_rows = []
    for case_key, group in ranked.groupby("case_key", sort=False):
        top = group.sort_values("polygon_rank").iloc[0]
        hit_rank = None
        if truth_enabled:
            hits = group[group["truth_intersects"]].sort_values("polygon_rank")
            if not hits.empty:
                hit_rank = int(hits.iloc[0]["polygon_rank"])
        case_rows.append(
            {
                "case_key": case_key,
                "base_key": top.get("base_key"),
                "original_address": top.get("original_address"),
                "best_confidence": top.get("best_confidence"),
                "polygon_hit_rank": hit_rank,
                "top1_candidate_id": top.get("candidate_id"),
                "top1_source": top.get("candidate_source"),
                "top1_bucket": top.get("polygon_selection_bucket"),
                "top1_truth_intersects": bool(top.get("truth_intersects")) if truth_enabled else None,
            }
        )
    pd.DataFrame(case_rows).to_csv(case_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"ranked_csv={ranked_csv}")
    print(f"selected_csv={selected_csv}")
    print(f"case_csv={case_csv}")
    print(f"summary_json={summary_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="No-AI polygon ranking stage.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--strategy",
        choices=[
            "rule",
            "conservative_road",
            "portfolio",
            "bounded_noncover",
            "diverse_footprint",
            "relation_evidence_mix",
            "centroid_cluster",
        ],
        default="centroid_cluster",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-top-n", type=int, default=100)
    parser.add_argument("--max-input-rank", type=int, default=200)
    parser.add_argument(
        "--candidate-source-filter",
        default="",
        help="Optional pipe-separated candidate_source allow-list, e.g. council_cadastral.",
    )
    parser.add_argument("--low-confidence-threshold", type=float, default=75.0)
    parser.add_argument("--preserve-rule-top-n", type=int, default=4)
    parser.add_argument("--max-road-distance-m", type=float, default=12.0)
    parser.add_argument("--max-point-distance-m", type=float, default=300.0)
    parser.add_argument("--max-area-m2", type=float, default=100000.0)
    parser.add_argument("--alternate-confidence-threshold", type=float, default=80.0)
    parser.add_argument("--max-alternate-rule-rank", type=int, default=200)
    parser.add_argument("--diversity-centroid-distance-m", type=float, default=8.0)
    parser.add_argument("--diversity-area-ratio", type=float, default=5.0)
    parser.add_argument("--max-diversity-rule-rank", type=int, default=8)
    parser.add_argument("--cluster-centroid-distance-m", type=float, default=80.0)
    parser.add_argument("--max-cluster-rule-rank", type=int, default=100)
    parser.add_argument("--cluster-order", choices=["min_rank", "max_score", "sum_score", "center"], default="sum_score")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
