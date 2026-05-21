#!/usr/bin/env python3
"""Adapt ``1_address_to_point_gemini.py`` output into candidate-box inputs.

This stage treats ``spatial_capture_production/1_address_to_point_gemini.py``
as an upstream black box.  It does not import or modify that script; it only
normalizes the JSON/JSONL evidence it writes into the tabular contracts used by
the geocoding-box polygon pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence_roi as roi_v1
from stage_config import parse_bbox, parse_names


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def parse_float(value: Any) -> float | None:
    return roi_v1.parse_float(value)


def suffix(prefix: Path, ending: str) -> Path:
    return prefix.with_name(prefix.name + ending)


def load_address_point_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload_rows = payload.get("rows") or payload.get("results") or []
    else:
        payload_rows = payload
    return [item for item in payload_rows if isinstance(item, dict)] if isinstance(payload_rows, list) else []


def row_key(row: dict[str, Any], idx: int) -> str:
    key = clean_text(row.get("variant_key") or row.get("key") or row.get("unique_key"))
    return key or str(idx)


def base_key(row: dict[str, Any], key: str) -> str:
    if row.get("_address_point_gpkg_primary"):
        return clean_text(row.get("unique_key") or row.get("key") or key) or key
    return clean_text(row.get("_source_unique_key") or row.get("unique_key") or key) or key


def json_group_key(row: dict[str, Any]) -> str:
    key = clean_text(row.get("_source_unique_key") or row.get("unique_key") or row.get("key") or row.get("variant_key"))
    if not key:
        return ""
    return key.split("_", 1)[0]


def json_variant_key(row: dict[str, Any]) -> str:
    return clean_text(row.get("variant_key") or row.get("key") or row.get("unique_key"))


def confidence_value(row: dict[str, Any]) -> float:
    value = parse_float(row.get("best_confidence"))
    return value if value is not None else -1.0


def has_best_point(row: dict[str, Any]) -> bool:
    return parse_float(first_nonempty(row, ["best_easting_27700_final", "best_anchor_easting_27700"])) is not None and parse_float(
        first_nonempty(row, ["best_northing_27700_final", "best_anchor_northing_27700"])
    ) is not None


ROAD_SUFFIX_RE = re.compile(
    r"\b(ROAD|STREET|LANE|AVENUE|CLOSE|WAY|DRIVE|GROVE|CRESCENT|HILL|GATE|WALK|PLACE|COURT|PARK|TERRACE|ROW|YARD|MEWS|RISE|VIEW|SQUARE)\b",
    re.I,
)
HOUSE_RANGE_RE = re.compile(r"\b(\d{1,4})[A-Z]?\s*(?:-|–|—|TO)\s*(\d{1,4})[A-Z]?\b", re.I)
INTERNAL_RANGE_PREFIX_RE = re.compile(r"\b(UNIT|UNITS|PLOT|PLOTS|BLOCK|FLAT|FLATS|APARTMENT|APARTMENTS|PHASE|PARCEL)\b", re.I)
LEADING_NUMBER_RE = re.compile(r"^\s*(\d{1,4})[A-Z]?\b", re.I)


def road_same_parity_ranges(address: Any) -> list[dict[str, Any]]:
    """Find conservative house-number ranges such as ``56-58 Westfield Lane``.

    UK road addresses usually keep odd/even numbers on opposite sides.  For a
    same-parity road range, expanding every integer can invent a wrong middle
    address on the other side of town.  This helper only fires when the range is
    followed by a road suffix and is not immediately introduced as Unit/Plot.
    """
    text = clean_text(address).upper()
    specs: list[dict[str, Any]] = []
    for match in HOUSE_RANGE_RE.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2))
        low, high = sorted((start, end))
        if not (0 < low < 1000 and 0 < high < 1000 and high - low <= 20):
            continue
        if start % 2 != end % 2:
            continue
        prefix = text[max(0, match.start() - 36) : match.start()]
        if INTERNAL_RANGE_PREFIX_RE.search(prefix):
            continue
        tail = text[match.end() : match.end() + 90]
        if not ROAD_SUFFIX_RE.search(tail):
            continue
        step = 2
        specs.append(
            {
                "low": low,
                "high": high,
                "start": start,
                "end": end,
                "numbers": list(range(low, high + 1, step)),
                "range_text": f"{low}-{high}",
            }
        )
    return specs


def leading_house_number(value: Any) -> int | None:
    match = LEADING_NUMBER_RE.search(clean_text(value))
    if not match:
        return None
    number = int(match.group(1))
    return number if 0 < number < 1000 else None


def best_point_from_row(row: dict[str, Any]) -> Point | None:
    x = parse_float(first_nonempty(row, ["best_easting_27700_final", "best_anchor_easting_27700"]))
    y = parse_float(first_nonempty(row, ["best_northing_27700_final", "best_anchor_northing_27700"]))
    if x is None or y is None:
        return None
    return Point(x, y)


def max_pairwise_distance(points: list[Point]) -> float:
    if len(points) <= 1:
        return 0.0
    out = 0.0
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            out = max(out, float(points[left].distance(points[right])))
    return out


def range_parity_corrections(case_row: dict[str, Any], evidence_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    address = first_nonempty(case_row, ["original_address", "raw_address", "chargegeog", "address"])
    specs = road_same_parity_ranges(address)
    if not specs or not evidence_group:
        return []

    corrections: list[dict[str, Any]] = []
    for spec in specs:
        expected_numbers = set(spec["numbers"])
        all_numbers = set(range(spec["low"], spec["high"] + 1))
        expected_rows: list[tuple[int, dict[str, Any], Point]] = []
        other_points: list[Point] = []
        for child in evidence_group:
            child_address = first_nonempty(child, ["original_address", "raw_address", "best_address_final", "address"])
            if HOUSE_RANGE_RE.search(clean_text(child_address)):
                continue
            number = leading_house_number(child_address)
            if number is None or number not in all_numbers:
                continue
            point = best_point_from_row(child)
            if point is None:
                continue
            if number in expected_numbers:
                expected_rows.append((number, child, point))
            else:
                other_points.append(point)

        unique_expected = {number for number, _, _ in expected_rows}
        if len(unique_expected) < 2:
            continue
        expected_points = [point for _, _, point in expected_rows]
        radius = max_pairwise_distance(expected_points)
        if radius > 180:
            continue
        other_distance_value = None
        other_distance = ""
        if other_points:
            other_distance_value = min(point.distance(other) for point in expected_points for other in other_points)
            other_distance = round(other_distance_value, 3)
        if other_distance_value is None or other_distance_value < 70:
            continue

        cx = sum(point.x for point in expected_points) / len(expected_points)
        cy = sum(point.y for point in expected_points) / len(expected_points)
        label = f"range parity {spec['range_text']} via {','.join(str(n) for n in sorted(unique_expected))}"
        corrections.append(
            {
                "address": label,
                "range_text": spec["range_text"],
                "expected_numbers": sorted(expected_numbers),
                "matched_numbers": sorted(unique_expected),
                "point_role": "range_parity_centroid",
                "easting_27700": round(cx, 3),
                "northing_27700": round(cy, 3),
                "weight": 12.0,
                "expected_cluster_radius_m": round(radius, 3),
                "other_to_expected_min_m": other_distance,
                "reason": "same_parity_road_range_children",
            }
        )
        for number, child, point in sorted(expected_rows, key=lambda item: item[0])[:6]:
            corrections.append(
                {
                    "address": first_nonempty(child, ["original_address", "best_address_final", "raw_address"]) or label,
                    "number": number,
                    "range_text": spec["range_text"],
                    "expected_numbers": sorted(expected_numbers),
                    "matched_numbers": sorted(unique_expected),
                    "point_role": "range_parity_child",
                    "easting_27700": round(float(point.x), 3),
                    "northing_27700": round(float(point.y), 3),
                    "weight": 8.5,
                    "expected_cluster_radius_m": round(radius, 3),
                    "other_to_expected_min_m": other_distance,
                    "reason": "same_parity_road_range_child",
                }
            )
    return corrections


def index_address_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = json_group_key(row)
        if key:
            grouped.setdefault(key, []).append(row)
    return grouped


def choose_address_evidence(case_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    exact = [
        row
        for row in rows
        if json_variant_key(row) == case_key
        or clean_text(row.get("key")) == case_key
        or clean_text(row.get("unique_key")) == case_key
    ]
    candidates = exact or rows
    return max(candidates, key=lambda row: (json_variant_key(row) == case_key, has_best_point(row), confidence_value(row)))


def serializable_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value


def load_address_point_gpkg_rows(path: Path, layer: str, key_column: str, address_column: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gdf = gpd.read_file(path, layer=layer)
    if key_column not in gdf.columns:
        raise ValueError(f"{path}:{layer} has no key column {key_column!r}")
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    geom_col = gdf.geometry.name
    rows: list[dict[str, Any]] = []
    geoms: dict[str, Any] = {}
    for _, source in gdf.iterrows():
        key = clean_text(source.get(key_column))
        if not key:
            continue
        record = {
            column: serializable_value(source.get(column))
            for column in gdf.columns
            if column != geom_col
        }
        address = clean_text(
            record.get(address_column)
            or record.get("original_address")
            or record.get("raw_address")
            or record.get("chargegeog")
            or record.get("address")
        )
        record["key"] = key
        record["unique_key"] = key
        record["_source_unique_key"] = key
        record["variant_key"] = key
        record["original_address"] = address
        if "raw_address" not in record or not clean_text(record.get("raw_address")):
            record["raw_address"] = address
        if address_column and address_column not in record:
            record[address_column] = address
        rows.append(record)
        geom = source.geometry
        if geom is not None and not geom.is_empty:
            geoms[key] = geom
    return rows, geoms


def merge_case_rows_with_address_evidence(
    case_rows: list[dict[str, Any]],
    address_rows: list[dict[str, Any]],
    enable_range_parity_correction: bool = True,
) -> list[dict[str, Any]]:
    grouped = index_address_rows(address_rows)
    merged: list[dict[str, Any]] = []
    for case_row in case_rows:
        key = clean_text(case_row.get("unique_key") or case_row.get("key"))
        evidence_group = grouped.get(key, [])
        evidence = choose_address_evidence(key, evidence_group)
        row = dict(evidence)
        for field, value in case_row.items():
            if clean_text(row.get(field)) == "":
                row[field] = value
        row["key"] = key
        row["unique_key"] = key
        row["_source_unique_key"] = key
        row["variant_key"] = key
        row["_address_point_gpkg_primary"] = True
        row["_address_point_json_rows"] = len(evidence_group)
        row["_address_point_json_variant_keys"] = "|".join(
            item for item in (json_variant_key(candidate) for candidate in evidence_group) if item
        )
        if clean_text(case_row.get("original_address")):
            row["original_address"] = case_row["original_address"]
        if clean_text(case_row.get("raw_address")):
            row["raw_address"] = case_row["raw_address"]
        corrections = range_parity_corrections(row, evidence_group) if enable_range_parity_correction else []
        row["range_parity_points_json"] = json.dumps(corrections, ensure_ascii=False)
        row["_range_parity_correction_count"] = len(corrections)
        row["_range_parity_correction_reason"] = "|".join(sorted({clean_text(item.get("reason")) for item in corrections if item.get("reason")}))
        merged.append(row)
    return merged


def json_list_text(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "[]"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return "[]"
        return json.dumps(parsed if isinstance(parsed, list) else [], ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps([item for item in value if isinstance(item, dict)], ensure_ascii=False)
    return "[]"


def first_nonempty(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if clean_text(value) != "":
            return value
    return ""


def normalized_raw_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    key = row_key(row, idx)
    parent = base_key(row, key)
    out = dict(row)
    out["key"] = key
    out["unique_key"] = clean_text(out.get("unique_key") or parent)
    out["_source_unique_key"] = parent
    out["variant_key"] = clean_text(out.get("variant_key") or key)
    out["original_address"] = clean_text(
        first_nonempty(out, ["original_address", "raw_address", "chargegeog", "address"])
    )
    for field in ("os_candidates_json", "gog_candidates_json", "os_fallback_pool_json", "range_parity_points_json"):
        out[field] = json_list_text(out.get(field))
    return out


def collect_roads_from_row(row: dict[str, Any]) -> list[str]:
    roads: list[str] = []
    for field in (
        "best_road_name",
        "best_road_anchor_name",
        "lexicon_road",
        "gog_best_road_name",
        "os_best_road_name",
        "os_raw_best_road_name",
    ):
        for part in clean_text(row.get(field)).split("|"):
            road = roi_v1.norm(part)
            if road and road not in roads:
                roads.append(road)

    for item in roi_v1.safe_json_list(row.get("road_candidates_json"))[:12]:
        for field in ("road_name", "matched_phrase"):
            road = roi_v1.norm(item.get(field))
            if road and road not in roads:
                roads.append(road)
    return roads


def v10_row(row: dict[str, Any]) -> dict[str, Any]:
    key = clean_text(row["key"])
    parent = base_key(row, key)
    best_x = first_nonempty(
        row,
        [
            "best_easting_27700_final",
            "best_anchor_easting_27700",
            "os_address_gemini_easting_27700",
            "gog_best_easting_27700",
            "lexicon_easting_27700",
        ],
    )
    best_y = first_nonempty(
        row,
        [
            "best_northing_27700_final",
            "best_anchor_northing_27700",
            "os_address_gemini_northing_27700",
            "gog_best_northing_27700",
            "lexicon_northing_27700",
        ],
    )
    source = clean_text(first_nonempty(row, ["best_source_final", "best_anchor_source", "selected_source"]))
    address = clean_text(first_nonempty(row, ["best_address_final", "best_anchor_name", "os_address_gemini"]))
    return {
        "key": key,
        "base_key": parent,
        "original_address": row.get("original_address") or "",
        "baseline_easting": best_x,
        "baseline_northing": best_y,
        "selected_easting": best_x,
        "selected_northing": best_y,
        "selected_source": source,
        "selected_address": address,
        "v7_selected_easting": best_x,
        "v7_selected_northing": best_y,
        "v7_selected_source": source,
        "v7_selected_address": address,
        "v7_selected_distance_m": row.get("best_distance_to_polygon_m_final") or "",
        "ocr_grid_raw": row.get("ocr_grid_raw") or "",
        "ocr_grid_method": row.get("ocr_grid_method") or "",
        "ocr_grid_gate": row.get("ocr_grid_gate") or "",
    }


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    key = clean_text(row["key"])
    parent = base_key(row, key)
    return {
        "key": key,
        "base_key": parent,
        "original_address": row.get("original_address") or "",
        "matched_roads": " | ".join(collect_roads_from_row(row)),
        "ocr_anchor_count": "",
        "notes": "from_address_point_evidence",
    }


def case_summary_row(row: dict[str, Any], points: list[roi_v1.EvidencePoint]) -> dict[str, Any]:
    key = clean_text(row["key"])
    parent = base_key(row, key)
    return {
        "key": key,
        "base_key": parent,
        "status": "ok" if points else "no_evidence_points",
        "sample_split": row.get("sample_split") or "",
        "original_address": row.get("original_address") or "",
        "best_confidence": row.get("best_confidence") or "",
        "best_source_final": row.get("best_source_final") or "",
        "best_address_final": row.get("best_address_final") or "",
        "range_parity_correction_count": row.get("_range_parity_correction_count") or 0,
        "evidence_point_count": len(points),
        "evidence_sources": "|".join(sorted({point.source for point in points})),
    }


def load_case_geometries(path: Path | None, layer: str | None, key_column: str) -> dict[str, Any]:
    if not path or not layer or not path.exists():
        return {}
    gdf = gpd.read_file(path, layer=layer, columns=[key_column])
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return {str(row[key_column]): row.geometry for _, row in gdf.iterrows()}


def roi_truth_metrics(geom: Any, target_geom: Any) -> dict[str, Any]:
    if target_geom is None or geom is None or geom.is_empty:
        return {
            "roi_intersects_target_polygon": "",
            "roi_contains_target_centroid": "",
            "roi_distance_to_target_m": "",
        }
    return {
        "roi_intersects_target_polygon": bool(geom.intersects(target_geom)),
        "roi_contains_target_centroid": bool(geom.contains(target_geom.centroid)),
        "roi_distance_to_target_m": round(float(geom.distance(target_geom)), 3),
    }


def rank_evidence_rois(
    points: list[roi_v1.EvidencePoint],
    side: float,
    max_rois: int,
    min_center_distance: float,
) -> list[dict[str, Any]]:
    if not points:
        return []
    half = side / 2.0
    centers: list[tuple[float, float, str]] = [(point.x, point.y, point.source) for point in points]
    for point in points:
        local = [other for other in points if abs(other.x - point.x) <= 140 and abs(other.y - point.y) <= 140]
        if len({other.source for other in local}) >= 2:
            total_weight = sum(other.weight for other in local)
            if total_weight:
                centers.append(
                    (
                        sum(other.x * other.weight for other in local) / total_weight,
                        sum(other.y * other.weight for other in local) / total_weight,
                        "local_centroid",
                    )
                )

    scored: list[dict[str, Any]] = []
    for cx, cy, center_source in centers:
        inside = [point for point in points if abs(point.x - cx) <= half and abs(point.y - cy) <= half]
        near = [point for point in points if math.hypot(point.x - cx, point.y - cy) <= 140]
        sources = {point.source for point in inside}
        source_bonus = len(sources) * 4.0
        agreement_bonus = 6.0 if len(sources & {"os", "gog", "fallback", "current", "ocr_grid", "range_parity"}) >= 2 else 0.0
        anchor_bonus = 5.0 if sources & {"road_zone", "road_midpoint", "range_parity"} and len(sources) >= 2 else 0.0
        score = (
            sum(point.weight for point in inside)
            + 0.35 * sum(point.weight for point in near)
            + source_bonus
            + agreement_bonus
            + anchor_bonus
        )
        scored.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "score": float(score),
                "inside_weight": float(sum(point.weight for point in inside)),
                "inside_count": len(inside),
                "near_count": len(near),
                "sources": "|".join(sorted(sources)),
                "center_source": center_source,
                "evidence_labels": " || ".join(
                    f"{point.source}:{point.label}@{point.x:.0f},{point.y:.0f} w={point.weight:.1f}"
                    for point in sorted(inside, key=lambda item: -item.weight)[:12]
                ),
            }
        )
    scored.sort(key=lambda item: (-item["score"], -item["inside_count"], item["cx"], item["cy"]))

    selected: list[dict[str, Any]] = []
    for item in scored:
        if any(math.hypot(item["cx"] - prev["cx"], item["cy"] - prev["cy"]) < min_center_distance for prev in selected):
            continue
        selected.append(item)
        if len(selected) >= max_rois:
            break
    return selected


def write_outputs(
    rows: list[dict[str, Any]],
    output_prefix: Path,
    road_geoms: dict[str, Any],
    target_geoms: dict[str, Any],
    roi_side: float,
    max_rois: int,
    min_center_distance: float,
) -> dict[str, Path]:
    raw_rows: list[dict[str, Any]] = []
    v10_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    roi_features: list[dict[str, Any]] = []

    for idx, source_row in enumerate(rows, start=1):
        raw = normalized_raw_row(source_row, idx)
        raw_rows.append(raw)
        v10 = v10_row(raw)
        audit = audit_row(raw)
        points = roi_v1.evidence_points_for_case(pd.Series(v10), raw, pd.Series(audit), road_geoms)
        v10_rows.append(v10)
        audit_rows.append(audit)
        summary = case_summary_row(raw, points)
        rois = rank_evidence_rois(points, roi_side, max_rois, min_center_distance)
        base = base_key(raw, raw["key"])
        target = target_geoms.get(base)
        roi_geoms = []
        for rank, item in enumerate(rois, start=1):
            geom = box(
                item["cx"] - roi_side / 2,
                item["cy"] - roi_side / 2,
                item["cx"] + roi_side / 2,
                item["cy"] + roi_side / 2,
            )
            roi_geoms.append(geom)
            roi_row = {
                "case_key": raw["key"],
                "base_key": base,
                "roi_rank": rank,
                "roi_reason": "evidence",
                "roi_sources": item["sources"],
                "roi_score": round(item["score"], 6),
                "roi_center_source": item["center_source"],
                "roi_inside_count": item["inside_count"],
                "roi_near_count": item["near_count"],
                "roi_minx": round(float(geom.bounds[0]), 3),
                "roi_miny": round(float(geom.bounds[1]), 3),
                "roi_maxx": round(float(geom.bounds[2]), 3),
                "roi_maxy": round(float(geom.bounds[3]), 3),
                "roi_center_easting": round(item["cx"], 3),
                "roi_center_northing": round(item["cy"], 3),
                "evidence_labels": item["evidence_labels"],
                **roi_truth_metrics(geom, target),
            }
            roi_rows.append(roi_row)
            feature = dict(roi_row)
            feature["geometry"] = geom
            roi_features.append(feature)
        if roi_geoms and target is not None:
            union = unary_union(roi_geoms)
            summary["union_intersects_target_polygon"] = bool(union.intersects(target))
            summary["union_contains_target_centroid"] = bool(union.contains(target.centroid))
        else:
            summary["union_intersects_target_polygon"] = ""
            summary["union_contains_target_centroid"] = ""
        summary_rows.append(summary)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_candidates_json": suffix(output_prefix, "_input_candidates.json"),
        "v10_csv": suffix(output_prefix, "_v10.csv"),
        "audit_csv": suffix(output_prefix, "_audit.csv"),
        "case_summary_csv": suffix(output_prefix, "_case_summary.csv"),
        "candidate_boxes_csv": suffix(output_prefix, "_candidate_boxes.csv"),
        "candidate_boxes_gpkg": suffix(output_prefix, "_candidate_boxes.gpkg"),
        "manifest_json": suffix(output_prefix, "_manifest.json"),
    }
    paths["input_candidates_json"].write_text(
        json.dumps({"rows": raw_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(v10_rows).to_csv(paths["v10_csv"], index=False)
    pd.DataFrame(audit_rows).to_csv(paths["audit_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["case_summary_csv"], index=False)
    pd.DataFrame(roi_rows).to_csv(paths["candidate_boxes_csv"], index=False)
    if roi_features:
        gpd.GeoDataFrame(roi_features, geometry="geometry", crs="EPSG:27700").to_file(
            paths["candidate_boxes_gpkg"],
            layer="candidate_boxes",
            driver="GPKG",
        )

    manifest = {
        "pipeline_stage": "address_point_evidence_adapter",
        "rows": len(rows),
        "cases_ok": sum(1 for row in summary_rows if row.get("status") == "ok"),
        "candidate_boxes": len(roi_rows),
        "roi_side_m": roi_side,
        "max_rois": max_rois,
        "outputs": {key: str(value) for key, value in paths.items()},
        "upstream_contract": "address-point GPKG is the primary case layer; 1_address JSON/JSONL is optional supplemental evidence",
    }
    paths["manifest_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt 1_address_to_point_gemini output into geocoding-box boxes.")
    parser.add_argument("--address-point-json", type=Path, help="Optional JSON or JSONL from 1_address_to_point_gemini.py.")
    parser.add_argument("--address-point-gpkg", type=Path, help="Primary 1_address/input GPKG. unique_key and case geometry come from here.")
    parser.add_argument("--address-point-layer", help="Layer in --address-point-gpkg.")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--open-roads", type=Path, required=True)
    parser.add_argument("--case-gpkg", type=Path, help="Optional evaluation GPKG for offline box-hit metrics only.")
    parser.add_argument("--case-layer", help="Layer in --case-gpkg.")
    parser.add_argument("--key-column", default="unique_key")
    parser.add_argument("--address-column", default="chargegeog")
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--generic-place-names", default="")
    parser.add_argument("--roi-side", type=float, default=180.0)
    parser.add_argument("--max-rois", type=int, default=8)
    parser.add_argument("--min-center-distance", type=float, default=30.0)
    parser.add_argument("--disable-range-parity-correction", action="store_true")
    args = parser.parse_args()

    roi_v1.MANSFIELD_BBOX = parse_bbox(args.local_bbox, roi_v1.MANSFIELD_BBOX)
    roi_v1.TOKEN_STOP = set(roi_v1.TOKEN_STOP) | set(parse_names(args.generic_place_names, []))

    address_rows = load_address_point_rows(args.address_point_json) if args.address_point_json else []
    if args.address_point_gpkg:
        if not args.address_point_layer:
            parser.error("--address-point-layer is required with --address-point-gpkg")
        case_rows, address_geoms = load_address_point_gpkg_rows(
            args.address_point_gpkg,
            args.address_point_layer,
            args.key_column,
            args.address_column,
        )
        rows = merge_case_rows_with_address_evidence(
            case_rows,
            address_rows,
            enable_range_parity_correction=not args.disable_range_parity_correction,
        )
        default_geoms = address_geoms
    elif address_rows:
        rows = address_rows
        default_geoms = {}
    else:
        parser.error("Provide --address-point-gpkg, --address-point-json, or both")

    road_geoms = roi_v1.load_road_geoms(args.open_roads)
    target_geoms = load_case_geometries(args.case_gpkg, args.case_layer, args.key_column) or default_geoms
    paths = write_outputs(
        rows=rows,
        output_prefix=args.output_prefix,
        road_geoms=road_geoms,
        target_geoms=target_geoms,
        roi_side=args.roi_side,
        max_rois=args.max_rois,
        min_center_distance=args.min_center_distance,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
