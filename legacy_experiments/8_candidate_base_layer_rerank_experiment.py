#!/usr/bin/env python3
"""Rerank production-visible base-map candidates for Mansfield cases.

Important boundary:
- Candidate polygons come from production-visible base layers only:
  `/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg`.
- OS Open UPRN points are used as supporting point evidence only.
- The manual Mansfield polygon-link layer is used only as offline truth for
  evaluation; it is never used to generate candidates.
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
DEFAULT_TRUTH_GPKG = TMP / f"{DEFAULT_TAG}.gpkg"
DEFAULT_TRUTH_LAYER = "random1200_seed42_43_combined"
DEFAULT_WFS_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg")
DEFAULT_WFS_LAYER = "mansfield_polygons_in_buffers"
DEFAULT_UPRN_GPKG = Path("/data/base-data/osopenuprn_202602.gpkg")
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OUTPUT_PREFIX = TMP / f"{DEFAULT_TAG}_base_layer_rerank_v2_theme_land_building"
MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)
POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
ROAD_TOKEN_NORMALIZATION = {
    "RD": "ROAD",
    "ST": "STREET",
    "LN": "LANE",
    "AVE": "AVENUE",
    "AV": "AVENUE",
    "DR": "DRIVE",
    "CT": "COURT",
    "PL": "PLACE",
}


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


def load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = clean_text(
            row.get("key")
            or row.get("variant_key")
            or row.get("unique_key")
            or row.get("_source_unique_key")
        )
        if key:
            out[key] = row
    return out


def load_truth(path: Path, layer: str) -> dict[str, Any]:
    gdf = gpd.read_file(path, layer=layer, columns=["unique_key"])
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return {str(row["unique_key"]): row.geometry for _, row in gdf.iterrows()}


def load_wfs_polygons(path: Path, layer: str, theme_regex: str | None) -> gpd.GeoDataFrame:
    columns = [
        "GmlID",
        "OBJECTID",
        "TOID",
        "Theme",
        "DescriptiveGroup",
        "DescriptiveTerm",
        "Make",
        "Shape_Area",
    ]
    gdf = gpd.read_file(path, layer=layer, bbox=MANSFIELD_BBOX, columns=columns)
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if theme_regex:
        gdf = gdf[gdf["Theme"].fillna("").str.contains(theme_regex, case=False, regex=True)].copy()
    gdf["candidate_id"] = gdf["TOID"].fillna("").astype(str)
    missing = gdf["candidate_id"].eq("") | gdf["candidate_id"].eq("nan")
    gdf.loc[missing, "candidate_id"] = gdf.loc[missing, "GmlID"].fillna(gdf.loc[missing, "OBJECTID"]).astype(str)
    gdf["candidate_area_m2"] = gdf.geometry.area
    centroids = gdf.geometry.centroid
    gdf["candidate_centroid_easting"] = centroids.x
    gdf["candidate_centroid_northing"] = centroids.y
    return gdf.reset_index(drop=True)


def load_uprn_points(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, bbox=MANSFIELD_BBOX, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["UPRN"] = gdf["UPRN"].astype(str)
    return gdf.reset_index(drop=True)


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
        found = roi_v1.extract_address_roads(source, road_names) if source_type == "address" else roi_v1.split_anchors(source)
        for road in found:
            if road in road_names and road not in roads:
                roads.append(road)
    return roads


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" ,.")


def norm_road_key(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", clean_text(value).upper()).strip()
    tokens = [ROAD_TOKEN_NORMALIZATION.get(token, token) for token in text.split()]
    return "".join(tokens)


def norm_compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", clean_text(value).upper())


ROAD_SUFFIX_RE = re.compile(
    r"\b(ROAD|STREET|LANE|AVENUE|CLOSE|WAY|DRIVE|GROVE|CRESCENT|HILL|GATE|WALK|PLACE|COURT|PARK|TERRACE|ROW|YARD|MEWS|RISE|VIEW|SQUARE)\b",
    re.I,
)
HOUSE_RANGE_RE = re.compile(r"\b(\d{1,4})[A-Z]?\s*(?:-|–|—|TO)\s*(\d{1,4})[A-Z]?\b", re.I)
INTERNAL_RANGE_PREFIX_RE = re.compile(r"\b(UNIT|UNITS|PLOT|PLOTS|BLOCK|FLAT|FLATS|APARTMENT|APARTMENTS|PHASE|PARCEL)\b", re.I)


def same_parity_road_range_context(raw_address: Any) -> dict[str, Any] | None:
    text = POSTCODE_RE.sub(" ", clean_text(raw_address).upper())
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
        numbers = set(range(low, high + 1, 2))
        return {
            "kind": "range",
            "numbers": numbers,
            "endpoints": {low, high},
            "range_texts": {f"{low}-{high}", f"{low}TO{high}", f"{low}AND{high}"},
            "parity_corrected": True,
        }
    return None


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def numbers_from_text(value: Any, *, expand_ranges: bool = True) -> set[int]:
    text = POSTCODE_RE.sub(" ", clean_text(value).upper())
    out: set[int] = set()
    if expand_ranges:
        for start_s, end_s in re.findall(r"\b(\d{1,4})\s*(?:-|TO|AND)\s*(\d{1,4})\b", text):
            start = int(start_s)
            end = int(end_s)
            if not (0 < start < 1000 and 0 < end < 1000):
                continue
            low, high = sorted((start, end))
            if high - low <= 20:
                out.update(range(low, high + 1))
    for number_s in re.findall(r"\b\d{1,4}[A-Z]?\b", text):
        number = int(re.match(r"\d+", number_s).group(0))
        if 0 < number < 1000:
            out.add(number)
    return out


def requested_number_context(raw_address: Any) -> dict[str, Any]:
    text = POSTCODE_RE.sub(" ", clean_text(raw_address).upper())
    range_match = re.search(r"\b(\d{1,4})\s*(?:-|TO)\s*(\d{1,4})\b", text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        low, high = sorted((start, end))
        if 0 < low < 1000 and 0 < high < 1000 and high - low <= 20:
            return {
                "kind": "range",
                "numbers": set(range(low, high + 1)),
                "endpoints": {low, high},
                "range_texts": {f"{low}-{high}", f"{low}TO{high}", f"{low}AND{high}"},
                "parity_corrected": False,
            }
    numbers = numbers_from_text(text, expand_ranges=False)
    if len(numbers) >= 2:
        return {"kind": "list", "numbers": numbers, "endpoints": set(numbers), "range_texts": set(), "parity_corrected": False}
    return {"kind": "none", "numbers": numbers, "endpoints": set(numbers), "range_texts": set(), "parity_corrected": False}


def range_relation(raw_address: Any) -> str:
    lowered = clean_text(raw_address).lower()
    if "between" in lowered:
        return "between"
    if "rear" in lowered or "behind" in lowered:
        return "rear"
    if "adjacent" in lowered or "adjoining" in lowered:
        return "adjacent"
    return "numbered"


def xy_from_candidate(candidate: dict[str, Any]) -> tuple[float | None, float | None]:
    x = parse_float(candidate.get("easting_27700") or candidate.get("X_COORDINATE"))
    y = parse_float(candidate.get("northing_27700") or candidate.get("Y_COORDINATE"))
    return x, y


def candidate_address(candidate: dict[str, Any]) -> str:
    return clean_text(candidate.get("address") or candidate.get("ADDRESS"))


def candidate_road(candidate: dict[str, Any]) -> str:
    return clean_text(candidate.get("road_name") or candidate.get("ROAD_NAME"))


def candidate_place(candidate: dict[str, Any]) -> str:
    return clean_text(candidate.get("place") or candidate.get("PLACE"))


def candidate_matches_road(candidate: dict[str, Any], address: str, road_key: str) -> bool:
    if not road_key:
        return True
    cand_road_key = norm_road_key(candidate_road(candidate))
    addr_road_key = norm_road_key(address)
    return cand_road_key == road_key or road_key in addr_road_key


def candidate_matches_place(candidate: dict[str, Any], address: str, place_key: str) -> bool:
    if not place_key:
        return "MANSFIELD" in clean_text(address).upper() or "MANSFIELD" in clean_text(candidate_place(candidate)).upper()
    return place_key in norm_compact(address) or place_key in norm_compact(candidate_place(candidate))


def range_text_exact_hit(address: str, range_texts: set[str]) -> bool:
    compact = norm_compact(address)
    return any(text and text in compact for text in range_texts)


def add_range_candidate(
    out: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    source: str,
    origin_key: str,
    origin_role: str,
    context: dict[str, Any],
    road_key: str,
    place_key: str,
    base_key: str,
) -> None:
    address = candidate_address(candidate)
    x, y = xy_from_candidate(candidate)
    if not address or x is None or y is None:
        return
    if not candidate_matches_road(candidate, address, road_key):
        return
    if not candidate_matches_place(candidate, address, place_key):
        return

    requested = set(context["numbers"])
    endpoints = set(context["endpoints"])
    cand_numbers = numbers_from_text(address, expand_ranges=True)
    if not cand_numbers or not (cand_numbers & requested):
        return

    exact = cand_numbers == requested or range_text_exact_hit(address, context.get("range_texts") or set())
    subset = bool(cand_numbers.issubset(requested))
    endpoint_hits = cand_numbers & endpoints
    overlap = cand_numbers & requested
    extras = cand_numbers - requested

    score = 0.0
    if exact:
        score += 170.0
    if subset:
        score += 42.0
    score += 12.0 * len(overlap)
    score += 24.0 * len(endpoint_hits)
    score -= 22.0 * len(extras)
    if source == "os":
        score += 8.0
    elif source == "gog":
        score += 5.0
    elif source == "range_parity":
        score += 22.0
    if origin_role == "parent":
        score += 18.0
    elif origin_role == "child_best":
        score += 10.0
    elif origin_role == "range_parity":
        score += 24.0
    if str(origin_key) == str(base_key):
        score += 5.0

    out.append(
        {
            "address": address,
            "x": float(x),
            "y": float(y),
            "source": source,
            "origin_key": origin_key,
            "origin_role": origin_role,
            "numbers": sorted(cand_numbers),
            "overlap_numbers": sorted(overlap),
            "endpoint_hits": sorted(endpoint_hits),
            "exact": exact,
            "subset": subset,
            "score": score,
        }
    )


def collect_range_candidates(parent_row: dict[str, Any], family_rows: list[dict[str, Any]], base_key: str) -> list[dict[str, Any]]:
    raw_address = clean_text(parent_row.get("original_address"))
    context = requested_number_context(raw_address)
    if len(context["numbers"]) < 2:
        return []
    lexicon_parts = [clean_text(part) for part in clean_text(parent_row.get("lexicon_road")).split("|")]
    road_key = norm_road_key(lexicon_parts[0] if lexicon_parts else "")
    place_key = norm_compact(lexicon_parts[1] if len(lexicon_parts) > 1 else "Mansfield")
    out: list[dict[str, Any]] = []

    rows_to_scan = [parent_row] + [row for row in family_rows if row is not parent_row]
    for row in rows_to_scan:
        row_key = clean_text(row.get("key"))
        role = "parent" if row_key == str(base_key) else "child"
        for source, field in [
            ("os", "os_candidates_json"),
            ("gog", "gog_candidates_json"),
            ("fallback", "os_fallback_pool_json"),
        ]:
            for candidate in parse_json_list(row.get(field)):
                add_range_candidate(
                    out,
                    candidate=candidate,
                    source=source,
                    origin_key=row_key,
                    origin_role=role,
                    context=context,
                    road_key=road_key,
                    place_key=place_key,
                    base_key=base_key,
                )

        for item in parse_json_list(row.get("range_parity_points_json")):
            range_candidate = {
                "address": item.get("address") or item.get("range_text"),
                "road_name": lexicon_parts[0] if lexicon_parts else "",
                "place": lexicon_parts[1] if len(lexicon_parts) > 1 else "Mansfield",
                "easting_27700": item.get("easting_27700"),
                "northing_27700": item.get("northing_27700"),
            }
            add_range_candidate(
                out,
                candidate=range_candidate,
                source="range_parity",
                origin_key=row_key,
                origin_role="range_parity",
                context=context,
                road_key=road_key,
                place_key=place_key,
                base_key=base_key,
            )

        best_candidate = {
            "address": row.get("best_address_final"),
            "road_name": lexicon_parts[0] if lexicon_parts else "",
            "place": lexicon_parts[1] if len(lexicon_parts) > 1 else "Mansfield",
            "easting_27700": row.get("best_easting_27700_final"),
            "northing_27700": row.get("best_northing_27700_final"),
        }
        add_range_candidate(
            out,
            candidate=best_candidate,
            source=clean_text(row.get("best_source_final")) or "best",
            origin_key=row_key,
            origin_role="parent_best" if role == "parent" else "child_best",
            context=context,
            road_key=road_key,
            place_key=place_key,
            base_key=base_key,
        )
    return out


def cluster_range_candidates(candidates: list[dict[str, Any]], threshold_m: float = 95.0) -> list[list[dict[str, Any]]]:
    if not candidates:
        return []
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            dist = math.hypot(candidates[left]["x"] - candidates[right]["x"], candidates[left]["y"] - candidates[right]["y"])
            if dist <= threshold_m:
                union(left, right)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for idx, candidate in enumerate(candidates):
        grouped.setdefault(find(idx), []).append(candidate)
    return list(grouped.values())


def range_cluster_radius(cluster: list[dict[str, Any]]) -> float:
    if len(cluster) <= 1:
        return 0.0
    max_distance = 0.0
    for left in range(len(cluster)):
        for right in range(left + 1, len(cluster)):
            max_distance = max(
                max_distance,
                math.hypot(cluster[left]["x"] - cluster[right]["x"], cluster[left]["y"] - cluster[right]["y"]),
            )
    return max_distance


def build_range_anchor(parent_row: dict[str, Any], family_rows: list[dict[str, Any]], base_key: str) -> dict[str, Any] | None:
    raw_address = clean_text(parent_row.get("original_address"))
    context = requested_number_context(raw_address)
    if len(context["numbers"]) < 2:
        return None
    candidates = collect_range_candidates(parent_row, family_rows, base_key)
    if not candidates:
        return None

    requested = set(context["numbers"])
    endpoints = set(context["endpoints"])
    clusters = cluster_range_candidates(candidates)
    if not clusters:
        return None

    def cluster_sort_key(cluster: list[dict[str, Any]]) -> tuple[float, float, float, float, float]:
        covered = set()
        endpoint_hits = set()
        exact_count = 0
        subset_count = 0
        total_score = 0.0
        for candidate in cluster:
            covered.update(set(candidate["overlap_numbers"]))
            endpoint_hits.update(set(candidate["endpoint_hits"]))
            exact_count += int(bool(candidate["exact"]))
            subset_count += int(bool(candidate["subset"]))
            total_score += float(candidate["score"])
        return (
            float(exact_count > 0),
            float(len(endpoint_hits)),
            float(len(covered)),
            float(subset_count),
            total_score - range_cluster_radius(cluster) * 0.4,
        )

    clusters.sort(key=cluster_sort_key, reverse=True)
    cluster = clusters[0]
    useful = [
        candidate
        for candidate in cluster
        if candidate["exact"] or candidate["subset"] or set(candidate["endpoint_hits"]) or float(candidate["score"]) >= 40
    ]
    if not useful:
        useful = cluster

    weights = [max(1.0, float(candidate["score"])) for candidate in useful]
    total_weight = sum(weights)
    x = sum(candidate["x"] * weight for candidate, weight in zip(useful, weights)) / total_weight
    y = sum(candidate["y"] * weight for candidate, weight in zip(useful, weights)) / total_weight
    covered_numbers = sorted({number for candidate in useful for number in candidate["overlap_numbers"]})
    endpoint_hits = sorted({number for candidate in useful for number in candidate["endpoint_hits"]})
    parity_corrected = bool(context.get("parity_corrected")) or any(candidate.get("source") == "range_parity" for candidate in useful)
    if context["kind"] == "range":
        complete = len(endpoint_hits) >= len(context["endpoints"])
    else:
        complete = len(covered_numbers) >= len(context["numbers"])
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "relation": range_relation(raw_address),
        "number_kind": context["kind"],
        "parity_corrected": parity_corrected,
        "requested_numbers": ",".join(str(number) for number in sorted(requested)),
        "covered_numbers": ",".join(str(number) for number in covered_numbers),
        "endpoint_hits": ",".join(str(number) for number in endpoint_hits),
        "candidate_count": len(candidates),
        "cluster_count": len(useful),
        "cluster_radius_m": round(range_cluster_radius(useful), 1),
        "cluster_score": round(sum(float(candidate["score"]) for candidate in useful), 1),
        "complete": bool(complete),
        "top_addresses": " | ".join(candidate["address"] for candidate in sorted(useful, key=lambda item: item["score"], reverse=True)[:5]),
    }


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


def uprn_features(geom: Any, uprn: gpd.GeoDataFrame, uprn_sindex: Any, best_point: Point | None, v7_point: Point | None) -> dict[str, Any]:
    idx = list(uprn_sindex.query(geom, predicate="intersects"))
    points = uprn.iloc[idx] if idx else uprn.iloc[[]]
    nearest_best = None
    nearest_v7 = None
    if len(points):
        if best_point is not None:
            nearest_best = float(points.geometry.distance(best_point).min())
        if v7_point is not None:
            nearest_v7 = float(points.geometry.distance(v7_point).min())
    return {
        "uprn_count": int(len(points)),
        "nearest_uprn_to_best_point_m": nearest_best,
        "nearest_uprn_to_v7_point_m": nearest_v7,
    }


def base_type_score(theme: Any, group: Any, term: Any, area: float) -> tuple[float, str]:
    theme_s = str(theme or "").upper()
    group_s = str(group or "").upper()
    term_s = str(term or "").upper()
    score = 0.0
    reasons: list[str] = []
    if "BUILDING" in theme_s or "BUILDING" in group_s:
        score += 28.0
        reasons.append("type_building:+28")
    elif "GENERAL SURFACE" in group_s:
        score += 9.0
        reasons.append("type_general_surface:+9")
    elif "STRUCTURE" in group_s:
        score += 8.0
        reasons.append("type_structure:+8")

    if "ROADS TRACKS" in theme_s or "ROAD" in group_s or "ROADSIDE" in group_s:
        score -= 28.0
        reasons.append("type_road_or_roadside:-28")
    if "WATER" in theme_s or "WATER" in group_s:
        score -= 18.0
        reasons.append("type_water:-18")
    if "AGRICULTURAL" in term_s:
        score -= 6.0
        reasons.append("type_agricultural:-6")

    if 15 <= area <= 2500:
        score += 8.0
        reasons.append("area_reasonable:+8")
    elif 2500 < area <= 12000:
        score += 2.0
        reasons.append("area_medium:+2")
    elif area > 40000:
        score -= 14.0
        reasons.append("area_huge:-14")
    elif area > 15000:
        score -= 5.0
        reasons.append("area_large:-5")
    return score, ";".join(reasons)


def score_candidate(row: dict[str, Any]) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []

    type_score, type_reasons = base_type_score(
        row["candidate_theme"],
        row["candidate_descriptive_group"],
        row["candidate_descriptive_term"],
        row["candidate_area_m2"],
    )
    score += type_score
    if type_reasons:
        reasons.append(type_reasons)

    if row["covers_v7_point"]:
        score += 80.0
        reasons.append("covers_v7_point:+80")
    else:
        d = row["distance_to_v7_point_m"]
        if d is not None:
            delta = max(-24.0, -0.05 * min(d, 480.0))
            score += delta
            reasons.append(f"distance_to_v7:{delta:.1f}")
    if row["covers_best_point"]:
        score += 58.0
        reasons.append("covers_best_point:+58")
    else:
        d = row["distance_to_best_point_m"]
        if d is not None:
            delta = max(-18.0, -0.035 * min(d, 520.0))
            score += delta
            reasons.append(f"distance_to_best:{delta:.1f}")

    uprn_count = int(row["uprn_count"])
    if uprn_count:
        gain = min(24.0, 8.0 + 4.0 * uprn_count)
        score += gain
        reasons.append(f"uprn_inside:+{gain:.0f}")
    nearest_v7 = row["nearest_uprn_to_v7_point_m"]
    nearest_best = row["nearest_uprn_to_best_point_m"]
    nearest = min([d for d in [nearest_v7, nearest_best] if d is not None], default=None)
    if nearest is not None:
        if nearest <= 5:
            score += 24.0
            reasons.append("uprn_near_point_5m:+24")
        elif nearest <= 20:
            score += 14.0
            reasons.append("uprn_near_point_20m:+14")
        elif nearest <= 50:
            score += 6.0
            reasons.append("uprn_near_point_50m:+6")

    if row["intersects_top1_roi"]:
        score += 18.0
        reasons.append("top1_roi:+18")
    if row["intersects_protected_current_roi"]:
        score += 8.0
        reasons.append("protected_current_roi:+8")
    if row["intersects_evidence_roi"]:
        score += 9.0
        reasons.append("evidence_roi:+9")
    if row["intersects_corridor_roi"]:
        score += 8.0
        reasons.append("corridor_roi:+8")
    score += min(25.0, row["roi_total_intersection_area_m2"] / 90.0)
    score += min(12.0, row["roi_rank_weighted_score"] / 4.0)
    score += max(-10.0, -0.015 * min(row["distance_to_nearest_roi_center_m"], 660.0))

    road_distance = row["distance_to_mentioned_road_m"]
    if road_distance is not None:
        if road_distance <= 20:
            score += 10.0
            reasons.append("near_mentioned_road_20m:+10")
        elif road_distance <= 50:
            score += 7.0
            reasons.append("near_mentioned_road_50m:+7")
        elif road_distance <= 100:
            score += 3.0
            reasons.append("near_mentioned_road_100m:+3")

    if row.get("has_range_anchor"):
        relation = str(row.get("range_anchor_relation") or "numbered")
        complete_anchor = bool(row.get("range_anchor_complete"))
        anchor_distance = row.get("distance_to_range_anchor_m")
        covers_anchor = bool(row.get("covers_range_anchor"))
        group = str(row.get("candidate_descriptive_group") or "").upper()
        theme = str(row.get("candidate_theme") or "").upper()
        is_building = "BUILDING" in theme or "BUILDING" in group
        is_surface = "GENERAL SURFACE" in group or "LAND" in theme

        if relation in {"rear", "adjacent"}:
            if anchor_distance is not None:
                if anchor_distance <= 20:
                    score += 36.0
                    reasons.append("range_anchor_rear_near_20m:+36")
                elif anchor_distance <= 50:
                    score += 28.0
                    reasons.append("range_anchor_rear_near_50m:+28")
                elif anchor_distance <= 90:
                    score += 12.0
                    reasons.append("range_anchor_rear_near_90m:+12")
                else:
                    delta = max(-24.0, -0.05 * min(anchor_distance, 480.0))
                    score += delta
                    reasons.append(f"range_anchor_rear_distance:{delta:.1f}")
            if is_surface:
                score += 10.0
                reasons.append("range_anchor_rear_surface:+10")
            if covers_anchor and is_building:
                score += 8.0
                reasons.append("range_anchor_rear_building_anchor:+8")
        elif relation == "between":
            if covers_anchor:
                score += 52.0
                reasons.append("range_anchor_between_covers:+52")
            elif anchor_distance is not None:
                if anchor_distance <= 20:
                    score += 42.0
                    reasons.append("range_anchor_between_near_20m:+42")
                elif anchor_distance <= 50:
                    score += 26.0
                    reasons.append("range_anchor_between_near_50m:+26")
                elif anchor_distance <= 90:
                    score += 10.0
                    reasons.append("range_anchor_between_near_90m:+10")
                else:
                    delta = max(-26.0, -0.06 * min(anchor_distance, 500.0))
                    score += delta
                    reasons.append(f"range_anchor_between_distance:{delta:.1f}")
            if is_surface:
                score += 14.0
                reasons.append("range_anchor_between_surface:+14")
            elif is_building:
                score -= 4.0
                reasons.append("range_anchor_between_building:-4")
        else:
            if covers_anchor:
                score += 76.0
                reasons.append("range_anchor_covers:+76")
                if complete_anchor:
                    score += 80.0
                    reasons.append("range_anchor_complete_covers:+80")
            elif anchor_distance is not None:
                if anchor_distance <= 10:
                    score += 62.0
                    reasons.append("range_anchor_near_10m:+62")
                elif anchor_distance <= 30:
                    score += 42.0
                    reasons.append("range_anchor_near_30m:+42")
                elif anchor_distance <= 60:
                    score += 18.0
                    reasons.append("range_anchor_near_60m:+18")
                else:
                    delta = max(-30.0, -0.075 * min(anchor_distance, 520.0))
                    score += delta
                    reasons.append(f"range_anchor_distance:{delta:.1f}")
            if complete_anchor and (row.get("covers_best_point") or row.get("covers_v7_point")) and not covers_anchor:
                if anchor_distance is not None and anchor_distance > 15:
                    score -= 80.0
                    reasons.append("range_anchor_complete_conflicts_old_point:-80")

        if relation == "between" and complete_anchor:
            if covers_anchor:
                score += 100.0
                reasons.append("range_anchor_complete_between_covers:+100")
            elif row.get("covers_best_point") or row.get("covers_v7_point"):
                score -= 80.0
                reasons.append("range_anchor_complete_between_conflicts_old_point:-80")

    return score, ";".join(reasons)


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
        "mean_candidate_count": float(pd.to_numeric(ok["candidate_polygon_count"], errors="coerce").mean()),
        "median_candidate_count": float(pd.to_numeric(ok["candidate_polygon_count"], errors="coerce").median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=DEFAULT_V10_CSV)
    parser.add_argument("--case-summary-csv", type=Path, default=DEFAULT_CASE_SUMMARY_CSV)
    parser.add_argument("--rois-csv", type=Path, default=DEFAULT_ROIS_CSV)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKG)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--wfs-gpkg", type=Path, default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--uprn-gpkg", type=Path, default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--theme-filter-regex", default="Land|Building")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--disable-range-anchor", action="store_true")
    args = parser.parse_args()

    print("loading tabular inputs...")
    raw_rows = load_raw_rows(args.input_json)
    raw_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows.values():
        source_key = clean_text(row.get("_source_unique_key") or row.get("key"))
        if source_key:
            raw_rows_by_source.setdefault(source_key, []).append(row)
    range_anchor_cache: dict[str, dict[str, Any] | None] = {}
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    v10_by_key = {str(row["key"]): row for _, row in v10.iterrows()}
    case_summary = pd.read_csv(args.case_summary_csv, dtype={"key": str, "base_key": str})
    rois = pd.read_csv(args.rois_csv, dtype={"case_key": str, "base_key": str})
    rois["roi_rank"] = pd.to_numeric(rois["roi_rank"], errors="coerce")
    rois["roi_score"] = pd.to_numeric(rois["roi_score"], errors="coerce").fillna(0.0)

    print("loading production-visible base layers...")
    wfs = load_wfs_polygons(args.wfs_gpkg, args.wfs_layer, args.theme_filter_regex)
    uprn = load_uprn_points(args.uprn_gpkg, args.uprn_layer)
    road_geoms = roi_v1.load_road_geoms(args.open_roads)
    road_names = set(road_geoms)
    truth = load_truth(args.truth_gpkg, args.truth_layer)
    wfs_sindex = wfs.sindex
    uprn_sindex = uprn.sindex
    print(f"loaded wfs_polygons={len(wfs)} uprn_points={len(uprn)} truth_polygons={len(truth)}")

    candidate_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
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
        candidate_idx = list(wfs_sindex.query(union_roi, predicate="intersects"))
        candidates = wfs.iloc[candidate_idx].copy()
        if candidates.empty:
            case_rows.append({"key": key, "base_key": base_key, "status": "no_candidates"})
            continue

        raw = raw_rows.get(key) or raw_rows.get(base_key) or {}
        v10_row = v10_by_key.get(key)
        best_point = point_from_xy(raw.get("best_easting_27700_final"), raw.get("best_northing_27700_final"))
        if best_point is None and v10_row is not None:
            best_point = point_from_xy(v10_row.get("baseline_easting"), v10_row.get("baseline_northing"))
        v7_point = point_from_xy(v10_row.get("v7_selected_easting"), v10_row.get("v7_selected_northing")) if v10_row is not None else None
        roads = build_case_roads(case, v10_row, road_names)
        parent_raw = raw_rows.get(base_key) or raw
        range_anchor = None
        range_anchor_point = None
        if not args.disable_range_anchor and parent_raw:
            if base_key not in range_anchor_cache:
                range_anchor_cache[base_key] = build_range_anchor(parent_raw, raw_rows_by_source.get(base_key, []), base_key)
            range_anchor = range_anchor_cache.get(base_key)
            if range_anchor:
                range_anchor_point = Point(float(range_anchor["x"]), float(range_anchor["y"]))

        per_case_rows: list[dict[str, Any]] = []
        for _, cand in candidates.iterrows():
            geom = cand.geometry
            centroid = geom.centroid
            roi_hits = [item for item in roi_items if geom.intersects(item["geom"])]
            if not roi_hits:
                continue
            intersection_areas = [float(geom.intersection(item["geom"]).area) for item in roi_hits]
            center_distances = [float(centroid.distance(item["center"])) for item in roi_hits]
            road_distance, road_near_50, road_near_100 = road_features(geom, roads, road_geoms)
            uprn_data = uprn_features(geom, uprn, uprn_sindex, best_point, v7_point)
            truth_intersects = bool(target_geom is not None and geom.intersects(target_geom))
            truth_centroid_inside = bool(target_geom is not None and target_geom.contains(centroid))
            truth_overlap_area = float(geom.intersection(target_geom).area) if target_geom is not None and truth_intersects else 0.0
            truth_candidate_overlap_ratio = truth_overlap_area / geom.area if geom.area else 0.0

            row = {
                "case_key": key,
                "base_key": base_key,
                "sample_split": case.get("sample_split"),
                "original_address": case.get("original_address"),
                "best_confidence": parse_float(raw.get("best_confidence")) or parse_float(case.get("best_confidence")),
                "best_source_final": raw.get("best_source_final"),
                "best_selection_category": raw.get("best_selection_category"),
                "candidate_id": cand.get("candidate_id"),
                "candidate_toid": cand.get("TOID"),
                "candidate_gmlid": cand.get("GmlID"),
                "candidate_objectid": cand.get("OBJECTID"),
                "candidate_theme": cand.get("Theme"),
                "candidate_descriptive_group": cand.get("DescriptiveGroup"),
                "candidate_descriptive_term": cand.get("DescriptiveTerm"),
                "candidate_make": cand.get("Make"),
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
                "truth_intersects": truth_intersects,
                "truth_centroid_inside": truth_centroid_inside,
                "truth_overlap_area_m2": truth_overlap_area,
                "truth_candidate_overlap_ratio": truth_candidate_overlap_ratio,
                **uprn_data,
            }
            row["rule_score"], row["rule_score_reasons"] = score_candidate(row)
            per_case_rows.append(row)

        if not per_case_rows:
            case_rows.append({"key": key, "base_key": base_key, "status": "no_intersecting_candidates"})
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
        for idx, row in enumerate(per_case_rows, start=1):
            row["rule_rank"] = idx
            if intersects_rank is None and row["truth_intersects"]:
                intersects_rank = idx
            if centroid_rank is None and row["truth_centroid_inside"]:
                centroid_rank = idx
            if idx <= 20:
                top_rows.append(row.copy())
            candidate_rows.append(row)

        top1 = per_case_rows[0]
        top2_score = per_case_rows[1]["rule_score"] if len(per_case_rows) > 1 else None
        case_rows.append(
            {
                "key": key,
                "base_key": base_key,
                "status": "ok",
                "sample_split": case.get("sample_split"),
                "original_address": case.get("original_address"),
                "best_confidence": parse_float(raw.get("best_confidence")) or parse_float(case.get("best_confidence")),
                "roi_union_intersects_truth": safe_bool(case.get("union_intersects_target_polygon")),
                "candidate_polygon_count": len(per_case_rows),
                "intersects_present": intersects_rank is not None,
                "intersects_rank": intersects_rank,
                "centroid_present": centroid_rank is not None,
                "centroid_rank": centroid_rank,
                "top1_candidate_id": top1["candidate_id"],
                "top1_theme": top1["candidate_theme"],
                "top1_group": top1["candidate_descriptive_group"],
                "top1_intersects_truth": bool(top1["truth_intersects"]),
                "top1_centroid_inside_truth": bool(top1["truth_centroid_inside"]),
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

    candidate_df = pd.DataFrame(candidate_rows)
    top_df = pd.DataFrame(top_rows)
    case_df = pd.DataFrame(case_rows)

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

    ok = case_df[case_df["status"] == "ok"].copy()
    low_conf = ok[pd.to_numeric(ok.get("best_confidence"), errors="coerce") < 75].copy()
    high_conf = ok[pd.to_numeric(ok.get("best_confidence"), errors="coerce") >= 80].copy()
    summary = {
        "candidate_source": str(args.wfs_gpkg),
        "candidate_layer": args.wfs_layer,
        "candidate_theme_filter_regex": args.theme_filter_regex,
        "uprn_source": str(args.uprn_gpkg),
        "truth_source_for_evaluation_only": str(args.truth_gpkg),
        "wfs_polygon_count_loaded": int(len(wfs)),
        "uprn_point_count_loaded": int(len(uprn)),
        "all_cases_intersects_metric": summarize(case_df, "intersects"),
        "all_cases_centroid_metric": summarize(case_df, "centroid"),
        "low_confidence_lt75_intersects_metric": summarize(low_conf, "intersects"),
        "high_confidence_ge80_intersects_metric": summarize(high_conf, "intersects"),
        "output_candidates_csv": str(candidates_csv),
        "output_top20_csv": str(top20_csv),
        "output_cases_csv": str(cases_csv),
        "output_xlsx": str(xlsx_path),
        "candidate_rows": int(len(candidate_df)),
        "range_anchor_cases": int(ok["has_range_anchor"].fillna(False).sum()) if "has_range_anchor" in ok else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
