#!/usr/bin/env python3
"""Cluster heterogeneous candidate evidence into a small ROI, then retrieve polygons.

Evidence sources:
- current/v10 selected coordinate
- original Gemini OS/GOG/fallback/web candidates
- OCR geo-code candidates decoded from the OCR grid field
- plan/address road-anchor pair zones

The experiment chooses a 150m x 150m evidence ROI per case and evaluates
whether the manual polygon centroid/intersection falls inside that ROI, plus
how many full-layer polygons remain as candidates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox, parse_names


ROOT = Path(__file__).resolve().parent
OCR_SCRIPT = ROOT / "ocr_candidate_rerank.py"
spec = importlib.util.spec_from_file_location("ocr_v2", OCR_SCRIPT)
ocr_v2 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["ocr_v2"] = ocr_v2
spec.loader.exec_module(ocr_v2)


DEFAULT_INPUT_JSON = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_gemini.json"
)
DEFAULT_V10_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_openroads_v10.csv"
)
DEFAULT_AUDIT_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_plan_ocr_anchor_audit.csv"
)
DEFAULT_FULL_GPKG = Path("/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg")
DEFAULT_FULL_LAYER = "mansfield-manual-polygon-link"
DEFAULT_SAMPLE_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42.gpkg"
)
DEFAULT_SAMPLE_LAYER = "mansfield-manual-polygon-link-random200"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_candidate_evidence_roi_v1"
)

MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)


def norm(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", norm(text))


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def in_bbox(easting: float, northing: float, pad: float = 5000) -> bool:
    minx, miny, maxx, maxy = MANSFIELD_BBOX
    return minx - pad <= easting <= maxx + pad and miny - pad <= northing <= maxy + pad


def safe_json_list(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", "[]") or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def split_anchors(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    out = []
    for part in str(value).split("|"):
        n = norm(part)
        if n and n not in out:
            out.append(n)
    return out


TOKEN_STOP = {
    "LAND",
    "SITE",
    "PLOT",
    "PLOTS",
    "UNIT",
    "UNITS",
    "ADJACENT",
    "REAR",
    "OFF",
    "THE",
    "AND",
    "FOR",
    "ROAD",
    "STREET",
    "LANE",
    "AVENUE",
    "CLOSE",
    "WAY",
    "DRIVE",
    "GROVE",
    "CRESCENT",
    "HILL",
    "GATE",
    "SIDE",
    "PARK",
    "MANSFIELD",
    "NOTTS",
    "NOTTINGHAMSHIRE",
    "FOREST",
    "TOWN",
    "WOODHOUSE",
    "WARSOP",
}

DIRECTION_SUFFIXES = {"EAST", "WEST", "NORTH", "SOUTH"}
ROAD_SUFFIXES = {
    "ROAD",
    "STREET",
    "LANE",
    "AVENUE",
    "CLOSE",
    "WAY",
    "DRIVE",
    "GROVE",
    "CRESCENT",
    "HILL",
    "GATE",
    "SIDE",
    "WALK",
    "PLACE",
    "SQUARE",
    "COURT",
    "PARK",
    "RISE",
    "VIEW",
    "TERRACE",
    "ROW",
    "YARD",
    "CROFT",
}


def sig_tokens(text: Any) -> set[str]:
    return {t for t in norm(text).split() if len(t) >= 4 and t not in TOKEN_STOP and not t.isdigit()}


def token_overlap(a: Any, b: Any) -> int:
    return len(sig_tokens(a) & sig_tokens(b))


def road_core_tokens(value: Any) -> tuple[str, ...]:
    tokens = norm(value).split()
    if len(tokens) >= 3 and tokens[-1] in DIRECTION_SUFFIXES and tokens[-2] in ROAD_SUFFIXES:
        tokens = tokens[:-1]
    if len(tokens) >= 2 and tokens[-1] in ROAD_SUFFIXES:
        return tuple(tokens[:-1])
    return tuple(tokens)


def road_stem_compact(value: Any) -> str:
    tokens = norm(value).split()
    if len(tokens) >= 3 and tokens[-1] in DIRECTION_SUFFIXES and tokens[-2] in ROAD_SUFFIXES:
        tokens = tokens[:-1]
    return "".join(tokens)


def roadlike_phrases(text: Any) -> list[str]:
    words = norm(text).split()
    out: list[str] = []
    for idx, token in enumerate(words):
        if token not in ROAD_SUFFIXES:
            continue
        for width in range(2, 6):
            start = idx - width + 1
            if start < 0:
                continue
            phrase = " ".join(words[start : idx + 1])
            if phrase not in out:
                out.append(phrase)
    return out


@dataclass
class EvidencePoint:
    x: float
    y: float
    weight: float
    source: str
    label: str


def add_point(points: list[EvidencePoint], x: Any, y: Any, weight: float, source: str, label: Any) -> None:
    px = parse_float(x)
    py = parse_float(y)
    if px is None or py is None or not in_bbox(px, py):
        return
    points.append(EvidencePoint(px, py, weight, source, str(label or source)[:160]))


def load_road_geoms(open_roads: Path) -> dict[str, Any]:
    roads = gpd.read_file(
        open_roads,
        layer="road_link",
        bbox=MANSFIELD_BBOX,
        columns=["name_1", "name_2"],
    )
    if roads.crs is not None:
        roads = roads.to_crs(27700)
    groups: dict[str, list[Any]] = {}
    for _, row in roads.iterrows():
        for col in ("name_1", "name_2"):
            name = norm(row.get(col))
            if name:
                groups.setdefault(name, []).append(row.geometry)
    return {name: unary_union(geoms) for name, geoms in groups.items()}


def extract_address_roads(original_address: Any, road_names: set[str]) -> list[str]:
    caddr = compact(original_address)
    phrase_cores = [road_core_tokens(phrase) for phrase in roadlike_phrases(original_address)]
    found: list[str] = []
    exact_cores: set[tuple[str, ...]] = set()
    for road in sorted(road_names, key=lambda item: (-len(item), item)):
        croad = compact(road)
        cstem = road_stem_compact(road)
        if len(croad) >= 7 and (croad in caddr or (len(cstem) >= 7 and cstem in caddr)):
            core = road_core_tokens(road)
            if core and any(len(existing) > len(core) and existing[-len(core) :] == core for existing in exact_cores):
                continue
            found.append(road)
            if core:
                exact_cores.add(core)
        if len(found) >= 8:
            break

    for road in sorted(road_names, key=lambda item: (-len(item), item)):
        if road in found:
            continue
        road_core = road_core_tokens(road)
        fuzzy_suffix_hit = bool(
            road_core
            and road_core not in exact_cores
            and any((len(core) >= 2 or (len(core) == 1 and len(core[0]) >= 8)) and core == road_core for core in phrase_cores)
        )
        if fuzzy_suffix_hit:
            found.append(road)
        if len(found) >= 8:
            break
    return found


def representative_points(geom: Any) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [part if part.geom_type == "Point" else part.centroid for part in geom.geoms if not part.is_empty]
    return [geom.centroid]


def pair_zone_points(roads: list[str], road_geoms: dict[str, Any], limit: int = 60) -> list[tuple[Point, int, str]]:
    zones: list[tuple[Point, int, str]] = []
    for a, b in combinations(roads, 2):
        ga = road_geoms.get(a)
        gb = road_geoms.get(b)
        if ga is None or gb is None:
            continue
        inter = ga.intersection(gb)
        if inter.is_empty:
            p1, p2 = nearest_points(ga, gb)
            points = [LineString([p1, p2]).centroid]
        else:
            points = representative_points(inter)
        for point in points:
            support = 0
            for road in roads:
                geom = road_geoms.get(road)
                if geom is not None and point.distance(geom) <= 350:
                    support += 1
            zones.append((point, support, f"{a}+{b}"))
    zones.sort(key=lambda item: (-item[1], item[0].x, item[0].y))
    deduped: list[tuple[Point, int, str]] = []
    for zone in zones:
        if all(zone[0].distance(existing[0]) > 45 for existing in deduped):
            deduped.append(zone)
        if len(deduped) >= limit:
            break
    return deduped


def road_midpoint_points(
    roads: list[str],
    road_geoms: dict[str, Any],
    reference_points: list[EvidencePoint],
    per_road_limit: int = 3,
    total_limit: int = 10,
) -> list[tuple[Point, float, str]]:
    if not roads:
        return []
    reference = None
    if reference_points:
        total_weight = sum(max(0.1, point.weight) for point in reference_points)
        reference = Point(
            sum(point.x * max(0.1, point.weight) for point in reference_points) / total_weight,
            sum(point.y * max(0.1, point.weight) for point in reference_points) / total_weight,
        )

    candidates: list[tuple[float, Point, float, str]] = []
    for road in roads:
        geom = road_geoms.get(road)
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        road_length = float(getattr(geom, "length", 0.0) or 0.0)
        road_items: list[tuple[float, Point, float, str]] = []
        for part in parts:
            length = float(getattr(part, "length", 0.0) or 0.0)
            if length <= 8:
                continue
            point = part.interpolate(length / 2.0)
            distance_to_reference = point.distance(reference) if reference is not None else 0.0
            compactness_bonus = 4.2 if road_length <= 350 else 1.1 if road_length <= 900 else 0.0
            distance_penalty = min(0.8, distance_to_reference / 1200.0) if road_length <= 350 else min(2.5, distance_to_reference / 700.0)
            score = compactness_bonus + min(1.8, length / 170.0) - distance_penalty
            road_items.append((score, point, road_length, road))
        road_items.sort(key=lambda item: (-item[0], item[1].x, item[1].y))
        candidates.extend(road_items[:per_road_limit])

    candidates.sort(key=lambda item: (-item[0], item[3], item[1].x, item[1].y))
    selected: list[tuple[Point, float, str]] = []
    for score, point, road_length, road in candidates:
        if any(point.distance(old_point) < 95 for old_point, _, _ in selected):
            continue
        weight = 3.4 + max(0.0, score)
        selected.append((point, weight, road))
        if len(selected) >= total_limit:
            break
    return selected


def decode_ocr_grid(raw: Any, method: Any, reference_points: list[EvidencePoint]) -> list[tuple[float, float]]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    method = str(method or "").lower()
    decoded: list[tuple[float, float]] = []
    if method == "slash":
        pt = ocr_v2.decode_slash_grid(raw)
        if pt:
            decoded.append(pt)
    elif method == "compact":
        decoded.extend(ocr_v2.decode_compact_grid(raw))
    else:
        groups = re.findall(r"[A-Z0-9]{4,6}", raw.upper())
        if len(groups) >= 2:
            pt = ocr_v2.decode_pair_grid(groups[0], groups[1])
            if pt:
                decoded.append(pt)
        decoded.extend(ocr_v2.decode_compact_grid(raw))

    decoded = [(x, y) for x, y in decoded if in_bbox(x, y, pad=0)]
    if len(decoded) <= 1 or not reference_points:
        return decoded
    # For ambiguous compact grids, keep the band nearest to the existing
    # candidate cloud; this mirrors the previous OCR geocode experiment.
    rx = sum(p.x * p.weight for p in reference_points) / sum(p.weight for p in reference_points)
    ry = sum(p.y * p.weight for p in reference_points) / sum(p.weight for p in reference_points)
    decoded.sort(key=lambda pt: math.hypot(pt[0] - rx, pt[1] - ry))
    return decoded[:1]


def candidate_weight(source: str, rank: int, item: dict[str, Any], original_address: str) -> float:
    label = item.get("address") or item.get("name") or ""
    overlap = token_overlap(label, original_address)
    if source == "road_sample":
        return max(3.8, 6.2 - rank * 0.06 + min(1.0, overlap * 0.2))
    if source == "os":
        return max(0.8, 4.8 - rank * 0.18 + min(2.0, overlap * 0.55))
    if source == "gog":
        match = parse_float(item.get("match")) or parse_float(item.get("road_score")) or 0.5
        return max(0.8, 4.3 - rank * 0.35 + match * 1.3 + min(1.5, overlap * 0.45))
    if source == "fallback":
        score = parse_float(item.get("fallback_score")) or 55
        return max(0.8, min(5.0, score / 22.0) - rank * 0.22 + min(1.2, overlap * 0.35))
    return 1.0


def evidence_points_for_case(
    row: pd.Series,
    raw: dict[str, Any] | None,
    audit_row: pd.Series | None,
    road_geoms: dict[str, Any],
) -> list[EvidencePoint]:
    points: list[EvidencePoint] = []
    original_address = str(row.get("original_address") or "")
    best_conf = parse_float(raw.get("best_confidence") if raw else None) or 0

    # Current best point is useful, but low-confidence current geocodes should
    # not dominate the evidence cluster.
    current_weight = 3.0 + min(2.5, max(0.0, best_conf - 50) / 18.0)
    source = str(row.get("v7_selected_source") or row.get("selected_source") or "current")
    if source == "ocr_geocode":
        current_weight += 3.0
    elif source in {"openname_place", "openname_named"}:
        current_weight += 1.0
    add_point(
        points,
        row.get("v7_selected_easting") or row.get("selected_easting"),
        row.get("v7_selected_northing") or row.get("selected_northing"),
        current_weight,
        "current",
        row.get("v7_selected_address") or row.get("selected_address"),
    )

    if raw:
        for idx, item in enumerate(safe_json_list(raw.get("os_candidates_json"))[:25]):
            point_source = "road_sample" if item.get("candidate_pool_source") == "openroads_road_sample" else "os"
            add_point(
                points,
                item.get("easting_27700"),
                item.get("northing_27700"),
                candidate_weight(point_source, idx, item, original_address),
                point_source,
                item.get("address"),
            )
        for idx, item in enumerate(safe_json_list(raw.get("gog_candidates_json"))[:10]):
            add_point(
                points,
                item.get("easting_27700"),
                item.get("northing_27700"),
                candidate_weight("gog", idx, item, original_address),
                "gog",
                item.get("address"),
            )
        for idx, item in enumerate(safe_json_list(raw.get("os_fallback_pool_json"))[:20]):
            add_point(
                points,
                item.get("easting_27700"),
                item.get("northing_27700"),
                candidate_weight("fallback", idx, item, original_address),
                "fallback",
                item.get("address"),
            )
        add_point(
            points,
            raw.get("web_research_best_easting_27700"),
            raw.get("web_research_best_northing_27700"),
            3.2,
            "web",
            raw.get("web_research_best_address"),
        )
        for idx, item in enumerate(safe_json_list(raw.get("range_parity_points_json"))[:12]):
            weight = parse_float(item.get("weight"))
            if weight is None:
                weight = 10.5 if item.get("point_role") == "range_parity_centroid" else 7.8
            add_point(
                points,
                item.get("easting_27700"),
                item.get("northing_27700"),
                max(1.0, weight - min(1.5, idx * 0.15)),
                "range_parity",
                item.get("address") or item.get("range_text"),
            )

    for x, y in decode_ocr_grid(row.get("ocr_grid_raw"), row.get("ocr_grid_method"), points):
        weight = 9.0 if str(row.get("ocr_grid_gate") or "").startswith(("low", "baseline_outside", "accepted")) else 5.5
        add_point(points, x, y, weight, "ocr_grid", f"OCR grid {row.get('ocr_grid_raw')}")

    plan_roads = split_anchors(audit_row.get("matched_roads") if audit_row is not None else "")
    address_roads = extract_address_roads(original_address, set(road_geoms))
    roads = []
    for road in [*address_roads, *plan_roads]:
        if road in road_geoms and road not in roads:
            roads.append(road)
    for zone, support, label in pair_zone_points(roads, road_geoms, limit=40):
        # Road-pair zones are approximate.  They should influence a cluster only
        # when other candidates agree with the same area.
        weight = 2.8 + min(3.2, support * 0.8)
        add_point(points, zone.x, zone.y, weight, "road_zone", label)

    for point, weight, label in road_midpoint_points(roads, road_geoms, points):
        add_point(points, point.x, point.y, weight, "road_midpoint", label)

    return points


def choose_roi(points: list[EvidencePoint], side: float = 150.0) -> tuple[box | None, dict[str, Any]]:
    if not points:
        return None, {"status": "no_points"}
    half = side / 2.0

    centers: list[tuple[float, float, str]] = [(p.x, p.y, p.source) for p in points]
    # Add local weighted centroids as candidate centers.
    for p in points:
        local = [q for q in points if abs(q.x - p.x) <= 140 and abs(q.y - p.y) <= 140]
        if len({q.source for q in local}) >= 2:
            sw = sum(q.weight for q in local)
            centers.append((sum(q.x * q.weight for q in local) / sw, sum(q.y * q.weight for q in local) / sw, "local_centroid"))

    best = None
    for cx, cy, center_source in centers:
        inside = [p for p in points if abs(p.x - cx) <= half and abs(p.y - cy) <= half]
        near = [p for p in points if math.hypot(p.x - cx, p.y - cy) <= 140]
        sources = {p.source for p in inside}
        source_bonus = len(sources) * 4.0
        agreement_bonus = 6.0 if len(sources & {"os", "gog", "fallback", "current", "ocr_grid", "range_parity"}) >= 2 else 0.0
        anchor_bonus = 5.0 if sources & {"road_zone", "road_midpoint", "range_parity"} and len(sources) >= 2 else 0.0
        score = (
            sum(p.weight for p in inside)
            + 0.35 * sum(p.weight for p in near)
            + source_bonus
            + agreement_bonus
            + anchor_bonus
        )
        item = {
            "cx": cx,
            "cy": cy,
            "score": score,
            "inside_weight": sum(p.weight for p in inside),
            "inside_count": len(inside),
            "near_count": len(near),
            "sources": "|".join(sorted(sources)),
            "center_source": center_source,
            "evidence_labels": " || ".join(f"{p.source}:{p.label}@{p.x:.0f},{p.y:.0f} w={p.weight:.1f}" for p in sorted(inside, key=lambda p: -p.weight)[:12]),
        }
        if best is None or score > best["score"]:
            best = item

    assert best is not None
    roi = box(best["cx"] - half, best["cy"] - half, best["cx"] + half, best["cy"] + half)
    best["status"] = "ok"
    return roi, best


def rank_roi_polygons(polygons: gpd.GeoDataFrame, roi: Any, road_names: list[str], road_geoms: dict[str, Any]) -> pd.DataFrame:
    center = roi.centroid
    candidates = polygons[polygons.geometry.intersects(roi)].copy()
    if candidates.empty:
        candidates = polygons[polygons.geometry.distance(center) <= 75].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["candidate_key"] = candidates["unique_key"].astype(str)
    candidates["centroid_dist_m"] = candidates.geometry.centroid.distance(center)
    candidates["roi_intersection_area"] = candidates.geometry.intersection(roi).area
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
        - candidates["centroid_dist_m"].clip(upper=250) * 0.015
    )
    return candidates.sort_values(["rank_score", "roi_intersection_area"], ascending=[False, False]).reset_index(drop=True)


def load_truth(sample_gpkg: Path, sample_layer: str) -> dict[str, Any]:
    gdf = gpd.read_file(sample_gpkg, layer=sample_layer)
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    return {str(row["unique_key"]): row.geometry for _, row in gdf.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=DEFAULT_V10_CSV)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--full-gpkg", type=Path, default=DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=DEFAULT_FULL_LAYER)
    parser.add_argument("--sample-gpkg", type=Path, default=DEFAULT_SAMPLE_GPKG)
    parser.add_argument("--sample-layer", default=DEFAULT_SAMPLE_LAYER)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--roi-side", type=float, default=150.0)
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--generic-place-names", default="")
    args = parser.parse_args()
    global MANSFIELD_BBOX, TOKEN_STOP
    MANSFIELD_BBOX = parse_bbox(args.local_bbox, MANSFIELD_BBOX)
    TOKEN_STOP = set(TOKEN_STOP) | set(parse_names(args.generic_place_names, []))

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
    truth = load_truth(args.sample_gpkg, args.sample_layer)
    road_geoms = load_road_geoms(args.open_roads)

    rows = []
    top_rows = []
    evidence_rows = []
    for _, row in v10.iterrows():
        key = str(row["key"])
        base_key = str(row.get("base_key") or key).split("_", 1)[0]
        raw = raw_rows.get(key)
        audit_row = audit_by_key.get(key)
        target_geom = truth.get(base_key)
        if target_geom is None:
            continue
        points = evidence_points_for_case(row, raw, audit_row, road_geoms)
        roi, info = choose_roi(points, side=args.roi_side)
        best_conf = parse_float(raw.get("best_confidence") if raw else None)
        plan_roads = split_anchors(audit_row.get("matched_roads") if audit_row is not None else "")
        address_roads = extract_address_roads(row.get("original_address"), set(road_geoms))
        all_roads = []
        for road in [*address_roads, *plan_roads]:
            if road not in all_roads:
                all_roads.append(road)

        if roi is None:
            rows.append({"key": key, "base_key": base_key, "status": info.get("status"), "best_confidence": best_conf})
            continue

        ranked_polys = rank_roi_polygons(polygons, roi, all_roads, road_geoms)
        candidate_count = int(len(ranked_polys))
        target_hit = ranked_polys[ranked_polys["candidate_key"] == base_key] if not ranked_polys.empty else pd.DataFrame()
        target_rank = int(target_hit.index[0] + 1) if not target_hit.empty else None
        centroid_inside = roi.contains(target_geom.centroid)
        polygon_intersects = roi.intersects(target_geom)
        target_area_inside_ratio = target_geom.intersection(roi).area / target_geom.area if target_geom.area else 0.0

        for _, cand in ranked_polys.head(10).iterrows():
            top_rows.append(
                {
                    "case_key": key,
                    "base_key": base_key,
                    "candidate_key": cand["candidate_key"],
                    "rank": int(cand.name + 1),
                    "is_target": cand["candidate_key"] == base_key,
                    "rank_score": float(cand["rank_score"]),
                    "roi_intersection_area": float(cand["roi_intersection_area"]),
                    "centroid_dist_m": float(cand["centroid_dist_m"]),
                }
            )
        for p in points:
            evidence_rows.append(
                {
                    "case_key": key,
                    "base_key": base_key,
                    "source": p.source,
                    "label": p.label,
                    "easting": p.x,
                    "northing": p.y,
                    "weight": p.weight,
                    "inside_roi": roi.contains(Point(p.x, p.y)),
                }
            )

        rows.append(
            {
                "key": key,
                "base_key": base_key,
                "original_address": row.get("original_address"),
                "best_confidence": best_conf,
                "current_distance_m": row.get("v7_selected_distance_m"),
                "status": info.get("status"),
                "roi_side_m": args.roi_side,
                "roi_minx": roi.bounds[0],
                "roi_miny": roi.bounds[1],
                "roi_maxx": roi.bounds[2],
                "roi_maxy": roi.bounds[3],
                "roi_center_easting": info.get("cx"),
                "roi_center_northing": info.get("cy"),
                "roi_score": info.get("score"),
                "roi_sources": info.get("sources"),
                "roi_inside_evidence_count": info.get("inside_count"),
                "roi_inside_weight": info.get("inside_weight"),
                "evidence_point_count": len(points),
                "plan_roads": " | ".join(plan_roads),
                "address_roads": " | ".join(address_roads),
                "all_roads": " | ".join(all_roads),
                "roi_contains_target_centroid": centroid_inside,
                "roi_intersects_target_polygon": polygon_intersects,
                "target_area_inside_ratio": target_area_inside_ratio,
                "roi_candidate_polygon_count": candidate_count,
                "target_rank_in_roi_candidates": target_rank,
                "top1_candidate_key": ranked_polys.iloc[0]["candidate_key"] if not ranked_polys.empty else None,
                "top5_contains_target": target_rank is not None and target_rank <= 5,
                "top10_contains_target": target_rank is not None and target_rank <= 10,
                "top20_contains_target": target_rank is not None and target_rank <= 20,
                "evidence_labels": info.get("evidence_labels"),
            }
        )

    out = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)
    evidence = pd.DataFrame(evidence_rows)
    csv_path = args.output_prefix.with_suffix(".csv")
    xlsx_path = args.output_prefix.with_suffix(".xlsx")
    top_path = args.output_prefix.with_name(args.output_prefix.name + "_top10.csv")
    evidence_path = args.output_prefix.with_name(args.output_prefix.name + "_evidence.csv")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    out.to_csv(csv_path, index=False)
    top.to_csv(top_path, index=False)
    evidence.to_csv(evidence_path, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        out.to_excel(writer, sheet_name="roi_summary", index=False)
        top.to_excel(writer, sheet_name="top10_polygons", index=False)
        evidence.head(10000).to_excel(writer, sheet_name="evidence_points", index=False)

    ok = out[out["status"] == "ok"].copy()
    low = ok[pd.to_numeric(ok["best_confidence"], errors="coerce") < 75].copy()
    two_anchor = ok[ok["all_roads"].fillna("").map(lambda x: len([p for p in str(x).split("|") if p.strip()]) >= 2)].copy()

    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "cases": int(len(df)),
            "centroid_inside_roi": int(df["roi_contains_target_centroid"].fillna(False).sum()),
            "polygon_intersects_roi": int(df["roi_intersects_target_polygon"].fillna(False).sum()),
            "top1_polygon": int((pd.to_numeric(df["target_rank_in_roi_candidates"], errors="coerce") <= 1).sum()),
            "top5_polygon": int(df["top5_contains_target"].fillna(False).sum()),
            "top10_polygon": int(df["top10_contains_target"].fillna(False).sum()),
            "top20_polygon": int(df["top20_contains_target"].fillna(False).sum()),
            "mean_roi_candidate_count": float(pd.to_numeric(df["roi_candidate_polygon_count"], errors="coerce").mean()),
            "median_roi_candidate_count": float(pd.to_numeric(df["roi_candidate_polygon_count"], errors="coerce").median()),
            "mean_current_distance_m": float(pd.to_numeric(df["current_distance_m"], errors="coerce").mean()),
        }

    summary = {
        "full_polygon_count": int(len(polygons)),
        "roi_side_m": args.roi_side,
        "all_cases": summarize(ok),
        "low_confidence_cases": summarize(low),
        "cases_with_2plus_road_anchors": summarize(two_anchor),
        "output_csv": str(csv_path),
        "output_xlsx": str(xlsx_path),
        "output_top10_csv": str(top_path),
        "output_evidence_csv": str(evidence_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
