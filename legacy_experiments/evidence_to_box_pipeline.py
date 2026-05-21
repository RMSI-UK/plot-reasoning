#!/usr/bin/env python3
"""Generate production-visible evidence boxes for planning parcel search.

This is a target-oriented alternative to "pick one best geocode".  It does not
ask an LLM to choose a final OS address.  Instead it builds multiple candidate
boxes from locally available evidence:

- current input point/polygon as a protected prior
- address road/locality/property tokens
- OpenRoads named road geometry
- OpenNames named places/properties/roads
- Monmouthshire WFS Land/Building polygons near evidence anchors

The hidden/manual polygon is never used to generate candidates.  If the input
layer already has geometry, it is used only for offline evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union


DEFAULT_INPUT_GPKG = Path("/data/monmouthshire/spatial/base-map/tmp_output/monmouthshire_input_layer_random200_seed42_20260505.gpkg")
DEFAULT_INPUT_LAYER = "features"
DEFAULT_WFS_GPKG = Path("/data/monmouthshire/spatial/base-map/monmouthshire_os_wfs_merge.gpkg")
DEFAULT_WFS_LAYER = "monmouthshire_polygons_in_buffers"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_OUTPUT_PREFIX = Path("/data/monmouthshire/spatial/base-map/tmp_output/monmouthshire_evidence_boxes_random200_seed42_v1")
MONMOUTHSHIRE_BBOX = (320000.0, 179000.0, 357000.0, 233500.0)

ROAD_SUFFIXES = {
    "STREET", "ST", "ROAD", "RD", "LANE", "LN", "CLOSE", "COURT", "AVENUE", "AVE",
    "DRIVE", "DR", "WAY", "PLACE", "PL", "GATE", "GROVE", "TERRACE", "VIEW", "HILL",
    "GARDENS", "CRESCENT", "WALK", "RISE", "MEWS", "PARADE", "SQUARE", "ROW", "BANK",
    "CHASE", "CROFT", "GREEN", "PARK", "FEE",
}
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
GENERIC_NAME_TOKENS = {
    "LAND", "SITE", "PLOT", "PLOTS", "ADJACENT", "ADJOINING", "REAR", "FRONT",
    "FORMER", "NEW", "OLD", "THE", "AT", "OFF", "OF", "TO", "AND", "BETWEEN",
    "ROAD", "STREET", "LANE", "CLOSE", "WAY", "MONMOUTHSHIRE",
}
POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
ROAD_RE = re.compile(
    r"\b([A-Z0-9][A-Z0-9'&.\- ]+?\b(?:"
    + "|".join(sorted(ROAD_SUFFIXES, key=len, reverse=True))
    + r"))\b",
    re.I,
)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" ,.")


def norm(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", clean_text(value).upper())


def norm_road(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", clean_text(value).upper()).strip()
    tokens = [ROAD_TOKEN_NORMALIZATION.get(token, token) for token in text.split()]
    return "".join(tokens)


def parse_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def derive_case_key(row: pd.Series, idx: int) -> str:
    for field in ["unique_key", "key", "further-information-reference", "geomid", "queryid"]:
        if field in row and clean_text(row.get(field)):
            return clean_text(row.get(field))
    return str(idx)


def make_square(cx: float, cy: float, side: float) -> Any:
    half = side / 2.0
    return box(cx - half, cy - half, cx + half, cy + half)


def make_bounds_box(geom: Any, margin: float, max_side: float = 260.0) -> Any:
    minx, miny, maxx, maxy = geom.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    side = max(maxx - minx, maxy - miny) + 2 * margin
    side = min(max(side, 80.0), max_side)
    return make_square(cx, cy, side)


@dataclass
class EvidencePoint:
    x: float
    y: float
    source: str
    label: str
    weight: float


@dataclass
class CandidateBox:
    case_key: str
    rank: int
    score: float
    source: str
    reason: str
    center_x: float
    center_y: float
    side_m: float
    evidence_sources: str
    evidence_labels: str
    geometry: Any


def extract_roads(raw_address: str, road_lookup: dict[str, str]) -> list[str]:
    roads: list[str] = []
    for match in ROAD_RE.finditer(clean_text(raw_address).upper()):
        phrase = clean_text(match.group(1))
        key = norm_road(phrase)
        if key in road_lookup and road_lookup[key] not in roads:
            roads.append(road_lookup[key])
    # Handle "3 Hillside, Mitchel Troy" where the suffix is missing in raw.
    raw_key = norm_road(raw_address)
    for key, road in road_lookup.items():
        if len(key) >= 8 and key in raw_key and road not in roads:
            roads.append(road)
    return roads[:5]


def extract_name_phrases(raw_address: str, roads: list[str]) -> list[str]:
    text = POSTCODE_RE.sub(" ", clean_text(raw_address))
    parts = [clean_text(part) for part in re.split(r"[,;/()]+", text) if clean_text(part)]
    road_keys = {norm(road) for road in roads}
    phrases: list[str] = []
    for part in parts[:5]:
        cleaned = re.sub(
            r"(?i)\b(?:land|site|plot|plots|adjacent|adjoining|rear|front|former|new|old|at|off|of|to|between|the)\b",
            " ",
            part,
        )
        cleaned = re.sub(r"\b\d+[A-Z]?(?:\s*-\s*\d+[A-Z]?)?\b", " ", cleaned, flags=re.I)
        cleaned = clean_text(cleaned)
        if not cleaned or norm(cleaned) in road_keys:
            continue
        tokens = [tok for tok in re.findall(r"[A-Za-z0-9]+", cleaned) if tok.upper() not in GENERIC_NAME_TOKENS]
        if not tokens:
            continue
        phrase = clean_text(" ".join(tokens))
        if len(norm(phrase)) >= 4 and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:6]


def extract_numbers(raw_address: str) -> set[int]:
    text = POSTCODE_RE.sub(" ", clean_text(raw_address).upper())
    out: set[int] = set()
    for start_s, end_s in re.findall(r"\b(\d{1,4})\s*(?:-|TO|AND)\s*(\d{1,4})\b", text):
        start = int(start_s)
        end = int(end_s)
        low, high = sorted((start, end))
        if 0 < low < 1000 and 0 < high < 1000 and high - low <= 20:
            out.update(range(low, high + 1))
    for number_s in re.findall(r"\b\d{1,4}[A-Z]?\b", text):
        number = int(re.match(r"\d+", number_s).group(0))
        if 0 < number < 1000:
            out.add(number)
    return out


def load_target(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(27700)
    else:
        gdf = gdf.to_crs(27700)
    return gdf.reset_index(drop=True)


def load_roads(path: Path, bbox_tuple: tuple[float, float, float, float]) -> tuple[dict[str, Any], dict[str, str]]:
    roads = gpd.read_file(
        path,
        layer="road_link",
        bbox=bbox_tuple,
        columns=["id", "name_1", "road_classification_number", "road_function", "length"],
    )
    roads = roads.to_crs(27700)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty & roads["name_1"].notna()].copy()
    geoms: dict[str, list[Any]] = {}
    display: dict[str, str] = {}
    for _, row in roads.iterrows():
        name = clean_text(row.get("name_1"))
        key = norm_road(name)
        if not key:
            continue
        geoms.setdefault(name, []).append(row.geometry)
        display[key] = name
    merged = {name: unary_union(items) for name, items in geoms.items()}
    return merged, display


def load_open_names(path: Path, bbox_tuple: tuple[float, float, float, float]) -> pd.DataFrame:
    minx, miny, maxx, maxy = bbox_tuple
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            """
            select NAME1, LOCAL_TYPE, GEOMETRY_X, GEOMETRY_Y, MBR_XMIN, MBR_YMIN, MBR_XMAX, MBR_YMAX,
                   POPULATED_PLACE, DISTRICT_BOROUGH, COUNTY_UNITARY, POSTCODE_DISTRICT
            from open_names
            where cast(GEOMETRY_X as real) between ? and ?
              and cast(GEOMETRY_Y as real) between ? and ?
            """,
            con,
            params=(minx, maxx, miny, maxy),
        )
    finally:
        con.close()
    df["name_key"] = df["NAME1"].map(norm)
    df["x"] = pd.to_numeric(df["GEOMETRY_X"], errors="coerce")
    df["y"] = pd.to_numeric(df["GEOMETRY_Y"], errors="coerce")
    return df[df["x"].notna() & df["y"].notna()].reset_index(drop=True)


def load_wfs(path: Path, layer: str, bbox_tuple: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    columns = ["GmlID", "OBJECTID", "TOID", "Theme", "DescriptiveGroup", "DescriptiveTerm", "Make", "Shape_Area"]
    gdf = gpd.read_file(path, layer=layer, bbox=bbox_tuple, columns=columns)
    if gdf.crs is None:
        gdf = gdf.set_crs(27700)
    else:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf["Theme"].fillna("").str.contains("Land|Building", case=False, regex=True)].copy()
    gdf["wfs_id"] = gdf["TOID"].fillna("").astype(str)
    missing = gdf["wfs_id"].isin(["", "nan", "None"])
    gdf.loc[missing, "wfs_id"] = gdf.loc[missing, "GmlID"].fillna(gdf.loc[missing, "OBJECTID"]).astype(str)
    gdf["area_m2"] = gdf.geometry.area
    return gdf.reset_index(drop=True)


def add_box(
    boxes_out: list[dict[str, Any]],
    case_key: str,
    geom: Any,
    source: str,
    reason: str,
    score: float,
    evidence: list[EvidencePoint],
    side_m: float | None = None,
) -> None:
    if geom is None or getattr(geom, "is_empty", False):
        return
    center = geom.centroid
    labels = []
    sources = []
    for item in evidence:
        if item.source not in sources:
            sources.append(item.source)
        if item.label and item.label not in labels:
            labels.append(item.label)
    if side_m is None:
        minx, miny, maxx, maxy = geom.bounds
        side_m = max(maxx - minx, maxy - miny)
    boxes_out.append(
        {
            "case_key": case_key,
            "score": round(float(score), 3),
            "source": source,
            "reason": reason,
            "center_x": round(center.x, 3),
            "center_y": round(center.y, 3),
            "side_m": round(float(side_m), 3),
            "evidence_sources": "|".join(sources),
            "evidence_labels": "|".join(labels[:12]),
            "geometry": geom,
        }
    )


def query_wfs_near(wfs: gpd.GeoDataFrame, geom: Any) -> gpd.GeoDataFrame:
    idx = list(wfs.sindex.query(geom, predicate="intersects"))
    if not idx:
        return wfs.iloc[[]]
    return wfs.iloc[idx].copy()


def candidate_boxes_for_case(
    case_key: str,
    raw_address: str,
    current_geom: Any,
    roads_by_name: dict[str, Any],
    road_lookup: dict[str, str],
    open_names: pd.DataFrame,
    wfs: gpd.GeoDataFrame,
    roi_side: float,
    include_current_boxes: bool,
) -> list[dict[str, Any]]:
    evidence: list[EvidencePoint] = []
    boxes_out: list[dict[str, Any]] = []
    current_point = current_geom.representative_point() if current_geom is not None and not current_geom.is_empty else None

    if current_point is not None:
        evidence.append(EvidencePoint(current_point.x, current_point.y, "current", "current_geometry", 30.0))
        if include_current_boxes and current_geom.geom_type in {"Polygon", "MultiPolygon"}:
            geom_box = make_bounds_box(current_geom, 20.0, max_side=320.0)
            add_box(boxes_out, case_key, geom_box, "current_polygon", "protected_current_polygon_bounds", 95.0, evidence, None)
        if include_current_boxes:
            add_box(
                boxes_out,
                case_key,
                make_square(current_point.x, current_point.y, 90.0),
                "current_point",
                "protected_current_90m",
                90.0,
                evidence,
                90.0,
            )
            add_box(
                boxes_out,
                case_key,
                make_square(current_point.x, current_point.y, roi_side),
                "current_point",
                f"protected_current_{int(roi_side)}m",
                72.0,
                evidence,
                roi_side,
            )

    roads = extract_roads(raw_address, road_lookup)
    name_phrases = extract_name_phrases(raw_address, roads)
    numbers = extract_numbers(raw_address)

    road_evidence: list[EvidencePoint] = []
    for road in roads:
        road_geom = roads_by_name.get(road)
        if road_geom is None:
            continue
        if current_point is not None:
            _, on_road = nearest_points(current_point, road_geom)
        else:
            on_road = road_geom.centroid
        ep = EvidencePoint(on_road.x, on_road.y, "road", road, 20.0)
        evidence.append(ep)
        road_evidence.append(ep)
        add_box(
            boxes_out,
            case_key,
            make_square(on_road.x, on_road.y, roi_side),
            "road_nearest",
            f"nearest_point_on_{road}",
            54.0 + (8.0 if current_point is not None and current_point.distance(on_road) <= 80 else 0.0),
            [ep],
            roi_side,
        )

    if len(roads) >= 2:
        for i, road_a in enumerate(roads[:3]):
            for road_b in roads[i + 1 : 4]:
                geom_a = roads_by_name.get(road_a)
                geom_b = roads_by_name.get(road_b)
                if geom_a is None or geom_b is None:
                    continue
                p1, p2 = nearest_points(geom_a, geom_b)
                mid = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
                if p1.distance(p2) <= 180.0:
                    ep = EvidencePoint(mid.x, mid.y, "road_pair", f"{road_a}+{road_b}", 30.0)
                    evidence.append(ep)
                    add_box(
                        boxes_out,
                        case_key,
                        make_square(mid.x, mid.y, roi_side),
                        "road_pair",
                        f"closest_road_pair_{road_a}_{road_b}",
                        78.0 - min(p1.distance(p2), 180.0) * 0.08,
                        [ep],
                        roi_side,
                    )

    raw_key = norm(raw_address)
    name_hits = open_names[open_names["name_key"].map(lambda k: bool(k and len(k) >= 4 and k in raw_key))].copy()
    if name_phrases:
        phrase_keys = {norm(item) for item in name_phrases if len(norm(item)) >= 4}
        more = open_names[open_names["name_key"].isin(phrase_keys)].copy()
        if not more.empty:
            name_hits = pd.concat([name_hits, more], ignore_index=True).drop_duplicates()
    if not name_hits.empty:
        if current_point is not None:
            name_hits["dist_current"] = ((name_hits["x"] - current_point.x) ** 2 + (name_hits["y"] - current_point.y) ** 2) ** 0.5
            name_hits = name_hits.sort_values(["dist_current"]).head(10)
        else:
            name_hits = name_hits.head(10)
    for _, hit in name_hits.iterrows():
        local_type = clean_text(hit.get("LOCAL_TYPE"))
        label = f"{clean_text(hit.get('NAME1'))} [{local_type}]"
        ep = EvidencePoint(float(hit["x"]), float(hit["y"]), "open_names", label, 16.0)
        evidence.append(ep)
        score = 50.0
        if local_type in {"Named Road", "Section Of Named Road"}:
            score -= 6.0
        if local_type in {"Village", "Hamlet", "Suburban Area", "Other Settlement"}:
            score -= 10.0
        add_box(
            boxes_out,
            case_key,
            make_square(ep.x, ep.y, roi_side),
            "open_names",
            f"open_names_{local_type}",
            score,
            [ep],
            roi_side,
        )

    # Evidence consensus boxes: local weighted centroids where multiple sources agree.
    for ep in list(evidence):
        local = [item for item in evidence if math.hypot(item.x - ep.x, item.y - ep.y) <= 160.0]
        if len({item.source for item in local}) >= 2:
            sw = sum(item.weight for item in local)
            cx = sum(item.x * item.weight for item in local) / sw
            cy = sum(item.y * item.weight for item in local) / sw
            add_box(
                boxes_out,
                case_key,
                make_square(cx, cy, roi_side),
                "evidence_consensus",
                "multi_source_local_centroid",
                66.0 + len({item.source for item in local}) * 5.0 + min(len(local), 5),
                local,
                roi_side,
            )

    # WFS boxes near all strong anchors.  These are not final polygon choices;
    # they convert visible Land/Building base geometry into additional boxes.
    anchor_points = [Point(item.x, item.y) for item in evidence if item.source in {"current", "road", "road_pair", "open_names"}]
    seen_wfs: set[str] = set()
    for anchor in anchor_points[:12]:
        near = query_wfs_near(wfs, anchor.buffer(100.0))
        if near.empty:
            near = query_wfs_near(wfs, anchor.buffer(180.0))
        if near.empty:
            continue
        near["dist_anchor"] = near.geometry.distance(anchor)
        near = near.sort_values(["dist_anchor", "area_m2"]).head(8)
        for _, poly in near.iterrows():
            wfs_id = str(poly.get("wfs_id"))
            if wfs_id in seen_wfs:
                continue
            seen_wfs.add(wfs_id)
            geom_box = make_bounds_box(poly.geometry, 18.0, max_side=260.0)
            dist = float(poly.get("dist_anchor") or 0.0)
            score = 44.0 + max(0.0, 18.0 - dist * 0.12)
            if "Building" in clean_text(poly.get("Theme")):
                score += 4.0
            add_box(
                boxes_out,
                case_key,
                geom_box,
                "wfs_land_building",
                f"near_anchor_{wfs_id}",
                score,
                [EvidencePoint(anchor.x, anchor.y, "wfs_anchor", wfs_id, 8.0)],
                None,
            )

    # Deduplicate highly overlapping boxes and rank.
    boxes_out.sort(key=lambda item: item["score"], reverse=True)
    deduped: list[dict[str, Any]] = []
    for item in boxes_out:
        geom = item["geometry"]
        duplicate = False
        for old in deduped:
            old_geom = old["geometry"]
            inter = geom.intersection(old_geom).area
            union = geom.union(old_geom).area
            if union > 0 and inter / union >= 0.82:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
        if len(deduped) >= 40:
            break
    for rank, item in enumerate(deduped, start=1):
        item["rank"] = rank
        item["address_roads"] = "|".join(roads)
        item["address_names"] = "|".join(name_phrases)
        item["address_numbers"] = "|".join(str(n) for n in sorted(numbers))
    return deduped


def evaluate_case(truth_geom: Any, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"box_count": len(boxes)}
    if truth_geom is None or getattr(truth_geom, "is_empty", True) or not boxes:
        for k in [1, 3, 5, 10, 20]:
            out[f"top{k}_hit"] = 0
        out["union_hit"] = 0
        out["union_area_m2"] = 0.0
        return out
    union_geom = unary_union([item["geometry"] for item in boxes])
    out["union_hit"] = int(union_geom.intersects(truth_geom))
    out["union_area_m2"] = round(float(union_geom.area), 3)
    for k in [1, 3, 5, 10, 20]:
        out[f"top{k}_hit"] = int(any(item["geometry"].intersects(truth_geom) for item in boxes[:k]))
    return out


def load_truth_geometries(path: Path | None, layer: str | None, fallback: gpd.GeoDataFrame) -> list[Any]:
    if path is None:
        return list(fallback.geometry)
    truth = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if truth.crs is None:
        truth = truth.set_crs(27700)
    else:
        truth = truth.to_crs(27700)
    if len(truth) != len(fallback):
        raise SystemExit(f"Truth row count {len(truth)} does not match input row count {len(fallback)}")
    return list(truth.geometry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-driven candidate boxes without LLM best-address selection.")
    parser.add_argument("--input-gpkg", type=Path, default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--input-layer", default=DEFAULT_INPUT_LAYER)
    parser.add_argument("--address-column", default="charge-geographic-description")
    parser.add_argument("--wfs-gpkg", type=Path, default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--open-names", type=Path, default=DEFAULT_OPEN_NAMES)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--roi-side", type=float, default=150.0)
    parser.add_argument("--bbox", default=",".join(str(v) for v in MONMOUTHSHIRE_BBOX))
    parser.add_argument("--truth-gpkg", type=Path, help="Optional offline truth/proxy polygon GPKG for evaluation only.")
    parser.add_argument("--truth-layer", help="Optional truth layer name.")
    parser.add_argument("--no-current-boxes", action="store_true", help="Use current geometry as context but do not emit protected current boxes.")
    args = parser.parse_args()

    bbox_tuple = tuple(float(part) for part in args.bbox.split(","))
    if len(bbox_tuple) != 4:
        raise SystemExit("--bbox must be minx,miny,maxx,maxy")

    target = load_target(args.input_gpkg, args.input_layer)
    truth_geoms = load_truth_geometries(args.truth_gpkg, args.truth_layer, target)
    roads_by_name, road_lookup = load_roads(args.open_roads, bbox_tuple)  # type: ignore[arg-type]
    open_names = load_open_names(args.open_names, bbox_tuple)  # type: ignore[arg-type]
    wfs = load_wfs(args.wfs_gpkg, args.wfs_layer, bbox_tuple)  # type: ignore[arg-type]

    all_boxes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for idx, row in target.iterrows():
        case_key = derive_case_key(row, idx + 1)
        raw_address = clean_text(row.get(args.address_column))
        case_boxes = candidate_boxes_for_case(
            case_key=case_key,
            raw_address=raw_address,
            current_geom=row.geometry,
            roads_by_name=roads_by_name,
            road_lookup=road_lookup,
            open_names=open_names,
            wfs=wfs,
            roi_side=float(args.roi_side),
            include_current_boxes=not args.no_current_boxes,
        )
        metrics = evaluate_case(truth_geoms[idx], case_boxes)
        summaries.append(
            {
                "case_key": case_key,
                "row_index": idx + 1,
                "original_address": raw_address,
                "input_geom_type": row.geometry.geom_type if row.geometry is not None else "",
                "truth_geom_type": truth_geoms[idx].geom_type if truth_geoms[idx] is not None else "",
                **metrics,
                "top1_source": case_boxes[0]["source"] if case_boxes else "",
                "top1_reason": case_boxes[0]["reason"] if case_boxes else "",
                "top1_score": case_boxes[0]["score"] if case_boxes else "",
                "address_roads": case_boxes[0].get("address_roads", "") if case_boxes else "",
                "address_names": case_boxes[0].get("address_names", "") if case_boxes else "",
            }
        )
        all_boxes.extend(case_boxes)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    boxes_df = pd.DataFrame([{k: v for k, v in item.items() if k != "geometry"} for item in all_boxes])
    summary_df = pd.DataFrame(summaries)
    boxes_csv = args.output_prefix.with_name(args.output_prefix.name + "_boxes.csv")
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_case_summary.csv")
    boxes_gpkg = args.output_prefix.with_name(args.output_prefix.name + "_boxes.gpkg")
    boxes_df.to_csv(boxes_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    if all_boxes:
        boxes_gdf = gpd.GeoDataFrame(boxes_df, geometry=[item["geometry"] for item in all_boxes], crs="EPSG:27700")
        if boxes_gpkg.exists():
            boxes_gpkg.unlink()
        boxes_gdf.to_file(boxes_gpkg, layer="candidate_boxes", driver="GPKG")

    print(f"cases: {len(summary_df)}")
    print(f"boxes: {len(boxes_df)}")
    for k in [1, 3, 5, 10, 20]:
        hit = int(summary_df[f"top{k}_hit"].sum())
        print(f"top{k}: {hit}/{len(summary_df)} ({hit / len(summary_df) * 100:.2f}%)")
    union_hit = int(summary_df["union_hit"].sum())
    print(f"union: {union_hit}/{len(summary_df)} ({union_hit / len(summary_df) * 100:.2f}%)")
    print(f"median boxes/case: {summary_df['box_count'].median():.1f}")
    print(f"median union area m2: {summary_df['union_area_m2'].median():.1f}")
    print(f"boxes csv: {boxes_csv}")
    print(f"summary csv: {summary_csv}")
    print(f"boxes gpkg: {boxes_gpkg}")


if __name__ == "__main__":
    main()
