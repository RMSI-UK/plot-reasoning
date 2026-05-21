#!/usr/bin/env python3
"""Experiment: use OS Open Roads line geometry for OCR relation candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox, parse_names


ROOT = Path(__file__).resolve().parent
OPENNAME_SCRIPT = ROOT / "openname_anchor.py"
spec = importlib.util.spec_from_file_location("ocr_openname", OPENNAME_SCRIPT)
ocr_openname = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["ocr_openname"] = ocr_openname
spec.loader.exec_module(ocr_openname)


DEFAULT_PREV_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_openname_v6.csv"
)
DEFAULT_INPUT_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42.gpkg"
)
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_openroads_v7"
)

MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)
LOW_TRUST_SOURCES = {"lexicon", "fallback", "none", "parent_consensus"}
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
}
ROAD_LEADING_STOP = {
    "LAND",
    "BETWEEN",
    "AND",
    "AT",
    "OF",
    "TO",
    "LOCATION",
    "OFF",
    "REAR",
    "ADJACENT",
    "ADJ",
    "SITE",
    "ON",
    "THE",
}


def norm(text: Any) -> str:
    return ocr_openname.norm(text)


def parse_float(value: Any) -> float | None:
    return ocr_openname.parse_float(value)


def in_bbox(easting: float, northing: float) -> bool:
    minx, miny, maxx, maxy = MANSFIELD_BBOX
    return minx <= easting <= maxx and miny <= northing <= maxy


def load_truth(gpkg_path: Path, layer: str) -> dict[str, Any]:
    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    return {str(row["unique_key"]): row.geometry for _, row in gdf.iterrows()}


def distance_to_geom(easting: Any, northing: Any, geom: Any) -> float | None:
    e = parse_float(easting)
    n = parse_float(northing)
    if e is None or n is None:
        return None
    return Point(e, n).distance(geom)


@dataclass
class RoadCandidate:
    source: str
    name: str
    easting: float
    northing: float
    score: float
    reason: str


def load_roads(open_roads: Path) -> dict[str, Any]:
    gdf = gpd.read_file(
        open_roads,
        layer="road_link",
        bbox=MANSFIELD_BBOX,
        columns=["name_1", "name_2", "length"],
    )
    groups: dict[str, list[Any]] = {}
    for _, row in gdf.iterrows():
        for col in ("name_1", "name_2"):
            name = row.get(col)
            if isinstance(name, str) and name.strip():
                groups.setdefault(norm(name), []).append(row.geometry)
    return {name: unary_union(geoms) for name, geoms in groups.items()}


def load_specific_places(con: sqlite3.Connection, known_places: list[str]) -> dict[str, Point]:
    out: dict[str, Point] = {}
    for place in known_places:
        if place in ocr_openname.GENERIC_PLACES:
            continue
        rows = con.execute(
            """
            SELECT NAME1, GEOMETRY_X, GEOMETRY_Y
            FROM open_names
            WHERE NAME1 = ? COLLATE NOCASE
              AND TYPE = 'populatedPlace'
              AND CAST(GEOMETRY_X AS REAL) BETWEEN ? AND ?
              AND CAST(GEOMETRY_Y AS REAL) BETWEEN ? AND ?
            LIMIT 1
            """,
            (place, MANSFIELD_BBOX[0], MANSFIELD_BBOX[2], MANSFIELD_BBOX[1], MANSFIELD_BBOX[3]),
        ).fetchall()
        if rows:
            _, x, y = rows[0]
            x = parse_float(x)
            y = parse_float(y)
            if x is not None and y is not None:
                out[place] = Point(x, y)
    return out


def nearest_on_road(road_geom: Any, anchor: Point) -> Point:
    return nearest_points(anchor, road_geom)[1]


def context(prev_row: pd.Series, known_places: list[str]) -> tuple[list[str], list[str], set[str]]:
    text = " | ".join(
        [
            str(prev_row.get("original_address") or ""),
            str(prev_row.get("ocr_roads") or ""),
            str(prev_row.get("ocr_location_lines") or ""),
        ]
    )
    roads = []
    for road in ocr_openname.extract_road_phrases(text):
        nroad = norm(road)
        if nroad not in roads:
            roads.append(nroad)
    places = ocr_openname.extract_localities(text, known_places)
    relations = {r for r in str(prev_row.get("ocr_relations") or "").split(",") if r}
    return roads, places, relations


def road_name_variants(phrase: str, road_geoms: dict[str, Any]) -> list[str]:
    phrase = norm(phrase)
    tokens = phrase.split()
    variants = []

    for idx, token in enumerate(tokens):
        if token not in ROAD_SUFFIXES:
            continue
        for start in range(max(0, idx - 4), idx):
            candidate_tokens = tokens[start : idx + 1]
            while candidate_tokens and candidate_tokens[0] in ROAD_LEADING_STOP:
                candidate_tokens = candidate_tokens[1:]
            if len(candidate_tokens) >= 2:
                variants.append(" ".join(candidate_tokens))
        if idx >= 1:
            variants.append(" ".join(tokens[idx - 1 : idx + 1]))
        if idx >= 2:
            variants.append(" ".join(tokens[idx - 2 : idx + 1]))
    out = []
    for item in variants:
        if item in road_geoms and item not in out:
            out.append(item)
    return out


def matched_road_names(raw_roads: list[str], road_geoms: dict[str, Any]) -> list[str]:
    out = []
    for road in raw_roads:
        for candidate in [road, *road_name_variants(road, road_geoms)]:
            if candidate in road_geoms and candidate not in out:
                out.append(candidate)
    return out


def build_candidates(
    prev_row: pd.Series,
    road_geoms: dict[str, Any],
    place_points: dict[str, Point],
    known_places: list[str],
) -> list[RoadCandidate]:
    raw_roads, places, relations = context(prev_row, known_places)
    roads = matched_road_names(raw_roads, road_geoms)
    candidates: list[RoadCandidate] = []

    # Road snapped to a specific locality, e.g. "Mansfield Road, Spion Kop".
    for road in roads:
        geom = road_geoms.get(road)
        if geom is None:
            continue
        for place in places:
            anchor = place_points.get(place)
            if anchor is None:
                continue
            pt = nearest_on_road(geom, anchor)
            dist = pt.distance(anchor)
            if dist <= 1500:
                candidates.append(
                    RoadCandidate(
                        "openroads_road_near_place",
                        f"{road} near {place}",
                        pt.x,
                        pt.y,
                        40 - min(dist / 100, 15),
                        f"road_near_place;road={road};place={place};dist={dist:.1f}",
                    )
                )

    # Junction/nearest relation for explicit two-road contexts.
    if relations & {"junction", "between"} and len(roads) >= 2:
        for i, road_a in enumerate(roads[:6]):
            geom_a = road_geoms.get(road_a)
            if geom_a is None:
                continue
            for road_b in roads[i + 1 : 7]:
                geom_b = road_geoms.get(road_b)
                if geom_b is None:
                    continue
                pa, pb = nearest_points(geom_a, geom_b)
                gap = pa.distance(pb)
                if gap <= 800:
                    mid = Point((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
                    src = "openroads_junction" if "junction" in relations and gap <= 80 else "openroads_between"
                    candidates.append(
                        RoadCandidate(
                            src,
                            f"{road_a} / {road_b}",
                            mid.x,
                            mid.y,
                            35 - min(gap / 50, 15),
                            f"{src};a={road_a};b={road_b};gap={gap:.1f}",
                        )
                    )

    # For low-trust road-only cases, road centroid/representative point can be
    # a better point than a same-name road in the wrong locality.  This is kept
    # low score and only accepted by a strict gate.
    if len(roads) == 1:
        geom = road_geoms.get(roads[0])
        if geom is not None:
            pt = geom.representative_point()
            candidates.append(
                RoadCandidate(
                    "openroads_single_road",
                    roads[0],
                    pt.x,
                    pt.y,
                    18,
                    f"single_road_representative;road={roads[0]}",
                )
            )

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def should_accept(prev_row: pd.Series, cand: RoadCandidate | None, known_places: list[str]) -> tuple[bool, str]:
    if cand is None:
        return False, "no_openroads_candidate"
    if str(prev_row.get("v3_selected_source") or "") == "ocr_geocode":
        return False, "keep_ocr_geocode"
    if str(prev_row.get("v3_selected_source") or "") != "baseline":
        return False, "keep_prior_non_baseline_refinement"

    baseline_source = str(prev_row.get("baseline_source") or "").lower()
    selected_source = str(prev_row.get("v3_selected_source") or "")
    selected_e = parse_float(prev_row.get("v3_selected_easting"))
    selected_n = parse_float(prev_row.get("v3_selected_northing"))
    selected_in_bbox = selected_e is not None and selected_n is not None and in_bbox(selected_e, selected_n)
    road_names, places, relations = context(prev_row, known_places)
    selected_dist_to_candidate = (
        math.hypot(cand.easting - selected_e, cand.northing - selected_n)
        if selected_e is not None and selected_n is not None
        else None
    )

    if not selected_in_bbox and cand.score >= 25:
        return True, "selected_outside_bbox"
    if baseline_source not in LOW_TRUST_SOURCES and selected_source == "baseline":
        return False, "keep_high_trust_baseline"
    if cand.source == "openroads_single_road":
        return False, "single_road_not_precise_enough"
    if cand.source == "openroads_road_near_place" and baseline_source in LOW_TRUST_SOURCES and cand.score >= 25:
        return True, "low_trust_road_near_specific_place"
    if cand.source in {"openroads_junction", "openroads_between"}:
        return False, "relation_roads_need_stronger_gate"
    if (
        selected_dist_to_candidate is not None
        and selected_dist_to_candidate <= 80
        and baseline_source in LOW_TRUST_SOURCES
        and cand.score >= 20
    ):
        return True, "near_existing_low_trust_refinement"
    return False, "gate_keep_previous"


def run(args: argparse.Namespace) -> None:
    prev = pd.read_csv(args.prev_csv)
    truth = load_truth(Path(args.input_gpkg), args.layer)
    con = sqlite3.connect(args.open_names)
    known_places = ocr_openname.load_local_place_names(con)
    place_points = load_specific_places(con, known_places)
    con.close()
    road_geoms = load_roads(Path(args.open_roads))

    rows = []
    for _, row in prev.iterrows():
        key = str(row["key"])
        geom = truth.get(key.split("_", 1)[0])
        candidates = build_candidates(row, road_geoms, place_points, known_places)
        best = candidates[0] if candidates else None
        accept, gate = should_accept(row, best, known_places)
        old_e = parse_float(row.get("v3_selected_easting"))
        old_n = parse_float(row.get("v3_selected_northing"))
        new_e = best.easting if best and accept else old_e
        new_n = best.northing if best and accept else old_n
        old_dist = distance_to_geom(old_e, old_n, geom) if geom is not None else None
        new_dist = distance_to_geom(new_e, new_n, geom) if geom is not None else None
        delta = old_dist - new_dist if old_dist is not None and new_dist is not None else None
        rows.append(
            {
                **row.to_dict(),
                "v7_selected_source": best.source if best and accept else row.get("v3_selected_source"),
                "v7_selected_address": best.name if best and accept else row.get("v3_selected_address"),
                "v7_selected_easting": new_e,
                "v7_selected_northing": new_n,
                "v7_selected_distance_m": new_dist,
                "v7_delta_vs_v6_m": delta,
                "v7_gate": gate,
                "v7_best_source": best.source if best else None,
                "v7_best_name": best.name if best else None,
                "v7_best_easting": best.easting if best else None,
                "v7_best_northing": best.northing if best else None,
                "v7_best_score": best.score if best else None,
                "v7_best_reason": best.reason if best else None,
                "v7_candidate_count": len(candidates),
            }
        )

    df = pd.DataFrame(rows)
    summary = summarize(df)
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    df.to_excel(out.with_suffix(".xlsx"), index=False)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out.with_suffix('.csv')}")
    print(f"Wrote {out.with_suffix('.xlsx')}")
    print(f"Wrote {out.with_suffix('.json')}")


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    comp = df[df["v3_selected_distance_m"].notna() & df["v7_selected_distance_m"].notna()].copy()
    improved = comp[comp["v7_delta_vs_v6_m"] > 1]
    worse = comp[comp["v7_delta_vs_v6_m"] < -1]
    same = comp[comp["v7_delta_vs_v6_m"].abs() <= 1]

    def mean(col: str) -> float | None:
        vals = pd.to_numeric(comp[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else None

    return {
        "rows": int(len(df)),
        "comparable_rows": int(len(comp)),
        "improved_vs_v6": int(len(improved)),
        "worse_vs_v6": int(len(worse)),
        "same_vs_v6": int(len(same)),
        "v6_mean_distance_m": mean("v3_selected_distance_m"),
        "v7_mean_distance_m": mean("v7_selected_distance_m"),
        "mean_delta_vs_v6_m": mean("v7_delta_vs_v6_m"),
        "v6_over_100m": int((comp["v3_selected_distance_m"] > 100).sum()),
        "v7_over_100m": int((comp["v7_selected_distance_m"] > 100).sum()),
        "v6_over_500m": int((comp["v3_selected_distance_m"] > 500).sum()),
        "v7_over_500m": int((comp["v7_selected_distance_m"] > 500).sum()),
        "selected_source_counts": {
            str(k): int(v) for k, v in df["v7_selected_source"].value_counts(dropna=False).to_dict().items()
        },
        "top_improvements": [
            {
                "key": str(r["key"]),
                "delta_m": float(r["v7_delta_vs_v6_m"]),
                "v6_m": float(r["v3_selected_distance_m"]),
                "v7_m": float(r["v7_selected_distance_m"]),
                "source": r["v7_selected_source"],
                "name": r["v7_selected_address"],
                "address": r["original_address"],
            }
            for _, r in improved.sort_values("v7_delta_vs_v6_m", ascending=False).head(15).iterrows()
        ],
        "top_regressions": [
            {
                "key": str(r["key"]),
                "delta_m": float(r["v7_delta_vs_v6_m"]),
                "v6_m": float(r["v3_selected_distance_m"]),
                "v7_m": float(r["v7_selected_distance_m"]),
                "source": r["v7_selected_source"],
                "name": r["v7_selected_address"],
                "address": r["original_address"],
            }
            for _, r in worse.sort_values("v7_delta_vs_v6_m").head(15).iterrows()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prev-csv", default=str(DEFAULT_PREV_CSV))
    parser.add_argument("--input-gpkg", default=str(DEFAULT_INPUT_GPKG))
    parser.add_argument("--open-names", default=str(DEFAULT_OPEN_NAMES))
    parser.add_argument("--open-roads", default=str(DEFAULT_OPEN_ROADS))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--layer", default="mansfield-manual-polygon-link-random200")
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--generic-place-names", default="")
    args = parser.parse_args()
    global MANSFIELD_BBOX
    MANSFIELD_BBOX = parse_bbox(args.local_bbox, MANSFIELD_BBOX)
    ocr_openname.MANSFIELD_BBOX = MANSFIELD_BBOX
    if args.generic_place_names:
        ocr_openname.GENERIC_PLACES = set(parse_names(args.generic_place_names, ocr_openname.GENERIC_PLACES))
    return args


if __name__ == "__main__":
    run(parse_args())
