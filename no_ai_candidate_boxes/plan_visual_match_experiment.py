#!/usr/bin/env python3
"""Direct plan-to-map visual matching experiment.

This is a no-API experiment.  For each candidate polygon we render a small
grayscale vector map around the candidate from local WFS polygons and OpenRoads,
then compare it to the selected plan crop using ORB feature matching.

The goal is not production quality yet; it tests whether the plan image carries
enough visual signal to rerank candidate polygons directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "tmp_results" / "mansfield_random1200_hybrid_top200_v1_top200.csv"
DEFAULT_PLAN_FEATURES = ROOT / "tmp_results" / "mansfield_plan_shape_rerank_v1_rank50_plan_features_cache_rank50.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "tmp_results" / "mansfield_plan_visual_match_v1"
DEFAULT_COUNCIL_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_councils_land.gpkg")
DEFAULT_WFS_RAW_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg")
DEFAULT_WFS_MERGED_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg")
DEFAULT_OPENROADS_GPKG = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
LOCAL_BBOX = (449000, 343000, 462500, 371000)


def geom_descriptors(geom: Any) -> dict[str, float]:
    if geom is None or geom.is_empty:
        return {}
    minx, miny, maxx, maxy = geom.bounds
    return {
        "candidate_centroid_easting": float(geom.centroid.x),
        "candidate_centroid_northing": float(geom.centroid.y),
        "candidate_geom_area": float(geom.area),
        "candidate_geom_length": float(geom.length),
        "candidate_geom_aspect": float(max((maxx - minx) / max(maxy - miny, 1e-6), (maxy - miny) / max(maxx - minx, 1e-6))),
    }


def source_ids(df: pd.DataFrame, source: str) -> list[int]:
    values = pd.to_numeric(df.loc[df["candidate_source"] == source, "candidate_objectid"], errors="coerce")
    return sorted({int(v) for v in values.dropna().tolist()})


def read_wfs_geoms(path: Path, layer: str, ids: set[int], source: str) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    print(f"reading candidate geoms: {source} ids={len(ids)}", flush=True)
    gdf = pyogrio.read_dataframe(path, layer=layer, columns=["OBJECTID", "geometry"])
    gdf = gdf[gdf["OBJECTID"].isin(ids)].copy()
    rows = []
    for _, row in gdf.iterrows():
        item = {"candidate_source": source, "candidate_objectid": int(row["OBJECTID"]), "candidate_geometry": row.geometry}
        item.update(geom_descriptors(row.geometry))
        rows.append(item)
    return pd.DataFrame(rows)


def read_council_geoms(path: Path, ids: list[int], chunk_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not ids:
        return pd.DataFrame()
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        where = "LABEL IN (" + ",".join(repr(str(v)) for v in chunk) + ")"
        gdf = pyogrio.read_dataframe(path, layer="cadastral_parcels", columns=["LABEL", "geometry"], where=where)
        for _, row in gdf.iterrows():
            item = {"candidate_source": "council_cadastral", "candidate_objectid": int(row["LABEL"]), "candidate_geometry": row.geometry}
            item.update(geom_descriptors(row.geometry))
            rows.append(item)
        print(f"council candidate geoms: {min(start + chunk_size, len(ids))}/{len(ids)}", flush=True)
    return pd.DataFrame(rows)


def load_candidate_geometries(df: pd.DataFrame, max_rank: int, council_chunk_size: int) -> pd.DataFrame:
    subset = df[pd.to_numeric(df["rule_rank"], errors="coerce") <= max_rank].copy()
    frames = [
        read_wfs_geoms(DEFAULT_WFS_RAW_GPKG, "mansfield_polygons_in_buffers", set(source_ids(subset, "wfs_raw")), "wfs_raw"),
        read_wfs_geoms(DEFAULT_WFS_MERGED_GPKG, "mansfield_polygons_in_buffers_merged", set(source_ids(subset, "wfs_merged")), "wfs_merged"),
        read_council_geoms(DEFAULT_COUNCIL_GPKG, source_ids(subset, "council_cadastral"), council_chunk_size),
    ]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = out.drop_duplicates(["candidate_source", "candidate_objectid"])
    return out


def load_context() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, STRtree, STRtree, dict[int, int], dict[int, int]]:
    print("loading local WFS/roads context", flush=True)
    wfs = pyogrio.read_dataframe(
        DEFAULT_WFS_MERGED_GPKG,
        layer="mansfield_polygons_in_buffers_merged",
        columns=["OBJECTID", "Theme", "DescriptiveGroup", "geometry"],
        bbox=LOCAL_BBOX,
    )
    roads = pyogrio.read_dataframe(
        DEFAULT_OPENROADS_GPKG,
        layer="road_link",
        columns=["geometry"],
        bbox=LOCAL_BBOX,
    )
    wfs_geoms = list(wfs.geometry)
    road_geoms = list(roads.geometry)
    wfs_tree = STRtree(wfs_geoms)
    road_tree = STRtree(road_geoms)
    # STRtree in Shapely 2 returns integer indices; keep maps for compatibility.
    wfs_index = {id(geom): i for i, geom in enumerate(wfs_geoms)}
    road_index = {id(geom): i for i, geom in enumerate(road_geoms)}
    print(f"context loaded: wfs={len(wfs)} roads={len(roads)}", flush=True)
    return wfs, roads, wfs_tree, road_tree, wfs_index, road_index


def transform_point(x: float, y: float, bounds: tuple[float, float, float, float], size: int) -> tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    px = (x - minx) / max(maxx - minx, 1e-6) * (size - 1)
    py = (maxy - y) / max(maxy - miny, 1e-6) * (size - 1)
    return px, py


def draw_lines(draw: ImageDraw.ImageDraw, geom: Any, bounds: tuple[float, float, float, float], size: int, fill: int, width: int) -> None:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        pts = [transform_point(x, y, bounds, size) for x, y in geom.coords]
        if len(pts) >= 2:
            draw.line(pts, fill=fill, width=width)
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            draw_lines(draw, part, bounds, size, fill, width)


def draw_polygon_outline(draw: ImageDraw.ImageDraw, geom: Any, bounds: tuple[float, float, float, float], size: int, fill: int, width: int) -> None:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        pts = [transform_point(x, y, bounds, size) for x, y in geom.exterior.coords]
        if len(pts) >= 2:
            draw.line(pts, fill=fill, width=width, joint="curve")
        for interior in geom.interiors:
            ipts = [transform_point(x, y, bounds, size) for x, y in interior.coords]
            if len(ipts) >= 2:
                draw.line(ipts, fill=fill, width=max(1, width - 1))
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            draw_polygon_outline(draw, part, bounds, size, fill, width)


def query_tree(tree: STRtree, geoms: list[Any], window: Any) -> list[Any]:
    hits = tree.query(window)
    out = []
    for hit in hits:
        if isinstance(hit, (int, np.integer)):
            geom = geoms[int(hit)]
        else:
            geom = hit
        if geom.intersects(window):
            out.append(geom)
    return out


def render_candidate_map(
    candidate_geom: Any,
    wfs_geoms: list[Any],
    road_geoms: list[Any],
    wfs_tree: STRtree,
    road_tree: STRtree,
    side_m: float,
    size: int,
) -> np.ndarray:
    cx = float(candidate_geom.centroid.x)
    cy = float(candidate_geom.centroid.y)
    half = side_m / 2.0
    bounds = (cx - half, cy - half, cx + half, cy + half)
    window = box(*bounds)
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    # Light parcel/building outlines, then roads, then candidate boundary.
    for geom in query_tree(wfs_tree, wfs_geoms, window)[:350]:
        draw_polygon_outline(draw, geom.intersection(window), bounds, size, fill=190, width=1)
    for geom in query_tree(road_tree, road_geoms, window)[:200]:
        draw_lines(draw, geom.intersection(window), bounds, size, fill=80, width=2)
    draw_polygon_outline(draw, candidate_geom.intersection(window), bounds, size, fill=0, width=4)
    return np.asarray(image)


def preprocess_plan(path: str, size: int) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    with Image.open(path) as image:
        image = image.convert("L")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (size, size), 255)
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        arr = np.asarray(canvas)
    # CLAHE helps weak scanned linework without using any learned model.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def orb_descriptors(image: np.ndarray, max_features: int) -> tuple[list[Any], Any]:
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=8, edgeThreshold=15)
    return orb.detectAndCompute(image, None)


def match_score(plan_desc: Any, map_desc: Any, plan_kp_count: int, map_kp_count: int) -> tuple[float, int]:
    if plan_desc is None or map_desc is None or plan_kp_count == 0 or map_kp_count == 0:
        return 0.0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(plan_desc, map_desc, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)
    denom = max(1, min(plan_kp_count, map_kp_count))
    return float(len(good) / denom), len(good)


def metric(df: pd.DataFrame, rank_col: str, ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": int(df["case_key"].nunique())}
    for k in ks:
        hit_cases = df[(pd.to_numeric(df[rank_col], errors="coerce") <= k) & df["truth_intersects_bool"]].groupby("case_key").size()
        hits = int((hit_cases > 0).sum())
        out[f"top{k}"] = hits
        out[f"top{k}_rate"] = hits / out["cases"] if out["cases"] else 0.0
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_prefix: Path = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    df["rule_rank"] = pd.to_numeric(df["rule_rank"], errors="coerce")
    df = df[df["rule_rank"] <= args.max_rank].copy()
    df["candidate_objectid"] = pd.to_numeric(df["candidate_objectid"], errors="coerce")
    df["truth_intersects_bool"] = df["truth_intersects"].astype(str).str.lower().isin({"true", "1", "yes"})

    plan = pd.read_csv(args.plan_features_csv, dtype={"case_key": str, "base_key": str})
    plan = plan[["case_key", "plan_available", "plan_best_crop", "mfd_ref", "plan_crop_count"]].copy()
    df = df.merge(plan, on="case_key", how="left")
    df["plan_available"] = df["plan_available"].fillna(False).astype(bool)
    df = df[df["plan_available"]].copy()

    cases = sorted(df["case_key"].unique())
    if args.sample_cases and args.sample_cases < len(cases):
        rng = np.random.default_rng(args.seed)
        cases = sorted(rng.choice(cases, size=args.sample_cases, replace=False).tolist())
        df = df[df["case_key"].isin(cases)].copy()
    if args.focus_rule_top5_misses:
        miss_cases = []
        for case_key, group in df.groupby("case_key"):
            if not bool(group[(group["rule_rank"] <= 5) & group["truth_intersects_bool"]].shape[0]):
                miss_cases.append(case_key)
        df = df[df["case_key"].isin(miss_cases)].copy()

    geoms = load_candidate_geometries(df, args.max_rank, args.council_chunk_size)
    df = df.merge(geoms, on=["candidate_source", "candidate_objectid"], how="left")
    df = df[~df["candidate_geometry"].isna()].copy()
    wfs, roads, wfs_tree, road_tree, _, _ = load_context()
    wfs_geoms = list(wfs.geometry)
    road_geoms = list(roads.geometry)

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    plan_cache: dict[str, tuple[int, Any, int]] = {}
    rows: list[dict[str, Any]] = []
    for i, (case_key, group) in enumerate(df.groupby("case_key"), start=1):
        plan_path = str(group["plan_best_crop"].iloc[0] or "")
        if plan_path not in plan_cache:
            plan_image = preprocess_plan(plan_path, args.image_size)
            if plan_image is None:
                plan_cache[plan_path] = (0, None, 0)
            else:
                kp, desc = orb_descriptors(plan_image, args.max_features)
                plan_cache[plan_path] = (len(kp), desc, int(plan_image.mean()))
        plan_kp_count, plan_desc, _ = plan_cache[plan_path]
        for _, row in group.iterrows():
            best_score = 0.0
            best_good = 0
            best_scale = 0.0
            geom = row["candidate_geometry"]
            for scale in scales:
                rendered = render_candidate_map(geom, wfs_geoms, road_geoms, wfs_tree, road_tree, scale, args.image_size)
                kp, desc = orb_descriptors(rendered, args.max_features)
                score, good = match_score(plan_desc, desc, plan_kp_count, len(kp))
                if score > best_score:
                    best_score = score
                    best_good = good
                    best_scale = scale
            out = row.drop(labels=["candidate_geometry"]).to_dict()
            out["visual_match_score"] = best_score
            out["visual_match_good"] = best_good
            out["visual_match_scale"] = best_scale
            out["plan_keypoints"] = plan_kp_count
            rows.append(out)
        if i % 25 == 0:
            print(f"visual match: {i}/{df['case_key'].nunique()} cases", flush=True)

    scored = pd.DataFrame(rows)
    scored["visual_rank"] = (
        scored.sort_values(["case_key", "visual_match_score", "rule_rank"], ascending=[True, False, True])
        .groupby("case_key")
        .cumcount()
        .add(1)
        .reindex(scored.index)
    )
    # A conservative hybrid: keep rule top4, then add the strongest visual
    # alternate.  This tests whether visual evidence can rescue misses without
    # throwing away the mature rule stack.
    hybrid_rows = []
    for case_key, group in scored.groupby("case_key"):
        chosen = []
        for idx in group.sort_values("rule_rank").head(4).index:
            chosen.append(idx)
        for idx in group.sort_values(["visual_match_score", "rule_rank"], ascending=[False, True]).index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= 5:
                break
        for rank, idx in enumerate(chosen, start=1):
            hybrid_rows.append((idx, rank))
    hybrid_rank = pd.Series(np.nan, index=scored.index)
    for idx, rank in hybrid_rows:
        hybrid_rank.loc[idx] = rank
    scored["rule4_visual1_rank"] = hybrid_rank

    ks = [1, 3, 5, 8, 10, args.max_rank]
    ks = sorted(set(k for k in ks if k <= args.max_rank))
    summary = {
        "input_csv": str(args.input_csv),
        "output_prefix": str(out_prefix),
        "max_rank": args.max_rank,
        "sample_cases": args.sample_cases,
        "focus_rule_top5_misses": args.focus_rule_top5_misses,
        "cases": int(scored["case_key"].nunique()),
        "rows": int(len(scored)),
        "rule": metric(scored, "rule_rank", ks),
        "visual_score": metric(scored, "visual_rank", ks),
        "rule4_visual1": metric(scored[~scored["rule4_visual1_rank"].isna()].copy(), "rule4_visual1_rank", [1, 3, 5]),
        "mean_plan_keypoints": float(scored.groupby("case_key")["plan_keypoints"].first().mean()) if not scored.empty else 0.0,
        "mean_visual_good_matches": float(scored["visual_match_good"].mean()) if not scored.empty else 0.0,
    }
    summary_json = out_prefix.with_suffix(".summary.json")
    scored_csv = out_prefix.with_name(out_prefix.name + "_scored.csv")
    top_csv = out_prefix.with_name(out_prefix.name + "_top.csv")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    keep_cols = [c for c in scored.columns if c != "candidate_geometry"]
    scored[keep_cols].to_csv(scored_csv, index=False)
    scored.sort_values(["case_key", "visual_rank"]).groupby("case_key").head(args.output_top_n)[keep_cols].to_csv(top_csv, index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"summary_json={summary_json}", flush=True)
    print(f"scored_csv={scored_csv}", flush=True)
    print(f"top_csv={top_csv}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct plan-to-map visual matching experiment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plan-features-csv", type=Path, default=DEFAULT_PLAN_FEATURES)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--sample-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--scales", default="160,260,420")
    parser.add_argument("--output-top-n", type=int, default=20)
    parser.add_argument("--council-chunk-size", type=int, default=500)
    parser.add_argument("--focus-rule-top5-misses", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
