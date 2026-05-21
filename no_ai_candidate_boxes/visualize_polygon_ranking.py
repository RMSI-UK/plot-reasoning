#!/usr/bin/env python3
"""Visualize ranked polygon candidates with local base-map context."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy_experiments"
CASCADE_SCRIPT = LEGACY / "10_cascade_polygon_selector.py"
spec = importlib.util.spec_from_file_location("cascade_selector_viz", CASCADE_SCRIPT)
cascade_selector = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["cascade_selector_viz"] = cascade_selector
spec.loader.exec_module(cascade_selector)
hybrid = cascade_selector.hybrid_rerank


DEFAULT_BASEMAP_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg")
DEFAULT_BASEMAP_LAYER = "mansfield_polygons_in_buffers_merged"

SOURCE_STYLE = {
    "council_cadastral": {"face": "#14b8a6", "edge": "#0f766e"},
    "wfs_merged": {"face": "#60a5fa", "edge": "#1d4ed8"},
    "wfs_raw": {"face": "#f59e0b", "edge": "#b45309"},
}


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def pad_bounds(bounds: tuple[float, float, float, float], pad_ratio: float, min_span: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    span = max(maxx - minx, maxy - miny, min_span)
    half = span * (0.5 + pad_ratio)
    return cx - half, cy - half, cx + half, cy + half


def load_truth(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, columns=["unique_key", "chargegeog"])
    if gdf.crs is None:
        gdf = gdf.set_crs(27700)
    else:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["unique_key"] = gdf["unique_key"].astype(str)
    return gdf


def load_basemap(path: Path, layer: str) -> gpd.GeoDataFrame:
    columns = ["TOID", "Theme", "DescriptiveGroup"]
    gdf = gpd.read_file(path, layer=layer, columns=columns)
    if gdf.crs is None:
        gdf = gdf.set_crs(27700)
    else:
        gdf = gdf.to_crs(27700)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if "Theme" in gdf:
        gdf = gdf[gdf["Theme"].fillna("").str.contains("Land|Building", case=False, regex=True)].copy()
    return gdf.reset_index(drop=True)


def make_contact_sheet(images: list[Path], out_path: Path, cols: int = 4, thumb_w: int = 560) -> None:
    if not images:
        return
    thumbs = []
    for path in images:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * ratio)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    rows = math.ceil(len(thumbs) / cols)
    thumb_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, img in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width - 1, y + img.height - 1], outline=(220, 220, 220), width=2)
    sheet.save(out_path, quality=90)


def roi_geoms_for_case(rois: pd.DataFrame | None, case_key: str, top_n: int) -> list[Any]:
    if rois is None or rois.empty:
        return []
    sub = rois[rois["case_key"].astype(str) == str(case_key)].copy()
    if sub.empty:
        return []
    sub["roi_rank"] = pd.to_numeric(sub["roi_rank"], errors="coerce")
    sub = sub[sub["roi_rank"] <= top_n].sort_values("roi_rank")
    out = []
    for _, row in sub.iterrows():
        out.append(
            box(
                float(row["roi_minx"]),
                float(row["roi_miny"]),
                float(row["roi_maxx"]),
                float(row["roi_maxy"]),
            )
        )
    return out


def split_member_ids(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [item for item in text.split("|") if item]


def geometry_ids_for_load(selected: pd.DataFrame) -> pd.DataFrame:
    wanted: set[str] = set(selected["candidate_id"].dropna().astype(str))
    if "polygon_cluster_member_ids" in selected:
        for value in selected["polygon_cluster_member_ids"]:
            wanted.update(split_member_ids(value))
    return pd.DataFrame({"candidate_id": sorted(wanted)})


def row_geometry(row: pd.Series, geom_map: dict[str, Any]) -> Any | None:
    if "polygon_cluster_member_ids" in row:
        member_geoms = [geom_map.get(member_id) for member_id in split_member_ids(row.get("polygon_cluster_member_ids"))]
        member_geoms = [geom for geom in member_geoms if geom is not None and not geom.is_empty]
        if member_geoms:
            return unary_union(member_geoms)
    return geom_map.get(str(row.get("candidate_id")))


def plot_case(
    *,
    key: str,
    rows: pd.DataFrame,
    truth_row: pd.Series,
    geom_map: dict[str, Any],
    basemap: gpd.GeoDataFrame,
    basemap_sindex: Any,
    roi_geoms: list[Any],
    out_dir: Path,
    top_k: int,
) -> tuple[Path, bool, str]:
    candidate_geoms: list[Any] = []
    candidate_rows: list[pd.Series] = []
    if not rows.empty:
        rows = rows.sort_values("polygon_rank" if "polygon_rank" in rows else "rule_rank").head(top_k)
        for _, row in rows.iterrows():
            geom = row_geometry(row, geom_map)
            if geom is None or geom.is_empty:
                continue
            candidate_geoms.append(geom)
            candidate_rows.append(row)

    truth_geom = truth_row.geometry
    extent_geoms = [truth_geom, *candidate_geoms, *roi_geoms]
    bounds = pad_bounds(tuple(gpd.GeoSeries(extent_geoms, crs=27700).total_bounds), pad_ratio=0.35, min_span=230.0)
    view = box(*bounds)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=145)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("#f8fafc")

    idx = list(basemap_sindex.query(view, predicate="intersects"))
    if idx:
        base = basemap.iloc[idx].copy()
        base.plot(ax=ax, facecolor="#e5e7eb", edgecolor="#cbd5e1", linewidth=0.35, alpha=0.72, zorder=1)

    if roi_geoms:
        roi_gdf = gpd.GeoDataFrame(geometry=roi_geoms, crs=27700)
        roi_gdf.boundary.plot(ax=ax, color="#7c3aed", linewidth=1.1, linestyle="--", alpha=0.42, zorder=4)

    if candidate_geoms:
        cand_gdf = gpd.GeoDataFrame(candidate_rows, geometry=candidate_geoms, crs=27700)
        for source, sub in cand_gdf.groupby("candidate_source", dropna=False):
            style = SOURCE_STYLE.get(str(source), {"face": "#c084fc", "edge": "#6d28d9"})
            sub.plot(ax=ax, facecolor=style["face"], edgecolor=style["edge"], linewidth=1.6, alpha=0.42, zorder=8)

        for _, row in cand_gdf.iterrows():
            pt = row.geometry.representative_point()
            rank = int(row.get("polygon_rank") or row.get("rule_rank"))
            hit = bool(row.get("truth_intersects"))
            ax.text(
                pt.x,
                pt.y,
                str(rank),
                ha="center",
                va="center",
                fontsize=11,
                weight="bold",
                color="#ffffff" if hit else "#111827",
                bbox={
                    "boxstyle": "circle,pad=0.22",
                    "facecolor": "#16a34a" if hit else "#f8fafc",
                    "edgecolor": "#111827",
                    "linewidth": 0.9,
                    "alpha": 0.94,
                },
                zorder=20,
            )

    truth_gdf = gpd.GeoDataFrame(geometry=[truth_geom], crs=27700)
    truth_gdf.plot(ax=ax, facecolor="#ef4444", edgecolor="none", alpha=0.10, zorder=12)
    truth_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=3.0, alpha=0.96, zorder=14)

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    hit_ranks: list[str] = []
    if not rows.empty and "truth_intersects" in rows:
        hit_ranks = rows.loc[bool_series(rows["truth_intersects"]), "polygon_rank"].fillna(rows.get("rule_rank")).astype(int).astype(str).tolist()
    hit_text = ",".join(hit_ranks) if hit_ranks else "none"
    address = ""
    if not rows.empty:
        address = str(rows.iloc[0].get("original_address") or "")
    if not address or address.lower() == "nan":
        address = str(truth_row.get("chargegeog") or "")
    if not address or address.lower() == "nan":
        address = "(no chargegeog / no candidate input)"
    wrapped = "\n".join(textwrap.wrap(address, width=76))
    ax.set_title(f"unique_key: {key}    top{top_k} hit rank: {hit_text}\n{wrapped}", loc="left", fontsize=10.5, pad=7)
    ax.text(
        0.01,
        0.015,
        "red = truth | numbered fills = ranked top candidates | dashed purple = top candidate boxes | grey = local WFS basemap",
        transform=ax.transAxes,
        fontsize=7.8,
        color="#111827",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.82, "pad": 2.5},
        zorder=30,
    )

    out_path = out_dir / f"{key}_top{top_k}.jpg"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08, pil_kwargs={"quality": 92})
    plt.close(fig)
    return out_path, bool(hit_ranks), hit_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize no-AI polygon ranking outputs.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--truth-gpkg", type=Path, required=True)
    parser.add_argument("--truth-layer", required=True)
    parser.add_argument("--rois-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--basemap-gpkg", type=Path, default=DEFAULT_BASEMAP_GPKG)
    parser.add_argument("--basemap-layer", default=DEFAULT_BASEMAP_LAYER)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--roi-top-n", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-random-state", type=int, default=55)
    parser.add_argument(
        "--stratified-sample",
        action="store_true",
        help="When sampling, keep roughly the same hit/miss ratio as the evaluated set.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    if "polygon_rank" not in selected:
        selected["polygon_rank"] = pd.to_numeric(selected["rule_rank"], errors="coerce")
    else:
        selected["polygon_rank"] = pd.to_numeric(selected["polygon_rank"], errors="coerce")
    if "truth_intersects" in selected:
        selected["truth_intersects"] = bool_series(selected["truth_intersects"])
    selected = selected[selected["polygon_rank"].between(1, args.top_k)].copy()

    truth = load_truth(args.truth_gpkg, args.truth_layer)
    rois = pd.read_csv(args.rois_csv, dtype={"case_key": str, "base_key": str}, low_memory=False) if args.rois_csv else None

    if args.sample_size and args.sample_size < len(truth):
        selected_by_key_for_sample = {str(key): group.copy() for key, group in selected.groupby("case_key", sort=False)}
        sample_index = pd.DataFrame({"unique_key": truth["unique_key"].astype(str)})
        sample_index["topk_intersects"] = sample_index["unique_key"].map(
            lambda key: bool(
                not selected_by_key_for_sample.get(key, selected.iloc[[]]).empty
                and bool_series(selected_by_key_for_sample[key]["truth_intersects"]).any()
                if key in selected_by_key_for_sample and "truth_intersects" in selected_by_key_for_sample[key]
                else False
            )
        )
        if args.stratified_sample:
            misses = sample_index[~sample_index["topk_intersects"]]
            hits = sample_index[sample_index["topk_intersects"]]
            miss_n = min(len(misses), max(1 if len(misses) else 0, round(args.sample_size * len(misses) / len(sample_index))))
            hit_n = args.sample_size - miss_n
            sampled = pd.concat(
                [
                    hits.sample(n=min(hit_n, len(hits)), random_state=args.sample_random_state),
                    misses.sample(n=miss_n, random_state=args.sample_random_state + 1) if miss_n else misses.iloc[[]],
                ],
                ignore_index=True,
            )
            if len(sampled) < args.sample_size:
                remainder = sample_index[~sample_index["unique_key"].isin(sampled["unique_key"])]
                sampled = pd.concat(
                    [sampled, remainder.sample(n=args.sample_size - len(sampled), random_state=args.sample_random_state + 2)],
                    ignore_index=True,
                )
        else:
            sampled = sample_index.sample(n=args.sample_size, random_state=args.sample_random_state)
        key_order = sampled["unique_key"].astype(str).tolist()
        truth = truth.set_index("unique_key").loc[key_order].reset_index()

    print("loading selected candidate geometries...", flush=True)
    geom_map = cascade_selector.load_candidate_geometries(geometry_ids_for_load(selected))
    print(f"candidate geometries loaded: {len(geom_map)}", flush=True)
    print("loading local WFS basemap...", flush=True)
    basemap = load_basemap(args.basemap_gpkg, args.basemap_layer)
    basemap_sindex = basemap.sindex
    print(f"basemap polygons loaded: {len(basemap)}", flush=True)

    image_paths: list[Path] = []
    index_rows: list[dict[str, Any]] = []
    selected_by_key = {str(key): group.copy() for key, group in selected.groupby("case_key", sort=False)}
    for i, truth_row in enumerate(truth.itertuples(), start=1):
        key = str(truth_row.unique_key)
        rows = selected_by_key.get(key, selected.iloc[[]])
        path, hit, hit_text = plot_case(
            key=key,
            rows=rows,
            truth_row=truth.iloc[i - 1],
            geom_map=geom_map,
            basemap=basemap,
            basemap_sindex=basemap_sindex,
            roi_geoms=roi_geoms_for_case(rois, key, args.roi_top_n),
            out_dir=args.out_dir,
            top_k=args.top_k,
        )
        image_paths.append(path)
        index_rows.append(
            {
                "unique_key": key,
                "has_ranked_candidates": not rows.empty,
                "topk_intersects": hit,
                "hit_ranks": hit_text,
                "image": str(path),
            }
        )
        if i % 25 == 0:
            print(f"visualized {i}/{len(truth)}", flush=True)

    index = pd.DataFrame(index_rows)
    index_path = args.out_dir / "index.csv"
    index.to_csv(index_path, index=False)

    make_contact_sheet(image_paths, args.out_dir / "contact_sheet_all_100.jpg", cols=4, thumb_w=560)
    miss_images = index[~index["topk_intersects"]]["image"].map(Path).tolist()
    make_contact_sheet(miss_images, args.out_dir / f"contact_sheet_top{args.top_k}_misses_{len(miss_images)}.jpg", cols=4, thumb_w=560)

    print(f"wrote images: {len(image_paths)}")
    print(f"index: {index_path}")
    print(f"contact sheet all: {args.out_dir / 'contact_sheet_all_100.jpg'}")
    print(f"contact sheet misses: {args.out_dir / f'contact_sheet_top{args.top_k}_misses_{len(miss_images)}.jpg'}")


if __name__ == "__main__":
    main()
