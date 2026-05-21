#!/usr/bin/env python3
"""Visualize Mansfield hybrid top5 polygon candidates against truth.

Each output image includes:
- OSM basemap;
- top5 candidate polygons from raw WFS / WFS-merged / council cadastral;
- manual truth polygon for offline evaluation;
- original address and unique key in the title.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent
CASCADE_SCRIPT = ROOT / "10_cascade_polygon_selector.py"
spec = importlib.util.spec_from_file_location("cascade_selector", CASCADE_SCRIPT)
cascade_selector = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["cascade_selector"] = cascade_selector
spec.loader.exec_module(cascade_selector)


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TAG = "mansfield-manual-polygon-link_random1200_seed42_43_combined"
DEFAULT_TOP = TMP / f"{DEFAULT_TAG}_hybrid_polygon_rerank_v1_top5roi_top50.csv"
DEFAULT_TRUTH = TMP / f"{DEFAULT_TAG}.gpkg"
DEFAULT_TRUTH_LAYER = "random1200_seed42_43_combined"
DEFAULT_OUT = ROOT / "tmp_results" / "mansfield_hybrid_top5_visuals"

T277_TO_3857 = Transformer.from_crs(27700, 3857, always_xy=True)
T3857_TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)

SOURCE_STYLE = {
    "council_cadastral": {"face": "#2dd4bf", "edge": "#0f766e"},
    "wfs_merged": {"face": "#60a5fa", "edge": "#1d4ed8"},
    "wfs_raw": {"face": "#fbbf24", "edge": "#b45309"},
}


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_3857(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    radius = 6378137.0
    origin = math.pi * radius
    tile_size = 2 * origin / (2**z)
    minx = -origin + x * tile_size
    maxx = -origin + (x + 1) * tile_size
    maxy = origin - y * tile_size
    miny = origin - (y + 1) * tile_size
    return minx, miny, maxx, maxy


def get_tile(x: int, y: int, z: int, cache_dir: Path) -> Image.Image:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{z}_{x}_{y}.png"
    if not path.exists():
        url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        req = urllib.request.Request(url, headers={"User-Agent": "MansfieldHybridTop5Visualizer/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            path.write_bytes(response.read())
    return Image.open(path).convert("RGB")


def basemap(bounds3857: tuple[float, float, float, float], out_dir: Path, z: int = 17):
    minx, miny, maxx, maxy = bounds3857
    lon_min, lat_min = T3857_TO_4326.transform(minx, miny)
    lon_max, lat_max = T3857_TO_4326.transform(maxx, maxy)
    x0, y1 = lonlat_to_tile(lon_min, lat_min, z)
    x1, y0 = lonlat_to_tile(lon_max, lat_max, z)
    xs = list(range(min(x0, x1), max(x0, x1) + 1))
    ys = list(range(min(y0, y1), max(y0, y1) + 1))
    mosaic = Image.new("RGB", (256 * len(xs), 256 * len(ys)))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            mosaic.paste(get_tile(x, y, z, out_dir / "tile_cache"), (ix * 256, iy * 256))
    ext_minx, ext_miny, _, _ = tile_bounds_3857(xs[0], ys[-1], z)
    _, _, ext_maxx, ext_maxy = tile_bounds_3857(xs[-1], ys[0], z)
    return mosaic, (ext_minx, ext_maxx, ext_miny, ext_maxy)


def pad_bounds(bounds: tuple[float, float, float, float], pad_ratio: float = 0.22, min_span: float = 170.0):
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    span = max(maxx - minx, maxy - miny, min_span)
    half = span * (0.5 + pad_ratio)
    return cx - half, cy - half, cx + half, cy + half


def to_3857_gdf(geoms: list[Any]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=geoms, crs=27700).to_crs(3857)


def load_truth(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, columns=["unique_key", "chargegeog"])
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf["unique_key"] = gdf["unique_key"].astype(str)
    return gdf


def make_contact_sheet(images: list[Path], out_path: Path, cols: int = 4, thumb_w: int = 560) -> None:
    if not images:
        return
    thumbs = []
    for path in images:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * ratio)), Image.Resampling.LANCZOS)
        thumbs.append((path, thumb))
    rows = math.ceil(len(thumbs) / cols)
    thumb_h = max(img.height for _, img in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (_, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width - 1, y + img.height - 1], outline=(220, 220, 220), width=2)
    sheet.save(out_path, quality=90)


def plot_case(
    key: str,
    rows: pd.DataFrame,
    truth_geom: Any,
    geom_map: dict[str, Any],
    out_dir: Path,
    *,
    no_basemap: bool,
    zoom: int,
) -> Path | None:
    rows = rows.sort_values("rule_rank").head(5).copy()
    geoms = []
    for _, row in rows.iterrows():
        geom = geom_map.get(str(row["candidate_id"]))
        if geom is not None and not geom.is_empty:
            geoms.append(geom)
    if truth_geom is not None and not truth_geom.is_empty:
        geoms.append(truth_geom)
    if not geoms:
        return None

    bounds27700 = pad_bounds(tuple(gpd.GeoSeries(geoms, crs=27700).total_bounds))
    bounds3857_geom = to_3857_gdf([gpd.GeoSeries([g], crs=27700).iloc[0] for g in geoms])
    bounds3857 = pad_bounds(tuple(bounds3857_geom.total_bounds), min_span=240.0)

    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=150)
    ax.set_axis_off()
    ax.set_aspect("equal")

    if not no_basemap:
        try:
            image, extent = basemap(bounds3857, out_dir, z=zoom)
            ax.imshow(image, extent=extent, origin="upper", alpha=0.92)
        except Exception as exc:
            ax.text(0.01, 0.01, f"basemap failed: {exc}", transform=ax.transAxes, fontsize=7)
            ax.set_facecolor("#f8fafc")
    else:
        ax.set_facecolor("#f8fafc")

    top_geoms = []
    top_rows = []
    for _, row in rows.iterrows():
        geom = geom_map.get(str(row["candidate_id"]))
        if geom is None or geom.is_empty:
            continue
        top_geoms.append(geom)
        top_rows.append(row)
    plot_gdf = gpd.GeoDataFrame(top_rows, geometry=top_geoms, crs=27700).to_crs(3857)

    for source, sub in plot_gdf.groupby("candidate_source", dropna=False):
        style = SOURCE_STYLE.get(str(source), {"face": "#c084fc", "edge": "#6d28d9"})
        sub.plot(ax=ax, facecolor=style["face"], edgecolor=style["edge"], linewidth=1.6, alpha=0.42)

    if truth_geom is not None and not truth_geom.is_empty:
        truth3857 = to_3857_gdf([truth_geom])
        truth3857.plot(ax=ax, facecolor="none", edgecolor="#dc2626", linewidth=3.0, alpha=0.95)
        truth3857.plot(ax=ax, facecolor="#ef4444", edgecolor="none", alpha=0.08)

    for _, row in plot_gdf.iterrows():
        pt = row.geometry.representative_point()
        rank = int(row["rule_rank"])
        hit = bool(row.get("truth_intersects"))
        ax.text(
            pt.x,
            pt.y,
            f"{rank}",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="#ffffff" if hit else "#111827",
            bbox={
                "boxstyle": "circle,pad=0.25",
                "facecolor": "#16a34a" if hit else "#f8fafc",
                "edgecolor": "#111827",
                "linewidth": 1.0,
                "alpha": 0.92,
            },
            zorder=20,
        )

    ax.set_xlim(bounds3857[0], bounds3857[2])
    ax.set_ylim(bounds3857[1], bounds3857[3])

    hit_ranks = rows.loc[rows["truth_intersects"], "rule_rank"].astype(int).tolist()
    hit_text = ",".join(map(str, hit_ranks[:5])) if hit_ranks else "none"
    address = str(rows.iloc[0].get("original_address") or "")
    title_addr = "\n".join(textwrap.wrap(address, width=82))
    title = f"unique_key: {key}    top5 truth-intersect ranks: {hit_text}\n{title_addr}"
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    ax.text(
        0.01,
        0.015,
        "red outline = truth polygon | numbered fills = top5 candidates | green number = candidate intersects truth",
        transform=ax.transAxes,
        fontsize=8,
        color="#111827",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.82, "pad": 3},
    )

    out_path = out_dir / f"{key}_top5.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize hybrid top5 candidate polygons.")
    parser.add_argument("--top-csv", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-keys", default="", help="Comma-separated unique_key values. Default: all cases.")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--no-basemap", action="store_true")
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top = pd.read_csv(args.top_csv, dtype={"case_key": str, "base_key": str})
    top["rule_rank"] = pd.to_numeric(top["rule_rank"], errors="coerce")
    top = top[top["rule_rank"].between(1, 5)].copy()
    top["truth_intersects"] = top["truth_intersects"].astype(str).str.lower().isin({"true", "1"})
    truth = load_truth(args.truth_gpkg, args.truth_layer)
    truth_by_key = {str(row["unique_key"]): row.geometry for _, row in truth.iterrows()}

    print("loading candidate geometries...")
    geom_map = cascade_selector.load_candidate_geometries(top)
    print(f"candidate geometries loaded: {len(geom_map)}")

    if args.case_keys:
        keys = [item.strip() for item in args.case_keys.split(",") if item.strip()]
    else:
        keys = list(dict.fromkeys(top["case_key"].astype(str)))
    if args.max_cases:
        keys = keys[: args.max_cases]

    image_paths: list[Path] = []
    index_rows = []
    for idx, key in enumerate(keys, start=1):
        rows = top[top["case_key"].astype(str) == key]
        if rows.empty:
            continue
        base_key = str(rows.iloc[0]["base_key"])
        path = plot_case(
            key,
            rows,
            truth_by_key.get(base_key),
            geom_map,
            args.out_dir,
            no_basemap=args.no_basemap,
            zoom=args.zoom,
        )
        if path is None:
            continue
        image_paths.append(path)
        hit_ranks = rows.loc[rows["truth_intersects"], "rule_rank"].astype(int).tolist()
        index_rows.append(
            {
                "unique_key": key,
                "base_key": base_key,
                "original_address": rows.iloc[0].get("original_address"),
                "top5_intersects": bool(hit_ranks),
                "truth_intersect_ranks": ",".join(map(str, hit_ranks)),
                "image": str(path),
            }
        )
        if idx % 50 == 0:
            print(f"visualized {idx}/{len(keys)}")

    index = pd.DataFrame(index_rows)
    index_path = args.out_dir / "index.csv"
    index.to_csv(index_path, index=False)
    if image_paths:
        sheet_paths = []
        limit = min(args.contact_sheet_limit, len(image_paths))
        if limit:
            sheet_path = args.out_dir / f"contact_sheet_first_{limit}.jpg"
            make_contact_sheet(image_paths[:limit], sheet_path)
            sheet_paths.append(sheet_path)
        misses = index[~index["top5_intersects"]]["image"].head(args.contact_sheet_limit).map(Path).tolist()
        if misses:
            miss_sheet = args.out_dir / f"contact_sheet_top5_misses_first_{len(misses)}.jpg"
            make_contact_sheet(misses, miss_sheet)
            sheet_paths.append(miss_sheet)
        for start in range(0, len(image_paths), 100):
            batch = image_paths[start : start + 100]
            make_contact_sheet(batch, args.out_dir / f"contact_sheet_batch_{start // 100 + 1:03d}.jpg")
    else:
        sheet_paths = []

    print(f"wrote images: {len(image_paths)}")
    print(f"index: {index_path}")
    for path in sheet_paths:
        print(f"contact sheet: {path}")


if __name__ == "__main__":
    main()
