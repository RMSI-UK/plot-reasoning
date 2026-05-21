#!/usr/bin/env python3
"""Visualize random Mansfield WFS top20 candidate polygons."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parent
RERANK_SCRIPT = ROOT / "8_candidate_base_layer_rerank_experiment.py"
spec = importlib.util.spec_from_file_location("base_rerank", RERANK_SCRIPT)
base_rerank = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["base_rerank"] = base_rerank
spec.loader.exec_module(base_rerank)


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TOP20 = TMP / (
    "mansfield-manual-polygon-link_random1200_seed42_43_combined_"
    "base_layer_rerank_v3_compressed_top50_top20.csv"
)
DEFAULT_CASES = TMP / (
    "mansfield-manual-polygon-link_random1200_seed42_43_combined_"
    "base_layer_rerank_v3_compressed_top50_cases.csv"
)
DEFAULT_INPUT_JSON = TMP / "mansfield-manual-polygon-link_random1200_seed42_43_combined_gemini.json"
DEFAULT_V10 = TMP / "mansfield-manual-polygon-link_random1200_seed42_43_combined_ocr_openroads_v10.csv"
DEFAULT_TRUTH_GPKG = TMP / "mansfield-manual-polygon-link_random1200_seed42_43_combined.gpkg"
DEFAULT_TRUTH_LAYER = "random1200_seed42_43_combined"
DEFAULT_WFS_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg")
DEFAULT_WFS_LAYER = "mansfield_polygons_in_buffers"
DEFAULT_OUT_DIR = TMP / "base_layer_top20_random50_visuals"


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return {str(row.get("key")): row for row in rows if isinstance(row, dict)}


def point_from_xy(x: Any, y: Any) -> Point | None:
    px = parse_float(x)
    py = parse_float(y)
    if px is None or py is None:
        return None
    return Point(px, py)


def load_truth(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, columns=["unique_key", "chargegeog"])
    if gdf.crs is not None:
        gdf = gdf.to_crs(27700)
    gdf["unique_key"] = gdf["unique_key"].astype(str)
    return gdf


def load_wfs(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = base_rerank.load_wfs_polygons(path, layer, "Land|Building")
    return gdf[["candidate_id", "TOID", "GmlID", "OBJECTID", "Theme", "DescriptiveGroup", "geometry"]].copy()


def pad_bounds(bounds: tuple[float, float, float, float], pad_ratio: float = 0.12, min_span: float = 180) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, min_span)
    height = max(maxy - miny, min_span)
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    span = max(width, height)
    pad = span * pad_ratio
    half = span / 2 + pad
    return cx - half, cy - half, cx + half, cy + half


def make_contact_sheet(images: list[Path], out_path: Path, cols: int = 2, thumb_w: int = 900) -> None:
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
    for idx, (path, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width - 1, y + img.height - 1], outline=(220, 220, 220), width=2)
    sheet.save(out_path, quality=92)


def plot_case(
    key: str,
    top20: pd.DataFrame,
    cases: pd.DataFrame,
    truth_by_key: dict[str, Any],
    wfs_by_id: gpd.GeoDataFrame,
    raw_rows: dict[str, dict[str, Any]],
    v10_by_key: dict[str, pd.Series],
    out_dir: Path,
) -> Path | None:
    rows = top20[top20["case_key"] == key].copy()
    if rows.empty:
        return None
    base_key = str(rows.iloc[0]["base_key"])
    truth_geom = truth_by_key.get(base_key)
    geoms = wfs_by_id[wfs_by_id["candidate_id"].isin(rows["candidate_id"].astype(str))]
    if geoms.empty:
        return None
    plot_df = geoms.merge(rows, on="candidate_id", how="inner", suffixes=("_wfs", ""))
    plot_df = plot_df.sort_values("candidate_rank")

    case_row = cases[cases["key"].astype(str) == key]
    case_row = case_row.iloc[0] if not case_row.empty else None
    raw = raw_rows.get(key) or raw_rows.get(base_key) or {}
    v10 = v10_by_key.get(key)
    best_pt = point_from_xy(raw.get("best_easting_27700_final"), raw.get("best_northing_27700_final"))
    v7_pt = point_from_xy(v10.get("v7_selected_easting"), v10.get("v7_selected_northing")) if v10 is not None else None
    range_anchor_pt = None
    if case_row is not None:
        range_anchor_pt = point_from_xy(case_row.get("range_anchor_easting"), case_row.get("range_anchor_northing"))

    geoms_for_bounds = list(plot_df.geometry)
    if truth_geom is not None:
        geoms_for_bounds.append(truth_geom)
    if range_anchor_pt is not None:
        geoms_for_bounds.append(range_anchor_pt)
    bounds = gpd.GeoSeries(geoms_for_bounds, crs=27700).total_bounds
    bounds = pad_bounds(tuple(bounds))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_axis_off()
    ax.set_facecolor("#fbfaf7")

    # Draw lower-ranked candidates first.
    rest = plot_df[plot_df["candidate_rank"] > 1]
    if not rest.empty:
        rest.plot(ax=ax, facecolor="#5aa9e6", edgecolor="#1e6091", linewidth=0.8, alpha=0.30)
    top1 = plot_df[plot_df["candidate_rank"] == 1]
    if not top1.empty:
        top1.plot(ax=ax, facecolor="#ffb703", edgecolor="#c2410c", linewidth=2.0, alpha=0.55)

    if truth_geom is not None:
        gpd.GeoSeries([truth_geom], crs=27700).plot(ax=ax, facecolor="none", edgecolor="#d00000", linewidth=2.4)

    if best_pt is not None:
        ax.scatter([best_pt.x], [best_pt.y], marker="x", s=45, c="#7b2cbf", linewidths=2.0, zorder=8)
    if v7_pt is not None:
        ax.scatter([v7_pt.x], [v7_pt.y], marker="o", s=26, c="#111111", edgecolors="white", linewidths=0.7, zorder=9)
    if range_anchor_pt is not None:
        ax.scatter([range_anchor_pt.x], [range_anchor_pt.y], marker="*", s=88, c="#008000", edgecolors="white", linewidths=0.7, zorder=11)

    for _, row in plot_df.iterrows():
        rank = int(row["candidate_rank"])
        pt = row.geometry.representative_point()
        color = "#7f1d1d" if rank == 1 else "#073b4c"
        ax.text(
            pt.x,
            pt.y,
            str(rank),
            ha="center",
            va="center",
            fontsize=7 if rank > 1 else 9,
            color=color,
            weight="bold" if rank == 1 else "normal",
            zorder=10,
        )

    top1_hit = str(case_row.get("top1_intersects_truth")) if case_row is not None else ""
    rank = case_row.get("intersects_rank") if case_row is not None else ""
    cand_count = case_row.get("original_candidate_count") if case_row is not None else ""
    address = rows.iloc[0].get("original_address") or ""
    address = "\n".join(textwrap.wrap(str(address), width=58))
    title = f"{key}  top1_hit={top1_hit}  truth_rank={rank}  candidates={cand_count}\n{address}"
    ax.set_title(title, fontsize=9, loc="left", pad=4)
    ax.text(
        0.01,
        0.01,
        "red=truth  orange=top1  blue=top20  green=range anchor  black=v7  purple=best",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 2},
    )

    out_path = out_dir / f"case_{key}_top20.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top20-csv", type=Path, default=DEFAULT_TOP20)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--v10-csv", type=Path, default=DEFAULT_V10)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKG)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--wfs-gpkg", type=Path, default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-keys", default="", help="Comma-separated case keys to visualize instead of random sampling.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top20 = pd.read_csv(args.top20_csv, dtype={"case_key": str, "base_key": str, "candidate_id": str})
    cases = pd.read_csv(args.cases_csv, dtype={"key": str, "base_key": str})
    v10 = pd.read_csv(args.v10_csv, dtype={"key": str, "base_key": str})
    raw_rows = load_raw_rows(args.input_json)
    truth = load_truth(args.truth_gpkg, args.truth_layer)
    truth_by_key = {str(row["unique_key"]): row.geometry for _, row in truth.iterrows()}
    v10_by_key = {str(row["key"]): row for _, row in v10.iterrows()}

    print("loading WFS geometries...")
    wfs = load_wfs(args.wfs_gpkg, args.wfs_layer)
    needed_ids = set(top20["candidate_id"].astype(str))
    wfs = wfs[wfs["candidate_id"].astype(str).isin(needed_ids)].copy()
    print(f"loaded candidate geometries: {len(wfs)}")

    case_keys = sorted(top20["case_key"].dropna().astype(str).unique())
    if args.case_keys:
        wanted = [value.strip() for value in args.case_keys.split(",") if value.strip()]
        sample = [value for value in wanted if value in set(case_keys)]
    else:
        sample = pd.Series(case_keys).sample(n=min(args.sample_size, len(case_keys)), random_state=args.seed).tolist()
    image_paths: list[Path] = []
    summary_rows = []
    for key in sample:
        path = plot_case(key, top20, cases, truth_by_key, wfs, raw_rows, v10_by_key, args.out_dir)
        if path:
            image_paths.append(path)
            row = cases[cases["key"] == key].iloc[0].to_dict()
            summary_rows.append({"key": key, "image": str(path), **row})

    for idx in range(0, len(image_paths), 10):
        sheet = args.out_dir / f"contact_sheet_{idx // 10 + 1:02d}.jpg"
        make_contact_sheet(image_paths[idx : idx + 10], sheet, cols=2, thumb_w=900)

    if image_paths:
        make_contact_sheet(image_paths, args.out_dir / "contact_sheet_all_50.jpg", cols=5, thumb_w=520)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "sampled_cases.csv", index=False)
    print(f"wrote {len(image_paths)} images to {args.out_dir}")
    print(f"contact sheets: {args.out_dir / 'contact_sheet_all_50.jpg'}")


if __name__ == "__main__":
    main()
