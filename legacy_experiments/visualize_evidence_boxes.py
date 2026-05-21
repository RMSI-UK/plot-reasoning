#!/usr/bin/env python3
"""Visualize Monmouthshire evidence-to-box candidate outputs."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from shapely.ops import unary_union


TMP = Path("/data/monmouthshire/spatial/base-map/tmp_output")
DEFAULT_PREFIX = TMP / "monmouthshire_evidence_boxes_random200_seed42_v1_no_current_boxes"
DEFAULT_INPUT_GPKG = TMP / "monmouthshire_input_layer_random200_seed42_20260505.gpkg"
DEFAULT_INPUT_LAYER = "features"
DEFAULT_TRUTH_GPKK = TMP / "monmouthshire_capture_random200_seed42_merge_direct_20260505.gpkg"
DEFAULT_TRUTH_LAYER = "capture_result"
DEFAULT_OUT_DIR = TMP / "monmouthshire_evidence_boxes_random200_seed42_v1_no_current_visuals"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def safe_filename(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return cleaned or "case"


def pad_bounds(bounds: tuple[float, float, float, float], pad_ratio: float = 0.14, min_span: float = 220.0) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, min_span)
    height = max(maxy - miny, min_span)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    span = max(width, height)
    half = span * (0.5 + pad_ratio)
    return cx - half, cy - half, cx + half, cy + half


def make_contact_sheet(images: list[Path], out_path: Path, cols: int = 3, thumb_w: int = 760) -> None:
    if not images:
        return
    thumbs: list[Image.Image] = []
    for path in images:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height * ratio)), Image.Resampling.LANCZOS))
    thumb_h = max(img.height for img in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, img in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width - 1, y + img.height - 1], outline=(215, 215, 215), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def make_contact_sheets(
    images: list[Path],
    out_base: Path,
    cols: int = 3,
    thumb_w: int = 760,
    page_size: int = 30,
) -> list[Path]:
    if not images:
        return []
    if page_size <= 0 or len(images) <= page_size:
        out_path = out_base.with_suffix(".jpg")
        make_contact_sheet(images, out_path, cols=cols, thumb_w=thumb_w)
        return [out_path]

    pages: list[Path] = []
    stem = out_base.name
    for start in range(0, len(images), page_size):
        page_no = start // page_size + 1
        chunk = images[start : start + page_size]
        out_path = out_base.with_name(f"{stem}_p{page_no:02d}").with_suffix(".jpg")
        make_contact_sheet(chunk, out_path, cols=cols, thumb_w=thumb_w)
        pages.append(out_path)
    return pages


def load_truth(path: Path, layer: str, target_len: int) -> gpd.GeoDataFrame:
    truth = gpd.read_file(path, layer=layer)
    if truth.crs is None:
        truth = truth.set_crs(27700)
    else:
        truth = truth.to_crs(27700)
    if len(truth) != target_len:
        raise SystemExit(f"truth rows={len(truth)} does not match input rows={target_len}")
    return truth.reset_index(drop=True)


def draw_case(
    row_index: int,
    summary_row: pd.Series,
    boxes: gpd.GeoDataFrame,
    input_gdf: gpd.GeoDataFrame,
    truth_gdf: gpd.GeoDataFrame,
    out_dir: Path,
    max_rank: int,
    zoom: str,
    near_radius_m: float,
) -> Path:
    case_key = str(summary_row["case_key"])
    case_boxes = boxes[(boxes["case_key"].astype(str) == case_key) & (boxes["rank"] <= max_rank)].copy()
    if case_boxes.empty:
        raise ValueError(f"No boxes for case {case_key}")
    case_boxes = case_boxes.sort_values("rank")
    input_geom = input_gdf.geometry.iloc[row_index - 1]
    truth_geom = truth_gdf.geometry.iloc[row_index - 1]

    case_boxes["truth_distance_m"] = case_boxes.geometry.distance(truth_geom)
    hit_boxes = case_boxes[case_boxes.geometry.intersects(truth_geom)]
    first_hit_rank = int(hit_boxes["rank"].min()) if not hit_boxes.empty else None
    top1_distance = float(case_boxes.loc[case_boxes["rank"] == 1, "truth_distance_m"].iloc[0]) if (case_boxes["rank"] == 1).any() else float("nan")

    visible_boxes = case_boxes
    if zoom == "truth":
        visible_boxes = case_boxes[case_boxes["truth_distance_m"] <= near_radius_m].copy()
        if visible_boxes.empty:
            visible_boxes = case_boxes.nsmallest(min(8, len(case_boxes)), "truth_distance_m").copy()

    bound_geoms = list(visible_boxes.geometry)
    if input_geom is not None and not input_geom.is_empty and (zoom != "truth" or input_geom.distance(truth_geom) <= near_radius_m):
        bound_geoms.append(input_geom)
    if truth_geom is not None and not truth_geom.is_empty:
        bound_geoms.append(truth_geom)
    bounds = pad_bounds(tuple(gpd.GeoSeries(bound_geoms, crs=27700).total_bounds))

    fig, ax = plt.subplots(figsize=(8.8, 8.8), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_axis_off()
    ax.set_facecolor("#fbfaf7")

    rank_groups = [
        (visible_boxes[visible_boxes["rank"] > 10], "#a8a29e", "#57534e", 0.14, 0.7),
        (visible_boxes[(visible_boxes["rank"] >= 4) & (visible_boxes["rank"] <= 10)], "#c084fc", "#7e22ce", 0.18, 0.9),
        (visible_boxes[(visible_boxes["rank"] >= 2) & (visible_boxes["rank"] <= 3)], "#86efac", "#15803d", 0.22, 1.3),
        (visible_boxes[visible_boxes["rank"] == 1], "#fbbf24", "#b45309", 0.38, 2.2),
    ]
    for subset, face, edge, alpha, linewidth in rank_groups:
        if not subset.empty:
            subset.plot(ax=ax, facecolor=face, edgecolor=edge, linewidth=linewidth, alpha=alpha, zorder=2)

    if truth_geom is not None and not truth_geom.is_empty:
        gpd.GeoSeries([truth_geom], crs=27700).plot(ax=ax, facecolor="none", edgecolor="#dc2626", linewidth=2.8, zorder=6)
    if input_geom is not None and not input_geom.is_empty:
        if input_geom.geom_type in {"Point", "MultiPoint"}:
            ax.scatter([input_geom.x], [input_geom.y], marker="x", s=54, c="#2563eb", linewidths=2.4, zorder=7)
        else:
            gpd.GeoSeries([input_geom], crs=27700).plot(ax=ax, facecolor="none", edgecolor="#2563eb", linewidth=1.8, linestyle="--", zorder=5)

    for _, box_row in visible_boxes.iterrows():
        rank = int(box_row["rank"])
        geom = box_row.geometry
        pt = geom.representative_point()
        color = "#7c2d12" if rank == 1 else "#1f2937"
        size = 9 if rank <= 3 else 7
        ax.text(pt.x, pt.y, str(rank), ha="center", va="center", fontsize=size, color=color, weight="bold", zorder=8)

    address = clean_text(summary_row.get("original_address"))
    address_lines = "\n".join(textwrap.wrap(address, width=70))
    title = (
        f"row={row_index} key={case_key} "
        f"top1={summary_row.get('top1_hit')} top3={summary_row.get('top3_hit')} "
        f"top10={summary_row.get('top10_hit')} boxes={summary_row.get('box_count')} "
        f"first_hit={first_hit_rank or '-'} top1_dist={top1_distance:.0f}m\n"
        f"{address_lines}"
    )
    ax.set_title(title, fontsize=9.5, loc="left", pad=5)
    ax.text(
        0.01,
        0.01,
        "red=truth proxy  blue=input geometry  orange=top1  green=top2-3  purple=top4-10  gray=top11-20",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#222",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2},
    )

    suffix = f"top{max_rank}" if zoom == "all" else f"top{max_rank}_truthzoom"
    out_path = out_dir / f"row_{row_index:03d}_{safe_filename(case_key)}_{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize evidence-to-box candidate outputs.")
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--boxes-gpkg", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--input-gpkg", type=Path, default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--input-layer", default=DEFAULT_INPUT_LAYER)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKK)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--zoom", choices=["all", "truth"], default="all")
    parser.add_argument("--near-radius-m", type=float, default=450.0)
    parser.add_argument("--case-rows", help="Comma-separated input row indices to render.")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--render-top1-misses", action="store_true")
    parser.add_argument("--render-top5-misses", action="store_true")
    parser.add_argument("--random-hits", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-page-size", type=int, default=30)
    args = parser.parse_args()

    boxes_path = args.boxes_gpkg or args.prefix.with_name(args.prefix.name + "_boxes.gpkg")
    summary_path = args.summary_csv or args.prefix.with_name(args.prefix.name + "_case_summary.csv")
    boxes = gpd.read_file(boxes_path, layer="candidate_boxes")
    boxes = boxes.to_crs(27700) if boxes.crs is not None else boxes.set_crs(27700)
    summary = pd.read_csv(summary_path)
    input_gdf = gpd.read_file(args.input_gpkg, layer=args.input_layer)
    input_gdf = input_gdf.to_crs(27700) if input_gdf.crs is not None else input_gdf.set_crs(27700)
    truth_gdf = load_truth(args.truth_gpkg, args.truth_layer, len(input_gdf))

    rows: list[int] = []
    if args.all_cases:
        rows.extend(summary["row_index"].astype(int).tolist())
    if args.case_rows:
        rows.extend(int(part.strip()) for part in args.case_rows.split(",") if part.strip())
    if args.render_top1_misses:
        rows.extend(summary.loc[summary["top1_hit"] == 0, "row_index"].astype(int).tolist())
    if args.render_top5_misses:
        rows.extend(summary.loc[summary["top5_hit"] == 0, "row_index"].astype(int).tolist())
    if args.random_hits > 0:
        hits = summary[summary["top1_hit"] == 1].sample(n=min(args.random_hits, int((summary["top1_hit"] == 1).sum())), random_state=args.seed)
        rows.extend(hits["row_index"].astype(int).tolist())

    deduped: list[int] = []
    for row in rows:
        if row not in deduped:
            deduped.append(row)
    if not deduped:
        deduped = summary.head(12)["row_index"].astype(int).tolist()

    miss_dir = args.out_dir / "top1_misses"
    top5_miss_dir = args.out_dir / "top5_misses"
    hit_dir = args.out_dir / "sample_hits"
    custom_dir = args.out_dir / "selected"
    all_dir = args.out_dir / "all_cases"
    miss_images: list[Path] = []
    top5_miss_images: list[Path] = []
    hit_images: list[Path] = []
    selected_images: list[Path] = []
    all_images: list[Path] = []

    top1_miss_rows = set(summary.loc[summary["top1_hit"] == 0, "row_index"].astype(int).tolist())
    top5_miss_rows = set(summary.loc[summary["top5_hit"] == 0, "row_index"].astype(int).tolist())
    for row_index in deduped:
        summary_row = summary.loc[summary["row_index"] == row_index]
        if summary_row.empty:
            print(f"skip missing row_index={row_index}")
            continue
        if args.all_cases:
            out_dir = all_dir
        elif args.render_top5_misses and row_index in top5_miss_rows:
            out_dir = top5_miss_dir
        else:
            out_dir = miss_dir if row_index in top1_miss_rows else hit_dir
        if args.case_rows and row_index not in top1_miss_rows and not args.all_cases:
            out_dir = custom_dir
        image = draw_case(
            row_index,
            summary_row.iloc[0],
            boxes,
            input_gdf,
            truth_gdf,
            out_dir,
            args.max_rank,
            args.zoom,
            args.near_radius_m,
        )
        if args.all_cases:
            all_images.append(image)
            if row_index in top5_miss_rows:
                top5_miss_images.append(image)
        elif row_index in top5_miss_rows and args.render_top5_misses:
            top5_miss_images.append(image)
        elif row_index in top1_miss_rows:
            miss_images.append(image)
        elif args.case_rows:
            selected_images.append(image)
        else:
            hit_images.append(image)
        print(image)

    if miss_images:
        make_contact_sheets(miss_images, args.out_dir / "top1_misses_contact", cols=3, page_size=args.contact_page_size)
    if top5_miss_images:
        make_contact_sheets(top5_miss_images, args.out_dir / "top5_misses_contact", cols=3, page_size=args.contact_page_size)
    if hit_images:
        make_contact_sheets(hit_images, args.out_dir / "sample_hits_contact", cols=3, page_size=args.contact_page_size)
    if selected_images:
        make_contact_sheets(selected_images, args.out_dir / "selected_contact", cols=3, page_size=args.contact_page_size)
    if all_images:
        make_contact_sheets(all_images, args.out_dir / "all_cases_contact", cols=3, page_size=args.contact_page_size)
    print(f"out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
