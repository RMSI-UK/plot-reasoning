#!/usr/bin/env python3
"""Visualize cases rescued by non-top1 ROI boxes.

Rescued means:
- roi_rank=1 does not intersect the target polygon
- at least one later ROI does intersect the target polygon

The plot keeps the visual vocabulary intentionally simple:
- truth polygon: red fill/border
- top1 ROI: black solid box
- non-top1 hit ROI: green solid box
- other non-top1 ROI: amber dashed box
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box


DEFAULT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined_corridor_augmented_roi_step50_v1"
)
DEFAULT_SAMPLE_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined.gpkg"
)
DEFAULT_SAMPLE_LAYER = "mansfield-manual-polygon-link-random1200-combined"
DEFAULT_OUT_DIR = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/rescued_non_top1_roi_visuals"
)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def expand_bounds(bounds, pad: float, min_size: float):
    minx, miny, maxx, maxy = bounds
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    width = max(maxx - minx, min_size)
    height = max(maxy - miny, min_size)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def make_contact_sheet(
    image_paths: list[Path],
    out_path: Path,
    cols: int,
    thumb_size: int,
    max_images: int | None = None,
) -> None:
    paths = image_paths[:max_images] if max_images else image_paths
    if not paths:
        return
    thumbs = []
    for idx, path in enumerate(paths, start=1):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        canvas.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, thumb_size - 1, thumb_size - 1), outline=(210, 210, 210), width=1)
        draw.text((8, 6), str(idx), fill=(20, 20, 20))
        thumbs.append(canvas)

    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb_size, (idx // cols) * thumb_size))
    sheet.save(out_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_PREFIX.with_suffix(".csv"))
    parser.add_argument(
        "--rois-csv",
        type=Path,
        default=DEFAULT_PREFIX.with_name(DEFAULT_PREFIX.name + "_rois.csv"),
    )
    parser.add_argument("--sample-gpkg", type=Path, default=DEFAULT_SAMPLE_GPKG)
    parser.add_argument("--sample-layer", default=DEFAULT_SAMPLE_LAYER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pad", type=float, default=70.0)
    parser.add_argument("--min-size", type=float, default=320.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv, dtype={"key": str, "base_key": str, "sample_split": str})
    rois = pd.read_csv(args.rois_csv, dtype={"case_key": str, "base_key": str, "sample_split": str})
    rois["roi_rank"] = pd.to_numeric(rois["roi_rank"], errors="coerce")
    rois["roi_intersects_target_polygon"] = bool_series(rois["roi_intersects_target_polygon"])
    rois["roi_contains_target_centroid"] = bool_series(rois["roi_contains_target_centroid"])
    summary["union_intersects_target_polygon"] = bool_series(summary["union_intersects_target_polygon"])
    summary["union_candidate_polygon_count"] = pd.to_numeric(
        summary["union_candidate_polygon_count"], errors="coerce"
    )

    sample = gpd.read_file(args.sample_gpkg, layer=args.sample_layer)
    if sample.crs is not None:
        sample = sample.to_crs(27700)
    sample["unique_key"] = sample["unique_key"].astype(str)
    truth_by_key = {str(row["unique_key"]): row.geometry for _, row in sample.iterrows()}

    top1 = rois[rois["roi_rank"] == 1][
        ["sample_split", "case_key", "roi_intersects_target_polygon"]
    ].rename(columns={"roi_intersects_target_polygon": "top1_hit"})
    rescued_keys = []
    for (split, case_key), group in rois.groupby(["sample_split", "case_key"], dropna=False):
        group = group.sort_values("roi_rank")
        top1_hit = bool(group.iloc[0]["roi_intersects_target_polygon"]) if len(group) else False
        later_hit = bool(group[group["roi_rank"] > 1]["roi_intersects_target_polygon"].any())
        if not top1_hit and later_hit:
            rescued_keys.append((str(split), str(case_key)))
    if args.limit > 0:
        rescued_keys = rescued_keys[: args.limit]

    rows = []
    image_paths: list[Path] = []
    for split, case_key in rescued_keys:
        row_df = summary[(summary["sample_split"].astype(str) == split) & (summary["key"] == case_key)]
        roi_df = rois[(rois["sample_split"].astype(str) == split) & (rois["case_key"] == case_key)].copy()
        if row_df.empty or roi_df.empty:
            continue
        row = row_df.iloc[0]
        base_key = str(row["base_key"]).split("_", 1)[0]
        truth = truth_by_key.get(base_key)
        if truth is None:
            continue

        roi_df = roi_df.sort_values("roi_rank")
        roi_geoms = [
            box(float(r["roi_minx"]), float(r["roi_miny"]), float(r["roi_maxx"]), float(r["roi_maxy"]))
            for _, r in roi_df.iterrows()
        ]
        bounds = gpd.GeoSeries([truth, *roi_geoms], crs=27700).total_bounds
        minx, miny, maxx, maxy = expand_bounds(bounds, args.pad, args.min_size)

        fig, ax = plt.subplots(figsize=(7.4, 7.4), dpi=180)
        ax.set_facecolor("white")

        for (_, r), geom in zip(roi_df.iterrows(), roi_geoms):
            rank = int(r["roi_rank"])
            hit = bool(r["roi_intersects_target_polygon"])
            if rank == 1:
                color = "#111827"
                linestyle = "-"
                linewidth = 3.3
            elif hit:
                color = "#16a34a"
                linestyle = "-"
                linewidth = 3.4
            else:
                color = "#f59e0b"
                linestyle = "--"
                linewidth = 2.4
            gpd.GeoSeries([geom], crs=27700).boundary.plot(
                ax=ax,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=3 if rank != 1 else 4,
            )
            ax.text(
                geom.centroid.x,
                geom.bounds[3] + 5,
                str(rank),
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold",
                zorder=6,
            )

        gpd.GeoSeries([truth], crs=27700).plot(
            ax=ax,
            facecolor=(0.9, 0.05, 0.05, 0.28),
            edgecolor="#dc2626",
            linewidth=3.4,
            zorder=7,
        )
        hit_ranks = [
            str(int(r["roi_rank"]))
            for _, r in roi_df[roi_df["roi_intersects_target_polygon"]].iterrows()
            if int(r["roi_rank"]) > 1
        ]
        hit_reasons = [
            f"{int(r['roi_rank'])}:{r.get('roi_reason', '')}/{r.get('roi_sources', '')}"
            for _, r in roi_df[roi_df["roi_intersects_target_polygon"]].iterrows()
            if int(r["roi_rank"]) > 1
        ]

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(
            f"{case_key}  hit ranks={','.join(hit_ranks)}  candidates={int(row['union_candidate_polygon_count'])}",
            fontsize=10,
            pad=4,
        )
        out = args.out_dir / f"{split}_{case_key}_rescued_roi.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        image_paths.append(out)
        rows.append(
            {
                "sample_split": split,
                "key": case_key,
                "base_key": base_key,
                "original_address": row.get("original_address"),
                "best_confidence": row.get("best_confidence"),
                "roi_count": len(roi_df),
                "hit_non_top1_ranks": " | ".join(hit_ranks),
                "hit_non_top1_reasons": " || ".join(hit_reasons),
                "union_candidate_polygon_count": int(row["union_candidate_polygon_count"]),
                "image": str(out),
            }
        )
        print(out, flush=True)

    out_summary = pd.DataFrame(rows)
    out_summary.to_csv(args.out_dir / "rescued_cases_summary.csv", index=False)
    if not out_summary.empty:
        with pd.ExcelWriter(args.out_dir / "rescued_cases_summary.xlsx") as writer:
            out_summary.to_excel(writer, sheet_name="rescued_cases", index=False)

    make_contact_sheet(image_paths, args.out_dir / "rescued_non_top1_contact_sheet_first30.jpg", 5, 460, 30)
    make_contact_sheet(image_paths, args.out_dir / "rescued_non_top1_contact_sheet_all.jpg", 6, 390, None)
    print(f"rescued_cases={len(rows)}")
    print(args.out_dir / "rescued_cases_summary.csv")
    print(args.out_dir / "rescued_non_top1_contact_sheet_first30.jpg")
    print(args.out_dir / "rescued_non_top1_contact_sheet_all.jpg")


if __name__ == "__main__":
    main()
