#!/usr/bin/env python3
"""Context visualization for non-top1 rescued ROI cases."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import Point, box


ROOT = Path(__file__).resolve().parent
VIS_SCRIPT = ROOT / "7_visualize_candidate_evidence_roi.py"
spec = importlib.util.spec_from_file_location("vis_v1", VIS_SCRIPT)
vis_v1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["vis_v1"] = vis_v1
spec.loader.exec_module(vis_v1)

DEFAULT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined_corridor_augmented_roi_step50_v1"
)
DEFAULT_EVIDENCE_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined_candidate_evidence_roi_v1_evidence.csv"
)
DEFAULT_SAMPLE_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined.gpkg"
)
DEFAULT_SAMPLE_LAYER = "mansfield-manual-polygon-link-random1200-combined"
DEFAULT_AUDIT_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined_plan_ocr_anchor_audit.csv"
)
DEFAULT_FULL_GPKG = Path("/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg")
DEFAULT_FULL_LAYER = "mansfield-manual-polygon-link"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_OUT_DIR = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/rescued_non_top1_roi_context_visuals"
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


def load_evidence() -> pd.DataFrame:
    """Combine evidence point CSVs for random200 and random1000 if available."""
    out_dir = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
    pairs = [
        (
            "random200_seed42",
            out_dir / "mansfield-manual-polygon-link_random200_seed42_candidate_evidence_roi_v1_evidence.csv",
        ),
        (
            "random1000_seed43_excl_random200",
            out_dir
            / "mansfield-manual-polygon-link_random1000_seed43_excl_random200_candidate_evidence_roi_v1_evidence.csv",
        ),
    ]
    frames = []
    for split, path in pairs:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"case_key": str, "base_key": str})
        df["sample_split"] = split
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["sample_split", "case_key", "base_key", "source", "label", "easting", "northing", "weight"]
        )
    return pd.concat(frames, ignore_index=True)


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int, thumb_size: int) -> None:
    if not image_paths:
        return
    thumbs = []
    for idx, path in enumerate(image_paths, start=1):
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
    parser.add_argument("--full-gpkg", type=Path, default=DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=DEFAULT_FULL_LAYER)
    parser.add_argument("--sample-gpkg", type=Path, default=DEFAULT_SAMPLE_GPKG)
    parser.add_argument("--sample-layer", default=DEFAULT_SAMPLE_LAYER)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--open-names", type=Path, default=DEFAULT_OPEN_NAMES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pad", type=float, default=90.0)
    parser.add_argument("--min-size", type=float, default=430.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv, dtype={"key": str, "base_key": str, "sample_split": str})
    rois = pd.read_csv(args.rois_csv, dtype={"case_key": str, "base_key": str, "sample_split": str})
    audit = pd.read_csv(args.audit_csv, dtype={"key": str, "base_key": str, "sample_split": str})
    evidence = load_evidence()

    rois["roi_rank"] = pd.to_numeric(rois["roi_rank"], errors="coerce")
    rois["roi_intersects_target_polygon"] = bool_series(rois["roi_intersects_target_polygon"])
    rois["roi_contains_target_centroid"] = bool_series(rois["roi_contains_target_centroid"])
    summary["union_intersects_target_polygon"] = bool_series(summary["union_intersects_target_polygon"])
    summary["union_candidate_polygon_count"] = pd.to_numeric(
        summary["union_candidate_polygon_count"], errors="coerce"
    )
    for col in ["easting", "northing", "weight"]:
        if col in evidence:
            evidence[col] = pd.to_numeric(evidence[col], errors="coerce")

    full = gpd.read_file(args.full_gpkg, layer=args.full_layer)
    if full.crs is not None:
        full = full.to_crs(27700)
    full["unique_key"] = full["unique_key"].astype(str)
    sample = gpd.read_file(args.sample_gpkg, layer=args.sample_layer)
    if sample.crs is not None:
        sample = sample.to_crs(27700)
    sample["unique_key"] = sample["unique_key"].astype(str)
    truth_by_key = {str(row["unique_key"]): row.geometry for _, row in sample.iterrows()}
    road_geoms = vis_v1.load_road_geoms(args.open_roads)
    feature_points = vis_v1.load_openname_points(args.open_names)
    audit_by_case = {
        (str(row.get("sample_split")), str(row["key"])): row
        for _, row in audit.iterrows()
        if str(row.get("sample_split")) != "nan"
    }
    # Fallback for CSVs without split.
    for _, row in audit.iterrows():
        audit_by_case.setdefault(("", str(row["key"])), row)

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
        window = box(minx, miny, maxx, maxy)

        fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=180)
        ax.set_facecolor("#fbfbfb")
        bg = full[full.geometry.intersects(window)].copy()
        if not bg.empty:
            bg.plot(ax=ax, facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.45, alpha=0.62, zorder=1)

        audit_row = audit_by_case.get((split, case_key))
        if audit_row is None:
            audit_row = audit_by_case.get(("", case_key))
        ocr_roads = vis_v1.split_anchors(audit_row.get("matched_roads") if audit_row is not None else "")
        ocr_features = vis_v1.split_anchors(audit_row.get("matched_features") if audit_row is not None else "")
        for idx, road in enumerate(ocr_roads[:10]):
            geom = road_geoms.get(road)
            if geom is None or geom.is_empty:
                continue
            clipped = geom.intersection(window)
            if clipped.is_empty:
                continue
            gpd.GeoSeries([clipped], crs=27700).plot(
                ax=ax,
                color=vis_v1.ANCHOR_LINE_COLORS[idx % len(vis_v1.ANCHOR_LINE_COLORS)],
                linewidth=3.8,
                alpha=0.92,
                zorder=4,
            )
            label_point = clipped.interpolate(0.5, normalized=True) if hasattr(clipped, "interpolate") else clipped.centroid
            ax.text(
                label_point.x,
                label_point.y,
                road.title()[:22],
                fontsize=6.5,
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
                zorder=9,
            )

        for feature in ocr_features[:10]:
            points = [p for p in feature_points.get(feature, []) if window.contains(p)]
            if not points:
                continue
            xs = [p.x for p in points[:6]]
            ys = [p.y for p in points[:6]]
            ax.scatter(xs, ys, marker="*", s=120, color="#facc15", edgecolors="#111827", linewidths=0.7, zorder=7)
            ax.text(xs[0], ys[0] + 7, feature.title()[:24], fontsize=6.5, color="#111827", zorder=9)

        ev = evidence[(evidence["sample_split"].astype(str) == split) & (evidence["case_key"].astype(str) == case_key)].copy()
        if not ev.empty:
            ev = ev.dropna(subset=["easting", "northing", "weight"])
            ev = ev[ev.apply(lambda r: window.contains(Point(r["easting"], r["northing"])), axis=1)]
            for source, group in ev.groupby("source"):
                ax.scatter(
                    group["easting"],
                    group["northing"],
                    s=(group["weight"].clip(lower=1, upper=9) * 15).values,
                    color=vis_v1.SOURCE_COLORS.get(source, "#6b7280"),
                    edgecolors="white",
                    linewidths=0.7,
                    alpha=0.93,
                    zorder=6,
                )

        for (_, r), geom in zip(roi_df.iterrows(), roi_geoms):
            rank = int(r["roi_rank"])
            hit = bool(r["roi_intersects_target_polygon"])
            if rank == 1:
                color = "#111827"
                linestyle = "-"
                linewidth = 3.2
            elif hit:
                color = "#16a34a"
                linestyle = "-"
                linewidth = 3.4
            else:
                color = "#f59e0b"
                linestyle = "--"
                linewidth = 2.2
            gpd.GeoSeries([geom], crs=27700).boundary.plot(
                ax=ax,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=10 if hit else 8,
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
                zorder=11,
            )

        gpd.GeoSeries([truth], crs=27700).plot(
            ax=ax,
            facecolor=(0.9, 0.05, 0.05, 0.28),
            edgecolor="#dc2626",
            linewidth=3.4,
            zorder=12,
        )
        hit_ranks = [
            str(int(r["roi_rank"]))
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
        out = args.out_dir / f"{split}_{case_key}_rescued_context.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        image_paths.append(out)
        rows.append(
            {
                "sample_split": split,
                "key": case_key,
                "base_key": base_key,
                "original_address": row.get("original_address"),
                "hit_non_top1_ranks": " | ".join(hit_ranks),
                "ocr_roads": " | ".join(ocr_roads),
                "ocr_features": " | ".join(ocr_features),
                "visible_evidence_points": int(len(ev)) if "ev" in locals() else 0,
                "union_candidate_polygon_count": int(row["union_candidate_polygon_count"]),
                "image": str(out),
            }
        )
        print(out, flush=True)

    out_summary = pd.DataFrame(rows)
    out_summary.to_csv(args.out_dir / "rescued_context_summary.csv", index=False)
    if image_paths:
        make_contact_sheet(image_paths[:30], args.out_dir / "rescued_context_contact_sheet_first30.jpg", 5, 460)
        make_contact_sheet(image_paths, args.out_dir / "rescued_context_contact_sheet_all.jpg", 6, 390)
    print(f"rescued_context_cases={len(rows)}")
    print(args.out_dir / "rescued_context_contact_sheet_first30.jpg")
    print(args.out_dir / "rescued_context_contact_sheet_all.jpg")


if __name__ == "__main__":
    main()
