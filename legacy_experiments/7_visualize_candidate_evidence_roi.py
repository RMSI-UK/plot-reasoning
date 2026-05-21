#!/usr/bin/env python3
"""Visualize evidence ROI output against manual target polygons."""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import nearest_points
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_ROI_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_candidate_evidence_roi_v1.csv"
)
DEFAULT_EVIDENCE_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_candidate_evidence_roi_v1_evidence.csv"
)
DEFAULT_FULL_GPKG = Path("/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg")
DEFAULT_FULL_LAYER = "mansfield-manual-polygon-link"
DEFAULT_SAMPLE_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42.gpkg"
)
DEFAULT_SAMPLE_LAYER = "mansfield-manual-polygon-link-random200"
DEFAULT_OUT_DIR = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/candidate_evidence_roi_visuals"
)
DEFAULT_AUDIT_CSV = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_plan_ocr_anchor_audit.csv"
)
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_CASES = ["2911", "5428", "4460", "5827", "6837", "1784", "2222", "4128", "7699"]


T277_TO_3857 = Transformer.from_crs(27700, 3857, always_xy=True)
T3857_TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)


SOURCE_COLORS = {
    "current": "#111827",
    "os": "#2563eb",
    "gog": "#7c3aed",
    "fallback": "#0891b2",
    "ocr_grid": "#f97316",
    "road_zone": "#16a34a",
    "web": "#dc2626",
}

ANCHOR_LINE_COLORS = [
    "#ef4444",
    "#22c55e",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#84cc16",
    "#ec4899",
    "#14b8a6",
]


def norm(text) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_anchors(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    out = []
    for part in str(value).split("|"):
        item = norm(part)
        if item and item not in out:
            out.append(item)
    return out


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
        req = urllib.request.Request(url, headers={"User-Agent": "MansfieldRoiVisualizer/1.0"})
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


def to_3857(geoms):
    return gpd.GeoDataFrame(geometry=list(geoms), crs=27700).to_crs(3857)


def expand_bounds(bounds, pad: float):
    minx, miny, maxx, maxy = bounds
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def load_road_geoms(open_roads: Path) -> dict[str, object]:
    roads = gpd.read_file(
        open_roads,
        layer="road_link",
        bbox=(449000, 343000, 462500, 371000),
        columns=["name_1", "name_2"],
    ).to_crs(27700)
    groups = {}
    for _, row in roads.iterrows():
        for col in ("name_1", "name_2"):
            name = norm(row.get(col))
            if name:
                groups.setdefault(name, []).append(row.geometry)
    return {name: gpd.GeoSeries(geoms, crs=27700).union_all() for name, geoms in groups.items()}


def load_openname_points(open_names: Path) -> dict[str, list[Point]]:
    con = sqlite3.connect(open_names)
    rows = con.execute(
        """
        SELECT NAME1, GEOMETRY_X, GEOMETRY_Y
        FROM open_names
        WHERE CAST(GEOMETRY_X AS REAL) BETWEEN 449000 AND 462500
          AND CAST(GEOMETRY_Y AS REAL) BETWEEN 343000 AND 371000
        """
    ).fetchall()
    out: dict[str, list[Point]] = {}
    for name, x, y in rows:
        try:
            point = Point(float(x), float(y))
        except (TypeError, ValueError):
            continue
        out.setdefault(norm(name), []).append(point)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI_CSV)
    parser.add_argument("--evidence-csv", type=Path, default=DEFAULT_EVIDENCE_CSV)
    parser.add_argument("--full-gpkg", type=Path, default=DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=DEFAULT_FULL_LAYER)
    parser.add_argument("--sample-gpkg", type=Path, default=DEFAULT_SAMPLE_GPKG)
    parser.add_argument("--sample-layer", default=DEFAULT_SAMPLE_LAYER)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--open-names", type=Path, default=DEFAULT_OPEN_NAMES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--no-osm", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi = pd.read_csv(args.roi_csv, dtype={"key": str, "base_key": str})
    evidence = pd.read_csv(args.evidence_csv, dtype={"case_key": str, "base_key": str})
    audit = pd.read_csv(args.audit_csv, dtype={"key": str, "base_key": str})
    audit_by_key = {str(row["key"]): row for _, row in audit.iterrows()}
    road_geoms = load_road_geoms(args.open_roads)
    feature_points = load_openname_points(args.open_names)
    full = gpd.read_file(args.full_gpkg, layer=args.full_layer).to_crs(27700)
    full["unique_key"] = full["unique_key"].astype(str)
    sample = gpd.read_file(args.sample_gpkg, layer=args.sample_layer).to_crs(27700)
    sample["unique_key"] = sample["unique_key"].astype(str)
    full_by_key = {str(row["unique_key"]): row.geometry for _, row in full.iterrows()}
    truth_by_key = {str(row["unique_key"]): row.geometry for _, row in sample.iterrows()}

    rows = []
    for case_key in args.cases:
        row = roi[roi["key"] == str(case_key)]
        if row.empty:
            continue
        row = row.iloc[0]
        base_key = str(row["base_key"]).split("_", 1)[0]
        truth = truth_by_key.get(base_key)
        top1_key = str(row.get("top1_candidate_key"))
        if top1_key.endswith(".0"):
            top1_key = top1_key[:-2]
        top1 = full_by_key.get(top1_key)
        if truth is None or top1 is None:
            continue

        roi_geom = box(row["roi_minx"], row["roi_miny"], row["roi_maxx"], row["roi_maxy"])
        audit_row = audit_by_key.get(str(case_key))
        ocr_roads = split_anchors(audit_row.get("matched_roads") if audit_row is not None else "")
        ocr_features = split_anchors(audit_row.get("matched_features") if audit_row is not None else "")
        case_evidence = evidence[evidence["case_key"] == str(case_key)].copy()
        for col in ["easting", "northing", "weight"]:
            case_evidence[col] = pd.to_numeric(case_evidence[col], errors="coerce")
        case_evidence = case_evidence.dropna(subset=["easting", "northing", "weight"])

        selected_geoms = [truth, top1, roi_geom]
        for road in ocr_roads:
            geom = road_geoms.get(road)
            if geom is not None and not geom.is_empty:
                selected_geoms.append(geom.intersection(box(*truth.buffer(900).bounds)))
        for feature in ocr_features:
            selected_geoms.extend(feature_points.get(feature, [])[:8])
        if not case_evidence.empty:
            ev_points = [Point(x, y) for x, y in zip(case_evidence["easting"], case_evidence["northing"])]
            selected_geoms.extend(ev_points[:80])
        bounds = gpd.GeoSeries(selected_geoms, crs=27700).total_bounds
        minx, miny, maxx, maxy = expand_bounds(bounds, 180)
        width = max(maxx - minx, 650)
        height = max(maxy - miny, 650)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        minx, maxx = cx - width / 2, cx + width / 2
        miny, maxy = cy - height / 2, cy + height / 2
        minx3857, miny3857 = T277_TO_3857.transform(minx, miny)
        maxx3857, maxy3857 = T277_TO_3857.transform(maxx, maxy)
        bounds3857 = (
            min(minx3857, maxx3857),
            min(miny3857, maxy3857),
            max(minx3857, maxx3857),
            max(miny3857, maxy3857),
        )

        fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
        if not args.no_osm:
            try:
                img, extent = basemap(bounds3857, args.out_dir)
                ax.imshow(img, extent=extent, interpolation="bilinear")
            except Exception:
                ax.set_facecolor("#f7f3ea")
        else:
            ax.set_facecolor("#f7f3ea")
            window277 = box(minx, miny, maxx, maxy)
            bg = full[full.geometry.intersects(window277)].copy()
            if not bg.empty:
                bg3857 = bg.to_crs(3857)
                bg3857.plot(ax=ax, facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.45, alpha=0.65)

        truth3857 = to_3857([truth]).geometry.iloc[0]
        top13857 = to_3857([top1]).geometry.iloc[0]
        roi3857 = to_3857([roi_geom]).geometry.iloc[0]
        gpd.GeoSeries([roi3857], crs=3857).boundary.plot(ax=ax, color="#f59e0b", linewidth=3.2, linestyle="--")
        gpd.GeoSeries([top13857], crs=3857).plot(ax=ax, facecolor=(0.1, 0.35, 1.0, 0.22), edgecolor="#2563eb", linewidth=3.0)
        gpd.GeoSeries([truth3857], crs=3857).plot(ax=ax, facecolor=(1.0, 0.0, 0.0, 0.24), edgecolor="#dc2626", linewidth=3.6)

        # OCR anchor line and point features.
        window277 = box(minx, miny, maxx, maxy)
        drawn_ocr_roads = []
        for idx, road in enumerate(ocr_roads[:8]):
            geom = road_geoms.get(road)
            if geom is None or geom.is_empty:
                continue
            clipped = geom.intersection(window277)
            if clipped.is_empty:
                continue
            color = ANCHOR_LINE_COLORS[idx % len(ANCHOR_LINE_COLORS)]
            gpd.GeoSeries([clipped], crs=27700).to_crs(3857).plot(
                ax=ax,
                color=color,
                linewidth=4.2,
                alpha=0.95,
                zorder=5,
            )
            drawn_ocr_roads.append((road, color))

        drawn_features = []
        for feature in ocr_features[:10]:
            points = [p for p in feature_points.get(feature, []) if window277.buffer(120).contains(p)]
            if not points:
                continue
            pts3857 = to_3857(points[:8])
            ax.scatter(
                pts3857.geometry.x,
                pts3857.geometry.y,
                marker="*",
                s=180,
                color="#facc15",
                edgecolors="#111827",
                linewidths=0.9,
                zorder=8,
            )
            drawn_features.append(feature)

        # Connect nearest points between selected top1 polygon and truth polygon.
        a, b = nearest_points(truth, top1)
        ab3857 = to_3857([a, b]).geometry
        ax.plot([ab3857.iloc[0].x, ab3857.iloc[1].x], [ab3857.iloc[0].y, ab3857.iloc[1].y], color="#111827", linewidth=2.1)

        # Evidence points.
        for source, group in case_evidence.groupby("source"):
            pts3857 = to_3857([Point(x, y) for x, y in zip(group["easting"], group["northing"])])
            sizes = (group["weight"].clip(lower=1, upper=10) * 18).values
            ax.scatter(
                pts3857.geometry.x,
                pts3857.geometry.y,
                s=sizes,
                color=SOURCE_COLORS.get(source, "#6b7280"),
                edgecolors="white",
                linewidths=0.8,
                alpha=0.9,
                zorder=6,
                label=source,
            )

        roi_dist = roi_geom.distance(truth)
        selected_dist = top1.distance(truth)
        centroid_dist = top1.centroid.distance(truth.centroid)
        text = (
            f"case {case_key}: {row['original_address']}\n"
            f"top1 polygon key: {top1_key}\n"
            f"top1-to-truth polygon distance: {selected_dist:.1f}m\n"
            f"top1 centroid-to-truth centroid: {centroid_dist:.1f}m\n"
            f"ROI-to-truth polygon distance: {roi_dist:.1f}m\n"
            f"ROI candidates: {int(row['roi_candidate_polygon_count'])}; target rank: {row.get('target_rank_in_roi_candidates')}\n"
            f"ROI sources: {row.get('roi_sources')}\n"
            f"OCR roads: {' | '.join(ocr_roads[:6]) or 'none'}\n"
            f"OCR features: {' | '.join(ocr_features[:4]) or 'none'}"
        )
        ax.text(
            0.012,
            0.012,
            text,
            transform=ax.transAxes,
            fontsize=8.8,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="#333", boxstyle="round,pad=0.45"),
        )

        legend_handles = [
            Patch(facecolor=(1, 0, 0, 0.24), edgecolor="#dc2626", label="manual truth polygon"),
            Patch(facecolor=(0.1, 0.35, 1.0, 0.22), edgecolor="#2563eb", label="new scheme top1 polygon"),
            Line2D([0], [0], color="#f59e0b", linewidth=3, linestyle="--", label="150m evidence ROI"),
            Line2D([0], [0], color="#111827", linewidth=2, label="nearest offset"),
        ]
        if drawn_ocr_roads:
            legend_handles.append(Line2D([0], [0], color="#ef4444", linewidth=4, label="OCR road anchor lines"))
        if drawn_features:
            legend_handles.append(Line2D([0], [0], marker="*", color="w", label="OCR feature anchor points", markerfacecolor="#facc15", markeredgecolor="#111827", markersize=12))
        for source, color in SOURCE_COLORS.items():
            if source in set(case_evidence["source"]):
                legend_handles.append(Line2D([0], [0], marker="o", color="w", label=source, markerfacecolor=color, markersize=7))
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.92, fontsize=8)
        ax.set_xlim(bounds3857[0], bounds3857[2])
        ax.set_ylim(bounds3857[1], bounds3857[3])
        ax.set_axis_off()
        ax.set_title(f"Candidate Evidence ROI vs Truth - case {case_key}", fontsize=14, pad=10)

        out = args.out_dir / f"{case_key}_evidence_roi_vs_truth.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        rows.append(
            {
                "key": case_key,
                "base_key": base_key,
                "original_address": row["original_address"],
                "top1_candidate_key": top1_key,
                "top1_to_truth_polygon_m": selected_dist,
                "top1_centroid_to_truth_centroid_m": centroid_dist,
                "roi_to_truth_polygon_m": roi_dist,
                "roi_intersects_target": bool(roi_geom.intersects(truth)),
                "roi_candidate_polygon_count": int(row["roi_candidate_polygon_count"]),
                "target_rank_in_roi": row.get("target_rank_in_roi_candidates"),
                "ocr_roads": " | ".join(ocr_roads),
                "ocr_features": " | ".join(ocr_features),
                "image": str(out),
            }
        )
        print(out, flush=True)

    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / "visualized_cases_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Make a compact contact sheet for quick scanning.
    images = [Image.open(path).convert("RGB") for path in summary["image"]] if not summary.empty else []
    thumbs = []
    for image in images:
        image.thumbnail((520, 520))
        canvas = Image.new("RGB", (520, 520), "white")
        canvas.paste(image, ((520 - image.width) // 2, (520 - image.height) // 2))
        thumbs.append(canvas)
    if thumbs:
        cols = 3
        rows_n = math.ceil(len(thumbs) / cols)
        sheet = Image.new("RGB", (cols * 520, rows_n * 520), "white")
        for idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((idx % cols) * 520, (idx // cols) * 520))
        sheet_path = args.out_dir / "candidate_evidence_roi_contact_sheet.jpg"
        sheet.save(sheet_path, quality=92)
        print(sheet_path, flush=True)


if __name__ == "__main__":
    main()
