#!/usr/bin/env python3
"""Build and validate a Mansfield merged polygon basemap.

The production-visible candidate base should not rely on the manual polygon
truth layer.  This script builds the WFS land/building shared-edge merge used
by the capture pipeline, and can optionally validate it against manual truth
alongside Mansfield cadastral parcels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "spatial_capture_production"
sys.path.insert(0, str(PROD))

from capture_wfs_merge import build_wfs_merge_gdf, filter_wfs_theme_features  # noqa: E402


DEFAULT_WFS_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg")
DEFAULT_WFS_LAYER = "mansfield_polygons_in_buffers"
DEFAULT_COUNCIL_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_councils_land.gpkg")
DEFAULT_COUNCIL_LAYER = "cadastral_parcels"
DEFAULT_OUTPUT_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg")
DEFAULT_OUTPUT_LAYER = "mansfield_polygons_in_buffers_merged"
DEFAULT_TRUTH_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random1200_seed42_43_combined.gpkg"
)
DEFAULT_TRUTH_LAYER = "random1200_seed42_43_combined"
DEFAULT_REPORT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield_wfs_polygon_merged_validation"
)


def load_27700(path: Path, layer: str, **kwargs: Any) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, **kwargs)
    if gdf.crs is None:
        gdf = gdf.set_crs(27700)
    else:
        gdf = gdf.to_crs(27700)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def candidate_id_series(gdf: gpd.GeoDataFrame, prefix: str) -> pd.Series:
    if "TOID" in gdf.columns:
        ids = gdf["TOID"].fillna("").astype(str)
    elif "gml_id" in gdf.columns:
        ids = gdf["gml_id"].fillna("").astype(str)
    elif "INSPIREID" in gdf.columns:
        ids = gdf["INSPIREID"].fillna("").astype(str)
    else:
        ids = pd.Series([""] * len(gdf), index=gdf.index)
    missing = ids.eq("") | ids.eq("nan")
    ids.loc[missing] = [f"{prefix}_{i}" for i in gdf.index[missing]]
    return ids


def best_intersection_metrics(
    truth: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cand = candidates[["candidate_id", "geometry"]].copy()
    cand["candidate_area_m2"] = cand.geometry.area
    joined = gpd.sjoin(
        truth[["truth_id", "truth_area_m2", "geometry"]],
        cand,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        summary = {"label": label, "candidate_rows": int(len(candidates)), "joined_rows": 0, "present": 0}
        return summary, pd.DataFrame()

    truth_geoms = truth.geometry
    cand_geoms = cand.geometry
    inter_areas: list[float] = []
    for truth_id, cand_idx in zip(joined["truth_id"].to_numpy(), joined["index_right"].to_numpy()):
        inter_areas.append(float(truth_geoms.iloc[int(truth_id)].intersection(cand_geoms.iloc[int(cand_idx)]).area))
    joined["inter_area_m2"] = inter_areas
    joined = joined[joined["inter_area_m2"] > 0].copy()
    joined["candidate_area_m2"] = joined["index_right"].map(cand["candidate_area_m2"])
    joined["iou"] = joined["inter_area_m2"] / (
        joined["truth_area_m2"] + joined["candidate_area_m2"] - joined["inter_area_m2"]
    )
    joined["truth_cover"] = joined["inter_area_m2"] / joined["truth_area_m2"]
    joined["candidate_cover"] = joined["inter_area_m2"] / joined["candidate_area_m2"]

    best = (
        joined.sort_values(
            ["truth_id", "iou", "truth_cover", "candidate_cover"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("truth_id", keep="first")
        .copy()
    )
    present = int(best["truth_id"].nunique())
    total = int(len(truth))
    summary = {
        "label": label,
        "candidate_rows": int(len(candidates)),
        "joined_rows": int(len(joined)),
        "present": present,
        "present_rate": present / total if total else 0.0,
        "median_candidates_per_truth": float(joined.groupby("truth_id").size().median()),
        "mean_candidates_per_truth": float(joined.groupby("truth_id").size().mean()),
        "median_best_iou": float(best["iou"].median()),
        "mean_best_iou": float(best["iou"].mean()),
        "median_truth_cover": float(best["truth_cover"].median()),
        "median_candidate_cover": float(best["candidate_cover"].median()),
        "iou_ge_0_50": int((best["iou"] >= 0.50).sum()),
        "iou_ge_0_65": int((best["iou"] >= 0.65).sum()),
        "iou_ge_0_80": int((best["iou"] >= 0.80).sum()),
        "iou_ge_0_90": int((best["iou"] >= 0.90).sum()),
    }
    best_out = best[
        [
            "truth_id",
            "candidate_id",
            "iou",
            "truth_cover",
            "candidate_cover",
            "truth_area_m2",
            "candidate_area_m2",
        ]
    ].copy()
    best_out.insert(0, "label", label)
    return summary, best_out


def validate_layers(
    wfs_raw: gpd.GeoDataFrame,
    wfs_merged: gpd.GeoDataFrame,
    args: argparse.Namespace,
) -> None:
    truth = load_27700(args.truth_gpkg, args.truth_layer)
    truth = truth.reset_index(drop=True)
    truth["truth_id"] = truth.index
    truth["truth_area_m2"] = truth.geometry.area
    bounds = truth.total_bounds
    pad = float(args.validation_bbox_pad_m)
    bbox = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    council = load_27700(args.council_gpkg, args.council_layer, bbox=bbox)
    council = council.reset_index(drop=True)

    raw = wfs_raw.copy().reset_index(drop=True)
    merged = wfs_merged.copy().reset_index(drop=True)
    raw["candidate_id"] = candidate_id_series(raw, "wfs_raw")
    merged["candidate_id"] = candidate_id_series(merged, "wfs_merge")
    council["candidate_id"] = candidate_id_series(council, "council")

    reports: list[dict[str, Any]] = []
    best_rows: list[pd.DataFrame] = []
    for label, layer in [
        ("wfs_raw_land_building", raw),
        ("wfs_land_building_merged", merged),
        ("cadastral_parcels", council),
    ]:
        summary, best = best_intersection_metrics(truth, layer, label)
        reports.append(summary)
        if not best.empty:
            best_rows.append(best)

    if best_rows:
        all_best = pd.concat(best_rows, ignore_index=True)
        pivot = all_best.pivot_table(index="truth_id", columns="label", values="iou", aggfunc="max").fillna(0.0)
        pivot["hybrid_best_iou"] = pivot.max(axis=1)
        reports.append(
            {
                "label": "hybrid_best_of_raw_merged_cadastral",
                "truth_rows": int(len(truth)),
                "median_best_iou": float(pivot["hybrid_best_iou"].median()),
                "mean_best_iou": float(pivot["hybrid_best_iou"].mean()),
                "iou_ge_0_50": int((pivot["hybrid_best_iou"] >= 0.50).sum()),
                "iou_ge_0_65": int((pivot["hybrid_best_iou"] >= 0.65).sum()),
                "iou_ge_0_80": int((pivot["hybrid_best_iou"] >= 0.80).sum()),
                "iou_ge_0_90": int((pivot["hybrid_best_iou"] >= 0.90).sum()),
            }
        )
        best_path = args.report_prefix.with_name(args.report_prefix.name + "_best_matches.csv")
        all_best.to_csv(best_path, index=False)

    report_path = args.report_prefix.with_suffix(".summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    print(f"validation_summary: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mansfield WFS land/building merged basemap.")
    parser.add_argument("--wfs-gpkg", type=Path, default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--council-gpkg", type=Path, default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--council-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--output-gpkg", type=Path, default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKG)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--report-prefix", type=Path, default=DEFAULT_REPORT_PREFIX)
    parser.add_argument("--validation-bbox-pad-m", type=float, default=1000.0)
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    print(f"loading_wfs: {args.wfs_gpkg}")
    wfs = load_27700(args.wfs_gpkg, args.wfs_layer)
    wfs = filter_wfs_theme_features(wfs).reset_index(drop=True)
    print(f"wfs_land_building_rows: {len(wfs)}")

    print("building_wfs_land_building_merge")
    merged = build_wfs_merge_gdf(wfs).reset_index(drop=True)
    merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty].copy()
    print(f"merged_rows: {len(merged)}")

    if not args.skip_write:
        args.output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        if args.output_gpkg.exists():
            args.output_gpkg.unlink()
        merged.to_file(args.output_gpkg, layer=args.output_layer, driver="GPKG")
        print(f"wrote: {args.output_gpkg} layer={args.output_layer}")

    if not args.skip_validation:
        validate_layers(wfs, merged, args)


if __name__ == "__main__":
    main()
