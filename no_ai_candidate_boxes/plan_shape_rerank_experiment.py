#!/usr/bin/env python3
"""Plan-aware polygon reranking experiment for Mansfield.

This is an offline experiment.  It uses the existing truth-labelled candidate
CSV only for training/evaluation labels, and tests whether cheap visual signals
from cropped plan images help choose the right polygon from an already generated
candidate set.

The visual signals are deliberately simple and local:
- red/blue/black line ratios and largest connected component geometry from plan
  crops;
- candidate polygon shape descriptors from WFS/cadastral geometry;
- explicit interactions such as candidate aspect vs. plan red-boundary aspect.

No external AI/API call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCRIPT = ROOT / "legacy_experiments" / "13_train_polygon_selector.py"
DEFAULT_INPUT = ROOT / "tmp_results" / "mansfield_random1200_hybrid_top200_v1_top200.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "tmp_results" / "mansfield_plan_shape_rerank_v1"
DEFAULT_FULL_GPKG = Path("/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg")
DEFAULT_FULL_LAYER = "mansfield-manual-polygon-link"
DEFAULT_PLAN_ROOT = Path("/data/mansfield/scan-images/cropped-plan/all")
DEFAULT_COUNCIL_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_councils_land.gpkg")
DEFAULT_WFS_RAW_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg")
DEFAULT_WFS_MERGED_GPKG = Path("/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg")

spec = importlib.util.spec_from_file_location("selector13_features", FEATURE_SCRIPT)
selector13 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["selector13_features"] = selector13
spec.loader.exec_module(selector13)


def fold_for_case(case_key: str, folds: int) -> int:
    return int(hashlib.md5(str(case_key).encode("utf-8")).hexdigest()[:8], 16) % folds


def parse_mfd_ref(value: Any) -> str:
    match = re.search(r"(MFD_[^/\\]+)", str(value or ""))
    return match.group(1) if match else ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def component_features(mask: np.ndarray) -> dict[str, float]:
    if mask.size == 0:
        return {
            "ratio": 0.0,
            "component_ratio": 0.0,
            "component_bbox_ratio": 0.0,
            "component_fill": 0.0,
            "component_aspect": 0.0,
            "component_count": 0.0,
        }
    ratio = float(mask.mean())
    if not mask.any():
        return {
            "ratio": ratio,
            "component_ratio": 0.0,
            "component_bbox_ratio": 0.0,
            "component_fill": 0.0,
            "component_aspect": 0.0,
            "component_count": 0.0,
        }
    labels, count = ndimage.label(mask)
    slices = ndimage.find_objects(labels)
    best_area = 0
    best_bbox_area = 0
    best_aspect = 0.0
    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp_area = int((labels[slc] == idx).sum())
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        bbox_area = width * height
        if comp_area > best_area:
            best_area = comp_area
            best_bbox_area = bbox_area
            best_aspect = max(width / height, height / width)
    image_area = float(mask.shape[0] * mask.shape[1])
    return {
        "ratio": ratio,
        "component_ratio": float(best_area / image_area),
        "component_bbox_ratio": float(best_bbox_area / image_area),
        "component_fill": float(best_area / best_bbox_area) if best_bbox_area else 0.0,
        "component_aspect": float(best_aspect),
        "component_count": float(count),
    }


def analyse_plan_image(path: Path, max_side: int = 640) -> dict[str, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        arr = np.asarray(image).astype("float32")
    if arr.size == 0:
        return {}
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    mean = arr.mean(axis=2)
    red = (r > 115) & (r > g * 1.25) & (r > b * 1.20)
    blue = (b > 105) & (b > r * 1.15) & (b > g * 1.10)
    black = (mean < 85) & ((arr.max(axis=2) - arr.min(axis=2)) < 55)
    grey = mean / 255.0
    gx = np.abs(np.diff(grey, axis=1)).mean() if grey.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(grey, axis=0)).mean() if grey.shape[0] > 1 else 0.0
    edge_density = float(gx + gy)
    out: dict[str, float] = {
        "plan_width": float(arr.shape[1]),
        "plan_height": float(arr.shape[0]),
        "plan_aspect": float(max(arr.shape[1] / max(arr.shape[0], 1), arr.shape[0] / max(arr.shape[1], 1))),
        "plan_edge_density": edge_density,
        "plan_dark_ratio": float(black.mean()),
    }
    for prefix, mask in [("red", red), ("blue", blue), ("black", black)]:
        feats = component_features(mask)
        for key, value in feats.items():
            out[f"plan_{prefix}_{key}"] = value
    return out


def aggregate_plan_features(paths: list[Path]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "plan_available": bool(paths),
        "plan_crop_count": len(paths),
        "plan_best_crop": "",
    }
    if not paths:
        return base
    rows: list[dict[str, float]] = []
    for path in paths:
        try:
            feats = analyse_plan_image(path)
        except Exception:
            continue
        if feats:
            feats["path_index"] = float(len(rows))
            rows.append(feats)
    if not rows:
        return base
    frame = pd.DataFrame(rows)
    # The crop with the clearest red component is the best proxy for a target
    # planning boundary.  If no red exists, fall back to the line-rich crop.
    red_score = frame.get("plan_red_component_ratio", pd.Series(0.0, index=frame.index)) * 3.0
    red_score += frame.get("plan_red_component_bbox_ratio", pd.Series(0.0, index=frame.index))
    red_score += frame.get("plan_edge_density", pd.Series(0.0, index=frame.index))
    best_idx = int(red_score.idxmax())
    best_path_idx = int(frame.loc[best_idx, "path_index"])
    base["plan_best_crop"] = str(paths[best_path_idx])
    for column in frame.columns:
        if column == "path_index":
            continue
        base[f"{column}_max"] = float(frame[column].max())
        base[f"{column}_mean"] = float(frame[column].mean())
        base[f"{column}_best"] = float(frame.loc[best_idx, column])
    base["plan_red_clear"] = bool(base.get("plan_red_component_ratio_best", 0.0) >= 0.002)
    base["plan_blue_clear"] = bool(base.get("plan_blue_component_ratio_best", 0.0) >= 0.002)
    return base


def build_case_plan_features(
    case_keys: set[str],
    full_gpkg: Path,
    full_layer: str,
    plan_root: Path,
    cache_csv: Path | None,
    force: bool,
) -> pd.DataFrame:
    if cache_csv and cache_csv.exists() and not force:
        cached = pd.read_csv(cache_csv, dtype={"base_key": str, "case_key": str})
        return cached[cached["case_key"].isin(case_keys)].copy()

    full = gpd.read_file(full_gpkg, layer=full_layer, columns=["unique_key", "FilePath"])
    full["base_key"] = full["unique_key"].astype(str)
    full["mfd_ref"] = full["FilePath"].map(parse_mfd_ref)
    ref_by_base = dict(zip(full["base_key"], full["mfd_ref"]))

    rows: list[dict[str, Any]] = []
    for i, case_key in enumerate(sorted(case_keys), start=1):
        base_key = str(case_key).split("_", 1)[0]
        ref = ref_by_base.get(base_key, "")
        paths = sorted((plan_root / ref).glob("*.jpg")) if ref else []
        row = {
            "case_key": case_key,
            "base_key": base_key,
            "mfd_ref": ref,
        }
        row.update(aggregate_plan_features(paths))
        rows.append(row)
        if i % 250 == 0:
            print(f"plan features: {i}/{len(case_keys)} cases")
    out = pd.DataFrame(rows)
    if cache_csv:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_csv, index=False)
    return out


def geometry_descriptors(geom: Any) -> dict[str, float]:
    if geom is None or geom.is_empty:
        return {}
    minx, miny, maxx, maxy = geom.bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    area = max(float(geom.area), 0.0)
    perimeter = max(float(geom.length), 1e-6)
    bbox_area = width * height
    parts = float(len(getattr(geom, "geoms", [geom])))
    return {
        "geom_area": area,
        "geom_perimeter": perimeter,
        "geom_bbox_width": width,
        "geom_bbox_height": height,
        "geom_bbox_area": bbox_area,
        "geom_aspect": max(width / height, height / width),
        "geom_fill": area / bbox_area if bbox_area else 0.0,
        "geom_compactness": (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter else 0.0,
        "geom_parts": parts,
    }


def source_ids(df: pd.DataFrame, source: str) -> list[int]:
    values = pd.to_numeric(df.loc[df["candidate_source"] == source, "candidate_objectid"], errors="coerce")
    return sorted({int(v) for v in values.dropna().tolist()})


def read_wfs_descriptors(path: Path, layer: str, ids: set[int], source: str) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    print(f"reading {source} full layer for {len(ids)} candidate ids")
    gdf = pyogrio.read_dataframe(path, layer=layer, columns=["OBJECTID", "geometry"])
    gdf = gdf[gdf["OBJECTID"].isin(ids)].copy()
    rows = []
    for _, row in gdf.iterrows():
        item = {"candidate_source": source, "candidate_objectid": int(row["OBJECTID"])}
        item.update(geometry_descriptors(row.geometry))
        rows.append(item)
    return pd.DataFrame(rows)


def read_council_descriptors(path: Path, ids: list[int], chunk_size: int) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        where = "LABEL IN (" + ",".join(repr(str(v)) for v in chunk) + ")"
        gdf = pyogrio.read_dataframe(path, layer="cadastral_parcels", columns=["LABEL", "geometry"], where=where)
        for _, row in gdf.iterrows():
            item = {"candidate_source": "council_cadastral", "candidate_objectid": int(row["LABEL"])}
            item.update(geometry_descriptors(row.geometry))
            rows.append(item)
        print(f"council geometry: {min(start + chunk_size, len(ids))}/{len(ids)} ids")
    return pd.DataFrame(rows)


def build_candidate_shape_features(
    candidates: pd.DataFrame,
    cache_csv: Path | None,
    force: bool,
    max_rank: int,
    council_gpkg: Path,
    wfs_raw_gpkg: Path,
    wfs_merged_gpkg: Path,
    council_chunk_size: int,
) -> pd.DataFrame:
    if cache_csv and cache_csv.exists() and not force:
        return pd.read_csv(cache_csv, dtype={"candidate_source": str})

    subset = candidates[pd.to_numeric(candidates["rule_rank"], errors="coerce") <= max_rank].copy()
    frames: list[pd.DataFrame] = []
    council_ids = source_ids(subset, "council_cadastral")
    raw_ids = set(source_ids(subset, "wfs_raw"))
    merged_ids = set(source_ids(subset, "wfs_merged"))
    frames.append(read_wfs_descriptors(wfs_raw_gpkg, "mansfield_polygons_in_buffers", raw_ids, "wfs_raw"))
    frames.append(read_wfs_descriptors(wfs_merged_gpkg, "mansfield_polygons_in_buffers_merged", merged_ids, "wfs_merged"))
    frames.append(read_council_descriptors(council_gpkg, council_ids, council_chunk_size))
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates(["candidate_source", "candidate_objectid"])
    if cache_csv:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_csv, index=False)
    return out


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(0.0, index=df.index)
    return df[column].astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_plan_shape_features(df: pd.DataFrame, train_mask: pd.Series, max_category_levels: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_features, _ = selector13.build_features(df, train_mask=train_mask, max_levels=max_category_levels)
    features = base_features.copy()

    # Geometry descriptors.  Some council geometries may be missing if the
    # source layer cannot be queried; fall back to candidate_area_m2 in that case.
    geom_area = numeric_series(df, "geom_area")
    candidate_area = numeric_series(df, "candidate_area_m2")
    area = geom_area.mask(geom_area <= 0, candidate_area)
    geom_aspect = numeric_series(df, "geom_aspect", 1.0).clip(lower=1.0, upper=100.0)
    geom_fill = numeric_series(df, "geom_fill").clip(lower=0.0, upper=1.0)
    geom_compactness = numeric_series(df, "geom_compactness").clip(lower=0.0, upper=1.0)

    plan_available = bool_series(df, "plan_available")
    plan_red_clear = bool_series(df, "plan_red_clear")
    plan_blue_clear = bool_series(df, "plan_blue_clear")
    plan_crop_count = numeric_series(df, "plan_crop_count")
    plan_red_aspect = numeric_series(df, "plan_red_component_aspect_best", 1.0).clip(lower=1.0, upper=100.0)
    plan_red_fill = numeric_series(df, "plan_red_component_fill_best").clip(lower=0.0, upper=1.0)
    plan_red_bbox = numeric_series(df, "plan_red_component_bbox_ratio_best").clip(lower=0.0, upper=1.0)
    plan_red_ratio = numeric_series(df, "plan_red_ratio_best").clip(lower=0.0, upper=1.0)
    plan_blue_ratio = numeric_series(df, "plan_blue_ratio_best").clip(lower=0.0, upper=1.0)
    plan_dark_ratio = numeric_series(df, "plan_dark_ratio_best").clip(lower=0.0, upper=1.0)
    plan_edge = numeric_series(df, "plan_edge_density_best").clip(lower=0.0, upper=1.0)

    source = df.get("candidate_source", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    theme = df.get("candidate_theme", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    group = df.get("candidate_descriptive_group", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    is_council = source.eq("council_cadastral").astype(float)
    is_wfs_merged = source.eq("wfs_merged").astype(float)
    is_wfs_raw = source.eq("wfs_raw").astype(float)
    is_land = theme.str.contains("land", regex=False).astype(float)
    is_building = (theme.str.contains("building", regex=False) | group.str.contains("building", regex=False)).astype(float)
    covers_best = bool_series(df, "covers_best_point")
    covers_range = bool_series(df, "covers_range_anchor")

    plan_extra = pd.DataFrame(index=df.index)
    plan_extra["plan_available"] = plan_available
    plan_extra["plan_crop_count_log"] = np.log1p(plan_crop_count)
    plan_extra["plan_red_clear"] = plan_red_clear
    plan_extra["plan_blue_clear"] = plan_blue_clear
    plan_extra["plan_red_ratio"] = plan_red_ratio
    plan_extra["plan_blue_ratio"] = plan_blue_ratio
    plan_extra["plan_dark_ratio"] = plan_dark_ratio
    plan_extra["plan_edge_density"] = plan_edge
    plan_extra["plan_red_bbox_ratio"] = plan_red_bbox
    plan_extra["plan_red_aspect_log"] = np.log1p(plan_red_aspect)
    plan_extra["plan_red_fill"] = plan_red_fill
    plan_extra["geom_aspect_log"] = np.log1p(geom_aspect)
    plan_extra["geom_fill"] = geom_fill
    plan_extra["geom_compactness"] = geom_compactness
    plan_extra["geom_area_log"] = np.log1p(area.clip(lower=0.0))
    plan_extra["geom_perimeter_log"] = np.log1p(numeric_series(df, "geom_perimeter").clip(lower=0.0))
    plan_extra["geom_shape_missing"] = (geom_area <= 0).astype(float)

    aspect_diff = np.abs(np.log1p(geom_aspect) - np.log1p(plan_red_aspect))
    fill_diff = np.abs(geom_fill - plan_red_fill)
    plan_extra["plan_red_geom_aspect_diff"] = plan_available * plan_red_clear * aspect_diff
    plan_extra["plan_red_geom_fill_diff"] = plan_available * plan_red_clear * fill_diff
    plan_extra["plan_red_bbox_x_area_log"] = plan_available * plan_red_bbox * np.log1p(area.clip(lower=0.0))
    plan_extra["plan_red_clear_x_council"] = plan_red_clear * is_council
    plan_extra["plan_red_clear_x_wfs_merged"] = plan_red_clear * is_wfs_merged
    plan_extra["plan_red_clear_x_wfs_raw"] = plan_red_clear * is_wfs_raw
    plan_extra["plan_red_clear_x_land"] = plan_red_clear * is_land
    plan_extra["plan_red_clear_x_building"] = plan_red_clear * is_building
    plan_extra["plan_red_clear_x_covers_best"] = plan_red_clear * covers_best
    plan_extra["plan_red_clear_x_covers_range"] = plan_red_clear * covers_range
    plan_extra["plan_blue_clear_x_building"] = plan_blue_clear * is_building
    plan_extra["plan_line_rich_x_compactness"] = plan_available * plan_edge * geom_compactness
    plan_extra = plan_extra.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    combined = pd.concat([base_features, plan_extra], axis=1)
    return base_features, combined


def train_oof(
    df: pd.DataFrame,
    features: pd.DataFrame,
    folds: int,
    seed: int,
    max_rank_for_top_neg: int,
) -> np.ndarray:
    case_keys = sorted(df["case_key"].unique())
    scores = np.full(len(df), np.nan, dtype="float64")
    x = features.to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype=bool)
    for fold in range(folds):
        train_cases = {key for key in case_keys if fold_for_case(key, folds) != fold}
        test_cases = {key for key in case_keys if fold_for_case(key, folds) == fold}
        train_mask = df["case_key"].isin(train_cases).to_numpy()
        test_mask = df["case_key"].isin(test_cases).to_numpy()
        rng = np.random.default_rng(seed + fold)
        train_idx = np.flatnonzero(train_mask)
        pos_idx = train_idx[y[train_idx]]
        train_df = df.iloc[train_idx]
        top_neg = train_df[(~train_df["label"]) & (pd.to_numeric(train_df["rule_rank"], errors="coerce") <= max_rank_for_top_neg)].index.to_numpy()
        other_neg = train_df[(~train_df["label"]) & (pd.to_numeric(train_df["rule_rank"], errors="coerce") > max_rank_for_top_neg)].index.to_numpy()
        sampled_other = rng.choice(other_neg, size=min(len(other_neg), max(len(pos_idx) * 6, 1)), replace=False)
        fit_idx = np.unique(np.concatenate([pos_idx, top_neg, sampled_other]))
        clf = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.045,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=seed + 100 + fold,
        )
        clf.fit(x[fit_idx], y[fit_idx])
        scores[test_mask] = clf.predict_proba(x[test_mask])[:, 1]
        print(f"fold {fold}: train_cases={len(train_cases)} test_cases={len(test_cases)} fit_rows={len(fit_idx)}")
    return scores


def rank_by_score(df: pd.DataFrame, score_column: str, rank_column: str) -> None:
    ranked = df.sort_values(["case_key", score_column, "rule_rank"], ascending=[True, False, True])
    df[rank_column] = ranked.groupby("case_key").cumcount().add(1).reindex(df.index)


def metric_for_rank(df: pd.DataFrame, rank_column: str, ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": int(df["case_key"].nunique())}
    for k in ks:
        hits = df[(pd.to_numeric(df[rank_column], errors="coerce") <= k) & df["label"]].groupby("case_key").size()
        count = int((hits > 0).sum())
        out[f"top{k}"] = count
        out[f"top{k}_rate"] = float(count / out["cases"]) if out["cases"] else 0.0
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_prefix: Path = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    df["rule_rank"] = pd.to_numeric(df["rule_rank"], errors="coerce")
    df = df[df["rule_rank"] <= args.max_rank].copy().reset_index(drop=True)
    df["candidate_objectid"] = pd.to_numeric(df["candidate_objectid"], errors="coerce")
    df["label"] = selector13.to_bool(df["truth_intersects"])
    case_keys = set(df["case_key"].astype(str))

    plan_cache = out_prefix.with_name(out_prefix.name + f"_plan_features_cache_rank{args.max_rank}.csv")
    shape_cache = out_prefix.with_name(out_prefix.name + f"_candidate_shape_cache_rank{args.max_rank}.csv")
    plan = build_case_plan_features(
        case_keys=case_keys,
        full_gpkg=args.full_gpkg,
        full_layer=args.full_layer,
        plan_root=args.plan_root,
        cache_csv=plan_cache,
        force=args.force_rebuild_features,
    )
    shapes = build_candidate_shape_features(
        candidates=df,
        cache_csv=shape_cache,
        force=args.force_rebuild_features,
        max_rank=args.max_rank,
        council_gpkg=args.council_gpkg,
        wfs_raw_gpkg=args.wfs_raw_gpkg,
        wfs_merged_gpkg=args.wfs_merged_gpkg,
        council_chunk_size=args.council_chunk_size,
    )

    df = df.merge(plan, on=["case_key", "base_key"], how="left")
    if not shapes.empty:
        shapes["candidate_objectid"] = pd.to_numeric(shapes["candidate_objectid"], errors="coerce")
        df = df.merge(shapes, on=["candidate_source", "candidate_objectid"], how="left")
    df["plan_available"] = df["plan_available"].fillna(False).astype(bool)
    if "plan_red_clear" in df:
        df["plan_red_clear"] = df["plan_red_clear"].fillna(False).astype(bool)
    else:
        df["plan_red_clear"] = False
    if "plan_blue_clear" in df:
        df["plan_blue_clear"] = df["plan_blue_clear"].fillna(False).astype(bool)
    else:
        df["plan_blue_clear"] = False

    train_mask = pd.Series(True, index=df.index)
    base_features, plan_features = build_plan_shape_features(df, train_mask, args.max_category_levels)
    print(f"rows={len(df)} cases={df['case_key'].nunique()} plan_available_cases={df[df['plan_available'].astype(bool)]['case_key'].nunique()}")
    print(f"base_features={base_features.shape[1]} plan_features={plan_features.shape[1]}")

    df["base_oof_score"] = train_oof(df, base_features, args.folds, args.seed, args.top_neg_rank)
    df["plan_oof_score"] = train_oof(df, plan_features, args.folds, args.seed + 1000, args.top_neg_rank)
    rank_by_score(df, "base_oof_score", "base_oof_rank")
    rank_by_score(df, "plan_oof_score", "plan_oof_rank")

    ks = [1, 3, 5, 8, 10, 20, 50, 100, args.max_rank]
    ks = sorted(set(k for k in ks if k <= args.max_rank))
    summary: dict[str, Any] = {
        "input_csv": str(args.input_csv),
        "output_prefix": str(out_prefix),
        "max_rank": args.max_rank,
        "folds": args.folds,
        "rows": int(len(df)),
        "cases": int(df["case_key"].nunique()),
        "plan_available_cases": int(df[df["plan_available"].astype(bool)]["case_key"].nunique()),
        "plan_red_clear_cases": int(df[df["plan_red_clear"]]["case_key"].nunique()),
        "baseline_rule": metric_for_rank(df, "rule_rank", ks),
        "base_model": metric_for_rank(df, "base_oof_rank", ks),
        "plan_model": metric_for_rank(df, "plan_oof_rank", ks),
    }
    for name, subset in [
        ("plan_available", df[df["plan_available"].astype(bool)].copy()),
        ("plan_missing", df[~df["plan_available"].astype(bool)].copy()),
        ("plan_red_clear", df[df["plan_red_clear"]].copy()),
        ("low_conf_lt75", df[pd.to_numeric(df["best_confidence"], errors="coerce") < 75].copy()),
    ]:
        if subset.empty:
            continue
        summary[name] = {
            "baseline_rule": metric_for_rank(subset, "rule_rank", ks),
            "base_model": metric_for_rank(subset, "base_oof_rank", ks),
            "plan_model": metric_for_rank(subset, "plan_oof_rank", ks),
        }

    case_rows: list[dict[str, Any]] = []
    for case_key, group in df.groupby("case_key"):
        base_top5 = bool(group[(group["base_oof_rank"] <= 5) & group["label"]].shape[0])
        plan_top5 = bool(group[(group["plan_oof_rank"] <= 5) & group["label"]].shape[0])
        rule_top5 = bool(group[(group["rule_rank"] <= 5) & group["label"]].shape[0])
        case_rows.append(
            {
                "case_key": case_key,
                "sample_split": group["sample_split"].iloc[0],
                "best_confidence": group["best_confidence"].iloc[0],
                "plan_available": bool(group["plan_available"].iloc[0]),
                "plan_red_clear": bool(group.get("plan_red_clear", pd.Series([False])).iloc[0]),
                "mfd_ref": group.get("mfd_ref", pd.Series([""])).iloc[0],
                "plan_best_crop": group.get("plan_best_crop", pd.Series([""])).iloc[0],
                "rule_top5_hit": rule_top5,
                "base_model_top5_hit": base_top5,
                "plan_model_top5_hit": plan_top5,
                "plan_recovered_vs_base_top5": plan_top5 and not base_top5,
                "plan_lost_vs_base_top5": base_top5 and not plan_top5,
                "plan_recovered_vs_rule_top5": plan_top5 and not rule_top5,
                "plan_lost_vs_rule_top5": rule_top5 and not plan_top5,
            }
        )
    case_df = pd.DataFrame(case_rows)
    summary["plan_vs_base_top5_recovered"] = int(case_df["plan_recovered_vs_base_top5"].sum())
    summary["plan_vs_base_top5_lost"] = int(case_df["plan_lost_vs_base_top5"].sum())
    summary["plan_vs_rule_top5_recovered"] = int(case_df["plan_recovered_vs_rule_top5"].sum())
    summary["plan_vs_rule_top5_lost"] = int(case_df["plan_lost_vs_rule_top5"].sum())

    summary_json = out_prefix.with_suffix(".summary.json")
    scored_csv = out_prefix.with_name(out_prefix.name + "_scored_top.csv")
    case_csv = out_prefix.with_name(out_prefix.name + "_case_metrics.csv")
    cols = [
        "case_key",
        "base_key",
        "sample_split",
        "original_address",
        "best_confidence",
        "candidate_source",
        "candidate_id",
        "candidate_theme",
        "candidate_descriptive_group",
        "candidate_area_m2",
        "truth_intersects",
        "rule_score",
        "rule_rank",
        "base_oof_score",
        "base_oof_rank",
        "plan_oof_score",
        "plan_oof_rank",
        "plan_available",
        "plan_red_clear",
        "mfd_ref",
        "plan_best_crop",
        "geom_aspect",
        "geom_fill",
        "geom_compactness",
    ]
    cols = [c for c in cols if c in df.columns]
    df.sort_values(["case_key", "plan_oof_rank"]).groupby("case_key").head(args.output_top_n)[cols].to_csv(scored_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_json={summary_json}")
    print(f"scored_csv={scored_csv}")
    print(f"case_csv={case_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-API plan visual feature polygon reranking experiment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--full-gpkg", type=Path, default=DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=DEFAULT_FULL_LAYER)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--council-gpkg", type=Path, default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--wfs-raw-gpkg", type=Path, default=DEFAULT_WFS_RAW_GPKG)
    parser.add_argument("--wfs-merged-gpkg", type=Path, default=DEFAULT_WFS_MERGED_GPKG)
    parser.add_argument("--output-top-n", type=int, default=20)
    parser.add_argument("--max-rank", type=int, default=100)
    parser.add_argument("--top-neg-rank", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-category-levels", type=int, default=20)
    parser.add_argument("--council-chunk-size", type=int, default=500)
    parser.add_argument("--force-rebuild-features", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
