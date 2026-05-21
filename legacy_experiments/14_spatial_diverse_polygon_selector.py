#!/usr/bin/env python3
"""Spatially diverse Top-K selector for Mansfield polygon candidates.

The hybrid ranker can spend several Top5 slots on near-duplicate polygons around
the same address point.  This post-processor keeps the original rule order but
skips candidates whose centroids are within a small distance of an already
selected candidate, then fills any remaining slots with the original order.

No AI/API calls are made.  Truth columns, when present, are used only for offline
evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "tmp_results" / "mansfield_random1200_hybrid_top200_v1_top200.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "tmp_results" / "mansfield_polygon_selector_spatial_diverse_v1"


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def select_group(group: pd.DataFrame, top_k: int, min_centroid_distance_m: float) -> list[int]:
    selected: list[int] = []
    selected_centroids: list[tuple[float, float]] = []
    ordered = group.sort_values(["rule_rank", "rule_score"], ascending=[True, False])

    for index, row in ordered.iterrows():
        if len(selected) >= top_k:
            break
        x = row.get("candidate_centroid_easting")
        y = row.get("candidate_centroid_northing")
        if pd.isna(x) or pd.isna(y):
            continue
        too_close = any(math.hypot(float(x) - px, float(y) - py) < min_centroid_distance_m for px, py in selected_centroids)
        if too_close:
            continue
        selected.append(index)
        selected_centroids.append((float(x), float(y)))

    # Preserve recall if a case has fewer than top_k spatially distinct rows.
    if len(selected) < top_k:
        for index, _ in ordered.iterrows():
            if len(selected) >= top_k:
                break
            if index not in selected:
                selected.append(index)
    return selected[:top_k]


def evaluate(selected: pd.DataFrame, all_cases: pd.Index, prefix: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"label": prefix, "cases": int(len(all_cases))}
    if "truth_intersects" not in selected:
        return summary
    selected = selected.copy()
    selected["hit"] = to_bool(selected["truth_intersects"])
    hits = selected.groupby("case_key")["hit"].any().reindex(all_cases, fill_value=False)
    summary["topk"] = int(hits.sum())
    summary["topk_rate"] = float(hits.mean())

    case_meta = selected.groupby("case_key").first(numeric_only=False).reindex(all_cases)
    if "sample_split" in case_meta:
        for split, split_cases in case_meta.groupby("sample_split", dropna=False):
            split_hits = hits.reindex(split_cases.index)
            summary[f"split_{split}"] = {
                "cases": int(len(split_cases)),
                "topk": int(split_hits.sum()),
                "topk_rate": float(split_hits.mean()),
            }
    if "best_confidence" in case_meta:
        conf = pd.to_numeric(case_meta["best_confidence"], errors="coerce")
        for name, mask in {
            "low_conf_lt75": conf < 75,
            "mid_conf_75_80": (conf >= 75) & (conf < 80),
            "high_conf_ge80": conf >= 80,
        }.items():
            split_hits = hits[mask.fillna(False)]
            summary[name] = {
                "cases": int(mask.fillna(False).sum()),
                "topk": int(split_hits.sum()),
                "topk_rate": float(split_hits.mean()) if len(split_hits) else 0.0,
            }
    return summary


def baseline_topk(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df[df["rule_rank"] <= top_k].copy()


def run(input_csv: Path, output_prefix: Path, top_k: int, min_centroid_distance_m: float) -> dict[str, Any]:
    df = pd.read_csv(input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    for column in [
        "rule_rank",
        "rule_score",
        "candidate_centroid_easting",
        "candidate_centroid_northing",
        "candidate_area_m2",
        "best_confidence",
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    all_cases = pd.Index(sorted(df["case_key"].unique(), key=lambda x: (len(str(x)), str(x))), name="case_key")
    selected_indices: list[int] = []
    for _, group in df.groupby("case_key", sort=False):
        selected_indices.extend(select_group(group, top_k=top_k, min_centroid_distance_m=min_centroid_distance_m))

    selected = df.loc[selected_indices].copy()
    selected = selected.sort_values(["case_key", "rule_rank"]).copy()
    selected["diverse_rank"] = selected.groupby("case_key").cumcount() + 1

    baseline = baseline_topk(df, top_k=top_k)
    summary = {
        "input_csv": str(input_csv),
        "output_prefix": str(output_prefix),
        "top_k": int(top_k),
        "min_centroid_distance_m": float(min_centroid_distance_m),
        "candidate_rows": int(len(df)),
        "cases": int(len(all_cases)),
        "baseline_rule": evaluate(baseline, all_cases, f"baseline_rule_top{top_k}"),
        "spatial_diverse": evaluate(selected, all_cases, f"spatial_diverse_top{top_k}"),
    }

    if "truth_intersects" in selected:
        baseline_hits = (
            to_bool(baseline["truth_intersects"]).groupby(baseline["case_key"]).any().reindex(all_cases, fill_value=False)
        )
        diverse_hits = (
            to_bool(selected["truth_intersects"]).groupby(selected["case_key"]).any().reindex(all_cases, fill_value=False)
        )
        summary["recovered_cases"] = int((diverse_hits & ~baseline_hits).sum())
        summary["lost_cases"] = int((baseline_hits & ~diverse_hits).sum())
        summary["net_gain_cases"] = int(diverse_hits.sum() - baseline_hits.sum())
        summary["recovered_case_keys"] = sorted((diverse_hits & ~baseline_hits)[lambda s: s].index.astype(str).tolist())
        summary["lost_case_keys"] = sorted((baseline_hits & ~diverse_hits)[lambda s: s].index.astype(str).tolist())

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    selected_csv = output_prefix.with_name(output_prefix.name + f"_top{top_k}.csv")
    summary_json = output_prefix.with_suffix(".summary.json")
    selected.to_csv(selected_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"selected_csv={selected_csv}")
    print(f"summary_json={summary_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatially diverse Top-K Mansfield polygon selector.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-centroid-distance-m", type=float, default=2.0)
    args = parser.parse_args()
    run(args.input_csv, args.output_prefix, args.top_k, args.min_centroid_distance_m)


if __name__ == "__main__":
    main()
