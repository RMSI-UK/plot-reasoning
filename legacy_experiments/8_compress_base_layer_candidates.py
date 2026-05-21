#!/usr/bin/env python3
"""Compress WFS base-layer candidates into a small production review pool.

Input is the production-visible WFS/UPRN candidate table produced by
`8_candidate_base_layer_rerank_experiment.py`.  This script keeps only the top
N candidates per case by the existing rule score, and reports offline metrics
against the manual truth layer labels already stored in that table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")
DEFAULT_TAG = "mansfield-manual-polygon-link_random1200_seed42_43_combined"
DEFAULT_INPUT = TMP / f"{DEFAULT_TAG}_base_layer_rerank_v2_theme_land_building_candidates.csv"
DEFAULT_OUTPUT_PREFIX = TMP / f"{DEFAULT_TAG}_base_layer_rerank_v3_compressed_top50"


SORT_COLS = [
    "case_key",
    "rule_score",
    "distance_to_v7_point_m",
    "distance_to_best_point_m",
    "candidate_area_m2",
    "candidate_id",
]
SORT_ASC = [True, False, True, True, True, True]


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def summarize(cases: pd.DataFrame, label: str) -> dict[str, Any]:
    if cases.empty:
        return {"cases": 0}
    rank_col = f"{label}_rank"
    present_col = f"{label}_present"
    rank = pd.to_numeric(cases[rank_col], errors="coerce")
    present = bool_series(cases[present_col])
    return {
        "cases": int(len(cases)),
        "present": int(present.sum()),
        "present_rate": float(present.mean()),
        "top1": int((rank == 1).sum()),
        "top1_rate": float((rank == 1).sum() / len(cases)),
        "top3": int((rank <= 3).sum()),
        "top3_rate": float((rank <= 3).sum() / len(cases)),
        "top5": int((rank <= 5).sum()),
        "top5_rate": float((rank <= 5).sum() / len(cases)),
        "top10": int((rank <= 10).sum()),
        "top10_rate": float((rank <= 10).sum() / len(cases)),
        "top20": int((rank <= 20).sum()),
        "top20_rate": float((rank <= 20).sum() / len(cases)),
        "top50": int((rank <= 50).sum()),
        "top50_rate": float((rank <= 50).sum() / len(cases)),
        "median_original_candidate_count": float(pd.to_numeric(cases["original_candidate_count"], errors="coerce").median()),
        "median_compressed_candidate_count": float(pd.to_numeric(cases["compressed_candidate_count"], errors="coerce").median()),
        "mean_original_candidate_count": float(pd.to_numeric(cases["original_candidate_count"], errors="coerce").mean()),
        "mean_compressed_candidate_count": float(pd.to_numeric(cases["compressed_candidate_count"], errors="coerce").mean()),
    }


def build_case_summary(df: pd.DataFrame, compressed: pd.DataFrame, max_rank: int) -> pd.DataFrame:
    original_counts = df.groupby("case_key").size().rename("original_candidate_count")
    compressed_counts = compressed.groupby("case_key").size().rename("compressed_candidate_count")

    truth_intersects = df[df["truth_intersects"]].groupby("case_key")["candidate_rank"].min().rename("intersects_rank")
    truth_centroid = df[df["truth_centroid_inside"]].groupby("case_key")["candidate_rank"].min().rename("centroid_rank")
    top1 = df[df["candidate_rank"] == 1].set_index("case_key")

    base = pd.DataFrame(index=original_counts.index)
    base = base.join(original_counts).join(compressed_counts)
    base["compressed_candidate_count"] = base["compressed_candidate_count"].fillna(0).astype(int)
    base = base.join(truth_intersects).join(truth_centroid)
    base["intersects_present"] = base["intersects_rank"].notna()
    base["centroid_present"] = base["centroid_rank"].notna()
    base["intersects_in_compressed"] = base["intersects_rank"].le(max_rank).fillna(False)
    base["centroid_in_compressed"] = base["centroid_rank"].le(max_rank).fillna(False)
    base["top1_candidate_id"] = top1["candidate_id"]
    base["top1_theme"] = top1["candidate_theme"]
    base["top1_group"] = top1["candidate_descriptive_group"]
    base["top1_score"] = top1["rule_score"]
    base["top1_intersects_truth"] = top1["truth_intersects"]
    base["top1_centroid_inside_truth"] = top1["truth_centroid_inside"]
    base["top1_covers_v7_point"] = top1["covers_v7_point"]
    base["top1_covers_best_point"] = top1["covers_best_point"]
    base["top1_uprn_count"] = top1["uprn_count"]

    meta_cols = [
        "base_key",
        "sample_split",
        "original_address",
        "best_confidence",
        "has_range_anchor",
        "range_anchor_relation",
        "range_anchor_easting",
        "range_anchor_northing",
        "range_anchor_covered_numbers",
        "range_anchor_endpoint_hits",
        "range_anchor_complete",
        "range_anchor_top_addresses",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]
    meta = df.groupby("case_key")[meta_cols].first()
    base = base.join(meta)
    base = base.reset_index().rename(columns={"case_key": "key"})
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-candidates", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-rank", type=int, default=50)
    args = parser.parse_args()

    print(f"loading {args.input_candidates}")
    df = pd.read_csv(args.input_candidates, dtype={"case_key": str, "base_key": str, "candidate_id": str}, engine="python")
    for col in [
        "rule_score",
        "distance_to_v7_point_m",
        "distance_to_best_point_m",
        "candidate_area_m2",
        "best_confidence",
        "uprn_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["truth_intersects", "truth_centroid_inside", "covers_v7_point", "covers_best_point"]:
        df[col] = bool_series(df[col])

    df = df.sort_values(SORT_COLS, ascending=SORT_ASC).copy()
    df["candidate_rank"] = df.groupby("case_key").cumcount() + 1
    compressed = df[df["candidate_rank"] <= args.max_rank].copy()
    top20 = df[df["candidate_rank"] <= min(20, args.max_rank)].copy()

    cases = build_case_summary(df, compressed, args.max_rank)
    low = cases[pd.to_numeric(cases["best_confidence"], errors="coerce") < 75].copy()
    high = cases[pd.to_numeric(cases["best_confidence"], errors="coerce") >= 80].copy()

    prefix = args.output_prefix
    cases_csv = prefix.with_name(prefix.name + "_cases.csv")
    top20_csv = prefix.with_name(prefix.name + "_top20.csv")
    topn_csv = prefix.with_name(prefix.name + f"_top{args.max_rank}.csv")
    xlsx_path = prefix.with_suffix(".xlsx")
    summary_path = prefix.with_suffix(".summary.json")

    cases.to_csv(cases_csv, index=False)
    top20.to_csv(top20_csv, index=False)
    compressed.to_csv(topn_csv, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        cases.to_excel(writer, sheet_name="cases", index=False)
        top20.to_excel(writer, sheet_name="top20", index=False)

    summary = {
        "source_candidates": str(args.input_candidates),
        "max_rank": args.max_rank,
        "all_cases_intersects": summarize(cases, "intersects"),
        "all_cases_centroid": summarize(cases, "centroid"),
        "low_confidence_lt75_intersects": summarize(low, "intersects"),
        "high_confidence_ge80_intersects": summarize(high, "intersects"),
        "output_cases_csv": str(cases_csv),
        "output_top20_csv": str(top20_csv),
        "output_topn_csv": str(topn_csv),
        "output_xlsx": str(xlsx_path),
        "compressed_rows": int(len(compressed)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
