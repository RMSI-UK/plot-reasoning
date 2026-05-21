#!/usr/bin/env python3
"""HGB out-of-fold experiment for Mansfield polygon selection.

This is an offline experiment, not a production dependency yet.  It trains a
HistGradientBoostingClassifier on production-visible candidate features and uses
case-level out-of-fold predictions to estimate generalization.

The experiment intentionally separates two jobs:
- HGB top1: choose one best polygon.
- spatial-diverse top5: keep high recall for review/second-stage validation.

Truth columns are used only as labels/evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
FEATURE_SCRIPT = ROOT / "13_train_polygon_selector.py"
DEFAULT_INPUT = ROOT / "tmp_results" / "mansfield_random1200_hybrid_top200_v1_top200.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "tmp_results" / "mansfield_polygon_selector_hgb_oof_combo_v1"

spec = importlib.util.spec_from_file_location("selector13_features", FEATURE_SCRIPT)
selector13 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["selector13_features"] = selector13
spec.loader.exec_module(selector13)


def fold_for_case(case_key: str, folds: int) -> int:
    return int(hashlib.md5(str(case_key).encode("utf-8")).hexdigest()[:8], 16) % folds


def train_fold(
    df: pd.DataFrame,
    train_cases: set[str],
    test_cases: set[str],
    fold_index: int,
    max_category_levels: int,
    seed: int,
) -> np.ndarray:
    train_mask = df["case_key"].isin(train_cases)
    features, _ = selector13.build_features(df, train_mask=train_mask, max_levels=max_category_levels)
    x = features.to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype=bool)

    rng = np.random.default_rng(seed + fold_index)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    pos_idx = train_idx[y[train_idx]]
    train_df = df.iloc[train_idx]
    top_neg = train_df[(~train_df["label"]) & (train_df["rule_rank"] <= 60)].index.to_numpy()
    other_neg = train_df[(~train_df["label"]) & (train_df["rule_rank"] > 60)].index.to_numpy()
    sampled_other = rng.choice(other_neg, size=min(len(other_neg), len(pos_idx) * 8), replace=False)
    fit_idx = np.unique(np.concatenate([pos_idx, top_neg, sampled_other]))

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=seed + 100 + fold_index,
    )
    clf.fit(x[fit_idx], y[fit_idx])
    test_mask = df["case_key"].isin(test_cases).to_numpy()
    scores = np.full(len(df), np.nan, dtype="float64")
    scores[test_mask] = clf.predict_proba(x[test_mask])[:, 1]
    return scores


def diverse_indices(group: pd.DataFrame, threshold_m: float, top_k: int) -> list[int]:
    selected: list[int] = []
    centroids: list[tuple[float, float]] = []
    ordered = group.sort_values(["rule_rank", "rule_score"], ascending=[True, False])
    for index, row in ordered.iterrows():
        if len(selected) >= top_k:
            break
        x = row.get("candidate_centroid_easting")
        y = row.get("candidate_centroid_northing")
        if pd.isna(x) or pd.isna(y):
            continue
        if any(math.hypot(float(x) - px, float(y) - py) < threshold_m for px, py in centroids):
            continue
        selected.append(index)
        centroids.append((float(x), float(y)))
    if len(selected) < top_k:
        for index, _ in ordered.iterrows():
            if len(selected) >= top_k:
                break
            if index not in selected:
                selected.append(index)
    return selected[:top_k]


def hit_rate(df: pd.DataFrame, column: str) -> dict[str, Any]:
    hits = df.groupby("case_key")[column].first()
    return {"hits": int(hits.sum()), "rate": float(hits.mean())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(args.input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    df = df[pd.to_numeric(df["rule_rank"], errors="coerce") <= args.max_rank].copy().reset_index(drop=True)
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
    df["label"] = selector13.to_bool(df["truth_intersects"])

    case_keys = sorted(df["case_key"].unique())
    oof = np.full(len(df), np.nan, dtype="float64")
    fold_logs: list[dict[str, Any]] = []
    for fold_index in range(args.folds):
        train_cases = {key for key in case_keys if fold_for_case(key, args.folds) != fold_index}
        test_cases = {key for key in case_keys if fold_for_case(key, args.folds) == fold_index}
        fold_scores = train_fold(df, train_cases, test_cases, fold_index, args.max_category_levels, args.seed)
        oof[~np.isnan(fold_scores)] = fold_scores[~np.isnan(fold_scores)]
        fold_logs.append({"fold": fold_index, "train_cases": len(train_cases), "test_cases": len(test_cases)})
        print(f"fold {fold_index} complete: train={len(train_cases)} test={len(test_cases)}")

    df["hgb_oof_score"] = oof
    df["hgb_oof_rank"] = (
        df.sort_values(["case_key", "hgb_oof_score", "rule_rank"], ascending=[True, False, True])
        .groupby("case_key")
        .cumcount()
        + 1
    )

    case_rows: list[dict[str, Any]] = []
    combo_rows: list[pd.Series] = []
    for case_key, group in df.groupby("case_key", sort=False):
        baseline_top5 = group[group["rule_rank"] <= 5].index.tolist()
        diverse_top5 = diverse_indices(group, threshold_m=args.diverse_threshold_m, top_k=5)
        hgb_top1 = [group.sort_values(["hgb_oof_score", "rule_rank"], ascending=[False, True]).index[0]]
        combo: list[int] = []
        for index in hgb_top1 + diverse_top5:
            if index not in combo:
                combo.append(index)
            if len(combo) >= 5:
                break
        case_rows.append(
            {
                "case_key": case_key,
                "sample_split": group["sample_split"].iloc[0],
                "best_confidence": group["best_confidence"].iloc[0],
                "baseline_top1_hit": bool(group[(group["rule_rank"] <= 1) & group["label"]].shape[0]),
                "hgb_top1_hit": bool(df.loc[hgb_top1, "label"].any()),
                "baseline_top5_hit": bool(df.loc[baseline_top5, "label"].any()),
                "diverse_top5_hit": bool(df.loc[diverse_top5, "label"].any()),
                "combo_top5_hit": bool(df.loc[combo, "label"].any()),
            }
        )
        for rank, index in enumerate(combo, 1):
            row = df.loc[index].copy()
            row["combo_rank"] = rank
            combo_rows.append(row)

    cases = pd.DataFrame(case_rows)
    summary: dict[str, Any] = {
        "input_csv": str(args.input_csv),
        "output_prefix": str(args.output_prefix),
        "folds": args.folds,
        "fold_logs": fold_logs,
        "cases": int(len(cases)),
        "baseline_top1": hit_rate(cases, "baseline_top1_hit"),
        "hgb_top1": hit_rate(cases, "hgb_top1_hit"),
        "baseline_top5": hit_rate(cases, "baseline_top5_hit"),
        "diverse_top5": hit_rate(cases, "diverse_top5_hit"),
        "combo_top5": hit_rate(cases, "combo_top5_hit"),
    }
    for split, split_df in cases.groupby("sample_split", dropna=False):
        summary[f"split_{split}"] = {
            "cases": int(len(split_df)),
            "baseline_top1": hit_rate(split_df, "baseline_top1_hit"),
            "hgb_top1": hit_rate(split_df, "hgb_top1_hit"),
            "baseline_top5": hit_rate(split_df, "baseline_top5_hit"),
            "diverse_top5": hit_rate(split_df, "diverse_top5_hit"),
            "combo_top5": hit_rate(split_df, "combo_top5_hit"),
        }

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_json = out_prefix.with_suffix(".summary.json")
    combo_csv = out_prefix.with_name(out_prefix.name + "_combo_top5.csv")
    scores_csv = out_prefix.with_name(out_prefix.name + "_oof_scores.csv")
    cases_csv = out_prefix.with_name(out_prefix.name + "_case_metrics.csv")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(combo_rows).to_csv(combo_csv, index=False)
    df[["case_key", "candidate_id", "truth_intersects", "rule_rank", "hgb_oof_score", "hgb_oof_rank"]].to_csv(
        scores_csv, index=False
    )
    cases.to_csv(cases_csv, index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_json={summary_json}")
    print(f"combo_csv={combo_csv}")
    print(f"scores_csv={scores_csv}")
    print(f"cases_csv={cases_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HGB OOF + spatial-diverse Mansfield polygon selector experiment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument("--max-category-levels", type=int, default=20)
    parser.add_argument("--diverse-threshold-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
