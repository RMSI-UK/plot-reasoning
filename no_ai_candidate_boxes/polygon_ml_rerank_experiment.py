#!/usr/bin/env python3
"""Offline polygon reranking experiment without AI/API calls.

The input is a production-visible polygon candidate table with offline truth
labels.  Truth columns are used only for training/evaluation.  The experiment
answers two questions:

1. Why do current top-k polygon selections miss?
2. Can a light tabular model improve top-k recall over the current
   ``polygon_rank`` ordering?

This is deliberately an experiment entrypoint.  It does not replace the
rule-based production selector.
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

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
LEGACY_FEATURES = PROJECT_ROOT / "legacy_experiments" / "13_train_polygon_selector.py"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "tmp_results"
    / "opt_final_1282_relation_mix_default_v1_ranked_top100.csv"
)
DEFAULT_OUTPUT_PREFIX = (
    PROJECT_ROOT
    / "tmp_results"
    / "polygon_ml_rerank_opt_final_1282_hgb_v1"
)

spec = importlib.util.spec_from_file_location("selector13_features", LEGACY_FEATURES)
selector13 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["selector13_features"] = selector13
spec.loader.exec_module(selector13)


TEXT_FLAGS = {
    "addr_land": r"\bland\b",
    "addr_rear": r"\brear\b|\bbehind\b",
    "addr_adjacent": r"\badjacent\b|\badjoining\b|\bnext\s+to\b",
    "addr_between": r"\bbetween\b",
    "addr_off": r"\boff\b",
    "addr_frontage": r"\bfronting\b|\bfrontage\b|\bfront\s+of\b",
    "addr_plot": r"\bplots?\b",
    "addr_site": r"\bsite\b",
    "addr_range": r"\b\d{1,4}\s*(?:-|to|/)\s*\d{1,4}\b",
    "addr_simple_number": r"^\s*\d{1,4}[A-Za-z]?\b",
}


def stable_hash_int(value: Any) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def fold_for_case(case_key: str, folds: int) -> int:
    return stable_hash_int(case_key) % folds


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "missing", "nan": "missing", "none": "missing"})
    )


def add_one_hot(
    features: pd.DataFrame,
    df: pd.DataFrame,
    column: str,
    train_mask: pd.Series,
    *,
    max_levels: int,
) -> None:
    if column not in df:
        return
    cleaned = clean_category(df[column])
    levels = cleaned[train_mask].value_counts().head(max_levels).index.tolist()
    for level in levels:
        safe_level = re.sub(r"[^a-z0-9]+", "_", level).strip("_")[:42] or "blank"
        features[f"{column}={safe_level}"] = (cleaned == level).astype("float32")


def build_model_features(
    df: pd.DataFrame,
    train_mask: pd.Series,
    *,
    baseline_rank_column: str,
    max_category_levels: int,
) -> tuple[pd.DataFrame, list[str]]:
    features, _ = selector13.build_features(df, train_mask=train_mask, max_levels=max_category_levels)
    features = features.astype("float32").copy()

    baseline_rank = pd.to_numeric(df[baseline_rank_column], errors="coerce").fillna(999.0)
    features["neg_baseline_rank"] = (-baseline_rank).astype("float32")
    features["baseline_rank_reciprocal"] = (1.0 / baseline_rank.clip(lower=1.0)).astype("float32")
    for rank in [1, 3, 5, 8, 10, 20]:
        features[f"baseline_top{rank}"] = baseline_rank.le(rank).astype("float32")

    rule_rank = pd.to_numeric(df.get("rule_rank", 999.0), errors="coerce").fillna(999.0)
    features["baseline_minus_rule_rank"] = (baseline_rank - rule_rank).astype("float32")

    address = df.get("original_address", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    for name, pattern in TEXT_FLAGS.items():
        features[name] = address.str.contains(pattern, regex=True, na=False).astype("float32")
    relation_like = (
        features["addr_land"]
        + features["addr_rear"]
        + features["addr_adjacent"]
        + features["addr_between"]
        + features["addr_off"]
        + features["addr_frontage"]
        + features["addr_plot"]
        + features["addr_site"]
    ).clip(upper=1)
    features["relation_like_case"] = relation_like.astype("float32")
    features["relation_like_x_not_cover_best"] = (
        relation_like * (1.0 - features.get("covers_best_point", pd.Series(0.0, index=df.index)))
    ).astype("float32")
    features["range_case_x_has_range_anchor"] = (
        features["addr_range"] * features.get("has_range_anchor", pd.Series(0.0, index=df.index))
    ).astype("float32")

    add_one_hot(features, df, "polygon_selection_bucket", train_mask, max_levels=max_category_levels)
    add_one_hot(features, df, "candidate_descriptive_group", train_mask, max_levels=max_category_levels)
    add_one_hot(features, df, "candidate_descriptive_term", train_mask, max_levels=max_category_levels)

    feature_names = list(features.columns)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0), feature_names


def load_data(path: Path, baseline_rank_column: str, max_rank: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_key": str, "base_key": str}, low_memory=False)
    if baseline_rank_column not in df:
        raise ValueError(f"{path} has no baseline rank column {baseline_rank_column!r}")
    df[baseline_rank_column] = pd.to_numeric(df[baseline_rank_column], errors="coerce")
    df = df[df[baseline_rank_column].le(max_rank)].copy().reset_index(drop=True)
    df["label"] = df["truth_intersects"].map(to_bool)
    for column in [
        "rule_rank",
        "rule_score",
        "polygon_rank",
        "best_confidence",
        "candidate_area_m2",
        "candidate_centroid_easting",
        "candidate_centroid_northing",
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def sample_fit_indices(
    df: pd.DataFrame,
    train_mask: pd.Series,
    *,
    baseline_rank_column: str,
    hard_negative_rank: int,
    other_negative_ratio: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    y = df["label"].to_numpy(dtype=bool)
    pos_idx = train_idx[y[train_idx]]
    train_df = df.iloc[train_idx]
    hard_neg = train_df[
        (~train_df["label"])
        & (pd.to_numeric(train_df[baseline_rank_column], errors="coerce") <= hard_negative_rank)
    ].index.to_numpy()
    other_neg = train_df[
        (~train_df["label"])
        & (pd.to_numeric(train_df[baseline_rank_column], errors="coerce") > hard_negative_rank)
    ].index.to_numpy()
    sampled_other = rng.choice(
        other_neg,
        size=min(len(other_neg), max(1, len(pos_idx) * other_negative_ratio)),
        replace=False,
    )
    return np.unique(np.concatenate([pos_idx, hard_neg, sampled_other]))


def train_predict_hgb(
    df: pd.DataFrame,
    train_cases: set[str],
    test_cases: set[str],
    *,
    baseline_rank_column: str,
    max_category_levels: int,
    seed: int,
    fold_index: int,
    max_iter: int,
    learning_rate: float,
    hard_negative_rank: int,
    other_negative_ratio: int,
) -> tuple[np.ndarray, list[str]]:
    train_mask = df["case_key"].isin(train_cases)
    test_mask = df["case_key"].isin(test_cases)
    features, feature_names = build_model_features(
        df,
        train_mask,
        baseline_rank_column=baseline_rank_column,
        max_category_levels=max_category_levels,
    )
    x = features.to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype=bool)
    fit_idx = sample_fit_indices(
        df,
        train_mask,
        baseline_rank_column=baseline_rank_column,
        hard_negative_rank=hard_negative_rank,
        other_negative_ratio=other_negative_ratio,
        seed=seed + fold_index,
    )
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=seed + 1000 + fold_index,
    )
    clf.fit(x[fit_idx], y[fit_idx])
    scores = np.full(len(df), np.nan, dtype="float64")
    test_idx = np.flatnonzero(test_mask.to_numpy())
    scores[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    return scores, feature_names


def assign_rank(
    df: pd.DataFrame,
    score_column: str,
    rank_column: str,
    *,
    baseline_rank_column: str,
) -> None:
    ordered = df.sort_values(
        ["case_key", score_column, baseline_rank_column, "candidate_id"],
        ascending=[True, False, True, True],
    ).copy()
    ordered[rank_column] = ordered.groupby("case_key").cumcount() + 1
    df[rank_column] = ordered.sort_index()[rank_column].to_numpy()


def portfolio_rank(
    df: pd.DataFrame,
    *,
    preserve_baseline: int,
    top_k: int,
    score_column: str,
    baseline_rank_column: str,
    out_column: str,
) -> None:
    ranks = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, group in df.groupby("case_key", sort=False):
        baseline_order = group.sort_values([baseline_rank_column, "candidate_id"], ascending=[True, True])
        ml_order = group.sort_values([score_column, baseline_rank_column, "candidate_id"], ascending=[False, True, True])
        selected: list[int] = []
        for index in baseline_order.index[:preserve_baseline]:
            selected.append(index)
            if len(selected) >= top_k:
                break
        for index in ml_order.index:
            if len(selected) >= top_k:
                break
            if index not in selected:
                selected.append(index)
        remainder = [index for index in baseline_order.index if index not in selected]
        final_order = selected + remainder
        ranks.loc[final_order] = np.arange(1, len(final_order) + 1, dtype="float64")
    df[out_column] = ranks.astype("int64")


def gated_portfolio_rank(
    df: pd.DataFrame,
    *,
    preserve_baseline: int,
    top_k: int,
    score_column: str,
    baseline_rank_column: str,
    score_margin: float,
    max_challenger_rank: int | None,
    max_candidate_area_m2: float | None,
    out_column: str,
) -> None:
    ranks = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, group in df.groupby("case_key", sort=False):
        baseline_order = group.sort_values([baseline_rank_column, "candidate_id"], ascending=[True, True])
        selected = list(baseline_order.index[:preserve_baseline])
        fill = list(baseline_order.index[preserve_baseline:top_k])
        ml_order = [
            index
            for index in group.sort_values([score_column, baseline_rank_column, "candidate_id"], ascending=[False, True, True]).index
            if index not in selected and index not in fill
        ]
        if max_challenger_rank is not None:
            ml_order = [
                index
                for index in ml_order
                if pd.notna(df.loc[index, baseline_rank_column])
                and float(df.loc[index, baseline_rank_column]) <= max_challenger_rank
            ]
        if max_candidate_area_m2 is not None:
            ml_order = [
                index
                for index in ml_order
                if "candidate_area_m2" in df
                and pd.notna(df.loc[index, "candidate_area_m2"])
                and float(df.loc[index, "candidate_area_m2"]) <= max_candidate_area_m2
            ]
        for pos in range(len(fill)):
            if not ml_order:
                break
            challenger = ml_order[0]
            incumbent = fill[pos]
            if float(df.loc[challenger, score_column]) >= float(df.loc[incumbent, score_column]) + score_margin:
                fill[pos] = challenger
                ml_order.pop(0)
        final_selected = selected + fill
        remainder = [index for index in baseline_order.index if index not in final_selected]
        final_order = final_selected + remainder
        ranks.loc[final_order] = np.arange(1, len(final_order) + 1, dtype="float64")
    df[out_column] = ranks.astype("int64")


def option_label(value: Any, none_label: str) -> str:
    if value is None:
        return none_label
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace("-", "neg").replace(".", "_")


def parse_optional_int_options(value: str) -> list[int | None]:
    parsed: list[int | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "all", "0"}:
            parsed.append(None)
        else:
            parsed.append(int(item))
    return parsed or [None]


def parse_optional_float_options(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "all", "0"}:
            parsed.append(None)
        else:
            parsed.append(float(item))
    return parsed or [None]


def summarize_ranking(df: pd.DataFrame, rank_column: str, cases: list[str], label: str) -> dict[str, Any]:
    sub = df[df["case_key"].isin(cases)].copy()
    n_cases = len(cases)
    summary: dict[str, Any] = {"label": label, "cases": int(n_cases)}
    for k in [1, 2, 3, 5, 8, 10, 20, 50, 100]:
        hits = sub[(sub[rank_column] <= k) & sub["label"]].groupby("case_key").size()
        count = int((hits > 0).sum())
        summary[f"top{k}"] = count
        summary[f"top{k}_rate"] = float(count / n_cases) if n_cases else 0.0
    present = sub[sub["label"]].groupby("case_key").size()
    summary["present"] = int((present > 0).sum())
    summary["present_rate"] = float(summary["present"] / n_cases) if n_cases else 0.0
    return summary


def first_hit_rank(df: pd.DataFrame, rank_column: str) -> pd.Series:
    hits = df[df["label"]].groupby("case_key")[rank_column].min()
    cases = pd.Index(sorted(df["case_key"].unique(), key=lambda item: (len(str(item)), str(item))))
    return hits.reindex(cases)


def classify_address(address: Any) -> dict[str, bool]:
    text = str(address or "").lower()
    return {name: bool(re.search(pattern, text)) for name, pattern in TEXT_FLAGS.items()}


def miss_analysis(
    df: pd.DataFrame,
    *,
    baseline_rank_column: str,
    candidate_rank_column: str,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_hit_rank = first_hit_rank(df, baseline_rank_column)
    candidate_hit_rank = first_hit_rank(df, candidate_rank_column)
    case_meta = df.groupby("case_key").first(numeric_only=False)
    rows: list[dict[str, Any]] = []
    for case_key in baseline_hit_rank.index:
        baseline_rank = baseline_hit_rank.get(case_key)
        candidate_rank = candidate_hit_rank.get(case_key)
        address = case_meta.loc[case_key, "original_address"] if case_key in case_meta.index else ""
        flags = classify_address(address)
        if pd.isna(baseline_rank):
            miss_type = "not_present_topN"
        elif int(baseline_rank) <= top_k:
            miss_type = "baseline_topk_hit"
        elif int(baseline_rank) <= 20:
            miss_type = "present_rank_6_20"
        elif int(baseline_rank) <= 100:
            miss_type = "present_rank_21_100"
        else:
            miss_type = "present_beyond_limit"
        rows.append(
            {
                "case_key": case_key,
                "sample_split": case_meta.loc[case_key, "sample_split"] if case_key in case_meta.index else "",
                "original_address": address,
                "best_confidence": case_meta.loc[case_key, "best_confidence"] if case_key in case_meta.index else "",
                "baseline_truth_rank": "" if pd.isna(baseline_rank) else int(baseline_rank),
                "candidate_truth_rank": "" if pd.isna(candidate_rank) else int(candidate_rank),
                "baseline_topk_hit": bool(pd.notna(baseline_rank) and int(baseline_rank) <= top_k),
                "candidate_topk_hit": bool(pd.notna(candidate_rank) and int(candidate_rank) <= top_k),
                "miss_type": miss_type,
                **flags,
            }
        )
    analysis = pd.DataFrame(rows)
    baseline_misses = analysis[~analysis["baseline_topk_hit"]]
    recovered = analysis[(~analysis["baseline_topk_hit"]) & (analysis["candidate_topk_hit"])]
    lost = analysis[(analysis["baseline_topk_hit"]) & (~analysis["candidate_topk_hit"])]
    summary: dict[str, Any] = {
        "cases": int(len(analysis)),
        "baseline_topk_hits": int(analysis["baseline_topk_hit"].sum()),
        "candidate_topk_hits": int(analysis["candidate_topk_hit"].sum()),
        "baseline_topk_misses": int((~analysis["baseline_topk_hit"]).sum()),
        "recovered_cases": int(len(recovered)),
        "lost_cases": int(len(lost)),
        "miss_type_counts": baseline_misses["miss_type"].value_counts().to_dict(),
        "recovered_case_keys": recovered["case_key"].astype(str).tolist(),
        "lost_case_keys": lost["case_key"].astype(str).tolist(),
    }
    for flag in TEXT_FLAGS:
        summary[f"baseline_misses_{flag}"] = int(baseline_misses[flag].sum()) if flag in baseline_misses else 0
    return analysis, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    df = load_data(args.input_csv, args.baseline_rank_column, args.max_rank)
    case_keys = sorted(df["case_key"].unique(), key=lambda item: (stable_hash_int(item), item))

    oof = np.full(len(df), np.nan, dtype="float64")
    fold_logs: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for fold_index in range(args.folds):
        train_cases = {key for key in case_keys if fold_for_case(key, args.folds) != fold_index}
        test_cases = {key for key in case_keys if fold_for_case(key, args.folds) == fold_index}
        scores, feature_names = train_predict_hgb(
            df,
            train_cases,
            test_cases,
            baseline_rank_column=args.baseline_rank_column,
            max_category_levels=args.max_category_levels,
            seed=args.seed,
            fold_index=fold_index,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            hard_negative_rank=args.hard_negative_rank,
            other_negative_ratio=args.other_negative_ratio,
        )
        mask = ~np.isnan(scores)
        oof[mask] = scores[mask]
        fold_logs.append({"fold": fold_index, "train_cases": len(train_cases), "test_cases": len(test_cases)})
        print(f"fold {fold_index} complete: train={len(train_cases)} test={len(test_cases)}", flush=True)

    df["hgb_oof_score"] = oof
    assign_rank(df, "hgb_oof_score", "hgb_oof_rank", baseline_rank_column=args.baseline_rank_column)

    portfolio_summaries: dict[str, Any] = {}
    best_portfolio_column = ""
    best_key = (-1, -1, -1, -1)
    preserve_options = [int(item) for item in args.preserve_baseline_options.split(",") if item.strip()]
    for preserve in preserve_options:
        rank_column = f"portfolio_p{preserve}_rank"
        portfolio_rank(
            df,
            preserve_baseline=preserve,
            top_k=args.top_k,
            score_column="hgb_oof_score",
            baseline_rank_column=args.baseline_rank_column,
            out_column=rank_column,
        )
        summary = summarize_ranking(df, rank_column, case_keys, f"portfolio_preserve_{preserve}")
        portfolio_summaries[f"preserve_{preserve}"] = summary
        key = (summary["top5"], summary["top3"], summary["top1"], -preserve)
        if key > best_key:
            best_key = key
            best_portfolio_column = rank_column

    gated_summaries: dict[str, Any] = {}
    best_gated_column = ""
    best_gated_key = (-1, -1, -999, -1)
    margins = [float(item) for item in args.gate_margins.split(",") if item.strip()]
    max_challenger_rank_options = parse_optional_int_options(args.gate_max_challenger_ranks)
    max_area_options = parse_optional_float_options(args.gate_max_candidate_area_m2_options)
    baseline_hits = first_hit_rank(df, args.baseline_rank_column)
    for margin in margins:
        margin_label = option_label(margin, "all")
        for max_challenger_rank in max_challenger_rank_options:
            rank_label = option_label(max_challenger_rank, "all")
            for max_candidate_area_m2 in max_area_options:
                area_label = option_label(max_candidate_area_m2, "all")
                rank_column = (
                    f"gated_p{args.gate_preserve_baseline}_"
                    f"m{margin_label}_r{rank_label}_a{area_label}_rank"
                )
                gated_portfolio_rank(
                    df,
                    preserve_baseline=args.gate_preserve_baseline,
                    top_k=args.top_k,
                    score_column="hgb_oof_score",
                    baseline_rank_column=args.baseline_rank_column,
                    score_margin=margin,
                    max_challenger_rank=max_challenger_rank,
                    max_candidate_area_m2=max_candidate_area_m2,
                    out_column=rank_column,
                )
                label = (
                    f"gated_p{args.gate_preserve_baseline}_m{margin}"
                    f"_r{rank_label}_a{area_label}"
                )
                summary = summarize_ranking(df, rank_column, case_keys, label)
                gated_hits = first_hit_rank(df, rank_column)
                recovered = sum(
                    (gated_hits.get(case, 999) <= args.top_k)
                    and not (baseline_hits.get(case, 999) <= args.top_k)
                    for case in case_keys
                )
                lost = sum(
                    (baseline_hits.get(case, 999) <= args.top_k)
                    and not (gated_hits.get(case, 999) <= args.top_k)
                    for case in case_keys
                )
                summary["score_margin"] = margin
                summary["max_challenger_rank"] = (
                    int(max_challenger_rank) if max_challenger_rank is not None else None
                )
                summary["max_candidate_area_m2"] = (
                    float(max_candidate_area_m2) if max_candidate_area_m2 is not None else None
                )
                summary["recovered_cases"] = int(recovered)
                summary["lost_cases"] = int(lost)
                summary["net_gain_cases"] = int(recovered - lost)
                gated_summaries[f"margin_{margin_label}_r{rank_label}_a{area_label}"] = summary
                key = (summary["top5"], -lost, recovered - lost, summary["top8"])
                if key > best_gated_key:
                    best_gated_key = key
                    best_gated_column = rank_column

    train_split_cases = sorted(
        df.loc[df["sample_split"].astype(str).eq(args.train_split), "case_key"].unique().tolist(),
        key=lambda item: (stable_hash_int(item), item),
    )
    test_split_cases = sorted(
        df.loc[df["sample_split"].astype(str).eq(args.test_split), "case_key"].unique().tolist(),
        key=lambda item: (stable_hash_int(item), item),
    )
    if not train_split_cases or not test_split_cases:
        train_split_cases = [key for index, key in enumerate(case_keys) if index % 5 != 0]
        test_split_cases = [key for index, key in enumerate(case_keys) if index % 5 == 0]

    baseline_summary_all = summarize_ranking(df, args.baseline_rank_column, case_keys, "baseline_all")
    ml_summary_all = summarize_ranking(df, "hgb_oof_rank", case_keys, "hgb_oof_all")
    best_portfolio_summary_all = summarize_ranking(df, best_portfolio_column, case_keys, "best_portfolio_all")
    analysis, analysis_summary = miss_analysis(
        df,
        baseline_rank_column=args.baseline_rank_column,
        candidate_rank_column=best_gated_column,
        top_k=args.top_k,
    )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_json = output_prefix.with_suffix(".summary.json")
    case_analysis_csv = output_prefix.with_name(output_prefix.name + "_case_miss_analysis.csv")
    scored_csv = output_prefix.with_name(output_prefix.name + "_scored_top100.csv")
    selected_csv = output_prefix.with_name(output_prefix.name + f"_best_portfolio_top{args.top_k}.csv")
    gated_selected_csv = output_prefix.with_name(output_prefix.name + f"_best_gated_top{args.top_k}.csv")
    gate_sweep_csv = output_prefix.with_name(output_prefix.name + "_gate_sweep.csv")

    compact_cols = [
        "case_key",
        "base_key",
        "sample_split",
        "original_address",
        "candidate_id",
        "candidate_source",
        "candidate_descriptive_group",
        "candidate_area_m2",
        "truth_intersects",
        args.baseline_rank_column,
        "hgb_oof_score",
        "hgb_oof_rank",
        best_portfolio_column,
        best_gated_column,
        "rule_rank",
        "rule_score",
        "polygon_selection_bucket",
    ]
    compact_cols = [column for column in compact_cols if column in df]
    df[compact_cols].to_csv(scored_csv, index=False)
    selected = df[df[best_portfolio_column] <= args.top_k].sort_values(["case_key", best_portfolio_column])
    selected.to_csv(selected_csv, index=False)
    gated_selected = df[df[best_gated_column] <= args.top_k].sort_values(["case_key", best_gated_column])
    gated_selected.to_csv(gated_selected_csv, index=False)
    pd.DataFrame(gated_summaries.values()).to_csv(gate_sweep_csv, index=False)
    analysis.to_csv(case_analysis_csv, index=False)

    summary: dict[str, Any] = {
        "input_csv": str(args.input_csv),
        "output_prefix": str(output_prefix),
        "baseline_rank_column": args.baseline_rank_column,
        "top_k": int(args.top_k),
        "max_rank": int(args.max_rank),
        "cases": int(len(case_keys)),
        "candidate_rows": int(len(df)),
        "feature_count": int(len(feature_names)),
        "folds": int(args.folds),
        "fold_logs": fold_logs,
        "baseline_all": baseline_summary_all,
        "hgb_oof_all": ml_summary_all,
        "portfolio_oof_all": portfolio_summaries,
        "best_portfolio_rank_column": best_portfolio_column,
        "best_portfolio_all": best_portfolio_summary_all,
        "gated_oof_all": gated_summaries,
        "best_gated_rank_column": best_gated_column,
        "best_gated_all": summarize_ranking(df, best_gated_column, case_keys, "best_gated_all"),
        "baseline_train_split": summarize_ranking(df, args.baseline_rank_column, train_split_cases, "baseline_train_split"),
        "baseline_test_split": summarize_ranking(df, args.baseline_rank_column, test_split_cases, "baseline_test_split"),
        "hgb_oof_train_split": summarize_ranking(df, "hgb_oof_rank", train_split_cases, "hgb_oof_train_split"),
        "hgb_oof_test_split": summarize_ranking(df, "hgb_oof_rank", test_split_cases, "hgb_oof_test_split"),
        "best_portfolio_train_split": summarize_ranking(df, best_portfolio_column, train_split_cases, "best_portfolio_train_split"),
        "best_portfolio_test_split": summarize_ranking(df, best_portfolio_column, test_split_cases, "best_portfolio_test_split"),
        "best_gated_train_split": summarize_ranking(df, best_gated_column, train_split_cases, "best_gated_train_split"),
        "best_gated_test_split": summarize_ranking(df, best_gated_column, test_split_cases, "best_gated_test_split"),
        "miss_analysis": analysis_summary,
        "outputs": {
            "summary_json": str(summary_json),
            "case_miss_analysis_csv": str(case_analysis_csv),
            "scored_csv": str(scored_csv),
            "selected_csv": str(selected_csv),
            "gated_selected_csv": str(gated_selected_csv),
            "gate_sweep_csv": str(gate_sweep_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={summary_json}")
    print(f"case_miss_analysis_csv={case_analysis_csv}")
    print(f"scored_csv={scored_csv}")
    print(f"selected_csv={selected_csv}")
    print(f"gated_selected_csv={gated_selected_csv}")
    print(f"gate_sweep_csv={gate_sweep_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline polygon ML rerank experiment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--baseline-rank-column", default="polygon_rank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-category-levels", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--hard-negative-rank", type=int, default=60)
    parser.add_argument("--other-negative-ratio", type=int, default=8)
    parser.add_argument("--preserve-baseline-options", default="0,1,2,3,4,5")
    parser.add_argument("--gate-preserve-baseline", type=int, default=3)
    parser.add_argument("--gate-margins", default="0.3,0.5,0.7")
    parser.add_argument("--gate-max-challenger-ranks", default="20,50")
    parser.add_argument("--gate-max-candidate-area-m2-options", default="0,5000")
    parser.add_argument("--train-split", default="random1000_seed43_excl_random200")
    parser.add_argument("--test-split", default="random200_seed42")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
