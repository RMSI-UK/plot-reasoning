#!/usr/bin/env python3
"""Train a non-AI Mansfield in-box polygon selector.

The input is the production-visible hybrid top-N candidate table emitted by
``9_hybrid_polygon_rerank.py``.  Truth columns are used only as labels and
diagnostics; they are deliberately excluded from the feature matrix.

The model is a small pairwise logistic ranker implemented with numpy/scipy so it
does not require sklearn/lightgbm/xgboost.  For every training case, positive
candidates are candidates that intersect the manual polygon.  The objective
learns weights that rank positive candidates above sampled negatives from the
same case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "tmp_results" / "mansfield_random1200_hybrid_top200_v1_top200.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "tmp_results" / "mansfield_polygon_selector_pairwise_v1"

TRUTH_COLUMNS = {
    "truth_intersects",
    "truth_centroid_inside",
    "truth_overlap_area_m2",
    "truth_candidate_overlap_ratio",
    "truth_cover_ratio",
    "truth_iou",
}

NUMERIC_COLUMNS = [
    "best_confidence",
    "candidate_area_m2",
    "distance_to_best_point_m",
    "distance_to_v7_point_m",
    "roi_intersects_count",
    "roi_min_rank",
    "roi_total_intersection_area_m2",
    "roi_rank_weighted_score",
    "distance_to_nearest_roi_center_m",
    "distance_to_mentioned_road_m",
    "mentioned_road_near_count_50m",
    "mentioned_road_near_count_100m",
    "range_anchor_endpoint_hits",
    "range_anchor_candidate_count",
    "range_anchor_cluster_count",
    "range_anchor_cluster_radius_m",
    "range_anchor_cluster_score",
    "distance_to_range_anchor_m",
    "uprn_count",
    "nearest_uprn_to_best_point_m",
    "nearest_uprn_to_v7_point_m",
    "rule_score",
    "rule_rank",
]

BOOL_COLUMNS = [
    "covers_best_point",
    "covers_v7_point",
    "intersects_top1_roi",
    "intersects_protected_current_roi",
    "intersects_evidence_roi",
    "intersects_corridor_roi",
    "has_range_anchor",
    "range_anchor_complete",
    "covers_range_anchor",
]

CATEGORICAL_COLUMNS = [
    "best_source_final",
    "best_selection_category",
    "candidate_source",
    "candidate_theme",
    "candidate_descriptive_group",
    "candidate_descriptive_term",
    "candidate_make",
    "roi_hit_reasons",
    "range_anchor_relation",
    "range_anchor_number_kind",
]

TEXT_PATTERNS = {
    "addr_land": r"\bland\b",
    "addr_rear": r"\brear\b|\bbehind\b",
    "addr_adjacent": r"\badj(?:acent)?\b|\badjoining\b|\bnext to\b",
    "addr_junction": r"\bjunction\b|\bcorner\b",
    "addr_between": r"\bbetween\b",
    "addr_frontage": r"\bfront(?:ing|age)?\b|\bfront of\b",
    "addr_plot": r"\bplots?\b",
    "addr_site": r"\bsite\b",
    "addr_former": r"\bformer\b",
    "addr_numbered": r"\b\d{1,4}\s*(?:-|to|/|&|and)?\s*\d{0,4}\b",
    "addr_range": r"\b\d{1,4}\s*(?:-|to|/)\s*\d{1,4}\b",
    "addr_unit": r"\bunit(?:s)?\b",
    "addr_farm": r"\bfarm\b",
    "addr_school": r"\bschool\b",
    "addr_church": r"\bchurch\b|\bchapel\b",
    "addr_pub": r"\bpub\b|\binn\b|\bhotel\b",
    "addr_retail": r"\bshop\b|\bsupermarket\b|\bretail\b|\bstores?\b",
}


def stable_hash_int(value: Any) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def log_feature(series: pd.Series, missing_value: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(missing_value).clip(lower=0)
    return np.log1p(numeric)


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "missing", "nan": "missing", "none": "missing"})
    )


def add_one_hot(
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    column: str,
    train_mask: pd.Series,
    max_levels: int,
) -> None:
    if column not in df:
        return
    cleaned = clean_category(df[column])
    levels = cleaned[train_mask].value_counts().head(max_levels).index.tolist()
    for level in levels:
        safe_level = re.sub(r"[^a-z0-9]+", "_", level).strip("_")[:42] or "blank"
        features[f"{column}={safe_level}"] = (cleaned == level).astype(float)


def build_features(df: pd.DataFrame, train_mask: pd.Series, max_levels: int = 16) -> tuple[pd.DataFrame, list[str]]:
    features: dict[str, pd.Series] = {}

    # Numeric features.  Distances use log transforms as the useful distinction is
    # usually near/far, not the exact metre value at long range.
    for column in NUMERIC_COLUMNS:
        value = safe_numeric(df, column)
        if column.startswith("distance") or column.startswith("nearest"):
            features[f"log_{column}"] = log_feature(value, 9999.0)
            features[f"{column}_missing"] = value.isna().astype(float)
        elif column in {"candidate_area_m2", "roi_total_intersection_area_m2"}:
            features[f"log_{column}"] = log_feature(value, 0.0)
            features[f"{column}_missing"] = value.isna().astype(float)
        elif column == "rule_rank":
            features["neg_rule_rank"] = -value.fillna(999.0)
            features["is_rule_top1"] = value.eq(1).astype(float)
            features["is_rule_top5"] = value.le(5).astype(float)
            features["is_rule_top10"] = value.le(10).astype(float)
        else:
            features[column] = value.fillna(0.0)
            features[f"{column}_missing"] = value.isna().astype(float)

    for column in BOOL_COLUMNS:
        if column in df:
            features[column] = to_bool(df[column]).astype(float)
        else:
            features[column] = pd.Series(0.0, index=df.index)

    # Production-visible address semantics.
    address = df.get("original_address", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    for name, pattern in TEXT_PATTERNS.items():
        features[name] = address.str.contains(pattern, regex=True, na=False).astype(float)

    # A few explicit interactions encode the main hypothesis: address points and
    # range anchors localize the area, but land/rear/adjacent cases often need a
    # neighbouring parcel rather than the exact building containing the point.
    source = clean_category(df.get("candidate_source", pd.Series("", index=df.index)))
    theme = clean_category(df.get("candidate_theme", pd.Series("", index=df.index)))
    group = clean_category(df.get("candidate_descriptive_group", pd.Series("", index=df.index)))
    land_like = (
        features["addr_land"]
        + features["addr_rear"]
        + features["addr_adjacent"]
        + features["addr_junction"]
        + features["addr_between"]
        + features["addr_plot"]
        + features["addr_site"]
    ).clip(upper=1)
    building_like_address = (
        features["addr_unit"]
        + features["addr_school"]
        + features["addr_church"]
        + features["addr_pub"]
        + features["addr_retail"]
    ).clip(upper=1)
    is_council = source.eq("council_cadastral").astype(float)
    is_wfs_merged = source.eq("wfs_merged").astype(float)
    is_wfs_raw = source.eq("wfs_raw").astype(float)
    is_theme_land = theme.str.contains("land", regex=False).astype(float)
    is_theme_building = theme.str.contains("building", regex=False).astype(float)
    is_group_building = group.str.contains("building", regex=False).astype(float)
    covers_best = features.get("covers_best_point", pd.Series(0.0, index=df.index))
    covers_range = features.get("covers_range_anchor", pd.Series(0.0, index=df.index))
    features["land_like_x_council"] = land_like * is_council
    features["land_like_x_theme_land"] = land_like * is_theme_land
    features["land_like_x_covers_best"] = land_like * covers_best
    features["land_like_x_covers_range"] = land_like * covers_range
    features["building_like_x_wfs_merged"] = building_like_address * is_wfs_merged
    features["building_like_x_building"] = building_like_address * (is_theme_building + is_group_building).clip(upper=1)
    features["source_council"] = is_council
    features["source_wfs_merged"] = is_wfs_merged
    features["source_wfs_raw"] = is_wfs_raw
    features["theme_land"] = is_theme_land
    features["theme_building"] = is_theme_building
    features["group_building"] = is_group_building

    for column in CATEGORICAL_COLUMNS:
        add_one_hot(features, df, column, train_mask, max_levels=max_levels)

    feature_df = pd.DataFrame(features, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = list(feature_df.columns)
    return feature_df, feature_names


def standardize_features(
    x: pd.DataFrame,
    train_mask: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = x.to_numpy(dtype="float64")
    train_values = values[train_mask.to_numpy()]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < 1e-9] = 1.0
    return (values - mean) / std, mean, std


def build_pairwise_matrix(
    df: pd.DataFrame,
    x: np.ndarray,
    train_cases: set[str],
    negative_ranks: int,
    negatives_per_positive: int,
    max_pairs: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs: list[np.ndarray] = []
    train_df = df[df["case_key"].isin(train_cases)].sort_values(["case_key", "rule_rank"])

    for _, group in train_df.groupby("case_key", sort=False):
        pos_indices = group.index[group["label"].to_numpy()].to_numpy()
        neg_group = group[~group["label"]].copy()
        if len(pos_indices) == 0 or neg_group.empty:
            continue
        priority_neg = neg_group[pd.to_numeric(neg_group["rule_rank"], errors="coerce") <= negative_ranks].index.to_numpy()
        other_neg = neg_group.index.to_numpy()
        for pos_idx in pos_indices[:20]:
            neg_choices: list[int] = []
            if len(priority_neg):
                take = min(len(priority_neg), max(1, negatives_per_positive // 2))
                neg_choices.extend(rng.choice(priority_neg, size=take, replace=False).tolist())
            remaining = negatives_per_positive - len(neg_choices)
            if remaining > 0 and len(other_neg):
                take = min(len(other_neg), remaining)
                neg_choices.extend(rng.choice(other_neg, size=take, replace=False).tolist())
            for neg_idx in neg_choices:
                pairs.append(x[int(pos_idx)] - x[int(neg_idx)])

    if not pairs:
        raise RuntimeError("No train pairs were built; check labels and split names.")
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in chosen]
    return np.vstack(pairs)


def fit_pairwise_logistic(pair_x: np.ndarray, l2: float, maxiter: int) -> np.ndarray:
    n_features = pair_x.shape[1]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        margin = pair_x @ weights
        # loss = log(1 + exp(-margin)), stable form.
        loss_terms = np.logaddexp(0.0, -margin)
        prob_neg = 1.0 / (1.0 + np.exp(np.clip(margin, -50, 50)))
        loss = float(loss_terms.mean() + 0.5 * l2 * np.dot(weights, weights))
        grad = -(pair_x.T @ prob_neg) / len(pair_x) + l2 * weights
        return loss, grad

    result = minimize(
        fun=lambda w: objective(w)[0],
        x0=np.zeros(n_features, dtype="float64"),
        jac=lambda w: objective(w)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-8},
    )
    if not result.success:
        print(f"warning: optimizer did not fully converge: {result.message}")
    return result.x


def assign_rank(scores: pd.Series, df: pd.DataFrame, rank_column: str) -> pd.Series:
    work = df[["case_key", "rule_rank"]].copy()
    work["_score"] = scores.to_numpy()
    work["_row_order"] = np.arange(len(work))
    ranked = work.sort_values(["case_key", "_score", "rule_rank"], ascending=[True, False, True]).copy()
    ranked[rank_column] = ranked.groupby("case_key").cumcount() + 1
    out = pd.Series(index=ranked["_row_order"].to_numpy(), data=ranked[rank_column].to_numpy())
    return out.sort_index()


def summarize_ranking(df: pd.DataFrame, rank_column: str, cases: list[str], label: str) -> dict[str, Any]:
    sub = df[df["case_key"].isin(cases)].copy()
    n_cases = len(cases)
    summary: dict[str, Any] = {"label": label, "cases": int(n_cases)}
    for k in [1, 2, 3, 5, 8, 10, 20, 50, 100, 200]:
        hits = sub[(sub[rank_column] <= k) & sub["label"]].groupby("case_key").size()
        count = int((hits > 0).sum())
        summary[f"top{k}"] = count
        summary[f"top{k}_rate"] = float(count / n_cases) if n_cases else 0.0
    present = sub[sub["label"]].groupby("case_key").size()
    summary["present"] = int((present > 0).sum())
    summary["present_rate"] = float(summary["present"] / n_cases) if n_cases else 0.0

    if "best_confidence" in sub:
        conf = pd.to_numeric(sub.groupby("case_key")["best_confidence"].first(), errors="coerce")
        for name, mask in {
            "low_conf_lt75": conf < 75,
            "mid_conf_75_80": (conf >= 75) & (conf < 80),
            "high_conf_ge80": conf >= 80,
        }.items():
            split_cases = conf[mask].index.tolist()
            if not split_cases:
                continue
            split_sub = sub[sub["case_key"].isin(split_cases)]
            split_hits = split_sub[(split_sub[rank_column] <= 5) & split_sub["label"]].groupby("case_key").size()
            summary[name] = {
                "cases": int(len(split_cases)),
                "top5": int((split_hits > 0).sum()),
                "top5_rate": float((split_hits > 0).sum() / len(split_cases)),
            }
    return summary


def tune_blend(
    df: pd.DataFrame,
    train_cases: list[str],
    ml_score: pd.Series,
    rule_score: pd.Series,
) -> tuple[float, dict[str, Any]]:
    train_mask = df["case_key"].isin(train_cases)
    rule = rule_score.copy()
    ml = ml_score.copy()
    # Per-case normalization keeps large score ranges from one case from
    # dominating another, and mirrors the ranking-only use of the score.
    for score in (rule, ml):
        grouped = score[train_mask].groupby(df.loc[train_mask, "case_key"])
        # no-op placeholder to keep intent visible; actual normalization below is global enough.
        _ = grouped.ngroups

    weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_weight = 0.0
    best_summary: dict[str, Any] | None = None
    best_tuple = (-1, -1, -1)
    for weight in weights:
        blended = rule + weight * ml
        rank_col = f"_tmp_blend_rank_{str(weight).replace('.', '_')}"
        df[rank_col] = assign_rank(blended, df, rank_col)
        summary = summarize_ranking(df, rank_col, train_cases, f"blend_train_{weight}")
        key = (summary["top5"], summary["top1"], summary["top10"])
        if key > best_tuple:
            best_tuple = key
            best_weight = weight
            best_summary = summary
        df.drop(columns=[rank_col], inplace=True)
    assert best_summary is not None
    return best_weight, best_summary


@dataclass
class RunConfig:
    input_csv: Path
    output_prefix: Path
    train_split: str
    test_split: str
    max_rank: int
    negative_ranks: int
    negatives_per_positive: int
    max_pairs: int
    l2: float
    maxiter: int
    seed: int


def run(config: RunConfig) -> dict[str, Any]:
    df = pd.read_csv(config.input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    df = df[pd.to_numeric(df["rule_rank"], errors="coerce") <= config.max_rank].copy()
    df.reset_index(drop=True, inplace=True)
    df["label"] = to_bool(df["truth_intersects"])
    df["rule_rank"] = pd.to_numeric(df["rule_rank"], errors="coerce")
    df["rule_score"] = pd.to_numeric(df["rule_score"], errors="coerce").fillna(-9999.0)

    all_cases = sorted(df["case_key"].unique(), key=lambda x: (stable_hash_int(x), x))
    train_cases = sorted(df.loc[df["sample_split"].eq(config.train_split), "case_key"].unique().tolist())
    test_cases = sorted(df.loc[df["sample_split"].eq(config.test_split), "case_key"].unique().tolist())
    if not train_cases or not test_cases:
        # Fallback to a stable 80/20 split if the requested split names are not present.
        train_cases = [case for i, case in enumerate(all_cases) if i % 5 != 0]
        test_cases = [case for i, case in enumerate(all_cases) if i % 5 == 0]

    train_mask = df["case_key"].isin(train_cases)
    features, feature_names = build_features(df, train_mask=train_mask)
    x, mean, std = standardize_features(features, train_mask=train_mask)
    pair_x = build_pairwise_matrix(
        df,
        x,
        set(train_cases),
        negative_ranks=config.negative_ranks,
        negatives_per_positive=config.negatives_per_positive,
        max_pairs=config.max_pairs,
        seed=config.seed,
    )
    weights = fit_pairwise_logistic(pair_x, l2=config.l2, maxiter=config.maxiter)
    ml_score = pd.Series(x @ weights, index=df.index)
    rule_score = pd.Series(df["rule_score"].to_numpy(dtype="float64"), index=df.index)

    # Blend uses a z-scored rule score so the learned model has a comparable range.
    rule_mean = rule_score[train_mask].mean()
    rule_std = rule_score[train_mask].std()
    if not np.isfinite(rule_std) or rule_std < 1e-9:
        rule_std = 1.0
    rule_z = (rule_score - rule_mean) / rule_std

    best_blend_weight, blend_train_summary = tune_blend(df, train_cases, ml_score, rule_z)
    df["ml_score"] = ml_score
    df["blend_score"] = rule_z + best_blend_weight * ml_score
    df["ml_rank"] = assign_rank(df["ml_score"], df, "ml_rank")
    df["blend_rank"] = assign_rank(df["blend_score"], df, "blend_rank")

    summary = {
        "input_csv": str(config.input_csv),
        "output_prefix": str(config.output_prefix),
        "candidate_rows": int(len(df)),
        "cases": int(len(all_cases)),
        "train_split": config.train_split,
        "test_split": config.test_split,
        "train_cases": int(len(train_cases)),
        "test_cases": int(len(test_cases)),
        "feature_count": int(len(feature_names)),
        "pair_count": int(len(pair_x)),
        "l2": config.l2,
        "best_blend_weight": float(best_blend_weight),
        "baseline_rule_train": summarize_ranking(df, "rule_rank", train_cases, "baseline_rule_train"),
        "baseline_rule_test": summarize_ranking(df, "rule_rank", test_cases, "baseline_rule_test"),
        "ml_train": summarize_ranking(df, "ml_rank", train_cases, "ml_train"),
        "ml_test": summarize_ranking(df, "ml_rank", test_cases, "ml_test"),
        "blend_train_tuning": blend_train_summary,
        "blend_train": summarize_ranking(df, "blend_rank", train_cases, "blend_train"),
        "blend_test": summarize_ranking(df, "blend_rank", test_cases, "blend_test"),
        "baseline_rule_all": summarize_ranking(df, "rule_rank", all_cases, "baseline_rule_all"),
        "ml_all": summarize_ranking(df, "ml_rank", all_cases, "ml_all"),
        "blend_all": summarize_ranking(df, "blend_rank", all_cases, "blend_all"),
    }

    weights_df = pd.DataFrame(
        {
            "feature": feature_names,
            "weight": weights,
            "abs_weight": np.abs(weights),
            "mean": mean,
            "std": std,
        }
    ).sort_values("abs_weight", ascending=False)

    out_prefix = config.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    scored_csv = out_prefix.with_name(out_prefix.name + "_scored_top200.csv")
    summary_json = out_prefix.with_suffix(".summary.json")
    weights_csv = out_prefix.with_name(out_prefix.name + "_weights.csv")
    compact_cols = [
        "case_key",
        "base_key",
        "sample_split",
        "original_address",
        "candidate_id",
        "candidate_source",
        "candidate_theme",
        "candidate_descriptive_group",
        "candidate_area_m2",
        "truth_intersects",
        "truth_iou",
        "rule_score",
        "rule_rank",
        "ml_score",
        "ml_rank",
        "blend_score",
        "blend_rank",
    ]
    existing_cols = [col for col in compact_cols if col in df.columns]
    df[existing_cols].to_csv(scored_csv, index=False)
    weights_df.to_csv(weights_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"scored_csv={scored_csv}")
    print(f"weights_csv={weights_csv}")
    print(f"summary_json={summary_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a Mansfield polygon pairwise selector.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--train-split", default="random1000_seed43_excl_random200")
    parser.add_argument("--test-split", default="random200_seed42")
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument("--negative-ranks", type=int, default=40)
    parser.add_argument("--negatives-per-positive", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=180000)
    parser.add_argument("--l2", type=float, default=0.04)
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        RunConfig(
            input_csv=args.input_csv,
            output_prefix=args.output_prefix,
            train_split=args.train_split,
            test_split=args.test_split,
            max_rank=args.max_rank,
            negative_ranks=args.negative_ranks,
            negatives_per_positive=args.negatives_per_positive,
            max_pairs=args.max_pairs,
            l2=args.l2,
            maxiter=args.maxiter,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
