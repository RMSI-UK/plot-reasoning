#!/usr/bin/env python3
"""ConvNeXt plan-to-candidate visual reranking experiment.

This is an offline, no-API experiment.  It uses a locally cached ImageNet
ConvNeXt-Tiny backbone to embed:
- the selected historical plan crop for a case;
- rendered local vector-map crops around each candidate polygon at several
  scales.

A small case-level out-of-fold torch classifier then tests whether visual
embedding relationships can improve candidate polygon ranking.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plan_visual_match_experiment as visual  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "tmp_results" / "mansfield_random1200_hybrid_top200_v1_top200.csv"
DEFAULT_PLAN_FEATURES = ROOT / "tmp_results" / "mansfield_plan_shape_rerank_v1_rank50_plan_features_cache_rank50.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "tmp_results" / "mansfield_plan_convnext_rerank_v1"


def fold_for_case(case_key: str, folds: int) -> int:
    import hashlib

    return int(hashlib.md5(str(case_key).encode("utf-8")).hexdigest()[:8], 16) % folds


def truth_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def preprocess_pil(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), "white")
    canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
    arr = np.asarray(canvas).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def preprocess_plan(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_pil(image, image_size)


def preprocess_array(arr: np.ndarray, image_size: int) -> torch.Tensor:
    if arr.ndim == 2:
        image = Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB")
    else:
        image = Image.fromarray(arr.astype("uint8")).convert("RGB")
    return preprocess_pil(image, image_size)


class ConvNextEmbedder:
    def __init__(self, device: str, image_size: int, batch_size: int):
        self.device = torch.device(device)
        self.image_size = image_size
        self.batch_size = batch_size
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        model.classifier = nn.Sequential(nn.Flatten())
        model.eval().to(self.device)
        self.model = model

    @torch.inference_mode()
    def embed_tensors(self, tensors: list[torch.Tensor]) -> np.ndarray:
        if not tensors:
            return np.zeros((0, 768), dtype="float32")
        outs: list[np.ndarray] = []
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
            emb = self.model(batch)
            emb = torch.nn.functional.normalize(emb, dim=1)
            outs.append(emb.detach().cpu().numpy().astype("float32"))
        return np.vstack(outs)


class LinearRanker(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def load_joined_candidates(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input_csv, dtype={"case_key": str, "base_key": str}, low_memory=False)
    df["rule_rank"] = pd.to_numeric(df["rule_rank"], errors="coerce")
    df = df[df["rule_rank"] <= args.max_rank].copy()
    df["candidate_objectid"] = pd.to_numeric(df["candidate_objectid"], errors="coerce")
    df["truth_intersects_bool"] = truth_bool(df["truth_intersects"])

    plan = pd.read_csv(args.plan_features_csv, dtype={"case_key": str, "base_key": str})
    plan = plan[["case_key", "plan_available", "plan_best_crop", "mfd_ref", "plan_crop_count"]].copy()
    df = df.merge(plan, on="case_key", how="left")
    df["plan_available"] = df["plan_available"].fillna(False).astype(bool)
    df = df[df["plan_available"]].copy()

    cases = sorted(df["case_key"].unique())
    if args.sample_cases and args.sample_cases < len(cases):
        rng = np.random.default_rng(args.seed)
        cases = sorted(rng.choice(cases, size=args.sample_cases, replace=False).tolist())
        df = df[df["case_key"].isin(cases)].copy()
    if args.focus_rule_top5_misses:
        miss_cases: list[str] = []
        for case_key, group in df.groupby("case_key"):
            if not bool(group[(group["rule_rank"] <= 5) & group["truth_intersects_bool"]].shape[0]):
                miss_cases.append(case_key)
        df = df[df["case_key"].isin(miss_cases)].copy()
    return df.reset_index(drop=True)


def build_embeddings(args: argparse.Namespace, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    cache = args.output_prefix.with_name(args.output_prefix.name + f"_embedding_cache_rank{args.max_rank}_sample{args.sample_cases}.npz")
    meta_cache = args.output_prefix.with_name(args.output_prefix.name + f"_embedding_cache_rank{args.max_rank}_sample{args.sample_cases}.csv")
    if cache.exists() and meta_cache.exists() and not args.force_rebuild_embeddings:
        meta = pd.read_csv(meta_cache, dtype={"case_key": str, "base_key": str})
        data = np.load(cache)
        return meta, data["plan_embeddings"], data["map_embeddings"], json.loads(str(data["info"].item()))

    geoms = visual.load_candidate_geometries(df, args.max_rank, args.council_chunk_size)
    df = df.merge(geoms, on=["candidate_source", "candidate_objectid"], how="left")
    df = df[~df["candidate_geometry"].isna()].copy().reset_index(drop=True)
    _, _, wfs_tree, road_tree, _, _ = visual.load_context()
    # Reload context geoms directly from trees' source by calling helper once.
    wfs = visual.pyogrio.read_dataframe(
        visual.DEFAULT_WFS_MERGED_GPKG,
        layer="mansfield_polygons_in_buffers_merged",
        columns=["geometry"],
        bbox=visual.LOCAL_BBOX,
    )
    roads = visual.pyogrio.read_dataframe(
        visual.DEFAULT_OPENROADS_GPKG,
        layer="road_link",
        columns=["geometry"],
        bbox=visual.LOCAL_BBOX,
    )
    wfs_geoms = list(wfs.geometry)
    road_geoms = list(roads.geometry)

    embedder = ConvNextEmbedder(args.device, args.image_size, args.embed_batch_size)
    unique_plan_paths = sorted({str(p) for p in df["plan_best_crop"].fillna("") if str(p)})
    plan_tensors: list[torch.Tensor] = []
    valid_plan_paths: list[str] = []
    for path in unique_plan_paths:
        if Path(path).exists():
            try:
                plan_tensors.append(preprocess_plan(path, args.image_size))
                valid_plan_paths.append(path)
            except Exception:
                pass
    plan_emb_array = embedder.embed_tensors(plan_tensors)
    plan_by_path = {path: plan_emb_array[i] for i, path in enumerate(valid_plan_paths)}
    embedding_dim = int(plan_emb_array.shape[1]) if len(plan_emb_array) else 768

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    map_embeddings = np.zeros((len(df), len(scales), embedding_dim), dtype="float32")
    tensors: list[torch.Tensor] = []
    tensor_row_scale: list[tuple[int, int]] = []

    def flush_tensor_batch() -> None:
        nonlocal tensors, tensor_row_scale
        if not tensors:
            return
        batch_embeddings = embedder.embed_tensors(tensors)
        for emb_idx, (row_idx, scale_idx) in enumerate(tensor_row_scale):
            map_embeddings[row_idx, scale_idx] = batch_embeddings[emb_idx]
        tensors = []
        tensor_row_scale = []
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    for i, row in df.iterrows():
        geom = row["candidate_geometry"]
        for scale_idx, scale in enumerate(scales):
            rendered = visual.render_candidate_map(
                geom,
                wfs_geoms,
                road_geoms,
                wfs_tree,
                road_tree,
                side_m=scale,
                size=args.image_size,
            )
            tensors.append(preprocess_array(rendered, args.image_size))
            tensor_row_scale.append((i, scale_idx))
            if len(tensors) >= args.render_embed_batch_size:
                flush_tensor_batch()
        if (i + 1) % 1000 == 0:
            print(f"rendered candidate maps: {i + 1}/{len(df)} rows", flush=True)
    flush_tensor_batch()

    plan_embeddings = np.zeros((len(df), embedding_dim), dtype="float32")
    for i, path in enumerate(df["plan_best_crop"].fillna("").astype(str)):
        plan_embeddings[i] = plan_by_path.get(path, np.zeros(embedding_dim, dtype="float32"))

    meta = df.drop(columns=["candidate_geometry"]).copy()
    info = {"scales": scales, "embedding_dim": embedding_dim, "rows": int(len(meta)), "streamed": True}
    np.savez_compressed(cache, plan_embeddings=plan_embeddings, map_embeddings=map_embeddings, info=json.dumps(info))
    meta.to_csv(meta_cache, index=False)
    return meta, plan_embeddings, map_embeddings, info


def visual_feature_matrix(plan_embeddings: np.ndarray, map_embeddings: np.ndarray, include_raw: bool) -> np.ndarray:
    parts: list[np.ndarray] = []
    for scale_idx in range(map_embeddings.shape[1]):
        m = map_embeddings[:, scale_idx, :]
        diff = np.abs(plan_embeddings - m)
        prod = plan_embeddings * m
        cos = (plan_embeddings * m).sum(axis=1, keepdims=True)
        parts.extend([diff, prod, cos])
        if include_raw:
            parts.extend([plan_embeddings, m])
    return np.concatenate(parts, axis=1).astype("float32")


def structured_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [
        "rule_score",
        "rule_rank",
        "best_confidence",
        "candidate_area_m2",
        "distance_to_best_point_m",
        "distance_to_v7_point_m",
        "roi_intersects_count",
        "roi_min_rank",
        "roi_rank_weighted_score",
        "distance_to_nearest_roi_center_m",
        "distance_to_mentioned_road_m",
        "uprn_count",
        "nearest_uprn_to_best_point_m",
        "nearest_uprn_to_v7_point_m",
    ]
    arr = []
    used = []
    for col in cols:
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype="float32")
            if "rank" in col:
                s = -s
            elif "distance" in col or "nearest" in col or "area" in col:
                s = np.log1p(np.clip(s, 0, None))
            arr.append(s[:, None])
            used.append(col)
    for col in [
        "covers_best_point",
        "covers_v7_point",
        "intersects_top1_roi",
        "intersects_protected_current_roi",
        "intersects_evidence_roi",
        "intersects_corridor_roi",
        "covers_range_anchor",
    ]:
        if col in df:
            arr.append(df[col].astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy(dtype="float32")[:, None])
            used.append(col)
    return np.concatenate(arr, axis=1).astype("float32"), used


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def train_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tx, vx = standardize(x[train_idx], x[test_idx])
    ty = y[train_idx].astype("float32")
    positives = max(float(ty.sum()), 1.0)
    negatives = max(float(len(ty) - ty.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    model = LinearRanker(tx.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_tensor = torch.from_numpy(tx.astype("float32"))
    label_tensor = torch.from_numpy(ty)
    for _ in range(epochs):
        order = rng.permutation(len(tx))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = train_tensor[idx].to(device)
            yb = label_tensor[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    with torch.inference_mode():
        logits = model(torch.from_numpy(vx.astype("float32")).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def oof_scores(df: pd.DataFrame, x: np.ndarray, seed: int, folds: int, args: argparse.Namespace) -> np.ndarray:
    y = df["truth_intersects_bool"].to_numpy(dtype=bool)
    scores = np.full(len(df), np.nan, dtype="float32")
    device = torch.device(args.device)
    for fold in range(folds):
        train_cases = {c for c in df["case_key"].unique() if fold_for_case(c, folds) != fold}
        test_cases = {c for c in df["case_key"].unique() if fold_for_case(c, folds) == fold}
        train_idx = np.flatnonzero(df["case_key"].isin(train_cases).to_numpy())
        test_idx = np.flatnonzero(df["case_key"].isin(test_cases).to_numpy())
        scores[test_idx] = train_fold(
            x,
            y,
            train_idx,
            test_idx,
            device=device,
            seed=seed + fold,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.train_batch_size,
        )
        print(f"fold {fold}: train_cases={len(train_cases)} test_cases={len(test_cases)}", flush=True)
    return scores


def rank_scores(df: pd.DataFrame, score_col: str, rank_col: str) -> None:
    ranked = df.sort_values(["case_key", score_col, "rule_rank"], ascending=[True, False, True])
    df[rank_col] = ranked.groupby("case_key").cumcount().add(1).reindex(df.index)


def metric(df: pd.DataFrame, rank_col: str, ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": int(df["case_key"].nunique())}
    for k in ks:
        hit_cases = df[(pd.to_numeric(df[rank_col], errors="coerce") <= k) & df["truth_intersects_bool"]].groupby("case_key").size()
        hits = int((hit_cases > 0).sum())
        out[f"top{k}"] = hits
        out[f"top{k}_rate"] = hits / out["cases"] if out["cases"] else 0.0
    return out


def conservative_visual_rank(df: pd.DataFrame, visual_rank_col: str, out_col: str, preserve_n: int = 4) -> None:
    ranks = pd.Series(np.nan, index=df.index)
    for _, group in df.groupby("case_key"):
        chosen: list[int] = []
        for idx in group.sort_values("rule_rank").head(preserve_n).index:
            chosen.append(idx)
        for idx in group.sort_values([visual_rank_col, "rule_rank"], ascending=[True, True]).index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= 5:
                break
        for rank, idx in enumerate(chosen, start=1):
            ranks.loc[idx] = rank
    df[out_col] = ranks


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, min(2, int(args.torch_threads))))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = load_joined_candidates(args)
    if len(df) > args.max_candidate_rows and not args.allow_large_run:
        raise SystemExit(
            f"Refusing to run {len(df)} candidate rows without --allow-large-run. "
            f"Use --sample-cases/--max-rank to keep <= {args.max_candidate_rows}, "
            "or pass --allow-large-run after confirming the run is safe."
        )
    df, plan_emb, map_emb, info = build_embeddings(args, df)
    visual_x = visual_feature_matrix(plan_emb, map_emb, include_raw=args.include_raw_embeddings)
    struct_x, struct_cols = structured_matrix(df)
    hybrid_x = np.concatenate([visual_x, struct_x], axis=1).astype("float32")

    df["vision_oof_score"] = oof_scores(df, visual_x, args.seed, args.folds, args)
    df["hybrid_oof_score"] = oof_scores(df, hybrid_x, args.seed + 1000, args.folds, args)
    rank_scores(df, "vision_oof_score", "vision_oof_rank")
    rank_scores(df, "hybrid_oof_score", "hybrid_oof_rank")
    conservative_visual_rank(df, "vision_oof_rank", "rule4_vision1_rank")
    conservative_visual_rank(df, "hybrid_oof_rank", "rule4_hybrid1_rank")

    ks = sorted(set(k for k in [1, 3, 5, 8, 10, args.max_rank] if k <= args.max_rank))
    summary = {
        "input_csv": str(args.input_csv),
        "output_prefix": str(args.output_prefix),
        "cases": int(df["case_key"].nunique()),
        "rows": int(len(df)),
        "max_rank": args.max_rank,
        "sample_cases": args.sample_cases,
        "focus_rule_top5_misses": args.focus_rule_top5_misses,
        "embedding_info": info,
        "structured_cols": struct_cols,
        "rule": metric(df, "rule_rank", ks),
        "vision_model": metric(df, "vision_oof_rank", ks),
        "hybrid_model": metric(df, "hybrid_oof_rank", ks),
        "rule4_vision1": metric(df[~df["rule4_vision1_rank"].isna()].copy(), "rule4_vision1_rank", [1, 3, 5]),
        "rule4_hybrid1": metric(df[~df["rule4_hybrid1_rank"].isna()].copy(), "rule4_hybrid1_rank", [1, 3, 5]),
    }
    case_rows = []
    for case_key, group in df.groupby("case_key"):
        rule = bool(group[(group["rule_rank"] <= 5) & group["truth_intersects_bool"]].shape[0])
        vision = bool(group[(group["vision_oof_rank"] <= 5) & group["truth_intersects_bool"]].shape[0])
        hybrid = bool(group[(group["hybrid_oof_rank"] <= 5) & group["truth_intersects_bool"]].shape[0])
        case_rows.append(
            {
                "case_key": case_key,
                "sample_split": group["sample_split"].iloc[0],
                "best_confidence": group["best_confidence"].iloc[0],
                "mfd_ref": group.get("mfd_ref", pd.Series([""])).iloc[0],
                "plan_best_crop": group.get("plan_best_crop", pd.Series([""])).iloc[0],
                "rule_top5_hit": rule,
                "vision_top5_hit": vision,
                "hybrid_top5_hit": hybrid,
                "vision_recovers_rule": vision and not rule,
                "vision_loses_rule": rule and not vision,
                "hybrid_recovers_rule": hybrid and not rule,
                "hybrid_loses_rule": rule and not hybrid,
            }
        )
    case_df = pd.DataFrame(case_rows)
    for prefix in ["vision", "hybrid"]:
        summary[f"{prefix}_recovers_rule_top5"] = int(case_df[f"{prefix}_recovers_rule"].sum())
        summary[f"{prefix}_loses_rule_top5"] = int(case_df[f"{prefix}_loses_rule"].sum())

    summary_json = args.output_prefix.with_suffix(".summary.json")
    scored_csv = args.output_prefix.with_name(args.output_prefix.name + "_scored.csv")
    case_csv = args.output_prefix.with_name(args.output_prefix.name + "_case_metrics.csv")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    keep_cols = [
        "case_key",
        "base_key",
        "sample_split",
        "original_address",
        "best_confidence",
        "candidate_source",
        "candidate_id",
        "candidate_theme",
        "candidate_area_m2",
        "truth_intersects",
        "truth_intersects_bool",
        "rule_rank",
        "rule_score",
        "vision_oof_score",
        "vision_oof_rank",
        "hybrid_oof_score",
        "hybrid_oof_rank",
        "rule4_vision1_rank",
        "rule4_hybrid1_rank",
        "mfd_ref",
        "plan_best_crop",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df.sort_values(["case_key", "hybrid_oof_rank"])[keep_cols].to_csv(scored_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"summary_json={summary_json}", flush=True)
    print(f"scored_csv={scored_csv}", flush=True)
    print(f"case_csv={case_csv}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ConvNeXt plan visual reranking experiment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plan-features-csv", type=Path, default=DEFAULT_PLAN_FEATURES)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--sample-cases", type=int, default=50)
    parser.add_argument("--focus-rule-top5-misses", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--scales", default="160,260,420")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--render-embed-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--council-chunk-size", type=int, default=500)
    parser.add_argument("--include-raw-embeddings", action="store_true")
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--max-candidate-rows", type=int, default=5000)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
