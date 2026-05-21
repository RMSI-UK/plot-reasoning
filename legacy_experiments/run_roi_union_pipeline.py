#!/usr/bin/env python3
"""Run the Mansfield ROI-union candidate pipeline end to end.

This script wraps the experiment chain that produced the current 94.85%
multi-box union recall:

1. optionally sample base cases from the full GPKG
2. run OS/Gemini/GOG address candidates
3. run OCR grid rerank
4. run OpenNames and OpenRoads refinements
5. audit plan OCR anchors
6. export evidence points for visualization
7. build protected-current + evidence multi-ROIs
8. add road-corridor ROIs
9. optionally combine with one or more existing runs

The default arguments reproduce the new random1000 run and combine it with the
existing random200 baseline.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
NO_AI_STAGES = PROJECT_ROOT / "no_ai_candidate_boxes" / "stages"
TMP = Path("/data/mansfield/spatial/polygon-layer/tmp_output")

DEFAULT_FULL_GPKG = Path("/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg")
DEFAULT_FULL_LAYER = "mansfield-manual-polygon-link"
DEFAULT_EXISTING_SAMPLE_GPKG = TMP / "mansfield-manual-polygon-link_random200_seed42.gpkg"
DEFAULT_EXISTING_SAMPLE_LAYER = "mansfield-manual-polygon-link-random200"
DEFAULT_EXISTING_TAG = "mansfield-manual-polygon-link_random200_seed42"
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OPEN_NAMES = Path("/data/base-data/opname_csv_gb/os_open_names_uk.sqlite")
DEFAULT_OCR_JSONL = Path("/data/mansfield/ocr/all_v5ocr.jsonl")
DEFAULT_FIND_PLAN = Path("/data/mansfield/skill/find-plan/full-scanraw/find_plan_results.jsonl")
DEFAULT_PLAN_IMAGES = Path("/data/mansfield/backup/plan_images.csv")
DEFAULT_KEYS_FILE = Path("/env/key")
DEFAULT_GEOCODER = Path(__file__).resolve().parent / "1_address_to_point_gemini.py"


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def tag_paths(tag: str) -> dict[str, Path]:
    return {
        "sample_gpkg": TMP / f"{tag}.gpkg",
        "gemini_json": TMP / f"{tag}_gemini.json",
        "gemini_xlsx": TMP / f"{tag}_gemini.xlsx",
        "ocr": TMP / f"{tag}_ocr_rerank_v2",
        "openname": TMP / f"{tag}_ocr_openname_v6",
        "openroads": TMP / f"{tag}_ocr_openroads_v10",
        "audit": TMP / f"{tag}_plan_ocr_anchor_audit",
        "evidence": TMP / f"{tag}_candidate_evidence_roi_v1",
        "multi": TMP / f"{tag}_multi_roi_top2_v2",
        "corridor": TMP / f"{tag}_corridor_augmented_roi_step50_v1",
    }


def suffix_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def create_sample(args: argparse.Namespace, tag: str, layer: str) -> Path:
    paths = tag_paths(tag)
    out = paths["sample_gpkg"]
    if args.sample_gpkg:
        return Path(args.sample_gpkg)
    if out.exists() and args.resume:
        print(f"reuse sample: {out}", flush=True)
        return out
    if args.dry_run:
        print(f"would create sample: {out}", flush=True)
        return out

    full = gpd.read_file(args.full_gpkg, layer=args.full_layer)
    full["unique_key"] = full["unique_key"].astype(str)
    exclude_keys: set[str] = set()
    for gpkg, lyr in zip(args.exclude_gpkg, args.exclude_layer):
        gdf = gpd.read_file(gpkg, layer=lyr)
        exclude_keys.update(gdf["unique_key"].astype(str))

    remaining = full[~full["unique_key"].isin(exclude_keys)].copy()
    if len(remaining) < args.sample_size:
        raise SystemExit(f"Only {len(remaining)} rows remain after exclusions; need {args.sample_size}")
    sample = remaining.sample(n=args.sample_size, random_state=args.seed).sort_values("unique_key")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    sample.to_file(out, layer=layer, driver="GPKG")
    print(
        f"sample written: {out} layer={layer} rows={len(sample)} "
        f"excluded={len(exclude_keys)} overlap={sample['unique_key'].isin(exclude_keys).sum()}",
        flush=True,
    )
    return out


def maybe_run(cmd: list[str], output: Path, args: argparse.Namespace) -> None:
    if args.resume and output.exists():
        print(f"skip existing: {output}", flush=True)
        return
    run(cmd, dry_run=args.dry_run)


def run_one_pipeline(args: argparse.Namespace, tag: str, sample_gpkg: Path, sample_layer: str) -> dict[str, Path]:
    p = tag_paths(tag)

    maybe_run(
        [
            sys.executable,
            str(args.geocoder_script),
            "--input-gpkg",
            str(sample_gpkg),
            "--output-json",
            str(p["gemini_json"]),
            "--output-xlsx",
            str(p["gemini_xlsx"]),
            "--keys-file",
            str(args.keys_file),
            "--workers",
            str(args.geocode_workers),
            "--web-research-workers",
            str(args.web_research_workers),
        ]
        + (["--enable-web-research"] if args.enable_web_research else []),
        p["gemini_json"],
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "ocr_candidate_rerank.py"),
            "--input-json",
            str(p["gemini_json"]),
            "--input-gpkg",
            str(sample_gpkg),
            "--layer",
            sample_layer,
            "--ocr-jsonl",
            str(args.ocr_jsonl),
            "--output-prefix",
            str(p["ocr"]),
        ],
        p["ocr"].with_suffix(".csv"),
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "openname_anchor.py"),
            "--prev-csv",
            str(p["ocr"].with_suffix(".csv")),
            "--input-json",
            str(p["gemini_json"]),
            "--input-gpkg",
            str(sample_gpkg),
            "--layer",
            sample_layer,
            "--open-names",
            str(args.open_names),
            "--output-prefix",
            str(p["openname"]),
        ],
        p["openname"].with_suffix(".csv"),
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "openroads_relation.py"),
            "--prev-csv",
            str(p["openname"].with_suffix(".csv")),
            "--input-gpkg",
            str(sample_gpkg),
            "--layer",
            sample_layer,
            "--open-names",
            str(args.open_names),
            "--open-roads",
            str(args.open_roads),
            "--output-prefix",
            str(p["openroads"]),
        ],
        p["openroads"].with_suffix(".csv"),
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "plan_ocr_anchor_audit.py"),
            "--gpkg",
            str(sample_gpkg),
            "--layer",
            sample_layer,
            "--prev-csv",
            str(p["openroads"].with_suffix(".csv")),
            "--find-plan",
            str(args.find_plan),
            "--plan-images",
            str(args.plan_images),
            "--ocr",
            str(args.ocr_jsonl),
            "--open-roads",
            str(args.open_roads),
            "--open-names",
            str(args.open_names),
            "--output-prefix",
            str(p["audit"]),
        ],
        p["audit"].with_suffix(".csv"),
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "evidence_roi.py"),
            "--input-json",
            str(p["gemini_json"]),
            "--v10-csv",
            str(p["openroads"].with_suffix(".csv")),
            "--audit-csv",
            str(p["audit"].with_suffix(".csv")),
            "--full-gpkg",
            str(args.full_gpkg),
            "--full-layer",
            args.full_layer,
            "--sample-gpkg",
            str(sample_gpkg),
            "--sample-layer",
            sample_layer,
            "--open-roads",
            str(args.open_roads),
            "--output-prefix",
            str(p["evidence"]),
        ],
        suffix_path(p["evidence"], "_evidence.csv"),
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "multi_roi_candidates.py"),
            "--input-json",
            str(p["gemini_json"]),
            "--v10-csv",
            str(p["openroads"].with_suffix(".csv")),
            "--audit-csv",
            str(p["audit"].with_suffix(".csv")),
            "--full-gpkg",
            str(args.full_gpkg),
            "--full-layer",
            args.full_layer,
            "--sample-gpkg",
            str(sample_gpkg),
            "--sample-layer",
            sample_layer,
            "--open-roads",
            str(args.open_roads),
            "--evidence-roi-count",
            str(args.evidence_roi_count),
            "--output-prefix",
            str(p["multi"]),
        ],
        p["multi"].with_suffix(".csv"),
        args,
    )

    maybe_run(
        [
            sys.executable,
            str(NO_AI_STAGES / "corridor_augmented_roi.py"),
            "--input-json",
            str(p["gemini_json"]),
            "--v10-csv",
            str(p["openroads"].with_suffix(".csv")),
            "--audit-csv",
            str(p["audit"].with_suffix(".csv")),
            "--base-csv",
            str(p["multi"].with_suffix(".csv")),
            "--base-rois",
            str(suffix_path(p["multi"], "_rois.csv")),
            "--full-gpkg",
            str(args.full_gpkg),
            "--full-layer",
            args.full_layer,
            "--sample-gpkg",
            str(sample_gpkg),
            "--sample-layer",
            sample_layer,
            "--open-roads",
            str(args.open_roads),
            "--step-m",
            str(args.corridor_step_m),
            "--pad-m",
            str(args.corridor_pad_m),
            "--max-per-road",
            str(args.max_per_road),
            "--max-total-samples",
            str(args.max_total_samples),
            "--output-prefix",
            str(p["corridor"]),
        ],
        p["corridor"].with_suffix(".csv"),
        args,
    )
    return p


def load_json_rows(path: Path, split: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("rows", data if isinstance(data, list) else [])
    for row in rows:
        row["sample_split"] = split
    return rows


def combine_runs(
    args: argparse.Namespace,
    combined_tag: str,
    run_specs: list[tuple[str, str, Path, str]],
) -> None:
    if args.dry_run:
        print(f"would combine into tag: {combined_tag}", flush=True)
        return

    combined_gpkg = TMP / f"{combined_tag}.gpkg"
    combined_layer = f"{combined_tag.replace('mansfield-manual-polygon-link_', '')[:50]}"
    gdfs = []
    for split, _tag, gpkg, layer in run_specs:
        gdf = gpd.read_file(gpkg, layer=layer)
        gdf["sample_split"] = split
        gdfs.append(gdf)
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    if combined_gpkg.exists():
        combined_gpkg.unlink()
    combined_gdf.to_file(combined_gpkg, layer=combined_layer, driver="GPKG")

    # Main row-level artifacts.
    suffixes = [
        "_ocr_openroads_v10.csv",
        "_plan_ocr_anchor_audit.csv",
        "_candidate_evidence_roi_v1_evidence.csv",
        "_multi_roi_top2_v2.csv",
        "_multi_roi_top2_v2_rois.csv",
        "_multi_roi_top2_v2_top20.csv",
        "_corridor_augmented_roi_step50_v1.csv",
        "_corridor_augmented_roi_step50_v1_rois.csv",
    ]
    for suffix in suffixes:
        frames = []
        for split, tag, _gpkg, _layer in run_specs:
            path = TMP / f"{tag}{suffix}"
            if not path.exists():
                print(f"missing combined artifact: {path}", flush=True)
                continue
            df = pd.read_csv(path, dtype=str)
            df["sample_split"] = split
            frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(TMP / f"{combined_tag}{suffix}", index=False)

    json_rows: list[dict[str, Any]] = []
    for split, tag, _gpkg, _layer in run_specs:
        path = TMP / f"{tag}_gemini.json"
        if path.exists():
            json_rows.extend(load_json_rows(path, split))
    (TMP / f"{combined_tag}_gemini.json").write_text(
        json.dumps({"meta": {"rows": len(json_rows), "sources": [spec[1] for spec in run_specs]}, "rows": json_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_combined_summary(combined_tag, run_specs)


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def write_combined_summary(combined_tag: str, run_specs: list[tuple[str, str, Path, str]]) -> None:
    final_csv = TMP / f"{combined_tag}_corridor_augmented_roi_step50_v1.csv"
    if not final_csv.exists():
        return
    final = pd.read_csv(final_csv)
    for col in ["union_contains_target_centroid", "union_intersects_target_polygon"]:
        final[col] = normalize_bool(final[col])
    for col in ["best_confidence", "union_candidate_polygon_count", "added_corridor_roi_count"]:
        final[col] = pd.to_numeric(final[col], errors="coerce")
    ok = final[final["status"].astype(str) == "ok"].copy()
    low = ok[ok["best_confidence"] < 75].copy()

    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(df)),
            "polygon_intersects_union": int(df["union_intersects_target_polygon"].sum()) if len(df) else 0,
            "centroid_inside_union": int(df["union_contains_target_centroid"].sum()) if len(df) else 0,
            "hit_rate": float(df["union_intersects_target_polygon"].sum() / len(df)) if len(df) else 0.0,
            "median_candidate_count": float(df["union_candidate_polygon_count"].median()) if len(df) else 0.0,
            "mean_candidate_count": float(df["union_candidate_polygon_count"].mean()) if len(df) else 0.0,
        }

    summary = {
        "base_case_count": int(sum(len(gpd.read_file(gpkg, layer=layer)) for _split, _tag, gpkg, layer in run_specs)),
        "expanded_final_rows": int(len(final)),
        "ok_rows": int(len(ok)),
        "ok_only": summarize(ok),
        "low_confidence_ok": summarize(low),
        "splits": {str(k): int(v) for k, v in final["sample_split"].value_counts(dropna=False).to_dict().items()},
        "output_final_csv": str(final_csv),
        "output_rois_csv": str(TMP / f"{combined_tag}_corridor_augmented_roi_step50_v1_rois.csv"),
    }
    (TMP / f"{combined_tag}_corridor_augmented_roi_step50_v1.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    xlsx = TMP / f"{combined_tag}_corridor_augmented_roi_step50_v1.xlsx"
    rois_csv = TMP / f"{combined_tag}_corridor_augmented_roi_step50_v1_rois.csv"
    with pd.ExcelWriter(xlsx) as writer:
        final.to_excel(writer, sheet_name="summary", index=False)
        if rois_csv.exists():
            pd.read_csv(rois_csv).to_excel(writer, sheet_name="rois", index=False)
        pd.DataFrame(
            [
                {"slice": "ok_only", **summary["ok_only"]},
                {"slice": "low_confidence_ok", **summary["low_confidence_ok"]},
            ]
        ).to_excel(writer, sheet_name="metrics", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_existing_run(value: str) -> tuple[str, str, Path, str]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("existing run must be split:tag:gpkg:layer")
    return parts[0], parts[1], Path(parts[2]), parts[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-gpkg", type=Path, default=DEFAULT_FULL_GPKG)
    parser.add_argument("--full-layer", default=DEFAULT_FULL_LAYER)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--tag", default="mansfield-manual-polygon-link_random1000_seed43_excl_random200")
    parser.add_argument("--sample-layer", default="mansfield-manual-polygon-link-random1000-seed43")
    parser.add_argument("--sample-gpkg", type=Path, help="Use an existing sample GPKG instead of sampling.")
    parser.add_argument("--exclude-gpkg", type=Path, action="append", default=[DEFAULT_EXISTING_SAMPLE_GPKG])
    parser.add_argument("--exclude-layer", action="append", default=[DEFAULT_EXISTING_SAMPLE_LAYER])
    parser.add_argument("--keys-file", type=Path, default=DEFAULT_KEYS_FILE)
    parser.add_argument("--geocoder-script", type=Path, default=DEFAULT_GEOCODER)
    parser.add_argument("--geocode-workers", type=int, default=16)
    parser.add_argument("--web-research-workers", type=int, default=8)
    parser.add_argument("--enable-web-research", action="store_true")
    parser.add_argument("--ocr-jsonl", type=Path, default=DEFAULT_OCR_JSONL)
    parser.add_argument("--find-plan", type=Path, default=DEFAULT_FIND_PLAN)
    parser.add_argument("--plan-images", type=Path, default=DEFAULT_PLAN_IMAGES)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--open-names", type=Path, default=DEFAULT_OPEN_NAMES)
    parser.add_argument("--evidence-roi-count", type=int, default=2)
    parser.add_argument("--corridor-step-m", type=float, default=50.0)
    parser.add_argument("--corridor-pad-m", type=float, default=180.0)
    parser.add_argument("--max-per-road", type=int, default=2)
    parser.add_argument("--max-total-samples", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--combined-tag", default="mansfield-manual-polygon-link_random1200_seed42_43_combined")
    parser.add_argument(
        "--existing-run",
        action="append",
        type=parse_existing_run,
        default=[
            (
                "random200_seed42",
                DEFAULT_EXISTING_TAG,
                DEFAULT_EXISTING_SAMPLE_GPKG,
                DEFAULT_EXISTING_SAMPLE_LAYER,
            )
        ],
        help="Existing run to combine, as split:tag:gpkg:layer. Can repeat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.exclude_gpkg) != len(args.exclude_layer):
        raise SystemExit("--exclude-gpkg and --exclude-layer counts must match")
    sample_gpkg = create_sample(args, args.tag, args.sample_layer)
    run_one_pipeline(args, args.tag, sample_gpkg, args.sample_layer)
    if not args.skip_combine:
        current = (args.tag.replace("mansfield-manual-polygon-link_", ""), args.tag, sample_gpkg, args.sample_layer)
        combine_runs(args, args.combined_tag, [*args.existing_run, current])


if __name__ == "__main__":
    main()
