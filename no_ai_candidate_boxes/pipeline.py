#!/usr/bin/env python3
"""Run the full A-stage evidence pipeline without Gemini.

The old 1200-case workflow used a rich A-stage:
OCR candidate rerank -> OpenNames -> OpenRoads -> plan OCR audit ->
multi ROI -> corridor augmented ROI.

This runner keeps that evidence chain and can either build a local geocoder-like
JSON or consume an existing ``1_address_to_point_gemini.py`` JSON/JSONL as the
upstream address-point evidence.  It does not call Gemini or any external model
API.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_CONFIG, bbox_arg, cfg_layer, cfg_param, cfg_path, names_arg, resolved_config


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
STAGES = PACKAGE_ROOT / "stages"


def suffix(prefix: Path, ending: str) -> Path:
    return prefix.with_name(prefix.name + ending)


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def copy_alias(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"alias {src} -> {dst}", flush=True)
        return
    if src.exists():
        shutil.copyfile(src, dst)


def add_common_spatial_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    if args.local_bbox:
        cmd.extend(["--local-bbox", args.local_bbox])
    return cmd


def add_name_args(cmd: list[str], args: argparse.Namespace, names: list[str]) -> list[str]:
    for name in names:
        value = getattr(args, name)
        if value:
            cmd.extend([f"--{name.replace('_', '-')}", value])
    return cmd


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Council profile JSON.")
    config_parser.add_argument("--council-root", type=Path, help="Override {council_root} in the config profile.")
    config_parser.add_argument("--base-data-root", type=Path, help="Override {base_data_root} in the config profile.")
    config_parser.add_argument("--output-root", type=Path, help="Override {output_root} in the config profile.")
    config_args, _ = config_parser.parse_known_args()
    config = resolved_config(
        config_args.config,
        council_root=config_args.council_root,
        base_data_root=config_args.base_data_root,
        output_root=config_args.output_root,
    )

    parser = argparse.ArgumentParser(
        description="Run the no-AI candidate-box pipeline.",
        parents=[config_parser],
    )
    parser.add_argument("--truth-gpkg", type=Path, default=cfg_path(config, "truth_gpkg"))
    parser.add_argument("--truth-layer", default=cfg_layer(config, "truth_layer"))
    parser.add_argument("--output-prefix", type=Path, default=cfg_path(config, "output_prefix"))
    parser.add_argument(
        "--address-point-json",
        type=Path,
        help="Optional JSON/JSONL from spatial_capture_production/1_address_to_point_gemini.py; replaces build_input_candidates.",
    )
    parser.add_argument(
        "--address-point-gpkg",
        type=Path,
        help="Primary 1_address/input GPKG. When supplied, all local stages use this as the case layer.",
    )
    parser.add_argument("--address-point-layer", help="Layer in --address-point-gpkg.")
    parser.add_argument("--address-column", default=cfg_param(config, "address_column", "chargegeog"))
    parser.add_argument("--text-xlsx", type=Path, default=cfg_path(config, "text_xlsx"))
    parser.add_argument("--ocr-jsonl", type=Path, default=cfg_path(config, "ocr_jsonl"))
    parser.add_argument("--open-roads", type=Path, default=cfg_path(config, "open_roads"))
    parser.add_argument("--open-names", type=Path, default=cfg_path(config, "open_names"))
    parser.add_argument("--open-names-dir", type=Path, default=cfg_path(config, "open_names_dir"))
    parser.add_argument("--find-plan", type=Path, default=cfg_path(config, "find_plan"))
    parser.add_argument("--plan-images", type=Path, default=cfg_path(config, "plan_images"))
    parser.add_argument("--full-gpkg", type=Path, default=cfg_path(config, "full_gpkg"))
    parser.add_argument("--full-layer", default=cfg_layer(config, "full_layer"))
    parser.add_argument("--local-bbox", default=bbox_arg(config.get("local_bbox")))
    parser.add_argument("--locality-names", default=names_arg(config.get("locality_names")))
    parser.add_argument("--generic-place-names", default=names_arg(config.get("generic_place_names")))
    parser.add_argument("--build-generic-place-names", default=names_arg(config.get("build_generic_place_names", config.get("generic_place_names"))))
    parser.add_argument("--openname-generic-place-names", default=names_arg(config.get("openname_generic_place_names", config.get("generic_place_names"))))
    parser.add_argument("--plan-generic-place-names", default=names_arg(config.get("plan_generic_place_names", config.get("generic_place_names"))))
    parser.add_argument("--allowed-locality-names", default=names_arg(config.get("allowed_locality_names")))
    parser.add_argument("--bad-locality-names", default=names_arg(config.get("bad_locality_names")))
    parser.add_argument("--county-names", default=names_arg(config.get("county_names")))
    parser.add_argument("--district-names", default=names_arg(config.get("district_names")))
    parser.add_argument("--max-rois", type=int, default=cfg_param(config, "max_rois", 8))
    parser.add_argument("--max-text-candidates", type=int, default=cfg_param(config, "max_text_candidates", 20))
    parser.add_argument("--evidence-roi-count", type=int, default=cfg_param(config, "evidence_roi_count", 5))
    parser.add_argument(
        "--roi-side",
        type=float,
        default=cfg_param(config, "roi_side", 180.0),
        help="Side length in metres for evidence boxes. 180m keeps the box small while tolerating adjacent/rear/plot offsets.",
    )
    parser.add_argument("--corridor-step-m", type=float, default=cfg_param(config, "corridor_step_m", 50.0))
    parser.add_argument("--corridor-pad-m", type=float, default=cfg_param(config, "corridor_pad_m", 180.0))
    parser.add_argument("--max-per-road", type=int, default=cfg_param(config, "max_per_road", 2))
    parser.add_argument("--max-total-samples", type=int, default=cfg_param(config, "max_total_samples", 8))
    parser.add_argument("--no-clean-aliases", action="store_true", help="Do not write no_ai_* aliases for the legacy output files.")
    parser.add_argument("--disable-range-parity-correction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    case_gpkg = args.address_point_gpkg or args.truth_gpkg
    case_layer = args.address_point_layer or args.truth_layer
    if not case_gpkg:
        parser.error("--address-point-gpkg or --truth-gpkg is required")
    if not case_layer:
        parser.error("--address-point-layer or --truth-layer is required")
    if args.address_point_gpkg and not args.address_point_layer:
        parser.error("--address-point-layer is required with --address-point-gpkg")
    if not args.output_prefix:
        parser.error("--output-prefix is required unless paths.output_prefix is set in --config")
    if args.address_point_json and not args.dry_run and not args.address_point_json.exists():
        parser.error(f"--address-point-json does not exist: {args.address_point_json}")
    if not args.dry_run and args.address_point_gpkg and not args.address_point_gpkg.exists():
        parser.error(f"--address-point-gpkg does not exist: {args.address_point_gpkg}")

    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)

    geocoder_prefix = suffix(out, "_geocoder")
    ocr_prefix = suffix(out, "_ocr")
    openname_prefix = suffix(out, "_openname")
    openroads_prefix = suffix(out, "_openroads")
    audit_prefix = suffix(out, "_audit")
    multi_prefix = suffix(out, "_multi_roi_top2")
    corridor_prefix = suffix(out, "_corridor_step50")

    geocoder_json = (
        suffix(geocoder_prefix, "_input_candidates.json")
        if args.address_point_json or args.address_point_gpkg
        else suffix(geocoder_prefix, "_gemini_like.json")
    )
    openroads_csv = openroads_prefix.with_suffix(".csv")
    audit_csv = audit_prefix.with_suffix(".csv")
    multi_csv = multi_prefix.with_suffix(".csv")
    multi_rois = suffix(multi_prefix, "_rois.csv")
    final_case_summary = corridor_prefix.with_suffix(".csv")
    final_rois = suffix(corridor_prefix, "_rois.csv")

    if args.address_point_json or args.address_point_gpkg:
        cmd = [
            sys.executable,
            str(STAGES / "address_point_evidence_adapter.py"),
            "--output-prefix",
            str(geocoder_prefix),
            "--open-roads",
            str(args.open_roads),
            "--roi-side",
            str(args.roi_side),
            "--max-rois",
            str(args.max_rois),
        ]
        if args.address_point_json:
            cmd.extend(["--address-point-json", str(args.address_point_json)])
        if args.address_point_gpkg:
            cmd.extend(["--address-point-gpkg", str(args.address_point_gpkg), "--address-point-layer", args.address_point_layer])
            cmd.extend(["--address-column", args.address_column])
        else:
            cmd.extend(["--case-gpkg", str(case_gpkg), "--case-layer", case_layer])
        add_common_spatial_args(cmd, args)
        if args.build_generic_place_names:
            cmd.extend(["--generic-place-names", args.build_generic_place_names])
        if args.disable_range_parity_correction:
            cmd.append("--disable-range-parity-correction")
        run(cmd, args.dry_run)
    else:
        cmd = [
            sys.executable,
            str(STAGES / "build_input_candidates.py"),
            "--truth-gpkg",
            str(case_gpkg),
            "--truth-layer",
            case_layer,
            "--text-xlsx",
            str(args.text_xlsx),
            "--ocr-jsonl",
            str(args.ocr_jsonl),
            "--open-roads",
            str(args.open_roads),
            "--open-names-dir",
            str(args.open_names_dir),
            "--output-prefix",
            str(geocoder_prefix),
            "--max-rois",
            str(args.max_rois),
            "--max-text-candidates",
            str(args.max_text_candidates),
        ]
        add_common_spatial_args(cmd, args)
        if args.build_generic_place_names:
            cmd.extend(["--generic-place-names", args.build_generic_place_names])
        run(cmd, args.dry_run)

    cmd = [
            sys.executable,
            str(STAGES / "ocr_candidate_rerank.py"),
            "--input-json",
            str(geocoder_json),
            "--input-gpkg",
            str(case_gpkg),
            "--layer",
            case_layer,
            "--ocr-jsonl",
            str(args.ocr_jsonl),
            "--output-prefix",
            str(ocr_prefix),
        ]
    add_common_spatial_args(cmd, args)
    add_name_args(cmd, args, ["locality_names", "allowed_locality_names", "bad_locality_names"])
    run(cmd, args.dry_run)

    cmd = [
            sys.executable,
            str(STAGES / "openname_anchor.py"),
            "--prev-csv",
            str(ocr_prefix.with_suffix(".csv")),
            "--input-json",
            str(geocoder_json),
            "--input-gpkg",
            str(case_gpkg),
            "--layer",
            case_layer,
            "--open-names",
            str(args.open_names),
            "--output-prefix",
            str(openname_prefix),
        ]
    add_common_spatial_args(cmd, args)
    if args.openname_generic_place_names:
        cmd.extend(["--generic-place-names", args.openname_generic_place_names])
    add_name_args(cmd, args, ["county_names", "district_names"])
    run(cmd, args.dry_run)

    cmd = [
            sys.executable,
            str(STAGES / "openroads_relation.py"),
            "--prev-csv",
            str(openname_prefix.with_suffix(".csv")),
            "--input-gpkg",
            str(case_gpkg),
            "--layer",
            case_layer,
            "--open-names",
            str(args.open_names),
            "--open-roads",
            str(args.open_roads),
            "--output-prefix",
            str(openroads_prefix),
        ]
    add_common_spatial_args(cmd, args)
    if args.openname_generic_place_names:
        cmd.extend(["--generic-place-names", args.openname_generic_place_names])
    run(cmd, args.dry_run)

    cmd = [
            sys.executable,
            str(STAGES / "plan_ocr_anchor_audit.py"),
            "--gpkg",
            str(case_gpkg),
            "--layer",
            case_layer,
            "--prev-csv",
            str(openroads_csv),
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
            str(audit_prefix),
        ]
    add_common_spatial_args(cmd, args)
    if args.plan_generic_place_names:
        cmd.extend(["--generic-place-names", args.plan_generic_place_names])
    run(cmd, args.dry_run)

    cmd = [
            sys.executable,
            str(STAGES / "multi_roi_candidates.py"),
            "--input-json",
            str(geocoder_json),
            "--v10-csv",
            str(openroads_csv),
            "--audit-csv",
            str(audit_csv),
            "--full-gpkg",
            str(args.full_gpkg),
            "--full-layer",
            args.full_layer,
            "--sample-gpkg",
            str(case_gpkg),
            "--sample-layer",
            case_layer,
            "--open-roads",
            str(args.open_roads),
            "--evidence-roi-count",
            str(args.evidence_roi_count),
            "--roi-side",
            str(args.roi_side),
            "--output-prefix",
            str(multi_prefix),
        ]
    add_common_spatial_args(cmd, args)
    run(cmd, args.dry_run)

    cmd = [
            sys.executable,
            str(STAGES / "corridor_augmented_roi.py"),
            "--input-json",
            str(geocoder_json),
            "--v10-csv",
            str(openroads_csv),
            "--audit-csv",
            str(audit_csv),
            "--base-csv",
            str(multi_csv),
            "--base-rois",
            str(multi_rois),
            "--full-gpkg",
            str(args.full_gpkg),
            "--full-layer",
            args.full_layer,
            "--sample-gpkg",
            str(case_gpkg),
            "--sample-layer",
            case_layer,
            "--open-roads",
            str(args.open_roads),
            "--step-m",
            str(args.corridor_step_m),
            "--pad-m",
            str(args.corridor_pad_m),
            "--roi-side",
            str(args.roi_side),
            "--max-per-road",
            str(args.max_per_road),
            "--max-total-samples",
            str(args.max_total_samples),
            "--output-prefix",
            str(corridor_prefix),
        ]
    add_common_spatial_args(cmd, args)
    run(cmd, args.dry_run)

    clean_candidates = suffix(out, "_no_ai_input_candidates.json")
    clean_summary = suffix(out, "_no_ai_case_summary.csv")
    clean_boxes = suffix(out, "_no_ai_candidate_boxes.csv")
    clean_manifest = suffix(out, "_no_ai_manifest.json")

    if not args.no_clean_aliases:
        copy_alias(geocoder_json, clean_candidates, args.dry_run)
        copy_alias(final_case_summary, clean_summary, args.dry_run)
        copy_alias(final_rois, clean_boxes, args.dry_run)
        manifest = {
            "pipeline_name": "no_ai_candidate_boxes",
            "profile_name": config.get("profile_name"),
            "council_name": config.get("council_name"),
            "config": str(args.config),
            "no_ai_api_calls": True,
            "address_point_json": str(args.address_point_json) if args.address_point_json else "",
            "entrypoint": str(PROJECT_ROOT / "run_no_ai_candidate_boxes.py"),
            "pipeline_script": str(PACKAGE_ROOT / "pipeline.py"),
            "parameters": {
                "truth_gpkg": str(args.truth_gpkg),
                "truth_layer": args.truth_layer,
                "address_point_gpkg": str(args.address_point_gpkg) if args.address_point_gpkg else "",
                "address_point_layer": args.address_point_layer or "",
                "case_gpkg_used": str(case_gpkg),
                "case_layer_used": case_layer,
                "local_bbox": args.local_bbox,
                "locality_names": args.locality_names,
                "roi_side": args.roi_side,
                "evidence_roi_count": args.evidence_roi_count,
                "corridor_step_m": args.corridor_step_m,
                "corridor_pad_m": args.corridor_pad_m,
                "max_per_road": args.max_per_road,
                "max_total_samples": args.max_total_samples,
                "range_parity_correction": not args.disable_range_parity_correction,
            },
            "clean_outputs": {
                "input_candidates_json": str(clean_candidates),
                "case_summary_csv": str(clean_summary),
                "candidate_boxes_csv": str(clean_boxes),
            },
            "legacy_outputs": {
                "geocoder_json": str(geocoder_json),
                "case_summary_csv": str(final_case_summary),
                "candidate_boxes_csv": str(final_rois),
                "multi_roi_csv": str(multi_csv),
                "multi_roi_boxes_csv": str(multi_rois),
            },
            "notes": [
                "Legacy geocoder_json may still contain 'gemini_like' in its filename for schema compatibility only.",
                "The pipeline does not call Gemini, DeepSeek, OpenAI, or external model APIs.",
                "When address_point_json is supplied, that upstream file may have been produced by an API-enabled stage.",
            ],
        }
        if args.dry_run:
            print(f"write manifest {clean_manifest}", flush=True)
        else:
            clean_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"no-AI input candidates: {clean_candidates if not args.no_clean_aliases else geocoder_json}", flush=True)
    print(f"no-AI case summary: {clean_summary if not args.no_clean_aliases else final_case_summary}", flush=True)
    print(f"no-AI candidate boxes: {clean_boxes if not args.no_clean_aliases else final_rois}", flush=True)
    if not args.no_clean_aliases:
        print(f"no-AI manifest: {clean_manifest}", flush=True)


if __name__ == "__main__":
    main()
