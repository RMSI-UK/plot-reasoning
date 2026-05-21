#!/usr/bin/env python3
"""Run the address-point evidence -> box -> council parcel pipeline.

This runner is an API firewall: it consumes an existing JSON/JSONL produced by
``spatial_capture_production/1_address_to_point_gemini.py`` and never invokes
that upstream script itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PACKAGE_ROOT = ROOT / "no_ai_candidate_boxes"
STAGES = PACKAGE_ROOT / "stages"
LEGACY = ROOT / "legacy_experiments"

sys.path.insert(0, str(PACKAGE_ROOT))
from config import DEFAULT_CONFIG, bbox_arg, cfg_layer, cfg_param, cfg_path, names_arg, resolved_config


def suffix(prefix: Path, ending: str) -> Path:
    return prefix.with_name(prefix.name + ending)


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def add_if(cmd: list[str], flag: str, value: object | None) -> None:
    if value not in (None, ""):
        cmd.extend([flag, str(value)])


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
        description="Run address-point evidence, candidate boxes, and council parcel ranking.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--address-point-json",
        type=Path,
        help="Existing JSON/JSONL from 1_address_to_point_gemini.py. Optional evidence; geocoding-box never calls APIs.",
    )
    parser.add_argument(
        "--address-point-gpkg",
        type=Path,
        help="Primary 1_address/input GPKG. unique_key and case geometry come from this layer.",
    )
    parser.add_argument("--address-point-layer", help="Layer in --address-point-gpkg.")
    parser.add_argument("--output-prefix", type=Path, default=cfg_path(config, "address_point_pipeline_output_prefix", cfg_path(config, "output_prefix")))
    parser.add_argument("--open-roads", type=Path, default=cfg_path(config, "open_roads"))
    parser.add_argument("--open-names", type=Path, default=cfg_path(config, "open_names"))
    parser.add_argument("--raw-wfs-gpkg", type=Path, default=cfg_path(config, "raw_wfs_gpkg"))
    parser.add_argument("--raw-wfs-layer", default=cfg_layer(config, "raw_wfs_layer"))
    parser.add_argument("--merged-wfs-gpkg", type=Path, default=cfg_path(config, "merged_wfs_gpkg"))
    parser.add_argument("--merged-wfs-layer", default=cfg_layer(config, "merged_wfs_layer"))
    parser.add_argument("--council-gpkg", type=Path, default=cfg_path(config, "council_gpkg"))
    parser.add_argument("--council-layer", default=cfg_layer(config, "council_layer"))
    parser.add_argument("--uprn-gpkg", type=Path, default=cfg_path(config, "uprn_gpkg"))
    parser.add_argument("--uprn-layer", default=cfg_layer(config, "uprn_layer", "osopenuprn_address"))
    parser.add_argument(
        "--case-gpkg",
        type=Path,
        help="Optional case/input GPKG used by the local OCR/OpenNames/OpenRoads evidence chain.",
    )
    parser.add_argument("--case-layer", help="Layer in --case-gpkg.")
    parser.add_argument(
        "--truth-gpkg",
        type=Path,
        help="Optional offline evaluation truth GPKG. Also accepted as a legacy alias for --case-gpkg.",
    )
    parser.add_argument(
        "--truth-layer",
        help="Optional offline evaluation truth layer. Also accepted as a legacy alias for --case-layer.",
    )
    parser.add_argument("--case-key-column", default="unique_key")
    parser.add_argument("--address-column", default=cfg_param(config, "address_column", "chargegeog"))
    parser.add_argument("--local-bbox", default=bbox_arg(config.get("local_bbox")))
    parser.add_argument("--generic-place-names", default=names_arg(config.get("generic_place_names")))
    parser.add_argument("--roi-side", type=float, default=cfg_param(config, "roi_side", 180.0))
    parser.add_argument("--max-rois", type=int, default=cfg_param(config, "max_rois", 8))
    parser.add_argument("--max-roi-rank", type=int, default=cfg_param(config, "polygon_max_roi_rank", 5))
    parser.add_argument("--polygon-output-rank", type=int, default=cfg_param(config, "polygon_output_rank", 200))
    parser.add_argument("--top-k", type=int, default=cfg_param(config, "polygon_top_k", 5))
    parser.add_argument("--theme-filter-regex", default=cfg_param(config, "theme_filter_regex", "Land|Building"))
    parser.add_argument(
        "--box-mode",
        choices=["auto", "direct", "full-a"],
        default="auto",
        help="auto uses full-a when --address-point-gpkg/--address-point-layer are supplied, otherwise direct.",
    )
    parser.add_argument("--disable-range-parity-correction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case_gpkg = args.address_point_gpkg or args.case_gpkg or args.truth_gpkg
    case_layer = args.address_point_layer or args.case_layer or args.truth_layer
    eval_truth_gpkg = args.truth_gpkg
    eval_truth_layer = args.truth_layer

    if not args.output_prefix:
        parser.error("--output-prefix is required unless configured in paths.output_prefix")
    if not args.address_point_gpkg:
        parser.error("--address-point-gpkg is required for the current workflow; --address-point-json may be supplied as evidence")
    if args.address_point_gpkg and not args.address_point_layer:
        parser.error("--address-point-layer is required with --address-point-gpkg")
    if args.address_point_json and not args.dry_run and not args.address_point_json.exists():
        parser.error(f"--address-point-json does not exist: {args.address_point_json}")
    if not args.dry_run and args.address_point_gpkg and not args.address_point_gpkg.exists():
        parser.error(f"--address-point-gpkg does not exist: {args.address_point_gpkg}")
    if not args.dry_run and case_gpkg and not case_gpkg.exists():
        parser.error(f"--case-gpkg does not exist: {case_gpkg}")
    if not args.dry_run and eval_truth_gpkg and not eval_truth_gpkg.exists():
        parser.error(f"--truth-gpkg does not exist: {eval_truth_gpkg}")
    if bool(case_gpkg) != bool(case_layer):
        parser.error("--case-gpkg/--case-layer must be supplied together")
    if bool(eval_truth_gpkg) != bool(eval_truth_layer):
        parser.error("--truth-gpkg/--truth-layer must be supplied together")
    if not args.open_roads:
        parser.error("--open-roads is required unless configured in paths.open_roads")
    if not args.raw_wfs_gpkg or not args.raw_wfs_layer:
        parser.error("--raw-wfs-gpkg and --raw-wfs-layer are required for polygon extraction")
    if not args.merged_wfs_gpkg or not args.merged_wfs_layer:
        parser.error("--merged-wfs-gpkg and --merged-wfs-layer are required for polygon extraction")
    if not args.council_gpkg or not args.council_layer:
        parser.error("--council-gpkg and --council-layer are required for council parcel ranking")
    if not args.uprn_gpkg or not args.uprn_layer:
        parser.error("--uprn-gpkg and --uprn-layer are required")

    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)

    address_json = args.address_point_json
    print(f"using address-point GPKG as case base: {case_gpkg}:{case_layer}", flush=True)
    if address_json:
        print(f"using existing address-point JSON evidence: {address_json}", flush=True)

    box_mode = args.box_mode
    if box_mode == "auto":
        box_mode = "full-a" if case_gpkg and case_layer else "direct"

    if box_mode == "full-a":
        if not case_gpkg or not case_layer:
            parser.error("--box-mode full-a requires --case-gpkg and --case-layer")
        box_prefix = suffix(out, "_fulla_boxes")
        box_cmd = [
            sys.executable,
            str(ROOT / "run_no_ai_candidate_boxes.py"),
            "--config",
            str(args.config),
            "--address-point-gpkg",
            str(case_gpkg),
            "--address-point-layer",
            case_layer,
            "--address-column",
            args.address_column,
            "--truth-gpkg",
            str(case_gpkg),
            "--truth-layer",
            case_layer,
            "--output-prefix",
            str(box_prefix),
        ]
        add_if(box_cmd, "--council-root", args.council_root)
        add_if(box_cmd, "--base-data-root", args.base_data_root)
        add_if(box_cmd, "--output-root", args.output_root)
        add_if(box_cmd, "--address-point-json", address_json)
        if args.disable_range_parity_correction:
            box_cmd.append("--disable-range-parity-correction")
        run(box_cmd, args.dry_run)
        input_candidates = suffix(box_prefix, "_no_ai_input_candidates.json")
        v10_csv = suffix(box_prefix, "_openroads.csv")
        case_summary = suffix(box_prefix, "_no_ai_case_summary.csv")
        boxes_csv = suffix(box_prefix, "_no_ai_candidate_boxes.csv")
        boxes_gpkg = suffix(box_prefix, "_geocoder_candidate_boxes.gpkg")
    else:
        adapter_cmd = [
            sys.executable,
            str(STAGES / "address_point_evidence_adapter.py"),
            "--output-prefix",
            str(out),
            "--open-roads",
            str(args.open_roads),
            "--roi-side",
            str(args.roi_side),
            "--max-rois",
            str(args.max_rois),
        ]
        add_if(adapter_cmd, "--address-point-json", address_json)
        if args.address_point_gpkg:
            adapter_cmd.extend(["--address-point-gpkg", str(args.address_point_gpkg), "--address-point-layer", args.address_point_layer])
            adapter_cmd.extend(["--address-column", args.address_column])
        if args.disable_range_parity_correction:
            adapter_cmd.append("--disable-range-parity-correction")
        add_if(adapter_cmd, "--local-bbox", args.local_bbox)
        add_if(adapter_cmd, "--generic-place-names", args.generic_place_names)
        if eval_truth_gpkg and eval_truth_layer:
            adapter_cmd.extend(["--case-gpkg", str(eval_truth_gpkg), "--case-layer", eval_truth_layer])
            adapter_cmd.extend(["--key-column", args.case_key_column])
        run(adapter_cmd, args.dry_run)
        input_candidates = suffix(out, "_input_candidates.json")
        v10_csv = suffix(out, "_v10.csv")
        case_summary = suffix(out, "_case_summary.csv")
        boxes_csv = suffix(out, "_candidate_boxes.csv")
        boxes_gpkg = suffix(out, "_candidate_boxes.gpkg")
    polygon_prefix = suffix(out, "_polygon_candidates")
    polygon_cmd = [
        sys.executable,
        str(LEGACY / "9_hybrid_polygon_rerank.py"),
        "--input-json",
        str(input_candidates),
        "--v10-csv",
        str(v10_csv),
        "--case-summary-csv",
        str(case_summary),
        "--rois-csv",
        str(boxes_csv),
        "--raw-wfs-gpkg",
        str(args.raw_wfs_gpkg),
        "--raw-wfs-layer",
        args.raw_wfs_layer,
        "--merged-wfs-gpkg",
        str(args.merged_wfs_gpkg),
        "--merged-wfs-layer",
        args.merged_wfs_layer,
        "--council-gpkg",
        str(args.council_gpkg),
        "--council-layer",
        args.council_layer,
        "--uprn-gpkg",
        str(args.uprn_gpkg),
        "--uprn-layer",
        args.uprn_layer,
        "--open-roads",
        str(args.open_roads),
        "--output-prefix",
        str(polygon_prefix),
        "--max-roi-rank",
        str(args.max_roi_rank),
        "--max-output-rank",
        str(args.polygon_output_rank),
        "--theme-filter-regex",
        args.theme_filter_regex,
    ]
    add_if(polygon_cmd, "--local-bbox", args.local_bbox)
    if eval_truth_gpkg and eval_truth_layer:
        polygon_cmd.extend(["--truth-gpkg", str(eval_truth_gpkg), "--truth-layer", eval_truth_layer])
    else:
        polygon_cmd.append("--no-truth")
    run(polygon_cmd, args.dry_run)

    polygon_top_csv = suffix(polygon_prefix, f"_top{args.polygon_output_rank}.csv")
    ranking_prefix = suffix(out, f"_council_atomic_top{args.top_k}")
    ranking_cmd = [
        sys.executable,
        str(PACKAGE_ROOT / "polygon_ranking.py"),
        "--input-csv",
        str(polygon_top_csv),
        "--output-prefix",
        str(ranking_prefix),
        "--strategy",
        "relation_evidence_mix",
        "--candidate-source-filter",
        "council_cadastral",
        "--top-k",
        str(args.top_k),
        "--output-top-n",
        "100",
    ]
    run(ranking_cmd, args.dry_run)

    manifest = {
        "pipeline_name": "address_point_council_pipeline",
        "config": str(args.config),
        "box_mode": box_mode,
        "api_calls_from_geocoding_box": False,
        "address_point_json": str(address_json) if address_json else None,
        "address_point_gpkg": str(args.address_point_gpkg) if args.address_point_gpkg else None,
        "address_point_layer": args.address_point_layer,
        "range_parity_correction": not args.disable_range_parity_correction,
        "case_gpkg": str(case_gpkg) if case_gpkg else None,
        "case_layer": case_layer,
        "truth_gpkg": str(eval_truth_gpkg) if eval_truth_gpkg else None,
        "truth_layer": eval_truth_layer,
        "outputs": {
            "input_candidates_json": str(input_candidates),
            "v10_csv": str(v10_csv),
            "case_summary_csv": str(case_summary),
            "candidate_boxes_csv": str(boxes_csv),
            "candidate_boxes_gpkg": str(boxes_gpkg),
            "polygon_candidates_top_csv": str(polygon_top_csv),
            "selected_parcels_csv": str(suffix(ranking_prefix, f"_selected_top{args.top_k}.csv")),
            "ranked_parcels_csv": str(suffix(ranking_prefix, "_ranked_top100.csv")),
        },
    }
    manifest_path = suffix(out, "_pipeline_manifest.json")
    if args.dry_run:
        print(f"write manifest {manifest_path}", flush=True)
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pipeline manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
