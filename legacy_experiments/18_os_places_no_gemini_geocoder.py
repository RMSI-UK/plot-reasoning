#!/usr/bin/env python3
"""Generate Mansfield geocoder JSON with OS Places and no Gemini.

This keeps the rich OS candidate pool from the older address-to-point pipeline
but replaces the model selection step with deterministic ranking and final
selection rules.  It intentionally does not load or call Gemini.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
ADDR_SCRIPT = ROOT / "1_address_to_point_gemini.py"
spec = importlib.util.spec_from_file_location("addr_pipeline", ADDR_SCRIPT)
addr = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["addr_pipeline"] = addr
spec.loader.exec_module(addr)
V3 = addr.V3


def cache_path(cache_dir: Path, query: str) -> Path:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def query_os_cached(session: Any, api_key: str, query: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.http_cache_dir:
        args.http_cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_path(args.http_cache_dir, query)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    rows = V3._query_find(
        session=session,
        api_key=api_key,
        query=query,
        timeout_seconds=int(args.timeout_seconds),
        max_retries=int(args.max_retries),
    )
    if args.http_cache_dir:
        cache_path(args.http_cache_dir, query).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def build_row(idx: int, row: dict[str, Any], args: argparse.Namespace, os_api_key: str) -> dict[str, Any]:
    session, matcher = addr._get_thread_resources(args)
    raw_address = addr.clean_text(str(row.get(args.address_column, "") or "").replace("\r", ", ").replace("\n", ", "))
    corrected_raw_address, correction_notes = V3._correct_raw_address_spelling(raw_address, matcher)
    best_match, candidates, debug = matcher.best_match(raw_address, max_candidates=5)
    preferred_match, access_match = addr._reference_road_hierarchy_override(raw_address, candidates, debug)
    effective_match = preferred_match or best_match

    road_phrases = [item.matched_phrase for item in candidates if getattr(item, "matched_phrase", "")]
    if not road_phrases:
        road_phrases = list(debug.get("expanded_segments", []) or [])
    if preferred_match is not None:
        preferred_phrase = addr.clean_text(getattr(preferred_match, "matched_phrase", "")) or addr.clean_text(
            getattr(preferred_match, "road_name", "")
        )
        access_phrase = ""
        if access_match is not None:
            access_phrase = addr.clean_text(getattr(access_match, "matched_phrase", "")) or addr.clean_text(
                getattr(access_match, "road_name", "")
            )
        reordered: list[str] = []
        for phrase in [preferred_phrase, access_phrase, *road_phrases]:
            cleaned = addr.clean_text(phrase)
            if cleaned and cleaned not in reordered:
                reordered.append(cleaned)
        road_phrases = reordered

    place_hints = list(debug.get("place_hints", []) or [])
    existing_place_keys = {V3._norm_compact(value) for value in place_hints}
    for hint in V3._extract_freeform_place_hints(raw_address, road_phrases):
        key = V3._norm_compact(hint)
        if key and key not in existing_place_keys:
            existing_place_keys.add(key)
            place_hints.append(hint)

    os_query = V3._compose_os_query(corrected_raw_address or raw_address, args.council, args.county)
    os_query_variants = V3._expand_os_query_variants(os_query)
    merged_candidates: list[dict[str, Any]] = []
    for query_variant in os_query_variants:
        merged_candidates.extend(query_os_cached(session, os_api_key, query_variant, args))
    os_candidates_raw = V3._dedupe_os_candidates(merged_candidates)

    os_ranked: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for candidate in os_candidates_raw:
        rank_tuple, rank_debug = V3._rank_os_candidate(
            candidate,
            corrected_raw_address or raw_address,
            road_phrases,
            place_hints,
            args.council,
        )
        os_ranked.append((rank_tuple, rank_debug, candidate))
    os_ranked.sort(key=lambda item: item[0], reverse=True)
    os_ranked_valid = V3._valid_os_ranked(os_ranked)

    os_raw_best_rank = os_ranked_valid[0][0] if os_ranked_valid else None
    os_raw_best_rank_debug = os_ranked_valid[0][1] if os_ranked_valid else None
    os_raw_best = os_ranked_valid[0][2] if os_ranked_valid else None
    guarded_source, guarded_reason, guarded_lexicon, os_best_rank, os_best_rank_debug, os_best = V3._select_guarded_anchor(
        effective_match,
        candidates,
        os_ranked_valid,
    )
    os_confidence_score, os_confidence_band, os_confidence_reason = V3._compute_os_confidence(
        raw_address,
        effective_match,
        candidates,
        os_ranked_valid,
        guarded_source,
        guarded_reason,
        os_best,
        os_best_rank_debug,
    )

    road_name = addr.clean_text(getattr(effective_match, "road_name", "")) if effective_match else ""
    road_place = addr.clean_text(getattr(effective_match, "populated_place", "")) if effective_match else ""
    road_easting = effective_match.geometry_x if effective_match else ""
    road_northing = effective_match.geometry_y if effective_match else ""
    serialized_candidates = [
        V3._serialize_os_candidate(candidate, rank_tuple, rank_debug, place_hints)
        for rank_tuple, rank_debug, candidate in os_ranked
    ]
    fallback_pool = addr._build_fallback_pool(
        raw_address,
        road_name,
        road_place,
        road_easting,
        road_northing,
        serialized_candidates,
        args.council,
        args.county,
    )

    key = str(row.get("key") or row.get("unique_key") or idx).strip()
    source_key = str(row.get("_source_unique_key") or row.get("unique_key") or key).strip()
    output = {
        "idx": idx,
        "key": key,
        "_source_unique_key": source_key,
        "_is_expanded_case": row.get("_is_expanded_case") or False,
        "_expanded_from": row.get("_expanded_from") or "",
        "_expanded_variant": row.get("_expanded_variant") or "",
        "original_address": raw_address,
        "raw_address": raw_address,
        "chargegeog": raw_address,
        "corrected_raw_address": corrected_raw_address,
        "spelling_correction_notes_json": json.dumps(correction_notes, ensure_ascii=False),
        "lexicon_road": " | ".join(part for part in [road_name, road_place] if part),
        "lexicon_easting_27700": road_easting,
        "lexicon_northing_27700": road_northing,
        "os_query": os_query,
        "os_query_variants_json": json.dumps(os_query_variants, ensure_ascii=False),
        "os_candidate_count": len(os_ranked),
        "os_candidates_json": json.dumps(serialized_candidates, ensure_ascii=False),
        "os_candidates_full_json": json.dumps(os_candidates_raw, ensure_ascii=False),
        "os_guarded_reason": guarded_reason,
        "os_confidence_score": os_confidence_score,
        "os_confidence_band": os_confidence_band,
        "os_confidence_reason": os_confidence_reason,
        "os_address_gemini": V3._os_candidate_address(os_best) if os_best else "",
        "os_address_gemini_easting_27700": V3._to_float(os_best.get("X_COORDINATE")) if os_best else "",
        "os_address_gemini_northing_27700": V3._to_float(os_best.get("Y_COORDINATE")) if os_best else "",
        "gemini_status": "skipped_no_gemini",
        "gemini_reason": "deterministic_os_places_selection",
        "gog_query": "",
        "gog_candidate_count": 0,
        "gog_confidence_score": 0,
        "gog_confidence_band": "not_requested",
        "gog_confidence_reason": "disabled",
        "gog_candidates_json": "[]",
        "gog_best_address": "",
        "gog_best_easting_27700": "",
        "gog_best_northing_27700": "",
        "os_fallback_pool_json": json.dumps(fallback_pool, ensure_ascii=False),
        "os_raw_best_address": V3._os_candidate_address(os_raw_best) if os_raw_best else "",
        "os_raw_best_road_name": V3._os_candidate_road(os_raw_best) if os_raw_best else "",
        "os_raw_best_place": V3._best_os_place_field(os_raw_best, place_hints) if os_raw_best else "",
        "os_raw_best_easting_27700": V3._to_float(os_raw_best.get("X_COORDINATE")) if os_raw_best else "",
        "os_raw_best_northing_27700": V3._to_float(os_raw_best.get("Y_COORDINATE")) if os_raw_best else "",
        "os_raw_best_rank_tuple_json": json.dumps(list(os_raw_best_rank), ensure_ascii=False) if os_raw_best_rank else "",
        "os_best_address": V3._os_candidate_address(os_best) if os_best else "",
        "os_best_road_name": V3._os_candidate_road(os_best) if os_best else "",
        "os_best_place": V3._best_os_place_field(os_best, place_hints) if os_best else "",
        "os_best_easting_27700": V3._to_float(os_best.get("X_COORDINATE")) if os_best else "",
        "os_best_northing_27700": V3._to_float(os_best.get("Y_COORDINATE")) if os_best else "",
        "os_best_rank_tuple_json": json.dumps(list(os_best_rank), ensure_ascii=False) if os_best_rank else "",
        "best_road_anchor_name": road_name,
        "best_road_anchor_place": road_place,
        "best_road_anchor_easting_27700": road_easting,
        "best_road_anchor_northing_27700": road_northing,
        "best_road_anchor_confidence": V3._anchor_confidence(effective_match, candidates),
        "best_road_anchor_type": V3._anchor_type(raw_address, effective_match),
        "best_road_name": road_name,
        "best_road_place": road_place,
        "best_road_easting_27700": road_easting,
        "best_road_northing_27700": road_northing,
        "best_anchor_source": guarded_source or "none",
        "best_anchor_name": V3._os_candidate_road(os_best) if guarded_source == "os" and os_best else road_name,
        "best_anchor_place": V3._best_os_place_field(os_best, place_hints) if guarded_source == "os" and os_best else road_place,
        "best_anchor_easting_27700": V3._to_float(os_best.get("X_COORDINATE"))
        if guarded_source == "os" and os_best
        else road_easting,
        "best_anchor_northing_27700": V3._to_float(os_best.get("Y_COORDINATE"))
        if guarded_source == "os" and os_best
        else road_northing,
        "best_anchor_choice_reason": guarded_reason,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="OS Places no-Gemini geocoder JSON for Mansfield.")
    parser.add_argument("--input-gpkg", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--address-column", default="chargegeog")
    parser.add_argument("--keys-file", default="/env/key")
    parser.add_argument("--api-key")
    parser.add_argument("--os-open-names-db", default=V3.DEFAULT_OS_OPEN_NAMES_DB)
    parser.add_argument("--county", default=V3.DEFAULT_COUNTY)
    parser.add_argument("--council", default=V3.DEFAULT_COUNCIL)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=V3.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=V3.DEFAULT_MAX_RETRIES)
    parser.add_argument("--http-cache-dir", type=Path, default=ROOT / "tmp_results" / "os_places_no_gemini_cache")
    args = parser.parse_args()

    os_api_key = args.api_key or V3._load_os_api_key_from_keys_file(args.keys_file)
    if not os_api_key:
        raise SystemExit("Missing OS API key. Use --api-key or --keys-file.")

    base_rows = addr.load_input_rows(args.input_gpkg, int(args.row_limit or 0), args.address_column)
    rows = addr.expand_main_rows(base_rows, args.address_column)
    indexed_rows = list(enumerate(rows, start=1))

    output_rows: list[dict[str, Any] | None] = [None] * len(indexed_rows)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(build_row, idx, row, args, os_api_key): idx - 1
            for idx, row in indexed_rows
        }
        for done_idx, future in enumerate(as_completed(futures), start=1):
            output_rows[futures[future]] = future.result()
            if done_idx % 25 == 0 or done_idx == len(futures):
                print(f"Progress: {done_idx}/{len(futures)}", flush=True)

    final_rows = addr._apply_final_best_selection([row for row in output_rows if row is not None])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": final_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output_json}", flush=True)
    print("No Gemini calls were made.", flush=True)


if __name__ == "__main__":
    main()
