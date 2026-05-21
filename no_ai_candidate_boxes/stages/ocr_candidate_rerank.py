#!/usr/bin/env python3
"""Experiment: use existing OCR evidence to refine Mansfield candidate points.

This is intentionally an experiment script, not a production replacement.
It tests two cheap OCR signals against the current Gemini/geocoder output:

1. Planning-form "Geo Code" fields.  Mansfield forms often encode a local
   grid reference such as 52576175 -> EPSG:27700 (452576, 361750).
2. OCR text/context reranking for existing OS/Google/fallback candidates.

The script writes row-level comparison outputs with baseline vs OCR-refined
distance to the manual polygon layer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from stage_config import parse_bbox, parse_names


DEFAULT_INPUT_JSON = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_gemini.json"
)
DEFAULT_INPUT_GPKG = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42.gpkg"
)
DEFAULT_OCR_JSONL = Path("/data/mansfield/ocr/all_v5ocr.jsonl")
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/mansfield/spatial/polygon-layer/tmp_output/"
    "mansfield-manual-polygon-link_random200_seed42_ocr_rerank_v0"
)

MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)

ROAD_WORDS = (
    "ROAD",
    "STREET",
    "LANE",
    "AVENUE",
    "CLOSE",
    "WAY",
    "DRIVE",
    "GROVE",
    "CRESCENT",
    "HILL",
    "GATE",
    "SIDE",
    "WALK",
    "PLACE",
    "SQUARE",
    "COURT",
    "PARK",
)

LOCALITIES = (
    "MANSFIELD WOODHOUSE",
    "FOREST TOWN",
    "MARKET WARSOP",
    "WARSOP",
    "MEDEN VALE",
    "SPION KOP",
    "CLIPSTONE",
    "MANSFIELD",
)
ALLOWED_LOCALITY_NAMES = LOCALITIES + ("NOTTINGHAMSHIRE",)

BAD_LOCALITIES = (
    "EAST LEAKE",
    "DERBY",
    "OXFORD",
    "DUNDEE",
    "HACKNEY",
    "LONDON",
    "FAREHAM",
    "REDDITCH",
    "SALTASH",
    "BIRMINGHAM",
    "MARCH",
    "BANGOR",
    "FELIXSTOWE",
    "NOTTINGHAM CITY",
)

RELATION_PATTERNS = {
    "adjacent": re.compile(r"\b(ADJ\.?|ADJACENT|ADJOINING)\b"),
    "rear": re.compile(r"\b(REAR|BEHIND)\b"),
    "land_off": re.compile(r"\bLAND\s+OFF\b|\bOFF\s+[A-Z0-9 .'-]+(" + "|".join(ROAD_WORDS) + r")\b"),
    "junction": re.compile(r"\bJUNCTION\b"),
    "between": re.compile(r"\bBETWEEN\b"),
    "plot": re.compile(r"\bPLOT(?:S)?\b"),
    "unit": re.compile(r"\bUNIT(?:S)?\b"),
    "former": re.compile(r"\bFORMER\b"),
}


def norm(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = text.replace("0", "O")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", norm(text))


def base_key(key: Any) -> str:
    return str(key).split("_", 1)[0]


def in_mansfield_bbox(easting: float, northing: float) -> bool:
    minx, miny, maxx, maxy = MANSFIELD_BBOX
    return minx <= easting <= maxx and miny <= northing <= maxy


def distance_to_geom(easting: Any, northing: Any, geom: Any) -> float | None:
    try:
        e = float(easting)
        n = float(northing)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(e) or not math.isfinite(n):
        return None
    return Point(e, n).distance(geom)


def safe_json(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", "[]"):
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [v for v in parsed if isinstance(v, dict)]
    return []


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def ocr_digit_fix(text: str) -> str:
    table = str.maketrans(
        {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "|": "1",
            "S": "5",
            "Z": "2",
            "B": "8",
            "G": "6",
            "Y": "7",
        }
    )
    return text.upper().translate(table)


def decode_slash_grid(raw: str) -> tuple[float, float] | None:
    cleaned = ocr_digit_fix(raw)
    m = re.search(r"([0-9]{4,6})\s*/\s*([0-9]{4,6})", cleaned)
    if not m:
        return None
    left = re.sub(r"\D", "", m.group(1))
    right = re.sub(r"\D", "", m.group(2))
    if len(left) < 4 or len(right) < 4:
        return None
    e5 = int(left[-5:]) if len(left) >= 5 else int(left) * 10
    n5 = int(right[-5:]) if len(right) >= 5 else int(right) * 10
    return float(400000 + e5), float(300000 + n5)


def decode_compact_grid(raw: str) -> list[tuple[float, float]]:
    digits = re.sub(r"\D", "", ocr_digit_fix(raw))
    if len(digits) == 8:
        # Mansfield planning form shorthand: EEEEE NNN at 10m precision.
        # The missing northing 10km band must be resolved against candidate
        # evidence: 52576175 -> 452576, 361750 while 55315893 ->
        # 455315, 358930.
        e = 400000 + int(digits[:5])
        n_suffix = int(digits[5:]) * 10
        return [
            (float(e), float(base + n_suffix))
            for base in (340000, 350000, 360000)
            if in_mansfield_bbox(float(e), float(base + n_suffix))
        ]
    if len(digits) == 10:
        # Full local 5+5 reference.
        e = 400000 + int(digits[:5])
        n = 300000 + int(digits[5:])
        return [(float(e), float(n))] if in_mansfield_bbox(float(e), float(n)) else []
    return []


def decode_pair_grid(left: str, right: str) -> tuple[float, float] | None:
    left_digits = re.sub(r"\D", "", ocr_digit_fix(left))
    right_digits = re.sub(r"\D", "", ocr_digit_fix(right))
    if len(left_digits) < 5:
        return None
    e = 400000 + int(left_digits[-5:])
    if len(right_digits) >= 5:
        n5 = int(right_digits[-5:])
        n = 300000 + n5
    elif len(right_digits) == 3:
        n = 360000 + int(right_digits) * 10
    else:
        return None
    return float(e), float(n)


@dataclass
class OcrGridCandidate:
    easting: float
    northing: float
    raw: str
    image: str
    confidence: float
    method: str

    @property
    def key(self) -> tuple[int, int]:
        return round(self.easting / 10), round(self.northing / 10)


@dataclass
class OcrEvidence:
    prefix: str
    pages: int = 0
    high_text_count: int = 0
    relation_counts: dict[str, int] | None = None
    road_phrases: list[str] | None = None
    location_lines: list[str] | None = None
    plan_pages: list[str] | None = None
    grid_candidates: list[OcrGridCandidate] | None = None

    def __post_init__(self) -> None:
        self.relation_counts = self.relation_counts or {}
        self.road_phrases = self.road_phrases or []
        self.location_lines = self.location_lines or []
        self.plan_pages = self.plan_pages or []
        self.grid_candidates = self.grid_candidates or []


def sorted_ocr_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def pos(item: dict[str, Any]) -> tuple[float, float]:
        bbox = item.get("bbox") or [0, 0, 0, 0]
        return float(bbox[1]), float(bbox[0])

    return sorted(results, key=pos)


def extract_grids_from_page(image: str, results: list[dict[str, Any]]) -> list[OcrGridCandidate]:
    out: list[OcrGridCandidate] = []
    ordered = sorted_ocr_results(results)
    texts = [str(r.get("text") or "") for r in ordered]
    confs = [float(r.get("confidence") or 0) for r in ordered]
    page_join = " ".join(texts)

    # Direct slash forms can appear as "Geo Code: 57040/62545".
    for m in re.finditer(r"[A-Z0-9]{4,6}\s*/\s*[A-Z0-9]{4,6}", page_join, re.I):
        decoded = decode_slash_grid(m.group(0))
        if decoded and in_mansfield_bbox(*decoded):
            out.append(OcrGridCandidate(decoded[0], decoded[1], m.group(0), image, 0.85, "slash"))

    for idx, text in enumerate(texts):
        is_label = bool(re.search(r"GEO\s*CODE|GEOCODE|GEOCOLE|GEOC0DE", text, re.I))
        text_has_grid = bool(re.search(r"\b\d{8,10}\b", text))
        if not is_label and not text_has_grid:
            continue

        window = texts[idx : idx + 8] if is_label else [text]
        window_text = " ".join(window)
        window_conf = max(confs[idx : idx + 8] or [confs[idx]])

        for raw in re.findall(r"\b[A-Z0-9]{8,10}\b", window_text, re.I):
            decoded_points = decode_compact_grid(raw)
            for decoded in decoded_points:
                out.append(
                    OcrGridCandidate(decoded[0], decoded[1], raw, image, window_conf, "compact")
                )

        # Some forms split the coordinate into two nearby OCR tokens.
        nums = re.findall(r"\b[A-Z0-9]{3,6}\b", window_text, re.I)
        for left, right in zip(nums, nums[1:]):
            decoded = decode_pair_grid(left, right)
            if decoded and in_mansfield_bbox(*decoded):
                out.append(
                    OcrGridCandidate(
                        decoded[0],
                        decoded[1],
                        f"{left} {right}",
                        image,
                        window_conf,
                        "pair",
                    )
                )

    return out


def extract_road_phrases(text: str) -> list[str]:
    words = "|".join(ROAD_WORDS)
    pattern = re.compile(r"\b([A-Z0-9][A-Z0-9 .'&-]{1,45}?\s+(?:" + words + r"))\b")
    phrases = []
    for match in pattern.finditer(norm(text)):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if len(phrase) >= 6 and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def build_prefix_map(gpkg_path: Path, layer: str, json_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    wanted = {base_key(row.get("key")) for row in json_rows}
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    sql = (
        f'SELECT unique_key, chargegeog, supplement, FilePath, "supplementary-information" '
        f'FROM "{layer}"'
    )
    out: dict[str, dict[str, Any]] = {}
    for unique_key, chargegeog, supplement, file_path, supplementary in cur.execute(sql):
        key = str(unique_key)
        if key not in wanted:
            continue
        prefix = os.path.basename(file_path or "")
        if not prefix:
            continue
        out[prefix] = {
            "base_key": key,
            "chargegeog": chargegeog or "",
            "supplement": supplement or supplementary or "",
            "file_path": file_path or "",
        }
    con.close()
    return out


def load_ocr_evidence(ocr_path: Path, prefix_map: dict[str, dict[str, Any]]) -> dict[str, OcrEvidence]:
    evidence = {prefix: OcrEvidence(prefix=prefix) for prefix in prefix_map}
    prefixes = set(prefix_map)

    with ocr_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            image_path = rec.get("image_path") or ""
            if "/" not in image_path:
                continue
            prefix = image_path.split("/", 1)[0]
            if prefix not in prefixes:
                continue

            ev = evidence[prefix]
            ev.pages += 1
            image = rec.get("image") or image_path
            results = rec.get("results") or []
            texts = []
            for item in results:
                text = str(item.get("text") or "")
                conf = float(item.get("confidence") or 0)
                if conf >= 0.85:
                    ev.high_text_count += 1
                if conf >= 0.75 and text:
                    texts.append(text)

            page_text = " | ".join(texts)
            page_norm = norm(page_text)
            if re.search(r"\b(SITE\s+PLAN|LOCATION\s*PLAN|PLANS?|DRAWING|DRG|SCALE)\b", page_norm):
                if len(ev.plan_pages) < 8:
                    ev.plan_pages.append(image)

            if re.search(r"\b(LOCATION|FULL ADDRESS|PROPOSAL|SITE AREA|GEO\s*CODE|GEOCODE)\b", page_norm):
                for text in texts:
                    ntext = norm(text)
                    if re.search(r"\b(LOCATION|FULL ADDRESS|PROPOSAL|SITE AREA|GEO\s*CODE|GEOCODE)\b", ntext):
                        if ntext not in ev.location_lines and len(ev.location_lines) < 40:
                            ev.location_lines.append(ntext)

            for rel, pattern in RELATION_PATTERNS.items():
                hits = len(pattern.findall(page_norm))
                if hits:
                    ev.relation_counts[rel] = ev.relation_counts.get(rel, 0) + hits

            for phrase in extract_road_phrases(page_text):
                if phrase not in ev.road_phrases and len(ev.road_phrases) < 80:
                    ev.road_phrases.append(phrase)

            ev.grid_candidates.extend(extract_grids_from_page(image, results))

    return evidence


def nearest_pool_distance(candidate: OcrGridCandidate, pool: list["Candidate"]) -> float | None:
    distances = [
        math.hypot(candidate.easting - item.easting, candidate.northing - item.northing)
        for item in pool
    ]
    return min(distances) if distances else None


def choose_ocr_grid(ev: OcrEvidence, pool: list["Candidate"]) -> tuple[OcrGridCandidate | None, float | None]:
    if not ev.grid_candidates:
        return None, None

    buckets: dict[tuple[int, int], list[OcrGridCandidate]] = defaultdict(list)
    for cand in ev.grid_candidates:
        buckets[cand.key].append(cand)

    eligible: list[tuple[list[OcrGridCandidate], float | None]] = []
    for items in buckets.values():
        best_dist = min(
            (
                d
                for d in (nearest_pool_distance(item, pool) for item in items)
                if d is not None
            ),
            default=None,
        )
        repeated = len(items) >= 2
        strong_method = any(item.method in {"slash", "pair"} for item in items)
        compact10 = any(len(re.sub(r"\D", "", item.raw)) == 10 for item in items)

        # This is the critical anti-regression gate: accept OCR grid points
        # when they are supported by the existing candidate cloud, or when a
        # stronger slash/pair grid appears and no usable geocoder candidate is
        # available. Compact 10-digit strings are often unrelated IDs, so they
        # need especially close support unless repeated.
        if best_dist is None:
            if strong_method:
                eligible.append((items, best_dist))
        elif best_dist <= 750:
            eligible.append((items, best_dist))
        elif strong_method and best_dist <= 1500:
            eligible.append((items, best_dist))
        elif repeated and not compact10 and best_dist <= 1500:
            eligible.append((items, best_dist))

    if not eligible:
        return None, None

    def bucket_score(pair: tuple[list[OcrGridCandidate], float | None]) -> tuple[float, float, float]:
        items, nearest = pair
        score = len(items) * 2.0
        score += max(item.confidence for item in items)
        if any(item.method == "compact" for item in items):
            score += 0.5
        if any(item.method == "slash" for item in items):
            score += 0.5
        if any(item.method == "pair" for item in items):
            score += 0.5
        if nearest is not None:
            score -= min(nearest / 500.0, 5.0)
        return score, max(item.confidence for item in items), -(nearest or 999999)

    best_items, best_nearest = max(eligible, key=bucket_score)
    best = max(best_items, key=lambda item: item.confidence)
    return best, best_nearest


@dataclass
class Candidate:
    source: str
    address: str
    easting: float
    northing: float
    score: float = 0.0
    reason: str = ""


def add_candidate(
    out: list[Candidate],
    source: str,
    address: Any,
    easting: Any,
    northing: Any,
    base_score: float = 0.0,
) -> None:
    e = parse_float(easting)
    n = parse_float(northing)
    if e is None or n is None or not in_mansfield_bbox(e, n):
        return
    addr = str(address or source)
    key = (source, round(e, 2), round(n, 2), norm(addr))
    for existing in out:
        if (existing.source, round(existing.easting, 2), round(existing.northing, 2), norm(existing.address)) == key:
            return
    out.append(Candidate(source, addr, e, n, base_score))


def extract_candidate_pool(row: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    add_candidate(
        candidates,
        "current_best",
        row.get("best_address_final"),
        row.get("best_easting_27700_final"),
        row.get("best_northing_27700_final"),
        10.0,
    )
    add_candidate(
        candidates,
        "lexicon",
        row.get("lexicon_road"),
        row.get("lexicon_easting_27700"),
        row.get("lexicon_northing_27700"),
        3.0,
    )
    add_candidate(
        candidates,
        "google_best",
        row.get("gog_best_address"),
        row.get("gog_best_easting_27700"),
        row.get("gog_best_northing_27700"),
        5.0,
    )
    add_candidate(
        candidates,
        "web_best",
        row.get("web_research_best_address"),
        row.get("web_research_best_easting_27700"),
        row.get("web_research_best_northing_27700"),
        6.0,
    )
    add_candidate(
        candidates,
        "os_gemini",
        row.get("os_address_gemini"),
        row.get("os_address_gemini_easting_27700"),
        row.get("os_address_gemini_northing_27700"),
        5.0,
    )

    for field, source, base in [
        ("os_candidates_json", "os", 4.0),
        ("gog_candidates_json", "google", 3.5),
        ("os_fallback_pool_json", "fallback", 2.5),
        ("web_research_results_json", "web_result", 2.0),
    ]:
        for item in safe_json(row.get(field)):
            address = (
                item.get("address")
                or item.get("ADDRESS")
                or item.get("formatted_address")
                or item.get("title")
                or item.get("name")
                or item.get("_google_name")
            )
            easting = (
                item.get("easting_27700")
                or item.get("X_COORDINATE")
                or item.get("x")
                or item.get("easting")
            )
            northing = (
                item.get("northing_27700")
                or item.get("Y_COORDINATE")
                or item.get("y")
                or item.get("northing")
            )
            add_candidate(candidates, source, address, easting, northing, base)
    return candidates


def expected_context(row: dict[str, Any], meta: dict[str, Any], ev: OcrEvidence | None) -> dict[str, Any]:
    texts = [
        row.get("original_address") or "",
        meta.get("chargegeog") or "",
        meta.get("supplement") or "",
    ]
    if ev:
        texts.extend(ev.location_lines)
        # Road phrases are useful but noisy; keep them separate too.
        texts.extend(ev.road_phrases[:20])
    combined = norm(" | ".join(texts))
    localities = [loc for loc in LOCALITIES if loc in combined]
    roads = []
    for text in texts:
        for phrase in extract_road_phrases(text):
            if phrase not in roads:
                roads.append(phrase)
    relations = []
    for rel, pattern in RELATION_PATTERNS.items():
        if pattern.search(combined):
            relations.append(rel)
    return {
        "combined": combined,
        "localities": localities,
        "roads": roads,
        "relations": relations,
    }


def token_similarity(a: str, b: str) -> float:
    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def score_candidate(candidate: Candidate, ctx: dict[str, Any], row: dict[str, Any]) -> Candidate:
    addr_norm = norm(candidate.address)
    score = candidate.score
    reasons = []

    expected_localities = ctx["localities"] or list(LOCALITIES[:1])
    if any(loc in addr_norm for loc in expected_localities):
        score += 25
        reasons.append("locality_match")
    elif any(bad in addr_norm for bad in BAD_LOCALITIES):
        score -= 80
        reasons.append("bad_locality")
    elif not any(place in addr_norm for place in ALLOWED_LOCALITY_NAMES):
        score -= 15
        reasons.append("weak_locality")

    road_scores = []
    for road in ctx["roads"]:
        sim = token_similarity(road, addr_norm)
        if sim >= 0.92 or compact_text(road) in compact_text(addr_norm):
            road_scores.append(1.0)
        elif sim >= 0.72:
            road_scores.append(sim)
    if road_scores:
        score += 22 * max(road_scores)
        reasons.append("road_match")
    elif ctx["roads"] and candidate.source in {"lexicon", "current_best", "google_best", "os"}:
        score -= 10
        reasons.append("missing_ocr_road")

    original = norm(row.get("original_address") or "")
    address_tokens = {t for t in re.findall(r"[A-Z0-9]{3,}", original) if t not in {"THE", "AND", "NOTTS"}}
    cand_tokens = set(re.findall(r"[A-Z0-9]{3,}", addr_norm))
    if address_tokens:
        overlap = len(address_tokens & cand_tokens) / max(len(address_tokens), 1)
        score += 20 * min(overlap, 1.0)
        if overlap >= 0.35:
            reasons.append("original_token_overlap")

    relations = set(ctx["relations"])
    if relations & {"adjacent", "rear", "land_off", "between", "plot"}:
        # For land relation cases, an exact property/address match is an anchor,
        # not automatically the target.  Keep it viable but lower certainty.
        if candidate.source == "current_best" and re.match(r"^\d+[A-Z]?,", addr_norm):
            score -= 8
            reasons.append("relation_property_anchor_penalty")
        if any(word in addr_norm for word in ("STREET RECORD", "ACCESS ROAD", "LAND", "SITE")):
            score += 8
            reasons.append("relation_landlike_candidate")

    candidate.score = score
    candidate.reason = ",".join(reasons)
    return candidate


def should_accept_ocr_grid(
    row: dict[str, Any],
    baseline_e: float | None,
    baseline_n: float | None,
    baseline_address: Any,
    grid_nearest_pool_distance: float | None,
    ctx: dict[str, Any],
) -> tuple[bool, str]:
    source = str(row.get("best_source_final") or "").lower()
    category = str(row.get("best_selection_category") or "").lower()
    confidence = parse_float(row.get("best_confidence")) or 0.0
    base_addr = norm(baseline_address)
    relations = set(ctx.get("relations") or [])
    has_land_relation = bool(relations & {"adjacent", "rear", "land_off", "between", "plot"})

    baseline_valid = (
        baseline_e is not None
        and baseline_n is not None
        and in_mansfield_bbox(baseline_e, baseline_n)
    )
    bad_locality = any(place in base_addr for place in BAD_LOCALITIES)
    nearest = grid_nearest_pool_distance

    if not baseline_valid:
        return True, "baseline_outside_mansfield_bbox"
    if bad_locality:
        return True, "baseline_bad_locality"
    if nearest is None:
        return source in {"fallback", "lexicon", "none"}, "no_candidate_pool_low_trust_baseline"

    if source in {"fallback", "lexicon", "none", "parent_consensus"} and nearest <= 1500:
        return True, "low_trust_baseline_grid_near_candidate"
    if confidence < 75 and nearest <= 1000:
        return True, "low_confidence_grid_near_candidate"
    if has_land_relation and source not in {"os", "gog", "google", "consensus"} and nearest <= 1500:
        return True, "relation_case_non_exact_baseline"

    # High-confidence OS/Google exact matches are usually already address-level.
    # Do not replace them with OCR grid unless the existing candidate cloud is
    # essentially at the same spot and the case text says this is a land
    # relation rather than the named property itself.
    if (
        has_land_relation
        and source in {"os", "gog", "google", "consensus"}
        and nearest <= 75
        and category != "exact"
    ):
        return True, "relation_case_supported_by_exact_candidate_cloud"

    return False, "kept_baseline_gate"


def load_truth_geometries(gpkg_path: Path, layer: str) -> dict[str, Any]:
    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.crs is not None and str(gdf.crs).lower() not in {"epsg:27700", "osgb36 / british national grid"}:
        gdf = gdf.to_crs(27700)
    return {str(row["unique_key"]): row.geometry for _, row in gdf.iterrows()}


def run(args: argparse.Namespace) -> None:
    input_json = Path(args.input_json)
    input_gpkg = Path(args.input_gpkg)
    ocr_jsonl = Path(args.ocr_jsonl)
    output_prefix = Path(args.output_prefix)
    layer = args.layer

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["rows"]
    prefix_map = build_prefix_map(input_gpkg, layer, rows)
    prefix_to_key = {prefix: meta["base_key"] for prefix, meta in prefix_map.items()}
    key_to_prefix = {meta["base_key"]: prefix for prefix, meta in prefix_map.items()}
    truth = load_truth_geometries(input_gpkg, layer)
    ocr = load_ocr_evidence(ocr_jsonl, prefix_map)

    out_rows: list[dict[str, Any]] = []
    improvements = Counter()

    for row in rows:
        key = str(row.get("key"))
        bkey = base_key(key)
        prefix = key_to_prefix.get(bkey)
        meta = prefix_map.get(prefix or "", {})
        ev = ocr.get(prefix or "")
        geom = truth.get(bkey)
        ctx = expected_context(row, meta, ev)

        baseline_e = parse_float(row.get("best_easting_27700_final"))
        baseline_n = parse_float(row.get("best_northing_27700_final"))
        baseline_distance = distance_to_geom(baseline_e, baseline_n, geom) if geom is not None else None

        pool = extract_candidate_pool(row)
        grid, grid_nearest_pool_distance = choose_ocr_grid(ev, pool) if ev else (None, None)
        grid_distance = distance_to_geom(grid.easting, grid.northing, geom) if grid and geom is not None else None

        scored_pool = [score_candidate(c, ctx, row) for c in pool]
        text_best = max(scored_pool, key=lambda c: c.score) if scored_pool else None
        text_distance = (
            distance_to_geom(text_best.easting, text_best.northing, geom)
            if text_best and geom is not None
            else None
        )

        selected_source = "baseline"
        selected_e = baseline_e
        selected_n = baseline_n
        selected_address = row.get("best_address_final")
        selected_reason = "kept_current_best"

        accept_grid, accept_grid_reason = (
            should_accept_ocr_grid(
                row,
                baseline_e,
                baseline_n,
                row.get("best_address_final"),
                grid_nearest_pool_distance,
                ctx,
            )
            if grid is not None
            else (False, "no_ocr_grid")
        )

        if grid is not None and accept_grid:
            selected_source = "ocr_geocode"
            selected_e = grid.easting
            selected_n = grid.northing
            selected_address = f"OCR Geo Code {grid.raw}"
            selected_reason = f"{accept_grid_reason};{grid.method}:{grid.image}"
        elif (
            text_best is not None
            and text_best.score >= 35
            and baseline_e is not None
            and baseline_n is not None
            and not in_mansfield_bbox(baseline_e, baseline_n)
        ):
            selected_source = f"ocr_text_rerank:{text_best.source}"
            selected_e = text_best.easting
            selected_n = text_best.northing
            selected_address = text_best.address
            selected_reason = text_best.reason

        selected_distance = (
            distance_to_geom(selected_e, selected_n, geom) if geom is not None else None
        )

        delta = None
        if baseline_distance is not None and selected_distance is not None:
            delta = baseline_distance - selected_distance
            if delta > 1:
                improvements["improved"] += 1
            elif delta < -1:
                improvements["worse"] += 1
            else:
                improvements["same"] += 1

        out = {
            "key": key,
            "base_key": bkey,
            "original_address": row.get("original_address"),
            "baseline_source": row.get("best_source_final"),
            "baseline_address": row.get("best_address_final"),
            "baseline_easting": baseline_e,
            "baseline_northing": baseline_n,
            "baseline_distance_m": baseline_distance,
            "selected_source": selected_source,
            "selected_address": selected_address,
            "selected_easting": selected_e,
            "selected_northing": selected_n,
            "selected_distance_m": selected_distance,
            "delta_improvement_m": delta,
            "selected_reason": selected_reason,
            "ocr_grid_gate": accept_grid_reason,
            "ocr_prefix": prefix,
            "ocr_pages": ev.pages if ev else 0,
            "ocr_high_text_count": ev.high_text_count if ev else 0,
            "ocr_grid_raw": grid.raw if grid else None,
            "ocr_grid_method": grid.method if grid else None,
            "ocr_grid_image": grid.image if grid else None,
            "ocr_grid_distance_m": grid_distance,
            "ocr_grid_nearest_candidate_m": grid_nearest_pool_distance,
            "ocr_relations": ",".join(ctx["relations"]),
            "ocr_roads": " | ".join(ctx["roads"][:8]),
            "ocr_location_lines": " | ".join((ev.location_lines if ev else [])[:10]),
            "text_best_source": text_best.source if text_best else None,
            "text_best_address": text_best.address if text_best else None,
            "text_best_score": text_best.score if text_best else None,
            "text_best_reason": text_best.reason if text_best else None,
            "text_best_distance_m": text_distance,
            "candidate_count": len(pool),
        }
        out_rows.append(out)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_out = output_prefix.with_suffix(".json")
    csv_out = output_prefix.with_suffix(".csv")
    xlsx_out = output_prefix.with_suffix(".xlsx")

    summary = summarize(out_rows)
    json_out.write_text(
        json.dumps({"meta": {"input_json": str(input_json), "input_gpkg": str(input_gpkg)}, "summary": summary, "rows": out_rows}, indent=2),
        encoding="utf-8",
    )
    df = pd.DataFrame(out_rows)
    df.to_csv(csv_out, index=False)
    df.to_excel(xlsx_out, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {xlsx_out}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        r
        for r in rows
        if r.get("baseline_distance_m") is not None and r.get("selected_distance_m") is not None
    ]
    improved = [r for r in comparable if (r["delta_improvement_m"] or 0) > 1]
    worse = [r for r in comparable if (r["delta_improvement_m"] or 0) < -1]
    same = [r for r in comparable if abs(r["delta_improvement_m"] or 0) <= 1]
    selected_ocr_grid = [r for r in rows if r.get("selected_source") == "ocr_geocode"]
    selected_text = [r for r in rows if str(r.get("selected_source") or "").startswith("ocr_text_rerank")]

    def mean(values: list[float]) -> float | None:
        vals = [v for v in values if v is not None and math.isfinite(v)]
        return sum(vals) / len(vals) if vals else None

    def median(values: list[float]) -> float | None:
        vals = sorted(v for v in values if v is not None and math.isfinite(v))
        if not vals:
            return None
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2

    by_source = Counter(r.get("selected_source") for r in rows)
    return {
        "rows": len(rows),
        "comparable_rows": len(comparable),
        "improved": len(improved),
        "worse": len(worse),
        "same": len(same),
        "selected_ocr_geocode": len(selected_ocr_grid),
        "selected_ocr_text_rerank": len(selected_text),
        "baseline_mean_distance_m": mean([r["baseline_distance_m"] for r in comparable]),
        "selected_mean_distance_m": mean([r["selected_distance_m"] for r in comparable]),
        "baseline_median_distance_m": median([r["baseline_distance_m"] for r in comparable]),
        "selected_median_distance_m": median([r["selected_distance_m"] for r in comparable]),
        "mean_delta_improvement_m": mean([r["delta_improvement_m"] for r in comparable]),
        "selected_source_counts": dict(by_source),
        "top_improvements": [
            {
                "key": r["key"],
                "delta_m": r["delta_improvement_m"],
                "baseline_m": r["baseline_distance_m"],
                "selected_m": r["selected_distance_m"],
                "source": r["selected_source"],
                "address": r["original_address"],
            }
            for r in sorted(improved, key=lambda x: x["delta_improvement_m"], reverse=True)[:15]
        ],
        "top_regressions": [
            {
                "key": r["key"],
                "delta_m": r["delta_improvement_m"],
                "baseline_m": r["baseline_distance_m"],
                "selected_m": r["selected_distance_m"],
                "source": r["selected_source"],
                "address": r["original_address"],
            }
            for r in sorted(worse, key=lambda x: x["delta_improvement_m"])[:15]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--input-gpkg", default=str(DEFAULT_INPUT_GPKG))
    parser.add_argument("--ocr-jsonl", default=str(DEFAULT_OCR_JSONL))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--layer", default="mansfield-manual-polygon-link-random200")
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--locality-names", default="")
    parser.add_argument("--allowed-locality-names", default="")
    parser.add_argument("--bad-locality-names", default="")
    args = parser.parse_args()
    global MANSFIELD_BBOX, LOCALITIES, ALLOWED_LOCALITY_NAMES, BAD_LOCALITIES
    MANSFIELD_BBOX = parse_bbox(args.local_bbox, MANSFIELD_BBOX)
    LOCALITIES = tuple(parse_names(args.locality_names, LOCALITIES))
    ALLOWED_LOCALITY_NAMES = tuple(parse_names(args.allowed_locality_names, ALLOWED_LOCALITY_NAMES))
    BAD_LOCALITIES = tuple(parse_names(args.bad_locality_names, BAD_LOCALITIES))
    return args


if __name__ == "__main__":
    run(parse_args())
