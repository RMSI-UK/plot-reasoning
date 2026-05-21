#!/usr/bin/env python3
"""Build a no-Gemini enriched A-stage for Mansfield polygon experiments.

This script generates the four inputs consumed by ``9_hybrid_polygon_rerank.py``:

- gemini-like JSON with production-visible address/geocode candidate fields;
- v10-like CSV with selected coordinates and OCR roads;
- case summary CSV;
- ROI CSV.

It does not call Gemini or any external API.  Evidence comes from local files:

- the Mansfield truth GPKG row itself (`chargegeog`, `FilePath`);
- the full textual spreadsheet OS geocode point(s);
- OCR text from `/data/mansfield/ocr/all_v5ocr.jsonl`;
- OS OpenRoads;
- OS OpenNames CSV tiles.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_config import parse_bbox, parse_names


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROI_SCRIPT = ROOT / "evidence_roi.py"
spec = importlib.util.spec_from_file_location("roi_v1", ROI_SCRIPT)
roi_v1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["roi_v1"] = roi_v1
spec.loader.exec_module(roi_v1)


DEFAULT_TRUTH_GPKG = PROJECT_ROOT / "tmp_results" / "mansfield_random500_excl1200_seed44_ospoint_v1_truth.gpkg"
DEFAULT_TRUTH_LAYER = "random500_excl1200_seed44"
DEFAULT_TEXT_XLSX = Path("/data/mansfield/textual/mansfield_cr&txt_with_points.xlsx")
DEFAULT_OCR_JSONL = Path("/data/mansfield/ocr/all_v5ocr.jsonl")
DEFAULT_OPEN_ROADS = Path("/data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg")
DEFAULT_OPEN_NAMES_DIR = Path("/data/base-data/opname_csv_gb/Data")
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "tmp_results" / "mansfield_random500_excl1200_seed44_no_gemini_enriched_a_v1"

MANSFIELD_BBOX = (449000.0, 343000.0, 462500.0, 371000.0)
GENERIC_PLACE_NAMES = {"MANSFIELD", "WARSOP"}
STOP_TOKENS = {
    "THE",
    "AND",
    "FOR",
    "MANSFIELD",
    "NOTTS",
    "NOTTINGHAMSHIRE",
    "ROAD",
    "STREET",
    "LANE",
    "AVENUE",
    "CLOSE",
    "WAY",
    "DRIVE",
    "GROVE",
    "CRESCENT",
    "PLACE",
    "COURT",
    "PARK",
}
POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
PARCEL_RELATION_RE = re.compile(r"\b(REAR|ADJACENT|ADJOINING|LAND|PLOT|PLOTS|PHASE|UNIT|UNITS|OFF|SITE)\b", re.I)


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(text or "").upper())).strip()


def compact(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper())


def tokens(text: Any) -> set[str]:
    return {
        token
        for token in norm(POSTCODE_RE.sub(" ", str(text or ""))).split()
        if len(token) >= 3 and token not in STOP_TOKENS and not token.isdigit()
    }


def postcodes(text: Any) -> set[str]:
    return {re.sub(r"\s+", "", item.upper()) for item in POSTCODE_RE.findall(str(text or ""))}


def postcode_outcodes(text: Any) -> set[str]:
    out = set()
    for code in postcodes(text):
        match = re.match(r"^([A-Z]{1,2}\d[A-Z\d]?)", code)
        if match:
            out.add(match.group(1))
    return out


def number_proximity(requested: set[int], candidate: set[int]) -> float:
    if not requested or not candidate:
        return 0.0
    delta = min(abs(left - right) for left in requested for right in candidate)
    return max(0.0, 1.0 - min(delta, 140) / 140.0)


def jaccard(left: Any, right: Any) -> float:
    lt = tokens(left)
    rt = tokens(right)
    return len(lt & rt) / len(lt | rt) if lt and rt else 0.0


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def in_work_bbox(x: float, y: float, pad: float = 5000.0) -> bool:
    minx, miny, maxx, maxy = MANSFIELD_BBOX
    return minx - pad <= x <= maxx + pad and miny - pad <= y <= maxy + pad


def roi_square(point: Point, side: float = 150.0) -> Any:
    half = side / 2.0
    return box(point.x - half, point.y - half, point.x + half, point.y + half)


def folder_from_filepath(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip("/")
    return text.split("/")[-1] if text else ""


def case_address(row: pd.Series | dict[str, Any]) -> str:
    def clean_value(col: str) -> str:
        value = row.get(col) if hasattr(row, "get") else None
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "<na>"} else text

    primary = clean_value("chargegeog")
    supplement = clean_value("supplementary-information")
    if primary:
        has_location_token = bool(re.search(r"\d", primary)) or bool(
            re.search(
                r"\b(ROAD|STREET|LANE|AVENUE|CLOSE|WAY|DRIVE|GROVE|CRESCENT|PLACE|COURT|PARK|GATE|HILL|TERRACE|ROW|YARD|CROFT)\b",
                primary,
                re.I,
            )
        )
        if not has_location_token and supplement:
            return f"{primary}, {supplement}"
        return primary

    parts = []
    for col in ("chargeaddr", "chargead00", "chargead01", "chargead02", "postcode", "supplementary-information"):
        text = clean_value(col)
        if text:
            parts.append(text)
    return ", ".join(dict.fromkeys(parts))


def road_name_from_address(address: str, road_names: set[str]) -> str:
    roads = roi_v1.extract_address_roads(address, road_names)
    return roads[0] if roads else ""


def numbers_from_text(value: Any) -> set[int]:
    text = POSTCODE_RE.sub(" ", norm(value))
    out: set[int] = set()
    for start_s, end_s in re.findall(r"\b(\d{1,4})\s*(?:-|TO|AND|/)\s*(\d{1,4})\b", text):
        start = int(start_s)
        end = int(end_s)
        low, high = sorted((start, end))
        if 0 < low < 1000 and high - low <= 30:
            out.update(range(low, high + 1))
    for number_s in re.findall(r"\b(\d{1,4})[A-Z]?\b", text):
        number = int(number_s)
        if 0 < number < 1000:
            out.add(number)
    return out


def _first_column(frame: pd.DataFrame, names: list[str], default: Any = "") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _geometry_xy(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "geometry" not in frame.columns:
        empty = pd.Series([None] * len(frame), index=frame.index)
        return empty, empty
    xs: list[float | None] = []
    ys: list[float | None] = []
    for geom in frame["geometry"]:
        if geom is None or getattr(geom, "is_empty", True):
            xs.append(None)
            ys.append(None)
            continue
        point = geom if getattr(geom, "geom_type", "") == "Point" else geom.representative_point()
        xs.append(float(point.x))
        ys.append(float(point.y))
    return pd.Series(xs, index=frame.index), pd.Series(ys, index=frame.index)


def _standardize_text_points(raw: pd.DataFrame) -> pd.DataFrame:
    geom_x, geom_y = _geometry_xy(raw)
    text = pd.DataFrame(index=raw.index)
    text["Index"] = _first_column(
        raw,
        ["Index", "sample_original_index", "further-information-reference", "originating-authority-charge-identifier"],
        "",
    )
    if text["Index"].astype(str).str.strip().eq("").all():
        text["Index"] = list(range(1, len(raw) + 1))
    text["charge-geographic-description"] = _first_column(
        raw,
        ["charge-geographic-description", "chargegeog", "charge-address", "ext-charge-geographic-description"],
        "",
    )
    text["FilePath"] = _first_column(
        raw,
        ["FilePath", "filepath", "source", "decision_file_name", "further-information-reference"],
        "",
    )
    text["os_geocode_status"] = _first_column(raw, ["os_geocode_status"], "matched")
    text["os_matched_address"] = _first_column(
        raw,
        ["os_matched_address", "match_address", "matched_address", "charge-address", "charge-geographic-description"],
        "",
    )
    text["os_match_score"] = _first_column(raw, ["os_match_score", "address_confidence"], 0.0)
    text["os_uprn"] = _first_column(raw, ["os_uprn", "uprn"], "")
    text["os_easting_27700"] = _first_column(raw, ["os_easting_27700", "easting_27700", "x"], None)
    text["os_northing_27700"] = _first_column(raw, ["os_northing_27700", "northing_27700", "y"], None)
    text["os_easting_27700"] = text["os_easting_27700"].where(text["os_easting_27700"].notna(), geom_x)
    text["os_northing_27700"] = text["os_northing_27700"].where(text["os_northing_27700"].notna(), geom_y)
    return text


def load_text_points(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".gpkg":
        layers = gpd.list_layers(path)
        layer = str(layers["name"].iloc[0])
        raw = gpd.read_file(path, layer=layer)
        text = _standardize_text_points(raw)
    else:
        try:
            text = pd.read_excel(
                path,
                usecols=[
                    "Index",
                    "charge-geographic-description",
                    "FilePath",
                    "os_geocode_status",
                    "os_matched_address",
                    "os_match_score",
                    "os_uprn",
                    "os_easting_27700",
                    "os_northing_27700",
                ],
            )
        except ValueError:
            text = _standardize_text_points(pd.read_excel(path))
    text["x"] = pd.to_numeric(text["os_easting_27700"], errors="coerce")
    text["y"] = pd.to_numeric(text["os_northing_27700"], errors="coerce")
    text = text[text["x"].notna() & text["y"].notna()].copy()
    text = text[text.apply(lambda row: in_work_bbox(float(row["x"]), float(row["y"])), axis=1)].copy()
    text["folder"] = text["FilePath"].map(folder_from_filepath)
    return text


def prepare_global_text_index(text: pd.DataFrame, road_names: set[str]) -> pd.DataFrame:
    indexed = text.copy()
    indexed["_candidate_address_text"] = (
        indexed.get("os_matched_address", pd.Series([""] * len(indexed))).fillna("").astype(str)
        + " "
        + indexed.get("charge-geographic-description", pd.Series([""] * len(indexed))).fillna("").astype(str)
    )
    indexed["_roads"] = indexed["_candidate_address_text"].map(lambda value: roi_v1.extract_address_roads(value, road_names)[:4])
    indexed["_numbers"] = indexed["_candidate_address_text"].map(numbers_from_text)
    indexed["_tokens"] = indexed["_candidate_address_text"].map(tokens)
    indexed["_postcodes"] = indexed["_candidate_address_text"].map(postcodes)
    indexed["_postcode_outcodes"] = indexed["_candidate_address_text"].map(postcode_outcodes)
    return indexed


def merge_candidate_lists(*candidate_lists: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for candidates in candidate_lists:
        for item in candidates:
            key = (
                round(float(item["easting_27700"]), 1),
                round(float(item["northing_27700"]), 1),
                compact(item.get("address") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_candidates:
                return merged
    return merged


def collect_text_candidates(
    case_row: pd.Series,
    text_groups: dict[str, pd.DataFrame],
    road_names: set[str],
    max_candidates: int,
) -> list[dict[str, Any]]:
    folder = folder_from_filepath(case_row.get("FilePath"))
    address = case_address(case_row)
    group = text_groups.get(folder)
    if group is None or group.empty:
        return []

    requested_numbers = numbers_from_text(address)
    scored: list[tuple[tuple[float, float, int, float], dict[str, Any]]] = []
    for _, candidate in group.iterrows():
        cand_address = str(candidate.get("os_matched_address") or candidate.get("charge-geographic-description") or "")
        sim = jaccard(address, candidate.get("charge-geographic-description") or cand_address)
        cand_numbers = numbers_from_text(cand_address)
        number_overlap = len(requested_numbers & cand_numbers) if requested_numbers else 0
        os_score = parse_float(candidate.get("os_match_score")) or 0.0
        match_bonus = 1 if str(candidate.get("os_geocode_status")).lower() == "matched" else 0
        road = road_name_from_address(cand_address or address, road_names)
        item = {
            "address": cand_address or address,
            "easting_27700": float(candidate["x"]),
            "northing_27700": float(candidate["y"]),
            "road_name": road,
            "place": "Mansfield",
            "uprn": str(candidate.get("os_uprn") or ""),
            "text_index": int(candidate.get("Index")) if pd.notna(candidate.get("Index")) else None,
            "text_similarity": round(sim, 4),
            "number_overlap": number_overlap,
            "os_match_score": os_score,
        }
        scored.append(((sim, float(number_overlap), match_bonus, os_score), item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Keep the best rows plus rows that materially match the case wording.
    kept: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for rank, (score_tuple, item) in enumerate(scored):
        if rank >= max_candidates and score_tuple[0] < 0.35 and score_tuple[1] <= 0:
            continue
        key = (round(float(item["easting_27700"]), 1), round(float(item["northing_27700"]), 1), item["address"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
        if len(kept) >= max_candidates:
            break
    return kept


def collect_global_text_candidates(
    case_row: pd.Series,
    text_index: pd.DataFrame,
    road_names: set[str],
    max_candidates: int,
) -> list[dict[str, Any]]:
    address = case_address(case_row)
    address_roads = roi_v1.extract_address_roads(address, road_names)
    requested_numbers = numbers_from_text(address)
    requested_tokens = tokens(address)
    requested_postcodes = postcodes(address)
    requested_outcodes = postcode_outcodes(address)
    if not address_roads and not requested_tokens:
        return []

    scored: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    weak_scored: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    requested_road_set = set(address_roads)
    for _, candidate in text_index.iterrows():
        candidate_roads = set(candidate.get("_roads") or [])
        road_overlap = len(requested_road_set & candidate_roads)

        candidate_numbers = candidate.get("_numbers") or set()
        number_overlap = len(requested_numbers & candidate_numbers) if requested_numbers else 0
        number_near = number_proximity(requested_numbers, candidate_numbers)
        candidate_tokens = candidate.get("_tokens") or set()
        token_sim = len(requested_tokens & candidate_tokens) / len(requested_tokens | candidate_tokens) if requested_tokens and candidate_tokens else 0.0
        candidate_postcodes = candidate.get("_postcodes") or set()
        candidate_outcodes = candidate.get("_postcode_outcodes") or set()
        postcode_match = bool(requested_postcodes & candidate_postcodes)
        outcode_match = bool(requested_outcodes & candidate_outcodes)
        source_address = str(candidate.get("charge-geographic-description") or "")
        source_sim = jaccard(address, source_address)

        if address_roads and road_overlap == 0:
            source_is_strong = source_sim >= 0.62 or (source_sim >= 0.45 and number_overlap > 0)
            if not source_is_strong:
                continue

        # Numeric address/range cases are only useful when the candidate agrees
        # on a number or has very strong token overlap.  Non-numeric land/site
        # cases can still use road-local candidates, but at lower priority.
        strong_match = True
        if requested_numbers:
            if number_overlap <= 0 and token_sim < 0.45:
                strong_match = False
        elif token_sim < 0.25 and road_overlap < 2:
            strong_match = False

        os_score = parse_float(candidate.get("os_match_score")) or 0.0
        matched_address = str(candidate.get("os_matched_address") or "")
        output_address = matched_address or source_address or address
        global_score = (
            road_overlap * 8.0
            + number_overlap * 12.0
            + number_near * 9.0
            + token_sim * 22.0
            + source_sim * 18.0
            + (8.0 if postcode_match else 3.0 if outcode_match else 0.0)
            + min(8.0, os_score * 8.0)
        )
        if requested_numbers and number_overlap:
            global_score += 10.0
        if source_address and compact(address) and compact(address) in compact(source_address):
            global_score += 8.0
        weak_reason = ""
        if not strong_match:
            # Keep a thin fallback layer for cases where OS has no exact
            # house-number row but the road/postcode context is still useful.
            if requested_numbers:
                allow_weak = number_near >= 0.35 or postcode_match or (outcode_match and token_sim >= 0.2)
            else:
                allow_weak = road_overlap > 0 and (postcode_match or outcode_match or token_sim >= 0.12 or requested_tokens)
            if not allow_weak:
                continue
            weak_reason = "road_postcode_number_fallback"
            global_score = (
                road_overlap * 5.5
                + number_near * 10.0
                + token_sim * 12.0
                + source_sim * 14.0
                + (10.0 if postcode_match else 4.0 if outcode_match else 0.0)
                + min(4.0, os_score * 4.0)
            )
        item = {
            "address": output_address,
            "easting_27700": float(candidate["x"]),
            "northing_27700": float(candidate["y"]),
            "road_name": next(iter(requested_road_set & candidate_roads), ""),
            "place": "Mansfield",
            "uprn": str(candidate.get("os_uprn") or ""),
            "text_index": int(candidate.get("Index")) if pd.notna(candidate.get("Index")) else None,
            "text_similarity": round(token_sim, 4),
            "source_text_similarity": round(source_sim, 4),
            "number_overlap": number_overlap,
            "number_proximity": round(number_near, 4),
            "road_overlap": road_overlap,
            "postcode_match": postcode_match,
            "postcode_outcode_match": outcode_match,
            "os_match_score": os_score,
            "candidate_pool_source": "global_text_os" if strong_match else "global_text_os_weak",
            "global_text_score": round(global_score, 4),
            "source_chargegeog": source_address,
            "weak_reason": weak_reason,
        }
        target_list = scored if strong_match else weak_scored
        target_list.append(((global_score, float(number_overlap), float(road_overlap), token_sim), item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    weak_scored.sort(key=lambda pair: pair[0], reverse=True)
    strong_items = [item for _, item in scored[:max_candidates]]
    if len(strong_items) >= max_candidates:
        return strong_items
    weak_items = [item for _, item in weak_scored[: max_candidates - len(strong_items)]]
    return merge_candidate_lists(strong_items, weak_items, max_candidates=max_candidates)


def load_ocr_roads(ocr_jsonl: Path, folders: set[str], road_names: set[str], max_lines: int = 0) -> dict[str, list[str]]:
    if not folders or not ocr_jsonl.exists():
        return {}
    found_text: dict[str, list[str]] = {folder: [] for folder in folders}
    with ocr_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_lines and line_no > max_lines:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            folder = str(row.get("image_path") or "").split("/", 1)[0]
            if folder not in found_text:
                continue
            parts = []
            for result in row.get("results") or []:
                try:
                    conf = float(result.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0
                if conf >= 0.35:
                    parts.append(str(result.get("text") or ""))
            if parts:
                found_text[folder].append(" ".join(parts))
    out: dict[str, list[str]] = {}
    for folder, chunks in found_text.items():
        text = " | ".join(chunks)
        out[folder] = roi_v1.extract_address_roads(text, road_names)
    return out


def load_open_names(open_names_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(open_names_dir.glob("SK*.csv")):
        try:
            frame = pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig")
        except Exception:
            continue
        if frame.shape[1] < 10:
            continue
        frame = frame[[0, 2, 6, 7, 8, 9]].copy()
        frame.columns = ["id", "name", "type", "local_type", "x", "y"]
        frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        minx, miny, maxx, maxy = MANSFIELD_BBOX
        frame = frame[frame["x"].between(minx - 4000, maxx + 4000) & frame["y"].between(miny - 4000, maxy + 4000)]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["id", "name", "type", "local_type", "x", "y"])


def openname_points(address: str, ocr_roads: list[str], open_names: pd.DataFrame, max_points: int = 4) -> list[dict[str, Any]]:
    haystack = compact(address + " " + " ".join(ocr_roads))
    scored = []
    for _, row in open_names.iterrows():
        name = str(row.get("name") or "")
        cname = compact(name)
        if len(cname) < 5 or cname in GENERIC_PLACE_NAMES:
            continue
        if cname and cname in haystack:
            scored.append((len(cname), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for _, row in scored[:max_points]:
        out.append(
            {
                "point": Point(float(row["x"]), float(row["y"])),
                "source": "openname",
                "label": str(row["name"]),
                "score": 34.0,
            }
        )
    return out


def road_intersection_points(roads: list[str], road_geoms: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    used: set[tuple[str, str]] = set()
    for i, left in enumerate(roads[:5]):
        for right in roads[i + 1 : 6]:
            key = tuple(sorted((left, right)))
            if key in used:
                continue
            used.add(key)
            lg = road_geoms.get(left)
            rg = road_geoms.get(right)
            if lg is None or rg is None or lg.is_empty or rg.is_empty:
                continue
            try:
                if lg.distance(rg) > 80:
                    continue
                p1, p2 = nearest_points(lg, rg)
                point = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            except Exception:
                continue
            out.append({"point": point, "source": "road_intersection", "label": f"{left} & {right}", "score": 48.0})
    return out


def road_geometry_candidates(
    address: str,
    road_geoms: dict[str, Any],
    road_names: set[str],
    max_candidates: int = 10,
) -> list[dict[str, Any]]:
    roads = roi_v1.extract_address_roads(address, road_names)
    if not roads:
        return []
    relation_like = bool(PARCEL_RELATION_RE.search(address)) or len(roads) >= 2
    if not relation_like:
        return []

    candidates: list[dict[str, Any]] = []
    for road in roads[:6]:
        geom = road_geoms.get(road)
        if geom is None or geom.is_empty or getattr(geom, "length", 0.0) <= 0:
            continue
        length = float(geom.length)
        if length <= 800:
            fractions = [0.12, 0.32, 0.52, 0.72, 0.88]
        elif length <= 1800:
            fractions = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
        elif length <= 3500:
            fractions = [0.08, 0.20, 0.34, 0.48, 0.62, 0.76, 0.90]
        else:
            fractions = [0.12, 0.30, 0.50, 0.70, 0.88]
        for frac in fractions:
            try:
                point = geom.interpolate(max(0.0, min(length, length * frac)))
            except Exception:
                continue
            candidates.append(
                {
                    "address": f"OpenRoads sample {road} {frac:.2f}",
                    "easting_27700": float(point.x),
                    "northing_27700": float(point.y),
                    "road_name": road,
                    "place": "Mansfield",
                    "uprn": "",
                    "text_index": None,
                    "text_similarity": 0.0,
                    "source_text_similarity": 0.0,
                    "number_overlap": 0,
                    "number_proximity": 0.0,
                    "road_overlap": 1,
                    "postcode_match": False,
                    "postcode_outcode_match": False,
                    "os_match_score": 0.0,
                    "candidate_pool_source": "openroads_road_sample",
                    "global_text_score": round(14.0 - abs(frac - 0.5) * 4.0, 4),
                    "source_chargegeog": address,
                    "weak_reason": "road_geometry_relation_fallback",
                }
            )
    return candidates[:max_candidates]


def road_corridor_points(points: list[dict[str, Any]], roads: list[str], road_geoms: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for road in roads[:4]:
        geom = road_geoms.get(road)
        if geom is None or geom.is_empty or getattr(geom, "length", 0.0) <= 0:
            continue
        measures = []
        for item in points:
            point = item["point"]
            try:
                projected = nearest_points(point, geom)[1]
                if point.distance(projected) <= 260:
                    measures.append((geom.project(projected), float(item.get("score", 10.0))))
            except Exception:
                continue
        if not measures:
            continue
        measures.sort()
        anchors = [measures[0][0], measures[-1][0]]
        if len(measures) >= 2:
            anchors.append(sum(m * max(w, 1.0) for m, w in measures) / sum(max(w, 1.0) for _, w in measures))
        else:
            anchors.extend([measures[0][0] - 120, measures[0][0] + 120])
        for measure in anchors:
            measure = max(0.0, min(float(geom.length), float(measure)))
            try:
                point = geom.interpolate(measure)
            except Exception:
                continue
            out.append({"point": point, "source": "road_corridor", "label": road, "score": 28.0})
    return out


def range_anchor_points(candidates: list[dict[str, Any]], address: str) -> list[dict[str, Any]]:
    requested = numbers_from_text(address)
    if len(requested) < 2:
        return []
    hits = []
    for candidate in candidates:
        numbers = numbers_from_text(candidate.get("address"))
        overlap = numbers & requested
        if not overlap:
            continue
        score = 25.0 + 8.0 * len(overlap)
        if numbers.issubset(requested):
            score += 16.0
        hits.append((score, Point(float(candidate["easting_27700"]), float(candidate["northing_27700"])), candidate.get("address")))
    if not hits:
        return []
    hits.sort(key=lambda item: item[0], reverse=True)
    top = hits[:8]
    total = sum(max(1.0, score) for score, _, _ in top)
    point = Point(
        sum(point.x * max(1.0, score) for score, point, _ in top) / total,
        sum(point.y * max(1.0, score) for score, point, _ in top) / total,
    )
    return [{"point": point, "source": "range_anchor", "label": " | ".join(str(label) for _, _, label in top[:4]), "score": 62.0}]


def choose_rois(evidence: list[dict[str, Any]], target_geom: Any, max_rois: int) -> list[dict[str, Any]]:
    source_priority = {
        "protected_current": 0,
        "range_anchor": 1,
        "road_intersection": 2,
        "text_os_candidate": 3,
        "road_corridor": 4,
        "openname": 9,
    }
    protected = [item for item in evidence if item.get("source") == "protected_current"]
    rest = [item for item in evidence if item.get("source") != "protected_current"]
    evidence = protected[:1] + sorted(
        rest,
        key=lambda item: (
            source_priority.get(str(item.get("source") or ""), 6),
            -float(item.get("score", 0.0)),
            str(item.get("label") or ""),
        ),
    )
    rois = []
    for item in evidence:
        point = item["point"]
        geom = roi_square(point, 150.0)
        if any(geom.centroid.distance(old["geom"].centroid) < 65 for old in rois):
            continue
        rois.append({**item, "geom": geom})
        if len(rois) >= max_rois:
            break
    # Keep at least one ROI even when all evidence was deduped.
    if not rois and evidence:
        item = evidence[0]
        rois.append({**item, "geom": roi_square(item["point"], 150.0)})
    for rank, item in enumerate(rois, 1):
        item["rank"] = rank
        item["intersects"] = bool(item["geom"].intersects(target_geom))
        item["contains_centroid"] = bool(item["geom"].contains(target_geom.centroid))
    return rois


def run(args: argparse.Namespace) -> None:
    truth = gpd.read_file(args.truth_gpkg, layer=args.truth_layer)
    if truth.crs is None:
        truth = truth.set_crs(27700)
    else:
        truth = truth.to_crs(27700)
    truth["unique_key"] = truth["unique_key"].astype(str)

    road_geoms = roi_v1.load_road_geoms(args.open_roads)
    road_names = set(road_geoms)
    text = load_text_points(args.text_xlsx)
    text_index = prepare_global_text_index(text, road_names)
    text_groups = {folder: group for folder, group in text.groupby("folder")}
    open_names = load_open_names(args.open_names_dir) if args.open_names_dir.exists() else pd.DataFrame()
    folders = {folder_from_filepath(value) for value in truth.get("FilePath", pd.Series(dtype=str))}
    ocr_roads_by_folder = load_ocr_roads(args.ocr_jsonl, folders, road_names, max_lines=args.max_ocr_lines)

    raw_rows = []
    v10_rows = []
    case_rows = []
    roi_rows = []

    for _, row in truth.iterrows():
        key = str(row["unique_key"])
        address = case_address(row)
        folder = folder_from_filepath(row.get("FilePath"))
        folder_candidates = collect_text_candidates(row, text_groups, road_names, args.max_text_candidates)
        global_candidates = collect_global_text_candidates(row, text_index, road_names, args.max_global_text_candidates)
        road_candidates = road_geometry_candidates(address, road_geoms, road_names, max_candidates=10)
        relation_like = bool(PARCEL_RELATION_RE.search(address))
        if relation_like:
            ordered_global = [c for c in global_candidates if c.get("candidate_pool_source") != "global_text_os_weak"]
            ordered_global.extend(road_candidates)
            ordered_global.extend(c for c in global_candidates if c.get("candidate_pool_source") == "global_text_os_weak")
        else:
            ordered_global = [*global_candidates, *road_candidates]
        text_candidates = merge_candidate_lists(
            ordered_global,
            folder_candidates,
            max_candidates=args.max_text_candidates,
        )
        if not text_candidates:
            continue

        best = text_candidates[0]
        best_point = Point(float(best["easting_27700"]), float(best["northing_27700"]))
        address_roads = roi_v1.extract_address_roads(address, road_names)
        ocr_roads = ocr_roads_by_folder.get(folder, [])
        roads: list[str] = []
        for road in [*address_roads, *ocr_roads]:
            if road in road_names and road not in roads:
                roads.append(road)

        evidence: list[dict[str, Any]] = [
            {"point": best_point, "source": "protected_current", "label": "best_text_os", "score": 70.0}
        ]
        for idx, candidate in enumerate(text_candidates[: args.max_text_candidates]):
            point = Point(float(candidate["easting_27700"]), float(candidate["northing_27700"]))
            score = 55.0 - idx * 2.0 + float(candidate.get("text_similarity") or 0.0) * 18.0
            if candidate.get("number_overlap"):
                score += 12.0 + 4.0 * int(candidate["number_overlap"])
            evidence.append({"point": point, "source": "text_os_candidate", "label": candidate["address"], "score": score})

        evidence.extend(range_anchor_points(text_candidates, address))
        evidence.extend(road_intersection_points(roads, road_geoms))
        corridor_roads = address_roads if address_roads else roads[:3]
        evidence.extend(road_corridor_points(evidence, corridor_roads, road_geoms))
        evidence.extend(openname_points(address, ocr_roads, open_names))
        rois = choose_rois(evidence, row.geometry, args.max_rois)
        union = unary_union([item["geom"] for item in rois]) if rois else None

        conf = max(50.0, min(95.0, float(best.get("os_match_score") or 0.72) * 100.0))
        raw_rows.append(
            {
                "key": key,
                "_source_unique_key": key,
                "_is_expanded_case": False,
                "original_address": address,
                "raw_address": address,
                "chargegeog": address,
                "lexicon_road": (roads[0] + " | Mansfield") if roads else "Mansfield",
                "os_query": address,
                "os_candidates_json": json.dumps(text_candidates, ensure_ascii=False),
                "gog_candidates_json": "[]",
                "os_fallback_pool_json": json.dumps(text_candidates[1:], ensure_ascii=False),
                "best_source_final": "os_text_local",
                "best_selection_category": "no_gemini_enriched",
                "best_address_final": best["address"],
                "best_easting_27700_final": best["easting_27700"],
                "best_northing_27700_final": best["northing_27700"],
                "best_confidence": round(conf, 1),
                "best_confidence_reason": "local_text_os_plus_ocr_roads_no_gemini",
            }
        )
        v10_rows.append(
            {
                "key": key,
                "base_key": key,
                "original_address": address,
                "baseline_easting": best["easting_27700"],
                "baseline_northing": best["northing_27700"],
                "v7_selected_easting": best["easting_27700"],
                "v7_selected_northing": best["northing_27700"],
                "ocr_roads": " | ".join(ocr_roads),
            }
        )
        case_rows.append(
            {
                "key": key,
                "base_key": key,
                "original_address": address,
                "best_confidence": round(conf, 1),
                "status": "ok",
                "roi_count": len(rois),
                "added_corridor_roi_count": sum(1 for item in rois if item["source"] in {"road_corridor", "road_intersection"}),
                "all_roads": " | ".join(roads),
                "union_contains_target_centroid": bool(union and union.contains(row.geometry.centroid)),
                "union_intersects_target_polygon": bool(union and union.intersects(row.geometry)),
                "union_candidate_polygon_count": None,
                "target_rank_in_union_candidates": None,
                "base_union_hit": None,
                "sample_split": args.sample_split,
            }
        )
        for item in rois:
            geom = item["geom"]
            roi_rows.append(
                {
                    "case_key": key,
                    "base_key": key,
                    "roi_rank": item["rank"],
                    "roi_reason": item["source"] if item["source"] != "protected_current" else "protected_current",
                    "roi_sources": item["label"],
                    "roi_score": item["score"],
                    "roi_minx": geom.bounds[0],
                    "roi_miny": geom.bounds[1],
                    "roi_maxx": geom.bounds[2],
                    "roi_maxy": geom.bounds[3],
                    "roi_center_easting": geom.centroid.x,
                    "roi_center_northing": geom.centroid.y,
                    "roi_intersects_target_polygon": item["intersects"],
                    "roi_contains_target_centroid": item["contains_centroid"],
                    "sample_split": args.sample_split,
                }
            )

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_name(out_prefix.name + "_gemini_like.json")
    v10_path = out_prefix.with_name(out_prefix.name + "_v10.csv")
    case_path = out_prefix.with_name(out_prefix.name + "_case_summary.csv")
    rois_path = out_prefix.with_name(out_prefix.name + "_rois.csv")
    summary_path = out_prefix.with_suffix(".summary.json")
    json_path.write_text(json.dumps({"rows": raw_rows}, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(v10_rows).to_csv(v10_path, index=False)
    pd.DataFrame(case_rows).to_csv(case_path, index=False)
    pd.DataFrame(roi_rows).to_csv(rois_path, index=False)
    case_df = pd.DataFrame(case_rows)
    summary = {
        "truth_gpkg": str(args.truth_gpkg),
        "truth_layer": args.truth_layer,
        "cases_with_inputs": int(len(case_rows)),
        "roi_rows": int(len(roi_rows)),
        "roi_union_hits": int(case_df["union_intersects_target_polygon"].sum()) if len(case_df) else 0,
        "roi_union_hit_rate": float(case_df["union_intersects_target_polygon"].mean()) if len(case_df) else 0.0,
        "outputs": {
            "json": str(json_path),
            "v10_csv": str(v10_path),
            "case_summary_csv": str(case_path),
            "rois_csv": str(rois_path),
        },
        "no_gemini_api": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-Gemini enriched Mansfield A-stage inputs.")
    parser.add_argument("--truth-gpkg", type=Path, default=DEFAULT_TRUTH_GPKG)
    parser.add_argument("--truth-layer", default=DEFAULT_TRUTH_LAYER)
    parser.add_argument("--text-xlsx", type=Path, default=DEFAULT_TEXT_XLSX)
    parser.add_argument("--ocr-jsonl", type=Path, default=DEFAULT_OCR_JSONL)
    parser.add_argument("--open-roads", type=Path, default=DEFAULT_OPEN_ROADS)
    parser.add_argument("--open-names-dir", type=Path, default=DEFAULT_OPEN_NAMES_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--sample-split", default="random500_excl1200_seed44_no_gemini_enriched")
    parser.add_argument("--max-rois", type=int, default=8)
    parser.add_argument("--max-text-candidates", type=int, default=20)
    parser.add_argument("--max-global-text-candidates", type=int, default=12)
    parser.add_argument("--max-ocr-lines", type=int, default=0, help="Debug only; 0 streams the full OCR JSONL.")
    parser.add_argument("--local-bbox", default="")
    parser.add_argument("--generic-place-names", default="")
    args = parser.parse_args()
    global MANSFIELD_BBOX, GENERIC_PLACE_NAMES, STOP_TOKENS
    MANSFIELD_BBOX = parse_bbox(args.local_bbox, MANSFIELD_BBOX)
    GENERIC_PLACE_NAMES = set(parse_names(args.generic_place_names, GENERIC_PLACE_NAMES))
    STOP_TOKENS = set(STOP_TOKENS) | GENERIC_PLACE_NAMES
    run(args)


if __name__ == "__main__":
    main()
