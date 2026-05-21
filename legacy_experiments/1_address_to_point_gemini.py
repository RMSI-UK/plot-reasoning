#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape as html_unescape
import json
import math
import re
import sys
import threading
import types
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import requests

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None  # type: ignore


HERE = Path(__file__).resolve()

_EMBEDDED_V3_SOURCE = '#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport json\nimport os\nimport re\nimport sqlite3\nimport time\nimport zipfile\nfrom dataclasses import dataclass\nfrom difflib import get_close_matches\nfrom difflib import SequenceMatcher\nfrom pathlib import Path\nfrom typing import Any, Iterable\nfrom xml.sax.saxutils import escape as xml_escape\n\nimport requests\nfrom pyproj import Transformer\nfrom requests.adapters import HTTPAdapter\ntry:\n    from google import genai as _google_genai\nexcept Exception:\n    _google_genai = None  # type: ignore\n\n\nDEFAULT_INPUT = "/data/mansfield/spatial/polygon-layer/mansfield-manual-polygon.csv"\nDEFAULT_KEYS_FILE = "/env/key"\nDEFAULT_ADDRESS_COLUMN = "chargegeog"\nDEFAULT_OUTPUT_CSV = "/data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon_point_v3_road.csv"\nDEFAULT_OUTPUT_JSON = "/data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon_point_v3_road.json"\nDEFAULT_OUTPUT_XLSX = "/data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon_point_v3_road_summary.xlsx"\nDEFAULT_OS_OPEN_NAMES_DB = "/data/base-data/opname_csv_gb/os_open_names_uk.sqlite"\nDEFAULT_COUNTY = "Nottinghamshire"\nDEFAULT_COUNCIL = "Mansfield"\nDEFAULT_TIMEOUT_SECONDS = 20\nDEFAULT_MAX_RETRIES = 3\nDEFAULT_GOOGLE_LOW_CONFIDENCE_THRESHOLD = 75\nDEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"\nUSER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"\nOS_FIND_ENDPOINT = "https://api.os.uk/search/places/v1/find"\nGOOGLE_PLACES_TEXTSEARCH_ENDPOINT = "https://maps.googleapis.com/maps/api/place/textsearch/json"\nGOOGLE_REGION = "uk"\nGOOGLE_COMPONENTS = "country:GB"\nDDG_SEARCH_URL = "https://lite.duckduckgo.com/lite/"\nOS_MAX_RESULTS = 100\nDATASET = "DPA,LPI"\nPLACE_MODIFIER_TOKENS = {\n    "MARKET", "CHURCH", "UPPER", "LOWER", "GREAT", "LITTLE", "OLD", "NEW",\n    "NORTH", "SOUTH", "EAST", "WEST", "UPON", "IN", "UNDER", "SUPER", "LE", "LA", "DE",\n}\nLEADING_ROAD_QUALIFIER_TOKENS = {"LITTLE", "GREAT", "UPPER", "LOWER", "OLD", "NEW"}\nTRAILING_ROAD_DIRECTION_TOKENS = {"NORTH", "SOUTH", "EAST", "WEST"}\n\nROAD_SUFFIXES = {\n    "STREET", "ST", "ROAD", "RD", "LANE", "LN", "CLOSE", "COURT", "AVENUE", "AVE",\n    "DRIVE", "DR", "WAY", "PLACE", "PL", "GATE", "GROVE", "TERRACE", "VIEW", "HILL",\n    "GARDENS", "CRESCENT", "WALK", "RISE", "MEWS", "BOULEVARD", "PARADE", "SQUARE",\n    "ROW", "END", "BANK", "CHASE", "CROFT", "GREEN", "PARK", "FIELD", "FIELDS",\n}\n\nLEADING_NOISE_PATTERNS = [\n    r"^(?:land|plot(?:\\s+of\\s+land)?)\\s+(?:at|off|adjacent\\s+to|adjoining|near|fronting|opposite)\\s+",\n    r"^(?:forecourt|curtilage|rear|garden|garage|outbuilding|car\\s+park|parking\\s+area|access\\s+road)\\s+(?:of|to|at)\\s+",\n    r"^(?:former|ex)[-\\s]+",\n    r"^(?:the\\s+corner\\s+of|corner\\s+of|junction\\s+of)\\s+",\n    r"^(?:at\\s+the\\s+junction\\s+of|land\\s+at\\s+junction)\\s+",\n    r"^(?:unit|units|flat|flats|suite|shop|building|plot|plots)\\s+[A-Z0-9]+(?:\\s*(?:&|and|-)\\s*[A-Z0-9]+)*\\s+",\n    r"^(?:off|at)\\s+",\n]\n\nSPLIT_CONNECTORS_RE = re.compile(r"\\s*(?:/|&|\\band\\b)\\s*", re.IGNORECASE)\nPOSTCODE_DISTRICT_RE = re.compile(r"\\b([A-Z]{1,2}\\d{1,2}[A-Z]?)\\b", re.IGNORECASE)\nLEADING_NUMBER_RE = re.compile(r"^\\d+[A-Z]?(?:\\s*-\\s*\\d+[A-Z]?)?\\s+")\nROAD_WITH_DIR_RE = re.compile(\n    r"\\b([A-Z0-9][A-Z0-9\'&.\\-\\s]+?\\b(?:"\n    + "|".join(sorted(ROAD_SUFFIXES))\n    + r")(?:\\s+(?:NORTH|SOUTH|EAST|WEST))?)\\b",\n    re.IGNORECASE,\n)\nTRAILING_ROAD_AFTER_PREP_RE = re.compile(\n    r"\\b(?:ON|OFF|TO|AT)\\s+([A-Z0-9][A-Z0-9\'&.\\-\\s]+?\\b(?:"\n    + "|".join(sorted(ROAD_SUFFIXES))\n    + r")(?:\\s+(?:NORTH|SOUTH|EAST|WEST))?)\\b",\n    re.IGNORECASE,\n)\nLEADING_HOUSE_RANGE_RE = re.compile(r"^(\\d+)\\s*-\\s*(\\d+)([A-Z]?)\\b", re.IGNORECASE)\n\nANCHOR_RISK_PATTERNS = [\n    r"\\bland\\b",\n    r"\\bplot(?:s)?\\b",\n    r"\\badjacent\\b",\n    r"\\brear\\b",\n    r"\\bforecourt\\b",\n    r"\\bcurtilage\\b",\n    r"\\bjunction\\b",\n    r"\\bcorner\\b",\n    r"\\boff\\b",\n    r"\\bsite\\b",\n    r"\\bindustrial\\s+estate\\b",\n    r"\\bpublic\\s+house\\b",\n    r"\\bsports\\s+ground\\b",\n    r"\\bpavilion\\b",\n    r"\\bschool\\b",\n    r"\\bhotel\\b",\n    r"\\bfarm\\b",\n    r"\\bcentre\\b",\n    r"\\bhospital\\b",\n    r"\\bcar\\s+park\\b",\n]\n\n_wgs84_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)\n\nTOKEN_NORMALIZATION_MAP = {\n    "LN": "LANE",\n    "RD": "ROAD",\n    "ST": "STREET",\n    "AVE": "AVENUE",\n    "AV": "AVENUE",\n    "DR": "DRIVE",\n    "CT": "COURT",\n    "PL": "PLACE",\n    "SQ": "SQUARE",\n    "TER": "TERRACE",\n    "CRES": "CRESCENT",\n    "GDNS": "GARDENS",\n    "GDN": "GARDENS",\n    "PK": "PARK",\n    "MT": "MOUNT",\n}\n\nROAD_TOKEN_NORMALIZATION_MAP = {\n    **TOKEN_NORMALIZATION_MAP,\n    "N": "NORTH",\n    "S": "SOUTH",\n    "E": "EAST",\n    "W": "WEST",\n}\n\n\ndef _clean_text(value: Any) -> str:\n    return re.sub(r"\\s+", " ", str(value or "")).strip(" ,.")\n\n\ndef _norm(value: Any) -> str:\n    text = re.sub(r"[^A-Z0-9]+", " ", _clean_text(value).upper()).strip()\n    if not text:\n        return ""\n    tokens = [TOKEN_NORMALIZATION_MAP.get(token, token) for token in text.split()]\n    return " ".join(tokens).strip()\n\n\ndef _norm_compact(value: Any) -> str:\n    return re.sub(r"[^A-Z0-9]+", "", _clean_text(value).upper())\n\n\ndef _norm_road(value: Any) -> str:\n    text = re.sub(r"[^A-Z0-9]+", " ", _clean_text(value).upper()).strip()\n    if not text:\n        return ""\n    tokens = [ROAD_TOKEN_NORMALIZATION_MAP.get(token, token) for token in text.split()]\n    return " ".join(tokens).strip()\n\n\ndef _norm_road_compact(value: Any) -> str:\n    return re.sub(r"[^A-Z0-9]+", "", _norm_road(value))\n\n\ndef _road_similarity(left: str, right: str) -> float:\n    return SequenceMatcher(None, _norm_road(left), _norm_road(right)).ratio()\n\n\ndef _similarity(left: str, right: str) -> float:\n    return SequenceMatcher(None, _norm(left), _norm(right)).ratio()\n\n\ndef _normalize_postcode_district(value: Any) -> str:\n    return re.sub(r"\\s+", "", _clean_text(value).upper())\n\n\ndef _load_os_api_key_from_keys_file(keys_file: str | None) -> str | None:\n    if not keys_file:\n        return None\n    root = Path(keys_file)\n    if not root.exists():\n        return None\n    patterns = [\n        r"(?im)^\\s*Ordnance\\s+Survey\\s+key\\s*=\\s*(\\S+)\\s*$",\n        r"(?im)^\\s*os\\s*map\\s*[:=]\\s*(\\S+)\\s*$",\n        r"(?im)^\\s*os\\s*[:=]\\s*(\\S+)\\s*$",\n    ]\n    candidate_paths: list[Path] = []\n    if root.is_file():\n        candidate_paths.append(root)\n    else:\n        for extension in ("*.md", "*.txt", "*.env", "*"):\n            candidate_paths.extend(sorted(root.glob(extension)))\n    seen: set[Path] = set()\n    for path in candidate_paths:\n        if path in seen or not path.is_file():\n            continue\n        seen.add(path)\n        try:\n            content = path.read_text(encoding="utf-8")\n        except OSError:\n            continue\n        for pattern in patterns:\n            match = re.search(pattern, content)\n            if match:\n                return match.group(1).strip()\n    return None\n\n\ndef _load_google_api_key_from_keys_file(keys_file: str | None) -> str | None:\n    if not keys_file:\n        return None\n    root = Path(keys_file)\n    if not root.exists():\n        return None\n    patterns = [\n        r"(?im)^\\s*Google\\s+Maps\\s+key\\s*=\\s*(\\S+)\\s*$",\n        r"(?im)^\\s*google\\s*map(?:s)?\\s*[:=]\\s*(\\S+)\\s*$",\n        r"(?im)^\\s*google\\s*[:=]\\s*(\\S+)\\s*$",\n    ]\n    candidate_paths: list[Path] = []\n    if root.is_file():\n        candidate_paths.append(root)\n    else:\n        for extension in ("*.md", "*.txt", "*.env", "*"):\n            candidate_paths.extend(sorted(root.glob(extension)))\n    seen: set[Path] = set()\n    for path in candidate_paths:\n        if path in seen or not path.is_file():\n            continue\n        seen.add(path)\n        try:\n            content = path.read_text(encoding="utf-8")\n        except OSError:\n            continue\n        for pattern in patterns:\n            match = re.search(pattern, content)\n            if match:\n                return match.group(1).strip()\n    return None\n\n\ndef _load_gemini_api_key_from_keys_file(keys_file: str | None) -> str | None:\n    if not keys_file:\n        return None\n    root = Path(keys_file)\n    if not root.exists():\n        return None\n    patterns = [\n        r"(?im)^\\s*gemini\\s*[:=]\\s*(\\S+)\\s*$",\n        r"(?im)^\\s*gemini_api_key\\s*[:=]\\s*(\\S+)\\s*$",\n        r"(?im)^\\s*google\\s+gemini\\s+key\\s*[:=]\\s*(\\S+)\\s*$",\n    ]\n    candidate_paths: list[Path] = []\n    if root.is_file():\n        candidate_paths.append(root)\n    else:\n        for extension in ("*.md", "*.txt", "*.env", "*"):\n            candidate_paths.extend(sorted(root.glob(extension)))\n    seen: set[Path] = set()\n    for path in candidate_paths:\n        if path in seen or not path.is_file():\n            continue\n        seen.add(path)\n        try:\n            content = path.read_text(encoding="utf-8")\n        except OSError:\n            continue\n        for pattern in patterns:\n            match = re.search(pattern, content)\n            if match:\n                return match.group(1).strip()\n    return None\n\n\ndef _build_session() -> requests.Session:\n    session = requests.Session()\n    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)\n    session.mount("http://", adapter)\n    session.mount("https://", adapter)\n    return session\n\n\ndef _request_json_with_retries(\n    session: requests.Session,\n    url: str,\n    params: dict[str, Any],\n    timeout_seconds: int,\n    max_retries: int,\n) -> dict[str, Any]:\n    last_error: str | None = None\n    for attempt in range(max_retries):\n        try:\n            response = session.get(url, params=params, timeout=timeout_seconds)\n            if response.status_code == 429:\n                retry_after = int(response.headers.get("Retry-After", "2"))\n                time.sleep(max(1, retry_after))\n                continue\n            if 500 <= response.status_code < 600:\n                time.sleep(min(2 ** attempt, 8))\n                continue\n            response.raise_for_status()\n            payload = response.json()\n            return payload if isinstance(payload, dict) else {}\n        except requests.RequestException as exc:\n            last_error = str(exc)\n            if attempt < max_retries - 1:\n                time.sleep(min(2 ** attempt, 8))\n                continue\n            raise RuntimeError(last_error or "OS request failed") from exc\n    return {}\n\n\ndef _extract_os_candidates(results: Any) -> list[dict[str, Any]]:\n    if not isinstance(results, list):\n        return []\n    out: list[dict[str, Any]] = []\n    for item in results:\n        if not isinstance(item, dict):\n            continue\n        dpa = item.get("DPA")\n        lpi = item.get("LPI")\n        if isinstance(dpa, dict) and dpa:\n            row = dict(dpa)\n            row["_record_type"] = "DPA"\n            out.append(row)\n        elif isinstance(lpi, dict) and lpi:\n            row = dict(lpi)\n            row["_record_type"] = "LPI"\n            out.append(row)\n    return out\n\n\ndef _query_find(\n    session: requests.Session,\n    api_key: str,\n    query: str,\n    timeout_seconds: int,\n    max_retries: int,\n) -> list[dict[str, Any]]:\n    payload = _request_json_with_retries(\n        session=session,\n        url=OS_FIND_ENDPOINT,\n        params={\n            "query": query,\n            "key": api_key,\n            "dataset": DATASET,\n            "maxresults": OS_MAX_RESULTS,\n            "output_srs": "EPSG:27700",\n        },\n        timeout_seconds=timeout_seconds,\n        max_retries=max_retries,\n    )\n    return _extract_os_candidates(payload.get("results"))\n\n\ndef _google_latlng_to_27700(lat: Any, lng: Any) -> tuple[float | None, float | None]:\n    lat_v = _to_float(lat)\n    lng_v = _to_float(lng)\n    if lat_v is None or lng_v is None:\n        return None, None\n    try:\n        easting, northing = _wgs84_to_bng.transform(float(lng_v), float(lat_v))\n    except Exception:\n        return None, None\n    return round(float(easting), 3), round(float(northing), 3)\n\n\ndef _extract_postcode_from_google_components(components: Any) -> str:\n    if not isinstance(components, list):\n        return ""\n    for comp in components:\n        if not isinstance(comp, dict):\n            continue\n        types = comp.get("types")\n        if isinstance(types, list) and "postal_code" in types:\n            return _normalize_postcode_district(comp.get("long_name") or comp.get("short_name"))\n    return ""\n\n\ndef _extract_postcode_from_text(text: str) -> str:\n    m = re.search(r"\\b([A-Z]{1,2}\\d{1,2}[A-Z]?)\\s*\\d[A-Z]{2}\\b", _clean_text(text).upper())\n    if m:\n        return _normalize_postcode_district(m.group(1))\n    m = POSTCODE_DISTRICT_RE.search(_clean_text(text).upper())\n    return _normalize_postcode_district(m.group(1)) if m else ""\n\n\ndef _google_location_quality_score(location_type: Any) -> float:\n    location_type_text = str(location_type or "").upper().strip()\n    if location_type_text == "ROOFTOP":\n        return 1.0\n    if location_type_text == "RANGE_INTERPOLATED":\n        return 0.75\n    if location_type_text == "GEOMETRIC_CENTER":\n        return 0.55\n    if location_type_text == "APPROXIMATE":\n        return 0.35\n    return 0.0\n\n\ndef _google_places_match_score(types: Any) -> float:\n    type_list = [str(t).strip().lower() for t in types] if isinstance(types, list) else []\n    if "route" in type_list and len(type_list) == 1:\n        return 0.55\n    if any(t in type_list for t in {"street_address", "premise", "subpremise", "establishment", "point_of_interest", "store", "supermarket"}):\n        return 0.85\n    return 0.7\n\n\ndef _synth_google_components_from_places(formatted_address: str) -> list[dict[str, Any]]:\n    text = _clean_text(formatted_address)\n    if not text:\n        return []\n    parts = [p.strip() for p in text.split(",") if _clean_text(p)]\n    comps: list[dict[str, Any]] = []\n    if parts:\n        route_part = re.sub(r"^\\d+[A-Z]?(?:\\s*(?:/|-)\\s*\\d+[A-Z]?)?\\s+", "", parts[0], flags=re.I).strip()\n        route_part = _clean_text(route_part or parts[0])\n        comps.append({"long_name": route_part, "short_name": route_part, "types": ["route"]})\n    if len(parts) >= 2:\n        locality_part = re.sub(r"\\b[A-Z]{1,2}\\d{1,2}[A-Z]?\\s*\\d[A-Z]{2}\\b", "", parts[1], flags=re.I).strip(" ,")\n        locality_part = _clean_text(locality_part or parts[1])\n        comps.append({"long_name": locality_part, "short_name": locality_part, "types": ["locality", "political"]})\n    if len(parts) >= 3:\n        town_part = re.sub(r"\\b[A-Z]{1,2}\\d{1,2}[A-Z]?\\s*\\d[A-Z]{2}\\b", "", parts[2], flags=re.I).strip(" ,")\n        if town_part:\n            comps.append({"long_name": town_part, "short_name": town_part, "types": ["postal_town", "political"]})\n        postcode = _extract_postcode_from_text(parts[2])\n        if postcode:\n            comps.append({"long_name": postcode, "short_name": postcode, "types": ["postal_code", "postal_code_prefix"]})\n    if len(parts) >= 4:\n        comps.append({"long_name": parts[3], "short_name": parts[3], "types": ["administrative_area_level_2", "political"]})\n    return comps\n\n\ndef _query_google_candidates_with_variants(\n    session: requests.Session,\n    google_api_key: str,\n    query: str,\n    timeout_seconds: int,\n    max_retries: int,\n) -> list[dict[str, Any]]:\n    variants: list[str] = []\n    tried: set[str] = set()\n    base = re.sub(r"\\s+", " ", str(query or "")).strip()\n    if base:\n        variants.append(base)\n    no_tail_punct = re.sub(r"[.,;:]+$", "", base).strip()\n    if no_tail_punct and no_tail_punct != base:\n        variants.append(no_tail_punct)\n    slash_as_space = re.sub(r"\\s+", " ", no_tail_punct.replace("/", " ")).strip()\n    if slash_as_space and slash_as_space != no_tail_punct:\n        variants.append(slash_as_space)\n    if base and "," not in base:\n        variants.append(f"{base}, {DEFAULT_COUNCIL}")\n    out: list[dict[str, Any]] = []\n    for q in variants:\n        if not q or q in tried:\n            continue\n        tried.add(q)\n        try:\n            payload = _request_json_with_retries(\n                session=session,\n                url=GOOGLE_PLACES_TEXTSEARCH_ENDPOINT,\n                params={\n                    "query": q,\n                    "key": google_api_key,\n                    "region": GOOGLE_REGION,\n                },\n                timeout_seconds=timeout_seconds,\n                max_retries=max_retries,\n            )\n        except Exception:\n            continue\n        status = str(payload.get("status") or "").upper().strip()\n        if status != "OK":\n            continue\n        raw_results = payload.get("results")\n        if not isinstance(raw_results, list):\n            continue\n        for item in raw_results:\n            if not isinstance(item, dict):\n                continue\n            geometry = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}\n            location = geometry.get("location") if isinstance(geometry.get("location"), dict) else {}\n            formatted_address = _clean_text(item.get("formatted_address"))\n            place_name = _clean_text(item.get("name"))\n            address_text = formatted_address\n            if place_name:\n                place_key = _norm_compact(place_name)\n                addr_key = _norm_compact(formatted_address)\n                if place_key and place_key not in addr_key:\n                    address_text = f"{place_name}, {formatted_address}" if formatted_address else place_name\n                else:\n                    address_text = formatted_address or place_name\n            components = _synth_google_components_from_places(formatted_address)\n            g_lat = _to_float(location.get("lat"))\n            g_lng = _to_float(location.get("lng"))\n            g_x, g_y = _google_latlng_to_27700(g_lat, g_lng)\n            out.append(\n                {\n                    "_record_type": "GOOGLE",\n                    "ADDRESS": address_text,\n                    "MATCH": _google_places_match_score(item.get("types")),\n                    "_google_partial_match": False,\n                    "_google_location_type": "PLACE_TEXT_SEARCH",\n                    "_google_place_id": item.get("place_id"),\n                    "_google_types": item.get("types") if isinstance(item.get("types"), list) else [],\n                    "_google_postcode": _extract_postcode_from_google_components(components),\n                    "_google_address_components": components if isinstance(components, list) else [],\n                    "_google_name": place_name,\n                    "_google_lat": g_lat,\n                    "_google_lng": g_lng,\n                    "X_COORDINATE": g_x,\n                    "Y_COORDINATE": g_y,\n                    "_google_raw": item,\n                }\n            )\n        if out:\n            break\n    return out\n\n\n@dataclass\nclass RoadRow:\n    road_name: str\n    populated_place: str\n    district_borough: str\n    county_unitary: str\n    postcode_district: str\n    geometry_x: float | None\n    geometry_y: float | None\n    local_type: str\n    source_tile: str\n\n\n@dataclass\nclass RoadMatch:\n    road_name: str\n    populated_place: str\n    district_borough: str\n    county_unitary: str\n    postcode_district: str\n    geometry_x: float | None\n    geometry_y: float | None\n    local_type: str\n    matched_phrase: str\n    match_type: str\n    score: float\n    source_tile: str\n\n\nclass OpenNamesRoadMatcher:\n    def __init__(self, db_path: str, county_name: str = "", council_name: str = "") -> None:\n        self.db_path = db_path\n        self.county_name = _clean_text(county_name)\n        self.county_norm = _norm(self.county_name)\n        self.council_name = _clean_text(council_name)\n        self.council_norm = _norm(self.council_name)\n        self.conn = sqlite3.connect(db_path)\n        self.conn.row_factory = sqlite3.Row\n        self._road_names_cache: list[str] | None = None\n        self._road_rows_cache: dict[str, list[RoadRow]] = {}\n        self._place_names_cache: list[str] | None = None\n        self._place_centroid_cache: dict[str, tuple[float, float]] | None = None\n\n    def close(self) -> None:\n        try:\n            self.conn.close()\n        except Exception:\n            pass\n\n    def _query_rows(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:\n        return list(self.conn.execute(sql, params).fetchall())\n\n    def place_centroid_map(self) -> dict[str, tuple[float, float]]:\n        """Return {norm_compact(place_name): (x, y)} from populatedPlace records."""\n        if self._place_centroid_cache is not None:\n            return self._place_centroid_cache\n        if self.county_norm:\n            rows = self._query_rows(\n                """\n                SELECT NAME1, GEOMETRY_X, GEOMETRY_Y\n                FROM open_names\n                WHERE TYPE = \'populatedPlace\'\n                  AND UPPER(REPLACE(COUNTY_UNITARY, \' \', \'\')) LIKE ?\n                  AND GEOMETRY_X != \'\' AND GEOMETRY_Y != \'\'\n                """,\n                (f"%{_norm_compact(self.county_name)}%",),\n            )\n        else:\n            rows = self._query_rows(\n                "SELECT NAME1, GEOMETRY_X, GEOMETRY_Y FROM open_names WHERE TYPE=\'populatedPlace\' AND GEOMETRY_X != \'\'",\n                (),\n            )\n        result: dict[str, tuple[float, float]] = {}\n        for row in rows:\n            x, y = _to_float(row[1]), _to_float(row[2])\n            if x is not None and y is not None:\n                result[_norm_compact(row[0])] = (x, y)\n        self._place_centroid_cache = result\n        return result\n\n    def place_names(self) -> list[str]:\n        if self._place_names_cache is not None:\n            return self._place_names_cache\n        if self.county_norm:\n            rows = self._query_rows(\n                """\n                SELECT DISTINCT NAME1\n                FROM open_names\n                WHERE TYPE = \'populatedPlace\'\n                  AND UPPER(REPLACE(COUNTY_UNITARY, \' \', \'\')) LIKE ?\n                ORDER BY LENGTH(NAME1) DESC, NAME1\n                """,\n                (f"%{_norm_compact(self.county_name)}%",),\n            )\n        else:\n            rows = self._query_rows(\n                "SELECT DISTINCT NAME1 FROM open_names WHERE TYPE = \'populatedPlace\' ORDER BY LENGTH(NAME1) DESC, NAME1",\n                (),\n            )\n        self._place_names_cache = [_clean_text(row[0]) for row in rows if _clean_text(row[0])]\n        return self._place_names_cache\n\n    def road_names(self) -> list[str]:\n        if self._road_names_cache is not None:\n            return self._road_names_cache\n        if self.county_norm:\n            rows = self._query_rows(\n                """\n                SELECT DISTINCT NAME1\n                FROM open_names\n                WHERE TYPE = \'transportNetwork\'\n                  AND LOCAL_TYPE IN (\'Named Road\', \'Section Of Named Road\')\n                  AND UPPER(REPLACE(COUNTY_UNITARY, \' \', \'\')) LIKE ?\n                ORDER BY NAME1\n                """,\n                (f"%{_norm_compact(self.county_name)}%",),\n            )\n        else:\n            rows = self._query_rows(\n                "SELECT DISTINCT NAME1 FROM open_names WHERE TYPE = \'transportNetwork\' AND LOCAL_TYPE IN (\'Named Road\', \'Section Of Named Road\') ORDER BY NAME1",\n                (),\n            )\n        self._road_names_cache = [_clean_text(row[0]) for row in rows if _clean_text(row[0])]\n        return self._road_names_cache\n\n    def rows_for_road(self, road_name: str) -> list[RoadRow]:\n        key = _norm_compact(road_name)\n        if key in self._road_rows_cache:\n            return self._road_rows_cache[key]\n        if self.county_norm:\n            rows = self._query_rows(\n                """\n                SELECT NAME1, POPULATED_PLACE, DISTRICT_BOROUGH, COUNTY_UNITARY,\n                       POSTCODE_DISTRICT, GEOMETRY_X, GEOMETRY_Y, LOCAL_TYPE, \'\'\n                FROM open_names\n                WHERE TYPE = \'transportNetwork\'\n                  AND LOCAL_TYPE IN (\'Named Road\', \'Section Of Named Road\')\n                  AND UPPER(REPLACE(NAME1, \' \', \'\')) = ?\n                  AND UPPER(REPLACE(COUNTY_UNITARY, \' \', \'\')) LIKE ?\n                """,\n                (_norm_compact(road_name), f"%{_norm_compact(self.county_name)}%"),\n            )\n        else:\n            rows = self._query_rows(\n                """\n                SELECT NAME1, POPULATED_PLACE, DISTRICT_BOROUGH, COUNTY_UNITARY,\n                       POSTCODE_DISTRICT, GEOMETRY_X, GEOMETRY_Y, LOCAL_TYPE, \'\'\n                FROM open_names\n                WHERE TYPE = \'transportNetwork\'\n                  AND LOCAL_TYPE IN (\'Named Road\', \'Section Of Named Road\')\n                  AND UPPER(REPLACE(NAME1, \' \', \'\')) = ?\n                """,\n                (_norm_compact(road_name),),\n            )\n        mapped = [\n            RoadRow(\n                road_name=_clean_text(row[0]),\n                populated_place=_clean_text(row[1]),\n                district_borough=_clean_text(row[2]),\n                county_unitary=_clean_text(row[3]),\n                postcode_district=_clean_text(row[4]),\n                geometry_x=_to_float(row[5]),\n                geometry_y=_to_float(row[6]),\n                local_type=_clean_text(row[7]),\n                source_tile=_clean_text(row[8]),\n            )\n            for row in rows\n        ]\n        # When the same (road, populated_place) has multiple rows at different\n        # coordinates (e.g. two sections of Church Street in Mansfield), keep\n        # only the row closest to the populated_place centroid.  This avoids\n        # arbitrarily picking whichever section DB returns first.\n        centroid_map = self.place_centroid_map()\n        best_per_place: dict[str, RoadRow] = {}\n        for road_row in mapped:\n            place_key = _norm_compact(road_row.populated_place)\n            dedup_key = f"{_norm_compact(road_row.road_name)}|{place_key}"\n            if dedup_key not in best_per_place:\n                best_per_place[dedup_key] = road_row\n            else:\n                centroid = centroid_map.get(place_key)\n                if centroid and road_row.geometry_x and road_row.geometry_y:\n                    cx, cy = centroid\n                    existing = best_per_place[dedup_key]\n                    ex, ey = existing.geometry_x or cx, existing.geometry_y or cy\n                    d_new = (road_row.geometry_x - cx) ** 2 + (road_row.geometry_y - cy) ** 2\n                    d_old = (ex - cx) ** 2 + (ey - cy) ** 2\n                    if d_new < d_old:\n                        best_per_place[dedup_key] = road_row\n        mapped = list(best_per_place.values())\n        self._road_rows_cache[key] = mapped\n        return mapped\n\n    def rows_for_road_prefix(self, phrase: str) -> list[tuple[str, RoadRow]]:\n        """Return (matched_road_name, row) pairs where phrase is a word-boundary\n        prefix of an actual road name.  E.g. "Southwell Road" matches\n        "Southwell Road West" and "Southwell Road East".\n        Only applies when the phrase itself ends with a road suffix — this\n        prevents place names ("Warsop") or site words ("Colliery") from being\n        spuriously extended to road names ("Warsop Road", "Colliery Way").\n        """\n        if not _road_like_suffix(phrase):\n            return []\n        phrase_words = _norm(phrase).split()\n        n = len(phrase_words)\n        if n == 0:\n            return []\n        results: list[tuple[str, RoadRow]] = []\n        seen: set[str] = set()\n        for road_name in self.road_names():\n            road_words = _norm(road_name).split()\n            if len(road_words) <= n:\n                continue\n            if road_words[:n] != phrase_words:\n                continue\n            key = _norm_compact(road_name)\n            if key in seen:\n                continue\n            seen.add(key)\n            for row in self.rows_for_road(road_name):\n                results.append((road_name, row))\n        return results\n\n    def best_match(\n        self,\n        raw_address: str,\n        max_candidates: int = 5,\n    ) -> tuple[RoadMatch | None, list[RoadMatch], dict[str, Any]]:\n        phrases, debug = _extract_road_phrases_stage1(raw_address, self.place_names())\n        best, limited, debug = self._match_from_phrases(\n            raw_address=raw_address,\n            phrases=phrases,\n            debug=debug,\n            max_candidates=max_candidates,\n        )\n        if best is not None:\n            debug["match_stage"] = "stage1"\n            return best, limited, debug\n\n        fallback_phrases, fallback_debug = _extract_road_phrases(raw_address, self.place_names())\n        fallback_best, fallback_limited, fallback_debug = self._match_from_phrases(\n            raw_address=raw_address,\n            phrases=fallback_phrases,\n            debug=fallback_debug,\n            max_candidates=max_candidates,\n        )\n        fallback_debug["match_stage"] = "stage2_fallback"\n        return fallback_best, fallback_limited, fallback_debug\n\n    def _match_from_phrases(\n        self,\n        raw_address: str,\n        phrases: list[str],\n        debug: dict[str, Any],\n        max_candidates: int,\n    ) -> tuple[RoadMatch | None, list[RoadMatch], dict[str, Any]]:\n        phrase_rank = {phrase: idx for idx, phrase in enumerate(phrases)}\n        standalone_segments = {_norm_compact(seg) for seg in debug.get("road_like_segments", [])}\n        place_hints = _extract_place_hints(raw_address, self.place_names(), phrases)\n        postcode_hints = sorted(set(m.upper() for m in POSTCODE_DISTRICT_RE.findall(raw_address or "")))\n        matches: list[RoadMatch] = []\n        seen: set[tuple[str, str]] = set()\n        exact_phrase_hits: dict[str, bool] = {}\n\n        for phrase in phrases:\n            exact_rows = self.rows_for_road(phrase)\n            exact_phrase_hits[phrase] = bool(exact_rows)\n            if exact_rows:\n                for row in exact_rows:\n                    key = (_norm_compact(row.road_name), _norm_compact(row.populated_place))\n                    if key in seen:\n                        continue\n                    seen.add(key)\n                    matches.append(\n                        RoadMatch(\n                            road_name=row.road_name,\n                            populated_place=row.populated_place,\n                            district_borough=row.district_borough,\n                            county_unitary=row.county_unitary,\n                            postcode_district=row.postcode_district,\n                            geometry_x=row.geometry_x,\n                            geometry_y=row.geometry_y,\n                            local_type=row.local_type,\n                            matched_phrase=phrase,\n                            match_type="exact",\n                            score=_score_match(\n                                phrase,\n                                row,\n                                place_hints,\n                                postcode_hints,\n                                self.council_name,\n                                _norm_compact(phrase) in standalone_segments,\n                                True,\n                                phrase_rank.get(phrase, 0),\n                            ),\n                            source_tile=row.source_tile,\n                        )\n                    )\n\n        for phrase in phrases:\n            if exact_phrase_hits.get(phrase):\n                continue\n            for simplified in _fallback_simplified_road_phrases(phrase):\n                fallback_rows = self.rows_for_road(simplified)\n                if not fallback_rows:\n                    continue\n                for row in fallback_rows:\n                    key = (_norm_compact(row.road_name), _norm_compact(row.populated_place))\n                    if key in seen:\n                        continue\n                    seen.add(key)\n                    matches.append(\n                        RoadMatch(\n                            road_name=row.road_name,\n                            populated_place=row.populated_place,\n                            district_borough=row.district_borough,\n                            county_unitary=row.county_unitary,\n                            postcode_district=row.postcode_district,\n                            geometry_x=row.geometry_x,\n                            geometry_y=row.geometry_y,\n                            local_type=row.local_type,\n                            matched_phrase=phrase,\n                            match_type=f"simplified:{simplified}",\n                            score=_score_match(\n                                simplified,\n                                row,\n                                place_hints,\n                                postcode_hints,\n                                self.council_name,\n                                _norm_compact(phrase) in standalone_segments,\n                                False,\n                                phrase_rank.get(phrase, 0),\n                            ) - 2.0,\n                            source_tile=row.source_tile,\n                        )\n                    )\n\n        # Prefix-extension pass: "Southwell Road" → "Southwell Road West/East".\n        # Runs always (not just when exact fails) so that a well-placed prefix match\n        # can outscore a poorly-placed exact match.\n        for phrase in phrases:\n            for road_name, row in self.rows_for_road_prefix(phrase):\n                key = (_norm_compact(row.road_name), _norm_compact(row.populated_place))\n                if key in seen:\n                    continue\n                seen.add(key)\n                matches.append(\n                    RoadMatch(\n                        road_name=row.road_name,\n                        populated_place=row.populated_place,\n                        district_borough=row.district_borough,\n                        county_unitary=row.county_unitary,\n                        postcode_district=row.postcode_district,\n                        geometry_x=row.geometry_x,\n                        geometry_y=row.geometry_y,\n                        local_type=row.local_type,\n                        matched_phrase=phrase,\n                        match_type="prefix_ext",\n                        score=_score_match(\n                            phrase,\n                            row,\n                            place_hints,\n                            postcode_hints,\n                            self.council_name,\n                            _norm_compact(phrase) in standalone_segments,\n                            False,  # not exact\n                            phrase_rank.get(phrase, 0),\n                        ),\n                        source_tile=row.source_tile,\n                    )\n                )\n\n        if not matches and phrases:\n            road_names = self.road_names()\n            best_name = ""\n            best_phrase = ""\n            best_score = 0.0\n            for phrase in phrases:\n                for road_name in road_names:\n                    score = _similarity(phrase, road_name)\n                    if score > best_score:\n                        best_score = score\n                        best_name = road_name\n                        best_phrase = phrase\n            if best_name and best_score >= 0.90:\n                for row in self.rows_for_road(best_name):\n                    key = (_norm_compact(row.road_name), _norm_compact(row.populated_place))\n                    if key in seen:\n                        continue\n                    seen.add(key)\n                    matches.append(\n                        RoadMatch(\n                            road_name=row.road_name,\n                            populated_place=row.populated_place,\n                            district_borough=row.district_borough,\n                            county_unitary=row.county_unitary,\n                            postcode_district=row.postcode_district,\n                            geometry_x=row.geometry_x,\n                            geometry_y=row.geometry_y,\n                            local_type=row.local_type,\n                            matched_phrase=best_phrase,\n                            match_type=f"fuzzy:{best_score:.3f}",\n                            score=_score_match(\n                                best_phrase,\n                                row,\n                                place_hints,\n                                postcode_hints,\n                                self.council_name,\n                                _norm_compact(best_phrase) in standalone_segments,\n                                False,\n                                phrase_rank.get(best_phrase, 0),\n                            ) + round(best_score * 10, 3),\n                            source_tile=row.source_tile,\n                        )\n                    )\n\n        matches.sort(key=lambda item: (item.score, item.match_type == "exact"), reverse=True)\n        limited = matches[:max_candidates]\n        best = limited[0] if limited else None\n        debug.update(\n            {\n                "place_hints": place_hints,\n                "postcode_hints": postcode_hints,\n                "candidate_phrase_count": len(phrases),\n            }\n        )\n        return best, limited, debug\n\n\ndef _duckduckgo_search(session: requests.Session, query: str, n: int = 10) -> list[dict[str, str]]:\n    try:\n        response = session.post(\n            DDG_SEARCH_URL,\n            data={"q": query, "kl": "uk-en"},\n            headers={"User-Agent": USER_AGENT},\n            timeout=20,\n            allow_redirects=True,\n        )\n        html = response.text\n    except Exception:\n        return []\n    results: list[dict[str, str]] = []\n    anchor_pattern = re.compile(r\'<a\\b[^>]*class=[\\\'"]result-link[\\\'"][^>]*>(.*?)</a>\', re.S)\n    href_pattern = re.compile(r\'\\bhref=[\\\'"]([^\\\'"]+)[\\\'"]\')\n    snippet_pattern = re.compile(r\'class=[\\\'"]result-snippet[\\\'"][^>]*>(.*?)</td>\', re.S)\n    anchors = anchor_pattern.findall(html)\n    full_anchors = re.findall(r\'<a\\b[^>]*class=[\\\'"]result-link[\\\'"][^>]*>\', html, re.S)\n    snippets = [re.sub(r"<[^>]+>", " ", s).strip() for s in snippet_pattern.findall(html)]\n    for i, (full_tag, inner) in enumerate(zip(full_anchors[:n], anchors[:n])):\n        href_m = href_pattern.search(full_tag)\n        url = href_m.group(1) if href_m else ""\n        title_clean = re.sub(r"<[^>]+>", "", inner).strip()\n        snippet = snippets[i] if i < len(snippets) else ""\n        snippet_clean = re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", snippet)).strip()\n        results.append({"title": title_clean, "url": url, "snippet": snippet_clean})\n    return results[:n]\n\n\ndef _gemini_extract_address_and_coords_27700(\n    gemini_api_key: str,\n    gemini_model: str,\n    ddg_results: list[dict[str, str]],\n    charge_desc: str,\n    council: str,\n    county: str,\n) -> dict[str, Any]:\n    if _google_genai is None:\n        return {"status": "unavailable", "raw_text": "", "matched_address": "", "easting_27700": None, "northing_27700": None, "confidence": 0}\n    search_block = json.dumps(ddg_results, ensure_ascii=False)\n    prompt = (\n        f"You are geocoding a UK planning-related address in {council}, {county}.\\n\\n"\n        f"Original address:\\n{charge_desc}\\n\\n"\n        f"Web search evidence (DuckDuckGo results as JSON):\\n{search_block}\\n\\n"\n        "Return ONLY JSON with this schema:\\n"\n        \'{"matched_address":"...", "easting_27700":123456.0, "northing_27700":654321.0, "confidence":0-100, "reason":"..."}\\n\\n\'\n        "Rules:\\n"\n        "- Use EPSG:27700 British National Grid coordinates.\\n"\n        "- The matched_address should be the best normalized address you infer from the evidence.\\n"\n        "- If uncertain, lower confidence.\\n"\n        "- If you cannot determine a plausible UK address and coordinates, return matched_address as empty string, coordinates as null, confidence as 0.\\n"\n        "- Prefer being conservative.\\n"\n    )\n    try:\n        client = _google_genai.Client(api_key=gemini_api_key)\n        response = client.models.generate_content(model=gemini_model, contents=prompt)\n        text = (response.text or "").strip()\n    except Exception as exc:\n        return {"status": "error", "raw_text": str(exc), "matched_address": "", "easting_27700": None, "northing_27700": None, "confidence": 0}\n    payload: dict[str, Any] | None = None\n    try:\n        payload = json.loads(text)\n    except Exception:\n        m = re.search(r"\\{.*\\}", text, re.S)\n        if m:\n            try:\n                payload = json.loads(m.group(0))\n            except Exception:\n                payload = None\n    if not isinstance(payload, dict):\n        nums = re.findall(r"[-+]?\\d+(?:\\.\\d+)?", text)\n        if len(nums) >= 2:\n            payload = {\n                "matched_address": "",\n                "easting_27700": float(nums[0]),\n                "northing_27700": float(nums[1]),\n                "confidence": 50,\n                "reason": "parsed_from_free_text",\n            }\n        else:\n            return {"status": "parse_error", "raw_text": text, "matched_address": "", "easting_27700": None, "northing_27700": None, "confidence": 0}\n    return {\n        "status": "matched" if payload.get("easting_27700") not in (None, "") and payload.get("northing_27700") not in (None, "") else "no_match",\n        "raw_text": text,\n        "matched_address": _clean_text(payload.get("matched_address")),\n        "easting_27700": _to_float(payload.get("easting_27700")),\n        "northing_27700": _to_float(payload.get("northing_27700")),\n        "confidence": max(0, min(100, int(round(_to_float(payload.get("confidence")) or 0)))),\n        "reason": _clean_text(payload.get("reason")),\n    }\n\n\ndef _to_float(value: Any) -> float | None:\n    try:\n        if value is None or value == "":\n            return None\n        return float(value)\n    except (TypeError, ValueError):\n        return None\n\n\ndef _score_match(\n    phrase: str,\n    row: RoadRow,\n    place_hints: list[str],\n    postcode_hints: list[str],\n    council_name: str,\n    is_standalone_segment: bool,\n    exact: bool,\n    phrase_position: int,\n) -> float:\n    score = 100.0 if exact else 80.0\n    place_score = _best_place_score(place_hints, row.populated_place)\n    score += place_score\n    if postcode_hints and row.postcode_district.upper() in postcode_hints:\n        score += 8.0\n    if council_name and _norm(row.district_borough) == _norm(council_name):\n        score += 0.5\n    if is_standalone_segment:\n        score += 3.0\n    if row.populated_place:\n        score += 2.0\n    if row.local_type == "Named Road":\n        score += 1.5\n    score += round(_similarity(phrase, row.road_name) * 10.0, 3)\n    score += min(phrase_position, 5) * 0.25\n    return round(score, 3)\n\n\ndef _norm_tokens(value: Any) -> list[str]:\n    return [token for token in _norm(value).split() if token]\n\n\ndef _core_place_tokens(value: Any) -> list[str]:\n    tokens = [token for token in _norm_tokens(value) if token not in PLACE_MODIFIER_TOKENS]\n    return tokens or _norm_tokens(value)\n\n\ndef _place_pair_score(hint: str, candidate: str) -> float:\n    hint_clean = _clean_text(hint)\n    cand_clean = _clean_text(candidate)\n    if not hint_clean or not cand_clean:\n        return 0.0\n    hint_compact = _norm_compact(hint_clean)\n    cand_compact = _norm_compact(cand_clean)\n    if hint_compact == cand_compact:\n        return 24.0\n\n    hint_core = _core_place_tokens(hint_clean)\n    cand_core = _core_place_tokens(cand_clean)\n    hint_core_set = set(hint_core)\n    cand_core_set = set(cand_core)\n    if hint_core_set and hint_core_set == cand_core_set:\n        return 22.0\n    if hint_core_set and cand_core_set and hint_core_set.issubset(cand_core_set):\n        return 20.0\n    if hint_core_set and cand_core_set and cand_core_set.issubset(hint_core_set):\n        return 18.0\n\n    overlap = hint_core_set & cand_core_set\n    if overlap:\n        coverage = len(overlap) / max(1, len(hint_core_set))\n        return round(10.0 + coverage * 6.0, 3)\n\n    sim = _similarity(hint_clean, cand_clean)\n    if sim >= 0.92:\n        return round(8.0 + sim * 4.0, 3)\n    return 0.0\n\n\ndef _best_place_score(place_hints: list[str], populated_place: str) -> float:\n    """Score how well populated_place matches the place hints.\n\n    Hints are ordered from most specific (leftmost in address) to broadest.\n    The first hint gets full weight; each subsequent hint gets 0.4x weight.\n    This prevents a broad city name (Mansfield) from outscoring a more specific\n    sub-place (Market Warsop) just because it happens to exact-match.\n    """\n    if not place_hints:\n        return 0.0\n    best = 0.0\n    for i, hint in enumerate(place_hints):\n        raw = _place_pair_score(hint, populated_place)\n        weight = 1.0 if i == 0 else 0.4\n        best = max(best, raw * weight)\n    return best\n\n\ndef _extract_place_hints(raw_address: str, place_names: Iterable[str], road_phrases: list[str]) -> list[str]:\n    """Extract place hints in segment order (leftmost = most specific = first).\n\n    For each comma segment that is not a road phrase:\n    - Try exact match first; if found, add it.\n    - If no exact match, try alias/token-subset expansion and add best results.\n\n    Processing left-to-right preserves address hierarchy so that sub-place names\n    (e.g. Market Warsop from "Warsop") appear before broader city names (Mansfield).\n    This ordering is used by _best_place_score to weight specific hints higher.\n    """\n    place_list = list(place_names)\n    road_phrase_keys = {_norm_compact(value) for value in road_phrases}\n    base_segments = [_clean_text(part) for part in _clean_text(raw_address).split(",") if _clean_text(part)]\n    found: list[str] = []\n    seen: set[str] = set()\n\n    for segment in base_segments:\n        seg_key = _norm_compact(segment)\n        if not seg_key:\n            continue\n        # Only skip segments that look like roads (have a road suffix).\n        # A segment like "Warsop" has no road suffix; it won\'t be in the place DB\n        # but is still a place alias — don\'t discard it.\n        if seg_key in road_phrase_keys and _road_like_suffix(segment):\n            continue\n\n        # Try exact match\n        exact_matched = False\n        for place in place_list:\n            if _norm_compact(place) == seg_key:\n                key = _norm_compact(place)\n                if key not in seen:\n                    seen.add(key)\n                    found.append(place)\n                exact_matched = True\n\n        # If no exact match, try alias/token-subset expansion\n        if not exact_matched:\n            segment_core = set(_core_place_tokens(segment))\n            if not segment_core:\n                continue\n            candidates: list[tuple[float, str]] = []\n            for place in place_list:\n                place_core = set(_core_place_tokens(place))\n                if not place_core:\n                    continue\n                if segment_core == place_core or segment_core.issubset(place_core):\n                    s = _place_pair_score(segment, place)\n                    if s > 0:\n                        candidates.append((s, place))\n            candidates.sort(key=lambda item: item[0], reverse=True)\n            for _, place in candidates[:3]:\n                key = _norm_compact(place)\n                if key not in seen:\n                    seen.add(key)\n                    found.append(place)\n\n    return found\n\n\ndef _extract_freeform_place_hints(raw_address: str, road_phrases: list[str]) -> list[str]:\n    segments = [_clean_text(part) for part in _clean_text(raw_address).split(",") if _clean_text(part)]\n    if not segments:\n        return []\n    road_keys = {_norm_compact(value) for value in road_phrases if _norm_compact(value)}\n    road_index = -1\n    for idx, segment in enumerate(segments):\n        seg_key = _norm_compact(segment)\n        if seg_key in road_keys or _road_like_suffix(segment):\n            road_index = idx\n            break\n    if road_index < 0:\n        return []\n\n    out: list[str] = []\n    seen: set[str] = set()\n    for segment in segments[road_index + 1:]:\n        if re.search(r"\\d", segment):\n            continue\n        if _road_like_suffix(segment):\n            continue\n        token_count = len(_norm(segment).split())\n        if token_count == 0 or token_count > 4:\n            continue\n        key = _norm_compact(segment)\n        if key and key not in seen:\n            seen.add(key)\n            out.append(segment)\n    return out\n\n\ndef _best_fuzzy_choice(query: str, candidates: list[str], min_score: float, min_margin: float) -> str:\n    clean_query = _clean_text(query)\n    if not clean_query:\n        return ""\n    matches = get_close_matches(clean_query, candidates, n=3, cutoff=min_score)\n    if not matches:\n        return ""\n    best = matches[0]\n    best_score = _similarity(clean_query, best)\n    second_score = _similarity(clean_query, matches[1]) if len(matches) > 1 else 0.0\n    if best_score < min_score:\n        return ""\n    if second_score and (best_score - second_score) < min_margin:\n        return ""\n    return best\n\n\ndef _replace_phrase_preserving_leading_number(updated: str, phrase: str, road_fix: str) -> str:\n    """Replace a matched road phrase without dropping a leading property number/range."""\n    clean_phrase = _clean_text(phrase)\n    clean_fix = _clean_text(road_fix)\n    if not clean_phrase or not clean_fix:\n        return updated\n    match = re.match(r"^(\\d+[A-Z]?(?:\\s*-\\s*\\d+[A-Z]?)?)\\s+(.+)$", clean_phrase, flags=re.IGNORECASE)\n    if match:\n        prefix, remainder = match.groups()\n        replacement = f"{prefix} {clean_fix}"\n        return re.sub(re.escape(phrase), replacement, updated, flags=re.IGNORECASE)\n    return re.sub(re.escape(phrase), clean_fix, updated, flags=re.IGNORECASE)\n\n\ndef _correct_raw_address_spelling(raw_address: str, matcher: OpenNamesRoadMatcher) -> tuple[str, list[str]]:\n    segments = [_clean_text(part) for part in _clean_text(raw_address).split(",") if _clean_text(part)]\n    corrected: list[str] = []\n    notes: list[str] = []\n    road_names = matcher.road_names()\n    place_names = matcher.place_names()\n    for segment in segments:\n        updated = segment\n        fixed = False\n        for phrase in _phrase_variants(segment):\n            road_fix = _best_fuzzy_choice(phrase, road_names, min_score=0.94, min_margin=0.02)\n            if road_fix and _norm_compact(road_fix) != _norm_compact(phrase):\n                updated = _replace_phrase_preserving_leading_number(updated, phrase, road_fix)\n                notes.append(f"road:{phrase}->{road_fix}")\n                fixed = True\n                break\n        if not fixed:\n            place_fix = _best_fuzzy_choice(segment, place_names, min_score=0.94, min_margin=0.02)\n            if place_fix and _norm_compact(place_fix) != _norm_compact(segment):\n                updated = place_fix\n                notes.append(f"place:{segment}->{place_fix}")\n        corrected.append(updated)\n    return ", ".join(corrected), notes\n\n\ndef _compose_os_query(corrected_raw_address: str, council_name: str, county_name: str) -> str:\n    return _clean_text(corrected_raw_address)\n\n\ndef _expand_os_query_variants(corrected_raw_address: str) -> list[str]:\n    """Keep the original query, then add small house-number range expansions.\n\n    Example:\n    11-15 Ratcliffe Gate, Mansfield -> [\n        "11-15 Ratcliffe Gate, Mansfield",\n        "11 Ratcliffe Gate, Mansfield",\n        "12 Ratcliffe Gate, Mansfield",\n        ...\n    ]\n    """\n    base = _clean_text(corrected_raw_address)\n    if not base:\n        return []\n    variants = [base]\n\n    match = LEADING_HOUSE_RANGE_RE.match(base)\n    if not match:\n        return variants\n\n    start = int(match.group(1))\n    end = int(match.group(2))\n    suffix = (match.group(3) or "").upper()\n    if suffix:\n        return variants\n    if end < start:\n        start, end = end, start\n    span = end - start + 1\n    if span < 2 or span > 10:\n        return variants\n\n    remainder = base[match.end():].lstrip(" ,")\n    for number in range(start, end + 1):\n        if remainder:\n            variants.append(f"{number} {remainder}")\n        else:\n            variants.append(str(number))\n\n    out: list[str] = []\n    seen: set[str] = set()\n    for value in variants:\n        cleaned = _clean_text(value)\n        key = _norm_compact(cleaned)\n        if cleaned and key and key not in seen:\n            seen.add(key)\n            out.append(cleaned)\n    return out\n\n\ndef _dedupe_os_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:\n    out: list[dict[str, Any]] = []\n    seen: set[str] = set()\n    for candidate in candidates:\n        key = "|".join(\n            [\n                _clean_text(candidate.get("UPRN")),\n                _clean_text(candidate.get("LPI_KEY")),\n                _clean_text(candidate.get("ADDRESS")),\n                str(candidate.get("X_COORDINATE") or ""),\n                str(candidate.get("Y_COORDINATE") or ""),\n            ]\n        )\n        if key in seen:\n            continue\n        seen.add(key)\n        out.append(candidate)\n    return out\n\n\ndef _extract_first_half_numbers(text: Any) -> set[str]:\n    s = _clean_text(text)\n    if not s:\n        return set()\n    half = s[: max(1, len(s) // 2)]\n    return set(re.findall(r"\\b\\d+[A-Z]?\\b", half.upper()))\n\n\ndef _parse_address_number_span(text: Any) -> tuple[int, int, str, str] | None:\n    s = _clean_text(text).upper()\n    if not s:\n        return None\n    match = re.search(r"\\b(\\d+)([A-Z]?)\\s*-\\s*(\\d+)([A-Z]?)\\b", s)\n    if match:\n        start = int(match.group(1))\n        start_suffix = match.group(2) or ""\n        end = int(match.group(3))\n        end_suffix = match.group(4) or ""\n        if end < start:\n            start, end = end, start\n            start_suffix, end_suffix = end_suffix, start_suffix\n        return start, end, start_suffix, end_suffix\n    match = re.search(r"\\b(\\d+)([A-Z]?)\\b", s)\n    if match:\n        number = int(match.group(1))\n        suffix = match.group(2) or ""\n        return number, number, suffix, suffix\n    return None\n\n\ndef _candidate_number_span(candidate: dict[str, Any]) -> tuple[int, int, str, str] | None:\n    start = _clean_text(candidate.get("PAO_START_NUMBER") or candidate.get("BUILDING_NUMBER"))\n    end = _clean_text(candidate.get("PAO_END_NUMBER"))\n    start_suffix = _clean_text(candidate.get("PAO_START_SUFFIX"))\n    end_suffix = _clean_text(candidate.get("PAO_END_SUFFIX"))\n    if start.isdigit():\n        start_num = int(start)\n        end_num = int(end) if end.isdigit() else start_num\n        if end_num < start_num:\n            start_num, end_num = end_num, start_num\n        return start_num, end_num, start_suffix.upper(), (end_suffix or start_suffix).upper()\n    return _parse_address_number_span(_os_candidate_address(candidate))\n\n\ndef _number_span_relation(\n    left: tuple[int, int, str, str] | None,\n    right: tuple[int, int, str, str] | None,\n) -> tuple[int, int]:\n    if left and right:\n        l0, l1, ls0, ls1 = left\n        r0, r1, rs0, rs1 = right\n        if left == right:\n            if l0 == l1:\n                return 3, 0\n            return 4, 0\n        if l0 == l1 and r0 <= l0 <= r1 and (not ls0 or not rs0 or ls0 == rs0):\n            return 2, 0\n        if r0 == r1 and l0 <= r0 <= l1 and (not ls0 or not rs0 or ls0 == rs0):\n            return 2, 0\n        if max(l0, r0) <= min(l1, r1):\n            return 2, 0\n        return 0, 1\n    if not left and not right:\n        return 1, 0\n    return 0, 1\n\n\ndef _numbers_mismatch(left: str, right: str) -> bool:\n    nums_left = _extract_first_half_numbers(left)\n    nums_right = _extract_first_half_numbers(right)\n    if not nums_left and not nums_right:\n        return False\n    return nums_left != nums_right\n\n\ndef _os_candidate_address(candidate: dict[str, Any]) -> str:\n    return _clean_text(candidate.get("ADDRESS"))\n\n\ndef _os_candidate_road(candidate: dict[str, Any]) -> str:\n    for key in ("STREET_DESCRIPTION", "THOROUGHFARE_NAME"):\n        value = _clean_text(candidate.get(key))\n        if value:\n            return value\n    return _os_candidate_address(candidate)\n\n\ndef _os_candidate_place(candidate: dict[str, Any]) -> str:\n    for key in ("TOWN_NAME", "POST_TOWN", "LOCALITY_NAME", "DEPENDENT_LOCALITY", "DOUBLE_DEPENDENT_LOCALITY"):\n        value = _clean_text(candidate.get(key))\n        if value:\n            return value\n    return ""\n\n\ndef _os_candidate_place_fields(candidate: dict[str, Any]) -> list[str]:\n    fields: list[str] = []\n    seen: set[str] = set()\n    for key in (\n        "DOUBLE_DEPENDENT_LOCALITY",\n        "DEPENDENT_LOCALITY",\n        "LOCALITY_NAME",\n        "TOWN_NAME",\n        "POST_TOWN",\n    ):\n        value = _clean_text(candidate.get(key))\n        compact = _norm_compact(value)\n        if value and compact and compact not in seen:\n            seen.add(compact)\n            fields.append(value)\n    return fields\n\n\ndef _os_candidate_locality_fields(candidate: dict[str, Any]) -> list[str]:\n    fields: list[str] = []\n    seen: set[str] = set()\n    for key in ("DOUBLE_DEPENDENT_LOCALITY", "DEPENDENT_LOCALITY", "LOCALITY_NAME"):\n        value = _clean_text(candidate.get(key))\n        compact = _norm_compact(value)\n        if value and compact and compact not in seen:\n            seen.add(compact)\n            fields.append(value)\n    return fields\n\n\ndef _os_candidate_town_fields(candidate: dict[str, Any]) -> list[str]:\n    fields: list[str] = []\n    seen: set[str] = set()\n    for key in ("TOWN_NAME", "POST_TOWN"):\n        value = _clean_text(candidate.get(key))\n        compact = _norm_compact(value)\n        if value and compact and compact not in seen:\n            seen.add(compact)\n            fields.append(value)\n    return fields\n\n\ndef _best_os_place_field(candidate: dict[str, Any], place_hints: list[str]) -> str:\n    place_fields = _os_candidate_place_fields(candidate)\n    if not place_fields:\n        return _os_candidate_place(candidate)\n    if not place_hints:\n        return place_fields[0]\n    ranked = sorted(\n        place_fields,\n        key=lambda value: (_best_place_score(place_hints, value), -place_fields.index(value)),\n        reverse=True,\n    )\n    return ranked[0]\n\n\ndef _os_candidate_postcode_district(candidate: dict[str, Any]) -> str:\n    postcode = _clean_text(candidate.get("POSTCODE_LOCATOR") or candidate.get("POSTCODE"))\n    match = POSTCODE_DISTRICT_RE.search(postcode.upper())\n    return match.group(1).upper() if match else ""\n\n\ndef _best_os_place_score(candidate: dict[str, Any], place_hints: list[str]) -> float:\n    return max((_best_place_score(place_hints, value) for value in _os_candidate_place_fields(candidate)), default=0.0)\n\n\ndef _best_field_score(hints: list[str], fields: list[str]) -> float:\n    return max((_best_place_score(hints, value) for value in fields), default=0.0)\n\n\ndef _field_exact_match(hints: list[str], fields: list[str]) -> int:\n    hint_keys = {_norm_compact(h) for h in hints if _norm_compact(h)}\n    field_keys = {_norm_compact(f) for f in fields if _norm_compact(f)}\n    return int(bool(hint_keys and field_keys and (hint_keys & field_keys)))\n\n\ndef _os_locality_rank(candidate: dict[str, Any], place_hints: list[str]) -> tuple[int, float]:\n    fields = _os_candidate_locality_fields(candidate)\n    score = _best_field_score(place_hints, fields)\n    if _field_exact_match(place_hints, fields):\n        tier = 3\n    elif score >= 18.0:\n        tier = 2\n    elif score >= 10.0:\n        tier = 1\n    else:\n        tier = 0\n    return tier, round(score, 3)\n\n\ndef _os_town_rank(candidate: dict[str, Any], place_hints: list[str]) -> tuple[int, float]:\n    fields = _os_candidate_town_fields(candidate)\n    score = _best_field_score(place_hints, fields)\n    if _field_exact_match(place_hints, fields):\n        tier = 3\n    elif score >= 18.0:\n        tier = 2\n    elif score >= 10.0:\n        tier = 1\n    else:\n        tier = 0\n    return tier, round(score, 3)\n\n\ndef _place_field_passes_gate(place_hints: list[str], fields: list[str]) -> bool:\n    if not fields:\n        return True\n    if _field_exact_match(place_hints, fields):\n        return True\n    return _best_field_score(place_hints, fields) >= 18.0\n\n\ndef _os_place_hard_gate(candidate: dict[str, Any], place_hints: list[str]) -> bool:\n    if not place_hints:\n        return True\n    locality_fields = _os_candidate_locality_fields(candidate)\n    town_fields = _os_candidate_town_fields(candidate)\n    return _place_field_passes_gate(place_hints, locality_fields) and _place_field_passes_gate(place_hints, town_fields)\n\n\ndef _os_admin_rank(candidate: dict[str, Any], council_name: str) -> int:\n    return int(bool(council_name and _norm(_clean_text(candidate.get("ADMINISTRATIVE_AREA"))) == _norm(council_name)))\n\n\ndef _os_road_rank(candidate: dict[str, Any], road_phrases: list[str]) -> tuple[int, float]:\n    road_name = _os_candidate_road(candidate)\n    if not road_name or not road_phrases:\n        return 0, 0.0\n    road_key = _norm_road_compact(road_name)\n    phrase_keys = [_norm_road_compact(p) for p in road_phrases if _norm_road_compact(p)]\n    if road_key and road_key in phrase_keys:\n        return 4, 1.0\n    best = max((_road_similarity(phrase, road_name) for phrase in road_phrases), default=0.0)\n    if best >= 0.94:\n        tier = 3\n    elif best >= 0.88:\n        tier = 2\n    elif best >= 0.80:\n        tier = 1\n    else:\n        tier = 0\n    return tier, round(best, 3)\n\n\ndef _os_number_rank(candidate: dict[str, Any], corrected_raw_address: str) -> tuple[int, int]:\n    candidate_span = _candidate_number_span(candidate)\n    input_span = _parse_address_number_span(corrected_raw_address)\n    return _number_span_relation(candidate_span, input_span)\n\n\ndef _os_address_rank(\n    candidate: dict[str, Any],\n    corrected_raw_address: str,\n    ) -> tuple[float, float]:\n    addr_score = round(_similarity(corrected_raw_address, _os_candidate_address(candidate)), 3)\n    try:\n        match_val = float(candidate.get("MATCH")) if candidate.get("MATCH") not in (None, "") else 0.0\n    except (TypeError, ValueError):\n        match_val = 0.0\n    return addr_score, round(match_val, 3)\n\n\ndef _os_record_rank(candidate: dict[str, Any]) -> int:\n    return 1 if candidate.get("_record_type") == "DPA" else 0\n\n\ndef _google_component_fields(candidate: dict[str, Any], wanted_types: set[str]) -> list[str]:\n    components = candidate.get("_google_address_components")\n    if not isinstance(components, list):\n        return []\n    fields: list[str] = []\n    seen: set[str] = set()\n    for comp in components:\n        if not isinstance(comp, dict):\n            continue\n        types = comp.get("types")\n        if not isinstance(types, list) or not (wanted_types & set(types)):\n            continue\n        value = _clean_text(comp.get("long_name") or comp.get("short_name"))\n        compact = _norm_compact(value)\n        if value and compact and compact not in seen:\n            seen.add(compact)\n            fields.append(value)\n    return fields\n\n\ndef _google_candidate_address(candidate: dict[str, Any]) -> str:\n    return _clean_text(candidate.get("ADDRESS"))\n\n\ndef _google_candidate_road(candidate: dict[str, Any]) -> str:\n    fields = _google_component_fields(candidate, {"route"})\n    return fields[0] if fields else _google_candidate_address(candidate)\n\n\ndef _google_candidate_locality_fields(candidate: dict[str, Any]) -> list[str]:\n    return _google_component_fields(\n        candidate,\n        {\n            "neighborhood",\n            "sublocality",\n            "sublocality_level_1",\n            "sublocality_level_2",\n            "locality",\n            "postal_town",\n        },\n    )\n\n\ndef _google_candidate_town_fields(candidate: dict[str, Any]) -> list[str]:\n    return _google_component_fields(candidate, {"postal_town", "locality", "administrative_area_level_2"})\n\n\ndef _best_google_place_field(candidate: dict[str, Any], place_hints: list[str]) -> str:\n    place_fields = _google_candidate_locality_fields(candidate) + [\n        f for f in _google_candidate_town_fields(candidate) if _norm_compact(f) not in {_norm_compact(x) for x in _google_candidate_locality_fields(candidate)}\n    ]\n    if not place_fields:\n        return ""\n    if not place_hints:\n        return place_fields[0]\n    ranked = sorted(\n        place_fields,\n        key=lambda value: (_best_place_score(place_hints, value), -place_fields.index(value)),\n        reverse=True,\n    )\n    return ranked[0]\n\n\ndef _google_place_hard_gate(candidate: dict[str, Any], place_hints: list[str]) -> bool:\n    if not place_hints:\n        return True\n    locality_fields = _google_candidate_locality_fields(candidate)\n    town_fields = _google_candidate_town_fields(candidate)\n    return _place_field_passes_gate(place_hints, locality_fields) and _place_field_passes_gate(place_hints, town_fields)\n\n\ndef _google_locality_rank(candidate: dict[str, Any], place_hints: list[str]) -> tuple[int, float]:\n    fields = _google_candidate_locality_fields(candidate)\n    score = _best_field_score(place_hints, fields)\n    if _field_exact_match(place_hints, fields):\n        tier = 3\n    elif score >= 18.0:\n        tier = 2\n    elif score >= 10.0:\n        tier = 1\n    else:\n        tier = 0\n    return tier, round(score, 3)\n\n\ndef _google_town_rank(candidate: dict[str, Any], place_hints: list[str]) -> tuple[int, float]:\n    fields = _google_candidate_town_fields(candidate)\n    score = _best_field_score(place_hints, fields)\n    if _field_exact_match(place_hints, fields):\n        tier = 3\n    elif score >= 18.0:\n        tier = 2\n    elif score >= 10.0:\n        tier = 1\n    else:\n        tier = 0\n    return tier, round(score, 3)\n\n\ndef _google_admin_rank(candidate: dict[str, Any], council_name: str) -> int:\n    fields = _google_component_fields(candidate, {"administrative_area_level_2"})\n    if not fields:\n        fields = _google_component_fields(candidate, {"administrative_area_level_1"})\n    return int(bool(council_name and _field_exact_match([council_name], fields)))\n\n\ndef _google_road_rank(candidate: dict[str, Any], road_phrases: list[str]) -> tuple[int, float]:\n    road_name = _google_candidate_road(candidate)\n    if not road_name or not road_phrases:\n        return 0, 0.0\n    road_key = _norm_road_compact(road_name)\n    phrase_keys = [_norm_road_compact(p) for p in road_phrases if _norm_road_compact(p)]\n    if road_key and road_key in phrase_keys:\n        return 4, 1.0\n    best = max((_road_similarity(phrase, road_name) for phrase in road_phrases), default=0.0)\n    if best >= 0.94:\n        tier = 3\n    elif best >= 0.88:\n        tier = 2\n    elif best >= 0.80:\n        tier = 1\n    else:\n        tier = 0\n    return tier, round(best, 3)\n\n\ndef _google_number_rank(candidate: dict[str, Any], corrected_raw_address: str) -> tuple[int, int]:\n    return _number_span_relation(_parse_address_number_span(_google_candidate_address(candidate)), _parse_address_number_span(corrected_raw_address))\n\n\ndef _google_address_rank(candidate: dict[str, Any], corrected_raw_address: str) -> tuple[float, float]:\n    addr_score = round(_similarity(corrected_raw_address, _google_candidate_address(candidate)), 3)\n    match_val = _to_float(candidate.get("MATCH")) or 0.0\n    return addr_score, round(match_val, 3)\n\n\ndef _google_record_rank(candidate: dict[str, Any]) -> int:\n    location_type = _clean_text(candidate.get("_google_location_type")).upper()\n    return 1 if location_type == "ROOFTOP" else 0\n\n\ndef _rank_google_candidate(\n    candidate: dict[str, Any],\n    corrected_raw_address: str,\n    road_phrases: list[str],\n    place_hints: list[str],\n    council_name: str,\n) -> tuple[tuple[int, int, int, int, int, int, float, float, float, float, float], dict[str, Any]]:\n    if not _google_place_hard_gate(candidate, place_hints):\n        debug = {\n            "locality_tier": 0,\n            "locality_score": 0.0,\n            "town_tier": 0,\n            "town_score": 0.0,\n            "admin_tier": 0,\n            "road_tier": 0,\n            "road_score": 0.0,\n            "number_tier": 0,\n            "number_mismatch": 1,\n            "record_tier": 0,\n            "address_score": 0.0,\n            "match_score": 0.0,\n            "place_gate_failed": 1,\n        }\n        return (-1, -1, -1, -1, -1, -1, -1.0, -1.0, -1.0, -1.0, -1.0), debug\n    locality_tier, locality_score = _google_locality_rank(candidate, place_hints)\n    town_tier, town_score = _google_town_rank(candidate, place_hints)\n    admin_tier = _google_admin_rank(candidate, council_name)\n    road_tier, road_score = _google_road_rank(candidate, road_phrases)\n    number_tier, number_mismatch = _google_number_rank(candidate, corrected_raw_address)\n    address_score, match_score = _google_address_rank(candidate, corrected_raw_address)\n    record_tier = _google_record_rank(candidate)\n    rank = (\n        road_tier,\n        number_tier,\n        locality_tier,\n        town_tier,\n        admin_tier,\n        record_tier,\n        locality_score,\n        town_score,\n        road_score,\n        address_score,\n        match_score,\n    )\n    debug = {\n        "locality_tier": locality_tier,\n        "locality_score": locality_score,\n        "town_tier": town_tier,\n        "town_score": town_score,\n        "admin_tier": admin_tier,\n        "road_tier": road_tier,\n        "road_score": road_score,\n        "number_tier": number_tier,\n        "number_mismatch": number_mismatch,\n        "record_tier": record_tier,\n        "address_score": address_score,\n        "match_score": match_score,\n    }\n    return rank, debug\n\n\ndef _serialize_google_candidate(\n    candidate: dict[str, Any],\n    rank: tuple[int, int, int, int, int, int, float, float, float, float, float],\n    rank_debug: dict[str, Any],\n    place_hints: list[str],\n) -> dict[str, Any]:\n    return {\n        "address": _google_candidate_address(candidate),\n        "road_name": _google_candidate_road(candidate),\n        "place": _best_google_place_field(candidate, place_hints),\n        "postcode_district": _extract_postcode_from_google_components(candidate.get("_google_address_components")),\n        "easting_27700": _to_float(candidate.get("X_COORDINATE")),\n        "northing_27700": _to_float(candidate.get("Y_COORDINATE")),\n        "match": _to_float(candidate.get("MATCH")),\n        "record_type": "GOOGLE",\n        "rank_tuple": list(rank),\n        **rank_debug,\n    }\n\n\ndef _compute_google_confidence(\n    raw_address: str,\n    best: RoadMatch | None,\n    candidates: list[RoadMatch],\n    google_ranked: list[tuple[tuple[int, int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]],\n    google_best: dict[str, Any] | None,\n    google_best_rank_debug: dict[str, Any] | None,\n) -> tuple[int, str, str]:\n    if not google_best or not google_best_rank_debug:\n        return 0, "very_low", "no_google_candidate"\n\n    score = 0.0\n    reasons: list[str] = []\n    road_tier = int(google_best_rank_debug.get("road_tier", 0))\n    locality_tier = int(google_best_rank_debug.get("locality_tier", 0))\n    town_tier = int(google_best_rank_debug.get("town_tier", 0))\n    admin_tier = int(google_best_rank_debug.get("admin_tier", 0))\n    number_tier = int(google_best_rank_debug.get("number_tier", 0))\n    number_mismatch = int(google_best_rank_debug.get("number_mismatch", 0))\n    record_tier = int(google_best_rank_debug.get("record_tier", 0))\n    road_score = float(google_best_rank_debug.get("road_score", 0.0))\n    has_input_number = bool(_extract_first_half_numbers(raw_address))\n    has_anchor_risk = _raw_address_has_anchor_risk(raw_address)\n    google_types = google_best.get("_google_types") if isinstance(google_best.get("_google_types"), list) else []\n    google_location_type = _clean_text(google_best.get("_google_location_type")).upper()\n    google_partial_match = bool(google_best.get("_google_partial_match"))\n    google_best_road_name = _google_candidate_road(google_best)\n    raw_pre_road = _extract_pre_road_label(raw_address, google_best_road_name)\n    best_pre_road = _extract_pre_road_label(_google_candidate_address(google_best), google_best_road_name)\n\n    score += {4: 40.0, 3: 30.0, 2: 20.0, 1: 10.0}.get(road_tier, 0.0)\n    reasons.append("road_exact" if road_tier >= 4 else "road_strong" if road_tier >= 3 else "road_weak")\n    score += {3: 15.0, 2: 11.0, 1: 5.0}.get(locality_tier, 0.0)\n    score += {3: 10.0, 2: 7.0, 1: 3.0}.get(town_tier, 0.0)\n    score += 3.0 if admin_tier else 0.0\n    if locality_tier >= 2:\n        reasons.append("locality_match")\n    elif town_tier >= 2:\n        reasons.append("town_match")\n    elif locality_tier or town_tier or admin_tier:\n        reasons.append("place_partial")\n    else:\n        reasons.append("place_weak")\n    if number_tier >= 4:\n        score += 24.0\n        reasons.append("number_range_exact")\n    elif number_tier == 3:\n        score += 20.0\n        reasons.append("number_match")\n    elif number_tier == 2:\n        score += 12.0\n        reasons.append("number_in_range")\n    elif number_tier == 1:\n        score += 8.0\n        reasons.append("number_neutral")\n    else:\n        reasons.append("number_weak")\n    if number_mismatch:\n        score -= 8.0\n        reasons.append("number_mismatch")\n    if has_input_number and road_tier >= 4 and number_tier >= 3 and (locality_tier >= 2 or town_tier >= 2):\n        score += 8.0\n        reasons.append("numbered_address_bonus")\n    elif not has_input_number:\n        reasons.append("no_input_number")\n    if record_tier:\n        score += 4.0\n        reasons.append("rooftop")\n    if google_partial_match:\n        score -= 10.0\n        reasons.append("partial_match")\n    if google_location_type == "GEOMETRIC_CENTER":\n        score -= 8.0\n        reasons.append("geometric_center")\n    elif google_location_type == "APPROXIMATE":\n        score -= 14.0\n        reasons.append("approximate")\n    elif google_location_type == "RANGE_INTERPOLATED":\n        score -= 3.0\n        reasons.append("range_interpolated")\n    if not has_input_number and not has_anchor_risk and raw_pre_road and best_pre_road:\n        site_sim = _similarity(raw_pre_road, best_pre_road)\n        if site_sim >= 0.85:\n            score += 6.0\n            reasons.append("site_name_match")\n        elif site_sim < 0.55:\n            score -= 18.0\n            reasons.append("site_name_mismatch")\n    lexicon_strong = _lexicon_guard_eligible(best, candidates)\n    if lexicon_strong and best:\n        lex_road_key = _norm_road_compact(best.road_name)\n        same_road = [\n            (rank_debug, candidate)\n            for _, rank_debug, candidate in google_ranked\n            if _norm_road_compact(_google_candidate_road(candidate)) == lex_road_key\n        ]\n        if not same_road:\n            score = min(score, 35.0)\n            reasons.append("no_same_road_against_strong_lexicon")\n        else:\n            score += 5.0\n            reasons.append("same_road_with_lexicon")\n    current_proxy = _candidate_confidence_proxy(google_best_rank_debug)\n    next_proxy = None\n    if len(google_ranked) >= 2:\n        next_proxy = _candidate_confidence_proxy(google_ranked[1][1])\n    if next_proxy is None:\n        score += 10.0\n        reasons.append("single_winner")\n    else:\n        gap = current_proxy - next_proxy\n        if gap >= 8.0:\n            score += 10.0\n            reasons.append("strong_margin")\n        elif gap >= 4.0:\n            score += 7.0\n            reasons.append("medium_margin")\n        elif gap >= 2.0:\n            score += 4.0\n            reasons.append("small_margin")\n        else:\n            score += 1.0\n            reasons.append("weak_margin")\n    if road_tier == 0 and road_score < 0.8:\n        score = min(score, 30.0)\n        reasons.append("wrong_road_risk")\n    if not has_input_number and google_types == ["route"]:\n        score = min(score, 68.0)\n        reasons.append("route_only_without_number")\n    if has_anchor_risk:\n        score -= 12.0\n        reasons.append("anchor_style_address")\n        if not has_input_number:\n            score -= 6.0\n            reasons.append("anchor_without_number")\n    final_score = max(0, min(100, int(round(score))))\n    return final_score, _confidence_band(final_score), " + ".join(reasons)\n\n\ndef _compute_gemini_confidence(\n    raw_address: str,\n    best: RoadMatch | None,\n    candidates: list[RoadMatch],\n    gemini_result: dict[str, Any] | None,\n    place_hints: list[str],\n    road_phrases: list[str],\n) -> tuple[int, str, str]:\n    if not gemini_result or gemini_result.get("status") != "matched":\n        return 0, "very_low", "no_gemini_candidate"\n    matched_address = _clean_text(gemini_result.get("matched_address"))\n    easting = _to_float(gemini_result.get("easting_27700"))\n    northing = _to_float(gemini_result.get("northing_27700"))\n    if easting is None or northing is None:\n        return 0, "very_low", "gemini_missing_coords"\n    score = float(gemini_result.get("confidence") or 0)\n    reasons: list[str] = []\n    road_score = 0.0\n    if matched_address:\n        road_score = max((_road_similarity(phrase, matched_address) for phrase in road_phrases), default=0.0)\n    if road_score >= 0.95:\n        reasons.append("road_exact")\n        score += 10.0\n    elif road_score >= 0.82:\n        reasons.append("road_strong")\n        score += 4.0\n    else:\n        reasons.append("road_weak")\n        score -= 18.0\n    place_hit = False\n    for hint in place_hints:\n        if _norm_compact(hint) and _norm_compact(hint) in _norm_compact(matched_address):\n            place_hit = True\n            break\n    if place_hit:\n        reasons.append("place_match")\n        score += 6.0\n    else:\n        reasons.append("place_weak")\n        score -= 8.0\n    number_tier, number_mismatch = _number_span_relation(_parse_address_number_span(matched_address), _parse_address_number_span(raw_address))\n    if number_tier >= 4:\n        reasons.append("number_range_exact")\n        score += 10.0\n    elif number_tier == 3:\n        reasons.append("number_match")\n        score += 8.0\n    elif number_tier == 2:\n        reasons.append("number_in_range")\n        score += 3.0\n    elif number_tier == 1:\n        reasons.append("number_neutral")\n    else:\n        reasons.append("number_weak")\n        if _extract_first_half_numbers(raw_address):\n            score -= 10.0\n    if number_mismatch:\n        reasons.append("number_mismatch")\n        score -= 8.0\n    if best and _lexicon_guard_eligible(best, candidates):\n        if _norm_road_compact(best.road_name) and _norm_road_compact(best.road_name) not in _norm_road_compact(matched_address):\n            score = min(score, 35.0)\n            reasons.append("no_same_road_against_strong_lexicon")\n        else:\n            reasons.append("same_road_with_lexicon")\n            score += 4.0\n    if _raw_address_has_anchor_risk(raw_address):\n        reasons.append("anchor_style_address")\n        score -= 10.0\n    final_score = max(0, min(100, int(round(score))))\n    return final_score, _confidence_band(final_score), " + ".join(reasons)\n\n\ndef _rank_os_candidate(\n    candidate: dict[str, Any],\n    corrected_raw_address: str,\n    road_phrases: list[str],\n    place_hints: list[str],\n    council_name: str,\n) -> tuple[tuple[int, int, int, int, int, float, float, float, float, float], dict[str, Any]]:\n    if not _os_place_hard_gate(candidate, place_hints):\n        debug = {\n            "locality_tier": 0,\n            "locality_score": 0.0,\n            "town_tier": 0,\n            "town_score": 0.0,\n            "admin_tier": 0,\n            "road_tier": 0,\n            "road_score": 0.0,\n            "number_tier": 0,\n            "number_mismatch": 1,\n            "record_tier": 0,\n            "address_score": 0.0,\n            "match_score": 0.0,\n            "place_gate_failed": 1,\n        }\n        return (-1, -1, -1, -1, -1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), debug\n\n    locality_tier, locality_score = _os_locality_rank(candidate, place_hints)\n    town_tier, town_score = _os_town_rank(candidate, place_hints)\n    admin_tier = _os_admin_rank(candidate, council_name)\n    road_tier, road_score = _os_road_rank(candidate, road_phrases)\n    number_tier, number_mismatch = _os_number_rank(candidate, corrected_raw_address)\n    address_score, match_score = _os_address_rank(candidate, corrected_raw_address)\n    record_tier = _os_record_rank(candidate)\n    rank = (\n        road_tier,\n        number_tier,\n        locality_tier,\n        town_tier,\n        admin_tier,\n        record_tier,\n        locality_score,\n        town_score,\n        road_score,\n        address_score,\n        match_score,\n    )\n    debug = {\n        "locality_tier": locality_tier,\n        "locality_score": locality_score,\n        "town_tier": town_tier,\n        "town_score": town_score,\n        "admin_tier": admin_tier,\n        "road_tier": road_tier,\n        "road_score": road_score,\n        "number_tier": number_tier,\n        "number_mismatch": number_mismatch,\n        "record_tier": record_tier,\n        "address_score": address_score,\n        "match_score": match_score,\n    }\n    return rank, debug\n\n\ndef _serialize_os_candidate(\n    candidate: dict[str, Any],\n    rank: tuple[int, int, int, int, int, float, float, float, float, float],\n    rank_debug: dict[str, Any],\n    place_hints: list[str],\n) -> dict[str, Any]:\n    return {\n        "address": _os_candidate_address(candidate),\n        "road_name": _os_candidate_road(candidate),\n        "place": _best_os_place_field(candidate, place_hints),\n        "place_fields": _os_candidate_place_fields(candidate),\n        "postcode_district": _os_candidate_postcode_district(candidate),\n        "easting_27700": _to_float(candidate.get("X_COORDINATE")),\n        "northing_27700": _to_float(candidate.get("Y_COORDINATE")),\n        "match": _to_float(candidate.get("MATCH")),\n        "record_type": _clean_text(candidate.get("_record_type")),\n        "uprn": _clean_text(candidate.get("UPRN")),\n        "rank_tuple": list(rank),\n        **rank_debug,\n    }\n\n\ndef _valid_os_ranked(\n    os_ranked: list[tuple[tuple[int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]],\n) -> list[tuple[tuple[int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]]:\n    return [item for item in os_ranked if not item[1].get("place_gate_failed")]\n\n\ndef _strip_noise(text: str) -> str:\n    clean = _clean_text(text)\n    if not clean:\n        return ""\n    changed = True\n    while changed:\n        changed = False\n        for pattern in LEADING_NOISE_PATTERNS:\n            newer = re.sub(pattern, "", clean, flags=re.IGNORECASE).strip(" ,.")\n            if newer != clean:\n                clean = newer\n                changed = True\n    return clean\n\n\ndef _road_like_suffix(text: str) -> bool:\n    words = _norm(text).split()\n    return bool(words) and (\n        words[-1] in ROAD_SUFFIXES\n        or (len(words) >= 2 and words[-2] in ROAD_SUFFIXES and words[-1] in TRAILING_ROAD_DIRECTION_TOKENS)\n    )\n\n\ndef _phrase_variants(segment: str) -> list[str]:\n    clean = _strip_noise(segment)\n    if not clean:\n        return []\n    variants = [clean]\n    without_num = LEADING_NUMBER_RE.sub("", clean).strip()\n    if without_num and without_num != clean:\n        variants.append(without_num)\n    for source in [clean, without_num]:\n        if not source:\n            continue\n        for match in TRAILING_ROAD_AFTER_PREP_RE.finditer(source.upper()):\n            start, end = match.span(1)\n            extracted = source[start:end].strip(" ,.-")\n            if extracted:\n                variants.append(extracted)\n        for match in ROAD_WITH_DIR_RE.finditer(source.upper()):\n            start, end = match.span(1)\n            extracted = source[start:end].strip(" ,.-")\n            if extracted:\n                variants.append(extracted)\n    words = clean.split()\n    has_complete_phrase = bool(without_num and _road_like_suffix(without_num))\n    if not has_complete_phrase:\n        for start in range(len(words)):\n            candidate = " ".join(words[start:]).strip()\n            if len(candidate.split()) >= 2 and _road_like_suffix(candidate):\n                variants.append(candidate)\n    out: list[str] = []\n    seen: set[str] = set()\n    for value in variants:\n        value = _clean_text(value)\n        key = _norm_compact(value)\n        if not value or not key or key in seen:\n            continue\n        seen.add(key)\n        out.append(value)\n    return out\n\n\ndef _phrase_variants_stage1(segment: str) -> list[str]:\n    clean = _strip_noise(segment)\n    if not clean:\n        return []\n    variants = [clean]\n    without_num = LEADING_NUMBER_RE.sub("", clean).strip()\n    if without_num and without_num != clean:\n        variants.append(without_num)\n    words = clean.split()\n    for start in range(len(words)):\n        candidate = " ".join(words[start:]).strip()\n        if len(candidate.split()) >= 2 and _road_like_suffix(candidate):\n            variants.append(candidate)\n    out: list[str] = []\n    seen: set[str] = set()\n    for value in variants:\n        value = _clean_text(value)\n        key = _norm_compact(value)\n        if not value or not key or key in seen:\n            continue\n        seen.add(key)\n        out.append(value)\n    return out\n\n\ndef _fallback_simplified_road_phrases(phrase: str) -> list[str]:\n    words = _norm(phrase).split()\n    if len(words) < 3:\n        return []\n    candidates: list[str] = []\n    if words[-1] in ROAD_SUFFIXES and words[0] in LEADING_ROAD_QUALIFIER_TOKENS and len(words) >= 4:\n        candidates.append(" ".join(words[1:]).title())\n    if len(words) >= 4 and words[-2] in ROAD_SUFFIXES and words[-1] in TRAILING_ROAD_DIRECTION_TOKENS:\n        candidates.append(" ".join(words[:-1]).title())\n    out: list[str] = []\n    seen: set[str] = set()\n    for candidate in candidates:\n        clean = _clean_text(candidate)\n        key = _norm_compact(clean)\n        if clean and key and key not in seen:\n            seen.add(key)\n            out.append(clean)\n    return out\n\n\ndef _extract_road_phrases_core(\n    raw_address: str,\n    place_names: Iterable[str],\n    variant_fn,\n) -> tuple[list[str], dict[str, Any]]:\n    raw = _clean_text(raw_address)\n    if not raw:\n        return [], {"segments": [], "expanded_segments": []}\n\n    base_segments = [_clean_text(part) for part in raw.split(",") if _clean_text(part)]\n    segments: list[str] = []\n    for segment in base_segments:\n        if segments and _norm(segment) in ROAD_SUFFIXES:\n            segments[-1] = _clean_text(f"{segments[-1]} {segment}")\n        else:\n            segments.append(segment)\n    expanded_segments: list[str] = []\n    for segment in segments:\n        core = _strip_noise(segment)\n        if not core:\n            continue\n        expanded_segments.append(core)\n        if re.search(r"\\b(?:junction|corner)\\b", core, re.IGNORECASE):\n            stripped = re.sub(r"(?i)\\b(?:junction|corner|of|the)\\b", " ", core)\n            for piece in SPLIT_CONNECTORS_RE.split(stripped):\n                piece = _clean_text(piece)\n                if piece:\n                    expanded_segments.append(piece)\n        elif re.search(r"[&/]", core):\n            for piece in SPLIT_CONNECTORS_RE.split(core):\n                piece = _clean_text(piece)\n                if piece:\n                    expanded_segments.append(piece)\n\n    phrases: list[str] = []\n    seen: set[str] = set()\n    place_norms = {_norm_compact(place) for place in place_names}\n    road_like_segments: list[str] = []\n    for segment in expanded_segments:\n        for phrase in variant_fn(segment):\n            key = _norm_compact(phrase)\n            if key in place_norms:\n                continue\n            if key in seen:\n                continue\n            seen.add(key)\n            phrases.append(phrase)\n            if _norm_compact(_clean_text(segment)) == key:\n                road_like_segments.append(phrase)\n\n    return phrases, {"segments": segments, "expanded_segments": expanded_segments, "road_like_segments": road_like_segments}\n\n\ndef _extract_road_phrases_stage1(raw_address: str, place_names: Iterable[str]) -> tuple[list[str], dict[str, Any]]:\n    return _extract_road_phrases_core(raw_address, place_names, _phrase_variants_stage1)\n\n\ndef _extract_road_phrases(raw_address: str, place_names: Iterable[str]) -> tuple[list[str], dict[str, Any]]:\n    return _extract_road_phrases_core(raw_address, place_names, _phrase_variants)\n\n\ndef _read_rows(input_csv: str, row_limit: int | None) -> list[dict[str, Any]]:\n    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as f:\n        reader = csv.DictReader(f)\n        rows = list(reader)\n    if row_limit is not None:\n        return rows[:row_limit]\n    return rows\n\n\ndef _expand_main_rows(rows: list[dict[str, Any]], address_column: str) -> list[dict[str, Any]]:\n    """Keep original rows and append expanded house-number variants.\n\n    Example:\n    unique_key=6, chargegeog="11-15 Ratcliffe Gate, ..."\n    ->\n    original row retained\n    plus 6_1..6_5 with 11..15 Ratcliffe Gate, ...\n    """\n    expanded: list[dict[str, Any]] = []\n    for row in rows:\n        base_row = dict(row)\n        base_row["_is_expanded_case"] = "no"\n        base_row["_source_unique_key"] = str(row.get("unique_key") or "").strip()\n        base_row["_expanded_from"] = ""\n        base_row["_expanded_variant"] = ""\n        expanded.append(base_row)\n\n        raw_address = _clean_text(row.get(address_column))\n        if not raw_address:\n            continue\n        match = LEADING_HOUSE_RANGE_RE.match(raw_address)\n        if not match:\n            continue\n        start = int(match.group(1))\n        end = int(match.group(2))\n        suffix = (match.group(3) or "").upper()\n        if suffix:\n            continue\n        if end < start:\n            start, end = end, start\n        span = end - start + 1\n        if span < 2 or span > 10:\n            continue\n        remainder = raw_address[match.end():].lstrip(" ,")\n        base_key = str(row.get("unique_key") or "").strip()\n        for offset, number in enumerate(range(start, end + 1), start=1):\n            child = dict(row)\n            child[address_column] = f"{number} {remainder}" if remainder else str(number)\n            if base_key:\n                child["unique_key"] = f"{base_key}_{offset}"\n            child["_is_expanded_case"] = "yes"\n            child["_source_unique_key"] = base_key\n            child["_expanded_from"] = raw_address\n            child["_expanded_variant"] = str(number)\n            expanded.append(child)\n    return expanded\n\n\ndef _infer_input_gpkg_path(input_csv: str) -> Path | None:\n    csv_path = Path(input_csv)\n    if not csv_path.suffix.lower() == ".csv":\n        return None\n    gpkg_path = csv_path.with_suffix(".gpkg")\n    if gpkg_path.exists():\n        return gpkg_path\n    stem = csv_path.stem\n    search_dirs = []\n    for parent in [csv_path.parent, *csv_path.parents]:\n        if parent not in search_dirs:\n            search_dirs.append(parent)\n        if len(search_dirs) >= 4:\n            break\n    candidates: list[Path] = []\n    for directory in search_dirs:\n        try:\n            candidates.extend(sorted(directory.glob("*.gpkg")))\n        except OSError:\n            continue\n    if not candidates:\n        return None\n    prefix_matches = [p for p in candidates if stem.startswith(p.stem)]\n    if prefix_matches:\n        prefix_matches.sort(key=lambda p: len(p.stem), reverse=True)\n        return prefix_matches[0]\n    return None\n\n\ndef _load_polygon_geometries(input_csv: str) -> dict[str, Any]:\n    gpkg_path = _infer_input_gpkg_path(input_csv)\n    if gpkg_path is None:\n        return {}\n    try:\n        import geopandas as gpd  # type: ignore\n    except Exception:\n        return {}\n    try:\n        layer_name = gpkg_path.stem\n        gdf = gpd.read_file(gpkg_path, layer=layer_name)\n    except Exception:\n        return {}\n    out: dict[str, Any] = {}\n    for _, row in gdf.iterrows():\n        out[str(row.get("unique_key") or "").strip()] = row.geometry\n    return out\n\n\ndef _distance_to_polygon_m(geometry_by_key: dict[str, Any], source_key: str, easting: Any, northing: Any) -> str:\n    geom = geometry_by_key.get(source_key)\n    if geom is None or easting in ("", None) or northing in ("", None):\n        return ""\n    try:\n        from shapely.geometry import Point  # type: ignore\n    except Exception:\n        return ""\n    try:\n        return str(round(geom.distance(Point(float(easting), float(northing))), 1))\n    except Exception:\n        return ""\n\n\ndef _xlsx_col_name(index: int) -> str:\n    out = ""\n    n = index\n    while n > 0:\n        n, rem = divmod(n - 1, 26)\n        out = chr(65 + rem) + out\n    return out\n\n\ndef _write_summary_xlsx(path: Path, rows: list[dict[str, Any]], input_csv: str) -> None:\n    geometry_by_key = _load_polygon_geometries(input_csv)\n    headers = [\n        "key",\n        "original_address",\n        "lexicon_road",\n        "lexicon_easting_27700",\n        "lexicon_northing_27700",\n        "lexicon_distance_to_polygon_m",\n        "os_query",\n        "os_best_address",\n        "os_best_distance_to_polygon_m",\n        "os_confidence_score",\n        "gog_best_address",\n        "gog_best_distance_to_polygon_m",\n        "gog_confidence_score",\n        "best_anchor_source",\n        "distance_to_polygon_m",\n    ]\n    widths = [12, 56, 32, 16, 16, 22, 58, 62, 22, 18, 62, 22, 18, 18, 20]\n\n    sheet_rows: list[list[str]] = [headers]\n    for row in rows:\n        key = str(row.get("unique_key") or "").strip()\n        source_key = str(row.get("_source_unique_key") or key).strip()\n        dist = _distance_to_polygon_m(\n            geometry_by_key,\n            source_key,\n            row.get("best_anchor_easting_27700"),\n            row.get("best_anchor_northing_27700"),\n        )\n        lex_dist = _distance_to_polygon_m(\n            geometry_by_key,\n            source_key,\n            row.get("best_road_anchor_easting_27700"),\n            row.get("best_road_anchor_northing_27700"),\n        )\n        os_dist = _distance_to_polygon_m(\n            geometry_by_key,\n            source_key,\n            row.get("os_best_easting_27700"),\n            row.get("os_best_northing_27700"),\n        )\n        gog_dist = _distance_to_polygon_m(\n            geometry_by_key,\n            source_key,\n            row.get("gog_best_easting_27700"),\n            row.get("gog_best_northing_27700"),\n        )\n        sheet_rows.append(\n            [\n                key,\n                _clean_text(row.get("chargegeog")),\n                " | ".join(\n                    part\n                    for part in [\n                        _clean_text(row.get("best_road_anchor_name")),\n                        _clean_text(row.get("best_road_anchor_place")),\n                    ]\n                    if part\n                ),\n                str(row.get("best_road_anchor_easting_27700") or ""),\n                str(row.get("best_road_anchor_northing_27700") or ""),\n                lex_dist,\n                _clean_text(row.get("os_query")),\n                _clean_text(row.get("os_best_address")) or _clean_text(row.get("best_anchor_name")),\n                os_dist,\n                str(row.get("os_confidence_score") or ""),\n                _clean_text(row.get("gog_best_address")),\n                gog_dist,\n                str(row.get("gog_confidence_score") or ""),\n                _clean_text(row.get("best_anchor_source")),\n                dist,\n            ]\n        )\n\n    def cell_xml(value: str) -> str:\n        text = xml_escape(str(value or ""))\n        return f\'<c t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>\'\n\n    rows_xml: list[str] = []\n    for idx, row_values in enumerate(sheet_rows, start=1):\n        cells = "".join(f\'<c r="{_xlsx_col_name(col_idx)}{idx}" t="inlineStr"><is><t xml:space="preserve">{xml_escape(str(value or ""))}</t></is></c>\' for col_idx, value in enumerate(row_values, start=1))\n        rows_xml.append(f\'<row r="{idx}">{cells}</row>\')\n    sheet_xml = (\n        \'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\'\n        \'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\'\n        \'<sheetViews><sheetView workbookViewId="0"/></sheetViews>\'\n        \'<sheetFormatPr defaultRowHeight="15"/>\'\n        \'<cols>\'\n        + "".join(\n            f\'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>\'\n            for i, width in enumerate(widths, start=1)\n        )\n        + \'</cols>\'\n        \'<sheetData>\'\n        + "".join(rows_xml)\n        + \'</sheetData>\'\n        \'</worksheet>\'\n    )\n    workbook_xml = (\n        \'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\'\n        \'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" \'\n        \'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\'\n        \'<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>\'\n        \'</workbook>\'\n    )\n    workbook_rels_xml = (\n        \'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\'\n        \'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\'\n        \'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>\'\n        \'</Relationships>\'\n    )\n    rels_xml = (\n        \'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\'\n        \'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\'\n        \'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\'\n        \'</Relationships>\'\n    )\n    content_types_xml = (\n        \'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\'\n        \'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\'\n        \'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\'\n        \'<Default Extension="xml" ContentType="application/xml"/>\'\n        \'<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\'\n        \'<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\'\n        \'</Types>\'\n    )\n\n    path.parent.mkdir(parents=True, exist_ok=True)\n    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:\n        zf.writestr("[Content_Types].xml", content_types_xml)\n        zf.writestr("_rels/.rels", rels_xml)\n        zf.writestr("xl/workbook.xml", workbook_xml)\n        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)\n        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)\n\n\ndef _serialize_match(match: RoadMatch) -> dict[str, Any]:\n    return {\n        "road_name": match.road_name,\n        "populated_place": match.populated_place,\n        "district_borough": match.district_borough,\n        "county_unitary": match.county_unitary,\n        "postcode_district": match.postcode_district,\n        "geometry_x": match.geometry_x,\n        "geometry_y": match.geometry_y,\n        "local_type": match.local_type,\n        "matched_phrase": match.matched_phrase,\n        "match_type": match.match_type,\n        "score": match.score,\n        "source_tile": match.source_tile,\n    }\n\n\ndef _lexicon_guard_eligible(best: RoadMatch | None, candidates: list[RoadMatch]) -> bool:\n    return bool(\n        best\n        and best.match_type == "exact"\n        and _anchor_confidence(best, candidates) == "high"\n        and _clean_text(best.road_name)\n    )\n\n\ndef _same_road_guard_rank(\n    rank_debug: dict[str, Any],\n) -> tuple[int, int, int, int, int, float, float, float, float]:\n    return (\n        int(rank_debug.get("locality_tier", 0)),\n        int(rank_debug.get("town_tier", 0)),\n        int(rank_debug.get("admin_tier", 0)),\n        int(rank_debug.get("number_tier", 0)),\n        int(rank_debug.get("record_tier", 0)),\n        float(rank_debug.get("locality_score", 0.0)),\n        float(rank_debug.get("town_score", 0.0)),\n        float(rank_debug.get("address_score", 0.0)),\n        float(rank_debug.get("match_score", 0.0)),\n    )\n\n\ndef _same_road_guard_place_ok(rank_debug: dict[str, Any]) -> bool:\n    locality_tier = int(rank_debug.get("locality_tier", 0))\n    town_tier = int(rank_debug.get("town_tier", 0))\n    admin_tier = int(rank_debug.get("admin_tier", 0))\n    return locality_tier >= 1 or town_tier >= 1 or admin_tier >= 1\n\n\ndef _select_guarded_anchor(\n    best: RoadMatch | None,\n    candidates: list[RoadMatch],\n    os_ranked: list[tuple[tuple[int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]],\n) -> tuple[str, str, RoadMatch | None, tuple[int, ...] | None, dict[str, Any] | None, dict[str, Any] | None]:\n    raw_os_best = os_ranked[0][2] if os_ranked else None\n    raw_os_rank = os_ranked[0][0] if os_ranked else None\n    raw_os_debug = os_ranked[0][1] if os_ranked else None\n    if not raw_os_best and not best:\n        return "none", "none", None, None, None, None\n    if not _lexicon_guard_eligible(best, candidates):\n        if raw_os_best:\n            return "os", "os_default", None, raw_os_rank, raw_os_debug, raw_os_best\n        return "lexicon", "lexicon_only", best, None, None, None\n\n    lex_road_key = _norm_road_compact(best.road_name)\n    same_road: list[tuple[tuple[int, int, int, int, int, float, float, float, float], dict[str, Any], dict[str, Any]]] = []\n    for rank_tuple, rank_debug, candidate in os_ranked:\n        if _norm_road_compact(_os_candidate_road(candidate)) == lex_road_key:\n            same_road.append((_same_road_guard_rank(rank_debug), rank_debug, candidate))\n\n    if not same_road:\n        return "lexicon", "lexicon_guard_no_same_road", best, None, None, None\n\n    same_road_place_ok = [item for item in same_road if _same_road_guard_place_ok(item[1])]\n    if not same_road_place_ok:\n        return "lexicon", "lexicon_guard_same_road_place_weak", best, None, None, None\n\n    same_road_place_ok.sort(key=lambda item: item[0], reverse=True)\n    guard_rank, guard_debug, guard_candidate = same_road_place_ok[0]\n    return "os", "os_same_road_guard", None, guard_rank, guard_debug, guard_candidate\n\n\ndef _candidate_confidence_proxy(rank_debug: dict[str, Any]) -> float:\n    return round(\n        float(rank_debug.get("road_tier", 0)) * 10.0\n        + float(rank_debug.get("number_tier", 0)) * 6.0\n        + float(rank_debug.get("locality_tier", 0)) * 4.0\n        + float(rank_debug.get("town_tier", 0)) * 3.0\n        + float(rank_debug.get("admin_tier", 0)) * 1.5\n        + float(rank_debug.get("record_tier", 0)) * 1.0\n        + float(rank_debug.get("road_score", 0.0)) * 10.0\n        + float(rank_debug.get("locality_score", 0.0)) * 0.4\n        + float(rank_debug.get("town_score", 0.0)) * 0.25\n        + float(rank_debug.get("address_score", 0.0)) * 8.0\n        + float(rank_debug.get("match_score", 0.0)) * 4.0,\n        3,\n    )\n\n\ndef _confidence_band(score: int) -> str:\n    if score >= 85:\n        return "high"\n    if score >= 70:\n        return "medium"\n    if score >= 50:\n        return "low"\n    return "very_low"\n\n\ndef _raw_address_has_anchor_risk(raw_address: str) -> bool:\n    text = _clean_text(raw_address).lower()\n    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ANCHOR_RISK_PATTERNS)\n\n\ndef _extract_pre_road_label(text: str, road_name: str) -> str:\n    source = _clean_text(text)\n    road = _clean_text(road_name)\n    if not source or not road:\n        return ""\n    match = re.search(rf"(?i)\\b{re.escape(road)}\\b", source)\n    if not match:\n        return ""\n    prefix = source[: match.start()].strip(" ,.-/")\n    prefix = re.sub(r"(?i)\\b(?:at|on|off|to|of|the)\\b\\s*$", "", prefix).strip(" ,.-/")\n    prefix = re.sub(r"(?i)^\\d+[A-Z]?(?:\\s*-\\s*\\d+[A-Z]?)?\\s*,?\\s*", "", prefix).strip(" ,.-/")\n    return prefix\n\n\ndef _compute_os_confidence(\n    raw_address: str,\n    best: RoadMatch | None,\n    candidates: list[RoadMatch],\n    os_ranked: list[tuple[tuple[int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]],\n    guarded_source: str,\n    guarded_reason: str,\n    os_best: dict[str, Any] | None,\n    os_best_rank_debug: dict[str, Any] | None,\n) -> tuple[int, str, str]:\n    if not os_best or not os_best_rank_debug:\n        return 0, "very_low", "no_os_candidate"\n\n    score = 0.0\n    reasons: list[str] = []\n\n    road_tier = int(os_best_rank_debug.get("road_tier", 0))\n    locality_tier = int(os_best_rank_debug.get("locality_tier", 0))\n    town_tier = int(os_best_rank_debug.get("town_tier", 0))\n    admin_tier = int(os_best_rank_debug.get("admin_tier", 0))\n    number_tier = int(os_best_rank_debug.get("number_tier", 0))\n    number_mismatch = int(os_best_rank_debug.get("number_mismatch", 0))\n    record_tier = int(os_best_rank_debug.get("record_tier", 0))\n    road_score = float(os_best_rank_debug.get("road_score", 0.0))\n    has_input_number = bool(_extract_first_half_numbers(raw_address))\n    has_anchor_risk = _raw_address_has_anchor_risk(raw_address)\n    os_best_road_name = _os_candidate_road(os_best)\n    raw_pre_road = _extract_pre_road_label(raw_address, os_best_road_name)\n    best_pre_road = _extract_pre_road_label(_os_candidate_address(os_best), os_best_road_name)\n\n    score += {4: 40.0, 3: 30.0, 2: 20.0, 1: 10.0}.get(road_tier, 0.0)\n    if road_tier >= 4:\n        reasons.append("road_exact")\n    elif road_tier >= 3:\n        reasons.append("road_strong")\n    else:\n        reasons.append("road_weak")\n\n    score += {3: 15.0, 2: 11.0, 1: 5.0}.get(locality_tier, 0.0)\n    score += {3: 10.0, 2: 7.0, 1: 3.0}.get(town_tier, 0.0)\n    score += 3.0 if admin_tier else 0.0\n    if locality_tier >= 2:\n        reasons.append("locality_match")\n    elif town_tier >= 2:\n        reasons.append("town_match")\n    elif locality_tier or town_tier or admin_tier:\n        reasons.append("place_partial")\n    else:\n        reasons.append("place_weak")\n\n    if number_tier >= 4:\n        score += 24.0\n        reasons.append("number_range_exact")\n    elif number_tier == 3:\n        score += 20.0\n        reasons.append("number_match")\n    elif number_tier == 2:\n        score += 12.0\n        reasons.append("number_in_range")\n    elif number_tier == 1:\n        score += 8.0\n        reasons.append("number_neutral")\n    else:\n        reasons.append("number_weak")\n    if number_mismatch:\n        score -= 8.0\n        reasons.append("number_mismatch")\n    if has_input_number and road_tier >= 4 and number_tier >= 3 and (locality_tier >= 2 or town_tier >= 2):\n        score += 8.0\n        reasons.append("numbered_address_bonus")\n    elif not has_input_number:\n        reasons.append("no_input_number")\n\n    if record_tier:\n        score += 4.0\n        reasons.append("dpa")\n\n    if (\n        not has_input_number\n        and not has_anchor_risk\n        and raw_pre_road\n        and best_pre_road\n    ):\n        site_sim = _similarity(raw_pre_road, best_pre_road)\n        if site_sim >= 0.85:\n            score += 6.0\n            reasons.append("site_name_match")\n        elif site_sim < 0.55:\n            score -= 18.0\n            reasons.append("site_name_mismatch")\n\n    lexicon_strong = _lexicon_guard_eligible(best, candidates)\n    if lexicon_strong:\n        lex_road_key = _norm_road_compact(best.road_name)\n        same_road = [\n            (rank_debug, candidate)\n            for _, rank_debug, candidate in os_ranked\n            if _norm_road_compact(_os_candidate_road(candidate)) == lex_road_key\n        ]\n        if not same_road:\n            score = min(score, 35.0)\n            reasons.append("no_same_road_against_strong_lexicon")\n        else:\n            score += 5.0\n            reasons.append("same_road_with_lexicon")\n            if guarded_reason == "os_same_road_guard":\n                score += 3.0\n                reasons.append("same_road_guard_applied")\n\n    comparable = os_ranked\n    if lexicon_strong and best:\n        lex_road_key = _norm_road_compact(best.road_name)\n        same_road_ranked = [\n            (rank_debug, candidate)\n            for _, rank_debug, candidate in os_ranked\n            if _norm_road_compact(_os_candidate_road(candidate)) == lex_road_key\n        ]\n        if same_road_ranked:\n            comparable = [(None, rd, c) for rd, c in same_road_ranked]\n\n    current_proxy = _candidate_confidence_proxy(os_best_rank_debug)\n    next_proxy = None\n    if len(comparable) >= 2:\n        if comparable[0][2] is os_best:\n            next_proxy = _candidate_confidence_proxy(comparable[1][1])\n        else:\n            comparable_sorted = sorted(\n                comparable,\n                key=lambda item: _candidate_confidence_proxy(item[1]),\n                reverse=True,\n            )\n            if comparable_sorted and comparable_sorted[0][2] is os_best and len(comparable_sorted) > 1:\n                next_proxy = _candidate_confidence_proxy(comparable_sorted[1][1])\n    if next_proxy is None and len(os_ranked) >= 2:\n        next_proxy = _candidate_confidence_proxy(os_ranked[1][1])\n\n    if next_proxy is None:\n        score += 10.0\n        reasons.append("single_winner")\n    else:\n        gap = current_proxy - next_proxy\n        if gap >= 8.0:\n            score += 10.0\n            reasons.append("strong_margin")\n        elif gap >= 4.0:\n            score += 7.0\n            reasons.append("medium_margin")\n        elif gap >= 2.0:\n            score += 4.0\n            reasons.append("small_margin")\n        else:\n            score += 1.0\n            reasons.append("weak_margin")\n\n    if guarded_source == "lexicon":\n        score = min(score, 45.0)\n        reasons.append("guarded_to_lexicon")\n    elif road_tier == 0 and road_score < 0.8:\n        score = min(score, 30.0)\n        reasons.append("wrong_road_risk")\n\n    if has_anchor_risk:\n        score -= 12.0\n        reasons.append("anchor_style_address")\n        if not has_input_number:\n            score -= 6.0\n            reasons.append("anchor_without_number")\n\n    final_score = max(0, min(100, int(round(score))))\n    return final_score, _confidence_band(final_score), " + ".join(reasons)\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(\n        description="Road-level Gazetteer matching only, using OS Open Names Named Road records."\n    )\n    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV. Default: {DEFAULT_INPUT}")\n    parser.add_argument("--address-column", default=DEFAULT_ADDRESS_COLUMN, help=f"Original address column. Default: {DEFAULT_ADDRESS_COLUMN}")\n    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help=f"Output JSON. Default: {DEFAULT_OUTPUT_JSON}")\n    parser.add_argument("--output-xlsx", default=DEFAULT_OUTPUT_XLSX, help=f"Summary XLSX. Default: {DEFAULT_OUTPUT_XLSX}")\n    parser.add_argument("--keys-file", default=DEFAULT_KEYS_FILE, help=f"Keys file or directory. Default: {DEFAULT_KEYS_FILE}")\n    parser.add_argument("--api-key", default="", help="OS Places API key. Overrides --keys-file if provided.")\n    parser.add_argument("--google-api-key", default="", help="Google Maps Places Text Search API key. Overrides --keys-file if provided.")\n    parser.add_argument("--os-open-names-db", default=DEFAULT_OS_OPEN_NAMES_DB, help=f"OS Open Names SQLite DB. Default: {DEFAULT_OS_OPEN_NAMES_DB}")\n    parser.add_argument("--county", default=DEFAULT_COUNTY, help=f"County filter for Named Road lookup. Default: {DEFAULT_COUNTY}")\n    parser.add_argument("--council", default=DEFAULT_COUNCIL, help=f"Council label for metadata only. Default: {DEFAULT_COUNCIL}")\n    parser.add_argument("--row-limit", type=int, default=None, help="Optional row limit.")\n    parser.add_argument("--max-candidates", type=int, default=5, help="How many road candidates to keep per row.")\n    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=f"HTTP timeout. Default: {DEFAULT_TIMEOUT_SECONDS}")\n    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"HTTP max retries. Default: {DEFAULT_MAX_RETRIES}")\n    parser.add_argument(\n        "--google-low-confidence-threshold",\n        type=int,\n        default=DEFAULT_GOOGLE_LOW_CONFIDENCE_THRESHOLD,\n        help=f"Request Google when os_confidence_score is below this threshold. Default: {DEFAULT_GOOGLE_LOW_CONFIDENCE_THRESHOLD}",\n    )\n    return parser.parse_args()\n\n\ndef _anchor_confidence(best: RoadMatch | None, candidates: list[RoadMatch]) -> str:\n    if best is None:\n        return "none"\n    if not candidates:\n        return "low"\n    if len(candidates) == 1:\n        return "high"\n    margin = best.score - candidates[1].score\n    if margin >= 8:\n        return "high"\n    if margin >= 3:\n        return "medium"\n    return "low"\n\n\ndef _anchor_type(raw_address: str, best: RoadMatch | None) -> str:\n    text = _clean_text(raw_address).lower()\n    if not best:\n        return "none"\n    if any(token in text for token in ["junction", "corner of"]):\n        return "junction"\n    if any(token in text for token in ["land at", "land off", "adjacent to", "rear of", "forecourt of", "curtilage of"]):\n        return "relative-road-anchor"\n    return "single-road-anchor"\n\n\ndef main() -> None:\n    args = parse_args()\n    base_rows = _read_rows(args.input, args.row_limit)\n    rows = _expand_main_rows(base_rows, args.address_column)\n    os_api_key = args.api_key or _load_os_api_key_from_keys_file(args.keys_file)\n    google_api_key = args.google_api_key or _load_google_api_key_from_keys_file(args.keys_file)\n    matcher = OpenNamesRoadMatcher(args.os_open_names_db, county_name=args.county, council_name=args.council)\n    session = _build_session() if os_api_key else None\n    output_rows: list[dict[str, Any]] = []\n\n    try:\n        for idx, row in enumerate(rows, start=1):\n            raw_value = str(row.get(args.address_column, "") or "")\n            raw_value = re.sub(r"[\\r\\n]+", ", ", raw_value)\n            raw_address = _clean_text(raw_value)\n            corrected_raw_address, correction_notes = _correct_raw_address_spelling(raw_address, matcher)\n            best, candidates, debug = matcher.best_match(raw_address, max_candidates=args.max_candidates)\n            road_phrases = [item.matched_phrase for item in candidates if getattr(item, "matched_phrase", "")]\n            if not road_phrases:\n                road_phrases = debug.get("expanded_segments", [])\n            place_hints = list(debug.get("place_hints", []))\n            for hint in _extract_freeform_place_hints(raw_address, road_phrases):\n                if _norm_compact(hint) not in {_norm_compact(value) for value in place_hints}:\n                    place_hints.append(hint)\n            os_query = _compose_os_query(corrected_raw_address or raw_address, args.council, args.county)\n            os_query_variants = _expand_os_query_variants(os_query)\n            os_candidates_raw: list[dict[str, Any]] = []\n            if session is not None and os_query_variants:\n                merged_candidates: list[dict[str, Any]] = []\n                for query_variant in os_query_variants:\n                    merged_candidates.extend(\n                        _query_find(\n                            session=session,\n                            api_key=os_api_key,\n                            query=query_variant,\n                            timeout_seconds=args.timeout_seconds,\n                            max_retries=args.max_retries,\n                        )\n                    )\n                os_candidates_raw = _dedupe_os_candidates(merged_candidates)\n            os_ranked: list[tuple[tuple[int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]] = []\n            for candidate in os_candidates_raw:\n                rank_tuple, rank_debug = _rank_os_candidate(\n                    candidate,\n                    corrected_raw_address or raw_address,\n                    road_phrases,\n                    place_hints,\n                    args.council,\n                )\n                os_ranked.append((rank_tuple, rank_debug, candidate))\n            os_ranked.sort(key=lambda item: item[0], reverse=True)\n            os_ranked_valid = _valid_os_ranked(os_ranked)\n            os_raw_best_rank = os_ranked_valid[0][0] if os_ranked_valid else None\n            os_raw_best_rank_debug = os_ranked_valid[0][1] if os_ranked_valid else None\n            os_raw_best = os_ranked_valid[0][2] if os_ranked_valid else None\n            guarded_source, guarded_reason, guarded_lexicon, os_best_rank, os_best_rank_debug, os_best = _select_guarded_anchor(\n                best,\n                candidates,\n                os_ranked_valid,\n            )\n            os_confidence_score, os_confidence_band, os_confidence_reason = _compute_os_confidence(\n                raw_address,\n                best,\n                candidates,\n                os_ranked_valid,\n                guarded_source,\n                guarded_reason,\n                os_best,\n                os_best_rank_debug,\n            )\n            google_query = ""\n            google_candidates_raw: list[dict[str, Any]] = []\n            google_ranked: list[tuple[tuple[int, int, int, int, int, int, float, float, float, float, float], dict[str, Any], dict[str, Any]]] = []\n            google_best_rank = None\n            google_best_rank_debug = None\n            google_best = None\n            google_confidence_score = 0\n            google_confidence_band = "very_low"\n            google_confidence_reason = "not_requested"\n            if (\n                session is not None\n                and int(os_confidence_score) < int(args.google_low_confidence_threshold)\n            ):\n                if google_api_key:\n                    google_query = os_query\n                    google_candidates_raw = _query_google_candidates_with_variants(\n                        session=session,\n                        google_api_key=google_api_key,\n                        query=google_query,\n                        timeout_seconds=args.timeout_seconds,\n                        max_retries=args.max_retries,\n                    )\n                    for candidate in google_candidates_raw:\n                        rank_tuple, rank_debug = _rank_google_candidate(\n                            candidate,\n                            corrected_raw_address or raw_address,\n                            road_phrases,\n                            place_hints,\n                            args.council,\n                        )\n                        google_ranked.append((rank_tuple, rank_debug, candidate))\n                    google_ranked.sort(key=lambda item: item[0], reverse=True)\n                    google_ranked_valid = _valid_os_ranked(google_ranked)\n                    google_best_rank = google_ranked_valid[0][0] if google_ranked_valid else None\n                    google_best_rank_debug = google_ranked_valid[0][1] if google_ranked_valid else None\n                    google_best = google_ranked_valid[0][2] if google_ranked_valid else None\n                    google_confidence_score, google_confidence_band, google_confidence_reason = _compute_google_confidence(\n                        raw_address,\n                        best,\n                        candidates,\n                        google_ranked_valid,\n                        google_best,\n                        google_best_rank_debug,\n                    )\n            final_source = guarded_source\n            final_name = ""\n            final_place = ""\n            final_easting = ""\n            final_northing = ""\n            final_score: Any = ""\n            final_choice_reason = guarded_reason\n            if guarded_source == "os" and os_best:\n                final_source = "os"\n                final_name = _os_candidate_road(os_best)\n                final_place = _os_candidate_place(os_best)\n                final_easting = _to_float(os_best.get("X_COORDINATE"))\n                final_northing = _to_float(os_best.get("Y_COORDINATE"))\n                final_score = json.dumps(list(os_best_rank), ensure_ascii=False) if os_best_rank else ""\n            elif guarded_source == "lexicon" and guarded_lexicon:\n                final_source = "lexicon"\n                final_name = guarded_lexicon.road_name\n                final_place = guarded_lexicon.populated_place\n                final_easting = guarded_lexicon.geometry_x\n                final_northing = guarded_lexicon.geometry_y\n                final_score = guarded_lexicon.score\n            if (\n                google_best\n                and int(google_confidence_score) >= int(args.google_low_confidence_threshold)\n                and int(google_confidence_score) > int(os_confidence_score)\n            ):\n                final_source = "gog"\n                final_name = _google_candidate_road(google_best)\n                final_place = _best_google_place_field(google_best, place_hints)\n                final_easting = _to_float(google_best.get("X_COORDINATE"))\n                final_northing = _to_float(google_best.get("Y_COORDINATE"))\n                final_score = json.dumps(list(google_best_rank), ensure_ascii=False) if google_best_rank else ""\n                final_choice_reason = "google_over_os"\n            output = dict(row)\n            output["idx"] = idx\n            output["raw_address"] = raw_address\n            output["corrected_raw_address"] = corrected_raw_address\n            output["spelling_correction_notes_json"] = json.dumps(correction_notes, ensure_ascii=False)\n            output["gazetteer_scope_county"] = args.county\n            output["gazetteer_scope_council"] = args.council\n            output["road_match_status"] = "matched" if best else "no_match"\n            output["road_candidate_phrases_json"] = json.dumps(debug.get("expanded_segments", []), ensure_ascii=False)\n            output["road_match_debug_json"] = json.dumps(debug, ensure_ascii=False)\n            output["road_candidates_json"] = json.dumps([_serialize_match(item) for item in candidates], ensure_ascii=False)\n            output["os_query"] = os_query\n            output["os_query_variants_json"] = json.dumps(os_query_variants, ensure_ascii=False)\n            output["os_candidate_count"] = len(os_ranked)\n            output["os_guarded_reason"] = guarded_reason\n            output["os_confidence_score"] = os_confidence_score\n            output["os_confidence_band"] = os_confidence_band\n            output["os_confidence_reason"] = os_confidence_reason\n            output["os_candidates_json"] = json.dumps(\n                [\n                    _serialize_os_candidate(candidate, rank_tuple, rank_debug, place_hints)\n                    for rank_tuple, rank_debug, candidate in os_ranked\n                ],\n                ensure_ascii=False,\n            )\n            output["os_candidates_full_json"] = json.dumps(os_candidates_raw, ensure_ascii=False)\n            output["gog_query"] = google_query\n            output["gog_candidate_count"] = len(google_ranked)\n            output["gog_confidence_score"] = google_confidence_score\n            output["gog_confidence_band"] = google_confidence_band\n            output["gog_confidence_reason"] = google_confidence_reason\n            output["gog_candidates_json"] = json.dumps(\n                [\n                    _serialize_google_candidate(candidate, rank_tuple, rank_debug, place_hints)\n                    for rank_tuple, rank_debug, candidate in google_ranked\n                ],\n                ensure_ascii=False,\n            )\n            output["gog_candidates_full_json"] = json.dumps(google_candidates_raw, ensure_ascii=False)\n            output["gog_best_address"] = _google_candidate_address(google_best) if google_best else ""\n            output["gog_best_road_name"] = _google_candidate_road(google_best) if google_best else ""\n            output["gog_best_place"] = _best_google_place_field(google_best, place_hints) if google_best else ""\n            output["gog_best_easting_27700"] = _to_float(google_best.get("X_COORDINATE")) if google_best else ""\n            output["gog_best_northing_27700"] = _to_float(google_best.get("Y_COORDINATE")) if google_best else ""\n            output["gog_best_rank_tuple_json"] = json.dumps(list(google_best_rank), ensure_ascii=False) if google_best_rank else ""\n            output["os_raw_best_address"] = _os_candidate_address(os_raw_best) if os_raw_best else ""\n            output["os_raw_best_road_name"] = _os_candidate_road(os_raw_best) if os_raw_best else ""\n            output["os_raw_best_place"] = _best_os_place_field(os_raw_best, place_hints) if os_raw_best else ""\n            output["os_raw_best_easting_27700"] = _to_float(os_raw_best.get("X_COORDINATE")) if os_raw_best else ""\n            output["os_raw_best_northing_27700"] = _to_float(os_raw_best.get("Y_COORDINATE")) if os_raw_best else ""\n            output["os_raw_best_rank_tuple_json"] = json.dumps(list(os_raw_best_rank), ensure_ascii=False) if os_raw_best_rank else ""\n            output["os_best_address"] = _os_candidate_address(os_best) if os_best else ""\n            output["os_best_road_name"] = _os_candidate_road(os_best) if os_best else ""\n            output["os_best_place"] = _best_os_place_field(os_best, place_hints) if os_best else ""\n            output["os_best_postcode_district"] = _os_candidate_postcode_district(os_best) if os_best else ""\n            output["os_best_easting_27700"] = _to_float(os_best.get("X_COORDINATE")) if os_best else ""\n            output["os_best_northing_27700"] = _to_float(os_best.get("Y_COORDINATE")) if os_best else ""\n            output["os_best_rank_tuple_json"] = json.dumps(list(os_best_rank), ensure_ascii=False) if os_best_rank else ""\n            output["os_best_locality_bucket"] = os_best_rank_debug.get("locality_tier", "") if os_best_rank_debug else ""\n            output["os_best_locality_score"] = os_best_rank_debug.get("locality_score", "") if os_best_rank_debug else ""\n            output["os_best_town_bucket"] = os_best_rank_debug.get("town_tier", "") if os_best_rank_debug else ""\n            output["os_best_town_score"] = os_best_rank_debug.get("town_score", "") if os_best_rank_debug else ""\n            output["os_best_admin_bucket"] = os_best_rank_debug.get("admin_tier", "") if os_best_rank_debug else ""\n            output["os_best_road_bucket"] = os_best_rank_debug.get("road_tier", "") if os_best_rank_debug else ""\n            output["os_best_road_score"] = os_best_rank_debug.get("road_score", "") if os_best_rank_debug else ""\n            output["os_best_number_bucket"] = os_best_rank_debug.get("number_tier", "") if os_best_rank_debug else ""\n            output["os_best_number_mismatch"] = os_best_rank_debug.get("number_mismatch", "") if os_best_rank_debug else ""\n            output["os_best_address_score"] = os_best_rank_debug.get("address_score", "") if os_best_rank_debug else ""\n            output["os_best_match_score"] = os_best_rank_debug.get("match_score", "") if os_best_rank_debug else ""\n            if best:\n                output["best_road_anchor_name"] = best.road_name\n                output["best_road_anchor_place"] = best.populated_place\n                output["best_road_anchor_easting_27700"] = best.geometry_x\n                output["best_road_anchor_northing_27700"] = best.geometry_y\n                output["best_road_anchor_confidence"] = _anchor_confidence(best, candidates)\n                output["best_road_anchor_type"] = _anchor_type(raw_address, best)\n                output["best_road_name"] = best.road_name\n                output["best_road_place"] = best.populated_place\n                output["best_road_district"] = best.district_borough\n                output["best_road_county"] = best.county_unitary\n                output["best_road_postcode_district"] = best.postcode_district\n                output["best_road_easting_27700"] = best.geometry_x\n                output["best_road_northing_27700"] = best.geometry_y\n                output["best_road_score"] = best.score\n                output["best_road_match_type"] = best.match_type\n                output["best_road_matched_phrase"] = best.matched_phrase\n                output["best_road_local_type"] = best.local_type\n            else:\n                output["best_road_anchor_name"] = ""\n                output["best_road_anchor_place"] = ""\n                output["best_road_anchor_easting_27700"] = ""\n                output["best_road_anchor_northing_27700"] = ""\n                output["best_road_anchor_confidence"] = "none"\n                output["best_road_anchor_type"] = "none"\n                output["best_road_name"] = ""\n                output["best_road_place"] = ""\n                output["best_road_district"] = ""\n                output["best_road_county"] = ""\n                output["best_road_postcode_district"] = ""\n                output["best_road_easting_27700"] = ""\n                output["best_road_northing_27700"] = ""\n                output["best_road_score"] = ""\n                output["best_road_match_type"] = ""\n                output["best_road_matched_phrase"] = ""\n                output["best_road_local_type"] = ""\n            output["best_anchor_source"] = final_source if final_source else "none"\n            output["best_anchor_name"] = final_name\n            output["best_anchor_place"] = final_place\n            output["best_anchor_easting_27700"] = final_easting\n            output["best_anchor_northing_27700"] = final_northing\n            output["best_anchor_score"] = final_score\n            output["best_anchor_choice_reason"] = final_choice_reason\n            output_rows.append(output)\n    finally:\n        matcher.close()\n        if session is not None:\n            session.close()\n\n    output_json = Path(args.output_json)\n    output_xlsx = Path(args.output_xlsx)\n    output_json.parent.mkdir(parents=True, exist_ok=True)\n    output_xlsx.parent.mkdir(parents=True, exist_ok=True)\n\n    output_json.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")\n    _write_summary_xlsx(output_xlsx, output_rows, args.input)\n\n    matched = sum(1 for row in output_rows if row.get("road_match_status") == "matched")\n    print(f"Rows processed: {len(output_rows)}")\n    print(f"Road matched: {matched}")\n    print(f"Road unmatched: {len(output_rows) - matched}")\n    print(f"JSON written: {output_json}")\n    print(f"XLSX written: {output_xlsx}")\n\n\nif __name__ == "__main__":\n    main()\n'
_embedded_v3_name = "embedded_address_to_point_v3_module"
_embedded_v3_module = types.ModuleType(_embedded_v3_name)
_embedded_v3_module.__dict__["__name__"] = _embedded_v3_name
sys.modules[_embedded_v3_name] = _embedded_v3_module
_v3_ns: dict[str, Any] = _embedded_v3_module.__dict__
exec(_EMBEDDED_V3_SOURCE, _v3_ns)
V3 = type("EmbeddedV3", (), {k: v for k, v in _v3_ns.items() if not k.startswith("__")})
_thread_local = threading.local()

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_COUNCIL_PROFILE = "monmouthshire"
COUNCIL_PROFILES: dict[str, dict[str, Any]] = {
    "mansfield": {
        "county": "Nottinghamshire",
        "council": "Mansfield",
        "address_column": "chargegeog",
        "preferred_layers": ["Mansfield"],
    },
    "monmouthshire": {
        "county": "Monmouthshire",
        "council": "Monmouthshire",
        "address_column": "charge-geographic-description",
        "preferred_layers": ["features", "capture_opti_result", "capture_result", "Monmouthshire"],
    },
}
GEMINI_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LEADING_PLOT_RANGE_RE = re.compile(r"^\s*plots?\s+(\d+)\s*-\s*(\d+)\b\s*(.*)$", re.IGNORECASE)
LEADING_PLOT_LIST_RE = re.compile(
    r"^\s*plots?\s+((?:\d+[A-Z]?)(?:\s*(?:,|&|and)\s*(?:\d+[A-Z]?))+)\s+(.*)$",
    re.IGNORECASE,
)
GENERIC_OS_TOKENS = {
    "street",
    "road",
    "lane",
    "gate",
    "avenue",
    "drive",
    "close",
    "court",
    "way",
    "hill",
    "place",
    "row",
    "off",
    "land",
    "plot",
    "part",
    "of",
    "the",
    "at",
    "junction",
    "forecourt",
    "adjacent",
}
GENERIC_OS_ADDRESS_MARKERS = (
    "STREET RECORD",
    "POST BOX",
    "PUBLIC TELEPHONE",
    "ELECTRICITY SUB STATION",
    "SHELTER",
    "TANK ",
    "CHIMNEY ",
)
UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
LAT_LNG_PAIR_RE = re.compile(r"(-?\d{1,2}\.\d{4,})\s*[,/ ]\s*(-?\d{1,3}\.\d{4,})")
WEBGIS_LINK_RE = re.compile(
    r"https?://[^\s\"'<>]+(?:arcgis|webgis|mapserver|osmaps|google\\.[^/]+/maps|bing\\.com/maps)[^\s\"'<>]*",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal lexicon + OS + Gemini address-to-point pipeline.")
    parser.add_argument("--input-gpkg", required=True, help="Input GPKG path.")
    parser.add_argument("--input-layer", help="Optional input GPKG layer. Defaults to the council profile preference.")
    parser.add_argument("--output-json", help="Optional output JSON path.")
    parser.add_argument("--output-xlsx", help="Optional output XLSX path.")
    parser.add_argument("--address-column", help="Address column name. Defaults to the council profile.")
    parser.add_argument("--keys-file", default="/env/key", help="Key file or directory.")
    parser.add_argument("--api-key", help="OS API key override.")
    parser.add_argument("--gemini-api-key", help="Gemini API key override.")
    parser.add_argument("--google-api-key", help="Google Places API key override.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--gemini-service-tier",
        choices=["standard", "flex"],
        default="standard",
        help="Gemini inference tier. Use flex for cheaper latency-tolerant runs.",
    )
    parser.add_argument(
        "--gemini-timeout-ms",
        type=int,
        default=60000,
        help="Per-request Gemini timeout in milliseconds. Flex runs usually need 600000+.",
    )
    parser.add_argument("--os-open-names-db", default=V3.DEFAULT_OS_OPEN_NAMES_DB, help="OS Open Names SQLite DB.")
    parser.add_argument(
        "--council-profile",
        choices=sorted(COUNCIL_PROFILES),
        default=DEFAULT_COUNCIL_PROFILE,
        help="Council defaults for county/council/address column/layer.",
    )
    parser.add_argument("--county", help="County scope. Overrides --council-profile.")
    parser.add_argument("--council", help="Council scope. Overrides --council-profile.")
    parser.add_argument("--row-limit", type=int, default=0, help="Optional row limit.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers.")
    parser.add_argument("--web-research-threshold", type=int, default=75, help="Run DDG/Gemini web research below this confidence.")
    parser.add_argument("--web-research-results", type=int, default=5, help="Max DuckDuckGo lite results to inspect.")
    parser.add_argument("--web-research-pages", type=int, default=4, help="Max result pages to fetch for evidence.")
    parser.add_argument("--web-research-workers", type=int, default=4, help="Concurrent workers for low-confidence web research.")
    parser.add_argument("--enable-web-research", action="store_true", help="Enable low-confidence DuckDuckGo/Gemini web research.")
    return _apply_council_profile(parser.parse_args())


def _set_embedded_default_scope(county: str, council: str) -> None:
    V3.DEFAULT_COUNTY = county
    V3.DEFAULT_COUNCIL = council
    _v3_ns["DEFAULT_COUNTY"] = county
    _v3_ns["DEFAULT_COUNCIL"] = council


def _apply_council_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile_name = clean_text(getattr(args, "council_profile", "")) or DEFAULT_COUNCIL_PROFILE
    profile = COUNCIL_PROFILES.get(profile_name.lower(), COUNCIL_PROFILES[DEFAULT_COUNCIL_PROFILE])
    args.council_profile = profile_name.lower()
    if not clean_text(getattr(args, "county", "")):
        args.county = profile["county"]
    if not clean_text(getattr(args, "council", "")):
        args.council = profile["council"]
    if not clean_text(getattr(args, "address_column", "")):
        args.address_column = profile["address_column"]
    _set_embedded_default_scope(clean_text(args.county), clean_text(args.council))
    return args


def clean_text(value: Any) -> str:
    return V3._clean_text(value)


def normalize_gemini_model(model: str) -> str:
    cleaned = clean_text(model)
    if cleaned == "gemini-3-preview":
        return "gemini-3-pro-preview"
    return cleaned or DEFAULT_GEMINI_MODEL


def _pick_gpkg_layer(path: Path, preferred_layer: str | None = None, council_profile: str | None = None) -> str | None:
    if gpd is None:
        raise SystemExit("This script requires geopandas.")
    layers = gpd.list_layers(path)
    if layers is None or len(layers) == 0:
        return None
    names = [str(value) for value in layers["name"].tolist()]
    preferred = clean_text(preferred_layer)
    if preferred:
        if preferred in names:
            return preferred
        raise SystemExit(f"GPKG layer not found: {preferred}. Available layers: {', '.join(names)}")
    if path.stem in names:
        return path.stem
    profile = COUNCIL_PROFILES.get(clean_text(council_profile).lower(), {})
    for layer_name in profile.get("preferred_layers", []):
        if layer_name in names:
            return layer_name
    for layer_name in ["features", "capture_opti_result", "capture_result", "Monmouthshire", "Mansfield"]:
        if layer_name in names:
            return layer_name
    return names[0]


def _derive_key_value(record: dict[str, Any], fallback_index: int) -> str:
    for field in ["unique_key", "key", "geomid", "queryid", "chargeid", "oachargeid"]:
        value = record.get(field)
        text = clean_text(value)
        if text:
            return text
    return str(fallback_index)


def load_input_rows(
    path: Path,
    row_limit: int,
    address_column: str,
    input_layer: str | None = None,
    council_profile: str | None = None,
) -> list[dict[str, Any]]:
    if gpd is None:
        raise SystemExit("This script requires geopandas.")
    layer = _pick_gpkg_layer(path, input_layer, council_profile)
    gdf = gpd.read_file(path, layer=layer)
    if row_limit > 0:
        gdf = gdf.head(row_limit)
    records = gdf.to_dict(orient="records")
    normalized_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        normalized = dict(record)
        normalized["key"] = _derive_key_value(normalized, idx)
        if "unique_key" not in normalized or clean_text(normalized.get("unique_key")) == "":
            normalized["unique_key"] = normalized["key"]
        if address_column not in normalized:
            raise SystemExit(f"Address column not found in GPKG: {address_column}")
        normalized_rows.append(normalized)
    return normalized_rows


def load_polygon_geometries(
    path: Path,
    input_layer: str | None = None,
    council_profile: str | None = None,
) -> dict[str, Any]:
    if gpd is None:
        raise SystemExit("This script requires geopandas.")
    layer = _pick_gpkg_layer(path, input_layer, council_profile)
    gdf = gpd.read_file(path, layer=layer)
    geometry_by_key: dict[str, Any] = {}
    for idx, (_, row) in enumerate(gdf.iterrows(), start=1):
        record = row.to_dict()
        key = _derive_key_value(record, idx)
        geometry = row.geometry
        if geometry is not None and not getattr(geometry, "is_empty", True):
            geometry_by_key[str(key)] = geometry
    return geometry_by_key


def expand_main_rows(rows: list[dict[str, Any]], address_column: str) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "unique_key" not in item and "key" in item:
            item["unique_key"] = item.get("key")
        normalized_rows.append(item)
    expanded = V3._expand_main_rows(normalized_rows, address_column)
    final_rows: list[dict[str, Any]] = []
    for row in expanded:
        raw_address = clean_text(row.get(address_column))
        parent_key = str(row.get("key") or row.get("unique_key") or "").strip()
        source_key = str(row.get("_source_unique_key") or row.get("unique_key") or row.get("key") or "").strip()

        match = LEADING_PLOT_RANGE_RE.match(raw_address)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start:
                start, end = end, start
            if 2 <= (end - start + 1) <= 20:
                suffix = clean_text(match.group(3)).lstrip(", ")
                final_rows.append(row)
                for offset, number in enumerate(range(start, end + 1), start=1):
                    child = dict(row)
                    child[address_column] = f"Plots {number} {suffix}".strip()
                    if parent_key:
                        child["key"] = f"{parent_key}_{offset}"
                    if row.get("unique_key"):
                        child["unique_key"] = child.get("key") or row.get("unique_key")
                    child["_source_unique_key"] = source_key
                    final_rows.append(child)
                continue

        list_match = LEADING_PLOT_LIST_RE.match(raw_address)
        if list_match and not LEADING_PLOT_RANGE_RE.match(raw_address):
            numbers_part = clean_text(list_match.group(1))
            suffix = clean_text(list_match.group(2)).lstrip(", ")
            numbers = re.findall(r"\b\d+[A-Z]?\b", numbers_part.upper())
            if 2 <= len(numbers) <= 20:
                final_rows.append(row)
                for offset, number in enumerate(numbers, start=1):
                    child = dict(row)
                    child[address_column] = f"Plots {number} {suffix}".strip()
                    if parent_key:
                        child["key"] = f"{parent_key}_{offset}"
                    if row.get("unique_key"):
                        child["unique_key"] = child.get("key") or row.get("unique_key")
                    child["_source_unique_key"] = source_key
                    final_rows.append(child)
                continue

        final_rows.append(row)
    return final_rows


def _get_thread_resources(args: argparse.Namespace) -> tuple[Any, Any]:
    session = getattr(_thread_local, "session", None)
    matcher = getattr(_thread_local, "matcher", None)
    matcher_key = getattr(_thread_local, "matcher_key", None)
    desired_key = (args.os_open_names_db, args.county, args.council)
    if session is None:
        session = V3._build_session()
        _thread_local.session = session
    if matcher is None or matcher_key != desired_key:
        if matcher is not None:
            try:
                matcher.close()
            except Exception:
                pass
        matcher = V3.OpenNamesRoadMatcher(args.os_open_names_db, county_name=args.county, council_name=args.council)
        _thread_local.matcher = matcher
        _thread_local.matcher_key = desired_key
    return session, matcher


def _serialize_os_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "address": V3._os_candidate_address(candidate),
        "road_name": V3._os_candidate_road(candidate),
        "place": V3._os_candidate_place(candidate),
        "easting_27700": V3._to_float(candidate.get("X_COORDINATE")),
        "northing_27700": V3._to_float(candidate.get("Y_COORDINATE")),
    }


def _google_textsearch_best(
    session: requests.Session,
    google_api_key: str,
    raw_address: str,
    corrected_raw_address: str,
    os_query: str,
    road_phrases: list[str],
    place_hints: list[str],
    council_name: str,
) -> dict[str, Any]:
    google_query = os_query or corrected_raw_address or raw_address
    if not clean_text(google_query):
        return {
            "status": "skipped",
            "query": "",
            "candidate_count": 0,
            "confidence_score": 0,
            "confidence_reason": "empty_query",
            "candidates_json": "[]",
            "best_address": "",
            "best_easting_27700": "",
            "best_northing_27700": "",
        }

    google_candidates_raw = V3._query_google_candidates_with_variants(
        session=session,
        google_api_key=google_api_key,
        query=google_query,
        timeout_seconds=V3.DEFAULT_TIMEOUT_SECONDS,
        max_retries=V3.DEFAULT_MAX_RETRIES,
    )
    serialized: list[dict[str, Any]] = []
    google_best = None
    for idx, candidate in enumerate(google_candidates_raw):
        place_ok = V3._google_place_hard_gate(candidate, place_hints)
        road_tier, road_score = V3._google_road_rank(candidate, road_phrases)
        serialized.append(
            {
                "address": V3._google_candidate_address(candidate),
                "road_name": V3._google_candidate_road(candidate),
                "place": V3._best_google_place_field(candidate, place_hints),
                "postcode_district": V3._extract_postcode_from_google_components(candidate.get("_google_address_components")),
                "easting_27700": V3._to_float(candidate.get("X_COORDINATE")),
                "northing_27700": V3._to_float(candidate.get("Y_COORDINATE")),
                "match": V3._to_float(candidate.get("MATCH")),
                "record_type": "GOOGLE",
                "rank_index": idx + 1,
                "place_gate_ok": 1 if place_ok else 0,
                "road_tier": road_tier,
                "road_score": road_score,
            }
        )
        if google_best is None and place_ok and int(road_tier) >= 4:
            google_best = candidate
    return {
        "status": "matched" if google_best else "no_match",
        "query": google_query,
        "candidate_count": len(google_candidates_raw),
        "confidence_score": 0,
        "confidence_reason": "google_first_place_and_road_match" if google_best else "no_google_candidate_with_place_and_road_match",
        "candidates_json": json.dumps(serialized, ensure_ascii=False),
        "best_address": V3._google_candidate_address(google_best) if google_best else "",
        "best_easting_27700": V3._to_float(google_best.get("X_COORDINATE")) if google_best else "",
        "best_northing_27700": V3._to_float(google_best.get("Y_COORDINATE")) if google_best else "",
    }


def _build_prompt(raw_address: str, council_name: str, lexicon_match: Any, os_candidates: list[dict[str, Any]]) -> str:
    lines = [f"这是一个{council_name}地区的历史地址：{raw_address}", ""]
    if lexicon_match is not None:
        lines.append(
            "这是这个历史地址词库的匹配结果："
            f"{clean_text(lexicon_match.road_name)} | {clean_text(lexicon_match.populated_place)}，"
            f"坐标=({lexicon_match.geometry_x}, {lexicon_match.geometry_y})"
        )
        lines.append("")
    lines.append("这是我在os api匹配到的所有结果：")
    if os_candidates:
        for candidate in os_candidates:
            lines.append(
                f"{clean_text(candidate.get('address'))}, "
                f"({candidate.get('easting_27700')}, {candidate.get('northing_27700')})"
            )
    else:
        lines.append("没有匹配结果")
    lines.append("")
    lines.append("请你根据你的专业知识，看看原始地址在os api中能否找到最佳的匹配。")
    lines.append("请只从上面的 OS API 结果里选择一个最佳匹配；如果都不合适，就返回 no_match。")
    lines.append("请你返回最佳的匹配。")
    lines.append("")
    lines.append("请只返回 JSON，不要返回别的内容。格式如下：")
    lines.append(
        '{"status":"matched或no_match","best_address":"必须和上面某条OS地址完全一致或空字符串",'
        '"best_easting_27700":123456.0或null,"best_northing_27700":654321.0或null,'
        '"reason":"一句简短中文理由"}'
    )
    return "\n".join(lines)


def _build_prompt_with_context(
    raw_address: str,
    council_name: str,
    lexicon_match: Any,
    os_candidates: list[dict[str, Any]],
    extra_notes: list[str] | None = None,
) -> str:
    prompt = _build_prompt(raw_address, council_name, lexicon_match, os_candidates)
    notes = [clean_text(note) for note in (extra_notes or []) if clean_text(note)]
    if not notes:
        return prompt
    insert_text = "\n".join(f"补充线索：{note}" for note in notes) + "\n"
    marker = "\n这是我在os api匹配到的所有结果："
    if marker in prompt:
        return prompt.replace(marker, "\n" + insert_text + "这是我在os api匹配到的所有结果：", 1)
    return prompt + "\n" + insert_text


def _parse_gemini_response(text: str) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    try:
        payload = json.loads(text)
    except Exception:
        import re

        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return {
            "status": "parse_error",
            "best_address": "",
            "best_easting_27700": None,
            "best_northing_27700": None,
            "reason": "",
        }
    return {
        "status": clean_text(payload.get("status")).lower() or "no_match",
        "best_address": clean_text(payload.get("best_address")),
        "best_easting_27700": V3._to_float(payload.get("best_easting_27700")),
        "best_northing_27700": V3._to_float(payload.get("best_northing_27700")),
        "reason": clean_text(payload.get("reason")),
    }


def _sanitize_api_error(exc: Exception) -> str:
    message = clean_text(str(exc))
    return re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", message)


def _call_gemini(
    gemini_api_key: str,
    gemini_model: str,
    prompt: str,
    service_tier: str = "standard",
    timeout_ms: int = 60000,
) -> dict[str, Any]:
    model_name = normalize_gemini_model(gemini_model)
    tier = clean_text(service_tier).lower()
    use_flex = tier == "flex"
    timeout_ms = max(1000, int(timeout_ms or 60000))
    try:
        text = ""
        if V3._google_genai is not None and not use_flex:
            client = V3._google_genai.Client(api_key=gemini_api_key)
            config = {"http_options": {"timeout": timeout_ms}}
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            text = clean_text(getattr(response, "text", "") or "")
        if not text:
            body: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
            timeout_seconds = max(60, int(math.ceil(timeout_ms / 1000)))
            if use_flex:
                body["service_tier"] = "flex"
                headers["X-Server-Timeout"] = str(timeout_seconds)
            response = requests.post(
                GEMINI_GENERATE_CONTENT_URL.format(model=model_name),
                params={"key": gemini_api_key},
                headers=headers,
                json=body,
                timeout=timeout_seconds + 5,
            )
            response.raise_for_status()
            payload = response.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = clean_text(" ".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)))
    except Exception as exc:
        return {
            "status": "error",
            "best_address": "",
            "best_easting_27700": None,
            "best_northing_27700": None,
            "reason": _sanitize_api_error(exc),
        }
    parsed = _parse_gemini_response(text)
    if parsed["status"] not in {"matched", "no_match"}:
        parsed["status"] = "matched" if parsed.get("best_address") else "parse_error"
    return parsed


def _strip_html_text(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html_unescape(text)
    return clean_text(text)


def _fetch_text_url(session: requests.Session, url: str, timeout_seconds: int = 20) -> tuple[str, str]:
    try:
        response = session.get(url, timeout=timeout_seconds, headers={"User-Agent": V3.USER_AGENT}, allow_redirects=True)
        response.raise_for_status()
    except Exception:
        return "", ""
    content_type = str(response.headers.get("Content-Type") or "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return str(response.url or url), ""
    text = response.text or ""
    if len(text) > 250000:
        text = text[:250000]
    return str(response.url or url), text


def _ddg_lite_search(session: requests.Session, query: str, max_results: int = 5) -> list[dict[str, Any]]:
    query = clean_text(query)
    if not query:
        return []
    try:
        response = session.post(
            V3.DDG_SEARCH_URL,
            data={"q": query, "kl": "uk-en"},
            timeout=30,
            headers={"User-Agent": V3.USER_AGENT},
        )
        response.raise_for_status()
        html = response.text or ""
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for anchor_match in re.finditer(r"(<a[^>]*class=['\"]result-link['\"][^>]*>.*?</a>)", html, re.I | re.S):
        anchor_html = anchor_match.group(1)
        href_match = re.search(r'href=\"([^\"]+)\"', anchor_html, re.I)
        if not href_match:
            continue
        url = href_match.group(1)
        title = _strip_html_text(anchor_html)
        trailing = html[anchor_match.end() : anchor_match.end() + 2000]
        snippet_match = re.search(r"<td class=['\"]result-snippet['\"]>(.*?)</td>", trailing, re.I | re.S)
        snippet_html = snippet_match.group(1) if snippet_match else ""
        title = _strip_html_text(anchor_html)
        snippet = _strip_html_text(snippet_html or "")
        results.append(
            {
                "url": clean_text(url),
                "title": title,
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break
    return results


def _extract_web_page_evidence(url: str, html: str) -> dict[str, Any]:
    plain_text = _strip_html_text(html)
    postcodes = []
    seen_postcodes: set[str] = set()
    for match in UK_POSTCODE_RE.findall((html or "") + "\n" + plain_text):
        normalized = clean_text(match).upper()
        if normalized and normalized not in seen_postcodes:
            seen_postcodes.add(normalized)
            postcodes.append(normalized)
        if len(postcodes) >= 8:
            break

    lat_lng_pairs: list[dict[str, float]] = []
    seen_pairs: set[tuple[float, float]] = set()
    for lat_text, lng_text in LAT_LNG_PAIR_RE.findall((html or "")[:120000]):
        try:
            lat = float(lat_text)
            lng = float(lng_text)
        except Exception:
            continue
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            continue
        pair_key = (round(lat, 6), round(lng, 6))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        easting, northing = V3._google_latlng_to_27700(lat, lng)
        lat_lng_pairs.append(
            {
                "lat": lat,
                "lng": lng,
                "easting_27700": easting,
                "northing_27700": northing,
            }
        )
        if len(lat_lng_pairs) >= 6:
            break

    webgis_links = []
    seen_links: set[str] = set()
    for link in WEBGIS_LINK_RE.findall(html or ""):
        link = clean_text(link)
        if link and link not in seen_links:
            seen_links.add(link)
            webgis_links.append(link)
        if len(webgis_links) >= 6:
            break

    excerpt = clean_text(plain_text[:1800])
    return {
        "url": url,
        "excerpt": excerpt,
        "postcodes": postcodes,
        "lat_lng_pairs": lat_lng_pairs,
        "webgis_links": webgis_links,
    }


def _build_web_research_prompt(row: dict[str, Any], ddg_results: list[dict[str, Any]], page_evidence: list[dict[str, Any]]) -> str:
    lines = [
        "你是一个英国历史地址定位研究助手。",
        f"原始历史地址：{clean_text(row.get('original_address'))}",
        f"当前最佳来源：{clean_text(row.get('best_source_final'))}",
        f"当前最佳地址：{clean_text(row.get('best_address_final'))}",
        f"当前置信度：{row.get('best_confidence')}",
        f"词库道路锚点：{clean_text(row.get('lexicon_road'))}",
        f"OS Gemini 地址：{clean_text(row.get('os_address_gemini'))}",
        f"Google 地址：{clean_text(row.get('gog_best_address'))}",
        "",
        "下面是把原始地址直接输入 DuckDuckGo lite 后的搜索结果：",
    ]
    if ddg_results:
        for index, result in enumerate(ddg_results, start=1):
            lines.append(f"{index}. 标题：{clean_text(result.get('title'))}")
            lines.append(f"   URL：{clean_text(result.get('url'))}")
            lines.append(f"   摘要：{clean_text(result.get('snippet'))}")
    else:
        lines.append("没有搜索结果。")
    lines.append("")
    lines.append("下面是抓取部分网页后的证据提取：")
    if page_evidence:
        for index, item in enumerate(page_evidence, start=1):
            lines.append(f"页面{index} URL：{clean_text(item.get('url'))}")
            if item.get("postcodes"):
                lines.append(f"页面{index} 邮编：{', '.join(item.get('postcodes') or [])}")
            if item.get("lat_lng_pairs"):
                pairs = item.get("lat_lng_pairs") or []
                pair_text = "; ".join(
                    f"lat={pair.get('lat')}, lng={pair.get('lng')}, e={pair.get('easting_27700')}, n={pair.get('northing_27700')}"
                    for pair in pairs[:3]
                )
                lines.append(f"页面{index} 坐标线索：{pair_text}")
            if item.get("webgis_links"):
                lines.append(f"页面{index} webgis/map链接：{'; '.join((item.get('webgis_links') or [])[:3])}")
            lines.append(f"页面{index} 摘要：{clean_text(item.get('excerpt'))}")
            lines.append("")
    else:
        lines.append("没有网页证据。")
        lines.append("")
    lines.append("请判断这些网页证据是否提供了比当前最佳结果更明确的定位线索。")
    lines.append("如果没有足够证据，请返回 no_improvement。")
    lines.append("如果有，请尽量返回更明确的地址、邮编、坐标或支持链接。")
    lines.append("只返回 JSON：")
    lines.append(
        '{"status":"improved或no_improvement","best_address":"",'
        '"postcode":"",'
        '"best_easting_27700":123456.0或null,'
        '"best_northing_27700":654321.0或null,'
        '"lat":53.12345或null,'
        '"lng":-1.23456或null,'
        '"supporting_url":"",'
        '"reason":"一句中文理由"}'
    )
    return "\n".join(lines)


def _parse_web_research_response(text: str) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return {
            "status": "parse_error",
            "best_address": "",
            "postcode": "",
            "best_easting_27700": None,
            "best_northing_27700": None,
            "lat": None,
            "lng": None,
            "supporting_url": "",
            "reason": "",
        }
    best_easting = V3._to_float(payload.get("best_easting_27700"))
    best_northing = V3._to_float(payload.get("best_northing_27700"))
    lat = V3._to_float(payload.get("lat"))
    lng = V3._to_float(payload.get("lng"))
    if (best_easting is None or best_northing is None) and lat is not None and lng is not None:
        best_easting, best_northing = V3._google_latlng_to_27700(lat, lng)
    return {
        "status": clean_text(payload.get("status")).lower() or "no_improvement",
        "best_address": clean_text(payload.get("best_address")),
        "postcode": clean_text(payload.get("postcode")).upper(),
        "best_easting_27700": best_easting,
        "best_northing_27700": best_northing,
        "lat": lat,
        "lng": lng,
        "supporting_url": clean_text(payload.get("supporting_url")),
        "reason": clean_text(payload.get("reason")),
    }


def _call_gemini_web_research(
    gemini_api_key: str,
    gemini_model: str,
    prompt: str,
    service_tier: str = "standard",
    timeout_ms: int = 60000,
) -> dict[str, Any]:
    model_name = normalize_gemini_model(gemini_model)
    tier = clean_text(service_tier).lower()
    use_flex = tier == "flex"
    timeout_ms = max(1000, int(timeout_ms or 60000))
    try:
        text = ""
        if V3._google_genai is not None and not use_flex:
            client = V3._google_genai.Client(api_key=gemini_api_key)
            config = {"http_options": {"timeout": timeout_ms}}
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            text = clean_text(getattr(response, "text", "") or "")
        if not text:
            body: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
            timeout_seconds = max(60, int(math.ceil(timeout_ms / 1000)))
            if use_flex:
                body["service_tier"] = "flex"
                headers["X-Server-Timeout"] = str(timeout_seconds)
            response = requests.post(
                GEMINI_GENERATE_CONTENT_URL.format(model=model_name),
                params={"key": gemini_api_key},
                headers=headers,
                json=body,
                timeout=timeout_seconds + 5,
            )
            response.raise_for_status()
            payload = response.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = clean_text(" ".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)))
    except Exception as exc:
        return {
            "status": "error",
            "best_address": "",
            "postcode": "",
            "best_easting_27700": None,
            "best_northing_27700": None,
            "lat": None,
            "lng": None,
            "supporting_url": "",
            "reason": _sanitize_api_error(exc),
        }
    parsed = _parse_web_research_response(text)
    if parsed["status"] not in {"improved", "no_improvement"}:
        parsed["status"] = "improved" if parsed.get("best_address") or parsed.get("best_easting_27700") else "parse_error"
    return parsed


def _match_returned_candidate(gemini_result: dict[str, Any], os_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_address = clean_text(gemini_result.get("best_address"))
    if not best_address:
        return None
    best_key = V3._norm_compact(best_address)
    for candidate in os_candidates:
        if V3._norm_compact(candidate.get("address")) == best_key:
            return candidate
    target_e = V3._to_float(gemini_result.get("best_easting_27700"))
    target_n = V3._to_float(gemini_result.get("best_northing_27700"))
    if target_e is None or target_n is None:
        return None
    for candidate in os_candidates:
        cand_e = V3._to_float(candidate.get("easting_27700"))
        cand_n = V3._to_float(candidate.get("northing_27700"))
        if cand_e == target_e and cand_n == target_n:
            return candidate
    return None


def _compact(value: Any) -> str:
    return V3._norm_compact(value)


def _tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_text(value).lower())


def _extract_raw_number_tokens(raw_address: str) -> set[str]:
    match = V3.LEADING_HOUSE_RANGE_RE.match(raw_address)
    if match and not (match.group(3) or "").strip():
        start = int(match.group(1))
        end = int(match.group(2))
        if end < start:
            start, end = end, start
        if 2 <= (end - start + 1) <= 10:
            return {str(number) for number in range(start, end + 1)}
    single_match = re.match(r"^\s*(\d+[A-Z]?)\b", raw_address.upper())
    if single_match:
        return {single_match.group(1)}
    labeled_numbers = {
        token.upper()
        for token in re.findall(r"\b(?:UNIT|NO|NUMBER)\s+(\d+[A-Z]?)\b", raw_address.upper())
    }
    if labeled_numbers:
        return labeled_numbers
    if not (_is_reference_style_address(raw_address) or _is_plot_style_address(raw_address)):
        postcode_stripped = re.sub(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", " ", raw_address.upper())
        inline_numbers = {token.upper() for token in re.findall(r"\b\d+[A-Z]?\b", postcode_stripped)}
        if inline_numbers:
            return inline_numbers
    if _is_reference_style_address(raw_address) or _is_plot_style_address(raw_address):
        return {token.upper() for token in re.findall(r"\b\d+[A-Z]?\b", raw_address.upper())}
    return set()


def _extract_candidate_number_tokens(candidate_address: str) -> set[str]:
    return {token.upper() for token in re.findall(r"\b\d+[A-Z]?\b", candidate_address.upper())}


def _number_bases(tokens: set[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        match = re.match(r"^(\d+)", str(token).upper())
        if match:
            out.add(match.group(1))
    return out


def _is_reference_style_address(raw_address: str) -> bool:
    lowered = clean_text(raw_address).lower()
    return any(
        token in lowered
        for token in [
            "land ",
            "land off",
            "plot ",
            "plots ",
            "forecourt",
            "junction",
            "adjacent",
            "adj ",
            "rear of",
            "part of",
            "site ",
            "site at",
            "off ",
        ]
    )


def _is_range_address(raw_address: str) -> bool:
    match = V3.LEADING_HOUSE_RANGE_RE.match(clean_text(raw_address))
    return bool(match and not (match.group(3) or "").strip())


def _anchor_gate_limit_m(raw_address: str, has_numbers: bool) -> float:
    if _is_reference_style_address(raw_address):
        return 250.0
    if _is_range_address(raw_address):
        return 120.0
    if not has_numbers:
        return 200.0
    return 120.0


def _build_site_tokens(raw_address: str, lexicon_road: str, lexicon_place: str, council: str, county: str) -> set[str]:
    tokens = set(_tokenize(raw_address))
    tokens -= set(_tokenize(lexicon_road))
    tokens -= set(_tokenize(lexicon_place))
    tokens -= set(_tokenize(council))
    tokens -= set(_tokenize(county))
    tokens -= {"notts", "nottinghamshire", "mans", "mansfieldshire"}
    return {token for token in tokens if token not in GENERIC_OS_TOKENS and not token.isdigit()}


def _is_generic_os_address(candidate_address: str) -> bool:
    return clean_text(candidate_address).upper().startswith(GENERIC_OS_ADDRESS_MARKERS)


def _split_lexicon_road(lexicon_road: str) -> tuple[str, str]:
    parts = [clean_text(part) for part in clean_text(lexicon_road).split("|")]
    road = parts[0] if parts else ""
    place = parts[1] if len(parts) > 1 else ""
    return road, place


def _raw_place_hints(raw_address: str, lexicon_road: str) -> list[str]:
    hints: list[str] = []
    road_key = V3._norm_road_compact(lexicon_road)
    for segment in [clean_text(part) for part in clean_text(raw_address).split(",") if clean_text(part)]:
        seg_key = V3._norm_road_compact(segment)
        if road_key and seg_key == road_key:
            continue
        if road_key and road_key and road_key in seg_key:
            continue
        if segment.lower().startswith(("plot ", "plots ")):
            continue
        if segment and segment not in hints:
            hints.append(segment)
    if hints and clean_text(hints[0]) == clean_text(raw_address):
        return []
    if hints:
        first = clean_text(hints[0])
        raw_first = clean_text(raw_address.split(",")[0])
        if first == raw_first:
            hints = hints[1:]
    return hints


def _candidate_road_matches(lexicon_road: str, candidate_address: str, candidate_road: str = "") -> bool:
    road = clean_text(lexicon_road)
    road_key = V3._norm_road_compact(road)
    cand_value = clean_text(candidate_road) or clean_text(candidate_address)
    cand_key = V3._norm_road_compact(cand_value)
    if road_key and cand_key and (road_key == cand_key or road_key in cand_key):
        return True
    return bool(road and cand_value and V3._road_similarity(road, cand_value) >= 0.94)


def _single_place_match(place: str, candidate_address: str, candidate_place: str = "") -> bool:
    place = clean_text(place)
    if not place:
        return False
    place_key = V3._norm_compact(place)
    cand_value = clean_text(candidate_place) or clean_text(candidate_address)
    cand_key = V3._norm_compact(cand_value)
    if place_key in cand_key:
        return True
    place_tokens = [token for token in _tokenize(place) if len(token) >= 4 and token not in {"market"}]
    cand_tokens = set(_tokenize(cand_value)) | set(_tokenize(candidate_address))
    return bool(place_tokens and any(token in cand_tokens for token in place_tokens))


def _candidate_place_matches(raw_address: str, lexicon_place: str, lexicon_road: str, candidate_address: str, candidate_place: str = "") -> bool:
    place_hints = [clean_text(lexicon_place)] + _raw_place_hints(raw_address, lexicon_road)
    seen: set[str] = set()
    filtered_hints: list[str] = []
    for hint in place_hints:
        key = V3._norm_compact(hint)
        if key and key not in seen:
            seen.add(key)
            filtered_hints.append(hint)
    if not filtered_hints:
        return True
    return any(_single_place_match(hint, candidate_address, candidate_place) for hint in filtered_hints)


def _has_input_number(raw_address: str) -> bool:
    return bool(_extract_raw_number_tokens(raw_address))


def _number_matches_candidate(raw_address: str, candidate_address: str) -> bool:
    raw_numbers = _extract_raw_number_tokens(raw_address)
    candidate_numbers = _extract_candidate_number_tokens(candidate_address)
    if not raw_numbers:
        return True
    if raw_numbers & candidate_numbers:
        return True
    raw_bases = _number_bases(raw_numbers)
    candidate_bases = _number_bases(candidate_numbers)
    if raw_bases & candidate_bases:
        return True
    if _is_range_address(raw_address):
        match = V3.LEADING_HOUSE_RANGE_RE.match(clean_text(raw_address))
        if match and candidate_bases:
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start:
                start, end = end, start
            for base in candidate_bases:
                try:
                    value = int(base)
                except Exception:
                    continue
                if start <= value <= end:
                    return True
    return False


def _site_tokens_overlap(raw_address: str, lexicon_road: str, lexicon_place: str, candidate_address: str) -> int:
    return len(_build_site_tokens(raw_address, lexicon_road, lexicon_place, "", "") & set(_tokenize(candidate_address)))


def _site_prefix_overlap(raw_address: str, lexicon_road: str, lexicon_place: str, candidate_address: str) -> int:
    first_segment = clean_text(clean_text(candidate_address).split(",")[0])
    if not first_segment:
        return 0
    return len(_build_site_tokens(raw_address, lexicon_road, lexicon_place, "", "") & set(_tokenize(first_segment)))


def _is_plot_style_address(raw_address: str) -> bool:
    lowered = clean_text(raw_address).lower()
    return lowered.startswith("plot ") or lowered.startswith("plots ")


def _is_complex_site_address(raw_address: str) -> bool:
    lowered = clean_text(raw_address).lower()
    return any(
        token in lowered
        for token in [
            "unit ",
            "business park",
            "industrial estate",
            " centre",
            " center",
            " depot",
            " works",
        ]
    )


def _is_road_only_address(raw_address: str, lexicon_road: str, lexicon_place: str) -> bool:
    if _is_reference_style_address(raw_address) or _is_plot_style_address(raw_address):
        return False
    if _has_input_number(raw_address):
        return False
    leading = clean_text(clean_text(raw_address).split(",")[0])
    if leading and lexicon_road and V3._road_similarity(leading, lexicon_road) >= 0.94:
        return True
    return not _build_site_tokens(raw_address, lexicon_road, lexicon_place, "", "")


def _address_category(raw_address: str, lexicon_road: str, lexicon_place: str) -> str:
    if _is_range_address(raw_address) or _is_plot_style_address(raw_address):
        return "range"
    if _is_reference_style_address(raw_address):
        return "reference"
    if _is_road_only_address(raw_address, lexicon_road, lexicon_place):
        return "road_only"
    return "exact"


def _find_candidate_for_phrase(candidates: list[Any], phrase: str) -> Any | None:
    variants = [clean_text(phrase)]
    stripped_number = re.sub(r"^\d+[A-Z]?\s+", "", clean_text(phrase), flags=re.I)
    stripped_plot = re.sub(r"^(?:plot|plots)\s+\d+[A-Z]?\s+", "", clean_text(phrase), flags=re.I)
    for variant in [stripped_number, stripped_plot]:
        if clean_text(variant) and clean_text(variant) not in variants:
            variants.append(clean_text(variant))
    phrase_keys = [V3._norm_road_compact(value) for value in variants if V3._norm_road_compact(value)]
    if not phrase_keys:
        return None
    for candidate in candidates:
        matched_phrase_key = V3._norm_road_compact(getattr(candidate, "matched_phrase", ""))
        road_name_key = V3._norm_road_compact(getattr(candidate, "road_name", ""))
        if any(key == matched_phrase_key or key == road_name_key for key in phrase_keys):
            return candidate
    return None


def _reference_road_hierarchy_override(
    raw_address: str,
    candidates: list[Any],
    debug: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    lowered = clean_text(raw_address).lower()
    if not _is_reference_style_address(raw_address) or " off " not in lowered:
        return None, None
    road_like_segments = [clean_text(value) for value in (debug.get("road_like_segments") or []) if clean_text(value)]
    if len(road_like_segments) < 2:
        return None, None
    primary_phrase = road_like_segments[0]
    access_phrase = road_like_segments[1]
    primary_candidate = _find_candidate_for_phrase(candidates, primary_phrase)
    access_candidate = _find_candidate_for_phrase(candidates, access_phrase)
    if not primary_candidate or not access_candidate:
        return None, None
    if clean_text(getattr(primary_candidate, "populated_place", "")) != clean_text(getattr(access_candidate, "populated_place", "")):
        return None, None
    if clean_text(getattr(primary_candidate, "match_type", "")) != "exact":
        return None, None
    if clean_text(getattr(access_candidate, "match_type", "")) != "exact":
        return None, None
    if getattr(primary_candidate, "road_name", "") == getattr(access_candidate, "road_name", ""):
        return None, None
    return primary_candidate, access_candidate


def _compose_reference_query_variants(
    raw_address: str,
    primary_match: Any | None,
    access_match: Any | None,
    council_name: str,
) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        query = clean_text(value)
        key = V3._norm_compact(query)
        if query and key and key not in seen:
            seen.add(key)
            variants.append(query)

    add(raw_address)
    if primary_match is not None:
        primary_place = clean_text(getattr(primary_match, "populated_place", "")) or council_name
        add(f"{clean_text(getattr(primary_match, 'road_name', ''))}, {primary_place}")
        if access_match is not None:
            add(
                f"{clean_text(getattr(primary_match, 'road_name', ''))}, "
                f"{clean_text(getattr(access_match, 'road_name', ''))}, {primary_place}"
            )
    return variants


def _parse_candidate_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _find_candidate_by_address(best_address: str, candidate_json: Any) -> dict[str, Any] | None:
    best_key = V3._norm_compact(best_address)
    if not best_key:
        return None
    for candidate in _parse_candidate_json(candidate_json):
        if V3._norm_compact(candidate.get("address")) == best_key:
            return candidate
    return None


def _source_candidate_valid(
    source: str,
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    candidate_address: str,
    category: str,
    candidate: dict[str, Any] | None = None,
) -> bool:
    candidate_address = clean_text(candidate_address)
    if not candidate_address:
        return False
    candidate_road = clean_text((candidate or {}).get("road_name"))
    candidate_place = clean_text((candidate or {}).get("place"))
    road_ok = (not lexicon_road) or _candidate_road_matches(lexicon_road, candidate_address, candidate_road)
    place_ok = _candidate_place_matches(raw_address, lexicon_place, lexicon_road, candidate_address, candidate_place)
    site_overlap = _site_tokens_overlap(raw_address, lexicon_road, lexicon_place, candidate_address)
    site_prefix_overlap = _site_prefix_overlap(raw_address, lexicon_road, lexicon_place, candidate_address)
    road_level = _candidate_is_road_level(candidate_address, candidate_road)

    if source == "os" and category == "exact" and site_prefix_overlap >= 2:
        return True
    if source == "os" and category == "exact" and site_overlap >= 2 and (road_ok or place_ok):
        return True

    if not road_ok:
        return False
    if not place_ok:
        return False
    if category in {"road_only", "reference", "range"}:
        return True
    if _has_input_number(raw_address) and _number_matches_candidate(raw_address, candidate_address):
        return True
    if source == "gog" and _is_complex_site_address(raw_address) and not road_level:
        return True
    if source == "gog" and not _has_input_number(raw_address) and _build_site_tokens(raw_address, lexicon_road, lexicon_place, "", "") and not road_level:
        return True
    if site_overlap > 0:
        return True
    return not _has_input_number(raw_address) and not _build_site_tokens(raw_address, lexicon_road, lexicon_place, "", "")


def _candidate_is_specific(
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    candidate_address: str,
) -> bool:
    if _has_input_number(raw_address) and _number_matches_candidate(raw_address, candidate_address):
        return True
    if _site_prefix_overlap(raw_address, lexicon_road, lexicon_place, candidate_address) >= 2:
        return True
    return _site_tokens_overlap(raw_address, lexicon_road, lexicon_place, candidate_address) >= 2


def _candidate_is_road_level(candidate_address: str, candidate_road: str = "") -> bool:
    first_segment = clean_text(clean_text(candidate_address).split(",")[0])
    road = clean_text(candidate_road)
    if not first_segment:
        return False
    if road and V3._road_similarity(first_segment, road) >= 0.94:
        return True
    return bool(road and V3._norm_road_compact(first_segment) == V3._norm_road_compact(road))


def _fallback_pool_is_compact(row: dict[str, Any]) -> bool:
    pool = _parse_candidate_json(row.get("os_fallback_pool_json"))[:3]
    if not pool:
        return False
    if len(pool) == 1:
        return True
    road_keys = [V3._norm_road_compact(candidate.get("road_name") or candidate.get("address")) for candidate in pool]
    road_keys = [key for key in road_keys if key]
    if road_keys and len(set(road_keys)) != 1:
        return False
    generic_count = sum(1 for candidate in pool if _is_generic_os_address(candidate.get("address")))
    if generic_count >= 2 and not _is_reference_style_address(clean_text(row.get("original_address"))):
        return False
    return True


def _candidate_point(entry_source: str, family: str, address: Any, easting: Any, northing: Any, score: int = 0) -> dict[str, Any] | None:
    x = V3._to_float(easting)
    y = V3._to_float(northing)
    if x is None or y is None:
        return None
    return {
        "source": entry_source,
        "family": family,
        "address": clean_text(address),
        "x": float(x),
        "y": float(y),
        "score": int(score or 0),
    }


def _point_distance_m(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))


def _cluster_consensus_points(points: list[dict[str, Any]], threshold_m: float) -> list[list[dict[str, Any]]]:
    if not points:
        return []
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if _point_distance_m(points[left], points[right]) <= threshold_m:
                union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, point in enumerate(points):
        grouped.setdefault(find(index), []).append(point)
    return list(grouped.values())


def _agreement_threshold_m(category: str) -> float:
    if category == "exact":
        return 30.0
    if category == "road_only":
        return 80.0
    return 120.0


def _best_agreement_cluster(points: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    clusters = [
        cluster
        for cluster in _cluster_consensus_points(points, _agreement_threshold_m(category))
        if len(cluster) >= 2 and len({point["family"] for point in cluster}) >= 2
    ]
    if not clusters:
        return []
    clusters.sort(
        key=lambda cluster: (
            len({point["family"] for point in cluster}),
            len(cluster),
            max(int(point.get("score") or 0) for point in cluster),
        ),
        reverse=True,
    )
    return clusters[0]


def _cluster_radius_m(cluster: list[dict[str, Any]]) -> float:
    if len(cluster) <= 1:
        return 0.0
    max_distance = 0.0
    for left in range(len(cluster)):
        for right in range(left + 1, len(cluster)):
            max_distance = max(max_distance, _point_distance_m(cluster[left], cluster[right]))
    return round(max_distance, 1)


def _collect_agreement_points(
    row: dict[str, Any],
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    category: str,
    os_address: str,
    gog_address: str,
    os_candidate: dict[str, Any] | None,
    gog_candidate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    lexicon_point = _candidate_point(
        "lexicon",
        "lexicon",
        row.get("lexicon_road"),
        row.get("lexicon_easting_27700"),
        row.get("lexicon_northing_27700"),
        score=50,
    )
    if lexicon_point:
        points.append(lexicon_point)

    if _source_candidate_valid("os", raw_address, lexicon_road, lexicon_place, os_address, category, os_candidate):
        os_point = _candidate_point(
            "os",
            "os",
            os_address,
            row.get("os_address_gemini_easting_27700"),
            row.get("os_address_gemini_northing_27700"),
            score=90,
        )
        if os_point:
            points.append(os_point)

    if _source_candidate_valid("gog", raw_address, lexicon_road, lexicon_place, gog_address, category, gog_candidate):
        gog_point = _candidate_point(
            "gog",
            "gog",
            gog_address,
            row.get("gog_best_easting_27700"),
            row.get("gog_best_northing_27700"),
            score=80,
        )
        if gog_point:
            points.append(gog_point)

    for index, candidate in enumerate(_parse_candidate_json(row.get("os_fallback_pool_json"))[:3], start=1):
        fallback_point = _candidate_point(
            f"fallback_{index}",
            "fallback",
            candidate.get("address"),
            candidate.get("easting_27700"),
            candidate.get("northing_27700"),
            score=int(candidate.get("fallback_score") or 0),
        )
        if fallback_point:
            points.append(fallback_point)
    return points


def _reference_consensus_override(
    row: dict[str, Any],
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    category: str,
    os_address: str,
    gog_address: str,
    os_candidate: dict[str, Any] | None,
    gog_candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if category not in {"reference", "range"}:
        return None
    if clean_text(row.get("best_source_final")) != "lexicon":
        return None
    if int(row.get("best_confidence") or 0) >= 70:
        return None

    points = _collect_agreement_points(
        row=row,
        raw_address=raw_address,
        lexicon_road=lexicon_road,
        lexicon_place=lexicon_place,
        category=category,
        os_address=os_address,
        gog_address=gog_address,
        os_candidate=os_candidate,
        gog_candidate=gog_candidate,
    )
    if len(points) < 2:
        return None

    cluster = _best_agreement_cluster(points, category)
    if not cluster:
        return None
    if "lexicon" not in {point["family"] for point in cluster}:
        return None

    weights_map = {"lexicon": 1.0, "fallback": 2.0, "os": 2.1, "gog": 1.8}
    weighted_total = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    addresses: list[str] = []
    for point in cluster:
        weight = float(weights_map.get(point["family"], 1.0))
        weighted_total += weight
        weighted_x += point["x"] * weight
        weighted_y += point["y"] * weight
        if point["address"]:
            addresses.append(point["address"])
    if weighted_total <= 0:
        return None
    return {
        "source": "consensus",
        "address": " / ".join(addresses),
        "easting_27700": round(weighted_x / weighted_total, 3),
        "northing_27700": round(weighted_y / weighted_total, 3),
        "reason": "consensus_reference_cluster",
        "agreement_count": len(cluster),
        "agreement_sources": ",".join(sorted({point["family"] for point in cluster})),
        "agreement_radius_m": _cluster_radius_m(cluster),
    }


def _clamp_confidence(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _compute_best_confidence(
    chosen: str,
    category: str,
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    chosen_address: str,
    chosen_candidate: dict[str, Any] | None,
    os_valid: bool,
    gog_valid: bool,
    fallback_valid: bool,
    fallback_compact: bool,
) -> tuple[int, str]:
    reasons: list[str] = []
    if chosen == "none":
        return 0, "no_selection"

    has_number = _has_input_number(raw_address)
    site_tokens = _build_site_tokens(raw_address, lexicon_road, lexicon_place, "", "")
    site_overlap = _site_tokens_overlap(raw_address, lexicon_road, lexicon_place, chosen_address)
    site_prefix_overlap = _site_prefix_overlap(raw_address, lexicon_road, lexicon_place, chosen_address)
    candidate_road = clean_text((chosen_candidate or {}).get("road_name"))
    candidate_place = clean_text((chosen_candidate or {}).get("place"))
    road_ok = (not lexicon_road) or _candidate_road_matches(lexicon_road, chosen_address, candidate_road)
    place_ok = _candidate_place_matches(raw_address, lexicon_place, lexicon_road, chosen_address, candidate_place)
    number_ok = _number_matches_candidate(raw_address, chosen_address)
    road_level = _candidate_is_road_level(chosen_address, candidate_road)
    generic = _is_generic_os_address(chosen_address)

    if chosen == "os":
        score = 58.0
        reasons.append("os_selected")
        if road_ok:
            score += 9.0
            reasons.append("road_match")
        if place_ok:
            score += 8.0
            reasons.append("place_match")
        if has_number:
            if number_ok:
                score += 12.0
                reasons.append("number_match")
            else:
                score -= 20.0
                reasons.append("number_miss")
        if site_prefix_overlap >= 2:
            score += 8.0
            reasons.append("site_prefix_strong")
        elif site_overlap > 0:
            score += 4.0
            reasons.append("site_token_match")
        if generic:
            score -= 14.0
            reasons.append("generic_record")
        if category in {"reference", "range"} and number_ok:
            score += 5.0
            reasons.append("reference_number_anchor")
        if category == "road_only":
            score -= 8.0
            reasons.append("road_only_specific_point")
        return _clamp_confidence(score), ",".join(reasons)

    if chosen == "gog":
        score = 52.0
        reasons.append("google_selected")
        if road_ok:
            score += 9.0
            reasons.append("road_match")
        if place_ok:
            score += 8.0
            reasons.append("place_match")
        if has_number:
            if number_ok:
                score += 11.0
                reasons.append("number_match")
            else:
                score -= 20.0
                reasons.append("number_miss")
        if site_prefix_overlap >= 2:
            score += 8.0
            reasons.append("site_prefix_strong")
        elif site_overlap > 0:
            score += 4.0
            reasons.append("site_token_match")
        if road_level and category == "exact" and site_tokens:
            score -= 10.0
            reasons.append("road_level_google")
        if category in {"reference", "range"} and _candidate_is_specific(raw_address, lexicon_road, lexicon_place, chosen_address):
            score += 5.0
            reasons.append("reference_specific")
        return _clamp_confidence(score), ",".join(reasons)

    if chosen == "lexicon":
        score = 54.0
        reasons.append("lexicon_selected")
        if category == "road_only":
            score += 16.0
            reasons.append("road_only_anchor")
        elif category in {"reference", "range"}:
            score += 6.0
            reasons.append("reference_anchor")
        if has_number:
            score -= 16.0
            reasons.append("missing_property_number")
        if site_tokens:
            score -= 8.0
            reasons.append("site_name_unresolved")
        if not os_valid and not gog_valid:
            score += 8.0
            reasons.append("no_stronger_online_match")
        else:
            score -= 6.0
            reasons.append("specific_candidate_exists")
        return _clamp_confidence(score), ",".join(reasons)

    if chosen == "fallback":
        score = 44.0
        reasons.append("fallback_selected")
        if fallback_valid:
            score += 4.0
            reasons.append("fallback_available")
        if fallback_compact:
            score += 12.0
            reasons.append("fallback_compact")
        if category in {"reference", "range"}:
            score += 8.0
            reasons.append("reference_nearby_cluster")
        if category == "exact":
            score -= 8.0
            reasons.append("exact_without_exact_match")
        return _clamp_confidence(score), ",".join(reasons)

    if chosen == "consensus":
        score = 60.0
        reasons.append("consensus_selected")
        if road_ok:
            score += 8.0
            reasons.append("road_match")
        if place_ok:
            score += 8.0
            reasons.append("place_match")
        if category in {"reference", "range"}:
            score += 8.0
            reasons.append("reference_consensus")
        if fallback_compact:
            score += 6.0
            reasons.append("fallback_compact")
        if os_valid:
            score += 4.0
            reasons.append("os_support")
        if gog_valid:
            score += 3.0
            reasons.append("google_support")
        return _clamp_confidence(score), ",".join(reasons)

    return 0, "no_selection"


def _apply_choice_to_row(
    row: dict[str, Any],
    chosen: str,
    lexicon_road_full: str,
    os_address: str,
    gog_address: str,
    consensus_override: dict[str, Any] | None = None,
) -> None:
    row["best_source_final"] = chosen
    if chosen == "os":
        row["best_address_final"] = os_address
        row["best_easting_27700_final"] = row.get("os_address_gemini_easting_27700") or ""
        row["best_northing_27700_final"] = row.get("os_address_gemini_northing_27700") or ""
    elif chosen == "gog":
        row["best_address_final"] = gog_address
        row["best_easting_27700_final"] = row.get("gog_best_easting_27700") or ""
        row["best_northing_27700_final"] = row.get("gog_best_northing_27700") or ""
    elif chosen == "fallback":
        fallback_address = clean_text(row.get("os_fallback_1_address"))
        row["best_address_final"] = fallback_address
        row["best_easting_27700_final"] = row.get("os_fallback_1_easting_27700") or ""
        row["best_northing_27700_final"] = row.get("os_fallback_1_northing_27700") or ""
    elif chosen == "lexicon":
        row["best_address_final"] = lexicon_road_full
        row["best_easting_27700_final"] = row.get("lexicon_easting_27700") or ""
        row["best_northing_27700_final"] = row.get("lexicon_northing_27700") or ""
    elif chosen == "consensus" and consensus_override:
        row["best_address_final"] = consensus_override.get("address") or ""
        row["best_easting_27700_final"] = consensus_override.get("easting_27700") or ""
        row["best_northing_27700_final"] = consensus_override.get("northing_27700") or ""
    else:
        row["best_address_final"] = ""
        row["best_easting_27700_final"] = ""
        row["best_northing_27700_final"] = ""


def _apply_agreement_metrics(
    row: dict[str, Any],
    category: str,
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    os_address: str,
    gog_address: str,
    os_candidate: dict[str, Any] | None,
    gog_candidate: dict[str, Any] | None,
) -> None:
    points = _collect_agreement_points(
        row=row,
        raw_address=raw_address,
        lexicon_road=lexicon_road,
        lexicon_place=lexicon_place,
        category=category,
        os_address=os_address,
        gog_address=gog_address,
        os_candidate=os_candidate,
        gog_candidate=gog_candidate,
    )
    cluster = _best_agreement_cluster(points, category)
    if not cluster:
        row["best_agreement_count"] = 0
        row["best_agreement_sources"] = ""
        row["best_agreement_radius_m"] = ""
        return
    row["best_agreement_count"] = len(cluster)
    row["best_agreement_sources"] = ",".join(sorted({point["family"] for point in cluster}))
    row["best_agreement_radius_m"] = _cluster_radius_m(cluster)


def _pick_final_source_for_row(row: dict[str, Any]) -> None:
    raw_address = clean_text(row.get("original_address"))
    lexicon_road_full = clean_text(row.get("lexicon_road"))
    lexicon_road, lexicon_place = _split_lexicon_road(lexicon_road_full)
    category = _address_category(raw_address, lexicon_road, lexicon_place)
    os_address = clean_text(row.get("os_address_gemini"))
    gog_address = clean_text(row.get("gog_best_address"))
    fallback_address = clean_text(row.get("os_fallback_1_address"))
    os_candidate = _find_candidate_by_address(os_address, row.get("os_candidates_json"))
    gog_candidate = _find_candidate_by_address(gog_address, row.get("gog_candidates_json"))
    os_valid = _source_candidate_valid("os", raw_address, lexicon_road, lexicon_place, os_address, category, os_candidate)
    gog_valid = _source_candidate_valid("gog", raw_address, lexicon_road, lexicon_place, gog_address, category, gog_candidate)
    fallback_valid = bool(fallback_address)
    lexicon_valid = bool(lexicon_road_full and row.get("lexicon_easting_27700") not in ("", None) and row.get("lexicon_northing_27700") not in ("", None))
    os_specific = os_valid and _candidate_is_specific(raw_address, lexicon_road, lexicon_place, os_address)
    gog_specific = gog_valid and _candidate_is_specific(raw_address, lexicon_road, lexicon_place, gog_address)
    fallback_compact = fallback_valid and _fallback_pool_is_compact(row)

    chosen = "none"
    if category == "exact":
        if os_valid:
            chosen = "os"
        elif gog_valid:
            chosen = "gog"
        elif lexicon_valid:
            chosen = "lexicon"
        elif fallback_compact:
            chosen = "fallback"
        elif fallback_valid:
            chosen = "fallback"
    elif category == "road_only":
        if lexicon_valid:
            chosen = "lexicon"
        elif gog_valid:
            chosen = "gog"
        elif os_valid:
            chosen = "os"
        elif fallback_compact:
            chosen = "fallback"
        elif fallback_valid:
            chosen = "fallback"
    else:
        if os_specific:
            chosen = "os"
        elif gog_specific:
            chosen = "gog"
        elif fallback_compact:
            chosen = "fallback"
        elif lexicon_valid:
            chosen = "lexicon"
        elif os_valid:
            chosen = "os"
        elif gog_valid:
            chosen = "gog"
        elif fallback_valid:
            chosen = "fallback"

    row["best_selection_category"] = category
    row["best_os_valid"] = int(bool(os_valid))
    row["best_gog_valid"] = int(bool(gog_valid))
    row["best_fallback_valid"] = int(bool(fallback_valid))
    row["best_os_specific"] = int(bool(os_specific))
    row["best_gog_specific"] = int(bool(gog_specific))
    row["best_fallback_compact"] = int(bool(fallback_compact))
    _apply_agreement_metrics(
        row=row,
        category=category,
        raw_address=raw_address,
        lexicon_road=lexicon_road,
        lexicon_place=lexicon_place,
        os_address=os_address,
        gog_address=gog_address,
        os_candidate=os_candidate,
        gog_candidate=gog_candidate,
    )
    row["best_selection_reason"] = row.get("best_selection_reason") or ""
    _apply_choice_to_row(row, chosen, lexicon_road_full, os_address, gog_address)

    chosen_candidate = None
    if chosen == "os":
        chosen_candidate = os_candidate
    elif chosen == "gog":
        chosen_candidate = gog_candidate
    elif chosen == "fallback":
        pool = _parse_candidate_json(row.get("os_fallback_pool_json"))
        chosen_candidate = pool[0] if pool else None
    confidence, confidence_reason = _compute_best_confidence(
        chosen=chosen,
        category=category,
        raw_address=raw_address,
        lexicon_road=lexicon_road,
        lexicon_place=lexicon_place,
        chosen_address=clean_text(row.get("best_address_final")),
        chosen_candidate=chosen_candidate,
        os_valid=os_valid,
        gog_valid=gog_valid,
        fallback_valid=fallback_valid,
        fallback_compact=fallback_compact,
    )
    row["best_confidence"] = confidence
    row["best_confidence_reason"] = confidence_reason

    consensus_override = _reference_consensus_override(
        row=row,
        raw_address=raw_address,
        lexicon_road=lexicon_road,
        lexicon_place=lexicon_place,
        category=category,
        os_address=os_address,
        gog_address=gog_address,
        os_candidate=os_candidate,
        gog_candidate=gog_candidate,
    )
    if consensus_override:
        _apply_choice_to_row(row, "consensus", lexicon_road_full, os_address, gog_address, consensus_override=consensus_override)
        row["best_selection_reason"] = consensus_override.get("reason") or "consensus_override"
        row["best_agreement_count"] = consensus_override.get("agreement_count") or row.get("best_agreement_count") or 0
        row["best_agreement_sources"] = consensus_override.get("agreement_sources") or row.get("best_agreement_sources") or ""
        row["best_agreement_radius_m"] = consensus_override.get("agreement_radius_m") or row.get("best_agreement_radius_m") or ""
        confidence, confidence_reason = _compute_best_confidence(
            chosen="consensus",
            category=category,
            raw_address=raw_address,
            lexicon_road=lexicon_road,
            lexicon_place=lexicon_place,
            chosen_address=clean_text(row.get("best_address_final")),
            chosen_candidate=None,
            os_valid=os_valid,
            gog_valid=gog_valid,
            fallback_valid=fallback_valid,
            fallback_compact=fallback_compact,
        )
        row["best_confidence"] = confidence
        row["best_confidence_reason"] = confidence_reason


def _apply_parent_best_selection(rows: list[dict[str, Any]]) -> None:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_key = clean_text(row.get("_source_unique_key") or row.get("key"))
        by_source.setdefault(source_key, []).append(row)

    for source_key, group in by_source.items():
        parent = next((row for row in group if clean_text(row.get("key")) == source_key), None)
        if not parent:
            continue
        category = clean_text(parent.get("best_selection_category"))
        if category not in {"range", "reference"}:
            continue
        children = [row for row in group if clean_text(row.get("key")) != source_key]
        valid_children = [
            row
            for row in children
            if clean_text(row.get("best_source_final")) != "none"
            and row.get("best_easting_27700_final") not in ("", None)
            and row.get("best_northing_27700_final") not in ("", None)
        ]
        if not valid_children:
            continue

        child_points: list[dict[str, Any]] = []
        for child in valid_children:
            point = _candidate_point(
                clean_text(child.get("key")),
                clean_text(child.get("best_source_final")),
                child.get("best_address_final"),
                child.get("best_easting_27700_final"),
                child.get("best_northing_27700_final"),
                score=int(child.get("best_confidence") or 0),
            )
            if point:
                child_points.append(point)

        cluster = _best_agreement_cluster(child_points, category) if len(child_points) >= 2 else []
        if cluster:
            total_weight = 0.0
            total_x = 0.0
            total_y = 0.0
            addresses: list[str] = []
            for point in cluster:
                weight = max(1.0, float(point.get("score") or 0) / 25.0)
                total_weight += weight
                total_x += point["x"] * weight
                total_y += point["y"] * weight
                if point["address"]:
                    addresses.append(point["address"])
            if total_weight > 0:
                parent["best_source_final"] = "parent_consensus"
                parent["best_selection_category"] = category
                parent["best_address_final"] = " / ".join(addresses)
                parent["best_easting_27700_final"] = round(total_x / total_weight, 3)
                parent["best_northing_27700_final"] = round(total_y / total_weight, 3)
                parent["best_confidence"] = max(int(child.get("best_confidence") or 0) for child in valid_children)
                parent["best_confidence_reason"] = "parent_child_consensus"
                parent["best_selection_reason"] = "parent_from_child_consensus"
                parent["best_agreement_count"] = len(cluster)
                parent["best_agreement_sources"] = ",".join(sorted({point["family"] for point in cluster}))
                parent["best_agreement_radius_m"] = _cluster_radius_m(cluster)
                continue

        def _variant_sort_value(item: dict[str, Any]) -> tuple[int, str]:
            variant = clean_text(item.get("_expanded_variant"))
            match = re.match(r"^(\d+)", variant)
            return (int(match.group(1)) if match else 10**9, variant)

        valid_children.sort(key=_variant_sort_value)
        chosen_child = valid_children[len(valid_children) // 2]
        parent["best_source_final"] = clean_text(chosen_child.get("best_source_final")) or parent.get("best_source_final") or "none"
        parent["best_selection_category"] = category
        parent["best_address_final"] = chosen_child.get("best_address_final") or ""
        parent["best_easting_27700_final"] = chosen_child.get("best_easting_27700_final") or ""
        parent["best_northing_27700_final"] = chosen_child.get("best_northing_27700_final") or ""
        parent["best_confidence"] = chosen_child.get("best_confidence") or 0
        parent["best_confidence_reason"] = chosen_child.get("best_confidence_reason") or ""
        parent["best_selection_reason"] = f"parent_from_child_{clean_text(chosen_child.get('key'))}"


def _apply_final_best_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        _pick_final_source_for_row(row)
        row.setdefault("best_selection_reason", "")
    _apply_parent_best_selection(rows)
    return rows


def _research_single_low_confidence_row(row: dict[str, Any], args: argparse.Namespace, gemini_api_key: str) -> dict[str, Any]:
    row = dict(row)
    confidence = int(row.get("best_confidence") or 0)
    if confidence >= int(args.web_research_threshold or 75):
        row["web_research_status"] = "skipped_high_confidence"
        row["web_research_query"] = ""
        row["web_research_result_count"] = 0
        row["web_research_pages_checked"] = 0
        row["web_research_best_address"] = ""
        row["web_research_postcode"] = ""
        row["web_research_best_easting_27700"] = ""
        row["web_research_best_northing_27700"] = ""
        row["web_research_supporting_url"] = ""
        row["web_research_reason"] = ""
        row["web_research_results_json"] = "[]"
        row["web_research_pages_json"] = "[]"
        return row

    query = clean_text(row.get("original_address"))
    session = V3._build_session()
    try:
        ddg_results = _ddg_lite_search(session, query, max_results=max(1, int(args.web_research_results or 5)))
        page_evidence: list[dict[str, Any]] = []
        for result in ddg_results[: max(1, int(args.web_research_pages or 4))]:
            parsed = urlparse(clean_text(result.get("url")))
            if parsed.scheme not in {"http", "https"}:
                continue
            final_url, html = _fetch_text_url(session, clean_text(result.get("url")))
            if not html:
                continue
            page_evidence.append(_extract_web_page_evidence(final_url, html))

        prompt = _build_web_research_prompt(row, ddg_results, page_evidence)
        research = _call_gemini_web_research(
            gemini_api_key,
            args.gemini_model,
            prompt,
            args.gemini_service_tier,
            args.gemini_timeout_ms,
        )
        row["web_research_status"] = research.get("status") or "no_improvement"
        row["web_research_query"] = query
        row["web_research_result_count"] = len(ddg_results)
        row["web_research_pages_checked"] = len(page_evidence)
        row["web_research_best_address"] = research.get("best_address") or ""
        row["web_research_postcode"] = research.get("postcode") or ""
        row["web_research_best_easting_27700"] = research.get("best_easting_27700") or ""
        row["web_research_best_northing_27700"] = research.get("best_northing_27700") or ""
        row["web_research_supporting_url"] = research.get("supporting_url") or ""
        row["web_research_reason"] = research.get("reason") or ""
        row["web_research_results_json"] = json.dumps(ddg_results, ensure_ascii=False)
        row["web_research_pages_json"] = json.dumps(page_evidence, ensure_ascii=False)
        return row
    finally:
        try:
            session.close()
        except Exception:
            pass


def _apply_low_confidence_web_research(rows: list[dict[str, Any]], args: argparse.Namespace, gemini_api_key: str) -> list[dict[str, Any]]:
    if not args.enable_web_research:
        for row in rows:
            row["web_research_status"] = "disabled"
            row["web_research_query"] = ""
            row["web_research_result_count"] = 0
            row["web_research_pages_checked"] = 0
            row["web_research_best_address"] = ""
            row["web_research_postcode"] = ""
            row["web_research_best_easting_27700"] = ""
            row["web_research_best_northing_27700"] = ""
            row["web_research_supporting_url"] = ""
            row["web_research_reason"] = ""
            row["web_research_results_json"] = "[]"
            row["web_research_pages_json"] = "[]"
        return rows

    indexed_rows = list(enumerate(rows))
    if max(1, int(args.web_research_workers or 4)) <= 1:
        return [_research_single_low_confidence_row(row, args, gemini_api_key) for row in rows]

    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, int(args.web_research_workers or 4))) as executor:
        future_map = {
            executor.submit(_research_single_low_confidence_row, row, args, gemini_api_key): idx
            for idx, row in indexed_rows
        }
        total = len(future_map)
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            output_rows[future_map[future]] = future.result()
            if done_idx % 25 == 0 or done_idx == total:
                print(f"Web research progress: {done_idx}/{total}")
    return [row for row in output_rows if row is not None]


def _score_fallback_candidate(
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    lexicon_easting: Any,
    lexicon_northing: Any,
    candidate: dict[str, Any],
    council: str,
    county: str,
) -> tuple[int, list[str]]:
    candidate_address = clean_text(candidate.get("address"))
    candidate_road = clean_text(candidate.get("road_name"))
    candidate_place = clean_text(candidate.get("place"))
    raw_compact = _compact(raw_address)
    road_compact = _compact(lexicon_road)
    place_compact = _compact(lexicon_place)
    candidate_road_compact = _compact(candidate_road)
    candidate_place_compact = _compact(candidate_place)

    score = 0
    reasons: list[str] = []

    if road_compact and candidate_road_compact == road_compact:
        score += 70
        reasons.append("road_exact")
    elif road_compact and road_compact and road_compact in _compact(candidate_address):
        score += 55
        reasons.append("road_in_address")
    elif candidate_road_compact and candidate_road_compact in raw_compact:
        score += 45
        reasons.append("road_from_raw")

    if place_compact and candidate_place_compact == place_compact:
        score += 25
        reasons.append("place_exact")
    elif candidate_place_compact and candidate_place_compact in raw_compact:
        score += 20
        reasons.append("place_in_raw")
    elif candidate_place_compact and candidate_place_compact in _compact(council):
        score += 15
        reasons.append("place_matches_council")
    elif candidate_place_compact and candidate_place_compact in _compact(county):
        score += 10
        reasons.append("place_matches_county")

    raw_numbers = _extract_raw_number_tokens(raw_address)
    candidate_numbers = _extract_candidate_number_tokens(candidate_address)
    raw_number_bases = _number_bases(raw_numbers)
    candidate_number_bases = _number_bases(candidate_numbers)
    if raw_numbers and candidate_numbers:
        if _is_range_address(raw_address):
            if len(candidate_numbers) == 1 and candidate_number_bases & raw_number_bases:
                score += 30
                reasons.append("range_internal_single")
            elif candidate_number_bases and candidate_number_bases.issubset(raw_number_bases):
                score += 22
                reasons.append("range_internal_multi")
            elif candidate_number_bases & raw_number_bases:
                score += 12
                reasons.append("range_partial_overlap")
        elif raw_numbers & candidate_numbers:
            score += 20
            reasons.append("number_match")
        elif raw_number_bases & candidate_number_bases:
            score += 18
            reasons.append("number_base_match")
        elif len(raw_numbers) > 1:
            score += 5
            reasons.append("range_related")

    site_tokens = _build_site_tokens(raw_address, lexicon_road, lexicon_place, council, county)
    candidate_tokens = set(_tokenize(candidate_address))
    site_overlap = site_tokens & candidate_tokens
    if len(site_overlap) >= 2:
        score += 15
        reasons.append("site_overlap_multi")
    elif len(site_overlap) == 1:
        score += 8
        reasons.append("site_overlap_single")

    if _is_generic_os_address(candidate_address):
        score -= 8
        reasons.append("generic_infrastructure_penalty")

    anchor_distance = None
    cand_e = V3._to_float(candidate.get("easting_27700"))
    cand_n = V3._to_float(candidate.get("northing_27700"))
    lex_e = V3._to_float(lexicon_easting)
    lex_n = V3._to_float(lexicon_northing)
    if None not in (cand_e, cand_n, lex_e, lex_n):
        dx = float(cand_e) - float(lex_e)
        dy = float(cand_n) - float(lex_n)
        anchor_distance = (dx * dx + dy * dy) ** 0.5
        if anchor_distance <= 25:
            reasons.append("anchor_very_close")
        elif anchor_distance <= 50:
            reasons.append("anchor_close")
        elif anchor_distance <= 100:
            reasons.append("anchor_near")
        elif anchor_distance <= 200:
            reasons.append("anchor_mid")
        else:
            reasons.append("anchor_far_but_ok")

    return score, reasons


def _build_fallback_pool(
    raw_address: str,
    lexicon_road: str,
    lexicon_place: str,
    lexicon_easting: Any,
    lexicon_northing: Any,
    os_candidates: list[dict[str, Any]],
    council: str,
    county: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    reference_style = _is_reference_style_address(raw_address)
    range_style = _is_range_address(raw_address)
    has_numbers = bool(_extract_raw_number_tokens(raw_address))
    gate_limit = _anchor_gate_limit_m(raw_address, has_numbers)
    for candidate in os_candidates:
        score, reasons = _score_fallback_candidate(
            raw_address,
            lexicon_road,
            lexicon_place,
            lexicon_easting,
            lexicon_northing,
            candidate,
            council,
            county,
        )
        if score < 35:
            continue
        cand_e = V3._to_float(candidate.get("easting_27700"))
        cand_n = V3._to_float(candidate.get("northing_27700"))
        lex_e = V3._to_float(lexicon_easting)
        lex_n = V3._to_float(lexicon_northing)
        anchor_distance = None
        if None not in (cand_e, cand_n, lex_e, lex_n):
            dx = float(cand_e) - float(lex_e)
            dy = float(cand_n) - float(lex_n)
            anchor_distance = round((dx * dx + dy * dy) ** 0.5, 1)
            if anchor_distance > gate_limit:
                continue
        enriched = dict(candidate)
        enriched["fallback_score"] = score
        enriched["fallback_reasons"] = reasons
        enriched["fallback_anchor_distance_m"] = anchor_distance
        enriched["fallback_is_generic"] = 1 if _is_generic_os_address(candidate.get("address")) else 0
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            int(item.get("fallback_score") or 0),
            1 if item.get("road_name") else 0,
            1 if item.get("place") else 0,
            -1 * float(item.get("fallback_anchor_distance_m") or 999999),
        ),
        reverse=True,
    )
    if not ranked:
        return []

    deduped: list[dict[str, Any]] = []
    seen_clusters: set[tuple[int, int]] = set()
    for item in ranked:
        easting = V3._to_float(item.get("easting_27700"))
        northing = V3._to_float(item.get("northing_27700"))
        if easting is not None and northing is not None:
            cluster = (round(float(easting) / 5), round(float(northing) / 5))
            if cluster in seen_clusters:
                continue
            seen_clusters.add(cluster)
        deduped.append(item)

    if not deduped:
        return []

    non_generic = [item for item in deduped if not int(item.get("fallback_is_generic") or 0)]
    if non_generic and not reference_style:
        deduped = non_generic

    top_score = int(deduped[0].get("fallback_score") or 0)
    cutoff = max(40, top_score - 8)
    eligible = [item for item in deduped if int(item.get("fallback_score") or 0) >= cutoff]
    if not eligible:
        eligible = [deduped[0]]

    if range_style:
        limit = 3
    elif reference_style:
        limit = 2
    else:
        limit = 2

    if len(eligible) >= 2:
        first_score = int(eligible[0].get("fallback_score") or 0)
        second_score = int(eligible[1].get("fallback_score") or 0)
        if first_score - second_score >= 8:
            limit = 1

    if limit == 2 and len(eligible) >= 3:
        second_score = int(eligible[1].get("fallback_score") or 0)
        third_score = int(eligible[2].get("fallback_score") or 0)
        if second_score - third_score >= 8:
            limit = 2

    return eligible[:limit]


def _build_row(
    idx: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    os_api_key: str,
    gemini_api_key: str,
    google_api_key: str,
) -> dict[str, Any]:
    session, matcher = _get_thread_resources(args)
    raw_value = str(row.get(args.address_column, "") or "")
    raw_address = clean_text(raw_value.replace("\r", ", ").replace("\n", ", "))
    corrected_raw_address, _ = V3._correct_raw_address_spelling(raw_address, matcher)
    best_match, candidates, debug = matcher.best_match(raw_address, max_candidates=5)
    preferred_match, access_match = _reference_road_hierarchy_override(raw_address, candidates, debug)
    effective_match = preferred_match or best_match
    road_phrases = [item.matched_phrase for item in candidates if getattr(item, "matched_phrase", "")]
    if not road_phrases:
        road_phrases = list(debug.get("expanded_segments", []) or [])
    if preferred_match is not None:
        preferred_phrase = clean_text(getattr(preferred_match, "matched_phrase", "")) or clean_text(getattr(preferred_match, "road_name", ""))
        access_phrase = ""
        if access_match is not None:
            access_phrase = clean_text(getattr(access_match, "matched_phrase", "")) or clean_text(getattr(access_match, "road_name", ""))
        reordered: list[str] = []
        for phrase in [preferred_phrase, access_phrase] + road_phrases:
            cleaned = clean_text(phrase)
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
    merged_candidates = []
    query_variants = list(V3._expand_os_query_variants(os_query))
    if preferred_match is not None:
        preferred_variants = _compose_reference_query_variants(raw_address, preferred_match, access_match, args.council)
        query_variants = preferred_variants + query_variants
    seen_query_keys: set[str] = set()
    deduped_query_variants: list[str] = []
    for query_variant in query_variants:
        query_key = V3._norm_compact(query_variant)
        if query_key and query_key not in seen_query_keys:
            seen_query_keys.add(query_key)
            deduped_query_variants.append(query_variant)
    for query_variant in deduped_query_variants:
        merged_candidates.extend(
            V3._query_find(
                session=session,
                api_key=os_api_key,
                query=query_variant,
                timeout_seconds=V3.DEFAULT_TIMEOUT_SECONDS,
                max_retries=V3.DEFAULT_MAX_RETRIES,
            )
        )
    os_candidates = [_serialize_os_candidate(item) for item in V3._dedupe_os_candidates(merged_candidates)]
    prompt_notes: list[str] = []
    if preferred_match is not None:
        access_road_note = clean_text(getattr(access_match, "road_name", "")) if access_match is not None else ""
        prompt_notes.append(
            f"这个地址里更低一级、更具体的内部道路更可能是 {clean_text(getattr(preferred_match, 'road_name', ''))}；"
            f"{access_road_note} 更像接入主路。"
        )
    prompt = _build_prompt_with_context(raw_address, args.council, effective_match, os_candidates, prompt_notes)
    gemini_result = _call_gemini(
        gemini_api_key,
        args.gemini_model,
        prompt,
        args.gemini_service_tier,
        args.gemini_timeout_ms,
    )
    selected = _match_returned_candidate(gemini_result, os_candidates)

    road_name = clean_text(effective_match.road_name) if effective_match else ""
    road_place = clean_text(effective_match.populated_place) if effective_match else ""
    road_easting = effective_match.geometry_x if effective_match else ""
    road_northing = effective_match.geometry_y if effective_match else ""
    fallback_pool = (
        _build_fallback_pool(
            raw_address,
            road_name,
            road_place,
            road_easting,
            road_northing,
            os_candidates,
            args.council,
            args.county,
        )
        if not selected and str(gemini_result.get("status") or "") == "no_match"
        else []
    )
    google_result = (
        _google_textsearch_best(
            session=session,
            google_api_key=google_api_key,
            raw_address=raw_address,
            corrected_raw_address=corrected_raw_address,
            os_query=os_query,
            road_phrases=road_phrases,
            place_hints=place_hints,
            council_name=args.council,
        )
        if (not selected and google_api_key)
        else {
            "status": "skipped",
            "query": "",
            "candidate_count": 0,
            "confidence_score": 0,
            "confidence_reason": "not_requested",
            "candidates_json": "[]",
            "best_address": "",
            "best_easting_27700": "",
            "best_northing_27700": "",
        }
    )

    output = {
        "idx": idx,
        "key": str(row.get("unique_key") or row.get("key") or "").strip(),
        "_source_unique_key": str(row.get("_source_unique_key") or row.get("unique_key") or row.get("key") or "").strip(),
        "_is_expanded_case": clean_text(row.get("_is_expanded_case") or ""),
        "_expanded_from": clean_text(row.get("_expanded_from") or ""),
        "_expanded_variant": clean_text(row.get("_expanded_variant") or ""),
        "original_address": raw_address,
        "lexicon_road": " | ".join(part for part in [road_name, road_place] if part),
        "lexicon_easting_27700": road_easting,
        "lexicon_northing_27700": road_northing,
        "os_query": os_query,
        "os_query_variants_json": json.dumps(deduped_query_variants, ensure_ascii=False),
        "os_candidates_json": json.dumps(os_candidates, ensure_ascii=False),
        "os_address_gemini": selected.get("address") if selected else "",
        "os_address_gemini_easting_27700": selected.get("easting_27700") if selected else "",
        "os_address_gemini_northing_27700": selected.get("northing_27700") if selected else "",
        "gemini_status": gemini_result.get("status"),
        "gemini_reason": gemini_result.get("reason") or "",
        "gog_query": google_result.get("query") or "",
        "gog_candidate_count": google_result.get("candidate_count") or 0,
        "gog_confidence_score": google_result.get("confidence_score") or 0,
        "gog_confidence_reason": google_result.get("confidence_reason") or "",
        "gog_candidates_json": google_result.get("candidates_json") or "[]",
        "gog_best_address": google_result.get("best_address") or "",
        "gog_best_easting_27700": google_result.get("best_easting_27700") or "",
        "gog_best_northing_27700": google_result.get("best_northing_27700") or "",
        "os_fallback_pool_json": json.dumps(fallback_pool, ensure_ascii=False),
        "reference_hierarchy_primary_road": clean_text(getattr(preferred_match, "road_name", "")) if preferred_match is not None else "",
        "reference_hierarchy_access_road": clean_text(getattr(access_match, "road_name", "")) if access_match is not None else "",
    }

    for index, candidate in enumerate(fallback_pool, start=1):
        output[f"os_fallback_{index}_address"] = candidate.get("address") or ""
        output[f"os_fallback_{index}_easting_27700"] = candidate.get("easting_27700") or ""
        output[f"os_fallback_{index}_northing_27700"] = candidate.get("northing_27700") or ""
        output[f"os_fallback_{index}_score"] = candidate.get("fallback_score") or ""
        output[f"os_fallback_{index}_reasons"] = ",".join(candidate.get("fallback_reasons") or [])
    return output

def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    _apply_council_profile(args)
    input_path = Path(args.input_gpkg).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input GPKG not found: {input_path}")
    os_api_key = args.api_key or V3._load_os_api_key_from_keys_file(args.keys_file)
    if not os_api_key:
        raise SystemExit("Missing OS API key. Use --api-key or --keys-file.")
    gemini_api_key = args.gemini_api_key or V3._load_gemini_api_key_from_keys_file(args.keys_file)
    if not gemini_api_key:
        raise SystemExit("Missing Gemini API key. Use --gemini-api-key or --keys-file.")
    google_api_key = args.google_api_key or V3._load_google_api_key_from_keys_file(args.keys_file) or ""

    rows = load_input_rows(
        input_path,
        int(args.row_limit or 0),
        args.address_column,
        args.input_layer,
        args.council_profile,
    )
    rows = expand_main_rows(rows, args.address_column)
    indexed_rows = list(enumerate(rows, start=1))
    if max(1, int(args.workers)) <= 1:
        selected_rows = _apply_final_best_selection(
            [_build_row(idx, row, args, os_api_key, gemini_api_key, google_api_key) for idx, row in indexed_rows]
        )
        return _apply_low_confidence_web_research(selected_rows, args, gemini_api_key)

    output_rows: list[dict[str, Any] | None] = [None] * len(indexed_rows)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {
            executor.submit(_build_row, idx, row, args, os_api_key, gemini_api_key, google_api_key): idx - 1
            for idx, row in indexed_rows
        }
        total = len(future_map)
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            output_rows[future_map[future]] = future.result()
            if done_idx % 25 == 0 or done_idx == total:
                print(f"Progress: {done_idx}/{total}")
    selected_rows = _apply_final_best_selection([row for row in output_rows if row is not None])
    return _apply_low_confidence_web_research(selected_rows, args, gemini_api_key)
def write_xlsx(
    path: Path,
    rows: list[dict[str, Any]],
    input_gpkg: str,
    input_layer: str | None = None,
    council_profile: str | None = None,
) -> None:
    if pd is None:
        raise SystemExit("XLSX output requires pandas.")
    geometry_by_key = load_polygon_geometries(Path(input_gpkg).expanduser().resolve(), input_layer, council_profile)
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        source_key = str(row.get("_source_unique_key") or row.get("key") or "").strip()
        road_distance = V3._distance_to_polygon_m(
            geometry_by_key,
            source_key,
            row.get("lexicon_easting_27700"),
            row.get("lexicon_northing_27700"),
        )
        gemini_distance = V3._distance_to_polygon_m(
            geometry_by_key,
            source_key,
            row.get("os_address_gemini_easting_27700"),
            row.get("os_address_gemini_northing_27700"),
        )
        google_distance = V3._distance_to_polygon_m(
            geometry_by_key,
            source_key,
            row.get("gog_best_easting_27700"),
            row.get("gog_best_northing_27700"),
        )
        final_distance = V3._distance_to_polygon_m(
            geometry_by_key,
            source_key,
            row.get("best_easting_27700_final"),
            row.get("best_northing_27700_final"),
        )
        web_research_distance = V3._distance_to_polygon_m(
            geometry_by_key,
            source_key,
            row.get("web_research_best_easting_27700"),
            row.get("web_research_best_northing_27700"),
        )
        fallback_avg_distance = ""
        try:
            fallback_pool = json.loads(row.get("os_fallback_pool_json") or "[]")
        except Exception:
            fallback_pool = []
        fallback_distances: list[float] = []
        for candidate in fallback_pool[:3]:
            distance = V3._distance_to_polygon_m(
                geometry_by_key,
                source_key,
                candidate.get("easting_27700"),
                candidate.get("northing_27700"),
            )
            if distance not in ("", None):
                try:
                    fallback_distances.append(float(distance))
                except Exception:
                    pass
        if fallback_distances:
            fallback_avg_distance = round(sum(fallback_distances) / len(fallback_distances), 1)
        summary_rows.append(
            {
                "key": row.get("key") or "",
                "original_address": row.get("original_address") or "",
                "lexicon_road": row.get("lexicon_road") or "",
                "lexicon_distance_to_polygon_m": road_distance or "",
                "os_query": row.get("os_query") or "",
                "os_address_gemini": row.get("os_address_gemini") or "",
                "os_address_gemini_distance_to_polygon_m": gemini_distance or "",
                "gog_best_address": row.get("gog_best_address") or "",
                "gog_best_distance_to_polygon_m": google_distance or "",
                "os_fallback_avg_distance_to_polygon_m": fallback_avg_distance,
                "best_source_final": row.get("best_source_final") or "",
                "best_confidence": row.get("best_confidence") or 0,
                "best_selection_category": row.get("best_selection_category") or "",
                "best_address_final": row.get("best_address_final") or "",
                "best_distance_to_polygon_m_final": final_distance or "",
                "web_research_status": row.get("web_research_status") or "",
                "web_research_best_address": row.get("web_research_best_address") or "",
                "web_research_postcode": row.get("web_research_postcode") or "",
                "web_research_supporting_url": row.get("web_research_supporting_url") or "",
                "web_research_distance_to_polygon_m": web_research_distance or "",
            }
        )
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg).expanduser().resolve()
    default_output_dir = input_gpkg.parent
    default_stem = f"{input_gpkg.stem}_gemini"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else (default_output_dir / f"{default_stem}.json")
    output_xlsx = Path(args.output_xlsx).expanduser().resolve() if args.output_xlsx else (default_output_dir / f"{default_stem}.xlsx")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)
    payload = {
        "meta": {
            "input_gpkg": str(input_gpkg),
            "mode": "lexicon_os_gemini_minimal",
            "rows": len(rows),
            "council_profile": args.council_profile,
            "county": args.county,
            "council": args.council,
            "address_column": args.address_column,
            "input_layer": args.input_layer or "",
            "gemini_model": normalize_gemini_model(args.gemini_model),
        },
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_xlsx(output_xlsx, rows, str(input_gpkg), args.input_layer, args.council_profile)
    print(f"Rows processed: {len(rows)}")
    print(f"JSON written: {output_json}")
    print(f"XLSX written: {output_xlsx}")


if __name__ == "__main__":
    main()
