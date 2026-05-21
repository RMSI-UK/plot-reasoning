"""Direct no-AI address/OCR to EPSG:27700 point geocoder.

This module is the standalone point version of the current candidate-box
pipeline.  It reuses the same local Mansfield evidence sources but returns one
selected coordinate instead of boxes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import Point

from no_ai_candidate_boxes.config import (
    DEFAULT_CONFIG,
    bbox_arg,
    cfg_path,
    cfg_param,
    names_arg,
    resolved_config,
)
from no_ai_candidate_boxes.stages import build_input_candidates as build
from no_ai_candidate_boxes.stages import ocr_candidate_rerank as ocr


@dataclass
class PointCandidate:
    source: str
    address: str
    easting_27700: float
    northing_27700: float
    score: float
    reason: str
    metadata: dict[str, Any]

    def as_output(self) -> dict[str, Any]:
        out = asdict(self)
        out["easting_27700"] = round(self.easting_27700, 3)
        out["northing_27700"] = round(self.northing_27700, 3)
        out["score"] = round(self.score, 3)
        return out


def unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def parse_ocr_text(value: str) -> list[dict[str, Any]]:
    """Build fake OCR result records from plain text.

    The existing OCR grid parser expects a page result list.  Plain text is
    enough for grid strings and road phrases, so split into line-like records
    with high confidence.
    """

    rows = []
    for idx, part in enumerate(re.split(r"[\n|]+", value or "")):
        text = part.strip()
        if text:
            rows.append({"text": text, "confidence": 0.98, "bbox": [0, idx, 100, idx + 1]})
    if not rows and value.strip():
        rows.append({"text": value.strip(), "confidence": 0.98, "bbox": [0, 0, 100, 1]})
    return rows


class AddressPointGeocoder:
    def __init__(
        self,
        config_path: Path | None = DEFAULT_CONFIG,
        council_root: Path | None = None,
        base_data_root: Path | None = None,
        output_root: Path | None = None,
        local_bbox: str | None = None,
        locality_names: str | None = None,
        generic_place_names: str | None = None,
        max_text_candidates: int | None = None,
    ) -> None:
        self.config = resolved_config(
            config_path,
            council_root=council_root,
            base_data_root=base_data_root,
            output_root=output_root,
        )
        bbox_value = local_bbox or bbox_arg(self.config.get("local_bbox"))
        if bbox_value:
            bbox = tuple(float(part) for part in bbox_value.split(","))
            build.MANSFIELD_BBOX = bbox
            ocr.MANSFIELD_BBOX = bbox

        names = generic_place_names or names_arg(
            self.config.get("build_generic_place_names") or self.config.get("generic_place_names")
        )
        if names:
            build.GENERIC_PLACE_NAMES = {build.compact(item) for item in names.split("|") if item.strip()}

        locality_arg = locality_names or names_arg(self.config.get("locality_names"))
        if locality_arg:
            parsed = tuple(item.strip().upper() for item in locality_arg.split("|") if item.strip())
            ocr.LOCALITIES = parsed
            ocr.ALLOWED_LOCALITY_NAMES = parsed + tuple(
                item.strip().upper()
                for item in names_arg(self.config.get("allowed_locality_names")).split("|")
                if item.strip()
            )

        self.text_xlsx = cfg_path(self.config, "text_xlsx")
        self.open_roads = cfg_path(self.config, "open_roads")
        self.open_names_dir = cfg_path(self.config, "open_names_dir")
        if not self.text_xlsx or not self.text_xlsx.exists():
            raise FileNotFoundError(f"text_xlsx not found: {self.text_xlsx}")
        if not self.open_roads or not self.open_roads.exists():
            raise FileNotFoundError(f"open_roads not found: {self.open_roads}")

        self.max_text_candidates = int(max_text_candidates or cfg_param(self.config, "max_text_candidates", 20))
        self.road_geoms = build.roi_v1.load_road_geoms(self.open_roads)
        self.road_names = set(self.road_geoms)
        self.text = build.load_text_points(self.text_xlsx)
        self.text_index = build.prepare_global_text_index(self.text, self.road_names)
        self.text_groups = {folder: group for folder, group in self.text.groupby("folder")}
        self.open_names: pd.DataFrame | None = None

    def _get_open_names(self) -> pd.DataFrame:
        if self.open_names is None:
            if self.open_names_dir and self.open_names_dir.exists():
                self.open_names = build.load_open_names(self.open_names_dir)
            else:
                self.open_names = pd.DataFrame()
        return self.open_names

    def _row(self, address: str, folder: str | None = None) -> pd.Series:
        return pd.Series({"chargegeog": address, "FilePath": folder or ""})

    def _text_candidates(self, address: str, ocr_roads: list[str], folder: str | None) -> list[dict[str, Any]]:
        row = self._row(address, folder)
        folder_candidates = (
            build.collect_text_candidates(row, self.text_groups, self.road_names, self.max_text_candidates)
            if folder
            else []
        )
        global_candidates = build.collect_global_text_candidates(
            row,
            self.text_index,
            self.road_names,
            self.max_text_candidates,
        )

        combined_address = " ".join([address, *ocr_roads])
        ocr_global_candidates = []
        if ocr_roads:
            ocr_global_candidates = build.collect_global_text_candidates(
                self._row(combined_address, folder),
                self.text_index,
                self.road_names,
                max(6, self.max_text_candidates // 2),
            )
            for item in ocr_global_candidates:
                item["candidate_pool_source"] = f"ocr_{item.get('candidate_pool_source') or 'global_text_os'}"

        roads = unique_keep_order(
            [
                *build.roi_v1.extract_address_roads(address, self.road_names),
                *ocr_roads,
            ]
        )
        relation_like = bool(build.PARCEL_RELATION_RE.search(combined_address)) or len(roads) >= 2
        road_candidates = build.road_geometry_candidates(
            combined_address,
            self.road_geoms,
            self.road_names,
            max_candidates=10,
        )
        if relation_like:
            ordered_global = [c for c in global_candidates if c.get("candidate_pool_source") != "global_text_os_weak"]
            ordered_global.extend(ocr_global_candidates)
            ordered_global.extend(road_candidates)
            ordered_global.extend(c for c in global_candidates if c.get("candidate_pool_source") == "global_text_os_weak")
        else:
            ordered_global = [*global_candidates, *ocr_global_candidates, *road_candidates]

        return build.merge_candidate_lists(
            ordered_global,
            folder_candidates,
            max_candidates=self.max_text_candidates,
        )

    def _ocr_grid_candidate(
        self,
        ocr_text: str,
        text_candidates: list[dict[str, Any]],
    ) -> tuple[PointCandidate | None, float | None]:
        if not ocr_text.strip():
            return None, None
        results = parse_ocr_text(ocr_text)
        grids = ocr.extract_grids_from_page("inline_ocr", results)
        if not grids:
            return None, None
        pool = [
            ocr.Candidate(
                source=str(item.get("candidate_pool_source") or "text_os"),
                address=str(item.get("address") or ""),
                easting=float(item["easting_27700"]),
                northing=float(item["northing_27700"]),
                score=float(item.get("global_text_score") or 0.0),
            )
            for item in text_candidates
        ]
        ev = ocr.OcrEvidence(prefix="inline", grid_candidates=grids)
        grid, nearest = ocr.choose_ocr_grid(ev, pool)
        if grid is None:
            return None, nearest
        confidence = 92.0
        if nearest is not None:
            confidence += max(0.0, 5.0 - min(nearest / 100.0, 5.0))
        return (
            PointCandidate(
                source="ocr_grid",
                address=f"OCR Geo Code {grid.raw}",
                easting_27700=grid.easting,
                northing_27700=grid.northing,
                score=confidence,
                reason=f"{grid.method};nearest_text_candidate_m={nearest:.1f}" if nearest is not None else grid.method,
                metadata={"raw": grid.raw, "method": grid.method, "image": grid.image, "confidence": grid.confidence},
            ),
            nearest,
        )

    def _evidence_candidates(
        self,
        address: str,
        ocr_text: str,
        text_candidates: list[dict[str, Any]],
    ) -> tuple[list[PointCandidate], dict[str, Any]]:
        address_roads = build.roi_v1.extract_address_roads(address, self.road_names)
        ocr_roads = build.roi_v1.extract_address_roads(ocr_text, self.road_names) if ocr_text else []
        roads = unique_keep_order([*address_roads, *ocr_roads])
        requested_numbers = build.numbers_from_text(address)
        relation_like = bool(build.PARCEL_RELATION_RE.search(address)) or len(roads) >= 2

        candidates: list[PointCandidate] = []
        for idx, item in enumerate(text_candidates):
            score = 58.0 - idx * 2.5
            score += min(16.0, float(item.get("global_text_score") or 0.0) * 0.35)
            score += float(item.get("text_similarity") or 0.0) * 12.0
            score += float(item.get("source_text_similarity") or 0.0) * 8.0
            if item.get("number_overlap"):
                score += 14.0 + min(12.0, 4.0 * int(item.get("number_overlap") or 0))
            if item.get("postcode_match"):
                score += 8.0
            if item.get("road_overlap"):
                score += min(8.0, 3.0 * int(item.get("road_overlap") or 0))
            if relation_like and requested_numbers and not item.get("number_overlap"):
                score -= 8.0
            candidates.append(
                PointCandidate(
                    source=str(item.get("candidate_pool_source") or "text_os"),
                    address=str(item.get("address") or address),
                    easting_27700=float(item["easting_27700"]),
                    northing_27700=float(item["northing_27700"]),
                    score=score,
                    reason="local_text_os",
                    metadata={
                        key: item.get(key)
                        for key in (
                            "road_name",
                            "text_index",
                            "text_similarity",
                            "source_text_similarity",
                            "number_overlap",
                            "number_proximity",
                            "road_overlap",
                            "postcode_match",
                            "postcode_outcode_match",
                            "os_match_score",
                            "global_text_score",
                            "weak_reason",
                        )
                    },
                )
            )

        range_anchors = build.range_anchor_points(text_candidates, address)
        for item in range_anchors:
            point = item["point"]
            candidates.append(
                PointCandidate(
                    source="range_anchor",
                    address=str(item.get("label") or "range_anchor"),
                    easting_27700=float(point.x),
                    northing_27700=float(point.y),
                    score=94.0,
                    reason="weighted_number_range_from_local_os_points",
                    metadata={"label": item.get("label")},
                )
            )

        for item in build.road_intersection_points(roads, self.road_geoms):
            point = item["point"]
            score = 90.0 if relation_like and not requested_numbers else 78.0
            candidates.append(
                PointCandidate(
                    source="road_intersection",
                    address=str(item.get("label") or "road_intersection"),
                    easting_27700=float(point.x),
                    northing_27700=float(point.y),
                    score=score,
                    reason="address_ocr_roads_intersection",
                    metadata={"roads": item.get("label")},
                )
            )

        seed_points = [{"point": Point(c.easting_27700, c.northing_27700), "score": c.score} for c in candidates[:12]]
        corridor_roads = address_roads if address_roads else roads[:3]
        for item in build.road_corridor_points(seed_points, corridor_roads, self.road_geoms):
            point = item["point"]
            candidates.append(
                PointCandidate(
                    source="road_corridor",
                    address=str(item.get("label") or "road_corridor"),
                    easting_27700=float(point.x),
                    northing_27700=float(point.y),
                    score=63.0,
                    reason="projected_onto_named_road_corridor",
                    metadata={"road": item.get("label")},
                )
            )

        # OpenNames CSV loading is comparatively expensive.  Only add it when
        # the local text/road evidence has not already produced a strong point.
        best_so_far = max((candidate.score for candidate in candidates), default=0.0)
        if best_so_far < 85.0:
            for item in build.openname_points(address, ocr_roads, self._get_open_names()):
                point = item["point"]
                candidates.append(
                    PointCandidate(
                        source="openname",
                        address=str(item.get("label") or "openname"),
                        easting_27700=float(point.x),
                        northing_27700=float(point.y),
                        score=57.0,
                        reason="openname_anchor_in_address_or_ocr",
                        metadata={"label": item.get("label")},
                    )
                )

        context = {
            "address_roads": address_roads,
            "ocr_roads": ocr_roads,
            "all_roads": roads,
            "requested_numbers": sorted(requested_numbers),
            "relation_like": relation_like,
        }
        return candidates, context

    def geocode(
        self,
        address: str,
        ocr_text: str = "",
        folder: str | None = None,
        top_n: int = 10,
    ) -> dict[str, Any]:
        address = address.strip()
        if not address:
            raise ValueError("address is required")
        ocr_roads = build.roi_v1.extract_address_roads(ocr_text, self.road_names) if ocr_text else []
        text_candidates = self._text_candidates(address, ocr_roads, folder)
        if not text_candidates and not ocr_text.strip():
            return {
                "status": "no_candidate",
                "address": address,
                "easting_27700": None,
                "northing_27700": None,
                "source": None,
                "confidence": 0.0,
                "reason": "no local text/OS candidates and no OCR evidence",
                "candidates": [],
            }

        candidates, context = self._evidence_candidates(address, ocr_text, text_candidates)
        grid_candidate, grid_nearest = self._ocr_grid_candidate(ocr_text, text_candidates)
        if grid_candidate is not None:
            candidates.append(grid_candidate)

        deduped: list[PointCandidate] = []
        seen: set[tuple[int, int, str]] = set()
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            key = (
                round(candidate.easting_27700 / 2),
                round(candidate.northing_27700 / 2),
                candidate.source,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)

        if not deduped:
            return {
                "status": "no_candidate",
                "address": address,
                "easting_27700": None,
                "northing_27700": None,
                "source": None,
                "confidence": 0.0,
                "reason": "no usable evidence candidates",
                "context": context,
                "candidates": [],
            }

        best = deduped[0]
        confidence = max(0.0, min(98.0, best.score))
        return {
            "status": "ok",
            "address": address,
            "ocr_text_supplied": bool(ocr_text.strip()),
            "easting_27700": round(best.easting_27700, 3),
            "northing_27700": round(best.northing_27700, 3),
            "source": best.source,
            "matched_address": best.address,
            "confidence": round(confidence, 1),
            "reason": best.reason,
            "context": {
                **context,
                "ocr_grid_nearest_text_candidate_m": grid_nearest,
                "text_candidate_count": len(text_candidates),
            },
            "candidates": [item.as_output() for item in deduped[:top_n]],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-AI direct address/OCR to EPSG:27700 point geocoder.")
    parser.add_argument("--address", required=True, help="Planning address or location text.")
    parser.add_argument("--ocr-text", default="", help="Optional OCR text from plans/forms.")
    parser.add_argument("--ocr-text-file", type=Path, help="Optional text file containing OCR text.")
    parser.add_argument("--folder", default="", help="Optional Mansfield OCR/FilePath folder prefix for folder-local text candidates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Council profile JSON.")
    parser.add_argument("--council-root", type=Path, help="Override {council_root} in the config profile.")
    parser.add_argument("--base-data-root", type=Path, help="Override {base_data_root} in the config profile.")
    parser.add_argument("--output-root", type=Path, help="Override {output_root} in the config profile.")
    parser.add_argument("--local-bbox", default="", help="Override working bbox: minx,miny,maxx,maxy.")
    parser.add_argument("--locality-names", default="", help="Override locality names separated by |.")
    parser.add_argument("--generic-place-names", default="", help="Override generic place names separated by |.")
    parser.add_argument("--max-text-candidates", type=int, help="Maximum local text/OS candidates to inspect.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of debug candidates to emit.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    ocr_text = args.ocr_text or ""
    if args.ocr_text_file:
        ocr_text = args.ocr_text_file.read_text(encoding="utf-8")

    geocoder = AddressPointGeocoder(
        config_path=args.config,
        council_root=args.council_root,
        base_data_root=args.base_data_root,
        output_root=args.output_root,
        local_bbox=args.local_bbox or None,
        locality_names=args.locality_names or None,
        generic_place_names=args.generic_place_names or None,
        max_text_candidates=args.max_text_candidates,
    )
    result = geocoder.geocode(
        args.address,
        ocr_text=ocr_text,
        folder=args.folder or None,
        top_n=args.top_n,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
