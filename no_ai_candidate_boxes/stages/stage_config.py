"""Small CLI parsing helpers shared by pipeline stages."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_bbox(value: Any, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value in (None, ""):
        return fallback
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parts = json.loads(text)
        else:
            parts = re.split(r"[,| ]+", text)
    else:
        parts = value
    out = tuple(float(part) for part in parts if str(part) != "")
    if len(out) != 4:
        raise ValueError(f"Expected bbox as minx,miny,maxx,maxy, got {value!r}")
    return out  # type: ignore[return-value]


def parse_names(value: Any, fallback: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    if value in (None, ""):
        return [normalise_name(item) for item in fallback]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parts = json.loads(text)
        else:
            parts = re.split(r"[|,]", text)
    else:
        parts = value
    return [normalise_name(item) for item in parts if normalise_name(item)]


def normalise_name(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

