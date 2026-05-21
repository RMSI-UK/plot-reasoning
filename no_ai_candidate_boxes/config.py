"""Configuration helpers for the no-AI candidate-box pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mansfield.json"


def load_config(path: Path | None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data["_config_path"] = str(config_path)
    return data


def resolve_template(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [resolve_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: resolve_template(item, variables) for key, item in value.items()}
    return value


def resolved_config(
    path: Path | None,
    council_root: Path | None = None,
    base_data_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    raw = load_config(path)
    variables = dict(raw)
    if council_root is not None:
        variables["council_root"] = str(council_root)
    if base_data_root is not None:
        variables["base_data_root"] = str(base_data_root)
    if output_root is not None:
        variables["output_root"] = str(output_root)

    resolved = resolve_template(raw, variables)
    resolved["council_root"] = variables.get("council_root", resolved.get("council_root"))
    resolved["base_data_root"] = variables.get("base_data_root", resolved.get("base_data_root"))
    resolved["output_root"] = variables.get("output_root", resolved.get("output_root"))
    return resolved


def cfg_path(config: dict[str, Any], key: str, default: str | None = None) -> Path | None:
    value = (config.get("paths") or {}).get(key, default)
    return Path(value) if value else None


def cfg_layer(config: dict[str, Any], key: str, default: str | None = None) -> str | None:
    return (config.get("layers") or {}).get(key, default)


def cfg_param(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return (config.get("parameters") or {}).get(key, default)


def bbox_arg(value: Any) -> str:
    if not value:
        return ""
    return ",".join(str(float(item)) for item in value)


def names_arg(value: Any) -> str:
    if not value:
        return ""
    return "|".join(str(item) for item in value)
