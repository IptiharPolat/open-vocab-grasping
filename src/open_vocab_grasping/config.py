from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, resolving an optional `extends` path relative to the file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    parent = data.pop("extends", None)
    if parent is not None:
        parent_path = (config_path.parent / str(parent)).resolve()
        data = _merge(load_config(parent_path), data)
    data["_config_path"] = str(config_path)
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable, stream, sort_keys=False)

