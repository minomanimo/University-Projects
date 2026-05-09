from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    defaults = config.pop('defaults', None)
    if defaults and defaults.get('base_config'):
        base_path = Path(defaults['base_config'])
        if not base_path.is_absolute():
            base_path = (config_path.parent.parent / base_path).resolve()
        config = _deep_update(load_config(base_path), config)
    return config
