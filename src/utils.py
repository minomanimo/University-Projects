from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_results_dirs(results_dir: str | Path) -> dict[str, Path]:
    root = Path(results_dir)
    paths = {
        'root': root,
        'models': root / 'models',
        'predictions': root / 'predictions',
        'figures': root / 'figures',
        'tables': root / 'tables',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def init_wandb(config: dict[str, Any], run_name: str, extra_config: dict[str, Any] | None = None):
    try:
        import wandb  # type: ignore
    except Exception:
        return None

    if wandb is None:
        return None
    wandb_cfg = config.get('wandb', {})
    merged = dict(config)
    if extra_config:
        merged.update(extra_config)
    try:
        return wandb.init(
            project=wandb_cfg.get('project', 'bank-marketing-assignment'),
            entity=wandb_cfg.get('entity'),
            mode=wandb_cfg.get('mode', 'offline'),
            tags=wandb_cfg.get('tags'),
            config=merged,
            name=run_name,
        )
    except Exception:
        return None
