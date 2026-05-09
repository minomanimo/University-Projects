from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def load_schema_and_metadata(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = config['paths']
    schema = load_json(paths['schema_json'])
    metadata = load_json(paths['metadata_json'])
    return schema, metadata


def load_train_and_test(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    paths = config['paths']
    train_df = load_dataframe(paths['train_csv'])
    test_df = load_dataframe(paths['test_csv'])
    schema, metadata = load_schema_and_metadata(config)
    required = [schema['id_column'], schema['target_column'], *schema['feature_columns']]
    validate_columns(train_df, required)
    validate_columns(test_df, required)
    return train_df, test_df, schema, metadata


def make_train_val_split(df: pd.DataFrame, target_col: str, val_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
