from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import numpy as np

@dataclass
class FeatureState:
    balance_shift: float


def label_encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if out[target_col].dtype == object:
        out[target_col] = out[target_col].map({'no': 0, 'yes': 1}).astype(int)
    return out


def fit_feature_state(train_df: pd.DataFrame) -> FeatureState:
    # TODO:
    # Compute the training-only shift used for log_balance.
    # Rule from the assignment:
    #   balance_shifted = balance - min(balance)
    # Use the minimum balance value from the TRAINING set only.
    balance_shift = float(-train_df['balance'].min())
    return FeatureState(balance_shift=balance_shift)



def add_is_pdays_missing(df: pd.DataFrame) -> pd.DataFrame:
    # TODO:
    # Create a binary feature called is_pdays_missing.
    # It should be 1 if pdays == -1, and 0 otherwise.
    df['is_pdays_missing'] = (df['pdays'] == -1).astype(int)

    return df


def add_has_previous_contact(df: pd.DataFrame) -> pd.DataFrame:
    # TODO:
    # Create a binary feature called has_previous_contact.
    # It should be 1 if previous > 0, and 0 otherwise.
    df['has_previous_contact'] = (df['previous'] > 0).astype(int)
    return df

def add_log_balance(df: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    # TODO:
    # Create log_balance using the exact rule from the assignment:
    # 1) use the training-derived shift from fit_feature_state
    # 2) create balance_shifted = balance + state.balance_shift
    # 3) create log_balance = log1p(balance_shifted)
    # Use the same shift for the validation and test sets.
    df['balance_shifted'] = df['balance'] + state.balance_shift
    df['log_balance'] = np.log1p(df['balance_shifted'])
    return df


def add_campaign_group(df: pd.DataFrame) -> pd.DataFrame:
    # TODO:
    # Create a categorical feature called campaign_group using the following bins:
    # - '1'
    # - '2'
    # - '3_to_4'
    # - '5_or_more'
    df['campaign_group'] = '5_or_more'
    df.loc[df['campaign'] == 1, 'campaign_group'] = '1'
    df.loc[df['campaign'] == 2, 'campaign_group'] = '2'
    df.loc[df['campaign'].isin([3, 4]), 'campaign_group'] = '3_to_4'
    return df

def add_engineered_features(df: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    # TODO:
    # Apply all required engineered features and return the expanded dataframe.
    df = add_is_pdays_missing(df)
    df = add_has_previous_contact(df)
    df = add_log_balance(df, state)
    df = add_campaign_group(df)
    out = df.copy()
    return out

def get_expanded_feature_columns(schema: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    base_features = list(schema['feature_columns'])
    added_numeric = ['is_pdays_missing', 'has_previous_contact', 'log_balance']
    added_categorical = ['campaign_group']
    feature_cols = base_features + added_numeric + added_categorical
    categorical_cols = list(schema['categorical_columns']) + added_categorical
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, categorical_cols, numeric_cols
