from __future__ import annotations

import pandas as pd


def fit_autogluon(train_df: pd.DataFrame, label: str, path: str, presets: str, eval_metric: str, time_limit: int | None):
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(label=label, path=path, eval_metric=eval_metric)
    predictor.fit(train_data=train_df, presets=presets, time_limit=time_limit)
    return predictor
