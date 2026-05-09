from __future__ import annotations

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score


def run_catboost_optuna(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
    categorical_columns: list[str],
    seed: int,
    n_trials: int,
    timeout: int | None,
    search_space: dict,
):
    records = []

    def objective(trial: optuna.Trial) -> float:
        # Define a reasonable search space for CatBoost hyperparameters.
        # Use trial.suggest_* to sample values from search_space.
        # You may tune parameters such as:
        # - iterations
        # - learning_rate
        # - depth
        # - l2_leaf_reg
        # - random_strength
        # - bagging_temperature
        # - border_count
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'F1',
            'random_seed': seed,
            'verbose': False,
            'iterations': trial.suggest_int(
                'iterations',
                int(search_space['iterations'][0]),
                int(search_space['iterations'][1]),
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                float(search_space['learning_rate'][0]),
                float(search_space['learning_rate'][1]),
                log=True,
            ),
            'depth': trial.suggest_int(
                'depth',
                int(search_space['depth'][0]),
                int(search_space['depth'][1]),
            ),
            'l2_leaf_reg': trial.suggest_float(
                'l2_leaf_reg',
                float(search_space['l2_leaf_reg'][0]),
                float(search_space['l2_leaf_reg'][1]),
                log=True,
            ),
            'random_strength': trial.suggest_float(
                'random_strength',
                float(search_space['random_strength'][0]),
                float(search_space['random_strength'][1]),
            ),
            'bagging_temperature': trial.suggest_float(
                'bagging_temperature',
                float(search_space['bagging_temperature'][0]),
                float(search_space['bagging_temperature'][1]),
            ),
            'border_count': trial.suggest_int(
                'border_count',
                int(search_space['border_count'][0]),
                int(search_space['border_count'][1]),
            ),
        }

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_columns,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=50,
        )
        y_pred = model.predict(X_valid).astype(int)
        score = float(f1_score(y_valid, y_pred, zero_division=0))
        records.append({'trial': trial.number, 'f1': score, **trial.params})
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study, pd.DataFrame(records)
