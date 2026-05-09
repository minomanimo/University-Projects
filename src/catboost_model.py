from __future__ import annotations

from catboost import CatBoostClassifier


def build_catboost_model(config: dict, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(config.get('iterations', 500)),
        learning_rate=float(config.get('learning_rate', 0.05)),
        depth=int(config.get('depth', 6)),
        l2_leaf_reg=float(config.get('l2_leaf_reg', 3.0)),
        loss_function=config.get('loss_function', 'Logloss'),
        eval_metric=config.get('eval_metric', 'F1'),
        random_strength=float(config.get('random_strength', 1.0)),
        bagging_temperature=float(config.get('bagging_temperature', 0.0)),
        border_count=int(config.get('border_count', 128)),
        random_seed=seed,
        verbose=False,
    )
