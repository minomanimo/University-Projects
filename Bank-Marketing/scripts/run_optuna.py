from __future__ import annotations

import argparse
import pandas as pd

from catboost import CatBoostClassifier

from config import load_config
from data import load_train_and_test, load_dataframe
from features import add_engineered_features, fit_feature_state, get_expanded_feature_columns, label_encode_target
from optuna_search import run_catboost_optuna
from utils import classification_metrics, ensure_results_dirs, init_wandb, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config['seed']))
    results = ensure_results_dirs(config['paths']['results_dir'])
    run = init_wandb(config, config.get('run_name', 'optuna_catboost'))

    train_split = load_dataframe(config['paths']['split_train_csv'])
    val_split = load_dataframe(config['paths']['split_val_csv'])
    _, test_df, schema, _ = load_train_and_test(config)

    train_split = label_encode_target(train_split, schema['target_column'])
    val_split = label_encode_target(val_split, schema['target_column'])
    test_df = label_encode_target(test_df, schema['target_column'])

    state = fit_feature_state(train_split)
    train_split = add_engineered_features(train_split, state)
    val_split = add_engineered_features(val_split, state)
    test_df = add_engineered_features(test_df, state)

    feature_cols, categorical_cols, _ = get_expanded_feature_columns(schema)
    target_col = schema['target_column']

    study, trials_df = run_catboost_optuna(
        X_train=train_split[feature_cols],
        y_train=train_split[target_col],
        X_valid=val_split[feature_cols],
        y_valid=val_split[target_col],
        categorical_columns=categorical_cols,
        seed=int(config['seed']),
        n_trials=int(config['optuna']['n_trials']),
        timeout=config['optuna']['timeout'],
        search_space=config['search_space'],
    )
    trials_df.to_csv(results['tables'] / 'optuna_trials.csv', index=False)

    best_params = study.best_trial.params | {
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'random_seed': int(config['seed']),
        'verbose': False,
    }
    model = CatBoostClassifier(**best_params)
    model.fit(
        train_split[feature_cols],
        train_split[target_col],
        cat_features=categorical_cols,
        eval_set=(val_split[feature_cols], val_split[target_col]),
        use_best_model=True,
        early_stopping_rounds=50,
    )

    val_metrics = classification_metrics(val_split[target_col], model.predict(val_split[feature_cols]).astype(int))
    test_metrics = classification_metrics(test_df[target_col], model.predict(test_df[feature_cols]).astype(int))
    save_json({'best_params': study.best_trial.params, 'validation': val_metrics, 'test': test_metrics}, results['tables'] / 'optuna_best.json')
    model.save_model(results['models'] / 'catboost_tuned.cbm')
    if run is not None:
        run.log({
            'model_name': 'catboost_tuned',
            'feature_set_name': 'expanded',
            'seed': int(config['seed']),
            'validation_accuracy': val_metrics['accuracy'],
            'validation_f1': val_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
        })
        run.finish()
    print({'best_params': study.best_trial.params, 'validation': val_metrics, 'test': test_metrics})


if __name__ == '__main__':
    main()
