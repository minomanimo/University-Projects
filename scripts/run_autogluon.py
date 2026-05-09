from __future__ import annotations

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from autogluon_model import fit_autogluon
from config import load_config
from data import load_train_and_test, load_dataframe
from features import add_engineered_features, fit_feature_state, label_encode_target
from utils import ensure_results_dirs, init_wandb, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config['seed']))
    results = ensure_results_dirs(config['paths']['results_dir'])
    run = init_wandb(config, config.get('run_name', 'autogluon_benchmark'))

    train_split = load_dataframe(config['paths']['split_train_csv'])
    _, test_df, schema, _ = load_train_and_test(config)
    train_split = label_encode_target(train_split, schema['target_column'])
    test_df = label_encode_target(test_df, schema['target_column'])

    state = fit_feature_state(train_split)
    train_split = add_engineered_features(train_split, state)
    test_df = add_engineered_features(test_df, state)

    predictor = fit_autogluon(
        train_df=train_split,
        label=schema['target_column'],
        path=str(results['models'] / 'autogluon'),
        presets=config['autogluon']['presets'],
        eval_metric=config['autogluon']['eval_metric'],
        time_limit=config['autogluon']['time_limit'],
    )
    test_pred = predictor.predict(test_df)
    metrics = {
        'accuracy': float(accuracy_score(test_df[schema['target_column']], test_pred)),
        'f1': float(f1_score(test_df[schema['target_column']], test_pred, zero_division=0)),
    }
    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard.to_csv(results['tables'] / 'autogluon_leaderboard.csv', index=False)
    save_json(metrics, results['tables'] / 'autogluon_metrics.json')
    if run is not None:
        run.log({
            'model_name': 'autogluon',
            'feature_set_name': 'expanded',
            'seed': int(config['seed']),
            'test_accuracy': metrics['accuracy'],
            'test_f1': metrics['f1'],
        })
        run.finish()
    print(metrics)


if __name__ == '__main__':
    main()
