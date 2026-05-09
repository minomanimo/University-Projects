from __future__ import annotations

import argparse
import joblib
import pandas as pd

from config import load_config
from data import load_train_and_test, load_dataframe
from features import label_encode_target
from linear_model import build_linear_pipeline
from utils import classification_metrics, ensure_results_dirs, init_wandb, save_json, set_seed
from features import add_engineered_features, fit_feature_state, get_expanded_feature_columns, label_encode_target

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config['seed']))
    results = ensure_results_dirs(config['paths']['results_dir'])
    run = init_wandb(config, config.get('run_name', 'linear_baseline'))

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

    
    target_col = schema['target_column']
    feature_cols = list(schema['feature_columns'])
    categorical_cols = list(schema['categorical_columns'])
    numeric_cols = list(schema['numeric_columns'])
    target_col = schema['target_column']
    
    feature_cols, categorical_cols, numeric_cols = get_expanded_feature_columns(schema)
    model = build_linear_pipeline(categorical_cols, numeric_cols, config['model'])
    model.fit(train_split[feature_cols], train_split[target_col])

    metrics = {}
    for split_name, df in [('train', train_split), ('validation', val_split), ('test', test_df)]:
        y_pred = model.predict(df[feature_cols])
        metrics[split_name] = classification_metrics(df[target_col], y_pred)

    pd.DataFrame(metrics).T.to_csv(results['tables'] / 'linear_metrics.csv')
    joblib.dump(model, results['models'] / 'linear_baseline.joblib')
    save_json(metrics, results['tables'] / 'linear_metrics.json')

    if run is not None:
        run.log({
            'validation_accuracy': metrics['validation']['accuracy'],
            'validation_f1': metrics['validation']['f1'],
            'test_accuracy': metrics['test']['accuracy'],
            'test_f1': metrics['test']['f1'],
        })
        run.finish()
    print(metrics)


if __name__ == '__main__':
    main()
