from __future__ import annotations

import argparse
import pandas as pd

from config import load_config
from data import load_train_and_test, load_dataframe
from features import add_engineered_features, fit_feature_state, get_expanded_feature_columns, label_encode_target
from catboost_model import build_catboost_model
from utils import classification_metrics, ensure_results_dirs, init_wandb, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config['seed']))
    results = ensure_results_dirs(config['paths']['results_dir'])
    run = init_wandb(config, config.get('run_name', 'catboost_expanded'))

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

    # Choose the feature set to use for CatBoost.
    # For this assignment, use the expanded feature set.
    #
    # Define the categorical feature columns for CatBoost.
    # Use the schema file and include any engineered categorical features if needed.
    model = build_catboost_model(config['model'], int(config['seed']))

    # Train the CatBoost model using the training set and validation set.
    # Suggested inputs:
    # - train_split[feature_cols]
    # - train_split[target_col]
    # - cat_features=categorical_cols
    # - eval_set=(val_split[feature_cols], val_split[target_col])
    # - use_best_model=True
    # - early_stopping_rounds=50
    model.fit(
        train_split[feature_cols],
        train_split[target_col],
        cat_features=categorical_cols,
        eval_set=(val_split[feature_cols], val_split[target_col]),
        use_best_model=True,
        early_stopping_rounds=50,
    )

    # Evaluate the trained model on the validation set and the test set.
    # Report Accuracy and F1-score.
    val_pred = model.predict(val_split[feature_cols]).astype(int)
    test_pred = model.predict(test_df[feature_cols]).astype(int)
    metrics = {
        'validation': classification_metrics(val_split[target_col], val_pred),
        'test': classification_metrics(test_df[target_col], test_pred),
    }

    pd.DataFrame(metrics).T.to_csv(results['tables'] / 'catboost_metrics.csv')
    save_json(metrics, results['tables'] / 'catboost_metrics.json')

    # Save the trained CatBoost model to results/models/.
    model.save_model(str(results['models'] / 'catboost_expanded.cbm'))

    if run is not None:
        # Log validation/test Accuracy and F1-score to W&B.
        run.log({
            'model_name': 'catboost',
            'feature_set_name': 'expanded',
            'seed': int(config['seed']),
            'validation_accuracy': metrics['validation']['accuracy'],
            'validation_f1': metrics['validation']['f1'],
            'test_accuracy': metrics['test']['accuracy'],
            'test_f1': metrics['test']['f1'],
        })
        run.finish()

    print(metrics)


if __name__ == '__main__':
    main()
