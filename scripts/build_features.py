from __future__ import annotations

import argparse
import pandas as pd

from config import load_config
from data import load_train_and_test, load_dataframe
from features import add_engineered_features, fit_feature_state, label_encode_target
from utils import ensure_results_dirs, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    results = ensure_results_dirs(config['paths']['results_dir'])
    train_df = load_dataframe(config['paths']['split_train_csv'])
    val_df = load_dataframe(config['paths']['split_val_csv'])
    _, test_df, schema, _ = load_train_and_test(config)

    train_df = label_encode_target(train_df, schema['target_column'])
    val_df = label_encode_target(val_df, schema['target_column'])
    test_df = label_encode_target(test_df, schema['target_column'])

    state = fit_feature_state(train_df)

    train_feat = add_engineered_features(train_df, state)
    val_feat = add_engineered_features(val_df, state)
    test_feat = add_engineered_features(test_df, state)

    train_feat.to_csv(results['tables'] / 'train_expanded_preview.csv', index=False)
    val_feat.to_csv(results['tables'] / 'val_expanded_preview.csv', index=False)
    test_feat.to_csv(results['tables'] / 'test_expanded_preview.csv', index=False)

    feature_summary = pd.DataFrame([
        {'feature_name': 'is_pdays_missing', 'source_columns': 'pdays', 'feature_type': 'numeric', 'description': '1 if pdays == -1 else 0'},
        {'feature_name': 'has_previous_contact', 'source_columns': 'previous', 'feature_type': 'numeric', 'description': '1 if previous > 0 else 0'},
        {'feature_name': 'log_balance', 'source_columns': 'balance', 'feature_type': 'numeric', 'description': 'log1p(balance + training_shift)'},
        {'feature_name': 'campaign_group', 'source_columns': 'campaign', 'feature_type': 'categorical', 'description': 'campaign binned into 1 / 2 / 3_to_4 / 5_or_more'},
    ])
    feature_summary.to_csv(results['tables'] / 'feature_summary.csv', index=False)
    save_json({'balance_shift': state.balance_shift}, results['tables'] / 'feature_state.json')
    print('Saved expanded feature previews and feature summary.')


if __name__ == '__main__':
    main()
