from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import load_config
from data import load_train_and_test
from utils import ensure_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    results = ensure_results_dirs(config['paths']['results_dir'])
    train_df, test_df, schema, metadata = load_train_and_test(config)

    summary = pd.DataFrame(
        {
            'dataset': ['train', 'test'],
            'rows': [len(train_df), len(test_df)],
            'columns': [train_df.shape[1], test_df.shape[1]],
            'target_yes_ratio': [
                (train_df[schema['target_column']] == 'yes').mean(),
                (test_df[schema['target_column']] == 'yes').mean(),
            ],
        }
    )
    summary.to_csv(results['tables'] / 'data_audit_summary.csv', index=False)

    feature_types = pd.DataFrame(
        {
            'feature_name': schema['feature_columns'],
            'feature_type': [
                'categorical' if c in schema['categorical_columns'] else 'numeric'
                for c in schema['feature_columns']
            ],
        }
    )
    feature_types.to_csv(results['tables'] / 'feature_types.csv', index=False)
    print('Saved data audit tables.')


if __name__ == '__main__':
    main()
