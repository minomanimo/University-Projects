from __future__ import annotations

import argparse

from config import load_config
from data import load_train_and_test
from data import make_train_val_split

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train_df, _, schema, _ = load_train_and_test(config)

    # TODO:
    # Use sklearn.model_selection.train_test_split to split bank_train.csv
    # into a training set and a validation set.
    #
    # Requirements:
    # - test_size=0.2
    # - random_state=42
    # - stratify=train_df[schema['target_column']]
    #
    # Save the outputs to:
    # - config['paths']['split_train_csv']
    # - config['paths']['split_val_csv']
    split = config['split']

    ts_df, val_df = make_train_val_split(train_df, schema['target_column'], split['val_size'], split['random_state'])

    ts_df.to_csv(config['paths']['split_train_csv'], index=False)
    val_df.to_csv(config['paths']['split_val_csv'], index=False)


if __name__ == '__main__':
    main()
