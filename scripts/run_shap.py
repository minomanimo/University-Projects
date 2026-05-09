from __future__ import annotations

import argparse

from catboost import CatBoostClassifier

from config import load_config
from data import load_train_and_test, load_dataframe
from features import add_engineered_features, fit_feature_state, get_expanded_feature_columns, label_encode_target
from shap_analysis import make_tree_explainer, save_shap_dependence, save_shap_summary, save_shap_waterfall
from utils import ensure_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    results = ensure_results_dirs(config['paths']['results_dir'])

    train_split = load_dataframe(config['paths']['split_train_csv'])
    _, test_df, schema, _ = load_train_and_test(config)
    train_split = label_encode_target(train_split, schema['target_column'])
    test_df = label_encode_target(test_df, schema['target_column'])

    state = fit_feature_state(train_split)
    train_split = add_engineered_features(train_split, state)
    test_df = add_engineered_features(test_df, state)

    feature_cols, _, _ = get_expanded_feature_columns(schema)
    target_col = schema['target_column']

    # TODO:
    # Load your best CatBoost model from config['model_path'].
    model = CatBoostClassifier()
    model.load_model(config['model_path'])

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy()

    # Generate predictions and identify at least two misclassified test samples.
    # Save the selected row indices in a list such as misclassified_indices.
    y_pred = model.predict(X_test).astype(int)
    misclassified_indices = [idx for idx, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
    misclassified_indices = misclassified_indices[:2]
    if len(misclassified_indices) < 2:
        raise ValueError('Need at least two misclassified samples for local SHAP explanations.')

    explainer = make_tree_explainer(model)

    # Save one global feature importance plot.
    # Example target path:
    # results['figures'] / 'shap_summary.png'
    save_shap_summary(
        explainer,
        X_test,
        results['figures'] / 'shap_summary.png',
        max_display=int(config.get('max_display', 15)),
    )

    # Save at least two feature effect plots.
    # Choose informative features from the expanded feature set.
    for feature_name in ['log_balance', 'campaign']:
        save_shap_dependence(
            explainer,
            X_test,
            feature_name,
            results['figures'] / f'shap_dependence_{feature_name}.png',
        )

    # Save at least two local explanations for misclassified samples.
    # Use the selected misclassified row indices.
    for i, row_index in enumerate(misclassified_indices, start=1):
        save_shap_waterfall(
            explainer,
            X_test,
            row_index,
            results['figures'] / f'shap_waterfall_misclassified_{i}.png',
        )

    print({
        'misclassified_indices': misclassified_indices,
        'figures': [
            'shap_summary.png',
            'shap_dependence_log_balance.png',
            'shap_dependence_campaign.png',
            'shap_waterfall_misclassified_1.png',
            'shap_waterfall_misclassified_2.png',
        ],
    })


if __name__ == '__main__':
    main()
