# Bank Marketing Starter Code

This repository contains starter code for the Bank Marketing tabular modeling assignment.
Several files intentionally include `TODO` blocks. You are expected to complete those parts
according to the assignment PDF.

## Provided data
Keep the following files under `data/`:
- `bank_train.csv`
- `bank_eval.csv`
- `schema.json`
- `metadata.json`

## Expected workflow
1. Create and save the train/validation split.
2. Run the data audit.
3. Train the linear baseline.
4. Build the engineered features.
5. Train the CatBoost model.
6. Run SHAP analysis.
7. Run Optuna-based hyperparameter optimization.
8. Run the AutoGluon benchmark.

## Example commands
```bash
export PYTHONPATH=src
python scripts/make_train_val_split.py --config configs/base.yaml
python scripts/run_data_audit.py --config configs/base.yaml
python scripts/train_linear.py --config configs/linear.yaml
python scripts/build_features.py --config configs/catboost.yaml
python scripts/run_catboost.py --config configs/catboost.yaml
python scripts/run_shap.py --config configs/shap.yaml
python scripts/run_optuna.py --config configs/optuna.yaml
python scripts/run_autogluon.py --config configs/autogluon.yaml
```

## Files with major TODOs
- `scripts/make_train_val_split.py`
- `src/linear_model.py`
- `src/features.py`
- `scripts/run_catboost.py`
- `src/optuna_search.py`
- `scripts/run_shap.py`

## Notes
- `src/catboost_model.py` is mostly provided so that you can focus on the experiment flow.
- `src/shap_analysis.py` provides helper functions; you should complete the SHAP workflow in `scripts/run_shap.py`.
- `src/autogluon_model.py` is mostly provided because AutoGluon is used as a benchmark.
