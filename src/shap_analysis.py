from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def make_tree_explainer(model):
    return shap.TreeExplainer(model)


def save_shap_summary(explainer, X: pd.DataFrame, output_path: str | Path, max_display: int = 15):
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_shap_dependence(explainer, X: pd.DataFrame, feature_name: str, output_path: str | Path):
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.dependence_plot(feature_name, shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_shap_waterfall(explainer, X: pd.DataFrame, row_index: int, output_path: str | Path):
    shap_values = explainer(X)
    plt.figure()
    shap.plots.waterfall(shap_values[row_index], show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
