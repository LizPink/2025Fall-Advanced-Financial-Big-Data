# -*- coding: utf-8 -*-
"""Explainability utilities: permutation importance (unified) + TreeSHAP (tree enhancer)."""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    metric: str = "rmse",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance for tabular models.
    For sklearn-compatible estimators, this uses sklearn.inspection.permutation_importance.
    """
    try:
        from sklearn.inspection import permutation_importance
    except Exception as e:
        raise ImportError("scikit-learn is required for permutation_importance") from e

    # Define scoring
    if metric == "rmse":
        scoring = "neg_root_mean_squared_error"
    elif metric == "mse":
        scoring = "neg_mean_squared_error"
    elif metric == "mae":
        scoring = "neg_mean_absolute_error"
    else:
        # fallback to negative RMSE
        scoring = "neg_root_mean_squared_error"

    res = permutation_importance(
        model, X, y,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        scoring=scoring,
        n_jobs=1,
    )
    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df

def compute_treeshap(
    model,
    X: pd.DataFrame,
    top_k: int = 30,
) -> Optional[pd.DataFrame]:
    """Compute TreeSHAP mean(|shap|) importance for tree models, if shap is installed."""
    try:
        import shap  # type: ignore
    except Exception:
        return None

    # shap.TreeExplainer works for many tree models (sklearn RF, XGBoost, LightGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # shap_values may be list for multioutput; here single-output regression expected
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    vals = np.abs(np.asarray(shap_values))
    imp = vals.mean(axis=0)
    df = pd.DataFrame({
        "feature": X.columns,
        "treeshap_mean_abs": imp,
    }).sort_values("treeshap_mean_abs", ascending=False).head(int(top_k)).reset_index(drop=True)
    return df
