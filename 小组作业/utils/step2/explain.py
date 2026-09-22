# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import rmse


@dataclass
class ImportanceResult:
    method: str
    importances: np.ndarray  # (n_features,)
    feature_names: List[str]


def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    sample_n: Optional[int] = 1000,
    n_repeats: int = 5,
    random_state: int = 42,
) -> ImportanceResult:
    """Model-agnostic permutation importance using RMSE increase.

    Higher importance means larger deterioration when permuting the feature.
    """
    rng = np.random.default_rng(int(random_state))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(y)
    if sample_n is not None and n > int(sample_n):
        idx = rng.choice(n, size=int(sample_n), replace=False)
        idx.sort()
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    base_pred = model.predict(Xs)
    mask = ~np.isnan(base_pred)
    base = rmse(ys[mask], base_pred[mask])

    importances = np.zeros(Xs.shape[1], dtype=float)

    for j in range(Xs.shape[1]):
        scores = []
        for _ in range(int(n_repeats)):
            Xp = Xs.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, j] = Xp[perm, j]
            pred = model.predict(Xp)
            m = ~np.isnan(pred)
            scores.append(rmse(ys[m], pred[m]))
        importances[j] = float(np.mean(scores) - base)

    return ImportanceResult(method="permutation", importances=importances, feature_names=feature_names)


def treeshap_importance_xgb(model: Any, X: np.ndarray, feature_names: List[str]) -> Optional[ImportanceResult]:
    """Compute mean(|SHAP|) for XGBoost using pred_contribs=True (no shap dependency)."""
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return None

    try:
        booster = model
        # if it's a pipeline, get the underlying estimator
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            booster = model.named_steps["model"]
        if hasattr(booster, "get_booster"):
            dm = xgb.DMatrix(X, feature_names=feature_names)
            contrib = booster.get_booster().predict(dm, pred_contribs=True)
            # last column is bias
            vals = np.mean(np.abs(contrib[:, :-1]), axis=0)
            return ImportanceResult(method="treeshap_xgb", importances=vals, feature_names=feature_names)
    except Exception:
        return None
    return None


def treeshap_importance_lgbm(model: Any, X: np.ndarray, feature_names: List[str]) -> Optional[ImportanceResult]:
    """Compute mean(|SHAP|) for LightGBM using pred_contrib=True (no shap dependency)."""
    try:
        est = model
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            est = model.named_steps["model"]
        if hasattr(est, "predict"):
            contrib = est.predict(X, pred_contrib=True)
            contrib = np.asarray(contrib)
            vals = np.mean(np.abs(contrib[:, :-1]), axis=0)
            return ImportanceResult(method="treeshap_lgbm", importances=vals, feature_names=feature_names)
    except Exception:
        return None
    return None
