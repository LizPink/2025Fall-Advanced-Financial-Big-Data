# -*- coding: utf-8 -*-
"""Model factory: create estimators by name."""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

def make_model(name: str, params: Dict[str, Any] | None = None):
    name = str(name).lower().strip()
    params = params or {}

    if name == "baseline_zero":
        return BaselineZero()

    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)

    if name == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(**params)

    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**params)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        base = dict(
            random_state=42,
            n_jobs=-1,
        )
        base.update(params)
        return RandomForestRegressor(**base)

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:
            raise ImportError("xgboost is not installed. Please install xgboost.") from e
        base = dict(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
        base.update(params)
        return XGBRegressor(**base)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor  # type: ignore
        except Exception as e:
            raise ImportError("lightgbm is not installed. Please install lightgbm.") from e
        base = dict(
            random_state=42,
            n_jobs=1,
        )
        base.update(params)
        return LGBMRegressor(**base)

    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        base = dict(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
        )
        base.update(params)
        return MLPRegressor(**base)

    raise ValueError(f"Unknown model name: {name}")

class BaselineZero:
    """Return baseline for random-walk-without-drift in return space: predict 0."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        import numpy as np
        return np.zeros(len(X), dtype=float)
