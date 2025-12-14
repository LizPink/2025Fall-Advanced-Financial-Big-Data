# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None  # type: ignore

# torch models are optional
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .torch_models import TorchSequenceRegressor


@dataclass
class FittedModel:
    name: str
    model: Any
    is_sequence: bool = False
    seq_len: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FittedModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.model.predict(X)
        return np.asarray(pred, dtype=float).reshape(-1)


def _with_standardize(estimator: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_model(
    model_name: str,
    params: Dict[str, Any],
    random_seed: int = 42,
    standardize_linear: bool = True,
    standardize_mlp: bool = True,
    standardize_sequence: bool = True,
) -> FittedModel:
    """Factory returning a unified .fit/.predict model."""
    name = model_name

    if name == "Ridge":
        est = Ridge(random_state=random_seed, **params)
        if standardize_linear:
            est = _with_standardize(est)
        return FittedModel(name=name, model=est)

    if name == "Lasso":
        est = Lasso(random_state=random_seed, max_iter=20000, **params)
        if standardize_linear:
            est = _with_standardize(est)
        return FittedModel(name=name, model=est)

    if name == "ElasticNet":
        est = ElasticNet(random_state=random_seed, max_iter=20000, **params)
        if standardize_linear:
            est = _with_standardize(est)
        return FittedModel(name=name, model=est)

    if name == "RandomForest":
        est = RandomForestRegressor(random_state=random_seed, n_jobs=-1, **params)
        return FittedModel(name=name, model=est)

    if name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        est = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=-1,
            **params,
        )
        return FittedModel(name=name, model=est)

    if name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        est = LGBMRegressor(
            random_state=random_seed,
            n_jobs=-1,
            **params,
        )
        return FittedModel(name=name, model=est)

    if name == "MLP":
        est = MLPRegressor(
            random_state=random_seed,
            early_stopping=True,
            n_iter_no_change=20,
            **params,
        )
        if standardize_mlp:
            est = _with_standardize(est)
        return FittedModel(name=name, model=est)

    if name in ("LSTM", "Transformer"):
        if torch is None:
            raise ImportError("torch is not installed")
        # TorchSequenceRegressor expects params include seq_len and model-specific settings
        seq_len = int(params.get("seq_len", 20))
        torch_params = dict(params)
        torch_params.pop("seq_len", None)
        reg = TorchSequenceRegressor(
            arch=name,
            seq_len=seq_len,
            standardize=standardize_sequence,
            random_seed=random_seed,
            **torch_params,
        )
        return FittedModel(name=name, model=reg, is_sequence=True, seq_len=seq_len)

    raise ValueError(f"Unknown model: {model_name}")
