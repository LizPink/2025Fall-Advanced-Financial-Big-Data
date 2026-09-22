# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import logging

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

from .torch_models import TorchSequenceRegressor, TorchMLPRegressor

# avoid spamming logs on repeated factory calls (grid search / walk-forward)
_GPU_LOGGED = set()


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

class _FitFallback:
    """Try fitting a primary estimator; if it fails, use a fallback estimator.

    This is helpful when GPU is requested but the local XGBoost/LightGBM build
    does not support GPU or runs out of GPU memory.
    """

    def __init__(self, primary: Any, fallback: Any, tag: str, fallback_enabled: bool = True):
        self._primary = primary
        self._fallback = fallback
        self._tag = str(tag)
        self._fallback_enabled = bool(fallback_enabled)
        self._active = primary

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            self._active = self._primary
            self._active.fit(X, y)
            return self
        except Exception as e:
            if not self._fallback_enabled:
                raise
            log = logging.getLogger("step2")
            log.warning(f"[{self._tag}] GPU fit failed -> fallback to CPU. error={type(e).__name__}: {e}")
            self._active = self._fallback
            self._active.fit(X, y)
            return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._active.predict(X)


class _XGBDMatrixPredictor:
    """Force XGBoost prediction via Booster.predict(DMatrix) to avoid device-mismatch warning."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict(self, X):
        import xgboost as xgb  # 本地 import，避免全局依赖
        booster = self.est.get_booster()

        # 如果训练时有 feature_names，用同一套 names 构造 DMatrix（更稳）
        feat_names = getattr(booster, "feature_names", None)
        dm = xgb.DMatrix(X, feature_names=feat_names)
        return booster.predict(dm)

class _LGBMPredictor:
    """LightGBM sklearn wrapper predictor.
    Force numpy inputs for both fit/predict to avoid sklearn feature_names warnings.
    """

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict(X)


def build_model(
    model_name: str,
    params: Dict[str, Any],
    random_seed: int = 42,
    standardize_linear: bool = True,
    standardize_mlp: bool = True,
    standardize_sequence: bool = True,
    gpu: Optional[Dict[str, Any]] = None,
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

    gpu = gpu or {}
    gpu_enable = bool(gpu.get("enable", False))
    gpu_models = gpu.get("models", {}) or {}

    if name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")

        xgb_params = dict(params)

        # Inject GPU params (do not rely on signature)
        if gpu_enable and bool(gpu_models.get("XGBoost", False)):
            inject = dict(gpu.get("xgboost", {}) or {})

            for k in ("device", "tree_method", "gpu_id"):
                v = inject.get(k, None)
                if v is None:
                    continue
                if (k not in xgb_params) or (xgb_params.get(k) is None):
                    xgb_params[k] = v

            # Log injection once
            try:
                key = (
                    "XGBoost",
                    str(xgb_params.get("device")),
                    str(xgb_params.get("tree_method")),
                    str(xgb_params.get("predictor")),
                )
                if key not in _GPU_LOGGED:
                    logging.getLogger("step2").info(
                        f"[XGBoost] GPU requested: injected_params={{device:{xgb_params.get('device')}, "
                        f"tree_method:{xgb_params.get('tree_method')}, predictor:{xgb_params.get('predictor')}}}"
                    )
                    _GPU_LOGGED.add(key)
            except Exception:  # pragma: no cover
                pass

        # ---- Primary estimator (GPU if configured) ----
        primary = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=-1,
            **xgb_params,
        )
        primary = _XGBDMatrixPredictor(primary)

        # ---- Optional fallback to CPU ----
        if gpu_enable and bool(gpu_models.get("XGBoost", False)) and bool(gpu.get("fallback_to_cpu", True)):
            cpu_params = dict(xgb_params)
            for k in ("device", "tree_method", "predictor", "gpu_id"):
                cpu_params.pop(k, None)

            fallback = XGBRegressor(
                objective="reg:squarederror",
                random_state=random_seed,
                n_jobs=-1,
                **cpu_params,
            )
            fallback = _XGBDMatrixPredictor(fallback)

            return FittedModel(name=name, model=_FitFallback(primary, fallback, tag="XGBoost", fallback_enabled=True),)
        return FittedModel(name=name, model=primary)

    if name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")

        lgb_params = dict(params)

        # Inject GPU params (do not rely on inspect.signature)
        if gpu_enable and bool(gpu_models.get("LightGBM", False)):
            inject = dict(gpu.get("lightgbm", {}) or {})

            # device_type is preferred; some versions use `device`
            # Treat None as missing and allow overwrite
            for k in ("device_type", "device", "gpu_platform_id", "gpu_device_id"):
                v = inject.get(k, None)
                if v is None:
                    continue
                if (k not in lgb_params) or (lgb_params.get(k) is None):
                    lgb_params[k] = v

            # Log injection once
            try:
                key = (
                    "LightGBM",
                    str(lgb_params.get("device_type")),
                    str(lgb_params.get("device")),
                    str(lgb_params.get("gpu_platform_id")),
                    str(lgb_params.get("gpu_device_id")),
                )
                if key not in _GPU_LOGGED:
                    logging.getLogger("step2").info(
                        "[LightGBM] GPU requested: injected_params={"
                        f"device_type:{lgb_params.get('device_type')}, device:{lgb_params.get('device')}, "
                        f"gpu_platform_id:{lgb_params.get('gpu_platform_id')}, gpu_device_id:{lgb_params.get('gpu_device_id')}"
                        "}"
                    )
                    _GPU_LOGGED.add(key)
            except Exception:  # pragma: no cover
                pass

        # ---- Primary estimator (GPU if configured) ----
        primary = LGBMRegressor(random_state=random_seed, n_jobs=-1, **lgb_params,)
        primary = _LGBMPredictor(primary)

        # ---- Optional fallback to CPU ----
        if gpu_enable and bool(gpu_models.get("LightGBM", False)) and bool(gpu.get("fallback_to_cpu", True)):
            cpu_params = dict(lgb_params)
            for k in ("device_type", "device", "gpu_platform_id", "gpu_device_id"):
                cpu_params.pop(k, None)

            fallback = LGBMRegressor(
                random_state=random_seed,
                n_jobs=-1,
                **cpu_params,
            )
            fallback = _LGBMPredictor(fallback)

            return FittedModel(name=name, model=_FitFallback(primary, fallback, tag="LightGBM", fallback_enabled=True),)
        return FittedModel(name=name, model=primary)

    if name == "MLP":
        # If GPU is enabled for MLP, switch to TorchMLPRegressor.
        if torch is not None and gpu_enable and bool(gpu_models.get("MLP", False)):
            device = str((gpu.get("device") or params.get("device") or "cuda:0"))

            # Map sklearn-style grid names -> torch params
            hidden = params.get("hidden_layer_sizes", (64, 32))
            activation = params.get("activation", "relu")
            dropout = float(params.get("dropout", 0.0))
            # sklearn uses alpha / learning_rate_init / max_iter
            weight_decay = float(params.get("alpha", 0.0))
            lr = float(params.get("learning_rate_init", params.get("lr", 1e-3)))
            batch_size = int(params.get("batch_size", 128))
            epochs = int(params.get("max_iter", params.get("epochs", 30)))

            reg = TorchMLPRegressor(
                hidden_layer_sizes=tuple(int(x) for x in hidden),
                activation=str(activation),
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                standardize=bool(standardize_mlp),
                random_seed=random_seed,
                device=device,
            )
            return FittedModel(name=name, model=reg)

        # Default: sklearn MLP (CPU)
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
        # If GPU is enabled for this torch model, inject a default device unless user overrides.
        if gpu_enable and bool(gpu_models.get(name, False)):
            torch_params.setdefault("device", str(gpu.get("device") or "cuda:0"))

        reg = TorchSequenceRegressor(
            arch=name,
            seq_len=seq_len,
            standardize=standardize_sequence,
            random_seed=random_seed,
            **torch_params,
        )
        return FittedModel(name=name, model=reg, is_sequence=True, seq_len=seq_len)

    raise ValueError(f"Unknown model: {model_name}")
