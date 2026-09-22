# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_bench: np.ndarray) -> float:
    """Out-of-sample R2 vs benchmark prediction y_bench."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_bench = np.asarray(y_bench, dtype=float)
    sse_m = np.sum((y_true - y_pred) ** 2)
    sse_b = np.sum((y_true - y_bench) ** 2)
    if sse_b <= 1e-18:
        return float("nan")
    return float(1.0 - sse_m / sse_b)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bench0 = np.zeros_like(y_true)
    bench_mean = np.full_like(y_true, float(np.mean(y_true)))

    out = {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "oos_r2_vs0": oos_r2(y_true, y_pred, bench0),
        "oos_r2_vsmean": oos_r2(y_true, y_pred, bench_mean),
        "sign_acc": sign_accuracy(y_true, y_pred),
    }
    return out


def score_for_tuning(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return a scalar score where LOWER is better."""
    metric = metric.lower()
    if metric == "rmse":
        return rmse(y_true, y_pred)
    if metric == "mse":
        return mse(y_true, y_pred)
    if metric == "mae":
        return mae(y_true, y_pred)
    raise ValueError(f"Unknown tune_metric: {metric}")
