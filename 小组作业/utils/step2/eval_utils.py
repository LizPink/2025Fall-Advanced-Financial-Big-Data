# -*- coding: utf-8 -*-
"""Evaluation utilities: statistical metrics + simple economic layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def sign_acc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def oos_r2(y_true, y_pred, y_base) -> float:
    """Out-of-sample R^2 relative to a baseline prediction y_base."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_base) ** 2)
    if den <= 0:
        return np.nan
    return float(1.0 - num / den)

# ---- Simple economic layer ----
def annualize_return(r_daily: np.ndarray, freq: int = 252) -> float:
    return float(np.mean(r_daily) * freq)

def annualize_vol(r_daily: np.ndarray, freq: int = 252) -> float:
    return float(np.std(r_daily, ddof=1) * np.sqrt(freq))

def sharpe_ratio(r_daily: np.ndarray, freq: int = 252) -> float:
    vol = annualize_vol(r_daily, freq=freq)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(annualize_return(r_daily, freq=freq) / vol)

def max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(np.min(dd))

def economic_layer(y_true: np.ndarray, y_pred: np.ndarray, tc_bps: float = 0.0) -> Dict[str, float]:
    """A minimal trading rule: position = sign(y_pred), realized return = position * y_true - tc * turnover.
    NOTE: This is a toy layer for demonstration; extend it later.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pos = np.sign(y_pred)
    pos[np.isnan(pos)] = 0.0

    # turnover: abs change in position
    turnover = np.abs(np.diff(pos, prepend=pos[:1]))
    tc = (tc_bps / 10000.0) * turnover

    r = pos * y_true - tc
    cum = np.cumprod(1.0 + r)  # approximate, ok for small returns

    out = {
        "ann_ret": annualize_return(r),
        "ann_vol": annualize_vol(r),
        "sharpe": sharpe_ratio(r),
        "max_dd": max_drawdown(cum),
        "turnover_mean": float(np.mean(turnover)),
    }
    return out
