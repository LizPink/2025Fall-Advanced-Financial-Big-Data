# -*- coding: utf-8 -*-
"""Sequence dataset utilities for LSTM/Transformer."""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tabular arrays into (N, lookback, F) sequences aligned to target at time t (sequence ends at t)."""
    if lookback <= 1:
        # treat as single-step
        return X[:, None, :], y
    n = len(y)
    if n <= lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty((0,))
    X_seq = []
    y_seq = []
    for t in range(lookback - 1, n):
        X_seq.append(X[t - lookback + 1 : t + 1, :])
        y_seq.append(y[t])
    return np.asarray(X_seq, dtype=float), np.asarray(y_seq, dtype=float)
