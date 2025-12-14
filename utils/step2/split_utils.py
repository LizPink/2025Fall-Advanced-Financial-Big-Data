# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd


def train_test_split_time(
    dates: np.ndarray,
    mode: str = "last_ratio",
    test_ratio: float = 0.2,
    test_years: int = 3,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train_val and test by time order.

    dates: datetime64 array sorted ascending.
    Returns: (train_val_idx, test_idx)
    """
    n = len(dates)
    if n < 10:
        raise ValueError("Too few samples for splitting.")

    dt = pd.to_datetime(dates)

    if mode == "last_ratio":
        if not (0.0 < test_ratio < 0.9):
            raise ValueError("test_ratio must be in (0, 0.9)")
        cut = int(round(n * (1 - test_ratio)))
        cut = max(1, min(n - 1, cut))
        train_idx = np.arange(0, cut, dtype=int)
        test_idx = np.arange(cut, n, dtype=int)
        return train_idx, test_idx

    if mode == "last_years":
        end = dt.iloc[-1]
        start = end - pd.DateOffset(years=int(test_years))
        test_mask = dt >= start
        if test_mask.sum() < 10:
            # fallback to last_ratio
            return train_test_split_time(dates, mode="last_ratio", test_ratio=test_ratio)
        test_idx = np.where(test_mask.to_numpy())[0].astype(int)
        train_idx = np.where(~test_mask.to_numpy())[0].astype(int)
        return train_idx, test_idx

    if mode == "date_range":
        if test_start is None or test_end is None:
            raise ValueError("date_range mode requires test_start and test_end")
        start = pd.to_datetime(test_start)
        end = pd.to_datetime(test_end)
        test_mask = (dt >= start) & (dt <= end)
        if test_mask.sum() < 10:
            raise ValueError("date_range yields too few test samples")
        test_idx = np.where(test_mask.to_numpy())[0].astype(int)
        train_idx = np.where(~test_mask.to_numpy())[0].astype(int)
        return train_idx, test_idx

    raise ValueError(f"Unknown test_mode: {mode}")


@dataclass(frozen=True)
class TimeSeriesCVConfig:
    mode: str  # expanding | rolling
    initial_train_size: int
    val_size: int
    step_size: int
    train_window: Optional[int] = None  # rolling window length


def iter_time_series_splits(
    n_samples: int,
    cfg: TimeSeriesCVConfig,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, val_idx) splits inside a contiguous dataset [0..n_samples).

    Expanding:
        train: [0, t)
        val:   [t, t+val)
        t starts at initial_train_size and increases by step_size
    Rolling:
        train: [t-train_window, t)
        val:   [t, t+val)
    """
    if cfg.initial_train_size <= 0 or cfg.val_size <= 0 or cfg.step_size <= 0:
        raise ValueError("CV sizes must be positive")

    t = cfg.initial_train_size
    while t + cfg.val_size <= n_samples:
        if cfg.mode == "expanding":
            train_start = 0
        elif cfg.mode == "rolling":
            if cfg.train_window is None:
                raise ValueError("rolling requires train_window")
            train_start = max(0, t - int(cfg.train_window))
        else:
            raise ValueError(f"Unknown cv mode: {cfg.mode}")

        train_idx = np.arange(train_start, t, dtype=int)
        val_idx = np.arange(t, t + cfg.val_size, dtype=int)

        if len(train_idx) >= 30 and len(val_idx) >= 10:
            yield train_idx, val_idx

        t += cfg.step_size


def apply_embargo_purge(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    embargo_size: int = 0,
    purge: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optional defense against label overlap around split boundary.

    This implementation is conservative and simple:
    - embargo: drop last embargo_size samples from train
    - purge:   drop first embargo_size samples from val (same size for simplicity)

    NOTE: User currently prefers purge/embargo off by default.
    """
    embargo_size = int(embargo_size or 0)
    if embargo_size <= 0:
        return train_idx, val_idx

    if len(train_idx) > embargo_size:
        train_idx = train_idx[: len(train_idx) - embargo_size]
    else:
        train_idx = train_idx[:0]

    if purge:
        if len(val_idx) > embargo_size:
            val_idx = val_idx[embargo_size:]
        else:
            val_idx = val_idx[:0]
    return train_idx, val_idx
