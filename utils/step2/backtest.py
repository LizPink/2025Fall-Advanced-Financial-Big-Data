# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class BacktestResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    test_idx: np.ndarray


def walk_forward_predict(
    model_name: str,
    build_model_fn,
    X: np.ndarray,
    y: np.ndarray,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    best_params: Dict[str, Any],
    mode: str = "expanding",
    train_window: Optional[int] = None,
    refit_every: int = 1,
    random_seed: int = 42,
    verbose: bool = True,
) -> BacktestResult:
    """Walk-forward OOS prediction on test_idx.

    Each iteration fits on available data up to current test block start,
    then predicts next block of size refit_every (or remaining length).
    """
    mode = mode.lower()
    refit_every = max(1, int(refit_every))

    y_pred = np.full((len(test_idx),), np.nan, dtype=float)
    y_true = y[test_idx].astype(float)

    # We'll treat indices as ordered time indices
    test_positions = np.arange(len(test_idx))

    block_start = 0
    while block_start < len(test_idx):
        block_end = min(len(test_idx), block_start + refit_every)
        cur_test_idx = test_idx[block_start:block_end]

        # Available training indices include: train_val + all test points strictly before current block
        past_test_idx = test_idx[:block_start]
        avail = np.concatenate([train_val_idx, past_test_idx]).astype(int)
        avail = np.unique(avail)
        avail.sort()

        if mode == "expanding":
            tr_idx = avail
        elif mode == "rolling":
            if train_window is None:
                raise ValueError("rolling requires train_window")
            tr_idx = avail[-int(train_window) :] if len(avail) > int(train_window) else avail
        else:
            raise ValueError(f"Unknown backtest_mode: {mode}")

        model = build_model_fn(model_name, best_params, random_seed=random_seed)
        model.fit(X[tr_idx], y[tr_idx])
        pred_block = model.predict(X[cur_test_idx])

        # handle padded nan (sequence models)
        if np.isnan(pred_block).any():
            mask = ~np.isnan(pred_block)
            # only fill where valid
            y_pred[block_start:block_end][mask] = pred_block[mask]
        else:
            y_pred[block_start:block_end] = pred_block

        if verbose:
            print(f"[OOS] {model_name} predicted {block_end}/{len(test_idx)}")

        block_start = block_end

    return BacktestResult(y_true=y_true, y_pred=y_pred, test_idx=test_idx)
