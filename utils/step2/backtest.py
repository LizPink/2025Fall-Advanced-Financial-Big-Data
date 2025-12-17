# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

def _split_contiguous(idxs: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into contiguous segments."""
    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0:
        return []
    cuts = np.where(np.diff(idxs) != 1)[0] + 1
    return [seg for seg in np.split(idxs, cuts) if seg.size > 0]

def _predict_sequence_with_lookback(
    model,
    X_full: np.ndarray,
    pred_idx: np.ndarray,
) -> np.ndarray:
    """Predict for sequence models at arbitrary global indices with lookback context.

    TorchSequenceRegressor.predict(X_block) needs X_block length >= seq_len,
    otherwise it returns all-NaN due to warmup padding.
    """
    seq_len = int(getattr(model, "seq_len", 0) or 0)
    if seq_len <= 1:
        return model.predict(X_full[pred_idx])

    pred_idx = np.asarray(pred_idx, dtype=int)
    segs = _split_contiguous(pred_idx)
    out = np.full((len(pred_idx),), np.nan, dtype=float)

    # map global index -> position in pred_idx array
    pos_map = {int(ix): i for i, ix in enumerate(pred_idx.tolist())}

    for seg in segs:
        seg_start = int(seg[0])
        seg_end = int(seg[-1])
        start = max(0, seg_start - (seq_len - 1))
        end = seg_end

        X_slice = X_full[start : end + 1]
        pred_slice = np.asarray(model.predict(X_slice), dtype=float).reshape(-1)

        rel = seg - start
        seg_pred = pred_slice[rel]
        for ix, pv in zip(seg.tolist(), seg_pred.tolist()):
            out[pos_map[int(ix)]] = pv

    return out

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
        # Sequence models (LSTM/Transformer) need lookback context; predicting on a tiny
        # block (e.g., refit_every=1) without history will produce all-NaN outputs.
        if bool(getattr(model, "is_sequence", False)):
            pred_block = _predict_sequence_with_lookback(model, X_full=X, pred_idx=cur_test_idx)
        else:
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
