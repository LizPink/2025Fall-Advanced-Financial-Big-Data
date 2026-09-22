# -*- coding: utf-8 -*-
"""Hyperparameter search utilities (grid / random) for time-series training windows."""
from __future__ import annotations

from typing import Dict, Any, Iterable, List, Tuple
import numpy as np

def time_train_val_split(train_idx: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split a training index into (train_sub, val_sub) preserving time order."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")
    n = len(train_idx)
    cut = int(np.floor(n * (1.0 - val_ratio)))
    cut = max(1, min(cut, n - 1))
    return train_idx[:cut], train_idx[cut:]

def iter_param_candidates(method: str, grid: Dict[str, List[Any]], n_iter: int, seed: int = 42):
    """Yield parameter dicts for grid or random search."""
    method = str(method).lower().strip()
    keys = list(grid.keys())
    if len(keys) == 0:
        yield {}
        return

    if method == "none":
        yield {}
        return

    if method == "grid":
        # Cartesian product
        from itertools import product
        values = [grid[k] for k in keys]
        for combo in product(*values):
            yield {k: v for k, v in zip(keys, combo)}
        return

    if method == "random":
        rng = np.random.default_rng(seed)
        for _ in range(int(n_iter)):
            cand = {k: rng.choice(grid[k]) for k in keys}
            yield cand
        return

    raise ValueError(f"Unsupported search method: {method}")
