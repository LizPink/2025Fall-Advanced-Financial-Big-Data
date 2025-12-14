# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .metrics import score_for_tuning
from .split_utils import apply_embargo_purge


def _grid_to_param_list(grid: Dict[str, List[Any]], max_combinations: Optional[int] = None) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, prod)})
        if max_combinations is not None and len(combos) >= int(max_combinations):
            break
    return combos


@dataclass
class TuneResult:
    best_params: Dict[str, Any]
    best_score: float
    history: List[Dict[str, Any]]  # each record includes params, fold_scores, mean_score


def grid_search_cv(
    model_name: str,
    build_model_fn,
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    param_grid: Dict[str, List[Any]],
    tune_metric: str,
    random_seed: int,
    embargo_size: int = 0,
    purge: bool = False,
    max_combinations: Optional[int] = None,
    verbose: bool = True,
) -> TuneResult:
    """Brute-force grid search on time-series splits, minimizing tune_metric."""
    params_list = _grid_to_param_list(param_grid, max_combinations=max_combinations)
    if len(params_list) == 0:
        params_list = [{}]

    best_params: Dict[str, Any] = {}
    best_score = float("inf")
    history: List[Dict[str, Any]] = []

    for i, params in enumerate(params_list, start=1):
        fold_scores: List[float] = []
        for (tr_idx, va_idx) in splits:
            tr_idx2, va_idx2 = apply_embargo_purge(tr_idx, va_idx, embargo_size=embargo_size, purge=purge)
            if len(tr_idx2) < 30 or len(va_idx2) < 10:
                continue

            model = build_model_fn(model_name, params, random_seed=random_seed)
            model.fit(X[tr_idx2], y[tr_idx2])
            pred = model.predict(X[va_idx2])

            # Some sequence models pad initial nan; drop those
            mask = ~np.isnan(pred)
            if mask.sum() < 10:
                continue
            s = score_for_tuning(tune_metric, y[va_idx2][mask], pred[mask])
            fold_scores.append(float(s))

        mean_score = float(np.mean(fold_scores)) if len(fold_scores) else float("inf")
        record = {
            "model": model_name,
            "params": params,
            "fold_scores": fold_scores,
            "mean_score": mean_score,
        }
        history.append(record)

        if verbose:
            print(f"[TUNE] {model_name} combo {i}/{len(params_list)} mean_{tune_metric}={mean_score:.6g} params={params}")

        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return TuneResult(best_params=best_params, best_score=best_score, history=history)
