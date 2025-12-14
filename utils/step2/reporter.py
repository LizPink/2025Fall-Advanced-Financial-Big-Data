# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_dirs(out_dir: str) -> Dict[str, Path]:
    base = Path(out_dir).expanduser().resolve()
    dirs = {
        "datasets": base / "datasets",
        "figures": base / "figures",
        "tables": base / "tables",
        "meta": base / "meta",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def append_table_csv(path: str, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if p.exists():
        df0 = pd.read_csv(p)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(p, index=False, encoding="utf-8")


def save_predictions(
    path: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": y_true,
        "y_pred": y_pred,
        "resid": y_true - y_pred,
    })
    if extra:
        for k, v in extra.items():
            df[k] = v
    df.to_csv(path, index=False, encoding="utf-8")
