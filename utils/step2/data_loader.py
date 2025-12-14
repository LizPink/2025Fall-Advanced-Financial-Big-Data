# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadedDataset:
    H: int
    df: pd.DataFrame
    dates: np.ndarray  # datetime64
    X: np.ndarray      # float64
    y: np.ndarray      # float64
    feature_names: List[str]
    y_col: str
    date_col: str


def load_dataset(
    dataset_dir: str,
    filename_template: str,
    H: int,
    date_col: str = "date",
    y_prefix: str = "USD",
    feature_cols: Union[str, Sequence[str]] = "ALL",
    drop_cols: Optional[Sequence[str]] = None,
) -> LoadedDataset:
    """Load Step1 output Data_D_{H}.xlsx and return numpy arrays.

    Notes:
    - filename_template example: "Data_D_{H}.xlsx"
    - y column default: f"{y_prefix}_{H}" e.g. USD_5
    """
    drop_cols = list(drop_cols or [])
    y_col = f"{y_prefix}_{H}"

    path = Path(dataset_dir).expanduser().resolve() / filename_template.format(H=H)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_excel(path, sheet_name="dataset")
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not found in columns: {list(df.columns)}")
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not found in columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Select features
    if feature_cols == "ALL":
        feats = [c for c in df.columns if c not in ([date_col, y_col] + drop_cols)]
    else:
        feats = list(feature_cols)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise KeyError(f"Feature columns missing: {missing}")
        feats = [c for c in feats if c not in drop_cols and c not in (date_col, y_col)]

    X = df[feats].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    dates = df[date_col].to_numpy()

    # Safety checks
    if np.isnan(X).any() or np.isnan(y).any():
        # Step1 should have dropped missing rows; if still exists, raise to avoid silent leakage.
        raise ValueError("NaN detected in Step2 inputs. Please re-check Step1 outputs.")

    return LoadedDataset(
        H=H,
        df=df,
        dates=dates,
        X=X,
        y=y,
        feature_names=feats,
        y_col=y_col,
        date_col=date_col,
    )
