# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 关键：强制使用非GUI后端，避免 Tkinter 报错
import matplotlib.pyplot as plt


def plot_pred_vs_true(
    out_path: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(dates, y_true, label="y_true")
    plt.plot(dates, y_pred, label="y_pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residual_hist(
    out_path: str,
    residuals: np.ndarray,
    title: str,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    r = residuals[np.isfinite(residuals)]
    plt.hist(r, bins=50)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(
    out_path: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    m = np.isfinite(y_pred)
    plt.scatter(y_true[m], y_pred[m], s=6)
    plt.title(title)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance_bar(
    out_path: str,
    feature_names: List[str],
    importances: np.ndarray,
    top_k: int = 20,
    title: str = "Feature importance",
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imp = np.asarray(importances, dtype=float)
    order = np.argsort(-imp)[: int(top_k)]
    names = [feature_names[i] for i in order]
    vals = imp[order]

    plt.figure(figsize=(8, max(4, 0.25 * len(order) + 1)))
    plt.barh(range(len(order)), vals[::-1])
    plt.yticks(range(len(order)), names[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
