# -*- coding: utf-8 -*-
"""Step3：滚动重要性可视化

输入
- tables/rolling_importance_long.csv（由 rolling_importance.py 生成）

输出
- figures/rolling_importance_lines_*.png
- figures/rolling_importance_heatmap_*.png
- figures/rolling_importance_jaccard_*.png

图形
1) lines：top-k 特征的重要性随时间变化（窗口结束日期为横轴）
2) heatmap：top-k 特征 x 窗口 的重要性矩阵（便于观察结构变化）
3) jaccard：每个窗口 top-k 集合与上一窗口 top-k 的 Jaccard 相似度

所有注释采用中文。不使用 seaborn。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.step3.plot_utils import setup_matplotlib_cn, save_figure


def _normalize_within_window(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """对每个窗口内的重要性做归一化（可选）。"""
    how = str(how or "none").lower()
    if how in ("none", "off", "false"):
        return df

    out = df.copy()
    if how == "sum_to_one":
        denom = out.groupby(["model", "H", "method", "window_end"], dropna=False)["importance"].transform(lambda x: np.sum(x))
    elif how == "abs_sum_to_one":
        denom = out.groupby(["model", "H", "method", "window_end"], dropna=False)["importance"].transform(lambda x: np.sum(np.abs(x)))
    else:
        return out

    denom = denom.replace(0.0, np.nan)
    out["importance"] = out["importance"] / denom
    return out


def _smooth_by_feature(df: pd.DataFrame, smooth_windows: int) -> pd.DataFrame:
    k = int(smooth_windows or 1)
    if k <= 1:
        return df

    out = []
    for keys, g in df.groupby(["model", "H", "method", "feature"], dropna=False):
        gg = g.sort_values("window_end").copy()
        gg["importance"] = gg["importance"].rolling(k, min_periods=1).mean()
        out.append(gg)
    return pd.concat(out, ignore_index=True) if len(out) > 0 else df


def _plot_lines(sub: pd.DataFrame, top_k_lines: int, out_path: str) -> None:
    # 选择平均重要性最高的 top-k
    rank = (
        sub.groupby("feature", dropna=False)["importance"].mean()
        .sort_values(ascending=False)
        .head(int(top_k_lines))
    )
    feats = list(rank.index)

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(111)

    for f in feats:
        g = sub[sub["feature"] == f].sort_values("window_end")
        ax.plot(g["window_end"].to_numpy(), g["importance"].to_numpy(dtype=float), label=str(f))

    ax.set_title("滚动重要性（折线）")
    ax.set_xlabel("窗口结束日期")
    ax.set_ylabel("重要性")
    ax.legend(ncol=2, fontsize=9)

    save_figure(fig, out_path, dpi=220)


def _plot_heatmap(sub: pd.DataFrame, top_k: int, out_path: str) -> None:
    # 选择平均重要性最高的 top-k
    rank = (
        sub.groupby("feature", dropna=False)["importance"].mean()
        .sort_values(ascending=False)
        .head(int(top_k))
    )
    feats = list(rank.index)

    sub2 = sub[sub["feature"].isin(feats)].copy()
    sub2 = sub2.sort_values(["feature", "window_end"])

    # pivot: rows=feature, cols=window_end
    mat = sub2.pivot_table(index="feature", columns="window_end", values="importance", aggfunc="mean")
    mat = mat.reindex(index=feats)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_title("滚动重要性（热力图）")
    ax.set_ylabel("特征")
    ax.set_xlabel("窗口序号（按时间排序）")

    # x 轴刻度过密时，仅显示少量
    n_cols = mat.shape[1]
    if n_cols > 1:
        ticks = np.linspace(0, n_cols - 1, num=min(8, n_cols), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(mat.columns[i])[:10] for i in ticks], rotation=0)

    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([str(f) for f in feats])

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    save_figure(fig, out_path, dpi=220)


def _plot_jaccard(sub: pd.DataFrame, top_k: int, out_path: str) -> None:
    # 每个窗口取 top-k 特征集合，然后计算与上一窗口的 jaccard
    sub = sub.sort_values(["window_end", "importance"], ascending=[True, False]).copy()
    windows = sorted(sub["window_end"].unique())

    top_sets = {}
    for w in windows:
        g = sub[sub["window_end"] == w].sort_values("importance", ascending=False)
        top_sets[w] = set(g["feature"].head(int(top_k)).astype(str).tolist())

    xs = []
    ys = []
    for i in range(1, len(windows)):
        a = top_sets[windows[i - 1]]
        b = top_sets[windows[i]]
        inter = len(a & b)
        union = len(a | b)
        j = float(inter) / float(union) if union > 0 else float("nan")
        xs.append(windows[i])
        ys.append(j)

    fig = plt.figure(figsize=(10, 3.8))
    ax = fig.add_subplot(111)
    ax.plot(pd.to_datetime(xs), ys)
    ax.set_title(f"Top-{int(top_k)} 特征集合的 Jaccard 相似度（相邻窗口）")
    ax.set_xlabel("窗口结束日期")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, 1)

    save_figure(fig, out_path, dpi=220)


def run_rolling_importance_plots(
    run_cfg: Dict[str, Any],
    long_path: str,
    out_dirs: Dict[str, str],
) -> Dict[str, Any]:
    cfg = dict(run_cfg.get("rolling_importance_plots", {}) or {})
    if not bool(cfg.get("enable", True)):
        return {}

    tables_dir = Path(out_dirs["tables"])
    figures_dir = Path(out_dirs["figures"])

    df = pd.read_csv(long_path, encoding="utf-8")
    df["window_end"] = pd.to_datetime(df["window_end"])

    df = _normalize_within_window(df, how=cfg.get("normalize", "none"))
    df = _smooth_by_feature(df, smooth_windows=int(cfg.get("smooth_windows", 1)))

    methods = [str(x) for x in cfg.get("methods", [])]
    top_k = int(cfg.get("top_k", 20))
    top_k_lines = int(cfg.get("top_k_lines", 8))
    formats = [str(x).lower().lstrip(".") for x in cfg.get("formats", ["png"]) if str(x).strip()]

    written = []

    # 逐 (model,H,method) 画图
    for (model, H, method), sub in df.groupby(["model", "H", "method"], dropna=False):
        method = str(method)
        if methods and method not in methods:
            continue

        base = f"rolling_importance_{model}_H{int(H)}_{method}"

        # 1) 折线
        out1 = str(figures_dir / f"{base}_lines.{formats[0]}")
        _plot_lines(sub, top_k_lines=top_k_lines, out_path=out1)
        written.append(out1)

        # 2) 热力图
        out2 = str(figures_dir / f"{base}_heatmap.{formats[0]}")
        _plot_heatmap(sub, top_k=top_k, out_path=out2)
        written.append(out2)

        # 3) Jaccard
        out3 = str(figures_dir / f"{base}_jaccard.{formats[0]}")
        _plot_jaccard(sub, top_k=top_k, out_path=out3)
        written.append(out3)

    return {"figures": written, "tables": [], "datasets": []}
