# -*- coding: utf-8 -*-
"""Step3：滚动重要性可视化

输入
- tables/rolling_importance_long.csv（由 rolling_importance.py 生成）

输出
- figures/rolling_importance_{model}_H{H}_{method}_lines.{png/pdf}
- figures/rolling_importance_{model}_H{H}_{method}_heatmap.{png/pdf}
- figures/rolling_importance_{model}_H{H}_{method}_jaccard.{png/pdf}

图形
1) lines：top-k 特征的重要性随时间变化（窗口结束日期为横轴）
2) heatmap：top-k 特征 × 窗口 的重要性矩阵（便于观察结构变化）
3) jaccard：相邻窗口 top-k 集合的 Jaccard 相似度

工程约定
- 不使用 seaborn，仅使用 matplotlib（便于控风格）
- 该模块只读 CSV 画图，不重训模型

所有注释采用中文。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.step3.plot_utils import save_figure


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
    """对每条“特征-时间”序列做滚动均值平滑（可选）。"""
    k = int(smooth_windows or 1)
    if k <= 1:
        return df

    out = []
    for keys, g in df.groupby(["model", "H", "method", "feature"], dropna=False):
        gg = g.sort_values("window_end").copy()
        gg["importance"] = gg["importance"].rolling(k, min_periods=1).mean()
        out.append(gg)

    return pd.concat(out, ignore_index=True) if len(out) > 0 else df


def _select_top_features(sub: pd.DataFrame, top_k: int) -> List[str]:
    """按窗口内均值重要性排序，选 Top-K 特征。"""
    rank = (
        sub.groupby("feature", dropna=False)["importance"].mean()
        .sort_values(ascending=False)
        .head(int(top_k))
    )
    return [str(x) for x in rank.index]


def _plot_lines(sub: pd.DataFrame, top_k_lines: int, title: str, out_path: str) -> None:
    feats = _select_top_features(sub, top_k=top_k_lines)

    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(111)

    for f in feats:
        g = sub[sub["feature"].astype(str) == str(f)].sort_values("window_end")
        ax.plot(g["window_end"].to_numpy(), g["importance"].to_numpy(dtype=float), label=str(f))

    ax.set_title(title)
    ax.set_xlabel("窗口结束日期")
    ax.set_ylabel("重要性")
    ax.legend(ncol=2, fontsize=9)

    save_figure(fig, out_path, dpi=220)


def _plot_heatmap(sub: pd.DataFrame, top_k: int, title: str, out_path: str) -> None:
    feats = _select_top_features(sub, top_k=top_k)

    sub2 = sub[sub["feature"].astype(str).isin(feats)].copy()
    sub2 = sub2.sort_values(["feature", "window_end"])

    mat = sub2.pivot_table(index="feature", columns="window_end", values="importance", aggfunc="mean")
    mat = mat.reindex(index=feats)

    fig = plt.figure(figsize=(12.2, 5.2))
    ax = fig.add_subplot(111)

    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_ylabel("特征")
    ax.set_xlabel("窗口（按时间排序）")

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


def _plot_jaccard(sub: pd.DataFrame, top_k: int, title: str, out_path: str) -> None:
    sub = sub.sort_values(["window_end", "importance"], ascending=[True, False]).copy()
    windows = sorted(sub["window_end"].unique())

    top_sets: Dict[pd.Timestamp, set] = {}
    for w in windows:
        g = sub[sub["window_end"] == w].sort_values("importance", ascending=False)
        top_sets[w] = set(g["feature"].head(int(top_k)).astype(str).tolist())

    xs: List[pd.Timestamp] = []
    ys: List[float] = []
    for i in range(1, len(windows)):
        a = top_sets[windows[i - 1]]
        b = top_sets[windows[i]]
        inter = len(a & b)
        union = len(a | b)
        j = float(inter) / float(union) if union > 0 else float("nan")
        xs.append(pd.to_datetime(windows[i]))
        ys.append(j)

    fig = plt.figure(figsize=(10.2, 3.9))
    ax = fig.add_subplot(111)
    if len(xs) > 0:
        ax.plot(pd.to_datetime(xs), ys)
    ax.set_title(title)
    ax.set_xlabel("窗口结束日期")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, 1)

    save_figure(fig, out_path, dpi=220)


def run_rolling_importance_plots(
    run_cfg: Dict[str, Any],
    long_path: str,
    out_dirs: Dict[str, str],
) -> Dict[str, Any]:
    """滚动重要性可视化主入口。"""
    cfg = dict(run_cfg.get("rolling_importance_plots", {}) or {})
    if not bool(cfg.get("enable", True)):
        return {}

    figures_dir = Path(out_dirs["figures"]) 
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(long_path, encoding="utf-8")
    df["window_end"] = pd.to_datetime(df["window_end"])

    # 归一化与平滑
    df = _normalize_within_window(df, how=cfg.get("normalize", "none"))
    df = _smooth_by_feature(df, smooth_windows=int(cfg.get("smooth_windows", 1)))

    methods_allow = [str(x) for x in cfg.get("methods", [])]
    top_k = int(cfg.get("top_k", 20))
    top_k_lines = int(cfg.get("top_k_lines", 8))
    formats = [str(x).lower().lstrip(".") for x in cfg.get("formats", ["png"]) if str(x).strip()]
    if not formats:
        formats = ["png"]

    written: List[str] = []

    # 逐 (model,H,method) 画图
    for (model, H, method), sub in df.groupby(["model", "H", "method"], dropna=False):
        model = str(model)
        H = int(H)
        method = str(method)
        if methods_allow and method not in methods_allow:
            continue

        base = f"rolling_importance_{model}_H{H}_{method}"
        title_prefix = f"滚动重要性 | model={model} | H={H} | method={method}"

        for fmt in formats:
            # 1) 折线
            out1 = str(figures_dir / f"{base}_lines.{fmt}")
            _plot_lines(sub, top_k_lines=top_k_lines, title=f"{title_prefix}（Top-{top_k_lines} 折线）", out_path=out1)
            written.append(out1)

            # 2) 热力图
            out2 = str(figures_dir / f"{base}_heatmap.{fmt}")
            _plot_heatmap(sub, top_k=top_k, title=f"{title_prefix}（Top-{top_k} 热力图）", out_path=out2)
            written.append(out2)

            # 3) Jaccard
            out3 = str(figures_dir / f"{base}_jaccard.{fmt}")
            _plot_jaccard(sub, top_k=top_k, title=f"{title_prefix}（Top-{top_k} Jaccard）", out_path=out3)
            written.append(out3)

    return {"figures": written, "tables": [], "datasets": []}
