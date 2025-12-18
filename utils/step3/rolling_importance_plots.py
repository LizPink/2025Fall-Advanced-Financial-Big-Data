# -*- coding: utf-8 -*-
"""Step3：滚动重要性可视化。

本模块只读取 rolling_importance_long.csv 画图，不重复训练模型。
输出三类图（每个 model×H×method 一套）：
- 折线图：Top-K 特征重要性随时间变化
- 热力图：Top-K 特征 × 时间 的重要性矩阵
- 稳定性图：相邻窗口 Top-K 集合 Jaccard 相似度

注意：
- permutation 与 treeshap 数值量纲不同，不建议直接叠加比较。我们按 method 分开画。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    methods: List[str]
    top_k: int
    top_k_lines: int
    normalize: str
    smooth_windows: int
    formats: List[str]
    font_sans_serif: Optional[List[str]] = None
    fix_unicode_minus: bool = True


def _maybe_configure_fonts(font_sans_serif: Optional[List[str]], fix_unicode_minus: bool) -> None:
    """可选设置中文字体，避免中文乱码/负号显示异常。"""
    if font_sans_serif:
        matplotlib.rcParams["font.sans-serif"] = list(font_sans_serif)
    if fix_unicode_minus:
        matplotlib.rcParams["axes.unicode_minus"] = False


def _smooth(series: pd.Series, w: int) -> pd.Series:
    w = int(w)
    if w <= 1:
        return series
    return series.rolling(w, min_periods=1).mean()


def _normalize_window(df_win: pd.DataFrame, mode: str) -> pd.DataFrame:
    """对单个 window 内的 importance 做归一化（可选）。"""
    mode = str(mode or "none").lower()
    if mode == "none":
        return df_win

    vals = df_win["importance"].astype(float).to_numpy()
    if mode == "sum_to_one":
        s = float(np.sum(vals))
        if s == 0:
            return df_win
        df_win = df_win.copy()
        df_win["importance"] = df_win["importance"] / s
        return df_win

    if mode == "abs_sum_to_one":
        s = float(np.sum(np.abs(vals)))
        if s == 0:
            return df_win
        df_win = df_win.copy()
        df_win["importance"] = np.abs(df_win["importance"].to_numpy()) / s
        return df_win

    raise ValueError(f"未知 normalize: {mode}")


def build_topk_feature_list(df: pd.DataFrame, top_k: int) -> List[str]:
    """按平均重要性选取 Top-K 特征。"""
    top_k = int(top_k)
    g = df.groupby("feature")["importance"].mean().sort_values(ascending=False)
    return list(g.head(top_k).index)


def plot_importance_lines(df: pd.DataFrame, out_path: str, top_k_lines: int, smooth_windows: int) -> None:
    """Top-K 折线图：特征重要性随 window_end 变化。"""
    top_feats = build_topk_feature_list(df, top_k_lines)
    dfp = df[df["feature"].isin(top_feats)].copy()
    dfp = dfp.sort_values(["window_end", "feature"])

    pivot = dfp.pivot_table(index="window_end", columns="feature", values="importance", aggfunc="mean")
    pivot = pivot.sort_index()

    plt.figure(figsize=(10, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, _smooth(pivot[col], smooth_windows), label=str(col))
    plt.title("Top 特征重要性随时间变化")
    plt.xlabel("窗口结束日期")
    plt.ylabel("importance")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_importance_heatmap(df: pd.DataFrame, out_path: str, top_k: int) -> None:
    """热力图：Top-K 特征 × window_end 的重要性矩阵。"""
    top_feats = build_topk_feature_list(df, top_k)
    dfp = df[df["feature"].isin(top_feats)].copy()

    pivot = dfp.pivot_table(index="feature", columns="window_end", values="importance", aggfunc="mean")
    # 排序：按均值降序
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    mat = pivot.to_numpy()

    plt.figure(figsize=(12, max(4, 0.25 * len(order))))
    plt.imshow(mat, aspect="auto")
    plt.title("特征重要性热力图（Top-K）")
    plt.xlabel("窗口序号（按时间）")
    plt.ylabel("特征")
    plt.yticks(range(len(order)), [str(x) for x in order], fontsize=8)
    plt.colorbar(label="importance")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if len(sa | sb) == 0:
        return float("nan")
    return float(len(sa & sb) / len(sa | sb))


def plot_jaccard_stability(df: pd.DataFrame, out_path: str, top_k: int) -> None:
    """稳定性图：相邻窗口 Top-K 集合 Jaccard 相似度。"""
    top_k = int(top_k)
    windows = sorted(df["window_end"].unique())
    if len(windows) < 2:
        return

    topk_by_win: Dict[pd.Timestamp, List[str]] = {}
    for w in windows:
        sub = df[df["window_end"] == w]
        feats = build_topk_feature_list(sub, top_k)
        topk_by_win[pd.to_datetime(w)] = feats

    xs: List[pd.Timestamp] = []
    ys: List[float] = []
    for i in range(1, len(windows)):
        w0 = pd.to_datetime(windows[i - 1])
        w1 = pd.to_datetime(windows[i])
        xs.append(w1)
        ys.append(_jaccard(topk_by_win[w0], topk_by_win[w1]))

    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker="o")
    plt.title("解释稳定性：相邻窗口 Top-K Jaccard")
    plt.xlabel("窗口结束日期")
    plt.ylabel("Jaccard")
    plt.ylim(0, 1)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_all_plots(rolling_long_csv: str, out_dir_figures: str, cfg: PlotConfig) -> List[str]:
    """为 rolling_importance_long.csv 生成全部图像。

    返回：生成的图像路径列表。
    """
    _maybe_configure_fonts(cfg.font_sans_serif, cfg.fix_unicode_minus)

    df = pd.read_csv(rolling_long_csv)
    if df.empty:
        return []

    df["window_end"] = pd.to_datetime(df["window_end"])

    outputs: List[str] = []

    # 分组：每个 model×H×method 一套图
    group_cols = ["model", "H", "method"]
    for (model, H, method), g in df.groupby(group_cols):
        method_s = str(method)
        if method_s not in cfg.methods:
            continue

        # 归一化：按 window_end 分开归一化
        parts = []
        for w, gw in g.groupby("window_end"):
            parts.append(_normalize_window(gw.copy(), cfg.normalize))
        g2 = pd.concat(parts, ignore_index=True)

        tag = f"{model}_H{int(H)}_{method_s}"
        base = Path(out_dir_figures).expanduser().resolve()

        # 折线图
        p1 = base / f"imp_lines_{tag}.png"
        plot_importance_lines(g2, str(p1), top_k_lines=cfg.top_k_lines, smooth_windows=cfg.smooth_windows)
        outputs.append(str(p1))

        # 热力图
        p2 = base / f"imp_heatmap_{tag}.png"
        plot_importance_heatmap(g2, str(p2), top_k=cfg.top_k)
        outputs.append(str(p2))

        # Jaccard
        p3 = base / f"imp_jaccard_{tag}.png"
        plot_jaccard_stability(g2, str(p3), top_k=cfg.top_k)
        outputs.append(str(p3))

        # 额外格式（例如 pdf）
        for fmt in cfg.formats:
            fmt = str(fmt).lower()
            if fmt == "png":
                continue
            # 这里简单复制一份（重新保存），避免重复绘制
            # 若后续需要更细的控制，可以在 plot_* 内支持 format
            for src in [p1, p2, p3]:
                if not src.exists():
                    continue
                dst = src.with_suffix(f".{fmt}")
                # matplotlib 读写不方便直接转换，这里采用再次绘制的方式更稳；
                # 但为了避免重绘，我们保留 png 为默认，pdf 由你们按需启用。
                # 暂时跳过非 png。
                _ = dst

    return outputs

# =========================================================
# Step3 pipeline 入口：封装 make_all_plots
# =========================================================


def run_rolling_importance_plots(cfg_all: Dict[str, Any], dirs: Dict[str, str]) -> Dict[str, Any]:
    """在 Step3 pipeline 中生成滚动重要性图。

    读取：
    - tables/rolling_importance_long.csv

    输出：
    - figures/imp_lines_*.png
    - figures/imp_heatmap_*.png
    - figures/imp_jaccard_*.png
    """

    plot_cfg = PlotConfig.from_dict(cfg_all.get("rolling_importance_plots", {}) or {})
    long_csv = str(Path(dirs["tables"]).expanduser().resolve() / "rolling_importance_long.csv")
    if not Path(long_csv).exists():
        return {"figures": []}

    figs = make_all_plots(long_csv, dirs["figures"], plot_cfg)
    return {"figures": figs}
