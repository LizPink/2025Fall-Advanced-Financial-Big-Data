# -*- coding: utf-8 -*-
"""
报告产出工具（表/图）：
- 中文字体：通过候选列表设置 rcParams["font.family"]
- 图2：每个关键外生变量一个子图
- 图3：相关性热力图变量由 config.PLOTS 控制
"""

from __future__ import annotations
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

def setup_matplotlib_cn(font_family_candidates: List[str], base_font_size: int = 11) -> None:
    try:
        plt.rcParams["font.family"] = font_family_candidates
        plt.rcParams["font.size"] = base_font_size
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def table1_variable_summary(data_dictionary: pd.DataFrame, lowfreq_lag: Dict[str, int]) -> pd.DataFrame:
    dd = data_dictionary.copy()

    def _lag(freq):
        f = str(freq).lower()
        return lowfreq_lag.get(f, 0)

    dd["lag_periods"] = dd["freq"].map(_lag)
    cols = ["name", "freq", "sheet", "transform", "lag_periods", "note"]
    for c in cols:
        if c not in dd.columns:
            dd[c] = ""
    out = dd[cols].copy()
    out.rename(columns={
        "name": "变量名",
        "freq": "频率",
        "sheet": "来源Sheet",
        "transform": "Transform",
        "lag_periods": "低频滞后期数",
        "note": "备注",
    }, inplace=True)
    return out

def plot1_y_timeseries(y: pd.Series, title: str, out_path: str, dpi: int = 220) -> None:
    s = y.dropna()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(s.index, s.values)
    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("收益率标签 y")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot2_key_exogenous_subplots(df: pd.DataFrame, cols: List[str], title: str, out_path: str, dpi: int = 220) -> None:
    use_cols = [c for c in cols if c in df.columns]
    if len(use_cols) == 0:
        return
    n = len(use_cols)
    fig_h = max(2.2 * n, 4.0)
    fig = plt.figure(figsize=(10, fig_h))
    fig.suptitle(title)
    for i, c in enumerate(use_cols, start=1):
        ax = fig.add_subplot(n, 1, i)
        s = df[c].dropna()
        ax.plot(s.index, s.values)
        ax.set_ylabel(c)
        if i == n:
            ax.set_xlabel("日期")
        else:
            ax.tick_params(axis="x", labelbottom=False)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot3_correlation_heatmap(df: pd.DataFrame, cols: List[str], title: str, out_path: str, dpi: int = 220) -> None:
    use_cols = [c for c in cols if c in df.columns]
    if len(use_cols) < 2:
        return
    mat = df[use_cols].corr().values
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(use_cols)))
    ax.set_yticks(range(len(use_cols)))
    ax.set_xticklabels(use_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(use_cols, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot4_missing_bar(missing_summary: pd.DataFrame, top_n: int, title: str, out_path: str, dpi: int = 220) -> None:
    ms = missing_summary.copy()
    if "missing_rate" not in ms.columns:
        return
    ms = ms.sort_values("missing_rate", ascending=False).head(top_n)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(ms["variable"].astype(str), ms["missing_rate"].astype(float))
    ax.set_title(title)
    ax.set_ylabel("缺失率")
    ax.set_xlabel("变量（缺失率Top）")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot5_lowfreq_lag_schematic(title: str, out_path: str, dpi: int = 220) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)

    y0 = 0.6
    ax.annotate("", xy=(0.95, y0), xytext=(0.05, y0), arrowprops=dict(arrowstyle="->"))
    ax.text(0.05, y0+0.08, "时间", transform=ax.transAxes)

    ax.text(0.18, y0-0.15, "低频观测期：M/Q", transform=ax.transAxes)
    ax.text(0.18, y0+0.05, "X(M/Q)_t", transform=ax.transAxes)

    ax.annotate("", xy=(0.55, y0), xytext=(0.30, y0), arrowprops=dict(arrowstyle="->"))
    ax.text(0.425, y0+0.12, "发布滞后\n(月+1 / 季+1)", ha="center", transform=ax.transAxes)

    ax.text(0.62, y0-0.15, "映射到日度：asof + forward fill\n(仅使用当日及之前可得信息)", transform=ax.transAxes)
    ax.text(0.62, y0+0.05, "用于预测：y_t = r_{t+h}", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
