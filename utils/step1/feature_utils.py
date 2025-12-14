# -*- coding: utf-8 -*-
"""
特征工程工具：
- 技术指标：MA 偏离度、布林带宽（BBW）、收益滞后项
- 期限结构差（Term Spread）：仅对同时拥有 3M 和 10Y 的国家计算（US/UK/GER）
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict

def compute_usd_return(close: pd.Series) -> pd.Series:
    """计算 USD 的日对数收益率 r_t = log(S_t) - log(S_{t-1})。"""
    close2 = close.where(close > 0)
    r = np.log(close2).diff()
    r.name = "USD_log_return"
    return r

def compute_technical_indicators(
    close: pd.Series,
    tech_windows_days: List[int],
    lags_days: List[int],
) -> pd.DataFrame:
    """
    在 USD close 上构造技术指标：
    1) MA_gap_w = close/MA_w - 1
    2) BBW_w   = 4 * rolling_std(close,w) / MA_w   （k=2 => 上下轨差=4σ；再除以 MA 做归一化）
    3) return_lag_k = r_t.shift(k-1)，其中 r_t 是当日已知收益（从 t-1 到 t）
    """
    feats = {}
    # MA 与 BBW
    for w in tech_windows_days:
        ma = close.rolling(window=w, min_periods=w).mean()
        std = close.rolling(window=w, min_periods=w).std()
        feats[f"MA_gap_{w}"] = close / ma - 1.0
        feats[f"BBW_{w}"] = (4.0 * std) / ma  # k=2 => 4σ, 再除以MA归一化

    # 收益率滞后项
    r = compute_usd_return(close)
    for k in lags_days:
        feats[f"ret_lag_{k}"] = r.shift(k - 1)

    return pd.DataFrame(feats, index=close.index)

def compute_term_spreads(aligned_levels: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    计算期限利差（10Y - 3M），仅对 US/UK/GER。
    输入 aligned_levels：已对齐到 USD 主轴且完成 forward fill 的“水平值”序列字典。
    输出 DataFrame：包含 US_TermSpread / UK_TermSpread / GER_TermSpread（若可计算）。
    """
    out = {}
    for c in ["US", "UK", "GER"]:
        k3, k10 = f"{c}_3M", f"{c}_10Y"
        if k3 in aligned_levels and k10 in aligned_levels:
            out[f"{c}_TermSpread"] = aligned_levels[k10] - aligned_levels[k3]
    if not out:
        return pd.DataFrame(index=next(iter(aligned_levels.values())).index)
    return pd.DataFrame(out)
