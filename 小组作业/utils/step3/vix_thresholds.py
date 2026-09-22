# -*- coding: utf-8 -*-
"""VIX 分时期阈值与去抖工具

本模块提供两类功能：
1) 阈值计算（quantile / fixed），并支持无前视来源（train_only / expanding）
2) 根据 hysteresis（双阈值）生成稳定的 regime 序列（low/mid/high）

说明
- hysteresis 的核心作用：避免 VIX 在阈值附近来回跳导致状态频繁切换
- min_spell_days 的作用：把过短的 low/high 片段“压回 mid”，减少 episode 过碎

所有注释采用中文（按小组规范）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    enter_low: float
    exit_low: float
    enter_high: float
    exit_high: float


def thresholds_from_config(mode: str, cfg: Dict[str, float]) -> Thresholds:
    """从 config 字典解析阈值。"""
    try:
        return Thresholds(
            enter_low=float(cfg["enter_low"]),
            exit_low=float(cfg["exit_low"]),
            enter_high=float(cfg["enter_high"]),
            exit_high=float(cfg["exit_high"]),
        )
    except Exception as e:
        raise ValueError(f"阈值配置解析失败(mode={mode}): {e}")


def compute_thresholds_train_only(
    vix: pd.Series,
    train_mask: pd.Series,
    quantiles_cfg: Dict[str, float],
) -> Thresholds:
    """train_only：仅用训练/验证期样本计算分位数阈值。"""
    vv = vix.loc[train_mask].dropna()
    if len(vv) < 30:
        raise ValueError("训练期 VIX 样本过少，无法稳定估计分位数阈值")

    return Thresholds(
        enter_low=float(vv.quantile(float(quantiles_cfg["enter_low"]))),
        exit_low=float(vv.quantile(float(quantiles_cfg["exit_low"]))),
        enter_high=float(vv.quantile(float(quantiles_cfg["enter_high"]))),
        exit_high=float(vv.quantile(float(quantiles_cfg["exit_high"]))),
    )


def compute_thresholds_expanding(
    vix: pd.Series,
    quantiles_cfg: Dict[str, float],
    min_history: int = 252,
) -> pd.DataFrame:
    """expanding：对每个日期计算历史分位数阈值，并 shift(1) 天避免前视。

    返回 DataFrame，列为 enter_low/exit_low/enter_high/exit_high。
    """
    s = vix.astype(float).copy()

    def _q(q: float) -> pd.Series:
        return s.expanding(min_periods=int(min_history)).quantile(float(q)).shift(1)

    out = pd.DataFrame({
        "enter_low": _q(float(quantiles_cfg["enter_low"])),
        "exit_low": _q(float(quantiles_cfg["exit_low"])),
        "enter_high": _q(float(quantiles_cfg["enter_high"])),
        "exit_high": _q(float(quantiles_cfg["exit_high"])),
    })
    return out


def assign_regime_hysteresis(vix: pd.Series, thr: Thresholds) -> pd.Series:
    """使用 hysteresis（双阈值）为每个日期分配 regime：low/mid/high。"""
    idx = vix.index
    x = vix.astype(float).to_numpy()

    state = "mid"
    out = []

    for val in x:
        if np.isnan(val):
            out.append(state)
            continue

        if state == "low":
            if val > float(thr.exit_low):
                state = "mid"
        elif state == "high":
            if val < float(thr.exit_high):
                state = "mid"
        else:
            if val <= float(thr.enter_low):
                state = "low"
            elif val >= float(thr.enter_high):
                state = "high"

        out.append(state)

    return pd.Series(out, index=idx, name="regime")


def assign_regime_hysteresis_timevarying(
    vix: pd.Series,
    thr_df: pd.DataFrame,
    fallback_thr: Thresholds,
) -> pd.Series:
    """时间变阈值版本（用于 expanding 分位数）。

    - 当某天阈值为 NaN（历史不够长）时，使用 fallback_thr。
    """
    idx = vix.index
    x = vix.astype(float).to_numpy()

    state = "mid"
    out = []

    cols = ["enter_low", "exit_low", "enter_high", "exit_high"]
    for c in cols:
        if c not in thr_df.columns:
            raise KeyError(f"thr_df 缺少列: {c}")

    thr_vals = thr_df[cols].to_numpy(dtype=float)

    for i, val in enumerate(x):
        t = thr_vals[i]
        if np.isnan(t).any():
            thr = fallback_thr
        else:
            thr = Thresholds(enter_low=t[0], exit_low=t[1], enter_high=t[2], exit_high=t[3])

        if np.isnan(val):
            out.append(state)
            continue

        if state == "low":
            if val > float(thr.exit_low):
                state = "mid"
        elif state == "high":
            if val < float(thr.exit_high):
                state = "mid"
        else:
            if val <= float(thr.enter_low):
                state = "low"
            elif val >= float(thr.enter_high):
                state = "high"

        out.append(state)

    return pd.Series(out, index=idx, name="regime")


def apply_min_spell_days(regime: pd.Series, min_spell_days: int = 0) -> pd.Series:
    """把过短的 low/high 片段压回 mid，减少 episode 过碎。"""
    k = int(min_spell_days or 0)
    if k <= 1:
        return regime

    r = regime.astype(str).copy()
    if len(r) == 0:
        return r

    vals = r.to_numpy()
    starts = [0]
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            starts.append(i)
    starts.append(len(vals))

    out = vals.copy()
    for a, b in zip(starts[:-1], starts[1:]):
        state = vals[a]
        length = b - a
        if state in ("low", "high") and length < k:
            out[a:b] = "mid"

    return pd.Series(out, index=r.index, name=r.name)


def build_episodes(regime: pd.Series) -> pd.DataFrame:
    """把 regime 序列合并为连续 episode。"""
    r = regime.astype(str)
    if len(r) == 0:
        return pd.DataFrame(columns=["start", "end", "length", "regime"])

    idx = pd.to_datetime(r.index)
    vals = r.to_numpy()

    rows = []
    start_i = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[i - 1]:
            rows.append({
                "start": idx[start_i],
                "end": idx[i - 1],
                "length": int(i - start_i),
                "regime": str(vals[i - 1]),
            })
            start_i = i

    return pd.DataFrame(rows)
