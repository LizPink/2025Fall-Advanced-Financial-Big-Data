# -*- coding: utf-8 -*-
"""
质检工具（最小必要集）：
- 最终 index 必须与 USD 主轴一致
"""

from __future__ import annotations
import pandas as pd

def check_index_equals_usd(dataset_index: pd.DatetimeIndex, usd_index: pd.DatetimeIndex) -> None:
    """检查最终数据集 index 是否与 USD 主轴完全一致。"""
    if not dataset_index.equals(usd_index):
        raise ValueError("最终数据集 index 与 USD 主时间轴不一致（可能出现新增日期/重复/排序问题）。")

def check_no_weekends(idx: pd.DatetimeIndex) -> None:
    """检查 index 中是否包含周末（如你们希望主轴为交易日，应不含周末）。"""
    weekends = idx[idx.weekday >= 5]
    if len(weekends) > 0:
        raise ValueError(f"发现周末日期（示例前5个）：{list(weekends[:5])}。请检查主轴是否为交易日。")

def summarize_row_filtering(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """输出行筛选统计，便于写作与复现说明。"""
    return pd.DataFrame({
        "metric": ["rows_before", "rows_after", "rows_dropped"],
        "value": [len(df_before), len(df_after), len(df_before) - len(df_after)],
    })
