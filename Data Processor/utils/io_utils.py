# -*- coding: utf-8 -*-
"""
IO 工具：读取 Excel、统一日期列/数值列、数值清洗等。
所有注释采用中文，便于组内协作与论文复现。
"""

from __future__ import annotations
import re
from typing import Optional, Tuple
import pandas as pd
import numpy as np

def _parse_human_number(x):
    """
    将类似 '7.26K', '1.2M', '3B' 等格式转换为数值。
    若无法解析则返回原值（后续再 to_numeric 会变为 NaN）。
    """
    if isinstance(x, (int, float, np.number)) or x is None:
        return x
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)([KMB])", s, flags=re.IGNORECASE)
        if m:
            num = float(m.group(1))
            suf = m.group(2).upper()
            mult = {"K": 1e3, "M": 1e6, "B": 1e9}[suf]
            return num * mult
        # 普通数字字符串
        try:
            return float(s)
        except Exception:
            return x
    return x

def read_series_from_excel(
    excel_path: str,
    sheet_name: str,
    date_col: str,
    value_col: str,
) -> pd.Series:
    """
    读取指定 sheet 的 (date, value) 并返回按日期排序的一维 Series。
    - date_col：日期列名（可能为中文/英文）
    - value_col：数值列名（通常是'收盘'或某个指标列）
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if date_col not in df.columns:
        raise KeyError(f"[{sheet_name}] 找不到日期列：{date_col}，当前列：{list(df.columns)}")
    if value_col not in df.columns:
        raise KeyError(f"[{sheet_name}] 找不到数值列：{value_col}，当前列：{list(df.columns)}")

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = out["value"].map(_parse_human_number)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date")
    # 若存在重复日期，保留最后一个（通常代表最新修订或同日多行）
    out = out.drop_duplicates(subset=["date"], keep="last")
    s = out.set_index("date")["value"]
    s.name = sheet_name
    return s

def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    import os
    os.makedirs(path, exist_ok=True)
