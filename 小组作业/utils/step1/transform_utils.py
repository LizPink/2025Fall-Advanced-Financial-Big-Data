# -*- coding: utf-8 -*-
"""
数据变换工具：
- none：不处理
- diff：一阶差分
- log_diff：对数差分（要求>0）
- pct_change：简单收益率
"""

from __future__ import annotations
import numpy as np
import pandas as pd

ALLOWED_TRANSFORMS = {"none", "diff", "log_diff", "pct_change"}

def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    """
    对齐后的序列上执行变换（对齐与 forward fill 在 DataProcessor 中完成）。
    注意：log_diff 要求序列 > 0，否则将产生 NaN。
    """
    transform = transform.lower().strip()
    if transform not in ALLOWED_TRANSFORMS:
        raise ValueError(f"不支持的 transform：{transform}，可选：{sorted(ALLOWED_TRANSFORMS)}")

    if transform == "none":
        return s
    if transform == "diff":
        return s.diff()
    if transform == "pct_change":
        return s.pct_change()
    if transform == "log_diff":
        s2 = s.where(s > 0)  # 非正值无法取对数
        return np.log(s2).diff()
    raise RuntimeError("transform 分支未覆盖")
