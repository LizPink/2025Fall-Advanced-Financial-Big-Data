# -*- coding: utf-8 -*-
"""Step3 输入契约校验工具

设计目标
- Step3 只读取 Step1/Step2 的产出文件；因此必须在最早阶段做校验，避免“文件存在但内容异常”导致整条流水线结果不可用。

校验内容
1) Step2 predictions_*.csv
   - 必须包含：date, y_true, y_pred
   - date 可解析为 datetime，严格升序且无重复
2) （可选）y_true 与 Step1 数据集中 USD_{H} 的对齐检查
   - 抽样比对 date 对齐后的数值差

注：所有注释采用中文（按小组规范）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_PRED_COLS = ["date", "y_true", "y_pred"]


@dataclass(frozen=True)
class ContractReport:
    ok: bool
    message: str
    n_rows: int
    n_missing_date: int
    n_missing_y_true: int
    n_missing_y_pred: int


def read_predictions_csv(path: str) -> pd.DataFrame:
    """读取 Step2 的预测文件，并强制解析 date。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"预测文件不存在: {p}")

    df = pd.read_csv(p, encoding="utf-8")
    # 兼容可能的列名大小写差异
    for c in REQUIRED_PRED_COLS:
        if c not in df.columns:
            raise KeyError(f"预测文件缺少必要列 {c}: {p} | columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def validate_predictions_df(df: pd.DataFrame, path_hint: str = "") -> ContractReport:
    """校验 predictions DataFrame 的列、日期顺序、缺失值。"""
    for c in REQUIRED_PRED_COLS:
        if c not in df.columns:
            return ContractReport(False, f"缺少必要列: {c} | {path_hint}", 0, 0, 0, 0)

    n_rows = int(len(df))
    n_missing_date = int(df["date"].isna().sum())
    n_missing_y_true = int(pd.isna(df["y_true"]).sum())
    n_missing_y_pred = int(pd.isna(df["y_pred"]).sum())

    if n_rows <= 0:
        return ContractReport(False, f"文件为空 | {path_hint}", 0, n_missing_date, n_missing_y_true, n_missing_y_pred)

    if n_missing_date > 0:
        return ContractReport(False, f"date 解析失败（存在 NaT）| {path_hint}", n_rows, n_missing_date, n_missing_y_true, n_missing_y_pred)

    # 日期严格升序
    if not df["date"].is_monotonic_increasing:
        return ContractReport(False, f"date 非升序（需要按日期排序）| {path_hint}", n_rows, n_missing_date, n_missing_y_true, n_missing_y_pred)

    # 无重复
    if df["date"].duplicated().any():
        return ContractReport(False, f"date 存在重复值（同一天出现多行）| {path_hint}", n_rows, n_missing_date, n_missing_y_true, n_missing_y_pred)

    # 缺失值
    if n_missing_y_true > 0 or n_missing_y_pred > 0:
        return ContractReport(False, f"y_true 或 y_pred 存在缺失值 | {path_hint}", n_rows, n_missing_date, n_missing_y_true, n_missing_y_pred)

    return ContractReport(True, "OK", n_rows, n_missing_date, n_missing_y_true, n_missing_y_pred)


def check_y_true_alignment(
    pred_df: pd.DataFrame,
    step1_df: pd.DataFrame,
    y_col_step1: str,
    max_abs_diff: float = 1e-10,
    sample_n: Optional[int] = 200,
    random_state: int = 42,
) -> Tuple[bool, str]:
    """校验 predictions 的 y_true 是否与 Step1 的 USD_{H} 对齐。

    方法
    - 按 date 合并（inner join）
    - 抽样（或全量）比较 y_true 与 USD_{H}

    返回
    - ok: bool
    - msg: str（包含最大误差/样本量等）
    """
    if "date" not in step1_df.columns:
        return False, "Step1 数据缺少 date 列，无法对齐校验"
    if y_col_step1 not in step1_df.columns:
        return False, f"Step1 数据缺少目标列 {y_col_step1}，无法对齐校验"

    a = pred_df[["date", "y_true"]].copy()
    b = step1_df[["date", y_col_step1]].copy()

    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])

    m = a.merge(b, on="date", how="inner")
    if len(m) == 0:
        return False, "predictions 与 Step1 在 date 上没有交集，无法对齐校验"

    rng = np.random.default_rng(int(random_state))
    if sample_n is not None and len(m) > int(sample_n):
        idx = rng.choice(len(m), size=int(sample_n), replace=False)
        m = m.iloc[np.sort(idx)]

    diff = (m["y_true"].astype(float) - m[y_col_step1].astype(float)).to_numpy()
    max_abs = float(np.max(np.abs(diff)))

    ok = bool(max_abs <= float(max_abs_diff))
    msg = f"y_true 对齐检查 | n_check={len(m)} | max_abs_diff={max_abs:.3e} | threshold={float(max_abs_diff):.3e}"
    return ok, msg
