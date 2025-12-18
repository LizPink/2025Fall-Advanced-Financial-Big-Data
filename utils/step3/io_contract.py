# -*- coding: utf-8 -*-
"""Step3 输入契约与文件扫描/校验。

本模块的目的：
- 明确 Step3 读取的 Step2 输出文件格式（predictions_*.csv 等）。
- 在运行开始阶段尽早发现问题：缺列、日期无法解析、重复日期、排序问题等。

注意：
- Step2 的 predictions 文件允许 y_pred 出现 NaN（例如序列模型 warmup），Step3 会自动剔除 NaN 行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class PredictionsFile:
    """扫描到的 predictions 文件元信息。"""

    path: str
    model: str
    H: int


_PRED_PATTERN = re.compile(r"predictions_(?P<model>.+)_H(?P<H>\d+)\.csv$")


def scan_predictions(step2_output_dir: str) -> List[PredictionsFile]:
    """扫描 Step2 输出目录，找到全部 predictions_{model}_H{H}.csv。"""

    ds_dir = Path(step2_output_dir).expanduser().resolve() / "datasets"
    files = sorted(ds_dir.glob("predictions_*_H*.csv"))

    out: List[PredictionsFile] = []
    for f in files:
        m = _PRED_PATTERN.search(f.name)
        if not m:
            continue
        out.append(PredictionsFile(path=str(f), model=m.group("model"), H=int(m.group("H"))))
    return out


def filter_predictions(
    items: Sequence[PredictionsFile],
    models_include: object = "ALL",
    models_exclude: Optional[Sequence[str]] = None,
    H_include: object = "ALL",
) -> List[PredictionsFile]:
    """按 config 过滤扫描结果。

    - models_include: "ALL" 或 list[str]
    - models_exclude: list[str]
    - H_include: "ALL" 或 list[int]
    """

    models_exclude = list(models_exclude or [])

    allow_models: Optional[set] = None
    if models_include != "ALL":
        allow_models = set(str(x) for x in models_include)  # type: ignore

    allow_H: Optional[set] = None
    if H_include != "ALL":
        allow_H = set(int(x) for x in H_include)  # type: ignore

    out: List[PredictionsFile] = []
    for it in items:
        if it.model in models_exclude:
            continue
        if allow_models is not None and it.model not in allow_models:
            continue
        if allow_H is not None and it.H not in allow_H:
            continue
        out.append(it)

    # 排序：先 H 再 model，便于输出表格查看
    out.sort(key=lambda x: (x.H, x.model))
    return out


def read_and_validate_predictions(path: str) -> pd.DataFrame:
    """读取并校验 predictions 文件。

    必需列：date, y_true, y_pred

    返回：
    - date 转为 datetime
    - 按 date 升序
    - 剔除 y_pred 非有限（NaN/inf）行
    """

    df = pd.read_csv(path)
    need = ["date", "y_true", "y_pred"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"predictions 缺少必要列 {missing} | file={path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise ValueError(f"predictions 中存在无法解析的 date 行数={bad} | file={path}")

    # 去重与排序
    if df["date"].duplicated().any():
        ndup = int(df["date"].duplicated().sum())
        raise ValueError(f"predictions 中 date 存在重复，重复行数={ndup} | file={path}")

    df = df.sort_values("date").reset_index(drop=True)

    # y_pred 允许 NaN（序列模型 warmup），这里剔除
    df = df[pd.to_numeric(df["y_pred"], errors="coerce").notna()].copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    df = df[df["y_pred"].map(lambda x: pd.notna(x) and pd.notnull(x))].copy()
    df = df[df["y_true"].map(lambda x: pd.notna(x) and pd.notnull(x))].copy()

    # 再次严格剔除 inf
    df = df[df["y_true"].map(lambda x: x == x and x != float("inf") and x != float("-inf"))].copy()
    df = df[df["y_pred"].map(lambda x: x == x and x != float("inf") and x != float("-inf"))].copy()

    if len(df) < 10:
        raise ValueError(f"predictions 有效样本过少（len={len(df)}）| file={path}")

    return df


def check_merge_coverage(
    pred_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    key: str = "date",
    max_missing_ratio: float = 0.05,
    tag: str = "aux",
) -> Dict[str, float]:
    """检查 pred_df 与 aux_df 通过日期合并时的覆盖率。

    返回：
    - merged_n: 合并后样本数
    - missing_ratio: aux 关键列在合并后缺失比例

    说明：
    - Step3 常用该函数检查 VIX 合并缺失率，缺失过高时写入 manifest 并报警。
    """

    m = pred_df.merge(aux_df[[key]], on=key, how="left", indicator=True)
    missing_ratio = float((m["_merge"] != "both").mean())
    stats = {"missing_ratio": missing_ratio, "merged_n": float(len(m))}

    if missing_ratio > float(max_missing_ratio):
        # 不直接 raise，留给上层决定（有时缺失是正常的，例如预测期与VIX日期不完全一致）
        # 这里仅返回统计信息。
        pass
    return stats
