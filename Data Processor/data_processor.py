# -*- coding: utf-8 -*-
"""
DataProcessor：按既定原则构造“日度版本”建模数据集，并输出 Excel（含数据字典、缺失率、描述统计、配置快照等）。

核心口径（请勿在不同文件中重复实现，防止口径漂移）：
1) USD 作为日度主时间轴；
2) shift y：y_t = r_{t+h}（h=forecast_horizon_days）；
3) 低频（月/季）变量：先施加发布滞后（月滞后1、季滞后1），再映射到日度并 forward fill；
4) 跨市场节假日缺口：carry-forward（forward fill）；
5) 每个变量按 config 显式指定 transform（none/diff/log_diff/pct_change）；
6) 期限结构差：仅对同时有 3M 与 10Y 的国家（US/UK/GER）计算 10Y-3M。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from utils.io_utils import read_series_from_excel, ensure_dir
from utils.transform_utils import apply_transform
from utils.feature_utils import compute_technical_indicators, compute_term_spreads, compute_usd_return
from utils.check_utils import check_index_equals_usd, check_no_weekends, summarize_row_filtering


# ==============================
# 配置读取
# ==============================
def load_config() -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    """
    从 config.py 导入配置。你们后续若改为 yaml/json，也可在此处统一入口。
    """
    import importlib
    cfg = importlib.import_module("config")
    return cfg.RUN, cfg.FEATURES, cfg.VARIABLES, cfg.LOWFREQ_LAG


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """将嵌套字典展开成 (key, value) 列表，便于输出 config_snapshot。"""
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            rows.extend(_flatten_dict(v, key))
        else:
            rows.append((key, v))
    return rows


# ==============================
# 低频处理：先滞后，再 asof 映射到日度
# ==============================
def lowfreq_to_daily_asof(
    lowfreq_series: pd.Series,
    daily_index: pd.DatetimeIndex,
    lag_periods: int,
) -> pd.Series:
    """
    将低频序列映射到日度（USD 主轴）：
    1) 先按期滞后 lag_periods（避免前视偏差）
    2) 再以 merge_asof 的方式，将最近一期（<= 当日）的低频值映射到每个日度日期
    """
    s = lowfreq_series.sort_index().copy()
    if lag_periods > 0:
        s = s.shift(lag_periods)

    # merge_asof 需要 DataFrame
    low = s.reset_index()
    low.columns = ["date", "value"]
    low = low.dropna(subset=["date"]).sort_values("date")

    daily = pd.DataFrame({"date": daily_index}).sort_values("date")

    merged = pd.merge_asof(
        daily,
        low,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    out = merged.set_index("date")["value"]
    out = out.reindex(daily_index)
    # 合理的 carry-forward：若低频数据起点晚于日度主轴，起点之前仍会 NaN（这是正常信息缺失）
    out.name = lowfreq_series.name
    return out


# ==============================
# 数据字典与统计表
# ==============================
def build_data_dictionary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """将变量元信息整理成 data_dictionary。"""
    dd = pd.DataFrame(rows)
    # 固定列顺序（便于阅读与版本对比）
    cols = ["name", "freq", "sheet", "date_col", "value_col", "transform", "note"]
    for c in cols:
        if c not in dd.columns:
            dd[c] = ""
    dd = dd[cols]
    return dd


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """缺失统计：缺失数、缺失率、首末可用日期等。"""
    out = []
    for col in df.columns:
        s = df[col]
        miss_n = int(s.isna().sum())
        total = int(len(s))
        miss_rate = miss_n / total if total > 0 else np.nan

        non_na = s.dropna()
        first_date = non_na.index.min() if len(non_na) else pd.NaT
        last_date = non_na.index.max() if len(non_na) else pd.NaT

        # 最长连续缺失长度（简单实现：按 isna 分段计数）
        is_na = s.isna().astype(int)
        max_run = 0
        run = 0
        for v in is_na.values:
            if v == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0

        out.append({
            "variable": col,
            "missing_n": miss_n,
            "missing_rate": miss_rate,
            "first_non_missing_date": first_date,
            "last_non_missing_date": last_date,
            "max_consecutive_missing": max_run,
        })
    return pd.DataFrame(out).sort_values("missing_rate", ascending=False)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    描述统计：count/mean/std/min/25%/50%/75%/max，并补充 skew/kurt（有助于识别分布特征）。
    """
    desc = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    # 补充偏度与峰度
    desc["skew"] = df.skew(numeric_only=True)
    desc["kurt"] = df.kurt(numeric_only=True)
    return desc.reset_index().rename(columns={"index": "variable"})


# ==============================
# 主流程
# ==============================
def run_data_processor() -> str:
    RUN, FEATURES, VARIABLES, LOWFREQ_LAG = load_config()

    # 输出路径
    out_root = RUN["output_dir"]
    datasets_dir = os.path.join(out_root, "datasets")
    tables_dir = os.path.join(out_root, "tables")
    figures_dir = os.path.join(out_root, "figures")
    meta_dir = os.path.join(out_root, "meta")
    for d in [datasets_dir, tables_dir, figures_dir, meta_dir]:
        ensure_dir(d)

    # 1) 读取 USD 主轴（close 水平）
    usd_close = read_series_from_excel(
        RUN["daily_excel_path"],
        RUN["y_sheet"],
        RUN["y_date_col"],
        RUN["y_value_col"],
    ).sort_index()

    # 主时间轴 = USD 的日期（可能包含周末；是否保留由配置控制）
    master_index = pd.DatetimeIndex(usd_close.index).sort_values()
    if not RUN.get("keep_weekends", False):
        master_index = master_index[master_index.weekday < 5]

    # 2) 对齐所有日度变量到主轴（水平值），并 carry-forward
    aligned_levels: Dict[str, pd.Series] = {}
    var_meta_rows: List[Dict[str, Any]] = []

    for v in VARIABLES:
        name = v["name"]
        freq = v["freq"].lower().strip()

        if freq != "daily":
            continue

        s_raw = read_series_from_excel(
            RUN["daily_excel_path"],
            v["sheet"],
            v["date_col"],
            v["value_col"],
        ).sort_index()

        # 先对齐到 USD 主轴，并 forward fill（跨市场节假日缺口 carry-forward）
        s_aligned = s_raw.reindex(master_index).ffill()
        s_aligned.name = name
        aligned_levels[name] = s_aligned

        var_meta_rows.append({
            "name": name,
            "freq": "daily",
            "sheet": v["sheet"],
            "date_col": v["date_col"],
            "value_col": v["value_col"],
            "transform": v["transform"],
            "note": "日度变量；先对齐USD主轴并carry-forward，再执行transform",
        })

    # 注意：USD 自身 close 用于标签与技术指标（不作为普通自变量从 VARIABLES 中读取）
    usd_close_aligned = usd_close.reindex(master_index).ffill()
    usd_close_aligned.name = "USD_close"

    # 3) 构造日度自变量（对齐后执行 transform）
    X_parts = []
    for v in VARIABLES:
        if v["freq"].lower().strip() != "daily":
            continue
        name = v["name"]
        transform = v["transform"]
        level = aligned_levels.get(name)
        if level is None:
            continue
        x = apply_transform(level, transform)
        x.name = name
        X_parts.append(x)

    X_daily = pd.concat(X_parts, axis=1)

    # 4) 技术指标（基于 USD close）
    tech = compute_technical_indicators(
        close=usd_close_aligned,
        tech_windows_days=FEATURES["tech_windows_days"],
        lags_days=FEATURES["lags_days"],
    )
    # 技术指标元信息
    for col in tech.columns:
        var_meta_rows.append({
            "name": col,
            "freq": "derived",
            "sheet": RUN["y_sheet"],
            "date_col": RUN["y_date_col"],
            "value_col": RUN["y_value_col"],
            "transform": "derived",
            "note": "基于USD close/收益构造的技术指标",
        })

    # 5) 期限结构差（US/UK/GER）
    spreads = compute_term_spreads(aligned_levels={
        # 这里需要 US_3M/US_10Y 等“水平值”，而 aligned_levels 的键是 name（如 US_3M）
        k: v for k, v in aligned_levels.items()
    })
    for col in spreads.columns:
        var_meta_rows.append({
            "name": col,
            "freq": "derived",
            "sheet": "US/UK/GER rates",
            "date_col": "日期",
            "value_col": "收盘",
            "transform": "derived",
            "note": "期限结构差：10Y-3M，仅US/UK/GER",
        })

    # 6) 低频变量：先滞后，再映射到日度
    lowfreq_parts = []
    for v in VARIABLES:
        freq = v["freq"].lower().strip()
        if freq not in {"monthly", "quarterly"}:
            continue

        s_lf = read_series_from_excel(
            RUN["lowfreq_excel_path"],
            v["sheet"],
            v["date_col"],
            v["value_col"],
        ).sort_index()
        # transform 在低频自身上执行（这里 lowfreq 是一维，且不会涉及跨市场假期）
        s_lf_t = apply_transform(s_lf, v["transform"])
        s_lf_t.name = v["name"]

        lag_periods = LOWFREQ_LAG[freq]
        s_daily = lowfreq_to_daily_asof(
            lowfreq_series=s_lf_t,
            daily_index=master_index,
            lag_periods=lag_periods,
        )
        lowfreq_parts.append(s_daily)

        var_meta_rows.append({
            "name": v["name"],
            "freq": freq,
            "sheet": v["sheet"],
            "date_col": v["date_col"],
            "value_col": v["value_col"],
            "transform": v["transform"],
            "note": f"{freq}变量；先滞后{lag_periods}期，再asof映射到日度并carry-forward",
        })

    X_lowfreq = pd.concat(lowfreq_parts, axis=1) if lowfreq_parts else pd.DataFrame(index=master_index)

    # 7) 合并特征
    X = pd.concat([X_daily, tech, spreads, X_lowfreq], axis=1)

    # 8) 构造标签 y：y_t = r_{t+h}（shift y）
    h = int(RUN["forecast_horizon_days"])
    usd_r = compute_usd_return(usd_close_aligned)
    y = usd_r.shift(-h)
    y.name = f"USD_{h}"

    # 9) 合并为最终数据集（先保留未删行版本，用于缺失统计）
    dataset_raw = pd.concat([y, X], axis=1)
    # 质检：index 应与 USD 主轴一致
    check_index_equals_usd(dataset_raw.index, master_index)

    # 10) 生成缺失统计（原始）
    miss_raw = missing_summary(dataset_raw)
    # 11) 生成描述统计（对数值列）
    desc = descriptive_stats(dataset_raw.select_dtypes(include=[np.number]))

    # 12) 形成建模数据：默认策略 = 删除任意含 NaN 的行（保证模型训练不出错）
    dataset_model = dataset_raw.dropna(axis=0, how="any")
    row_filter = summarize_row_filtering(dataset_raw, dataset_model)

    # 13) 输出 Excel 工作簿
    file_stem = f"{RUN['dataset_prefix']}_{h}"
    out_path = os.path.join(datasets_dir, f"{file_stem}.xlsx")

    dd = build_data_dictionary(var_meta_rows)

    # 配置快照
    config_rows = _flatten_dict({"RUN": RUN, "FEATURES": FEATURES, "LOWFREQ_LAG": LOWFREQ_LAG})
    config_snapshot = pd.DataFrame(config_rows, columns=["key", "value"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        dataset_model.to_excel(writer, sheet_name="dataset", index=True)
        dd.to_excel(writer, sheet_name="data_dictionary", index=False)
        miss_raw.to_excel(writer, sheet_name="missing_summary", index=False)
        desc.to_excel(writer, sheet_name="descriptive_stats", index=False)
        config_snapshot.to_excel(writer, sheet_name="config_snapshot", index=False)
        row_filter.to_excel(writer, sheet_name="row_filtering", index=False)

    # 同时在 meta/ 下保存一份归档配置（避免覆盖导致不可追溯）
    meta_path = os.path.join(meta_dir, f"{file_stem}_config_snapshot.xlsx")
    with pd.ExcelWriter(meta_path, engine="openpyxl") as writer:
        config_snapshot.to_excel(writer, sheet_name="config_snapshot", index=False)
        dd.to_excel(writer, sheet_name="data_dictionary", index=False)

    return out_path


if __name__ == "__main__":
    out = run_data_processor()
    print(f"已生成：{out}")
