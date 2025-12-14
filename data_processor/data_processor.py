# -*- coding: utf-8 -*-
"""
DataProcessor：构造“日度版本”建模数据集，并固定输出：
1) Data/datasets/Data_D_X.xlsx（dataset / data_dictionary / missing_summary / descriptive_stats / config_snapshot / row_filtering）
2) Data/tables/ 表1-表4
3) Data/figures/ 图1-图5（由 enable_plots 控制）
4) Data/meta/ manifest 与配置归档
"""
from __future__ import annotations

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from utils.common.io_utils import read_series_from_excel, ensure_dir
from utils.step1.transform_utils import apply_transform
from utils.step1.feature_utils import compute_technical_indicators, compute_term_spreads, compute_usd_return
from utils.step1.check_utils import check_index_equals_usd, summarize_row_filtering
from utils.step1.report_utils import (
    setup_matplotlib_cn,
    table1_variable_summary,
    plot1_y_timeseries,
    plot2_key_exogenous_subplots,
    plot3_correlation_heatmap,
    plot4_missing_bar,
    plot5_lowfreq_lag_schematic,
)

def load_config() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    import importlib
    cfg = importlib.import_module("configs.config_step1")
    return cfg.RUN, cfg.FEATURES, cfg.PLOTS, cfg.VARIABLES, cfg.LOWFREQ_LAG

def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
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
    RUN, FEATURES, PLOTS, VARIABLES, LOWFREQ_LAG = load_config()

    # 中文字体设置
    setup_matplotlib_cn(PLOTS.get("font_family_candidates", ["Microsoft YaHei", "SimHei"]),
                        base_font_size=int(PLOTS.get("base_font_size", 11)))

    # 输出目录
    out_root = os.path.join(os.path.dirname(__file__), RUN["output_dir"])
    datasets_dir = os.path.join(out_root, "datasets")
    tables_dir = os.path.join(out_root, "tables")
    figures_dir = os.path.join(out_root, "figures")
    meta_dir = os.path.join(out_root, "meta")
    for d in [datasets_dir, tables_dir, figures_dir, meta_dir]:
        ensure_dir(d)

    # 读取路径
    daily_excel = os.path.join(os.path.dirname(__file__), RUN["daily_excel_path"])
    lowfreq_excel = os.path.join(os.path.dirname(__file__), RUN["lowfreq_excel_path"])

    # 1) USD close 与主轴
    usd_close = read_series_from_excel(daily_excel, RUN["y_sheet"], RUN["y_date_col"], RUN["y_value_col"]).sort_index()
    master_index = pd.DatetimeIndex(usd_close.index).sort_values()
    if not RUN.get("keep_weekends", False):
        master_index = master_index[master_index.weekday < 5]
    usd_close_aligned = usd_close.reindex(master_index).ffill()
    usd_close_aligned.name = "USD_close"

    # 2) 日度变量：先对齐再 transform
    aligned_levels: Dict[str, pd.Series] = {}
    meta_rows: List[Dict[str, Any]] = []
    X_daily_list = []

    for v in VARIABLES:
        if v["freq"].lower().strip() != "daily":
            continue
        s_raw = read_series_from_excel(daily_excel, v["sheet"], v["date_col"], v["value_col"]).sort_index()
        s_level = s_raw.reindex(master_index).ffill()
        s_level.name = v["name"]
        aligned_levels[v["name"]] = s_level

        x = apply_transform(s_level, v["transform"])
        x.name = v["name"]
        X_daily_list.append(x)

        meta_rows.append({
            "name": v["name"], "freq": "daily", "sheet": v["sheet"],
            "date_col": v["date_col"], "value_col": v["value_col"],
            "transform": v["transform"],
            "note": "日度：先对齐USD主轴并carry-forward，再执行transform",
        })

    X_daily = pd.concat(X_daily_list, axis=1) if X_daily_list else pd.DataFrame(index=master_index)

    # 3) 技术指标（基于 USD close）
    tech = compute_technical_indicators(
        close=usd_close_aligned,
        tech_windows_days=FEATURES["tech_windows_days"],
        lags_days=FEATURES["lags_days"],
    )
    for col in tech.columns:
        meta_rows.append({
            "name": col, "freq": "derived", "sheet": RUN["y_sheet"],
            "date_col": RUN["y_date_col"], "value_col": RUN["y_value_col"],
            "transform": "derived",
            "note": "技术指标：MA_gap/BBW/ret_lag（基于USD close/收益）",
        })

    # 4) 期限结构差
    spreads = compute_term_spreads(aligned_levels)
    for col in spreads.columns:
        meta_rows.append({
            "name": col, "freq": "derived", "sheet": "Rates",
            "date_col": "日期", "value_col": "收盘",
            "transform": "derived",
            "note": "期限结构差：10Y-3M（仅US/UK/GER）",
        })

    # 5) 低频：transform -> lag -> asof 映射
    low_list = []
    for v in VARIABLES:
        freq = v["freq"].lower().strip()
        if freq not in {"monthly", "quarterly"}:
            continue
        s_lf = read_series_from_excel(lowfreq_excel, v["sheet"], v["date_col"], v["value_col"]).sort_index()
        s_lf_t = apply_transform(s_lf, v["transform"])
        s_lf_t.name = v["name"]
        lag = int(LOWFREQ_LAG[freq])
        s_daily = lowfreq_to_daily_asof(s_lf_t, master_index, lag_periods=lag)
        low_list.append(s_daily)

        meta_rows.append({
            "name": v["name"], "freq": freq, "sheet": v["sheet"],
            "date_col": v["date_col"], "value_col": v["value_col"],
            "transform": v["transform"],
            "note": f"{freq}：先滞后{lag}期，再asof映射到日度并carry-forward",
        })
    X_low = pd.concat(low_list, axis=1) if low_list else pd.DataFrame(index=master_index)

    # 6) 合并特征
    X = pd.concat([X_daily, tech, spreads, X_low], axis=1)

    # 7) 构造标签 y（log(S_{t+h}) - log(S_t)）
    h = int(RUN["forecast_horizon_days"])
    close = usd_close_aligned.astype(float)
    close = close.where(close > 0)
    log_close = np.log(close)

    y = log_close.shift(-h) - log_close
    y.name = f"USD_{h}"

    dataset_raw = pd.concat([y, X], axis=1)
    check_index_equals_usd(dataset_raw.index, master_index)

    # 8) 汇总表
    miss_raw = missing_summary(dataset_raw)
    desc = descriptive_stats(dataset_raw.select_dtypes(include=[np.number]))
    dataset_model = dataset_raw.dropna(axis=0, how="any")
    row_filter = summarize_row_filtering(dataset_raw, dataset_model)

    # 9) 输出主工作簿
    file_stem = f"{RUN['dataset_prefix']}_{h}"
    out_path = os.path.join(datasets_dir, f"{file_stem}.xlsx")

    dd = build_data_dictionary(meta_rows)
    config_snapshot = pd.DataFrame(
        _flatten_dict({"RUN": RUN, "FEATURES": FEATURES, "PLOTS": PLOTS, "LOWFREQ_LAG": LOWFREQ_LAG}),
        columns=["key", "value"]
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        dataset_model.to_excel(writer, sheet_name="dataset", index=True)
        dd.to_excel(writer, sheet_name="data_dictionary", index=False)
        miss_raw.to_excel(writer, sheet_name="missing_summary", index=False)
        desc.to_excel(writer, sheet_name="descriptive_stats", index=False)
        config_snapshot.to_excel(writer, sheet_name="config_snapshot", index=False)
        row_filter.to_excel(writer, sheet_name="row_filtering", index=False)

    # 10) 表格输出
    t1 = table1_variable_summary(dd, LOWFREQ_LAG)
    t1.to_excel(os.path.join(tables_dir, "表1_变量清单与处理口径.xlsx"), index=False)
    miss_raw.to_excel(os.path.join(tables_dir, "表2_缺失率与样本覆盖.xlsx"), index=False)
    desc.to_excel(os.path.join(tables_dir, "表3_描述统计.xlsx"), index=False)
    row_filter.to_excel(os.path.join(tables_dir, "表4_样本筛选统计.xlsx"), index=False)

    # 11) 图形输出
    if RUN.get("enable_plots", False):
        dpi = int(PLOTS.get("dpi", 220))
        plot1_y_timeseries(y, f"图1 USD 持有期对数收益标签（h={h}）时间序列", os.path.join(figures_dir, "图1_USD持有期对数收益标签时间序列.png"),dpi=dpi)

        key_cols = PLOTS.get("plot2_vars", ["VIX", "US_TermSpread", "WTI", "GOLD", "EPU_D"])
        plot2_key_exogenous_subplots(dataset_raw, key_cols, "图2 关键外生变量时间序列", os.path.join(figures_dir, "图2_关键外生变量时间序列.png"), dpi=dpi)

        heat_cols = PLOTS.get("plot3_vars", [y.name, "VIX", "US_TermSpread", "WTI", "GOLD", "EPU_D"])
        plot3_correlation_heatmap(dataset_raw.dropna(axis=0, how="any"), heat_cols, "图3 核心变量相关性热力图", os.path.join(figures_dir, "图3_相关性热力图.png"), dpi=dpi)

        top_n = int(PLOTS.get("plot4_top_n", 30))
        plot4_missing_bar(miss_raw, top_n, f"图4 缺失率分布（Top {top_n}）", os.path.join(figures_dir, "图4_缺失分布可视化.png"), dpi=dpi)

        plot5_lowfreq_lag_schematic("图5 低频变量滞后与日度映射示意", os.path.join(figures_dir, "图5_低频滞后与日度映射示意图.png"), dpi=dpi)

    # 12) manifest + meta 归档
    manifest_df = pd.DataFrame([
        {"type": "table", "id": "T1", "file": "表1_变量清单与处理口径.xlsx"},
        {"type": "table", "id": "T2", "file": "表2_缺失率与样本覆盖.xlsx"},
        {"type": "table", "id": "T3", "file": "表3_描述统计.xlsx"},
        {"type": "table", "id": "T4", "file": "表4_样本筛选统计.xlsx"},
        {"type": "figure", "id": "F1", "file": "图1_USD持有期对数收益标签时间序列.png"},
        {"type": "figure", "id": "F2", "file": "图2_关键外生变量时间序列.png"},
        {"type": "figure", "id": "F3", "file": "图3_相关性热力图.png"},
        {"type": "figure", "id": "F4", "file": "图4_缺失分布可视化.png"},
        {"type": "figure", "id": "F5", "file": "图5_低频滞后与日度映射示意图.png"},
    ])
    with pd.ExcelWriter(os.path.join(meta_dir, f"{file_stem}_manifest.xlsx"), engine="openpyxl") as writer:
        manifest_df.to_excel(writer, sheet_name="manifest", index=False)
        config_snapshot.to_excel(writer, sheet_name="config_snapshot", index=False)
        dd.to_excel(writer, sheet_name="data_dictionary", index=False)

    return out_path

if __name__ == "__main__":
    p = run_data_processor()
    print(f"已生成：{p}")
