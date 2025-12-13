# -*- coding: utf-8 -*-
"""
配置文件（可直接修改）
说明：
1) 你们采用 USD 作为日度主时间轴；
2) 标签采用 shift-y：y_t = r_{t+h}；
3) 低频变量（按月/按季）先施加发布滞后（默认：月滞后1、季滞后1）再映射到日度并 forward fill；
4) 跨市场节假日缺口采用 carry-forward（即 forward fill）；
5) 每个变量需要显式指定 transform（none/diff/log_diff/pct_change）。
"""

from __future__ import annotations

# ==============================
# 运行参数
# ==============================
RUN = {
    # 输入数据路径（请按需修改）
    "daily_excel_path": r"Raw_Data/Raw_Data_Daily.xlsx",
    "lowfreq_excel_path": r"Raw_Data/Raw_Data_Monthly+.xlsx",

    # 输出目录（会在当前脚本同级创建 Data/）
    "output_dir": "Data",

    # 输出文件命名：Data_D_X，其中 X 为预测步长（交易日）
    "dataset_prefix": "Data_D",
    "forecast_horizon_days": 1,  # X

    # 因变量（默认 USD）
    "y_sheet": "USD",
    "y_value_col": "收盘",
    "y_date_col": "日期",

    # 是否输出图表（你们当前 Appendix B 未规划，默认关闭）
    "enable_plots": True,

    # 是否保留周末（若USD含周末行，建议默认剔除）
    "keep_weekends": False,

}

# ==============================
# 技术指标参数（默认值，可修改）
# ==============================
FEATURES = {
    # 日度技术指标窗口（交易日）
    "tech_windows_days": [5, 20, 60],
    # USD 收益率滞后项（return_lag_k = r_t.shift(k-1)）
    "lags_days": [1, 2, 5, 10, 20],
}

# ==============================
# 变量定义（建议按“变量一行”维护）
# - name: 输出列名
# - sheet: 来源 sheet
# - freq: daily/monthly/quarterly
# - date_col/value_col: 原始列名
# - transform: none/diff/log_diff/pct_change
# ==============================
VARIABLES = [
    # -------- 日度：汇率与资产价格/指数 --------
    {"name": "USD_Futures", "sheet": "USD_Futures", "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "USD_GBP",     "sheet": "USD_GBP",     "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "USD_EUR",     "sheet": "USD_EUR",     "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "USD_JPY",     "sheet": "USD_JPY",     "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "USD_CNY",     "sheet": "USD_CNY",     "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},

    {"name": "GOLD",        "sheet": "GOLD",        "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "WTI",         "sheet": "WTI",         "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},

    {"name": "US_Mkt_Index","sheet": "US_Mkt_Index","freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "UK_Mkt_Index","sheet": "UK_Mkt_Index","freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "GER_Mkt_Index","sheet":"GER_Mkt_Index","freq":"daily", "date_col":"日期", "value_col":"收盘", "transform":"log_diff"},
    {"name": "JP_Mkt_Index","sheet": "JP_Mkt_Index","freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "CN_Mkt_Index","sheet": "CN_Mkt_Index","freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},
    {"name": "HK_Mkt_Index","sheet": "HK_Mkt_Index","freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "log_diff"},

    # 情绪/不确定性（主线建议：VIX 用水平；EPU 用 log_diff）
    {"name": "VIX",         "sheet": "VIX",         "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "EPU_D",       "sheet": "EPU_Day",     "freq": "daily", "date_col": "Date", "value_col": "EPU_D", "transform": "log_diff"},

    # -------- 日度：利率（默认用水平）--------
    {"name": "US_3M",  "sheet": "US_3M",  "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "US_10Y", "sheet": "US_10Y", "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "UK_3M",  "sheet": "UK_3M",  "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "UK_10Y", "sheet": "UK_10Y", "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "GER_3M", "sheet": "GER_3M", "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "GER_10Y","sheet": "GER_10Y","freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "JP_10Y", "sheet": "JP_10Y", "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "CN_10Y", "sheet": "CN_10Y", "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},

    # -------- 低频：月度 / 季度 --------
    {"name": "GDP",        "sheet": "GDP",        "freq": "quarterly", "date_col": "Date",              "value_col": "GDP",        "transform": "none"},
    {"name": "Output_Gap", "sheet": "Output_Gap", "freq": "quarterly", "date_col": "observation_date", "value_col": "Output_Gap", "transform": "none"},

    {"name": "US_CPI",   "sheet": "CPI", "freq": "monthly", "date_col": "Date", "value_col": "US_CPI",   "transform": "none"},
    {"name": "Euro_CPI", "sheet": "CPI", "freq": "monthly", "date_col": "Date", "value_col": "Euro_CPI", "transform": "none"},
    {"name": "JP_CPI",   "sheet": "CPI", "freq": "monthly", "date_col": "Date", "value_col": "JP_CPI",   "transform": "none"},
    {"name": "CN_CPI",   "sheet": "CPI", "freq": "monthly", "date_col": "Date", "value_col": "CN_CPI",   "transform": "none"},

    #{"name": "EPU_M", "sheet": "EPU_Mon", "freq": "monthly", "date_col": "Date", "value_col": "EPU_M", "transform": "log_diff"},
]

# ==============================
# 低频发布滞后（固定规则）
# ==============================
LOWFREQ_LAG = {
    "monthly": 1,     # 月度变量滞后 1 期
    "quarterly": 1,   # 季度变量滞后 1 期
}
