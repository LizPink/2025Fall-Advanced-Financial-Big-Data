# -*- coding: utf-8 -*-
"""
配置文件
说明：
1) 我们小组采用USD作为日度主时间轴；
2) 标签采用“持有期对数收益”（h 个交易日）：y_t = log(S_{t+h}) - log(S_t)；
3) 低频（月/季）变量：先施加发布滞后（月滞后1、季滞后1），再映射到日度并 forward fill；
4) 缺口/跨市场节假日：carry-forward（forward fill）；
5) 每个变量需显式指定 transform：none/diff/log_diff/pct_change。
"""

from __future__ import annotations

# ==============================
# 运行参数
# ==============================
RUN = {
    # 输入数据路径（请按需修改）
    "daily_excel_path": r"../raw_data/Raw_Data_Daily.xlsx",
    "lowfreq_excel_path": r"../raw_data/Raw_Data_Monthly+.xlsx",

    # 输出目录（会在当前脚本同级创建 Data/）
    "output_dir": "../output/step1",
    "dataset_prefix": "Data_D",
    "forecast_horizon_days": 20,

    "y_sheet": "USD",
    "y_date_col": "日期",
    "y_value_col": "收盘",

    # 是否输出图表（图1-图5）
    "enable_plots": True,

    # 是否保留周末（一般建议 False）
    "keep_weekends": False,

    # 标签可得性强校验：若为 True，将强制检查 y 是否只在尾部出现缺失（且缺失应连续），
    # 用于防止样本截止日过近导致标签不可构造时被静默 drop。
    "enforce_label_availability": True,
}

# ==============================
# 技术指标参数（可改）
# ==============================
FEATURES = {
    # 日度技术指标窗口（交易日）
    "tech_windows_days": [5, 20, 60],
    # USD 收益率滞后项（return_lag_k = r_t.shift(k-1)）
    "lags_days": [1, 2, 5, 10, 20],
}

# ==============================
# 绘图参数（按你们反馈新增）
# ==============================
PLOTS = {
    "font_family_candidates": [
        "Microsoft YaHei",   # Windows
        "SimHei",            # Windows
        "SimSun",           # 宋体（可选）
        "Arial",            # 兜底
    ],
    "base_font_size": 11,
    "dpi": 220,

    # 图2：关键外生变量（每个变量一个子图）
    "plot2_vars": ["VIX", "US_TermSpread", "WTI", "GOLD", "EPU_D"],

    # 图3：相关性热力图变量（按论文/答辩需要自行精简）
    "plot3_vars": ["USD_1", "VIX", "US_TermSpread", "WTI", "GOLD", "EPU_D", "USD_EUR", "USD_JPY", "USD_GBP", "USD_CNY"],

    "plot4_top_n": 30,
}

# ==============================
# 变量定义
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

    # ---- 日度：情绪/不确定性 ----
    {"name": "VIX",   "sheet": "VIX",     "freq": "daily", "date_col": "日期", "value_col": "收盘", "transform": "none"},
    {"name": "EPU_D", "sheet": "EPU_Day", "freq": "daily", "date_col": "Date", "value_col": "EPU_D", "transform": "log_diff"},

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
