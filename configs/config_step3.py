# -*- coding: utf-8 -*-
"""Step3：拓展分析配置

本 Step3 主要做三件事：
1）经济层检验：把 Step2 的预测结果转换为可解释的交易绩效，并做稳健性（tau/baseline/相位 phase）。
2）VIX 分时期检验：用 VIX 划分风险状态，既输出 conditional（条件绩效）也输出 episode（连续区间）。
3）滚动窗口重要性稳定性：按时间滚动训练并计算 permutation importance /（树模型）TreeSHAP，并输出可视化。

运行方式：
    python run_step3.py

输出：
    output/step3/
        tables/   经济层、VIX 分时期、滚动重要性（长表/均值表）
        datasets/ 策略收益序列、VIX episode 列表等
        figures/  资金曲线、重要性折线/热力图/稳定性图
        meta/     config_snapshot_step3.json、run_manifest.json
        logs/     运行日志

注意：
- Step3 默认以“非重叠持有期（每 H 天再平衡一次）”作为经济层主口径。
- VIX 分位数阈值默认用 train_only（仅训练/验证期）以避免前视。
"""

from __future__ import annotations

from typing import Any, Dict


RUN: Dict[str, Any] = {
    # -----------------------------
    # 输入/输出路径
    # -----------------------------
    "step1_dataset_dir": "../output/step1/datasets",
    "step2_output_dir": "../output/step2",
    "output_dir": "../output/step3",

    # -----------------------------
    # 选择要做拓展分析的模型与预测期 H
    # - 默认扫描 step2/datasets 下的 predictions_{model}_H{H}.csv
    # - 再按 include/exclude 过滤
    # -----------------------------
    "selection": {
        "models_include": "ALL",   # "ALL" 或 ["Lasso", "XGBoost", ...]
        "models_exclude": [],      # 永远优先生效
        "H_include": "ALL",       # "ALL" 或 [1,5,10,20]
    },

    # -----------------------------
    # 经济层检验（Economic Layer）
    # -----------------------------
    "economic": {
        "enable": True,

        # 交易信号
        "allow_short": True,             # 是否允许做空（外汇通常允许）
        "tau_list": [0.0, 0.0005, 0.001],# 建/卖仓位阈值（敏感性分析，不在 test 上挑最优）

        # 基准策略（与模型同口径输出）
        # - cash: 全程空仓
        # - always_long: 全程做多
        # - always_short: 全程做空
        "baselines": ["cash", "always_long", "always_short"],

        # 非重叠持有期与相位稳健性
        "use_non_overlapping": True,     # True：每 H 天取一次样本
        "phase_sweep": True,             # True：phase=0..H-1 全扫

        # 成本与年化
        "tc_bps": 1.0,                   # 单边成本（bps），按仓位变化计费
        "annual_days": 252,

        # 输出开关
        "save_return_series": True,
        "make_equity_fig": True,

        # 可选：未来如果要做“重叠持有期 + HAC t-stat”检验
        "overlap_inference": {
            "enable_hac": False,
            "hac_lags": "H-1",          # 字符串便于表达；实现时解析
        },
    },

    # -----------------------------
    # VIX 分时期检验
    # -----------------------------
    "vix_regime": {
        "enable": True,
        "vix_col": "VIX",

        # 两种输出：
        # - conditional: 条件绩效（不改变时序，只做分组统计）
        # - episode: 连续区间（便于叙事与可视化）
        "methods": ["conditional", "episode"],

        # regime 贴标签口径：
        # - entry: 用 VIX_t 给 (t -> t+H) 这笔交易贴标签（无前视、推荐）
        "regime_on": "entry",

        # 阈值模式：quantile / fixed
        "mode": "quantile",

        # 阈值来源：train_only / expanding
        # - train_only: 用训练/验证期估阈值，固定后用于测试期
        # - expanding: 到 t 为止用历史估阈值（更严格，但计算更重）
        "threshold_source": "train_only",

        # 分位数阈值 + 去抖动（双阈值 hysteresis）
        "quantiles": {
            "enter_low": 0.30,
            "exit_low": 0.40,
            "enter_high": 0.70,
            "exit_high": 0.60,
        },

        # 固定阈值（可用于稳健性对照）
        "fixed_thresholds": {
            "enter_low": 15.0,
            "exit_low": 18.0,
            "enter_high": 25.0,
            "exit_high": 22.0,
        },

        # episode 平滑：持续天数太短的 episode 合并处理
        "min_spell_days": 5,

        # 输出与可视化
        "save_labeled_series": True,
        "make_shading_fig": True,
        "shading_fig_tau": 0.0,          # 只对某个 tau 画阴影资金曲线（避免图太多）
        "shading_fig_strategy": "model",# "model" or "always_long" etc.
    },

    # -----------------------------
    # 滚动窗口重要性（Rolling Importance）
    # -----------------------------
    "rolling_importance": {
        "enable": True,

        # 滚动参数
        "window": 252,
        "step": 21,
        "sample_n": 500,
        "n_repeats": 3,
        "random_state": 42,

        # 仅树模型输出 TreeSHAP
        "enable_treeshap_for_tree_models": True,

        # 控制算力（先 smoke test，避免一次性算太久）
        "max_windows": None,  # None 表示不限制

        # 如果只想对部分模型做滚动重要性，可填列表；None 表示对 selection 的全做
        "only_models": None,
    },

    # -----------------------------
    # 滚动重要性可视化（仅读取 rolling_importance_long.csv，不重复训练）
    # -----------------------------
    "rolling_importance_plots": {
        "enable": True,
        "methods": ["permutation", "treeshap"],
        "top_k": 20,
        "top_k_lines": 8,
        "normalize": "none",      # "none" | "sum_to_one" | "abs_sum_to_one"
        "smooth_windows": 1,        # 1=不平滑；3/5=轻度平滑
        "formats": ["png"],        # 可加 "pdf" 便于直接插入报告

        # 中文字体（按你们机器情况调整；若为空则不改 rcParams）
        "font_sans_serif": ["SimHei", "Arial Unicode MS", "Microsoft YaHei"],
        "fix_unicode_minus": True,
    },

    # -----------------------------
    # Manifest / 日志
    # -----------------------------
    "manifest": {
        "enable": True,
    },
    "log_level": "INFO",
}
