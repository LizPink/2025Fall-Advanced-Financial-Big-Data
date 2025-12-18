# -*- coding: utf-8 -*-
"""Step3: 拓展分析（extension_analysis）配置

本 Step3 的核心目标：在不改动 Step2 主流程的前提下，基于 Step2 的 OOS 预测结果开展拓展分析。

主要模块
1) 经济层检验（Economic Layer）
   - 允许做空（allow_short）
   - 基准策略（baselines）：cash / always_long / always_short
   - 开平仓阈值（tau_list）作为稳健性分析
   - 非重叠持有期评估（non-overlapping）+ 相位全扫（phase_sweep）
   - 交易成本（tc_bps）：按仓位变化计费

2) VIX 分时期检验（Regime / Subperiod）
   - conditional：按 VIX 状态对“交易样本”分组计算绩效（不改变时间顺序）
   - episode：把 VIX 状态合并成连续时段（便于叙事/可视化）
   - 阈值无前视：train_only（默认）/ expanding（可选）
   - 去抖：hysteresis（双阈值）+ min_spell_days（最短持续期）

3) 滚动窗口重要性稳定性
   - permutation importance（通用）
   - TreeSHAP（树模型：XGBoost/LightGBM）
   - 生成折线图/热力图/Jaccard 稳定性图

输出目录（默认）：../output/step3
"""

from __future__ import annotations

from typing import Any, Dict, List


RUN: Dict[str, Any] = {
    # -----------------------------
    # 路径（相对 extension_analysis/pipeline.py 解析）
    # -----------------------------
    "step1_dataset_dir": "../output/step1/datasets",
    "step2_output_dir": "../output/step2",
    "output_dir": "../output/step3",

    # 是否读取 Step2 的 config_snapshot.json 来复用 test split / backtest_mode 等口径
    # True：优先采用 Step2 口径，保证 Step3 与 Step2 可比
    # False：仅使用本 Step3 配置（不推荐）
    "inherit_step2_snapshot": True,

    # 日志
    "log_level": "INFO",

    # Matplotlib 中文字体候选（用于图标题/坐标轴中文）
    "cn_font_candidates": [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ],

    # -----------------------------
    # 选择：哪些模型 / 哪些 H 参与 Step3
    # 说明：Step3 会扫描 Step2 的 predictions_*.csv，再按这里筛选
    # -----------------------------
    "selection": {
        "models_include": ["Lasso", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM"],     # "ALL" 或 ["Lasso","XGBoost",...]
        "models_exclude": [],
        "H_include": "ALL",          # "ALL" 或 [1,5,10,20]
        "H_exclude": [],
    },

    # -----------------------------
    # 烟雾测试：快速验证流水线（建议首次跑通用 True）
    # -----------------------------
    "smoke_test": {
        "enable": False,
        "max_models": 2,
        "max_windows": 3,
    },

    # -----------------------------
    # 输入契约校验（强烈建议开启）
    # -----------------------------
    "io_contract": {
        "enable": True,
        "check_y_true_alignment": True,   # 把 predictions 的 y_true 与 Step1 的 USD_{H} 做对齐检查（抽样/全量）
        "alignment_max_abs_diff": 1e-10,  # 允许的最大绝对误差
        "alignment_sample_n": 200,        # 抽样检查条数；None 表示全量（可能慢）
    },

    # -----------------------------
    # 经济层检验
    # -----------------------------
    "economic": {
        "enable": True,

        # 交易规则
        "allow_short": True,
        "tau_list": [0.0, 0.0005, 0.001],  # 开平仓阈值：|y_pred| <= tau -> 空仓
        "tc_bps": 1.0,                     # 单边成本（bps），按仓位变化计费

        # 评估口径
        "annual_days": 252,
        "use_non_overlapping": True,
        "phase_sweep": True,               # phase=0..H-1 全扫

        # 基准策略
        "baselines": ["cash", "always_long", "always_short"],

        # 输出
        "save_trades_long": True,          # 保存“交易级别”长表（供 VIX/可视化复用）
        "save_equity_fig": True,
        "plot_phase": 0,                   # 画资金曲线时使用哪个相位（0..H-1）
        "plot_strategies": ["model", "always_long", "cash"],
    },

    # -----------------------------
    # VIX 分时期（Regime）
    # -----------------------------
    "vix_regime": {
        "enable": True,
        "vix_col": "VIX",                      # Step1 数据集中 VIX 列名
        "methods": ["conditional", "episode"],

        # regime 贴标时点：
        # - "entry"：用 VIX_t 给 t->t+H 这笔交易贴标签（推荐，避免持有期内信息）
        "regime_on": "entry",

        # 阈值模式：分位数 or 固定阈值
        "mode": "quantile",                    # "quantile" | "fixed"

        # 阈值来源（无前视）：
        # - "train_only"：用训练/验证期固定阈值，再用于测试期
        # - "expanding"：对每个日期用历史 expanding 分位数（并 shift 1 天）
        "threshold_source": "train_only",      # "train_only" | "expanding"

        # hysteresis（双阈值）：进入/退出 low/high 的分位数
        "quantiles": {
            "enter_low": 0.30,
            "exit_low": 0.40,
            "enter_high": 0.70,
            "exit_high": 0.60,
        },

        # 固定阈值（可作为稳健性对照）
        "fixed_thresholds": {
            "enter_low": 15.0,
            "exit_low": 18.0,
            "enter_high": 25.0,
            "exit_high": 22.0,
        },

        # episode 去抖
        "min_spell_days": 5,

        # 输出
        "save_equity_shading_fig": True,
        "plot_top_n": 0,                        # 0 表示生成“全部” vixshade 图片；>0 则仅生成 Sharpe 较高的前 N 个 (model,H)
    },

    # -----------------------------
    # 滚动窗口重要性
    # -----------------------------
    "rolling_importance": {
        "enable": True,
        "window": 252,
        "step": 21,
        "sample_n": 500,
        "n_repeats": 3,
        "random_state": 42,

        # 只对哪些模型计算 rolling importance（None 表示跟随 selection）
        "only_models": ["Lasso", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM"],

        # TreeSHAP（仅树模型）
        "enable_treeshap_for_tree_models": True,

        # 运行保护
        "max_windows": None,                    # 限制最多计算多少个窗口（None=全算）
        "min_train_size": 200,                  # 训练样本太少则跳过窗口

        # 进度监控：每隔多少个窗口输出一次 INFO 日志（1 表示每个窗口都输出）
        "progress_every_windows": 1,
    },

    # -----------------------------
    # 滚动重要性可视化（只读 CSV，不重训模型）
    # -----------------------------
    "rolling_importance_plots": {
        "enable": True,
        "methods": ["permutation", "treeshap_xgb", "treeshap_lgbm"],
        "top_k": 20,
        "top_k_lines": 8,
        "normalize": "abs_sum_to_one",   # "none" | "sum_to_one" | "abs_sum_to_one"
        "smooth_windows": 1,              # 1=不平滑；3/5=轻度平滑
        "formats": ["png"],             # 可加 "pdf" 直接用于论文
    },

    # -----------------------------
    # Manifest（实验元数据）
    # -----------------------------
    "manifest": {
        "enable": True,
        "capture_versions": True,
    },
}
