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

说明
- 你们已决定：Step3 仅评估 6 个机器学习模型（不含深度学习模型）。
- 所有注释采用中文（按小组规范）。
"""

from __future__ import annotations

from typing import Any, Dict


RUN: Dict[str, Any] = {
    # -----------------------------
    # 路径（相对 extension_analysis/pipeline.py 解析）
    # -----------------------------
    "step1_dataset_dir": "../output/step1/datasets",
    "step2_output_dir": "../output/step2",
    "output_dir": "../output/step3",

    # 是否读取 Step2 的 config_snapshot.json 来复用 test split / backtest_mode 等口径
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
        "models_include": ["Lasso", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM"],
        "models_exclude": [],
        "H_include": "ALL",
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
        "check_y_true_alignment": True,
        # 注意：CSV/Excel 往返可能带来极小数值误差；1e-8 仍然非常严格
        "alignment_max_abs_diff": 1e-8,
        "alignment_sample_n": 200,
    },

    # -----------------------------
    # 经济层检验
    # -----------------------------
    "economic": {
        "enable": True,

        # 交易规则
        "allow_short": True,
        "tau_list": [0.0, 0.0005, 0.001],
        "tc_bps": 1.0,

        # 评估口径
        "annual_days": 252,
        "use_non_overlapping": True,
        "phase_sweep": True,

        # 基准策略
        "baselines": ["cash", "always_long", "always_short"],

        # 输出
        "save_trades_long": True,
        "save_equity_fig": True,
        "plot_phase": 0,
        "plot_strategies": ["model", "always_long", "cash"],
    },

    # -----------------------------
    # VIX 分时期（Regime）
    # -----------------------------
    "vix_regime": {
        "enable": True,
        "vix_col": "VIX",
        "methods": ["conditional", "episode"],

        # regime 贴标时点：entry = 用 VIX_t 给 t->t+H 这笔交易贴标签
        "regime_on": "entry",

        "mode": "quantile",
        "threshold_source": "train_only",

        "quantiles": {
            "enter_low": 0.30,
            "exit_low": 0.40,
            "enter_high": 0.70,
            "exit_high": 0.60,
        },

        "fixed_thresholds": {
            "enter_low": 15.0,
            "exit_low": 18.0,
            "enter_high": 25.0,
            "exit_high": 22.0,
        },

        "min_spell_days": 5,

        "save_equity_shading_fig": True,
        "plot_top_n": 3,
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

        "enable_treeshap_for_tree_models": True,

        # 运行保护
        "max_windows": None,
        "min_train_size": 200,
    },

    # -----------------------------
    # 滚动重要性可视化（只读 CSV，不重训模型）
    # -----------------------------
    "rolling_importance_plots": {
        "enable": True,
        "methods": ["permutation", "treeshap_xgb", "treeshap_lgbm"],
        "top_k": 20,
        "top_k_lines": 8,
        "normalize": "abs_sum_to_one",
        "smooth_windows": 1,
        "formats": ["png"],
    },

    # -----------------------------
    # Manifest（实验元数据）
    # -----------------------------
    "manifest": {
        "enable": True,
        "capture_versions": True,
    },
}
