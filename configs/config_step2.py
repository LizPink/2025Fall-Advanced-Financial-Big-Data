# -*- coding: utf-8 -*-
"""Step2: 建模与回测配置

说明
- 文件名中的 'D' 仅表示 Daily 频率标签；预测期统一用 H (horizon) 表示。
- 本 Step2 对每个 H 读取 Step1 产出的 Data_D_{H}.xlsx，并使用其中的目标列 USD_{H} 作为累计对数收益标签。

强约束（建议写入报告）
- 超参选择仅在 train_val 内进行（时间序列 CV：expanding/rolling）。
- OOS 预测采用 walk-forward（每个预测点只使用当时可得的历史数据训练）。

依赖
- pandas/numpy/openpyxl/matplotlib
- scikit-learn/xgboost/lightgbm/torch
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# -----------------------------
# 路径与数据
# -----------------------------
RUN: Dict[str, Any] = {
    # Step1 输出目录（相对 modeling/modeling.py 所在目录解析；建议保持相对路径）
    "step1_dataset_dir": "../output/step1/datasets",
    "output_dir": "../output/step2",

    # 预测期列表（H=holding horizon, in trading days）
    "H_list": [1, 5, 10, 20],

    # 文件模板：Data_D_{H}.xlsx（D=Daily 标签，不代表 horizon）
    "dataset_file_template": "Data_D_{H}.xlsx",

    # 数据列
    "date_col": "date",
    "y_prefix": "USD",  # 目标列名：USD_{H}
    "drop_cols": [],    # 额外丢弃列（若有）
    "feature_cols": "ALL",  # "ALL" or list[str]

    # -------------------------
    # 样本划分：train_val / test
    # -------------------------
    # test_mode:
    # - "last_ratio": 取最后 test_ratio 作为测试集
    # - "last_years": 取最后 test_years 年作为测试集
    # - "date_range": 通过 test_start/test_end 指定测试期（闭区间）
    "test_mode": "last_ratio",
    "test_ratio": 0.2,
    "test_years": 3,
    "test_start": None,   # e.g. "2021-01-01"
    "test_end": None,     # e.g. "2025-12-31"

    # 防止标签重叠导致的乐观偏差（默认关闭）
    "purge": False,
    "embargo_size": 0,  # 建议：若启用，可设为 H 或 H-1

    # -------------------------
    # 超参搜索：train_val 内 TimeSeries CV
    # -------------------------
    "cv_mode": "expanding",  # "expanding" or "rolling"
    "cv_initial_train_size": 1500,  # 初始训练长度（行数）
    "cv_val_size": 252,             # 每折验证长度（行数）
    "cv_step_size": 252,            # 每折向前步长（行数）
    "cv_train_window": 1500,        # rolling 模式的训练窗口长度

    # 评分口径（用于调参）："rmse" | "mae" | "mse"
    "tune_metric": "rmse",

    # 网格搜索策略
    "grid_max_combinations": None,  # 若担心组合爆炸，可设置上限（None 表示不截断）
    "random_seed": 42,

    # -------------------------
    # OOS 预测：walk-forward
    # -------------------------
    "backtest_mode": "expanding",   # "expanding" or "rolling"
    "backtest_train_window": 2000,  # rolling 模式的训练窗口长度
    "refit_every": 1,               # 每隔 refit_every 个测试点重训一次（深度模型建议 > 1）

    # -------------------------
    # 模型库开关
    # -------------------------
    "models": {
        "RandomWalk": False,     # 基准模型
        "Ridge": False,
        "Lasso": False,
        "ElasticNet": False,
        "RandomForest": False,
        "XGBoost": False,
        "LightGBM": True,
        "MLP": True,
        "LSTM": True,          # 默认关闭（训练耗时较长，建议跑通后再打开）
        "Transformer": True,   # 默认关闭（训练耗时较长，建议跑通后再打开）
    },

    # -------------------------
    # 各模型超参网格（可以随时调整）
    # - 线性模型建议配合标准化
    # - 树模型对标准化不敏感
    # -------------------------
    "param_grids": {
        "Ridge": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        },
        "ElasticNet": {
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "RandomForest": {
            "n_estimators": [300],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt"],
        },
        "XGBoost": {
            "n_estimators": [500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 5.0],
        },
        "LightGBM": {
            "n_estimators": [1000],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0],
            "min_child_samples": [20, 50],   # 默认 20；有时调大反而更稳
            "min_split_gain": [0.0],         # 确保不要人为抬高分裂门槛
            "max_depth": [-1, 3, 5],
            "verbose": [-1],                 # 降噪（LightGBM 自己的日志）
            "bagging_freq": [1],             # 让 subsample 真正按迭代生效
            "force_col_wise": [True],
        },
        "MLP": {
            "hidden_layer_sizes": [(32,), (64,), (64,32), (128,64), (128,64,32)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [1e-3, 5e-4],
            "batch_size": [64, 128],
            "max_iter": [800, 1000],
        },
        # 序列模型：建议先固定少量组合，避免组合爆炸
        "LSTM": {
            "seq_len": [20, 40, 60],
            "hidden_size": [32, 64, 128],
            "num_layers": [1, 2, 3],
            "dropout": [0.1],
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128],
            "epochs": [20, 30],
        },
        "Transformer": {
            "seq_len": [20, 40, 80],
            "d_model": [32, 64, 128],
            "nhead": [2, 4, 8],
            "num_layers": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128, 256],
            "epochs": [20, 30],
        },
    },

    # -------------------------
    # 预处理
    # -------------------------
    "standardize_linear": True,  # Ridge/Lasso/EN
    "standardize_mlp": True,
    "standardize_sequence": True,

    # -------------------------
    # 评估与可视化
    # -------------------------
    "metrics": ["rmse", "mae", "mse", "oos_r2_vs0", "oos_r2_vsmean", "sign_acc"],
    "make_figures": True,
    "top_k_features": 20,

    # -------------------------
    # 解释：统一口径 + 树模型加强
    # -------------------------
    "explain": {
        "enable": True,
        "global_method": "permutation",  # all models
        "tree_method": "treeshap",       # xgboost/lightgbm
        "sample_n": 1000,                # 解释时抽样测试集样本数（None=全量）
        "random_state": 42,
    },

    # -------------------------
    # 运行时输出与日志
    # -------------------------
    "log_level": "INFO",
    "show_progress": True,
}
