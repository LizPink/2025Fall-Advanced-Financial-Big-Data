# -*- coding: utf-8 -*-
"""Step3：滚动窗口重要性稳定性（Rolling Feature Importance Stability）

目标
- 检查模型在不同时间窗口下的特征重要性是否稳定
- 输出可用于论文附录（例如“特征稳定性分析”）

方法
- permutation importance：通用（所有模型都可尝试），计算量较大但解释一致
- TreeSHAP：仅树模型（XGBoost/LightGBM），速度通常更快、解释更精细

重要说明（工程侧）
- deep 模型在滚动窗口上反复训练代价很大；建议在 config 里只做树模型
- 为了与 Step2 保持口径一致，默认读取 Step2 的 best_params.csv 作为窗口内训练的超参

输出
- tables/rolling_importance_long.csv：长表 (model,H,method,window_end,feature,importance)
- tables/rolling_importance_mean.csv：按 (model,H,method,feature) 聚合均值

所有注释采用中文。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.step2.data_loader import load_dataset
from utils.step2.split_utils import train_test_split_time
from utils.step2.model_registry import build_model
from utils.step2.explain import permutation_importance, treeshap_importance_xgb, treeshap_importance_lgbm


TREE_METHODS = {
    "XGBoost": "treeshap_xgb",
    "LightGBM": "treeshap_lgbm",
}


def _parse_best_params(best_params_csv: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """从 Step2 的 best_params.csv 读取 (model,H)->best_params。"""
    p = Path(best_params_csv)
    if not p.exists():
        raise FileNotFoundError(f"best_params.csv 不存在: {p}")
    df = pd.read_csv(p)
    if "model" not in df.columns or "H" not in df.columns or "params_json" not in df.columns:
        raise KeyError(f"best_params.csv 缺少必要列: model,H,params_json | columns={list(df.columns)}")

    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        model = str(row["model"])
        H = int(row["H"])
        try:
            params = json.loads(row["params_json"]) if isinstance(row["params_json"], str) else {}
        except Exception:
            params = {}
        out[(model, H)] = params
    return out


def _infer_is_tree(model_name: str) -> bool:
    return model_name in ("XGBoost", "LightGBM")


def run_rolling_importance(
    run_cfg: Dict[str, Any],
    combos: List[Dict[str, Any]],
    step1_dataset_dir: str,
    dataset_file_template: str,
    date_col: str,
    y_prefix: str,
    step2_output_dir: str,
    step2_snapshot: Optional[Dict[str, Any]],
    out_dirs: Dict[str, str],
) -> Dict[str, Any]:
    cfg = dict(run_cfg.get("rolling_importance", {}) or {})
    if not bool(cfg.get("enable", True)):
        return {}

    window = int(cfg.get("window", 252))
    step = int(cfg.get("step", 21))
    sample_n = cfg.get("sample_n", 500)
    n_repeats = int(cfg.get("n_repeats", 3))
    random_state = int(cfg.get("random_state", 42))

    only_models = cfg.get("only_models", None)
    if only_models is not None:
        only_models = [str(m) for m in only_models]

    enable_treeshap = bool(cfg.get("enable_treeshap_for_tree_models", True))
    max_windows = cfg.get("max_windows", None)
    if max_windows is not None:
        max_windows = int(max_windows)

    min_train_size = int(cfg.get("min_train_size", 200))

    tables_dir = Path(out_dirs["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    best_params_map = _parse_best_params(str(Path(step2_output_dir) / "tables" / "best_params.csv"))

    rows = []

    # 与 Step2 一致的测试期划分口径
    snap = step2_snapshot or {}
    test_mode = str(snap.get("test_mode", "last_ratio"))
    test_ratio = float(snap.get("test_ratio", 0.2))
    test_years = int(snap.get("test_years", 3))
    test_start = snap.get("test_start", None)
    test_end = snap.get("test_end", None)

    combos_sorted = sorted(combos, key=lambda x: (str(x.get("model")), int(x.get("H"))))

    for c in combos_sorted:
        model = str(c["model"])
        H = int(c["H"])

        if only_models is not None and model not in only_models:
            continue

        data = load_dataset(
            dataset_dir=step1_dataset_dir,
            filename_template=dataset_file_template,
            H=H,
            date_col=date_col,
            y_prefix=y_prefix,
            feature_cols="ALL",
            drop_cols=[],
        )

        # Step2 的 test 划分（用于定义“滚动窗口在测试期上滑动”）
        train_val_idx, test_idx = train_test_split_time(
            dates=data.dates,
            mode=test_mode,
            test_ratio=test_ratio,
            test_years=test_years,
            test_start=test_start,
            test_end=test_end,
        )

        best_params = best_params_map.get((model, H), {})

        # 仅在 test_idx 上滚动
        test_start_i = int(test_idx[0])
        test_end_i = int(test_idx[-1])

        # 窗口以 window_end 为右端点（包含），左端点为 window_end-window+1
        # 训练集使用：window_start 之前的所有样本（expanding），再按需要裁剪成 rolling
        window_end_candidates = list(range(test_start_i + window - 1, test_end_i + 1, step))
        if max_windows is not None:
            window_end_candidates = window_end_candidates[:max_windows]

        for wi, window_end in enumerate(window_end_candidates):
            window_start = window_end - window + 1
            win_idx = np.arange(window_start, window_end + 1, dtype=int)

            # 训练索引（只用 window_start 之前当时可得的历史）
            # 与 Step2 的 backtest_mode 尽量一致：
            # - expanding：0..window_start-1
            # - rolling：仅用最近 backtest_train_window 个样本
            bt_mode = str(snap.get("backtest_mode", "expanding"))
            bt_train_window = int(snap.get("backtest_train_window", 2000))
            if bt_mode == "rolling":
                left = max(0, window_start - bt_train_window)
                train_idx = np.arange(left, window_start, dtype=int)
            else:
                train_idx = np.arange(0, window_start, dtype=int)

            if len(train_idx) < min_train_size:
                continue

            X_train = data.X[train_idx]
            y_train = data.y[train_idx]
            X_win = data.X[win_idx]
            y_win = data.y[win_idx]

            # 建模（复用 Step2 build_model），并注入 best_params
            # 说明：Step3 只做 feature importance，因此不需要 walk-forward 的 refit_every 逻辑
            model_obj, _ = build_model(
                model_name=model,
                params=best_params,
                input_dim=X_train.shape[1],
                standardize_linear=True,
                standardize_mlp=True,
                standardize_sequence=True,
                gpu=snap.get("gpu", {}),
            )

            # 对于序列模型：本项目 build_model 返回的 wrapper 需要序列构造；
            # 为避免 Step3 大幅改造，我们在 Step3 中默认只对非序列模型做 rolling importance。
            if model in ("LSTM", "Transformer"):
                continue

            # 训练
            model_obj.fit(X_train, y_train)

            # permutation importance（通用）
            imp_perm = permutation_importance(
                model=model_obj,
                X_test=X_win,
                y_test=y_win,
                feature_names=data.feature_names,
                metric="rmse",
                n_repeats=n_repeats,
                sample_n=sample_n,
                random_seed=random_state,
            )
            for r in imp_perm:
                rows.append({
                    "model": model,
                    "H": H,
                    "method": "permutation",
                    "window_end": pd.to_datetime(data.dates[window_end]),
                    "window_start": pd.to_datetime(data.dates[window_start]),
                    "n_obs": int(window),
                    "feature": r.get("feature"),
                    "importance": float(r.get("importance", 0.0)),
                })

            # TreeSHAP（仅树模型）
            if enable_treeshap and _infer_is_tree(model):
                if model == "XGBoost":
                    imp_shap = treeshap_importance_xgb(model_obj, X_win, data.feature_names, sample_n=sample_n)
                    method = "treeshap_xgb"
                else:
                    imp_shap = treeshap_importance_lgbm(model_obj, X_win, data.feature_names, sample_n=sample_n)
                    method = "treeshap_lgbm"

                for r in imp_shap:
                    rows.append({
                        "model": model,
                        "H": H,
                        "method": method,
                        "window_end": pd.to_datetime(data.dates[window_end]),
                        "window_start": pd.to_datetime(data.dates[window_start]),
                        "n_obs": int(window),
                        "feature": r.get("feature"),
                        "importance": float(r.get("importance", 0.0)),
                    })

    long_df = pd.DataFrame(rows)
    long_path = str(tables_dir / "rolling_importance_long.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8")

    mean_df = (
        long_df
        .groupby(["model", "H", "method", "feature"], dropna=False)["importance"]
        .mean()
        .reset_index()
        .rename(columns={"importance": "importance_mean"})
    )
    mean_path = str(tables_dir / "rolling_importance_mean.csv")
    mean_df.to_csv(mean_path, index=False, encoding="utf-8")

    return {
        "tables": [long_path, mean_path],
        "datasets": [],
        "figures": [],
        "long_path": long_path,
        "mean_path": mean_path,
    }
