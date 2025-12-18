# -*- coding: utf-8 -*-
"""Step3：滚动窗口重要性稳定性（Rolling Feature Importance Stability）

目标
- 检查模型在不同时间窗口下的特征重要性是否稳定（是否随时间漂移）

方法
- permutation importance：通用（所有模型都可用），解释一致但计算量较大
- TreeSHAP：仅树模型（XGBoost/LightGBM），使用 pred_contrib(s) 计算 mean(|SHAP|)

重要说明（工程侧）
- 深度学习模型在滚动窗口内反复训练成本很高，你们已决定 Step3 仅跑 6 个机器学习模型，因此本模块默认跳过深度模型。
- 为与 Step2 口径一致，默认读取 Step2 的 best_params.csv 作为窗口内训练的超参数来源。

输出
- tables/rolling_importance_long.csv：长表 (model,H,method,window_start,window_end,feature,importance)
- tables/rolling_importance_mean.csv：按 (model,H,method,feature) 聚合均值

所有注释采用中文。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 复用 Step2 数据读取与拆分口径
from utils.step2.data_loader import load_dataset
from utils.step2.split_utils import train_test_split_time

# 复用 Step2 模型工厂与解释工具
from utils.step2.model_registry import build_model
from utils.step2.explain import permutation_importance, treeshap_importance_xgb, treeshap_importance_lgbm


TREE_SHAP_METHOD = {
    "XGBoost": "treeshap_xgb",
    "LightGBM": "treeshap_lgbm",
}


def _parse_best_params(best_params_csv: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """从 Step2 的 best_params.csv 读取 (model,H)->best_params。

    兼容策略
    - 若文件不存在：返回空字典并告警（Step3 将使用模型默认参数）
    - 若 params_json 解析失败：该行视为 {}，不中断全局流程
    """
    p = Path(best_params_csv)
    if not p.exists():
        logging.getLogger("step3").warning(f"best_params.csv 不存在，将使用默认参数: {p}")
        return {}

    df = pd.read_csv(p)
    required = {"model", "H", "params_json"}
    if not required.issubset(set(df.columns)):
        logging.getLogger("step3").warning(
            f"best_params.csv 缺少必要列 {sorted(required)}，将使用默认参数 | columns={list(df.columns)}"
        )
        return {}

    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        model = str(row["model"])
        H = int(row["H"])
        params_raw = row.get("params_json", "{}")
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else {}
            if not isinstance(params, dict):
                params = {}
        except Exception:
            params = {}
        out[(model, H)] = params
    return out


def _maybe_subsample_rows(X: np.ndarray, y: np.ndarray, sample_n: Optional[int], random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """为了加速 permutation/SHAP，在窗口内做一次可选的行抽样。"""
    if sample_n is None:
        return X, y

    n = int(len(y))
    k = int(sample_n)
    if n <= 0 or n <= k:
        return X, y

    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return X[idx], y[idx]


def _iter_window_ends(test_start: int, test_end: int, window: int, step: int) -> List[int]:
    """生成窗口右端点序列（包含右端点）。"""
    first = int(test_start) + int(window) - 1
    if first > int(test_end):
        return []
    return list(range(first, int(test_end) + 1, int(step)))


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
    """滚动窗口重要性主入口。"""
    cfg = dict(run_cfg.get("rolling_importance", {}) or {})
    if not bool(cfg.get("enable", True)):
        return {}

    log = logging.getLogger("step3")

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

    # 与 Step2 一致的测试期划分口径
    snap = step2_snapshot or {}
    test_mode = str(snap.get("test_mode", "last_ratio"))
    test_ratio = float(snap.get("test_ratio", 0.2))
    test_years = int(snap.get("test_years", 3))
    test_start = snap.get("test_start", None)
    test_end = snap.get("test_end", None)

    # backtest 口径（决定窗口内训练集怎么取）
    bt_mode = str(snap.get("backtest_mode", "expanding"))
    bt_train_window = int(snap.get("backtest_train_window", 2000))

    tables_dir = Path(out_dirs["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    best_params_map = _parse_best_params(str(Path(step2_output_dir) / "tables" / "best_params.csv"))

    combos_sorted = sorted(combos, key=lambda x: (str(x.get("model")), int(x.get("H"))))

    rows: List[Dict[str, Any]] = []
    skipped = 0

    for c in combos_sorted:
        model = str(c["model"])
        H = int(c["H"])

        if only_models is not None and model not in only_models:
            continue

        # Step1 数据读取
        data = load_dataset(
            dataset_dir=step1_dataset_dir,
            filename_template=dataset_file_template,
            H=H,
            date_col=date_col,
            y_prefix=y_prefix,
            feature_cols="ALL",
            drop_cols=[],
        )

        train_val_idx, test_idx = train_test_split_time(
            dates=data.dates,
            mode=test_mode,
            test_ratio=test_ratio,
            test_years=test_years,
            test_start=test_start,
            test_end=test_end,
        )

        if len(test_idx) == 0:
            log.warning(f"{model}-H{H} | 测试期为空，跳过 rolling importance")
            continue

        best_params = best_params_map.get((model, H), {})

        test_start_i = int(test_idx[0])
        test_end_i = int(test_idx[-1])

        window_ends = _iter_window_ends(test_start_i, test_end_i, window=window, step=step)
        if max_windows is not None:
            window_ends = window_ends[:max_windows]

        if len(window_ends) == 0:
            log.warning(f"{model}-H{H} | 测试期长度不足 window={window}，无可用窗口")
            continue

        for window_end in window_ends:
            window_start = int(window_end) - int(window) + 1
            win_idx = np.arange(window_start, window_end + 1, dtype=int)

            # 训练索引：只用 window_start 之前当时可得的历史
            if bt_mode == "rolling":
                left = max(0, window_start - bt_train_window)
                train_idx = np.arange(left, window_start, dtype=int)
            else:
                train_idx = np.arange(0, window_start, dtype=int)

            if len(train_idx) < min_train_size:
                skipped += 1
                continue

            X_train = data.X[train_idx]
            y_train = data.y[train_idx]
            X_win = data.X[win_idx]
            y_win = data.y[win_idx]

            # 保护：sample_n 不应超过窗口长度
            sample_n_eff = None if sample_n is None else int(min(int(sample_n), int(len(y_win))))

            # 建模与训练
            try:
                fm = build_model(
                    model_name=model,
                    params=best_params,
                    random_seed=int(snap.get("random_seed", 42)),
                    standardize_linear=True,
                    standardize_mlp=True,
                    standardize_sequence=True,
                    gpu=snap.get("gpu", None),
                )
                fm.fit(X_train, y_train)
            except Exception as e:
                skipped += 1
                log.warning(f"{model}-H{H} | 窗口训练失败，跳过该窗口 | err={type(e).__name__}: {e}")
                continue

            # ----------------------
            # permutation importance
            # ----------------------
            try:
                Xp, yp = _maybe_subsample_rows(X_win, y_win, sample_n=sample_n_eff, random_state=random_state)
                res = permutation_importance(
                    model=fm,
                    X=Xp,
                    y=yp,
                    feature_names=data.feature_names,
                    sample_n=None,              # 已在这里抽样，避免二次抽样
                    n_repeats=n_repeats,
                    random_state=random_state,
                )

                for feat, imp in zip(res.feature_names, res.importances):
                    rows.append({
                        "model": model,
                        "H": H,
                        "method": "permutation",
                        "window_start": pd.to_datetime(data.dates[window_start]),
                        "window_end": pd.to_datetime(data.dates[window_end]),
                        "n_obs": int(len(win_idx)),
                        "feature": str(feat),
                        "importance": float(imp),
                    })
            except Exception as e:
                skipped += 1
                log.warning(f"{model}-H{H} | permutation importance 失败 | err={type(e).__name__}: {e}")

            # ----------------------
            # TreeSHAP（树模型）
            # ----------------------
            if enable_treeshap and model in TREE_SHAP_METHOD:
                try:
                    Xs, _ = _maybe_subsample_rows(X_win, y_win, sample_n=sample_n_eff, random_state=random_state)
                    if model == "XGBoost":
                        shap_res = treeshap_importance_xgb(fm.model, Xs, data.feature_names)
                        method = "treeshap_xgb"
                    else:
                        shap_res = treeshap_importance_lgbm(fm.model, Xs, data.feature_names)
                        method = "treeshap_lgbm"

                    if shap_res is not None:
                        for feat, imp in zip(shap_res.feature_names, shap_res.importances):
                            rows.append({
                                "model": model,
                                "H": H,
                                "method": method,
                                "window_start": pd.to_datetime(data.dates[window_start]),
                                "window_end": pd.to_datetime(data.dates[window_end]),
                                "n_obs": int(len(win_idx)),
                                "feature": str(feat),
                                "importance": float(imp),
                            })
                except Exception as e:
                    skipped += 1
                    log.warning(f"{model}-H{H} | TreeSHAP 失败 | err={type(e).__name__}: {e}")

    long_df = pd.DataFrame(rows)

    long_path = str(tables_dir / "rolling_importance_long.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8")

    if len(long_df) > 0:
        mean_df = (
            long_df
            .groupby(["model", "H", "method", "feature"], dropna=False)["importance"]
            .mean()
            .reset_index()
            .rename(columns={"importance": "importance_mean"})
        )
    else:
        mean_df = pd.DataFrame(columns=["model", "H", "method", "feature", "importance_mean"])

    mean_path = str(tables_dir / "rolling_importance_mean.csv")
    mean_df.to_csv(mean_path, index=False, encoding="utf-8")

    if skipped > 0:
        log.info(f"rolling_importance | 跳过窗口数={skipped}（训练样本不足/训练失败/解释失败等）")

    return {
        "tables": [long_path, mean_path],
        "datasets": [],
        "figures": [],
        "long_path": long_path,
        "mean_path": mean_path,
    }
