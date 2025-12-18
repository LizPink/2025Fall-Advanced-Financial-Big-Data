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
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.step2.data_loader import load_dataset
from utils.step2.split_utils import train_test_split_time
import inspect
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

def _subsample_rows(X: np.ndarray, sample_n: Optional[int], random_state: int) -> np.ndarray:
    """对 X 做随机子采样（用于 TreeSHAP 加速；Permutation importance 内部已支持 sample_n）。"""
    if sample_n is None:
        return X
    n = int(len(X))
    k = int(sample_n)
    if n <= k:
        return X
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return np.asarray(X)[idx]


def _unwrap_estimator_for_treeshap(model_obj: Any) -> Any:
    """尽量把各种 wrapper 解包成“原生树模型”估计器，以便调用 pred_contrib(s)。

    说明
    - Step2 的 build_model 会包一层 FittedModel
    - XGBoost/LightGBM 还可能包 _FitFallback / _XGBDMatrixPredictor / _LGBMPredictor
    - 这里尽量向下拿到 XGBRegressor / LGBMRegressor（或 sklearn Pipeline 的最终 model）
    """
    est = model_obj
    # 1) Step2 的 FittedModel
    if hasattr(est, "model"):
        est = getattr(est, "model")
    # 2) GPU fallback wrapper
    if hasattr(est, "_active"):
        est = getattr(est, "_active")
    # 3) predictor wrapper（.est 指向真实 estimator）
    if hasattr(est, "est"):
        est = getattr(est, "est")
    # 4) sklearn Pipeline
    if hasattr(est, "named_steps") and "model" in est.named_steps:
        est = est.named_steps["model"]
    return est


def _build_model_compat(model: str, best_params: Dict[str, Any], X_train: np.ndarray, snap: Dict[str, Any]):
    """
    兼容不同版本/不同签名的 build_model：
    - 只向 build_model 传递其签名中存在的参数
    - 维度参数名可能是 input_dim / input_size / n_features / feature_dim 等
    """
    sig = inspect.signature(build_model)
    allowed = set(sig.parameters.keys())

    # 先准备候选参数
    kwargs = {
        "model_name": model,
        "params": best_params,
        "standardize_linear": True,
        "standardize_mlp": True,
        "standardize_sequence": True,
        "gpu": snap.get("gpu", {}),
    }

    # 自动匹配“输入维度”参数名（如果 build_model 需要）
    dim = int(X_train.shape[1])
    for dim_key in ("input_dim", "input_size", "n_features", "feature_dim"):
        if dim_key in allowed:
            kwargs[dim_key] = dim
            break

    # 过滤：只保留 build_model 签名允许的键
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return build_model(**kwargs)

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

    # 进度监控：每隔多少个窗口输出一次日志（1 表示每个窗口都输出）
    progress_every = cfg.get("progress_every_windows", 1)
    try:
        progress_every = int(progress_every)
    except Exception:
        progress_every = 1
    progress_every = max(1, progress_every)

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

        log.info(
            f"[滚动重要性] 开始: model={model} | H={H} | windows={len(window_end_candidates)} | "
            f"window={window} | step={step} | sample_n={sample_n} | repeats={n_repeats}"
        )

        t_model0 = time.time()

        for wi, window_end in enumerate(window_end_candidates):
            if (wi == 0) or ((wi + 1) % progress_every == 0) or (wi + 1 == len(window_end_candidates)):
                log.info(
                    f"[滚动重要性] 进度: {model}-H{H} 窗口 {wi+1}/{len(window_end_candidates)} | "
                    f"window_end={pd.to_datetime(data.dates[window_end]).date()}"
                )

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

            # 序列模型滚动训练成本高、且需要序列构造；Step3 默认跳过
            if model in ("LSTM", "Transformer"):
                continue

            # 建模（复用 Step2 build_model），并注入 best_params（兼容不同函数签名）
            model_obj = _build_model_compat(
                model=model,
                best_params=best_params,
                X_train=X_train,
                snap=snap,
            )
            # 训练
            t0 = time.time()
            model_obj.fit(X_train, y_train)

            log.debug(
                f"[滚动重要性] {model}-H{H} 窗口 {wi+1}/{len(window_end_candidates)} | "
                f"训练完成 | 用时 {time.time()-t0:.2f}s"
            )

            # permutation importance（通用）
            # 注意：这里复用 utils.step2.explain.permutation_importance 的签名（X/y，不是 X_test/y_test）。
            imp_perm = permutation_importance(
                model=model_obj,
                X=X_win,
                y=y_win,
                feature_names=data.feature_names,
                sample_n=sample_n,
                n_repeats=n_repeats,
                random_state=random_state,
            )

            log.debug(
                f"[滚动重要性] {model}-H{H} 窗口 {wi+1}/{len(window_end_candidates)} | "
                f"Permutation importance 完成 | 返回 {len(imp_perm.feature_names)} 个特征"
            )
            for feat, imp in zip(imp_perm.feature_names, imp_perm.importances):
                rows.append({
                    "model": model,
                    "H": H,
                    "method": "permutation",
                    "window_end": pd.to_datetime(data.dates[window_end]),
                    "window_start": pd.to_datetime(data.dates[window_start]),
                    "n_obs": int(window),
                    "feature": str(feat),
                    "importance": float(imp),
                })

            # TreeSHAP（仅树模型）
            # 说明：utils.step2.explain.treeshap_importance_* 依赖 pred_contrib(s)，需要尽量拿到“原生 estimator”。
            if enable_treeshap and _infer_is_tree(model):
                est_for_shap = _unwrap_estimator_for_treeshap(model_obj)
                X_shap = _subsample_rows(X_win, sample_n=sample_n, random_state=random_state)

                if model == "XGBoost":
                    imp_shap = treeshap_importance_xgb(est_for_shap, X_shap, data.feature_names)
                    method = "treeshap_xgb"
                else:
                    imp_shap = treeshap_importance_lgbm(est_for_shap, X_shap, data.feature_names)
                    method = "treeshap_lgbm"

                if imp_shap is not None:
                    log.debug(
                        f"[滚动重要性] {model}-H{H} 窗口 {wi+1}/{len(window_end_candidates)} | "
                        f"TreeSHAP 完成 | method={method} | 返回 {len(imp_shap.feature_names)} 个特征"
                    )
                    for feat, imp in zip(imp_shap.feature_names, imp_shap.importances):
                        rows.append({
                            "model": model,
                            "H": H,
                            "method": method,
                            "window_end": pd.to_datetime(data.dates[window_end]),
                            "window_start": pd.to_datetime(data.dates[window_start]),
                            "n_obs": int(window),
                            "feature": str(feat),
                            "importance": float(imp),
                        })
                else:
                    log.debug(
                        f"[滚动重要性] {model}-H{H} 窗口 {wi+1}/{len(window_end_candidates)} | "
                        f"TreeSHAP 跳过（当前环境/模型不支持 pred_contrib）"
                    )

        log.info(f"[滚动重要性] 完成: model={model} | H={H} | 用时 {time.time()-t_model0:.1f}s")

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
