# -*- coding: utf-8 -*-
"""Step3：滚动窗口重要性稳定性（Rolling Feature Importance）。

实现思路（与 Step2 的无前视一致）：
- 对测试期按 window/step 生成多个窗口。
- 每个窗口都只用“窗口起点之前可得数据”训练模型（expanding 或 rolling），再在窗口内评估重要性。

重要性方法：
- permutation importance：模型无关，使用 RMSE 增量。
- TreeSHAP（无 shap 依赖）：仅对 XGBoost/LightGBM，使用 pred_contrib(s) 计算 mean(|contrib|)。

输出：
- rolling_importance_long.csv：长表，每行 = (model,H,method,window_end,feature,importance)
- rolling_importance_mean.csv：对 window 平均后的重要性排行

依赖：
- 需要 Step2 的 build_model / load_dataset 等工具。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Step2 工具：你们项目中已存在 utils/step2/...
from utils.step2.data_loader import load_dataset
from utils.step2.model_registry import build_model
from utils.step2.split_utils import train_test_split_time
from utils.step2.explain import permutation_importance, treeshap_importance_xgb, treeshap_importance_lgbm


@dataclass(frozen=True)
class RollingWindow:
    """滚动窗口定义（以索引表示）。"""

    start: int
    end: int


def _make_windows(test_idx: np.ndarray, window: int, step: int, max_windows: Optional[int] = None) -> List[RollingWindow]:
    """在测试集索引上生成滚动窗口。"""
    test_idx = np.asarray(test_idx, dtype=int)
    if len(test_idx) < 10:
        return []

    w = int(window)
    s = int(step)
    if w <= 0 or s <= 0:
        raise ValueError("window/step 必须为正")

    wins: List[RollingWindow] = []
    # 在 test_idx 的“位置序列”上滑动
    pos = 0
    while pos + w <= len(test_idx):
        idx_start = int(test_idx[pos])
        idx_end = int(test_idx[pos + w - 1])
        wins.append(RollingWindow(start=idx_start, end=idx_end))
        pos += s
        if max_windows is not None and len(wins) >= int(max_windows):
            break
    return wins


def _safe_json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_best_params(best_params_csv: str) -> pd.DataFrame:
    df = pd.read_csv(best_params_csv)
    need = ["model", "H", "params_json"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"best_params.csv 缺列：{miss} | file={best_params_csv}")
    return df


def compute_rolling_importance(
    *,
    step1_dataset_dir: str,
    step2_output_dir: str,
    out_tables_dir: str,
    run_cfg_step2_snapshot_path: str,
    selection: List[Tuple[str, int]],
    rolling_cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """计算滚动窗口重要性并写表。

    参数：
    - selection: [(model,H), ...]

    返回：
    - (long_csv_path, mean_csv_path)
    """

    log = logging.getLogger("step3.rolling_importance")

    # Step2 配置快照：用于数据划分口径一致（train/test 边界）
    step2_run = _safe_json_load(run_cfg_step2_snapshot_path)

    best_params_csv = str(Path(step2_output_dir).expanduser().resolve() / "tables" / "best_params.csv")
    best_df = _read_best_params(best_params_csv)

    window = int(rolling_cfg.get("window", 252))
    step = int(rolling_cfg.get("step", 21))
    sample_n = rolling_cfg.get("sample_n", 500)
    n_repeats = int(rolling_cfg.get("n_repeats", 3))
    random_state = int(rolling_cfg.get("random_state", 42))
    max_windows = rolling_cfg.get("max_windows", None)

    only_models = rolling_cfg.get("only_models", None)
    if only_models is not None:
        only_models = set(str(m) for m in only_models)

    enable_tree = bool(rolling_cfg.get("enable_treeshap_for_tree_models", True))

    rows: List[Dict[str, Any]] = []

    for model_name, H in selection:
        if only_models is not None and model_name not in only_models:
            log.info(f"跳过滚动重要性（only_models 过滤）: {model_name} H={H}")
            continue

        sub = best_df[(best_df["model"] == model_name) & (best_df["H"] == H)]
        if sub.empty:
            log.warning(f"best_params.csv 未找到对应行：model={model_name} H={H} -> 跳过")
            continue

        # 取第一行（理论上同 model/H 只有一行）
        params = json.loads(str(sub.iloc[0]["params_json"]))

        # 加载 Step1 数据
        data = load_dataset(
            dataset_dir=step1_dataset_dir,
            filename_template=step2_run.get("dataset_file_template", "Data_D_{H}.xlsx"),
            H=int(H),
            date_col=step2_run.get("date_col", "date"),
            y_prefix=step2_run.get("y_prefix", "USD"),
            feature_cols=step2_run.get("feature_cols", "ALL"),
            drop_cols=step2_run.get("drop_cols", []),
        )

        # 与 Step2 相同的 train/test 划分
        train_val_idx, test_idx = train_test_split_time(
            data.dates,
            mode=step2_run.get("test_mode", "last_ratio"),
            test_ratio=float(step2_run.get("test_ratio", 0.2)),
            test_years=int(step2_run.get("test_years", 3)),
            test_start=step2_run.get("test_start", None),
            test_end=step2_run.get("test_end", None),
        )

        wins = _make_windows(test_idx=test_idx, window=window, step=step, max_windows=max_windows)
        if not wins:
            log.warning(f"测试集太短，无法生成滚动窗口：model={model_name} H={H}")
            continue

        log.info(f"滚动重要性：model={model_name} H={H} windows={len(wins)}")

        # 逐窗口训练 + 解释
        for w_i, w in enumerate(wins):
            # 训练集：窗口起点之前的所有样本（严格无前视）
            train_end = int(w.start)
            train_idx = np.arange(0, train_end, dtype=int)
            if len(train_idx) < 30:
                continue

            X_tr = data.X[train_idx]
            y_tr = data.y[train_idx]

            # 解释窗口：窗口内样本
            test_mask = (np.arange(len(data.y)) >= w.start) & (np.arange(len(data.y)) <= w.end)
            idx_te = np.where(test_mask)[0].astype(int)
            if len(idx_te) < 10:
                continue

            X_te = data.X[idx_te]
            y_te = data.y[idx_te]

            # 构建与训练
            fm = build_model(
                model_name=model_name,
                params=params,
                random_seed=int(step2_run.get("random_seed", 42)),
                standardize_linear=bool(step2_run.get("standardize_linear", True)),
                standardize_mlp=bool(step2_run.get("standardize_mlp", True)),
                standardize_sequence=bool(step2_run.get("standardize_sequence", True)),
                gpu=step2_run.get("gpu", {}),
            )
            fm.fit(X_tr, y_tr)

            # 1) permutation
            pi = permutation_importance(
                model=fm,
                X=X_te,
                y=y_te,
                feature_names=data.feature_names,
                sample_n=sample_n,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            window_end_date = pd.to_datetime(data.dates[w.end])
            for feat, imp in zip(pi.feature_names, pi.importances):
                rows.append({
                    "model": model_name,
                    "H": int(H),
                    "method": "permutation",
                    "window_index": int(w_i),
                    "window_start": str(pd.to_datetime(data.dates[w.start]).date()),
                    "window_end": str(window_end_date.date()),
                    "feature": feat,
                    "importance": float(imp),
                })

            # 2) TreeSHAP（树模型）
            if enable_tree and model_name in ("XGBoost", "LightGBM"):
                if model_name == "XGBoost":
                    tr = treeshap_importance_xgb(fm.model, X_te, data.feature_names)
                    method = "treeshap"
                else:
                    tr = treeshap_importance_lgbm(fm.model, X_te, data.feature_names)
                    method = "treeshap"

                if tr is not None:
                    for feat, imp in zip(tr.feature_names, tr.importances):
                        rows.append({
                            "model": model_name,
                            "H": int(H),
                            "method": method,
                            "window_index": int(w_i),
                            "window_start": str(pd.to_datetime(data.dates[w.start]).date()),
                            "window_end": str(window_end_date.date()),
                            "feature": feat,
                            "importance": float(imp),
                        })

    if not rows:
        # 仍然写出空表，避免后续绘图模块找不到文件
        df_long = pd.DataFrame(columns=[
            "model", "H", "method", "window_index", "window_start", "window_end", "feature", "importance"
        ])
    else:
        df_long = pd.DataFrame(rows)

    out_tables = Path(out_tables_dir).expanduser().resolve()
    out_tables.mkdir(parents=True, exist_ok=True)

    long_path = str(out_tables / "rolling_importance_long.csv")
    df_long.to_csv(long_path, index=False, encoding="utf-8")

    # 均值表：对 window 平均后排序
    if len(df_long) == 0:
        df_mean = pd.DataFrame(columns=["model", "H", "method", "feature", "importance_mean"])
    else:
        df_mean = (
            df_long
            .groupby(["model", "H", "method", "feature"], as_index=False)["importance"]
            .mean()
            .rename(columns={"importance": "importance_mean"})
            .sort_values(["model", "H", "method", "importance_mean"], ascending=[True, True, True, False])
        )
    mean_path = str(out_tables / "rolling_importance_mean.csv")
    df_mean.to_csv(mean_path, index=False, encoding="utf-8")

    log.info(f"滚动重要性输出：{long_path}")
    log.info(f"滚动重要性均值表输出：{mean_path}")

    return long_path, mean_path

# =========================================================
# Step3 pipeline 入口：封装 compute_rolling_importance
# =========================================================


def run_rolling_importance(
    step1_dataset_dir: str,
    step2_output_dir: str,
    cfg: Dict[str, Any],
    dirs: Dict[str, str],
    filtered_preds: "List[Any]",
) -> Dict[str, Any]:
    """在 Step3 pipeline 中运行滚动重要性。

    参数：
    - filtered_preds：预测文件列表（含 model,H），来自 io_contract.filter_predictions 的输出
    """

    log = logging.getLogger("step3.rolling_importance")

    rolling_cfg = (cfg.get("rolling_importance", {}) or {})
    step2_snapshot_path = str(Path(step2_output_dir).expanduser().resolve() / "meta" / "config_snapshot.json")
    if not Path(step2_snapshot_path).exists():
        # 兼容可能的文件名差异
        alt = str(Path(step2_output_dir).expanduser().resolve() / "meta" / "config_snapshot_step2.json")
        if Path(alt).exists():
            step2_snapshot_path = alt

    selection = [(str(x.model), int(x.H)) for x in filtered_preds]
    if len(selection) == 0:
        log.warning("滚动重要性：selection 为空，跳过")
        return {"tables": []}

    long_csv, mean_csv = compute_rolling_importance(
        step1_dataset_dir=step1_dataset_dir,
        step2_output_dir=step2_output_dir,
        out_tables_dir=dirs["tables"],
        run_cfg_step2_snapshot_path=step2_snapshot_path,
        selection=selection,
        rolling_cfg=rolling_cfg,
    )

    return {"tables": [long_csv, mean_csv]}
