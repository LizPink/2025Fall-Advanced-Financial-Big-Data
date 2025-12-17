# -*- coding: utf-8 -*-
"""Step2: 模型调参 + walk-forward 预测 + 评估 + 解释 + 可视化

运行方式（推荐）：
    python run_step2.py

输出：
    output/step2/
        tables/  metrics.csv, best_params.csv
        datasets/ predictions_{model}_H{H}.csv
        figures/  ...
        meta/     config_snapshot.json, tuning_history_*.json
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.step2.data_loader import load_dataset
from utils.step2.split_utils import TimeSeriesCVConfig, iter_time_series_splits, train_test_split_time
from utils.step2.tuning import grid_search_cv
from utils.step2.backtest import walk_forward_predict
from utils.step2.metrics import compute_metrics
from utils.step2.model_registry import build_model
from utils.step2.explain import permutation_importance, treeshap_importance_xgb, treeshap_importance_lgbm
from utils.step2.plotting import (
    plot_feature_importance_bar,
    plot_pred_vs_true,
    plot_residual_hist,
    plot_scatter,
)
from utils.step2.reporter import ensure_dirs, save_json, append_table_csv, save_predictions


def _setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _resolve(base_file: str, rel: str) -> str:
    """Resolve relative paths with respect to the caller file location."""
    return str(Path(base_file).resolve().parent.joinpath(rel).resolve())

def resolve_refit_every(run_cfg: Dict[str, Any], model_name: str, H: int) -> int:
    """
    解析 refit_every，支持：
    - int：直接使用
    - dict：by_model_h / by_model / by_h / by default(1)（优先级由高到低）
    """
    cfg = run_cfg.get("refit_every", 1)

    # 1) 兼容旧写法：refit_every = 1
    if isinstance(cfg, (int, float)):
        v = int(cfg)
        return max(1, v)

    # 2) 新写法：refit_every = {...}
    if isinstance(cfg, dict):
        default = cfg.get("default", 1)
        by_model_h = cfg.get("by_model_h", {}) or {}
        by_model = cfg.get("by_model", {}) or {}
        by_h = cfg.get("by_h", {}) or {}

        v = None

        # model + H（最高优先级）
        mh = by_model_h.get(model_name, None)
        if isinstance(mh, dict):
            if H in mh:
                v = mh[H]
            elif str(H) in mh:
                v = mh[str(H)]

        # model
        if v is None and model_name in by_model:
            v = by_model[model_name]

        # H
        if v is None:
            if H in by_h:
                v = by_h[H]
            elif str(H) in by_h:
                v = by_h[str(H)]

        # default
        if v is None:
            v = default

        try:
            v = int(v)
        except Exception:
            v = 1
        return max(1, v)

    # 3) 其他异常类型，兜底
    return 1

def main() -> None:
    cfg = importlib.import_module("configs.config_step2")
    RUN: Dict[str, Any] = getattr(cfg, "RUN")

    _setup_logger(RUN.get("log_level", "INFO"))
    log = logging.getLogger("step2")
    # Environment hint for torch/CUDA
    try:  # pragma: no cover
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            n = torch.cuda.device_count()
            name0 = torch.cuda.get_device_name(0) if n > 0 else "(unknown)"
            log.info(f"PyTorch CUDA available: True | device_count={n} | device0={name0}")
        else:
            log.info("PyTorch CUDA available: False")
    except Exception:
        log.info("PyTorch not installed; LSTM/Transformer will be unavailable")

    here = __file__
    dataset_dir = _resolve(here, RUN["step1_dataset_dir"])
    out_dir = _resolve(here, RUN["output_dir"])

    dirs = ensure_dirs(out_dir)

    # snapshot config
    save_json(str(dirs["meta"] / "config_snapshot.json"), RUN)

    H_list: List[int] = list(RUN["H_list"])
    model_switch: Dict[str, bool] = dict(RUN.get("models", {}))
    param_grids: Dict[str, Any] = dict(RUN.get("param_grids", {}))

    # global tables will be appended incrementally
    metrics_csv = str(dirs["tables"] / "metrics.csv")
    params_csv = str(dirs["tables"] / "best_params.csv")

    for H in H_list:
        log.info(f"=== Horizon H={H} ===")

        data = load_dataset(
            dataset_dir=dataset_dir,
            filename_template=RUN["dataset_file_template"],
            H=H,
            date_col=RUN["date_col"],
            y_prefix=RUN["y_prefix"],
            feature_cols=RUN.get("feature_cols", "ALL"),
            drop_cols=RUN.get("drop_cols", []),
        )

        # Split train_val/test
        train_val_idx, test_idx = train_test_split_time(
            dates=data.dates,
            mode=RUN.get("test_mode", "last_ratio"),
            test_ratio=float(RUN.get("test_ratio", 0.2)),
            test_years=int(RUN.get("test_years", 3)),
            test_start=RUN.get("test_start"),
            test_end=RUN.get("test_end"),
        )
        log.info(f"Split: train_val={len(train_val_idx)} test={len(test_idx)}")

        # CV splits inside train_val
        cv_cfg = TimeSeriesCVConfig(
            mode=RUN.get("cv_mode", "expanding"),
            initial_train_size=int(RUN.get("cv_initial_train_size", 1500)),
            val_size=int(RUN.get("cv_val_size", 252)),
            step_size=int(RUN.get("cv_step_size", 252)),
            train_window=int(RUN.get("cv_train_window", 1500)),
        )
        splits = list(iter_time_series_splits(len(train_val_idx), cv_cfg))
        if len(splits) == 0:
            raise ValueError("No CV splits generated; adjust cv_initial_train_size/cv_val_size.")

        # Convert splits from train_val-local to global indices
        splits_global: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr_local, va_local in splits:
            splits_global.append((train_val_idx[tr_local], train_val_idx[va_local]))

        for model_name, enabled in model_switch.items():
            if not enabled:
                continue
            if model_name == "RandomWalk":
                # 1) 直接预测：y_hat = 0（对应 RW-no-drift 对收益标签的预测）
                y_pred = np.zeros(len(test_idx), dtype=float)
                y_true = data.y[test_idx].astype(float)

                metrics = compute_metrics(y_true, y_pred)
                row = {"model": model_name, "H": H, **metrics, "n_test_eval": int(len(test_idx))}
                append_table_csv(metrics_csv, row)
                log.info(f"Metrics: {row}")

                # 2) 保存预测
                pred_path = str(dirs["datasets"] / f"predictions_{model_name}_H{H}.csv")
                save_predictions(
                    pred_path,
                    dates=data.dates[test_idx],
                    y_true=y_true,
                    y_pred=y_pred,
                    extra={"H": H, "model": model_name},
                )

                # 3) 可视化
                if bool(RUN.get("make_figures", True)):
                    fig_prefix = dirs["figures"] / f"{model_name}_H{H}"
                    plot_pred_vs_true(
                        out_path=str(fig_prefix) + "_pred_vs_true.png",
                        dates=data.dates[test_idx],
                        y_true=y_true,
                        y_pred=y_pred,
                        title=f"{model_name} | H={H} | y_pred vs y_true",
                    )
                    resid = y_true - y_pred
                    plot_residual_hist(
                        out_path=str(fig_prefix) + "_residual_hist.png",
                        residuals=resid,
                        title=f"{model_name} | H={H} | residuals",
                    )
                    plot_scatter(
                        out_path=str(fig_prefix) + "_scatter.png",
                        y_true=y_true,
                        y_pred=y_pred,
                        title=f"{model_name} | H={H} | scatter",
                    )
                # RandomWalk 没有特征重要性，直接进入下一个模型
                continue

            log.info(f"--- Model {model_name} (H={H}) ---")

            # Build model function wrapper with standardize flags
            def _build(name: str, params: Dict[str, Any], random_seed: int) -> Any:
                return build_model(
                    name,
                    params,
                    random_seed=random_seed,
                    standardize_linear=bool(RUN.get("standardize_linear", True)),
                    standardize_mlp=bool(RUN.get("standardize_mlp", True)),
                    standardize_sequence=bool(RUN.get("standardize_sequence", True)),
                    gpu=RUN.get("gpu", {}),
                )

            # 1) Hyperparameter tuning on train_val only
            grid = param_grids.get(model_name, {})
            tune = grid_search_cv(
                model_name=model_name,
                build_model_fn=_build,
                X=data.X,
                y=data.y,
                splits=splits_global,
                param_grid=grid,
                tune_metric=RUN.get("tune_metric", "rmse"),
                random_seed=int(RUN.get("random_seed", 42)),
                embargo_size=int(RUN.get("embargo_size", 0)),
                purge=bool(RUN.get("purge", False)),
                max_combinations=RUN.get("grid_max_combinations"),
                verbose=bool(RUN.get("show_progress", True)),
            )
            best_params = tune.best_params
            log.info(f"Best params: {best_params} | best_score={tune.best_score:.6g}")

            # Save tuning history
            save_json(
                str(dirs["meta"] / f"tuning_history_{model_name}_H{H}.json"),
                tune.history,
            )
            append_table_csv(params_csv, {
                "model": model_name,
                "H": H,
                "best_score": tune.best_score,
                "params_json": json.dumps(best_params, ensure_ascii=False),
            })

            # 2) Walk-forward OOS prediction on test
            refit_every = resolve_refit_every(RUN, model_name, H)
            log.info(
                f"Backtest settings | model={model_name} H={H} "
                f"mode={RUN.get('backtest_mode','expanding')} "
                f"train_window={int(RUN.get('backtest_train_window', 2000))} "
                f"refit_every={refit_every}"
            )
            bt = walk_forward_predict(
                model_name=model_name,
                build_model_fn=_build,
                X=data.X,
                y=data.y,
                train_val_idx=train_val_idx,
                test_idx=test_idx,
                best_params=best_params,
                mode=RUN.get("backtest_mode", "expanding"),
                train_window=int(RUN.get("backtest_train_window", 2000)),
                refit_every=refit_every,
                random_seed=int(RUN.get("random_seed", 42)),
                verbose=bool(RUN.get("show_progress", True)),
            )

            # remove possible nan predictions (sequence warmup)
            mask = np.isfinite(bt.y_pred)
            y_true_eval = bt.y_true[mask]
            y_pred_eval = bt.y_pred[mask]
            metrics = compute_metrics(y_true_eval, y_pred_eval)

            row = {"model": model_name, "H": H, **metrics, "n_test_eval": int(mask.sum())}
            append_table_csv(metrics_csv, row)
            log.info(f"Metrics: {row}")

            # Save predictions
            pred_path = str(dirs["datasets"] / f"predictions_{model_name}_H{H}.csv")
            save_predictions(
                pred_path,
                dates=data.dates[test_idx],
                y_true=bt.y_true,
                y_pred=bt.y_pred,
                extra={"H": H, "model": model_name},
            )

            # 3) Explain + Figures
            if bool(RUN.get("make_figures", True)):
                fig_prefix = dirs["figures"] / f"{model_name}_H{H}"
                plot_pred_vs_true(
                    out_path=str(fig_prefix) + "_pred_vs_true.png",
                    dates=data.dates[test_idx],
                    y_true=bt.y_true,
                    y_pred=bt.y_pred,
                    title=f"{model_name} | H={H} | y_pred vs y_true",
                )
                resid = bt.y_true - bt.y_pred
                plot_residual_hist(
                    out_path=str(fig_prefix) + "_residual_hist.png",
                    residuals=resid,
                    title=f"{model_name} | H={H} | residuals",
                )
                plot_scatter(
                    out_path=str(fig_prefix) + "_scatter.png",
                    y_true=bt.y_true,
                    y_pred=bt.y_pred,
                    title=f"{model_name} | H={H} | scatter",
                )

            # Fit once on full available data up to end of test for explanation (optional)
            explain_cfg = RUN.get("explain", {}) or {}
            if bool(explain_cfg.get("enable", True)):
                final_train_idx = np.concatenate([train_val_idx, test_idx]).astype(int)
                final_train_idx.sort()

                final_model = _build(model_name, best_params, random_seed=int(RUN.get("random_seed", 42)))
                final_model.fit(data.X[final_train_idx], data.y[final_train_idx])

                # Permutation importance on test (mask for seq warmup handled inside)
                imp = permutation_importance(
                    model=final_model,
                    X=data.X[test_idx],
                    y=data.y[test_idx],
                    feature_names=data.feature_names,
                    sample_n=explain_cfg.get("sample_n", 1000),
                    random_state=int(explain_cfg.get("random_state", 42)),
                )
                imp_row_path = str(dirs["tables"] / f"importance_permutation_{model_name}_H{H}.csv")
                pd.DataFrame({
                    "feature": imp.feature_names,
                    "importance": imp.importances,
                }).sort_values("importance", ascending=False).to_csv(imp_row_path, index=False, encoding="utf-8")

                if bool(RUN.get("make_figures", True)):
                    plot_feature_importance_bar(
                        out_path=str(dirs["figures"] / f"{model_name}_H{H}_perm_importance.png"),
                        feature_names=imp.feature_names,
                        importances=imp.importances,
                        top_k=int(RUN.get("top_k_features", 20)),
                        title=f"{model_name} | H={H} | Permutation importance (RMSE increase)",
                    )

                # TreeSHAP enhancement for XGBoost/LightGBM (no shap dependency)
                tree_imp = None
                if model_name == "XGBoost":
                    tree_imp = treeshap_importance_xgb(final_model.model, data.X[test_idx], data.feature_names)
                elif model_name == "LightGBM":
                    tree_imp = treeshap_importance_lgbm(final_model.model, data.X[test_idx], data.feature_names)

                if tree_imp is not None:
                    tree_path = str(dirs["tables"] / f"importance_treeshap_{model_name}_H{H}.csv")
                    pd.DataFrame({
                        "feature": tree_imp.feature_names,
                        "importance": tree_imp.importances,
                    }).sort_values("importance", ascending=False).to_csv(tree_path, index=False, encoding="utf-8")

                    if bool(RUN.get("make_figures", True)):
                        plot_feature_importance_bar(
                            out_path=str(dirs["figures"] / f"{model_name}_H{H}_treeshap_importance.png"),
                            feature_names=tree_imp.feature_names,
                            importances=tree_imp.importances,
                            top_k=int(RUN.get("top_k_features", 20)),
                            title=f"{model_name} | H={H} | TreeSHAP importance (mean |contrib|)",
                        )

    log.info(f"[OK] Step2 finished. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
