# -*- coding: utf-8 -*-
"""Step3：基于 VIX 的分时期（Regime）检验

输入
- econ_trades_long.csv：经济层模块产出的交易级别长表
- Step1 的 Data_D_{H}.xlsx（读取 date 与 VIX 列，用于贴 regime 标签）

两种视角
1) conditional：按 regime（low/mid/high）对交易样本分组计算绩效
   - 不会“打散”时间顺序；只是对同一条时间序列做子样本条件统计

2) episode：把 regime 合并为连续时段，再在每个 episode 内计算绩效
   - 方便在论文/答辩里叙事与可视化（例如“高 VIX 期间策略更好/更差”）

无前视
- train_only（默认）：用训练/验证期固定阈值，应用于测试期
- expanding：对每个日期用历史 expanding 分位数阈值（并 shift 1 天）

去抖
- hysteresis：双阈值状态机
- min_spell_days：把过短片段压回 mid

所有注释采用中文。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from utils.step2.data_loader import load_dataset
from utils.step2.split_utils import train_test_split_time

from utils.step3.vix_thresholds import (
    thresholds_from_config,
    compute_thresholds_train_only,
    compute_thresholds_expanding,
    assign_regime_hysteresis,
    assign_regime_hysteresis_timevarying,
    apply_min_spell_days,
    build_episodes,
)
from utils.step3.plot_utils import save_figure


def _ann_scale(annual_days: int, H: int) -> float:
    return math.sqrt(float(annual_days) / float(max(1, H)))


def _sharpe_ann(r: np.ndarray, ann_scale: float) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 3:
        return float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(mu / sd * ann_scale)


def _max_drawdown(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(np.min(dd))


def _compute_metrics_for_subset(r_net: np.ndarray, ann_scale: float) -> Dict[str, Any]:
    r_net = np.asarray(r_net, dtype=float)
    return {
        "n_obs": int(len(r_net)),
        "mean_log_ret": float(np.mean(r_net)) if len(r_net) > 0 else float("nan"),
        "std_log_ret": float(np.std(r_net, ddof=1)) if len(r_net) > 1 else float("nan"),
        "sharpe_ann": _sharpe_ann(r_net, ann_scale=ann_scale),
        "max_drawdown": _max_drawdown(r_net) if len(r_net) > 0 else float("nan"),
        "hit_rate": float(np.mean(r_net > 0.0)) if len(r_net) > 0 else float("nan"),
    }


def _prepare_vix_series(
    step1_dataset_dir: str,
    dataset_file_template: str,
    H: int,
    date_col: str,
    y_prefix: str,
    vix_col: str,
) -> pd.DataFrame:
    """从 Step1 的 Data_D_{H}.xlsx 读取 date 与 VIX 列。

    说明
    - 这里复用 utils.step2.data_loader.load_dataset 以保证读取/排序口径一致
    - load_dataset 会返回 df 全量，我们只取 date 与 vix_col
    """
    data = load_dataset(
        dataset_dir=step1_dataset_dir,
        filename_template=dataset_file_template,
        H=H,
        date_col=date_col,
        y_prefix=y_prefix,
        feature_cols="ALL",
        drop_cols=[],
    )
    df = data.df.copy()
    if vix_col not in df.columns:
        raise KeyError(f"Step1 数据缺少 VIX 列: {vix_col} | H={H}")
    out = df[[date_col, vix_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)
    return out


def run_vix_regime_eval(
    run_cfg: Dict[str, Any],
    combos: List[Dict[str, Any]],
    trades_long_path: str,
    step1_dataset_dir: str,
    dataset_file_template: str,
    date_col: str,
    y_prefix: str,
    step2_snapshot: Optional[Dict[str, Any]],
    out_dirs: Dict[str, str],
) -> Dict[str, Any]:
    cfg = dict(run_cfg.get("vix_regime", {}) or {})
    if not bool(cfg.get("enable", True)):
        return {}

    vix_col = str(cfg.get("vix_col", "VIX"))
    mode = str(cfg.get("mode", "quantile"))
    thr_source = str(cfg.get("threshold_source", "train_only"))
    min_spell_days = int(cfg.get("min_spell_days", 0))

    quantiles_cfg = dict(cfg.get("quantiles", {}) or {})
    fixed_thr_cfg = dict(cfg.get("fixed_thresholds", {}) or {})

    annual_days = int(run_cfg.get("economic", {}).get("annual_days", 252))

    trades = pd.read_csv(trades_long_path, encoding="utf-8", low_memory=False)
    trades["date"] = pd.to_datetime(trades["date"])

    tables_dir = Path(out_dirs["tables"])
    datasets_dir = Path(out_dirs["datasets"])
    figures_dir = Path(out_dirs["figures"])

    out_tables = []
    out_datasets = []
    out_figures = []

    # 为避免重复读取 Step1/VIX，按 H 缓存
    vix_cache: Dict[int, pd.DataFrame] = {}

    # 用于 train_only 阈值计算的 train_mask，需要 Step2 的 test split 口径
    def _get_train_mask_for_H(H: int, vix_df: pd.DataFrame) -> pd.Series:
        # 默认采用 Step2 口径；若没有 snapshot，则退化为 last_ratio(0.2)
        snap = step2_snapshot or {}
        test_mode = str(snap.get("test_mode", "last_ratio"))
        test_ratio = float(snap.get("test_ratio", 0.2))
        test_years = int(snap.get("test_years", 3))
        test_start = snap.get("test_start", None)
        test_end = snap.get("test_end", None)

        dt = vix_df[date_col].to_numpy()
        train_idx, test_idx = train_test_split_time(
            dates=dt,
            mode=test_mode,
            test_ratio=test_ratio,
            test_years=test_years,
            test_start=test_start,
            test_end=test_end,
        )
        mask = pd.Series(False, index=vix_df.index)
        mask.iloc[train_idx] = True
        return mask

    # 输出：交易样本按 regime 分组（conditional）
    cond_rows = []

    # 输出：episode 级别
    episode_rows = []

    # 输出：各 H 的 regime 序列与 episode 表
    regime_series_paths = []
    episode_table_paths = []

    H_list = sorted({int(c["H"]) for c in combos})

    for H in H_list:
        if H not in vix_cache:
            vix_cache[H] = _prepare_vix_series(
                step1_dataset_dir=step1_dataset_dir,
                dataset_file_template=dataset_file_template,
                H=H,
                date_col=date_col,
                y_prefix=y_prefix,
                vix_col=vix_col,
            )

        vix_df = vix_cache[H]
        vix_df = vix_df.rename(columns={date_col: "date"})
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_s = vix_df.set_index("date")[vix_col].astype(float)

        # 1) 阈值
        if mode == "fixed":
            fixed_thr = thresholds_from_config(mode, fixed_thr_cfg)
            regime = assign_regime_hysteresis(vix_s, fixed_thr)
        elif mode == "quantile":
            if thr_source == "train_only":
                train_mask = _get_train_mask_for_H(H, vix_cache[H])
                thr = compute_thresholds_train_only(
                    vix=vix_cache[H][vix_col],
                    train_mask=train_mask,
                    quantiles_cfg=quantiles_cfg,
                )
                regime = assign_regime_hysteresis(vix_s, thr)
            elif thr_source == "expanding":
                # fallback：使用 train_only 阈值避免早期 NaN
                train_mask = _get_train_mask_for_H(H, vix_cache[H])
                fallback_thr = compute_thresholds_train_only(
                    vix=vix_cache[H][vix_col],
                    train_mask=train_mask,
                    quantiles_cfg=quantiles_cfg,
                )
                thr_df = compute_thresholds_expanding(vix_s, quantiles_cfg=quantiles_cfg)
                regime = assign_regime_hysteresis_timevarying(vix_s, thr_df, fallback_thr=fallback_thr)
            else:
                raise ValueError(f"未知 threshold_source: {thr_source}")
        else:
            raise ValueError(f"未知 mode: {mode}")

        # 2) 去抖（最短持续期）
        regime = apply_min_spell_days(regime, min_spell_days=min_spell_days)

        # 保存 regime 序列
        regime_df = pd.DataFrame({"date": regime.index, "VIX": vix_s.reindex(regime.index).to_numpy(), "regime": regime.to_numpy()})
        regime_path = str(datasets_dir / f"vix_regime_H{H}.csv")
        regime_df.to_csv(regime_path, index=False, encoding="utf-8")
        regime_series_paths.append(regime_path)

        # episode 表
        episodes = build_episodes(regime)
        episodes_path = str(datasets_dir / f"vix_episodes_H{H}.csv")
        episodes.to_csv(episodes_path, index=False, encoding="utf-8")
        episode_table_paths.append(episodes_path)

        # 3) conditional 计算：只在 H 对应的交易样本上做
        sub_trades = trades[trades["H"] == H].copy()
        sub_trades = sub_trades.merge(regime_df[["date", "regime"]], on="date", how="left")

        ann_scale = _ann_scale(annual_days=annual_days, H=H)

        for keys, g in sub_trades.groupby(["model", "strategy", "tau", "phase"], dropna=False):
            model, strategy, tau, phase = keys
            for reg, gg in g.groupby("regime", dropna=False):
                m = _compute_metrics_for_subset(gg["r_net"].to_numpy(dtype=float), ann_scale=ann_scale)
                cond_rows.append({
                    "model": model,
                    "H": H,
                    "strategy": strategy,
                    "tau": float(tau),
                    "phase": int(phase),
                    "regime": str(reg),
                    **m,
                })

        # 4) episode 计算：以 episode 为单位，取该时间段内的交易样本
        if "episode" in [str(x) for x in cfg.get("methods", [])]:
            for _, ep in episodes.iterrows():
                start = pd.to_datetime(ep["start"])
                end = pd.to_datetime(ep["end"])
                reg = str(ep["regime"])

                in_ep = sub_trades[(sub_trades["date"] >= start) & (sub_trades["date"] <= end)]
                if len(in_ep) == 0:
                    continue

                for keys, g in in_ep.groupby(["model", "strategy", "tau", "phase"], dropna=False):
                    model, strategy, tau, phase = keys
                    m = _compute_metrics_for_subset(g["r_net"].to_numpy(dtype=float), ann_scale=ann_scale)
                    episode_rows.append({
                        "model": model,
                        "H": H,
                        "strategy": strategy,
                        "tau": float(tau),
                        "phase": int(phase),
                        "episode_start": start,
                        "episode_end": end,
                        "episode_len": int(ep["length"]),
                        "regime": reg,
                        **m,
                    })

    # 写出 conditional 表
    if "conditional" in [str(x) for x in cfg.get("methods", [])]:
        cond_df = pd.DataFrame(cond_rows)
        cond_path = str(tables_dir / "econ_by_vix_regime_conditional.csv")
        cond_df.to_csv(cond_path, index=False, encoding="utf-8")
        out_tables.append(cond_path)

    # 写出 episode 表
    if len(episode_rows) > 0:
        ep_df = pd.DataFrame(episode_rows)
        ep_path = str(tables_dir / "econ_by_vix_episode.csv")
        ep_df.to_csv(ep_path, index=False, encoding="utf-8")
        out_tables.append(ep_path)

    out_datasets.extend(regime_series_paths)
    out_datasets.extend(episode_table_paths)

    # 资金曲线 + 阴影（高 VIX）
    if bool(cfg.get("save_equity_shading_fig", True)):
        log = logging.getLogger("step3")
        try:
            # 与经济层资金曲线保持同一口径：相同的 phase、相同的 tau_for_plot、相同的策略线条
            econ_cfg = dict(run_cfg.get("economic", {}) or {})
            plot_phase = int(econ_cfg.get("plot_phase", 0))
            plot_strategies = list(econ_cfg.get("plot_strategies", ["model"]))
            tau_list = list(econ_cfg.get("tau_list", [0.0]))
            tau_for_plot = [float(tau_list[0])] if len(tau_list) > 0 else [0.0]

            baselines = list(econ_cfg.get("baselines", []))
            baseline_strategies = [str(s) for s in baselines if str(s) not in ("model", "")]

            # 默认：对所有 (model,H) 生成 vixshade 图；若 plot_top_n>0，则仅对 Sharpe 较高的前 N 个 (model,H) 生成
            plot_top_n = cfg.get("plot_top_n", 0)
            try:
                plot_top_n = int(plot_top_n)
            except Exception:
                plot_top_n = 0

            combos_unique = sorted({(str(c["model"]), int(c["H"])) for c in combos}, key=lambda x: (x[0], x[1]))

            if plot_top_n > 0:
                phase_summary_path = Path(out_dirs["tables"]) / "econ_metrics_phase_summary.csv"
                if phase_summary_path.exists():
                    summ = pd.read_csv(phase_summary_path)
                    # 仅用策略为 model 的行来排序（避免 baseline 影响）
                    summ = summ[(summ["strategy"] == "model") & (summ["model"] != "BASELINE")]
                    # 尽量使用 sharpe_ann_mean
                    sort_col = "sharpe_ann_mean" if "sharpe_ann_mean" in summ.columns else None
                    if sort_col is not None:
                        summ = summ.sort_values(sort_col, ascending=False)
                    top_pairs = []
                    for _, r in summ.iterrows():
                        pair = (str(r["model"]), int(r["H"]))
                        if pair not in top_pairs and pair in combos_unique:
                            top_pairs.append(pair)
                        if len(top_pairs) >= plot_top_n:
                            break
                    if len(top_pairs) > 0:
                        combos_unique = top_pairs

            trades_long = trades.copy()
            trades_long["date"] = pd.to_datetime(trades_long["date"])

            total_imgs = 0
            for (model, H) in combos_unique:
                for tau in tau_for_plot:
                    fig = plt.figure(figsize=(10, 4))
                    ax = fig.add_subplot(111)

                    # 逐条策略线画累计净值
                    date_min, date_max = None, None
                    any_line = False
                    for strategy in plot_strategies:
                        if str(strategy) in baseline_strategies:
                            sub = trades_long[
                                (trades_long["model"] == "BASELINE")
                                & (trades_long["H"] == H)
                                & (trades_long["phase"] == int(plot_phase))
                                & (trades_long["strategy"] == str(strategy))
                            ].sort_values("date")
                        else:
                            sub = trades_long[
                                (trades_long["model"] == model)
                                & (trades_long["H"] == H)
                                & (trades_long["tau"] == float(tau))
                                & (trades_long["phase"] == int(plot_phase))
                                & (trades_long["strategy"] == str(strategy))
                            ].sort_values("date")

                        if len(sub) == 0:
                            continue

                        any_line = True
                        eq = np.exp(np.cumsum(sub["r_net"].to_numpy(dtype=float)))
                        ax.plot(sub["date"].to_numpy(), eq, label=str(strategy))

                        d0 = pd.to_datetime(sub["date"].iloc[0])
                        d1 = pd.to_datetime(sub["date"].iloc[-1])
                        date_min = d0 if date_min is None else min(date_min, d0)
                        date_max = d1 if date_max is None else max(date_max, d1)

                    if not any_line or date_min is None or date_max is None:
                        plt.close(fig)
                        continue

                    # 阴影：仅对测试期（资金曲线区间）内的 high episode 进行 shading，避免把 x 轴“拉回训练期”
                    ep_path = Path(out_dirs["datasets"]) / f"vix_episodes_H{H}.csv"
                    if ep_path.exists():
                        eps = pd.read_csv(ep_path)
                        for _, ep in eps.iterrows():
                            if str(ep.get("regime")) != "high":
                                continue
                            s = pd.to_datetime(ep["start"])
                            e = pd.to_datetime(ep["end"])

                            # 裁剪到 [date_min, date_max]，确保与 equity 图片区间对齐
                            if e < date_min or s > date_max:
                                continue
                            s2 = max(s, date_min)
                            e2 = min(e, date_max)
                            if s2 <= e2:
                                ax.axvspan(s2, e2, alpha=0.15)

                    ax.set_title(f"资金曲线（高 VIX 阴影）| model={model} | H={H} | tau={tau} | phase={plot_phase}")
                    ax.set_xlabel("日期")
                    ax.set_ylabel("累计净值")
                    ax.set_xlim(date_min, date_max)
                    ax.legend()

                    out_path = str(figures_dir / f"equity_vixshade_{model}_H{H}_tau{tau}_phase{plot_phase}.png")
                    save_figure(fig, out_path, dpi=220)
                    out_figures.append(out_path)
                    total_imgs += 1

            log.info(f"VIX 阴影资金曲线已生成: {total_imgs} 张")
        except Exception as e:
            # 可视化失败不应影响核心表格输出
            log.warning(f"VIX 阴影资金曲线生成失败（已跳过）：{e}")

    return {
        "tables": out_tables,
        "datasets": out_datasets,
        "figures": out_figures,
    }
