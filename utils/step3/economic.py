# -*- coding: utf-8 -*-
"""Step3：经济层检验（Economic Layer）。

核心输入：
- Step2 的 predictions_{model}_H{H}.csv（date, y_true, y_pred）

核心输出：
- econ_metrics.csv：不同模型/不同 tau/baseline 的经济指标
- econ_metrics_by_phase.csv：在非重叠口径下，对 phase=0..H-1 分别计算
- econ_metrics_phase_summary.csv：对 phase 的均值/中位数/最小/最大汇总
- econ_returns_*.csv：策略收益时间序列（便于画图与复核）

注意：
- 本模块默认使用“非重叠持有期”：当 H>1 时，每 H 天再平衡一次，并对每个相位做 sweep。
- y_true 默认是 H 日累计对数收益（log holding-period return）。
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconMetrics:
    mean_log_ret: float
    std_log_ret: float
    sharpe_ann: float
    hit_rate: float
    turnover: float
    max_drawdown: float
    n_obs: int


def _max_drawdown_from_log_returns(r: np.ndarray) -> float:
    """根据对数收益序列计算最大回撤（以简单收益口径返回）。"""
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return float("nan")
    cs = np.cumsum(r)
    equity = np.exp(cs)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def _annual_scale(annual_days: int, H: int, non_overlapping: bool) -> float:
    """年化缩放因子。

    - 非重叠口径：每个观察值对应一笔 H 日持有期收益，因此年化用 sqrt(252/H)。
    - 重叠口径：严格年化需要更复杂的相关性修正；为避免误导，本项目默认不启用。
    """
    if non_overlapping and H > 0:
        return math.sqrt(float(annual_days) / float(H))
    return math.sqrt(float(annual_days))


def compute_econ_metrics(r_net: np.ndarray, pos: np.ndarray, annual_days: int, H: int, non_overlapping: bool) -> EconMetrics:
    """从净对数收益与仓位序列计算常用经济指标。"""
    r = np.asarray(r_net, dtype=float)
    p = np.asarray(pos, dtype=float)

    mask = np.isfinite(r)
    r = r[mask]
    p = p[mask]

    n = int(len(r))
    if n == 0:
        return EconMetrics(*(float("nan"),) * 6, n_obs=0)

    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    scale = _annual_scale(annual_days=annual_days, H=H, non_overlapping=non_overlapping)
    sharpe = float(mean / std * scale) if std > 0 else float("nan")

    hit = float(np.mean(r > 0))
    dp = np.abs(np.diff(p))
    turnover = float(np.mean(dp)) if len(dp) > 0 else 0.0
    mdd = _max_drawdown_from_log_returns(r)

    return EconMetrics(mean, std, sharpe, hit, turnover, mdd, n)


def build_position(y_pred: np.ndarray, tau: float, allow_short: bool) -> np.ndarray:
    """由预测值生成仓位：-1/0/1。

    规则：
    - |y_pred| <= tau：空仓 0
    - y_pred > tau：做多 1
    - y_pred < -tau：做空 -1（若 allow_short=False，则映射为 0）
    """
    yp = np.asarray(y_pred, dtype=float)
    pos = np.zeros_like(yp)
    pos[yp > float(tau)] = 1.0
    pos[yp < -float(tau)] = -1.0
    if not bool(allow_short):
        pos[pos < 0] = 0.0
    return pos


def apply_baseline(strategy: str, n: int) -> np.ndarray:
    """生成基准策略仓位序列。"""
    s = str(strategy).lower()
    if s == "cash":
        return np.zeros(int(n), dtype=float)
    if s == "always_long":
        return np.ones(int(n), dtype=float)
    if s == "always_short":
        return -np.ones(int(n), dtype=float)
    raise ValueError(f"未知 baseline: {strategy}")


def compute_net_returns(y_true: np.ndarray, pos: np.ndarray, tc_bps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 y_true、pos 计算 (gross, cost, net) 对数收益序列。

    成本模型：单边成本 tc_bps（bps），按仓位变化计费：
        cost_t = tc_bps/10000 * |pos_t - pos_{t-1}|
    解释：
    - 开仓/平仓：|Δpos|=1
    - 反手：|Δpos|=2（相当于先平再开）
    """
    yt = np.asarray(y_true, dtype=float)
    p = np.asarray(pos, dtype=float)

    gross = p * yt
    dpos = np.zeros_like(p)
    if len(p) > 1:
        dpos[1:] = np.abs(p[1:] - p[:-1])
    cost = (float(tc_bps) / 10000.0) * dpos
    net = gross - cost
    return gross, cost, net


def iter_phases(H: int, phase_sweep: bool) -> List[int]:
    """返回要计算的 phase 列表。"""
    if not phase_sweep or H <= 1:
        return [0]
    return list(range(int(H)))


def subset_non_overlapping(df: pd.DataFrame, H: int, phase: int) -> pd.DataFrame:
    """非重叠持有期子采样：每 H 个点取一个，从 phase 开始。"""
    if H <= 1:
        return df
    phase = int(phase)
    if phase < 0 or phase >= H:
        raise ValueError(f"phase 必须在 [0, H-1]，当前 phase={phase}, H={H}")
    return df.iloc[phase::H].copy()


def summarize_phase_table(df_phase: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """对 phase 表做摘要：mean/median/min/max。"""
    metrics_cols = [
        "mean_log_ret",
        "std_log_ret",
        "sharpe_ann",
        "hit_rate",
        "turnover",
        "max_drawdown",
        "n_obs",
    ]
    agg = {}
    for c in metrics_cols:
        if c == "n_obs":
            agg[c] = ["min", "max", "mean"]
        else:
            agg[c] = ["mean", "median", "min", "max"]

    out = df_phase.groupby(group_cols, as_index=False).agg(agg)

    # 扁平化列名
    out.columns = [
        (f"{a}_{b}" if b else a)
        for a, b in [(c[0], c[1] if len(c) > 1 else "") for c in out.columns.to_flat_index()]
    ]
    return out


def run_economic_eval_for_one(
    df_pred: pd.DataFrame,
    model: str,
    H: int,
    allow_short: bool,
    tau_list: List[float],
    baselines: List[str],
    use_non_overlapping: bool,
    phase_sweep: bool,
    tc_bps: float,
    annual_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """对单个 (model,H) 运行经济层评估。

    返回：
    - econ_total：不区分 phase 的总表（若 use_non_overlapping=True，则这里给出 phase_summary 的均值行）
    - econ_by_phase：逐 phase 指标
    - econ_phase_summary：对 phase 的汇总
    - return_series：key=(strategy,tau,phase) -> 含 date,pos,gross,cost,net 的 DataFrame（便于落盘）
    """
    df = df_pred.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # y_pred NaN 已在 io_contract 里过滤过；这里再次保险
    df = df[np.isfinite(df["y_pred"].to_numpy(float))].copy()

    phases = iter_phases(H=H, phase_sweep=phase_sweep if use_non_overlapping else False)

    rows_phase: List[Dict[str, object]] = []
    return_series: Dict[str, pd.DataFrame] = {}

    # 1) 模型策略
    for tau in tau_list:
        pos_all = build_position(df["y_pred"].to_numpy(float), tau=float(tau), allow_short=allow_short)
        df_model = df[["date", "y_true"]].copy()
        df_model["pos"] = pos_all

        for phase in phases:
            sub = subset_non_overlapping(df_model, H=H, phase=phase) if use_non_overlapping else df_model
            gross, cost, net = compute_net_returns(sub["y_true"].to_numpy(float), sub["pos"].to_numpy(float), tc_bps=tc_bps)
            m = compute_econ_metrics(net, sub["pos"].to_numpy(float), annual_days=annual_days, H=H, non_overlapping=use_non_overlapping)

            key = f"model|tau={tau}|phase={phase}"
            rs = sub[["date"]].copy()
            rs["pos"] = sub["pos"].to_numpy(float)
            rs["r_gross"] = gross
            rs["r_cost"] = cost
            rs["r_net"] = net
            return_series[key] = rs

            rows_phase.append({
                "model": model,
                "H": H,
                "strategy": "model",
                "tau": float(tau),
                "phase": int(phase),
                **m.__dict__,
                "tc_bps": float(tc_bps),
                "allow_short": bool(allow_short),
                "non_overlapping": bool(use_non_overlapping),
            })

    # 2) baseline 策略
    for b in baselines:
        pos_b = apply_baseline(b, n=len(df))
        df_b = df[["date", "y_true"]].copy()
        df_b["pos"] = pos_b

        # baseline 不依赖 tau，但为了“同表对齐”，仍对每个 tau 输出一份（便于报告横向对比）
        for tau in tau_list:
            for phase in phases:
                sub = subset_non_overlapping(df_b, H=H, phase=phase) if use_non_overlapping else df_b
                gross, cost, net = compute_net_returns(sub["y_true"].to_numpy(float), sub["pos"].to_numpy(float), tc_bps=tc_bps)
                m = compute_econ_metrics(net, sub["pos"].to_numpy(float), annual_days=annual_days, H=H, non_overlapping=use_non_overlapping)

                key = f"{b}|tau={tau}|phase={phase}"
                rs = sub[["date"]].copy()
                rs["pos"] = sub["pos"].to_numpy(float)
                rs["r_gross"] = gross
                rs["r_cost"] = cost
                rs["r_net"] = net
                return_series[key] = rs

                rows_phase.append({
                    "model": model,
                    "H": H,
                    "strategy": str(b),
                    "tau": float(tau),
                    "phase": int(phase),
                    **m.__dict__,
                    "tc_bps": float(tc_bps),
                    "allow_short": bool(allow_short),
                    "non_overlapping": bool(use_non_overlapping),
                })

    econ_by_phase = pd.DataFrame(rows_phase)

    group_cols = ["model", "H", "strategy", "tau", "tc_bps", "allow_short", "non_overlapping"]
    econ_phase_summary = summarize_phase_table(econ_by_phase, group_cols=group_cols) if use_non_overlapping else pd.DataFrame()

    # econ_total：如果做了 phase_sweep，则用 phase_summary 的 mean 口径作为“总表”；否则直接取 phase=0
    if use_non_overlapping:
        # 从 *_mean 列中取出一组更直观的总表
        cols_map = {
            "mean_log_ret_mean": "mean_log_ret",
            "std_log_ret_mean": "std_log_ret",
            "sharpe_ann_mean": "sharpe_ann",
            "hit_rate_mean": "hit_rate",
            "turnover_mean": "turnover",
            "max_drawdown_mean": "max_drawdown",
            "n_obs_mean": "n_obs",
        }
        keep = group_cols + list(cols_map.keys())
        tmp = econ_phase_summary[keep].copy() if not econ_phase_summary.empty else pd.DataFrame(columns=keep)
        tmp = tmp.rename(columns=cols_map)
        econ_total = tmp
    else:
        econ_total = econ_by_phase[econ_by_phase["phase"] == 0].drop(columns=["phase"]).copy()

    return econ_total, econ_by_phase, econ_phase_summary, return_series

# =========================================================
# Step3 pipeline 入口：批量运行 + 写文件
# =========================================================

def _plot_equity_curve(df_ret: pd.DataFrame, title: str, out_path: str) -> None:
    """简单资金曲线图：equity = exp(cumsum(r_net))。

    说明：
    - 这里使用对数收益的指数化作为简单资金曲线（忽略复利细节差异）。
    - 若你们后续想加入更严格的资金管理/杠杆/保证金，可在此基础上扩展。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    d = df_ret.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    equity = np.exp(np.cumsum(d["r_net"].to_numpy(float)))

    plt.figure(figsize=(10, 4))
    plt.plot(d["date"], equity)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (exp(cumsum log ret))")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_economic_layer(
    pred_frames: List[Dict[str, object]],
    cfg: Dict[str, object],
    dirs: Dict[str, str],
) -> Dict[str, List[str]]:
    """Step3：批量运行经济层检验并输出表格/收益序列/图。

    参数：
    - pred_frames: 来自 step3_pipeline 的列表，每个元素含 model/H/df
    - cfg: config_step3.RUN
    - dirs: Step3 输出目录（datasets/figures/tables/...）
    """

    econ_cfg = (cfg.get("economic", {}) or {})  # type: ignore

    out_tables: List[str] = []
    out_datasets: List[str] = []
    out_figures: List[str] = []

    totals: List[pd.DataFrame] = []
    by_phases: List[pd.DataFrame] = []
    phase_summaries: List[pd.DataFrame] = []

    save_series = bool(econ_cfg.get("save_return_series", True))
    make_fig = bool(econ_cfg.get("make_equity_fig", True))

    for it in pred_frames:
        model = str(it["model"])
        H = int(it["H"])
        df = it["df"]  # type: ignore

        econ_total, econ_by_phase, econ_phase_summary, return_series = run_economic_eval_for_one(
            df=df,
            model=model,
            H=H,
            econ_cfg=econ_cfg,
        )

        totals.append(econ_total)
        by_phases.append(econ_by_phase)
        if not econ_phase_summary.empty:
            phase_summaries.append(econ_phase_summary)

        # 保存收益序列（长表）：一份文件包含同 (model,H) 的所有 series
        if save_series:
            rows = []
            for key, rs in return_series.items():
                tmp = rs.copy()
                tmp["series"] = key
                tmp["model"] = model
                tmp["H"] = H
                rows.append(tmp)
            df_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            p = str(Path(dirs["datasets"]) / f"econ_returns_{model}_H{H}.csv")
            df_long.to_csv(p, index=False, encoding="utf-8")
            out_datasets.append(p)

        # 资金曲线：默认只画“模型策略 + tau_list[0] + phase=0”，避免图太多
        if make_fig:
            tau0 = float((econ_cfg.get("tau_list", [0.0]) or [0.0])[0])
            k = f"model|tau={tau0}|phase=0"
            if k in return_series:
                fig_p = str(Path(dirs["figures"]) / f"equity_{model}_H{H}_tau{tau0}_phase0.png")
                _plot_equity_curve(
                    return_series[k],
                    title=f"Equity | {model} | H={H} | tau={tau0} | phase=0",
                    out_path=fig_p,
                )
                out_figures.append(fig_p)

    # 写表
    econ_metrics_path = str(Path(dirs["tables"]) / "econ_metrics.csv")
    econ_by_phase_path = str(Path(dirs["tables"]) / "econ_metrics_by_phase.csv")
    econ_phase_summary_path = str(Path(dirs["tables"]) / "econ_metrics_phase_summary.csv")

    pd.concat(totals, ignore_index=True).to_csv(econ_metrics_path, index=False, encoding="utf-8")
    pd.concat(by_phases, ignore_index=True).to_csv(econ_by_phase_path, index=False, encoding="utf-8")
    if phase_summaries:
        pd.concat(phase_summaries, ignore_index=True).to_csv(econ_phase_summary_path, index=False, encoding="utf-8")
    else:
        # 若未启用 non_overlapping，则可能没有 phase_summary
        pd.DataFrame().to_csv(econ_phase_summary_path, index=False, encoding="utf-8")

    out_tables.extend([econ_metrics_path, econ_by_phase_path, econ_phase_summary_path])

    return {"tables": out_tables, "datasets": out_datasets, "figures": out_figures}
