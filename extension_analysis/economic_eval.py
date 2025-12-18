# -*- coding: utf-8 -*-
"""Step3 经济层检验（Economic Layer）

输入
- Step2 的 predictions_{model}_H{H}.csv：date, y_true, y_pred

输出
- datasets/econ_trades_long.csv：交易级别长表（供 VIX / 可视化复用）
- tables/econ_metrics_by_phase.csv：每个相位一行
- tables/econ_metrics_phase_summary.csv：对相位做 mean/median/min/max 汇总
- figures/equity_*.png：资金曲线（可选）

核心口径
- y_true 为持有期对数收益标签：log(S_{t+H})-log(S_t)
- 策略仓位 pos_t：由 y_pred 与阈值 tau 决定（-1/0/1 或 0/1）
- 交易成本 cost_t：按仓位变化计费 cost_t = tc * |pos_t-pos_{t-1}|
  其中 tc = tc_bps/10000（bps -> 对数收益近似扣减）

所有注释采用中文。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.step3.plot_utils import save_figure


def _build_position(y_pred: np.ndarray, tau: float, allow_short: bool) -> np.ndarray:
    """根据预测值生成仓位 pos（-1/0/1 或 0/1）。"""
    y_pred = np.asarray(y_pred, dtype=float)
    tau = float(tau)

    if allow_short:
        pos = np.zeros_like(y_pred, dtype=int)
        pos[y_pred > tau] = 1
        pos[y_pred < -tau] = -1
        return pos

    # 不允许做空：负信号直接空仓
    pos = np.zeros_like(y_pred, dtype=int)
    pos[y_pred > tau] = 1
    return pos


def _compute_cost(pos: np.ndarray, tc_bps: float) -> np.ndarray:
    """按仓位变化计费：cost_t = tc * |pos_t - pos_{t-1}|。

    说明
    - tc_bps 为单边成本（bps）
    - pos 从 1 -> -1 的“反手”变化幅度为 2，因此会收取 2 倍成本
    """
    tc = float(tc_bps) / 10000.0
    p = np.asarray(pos, dtype=float)
    dp = np.abs(np.diff(p, prepend=p[:1]))
    dp[0] = 0.0
    return tc * dp


def _max_drawdown_from_log_returns(r: np.ndarray) -> float:
    """由对数收益序列计算最大回撤（基于累计净值 exp(cumsum(r))）。"""
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return float("nan")
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(np.min(dd))


def _sharpe_ann_from_log_returns(r: np.ndarray, annual_scale: float) -> float:
    """年化 Sharpe（对数收益口径）。"""
    r = np.asarray(r, dtype=float)
    if len(r) < 3:
        return float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(mu / sd * float(annual_scale))


def compute_econ_metrics(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    strategy: str,
    H: int,
    tau: float,
    allow_short: bool,
    tc_bps: float,
    annual_days: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """计算交易序列与经济指标。

    返回
    - trades_df：包含 date/pos/r_net 等列
    - metrics：摘要指标字典

    注意
    - 该函数假设输入序列为“交易级别”：每行代表一笔持有期为 H 的交易
    """
    dates = pd.to_datetime(np.asarray(dates))
    y_true = np.asarray(y_true, dtype=float)
    if y_pred is None:
        y_pred = np.full_like(y_true, np.nan, dtype=float)

    if strategy == "model":
        pos = _build_position(y_pred, tau=tau, allow_short=allow_short)
    elif strategy == "cash":
        pos = np.zeros_like(y_true, dtype=int)
    elif strategy == "always_long":
        pos = np.ones_like(y_true, dtype=int)
    elif strategy == "always_short":
        pos = -np.ones_like(y_true, dtype=int) if allow_short else np.zeros_like(y_true, dtype=int)
    else:
        raise ValueError(f"未知策略: {strategy}")

    cost = _compute_cost(pos, tc_bps=tc_bps)
    r_gross = pos.astype(float) * y_true
    r_net = r_gross - cost

    trades = pd.DataFrame({
        "date": dates,
        "y_true": y_true,
        "y_pred": np.asarray(y_pred, dtype=float),
        "pos": pos,
        "r_gross": r_gross,
        "cost": cost,
        "r_net": r_net,
    })

    # 年化系数：每笔交易跨度 H 日，频率约为 annual_days/H
    ann_scale = math.sqrt(float(annual_days) / float(max(1, H)))

    mdd = _max_drawdown_from_log_returns(r_net)
    sharpe = _sharpe_ann_from_log_returns(r_net, annual_scale=ann_scale)
    hit_rate = float(np.mean(r_net > 0.0)) if len(r_net) > 0 else float("nan")
    turnover = float(np.mean(np.abs(np.diff(pos, prepend=pos[:1])))) if len(pos) > 0 else float("nan")

    metrics = {
        "n_obs": int(len(trades)),
        "mean_log_ret": float(np.mean(r_net)) if len(r_net) > 0 else float("nan"),
        "std_log_ret": float(np.std(r_net, ddof=1)) if len(r_net) > 1 else float("nan"),
        "sharpe_ann": sharpe,
        "max_drawdown": mdd,
        "hit_rate": hit_rate,
        "turnover": turnover,
        "tc_bps": float(tc_bps),
        "allow_short": bool(allow_short),
        "tau": float(tau),
    }

    return trades, metrics




def run_economic_layer(
    run_cfg: Dict[str, Any],
    combos: List[Dict[str, Any]],
    pred_frames: Dict[Tuple[str, int], pd.DataFrame],
    out_dirs: Dict[str, str],
) -> Dict[str, Any]:
    """经济层检验主入口。

    参数
    - combos: [{model, H, path}, ...]（已按 selection 过滤）
    - pred_frames: {(model,H): predictions_df}
    - out_dirs: {"tables":..., "datasets":..., "figures":...}

    返回
    - 输出文件路径集合（供 pipeline 注册到 manifest）
    """
    cfg = dict(run_cfg.get("economic", {}) or {})
    if not bool(cfg.get("enable", True)):
        return {}

    allow_short = bool(cfg.get("allow_short", True))
    tau_list = list(cfg.get("tau_list", [0.0]))
    tc_bps = float(cfg.get("tc_bps", 1.0))
    annual_days = int(cfg.get("annual_days", 252))
    use_non_overlapping = bool(cfg.get("use_non_overlapping", True))
    phase_sweep = bool(cfg.get("phase_sweep", True))

    # 基准策略（只依赖 y_true，不依赖模型），为避免在每个模型上重复输出：按 H 只算一次
    baselines = list(cfg.get("baselines", []))
    baseline_strategies = [s for s in baselines if str(s) not in ("model", "")]

    trades_long_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []

    combos_sorted = sorted(combos, key=lambda x: (str(x.get("model")), int(x.get("H"))))

    # 记录每个 H 的代表模型（用于取 dates/y_true 来算 baselines）
    rep_model_for_H: Dict[int, str] = {}
    for c in combos_sorted:
        H = int(c["H"])
        rep_model_for_H.setdefault(H, str(c["model"]))

    # -------------------
    # A) 模型策略：每个 (model,H) 都计算
    # -------------------
    for c in combos_sorted:
        model = str(c["model"])
        H = int(c["H"])

        df = pred_frames[(model, H)].copy()
        df = df.sort_values("date").reset_index(drop=True)

        dates = df["date"].to_numpy()
        y_true = df["y_true"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=float)

        # 相位列表
        if use_non_overlapping:
            phases = list(range(H)) if phase_sweep else [0]
        else:
            phases = [0]

        for tau in tau_list:
            for phase in phases:
                if use_non_overlapping:
                    idx = np.arange(phase, len(df), H, dtype=int)
                else:
                    idx = np.arange(len(df), dtype=int)

                d_i = dates[idx]
                y_i = y_true[idx]
                p_i = y_pred[idx]

                trades, metrics = compute_econ_metrics(
                    dates=d_i,
                    y_true=y_i,
                    y_pred=p_i,
                    strategy="model",
                    H=H,
                    tau=float(tau),
                    allow_short=allow_short,
                    tc_bps=tc_bps,
                    annual_days=annual_days,
                )

                if bool(cfg.get("save_trades_long", True)):
                    tmp = trades.copy()
                    tmp["model"] = model
                    tmp["H"] = H
                    tmp["strategy"] = "model"
                    tmp["tau"] = float(tau)
                    tmp["phase"] = int(phase)
                    tmp["allow_short"] = bool(allow_short)
                    tmp["tc_bps"] = float(tc_bps)
                    tmp["use_non_overlapping"] = bool(use_non_overlapping)
                    tmp["is_baseline"] = False
                    tmp["ref_model"] = ""
                    trades_long_rows.append(tmp)

                metric_rows.append({
                    "model": model,
                    "H": H,
                    "strategy": "model",
                    "tau": float(tau),
                    "phase": int(phase),
                    "use_non_overlapping": bool(use_non_overlapping),
                    "is_baseline": False,
                    **metrics,
                })

    # -------------------
    # B) 基准策略：按 H 只算一次（model 字段固定为 BASELINE）
    # -------------------
    if len(baseline_strategies) > 0:
        baseline_tau = 0.0  # 基准策略与 tau 无关，避免在多个 tau 上重复输出

        for H, rep_model in sorted(rep_model_for_H.items(), key=lambda x: x[0]):
            df = pred_frames[(rep_model, H)].copy()
            df = df.sort_values("date").reset_index(drop=True)

            dates = df["date"].to_numpy()
            y_true = df["y_true"].to_numpy(dtype=float)

            if use_non_overlapping:
                phases = list(range(H)) if phase_sweep else [0]
            else:
                phases = [0]

            for strategy in baseline_strategies:
                for phase in phases:
                    if use_non_overlapping:
                        idx = np.arange(phase, len(df), H, dtype=int)
                    else:
                        idx = np.arange(len(df), dtype=int)

                    d_i = dates[idx]
                    y_i = y_true[idx]

                    trades, metrics = compute_econ_metrics(
                        dates=d_i,
                        y_true=y_i,
                        y_pred=None,
                        strategy=str(strategy),
                        H=H,
                        tau=float(baseline_tau),
                        allow_short=allow_short,
                        tc_bps=tc_bps,
                        annual_days=annual_days,
                    )

                    if bool(cfg.get("save_trades_long", True)):
                        tmp = trades.copy()
                        tmp["model"] = "BASELINE"
                        tmp["H"] = H
                        tmp["strategy"] = str(strategy)
                        tmp["tau"] = float(baseline_tau)
                        tmp["phase"] = int(phase)
                        tmp["allow_short"] = bool(allow_short)
                        tmp["tc_bps"] = float(tc_bps)
                        tmp["use_non_overlapping"] = bool(use_non_overlapping)
                        tmp["is_baseline"] = True
                        tmp["ref_model"] = str(rep_model)
                        trades_long_rows.append(tmp)

                    metric_rows.append({
                        "model": "BASELINE",
                        "H": H,
                        "strategy": str(strategy),
                        "tau": float(baseline_tau),
                        "phase": int(phase),
                        "use_non_overlapping": bool(use_non_overlapping),
                        "is_baseline": True,
                        **metrics,
                    })

    tables_dir = Path(out_dirs["tables"])
    datasets_dir = Path(out_dirs["datasets"])
    figures_dir = Path(out_dirs["figures"])

    # 1) 每相位指标表
    metrics_by_phase = pd.DataFrame(metric_rows)
    metrics_by_phase_path = str(tables_dir / "econ_metrics_by_phase.csv")
    metrics_by_phase.to_csv(metrics_by_phase_path, index=False, encoding="utf-8")

    # 2) across-phase 汇总（mean/median/min/max）
    group_cols = ["model", "H", "strategy", "tau", "use_non_overlapping", "is_baseline"]

    def _named_agg(col: str) -> Dict[str, Tuple[str, str]]:
        return {
            f"{col}_mean": (col, "mean"),
            f"{col}_median": (col, "median"),
            f"{col}_min": (col, "min"),
            f"{col}_max": (col, "max"),
        }

    agg_spec: Dict[str, Tuple[str, str]] = {}
    for col in ["sharpe_ann", "mean_log_ret", "max_drawdown", "hit_rate", "turnover", "n_obs"]:
        agg_spec.update(_named_agg(col))

    summary = (
        metrics_by_phase
        .groupby(group_cols, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    metrics_phase_summary_path = str(tables_dir / "econ_metrics_phase_summary.csv")
    summary.to_csv(metrics_phase_summary_path, index=False, encoding="utf-8")

    outputs: Dict[str, Any] = {
        "tables": [metrics_by_phase_path, metrics_phase_summary_path],
        "datasets": [],
        "figures": [],
        "trades_long_path": None,
    }

    # 3) 保存交易级别长表
    if bool(cfg.get("save_trades_long", True)) and len(trades_long_rows) > 0:
        trades_long = pd.concat(trades_long_rows, ignore_index=True)
        trades_long_path = str(datasets_dir / "econ_trades_long.csv")
        trades_long.to_csv(trades_long_path, index=False, encoding="utf-8")
        outputs["datasets"].append(trades_long_path)
        outputs["trades_long_path"] = trades_long_path

    # 4) 资金曲线（可选）：使用指定 plot_phase + 第一组 tau（避免图片爆炸）
    if bool(cfg.get("save_equity_fig", True)) and outputs["trades_long_path"]:
        plot_phase = int(cfg.get("plot_phase", 0))
        plot_strategies = list(cfg.get("plot_strategies", ["model"]))
        tau_for_plot = [tau_list[0]] if len(tau_list) > 0 else [0.0]

        trades_long = pd.read_csv(str(outputs["trades_long_path"]), encoding="utf-8")
        trades_long["date"] = pd.to_datetime(trades_long["date"])

        for c in combos_sorted:
            model = str(c["model"])
            H = int(c["H"])
            for tau in tau_for_plot:
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(111)

                for strategy in plot_strategies:
                    # 基准策略从 model=BASELINE 取；模型策略从对应 model 取
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

                    eq = np.exp(np.cumsum(sub["r_net"].to_numpy(dtype=float)))
                    ax.plot(sub["date"].to_numpy(), eq, label=str(strategy))

                ax.set_title(f"资金曲线 | model={model} | H={H} | tau={tau} | phase={plot_phase}")
                ax.set_xlabel("日期")
                ax.set_ylabel("累计净值（exp(累计对数收益)）")
                ax.legend()

                out_path = str(figures_dir / f"equity_{model}_H{H}_tau{tau}_phase{plot_phase}.png")
                save_figure(fig, out_path, dpi=220)
                outputs["figures"].append(out_path)

    return outputs
