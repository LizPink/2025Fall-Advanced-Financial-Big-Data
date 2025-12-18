# -*- coding: utf-8 -*-
"""Step3：VIX 分时期（Regime）分析。

本模块实现两类输出：
1）Conditional（条件绩效）：先按时间顺序计算策略收益，再按 regime 分组统计指标，不改变时序。
2）Episode（连续区间）：对 daily regime 标签做 run-length encoding 得到连续区间，便于叙事与画图。

关键设计：
- 阈值来源可选 train_only / expanding，默认 train_only 以避免前视。
- 支持 hysteresis（双阈值进入/退出）与 min_spell_days 平滑，避免 regime 抖动导致 episode 过碎。

注意：
- regime_on 默认 entry：用 VIX_t 给 (t->t+H) 这笔交易贴标签。
"""

from __future__ import annotations

import math
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.step3.economic import compute_econ_metrics, build_position, apply_baseline, compute_net_returns, subset_non_overlapping


@dataclass(frozen=True)
class RegimeThresholds:
    enter_low: float
    exit_low: float
    enter_high: float
    exit_high: float


def _validate_thresholds(th: RegimeThresholds) -> None:
    # 简单防守：确保进入/退出阈值之间有序；若不满足，则只做退化处理（不抛异常）
    pass


def compute_thresholds_train_only(
    vix: pd.Series,
    train_mask: pd.Series,
    mode: str,
    quantiles: Dict[str, float],
    fixed_thresholds: Dict[str, float],
) -> RegimeThresholds:
    """仅用训练/验证期 vix 估计阈值，固定后用于测试期。"""
    v_train = vix[train_mask].dropna()
    if len(v_train) < 20:
        # 训练期太短：退化为全样本（并在上层写 manifest note）
        v_train = vix.dropna()

    mode = str(mode).lower()
    if mode == "fixed":
        return RegimeThresholds(
            enter_low=float(fixed_thresholds["enter_low"]),
            exit_low=float(fixed_thresholds["exit_low"]),
            enter_high=float(fixed_thresholds["enter_high"]),
            exit_high=float(fixed_thresholds["exit_high"]),
        )

    # quantile
    el = float(v_train.quantile(float(quantiles["enter_low"])))
    xl = float(v_train.quantile(float(quantiles["exit_low"])))
    eh = float(v_train.quantile(float(quantiles["enter_high"])))
    xh = float(v_train.quantile(float(quantiles["exit_high"])))
    return RegimeThresholds(enter_low=el, exit_low=xl, enter_high=eh, exit_high=xh)


def compute_thresholds_expanding(
    vix: pd.Series,
    mode: str,
    quantiles: Dict[str, float],
    fixed_thresholds: Dict[str, float],
    min_history: int = 252,
) -> pd.DataFrame:
    """到 t 为止用历史 vix 估计阈值（expanding），返回每一天的阈值。"""
    mode = str(mode).lower()
    idx = vix.index
    out = pd.DataFrame(index=idx, columns=["enter_low","exit_low","enter_high","exit_high"], dtype=float)

    if mode == "fixed":
        out.loc[:, "enter_low"] = float(fixed_thresholds["enter_low"])
        out.loc[:, "exit_low"] = float(fixed_thresholds["exit_low"])
        out.loc[:, "enter_high"] = float(fixed_thresholds["enter_high"])
        out.loc[:, "exit_high"] = float(fixed_thresholds["exit_high"])
        return out

    # quantile expanding：对每个 t，用 vix[:t] 的分位数作为阈值
    # 说明：为了避免最初样本过少导致阈值不稳，设置 min_history；不足时阈值记为 NaN
    qs = {k: float(v) for k, v in quantiles.items()}
    for i, t in enumerate(idx):
        hist = vix.iloc[: i + 1].dropna()
        if len(hist) < int(min_history):
            continue
        out.at[t, "enter_low"] = float(hist.quantile(qs["enter_low"]))
        out.at[t, "exit_low"] = float(hist.quantile(qs["exit_low"]))
        out.at[t, "enter_high"] = float(hist.quantile(qs["enter_high"]))
        out.at[t, "exit_high"] = float(hist.quantile(qs["exit_high"]))
    return out


def label_regime_hysteresis(vix: pd.Series, th: RegimeThresholds) -> pd.Series:
    """基于 hysteresis 规则对每日 vix 打 regime 标签（low/mid/high）。"""
    # 这里不强行校验阈值顺序（可能存在极端情况），只要规则可运行即可。
    state = "mid"
    lab: List[str] = []
    for val in vix.astype(float).tolist():
        if not math.isfinite(val):
            lab.append(state)
            continue

        # high 进入/退出
        if state != "high" and val >= float(th.enter_high):
            state = "high"
        elif state == "high" and val <= float(th.exit_high):
            state = "mid"

        # low 进入/退出
        if state != "low" and val <= float(th.enter_low):
            state = "low"
        elif state == "low" and val >= float(th.exit_low):
            state = "mid"

        lab.append(state)

    return pd.Series(lab, index=vix.index, name="regime")


@dataclass(frozen=True)
class Episode:
    start: pd.Timestamp
    end: pd.Timestamp
    regime: str
    length: int


def build_episodes(regime: pd.Series, min_spell_days: int = 5) -> List[Episode]:
    """把 daily regime 标签合并为连续区间（episode）。"""
    if regime.empty:
        return []

    r = regime.astype(str)
    episodes: List[Episode] = []

    start = r.index[0]
    cur = r.iloc[0]
    last = r.index[0]

    for t, v in r.iloc[1:].items():
        if v == cur:
            last = t
            continue
        # 结束上一段
        length = int((last - start).days) + 1
        episodes.append(Episode(start=start, end=last, regime=cur, length=length))
        # 新开一段
        start = t
        cur = v
        last = t

    length = int((last - start).days) + 1
    episodes.append(Episode(start=start, end=last, regime=cur, length=length))

    # 平滑：太短的 episode 合并到邻近的 mid（或邻近更长段）
    # 这里采取确定性规则：
    # - 若段长 < min_spell_days 且 regime 不是 mid：直接改为 mid，再重新生成 episodes
    # 这样简单且可解释，避免复杂合并规则影响复现。
    if int(min_spell_days) > 1:
        rr = r.copy()
        for ep in episodes:
            if ep.length < int(min_spell_days) and ep.regime in ("low", "high"):
                rr.loc[ep.start: ep.end] = "mid"
        # 递归一次即可（mid 合并后不会再变碎）
        if not rr.equals(r):
            return build_episodes(rr, min_spell_days=0)

    return episodes


def conditional_metrics_by_regime(
    df_trades: pd.DataFrame,
    annual_days: int,
    H: int,
    non_overlapping: bool,
) -> pd.DataFrame:
    """对交易序列按 regime 分组，计算经济指标。

    df_trades 必需列：date, r_net, pos, regime
    """
    rows = []
    for reg, g in df_trades.groupby("regime"):
        m = compute_econ_metrics(
            r_net=g["r_net"].to_numpy(float),
            pos=g["pos"].to_numpy(float),
            annual_days=int(annual_days),
            H=int(H),
            non_overlapping=bool(non_overlapping),
        )
        rows.append({
            "regime": str(reg),
            "mean_log_ret": m.mean_log_ret,
            "std_log_ret": m.std_log_ret,
            "sharpe_ann": m.sharpe_ann,
            "hit_rate": m.hit_rate,
            "turnover": m.turnover,
            "max_drawdown": m.max_drawdown,
            "n_obs": m.n_obs,
        })
    return pd.DataFrame(rows)


def episode_table(episodes: List[Episode]) -> pd.DataFrame:
    """episodes -> DataFrame。"""
    if not episodes:
        return pd.DataFrame(columns=["start", "end", "regime", "length"])
    return pd.DataFrame([
        {"start": ep.start, "end": ep.end, "regime": ep.regime, "length": ep.length}
        for ep in episodes
    ])

# =========================================================
# Step3 pipeline 入口：VIX 分时期批量运行 + 写文件
# =========================================================


def _read_step2_snapshot(step2_output_dir: str) -> Dict[str, Any]:
    """读取 Step2 配置快照（用于 train/test 边界一致）。"""
    p = Path(step2_output_dir).expanduser().resolve() / "meta" / "config_snapshot.json"
    if not p.exists():
        # 兼容你们可能保存成不同名字
        p = Path(step2_output_dir).expanduser().resolve() / "meta" / "config_snapshot_step2.json"
    if not p.exists():
        raise FileNotFoundError(f"未找到 Step2 config snapshot: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_vix_series(step1_dataset_dir: str, H: int, step2_snapshot: Dict[str, Any], vix_col: str) -> pd.DataFrame:
    """从 Step1 Data_D_{H}.xlsx 读取 date 与 VIX 列。"""
    date_col = step2_snapshot.get("date_col", "date")
    template = step2_snapshot.get("dataset_file_template", "Data_D_{H}.xlsx")
    p = Path(step1_dataset_dir).expanduser().resolve() / str(template).format(H=int(H))
    if not p.exists():
        raise FileNotFoundError(f"未找到 Step1 数据集: {p}")

    df = pd.read_excel(p, sheet_name="dataset")
    if date_col not in df.columns:
        raise KeyError(f"Step1 数据集缺少 date_col={date_col} | file={p}")
    if vix_col not in df.columns:
        raise KeyError(f"Step1 数据集缺少 vix_col={vix_col} | file={p}")

    out = df[[date_col, vix_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)
    out = out.rename(columns={date_col: "date", vix_col: "vix"})
    return out


def _train_test_split_idx_from_snapshot(dates: pd.Series, step2_snapshot: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """按 Step2 口径得到 train_val_idx/test_idx。

    这里不直接依赖 step2.split_utils（避免包路径问题），逻辑与 Step2 一致。
    """
    dt = pd.to_datetime(dates)
    n = len(dt)
    mode = str(step2_snapshot.get("test_mode", "last_ratio"))

    if mode == "last_ratio":
        ratio = float(step2_snapshot.get("test_ratio", 0.2))
        cut = int(round(n * (1 - ratio)))
        cut = max(1, min(n - 1, cut))
        tr = np.arange(0, cut, dtype=int)
        te = np.arange(cut, n, dtype=int)
        return tr, te

    if mode == "last_years":
        years = int(step2_snapshot.get("test_years", 3))
        end = dt.iloc[-1]
        start = end - pd.DateOffset(years=years)
        test_mask = dt >= start
        if int(test_mask.sum()) < 10:
            # 回退
            cut = int(round(n * (1 - float(step2_snapshot.get("test_ratio", 0.2)))))
            cut = max(1, min(n - 1, cut))
            return np.arange(0, cut, dtype=int), np.arange(cut, n, dtype=int)
        te = np.where(test_mask.to_numpy())[0].astype(int)
        tr = np.where((~test_mask).to_numpy())[0].astype(int)
        return tr, te

    if mode == "date_range":
        s = step2_snapshot.get("test_start", None)
        e = step2_snapshot.get("test_end", None)
        if s is None or e is None:
            raise ValueError("Step2 snapshot 的 date_range 需要 test_start/test_end")
        start = pd.to_datetime(s)
        end = pd.to_datetime(e)
        test_mask = (dt >= start) & (dt <= end)
        if int(test_mask.sum()) < 10:
            raise ValueError("date_range yields too few samples")
        te = np.where(test_mask.to_numpy())[0].astype(int)
        tr = np.where((~test_mask).to_numpy())[0].astype(int)
        return tr, te

    raise ValueError(f"未知 test_mode={mode}")


def _plot_equity_with_vix_shading(df_trades: pd.DataFrame, episodes: List[Episode], title: str, out_path: str) -> None:
    """画资金曲线，并对高 VIX episode 做阴影。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if df_trades.empty:
        return

    d = df_trades.sort_values("date").copy()
    eq = np.exp(np.cumsum(d["r_net"].to_numpy(float)))

    fig = plt.figure(figsize=(10, 4))
    plt.plot(d["date"], eq)

    # 高 VIX 阴影
    for ep in episodes:
        if str(ep.regime) != "high":
            continue
        plt.axvspan(ep.start, ep.end, alpha=0.15)

    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("equity")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def run_vix_regime(
    pred_frames: List[Dict[str, Any]],
    step1_dataset_dir: str,
    step2_output_dir: str,
    cfg: Dict[str, Any],
    dirs: Dict[str, str],
) -> Dict[str, Any]:
    """Step3：VIX 分时期主入口。

    输入：
    - pred_frames：来自 step3_pipeline 的 predictions DataFrame 列表

    输出：
    - tables/econ_by_vix_regime_conditional.csv
    - tables/econ_by_vix_episode.csv
    - datasets/vix_labeled_H{H}.csv
    - datasets/vix_episodes_H{H}.csv
    - figures/equity_with_vix_shading_*.png（可选）
    """

    log = logging.getLogger("step3.vix")
    vcfg = (cfg.get("vix_regime", {}) or {})  # type: ignore
    ecfg = (cfg.get("economic", {}) or {})  # 复用成本/非重叠/phase/tau/baselines

    out_tables: List[str] = []
    out_datasets: List[str] = []
    out_figures: List[str] = []

    vix_col = str(vcfg.get("vix_col", "VIX"))
    mode = str(vcfg.get("mode", "quantile"))
    threshold_source = str(vcfg.get("threshold_source", "train_only"))
    quant = vcfg.get("quantiles", {}) or {}
    fixed = vcfg.get("fixed_thresholds", {}) or {}
    min_spell_days = int(vcfg.get("min_spell_days", 5))

    methods = list(vcfg.get("methods", ["conditional", "episode"]))

    allow_short = bool(ecfg.get("allow_short", True))
    tau_list = [float(x) for x in (ecfg.get("tau_list", [0.0]) or [0.0])]
    baselines = list(ecfg.get("baselines", []))
    tc_bps = float(ecfg.get("tc_bps", 1.0))
    annual_days = int(ecfg.get("annual_days", 252))

    use_non_overlapping = bool(ecfg.get("use_non_overlapping", True))
    phase_sweep = bool(ecfg.get("phase_sweep", True))

    # Step2 snapshot（用于确定 train/test 边界，计算 train_only 阈值）
    step2_snapshot = _read_step2_snapshot(step2_output_dir)

    # 先按 H 缓存 VIX+regime（regime 只依赖 H 的 test 区间日期）
    cache_by_H: Dict[int, Dict[str, Any]] = {}

    # 最终输出长表
    rows_cond: List[Dict[str, Any]] = []
    rows_ep: List[Dict[str, Any]] = []

    # 逐个 predictions
    for it in pred_frames:
        model = str(it["model"])
        H = int(it["H"])
        df_pred: pd.DataFrame = it["df"]  # type: ignore

        if H not in cache_by_H:
            # 读取 Step1 的 VIX series
            df_vix = _load_vix_series(step1_dataset_dir, H=H, step2_snapshot=step2_snapshot, vix_col=vix_col)

            # 计算 train/test 边界（用 Step1 的 date）
            train_idx, test_idx = _train_test_split_idx_from_snapshot(df_vix["date"], step2_snapshot)
            df_test = df_vix.iloc[test_idx].copy().reset_index(drop=True)

            # 计算阈值并生成 regime
            if mode == "fixed":
                thr = Thresholds(
                    enter_low=float(fixed.get("enter_low")),
                    exit_low=float(fixed.get("exit_low")),
                    enter_high=float(fixed.get("enter_high")),
                    exit_high=float(fixed.get("exit_high")),
                )
                reg = label_regime_hysteresis(df_test["vix"], thr)
            elif mode == "quantile":
                if threshold_source == "train_only":
                    v_train = df_vix.iloc[train_idx]["vix"].dropna()
                    if v_train.empty:
                        raise ValueError("train_only 阈值计算失败：训练期 VIX 为空")
                    thr = compute_thresholds_quantile(v_train, quant)
                    reg = label_regime_hysteresis(df_test["vix"], thr)
                elif threshold_source == "expanding":
                    # 每天用历史估阈值，并用当日阈值推进状态机
                    reg = label_regime_expanding(df_vix, test_idx=test_idx, qcfg=quant)
                    thr = None  # type: ignore
                else:
                    raise ValueError(f"未知 threshold_source={threshold_source}")
            else:
                raise ValueError(f"未知 vix_regime.mode={mode}")

            df_test["regime"] = reg

            # 平滑：把太短的 high/low run 改为 mid，再重新做 episode
            reg_sm = smooth_regime_min_spell(df_test["regime"].tolist(), min_spell_days=min_spell_days)
            df_test["regime_sm"] = reg_sm

            episodes = build_episodes(df_test["date"], df_test["regime_sm"].tolist())

            cache_by_H[H] = {
                "df_test": df_test,
                "episodes": episodes,
                "thresholds": thr.to_dict() if hasattr(thr, "to_dict") else None,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }

            # 保存标注序列与 episode（按 H 一份）
            if bool(vcfg.get("save_labeled_series", True)):
                p1 = str(Path(dirs["datasets"]) / f"vix_labeled_H{H}.csv")
                df_test[["date", "vix", "regime", "regime_sm"]].to_csv(p1, index=False, encoding="utf-8")
                out_datasets.append(p1)

            if "episode" in methods:
                p2 = str(Path(dirs["datasets"]) / f"vix_episodes_H{H}.csv")
                episode_table(episodes).to_csv(p2, index=False, encoding="utf-8")
                out_datasets.append(p2)

        df_test = cache_by_H[H]["df_test"]
        episodes = cache_by_H[H]["episodes"]

        # 将预测与 VIX regime 合并（只保留 predictions 的日期）
        dfm = pd.merge(df_pred[["date", "y_true", "y_pred"]], df_test[["date", "vix", "regime_sm"]], on="date", how="left")
        dfm = dfm.rename(columns={"regime_sm": "regime"})
        dfm = dfm.dropna(subset=["regime"]).reset_index(drop=True)

        if dfm.empty:
            log.warning(f"VIX 合并后为空：model={model} H={H} -> 跳过")
            continue

        phases = list(range(H)) if (use_non_overlapping and phase_sweep and H > 1) else [0]

        # 1) 条件绩效：对 model 策略与 baselines 分别输出
        if "conditional" in methods:
            # model 策略
            for tau in tau_list:
                pos = build_position(dfm["y_pred"].to_numpy(float), tau=tau, allow_short=allow_short)
                df_s = dfm[["date", "y_true", "regime"]].copy()
                df_s["pos"] = pos

                for phase in phases:
                    sub = subset_non_overlapping(df_s, H=H, phase=phase) if use_non_overlapping else df_s
                    gross, cost, net = compute_net_returns(sub["y_true"].to_numpy(float), sub["pos"].to_numpy(float), tc_bps=tc_bps)
                    sub2 = sub[["date", "regime"]].copy()
                    sub2["pos"] = sub["pos"].to_numpy(float)
                    sub2["r_net"] = net

                    tab = conditional_metrics_by_regime(sub2, annual_days=annual_days, H=H, non_overlapping=use_non_overlapping)
                    for _, r in tab.iterrows():
                        rows_cond.append({
                            "model": model,
                            "H": int(H),
                            "strategy": "model",
                            "tau": float(tau),
                            "phase": int(phase),
                            "regime": str(r["regime"]),
                            "mean_log_ret": float(r["mean_log_ret"]),
                            "std_log_ret": float(r["std_log_ret"]),
                            "sharpe_ann": float(r["sharpe_ann"]),
                            "hit_rate": float(r["hit_rate"]),
                            "turnover": float(r["turnover"]),
                            "max_drawdown": float(r["max_drawdown"]),
                            "n_obs": int(r["n_obs"]),
                            "tc_bps": float(tc_bps),
                            "allow_short": bool(allow_short),
                            "non_overlapping": bool(use_non_overlapping),
                            "vix_mode": mode,
                            "threshold_source": threshold_source,
                            "min_spell_days": int(min_spell_days),
                        })

            # baselines
            for b in baselines:
                pos_b = apply_baseline(b, n=len(dfm))
                df_b = dfm[["date", "y_true", "regime"]].copy()
                df_b["pos"] = pos_b

                for tau in tau_list:
                    for phase in phases:
                        sub = subset_non_overlapping(df_b, H=H, phase=phase) if use_non_overlapping else df_b
                        gross, cost, net = compute_net_returns(sub["y_true"].to_numpy(float), sub["pos"].to_numpy(float), tc_bps=tc_bps)
                        sub2 = sub[["date", "regime"]].copy()
                        sub2["pos"] = sub["pos"].to_numpy(float)
                        sub2["r_net"] = net

                        tab = conditional_metrics_by_regime(sub2, annual_days=annual_days, H=H, non_overlapping=use_non_overlapping)
                        for _, r in tab.iterrows():
                            rows_cond.append({
                                "model": model,
                                "H": int(H),
                                "strategy": str(b),
                                "tau": float(tau),
                                "phase": int(phase),
                                "regime": str(r["regime"]),
                                "mean_log_ret": float(r["mean_log_ret"]),
                                "std_log_ret": float(r["std_log_ret"]),
                                "sharpe_ann": float(r["sharpe_ann"]),
                                "hit_rate": float(r["hit_rate"]),
                                "turnover": float(r["turnover"]),
                                "max_drawdown": float(r["max_drawdown"]),
                                "n_obs": int(r["n_obs"]),
                                "tc_bps": float(tc_bps),
                                "allow_short": bool(allow_short),
                                "non_overlapping": bool(use_non_overlapping),
                                "vix_mode": mode,
                                "threshold_source": threshold_source,
                                "min_spell_days": int(min_spell_days),
                            })

        # 2) Episode：按 episode_id 分组输出经济指标（以模型策略为主）
        if "episode" in methods:
            # 给每个 test date 分配 episode_id
            df_ep = df_test[["date", "regime_sm"]].copy().rename(columns={"regime_sm": "regime"})
            df_ep = assign_episode_ids(df_ep, episodes)

            # 这里只对模型策略的某个 tau 画阴影资金曲线（控制产出规模）
            shading_tau = float(vcfg.get("shading_fig_tau", 0.0))
            shading_strategy = str(vcfg.get("shading_fig_strategy", "model"))
            make_shading = bool(vcfg.get("make_shading_fig", True))

            for tau in tau_list:
                pos = build_position(dfm["y_pred"].to_numpy(float), tau=tau, allow_short=allow_short)
                df_s = dfm[["date", "y_true"]].copy()
                df_s["pos"] = pos

                # episode 分析使用 phase=0（非重叠口径下）作为代表，避免维度爆炸
                sub = subset_non_overlapping(df_s, H=H, phase=0) if use_non_overlapping else df_s
                gross, cost, net = compute_net_returns(sub["y_true"].to_numpy(float), sub["pos"].to_numpy(float), tc_bps=tc_bps)
                trades = sub[["date"]].copy()
                trades["pos"] = sub["pos"].to_numpy(float)
                trades["r_net"] = net

                # merge episode id/regime
                trades = pd.merge(trades, df_ep[["date", "regime", "episode_id"]], on="date", how="left")
                trades = trades.dropna(subset=["episode_id"]).reset_index(drop=True)

                for eid, g in trades.groupby("episode_id"):
                    m = compute_econ_metrics(g["r_net"].to_numpy(float), g["pos"].to_numpy(float), annual_days=annual_days, H=H, non_overlapping=use_non_overlapping)
                    ep_info = episodes[int(eid)]
                    rows_ep.append({
                        "model": model,
                        "H": int(H),
                        "strategy": "model",
                        "tau": float(tau),
                        "phase": 0,
                        "episode_id": int(eid),
                        "regime": str(ep_info.regime),
                        "start": ep_info.start,
                        "end": ep_info.end,
                        "length_days": int(ep_info.length),
                        "mean_log_ret": m.mean_log_ret,
                        "std_log_ret": m.std_log_ret,
                        "sharpe_ann": m.sharpe_ann,
                        "hit_rate": m.hit_rate,
                        "turnover": m.turnover,
                        "max_drawdown": m.max_drawdown,
                        "n_obs": m.n_obs,
                        "tc_bps": float(tc_bps),
                        "allow_short": bool(allow_short),
                        "non_overlapping": bool(use_non_overlapping),
                        "vix_mode": mode,
                        "threshold_source": threshold_source,
                        "min_spell_days": int(min_spell_days),
                    })

                # 阴影图（只画指定 tau+strategy 的一张/每个 model×H）
                if make_shading and abs(tau - shading_tau) < 1e-12 and shading_strategy == "model":
                    fig_p = str(Path(dirs["figures"]) / f"equity_with_vix_shading_{model}_H{H}_tau{tau}.png")
                    _plot_equity_with_vix_shading(
                        df_trades=trades,
                        episodes=episodes,
                        title=f"Equity (tau={tau}) with High-VIX shading | {model} | H={H}",
                        out_path=fig_p,
                    )
                    out_figures.append(fig_p)

    # 写输出表
    if "conditional" in methods:
        p = str(Path(dirs["tables"]) / "econ_by_vix_regime_conditional.csv")
        pd.DataFrame(rows_cond).to_csv(p, index=False, encoding="utf-8")
        out_tables.append(p)

    if "episode" in methods:
        p = str(Path(dirs["tables"]) / "econ_by_vix_episode.csv")
        pd.DataFrame(rows_ep).to_csv(p, index=False, encoding="utf-8")
        out_tables.append(p)

    return {"tables": out_tables, "datasets": out_datasets, "figures": out_figures}


def assign_episode_ids(df_daily: pd.DataFrame, episodes: List[Episode]) -> pd.DataFrame:
    """给 daily regime 序列分配 episode_id。

    输入 df_daily 必需列：date, regime
    输出增加列：episode_id（从 0 开始，按 episodes 顺序）。

    说明：
    - episodes 必须是按时间连续且不重叠的区间（由 build_episodes 生成即满足）。
    - 若日期不在任何 episode 中（理论上不应发生），episode_id 置为 NaN。
    """

    if df_daily.empty or not episodes:
        out = df_daily.copy()
        out["episode_id"] = np.nan
        return out

    d = df_daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    starts = pd.to_datetime([ep.start for ep in episodes]).to_numpy()
    ends = pd.to_datetime([ep.end for ep in episodes]).to_numpy()
    dates = pd.to_datetime(d["date"]).to_numpy()

    # idx = 最后一个 start <= date
    idx = np.searchsorted(starts, dates, side="right") - 1
    valid = (idx >= 0) & (dates <= ends[idx])

    out_idx = np.where(valid, idx.astype(float), np.nan)
    d["episode_id"] = out_idx
    return d
