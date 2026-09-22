# -*- coding: utf-8 -*-
"""Step3：拓展分析主流水线（extension_analysis）

运行方式
- 建议在项目根目录执行：python run_step3.py

主要流程
1) 扫描 Step2 输出 predictions_*.csv，并按 config 选择 (model,H)
2) 输入契约校验：列/日期/缺失值，以及（可选）y_true 与 Step1 目标对齐
3) 经济层检验：非重叠持有期 + 相位全扫 + tau/成本/基准策略
4) VIX 分时期检验：conditional 与 episode（不打散时间顺序）
5) 滚动窗口重要性：permutation + TreeSHAP（树模型）
6) 可视化：rolling importance 折线/热力图/Jaccard
7) 生成 manifest（实验元数据）

说明
- Step3 只读取 Step1/Step2 产出文件；不修改 Step2 训练流程。
- 所有注释采用中文（按小组规范）。
"""

from __future__ import annotations

import importlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.step3.io_contract import read_predictions_csv, validate_predictions_df, check_y_true_alignment
from utils.step3.manifest_utils import (
    ensure_dirs_step3,
    save_json,
    read_json,
    build_manifest_base,
    register_outputs,
)
from utils.step3.plot_utils import setup_matplotlib_cn

from extension_analysis.economic_eval import run_economic_layer
from extension_analysis.vix_regime_eval import run_vix_regime_eval
from extension_analysis.rolling_importance import run_rolling_importance
from extension_analysis.rolling_importance_plots import run_rolling_importance_plots


def _setup_logger(level: str, log_file: str) -> None:
    """同时输出到控制台与文件。"""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _resolve(base_file: str, rel: str) -> str:
    """以 base_file 所在目录为基准解析相对路径。"""
    return str(Path(base_file).resolve().parent.joinpath(rel).resolve())


def _scan_predictions(step2_output_dir: str) -> List[Dict[str, Any]]:
    """扫描 Step2 预测文件 predictions_{model}_H{H}.csv。"""
    ds_dir = Path(step2_output_dir) / "datasets"
    if not ds_dir.exists():
        raise FileNotFoundError(f"Step2 datasets 目录不存在: {ds_dir}")

    out: List[Dict[str, Any]] = []
    for p in sorted(ds_dir.glob("predictions_*_H*.csv")):
        name = p.name
        # 尽量从文件名右侧解析 H，避免 model 名包含 '_' 造成歧义
        m = re.match(r"^predictions_(.+)_H(\d+)\.csv$", name)
        if not m:
            continue
        model = str(m.group(1))
        H = int(m.group(2))
        out.append({"model": model, "H": H, "path": str(p)})
    return out


def _apply_selection(run_cfg: Dict[str, Any], combos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 config.selection 过滤 (model,H) 组合。"""
    sel = dict(run_cfg.get("selection", {}) or {})

    mi = sel.get("models_include", "ALL")
    me = set([str(x) for x in sel.get("models_exclude", [])])
    hi = sel.get("H_include", "ALL")
    he = set([int(x) for x in sel.get("H_exclude", [])])

    def model_ok(m: str) -> bool:
        if m in me:
            return False
        if mi == "ALL" or mi is None:
            return True
        return m in set([str(x) for x in mi])

    def H_ok(h: int) -> bool:
        if h in he:
            return False
        if hi == "ALL" or hi is None:
            return True
        return h in set([int(x) for x in hi])

    out = [c for c in combos if model_ok(str(c["model"])) and H_ok(int(c["H"]))]

    # 烟雾测试：只保留少量组合，快速验证工程跑通
    smoke = dict(run_cfg.get("smoke_test", {}) or {})
    if bool(smoke.get("enable", False)):
        max_models = int(smoke.get("max_models", 2))
        seen = {}
        filtered: List[Dict[str, Any]] = []
        for c in sorted(out, key=lambda x: (str(x["model"]), int(x["H"]))):
            m = str(c["model"])
            if m not in seen and len(seen) >= max_models:
                continue
            seen.setdefault(m, 0)
            if seen[m] >= 1:
                continue
            seen[m] += 1
            filtered.append(c)
        out = filtered

    return out


def main() -> None:
    # 1) 读取 config
    cfg_mod = importlib.import_module("configs.config_step3")
    RUN: Dict[str, Any] = getattr(cfg_mod, "RUN")

    here = __file__
    step1_dataset_dir = _resolve(here, RUN["step1_dataset_dir"])
    step2_output_dir = _resolve(here, RUN["step2_output_dir"])
    out_dir = _resolve(here, RUN["output_dir"])

    # 2) 输出目录与日志
    dirs = ensure_dirs_step3(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(dirs.logs / f"step3_{run_id}.log")
    _setup_logger(RUN.get("log_level", "INFO"), log_file=log_file)
    log = logging.getLogger("step3")

    # 3) 图形字体（全局一次）
    setup_matplotlib_cn(RUN.get("cn_font_candidates", []), base_font_size=11)

    # 4) 读取 Step2 config snapshot（可选）
    step2_snapshot_path = str(Path(step2_output_dir) / "meta" / "config_snapshot.json")
    step2_snapshot: Optional[Dict[str, Any]] = None
    if bool(RUN.get("inherit_step2_snapshot", True)) and Path(step2_snapshot_path).exists():
        step2_snapshot = read_json(step2_snapshot_path)
        log.info(f"读取 Step2 config_snapshot: {step2_snapshot_path}")
    else:
        log.info("未读取 Step2 config_snapshot（将使用 Step3 默认口径或配置中的回退参数）")

    # 5) 保存 Step3 config 快照
    config_snapshot_path = str(dirs.meta / "config_snapshot.json")
    save_json(config_snapshot_path, RUN)

    # 6) 扫描并筛选 combos
    combos_all = _scan_predictions(step2_output_dir)
    combos = _apply_selection(RUN, combos_all)
    if len(combos) == 0:
        raise RuntimeError("没有任何 predictions 文件被选中；请检查 Step2 输出与 Step3 selection 配置")

    log.info(f"Step3 扫描到 predictions 文件={len(combos_all)} | 选中={len(combos)}")
    log.info(f"示例 combo: {combos[0]}")

    # 7) manifest（基础信息先写入）
    manifest = build_manifest_base(
        cfg=RUN,
        combos=combos,
        step1_dataset_dir=step1_dataset_dir,
        step2_output_dir=step2_output_dir,
        out_dir=out_dir,
        step2_snapshot_path=step2_snapshot_path if step2_snapshot else None,
        extra={"logs": {"log_file": log_file}},
    )

    # 8) 读取 predictions 并做 IO 契约校验
    pred_frames: Dict[Tuple[str, int], pd.DataFrame] = {}
    io_cfg = dict(RUN.get("io_contract", {}) or {})

    # Step1/Step2 的数据模板信息（用于 y_true 对齐校验）
    dataset_file_template = str((step2_snapshot or {}).get("dataset_file_template", "Data_D_{H}.xlsx"))
    date_col = str((step2_snapshot or {}).get("date_col", "date"))
    y_prefix = str((step2_snapshot or {}).get("y_prefix", "USD"))

    # 为了避免 Step3 强耦合 Step2 内部实现：只有在需要对齐校验时才 import
    if bool(io_cfg.get("enable", True)) and bool(io_cfg.get("check_y_true_alignment", True)):
        try:
            from utils.step2.data_loader import load_dataset  # type: ignore
        except Exception as e:
            raise ImportError("Step3 需要 utils.step2.data_loader.load_dataset 以执行 y_true 对齐校验，但导入失败") from e

    for c in combos:
        model, H, path = str(c["model"]), int(c["H"]), str(c["path"])
        df = read_predictions_csv(path)
        df = df.sort_values("date").reset_index(drop=True)

        if bool(io_cfg.get("enable", True)):
            rep = validate_predictions_df(df, path_hint=path)
            if not rep.ok:
                raise RuntimeError(f"IO 契约校验失败: {rep.message}")

            # 可选：y_true 与 Step1 标签对齐
            if bool(io_cfg.get("check_y_true_alignment", True)):
                step1 = load_dataset(
                    dataset_dir=step1_dataset_dir,
                    filename_template=dataset_file_template,
                    H=H,
                    date_col=date_col,
                    y_prefix=y_prefix,
                    feature_cols="ALL",
                    drop_cols=[],
                ).df
                y_col_step1 = f"{y_prefix}_{H}"
                ok, msg = check_y_true_alignment(
                    pred_df=df,
                    step1_df=step1,
                    y_col_step1=y_col_step1,
                    max_abs_diff=float(io_cfg.get("alignment_max_abs_diff", 1e-8)),
                    sample_n=io_cfg.get("alignment_sample_n", 200),
                    random_state=42,
                )
                log.info(f"{model}-H{H} | {msg}")
                if not ok:
                    raise RuntimeError(f"y_true 对齐校验失败: {model}-H{H} | {msg}")

        pred_frames[(model, H)] = df

    # 9) 统一输出目录字典（供各模块写文件）
    out_dirs = {
        "datasets": str(dirs.datasets),
        "tables": str(dirs.tables),
        "figures": str(dirs.figures),
        "meta": str(dirs.meta),
        "logs": str(dirs.logs),
    }

    # -------------------
    # A) 经济层
    # -------------------
    econ_out = run_economic_layer(
        run_cfg=RUN,
        combos=combos,
        pred_frames=pred_frames,
        out_dirs=out_dirs,
    )
    if econ_out:
        register_outputs(manifest, "tables", econ_out.get("tables", []))
        register_outputs(manifest, "datasets", econ_out.get("datasets", []))
        register_outputs(manifest, "figures", econ_out.get("figures", []))

    trades_long_path = econ_out.get("trades_long_path") if econ_out else None

    # -------------------
    # B) VIX regime
    # -------------------
    if trades_long_path and bool(RUN.get("vix_regime", {}).get("enable", True)):
        vix_out = run_vix_regime_eval(
            run_cfg=RUN,
            combos=combos,
            trades_long_path=trades_long_path,
            step1_dataset_dir=step1_dataset_dir,
            dataset_file_template=dataset_file_template,
            date_col=date_col,
            y_prefix=y_prefix,
            step2_snapshot=step2_snapshot,
            out_dirs=out_dirs,
        )
        if vix_out:
            register_outputs(manifest, "tables", vix_out.get("tables", []))
            register_outputs(manifest, "datasets", vix_out.get("datasets", []))
            register_outputs(manifest, "figures", vix_out.get("figures", []))

    # -------------------
    # C) Rolling importance
    # -------------------
    roll_out: Dict[str, Any] = {}
    if bool(RUN.get("rolling_importance", {}).get("enable", True)):
        roll_out = run_rolling_importance(
            run_cfg=RUN,
            combos=combos,
            step1_dataset_dir=step1_dataset_dir,
            dataset_file_template=dataset_file_template,
            date_col=date_col,
            y_prefix=y_prefix,
            step2_output_dir=step2_output_dir,
            step2_snapshot=step2_snapshot,
            out_dirs=out_dirs,
        )
        if roll_out:
            register_outputs(manifest, "tables", roll_out.get("tables", []))
            register_outputs(manifest, "datasets", roll_out.get("datasets", []))
            register_outputs(manifest, "figures", roll_out.get("figures", []))

    # -------------------
    # D) Rolling importance plots（只读 CSV，不重训）
    # -------------------
    if roll_out and bool(RUN.get("rolling_importance_plots", {}).get("enable", True)):
        plots_out = run_rolling_importance_plots(
            run_cfg=RUN,
            long_path=str(roll_out.get("long_path")),
            out_dirs=out_dirs,
        )
        if plots_out:
            register_outputs(manifest, "figures", plots_out.get("figures", []))

    # 10) 收尾：写 manifest
    manifest_path = str(dirs.meta / "run_manifest.json")
    register_outputs(manifest, "meta", [manifest_path, config_snapshot_path])
    register_outputs(manifest, "logs", [log_file])
    save_json(manifest_path, manifest)

    log.info(f"Step3 完成 | 输出目录: {out_dir}")
    log.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
