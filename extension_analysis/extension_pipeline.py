# -*- coding: utf-8 -*-
"""Step3 主流程（拓展分析 Pipeline）。

run_step3.py 会调用本模块的 run_step3(cfg) 执行完整流程。

职责：
- 扫描 Step2 predictions 输出并按 config 过滤
- 输入契约校验
- 经济层评估（含 baseline / tau / phase）+ 可选资金曲线
- VIX 分时期评估（conditional + episode）+ 可选阴影图
- 滚动重要性计算 + 可视化
- 生成 manifest
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.step3.io_contract import scan_predictions, filter_predictions, read_and_validate_predictions
from utils.step3.manifest import RunManifest, make_run_id, collect_file_list
from utils.step3.economic import run_economic_layer
from utils.step3.vix_regime import run_vix_regime
from utils.step3.rolling_importance import run_rolling_importance
from utils.step3.rolling_importance_plots import run_rolling_importance_plots


def _save_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def ensure_step3_dirs(output_dir: str) -> Dict[str, str]:
    """创建 Step3 的标准输出目录，并返回路径字典。"""
    base = Path(output_dir).expanduser().resolve()
    dirs = {
        "datasets": str(base / "datasets"),
        "figures": str(base / "figures"),
        "tables": str(base / "tables"),
        "meta": str(base / "meta"),
        "logs": str(base / "logs"),
    }
    for p in dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return dirs


def run_step3(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """执行 Step3 完整流程。

    返回：
    - 一个摘要 dict（便于在 run_step3.py 打印）
    """

    log = logging.getLogger("step3")

    step1_dataset_dir = str(cfg.get("step1_dataset_dir", "../output/step1/datasets"))
    step2_output_dir = str(cfg.get("step2_output_dir", "../output/step2"))
    output_dir = str(cfg.get("output_dir", "../output/step3"))

    dirs = ensure_step3_dirs(output_dir)

    # 1) 保存配置快照
    config_snapshot_path = str(Path(dirs["meta"]) / "config_snapshot_step3.json")
    _save_json(config_snapshot_path, cfg)

    # 2) 初始化 manifest
    run_id = make_run_id("step3")
    manifest_path = str(Path(dirs["meta"]) / "run_manifest.json")
    mf = RunManifest(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        config_snapshot_path=config_snapshot_path,
    )

    mf.inputs = {
        "step1_dataset_dir": step1_dataset_dir,
        "step2_output_dir": step2_output_dir,
    }
    mf.settings = {
        "selection": cfg.get("selection", {}),
        "economic": cfg.get("economic", {}),
        "vix_regime": cfg.get("vix_regime", {}),
        "rolling_importance": cfg.get("rolling_importance", {}),
        "rolling_importance_plots": cfg.get("rolling_importance_plots", {}),
    }

    # 3) 扫描与过滤 predictions
    sel = cfg.get("selection", {}) or {}
    scanned = scan_predictions(step2_output_dir)
    filtered = filter_predictions(
        scanned,
        models_include=sel.get("models_include", "ALL"),
        models_exclude=sel.get("models_exclude", []),
        H_include=sel.get("H_include", "ALL"),
    )

    if len(filtered) == 0:
        raise FileNotFoundError(
            f"未找到符合条件的 predictions 文件。请检查 step2_output_dir={step2_output_dir} 及 selection 配置。"
        )

    mf.inputs["predictions_files"] = [x.path for x in filtered]
    log.info(f"扫描到 predictions 文件数={len(scanned)} | 过滤后={len(filtered)}")

    # 4) 逐个读取 predictions（在最早阶段做契约校验）
    pred_frames: List[Dict[str, Any]] = []
    for it in filtered:
        df = read_and_validate_predictions(it.path)
        df["model"] = it.model
        df["H"] = it.H
        pred_frames.append({"model": it.model, "H": it.H, "path": it.path, "df": df})

    # 5) 经济层
    if bool((cfg.get("economic", {}) or {}).get("enable", True)):
        log.info("Step3-经济层：开始")
        out = run_economic_layer(pred_frames, cfg, dirs)
        mf.notes.append(f"经济层完成：生成 tables={out.get('tables', [])}")
        log.info("Step3-经济层：完成")
    else:
        log.info("Step3-经济层：已跳过（economic.enable=False）")

    # 6) VIX 分时期
    if bool((cfg.get("vix_regime", {}) or {}).get("enable", True)):
        log.info("Step3-VIX：开始")
        out = run_vix_regime(pred_frames, step1_dataset_dir, step2_output_dir, cfg, dirs)
        mf.notes.append(f"VIX 分时期完成：生成 tables={out.get('tables', [])}")
        log.info("Step3-VIX：完成")
    else:
        log.info("Step3-VIX：已跳过（vix_regime.enable=False）")

    # 7) 滚动重要性计算
    if bool((cfg.get("rolling_importance", {}) or {}).get("enable", True)):
        log.info("Step3-滚动重要性：开始")
        out = run_rolling_importance(step1_dataset_dir, step2_output_dir, cfg, dirs, filtered)
        mf.notes.append(f"滚动重要性完成：生成 tables={out.get('tables', [])}")
        log.info("Step3-滚动重要性：完成")
    else:
        log.info("Step3-滚动重要性：已跳过（rolling_importance.enable=False）")

    # 8) 滚动重要性绘图
    if bool((cfg.get("rolling_importance_plots", {}) or {}).get("enable", True)):
        log.info("Step3-重要性可视化：开始")
        out = run_rolling_importance_plots(cfg, dirs)
        mf.notes.append(f"重要性可视化完成：生成 figures={out.get('figures', [])}")
        log.info("Step3-重要性可视化：完成")
    else:
        log.info("Step3-重要性可视化：已跳过（rolling_importance_plots.enable=False）")

    # 9) 收集输出文件清单并写入 manifest
    file_list = collect_file_list(output_dir)
    for kind, paths in file_list.items():
        for p in paths:
            mf.add_output(kind, p)

    mf.save(manifest_path)
    log.info(f"已写入 manifest: {manifest_path}")

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "manifest": manifest_path,
        "predictions_count": len(filtered),
    }
