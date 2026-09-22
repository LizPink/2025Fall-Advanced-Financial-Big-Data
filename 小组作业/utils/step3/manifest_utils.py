# -*- coding: utf-8 -*-
"""Step3 实验元数据（Manifest）工具

目的
- 记录 Step3 运行时的关键配置、输入依赖、输出文件、以及环境版本信息
- 便于复现实验与答辩追溯

约定
- Manifest 存放在 output/step3/meta/run_manifest.json
- Config 快照存放在 output/step3/meta/config_snapshot.json

所有注释采用中文（按小组规范）。
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Step3Dirs:
    base: Path
    datasets: Path
    figures: Path
    tables: Path
    meta: Path
    logs: Path


def ensure_dirs_step3(out_dir: str) -> Step3Dirs:
    """创建 Step3 输出目录结构。"""
    base = Path(out_dir).expanduser().resolve()
    datasets = base / "datasets"
    figures = base / "figures"
    tables = base / "tables"
    meta = base / "meta"
    logs = base / "logs"
    for p in (datasets, figures, tables, meta, logs):
        p.mkdir(parents=True, exist_ok=True)
    return Step3Dirs(base=base, datasets=datasets, figures=figures, tables=tables, meta=meta, logs=logs)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON 不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_versions() -> Dict[str, str]:
    """收集关键依赖版本（尽量不报错）。"""
    versions: Dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    def _try_pkg(name: str) -> None:
        try:
            mod = __import__(name)
            v = getattr(mod, "__version__", "(unknown)")
            versions[name] = str(v)
        except Exception:
            versions[name] = "(not installed)"

    for pkg in [
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "torch",
        "xgboost",
        "lightgbm",
    ]:
        _try_pkg(pkg)

    return versions


def build_manifest_base(
    cfg: Dict[str, Any],
    combos: List[Dict[str, Any]],
    step1_dataset_dir: str,
    step2_output_dir: str,
    out_dir: str,
    step2_snapshot_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建 Step3 manifest 的基础结构。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest: Dict[str, Any] = {
        "step": 3,
        "created_at": now,
        "paths": {
            "step1_dataset_dir": str(step1_dataset_dir),
            "step2_output_dir": str(step2_output_dir),
            "output_dir": str(out_dir),
            "step2_config_snapshot": str(step2_snapshot_path) if step2_snapshot_path else None,
        },
        "selection": {
            "n_combos": int(len(combos)),
            "combos": [{"model": c.get("model"), "H": int(c.get("H"))} for c in combos],
        },
        "config": {
            "economic": cfg.get("economic", {}),
            "vix_regime": cfg.get("vix_regime", {}),
            "rolling_importance": cfg.get("rolling_importance", {}),
            "rolling_importance_plots": cfg.get("rolling_importance_plots", {}),
            "io_contract": cfg.get("io_contract", {}),
        },
        "outputs": {
            "tables": [],
            "datasets": [],
            "figures": [],
            "meta": [],
            "logs": [],
        },
    }

    if extra:
        manifest.update(extra)

    if bool(cfg.get("manifest", {}).get("capture_versions", True)):
        manifest["versions"] = collect_versions()

    return manifest


def register_outputs(manifest: Dict[str, Any], category: str, paths: List[str]) -> None:
    """把输出文件路径注册到 manifest（去重、保持顺序）。"""
    if "outputs" not in manifest:
        manifest["outputs"] = {}
    if category not in manifest["outputs"]:
        manifest["outputs"][category] = []

    exist = set(manifest["outputs"][category])
    for p in paths:
        p = str(p)
        if p not in exist:
            manifest["outputs"][category].append(p)
            exist.add(p)
