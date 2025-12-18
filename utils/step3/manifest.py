# -*- coding: utf-8 -*-
"""Step3：实验元数据（Manifest）工具。

目标：
- 每次运行都生成 run_id 与 manifest（json），记录输入依赖、关键口径开关与输出文件清单。
- 避免“跑完了但不知道用的是什么配置/什么输入文件”的情况，方便写报告与答辩。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def make_run_id(prefix: str = "step3") -> str:
    """生成可读的 run_id（本地时间戳）。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


@dataclass
class RunManifest:
    """Step3 运行清单。"""

    run_id: str
    created_at: str
    config_snapshot_path: str

    inputs: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    outputs: Dict[str, List[str]] = field(default_factory=lambda: {
        "tables": [],
        "datasets": [],
        "figures": [],
        "meta": [],
        "logs": [],
    })

    notes: List[str] = field(default_factory=list)

    def add_output(self, kind: str, path: str) -> None:
        kind = str(kind)
        if kind not in self.outputs:
            self.outputs[kind] = []
        self.outputs[kind].append(str(path))

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config_snapshot_path": self.config_snapshot_path,
            "inputs": self.inputs,
            "settings": self.settings,
            "outputs": self.outputs,
            "notes": self.notes,
        }


def collect_file_list(root_dir: str, subdirs: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """收集输出目录下的文件列表，用于写入 manifest。

    参数：
    - root_dir: output/step3 根目录
    - subdirs: 指定要收集的子目录（例如 ["tables","figures"]）；None 表示默认五类
    """

    root = Path(root_dir).expanduser().resolve()
    defaults = ["tables", "datasets", "figures", "meta", "logs"]
    subdirs = subdirs or defaults

    out: Dict[str, List[str]] = {}
    for s in subdirs:
        d = root / s
        if not d.exists():
            out[s] = []
            continue
        out[s] = [str(p) for p in sorted(d.glob("**/*")) if p.is_file()]
    return out
