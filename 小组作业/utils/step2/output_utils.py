# -*- coding: utf-8 -*-
"""Output helpers for Step2."""
from __future__ import annotations

import os
from typing import Dict, Any
import pandas as pd

def ensure_dirs(out_root: str) -> Dict[str, str]:
    paths = {
        "root": out_root,
        "tables": os.path.join(out_root, "tables"),
        "predictions": os.path.join(out_root, "predictions"),
        "importance": os.path.join(out_root, "importance"),
        "meta": os.path.join(out_root, "meta"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def save_config_snapshot(cfg: Dict[str, Any], out_path: str) -> None:
    rows = []
    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                rows.append((key, v))
    _flatten(cfg)
    df = pd.DataFrame(rows, columns=["key", "value"])
    df.to_excel(out_path, index=False)
