# -*- coding: utf-8 -*-
"""Step3 绘图工具（matplotlib）

本模块封装：
- 中文字体设置（会过滤本机不存在的字体，避免大量 findfont warning）
- 统一保存图片（png/pdf），并在保存后释放内存

说明
- 你们要求“代码注释全部中文”，因此本模块也保持中文说明。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def setup_matplotlib_cn(font_family_candidates: List[str], base_font_size: int = 11) -> None:
    """设置 matplotlib 中文字体候选（自动过滤未安装字体）。

    参数
    - font_family_candidates: 配置里给出的字体候选列表（按优先级从高到低）
    - base_font_size: 全局默认字号
    """
    try:
        available = {f.name for f in fm.fontManager.ttflist}

        resolved: List[str] = []
        for cand in [str(x) for x in (font_family_candidates or [])]:
            if cand in available:
                resolved.append(cand)
                continue

            # 模糊匹配：例如系统字体族名可能带 Regular/SC 等变体
            cand_lower = cand.lower()
            matches = [name for name in available if cand_lower in name.lower()]
            if matches:
                resolved.append(sorted(matches)[0])

        # 去重且保持顺序
        seen = set()
        resolved = [x for x in resolved if not (x in seen or seen.add(x))]

        if not resolved:
            resolved = ["DejaVu Sans"]

        plt.rcParams["font.family"] = resolved
        plt.rcParams["font.size"] = int(base_font_size)
        plt.rcParams["axes.unicode_minus"] = False

        # 降噪：如果仍有少量字体查找日志，把 font_manager 日志级别调高
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

        logging.getLogger("step3").info(f"Matplotlib 字体候选={font_family_candidates} | 实际启用={resolved}")
    except Exception:
        # 字体设置失败时不抛错，避免影响主流程
        pass


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 220) -> None:
    """统一保存图片，并关闭 figure 释放内存。"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(p), dpi=int(dpi))
    plt.close(fig)
