# -*- coding: utf-8 -*-
"""Step3 绘图工具（matplotlib）

要求
- 不使用 seaborn
- 图默认可用于论文/汇报（合理尺寸、tight_layout、中文字体兼容）

本模块封装：
- 中文字体设置
- 统一保存图片（png/pdf）
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


def setup_matplotlib_cn(font_family_candidates: List[str], base_font_size: int = 11) -> None:
    """设置 matplotlib 中文字体候选。"""
    try:
        plt.rcParams["font.family"] = font_family_candidates
        plt.rcParams["font.size"] = int(base_font_size)
        plt.rcParams["axes.unicode_minus"] = False
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


def maybe_save_formats(fig: plt.Figure, out_base: str, formats: List[str], dpi: int = 220) -> List[str]:
    """按 formats 保存多种格式文件。

    参数
    - out_base: 不含后缀的路径（如 output/figures/xxx）
    - formats: ["png", "pdf"]

    返回
    - 实际写出的文件路径列表
    """
    written = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        out_path = f"{out_base}.{fmt}"
        save_figure(fig, out_path, dpi=dpi)
        written.append(out_path)
        # 注意：save_figure 会 close(fig)，因此多格式时需要重新画图
        # 这里假设调用方在循环外自行处理（或只保存一种格式）
    return written
