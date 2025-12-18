# utils/step3/plot_utils.py
# -*- coding: utf-8 -*-
"""Step3 绘图工具（matplotlib）

本模块封装：
- 中文字体设置（会过滤本机不存在的字体，避免 findfont warning）
- 统一保存图片（png/pdf）
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def setup_matplotlib_cn(font_family_candidates: List[str], base_font_size: int = 11) -> None:
    """设置 matplotlib 中文字体候选（自动过滤未安装字体，避免 findfont warning）。

    参数
    - font_family_candidates: 配置里给出的字体候选列表（按优先级从高到低）
    - base_font_size: 全局默认字号
    """
    try:
        # 1) 收集本机可用字体族名称
        available = {f.name for f in fm.fontManager.ttflist}

        # 2) 过滤：仅保留“确实存在”的字体；若没有精确命中，做一次模糊匹配
        resolved: List[str] = []
        for cand in [str(x) for x in (font_family_candidates or [])]:
            if cand in available:
                resolved.append(cand)
                continue

            # 模糊匹配：例如某些系统字体族名可能带有 Regular/SC 等变体
            cand_lower = cand.lower()
            matches = [name for name in available if cand_lower in name.lower()]
            if matches:
                # 取一个稳定的匹配（排序后取第一个）
                resolved.append(sorted(matches)[0])

        # 去重且保持顺序
        seen = set()
        resolved = [x for x in resolved if not (x in seen or seen.add(x))]

        # 3) 如果一个都匹配不到，至少保证 matplotlib 不报错（中文可能会变方块）
        if not resolved:
            resolved = ["DejaVu Sans"]

        # 4) 应用 rcParams
        plt.rcParams["font.family"] = resolved
        plt.rcParams["font.size"] = int(base_font_size)
        plt.rcParams["axes.unicode_minus"] = False

        # 5) 降噪：如果仍有少量字体查找日志，把 font_manager 日志级别调高
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


def maybe_save_formats(fig: plt.Figure, out_base: str, formats: List[str], dpi: int = 220) -> List[str]:
    """按 formats 保存多种格式文件。"""
    written = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        out_path = f"{out_base}.{fmt}"
        save_figure(fig, out_path, dpi=dpi)
        written.append(out_path)
        # 注意：save_figure 会 close(fig)，因此多格式时需要调用方重新生成 fig
    return written
