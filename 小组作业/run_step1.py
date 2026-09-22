# -*- coding: utf-8 -*-
"""Step1 入口：构造日度建模数据集并输出到 output/step1。"""

from __future__ import annotations

from data_processor.data_processor import run_data_processor

if __name__ == "__main__":
    out_path = run_data_processor()
    print(f"[OK] Step1 dataset generated: {out_path}")
