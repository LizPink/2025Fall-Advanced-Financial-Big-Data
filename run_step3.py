# -*- coding: utf-8 -*-
"""Step3 入口脚本。

运行方式：
    python run_step3.py
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from extension_analysis.extension_pipeline import run_step3


def _setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    cfg_mod = importlib.import_module("config.config_step3")
    RUN = getattr(cfg_mod, "RUN")

    _setup_logger(RUN.get("log_level", "INFO"))

    # 运行
    run_step3(RUN)


if __name__ == "__main__":
    main()
