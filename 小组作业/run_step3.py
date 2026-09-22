# -*- coding: utf-8 -*-
"""Step3 启动脚本

用法
- 在项目根目录执行：python run_step3.py

说明
- Step3 主逻辑在 extension_analysis/pipeline.py
- Step3 配置在 configs/config_step3.py

所有注释采用中文（按小组规范）。
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
from extension_analysis.pipeline import main


if __name__ == "__main__":
    main()
