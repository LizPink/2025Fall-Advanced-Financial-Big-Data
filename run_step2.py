# -*- coding: utf-8 -*-
"""Step2 入口：模型调参与 OOS 预测评估（输出到 output/step2）。"""

from __future__ import annotations
from modeling.modeling import main
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")


if __name__ == "__main__":
    main()
