# 作业：高级大数据与金融 — 汇率收益率预测（Step1/Step2/Step3）

本项目包含三个阶段：

- **Step1（data_processor）**：数据清洗、变量构造、生成建模数据集（例如 `output/step1/datasets/Data_D_{H}.xlsx`）。
- **Step2（modeling）**：模型调参 + walk-forward 预测 + 评估 + 解释，输出预测文件与最优超参数（例如 `output/step2/datasets/predictions_{model}_H{H}.csv`、`best_params.csv`、`metrics.csv`）。
- **Step3（extension_analysis）**：拓展分析，围绕 Step2 的预测结果进行经济层检验、分时期（VIX）检验，以及滚动重要性稳定性分析，并生成可视化结果（`output/step3/`）。

---

## Step3：拓展分析做了什么

Step3 的代码位于：`utils/step3/`，入口脚本为：`run_step3.py`，配置文件为：`config/config_step3.py`。

Step3 主要包含以下工作：

### 1. 经济层检验（Economic Layer）

把 Step2 输出的 `y_pred`（预测）与 `y_true`（H 日持有期对数收益标签）转化为交易策略收益序列，并输出绩效指标。

支持的扩展与稳健性：

- **做空开关**：`allow_short=True/False`
- **建/平仓阈值**：`tau_list=[...]`（作为敏感性分析，不在 test 上挑最优）
- **基准策略**：`cash / always_long / always_short`
- **非重叠持有期评估**：默认启用（每 H 天再平衡一次），并支持 **phase sweep**（对每个相位分别计算经济指标，避免“相位碰巧更好”）
- **交易成本**：按仓位变化计费（bps）

输出：
- `output/step3/tables/econ_metrics.csv`
- `output/step3/tables/econ_metrics_by_phase.csv`
- `output/step3/tables/econ_metrics_phase_summary.csv`
- （可选）资金曲线与收益序列文件

### 2. VIX 分时期检验（Regime / Subperiod）

用 VIX 划分市场风险状态，检验模型在不同风险状态下的经济表现是否存在系统差异。

两种互补口径：

- **conditional**：不改变时间顺序，只做“条件绩效”分组统计（High/Mid/Low VIX）。
- **episode**：把 regime 标签合并成连续区间（episode），用于叙事与可视化（可画资金曲线 + 高 VIX 阴影）。

关键设置：
- 阈值来源默认 `train_only`（仅训练/验证期估阈值，避免前视），并提供 `expanding` 可选。
- 支持 hysteresis（进入/退出双阈值）与 `min_spell_days` 平滑，减少 regime 抖动。

输出：
- `output/step3/tables/econ_by_vix_regime_conditional.csv`
- `output/step3/datasets/vix_episodes_*.csv`
- （可选）`equity_with_vix_shading_*.png`

### 3. 滚动窗口重要性稳定性（Rolling Feature Importance）

回答“模型在不同时间段依赖的关键特征是否稳定”。

做法：
- 在测试期上按 `window/step` 生成滚动窗口。
- 每个窗口使用“窗口起点之前可得数据”训练模型，再在窗口内计算重要性。

支持：
- **permutation importance**（通用）
- **TreeSHAP（无 shap 依赖）**（仅 XGBoost/LightGBM，使用 pred_contrib(s)）

并生成三类可视化：
- Top-K 特征重要性随时间折线图
- Top-K × 时间 热力图
- 相邻窗口 Top-K 集合 Jaccard 稳定性图

输出：
- `output/step3/tables/rolling_importance_long.csv`
- `output/step3/tables/rolling_importance_mean.csv`
- `output/step3/figures/imp_lines_*.png` / `imp_heatmap_*.png` / `imp_jaccard_*.png`

### 4. 实验元数据（Manifest）

每次运行 Step3 都会生成：
- `output/step3/meta/config_snapshot_step3.json`
- `output/step3/meta/run_manifest.json`

用于记录输入依赖、关键口径参数与输出文件清单，保证可复现。

---

## 如何运行 Step3

1. 确保 Step1 与 Step2 已运行，并且 Step2 的 predictions / best_params 等文件已生成。
2. 修改 `config/config_step3.py`（选择模型、H、tau、VIX 阈值来源、滚动窗口参数等）。
3. 执行：

```bash
python run_step3.py
```

结果将输出到：`output/step3/`。
