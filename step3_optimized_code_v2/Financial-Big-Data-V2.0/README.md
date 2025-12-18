# Financial-Big-Data-V2.0

本项目分为三个主要步骤：
- **Step1（data_processor）**：数据清洗、变量构造、输出可建模的数据集。
- **Step2（modeling）**：模型训练、超参数选择、输出样本外预测（OOS predictions）。
- **Step3（extension_analysis）**：在 Step2 预测结果之上做拓展分析（经济层、分时期、解释与稳定性）。

下面仅说明 Step3（extension_analysis）部分做了哪些事情，以及如何运行。

## Step3 做了哪些事情

Step3 的输入主要是 Step2 的样本外预测文件（例如 `predictions_{model}_H{H}.csv`），并在必要时读取 Step1 输出的数据集（用于 VIX、对齐校验与滚动重要性训练）。Step3 的核心产出是“可写入论文/答辩”的表格与图形。

Step3 包含四个模块（按默认流水线顺序）：

1. **输入契约校验（IO Contract）**
   - 检查预测文件是否包含必要列（date/y_true/y_pred）。
   - 检查 date 是否可解析、是否严格升序、是否无重复。
   - （可选）对齐检查：抽样比对 predictions 里的 y_true 是否与 Step1 的 `USD_{H}` 完全一致（容忍极小浮点误差）。

2. **经济层检验（Economic Layer）**
   - 将预测信号转为仓位 `pos_t`（允许做空时取值 -1/0/1；不允许做空时取值 0/1）。
   - 引入交易成本：`cost_t = tc * |pos_t - pos_{t-1}|`（反手会产生 2 倍成本）。
   - 支持“非重叠持有期”评估与“相位全扫”（phase=0..H-1），避免重叠收益导致的统计问题。
   - 输出每个相位的绩效指标表，以及跨相位汇总表；并可输出资金曲线图。

3. **VIX 分时期检验（Regime / Subperiod）**
   - conditional：按 VIX 状态（low/mid/high）对交易样本分组做条件绩效统计（不会打散时间顺序）。
   - episode：把 VIX 状态合并为连续时段，并在每个 episode 内计算绩效（便于叙事与可视化）。
   - 阈值避免前视：默认 `train_only`（用训练/验证期阈值固定后用于测试期），也可选 `expanding`（历史分位数并 shift 1 天）。
   - 支持去抖：hysteresis（双阈值）+ min_spell_days（最短持续期）。

4. **滚动窗口重要性稳定性（Rolling Importance）+ 可视化**
   - 在测试期上按窗口滑动，对每个窗口重新训练模型并计算重要性。
   - importance 方法：permutation importance（通用）与 TreeSHAP（仅 XGBoost/LightGBM）。
   - 自动生成可视化：折线图（重要性随时间）、热力图（特征×窗口）、Jaccard（相邻窗口 top-k 稳定性）。

## Step3 运行方式

在项目根目录执行：

```bash
python run_step3.py
```

配置文件：`configs/config_step3.py`
- 你们已在配置中限定 Step3 只评估 6 个机器学习模型：`Lasso, Ridge, ElasticNet, RandomForest, XGBoost, LightGBM`。
- 所有输出默认写入：`output/step3/{tables,datasets,figures,meta,logs}`。

## Step3 输出说明（默认）

- `output/step3/tables/`
  - `econ_metrics_by_phase.csv`：每个 (model,H,tau,phase,strategy) 一行
  - `econ_metrics_phase_summary.csv`：对相位做汇总（mean/median/min/max）
  - `econ_by_vix_regime_conditional.csv`：VIX 条件分组统计
  - `econ_by_vix_episode.csv`：episode 级别统计
  - `rolling_importance_long.csv`：滚动重要性长表
  - `rolling_importance_mean.csv`：滚动重要性均值表

- `output/step3/datasets/`
  - `econ_trades_long.csv`：交易级别长表（供分时期与可视化复用）
  - `vix_regime_H{H}.csv`：各 H 的 VIX 状态序列
  - `vix_episodes_H{H}.csv`：各 H 的 VIX episode 表

- `output/step3/figures/`
  - `equity_*.png`：资金曲线
  - `rolling_importance_*.(png/pdf)`：重要性稳定性图

- `output/step3/meta/`
  - `config_snapshot.json`：Step3 配置快照
  - `run_manifest.json`：运行元数据（输入/输出/版本信息）

