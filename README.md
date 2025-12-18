# Financial-Big-Data-V2.0：汇率收益率预测（Step1 数据处理）

本工程用于构造“日度版本”的建模数据集，并以 **Excel** 形式输出到 `output/step1/`，为后续 Step2（Ridge/LASSO/ElasticNet/RF/XGB/LGBM/MLP/LSTM/Transformer）建模与回测做准备。

## Step1 核心口径（已固化于代码与配置）

1. **USD** 作为日度主时间轴（交易日集合）。
2. **标签（持有期对数收益）**：
   \[
   y_t = \ln S_{t+h} - \ln S_t
   \]
   其中 `h = forecast_horizon_days`（交易日持有期）。
3. **低频变量（月度/季度）**：先施加发布滞后（默认：月滞后 1 期、季滞后 1 期），再映射到日度并 carry-forward。
4. **跨市场节假日缺口**：carry-forward（forward fill）。
5. 每个变量在 `configs/config_step1.py` 中显式指定 `transform`：`none / diff / log_diff / pct_change`。
6. **期限结构差**：仅对同时拥有 `3M` 与 `10Y` 的国家（US/UK/GER）计算 `10Y - 3M`。
7. **标签可得性护栏（可开关）**：`enforce_label_availability=True` 时，将强制检查标签缺失仅允许出现在尾部且应连续，避免因样本截止日过近导致标签不可构造被静默 drop。

## 目录结构（建议保持稳定）

- `configs/`
  - `config_step1.py`：Step1 运行参数、变量清单、transform、技术指标窗口、滞后规则、绘图配置等
  - `config_step2.py`：Step2（建模与回测）配置占位
- `data_processor/`
  - `data_processor.py`：Step1 主流程（读取 → 对齐 → transform → 特征工程 → 生成 y → 输出 Excel）
- `modeling/`
  - `modeling.py`：Step2（建模与回测）入口占位
- `raw_data/`：原始 Excel 数据（日度 + 低频）
- `output/step1/`：Step1 所有产出
  - `datasets/` `figures/` `tables/` `meta/`
- `utils/`
  - `common/`：通用 IO 工具
  - `step1/`：Step1 专用工具（transform/feature/check/report）
  - `step2/`：Step2 专用工具（占位）

## Step1 输出文件

运行 Step1 后生成：

- `output/step1/datasets/Data_D_<h>.xlsx`（其中 `<h> = forecast_horizon_days`）

工作簿固定包含：
- `dataset`：最终建模主表（features + y）
- `data_dictionary`：数据字典（变量来源、频率、transform、备注）
- `missing_summary`：缺失率统计（基于未删行版本）
- `descriptive_stats`：描述统计（含 skew/kurt）
- `config_snapshot`：本次运行配置快照
- `row_filtering`：行筛选统计（删除 NaN 前后对比）

同时在 `output/step1/meta/` 下保存 `manifest` 与配置归档。

## 运行方法

在工程根目录执行：

```bash
python run_step1.py
```

如需修改变量/transform/窗口/预测持有期，请编辑：`configs/config_step1.py`。


## Step2：建模、调参与回测（Walk-forward OOS）

Step2 读取 Step1 产出的 `output/step1/datasets/Data_D_<H>.xlsx`（其中 `D` 仅表示日度频率标签；`H` 表示预测期限/持有期），对每个 `H in RUN['H_list']` 执行：

1. `train_val/test` 时间顺序切分；
2. `train_val` 内时间序列 CV（expanding/rolling）进行网格调参；
3. 使用最优超参在测试期做 walk-forward 预测（expanding/rolling，可配置 `refit_every`）；
4. 输出预测、指标汇总、图表、特征重要性（Permutation Importance 统一口径；XGB/LGBM 额外输出 TreeSHAP 贡献度）。

### Step2 运行

在工程根目录执行：

```bash
python run_step2.py
```

### Step2 主要配置

编辑：`configs/config_step2.py`。建议优先关注：

- `H_list`：预测期限列表
- `test_mode/test_ratio`：测试集定义
- `cv_mode` 与 `cv_*`：调参时的时间序列 CV
- `backtest_mode/backtest_train_window/refit_every`：测试期 walk-forward 预测策略
- `models` 与 `param_grids`：模型开关与超参网格
- `explain`：Permutation Importance 与 TreeSHAP（XGB/LGBM）

### Step2 产出目录

- `output/step2/tables/`：`metrics.csv`, `best_params.csv`, `importance_*.csv`
- `output/step2/datasets/`：每个模型/期限的预测明细 `predictions_<model>_H<H>.csv`
- `output/step2/figures/`：预测对比、残差分布、重要性条形图等
- `output/step2/meta/`：配置快照、调参历史

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
