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
