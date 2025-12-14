# 汇率收益率预测项目：数据处理流水线（日度版本）

本工程用于构造“日度版本”的建模数据集，并以 **Excel** 形式输出到 `Data/` 目录，便于后续训练 LASSO/RF/XGBoost/LSTM/Transformer 等模型。

## 核心口径（已固化在 `data_processor.py`）

1. **USD** 作为日度主时间轴（交易日集合）。
2. **shift y**：标签定义为  
   \[
   y_t =\ln S_{t+h} - \ln S_{t}
   \]
   其中 `h = forecast_horizon_days`（默认 1 个交易日）。
3. **低频变量（月度/季度）**：先施加发布滞后（默认：月滞后 1 期、季滞后 1 期），再映射到日度并 **carry-forward**。
4. **跨市场节假日缺口**：carry-forward（forward fill）。
5. 每个变量在 `config.py` 中显式指定 `transform`：`none / diff / log_diff / pct_change`。
6. **期限结构差**：仅对同时拥有 `3M` 与 `10Y` 的国家（US/UK/GER）计算 `10Y - 3M`。

## 目录结构

- `config.py`：运行参数、变量清单、transform、技术指标窗口、滞后规则等
- `data_processor.py`：主流程（读取 → 对齐 → transform → 特征工程 → 生成 y → 输出 Excel）
- `utils/`：工具函数（IO/transform/feature/check）

## 输出文件

运行后会生成：

- `Data/datasets/Data_D_X.xlsx`  
  其中 `X = forecast_horizon_days`（预测步长，交易日）。

工作簿包含固定 Sheet：

- `dataset`：最终建模主表（features + y）
- `data_dictionary`：数据字典（变量来源、频率、transform、备注）
- `missing_summary`：缺失率统计（基于未删行版本）
- `descriptive_stats`：描述统计（含 skew/kurt）
- `config_snapshot`：本次运行配置快照
- `row_filtering`：行筛选统计（删除 NaN 前后对比）

同时会在 `Data/meta/` 下保存配置快照归档。

## 运行方法

在工程根目录执行：

```bash
python data_processor.py
```

如需修改变量/transform/窗口/预测步长，请编辑 `config.py`。
