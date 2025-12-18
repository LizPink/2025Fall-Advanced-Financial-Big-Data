# Step3：拓展分析（extension_analysis）

Step3 的定位：在不改动 Step2 主建模流程的前提下，直接复用 Step2 的 OOS 预测结果（predictions_*.csv），开展经济含义检验、分时期检验与特征稳定性分析。

## 运行

在项目根目录执行：

```bash
python run_step3.py
```

## 输入依赖

- Step1 产出：`output/step1/datasets/Data_D_{H}.xlsx`（用于读取 VIX 与日期对齐校验）
- Step2 产出：
  - `output/step2/datasets/predictions_{model}_H{H}.csv`
  - `output/step2/tables/best_params.csv`（滚动重要性训练时复用 Step2 最优超参）
  - `output/step2/meta/config_snapshot.json`（可选，Step3 默认继承 Step2 的 test split/backtest 设置）

## 输出结构（output/step3）

- `datasets/`
  - `econ_trades_long.csv`：交易级别长表（每行对应一笔 H 日交易，包含 pos、cost、r_net 等）
  - `vix_regime_H{H}.csv`：VIX 与 regime（low/mid/high）时间序列
  - `vix_episodes_H{H}.csv`：连续 episode 列表（start/end/length/regime）
- `tables/`
  - `econ_metrics_by_phase.csv`：每个相位的经济指标
  - `econ_metrics_phase_summary.csv`：相位 across-phase 汇总（mean/median/min/max）
  - `econ_by_vix_regime_conditional.csv`：按 VIX regime 条件分组的指标
  - `econ_by_vix_episode.csv`：按 episode 统计的指标
  - `rolling_importance_long.csv`：滚动重要性长表
  - `rolling_importance_mean.csv`：滚动重要性均值表
- `figures/`
  - `equity_*.png`：资金曲线（可选）
  - `equity_vixshade_*.png`：资金曲线 + 高 VIX 阴影（可选）
  - `rolling_importance_*.(png/pdf)`：折线/热力图/Jaccard（可选）
- `meta/`
  - `config_snapshot.json`：Step3 配置快照
  - `run_manifest.json`：实验元数据（输入路径/输出文件列表/依赖版本等）
- `logs/`
  - `step3.log`

## 配置说明（configs/config_step3.py）

- `selection`：选择要评估的模型与持有期 H
- `economic`：allow_short、tau_list、tc_bps、baselines、phase_sweep 等
- `vix_regime`：阈值模式（quantile/fixed）、阈值来源（train_only/expanding）、hysteresis/min_spell_days
- `rolling_importance`：window/step、是否启用 TreeSHAP、only_models 等
- `rolling_importance_plots`：top-k、归一化、平滑、输出格式

