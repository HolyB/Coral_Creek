# Qlib 因子/策略挖掘使用指南

## 1) 环境准备

```bash
pip install pyqlib
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us --interval 1d
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d
```

## 2) 一键运行挖掘

```bash
cd versions/v3
python scripts/run_qlib_mining.py --market US --segment ALL --days 730
```

按大/中/小市值批量挖掘并输出对比：

```bash
cd versions/v3
python scripts/run_qlib_mining.py --market US --run-segment-batch --days 730
```

常用参数：
- `--segment LARGE|MID|SMALL|ALL`：按市值分层测试（US）。
- `--symbols AAPL,MSFT,NVDA`：自定义股票池（覆盖 segment）。
- `--topk-grid 5,8,10,15`：策略持仓数网格。
- `--drop-grid 1,2,3`：策略换仓数网格。
- `--benchmark SPY`：自定义基准。

## 3) 与训练联动

```bash
cd versions/v3
python scripts/train_qlib_model.py --market US --symbols TECH_TOP20 --run-mining
```

## 4) 输出文件

默认输出目录：`versions/v3/ml/saved_models/qlib_us/`（或 `qlib_cn/`）

- `factor_mining_latest.csv`: 因子 IC/IR 排名
- `strategy_mining_latest.csv`: TopK/Dropout 策略排名
- `qlib_mining_summary_latest.json`: 汇总结果（最佳因子、最佳策略）
- `segment_strategy_compare_latest.csv`: 大/中/小市值最佳策略对比（批量模式）

## 5) Streamlit 页面

新增页面：`pages/qlib_mining.py`

- 展示 Qlib 安装与数据状态
- 展示最新因子/策略排名与图表
- 展示市值分层策略对比图
- 支持页面内触发挖掘命令
