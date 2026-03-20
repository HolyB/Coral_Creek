# Coral Creek V3 - ML量化交易系统

> 从量化工具箱到专业交易系统的进化

## 🎯 V3 定位

| 版本 | 定位 | 核心能力 |
|------|------|----------|
| V2 | 量化工具箱 | 扫描、可视化、回测、模拟交易 |
| **V3** | **ML量化系统** | 特征工程、模型训练、风险管理、实盘预备 |

## 📁 目录结构

```
versions/v3/
├── app.py                  # 精简版 Streamlit UI (6 Tabs)
├── data_fetcher.py         # 数据获取 (美股/A股/港股)
├── indicator_utils.py      # 技术指标计算
├── scanner.py              # 扫描引擎
├── chart_utils.py          # 图表工具
│
├── ml/                     # 🤖 机器学习模块
│   ├── features/           # 特征工程
│   │   ├── technical.py    # 100+ 技术因子
│   │   ├── fundamental.py  # 基本面因子 (TODO)
│   │   └── alternative.py  # 另类数据因子 (TODO)
│   ├── models/             # 模型
│   │   ├── xgboost_ranker.py
│   │   ├── lightgbm_model.py
│   │   └── nn_models.py
│   ├── pipeline/           # ML Pipeline
│   │   ├── train.py
│   │   ├── backtest.py
│   │   └── inference.py
│   └── monitoring/         # 模型监控
│       ├── drift_detector.py
│       └── performance_tracker.py
│
├── risk/                   # 🛡️ 风险管理模块
│   ├── risk_metrics.py     # VaR/CVaR/夏普/回撤
│   ├── position_sizer.py   # 仓位管理/凯利/风险预算
│   └── correlation.py      # 相关性监控
│
├── execution/              # 🔌 执行层 (实盘预备)
│   ├── broker_api.py       # 券商接口抽象
│   ├── order_manager.py    # 订单管理
│   └── tca.py              # 交易成本分析
│
├── backtest/               # 回测引擎
├── services/               # 业务服务
└── db/                     # 数据库
```

## 🖥️ UI 架构 (6 Tabs)

```
┌─────────────────────────────────────────────────────────────┐
│  Coral Creek V3                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📊 每日扫描      核心扫描页面，支持美股/A股/港股             │
│                                                              │
│  🔍 个股查询      单股详情、K线图、技术分析                   │
│                                                              │
│  📈 信号中心      信号追踪 + 验证 + 历史复盘 (合并)          │
│                                                              │
│  💼 组合管理      持仓 + 风控仪表盘 + 模拟交易 (合并)        │
│                                                              │
│  🧪 策略实验室    回测 + 参数优化 + 因子研究 (合并)          │
│                                                              │
│  🤖 AI 中心       AI决策 + ML模型 + 博主追踪 (合并)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🛡️ 风控模块

### 风险指标
- **VaR/CVaR**: 95%/99% 置信度下的最大损失估计
- **最大回撤**: 历史最大回撤及发生时间
- **波动率**: 滚动波动率监控
- **Sharpe/Sortino/Calmar**: 风险调整后收益

### 仓位管理
- **固定比例法**: 基于止损距离计算仓位
- **凯利公式**: 最优仓位计算
- **波动率目标**: 动态调整仓位使波动率接近目标
- **风险预算**: 总风险额度分配

### 组合风险
- **相关性矩阵**: 持仓相关性监控
- **集中度风险**: 单股/行业集中度告警
- **分散化比率**: 分散化效果评估

## 🤖 ML 模块

### 特征工程 (100+ 因子)
- **趋势类**: MA/EMA/MACD 及其衍生
- **动量类**: 收益率/RSI/KDJ
- **波动率**: 历史波动率/ATR/布林带
- **成交量**: 量价关系/OBV
- **形态类**: K线形态/新高新低

### 模型 (TODO)
- XGBoost/LightGBM 排序模型
- LSTM 时序预测
- Transformer 注意力模型

### 监控 (TODO)
- 特征漂移检测
- 模型性能追踪
- 自动再训练触发

## 🚀 快速开始

```bash
cd versions/v3
pip install -r requirements.txt
streamlit run app.py --server.port 8504
```

## ⚙️ 自动化工作流 (GitHub Actions)
项目重度依赖 GitHub Actions 来实现云端自动化数据获取、计算和模型训练，释放本地算力。

核心工作流清单 (`.github/workflows/`)：

### 1. 核心日常扫描 (Daily Scans)
* **`daily_scan.yml`** (美股日常扫描): 周一到周五运行。触发全量美股扫描，识别特征信号写入数据库。最近升级：新增自动触发 **盘后信号追踪(candidate_tracking)** 及 **全量预计算(precompute)**，实现前端秒速加载。
* **`daily_scan_cn.yml`** (A股日常扫描): 在北京时间的盘前、盘中和盘后自动执行，逻辑同美股，涵盖扫描及后续全量预计算，自动将数据库(包含 `precomputed.db`)推送到云端。
* **`baseline_scan_us.yml` / `baseline_scan_cn.yml`**: 基准大盘及特征扫描辅助。

### 2. 回填与回溯测试 (Backfill & Tracking)
* **`backfill.yml`** (手动回填工具): 手动触发，支持指定日期范围、市场 (US/CN) 进行历史数据回填。完美解决 A 股 5000 只股票等历史海量 K 线带来的本地机器卡死或 API 限流问题。**自带候选池重建和预计算步骤**。
* **`picks_tracker.yml`** (每日推荐追踪): 每日收盘后统计、预估和追踪系统金股预测收益，使用最新的重试/Rebase 机制防止冲突写入。
* **`rescan_v3.yml`**: 针对历史信号和因子进行 V3 引擎的集中重算更新。

### 3. 数据与智能洞察 (Intelligence & ML)
* **`ml_training.yml`** (ML模型训练): 每周自动拉取最新的股票 K 线并执行 XGBoost 及排序类模型的定期增量训练。
* **`daily_signal_report.yml`** (每日信号追踪报告): 生成并推送核心策略当天的表现。
* **`realtime_monitor.yml`**: 实时行情监控预警。

---

## 📋 V3 开发路线图

### Phase 1: 基础架构 ✅
- [x] 目录结构重组
- [x] 风险指标模块
- [x] 仓位管理模块
- [x] 技术因子库

### Phase 2: 风控仪表盘
- [ ] 风控 Dashboard UI
- [ ] 组合相关性热力图
- [ ] 回撤曲线可视化
- [ ] 风险预算管理

### Phase 3: ML Pipeline
- [ ] 特征工程流水线
- [ ] XGBoost 排序模型
- [ ] 回测集成
- [ ] 在线推理

### Phase 4: 实盘预备
- [ ] 券商 API 抽象
- [ ] 模拟撮合引擎
- [ ] 交易成本分析
- [ ] 实盘监控
