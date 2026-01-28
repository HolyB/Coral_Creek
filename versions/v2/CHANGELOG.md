# Coral Creek V2.x - Changelog

## 📱 V2.3 (2026-01-27) - 盘中实时监控

### 核心新功能

#### 📱 持仓监控预警
- **止损预警**: 亏损达 -7% 时紧急通知
- **止盈预警**: 盈利达 +15% 时提醒锁定利润
- **日涨预警**: 当日涨幅 >5% 关注
- **日跌预警**: 当日跌幅 >5% 紧急关注

#### ⏰ 自动化触发
- **GitHub Actions**: 美股盘中每30分钟自动检查
- **Telegram 通知**: 预警直接推送到手机

### 新增文件
```
scripts/intraday_monitor.py        # 监控脚本
.github/workflows/intraday_monitor.yml  # 定时任务
```

---

## 🤖 V2.2 (2026-01-27) - AI 决策仪表盘 + 实时新闻

### 核心新功能

#### 1. 📊 AI 决策仪表盘
- **醒目结论卡片**: 买入/卖出/观望 颜色编码，一目了然
- **狙击价位**: 显示建议买入价、止损价、目标价
- **交易检查清单** (✅/⚠️/❌):
  - BLUE 信号（超卖/观望/弱势）
  - 均线排列（多头/空头）
  - 乖离率（严禁追高 >5%）
  - 量价配合
  - 趋势强度 (RSI)
  - 舆情风控（利好/利空/风险）
- **持仓建议**: 分「空仓者」和「持仓者」给出不同操作建议

#### 2. 📰 实时新闻集成
- **Google News RSS**: 免费、稳定、无需 API Key
- **多语言支持**: 中文查询 A股新闻，英文查询美股新闻
- **舆情风控**: AI 结合新闻分析（减持、业绩雷、利好等）

#### 3. 🔧 Bug 修复
- 修复 `Unknown format code 'f'` 格式化错误
- 增强 `safe_float()` 处理逗号格式数字
- 优化 Gemini API 模型切换（2.0-flash → 2.5-flash）

### 新增/修改文件
```
versions/v2/
├── services/
│   └── search_service.py    # [NEW] Google News 搜索服务
├── ml/
│   └── llm_intelligence.py  # [MOD] 新增 generate_decision_dashboard()
└── app.py                   # [MOD] AI 仪表盘 UI 渲染
```

### 参考项目
- 参考了 `daily_stock_analysis` 项目的 AI 决策格式

---

## 🆕 V2.1 (2026-01-07) - 数据库存储与历史回溯

### 核心升级
- **🗄️ SQLite 数据库存储**: 扫描结果不再依赖 CSV 文件，改为存入 SQLite 数据库
  - 支持按日期查询历史扫描结果
  - 自动去重 (UPSERT)
  - 毫秒级查询性能

- **📅 日期选择器**: 前端新增日期选择功能
  - 可查看任意已扫描日期的数据
  - 显示数据库统计信息

- **📜 批量回填脚本**: 支持回填历史数据
  - `backfill.py`: 自动检测缺失日期并回填
  - 支持 `--dry-run` 预览模式

- **🔧 扫描服务重构**: 
  - `scan_service.py`: 支持指定任意日期扫描
  - `run_daily_scan.py`: 每日定时扫描脚本

### 新增文件
```
versions/v2/
├── db/
│   ├── database.py           # 数据库操作
│   └── coral_creek.db        # SQLite 数据库
├── services/
│   └── scan_service.py       # 扫描服务
└── scripts/
    ├── run_daily_scan.py     # 每日扫描
    └── backfill.py           # 批量回填
```

### 使用方法
```bash
# 每日扫描
python scripts/run_daily_scan.py

# 指定日期扫描
python services/scan_service.py --date 2026-01-05

# 批量回填
python scripts/backfill.py --start 2025-12-01 --end 2026-01-07
```

---

## 🚀 V2.0 (2026-01-05) - 自适应扫描引擎

### 1. 🌊 自适应扫描引擎 (Adaptive Scanner)
- **动态阈值**: 不再使用死板的 `BLUE > 100`。系统根据个股的**波动率 (Volatility)** 和 **趋势强度 (ADX)** 自动调整买入阈值。
    - 稳健蓝筹 (Low Vol): 阈值降至 **60-80**。
    - 妖股/高波股 (High Vol): 阈值升至 **110**。
    - 强趋势股 (Strong Trend): 只要趋势确立，阈值自动放宽。
- **波浪形态识别**: 引入 `ZigZag` + `MACD` 算法，自动识别当前处于 **主升浪 (Wave 3)** 还是 **底部反转 (Wave 1)**。

### 2. 🛡️ 专业风控体系 (Risk Management)
- **ATR 动态止损**: 止损位不再是拍脑袋的百分比，而是基于 ATR (平均真实波幅) 的倍数。
    - 稳健股: 1.8x ATR
    - 妖股: 3.5x ATR
- **仓位管理**: 基于 `$1000` (或自定义) 单笔风险敞口，自动计算建议买入股数。**波动越大的股票，建议仓位越小**。

### 3. 📊 白盒化看板 (Transparent Dashboard)
- **透明度报告**: 每一只入选股票都有详细的得分卡 (Score Card)，解释为何入选。
    - 拆解得分: 信号分 + 趋势分 + 筹码分。
    - 策略解释: 明确告知触发了 "趋势策略" 还是 "共振策略"。
- **增强图表**:
    - 🔴 **红色止损线**: 直观显示风控位置。
    - 📊 **智能筹码峰**: 自动标注 **POC (最长筹码峰)**，并用不同颜色区分获利盘(金)和套牢盘(蓝)。
    - 📅 **动态筹码分布**: 拖动日期滑块查看历史任意时点的筹码分布

### 4. 🏗️ 架构升级
- **多页面架构**: 将 "每日扫描" 和 "策略回测" 整合在一个应用中，无缝切换。
- **并发加速**: 扫描引擎支持多线程并发，1.2万只股票全量扫描仅需 15-20 分钟。
- **SMA 算法修正**: 修复了 SMA 计算错误 (原为简单移动平均，改为通达信加权移动平均)

---

## 📝 文件结构 (V2.1)

| 文件名 | 描述 |
| :--- | :--- |
| `app.py` | Streamlit 仪表盘主程序 |
| `scanner.py` | 核心扫描引擎 (CSV 输出) |
| `backtester.py` | 单股回测引擎 |
| `chart_utils.py` | 增强版图表库 |
| `indicator_utils.py` | 指标库 (BLUE, ZigZag, ADX, ATR, VP) |
| `data_fetcher.py` | 数据获取模块 (Polygon/Tushare) |
| `db/database.py` | 数据库操作模块 |
| `services/scan_service.py` | 扫描服务 |
| `scripts/run_daily_scan.py` | 每日扫描脚本 |
| `scripts/backfill.py` | 批量回填脚本 |

---

## 🔮 未来规划 (Roadmap)

- [x] ~~数据库集成~~: ✅ V2.1 已完成
- [ ] **通知系统**: 集成 Telegram/Email/钉钉机器人，扫描完成后自动推送到手机
- [ ] **盘中监控**: 目前是盘后扫描 (EOD)，未来可升级为盘中实时 (Intraday) 监控异动
- [ ] **A股全面适配**: 目前 V2 主要针对美股优化，需针对 A 股的涨跌停限制和 T+1 规则进行适配
- [ ] **定时任务**: 集成 APScheduler 实现自动每日扫描
