# Coral Creek 量化系统 - 版本管理

本目录包含 Coral Creek 系统的所有迭代版本。

## 📋 版本概览

| 版本 | 目录 | 状态 | 核心特性 |
| :--- | :--- | :--- | :--- |
| **V2.1** | `versions/v2/` | 🚀 **最新推荐** | **数据库存储 + 历史回溯 + 批量回填**。<br>支持按日期查询历史扫描结果。 |
| V2.0 | `versions/v2/` | ✅ 稳定 | 自适应扫描 + 波浪识别 + 白盒风控 + Dashboard |
| V1.x | `versions/v1/` | 🛑 存档 | 初始版本，基于固定阈值的简单扫描 |

---

## 🚀 快速开始 (V2.1)

### 1. 安装依赖
```bash
cd versions/v2
pip install pandas numpy plotly streamlit polygon-api-client tqdm
```

### 2. 初始化数据库
```bash
python -c "from db.database import init_db; init_db()"
```

### 3. 运行每日扫描
```bash
# 扫描今天的数据
python scripts/run_daily_scan.py

# 或指定日期
python services/scan_service.py --date 2026-01-07 --workers 30
```

### 4. 批量回填历史数据
```bash
# 查看缺失日期 (dry-run)
python scripts/backfill.py --start 2025-12-01 --end 2026-01-07 --dry-run

# 执行回填
python scripts/backfill.py --start 2025-12-01 --end 2026-01-07 --workers 30
```

### 5. 启动 Web 界面
```bash
streamlit run app.py --server.port 8502
```

---

## 📁 V2.1 目录结构

```
versions/v2/
├── app.py                    # Streamlit 前端 (支持日期选择)
├── scanner.py                # 扫描引擎 (CSV 输出，兼容旧版)
├── backtester.py             # 回测引擎
├── indicator_utils.py        # 技术指标计算
├── chart_utils.py            # 图表工具
├── data_fetcher.py           # 数据获取 (Polygon API)
│
├── db/                       # 📦 数据库模块 (V2.1 新增)
│   ├── database.py           # SQLite 操作
│   └── coral_creek.db        # 数据库文件
│
├── services/                 # 🔧 服务模块 (V2.1 新增)
│   └── scan_service.py       # 扫描服务 (支持指定日期)
│
└── scripts/                  # 📜 脚本 (V2.1 新增)
    ├── run_daily_scan.py     # 每日扫描
    └── backfill.py           # 批量回填
```

---

## 🗄️ 数据库设计 (V2.1)

### scan_results 表
存储每日扫描结果，支持按日期查询历史数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | VARCHAR | 股票代码 |
| scan_date | DATE | 扫描日期 (关键索引) |
| blue_daily | FLOAT | 日线 BLUE 信号 |
| blue_weekly | FLOAT | 周线 BLUE 信号 |
| blue_monthly | FLOAT | 月线 BLUE 信号 |
| adx | FLOAT | 趋势强度 |
| ... | ... | 其他指标 |

### scan_jobs 表
记录扫描任务状态。

| 字段 | 类型 | 说明 |
|------|------|------|
| scan_date | DATE | 扫描日期 |
| status | VARCHAR | pending/running/done/failed |
| signals_found | INT | 发现信号数 |

---

## 📝 详细变更日志

请查阅 [V2 Changelog](v2/CHANGELOG.md) 获取详细技术细节。
