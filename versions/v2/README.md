# 🌊 Coral Creek - AI 驱动的量化交易辅助系统 (v2.3)

Coral Creek 是一个集成了**传统技术指标**（BLUE、筹码分布）与**现代大语言模型**（LLM）的智能股票分析系统。它旨在帮助个人投资者发现交易机会、监控持仓风险，并提供客观的决策支持。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![Market](https://img.shields.io/badge/Market-US%20%7C%20CN-green.svg)

## ✨ 核心功能 (v2.3)

### 1. 🤖 AI 决策仪表盘
不仅仅是数据展示，更是**决策辅助**。
- **智能评分**: LLM 根据技术面、筹码面和舆情综合打分 (0-100)。
- **交易信号**: 明确给出 BUY (买入) / SELL (卖出) / HOLD (观望) 建议。
- **三位一体**: 结合 **BLUE 信号** (趋势)、**筹码分布** (成本) 和 **均线系统** (形态)。

### 2. 🌏 双市场支持 (美股 + A股)
一套系统，覆盖全球主要市场。
- **美股**: 通过 Polygon.io 获取实时/历史数据。
- **A股**: 集成 Tushare 和 AkShare (备选)，自动识别 6 位数字代码。
- **无缝切换**: 个股查询和扫描自动适配不同市场规则。

### 3. 📱 盘中实时监控
不错过任何关键行情。
- **实时预警**: 监控持仓的 **止盈/止损** 和 **暴涨/暴跌**。
- **BLUE 爆发监控**: 实时捕捉 BLUE 指标突破 100 或金叉的启动点。
- **自动化**: 基于 GitHub Actions，在美股和 A股交易时段自动运行。

### 4. 📊 专业级图表交互
- **动态 K 线图**: 支持日/周/月线切换，集成十字准线 (Crosshair)。
- **筹码分布 (Volume Profile)**: 右侧直观显示筹码峰、获利盘/套牢盘比例。
- **主力动向**: 分析近期筹码流动，识别主力吸筹或派发行为。

### 5. 🔔 智能通知系统
- **Telegram 推送**: 每日盘前发送精简日报。
- **详细决策卡**: 对高分潜力股生成包含入场价、止损价的详细分析卡片。
- **多渠道**: 支持 Telegram、邮件等多种通知方式。

---

## 🚀 快速开始

### 1. 环境只要
- Python 3.10+
- Tushare Token (用于 A股数据)
- Polygon API Key (用于 美股数据)
- Telegram Bot Token (用于通知)
- LLM API Key (OpenAI / Gemini / Anthropic)

### 2. 安装依赖
```bash
cd versions/v2
pip install -r requirements.txt
```

### 3. 配置环境变量
复制 `.env.example` 为 `.env` 并填入你的 API Keys：
```bash
cp .env.example .env
```
```ini
# .env
POLYGON_API_KEY=your_polygon_key
TUSHARE_TOKEN=your_tushare_token
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GEMINI_API_KEY=your_gemini_key  # 或 OPENAI_API_KEY
```

### 4. 运行系统
启动 Streamlit 前端界面：
```bash
streamlit run app.py
```

---

## 📂 项目结构

```
versions/v2/
├── app.py                  # Streamlit 主程序入口
├── chart_utils.py          # 图表绘制 (K线, 筹码分布)
├── data_fetcher.py         # 数据获取 (支持 US/CN 多源回退)
├── indicator_utils.py      # 技术指标计算 (BLUE, MA, RSI)
├── ml/
│   └── llm_intelligence.py # AI 分析核心逻辑
├── scripts/
│   ├── daily_scan.py       # 每日选股扫描脚本
│   ├── intraday_monitor.py # 盘中实时监控脚本
│   └── send_notification.py # 消息推送脚本
├── services/               # 基础服务 (DB, Notification)
└── .github/workflows/      # 自动化工作流配置
```

## 🤖 自动化部署 (GitHub Actions)
本项目包含两套自动化工作流：
1. **Daily Scan (`daily_scan.yml`)**: 
   - 每天美股/A股收盘后运行。
   - 执行全市场扫描，生成数据库，发送分析报告。
2. **Intraday Monitor (`intraday_monitor.yml`)**: 
   - 在美股 (9:30-16:00 ET) 和 A股 (9:30-15:00 CN) 交易时段运行。
   - 每 30 分钟检查一次持仓状态，触发预警。

---

## 📈 策略说明
系统核心基于 **BLUE 趋势策略**：
- **BLUE > 100**: 超卖区域，往往对应底部反转机会。
- **BLUE < 0**: 弱势区域，建议回避。
- **筹码峰突破**: 价格有效突破筹码峰 (POC)，且上方无明显套牢盘，视为强力买点。

---
*Created by Antigravity Agent*
