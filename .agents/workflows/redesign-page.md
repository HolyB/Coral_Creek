---
description: 重新设计或优化 Streamlit 页面 UI/UX
---

## 页面设计工作流

### 上下文
- 项目路径: `versions/v3/`
- 入口文件: `versions/v3/app.py` (16000+ 行单文件 Streamlit 应用)
- 导航结构: 4个主页面 (每日机会 / 全量扫描 / 个股研究 / 交易执行)
- 数据库: SQLite 本地 + Supabase 云端
- 核心指标: BLUE(多头) / LIRED(空头) / PINK(KDJ变体) / 黑马/掘地 / 多空王

### 设计原则
1. **行动优先**: 先看风险(卖出/止损)，再看机会(买入)
2. **分层展示**: 摘要卡片 → 分类信号 → 详情表格
3. **20年交易员视角**: SOP 驱动，不是数据倾倒
4. **深色主题**: 适合盯盘场景

### 步骤
1. 先查看当前页面代码 (`app.py` 中对应函数)
2. 理解数据源 (scan_results / candidate_tracking / portfolio)
3. 画出新布局方案 (用文字描述 + 框架图)
4. 用户确认后再写代码
5. 本地测试 → 推送

### 关键文件
- `versions/v3/app.py` - 主应用 (所有页面逻辑)
- `versions/v3/db/database.py` - 数据库查询
- `versions/v3/services/scan_service.py` - 扫描服务
- `versions/v3/indicator_utils.py` - 技术指标计算
- `versions/v3/services/candidate_tracking_service.py` - 候选追踪
