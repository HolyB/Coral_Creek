# BLUE信号扫描器 - 版本1 (v1)

## 📋 版本概述

这是BLUE信号扫描器的第一个正式版本，专门用于扫描BLUE信号（多头信号），完全移除了LIRED信号（空头信号）相关功能。

## 📁 文件列表

- `scan_signals_blue_only.py` - 美股BLUE信号扫描器
- `scan_cn_signals_blue_only.py` - A股BLUE信号扫描器

## 🎯 主要特性

### 美股版本 (`scan_signals_blue_only.py`)
- ✅ 仅扫描BLUE信号（多头）
- ✅ 支持日线/周线/全部周期过滤
- ✅ 支持黑马信号过滤
- ✅ 多线程并发扫描
- ✅ 邮件通知功能
- ✅ 使用Polygon API获取数据

### A股版本 (`scan_cn_signals_blue_only.py`)
- ✅ 仅扫描BLUE信号（多头）
- ✅ 支持日线/周线/全部周期过滤
- ✅ 支持黑马信号过滤
- ✅ 分批扫描功能
- ✅ 邮件通知功能
- ✅ 使用Tushare API获取数据

## 🚀 使用方法

### 美股版本

```bash
# 扫描所有BLUE信号（日线+周线）
python scan_signals_blue_only.py

# 只看日线BLUE信号
python scan_signals_blue_only.py --period daily

# 只看周线BLUE信号
python scan_signals_blue_only.py --period weekly

# 只看BLUE+黑马信号同时出现的股票
python scan_signals_blue_only.py --with-heima

# 限制扫描数量
python scan_signals_blue_only.py --limit 1000

# 不发送邮件
python scan_signals_blue_only.py --no-email
```

### A股版本

```bash
# 扫描所有BLUE信号（日线+周线）
python scan_cn_signals_blue_only.py

# 只看日线BLUE信号
python scan_cn_signals_blue_only.py --period daily

# 只看周线BLUE信号
python scan_cn_signals_blue_only.py --period weekly

# 只看BLUE+黑马信号同时出现的股票
python scan_cn_signals_blue_only.py --with-heima

# 指定批次范围
python scan_cn_signals_blue_only.py --start-batch 1 --end-batch 3

# 不发送邮件
python scan_cn_signals_blue_only.py --no-email
```

## 📊 信号阈值配置

### 默认阈值

**美股版本：**
- 日线BLUE阈值: 100，所需天数: 3
- 周线BLUE阈值: 130，所需周数: 2

**A股版本：**
- 日线BLUE阈值: 100，所需天数: 3
- 周线BLUE阈值: 100，所需周数: 2

## 🔧 技术实现

### 核心算法
- 使用富途函数库（REF, EMA, SMA, LLV, HHV等）
- BLUE信号计算公式：
  - VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
  - VAR2 = SMA(|LOW - VAR1|, 13) / SMA(max(LOW - VAR1, 0), 10)
  - VAR3 = EMA(VAR2, 10)
  - VAR4 = LLV(LOW, 33)
  - VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
  - VAR6 = POW(|VAR5|, 0.3) * sign(VAR5)
  - BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)

### 黑马信号
- 基于CCI类超卖指标（VAR2 < -110）
- 结合ZIG底部反转信号（VAR4 > 0）

## 📝 版本历史

### v1.0 (2025-12-15)
- ✅ 初始版本发布
- ✅ 移除所有LIRED信号相关代码
- ✅ 仅保留BLUE信号扫描功能
- ✅ 支持美股和A股两个版本
- ✅ 支持周期过滤和黑马信号过滤
- ✅ 完整的邮件通知功能

## 🔄 与原版对比

| 功能 | 原版 | v1.0 (BLUE专版) |
|------|------|-----------------|
| BLUE信号 | ✅ | ✅ |
| LIRED信号 | ✅ | ❌ 已移除 |
| 周期过滤 | ✅ | ✅ |
| 黑马信号 | ✅ | ✅ |
| 邮件通知 | ✅ | ✅ |
| 代码复杂度 | 高 | 低（简化） |

## 📌 注意事项

1. **数据源要求**：
   - 美股版本需要Polygon API密钥
   - A股版本需要Tushare API密钥

2. **性能优化**：
   - 美股版本默认限制20000只股票
   - A股版本支持分批扫描，避免API限制

3. **错误处理**：
   - 自动跳过无数据的股票
   - 单只股票失败不影响整体扫描

## 🎯 使用建议

1. **日常扫描**：使用默认参数，扫描全部股票
2. **快速测试**：使用 `--limit` 或 `--batch-size` 限制数量
3. **精准筛选**：结合 `--with-heima` 和周期过滤
4. **批量处理**：A股版本使用批次参数分段扫描

## 📞 问题反馈

如遇到问题，请检查：
1. API密钥是否有效
2. 网络连接是否正常
3. 依赖库是否安装完整
4. 股票数据是否可用

---

**版本创建日期**: 2025-12-15  
**维护状态**: 活跃维护中
