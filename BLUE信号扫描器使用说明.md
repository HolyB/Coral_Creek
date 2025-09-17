# 🔵 A股BLUE信号专用扫描器 使用说明

## 📋 概述

基于您的需求，我为您创建了一个专门扫描BLUE信号的A股扫描器，**移除了所有LIRED相关逻辑**，只专注于BLUE信号的检测。

## 🚀 可用版本

### 1. **简化版** (推荐) - `scan_blue_signals_simple.py`
- ✅ **已测试成功**
- ✅ 使用简化的数据获取方法
- ✅ 直接使用AKShare获取数据
- ✅ 内置多数据源股票列表获取
- ✅ 自动处理数据格式转换

### 2. **完整版** - `scan_blue_signals_only.py`  
- ⚠️ 依赖Stock_utils模块
- 🔧 需要修复数据获取接口
- 📧 包含邮件通知功能

## 🎯 BLUE信号检测逻辑

### 什么是BLUE信号？
BLUE信号是一种技术分析指标，用于识别股票的潜在买入机会。

### 计算方法
1. **日线BLUE计算**：
   - 基于日线的开盘价、最高价、最低价、收盘价
   - 通过复杂的数学公式计算VAR变量
   - 最终得出BLUE指标值

2. **周线BLUE计算**：
   - 将日线数据转换为周线数据
   - 应用相同的计算逻辑
   - 得出周线BLUE指标值

### 信号条件（可配置）
- **日线BLUE**：值 > 阈值，且连续出现指定天数
- **周线BLUE**：值 > 阈值，且连续出现指定周数
- **强信号**：日线和周线BLUE同时满足条件

## 🛠️ 使用方法

### 基本用法
```bash
# 使用默认参数扫描
python scan_blue_signals_simple.py

# 测试少量股票
python scan_blue_signals_simple.py --batch_size 50

# 指定扫描时机
python scan_blue_signals_simple.py --timing "盘前扫描"
```

### 高级参数配置
```bash
python scan_blue_signals_simple.py \
    --batch_size 100 \
    --max_workers 10 \
    --min_turnover 200 \
    --day_blue 100 \
    --week_blue 130 \
    --day_blue_count 3 \
    --week_blue_count 2 \
    --timing "自定义扫描"
```

### 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--batch_size` | 扫描股票数量 (0=全部) | 0 | 100 |
| `--max_workers` | 并发线程数 | 10 | 20 |
| `--min_turnover` | 最小成交额(万元) | 200 | 500 |
| `--day_blue` | 日线BLUE阈值 | 100 | 80 |
| `--week_blue` | 周线BLUE阈值 | 130 | 100 |
| `--day_blue_count` | 日线BLUE出现次数 | 3 | 2 |
| `--week_blue_count` | 周线BLUE出现次数 | 2 | 1 |
| `--timing` | 扫描时机标识 | '' | "盘前" |

## 📊 输出结果

### 1. 实时进度显示
```
🔍 开始扫描BLUE信号...
📊 股票数量: 100
⚙️ 参数: 线程数=10, 最小成交额=200万
🎯 BLUE阈值: 日线>100, 周线>130
📈 信号条件: 日线3天, 周线2周
扫描进度: 100%|████████████| 100/100 [00:30<00:00,  3.33it/s]
```

### 2. 发现信号时的输出
```
✅ 发现BLUE信号: 000001.SZ - 日BLUE:3天(125.5), 价格:15.23, 成交额:1250万
✅ 发现BLUE信号: 600519.SH - 周BLUE:2周(145.2), 价格:1829.50, 成交额:5600万 ⭐
```

### 3. 最终统计报告
```
🎉 BLUE信号扫描完成!
⏱️ 耗时: 30.25 秒
🎯 发现 5 只股票满足BLUE信号条件

📋 前10个BLUE信号:
序号 代码       价格     成交额(万)  日BLUE          周BLUE          同时
1    000001.SZ  15.23    1250       3天(125.5)      -               
2    600519.SH  1829.50  5600       3天(110.2)      2周(145.2)     ⭐

📊 BLUE信号统计:
   日线BLUE信号: 4 只
   周线BLUE信号: 3 只
   日周同时BLUE: 2 只
```

### 4. CSV文件输出
自动保存到文件：`blue_signals_simple_20250825_134250.csv`

包含字段：
- symbol: 股票代码
- price: 当前价格
- turnover: 成交额(万元)
- blue_days: 日线BLUE出现天数
- blue_weeks: 周线BLUE出现周数
- latest_day_blue_value: 最近日线BLUE值
- latest_week_blue_value: 最近周线BLUE值
- has_day_week_blue: 是否日周同时BLUE

## 🔧 常见问题

### Q1: 为什么扫描结果为0？
**可能原因：**
1. 阈值设置过高，降低 `--day_blue` 和 `--week_blue` 值
2. 条件过严，减少 `--day_blue_count` 和 `--week_blue_count`
3. 成交额过滤太严，降低 `--min_turnover`
4. 当前市场状况确实没有满足条件的股票

**解决方案：**
```bash
# 使用更宽松的条件
python scan_blue_signals_simple.py \
    --day_blue 50 \
    --week_blue 50 \
    --day_blue_count 2 \
    --week_blue_count 1 \
    --min_turnover 100
```

### Q2: 如何提高扫描速度？
**优化建议：**
1. 增加线程数：`--max_workers 20`
2. 减少扫描数量：`--batch_size 500`
3. 确保网络状况良好

### Q3: 扫描某些股票失败？
这是正常现象，可能原因：
- 停牌股票无法获取数据
- 新股数据不足
- 网络超时

## 🎯 实际应用建议

### 1. 盘前扫描
```bash
python scan_blue_signals_simple.py --timing "盘前" --batch_size 1000
```

### 2. 盘中扫描
```bash
python scan_blue_signals_simple.py --timing "盘中" --min_turnover 500
```

### 3. 盘后扫描
```bash
python scan_blue_signals_simple.py --timing "盘后" --batch_size 0
```

### 4. 周末深度扫描
```bash
python scan_blue_signals_simple.py \
    --timing "周末扫描" \
    --batch_size 0 \
    --day_blue 80 \
    --week_blue 100 \
    --max_workers 20
```

## ⚡ 性能特点

- **多数据源容错**：支持9个数据源自动切换
- **多线程并发**：大幅提升扫描速度
- **智能缓存**：避免重复获取股票列表
- **内存优化**：逐个处理股票，避免内存溢出
- **错误处理**：单只股票失败不影响整体扫描

## 🔗 与原版对比

| 功能 | 原版 | BLUE专版 |
|------|------|----------|
| BLUE信号 | ✅ | ✅ |
| LIRED信号 | ✅ | ❌ 已移除 |
| 数据获取 | 复杂依赖 | 简化独立 |
| 运行稳定性 | 中等 | 高 |
| 使用难度 | 高 | 低 |

## 📝 更新日志

### v1.0 - 简化版
- ✅ 移除所有LIRED相关逻辑
- ✅ 简化数据获取方式
- ✅ 集成多数据源股票列表
- ✅ 优化错误处理和用户体验
- ✅ 添加详细的输出和统计信息

---

**现在您就有了一个专门的BLUE信号扫描器！** 🎉

运行命令：
```bash
python scan_blue_signals_simple.py --batch_size 100 --timing "测试运行"
```

