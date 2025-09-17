# 增强版港股A股扫描定时任务系统

## 📋 系统概述

这是一个基于北京时间的智能股票扫描定时任务系统，支持A股和港股的全天候自动扫描，包含盘前、盘中、盘后等多个时间段的定制化扫描策略。

## 🔧 主要功能

### ⏰ 定时任务安排（北京时间）

**工作日扫描时间表：**
- **8:30** - 盘前早期扫描（顺序执行，获取隔夜消息）
- **9:00** - 盘前扫描（并行执行，开盘前最后扫描）
- **10:30** - 上午盘中扫描（并行执行，上午交易时段中期）
- **14:00** - 下午盘中扫描（并行执行，下午开盘后）
- **15:30** - 盘后扫描（顺序执行，收盘后立即扫描）
- **16:30** - 盘后深度扫描（并行执行，深度分析）

**周末扫描：**
- **周六 10:00** - 综合扫描
- **周日 20:00** - 为下周做准备的扫描

### 🎯 智能扫描策略

- **盘前扫描**：重点关注做多信号，成交量门槛较低
- **盘中扫描**：双向信号检测，成交量门槛较高
- **盘后扫描**：全面分析，平衡的成交量要求
- **周末扫描**：深度分析，较长的处理时间

### 🌏 时区支持

- 完全基于北京时间（Asia/Shanghai）
- 自动识别交易日和非交易日
- 实时显示交易时段状态

## 📁 文件结构

```
Coral_Creek/
├── scheduler_stock_scan_enhanced.py     # 增强版定时任务主程序
├── stock_scanner_config.json           # 配置文件
├── test_scheduler.py                    # 测试脚本
├── start_enhanced_stock_scanner.bat    # Windows启动脚本
├── README_增强版股票扫描系统.md         # 本说明文件
├── scan_cn_signals_multi_thread_tushare.py  # A股扫描脚本
└── scan_hk_signals_multi_thread_tushare.py  # 港股扫描脚本
```

## 🚀 快速开始

### 1. 环境准备

确保您已经安装了必要的Python包：

```bash
pip install schedule pytz tushare pandas numpy
```

### 2. 配置检查

运行测试脚本检查系统配置：

```bash
python test_scheduler.py
```

### 3. 启动定时任务

**方式一：使用批处理文件启动（推荐Windows用户）**
```bash
双击运行 start_enhanced_stock_scanner.bat
```

**方式二：直接运行Python脚本**
```bash
python scheduler_stock_scan_enhanced.py
```

## ⚙️ 配置说明

### 配置文件 `stock_scanner_config.json`

```json
{
    "scheduler_settings": {
        "timezone": "Asia/Shanghai",           // 时区设置
        "check_interval_seconds": 60,         // 任务检查间隔
        "status_report_interval_hours": 1     // 状态报告间隔
    },
    "scan_parameters": {
        "cn_stock": {                         // A股扫描参数
            "premarket": {
                "batch_size": 400,            // 批量大小
                "max_workers": 8,             // 最大工作线程数
                "signal_type": "bullish",     // 信号类型
                "min_turnover": 150           // 最小成交量
            }
        }
    }
}
```

### 主要参数说明

- **batch_size**: 每批处理的股票数量
- **max_workers**: 并行处理的线程数
- **signal_type**: 
  - `"bullish"` - 只检测做多信号
  - `"bearish"` - 只检测做空信号
  - `"both"` - 检测双向信号
- **min_turnover**: 最小成交量过滤器

## 📊 监控和日志

### 日志文件
系统会生成详细的日志文件：
- `enhanced_stock_scanner_scheduler.log` - 主要日志文件

### 实时状态监控
程序运行时会显示：
- 当前北京时间
- 交易时段状态（A股/港股）
- 即将执行的任务
- 任务执行状态

## 🔍 故障排除

### 常见问题

**1. 时区显示不正确**
- 检查系统时区设置
- 确保pytz库正确安装

**2. 扫描脚本找不到**
- 确保以下文件存在：
  - `scan_cn_signals_multi_thread_tushare.py`
  - `scan_hk_signals_multi_thread_tushare.py`

**3. 定时任务不执行**
- 检查是否为交易日
- 确认时间设置是否正确
- 查看日志文件获取详细错误信息

### 测试命令

```bash
# 运行系统测试
python test_scheduler.py

# 手动测试A股扫描
python scan_cn_signals_multi_thread_tushare.py --batch_size 100

# 手动测试港股扫描
python scan_hk_signals_multi_thread_tushare.py --batch_size 100
```

## 🎛️ 高级配置

### 自定义扫描时间

编辑 `scheduler_stock_scan_enhanced.py` 文件中的 `setup_enhanced_schedule()` 函数：

```python
# 添加新的扫描时间
schedule.every().monday.at("11:00").do(custom_scan_function)
```

### 调整扫描参数

根据市场情况和系统性能，可以调整：
- 批量大小（影响内存使用和速度）
- 工作线程数（影响CPU使用）
- 成交量门槛（影响信号质量）

## 📈 性能优化建议

1. **内存优化**：适当调整batch_size，避免内存不足
2. **网络优化**：控制max_workers数量，避免API限制
3. **磁盘优化**：定期清理旧的日志和结果文件
4. **时间优化**：根据实际需要调整扫描频率

## 🔐 安全建议

1. 确保API密钥安全存储
2. 定期检查日志文件，监控异常访问
3. 使用防火墙保护系统端口
4. 定期备份配置文件和重要数据

## 📞 支持和维护

### 系统状态检查
```bash
# 检查定时任务状态
python -c "import schedule; print(len(schedule.get_jobs()), '个任务已安排')"

# 检查北京时间
python -c "from datetime import datetime; import pytz; print('北京时间:', datetime.now(pytz.timezone('Asia/Shanghai')))"
```

### 日志分析
```bash
# 查看最近的扫描结果
tail -n 50 enhanced_stock_scanner_scheduler.log

# 搜索错误信息
grep -i "error\|异常\|失败" enhanced_stock_scanner_scheduler.log
```

## 📋 更新日志

### v2.0 (当前版本)
- ✅ 增加基于北京时间的精确时区控制
- ✅ 新增盘中扫描功能（10:30, 14:00）
- ✅ 优化并行和顺序扫描策略
- ✅ 增强交易时段智能识别
- ✅ 完善配置文件系统
- ✅ 新增系统测试功能

### v1.0 (原版本)
- ✅ 基础定时任务功能
- ✅ A股和港股扫描支持
- ✅ 简单的盘前盘后扫描

---

**注意**：使用本系统前，请确保您已经正确配置了股票数据源的API密钥，并了解相关的使用限制和费用。

**免责声明**：本系统仅用于技术分析和数据处理，不构成投资建议。投资有风险，决策需谨慎。 