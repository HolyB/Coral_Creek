# 股票扫描系统 - Windows任务计划程序配置指南

## 📋 概述

本指南帮助您在Windows环境中使用**任务计划程序（Task Scheduler）**设置自动化的股票扫描定时任务。这是Windows环境下crontab的替代方案。

## 🚀 快速安装

### 方法1：自动安装（推荐）

1. **以管理员身份运行PowerShell**
   ```powershell
   # 右键点击PowerShell图标，选择"以管理员身份运行"
   ```

2. **切换到项目目录**
   ```powershell
   cd D:\Cursor\Coral_Creek
   ```

3. **运行安装脚本**
   ```powershell
   PowerShell -ExecutionPolicy Bypass -File install_windows_scheduler.ps1
   ```

### 方法2：手动创建任务

打开任务计划程序：
```powershell
taskschd.msc
```

然后手动创建各个定时任务。

## 📅 定时任务安排

### 🏢 工作日扫描 (周一到周五)

| 时间 | A股任务 | 港股任务 | 说明 |
|------|---------|---------|------|
| 08:30 | 盘前早期扫描 | - | 获取隔夜消息 |
| 09:00 | 盘前扫描 | - | 开盘前最后扫描 |
| 09:15 | - | 港股盘前扫描 | 港股开盘前 |
| 10:30 | 上午盘中扫描 | - | A股上午交易时段 |
| 11:00 | - | 港股盘中扫描 | 港股上午交易时段 |
| 14:00 | 下午盘中扫描 | - | A股下午开盘后 |
| 14:30 | - | 港股下午扫描 | 港股下午交易时段 |
| 15:30 | 盘后扫描 | - | A股收盘后 |
| 16:30 | 盘后深度扫描 | 港股盘后扫描 | 深度分析 |

### 🏖️ 周末扫描
| 时间 | 任务名称 | 说明 |
|------|----------|------|
| 周六 10:00 | A股综合扫描 | 周末深度分析 |
| 周六 10:30 | 港股扫描 | 港股周末分析 |
| 周日 20:00 | 准备扫描 | 为下周做准备 |

## 🔧 管理命令

### 查看和管理任务

```powershell
# 查看所有股票扫描任务
Get-ScheduledTask -TaskPath "\StockScanning\"

# 查看特定任务
Get-ScheduledTask -TaskName "CN_Stock_PreMarket"

# 立即运行任务
Start-ScheduledTask -TaskName "CN_Stock_PreMarket"

# 停用任务
Disable-ScheduledTask -TaskName "CN_Stock_PreMarket"

# 启用任务
Enable-ScheduledTask -TaskName "CN_Stock_PreMarket"

# 删除特定任务
Unregister-ScheduledTask -TaskName "CN_Stock_PreMarket" -Confirm:$false

# 删除所有股票扫描任务
Get-ScheduledTask -TaskPath "\StockScanning\" | Unregister-ScheduledTask -Confirm:$false
```

### 图形界面管理

```powershell
# 打开任务计划程序图形界面
taskschd.msc
```

在图形界面中：
1. 导航到 `任务计划程序库 > StockScanning`
2. 可以查看、编辑、运行、删除任务
3. 查看任务历史记录和日志

## 📊 日志查看

### 查看任务历史

1. 打开任务计划程序 (`taskschd.msc`)
2. 选择具体任务
3. 点击"历史记录"选项卡
4. 查看执行状态和错误信息

### PowerShell查看日志

```powershell
# 查看任务事件
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-TaskScheduler/Operational'; ID=200,201} | 
Where-Object {$_.Message -like "*StockScanning*"} | 
Select-Object TimeCreated, Id, LevelDisplayName, Message | 
Format-Table -Wrap

# 查看最近的任务执行记录
Get-ScheduledTask -TaskPath "\StockScanning\" | Get-ScheduledTaskInfo
```

## 🔍 故障排除

### 常见问题

**1. 权限不足**
```powershell
# 确保以管理员身份运行PowerShell
# 右键PowerShell图标 -> "以管理员身份运行"
```

**2. 执行策略限制**
```powershell
# 临时绕过执行策略
PowerShell -ExecutionPolicy Bypass -File install_windows_scheduler.ps1

# 或者永久设置
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. Python路径问题**
```powershell
# 检查Python是否在PATH中
python --version
Get-Command python

# 如果不在PATH中，需要添加Python安装目录到环境变量
```

**4. 任务不执行**
- 检查系统时间和时区设置
- 确保计算机在预定时间是开机状态
- 检查任务的"条件"和"设置"选项卡

### 调试方法

**1. 手动测试脚本**
```powershell
cd D:\Cursor\Coral_Creek
python scan_cn_signals_multi_thread_tushare.py --batch_size 10 --timing "测试"
```

**2. 测试任务创建**
```powershell
# 创建一个简单的测试任务
$action = New-ScheduledTaskAction -Execute "notepad.exe"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
Register-ScheduledTask -TaskName "TestTask" -Action $action -Trigger $trigger
```

**3. 检查任务设置**
```powershell
# 查看任务详细信息
Get-ScheduledTask -TaskName "CN_Stock_PreMarket" | Get-ScheduledTaskInfo
```

## ⚙️ 高级配置

### 自定义扫描参数

编辑 `install_windows_scheduler.ps1` 文件中的任务定义：

```powershell
@{
    Name = "CN_Stock_PreMarket"
    Description = "A股盘前扫描"
    Time = "09:00"
    Script = $cnScript
    Args = "--batch_size 400 --max_workers 8 --signal_type bullish --timing `"盘前`""
    Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
}
```

可以修改的参数：
- `--batch_size`: 批次大小
- `--max_workers`: 并发数
- `--signal_type`: 信号类型 (bullish/bearish/both)
- `--min_turnover`: 最小成交额

### 添加新的扫描时间

```powershell
# 例如：添加中午12:00的扫描
@{
    Name = "CN_Stock_Noon"
    Description = "A股中午扫描"
    Time = "12:00"
    Script = $cnScript
    Args = "--batch_size 300 --max_workers 6 --timing `"中午扫描`""
    Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
}
```

### 邮件通知设置

任务失败时发送邮件通知：

1. 在任务计划程序中选择任务
2. 右键 -> 属性
3. 切换到"操作"选项卡
4. 添加"发送电子邮件"操作（需要配置SMTP）

## 🛡️ 安全考虑

### 文件权限

```powershell
# 限制脚本文件的访问权限
icacls "install_windows_scheduler.ps1" /grant:r "$env:USERNAME:(R)"
```

### 任务权限

- 任务以当前用户身份运行
- 确保Python脚本有必要的文件访问权限
- 定期检查和更新任务配置

## 📈 性能优化

### 系统资源管理

1. **调整并发数**：根据系统性能调整 `--max_workers` 参数
2. **错峰执行**：避免多个高负载任务同时运行
3. **资源监控**：使用任务管理器监控Python进程资源使用

### 定期维护

```powershell
# 每周清理日志文件（可以作为定时任务添加）
$cleanupScript = @"
Get-ChildItem -Path "D:\Cursor\Coral_Creek" -Filter "*.log" | 
Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | 
Remove-Item -Force
"@

# 创建清理任务
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-Command `"$cleanupScript`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "02:00"
Register-ScheduledTask -TaskName "LogCleanup" -TaskPath "\StockScanning\" -Action $action -Trigger $trigger
```

## 📞 支持与帮助

如果遇到问题：

1. 检查Windows事件查看器中的系统日志
2. 查看任务计划程序中的任务历史记录
3. 验证Python环境和脚本路径
4. 确认系统时区设置正确 (应为中国标准时间)

**记住：所有时间都基于本地系统时间，请确保系统时区设置为北京时间** 