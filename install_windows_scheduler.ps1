# 股票扫描系统 - Windows任务计划程序安装脚本
# 使用方法: PowerShell -ExecutionPolicy Bypass -File install_windows_scheduler.ps1

Write-Host "======================================"
Write-Host "   股票扫描系统 Windows任务计划程序安装"
Write-Host "======================================"

# 检查是否以管理员身份运行
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ 错误: 需要管理员权限运行此脚本" -ForegroundColor Red
    Write-Host "请右键点击PowerShell，选择'以管理员身份运行'" -ForegroundColor Yellow
    exit 1
}

# 获取当前工作目录
$CURRENT_DIR = Get-Location
Write-Host "📁 当前工作目录: $CURRENT_DIR"

# 检查必要文件是否存在
$cnScript = Join-Path $CURRENT_DIR "scan_cn_signals_multi_thread_tushare.py"
$hkScript = Join-Path $CURRENT_DIR "scan_hk_signals_multi_thread_tushare.py"

if (-not (Test-Path $cnScript)) {
    Write-Host "❌ 错误: A股扫描脚本不存在: $cnScript" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $hkScript)) {
    Write-Host "❌ 错误: 港股扫描脚本不存在: $hkScript" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 扫描脚本检查通过" -ForegroundColor Green

# 检查Python环境
try {
    $pythonPath = (Get-Command python).Source
    Write-Host "✅ Python路径: $pythonPath" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 找不到Python，请确保Python已安装并在PATH中" -ForegroundColor Red
    exit 1
}

# 创建任务计划程序任务
$taskName = "StockScanner"
$taskFolder = "\StockScanning\"

# 删除现有任务（如果存在）
Write-Host "📋 清理现有任务..."
Get-ScheduledTask -TaskPath $taskFolder -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false

# 创建任务文件夹
try {
    New-ScheduledTaskFolder -TaskPath $taskFolder -ErrorAction SilentlyContinue
} catch {
    # 文件夹可能已存在，忽略错误
}

# 定义任务列表
$tasks = @(
    @{
        Name = "CN_Stock_PreMarket_Early"
        Description = "A股盘前早期扫描"
        Time = "08:30"
        Script = $cnScript
        Args = "--batch_size 300 --max_workers 5 --timing `"盘前早期`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "CN_Stock_PreMarket"
        Description = "A股盘前扫描"
        Time = "09:00"
        Script = $cnScript
        Args = "--batch_size 400 --max_workers 8 --timing `"盘前`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "CN_Stock_Morning"
        Description = "A股上午盘中扫描"
        Time = "10:30"
        Script = $cnScript
        Args = "--batch_size 500 --max_workers 10 --timing `"上午盘中`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "CN_Stock_Afternoon"
        Description = "A股下午盘中扫描"
        Time = "14:00"
        Script = $cnScript
        Args = "--batch_size 500 --max_workers 10 --timing `"下午盘中`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "CN_Stock_PostMarket"
        Description = "A股盘后扫描"
        Time = "15:30"
        Script = $cnScript
        Args = "--batch_size 400 --max_workers 8 --timing `"盘后`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "CN_Stock_PostMarket_Deep"
        Description = "A股盘后深度扫描"
        Time = "16:30"
        Script = $cnScript
        Args = "--batch_size 400 --max_workers 8 --timing `"盘后深度`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "HK_Stock_PreMarket"
        Description = "港股盘前扫描"
        Time = "09:15"
        Script = $hkScript
        Args = "--batch_size 250 --max_workers 8 --timing `"港股盘前`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "HK_Stock_Morning"
        Description = "港股盘中扫描"
        Time = "11:00"
        Script = $hkScript
        Args = "--batch_size 300 --max_workers 10 --timing `"港股盘中`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "HK_Stock_Afternoon"
        Description = "港股下午扫描"
        Time = "14:30"
        Script = $hkScript
        Args = "--batch_size 300 --max_workers 10 --timing `"港股下午`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "HK_Stock_PostMarket"
        Description = "港股盘后扫描"
        Time = "16:30"
        Script = $hkScript
        Args = "--batch_size 250 --max_workers 8 --timing `"港股盘后`""
        Days = "Monday,Tuesday,Wednesday,Thursday,Friday"
    },
    @{
        Name = "CN_Stock_Weekend"
        Description = "A股周末综合扫描"
        Time = "10:00"
        Script = $cnScript
        Args = "--batch_size 600 --max_workers 12 --timing `"周六A股`""
        Days = "Saturday"
    },
    @{
        Name = "HK_Stock_Weekend"
        Description = "港股周末扫描"
        Time = "10:30"
        Script = $hkScript
        Args = "--batch_size 400 --max_workers 12 --timing `"周六港股`""
        Days = "Saturday"
    },
    @{
        Name = "Stock_Sunday_Prep"
        Description = "股票周日准备扫描"
        Time = "20:00"
        Script = $cnScript
        Args = "--batch_size 600 --max_workers 12 --timing `"周日准备`""
        Days = "Sunday"
    }
)

# 创建每个任务
foreach ($task in $tasks) {
    Write-Host "⚙️  创建任务: $($task.Name)" -ForegroundColor Cyan
    
    # 创建任务动作
    $action = New-ScheduledTaskAction -Execute $pythonPath -Argument "`"$($task.Script)`" $($task.Args)" -WorkingDirectory $CURRENT_DIR
    
    # 创建触发器
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $task.Days -At $task.Time
    
    # 创建任务设置
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    # 创建任务主体
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveOrPassword
    
    # 注册任务
    Register-ScheduledTask -TaskName $task.Name -TaskPath $taskFolder -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $task.Description
}

Write-Host ""
Write-Host "✅ Windows任务计划程序安装成功！" -ForegroundColor Green
Write-Host ""
Write-Host "📅 定时任务安排 (北京时间):" -ForegroundColor Yellow
Write-Host "  工作日 A股:" -ForegroundColor White
Write-Host "    08:30 - 盘前早期扫描" -ForegroundColor Gray
Write-Host "    09:00 - 盘前扫描" -ForegroundColor Gray
Write-Host "    10:30 - 上午盘中扫描" -ForegroundColor Gray
Write-Host "    14:00 - 下午盘中扫描" -ForegroundColor Gray
Write-Host "    15:30 - 盘后扫描" -ForegroundColor Gray
Write-Host "    16:30 - 盘后深度扫描" -ForegroundColor Gray
Write-Host ""
Write-Host "  工作日 港股:" -ForegroundColor White
Write-Host "    09:15 - 盘前扫描" -ForegroundColor Gray
Write-Host "    11:00 - 盘中扫描" -ForegroundColor Gray
Write-Host "    14:30 - 下午扫描" -ForegroundColor Gray
Write-Host "    16:30 - 盘后扫描" -ForegroundColor Gray
Write-Host ""
Write-Host "  周末:" -ForegroundColor White
Write-Host "    周六 10:00 - A股综合扫描" -ForegroundColor Gray
Write-Host "    周六 10:30 - 港股扫描" -ForegroundColor Gray
Write-Host "    周日 20:00 - 准备扫描" -ForegroundColor Gray
Write-Host ""
Write-Host "🔍 管理命令:" -ForegroundColor Yellow
Write-Host "  查看任务: Get-ScheduledTask -TaskPath '$taskFolder'" -ForegroundColor Cyan
Write-Host "  打开任务计划程序: taskschd.msc" -ForegroundColor Cyan
Write-Host "  删除所有任务: Get-ScheduledTask -TaskPath '$taskFolder' | Unregister-ScheduledTask -Confirm:`$false" -ForegroundColor Cyan
Write-Host ""
Write-Host "======================================"
Write-Host "  安装完成！系统将自动执行定时任务"
Write-Host "======================================" 