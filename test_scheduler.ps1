# 简化版股票扫描系统 - Windows任务计划程序测试脚本

Write-Host "======================================"
Write-Host "   测试 Windows任务计划程序安装"
Write-Host "======================================"

# 获取当前工作目录
$CURRENT_DIR = Get-Location
Write-Host "📁 当前工作目录: $CURRENT_DIR"

# 检查必要文件是否存在
$cnScript = Join-Path $CURRENT_DIR "scan_cn_signals_multi_thread_tushare.py"
$hkScript = Join-Path $CURRENT_DIR "scan_hk_signals_multi_thread_tushare.py"

Write-Host "🔍 检查扫描脚本..."
if (Test-Path $cnScript) {
    Write-Host "✅ A股扫描脚本存在: $cnScript" -ForegroundColor Green
} else {
    Write-Host "❌ A股扫描脚本不存在: $cnScript" -ForegroundColor Red
}

if (Test-Path $hkScript) {
    Write-Host "✅ 港股扫描脚本存在: $hkScript" -ForegroundColor Green  
} else {
    Write-Host "❌ 港股扫描脚本不存在: $hkScript" -ForegroundColor Red
}

# 检查Python环境
Write-Host "🐍 检查Python环境..."
try {
    $pythonPath = (Get-Command python).Source
    Write-Host "✅ Python路径: $pythonPath" -ForegroundColor Green
    
    # 测试Python版本
    $pythonVersion = python --version
    Write-Host "✅ Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 找不到Python，请确保Python已安装并在PATH中" -ForegroundColor Red
    exit 1
}

# 尝试创建一个测试任务
Write-Host "🧪 测试任务创建权限..."
try {
    # 创建简单的测试任务
    $testAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c echo Test Task Executed > test_output.txt"
    $testTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
    $testSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    
    # 尝试注册测试任务
    Register-ScheduledTask -TaskName "StockScannerTest" -Action $testAction -Trigger $testTrigger -Settings $testSettings -Force
    
    Write-Host "✅ 测试任务创建成功！" -ForegroundColor Green
    
    # 立即删除测试任务
    Unregister-ScheduledTask -TaskName "StockScannerTest" -Confirm:$false
    Write-Host "✅ 测试任务已清理" -ForegroundColor Green
    
} catch {
    Write-Host "❌ 任务创建失败: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "可能需要管理员权限或者有其他限制。" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "🎯 系统检查完成，可以安装股票扫描任务！" -ForegroundColor Green
Write-Host "   建议执行完整安装脚本:" -ForegroundColor Yellow  
Write-Host "   完整命令: PowerShell -ExecutionPolicy Bypass -File install_windows_scheduler.ps1" -ForegroundColor Cyan
Write-Host "" 