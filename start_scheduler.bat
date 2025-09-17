@echo off
echo 启动股票扫描定时任务...
echo 当前时间: %date% %time%
echo.

cd /d "%~dp0"

echo 检查Python环境...
python --version
if errorlevel 1 (
    echo Python未找到，请确保已安装Python并添加到PATH
    pause
    exit /b 1
)

echo.
echo 安装依赖包...
pip install -r requirements_scheduler.txt
if errorlevel 1 (
    echo 依赖包安装失败
    pause
    exit /b 1
)

echo.
echo 启动定时任务...
python scheduler_stock_scan.py

pause 