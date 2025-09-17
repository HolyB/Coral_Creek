@echo off
chcp 65001
echo ====================================
echo    增强版股票扫描定时任务启动器
echo ====================================
echo.

REM 设置Python环境路径（请根据实际情况修改）
set PYTHON_ENV=..\stock_env\Scripts\python.exe
set PYTHON_SCRIPT=scheduler_stock_scan_enhanced.py

REM 检查Python环境是否存在
if not exist "%PYTHON_ENV%" (
    echo 错误: 找不到Python环境 %PYTHON_ENV%
    echo 请检查股票环境路径是否正确
    pause
    exit /b 1
)

REM 检查Python脚本是否存在
if not exist "%PYTHON_SCRIPT%" (
    echo 错误: 找不到Python脚本 %PYTHON_SCRIPT%
    echo 请确保脚本在当前目录下
    pause
    exit /b 1
)

echo 正在启动增强版股票扫描定时任务...
echo 使用Python环境: %PYTHON_ENV%
echo 执行脚本: %PYTHON_SCRIPT%
echo.
echo 注意: 程序将在后台运行，请勿关闭此窗口
echo 按 Ctrl+C 可以停止程序
echo.

REM 启动定时任务
"%PYTHON_ENV%" "%PYTHON_SCRIPT%"

pause 