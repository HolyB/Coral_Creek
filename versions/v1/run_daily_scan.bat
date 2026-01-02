@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ==========================================
echo 正在启动 A股 扫描 (CN Stock Scan)...
echo ==========================================
python scan_cn_signals_blue_only.py --no-email

echo.
echo ==========================================
echo 正在启动 美股 扫描 (US Stock Scan)...
echo ==========================================
python scan_us_signals.py

echo.
echo ==========================================
echo 全部扫描完成! (All Scans Completed)
echo 请刷新网页查看最新数据。
echo ==========================================
pause


