@echo off
chcp 65001 >nul
cd /d "%~dp0"
:loop
cls
echo ============================================================
echo 扫描进度监控 - 每10秒刷新一次
echo ============================================================
echo.
python monitor_scan.py
echo.
echo 按 Ctrl+C 停止监控
timeout /t 10 /nobreak >nul 2>&1
goto loop


