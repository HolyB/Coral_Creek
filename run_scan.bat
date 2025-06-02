@echo off
chcp 65001 >nul
echo 开始运行扫描脚本 - %date% %time% >> D:\Cursor\Coral_Creek\scan_log.txt

:: 切换到正确的目录
cd /d D:\Cursor\Coral_Creek

:: 激活虚拟环境
call .\coral_creek_env\Scripts\activate

:: 设置Python默认编码为UTF-8
set PYTHONIOENCODING=utf-8

:: 运行 Python 脚本并记录输出
python scan_signals_multi_thread_claude.py >> D:\Cursor\Coral_Creek\scan_log.txt 2>&1

:: 记录完成时间
echo 扫描完成 - %date% %time% >> D:\Cursor\Coral_Creek\scan_log.txt
echo --------------------------- >> D:\Cursor\Coral_Creek\scan_log.txt