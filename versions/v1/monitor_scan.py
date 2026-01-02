#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""监控扫描进度"""
import sys
import os
import time
from datetime import datetime

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_manager import StockDatabase
import pandas as pd

print("=" * 60)
print("扫描进度监控")
print("=" * 60)
print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 检查数据库
db = StockDatabase()
dates = db.get_available_dates()

if dates:
    latest_date = dates[0]
    df = db.get_results_by_date(latest_date)
    
    print(f"最新扫描日期: {latest_date}")
    print(f"总记录数: {len(df)}")
    
    cn_count = len(df[df['market'] == 'CN'])
    us_count = len(df[df['market'] == 'US'])
    
    print(f"  A股: {cn_count} 只")
    print(f"  美股: {us_count} 只")
    
    # 检查有日期信息的记录
    if 'day_blue_dates' in df.columns:
        cn_with_dates = df[(df['market'] == 'CN') & (df['day_blue_dates'].notna())].shape[0]
        us_with_dates = df[(df['market'] == 'US') & (df['day_blue_dates'].notna())].shape[0]
        print(f"\n有日期信息的记录:")
        print(f"  A股: {cn_with_dates} 只")
        print(f"  美股: {us_with_dates} 只")
    
    # 信号统计
    if 'has_day_blue' in df.columns:
        day_blue_cn = len(df[(df['market'] == 'CN') & (df['has_day_blue'] == True)])
        day_blue_us = len(df[(df['market'] == 'US') & (df['has_day_blue'] == True)])
        print(f"\n日线BLUE信号:")
        print(f"  A股: {day_blue_cn} 只")
        print(f"  美股: {day_blue_us} 只")
    
    if 'has_week_blue' in df.columns:
        week_blue_cn = len(df[(df['market'] == 'CN') & (df['has_week_blue'] == True)])
        week_blue_us = len(df[(df['market'] == 'US') & (df['has_week_blue'] == True)])
        print(f"\n周线BLUE信号:")
        print(f"  A股: {week_blue_cn} 只")
        print(f"  美股: {week_blue_us} 只")
else:
    print("数据库中暂无数据")

# 检查临时文件
print("\n" + "=" * 60)
print("临时文件检查")
print("=" * 60)

interim_files = []
for pattern in ['cn_signals_blue_interim_*.csv', 'us_signals_*.csv']:
    import glob
    files = glob.glob(os.path.join(os.path.dirname(__file__), pattern))
    interim_files.extend(files)

if interim_files:
    print(f"找到 {len(interim_files)} 个临时文件:")
    for f in sorted(interim_files, key=os.path.getmtime, reverse=True)[:5]:
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)}: {size:,} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("未找到临时文件")

print("\n" + "=" * 60)


