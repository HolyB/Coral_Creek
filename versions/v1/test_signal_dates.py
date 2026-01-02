#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试信号日期功能"""
import sys
import os
import json

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_manager import StockDatabase
import pandas as pd

print("=" * 60)
print("测试信号日期功能")
print("=" * 60)

db = StockDatabase()

# 检查数据库结构
cursor = db.conn.cursor()
cursor.execute('PRAGMA table_info(scan_results)')
cols = cursor.fetchall()
print("\n数据库字段:")
for col in cols:
    print(f"  - {col[1]} ({col[2]})")

# 检查是否有数据
dates = db.get_available_dates()
if dates:
    print(f"\n可用扫描日期: {len(dates)} 个")
    print(f"最近日期: {dates[:3]}")
    
    # 检查最新日期的数据
    latest_date = dates[0]
    df = db.get_results_by_date(latest_date)
    print(f"\n{latest_date} 的数据:")
    print(f"  总记录数: {len(df)}")
    
    # 检查日期字段
    if 'day_blue_dates' in df.columns:
        has_dates = df['day_blue_dates'].notna().sum()
        print(f"  有日线BLUE日期的记录: {has_dates}")
        
        # 显示几个示例
        sample = df[df['day_blue_dates'].notna()].head(3)
        if not sample.empty:
            print("\n  示例数据:")
            for idx, row in sample.iterrows():
                symbol = row.get('symbol', 'N/A')
                try:
                    dates = json.loads(row['day_blue_dates']) if isinstance(row['day_blue_dates'], str) else row['day_blue_dates']
                    print(f"    {symbol}: {len(dates)} 个日期 - {dates[:3]}...")
                except Exception as e:
                    print(f"    {symbol}: 解析错误 - {e}")
    
    if 'week_blue_dates' in df.columns:
        has_dates = df['week_blue_dates'].notna().sum()
        print(f"  有周线BLUE日期的记录: {has_dates}")
    
    if 'heima_dates' in df.columns:
        has_dates = df['heima_dates'].notna().sum()
        print(f"  有黑马信号日期的记录: {has_dates}")
else:
    print("\n数据库中暂无数据")

print("\n" + "=" * 60)
print("测试完成")


