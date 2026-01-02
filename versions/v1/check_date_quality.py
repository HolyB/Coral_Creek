#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查信号日期数据质量"""
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
print("信号日期数据质量检查")
print("=" * 60)

db = StockDatabase()
df = db.get_results_by_date('2026-01-01')

if df.empty:
    print("今日暂无数据")
    sys.exit(0)

print(f"\n总记录数: {len(df)}")
print(f"A股: {len(df[df['market']=='CN'])} 只")
print(f"美股: {len(df[df['market']=='US'])} 只")

# 检查A股日期信息
print("\n" + "=" * 60)
print("A股日期信息")
print("=" * 60)
cn_df = df[df['market']=='CN']
if not cn_df.empty:
    cn_day_dates = cn_df[cn_df['day_blue_dates'].notna()]
    cn_week_dates = cn_df[cn_df['week_blue_dates'].notna()]
    cn_heima_dates = cn_df[cn_df['heima_dates'].notna()]
    
    print(f"有日线BLUE日期的: {len(cn_day_dates)} 只")
    print(f"有周线BLUE日期的: {len(cn_week_dates)} 只")
    print(f"有黑马信号日期的: {len(cn_heima_dates)} 只")
    
    if len(cn_day_dates) > 0:
        sample = cn_day_dates.head(1).iloc[0]
        print(f"\n示例股票: {sample['symbol']} ({sample['name']})")
        try:
            dates = json.loads(sample['day_blue_dates']) if isinstance(sample['day_blue_dates'], str) else sample['day_blue_dates']
            print(f"  日线BLUE日期 ({len(dates)}个): {dates[:5] if len(dates) > 5 else dates}")
        except Exception as e:
            print(f"  解析日期失败: {e}")
else:
    print("暂无A股数据")

# 检查美股日期信息
print("\n" + "=" * 60)
print("美股日期信息")
print("=" * 60)
us_df = df[df['market']=='US']
if not us_df.empty:
    us_day_dates = us_df[us_df['day_blue_dates'].notna()]
    us_week_dates = us_df[us_df['week_blue_dates'].notna()]
    us_heima_dates = us_df[us_df['heima_dates'].notna()]
    
    print(f"有日线BLUE日期的: {len(us_day_dates)} 只")
    print(f"有周线BLUE日期的: {len(us_week_dates)} 只")
    print(f"有黑马信号日期的: {len(us_heima_dates)} 只")
    
    if len(us_day_dates) > 0:
        sample = us_day_dates.head(1).iloc[0]
        print(f"\n示例股票: {sample['symbol']} ({sample['name']})")
        try:
            dates = json.loads(sample['day_blue_dates']) if isinstance(sample['day_blue_dates'], str) else sample['day_blue_dates']
            print(f"  日线BLUE日期 ({len(dates)}个): {dates[:5] if len(dates) > 5 else dates}")
        except Exception as e:
            print(f"  解析日期失败: {e}")
else:
    print("暂无美股数据")

print("\n" + "=" * 60)
print("检查完成")


