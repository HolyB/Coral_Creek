#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查同时有日线和周线BLUE信号的股票"""
import sys
import os

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_manager import StockDatabase
import pandas as pd

print("=" * 60)
print("日线和周线都有BLUE信号的股票")
print("=" * 60)

db = StockDatabase()
df = db.get_results_by_date('2026-01-01')

if df.empty:
    print("暂无数据")
    sys.exit(0)

print(f"\n总记录数: {len(df)}")
print(f"A股: {len(df[df['market']=='CN'])} 只")
print(f"美股: {len(df[df['market']=='US'])} 只")

# 筛选同时有日线和周线BLUE信号的股票
both_blue = df[(df['has_day_blue']==True) & (df['has_week_blue']==True)]

print(f"\n{'='*60}")
print(f"同时有日线和周线BLUE信号的股票: {len(both_blue)} 只")
print(f"{'='*60}")

if len(both_blue) > 0:
    cn_both = both_blue[both_blue['market']=='CN']
    us_both = both_blue[both_blue['market']=='US']
    
    print(f"A股: {len(cn_both)} 只")
    print(f"美股: {len(us_both)} 只")
    
    print(f"\n{'='*60}")
    print("示例股票（前10只）:")
    print(f"{'='*60}")
    
    for idx, row in both_blue.head(10).iterrows():
        print(f"\n{row['symbol']} ({row['name']}) - {row['market']}")
        print(f"  日线BLUE: {row['blue_days']}天, 周线BLUE: {row['blue_weeks']}周")
        print(f"  价格: {row['price']:.2f}, 成交额: {row['turnover']:.2f}万")
else:
    print("\n暂无同时有日线和周线BLUE信号的股票")

print("\n" + "=" * 60)


