#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试筛选功能"""
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
print("筛选功能测试")
print("=" * 60)

db = StockDatabase()
df = db.get_results_by_date('2026-01-01')

if df.empty:
    print("暂无数据")
    sys.exit(0)

print(f"\n原始数据: {len(df)} 只股票")
print(f"价格范围: {df['price'].min():.2f} - {df['price'].max():.2f}")
print(f"成交额范围: {df['turnover'].min():.2f} - {df['turnover'].max():.2f}")
print(f"日线BLUE范围: {df['blue_daily'].min():.2f} - {df['blue_daily'].max():.2f}")
print(f"周线BLUE范围: {df['blue_weekly'].min():.2f} - {df['blue_weekly'].max():.2f}")

# 测试1: 价格筛选
print("\n" + "=" * 60)
print("测试1: 价格筛选 (1.0 - 10.0)")
print("=" * 60)
filtered = df[(df['price'] >= 1.0) & (df['price'] <= 10.0)]
print(f"筛选结果: {len(filtered)} 只股票")
if len(filtered) > 0:
    print(f"价格范围: {filtered['price'].min():.2f} - {filtered['price'].max():.2f}")
    print("示例:", filtered[['symbol', 'name', 'price']].head(3).to_string(index=False))

# 测试2: 成交额筛选
print("\n" + "=" * 60)
print("测试2: 成交额筛选 (100 - 10000 万元)")
print("=" * 60)
filtered = df[(df['turnover'] >= 100) & (df['turnover'] <= 10000)]
print(f"筛选结果: {len(filtered)} 只股票")
if len(filtered) > 0:
    print(f"成交额范围: {filtered['turnover'].min():.2f} - {filtered['turnover'].max():.2f}")
    print("示例:", filtered[['symbol', 'name', 'turnover']].head(3).to_string(index=False))

# 测试3: BLUE数值筛选
print("\n" + "=" * 60)
print("测试3: 日线BLUE筛选 (150 - 300)")
print("=" * 60)
filtered = df[(df['blue_daily'] >= 150) & (df['blue_daily'] <= 300)]
print(f"筛选结果: {len(filtered)} 只股票")
if len(filtered) > 0:
    print(f"日线BLUE范围: {filtered['blue_daily'].min():.2f} - {filtered['blue_daily'].max():.2f}")
    print("示例:", filtered[['symbol', 'name', 'blue_daily']].head(3).to_string(index=False))

# 测试4: 信号强度筛选
print("\n" + "=" * 60)
print("测试4: 信号强度筛选 (日线>=4天, 周线>=3周)")
print("=" * 60)
filtered = df[(df['blue_days'] >= 4) & (df['blue_weeks'] >= 3)]
print(f"筛选结果: {len(filtered)} 只股票")
if len(filtered) > 0:
    print("示例:", filtered[['symbol', 'name', 'blue_days', 'blue_weeks']].head(3).to_string(index=False))

# 测试5: 组合筛选
print("\n" + "=" * 60)
print("测试5: 组合筛选 (价格1-10 AND 日线BLUE>=150)")
print("=" * 60)
filtered = df[
    (df['price'] >= 1.0) & (df['price'] <= 10.0) &
    (df['blue_daily'] >= 150)
]
print(f"筛选结果: {len(filtered)} 只股票")
if len(filtered) > 0:
    print("示例:", filtered[['symbol', 'name', 'price', 'blue_daily']].head(3).to_string(index=False))

# 测试6: 排序
print("\n" + "=" * 60)
print("测试6: 按日线BLUE降序排序")
print("=" * 60)
sorted_df = df.sort_values(by='blue_daily', ascending=False)
print("前5名:")
print(sorted_df[['symbol', 'name', 'blue_daily']].head(5).to_string(index=False))

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)


