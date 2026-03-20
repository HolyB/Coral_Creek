#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复 scan_results 中布尔字段被存为 blob 的问题"""
import sqlite3, os

db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "coral_creek.db")
conn = sqlite3.connect(db)

bool_cols = ['is_heima', 'is_juedi', 'heima_daily', 'heima_weekly', 'heima_monthly', 
             'juedi_daily', 'juedi_weekly', 'juedi_monthly', 'strat_d_trend', 'strat_c_resonance',
             'legacy_signal', 'duokongwang_buy', 'duokongwang_sell']

print("=== 修复 blob -> integer ===")
for col in bool_cols:
    n1 = conn.execute(f"UPDATE scan_results SET {col} = 1 WHERE typeof({col}) = 'blob' AND {col} = X'01'").rowcount
    n0 = conn.execute(f"UPDATE scan_results SET {col} = 0 WHERE typeof({col}) = 'blob' AND {col} = X'00'").rowcount
    if n1 > 0 or n0 > 0:
        print(f"  {col}: fixed {n1} true + {n0} false blobs")

conn.commit()
print("Done!")

# 验证
print("\n=== 验证 ===")
cur = conn.execute("""SELECT 
    SUM(CASE WHEN is_juedi = 1 THEN 1 ELSE 0 END) as juedi_true,
    SUM(CASE WHEN is_heima = 1 THEN 1 ELSE 0 END) as heima_true,
    SUM(CASE WHEN strat_d_trend = 1 THEN 1 ELSE 0 END) as strat_d,
    SUM(CASE WHEN legacy_signal = 1 THEN 1 ELSE 0 END) as legacy,
    SUM(CASE WHEN typeof(is_juedi) = 'blob' THEN 1 ELSE 0 END) as blobs_remaining
    FROM scan_results WHERE scan_date >= '2025-02-20'""")
r = cur.fetchone()
print(f"  is_juedi=1: {r[0]}")
print(f"  is_heima=1: {r[1]}")
print(f"  strat_d=1: {r[2]}")
print(f"  legacy=1: {r[3]}")
print(f"  blobs remaining: {r[4]}")
conn.close()
