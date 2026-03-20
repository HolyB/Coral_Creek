#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 backfill 数据质量"""
import sqlite3, os

db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "coral_creek.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# 1. 总体统计
cur = conn.execute("SELECT COUNT(*) as cnt, COUNT(DISTINCT scan_date) as dates, COUNT(DISTINCT symbol) as symbols FROM scan_results")
r = cur.fetchone()
print("=== scan_results 总览 ===")
print(f"  总记录: {r['cnt']:,}  |  日期数: {r['dates']}  |  股票数: {r['symbols']:,}")

# 2. 按日期统计
print("\n=== 各日期信号数 (最新20天) ===")
cur = conn.execute("SELECT scan_date, COUNT(*) as cnt FROM scan_results GROUP BY scan_date ORDER BY scan_date DESC LIMIT 20")
for r in cur.fetchall():
    print(f"  {r['scan_date']}: {r['cnt']:,} signals")

# 3. 抽样检查完整数据
print("\n=== 抽样数据 (2025-02-25, blue_daily 最高3条) ===")
cur = conn.execute("""SELECT symbol, scan_date, price, blue_daily, blue_weekly, blue_monthly, 
    adx, volatility, is_heima, is_juedi, strat_d_trend, strat_c_resonance, 
    legacy_signal, regime, duokongwang_buy, duokongwang_sell, vp_rating, wave_phase, chan_signal
    FROM scan_results WHERE scan_date = '2025-02-25' ORDER BY blue_daily DESC LIMIT 3""")
for r in cur.fetchall():
    print(f"  {r['symbol']} | price={r['price']} | blue_d={r['blue_daily']} blue_w={r['blue_weekly']} blue_m={r['blue_monthly']}")
    print(f"    ADX={r['adx']} vol={r['volatility']} heima={r['is_heima']} juedi={r['is_juedi']}")
    print(f"    strat_d={r['strat_d_trend']} strat_c={r['strat_c_resonance']} legacy={r['legacy_signal']}")
    print(f"    regime={r['regime']} dkw_buy={r['duokongwang_buy']} dkw_sell={r['duokongwang_sell']}")
    print(f"    vp={r['vp_rating']} wave={r['wave_phase']} chan={r['chan_signal']}")
    print()

# 4. 信号类型分布
print("=== 信号类型分布 (backfill 数据, >= 2025-02-20) ===")
cur = conn.execute("""SELECT 
    SUM(CASE WHEN strat_d_trend = 1 THEN 1 ELSE 0 END) as strat_d,
    SUM(CASE WHEN strat_c_resonance = 1 THEN 1 ELSE 0 END) as strat_c,
    SUM(CASE WHEN legacy_signal = 1 THEN 1 ELSE 0 END) as legacy,
    SUM(CASE WHEN duokongwang_buy = 1 THEN 1 ELSE 0 END) as dkw_buy,
    SUM(CASE WHEN duokongwang_sell = 1 THEN 1 ELSE 0 END) as dkw_sell,
    SUM(CASE WHEN is_heima = 1 THEN 1 ELSE 0 END) as heima,
    SUM(CASE WHEN is_juedi = 1 THEN 1 ELSE 0 END) as juedi,
    COUNT(*) as total
    FROM scan_results WHERE scan_date >= '2025-02-20' AND scan_date < '2025-03-01' """)
r = cur.fetchone()
print(f"  Strat D (趋势):     {r['strat_d'] or 0:,}")
print(f"  Strat C (共振):     {r['strat_c'] or 0:,}")
print(f"  Legacy (蓝色>=100): {r['legacy'] or 0:,}")
print(f"  多空王买:           {r['dkw_buy'] or 0:,}")
print(f"  多空王卖:           {r['dkw_sell'] or 0:,}")
print(f"  黑马:               {r['heima'] or 0:,}")
print(f"  掘地:               {r['juedi'] or 0:,}")
print(f"  总计:               {r['total'] or 0:,}")

# 5. 对比已有数据（之前每日扫描的结果）
print("\n=== 对比: 已有数据 vs backfill ===")
for date in ['2026-02-12', '2025-02-25']:
    cur = conn.execute("""SELECT COUNT(*) as cnt, 
        AVG(blue_daily) as avg_blue, AVG(price) as avg_price, AVG(adx) as avg_adx,
        MIN(blue_daily) as min_blue, MAX(blue_daily) as max_blue
        FROM scan_results WHERE scan_date = ?""", (date,))
    r = cur.fetchone()
    if r['cnt'] and r['cnt'] > 0:
        print(f"  {date}: {r['cnt']:,} signals | avg_blue={r['avg_blue']:.1f} (min={r['min_blue']:.0f}, max={r['max_blue']:.0f}) | avg_price=${r['avg_price']:.1f} | avg_adx={r['avg_adx']:.1f}")

# 6. 检查是否有 NULL 字段
print("\n=== NULL 字段检查 (2025-02-25) ===")
cur = conn.execute("""SELECT 
    SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) as null_price,
    SUM(CASE WHEN blue_daily IS NULL THEN 1 ELSE 0 END) as null_blue,
    SUM(CASE WHEN adx IS NULL THEN 1 ELSE 0 END) as null_adx,
    SUM(CASE WHEN regime IS NULL THEN 1 ELSE 0 END) as null_regime,
    SUM(CASE WHEN wave_phase IS NULL THEN 1 ELSE 0 END) as null_wave,
    SUM(CASE WHEN chan_signal IS NULL THEN 1 ELSE 0 END) as null_chan,
    COUNT(*) as total
    FROM scan_results WHERE scan_date = '2025-02-25' """)
r = cur.fetchone()
total = r['total'] or 0
print(f"  总数: {total}")
for field in ['null_price', 'null_blue', 'null_adx', 'null_regime', 'null_wave', 'null_chan']:
    val = r[field] or 0
    label = field.replace('null_', '')
    status = "✅" if val == 0 else f"⚠️ {val}/{total} null"
    print(f"  {label}: {status}")

conn.close()
