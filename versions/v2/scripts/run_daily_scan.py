#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日扫描脚本 - 可通过 cron 或手动运行
"""
import os
import sys
from datetime import datetime

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import init_db, get_db_stats, get_scan_job
from services.scan_service import run_scan_for_date


def main():
    print("="*60)
    print(f"🌊 Coral Creek Daily Scan")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 初始化数据库
    init_db()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 检查今天是否已扫描
    job = get_scan_job(today)
    if job and job['status'] == 'done':
        print(f"\n⚠️  Today ({today}) has already been scanned.")
        print(f"   Signals found: {job['signals_found']}")
        print(f"   Finished at: {job['finished_at']}")
        print("\nTo rescan, delete the job from the database first.")
        return
    
    # 运行扫描
    print(f"\n🚀 Starting scan for {today}...")
    results = run_scan_for_date(today, market='US', max_workers=30, save_to_db=True)
    
    # 显示结果摘要
    print("\n" + "="*60)
    print("📊 Scan Summary")
    print("="*60)
    
    if results:
        print(f"\n✅ Found {len(results)} candidates\n")
        
        # 按 BLUE 排序显示 Top 10
        print("🏆 Top 10 by Day BLUE:")
        sorted_by_blue = sorted(results, key=lambda x: x['Blue_Daily'], reverse=True)[:10]
        for i, r in enumerate(sorted_by_blue, 1):
            print(f"  {i:2}. {r['Symbol']:6} | ${r['Price']:8.2f} | Day: {r['Blue_Daily']:5.1f} | Week: {r['Blue_Weekly']:5.1f} | {r['Regime']}")
        
        # 策略分布
        strat_d = sum(1 for r in results if r.get('Strat_D_Trend'))
        strat_c = sum(1 for r in results if r.get('Strat_C_Resonance'))
        legacy = sum(1 for r in results if r.get('Legacy_Signal'))
        
        print(f"\n📈 Strategy Distribution:")
        print(f"   Strategy D (Trend):     {strat_d}")
        print(f"   Strategy C (Resonance): {strat_c}")
        print(f"   Legacy (BLUE > 100):    {legacy}")
    else:
        print("\n⚠️  No signals found today.")
    
    # 数据库统计
    stats = get_db_stats()
    print(f"\n📁 Database Stats:")
    print(f"   Total Records: {stats['total_records']:,}")
    print(f"   Total Dates:   {stats['total_dates']:,}")
    print(f"   Date Range:    {stats['min_date']} ~ {stats['max_date']}")


if __name__ == "__main__":
    main()



