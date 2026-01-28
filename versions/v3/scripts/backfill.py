#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量回填脚本 - 回填指定日期范围的扫描数据
"""
import os
import sys
import argparse
from datetime import datetime, timedelta

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import init_db, get_missing_dates, get_db_stats
from services.scan_service import run_scan_for_date, backfill_dates


def main():
    parser = argparse.ArgumentParser(description='Backfill scan data for date range')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--market', type=str, default='US', help='Market (US/CN)')
    parser.add_argument('--workers', type=int, default=30, help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=0, help='Limit stocks per day (0=all)')
    parser.add_argument('--dry-run', action='store_true', help='Only show missing dates, do not scan')
    
    args = parser.parse_args()
    
    # 初始化数据库
    init_db()
    
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')
    
    print(f"📅 Backfill range: {args.start} to {end_date}")
    
    # 检查缺失日期
    missing = get_missing_dates(args.start, end_date)
    
    if not missing:
        print("✅ No missing dates!")
        return
    
    print(f"📋 Found {len(missing)} missing trading days:")
    for d in missing:
        print(f"   - {d}")
    
    if args.dry_run:
        print("\n(Dry run - no scanning performed)")
        return
    
    # 开始回填
    print(f"\n🚀 Starting backfill with {args.workers} workers...")
    
    for i, date in enumerate(missing):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(missing)}] Backfilling {date}")
        print(f"{'='*60}")
        
        run_scan_for_date(date, market=args.market, limit=args.limit, max_workers=args.workers, save_to_db=True)
    
    print("\n" + "="*60)
    print("✅ Backfill complete!")
    print("="*60)
    
    # 显示统计
    stats = get_db_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Total Records: {stats['total_records']:,}")
    print(f"   Total Symbols: {stats['total_symbols']:,}")
    print(f"   Total Dates:   {stats['total_dates']:,}")
    print(f"   Date Range:    {stats['min_date']} ~ {stats['max_date']}")


if __name__ == "__main__":
    main()




