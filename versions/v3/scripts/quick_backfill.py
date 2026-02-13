#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速 backfill - 带进度输出，方便调试"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from db.database import get_scanned_dates, query_scan_results
from services.candidate_tracking_service import (
    capture_daily_candidates,
    refresh_candidate_tracking,
    _ensure_tracking_table,
)

def run(market="US", max_per_day=1200):
    _ensure_tracking_table()
    print(f"[{market}] 获取扫描日期...")
    dates = get_scanned_dates(market=market) or []
    print(f"[{market}] 共 {len(dates)} 个扫描日期")
    if not dates:
        return

    total_added = 0
    for i, d in enumerate(dates):
        t0 = time.time()
        rows = query_scan_results(scan_date=d, market=market, limit=max_per_day) or []
        t1 = time.time()
        if not rows:
            print(f"  [{i+1}/{len(dates)}] {d}: 0 rows (查询 {t1-t0:.1f}s)")
            continue
        added = capture_daily_candidates(rows=rows, market=market, signal_date=d, source="history_backfill")
        t2 = time.time()
        total_added += added
        print(f"  [{i+1}/{len(dates)}] {d}: {len(rows)} rows -> +{added} new (查询 {t1-t0:.1f}s, 写入 {t2-t1:.1f}s)")

    print(f"\n[{market}] 回填完成: 新增 {total_added} 条")
    print(f"[{market}] 刷新追踪数据...")
    t0 = time.time()
    refreshed = refresh_candidate_tracking(market=market, max_rows=50000)
    print(f"[{market}] 刷新完成: {refreshed} 条 ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    for m in ["US", "CN"]:
        run(market=m)
