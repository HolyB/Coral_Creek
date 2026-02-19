#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按日期范围重扫（可覆盖）
- 使用 v3 scan_service 的最新逻辑重算历史信号
- 可选先删除当日旧结果，避免历史脏数据残留
"""

import argparse
import os
import sys
import sqlite3
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from db.database import init_db, get_db
from services.scan_service import run_scan_for_date


def _load_dates_from_scan_results(market: str, start: str, end: str):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT scan_date
            FROM scan_results
            WHERE market = ? AND scan_date BETWEEN ? AND ?
            ORDER BY scan_date ASC
            """,
            (market, start, end),
        )
        return [str(r[0]) for r in cur.fetchall()]


def _purge_one_day(market: str, scan_date: str):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM scan_results WHERE market=? AND scan_date=?", (market, scan_date))
        deleted_scan = int(cur.rowcount or 0)
        cur.execute("DELETE FROM scan_jobs WHERE market=? AND scan_date=?", (market, scan_date))
        cur.execute("DELETE FROM candidate_tracking WHERE market=? AND signal_date=?", (market, scan_date))
        deleted_track = int(cur.rowcount or 0)
        conn.commit()
    return deleted_scan, deleted_track


def main():
    parser = argparse.ArgumentParser(description="Rescan date range with optional overwrite")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--market", default="US", choices=["US", "CN"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Delete old rows before rescan")
    args = parser.parse_args()

    init_db()

    # 仅重扫库里已有交易日，避免节假日回退到前一交易日造成重复
    dates = _load_dates_from_scan_results(args.market, args.start, args.end)
    print(f"[{args.market}] range={args.start}..{args.end}, dates_in_db={len(dates)}, overwrite={args.overwrite}")

    if not dates:
        print("No dates found in scan_results for this range.")
        return

    for i, d in enumerate(dates, 1):
        if args.overwrite:
            ds, dt = _purge_one_day(args.market, d)
            print(f"[{i}/{len(dates)}] purge {d}: scan_results={ds}, candidate_tracking={dt}")

        t0 = datetime.now()
        results = run_scan_for_date(
            target_date=d,
            market=args.market,
            max_workers=int(args.workers),
            limit=int(args.limit),
            save_to_db=True,
        )
        found = len(results or [])
        sec = (datetime.now() - t0).total_seconds()
        print(f"[{i}/{len(dates)}] rescan {d}: signals={found}, elapsed={sec:.1f}s")

    print("DONE")


if __name__ == "__main__":
    main()
