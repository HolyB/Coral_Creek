#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回填并刷新 candidate_tracking，用于策略统计口径稳定。
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.candidate_tracking_service import (
    backfill_candidates_from_scan_history,
    refresh_candidate_tracking,
)


def run_one_market(market: str, recent_days: int, max_per_day: int, refresh_rows: int) -> None:
    added = backfill_candidates_from_scan_history(
        market=market,
        recent_days=int(recent_days),
        max_per_day=int(max_per_day),
    )
    refreshed = refresh_candidate_tracking(
        market=market,
        max_rows=int(refresh_rows),
    )
    print(
        f"[{market}] candidate_tracking backfill added={added}, refreshed={refreshed}, "
        f"recent_days={recent_days}, max_per_day={max_per_day}, refresh_rows={refresh_rows}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill candidate_tracking from scan history and refresh pnl.")
    parser.add_argument("--market", type=str, default="ALL", choices=["US", "CN", "ALL"])
    parser.add_argument("--recent-days", type=int, default=420, help="回填最近N个扫描交易日")
    parser.add_argument("--max-per-day", type=int, default=800)
    parser.add_argument("--refresh-rows", type=int, default=12000)
    args = parser.parse_args()

    markets = ["US", "CN"] if args.market == "ALL" else [args.market]
    for m in markets:
        run_one_market(m, args.recent_days, args.max_per_day, args.refresh_rows)
