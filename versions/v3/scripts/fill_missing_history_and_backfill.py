#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键补齐历史K线并回填 scan_results 信号字段

流程:
1) 找出 scan_results 中仍缺 OHLC/多空王 的 symbol
2) 拉取这些 symbol 的历史K线并写入 stock_history.db
3) 执行回填 (OHLC + 多空王 + 可选筹码重算)
4) 输出回填后的剩余缺口统计
"""

import argparse
import os
import sys
import time
import sqlite3
from typing import Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from db.database import get_connection
from db.stock_history import save_stock_history, get_history_stats
from data_fetcher import get_stock_data
from scripts.backfill_scan_signal_fields import run_backfill, recalc_chip_latest_signals


def _missing_symbols(market: str) -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT symbol
        FROM scan_results
        WHERE market = ?
          AND (
            day_high IS NULL OR day_low IS NULL OR day_close IS NULL
            OR duokongwang_buy IS NULL OR duokongwang_sell IS NULL
          )
        ORDER BY symbol
        """,
        (market,),
    )
    rows = cur.fetchall()
    conn.close()
    return [str(r["symbol"]) for r in rows]


def _count_missing(market: str) -> Dict[str, int]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM scan_results WHERE market=? AND (day_high IS NULL OR day_low IS NULL OR day_close IS NULL)",
        (market,),
    )
    miss_ohlc = int(cur.fetchone()[0] or 0)
    cur.execute(
        "SELECT COUNT(*) FROM scan_results WHERE market=? AND (duokongwang_buy IS NULL OR duokongwang_sell IS NULL)",
        (market,),
    )
    miss_dkw = int(cur.fetchone()[0] or 0)
    conn.close()
    return {"missing_ohlc": miss_ohlc, "missing_dkw": miss_dkw}


def _fetch_history_for_symbols(market: str, symbols: List[str], days: int, delay: float) -> Dict[str, int]:
    ok = 0
    fail = 0
    saved_rows = 0

    total = len(symbols)
    if total == 0:
        return {"ok": 0, "fail": 0, "saved_rows": 0}

    print(f"\n[{market}] 待抓取 symbol: {total}")
    for i, sym in enumerate(symbols, 1):
        try:
            df = get_stock_data(sym, market=market, days=days)
            if df is None or len(df) < 20:
                fail += 1
            else:
                saved = save_stock_history(sym, market, df)
                saved_rows += int(saved or 0)
                ok += 1
        except Exception:
            fail += 1

        if i % 30 == 0 or i == total:
            print(f"[{market}] 进度 {i}/{total} | 成功 {ok} | 失败 {fail} | 写入行 {saved_rows}")
        if delay > 0:
            time.sleep(delay)
    return {"ok": ok, "fail": fail, "saved_rows": saved_rows}


def run_market(market: str, days: int, delay: float, recalc_chip: bool, chip_recent_days: int) -> Dict:
    before = _count_missing(market)
    symbols = _missing_symbols(market)
    fetch_stat = _fetch_history_for_symbols(market, symbols, days=days, delay=delay)

    # 重新回填
    run_backfill(market=market, profile="balanced", since_days=0)
    if recalc_chip:
        recalc_chip_latest_signals(market=market, recent_days=int(chip_recent_days))

    after = _count_missing(market)
    return {
        "market": market,
        "before": before,
        "after": after,
        "fetch": fetch_stat,
        "symbols_targeted": len(symbols),
    }


def main():
    parser = argparse.ArgumentParser(description="Fill missing history then backfill scan fields")
    parser.add_argument("--market", choices=["US", "CN", "ALL"], default="ALL")
    parser.add_argument("--days", type=int, default=420, help="每个symbol抓取历史天数")
    parser.add_argument("--delay", type=float, default=0.15, help="每个请求间隔秒数")
    parser.add_argument("--recalc-chip", action="store_true", help="回填后重算筹码字段")
    parser.add_argument("--chip-recent-days", type=int, default=365)
    args = parser.parse_args()

    markets = ["US", "CN"] if args.market == "ALL" else [args.market]
    results = []
    for m in markets:
        print(f"\n{'='*72}\n开始处理市场: {m}\n{'='*72}")
        results.append(
            run_market(
                market=m,
                days=int(args.days),
                delay=float(args.delay),
                recalc_chip=bool(args.recalc_chip),
                chip_recent_days=int(args.chip_recent_days),
            )
        )

    print(f"\n{'='*72}\n回填总结\n{'='*72}")
    for r in results:
        b = r["before"]
        a = r["after"]
        f = r["fetch"]
        print(
            f"[{r['market']}] 缺口 OHLC {b['missing_ohlc']} -> {a['missing_ohlc']} | "
            f"DKW {b['missing_dkw']} -> {a['missing_dkw']} | "
            f"抓取symbol {r['symbols_targeted']} (成功{f['ok']}/失败{f['fail']})"
        )

    hist = get_history_stats()
    print(f"\n历史库统计: symbols={hist.get('total_symbols')} records={hist.get('total_records')} range={hist.get('date_range')}")


if __name__ == "__main__":
    main()

