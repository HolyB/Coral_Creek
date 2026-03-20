#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三年历史回填脚本 - 两阶段:
  阶段1: 用 Polygon grouped daily API 批量拉每日全市场 OHLCV -> stock_history
  阶段2: 从本地 stock_history 读取数据，跑信号策略 -> scan_results -> candidate_tracking
"""
import os, sys, time, sqlite3
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(SCRIPT_DIR)  # versions/v3
sys.path.insert(0, V3_DIR)

# Supabase 表结构已同步，正常使用

from db.database import insert_scan_result, init_db, get_scanned_dates
from db.stock_history import get_history_db_path
from indicator_utils import (
    calculate_blue_signal_series, calculate_heima_signal_series,
    calculate_adx_series, calculate_volume_profile_metrics,
    calculate_volatility, analyze_elliott_wave_proxy, analyze_chanlun_proxy,
)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")


# ─────────────────────────────────────────────
# 阶段 1: 批量拉 grouped daily -> stock_history
# ─────────────────────────────────────────────

def _get_trading_dates(start: str, end: str):
    """生成工作日列表 (排除周末)"""
    from datetime import date as dt_date
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    dates = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:  # Mon-Fri
            dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def _ensure_history_table(conn):
    from db.stock_history import init_history_db
    init_history_db()


def _dates_already_in_history(conn, market="US"):
    """返回已拉取的日期集合"""
    cur = conn.execute(
        "SELECT DISTINCT trade_date FROM stock_history WHERE market = ?", (market,)
    )
    return {r[0] for r in cur.fetchall()}


def pull_grouped_daily(start_date: str, end_date: str, market: str = "US"):
    """
    阶段1: 逐天拉 Polygon grouped daily -> stock_history (SQLite)
    每天 1 次 API 调用，返回全市场 OHLCV。
    """
    if market != "US":
        print(f"[阶段1] grouped daily 仅支持 US 市场，跳过 {market}")
        return

    from polygon import RESTClient
    client = RESTClient(POLYGON_API_KEY)

    hist_path = get_history_db_path()
    conn = sqlite3.connect(hist_path)
    _ensure_history_table(conn)
    existing = _dates_already_in_history(conn, market)

    all_dates = _get_trading_dates(start_date, end_date)
    todo = [d for d in all_dates if d not in existing]
    print(f"[阶段1] US grouped daily: 共 {len(all_dates)} 工作日, 已有 {len(existing)}, 待拉取 {len(todo)}")

    for i, d in enumerate(todo):
        t0 = time.time()
        try:
            aggs = client.get_grouped_daily_aggs(d)
            if not aggs:
                print(f"  [{i+1}/{len(todo)}] {d}: 无数据 (可能非交易日)")
                continue

            rows = []
            for a in aggs:
                sym = getattr(a, "ticker", "")
                if not sym or len(sym) > 5:
                    continue
                rows.append((
                    sym, market, d,
                    getattr(a, "open", None),
                    getattr(a, "high", None),
                    getattr(a, "low", None),
                    getattr(a, "close", None),
                    getattr(a, "volume", None),
                    None,  # turnover
                ))

            conn.executemany(
                """INSERT OR IGNORE INTO stock_history
                   (symbol, market, trade_date, open, high, low, close, volume, turnover)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] {d}: {len(rows)} tickers ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] {d}: ERROR - {e}")
            time.sleep(1)

    conn.close()
    print(f"[阶段1] US grouped daily 完成")


def pull_cn_daily(start_date: str, end_date: str):
    """
    阶段1 CN: 用 Tushare daily API 逐天拉 A股全市场 -> stock_history
    """
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        print("[阶段1] TUSHARE_TOKEN 未设置，跳过 CN")
        return
    ts.set_token(token)
    pro = ts.pro_api()

    hist_path = get_history_db_path()
    conn = sqlite3.connect(hist_path)
    _ensure_history_table(conn)
    existing = _dates_already_in_history(conn, "CN")

    all_dates = _get_trading_dates(start_date, end_date)
    todo = [d for d in all_dates if d not in existing]
    print(f"[阶段1] CN daily: 共 {len(all_dates)} 工作日, 已有 {len(existing)}, 待拉取 {len(todo)}")

    call_count = 0
    for i, d in enumerate(todo):
        t0 = time.time()
        td = d.replace("-", "")
        try:
            call_count += 1
            if call_count % 450 == 0:
                print(f"  Tushare 限流暂停 60s...")
                time.sleep(60)

            df = pro.daily(trade_date=td, fields="ts_code,open,high,low,close,vol")
            if df is None or df.empty:
                print(f"  [{i+1}/{len(todo)}] {d}: 无数据")
                continue

            rows = []
            for _, r in df.iterrows():
                rows.append((
                    r["ts_code"], "CN", d,
                    r.get("open"), r.get("high"), r.get("low"), r.get("close"),
                    r.get("vol"),
                    None,  # turnover
                ))

            conn.executemany(
                """INSERT OR IGNORE INTO stock_history
                   (symbol, market, trade_date, open, high, low, close, volume, turnover)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] {d}: {len(rows)} tickers ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] {d}: ERROR - {e}")
            time.sleep(2)

    conn.close()
    print(f"[阶段1] CN daily 完成")



def scan_from_local_history(target_date: str, market: str = "US", max_workers: int = 15):
    """
    阶段2: 调用 scan_service 的逻辑（支持 Parquet 缓存）
    注意：这里不再强依赖 stock_history SQLite 表，而是依赖 DataCache
    """
    from services.scan_service import run_scan_for_date
    
    # max_workers 调高到 12 以加速计算（DataCache hit），SQLite 锁风险适中
    results = run_scan_for_date(target_date, market, max_workers=12, save_to_db=True)
    return len(results) if results else 0


def run_phase2(start_date: str, end_date: str, market: str = "US"):
    """阶段2: 批量扫描全部日期"""
    # 获取需要扫描的日期范围
    all_dates = _get_trading_dates(start_date, end_date)
    
    # 检查已完成的日期
    # from services.scan_service import _get_main_db_path
    def _get_main_db_path_local():
        return os.path.join(V3_DIR, "db", "coral_creek.db")
        
    main_db = _get_main_db_path_local()
    conn = sqlite3.connect(main_db)
    try:
        cur = conn.execute(
            "SELECT DISTINCT scan_date FROM scan_results WHERE market = ? ORDER BY scan_date",
            (market,)
        )
        existing_scan_dates = {r[0] for r in cur.fetchall()}
    except Exception:
        existing_scan_dates = set()
    conn.close()

    todo = [d for d in all_dates if d not in existing_scan_dates]
    print(f"[阶段2] {market} 信号扫描 (DataCache版): 共 {len(all_dates)} 交易日, 已扫描 {len(existing_scan_dates)}, 待扫描 {len(todo)}")

    for i, d in enumerate(todo):
        t0 = time.time()
        # 调用新版逻辑
        found = scan_from_local_history(d, market, max_workers=15)
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(todo)}] {d}: {found} signals ({elapsed:.1f}s)")
        sys.stdout.flush()



# ─────────────────────────────────────────────
# 阶段 3: SQLite -> Supabase 批量同步
# ─────────────────────────────────────────────

def sync_to_supabase(start_date: str, end_date: str, market: str = "US"):
    """阶段3: 将 SQLite scan_results 批量同步到 Supabase"""
    supa_url = os.environ.get('SUPABASE_URL')
    supa_key = os.environ.get('SUPABASE_KEY')
    if not supa_url or not supa_key:
        print("[阶段3] SUPABASE_URL/KEY 未设置，跳过同步")
        return

    from supabase import create_client
    client = create_client(supa_url, supa_key)
    from db.database import get_db

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM scan_results 
            WHERE scan_date BETWEEN ? AND ? AND market = ?
            ORDER BY scan_date
        """, (start_date, end_date, market))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    print(f"[阶段3] 同步 {len(rows)} 条到 Supabase ({start_date} ~ {end_date}, {market})")

    # 批量 upsert，每批 500 条
    batch_size = 500
    success = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        records = []
        for row in batch:
            record = dict(zip(columns, row))
            # 移除 id 列（Supabase 自生成）
            record.pop('id', None)
            records.append(record)
        
        try:
            client.table('scan_results').upsert(
                records, on_conflict='symbol,scan_date'
            ).execute()
            success += len(batch)
            print(f"  已同步 {success}/{len(rows)} 条")
            sys.stdout.flush()
        except Exception as e:
            print(f"  批次 {i//batch_size+1} 失败: {e}")

    print(f"[阶段3] Supabase 同步完成: {success}/{len(rows)}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="历史回填")
    parser.add_argument("--start", default="2021-02-12")
    parser.add_argument("--end", default="2026-02-12")
    parser.add_argument("--market", default="US", choices=["US", "CN", "ALL"])
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    markets = ["US", "CN"] if args.market == "ALL" else [args.market]

    for m in markets:
        if args.phase in ("1", "all"):
            if m == "US":
                pull_grouped_daily(args.start, args.end, "US")
            elif m == "CN":
                pull_cn_daily(args.start, args.end)

        if args.phase in ("2", "all"):
            run_phase2(args.start, args.end, m)

        if args.phase in ("3", "all"):
            sync_to_supabase(args.start, args.end, m)

    print("\n=== 全部完成 ===")

