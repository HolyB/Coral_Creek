#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补筹码特征 — 给已有 scan_results 补 profit_ratio + vp_rating
============================================================

用法:
    PYTHONPATH=. python scripts/backfill_chip_features.py --market CN --workers 12
    PYTHONPATH=. python scripts/backfill_chip_features.py --market US --workers 12
"""

import os, sys, time, sqlite3
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))
os.environ['PYTHONPATH'] = str(V3_DIR)

import pandas as pd
import numpy as np
from db.stock_history import get_history_db_path

_V3_DIR_STR = str(V3_DIR)


def _get_dates_needing_chip(market: str):
    """获取需要补筹码的日期列表"""
    db_path = V3_DIR / "db" / "coral_creek.db"
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """SELECT DISTINCT scan_date FROM scan_results 
               WHERE market = ? AND (profit_ratio IS NULL OR profit_ratio = 0)
               ORDER BY scan_date DESC""",
            (market,)
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _batch_load_histories(target_date: str, market: str, days_back: int = 200):
    """批量加载历史数据"""
    hist_path = get_history_db_path()
    conn = sqlite3.connect(hist_path)
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = (dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        df = pd.read_sql_query(
            """SELECT symbol, trade_date, close, volume 
               FROM stock_history 
               WHERE market = ? AND trade_date BETWEEN ? AND ?
               ORDER BY symbol, trade_date""",
            conn,
            params=(market, start, target_date)
        )
        
        if df.empty:
            return {}
        
        result = {}
        for symbol, group in df.groupby('symbol'):
            sdf = group.copy()
            if len(sdf) >= 30:
                result[symbol] = {
                    'closes': sdf['close'].values.astype(float),
                    'volumes': sdf['volume'].values.astype(float),
                    'current_price': float(sdf['close'].values[-1])
                }
        return result
    finally:
        conn.close()


def _get_symbols_for_date(target_date: str, market: str):
    """获取某天需要补筹码的 symbol 列表"""
    db_path = V3_DIR / "db" / "coral_creek.db"
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """SELECT symbol, price FROM scan_results 
               WHERE market = ? AND scan_date = ? AND (profit_ratio IS NULL OR profit_ratio = 0)""",
            (market, target_date)
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()


def _calc_chip(args):
    """计算单个 symbol 的筹码特征"""
    import sys as _sys
    _sys.path.insert(0, _V3_DIR_STR)
    
    symbol, closes, volumes, current_price = args
    try:
        from indicator_utils import calculate_volume_profile_metrics
        
        vp_res = calculate_volume_profile_metrics(closes, volumes, current_price)
        profit_ratio = vp_res['profit_ratio']
        
        if profit_ratio > 0.9:
            vp_rating = 'Excellent'
        elif profit_ratio > 0.7:
            vp_rating = 'Good'
        elif profit_ratio < 0.3:
            vp_rating = 'Poor'
        else:
            vp_rating = 'Normal'
        
        return (symbol, profit_ratio, vp_rating)
    except:
        return None


def process_one_day(target_date: str, market: str, n_workers: int):
    """处理一天的数据"""
    t0 = time.time()
    
    # 1. 获取需要补的 symbols
    symbols_prices = _get_symbols_for_date(target_date, market)
    if not symbols_prices:
        return 0
    
    # 2. 加载历史数据
    histories = _batch_load_histories(target_date, market)
    
    # 3. 准备计算参数
    work_args = []
    for symbol in symbols_prices:
        if symbol in histories:
            h = histories[symbol]
            work_args.append((symbol, h['closes'], h['volumes'], h['current_price']))
    
    if not work_args:
        return 0
    
    # 4. 并行计算
    results = []
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_calc_chip, work_args, chunksize=200):
            if result is not None:
                results.append(result)
    
    # 5. 批量更新
    if results:
        db_path = str(V3_DIR / "db" / "coral_creek.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        
        sql = """UPDATE scan_results SET profit_ratio = ?, vp_rating = ?
                 WHERE symbol = ? AND scan_date = ? AND market = ?"""
        rows = [(pr, vr, sym, target_date, market) for sym, pr, vr in results]
        conn.executemany(sql, rows)
        conn.commit()
        conn.close()
    
    elapsed = time.time() - t0
    print(f"   [{target_date}] updated {len(results)}/{len(symbols_prices)} ({elapsed:.1f}s)")
    return len(results)


def run(market: str, n_workers: int):
    print(f"\n{'='*65}")
    print(f"🔧 补筹码特征 — {market}, workers={n_workers}")
    print(f"{'='*65}")
    
    dates = _get_dates_needing_chip(market)
    print(f"   待补日期: {len(dates)} 天")
    
    if not dates:
        print("   ✅ 全部已有筹码数据!")
        return
    
    total_updated = 0
    t_start = time.time()
    
    for i, date in enumerate(dates):
        updated = process_one_day(date, market, n_workers)
        total_updated += updated
        
        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        eta = avg * (len(dates) - i - 1)
        print(f"   进度: {i+1}/{len(dates)} | 已更新: {total_updated} | ETA: {eta/60:.0f}min")
        sys.stdout.flush()
    
    print(f"\n✅ 完成! {len(dates)}天, 更新 {total_updated} 行, {(time.time()-t_start)/60:.1f}min")


if __name__ == '__main__':
    import argparse
    
    default_workers = max(2, (os.cpu_count() or 4) - 2)
    
    parser = argparse.ArgumentParser(description='补筹码特征')
    parser.add_argument('--market', required=True, choices=['CN', 'US'])
    parser.add_argument('--workers', type=int, default=default_workers)
    
    args = parser.parse_args()
    mp.set_start_method('fork', force=True)
    run(args.market, args.workers)
