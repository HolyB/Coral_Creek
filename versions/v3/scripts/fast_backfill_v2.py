#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超快回填 v2 — 预加载 + multiprocessing
========================================

核心优化:
1. 一次性从 stock_history 批量读取当天所有股票数据到内存
2. 用 multiprocessing.Pool 真并行计算指标 (绕过 GIL)
3. 每个 worker 只做 CPU 计算，不碰 SQLite

用法:
    PYTHONPATH=. python scripts/fast_backfill_v2.py --market CN --start 2025-06-01 --end 2025-09-30 --workers 6
"""

import os, sys, time, sqlite3
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))
os.environ['PYTHONPATH'] = str(V3_DIR)  # 确保 fork 的子进程也能找到模块

import pandas as pd
import numpy as np
from db.stock_history import get_history_db_path


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def _get_trading_dates(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    while d <= end_d:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _get_existing_dates(market: str) -> set:
    db_path = V3_DIR / "db" / "coral_creek.db"
    conn = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in conn.execute(
            "SELECT DISTINCT scan_date FROM scan_results WHERE market = ?", (market,)
        ).fetchall()}
    except:
        return set()
    finally:
        conn.close()


def _batch_load_histories(target_date: str, market: str, days_back: int = 750):
    """
    一次性从 stock_history 批量读取所有股票数据。
    返回 {symbol: DataFrame}
    """
    hist_path = get_history_db_path()
    conn = sqlite3.connect(hist_path)
    
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = (dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # 一次 SQL 读全部
        df = pd.read_sql_query(
            """SELECT symbol, trade_date, open, high, low, close, volume 
               FROM stock_history 
               WHERE market = ? AND trade_date BETWEEN ? AND ?
               ORDER BY symbol, trade_date""",
            conn,
            params=(market, start, target_date)
        )
        
        if df.empty:
            return {}
        
        # 按 symbol 分组
        result = {}
        for symbol, group in df.groupby('symbol'):
            sdf = group.copy()
            sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])
            sdf.set_index('trade_date', inplace=True)
            sdf.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            }, inplace=True)
            sdf = sdf[['Open', 'High', 'Low', 'Close', 'Volume']]
            if len(sdf) >= 60:
                result[symbol] = sdf
        
        return result
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Worker: 纯 CPU 计算 (不碰磁盘)
# ─────────────────────────────────────────────

_V3_DIR_STR = str(V3_DIR)  # resolved at import time

def _analyze_one(args):
    """在子进程中分析一只股票 (数据通过 args 传入)"""
    # 先确保路径 (fork 子进程可能丢失)
    import sys as _sys
    _sys.path.insert(0, _V3_DIR_STR)
    
    symbol, df_dict, target_date, market = args
    
    try:
        # 重建 DataFrame
        df = pd.DataFrame(df_dict)
        df.index = pd.to_datetime(df.index)
        
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        df = df[df.index <= target_dt]
        
        if len(df) < 60:
            return None
        
        # 延迟导入指标函数
        from indicator_utils import (
            calculate_blue_signal_series, calculate_heima_signal_series,
            calculate_phantom_indicator, calculate_volume_profile_metrics
        )
        
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        volumes = df['Volume'].values
        
        curr_price = closes[-1]
        latest_date = df.index[-1]
        
        # 日线 BLUE
        day_blue = calculate_blue_signal_series(opens, highs, lows, closes)
        day_blue_val = day_blue[-1] if len(day_blue) > 0 else 0
        
        # LIRED + PINK
        lired_daily_val = 0.0
        pink_daily_val = 50.0
        try:
            phantom = calculate_phantom_indicator(opens, highs, lows, closes, volumes)
            lired_daily_val = float(phantom['lired'][-1]) if len(phantom['lired']) > 0 else 0.0
            pink_daily_val = float(phantom['pink'][-1]) if len(phantom['pink']) > 0 else 50.0
        except:
            pass
        
        # 周线
        df_weekly = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        week_blue_val = 0
        if len(df_weekly) >= 10:
            week_blue = calculate_blue_signal_series(
                df_weekly['Open'].values, df_weekly['High'].values,
                df_weekly['Low'].values, df_weekly['Close'].values
            )
            week_blue_val = week_blue[-1] if len(week_blue) > 0 else 0
        
        # 月线
        df_monthly = df.resample('ME').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
        
        month_blue_val = 0
        if len(df_monthly) >= 6:
            month_blue = calculate_blue_signal_series(
                df_monthly['Open'].values, df_monthly['High'].values,
                df_monthly['Low'].values, df_monthly['Close'].values
            )
            month_blue_val = month_blue[-1] if len(month_blue) > 0 else 0
        
        # 黑马/掘地 日线
        heima_arr, juedi_arr = calculate_heima_signal_series(highs, lows, closes, opens)
        heima_daily = bool(heima_arr[-1]) if len(heima_arr) > 0 else False
        juedi_daily = bool(juedi_arr[-1]) if len(juedi_arr) > 0 else False
        
        # 黑马/掘地 周线
        heima_weekly = False
        juedi_weekly = False
        if len(df_weekly) >= 10:
            heima_w, juedi_w = calculate_heima_signal_series(
                df_weekly['High'].values, df_weekly['Low'].values,
                df_weekly['Close'].values, df_weekly['Open'].values
            )
            heima_weekly = bool(heima_w[-1]) if len(heima_w) > 0 else False
            juedi_weekly = bool(juedi_w[-1]) if len(juedi_w) > 0 else False
        
        # 多空王 (与 scan_service.py 完全一致的内联计算)
        duokongwang_buy = False
        duokongwang_sell = False
        try:
            n = len(closes)
            if n >= 30:
                def _ema_local(vals, period):
                    alpha = 2.0 / (period + 1.0)
                    out = [float(vals[0])]
                    for v in vals[1:]:
                        out.append(alpha * float(v) + (1 - alpha) * out[-1])
                    return out

                def _sma_cn_local(vals, period, m=1):
                    out = [float(vals[0])]
                    for x in vals[1:]:
                        out.append((m * float(x) + (period - m) * out[-1]) / float(period))
                    return out

                opens_proxy = [float(closes[0])] + [float(x) for x in closes[:-1]]
                up = _ema_local(highs, 13)
                dw = _ema_local(lows, 13)

                # KDJ(14,3,3)
                rsv = []
                for i in range(n):
                    s = max(0, i - 13)
                    llv = float(np.min(lows[s:i + 1]))
                    hhv = float(np.max(highs[s:i + 1]))
                    rsv.append(50.0 if hhv <= llv else (float(closes[i]) - llv) / (hhv - llv) * 100.0)
                k = _sma_cn_local(rsv, 3, 1)
                d_line = _sma_cn_local(k, 3, 1)
                j = [3.0 * k[i] - 2.0 * d_line[i] for i in range(n)]

                # RSI2(9)
                lc = [float(closes[0])] + [float(x) for x in closes[:-1]]
                up_move = [max(float(closes[i]) - lc[i], 0.0) for i in range(n)]
                abs_move = [abs(float(closes[i]) - lc[i]) for i in range(n)]
                rsi_num = _sma_cn_local(up_move, 9, 1)
                rsi_den = _sma_cn_local(abs_move, 9, 1)
                rsi2 = [(rsi_num[i] / rsi_den[i] * 100.0) if rsi_den[i] > 1e-12 else 50.0 for i in range(n)]

                # 九转计数
                nt = [0] * n
                nt0 = [0] * n
                for i in range(n):
                    a1 = i >= 4 and float(closes[i]) > float(closes[i - 4])
                    b1 = i >= 4 and float(closes[i]) < float(closes[i - 4])
                    nt[i] = (nt[i - 1] + 1) if (a1 and i > 0) else (1 if a1 else 0)
                    nt0[i] = (nt0[i - 1] + 1) if (b1 and i > 0) else (1 if b1 else 0)

                i = n - 1
                cond = (
                    (float(closes[i]) > opens_proxy[i] and (opens_proxy[i] > up[i] or float(closes[i]) < dw[i]))
                    or (float(closes[i]) < opens_proxy[i] and (opens_proxy[i] < dw[i] or float(closes[i]) > up[i]))
                )
                cond1 = bool(i >= 1 and up[i] > up[i - 1] and dw[i] > dw[i - 1])
                cond2 = bool(i >= 1 and up[i] < up[i - 1] and dw[i] < dw[i - 1])

                # balanced profile
                j_cross_level, j_oversold_prev = 30.0, 22.0
                rsi_prev_th, rsi_now_th, nine_min = 24.0, 20.0, 9

                kdj_cross_up = bool(i >= 1 and j[i - 1] <= j_cross_level and j[i] > j_cross_level)
                kdj_oversold_turn = bool(i >= 1 and j[i - 1] < j_oversold_prev and j[i] > j[i - 1])
                rsi_oversold_turn = bool(i >= 1 and rsi2[i - 1] <= rsi_prev_th and rsi2[i] > rsi_now_th)
                nine_down_exhaust = bool(nt0[i] >= nine_min)
                duokongwang_buy = bool(
                    (cond and cond1 and (kdj_cross_up or rsi_oversold_turn))
                    or kdj_oversold_turn
                    or rsi_oversold_turn
                    or nine_down_exhaust
                )

                kdj_overheat_fade = bool(i >= 1 and ((j[i - 1] >= 100.0 and j[i] < 95.0) or (j[i - 1] >= 90.0 and j[i] < j[i - 1] - 8.0)))
                rsi_overbought_turn = bool(i >= 1 and rsi2[i - 1] >= 79.0 and rsi2[i] < 80.0)
                nine_up_exhaust = bool(nt[i] >= 9 and i >= 1 and float(closes[i]) < float(closes[i - 1]))
                duokongwang_sell = bool((cond and cond2) or kdj_overheat_fade or rsi_overbought_turn or nine_up_exhaust)
        except Exception:
            duokongwang_buy = False
            duokongwang_sell = False
        
        # 筹码分布 (与 chart_utils.quick_chip_analysis 一致)
        chip_bottom_ratio = None
        chip_poc_position = None
        chip_max_pct = None
        chip_is_bottom_peak = False
        chip_is_strong_peak = False
        try:
            n_chip = len(closes)
            if n_chip >= 30:
                price_min = float(np.min(lows))
                price_max = float(np.max(highs))
                price_range = price_max - price_min
                if price_range > 0:
                    bins = 70
                    bin_size = price_range / bins
                    vp = np.zeros(bins)
                    bin_centers = np.linspace(price_min, price_max, bins + 1)
                    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
                    decay = 0.97
                    for ci in range(n_chip):
                        tw = decay ** (n_chip - 1 - ci)
                        wv = float(volumes[ci]) * tw
                        h_val, l_val, c_val = float(highs[ci]), float(lows[ci]), float(closes[ci])
                        if h_val == l_val or bin_size == 0:
                            bi = min(max(int((c_val - price_min) / bin_size), 0), bins - 1)
                            vp[bi] += wv
                        else:
                            sb = max(int((l_val - price_min) / bin_size), 0)
                            eb = min(int((h_val - price_min) / bin_size), bins - 1)
                            cb = min(max(int((c_val - price_min) / bin_size), sb), eb)
                            if sb == eb:
                                vp[sb] += wv
                            else:
                                for b in range(sb, eb + 1):
                                    d2c = abs(b - cb)
                                    md = max(cb - sb, eb - cb, 1)
                                    vp[b] += wv * (1.0 - 0.8 * (d2c / md))
                    tv = np.sum(vp)
                    if tv > 0:
                        poc_idx = int(np.argmax(vp))
                        poc_price = bin_centers[poc_idx]
                        chip_max_pct = float(np.max(vp) / tv * 100)
                        chip_poc_position = float((poc_price - price_min) / price_range * 100)
                        b30 = price_min + price_range * 0.30
                        chip_bottom_ratio = float(np.sum(vp[bin_centers <= b30]) / tv)
                        chip_is_strong_peak = bool(chip_poc_position < 30 and chip_bottom_ratio > 0.50 and chip_max_pct > 5)
                        chip_is_bottom_peak = bool(chip_poc_position < 35 and chip_bottom_ratio > 0.35)
        except Exception:
            pass
        
        # 筹码获利盘 (profit_ratio + vp_rating)
        profit_ratio = 0.0
        vp_rating = 'Normal'
        try:
            vp_res = calculate_volume_profile_metrics(closes, volumes, curr_price)
            profit_ratio = vp_res['profit_ratio']
            if profit_ratio > 0.9:
                vp_rating = 'Excellent'
            elif profit_ratio > 0.7:
                vp_rating = 'Good'
            elif profit_ratio < 0.3:
                vp_rating = 'Poor'
        except Exception:
            pass
        
        return {
            'Symbol': symbol,
            'Date': latest_date.strftime('%Y-%m-%d'),
            'scan_date': target_date,
            'Market': market,
            'Price': float(curr_price),
            'Volume': int(volumes[-1]) if len(volumes) > 0 else 0,
            'Blue_Daily': float(day_blue_val),
            'Blue_Weekly': float(week_blue_val),
            'Blue_Monthly': float(month_blue_val),
            'Lired_Daily': float(lired_daily_val),
            'Pink_Daily': float(pink_daily_val),
            'Is_Heima': heima_daily,
            'Is_Juedi': juedi_daily,
            'Heima_Daily': heima_daily,
            'Heima_Weekly': heima_weekly,
            'Juedi_Daily': juedi_daily,
            'Juedi_Weekly': juedi_weekly,
            'Duokongwang_Buy': duokongwang_buy,
            'Duokongwang_Sell': duokongwang_sell,
            'Chip_Bottom_Ratio': chip_bottom_ratio,
            'Chip_POC_Position': chip_poc_position,
            'Chip_Max_Pct': chip_max_pct,
            'Chip_Is_Bottom_Peak': chip_is_bottom_peak,
            'Chip_Is_Strong_Peak': chip_is_strong_peak,
            'Profit_Ratio': float(profit_ratio),
            'VP_Rating': vp_rating,
        }
    except Exception as e:
        if symbol.endswith('.SZ') and symbol.startswith('000001'):
            import traceback
            traceback.print_exc()
        return None


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def scan_one_day(target_date: str, market: str, n_workers: int) -> int:
    """扫描一天: 批量加载 → 多进程计算 → 批量写入"""
    
    # 1. 批量加载 (单次 SQL, 主进程)
    t0 = time.time()
    all_histories = _batch_load_histories(target_date, market)
    load_time = time.time() - t0
    
    if not all_histories:
        return 0
    
    n_stocks = len(all_histories)
    
    # 2. 准备 worker 参数 (传 dict 避免序列化问题)
    work_args = [
        (sym, df.to_dict(), target_date, market)
        for sym, df in all_histories.items()
    ]
    
    # 3. 多进程并行计算
    t1 = time.time()
    results = []
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_analyze_one, work_args, chunksize=100):
            if result is not None:
                results.append(result)
    calc_time = time.time() - t1
    
    # 4. 批量写入 (主进程, 直接 SQLite 绕过 get_db 开销)
    t2 = time.time()
    if results:
        import sqlite3 as _sq
        db_path = str(V3_DIR / "db" / "coral_creek.db")
        conn = _sq.connect(db_path)
        cols = ['symbol', 'scan_date', 'price', 'blue_daily', 'blue_weekly', 'blue_monthly',
                'lired_daily', 'pink_daily', 'is_heima', 'is_juedi',
                'heima_daily', 'heima_weekly', 'juedi_daily', 'juedi_weekly',
                'duokongwang_buy', 'duokongwang_sell',
                'chip_bottom_ratio', 'chip_poc_position', 'chip_max_pct',
                'chip_is_bottom_peak', 'chip_is_strong_peak',
                'profit_ratio', 'vp_rating',
                'market', 'updated_at']
        placeholders = ','.join(['?' for _ in cols])
        sql = f"INSERT OR IGNORE INTO scan_results ({','.join(cols)}) VALUES ({placeholders})"
        rows = []
        for r in results:
            rows.append((
                r.get('Symbol'), r.get('scan_date') or r.get('Date'),
                r.get('Price'), r.get('Blue_Daily'), r.get('Blue_Weekly'), r.get('Blue_Monthly'),
                r.get('Lired_Daily'), r.get('Pink_Daily'),
                r.get('Is_Heima'), r.get('Is_Juedi'),
                r.get('Heima_Daily'), r.get('Heima_Weekly'),
                r.get('Juedi_Daily'), r.get('Juedi_Weekly'),
                r.get('Duokongwang_Buy'), r.get('Duokongwang_Sell'),
                r.get('Chip_Bottom_Ratio'), r.get('Chip_POC_Position'), r.get('Chip_Max_Pct'),
                r.get('Chip_Is_Bottom_Peak'), r.get('Chip_Is_Strong_Peak'),
                r.get('Profit_Ratio'), r.get('VP_Rating'),
                r.get('Market', market),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
        conn.executemany(sql, rows)
        conn.commit()
        conn.close()
    write_time = time.time() - t2
    
    total = time.time() - t0
    speed = n_stocks / total if total > 0 else 0
    print(f"   [{target_date}] {len(results)} signals / {n_stocks} stocks "
          f"({total:.0f}s = load {load_time:.0f}s + calc {calc_time:.0f}s + write {write_time:.0f}s) "
          f"[{speed:.1f} stock/s]")
    
    return len(results)


def run(start: str, end: str, market: str, n_workers: int):
    print(f"\n{'='*65}")
    print(f"⚡ 超快回填 v2 — {market} ({start} ~ {end}) workers={n_workers}")
    print(f"{'='*65}")
    
    all_dates = _get_trading_dates(start, end)
    existing = _get_existing_dates(market)
    todo = sorted([d for d in all_dates if d not in existing], reverse=True)  # 从最新开始
    
    print(f"   总: {len(all_dates)}天, 已完成: {len(existing)}, 待扫: {len(todo)}")
    
    if not todo:
        print("   ✅ 全部完成!")
        return
    
    total_signals = 0
    t_start = time.time()
    
    for i, date in enumerate(todo):
        signals = scan_one_day(date, market, n_workers)
        total_signals += signals
        
        elapsed = time.time() - t_start
        avg_per_day = elapsed / (i + 1)
        eta = avg_per_day * (len(todo) - i - 1)
        print(f"   进度: {i+1}/{len(todo)} | ETA: {eta/60:.0f}min")
        sys.stdout.flush()
    
    print(f"\n✅ 完成! {len(todo)}天, {total_signals} signals, {(time.time()-t_start)/60:.1f}min")


if __name__ == '__main__':
    import argparse
    
    default_workers = max(2, (os.cpu_count() or 4) - 2)
    
    parser = argparse.ArgumentParser(description='超快回填 v2')
    parser.add_argument('--market', required=True, choices=['CN', 'US'])
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--workers', type=int, default=default_workers)
    
    args = parser.parse_args()
    mp.set_start_method('fork', force=True)
    run(args.start, args.end, args.market, args.workers)
