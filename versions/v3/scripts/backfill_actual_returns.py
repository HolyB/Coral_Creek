#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回填 ML 每日推荐的实际收益
================================
- 从 stock_history.db 或 Polygon/yfinance 获取买入后的实际价格
- 计算 actual_10d, actual_30d
- 如未到期，计算当前浮盈

用法:
    PYTHONPATH=. python scripts/backfill_actual_returns.py
    PYTHONPATH=. python scripts/backfill_actual_returns.py --market US
"""
import os, sys, sqlite3, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

PICKS_DB = os.path.join(parent_dir, 'db', 'ml_daily_picks.db')
HIST_DB = os.path.join(parent_dir, 'db', 'stock_history.db')


def get_price_on_date(symbol, target_date, market, hist_conn):
    """Get closing price on or near target_date from stock_history"""
    # Try exact date first, then nearest before
    row = hist_conn.execute(
        '''SELECT close FROM stock_history 
           WHERE symbol=? AND trade_date<=? 
           ORDER BY trade_date DESC LIMIT 1''',
        (symbol, target_date)
    ).fetchone()
    if row:
        return float(row[0])
    return None


def get_latest_price(symbol, market, hist_conn):
    """Get the most recent price from stock_history"""
    row = hist_conn.execute(
        '''SELECT close, trade_date FROM stock_history 
           WHERE symbol=? ORDER BY trade_date DESC LIMIT 1''',
        (symbol,)
    ).fetchone()
    if row:
        return float(row[0]), row[1]
    return None, None


def add_trading_days(start_date, n_days):
    """Approximate trading days (skip weekends)"""
    d = pd.Timestamp(start_date)
    added = 0
    while added < n_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d.strftime('%Y-%m-%d')


def backfill_returns(market=None):
    """Backfill actual returns for all picks"""
    if not os.path.exists(PICKS_DB):
        print("❌ ml_daily_picks.db not found")
        return
    
    picks_conn = sqlite3.connect(PICKS_DB)
    
    # Check table
    tables = [r[0] for r in picks_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    tbl = 'ml_picks_v2' if 'ml_picks_v2' in tables else 'ml_picks'
    
    # Get all picks that need updating
    where = f"WHERE market='{market}'" if market else ""
    picks = picks_conn.execute(f'''
        SELECT id, symbol, date, price, market, holding_period, actual_10d, actual_30d
        FROM {tbl} {where}
        ORDER BY date DESC
    ''').fetchall()
    
    print(f"📊 Total picks to check: {len(picks)}")
    
    # Open stock_history
    if not os.path.exists(HIST_DB):
        print("❌ stock_history.db not found")
        picks_conn.close()
        return
    
    hist_conn = sqlite3.connect(HIST_DB)
    
    today = datetime.now().strftime('%Y-%m-%d')
    updated_10d = 0
    updated_30d = 0
    updated_current = 0
    
    for pick_id, symbol, pick_date, buy_price, mkt, hold_period, existing_10d, existing_30d in picks:
        if not buy_price or buy_price <= 0:
            continue
        
        # Calculate target dates
        target_10d = add_trading_days(pick_date, 10)
        target_30d = add_trading_days(pick_date, 30)
        days_since = (pd.Timestamp(today) - pd.Timestamp(pick_date)).days
        
        updates = {}
        
        # --- actual_10d ---
        if existing_10d is None:
            if days_since >= 14:  # ~10 trading days
                price_10d = get_price_on_date(symbol, target_10d, mkt, hist_conn)
                if price_10d:
                    actual = (price_10d / buy_price - 1) * 100
                    updates['actual_10d'] = round(actual, 2)
                    updated_10d += 1
            else:
                # Not expired yet -> show current unrealized
                latest_price, latest_date = get_latest_price(symbol, mkt, hist_conn)
                if latest_price:
                    actual = (latest_price / buy_price - 1) * 100
                    updates['actual_10d'] = round(actual, 2)
                    updated_current += 1
        
        # --- actual_30d ---
        if existing_30d is None:
            if days_since >= 42:  # ~30 trading days
                price_30d = get_price_on_date(symbol, target_30d, mkt, hist_conn)
                if price_30d:
                    actual = (price_30d / buy_price - 1) * 100
                    updates['actual_30d'] = round(actual, 2)
                    updated_30d += 1
            else:
                latest_price, latest_date = get_latest_price(symbol, mkt, hist_conn)
                if latest_price:
                    actual = (latest_price / buy_price - 1) * 100
                    updates['actual_30d'] = round(actual, 2)
                    updated_current += 1
        
        if updates:
            set_clause = ', '.join(f"{k}={v}" for k, v in updates.items())
            picks_conn.execute(f"UPDATE {tbl} SET {set_clause} WHERE id=?", (pick_id,))
    
    picks_conn.commit()
    hist_conn.close()
    picks_conn.close()
    
    print(f"✅ 回填完成:")
    print(f"   10d 确定收益: {updated_10d}")
    print(f"   30d 确定收益: {updated_30d}")
    print(f"   当前浮盈:     {updated_current}")


def show_summary(market='US'):
    """Show prediction vs actual summary"""
    if not os.path.exists(PICKS_DB):
        return
    
    conn = sqlite3.connect(PICKS_DB)
    tbl = 'ml_picks_v2'
    
    df = pd.read_sql(f'''
        SELECT date, symbol, tier, price, primary_pred, pred_5d, pred_10d, pred_30d,
               actual_10d, actual_30d, holding_period
        FROM {tbl} WHERE market=? AND actual_10d IS NOT NULL
        ORDER BY date DESC
    ''', conn, params=[market])
    conn.close()
    
    if df.empty:
        print(f"⚠️ {market} 暂无实际收益数据")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 ML 预测 vs 实际 ({market})")
    print(f"{'='*60}")
    print(f"  样本数: {len(df)}")
    
    if market == 'US':
        pred_col = 'pred_10d'
        actual_col = 'actual_10d'
    else:
        pred_col = 'pred_30d' if 'pred_30d' in df.columns else 'pred_5d'
        actual_col = 'actual_30d' if df['actual_30d'].notna().sum() > 0 else 'actual_10d'
    
    valid = df[df[pred_col].notna() & df[actual_col].notna()]
    if valid.empty:
        print("  暂无可对比的样本")
        return
    
    avg_pred = valid[pred_col].mean()
    avg_actual = valid[actual_col].mean()
    
    # Direction accuracy
    correct_dir = ((valid[pred_col] > 0) & (valid[actual_col] > 0)) | ((valid[pred_col] <= 0) & (valid[actual_col] <= 0))
    dir_acc = correct_dir.mean() * 100
    
    # Win rate (actual > 0)
    win_rate = (valid[actual_col] > 0).mean() * 100
    
    print(f"  平均预测: {avg_pred:+.1f}%")
    print(f"  平均实际: {avg_actual:+.1f}%")
    print(f"  方向准确率: {dir_acc:.0f}%")
    print(f"  胜率: {win_rate:.0f}%")
    
    # By tier
    print(f"\n  按层级:")
    for tier, g in valid.groupby('tier'):
        n = len(g)
        avg_p = g[pred_col].mean()
        avg_a = g[actual_col].mean()
        wr = (g[actual_col] > 0).mean() * 100
        print(f"    {tier}: {n}只, 预测{avg_p:+.1f}%, 实际{avg_a:+.1f}%, 胜率{wr:.0f}%")
    
    # Latest day
    latest = df['date'].max()
    latest_df = df[df['date'] == latest]
    print(f"\n  📅 最新 ({latest}): {len(latest_df)} picks")
    for _, r in latest_df.head(5).iterrows():
        pred = r.get(pred_col, 0) or 0
        actual = r.get(actual_col, 0) or 0
        emoji = '✅' if actual > 0 else '❌'
        print(f"    {emoji} {r['symbol']:8s} 预测{pred:+.1f}% → 实际{actual:+.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default=None, choices=['US', 'CN'])
    args = parser.parse_args()
    
    backfill_returns(args.market)
    
    for m in (['US', 'CN'] if not args.market else [args.market]):
        show_summary(m)
