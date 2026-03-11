#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同步本地 scan_results 到 Supabase（最近 N 天，按天倒序）
"""

import os, sys, time, sqlite3, math
from datetime import datetime, timedelta
from pathlib import Path

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

from dotenv import load_dotenv
load_dotenv(V3_DIR / '.env')


def run(days_back: int = 180, batch_size: int = 500):
    from supabase import create_client
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
    if not url or not key:
        print("❌ SUPABASE_URL or SUPABASE_KEY not set")
        return
    
    sb = create_client(url, key)
    db_path = str(V3_DIR / "db" / "coral_creek.db")
    conn = sqlite3.connect(db_path)
    
    # 获取要同步的日期列表（倒序，最新的先）
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT scan_date FROM scan_results WHERE scan_date >= ? ORDER BY scan_date DESC",
        (cutoff,)
    ).fetchall()]
    
    print(f"📊 Syncing {len(dates)} days to Supabase (newest first)...")
    
    # 列定义 — 只包含 Supabase 表中存在的列
    sync_cols = [
        'symbol', 'scan_date', 'price', 'blue_daily', 'blue_weekly', 'blue_monthly',
        'lired_daily', 'pink_daily', 'adx', 'volatility', 'turnover_m',
        'is_heima', 'is_juedi', 'heima_daily', 'heima_weekly', 'juedi_daily', 'juedi_weekly',
        'duokongwang_buy', 'duokongwang_sell',
        'strat_d_trend', 'strat_c_resonance', 'legacy_signal',
        'regime', 'adaptive_thresh', 'vp_rating', 'profit_ratio',
        'market'
    ]
    
    cursor = conn.execute("PRAGMA table_info(scan_results)")
    local_cols = {row[1] for row in cursor.fetchall()}
    actual_cols = [c for c in sync_cols if c in local_cols]
    col_str = ','.join(actual_cols)
    
    bool_cols = {'is_heima', 'is_juedi', 'heima_daily', 'heima_weekly', 
                 'juedi_daily', 'juedi_weekly', 'duokongwang_buy', 'duokongwang_sell',
                 'strat_d_trend', 'strat_c_resonance', 'legacy_signal',
                 'chip_is_bottom_peak', 'chip_is_strong_peak'}
    
    total_uploaded = 0
    total_errors = 0
    t_start = time.time()
    
    for di, scan_date in enumerate(dates):
        # 1. 清理该日期的 Supabase 数据
        try:
            sb.table('scan_results').delete().eq('scan_date', scan_date).execute()
        except:
            pass  # 可能没数据
        
        # 2. 读取本地数据
        rows = conn.execute(f"SELECT {col_str} FROM scan_results WHERE scan_date = ?", (scan_date,)).fetchall()
        
        if not rows:
            continue
        
        # 3. 分批上传
        day_uploaded = 0
        for i in range(0, len(rows), batch_size):
            batch = []
            for row in rows[i:i+batch_size]:
                d = {}
                for j, col in enumerate(actual_cols):
                    val = row[j]
                    if val is None:
                        continue
                    if col in bool_cols:
                        val = bool(val)
                    elif isinstance(val, float):
                        if math.isnan(val) or math.isinf(val):
                            val = 0.0
                        else:
                            val = round(val, 6)
                    d[col] = val
                batch.append(d)
            
            try:
                sb.table('scan_results').insert(batch).execute()
                day_uploaded += len(batch)
            except Exception as e:
                total_errors += len(batch)
                if total_errors <= 3:
                    print(f"   ⚠️ {scan_date} batch error: {str(e)[:80]}")
        
        total_uploaded += day_uploaded
        elapsed = time.time() - t_start
        speed = total_uploaded / elapsed if elapsed > 0 else 0
        eta = (len(dates) - di - 1) * (elapsed / (di + 1)) / 60 if di > 0 else 0
        
        print(f"   [{di+1}/{len(dates)}] {scan_date}: {day_uploaded}/{len(rows)} | 累计 {total_uploaded:,} | ETA: {eta:.0f}min")
        sys.stdout.flush()
    
    conn.close()
    elapsed = time.time() - t_start
    print(f"\n✅ 完成! 上传 {total_uploaded:,} 行, 错误 {total_errors}, 耗时 {elapsed/60:.1f}min")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--batch', type=int, default=500)
    args = parser.parse_args()
    run(args.days, args.batch)
