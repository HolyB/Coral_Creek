#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supabase → 本地 SQLite 自动补缺
================================
检查本地 scan_results 和 Supabase 的日期差距，自动补缺。
每日扫描脚本调用此脚本，确保本地数据与云端同步。

用法:
    PYTHONPATH=. python scripts/sync_from_supabase.py
    PYTHONPATH=. python scripts/sync_from_supabase.py --days 30  # 只检查最近30天
"""

import os, sys, sqlite3, time
from pathlib import Path
from datetime import datetime, timedelta

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

# 加载 .env
env_file = V3_DIR / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k] = v


COLS = ('symbol,scan_date,price,turnover_m,blue_daily,blue_weekly,blue_monthly,'
        'adx,volatility,is_heima,is_juedi,strat_d_trend,strat_c_resonance,'
        'legacy_signal,regime,adaptive_thresh,vp_rating,profit_ratio,wave_phase,'
        'market,heima_daily,heima_weekly,heima_monthly,juedi_daily,juedi_weekly,'
        'juedi_monthly,day_high,day_low,day_close,duokongwang_buy,duokongwang_sell,'
        'ml_rank_score,lired_daily,pink_daily,company_name,industry,market_cap,'
        'cap_category,stop_loss,shares_rec,risk_reward_score')

COL_LIST = [c.strip() for c in COLS.split(',')]


def _row_to_tuple(row):
    return tuple(row.get(c) for c in COL_LIST)


def sync(days_back=90):
    from db.supabase_db import get_supabase
    sb = get_supabase()
    if not sb:
        print("❌ Supabase 不可用（检查 .env 中的 SUPABASE_URL/SUPABASE_KEY）")
        return

    db_path = str(V3_DIR / "db" / "coral_creek.db")
    conn = sqlite3.connect(db_path)

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    # 1. 获取本地每日计数
    local_counts = {}
    for r in conn.execute(
        "SELECT scan_date, market, COUNT(*) FROM scan_results "
        "WHERE scan_date >= ? GROUP BY scan_date, market", (cutoff,)
    ).fetchall():
        local_counts[(r[0], r[1])] = r[2]

    # 2. 获取 Supabase 每日计数（分页）
    remote_counts = {}
    for market in ['US', 'CN']:
        offset = 0
        while True:
            try:
                r = sb.table('scan_results').select('scan_date') \
                    .eq('market', market).gte('scan_date', cutoff) \
                    .order('scan_date').range(offset, offset + 999).execute()
            except Exception as e:
                print(f"  ⚠️ 查询失败: {e}")
                break
            if not r.data:
                break
            for row in r.data:
                key = (row['scan_date'], market)
                remote_counts[key] = remote_counts.get(key, 0) + 1
            if len(r.data) < 1000:
                break
            offset += 1000

    # 3. 找需要补填的日期（远端有但本地缺或少 >50%）
    to_fill = []
    for (dt, mkt), remote_cnt in sorted(remote_counts.items()):
        local_cnt = local_counts.get((dt, mkt), 0)
        if local_cnt < remote_cnt * 0.5:  # 本地少于远端 50%
            to_fill.append((dt, mkt, local_cnt, remote_cnt))

    if not to_fill:
        print(f"✅ 本地数据已同步（最近 {days_back} 天，{len(remote_counts)} 个日期-市场组合）")
        return

    print(f"📥 需补填: {len(to_fill)} 个日期-市场组合")

    # 4. 逐日拉取
    t0 = time.time()
    total = 0
    placeholders = ','.join(['?'] * len(COL_LIST))
    insert_sql = f"INSERT OR REPLACE INTO scan_results ({COLS}) VALUES ({placeholders})"

    for i, (dt, mkt, local_cnt, remote_cnt) in enumerate(to_fill):
        offset = 0
        day_count = 0
        while True:
            try:
                r = sb.table('scan_results').select(COLS) \
                    .eq('scan_date', dt).eq('market', mkt) \
                    .range(offset, offset + 999).execute()
            except Exception as e:
                print(f"  ⚠️ {dt} {mkt}: {e}")
                break
            if not r.data:
                break
            for row in r.data:
                try:
                    conn.execute(insert_sql, _row_to_tuple(row))
                    day_count += 1
                except:
                    pass
            if len(r.data) < 1000:
                break
            offset += 1000

        conn.commit()
        total += day_count

        if (i + 1) % 20 == 0 or i == len(to_fill) - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(to_fill)} 日期, {total:,} 行 ({elapsed:.0f}s)")

    conn.close()
    print(f"\n✅ 同步完成: {total:,} 行, {len(to_fill)} 天 ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=90, help='检查最近N天')
    args = p.parse_args()
    sync(args.days)
