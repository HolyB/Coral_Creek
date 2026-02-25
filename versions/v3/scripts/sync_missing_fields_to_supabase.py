#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 SQLite 中有值但 Supabase 中为 NULL 的字段同步过去。
主要是: stop_loss, shares_rec, wave_phase, wave_desc, chan_signal, chan_desc,
        regime, profit_ratio, vp_rating, risk_reward_score, strat_d_trend,
        strat_c_resonance, legacy_signal, adaptive_thresh
"""
import os
import sys
import sqlite3
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

os.environ.setdefault('SUPABASE_URL', 'https://worqpdsypymnzqjbidyz.supabase.co')
os.environ.setdefault('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndvcnFwZHN5cHltbnpxamJpZHl6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk4MTA5MjksImV4cCI6MjA4NTM4NjkyOX0.UzE54Q4QB1mQZqRp_jn4BWGFOtWN3GAscrmGpHpMG9U')

from db.supabase_db import get_supabase, _to_json_native

# 需要同步的字段
SYNC_FIELDS = [
    'stop_loss', 'shares_rec', 'wave_phase', 'wave_desc',
    'chan_signal', 'chan_desc', 'regime', 'profit_ratio',
    'vp_rating', 'risk_reward_score', 'strat_d_trend',
    'strat_c_resonance', 'legacy_signal', 'adaptive_thresh',
]

def main():
    db_path = os.path.join(parent_dir, 'db', 'coral_creek.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sb = get_supabase()
    if not sb:
        print("❌ Supabase not available")
        return

    # 获取所有日期
    cursor.execute("SELECT DISTINCT scan_date FROM scan_results ORDER BY scan_date DESC")
    dates = [r['scan_date'] for r in cursor.fetchall()]
    print(f"📅 Found {len(dates)} dates in SQLite")

    total_updated = 0
    total_errors = 0

    for date_idx, scan_date in enumerate(dates):
        cursor.execute(
            f"SELECT symbol, market, {', '.join(SYNC_FIELDS)} FROM scan_results WHERE scan_date = ?",
            (scan_date,)
        )
        rows = cursor.fetchall()

        batch_count = 0
        for row in rows:
            row_dict = dict(row)
            symbol = row_dict.pop('symbol')
            market = row_dict.pop('market') or 'US'

            # 只取非 None 的字段
            update = {}
            for field in SYNC_FIELDS:
                val = row_dict.get(field)
                if val is not None:
                    # SQLite bool 可能存为 bytes 或 int
                    if isinstance(val, bytes):
                        val = bool(int.from_bytes(val, 'little'))
                    elif isinstance(val, int) and field in ('strat_d_trend', 'strat_c_resonance', 'legacy_signal'):
                        val = bool(val)
                    update[field] = val

            if not update:
                continue

            update = _to_json_native(update)

            try:
                sb.table('scan_results').update(update).eq(
                    'symbol', symbol
                ).eq('scan_date', scan_date).eq('market', market).execute()
                batch_count += 1
            except Exception as e:
                total_errors += 1
                if total_errors <= 5:
                    print(f"  ⚠️ {symbol}/{scan_date}: {e}")

            # 速率限制
            if batch_count % 50 == 0:
                time.sleep(0.5)

        total_updated += batch_count
        print(f"  [{date_idx+1}/{len(dates)}] {scan_date}: synced {batch_count}/{len(rows)} rows")

    conn.close()
    print(f"\n✅ Done! Updated {total_updated} rows, {total_errors} errors")


if __name__ == '__main__':
    main()
