#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同步本地 SQLite 数据到 Supabase
用于 GitHub Actions 扫描后同步
"""
import os
import sys
import sqlite3

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ supabase module not installed")


def sync_to_supabase(db_path: str = None, days_back: int = 3):
    """同步最近N天的数据到 Supabase"""
    
    if not SUPABASE_AVAILABLE:
        print("❌ Supabase module not available")
        return False
    
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    
    if not url or not key:
        print("❌ SUPABASE_URL or SUPABASE_KEY not set")
        return False
    
    # 使用默认数据库路径
    if not db_path:
        db_path = os.path.join(parent_dir, 'db', 'coral_creek.db')
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print(f"🔗 Connecting to Supabase...")
    supabase = create_client(url, key)
    
    print(f"📂 Reading from: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取最近N天的数据
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    cursor.execute('''
        SELECT symbol, scan_date, price, turnover_m, blue_daily, blue_weekly, 
               blue_monthly, adx, volatility, is_heima, is_juedi, market, 
               company_name, industry
        FROM scan_results 
        WHERE scan_date >= ?
        ORDER BY scan_date DESC
    ''', (cutoff_date,))
    
    rows = cursor.fetchall()
    print(f"📊 Found {len(rows)} records from {cutoff_date}")
    
    if not rows:
        print("⚠️ No recent data to sync")
        conn.close()
        return True
    
    # 批量 upsert
    batch_size = 100
    total = 0
    errors = 0
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        records = []
        
        for row in batch:
            records.append({
                'symbol': row['symbol'],
                'scan_date': row['scan_date'],
                'price': row['price'],
                'turnover_m': row['turnover_m'],
                'blue_daily': row['blue_daily'],
                'blue_weekly': row['blue_weekly'],
                'blue_monthly': row['blue_monthly'],
                'adx': row['adx'],
                'volatility': row['volatility'],
                'is_heima': bool(row['is_heima']) if row['is_heima'] is not None else None,
                'is_juedi': bool(row['is_juedi']) if row['is_juedi'] is not None else None,
                'market': row['market'] or 'US',
                'company_name': row['company_name'],
                'industry': row['industry']
            })
        
        try:
            supabase.table('scan_results').upsert(
                records, 
                on_conflict='symbol,scan_date,market'
            ).execute()
            total += len(records)
        except Exception as e:
            errors += 1
            print(f"❌ Batch error: {e}")
    
    conn.close()
    
    print(f"✅ Synced {total} records to Supabase (errors: {errors})")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync SQLite to Supabase')
    parser.add_argument('--days', type=int, default=3, help='Days to sync')
    parser.add_argument('--db', type=str, default=None, help='Database path')
    
    args = parser.parse_args()
    
    success = sync_to_supabase(db_path=args.db, days_back=args.days)
    sys.exit(0 if success else 1)
