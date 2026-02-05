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
               company_name, industry, market_cap, cap_category,
               heima_daily, heima_weekly, heima_monthly,
               juedi_daily, juedi_weekly, juedi_monthly
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
    
    def sqlite_bool_to_python(val):
        """正确转换 SQLite 布尔值 (bytes b'\x00'/b'\x01' 或 int 0/1)"""
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, bytes):
            return val == b'\x01'  # b'\x00' -> False, b'\x01' -> True
        if isinstance(val, (int, float)):
            return val == 1
        return bool(val)
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        records = []
        
        for row in batch:
            record = {
                'symbol': row['symbol'],
                'scan_date': row['scan_date'],
                'price': row['price'],
                'turnover_m': row['turnover_m'],
                'blue_daily': row['blue_daily'],
                'blue_weekly': row['blue_weekly'],
                'blue_monthly': row['blue_monthly'],
                'adx': row['adx'],
                'volatility': row['volatility'],
                'is_heima': sqlite_bool_to_python(row['is_heima']),
                'is_juedi': sqlite_bool_to_python(row['is_juedi']),
                'heima_daily': sqlite_bool_to_python(row['heima_daily']) if 'heima_daily' in row.keys() else None,
                'heima_weekly': sqlite_bool_to_python(row['heima_weekly']) if 'heima_weekly' in row.keys() else None,
                'heima_monthly': sqlite_bool_to_python(row['heima_monthly']) if 'heima_monthly' in row.keys() else None,
                'juedi_daily': sqlite_bool_to_python(row['juedi_daily']) if 'juedi_daily' in row.keys() else None,
                'juedi_weekly': sqlite_bool_to_python(row['juedi_weekly']) if 'juedi_weekly' in row.keys() else None,
                'juedi_monthly': sqlite_bool_to_python(row['juedi_monthly']) if 'juedi_monthly' in row.keys() else None,
                'market': row['market'] or 'US',
                'company_name': row['company_name'],
                'industry': row['industry'],
            }
            # 可选字段 - 只在存在时添加
            if row['market_cap'] is not None:
                record['market_cap'] = row['market_cap']
            if row['cap_category'] is not None:
                record['cap_category'] = row['cap_category']
            records.append(record)
        
        try:
            # 先尝试完整记录
            supabase.table('scan_results').upsert(
                records, 
                on_conflict='symbol,scan_date,market'
            ).execute()
            total += len(records)
        except Exception as e:
            # 如果失败，尝试不带新字段
            if 'cap_category' in str(e) or 'market_cap' in str(e):
                print("⚠️ Supabase 表缺少 market_cap/cap_category 列，跳过这些字段...")
                for rec in records:
                    rec.pop('market_cap', None)
                    rec.pop('cap_category', None)
                try:
                    supabase.table('scan_results').upsert(
                        records, 
                        on_conflict='symbol,scan_date,market'
                    ).execute()
                    total += len(records)
                except Exception as e2:
                    errors += 1
                    print(f"❌ Batch error: {e2}")
            else:
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
