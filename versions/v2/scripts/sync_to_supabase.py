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
    
    # 检查哪些列存在
    cursor.execute("PRAGMA table_info(scan_results)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    
    # 基础列
    base_cols = ['symbol', 'scan_date', 'price', 'turnover_m', 'blue_daily', 'blue_weekly', 
                 'blue_monthly', 'adx', 'volatility', 'is_heima', 'is_juedi', 'market', 
                 'company_name', 'industry', 'market_cap', 'cap_category']
    
    # 可选列 (可能不存在于旧数据库)
    optional_cols = ['heima_daily', 'heima_weekly', 'heima_monthly',
                     'juedi_daily', 'juedi_weekly', 'juedi_monthly',
                     # 策略/分析字段
                     'wave_phase', 'wave_desc', 'chan_signal', 'chan_desc',
                     'stop_loss', 'shares_rec', 'regime',
                     'strat_d_trend', 'strat_c_resonance', 'legacy_signal',
                     'adaptive_thresh', 'vp_rating', 'profit_ratio', 'risk_reward_score']
    
    # 只选择存在的列
    select_cols = [c for c in base_cols if c in existing_cols]
    select_cols += [c for c in optional_cols if c in existing_cols]
    
    query = f"SELECT {', '.join(select_cols)} FROM scan_results WHERE scan_date >= ? ORDER BY scan_date DESC"
    cursor.execute(query, (cutoff_date,))
    
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
            row_keys = row.keys()
            # 布尔类型字段
            bool_cols = {'is_heima', 'is_juedi', 'heima_daily', 'heima_weekly', 'heima_monthly',
                         'juedi_daily', 'juedi_weekly', 'juedi_monthly',
                         'strat_d_trend', 'strat_c_resonance', 'legacy_signal'}
            record = {}
            for col in select_cols:
                if col not in row_keys:
                    continue
                val = row[col]
                if val is None:
                    continue  # 不上传 None 值
                if col in bool_cols:
                    record[col] = sqlite_bool_to_python(val)
                elif isinstance(val, bytes):
                    # 兜底：任何未知 bytes 字段当 bool 处理
                    record[col] = sqlite_bool_to_python(val)
                else:
                    record[col] = val
            # 确保必需字段
            record.setdefault('market', 'US')
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
