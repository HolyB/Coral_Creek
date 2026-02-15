"""
检查 Supabase 同步状态
"""
import os
import sys
from datetime import datetime

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import get_db, get_scan_date_counts_supabase, get_scan_date_counts

def check_sync():
    print("🔍 Checking Supabase vs SQLite synchronization...")
    
    # 1. Get local counts (SQLite)
    local_counts = get_scan_date_counts(limit=5)
    print("\n[Local SQLite] Recent Dates:")
    for row in local_counts:
        print(f"  {row['scan_date']}: {row['count']} signals")
        
    # 2. Get remote counts (Supabase)
    print("\n[Remote Supabase] Recent Dates:")
    try:
        remote_counts = get_scan_date_counts_supabase(limit=5)
        if not remote_counts:
            print("  (No data found or connection failed)")
        for row in remote_counts:
            # Supabase return dict or object? Usually dict from postgrest
            d = row.get('scan_date')
            c = row.get('count')
            print(f"  {d}: {c} signals")
            
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        print("   Make sure SUPABASE_URL and SUPABASE_KEY are set in .env")

if __name__ == "__main__":
    check_sync()
