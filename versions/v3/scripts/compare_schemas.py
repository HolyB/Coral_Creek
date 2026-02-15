#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 Supabase 和本地 SQLite 的 scan_results 表结构"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase import create_client
import sqlite3
from db.database import get_db

# Supabase
supa_url = os.environ.get('SUPABASE_URL')
supa_key = os.environ.get('SUPABASE_KEY')
client = create_client(supa_url, supa_key)

result = client.table('scan_results').select('*').limit(1).execute()
if result.data:
    supa_cols = set(result.data[0].keys())
    print(f"Supabase columns ({len(supa_cols)}):")
    for c in sorted(supa_cols):
        print(f"  {c}")
else:
    supa_cols = set()
    print("Supabase: no data")

# SQLite
with get_db() as conn:
    cur = conn.execute("PRAGMA table_info(scan_results)")
    sqlite_cols = {row[1] for row in cur.fetchall()}
    print(f"\nSQLite columns ({len(sqlite_cols)}):")
    for c in sorted(sqlite_cols):
        print(f"  {c}")

# 对比
missing_in_supa = sqlite_cols - supa_cols
missing_in_sqlite = supa_cols - sqlite_cols

if missing_in_supa:
    print(f"\n❌ Missing in Supabase ({len(missing_in_supa)}):")
    for c in sorted(missing_in_supa):
        print(f"  {c}")

if missing_in_sqlite:
    print(f"\n⚠️ Extra in Supabase ({len(missing_in_sqlite)}):")
    for c in sorted(missing_in_sqlite):
        print(f"  {c}")

if not missing_in_supa and not missing_in_sqlite:
    print("\n✅ Schemas match!")
