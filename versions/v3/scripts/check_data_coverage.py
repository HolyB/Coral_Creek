#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check scan_results date coverage in Supabase and local SQLite."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.supabase_db import get_scanned_dates_supabase
from db.database import get_scanned_dates

us_supa = get_scanned_dates_supabase(market="US") or []
cn_supa = get_scanned_dates_supabase(market="CN") or []
us_local = get_scanned_dates(market="US") or []
cn_local = get_scanned_dates(market="CN") or []

print("=== Supabase scan_results ===")
print(f"US: {len(us_supa)} dates")
if us_supa:
    print(f"  range: {us_supa[-1]} ~ {us_supa[0]}")
    print(f"  first 5: {us_supa[:5]}")
    print(f"  last 5:  {us_supa[-5:]}")
print(f"CN: {len(cn_supa)} dates")
if cn_supa:
    print(f"  range: {cn_supa[-1]} ~ {cn_supa[0]}")
    print(f"  first 5: {cn_supa[:5]}")
    print(f"  last 5:  {cn_supa[-5:]}")

print()
print("=== Local SQLite scan_results ===")
print(f"US: {len(us_local)} dates")
if us_local:
    print(f"  range: {us_local[-1]} ~ {us_local[0]}")
print(f"CN: {len(cn_local)} dates")
if cn_local:
    print(f"  range: {cn_local[-1]} ~ {cn_local[0]}")

# Also check stock_history
print()
print("=== stock_history (local) ===")
import sqlite3
from db.stock_history import get_history_db_path
try:
    hconn = sqlite3.connect(get_history_db_path())
    hcur = hconn.cursor()
    hcur.execute("SELECT market, COUNT(DISTINCT symbol) as symbols, COUNT(*) as rows, MIN(trade_date), MAX(trade_date) FROM stock_history GROUP BY market")
    for r in hcur.fetchall():
        print(f"  {r[0]}: {r[1]} symbols, {r[2]} rows, {r[3]} ~ {r[4]}")
    hconn.close()
except Exception as e:
    print(f"  Error: {e}")
