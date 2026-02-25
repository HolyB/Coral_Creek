import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_connection, get_stock_history

def get_signal_records(db_path, year=2025):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(f"SELECT symbol, generated_at as date, price, strategy FROM signals WHERE generated_at LIKE '{year}%'")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def restore_scan_results(year=2025):
    print(f"🔄 Restoring scan_results for {year} from signals...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sig_db = os.path.join(base_dir, 'db', 'signals.db')
    hist_db = os.path.join(base_dir, 'db', 'stock_history.db')
    
    # 1. Get all signals (which imply a successful scan result existed)
    signals = get_signal_records(sig_db, year)
    print(f"  - Found {len(signals)} signal records.")
    
    # Group by (symbol, date) to invoke efficiently
    import collections
    grouped = collections.defaultdict(list) # (symbol, date) -> strategies
    
    prices_map = {} # (symbol, date) -> price from signal (fallback)
    
    for s in signals:
        key = (s['symbol'], s['date'])
        grouped[key].append(s['strategy'])
        prices_map[key] = s['price']
        
    unique_keys = list(grouped.keys())
    print(f"  - {len(unique_keys)} unique (symbol, date) entries to verify/restore.")
    
    # 2. Check existing scan_results
    conn = get_connection()
    cursor = conn.cursor()
    
    # Batch check
    # Too many params for IN clause?
    # Iterate and check existence? Or bulk insert ignoring duplicates.
    # UNIQUE(symbol, scan_date, market) constraint exists.
    # We can use INSERT OR IGNORE.
    
    # But we need data to insert! (OHLC, Indicators).
    # Indicators are lost (except what we can infer).
    # We set duokongwang_buy = 1 if strategy has '多空王'.
    # We fetch OHLC from stock_history.
    
    batch_size = 500
    restored_count = 0
    
    for i in range(0, len(unique_keys), batch_size):
        batch_keys = unique_keys[i:i+batch_size]
        
        # Prepare data
        insert_data = []
        
        # We need to fetch OHLC for these symbols/dates?
        # That's slow (1 query per symbol).
        # We can just use the price from signal as close, and estimate high/low?
        # For backtest 'return', we need accurate close.
        # Signals have price. 'price' in signal is usually close (or entry).
        
        # Creating minimal rows
        for sym, dated in batch_keys:
            strats = grouped[(sym, dated)]
            
            is_dkw = any('多空王' in s for s in strats)
            is_heima = any('黑马' in s for s in strats)
            is_blue = any('BLUE' in s for s in strats)
            
            price = prices_map.get((sym, dated), 0)
            
            # minimal columns: symbol, scan_date, price, duokongwang_buy, is_heima, blue_daily
            # We assume day_close = price.
            
            insert_data.append((
                sym, dated, price, 
                1 if is_dkw else 0,
                1 if is_heima else 0,
                180 if is_blue else 0, # Fake blue value > 180 trigger
                'US'
            ))
            
        try:
            cursor.executemany('''
                INSERT OR IGNORE INTO scan_results 
                (symbol, scan_date, price, duokongwang_buy, is_heima, blue_daily, market)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            inserted = cursor.rowcount
            restored_count += inserted
            # Typically rowcount for INSERT OR IGNORE is reliable in recent sqlite3
            
        except Exception as e:
            print(f"Error inserting batch: {e}")
            
        if i % 5000 == 0 and i > 0:
            conn.commit()
            print(f"  - Processed {i} keys...")
            
    conn.commit()
    conn.close()
    
    print(f"✨ Restore complete. Inserted {restored_count} rows into scan_results.")

if __name__ == "__main__":
    restore_scan_results(2025)
