import sys
from scripts.fast_backfill_2025 import process_symbol

res = process_symbol('AAPL', '2025-01-01', '2025-11-30')
print(f"Signals for AAPL: {len(res)}")
from scripts.fast_backfill_2025 import insert_scan_result, get_connection
conn = get_connection()
cursor = conn.cursor()
for r in res:
    insert_scan_result(r)
conn.commit()
conn.close()
print("Saved AAPL successfully.")
