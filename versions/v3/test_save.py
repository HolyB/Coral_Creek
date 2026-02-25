import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.scan_service import run_scan_for_date

res = run_scan_for_date('2026-01-05', market='US', limit=10, save_to_db=True)
print([r['Symbol'] for r in res])
