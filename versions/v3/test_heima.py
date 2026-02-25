import sys, os
from datetime import datetime
import pandas as pd
from ml.data_cache import DataCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.scan_service import analyze_stock_for_date

res = analyze_stock_for_date('TSLA', '2026-02-05', market='US')
print(res)
