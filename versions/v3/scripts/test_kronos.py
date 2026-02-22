import sys
import os
import pandas as pd

# Add versions/v3 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.kronos_integration import get_kronos_engine
    from ml.data_cache import DataCache
except ImportError as e:
    print(f"Path err: {e}")
    sys.exit(1)

cache = DataCache()
# Fetch 300 days of data for Apple
df = cache.get_stock_history("AAPL", market="US", days=300)

if df is None or len(df) == 0:
    print("No data for AAPL")
    sys.exit(0)

print(f"Loaded {len(df)} rows for AAPL:")
print(df.tail(2))

print("\nInvoking Kronos for next 10 days...")
engine = get_kronos_engine()

# df has 'open','high','low','close','volume','date'
# we rename 'date' to 'timestamps' and lowercase all columns to match KronosEngine signature
df_input = df.copy()
df_input.rename(columns={'date': 'timestamps', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
# If original columns are Capitalized:
df_input.columns = [c.lower() for c in df_input.columns]

if 'timestamps' not in df_input.columns and 'date' in df_input.columns:
    df_input.rename(columns={'date': 'timestamps'}, inplace=True)

pred = engine.predict_future_klines(df_input, pred_len=10, temperature=0.5, top_p=0.8)
if pred is not None:
    print("\n✅ Kronos Prediction output for AAPL:")
    print(pred)
else:
    print("Failed")
