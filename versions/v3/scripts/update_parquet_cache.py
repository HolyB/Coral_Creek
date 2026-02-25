import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from ml.data_cache import DataCache

def update_us_parquet_for_date(target_date: str):
    """
    使用 Polygon 的 grouped_daily 接口拉取全市场数据，并高速追加到本地对应的 Parquet 缓存中。
    这避免了 daily scan 中分别发起 12,000+ 次 list_aggs 拉取请求引起的极长耗时和限流风险。
    """
    from polygon import RESTClient
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("❌ Missing POLYGON_API_KEY")
        return
        
    client = RESTClient(api_key)
    cache = DataCache()
    us_dir = cache.us_dir
    us_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📡 Fetching grouped daily for {target_date}...")
    try:
        aggs = client.get_grouped_daily_aggs(target_date)
        if not aggs:
            print(f"⚠️ No data found for {target_date}. Maybe a market holiday.")
            return
    except Exception as e:
        print(f"❌ API Error fetching grouped daily: {e}")
        return
        
    # parse new data
    updates = {}
    for a in aggs:
        sym = getattr(a, "ticker", "")
        if not sym or len(sym) > 5: continue
        sym = sym.replace("/", "_").upper()
        
        # safely convert to float properly
        def _sf(val):
            try: return float(val) if val is not None else 0.0
            except: return 0.0
            
        updates[sym] = {
            'date': pd.Timestamp(target_date + " 16:00:00-05:00").tz_convert('America/New_York'),
            'open': _sf(getattr(a, "open", None)),
            'high': _sf(getattr(a, "high", None)),
            'low': _sf(getattr(a, "low", None)),
            'close': _sf(getattr(a, "close", None)),
            'volume': _sf(getattr(a, "volume", None)),
            'vwap': _sf(getattr(a, "vwap", None)) if getattr(a, "vwap", None) is not None else _sf(getattr(a, "close", None))
        }
        
    print(f"✅ Downloaded daily cross-section for {len(updates)} tickers.")
    
    # helper for multithreaded fast writing
    def append_to_parquet(sym, new_row_dict):
        file_path = cache.get_file_path(sym, "US")
        new_df = pd.DataFrame([new_row_dict])
        new_df['symbol'] = sym
        
        if file_path.exists():
            try:
                # read existing
                existing_df = pd.read_parquet(file_path)
                # align timezone if naive existing
                if existing_df['date'].dt.tz is None:
                    # old parquets lack timezone
                    new_df['date'] = pd.Timestamp(target_date + " 21:00:00") # matching naive structure roughly
                    
                # drop exact duplicate dates if any
                existing_df = existing_df[existing_df['date'].dt.strftime('%Y-%m-%d') != target_date]
                
                # combine
                merged = pd.concat([existing_df, new_df], ignore_index=True)
                merged = merged.sort_values('date').reset_index(drop=True)
                merged.to_parquet(file_path, engine='pyarrow', compression='snappy')
            except Exception as e:
                # corrupted parquet, overwrite entirely using polygon next time
                file_path.unlink(missing_ok=True)
        else:
            # Note: We do NOT create strictly from 1 day to avoid breaking history needs
            # just skip, data cache will naturally lazy load the full 5-years later when requested!
            pass

    print(f"💾 Merging data into local Parquet caches...")
    symbols_present = list(updates.keys())
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(append_to_parquet, sym, updates[sym]): sym for sym in symbols_present}
        for _ in tqdm(futures, total=len(futures)):
            pass
            
    print("✅ Successfully updated local Parquet snapshot.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='Date to append (YYYY-MM-DD)')
    args = parser.parse_args()
    update_us_parquet_for_date(args.date)
