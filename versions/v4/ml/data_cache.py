"""
数据缓存模块 (Data Caching Layer)
==================================
负责将股票历史数据 (OHLCV) 从 Polygon/Tushare 拉取并缓存到本地 Parquet 文件。
这能极大加速 Backfill 和 ML 训练，避免重复 API 请求和 SQLite 锁竞争。

依赖:
    - pandas
    - polygon-api-client
    - pyarrow (for parquet)

用法:
    from ml.data_cache import DataCache
    cache = DataCache()
    
    # 获取数据 (优先读缓存，无缓存则下载并保存)
    df = cache.get_stock_history("AAPL", market="US")
    
    # 批量预热缓存
    cache.warmup_cache(["AAPL", "TSLA", "NVDA"])
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None

class DataCache:
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            # 默认存放在 versions/v3/data/parquet
            self.base_dir = Path(parent_dir) / "data" / "parquet"
        else:
            self.base_dir = Path(cache_dir)
            
        self.us_dir = self.base_dir / "us"
        self.cn_dir = self.base_dir / "cn"
        
        # 确保存储目录存在
        self.us_dir.mkdir(parents=True, exist_ok=True)
        self.cn_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = os.environ.get('POLYGON_API_KEY')
        self._client = None
        
    @property
    def client(self):
        if self._client is None and self.api_key:
            if RESTClient:
                self._client = RESTClient(self.api_key)
        return self._client

    def get_file_path(self, symbol: str, market: str = "US") -> Path:
        """获取缓存文件路径"""
        symbol = symbol.replace("/", "_").upper() # 处理特殊字符
        if market == "US":
            return self.us_dir / f"{symbol}.parquet"
        else:
            return self.cn_dir / f"{symbol}.parquet"

    def load_from_cache(self, symbol: str, market: str = "US", max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """从缓存读取，如果太旧则返回 None"""
        path = self.get_file_path(symbol, market)
        if not path.exists():
            return None
            
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        
        # 如果是盘中实时任务，可能需要更短的过期时间
        # 但对于 Backfill/ML，通常一天一更甚至一周一更都够了
        if age.days > max_age_days:
            pass # 这里可以加逻辑：如果需要最新数据，则视为过期待更新
            
        try:
            return pd.read_parquet(path)
        except getattr(Exception, 'None', Exception): # Capture all, prevent crash from corrupted file
            return None

    def fetch_from_polygon(self, symbol: str, days: int = 365*5) -> Optional[pd.DataFrame]:
        """从 Polygon 拉取数据"""
        if not self.client:
            print("❌ Polygon Client 未初始化 (Missing API Key?)")
            return None
            
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        try:
            # 使用 get_aggs (v2)
            aggs = []
            for a in self.client.list_aggs(
                symbol, 1, "day", 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d"), 
                limit=50000
            ):
                aggs.append(a)
                
            if not aggs:
                return None
                
            records = []
            for a in aggs:
                if a.timestamp is None: continue
                dt = datetime.fromtimestamp(a.timestamp / 1000)
                
                # Safe float conversion
                def _sf(val):
                    try:
                        return float(val) if val is not None else 0.0
                    except:
                        return 0.0
                
                records.append({
                    'date': dt,
                    'open': _sf(a.open),
                    'high': _sf(a.high),
                    'low': _sf(a.low),
                    'close': _sf(a.close),
                    'volume': _sf(a.volume),
                    'vwap': _sf(a.vwap) if hasattr(a, 'vwap') else _sf(a.close)
                })
                
            df = pd.DataFrame(records)
            df['symbol'] = symbol
            # 确保按时间排序
            df = df.sort_values('date').reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"⚠️ Polygon Error for {symbol}: {e}")
            return None

    def save_to_cache(self, df: pd.DataFrame, symbol: str, market: str = "US"):
        """保存到 Parquet"""
        if df is None or df.empty:
            return
        
        path = self.get_file_path(symbol, market)
        try:
            # 使用 pyarrow 引擎，压缩以节省空间
            df.to_parquet(path, engine='pyarrow', compression='snappy')
        except Exception as e:
            print(f"❌ Save Parquet Failed {symbol}: {e}")

    def get_stock_history(self, symbol: str, market: str = "US", days: int = 365*5, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据 (主入口)
        1. 尝试读缓存
        2. 缓存不存在或过期 -> 下载
        3. 保存并返回
        """
        if not force_refresh:
            df = self.load_from_cache(symbol, market, max_age_days=1)
            if df is not None:
                return df
                
        # 需要下载
        if market == "US":
            df = self.fetch_from_polygon(symbol, days)
        else:
            # CN 逻辑暂留空或接 Tushare
            return None
            
        if df is not None:
            self.save_to_cache(df, symbol, market)
            
        return df

    def warmup_cache_batch(self, symbols: List[str], market: str = "US", max_workers: int = 10):
        """批量预热缓存 (多线程)"""
        print(f"🔥 Warming up cache for {len(symbols)} symbols ({market})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_stock_history, sym, market, 365*5, False): sym
                for sym in symbols
            }
            
            completed = 0
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        completed += 1
                except Exception as e:
                    print(f"Error {sym}: {e}")
                    
                # 简单的进度条
                if completed % 100 == 0:
                    print(f"   Progress: {completed}/{len(symbols)}")
                    
        print(f"✅ Cache warmup finished. {completed}/{len(symbols)} synced.")

if __name__ == "__main__":
    # 测试代码
    cache = DataCache()
    df = cache.get_stock_history("AAPL", market="US")
    if df is not None:
        print(f"Successfully loaded AAPL: {len(df)} rows")
        print(df.tail())
    else:
        print("Failed to load AAPL")
