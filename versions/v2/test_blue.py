import sys
import os
import pandas as pd
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import get_us_stock_data
from indicator_utils import calculate_blue_signal_series

def test_stock(symbol):
    print(f"\n🔍 Testing {symbol}...")
    df = get_us_stock_data(symbol, days=1095)  # 3年数据
    if df is None or len(df) < 100:
        print(f"  ❌ Failed to fetch data for {symbol}")
        return
    
    print(f"  📊 Data loaded: {len(df)} days")
    
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    blue = calculate_blue_signal_series(opens, highs, lows, closes)
    
    print(f"  💰 Latest Price: ${closes[-1]:.2f}")
    print(f"  🟦 BLUE[-3]: {blue[-3]:.1f}")
    print(f"  🟦 BLUE[-2]: {blue[-2]:.1f}")
    print(f"  🟦 BLUE[-1] (Today): {blue[-1]:.1f}")

if __name__ == "__main__":
    # 测试几个关键股票
    test_stocks = ["CSCO", "NVDA", "AAPL", "TSLA", "META", "GOOGL"]
    
    print("=" * 50)
    print("🧪 BLUE Signal Validation Test (3-Year Data)")
    print("=" * 50)
    
    for symbol in test_stocks:
        test_stock(symbol)
    
    print("\n" + "=" * 50)
    print("✅ Test Complete!")



