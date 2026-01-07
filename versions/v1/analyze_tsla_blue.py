import sys
import os
import numpy as np
import pandas as pd
from data_fetcher import get_stock_data
from indicator_utils import calculate_blue_signal_series

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def analyze_blue_values():
    symbol = "TSLA"
    market = "US"
    print(f"Analyzing BLUE values for {symbol}...")
    
    df = get_stock_data(symbol, market, days=365)
    if df is None or df.empty:
        print("Failed to get data")
        return

    blue_series = calculate_blue_signal_series(
        df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    )
    
    # 筛选出有信号的日子
    signal_days = blue_series[blue_series > 0]
    
    print(f"\nTotal days: {len(blue_series)}")
    print(f"Days with BLUE signal > 0: {len(signal_days)}")
    
    if len(signal_days) > 0:
        print("\nBLUE Value Distribution:")
        print(pd.Series(signal_days).describe())
        print("\nUnique BLUE values:")
        print(np.unique(signal_days))
    else:
        print("No BLUE signals found in the last year.")

if __name__ == "__main__":
    analyze_blue_values()

