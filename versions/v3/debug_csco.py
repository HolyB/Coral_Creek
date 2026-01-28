import sys
import os
import pandas as pd
import numpy as np
from data_fetcher import get_us_stock_data
from indicator_utils import calculate_blue_signal_series

def main():
    symbol = "CSCO"
    print(f"🔍 Debugging {symbol} with FINAL FIXED LOGIC...")
    
    df = get_us_stock_data(symbol, days=365)
    if df is None: return

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # 直接调用 indicator_utils 里的函数
    blue = calculate_blue_signal_series(opens, highs, lows, closes)
    
    print("\n--- Final Calculation Result ---")
    print(f"BLUE[-5:]: {blue[-5:]}")
    print(f"BLUE[-1]: {blue[-1]}")

if __name__ == "__main__":
    main()
