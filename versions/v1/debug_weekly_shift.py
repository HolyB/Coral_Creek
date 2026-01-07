import sys
import os
import pandas as pd
from simple_backtest import SimpleBacktester
from indicator_utils import calculate_blue_signal_series

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def debug_weekly_logic():
    symbol = "LUNG"
    market = "US"
    
    # 1. 获取日线数据
    bt = SimpleBacktester(symbol, market, days=365)
    if not bt.load_data(): return
    df = bt.df
    
    # 2. 手动重算周线
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    df_weekly['Week_BLUE'] = calculate_blue_signal_series(
        df_weekly['Open'].values, df_weekly['High'].values, 
        df_weekly['Low'].values, df_weekly['Close'].values
    )
    
    print("\n=== 周线数据 (Raw Weekly Data) ===")
    # 打印 11月附近的周线
    target_period = df_weekly['2025-10-20':'2025-12-01']
    print(target_period[['Close', 'Week_BLUE']])
    
    # 3. 检查映射逻辑
    df_weekly_shifted = df_weekly.shift(1)
    mapped_daily = df_weekly_shifted['Week_BLUE'].reindex(df.index, method='ffill').fillna(0)
    
    print("\n=== 日线映射数据 (Mapped Daily Data) ===")
    target_daily = mapped_daily['2025-11-10':'2025-11-25']
    print(target_daily)

if __name__ == "__main__":
    debug_weekly_logic()

