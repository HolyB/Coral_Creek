import sys
import os
import pandas as pd
from simple_backtest import SimpleBacktester
from indicator_utils import calculate_blue_signal_series

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def diagnose_august():
    symbol = "LUNG"
    market = "US"
    print(f"Diagnosing {symbol} for August 2025...")
    
    # 1. 获取数据
    bt = SimpleBacktester(symbol, market, days=365)
    if not bt.load_data(): return
    
    # 2. 手动计算周线，看看 8/8 和 8/15 的情况
    df = bt.df
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_weekly['Week_BLUE'] = calculate_blue_signal_series(
        df_weekly['Open'].values, df_weekly['High'].values, 
        df_weekly['Low'].values, df_weekly['Close'].values
    )
    
    print("\n=== 8月份周线数据 ===")
    print(df_weekly.loc['2025-08-01':'2025-08-22', ['Close', 'Week_BLUE']])
    
    # 3. 计算日线信号
    bt.calculate_signals()
    df_daily = bt.df.loc['2025-08-01':'2025-08-20']
    
    print("\n=== 8月份日线详细数据 ===")
    # 打印关键列：收盘价，日线BLUE，周线引用值，黑马信号，持仓模拟
    print(f"{'Date':<12} {'Close':<6} {'Day_BLUE':<8} {'Week_Ref':<8} {'Heima':<5} {'Status'}")
    print("-" * 60)
    
    # 模拟持仓：假设5月28日买了
    position = 1 
    
    for date, row in df_daily.iterrows():
        d_str = date.strftime('%Y-%m-%d')
        close = row['Close']
        d_blue = row['Day_BLUE']
        w_ref = row['Week_BLUE_Ref']
        heima = "YES" if row['heima'] else "NO"
        
        # 简单的持仓逻辑推演
        status = "HOLDING (No Cash)" if position == 1 else "EMPTY (Can Buy)"
        
        # 如果是 8/27 卖出，这里还没到时间
        
        print(f"{d_str:<12} {close:<6.2f} {d_blue:<8.1f} {w_ref:<8.1f} {heima:<5} {status}")

if __name__ == "__main__":
    diagnose_august()

