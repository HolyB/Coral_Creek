import sys
import os
import pandas as pd
import numpy as np
from simple_backtest import SimpleBacktester

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def diagnose_lung():
    symbol = "LUNG"
    market = "US"
    print(f"Diagnosing {symbol} backtest logic...")
    
    # 初始化回测器
    bt = SimpleBacktester(symbol, market, days=365, blue_threshold=80)
    
    if not bt.load_data():
        print("Failed to load data")
        return

    bt.calculate_signals()
    
    df = bt.df
    print(f"\nTotal Data Points: {len(df)}")
    
    # 1. 检查是否有 BLUE 信号
    blue_days = df[df['Day_BLUE'] > 80]
    print(f"Days with BLUE > 80: {len(blue_days)}")
    if not blue_days.empty:
        print("Sample BLUE days:")
        print(blue_days[['Close', 'Day_BLUE', 'J']].head())
    
    # 2. 检查是否有卖出信号 (J > 100)
    sell_days = df[df['J'] > 100]
    print(f"Days with J > 100 (Sell Signal): {len(sell_days)}")
    
    # 3. 模拟简化的交易流
    print("\n--- Simulating Logic Step-by-Step ---")
    position = 0
    capital = 100000
    
    # 只看有信号的前后几天
    # 为了缩短输出，我们只打印关键节点
    
    for i in range(len(df)):
        date = df.index[i]
        d_blue = df['Day_BLUE'].iloc[i]
        j_val = df['J'].iloc[i]
        price = df['Close'].iloc[i]
        
        action = ""
        
        # 买入逻辑检查
        if position == 0 and d_blue > 80:
            action = " [BUY SIGNAL]"
            position = 1 # 模拟买入
            
        # 卖出逻辑检查
        elif position == 1 and j_val > 100:
            action = " [SELL SIGNAL]"
            position = 0 # 模拟卖出
            
        if action or d_blue > 10 or j_val > 90:
            print(f"{date.strftime('%Y-%m-%d')} | Price: {price:.2f} | BLUE: {d_blue:.1f} | J: {j_val:.1f} | Pos: {position} {action}")

if __name__ == "__main__":
    diagnose_lung()

