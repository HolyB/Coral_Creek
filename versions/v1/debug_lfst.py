import pandas as pd
import numpy as np
import sys
import os

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_fetcher import get_stock_data
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series

def debug_lfst():
    symbol = "LFST"
    market = "US"
    days = 1095 # 3年
    
    print(f"Loading data for {symbol}...")
    df = get_stock_data(symbol, market, days)
    
    if df is None or df.empty:
        print("Failed to load data")
        return

    # 1. 计算日线指标
    print("Calculating Daily Indicators...")
    df['Day_BLUE'] = calculate_blue_signal_series(
        df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    )
    df['heima'], df['juedi'] = calculate_heima_signal_series(
        df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    )
    
    # 2. 计算周线指标
    print("Calculating Weekly Indicators...")
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    df_weekly['Week_BLUE'] = calculate_blue_signal_series(
        df_weekly['Open'].values, df_weekly['High'].values, 
        df_weekly['Low'].values, df_weekly['Close'].values
    )
    
    # 计算周线黑马
    df_weekly['heima_week'], df_weekly['juedi_week'] = calculate_heima_signal_series(
        df_weekly['High'].values, df_weekly['Low'].values, df_weekly['Close'].values, df_weekly['Open'].values
    )

    # 3. 映射回日线
    # 严谨模式 (Shift 1): 只能看到上周
    df_weekly_shifted = df_weekly.shift(1)
    df['Week_BLUE_Strict'] = df_weekly_shifted['Week_BLUE'].reindex(df.index, method='ffill')
    
    # 未来模式 (No Shift): 可以看到本周最终值 (用户视角)
    df['Week_BLUE_Future'] = df_weekly['Week_BLUE'].reindex(df.index, method='ffill')
    df['Week_Heima_Future'] = df_weekly['heima_week'].reindex(df.index, method='ffill')
    df['Week_Juedi_Future'] = df_weekly['juedi_week'].reindex(df.index, method='ffill')
    
    # 截取 2025/7/20 - 2025/8/10 的数据
    mask = (df.index >= '2025-07-20') & (df.index <= '2025-08-10')
    target_df = df.loc[mask]
    
    print("\n" + "="*80)
    print(f"Debug Data for {symbol} (2025-07-20 to 2025-08-10)")
    print("="*80)
    
    # 格式化输出
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    cols = ['Open', 'Close', 'Day_BLUE', 'heima', 'juedi', 'Week_BLUE_Strict', 'Week_BLUE_Future', 'Week_Heima_Future', 'Week_Juedi_Future']
    
    for idx, row in target_df.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        print(f"{date_str} | Close: {row['Close']:<6.2f} | Day_BLUE: {row['Day_BLUE']:<6.1f} | Heima: {int(row['heima']):<1} | Juedi: {int(row['juedi']):<1} | Week_BLUE(Strict): {row['Week_BLUE_Strict']:<6.1f} | Week_BLUE(Future): {row['Week_BLUE_Future']:<6.1f} | W_Heima: {int(row['Week_Heima_Future']) if pd.notna(row['Week_Heima_Future']) else 0} | W_Juedi: {int(row['Week_Juedi_Future']) if pd.notna(row['Week_Juedi_Future']) else 0}")

if __name__ == "__main__":
    debug_lfst()



