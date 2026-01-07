import sys
import os
import pandas as pd
from simple_backtest import SimpleBacktester

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def verify_weekly():
    symbol = "LUNG"
    market = "US"
    print(f"Verifying Weekly BLUE for {symbol}...")
    
    # 加载数据
    bt = SimpleBacktester(symbol, market, days=365)
    if not bt.load_data():
        return

    bt.calculate_signals()
    df = bt.df
    
    # 提取信号
    day_blue_days = df[df['Day_BLUE'] > 80].index
    heima_days = df[df['heima'] == True].index
    week_blue_days = df[df['Week_BLUE_Ref'] > 80].index
    
    print(f"\n日线 BLUE 天数: {len(day_blue_days)}")
    print(f"周线 BLUE (Ref) 天数: {len(week_blue_days)}")
    print(f"黑马 信号天数: {len(heima_days)}")
    
    # 找交集
    # 1. 日线 + 周线
    overlap_dw = day_blue_days.intersection(week_blue_days)
    print(f"\n[日线 + 周线] 重叠天数: {len(overlap_dw)}")
    if len(overlap_dw) > 0:
        print(f"示例: {[d.strftime('%Y-%m-%d') for d in overlap_dw[:5]]}")
        
    # 2. 日线 + 黑马 (之前的成功组合)
    overlap_dh = day_blue_days.intersection(heima_days)
    print(f"\n[日线 + 黑马] 重叠天数: {len(overlap_dh)}")
    if len(overlap_dh) > 0:
        print(f"关键日期: {[d.strftime('%Y-%m-%d') for d in overlap_dh]}")
        
    # 3. 三者共振 (日线 + 周线 + 黑马)
    overlap_all = overlap_dh.intersection(week_blue_days)
    print(f"\n[三者共振] 重叠天数: {len(overlap_all)}")
    
    if len(overlap_all) == 0:
        print("❌ 没有任何一天同时满足三个条件！")
        print("详细分析关键日期（日线+黑马）的周线状态:")
        for date in overlap_dh:
            w_val = df.loc[date, 'Week_BLUE_Ref']
            print(f"  Date: {date.strftime('%Y-%m-%d')} | Week_BLUE: {w_val}")
    else:
        print(f"✅ 发现三者共振: {[d.strftime('%Y-%m-%d') for d in overlap_all]}")

if __name__ == "__main__":
    verify_weekly()

