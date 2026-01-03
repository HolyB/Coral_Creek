import sys
import os
import pandas as pd
from simple_backtest import SimpleBacktester

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def verify_overlap():
    symbol = "LUNG"
    market = "US"
    print(f"Checking signal overlap for {symbol}...")
    
    # 加载数据
    bt = SimpleBacktester(symbol, market, days=365)
    if not bt.load_data():
        return

    bt.calculate_signals()
    df = bt.df
    
    # 提取信号
    # 注意: 这里的阈值设为 80，与您回测时一致
    blue_days = df[df['Day_BLUE'] > 80].index
    heima_days = df[df['heima'] == True].index
    juedi_days = df[df['juedi'] == True].index
    
    print(f"\nBLUE 信号天数: {len(blue_days)}")
    print(f"黑马 信号天数: {len(heima_days)}")
    print(f"掘底 信号天数: {len(juedi_days)}")
    
    # 打印前几个日期
    if len(blue_days) > 0:
        print(f"BLUE 示例: {[d.strftime('%Y-%m-%d') for d in blue_days[:5]]}")
    if len(heima_days) > 0:
        print(f"黑马 示例: {[d.strftime('%Y-%m-%d') for d in heima_days[:5]]}")
    
    # 找交集
    overlap_heima = blue_days.intersection(heima_days)
    overlap_juedi = blue_days.intersection(juedi_days)
    
    print("\n" + "="*40)
    print("信号重叠分析")
    print("="*40)
    
    if len(overlap_heima) == 0 and len(overlap_juedi) == 0:
        print("❌ 没有任何重叠！")
        print("结论：在过去一年里，LUNG 从未在同一天同时触发 BLUE 和 黑马/掘底 信号。")
        print("这就是为什么勾选该选项后交易为 0 的原因。")
    else:
        print(f"✅ 发现重叠！")
        if len(overlap_heima) > 0:
            print(f"BLUE + 黑马: {[d.strftime('%Y-%m-%d') for d in overlap_heima]}")
        if len(overlap_juedi) > 0:
            print(f"BLUE + 掘底: {[d.strftime('%Y-%m-%d') for d in overlap_juedi]}")

if __name__ == "__main__":
    verify_overlap()

