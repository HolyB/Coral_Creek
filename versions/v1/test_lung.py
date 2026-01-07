import sys
import os
import numpy as np
from data_fetcher import get_stock_data
from indicator_utils import calculate_blue_signal_series

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_lung():
    symbol = "LUNG"
    market = "US"
    print(f"Testing data fetch for {symbol}...")
    
    try:
        # 1. 测试获取数据
        df = get_stock_data(symbol, market, days=365)
        
        if df is None or df.empty:
            print("❌ Failed to get data: DataFrame is empty or None")
            return
            
        print(f"✅ Data fetched successfully. Records: {len(df)}")
        print("Last 5 rows:")
        print(df.tail())
        
        # 2. 测试计算波动率
        df['returns'] = df['Close'].pct_change()
        volatility = df['returns'].std() * (252 ** 0.5)
        print(f"✅ Volatility calculated: {volatility:.2%}")
        
        # 3. 测试计算信号
        print("Calculating BLUE signals...")
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        )
        print(f"✅ BLUE signal calculated. Max value: {np.nanmax(blue):.2f}")
        
        # 4. 测试回测逻辑中的切片
        print("Testing backtest logic...")
        from simple_backtest import SimpleBacktester
        bt = SimpleBacktester(symbol, market, days=365, blue_threshold=80)
        bt.df = df
        bt.calculate_signals()
        print(f"✅ Signals calculated in Backtester. Max Day BLUE: {bt.df['Day_BLUE'].max()}")
        
        bt.run_backtest()
        print(f"✅ Backtest run. Total Trades: {len(bt.trades)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lung()

