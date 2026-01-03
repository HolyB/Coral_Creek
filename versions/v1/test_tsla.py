import sys
import os
from data_fetcher import get_stock_data

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_tsla():
    symbol = "TSLA"
    market = "US"
    print(f"Testing data fetch for {symbol}...")
    
    try:
        # 尝试获取数据
        df = get_stock_data(symbol, market, days=365)
        
        if df is None:
            print("❌ Result is None")
        elif df.empty:
            print("❌ DataFrame is empty")
        else:
            print(f"✅ Data fetched successfully. Records: {len(df)}")
            print("Last 5 rows:")
            print(df.tail())
            
    except Exception as e:
        print(f"❌ Exception caught: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tsla()

