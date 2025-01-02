import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

def test_smx_phantom():
    """测试 SMX 的幻影主力指标"""
    ticker = 'COIN'
    
    print(f"\n分析股票: {ticker}")
    
    try:
        # 修改数据获取方式，移除 start_date 和 end_date 参数
        fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
        data = fetcher.get_stock_data()
        
        if data is None or data.empty:
            print("无法获取数据")
            return
            
        print(f"获取到 {len(data)} 天的数据")
        
        # 分析数据
        analysis = StockAnalysis(data)
        result_df = analysis.calculate_all_strategies()
        
        # 获取最后一周的数据（5个交易日）
        last_week = result_df.tail(5)
        
        print("\n最后一周的幻影主力指标:")
        print("\n1. PINK线和信号:")
        print(last_week[['Close', 'PINK', '笑脸信号_做多', '笑脸信号_做空']].round(2))
        
        print("\n2. 资金力度指标:")
        print(last_week[['RED', 'YELLOW', 'GREEN', 'LIGHTBLUE']].round(2))
        
        print("\n3. 海底捞月指标:")
        print(last_week[['BLUE', 'LIRED']].round(2))
        
        # 检查最新信号
        latest = result_df.iloc[-1]
        signals = []
        
        if latest['笑脸信号_做多'] == 1:
            signals.append('PINK线上穿10信号 (做多)')
        if latest['笑脸信号_做空'] == 1:
            signals.append('PINK线下穿94信号 (做空)')
            
        if signals:
            print("\n⭐ 最新信号:")
            for signal in signals:
                print(f"- {signal}")
        else:
            print("\n未发现信号")
            
        # 显示当前 PINK 线位置
        current_pink = latest['PINK']
        print(f"\nPINK线当前位置: {current_pink:.2f}")
        if current_pink < 10:
            print("PINK线处于超卖区域，注意可能出现做多信号")
        elif current_pink > 94:
            print("PINK线处于超买区域，注意可能出现做空信号")
            
    except Exception as e:
        print(f"分析出错: {e}")

if __name__ == "__main__":
    test_smx_phantom() 