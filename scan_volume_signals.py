import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

def scan_volume_signals():
    """扫描成交量信号"""
    # 指定要检查的股票
    tickers = ['ETSY', 'TSLA', 'NVDA']
    
    results = []
    start_time = time.time()
    
    for symbol in tickers:
        try:
            print(f"\n处理 {symbol}")
            
            # 获取数据
            fetcher = StockDataFetcher(symbol, source='polygon')
            data = fetcher.get_stock_data()
            
            if data is None or data.empty:
                print(f"{symbol} 无法获取数据")
                continue
                
            print(f"{symbol} 获取到 {len(data)} 天的数据")
            
            # 计算指标
            analysis = StockAnalysis(data)
            analysis.calculate_heatmap_volume()
            df = analysis.df
            
            # 获取最近数据
            latest = df.iloc[-1]
            
            # 检查信号
            if (latest['HVOL_COLOR'] >= 4 or  # 高成交量
                latest['GOLD_VOL'] or  # 黄金柱
                latest['DOUBLE_VOL']):  # 倍量柱
                
                result = {
                    'symbol': symbol,
                    'price': latest['Close'],
                    'Volume': latest['Volume'],
                    'vol_times': latest['VOL_TIMES'],
                    'vol_color': latest['HVOL_COLOR'],
                    'vol_devbar': latest['HVOL_DEVBAR'],
                    'gold_vol': latest['GOLD_VOL'],
                    'double_vol': latest['DOUBLE_VOL']
                }
                results.append(result)
            
            # 打印详细信息
            print(f"热力值: {latest['HVOL_COLOR']}, "
                  f"成交倍数: {latest['VOL_TIMES']:.1f}, "
                  f"偏离度: {latest['HVOL_DEVBAR']:.2f}")
            
            if latest['GOLD_VOL']:
                print("发现黄金柱")
            if latest['DOUBLE_VOL']:
                print(f"发现倍量柱: {latest['VOL_TIMES']:.1f}倍")
            
        except Exception as e:
            print(f"{symbol} 处理出错: {e}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def main():
    """主函数"""
    start_time = time.time()
    
    # 扫描成交量信号
    print("\n开始扫描成交量信号...")
    results = scan_volume_signals()
    
    if not results.empty:
        # 显示结果
        print("\n发现信号的股票:")
        print("=" * 120)
        print(f"{'代码':<6} | {'价格':>8} | {'成交量':>12} | {'成交倍数':>8} | {'热力值':>8} | {'偏离度':>8} | {'信号':<30}")
        print("-" * 120)
        
        for _, row in results.iterrows():
            signals = []
            if row['vol_color'] == 5:
                signals.append('极高成交量')
            elif row['vol_color'] == 4:
                signals.append('高成交量')
            if row['gold_vol']:
                signals.append('黄金柱')
            if row['double_vol']:
                signals.append(f'倍量柱({row["vol_times"]:.1f}倍)')
            
            signals_str = ', '.join(signals)
            print(f"{row['symbol']:<6} | {row['price']:8.2f} | {row['Volume']:12.0f} | "
                  f"{row['vol_times']:8.1f} | {row['vol_color']:8.0f} | {row['vol_devbar']:8.2f} | "
                  f"{signals_str:<30}")
        
        print("=" * 120)
        print(f"共发现 {len(results)} 只股票有信号")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 