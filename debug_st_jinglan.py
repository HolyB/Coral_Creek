import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import requests

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

def debug_stock(symbol='000711', name='ST京蓝'):
    """调试单个股票"""
    print(f"\n开始调试 {symbol} {name}")
    
    max_retries = 5  # 最大重试次数
    retry_delay = 10  # 重试等待时间
    
    for attempt in range(max_retries):
        try:
            # 1. 获取日线数据
            print(f"\n第 {attempt + 1} 次尝试获取数据...")
            fetcher_daily = StockDataFetcher(symbol, source='akshare', interval='1d')
            data_daily = fetcher_daily.get_stock_data()
            
            if data_daily is None or data_daily.empty:
                print(f"无法获取历史数据，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                continue
                
            print(f"获取到 {len(data_daily)} 条日线数据")
            print("最新日期:", data_daily.index[-1].strftime('%Y-%m-%d'))
            
            # 重采样为周线数据
            data_weekly = data_daily.resample('W-MON').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'mean'
            }).dropna()
            
            print(f"重采样得到 {len(data_weekly)} 条周线数据")
            
            # 2. 计算指标
            print("\n2. 计算指标...")
            analysis_weekly = StockAnalysis(data_weekly)
            df_weekly = analysis_weekly.calculate_phantom_indicators()
            
            # 添加缩放比例计算
            max_blue = df_weekly['BLUE'].max()
            radio1 = 200 / max_blue if max_blue > 0 else 1
            df_weekly['BLUE'] = df_weekly['BLUE'] * radio1
            
            analysis_daily = StockAnalysis(data_daily)
            df_daily = analysis_daily.calculate_phantom_indicators()
            df_daily = analysis_daily.calculate_heatmap_volume()
            df_daily = analysis_daily.calculate_macd_signals()
            
            recent_weekly = df_weekly.tail(10)
            latest_weekly = df_weekly.iloc[-1]
            recent_daily = df_daily.tail(10)
            latest_daily = df_daily.iloc[-1]
            recent_5d = df_daily.tail(5)
            
            # 3. 检查信号条件
            print("\n3. 检查信号...")
            print("\n日线指标:")
            print(f"PINK: {latest_daily['PINK']:.2f}")
            print(f"BLUE: {latest_daily['BLUE']:.2f}")
            print(f"笑脸信号_做多: {latest_daily['笑脸信号_做多']}")
            print(f"笑脸信号_做空: {latest_daily['笑脸信号_做空']}")
            print(f"BLUE>150天数: {len(recent_daily[recent_daily['BLUE'] > 150])}")
            
            print("\n周线指标:")
            print(f"PINK: {latest_weekly['PINK']:.2f}")
            print(f"BLUE: {latest_weekly['BLUE']:.2f}")
            print(f"笑脸信号_做多: {latest_weekly['笑脸信号_做多']}")
            print(f"笑脸信号_做空: {latest_weekly['笑脸信号_做空']}")
            print(f"BLUE>150周数: {len(recent_weekly[recent_weekly['BLUE'] > 150])}")
            
            print("\n成交量指标:")
            print(f"VOL_TIMES: {latest_daily['VOL_TIMES']:.2f}")
            print(f"HVOL_COLOR: {latest_daily['HVOL_COLOR']}")
            print(f"GOLD_VOL: {latest_daily['GOLD_VOL']}")
            print(f"DOUBLE_VOL: {latest_daily['DOUBLE_VOL']}")
            print(f"最近5天黄金柱次数: {len(recent_5d[recent_5d['GOLD_VOL']])}")
            print(f"最近5天倍量柱次数: {len(recent_5d[recent_5d['DOUBLE_VOL']])}")
            
            print("\nMACD指标:")
            print(f"DIF: {latest_daily['DIF']:.4f}")
            print(f"DEA: {latest_daily['DEA']:.4f}")
            print(f"MACD: {latest_daily['MACD']:.4f}")
            print(f"EMAMACD: {latest_daily['EMAMACD']:.4f}")
            
            # 4. 检查信号条件
            has_daily_signal = (
                latest_daily['笑脸信号_做多'] == 1 or 
                latest_daily['笑脸信号_做空'] == 1 or 
                len(recent_daily[recent_daily['BLUE'] > 150]) >= 3
            )
            
            has_weekly_signal = (
                latest_weekly['笑脸信号_做多'] == 1 or 
                latest_weekly['笑脸信号_做空'] == 1 or 
                len(recent_weekly[recent_weekly['BLUE'] > 150]) >= 2
            )
            
            has_volume_signal = (
                latest_daily['GOLD_VOL'] or 
                latest_daily['DOUBLE_VOL']
            )
            
            print("\n4. 信号汇总:")
            print(f"日线信号: {has_daily_signal}")
            print(f"周线信号: {has_weekly_signal}")
            print(f"成交量信号: {has_volume_signal}")
            
            print("\n5. 最终判断:")
            if has_weekly_signal and (has_daily_signal or has_volume_signal):
                print("符合条件！应该被选中")
            else:
                print("不符合条件，原因：")
                if not has_weekly_signal:
                    print("- 缺少周线信号")
                if not has_daily_signal and not has_volume_signal:
                    print("- 缺少日线信号和成交量信号")
            
            # 如果成功获取和处理了数据，就跳出重试循环
            break
            
        except Exception as e:
            print(f"其他错误: {e}")
            import traceback
            print(traceback.format_exc())
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，退出")
                return None

if __name__ == "__main__":
    debug_stock() 