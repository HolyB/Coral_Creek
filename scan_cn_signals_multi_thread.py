import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import akshare as ak
import concurrent.futures
import threading
from tqdm import tqdm

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher
from scan_cn_signals import get_cn_tickers, get_realtime_price

# 创建一个线程锁用于打印
print_lock = threading.Lock()

def process_single_stock(stock):
    """处理单个股票"""
    symbol = stock['code']
    name = stock['name']
    
    try:
        # 1. 获取日线数据
        fetcher_daily = StockDataFetcher(symbol, source='akshare', interval='1d')
        data_daily = fetcher_daily.get_stock_data()
        
        if data_daily is None or data_daily.empty:
            return None
            
        # 重采样为周线数据，使用周一作为时间戳
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'  # 取平均
        }).dropna()  # 删除空值
        
        # 2. 计算周线指标，添加缩放比例
        analysis_weekly = StockAnalysis(data_weekly)
        df_weekly = analysis_weekly.calculate_phantom_indicators()
        
        # 添加缩放比例计算
        max_blue = df_weekly['BLUE'].max()  # 获取整个周期内的最大BLUE值
        radio1 = 200 / max_blue if max_blue > 0 else 1  # 根据原版公式设置缩放比例
        
        # 应用缩放
        df_weekly['BLUE'] = df_weekly['BLUE'] * radio1
        
        recent_weekly = df_weekly.tail(10)
        latest_weekly = df_weekly.iloc[-1]
        
        # 3. 检查周线信号
        blue_weeks = len(recent_weekly[recent_weekly['BLUE'] > 150])
        has_weekly_signal = (blue_weeks >= 2 or  
                           latest_weekly['笑脸信号_做多'] == 1 or 
                           latest_weekly['笑脸信号_做空'] == 1)
        
        # 调试信息：显示所有股票的周线数据
        with print_lock:
            print(f"\n{symbol} {name} 周线数据:")
            print(f"最近10周BLUE值:")
            print(recent_weekly[['Close', 'BLUE', 'Volume']].to_string())
            print(f"缩放比例 RADIO1: {radio1:.2f}")
            print(f"BLUE>150周数: {blue_weeks}")
            print(f"最新周BLUE值: {latest_weekly['BLUE']:.2f}")
            print(f"笑脸信号_做多: {latest_weekly['笑脸信号_做多']}")
            print(f"笑脸信号_做空: {latest_weekly['笑脸信号_做空']}")
            print(f"是否有周线信号: {has_weekly_signal}")
        
        # 计算日线指标
        analysis_daily = StockAnalysis(data_daily)
        df_daily = analysis_daily.calculate_phantom_indicators()
        analysis_daily.calculate_heatmap_volume()
        
        recent_daily = df_daily.tail(10)
        latest_daily = df_daily.iloc[-1]
        recent_5d = df_daily.tail(5)
        
        # 检查所有信号
        has_daily_signal = (len(recent_daily[recent_daily['BLUE'] > 150]) >= 3 or
                          latest_daily['笑脸信号_做多'] == 1 or 
                          latest_daily['笑脸信号_做空'] == 1)
        
        has_volume_signal = (
            len(recent_5d[recent_5d['DOUBLE_VOL'] | recent_5d['GOLD_VOL']]) > 0
        )
        
        # 只在所有信号都满足时返回结果
        if has_daily_signal and has_weekly_signal and has_volume_signal:
            result = {
                'symbol': symbol,
                'name': name,
                'pink_daily': latest_daily['PINK'],
                'blue_daily': latest_daily['BLUE'],
                'max_blue_daily': recent_daily['BLUE'].max(),
                'blue_days': len(recent_daily[recent_daily['BLUE'] > 150]),
                'pink_weekly': latest_weekly['PINK'],
                'blue_weekly': latest_weekly['BLUE'],
                'max_blue_weekly': recent_weekly['BLUE'].max(),
                'blue_weeks': len(recent_weekly[recent_weekly['BLUE'] > 150]),
                'smile_long_daily': latest_daily['笑脸信号_做多'],
                'smile_short_daily': latest_daily['笑脸信号_做空'],
                'smile_long_weekly': latest_weekly['笑脸信号_做多'],
                'smile_short_weekly': latest_weekly['笑脸信号_做空'],
                'vol_times': latest_daily['VOL_TIMES'],
                'vol_color': latest_daily['HVOL_COLOR'],
                'vol_devbar': latest_daily['HVOL_DEVBAR'],
                'gold_vol_count': len(recent_5d[recent_5d['GOLD_VOL']]),  # 5天内黄金柱次数
                'double_vol_count': len(recent_5d[recent_5d['DOUBLE_VOL']])  # 5天内倍量柱次数
            }
            
            # 打印信号
            signals = []
            if latest_daily['笑脸信号_做多'] == 1:
                signals.append('日线PINK上穿10')
            if latest_daily['笑脸信号_做空'] == 1:
                signals.append('日线PINK下穿94')
            if len(recent_daily[recent_daily['BLUE'] > 150]) >= 3:
                signals.append(f'日线BLUE>150 ({result["blue_days"]}天)')
            if latest_weekly['笑脸信号_做多'] == 1:
                signals.append('周线PINK上穿10')
            if latest_weekly['笑脸信号_做空'] == 1:
                signals.append('周线PINK下穿94')
            if len(recent_weekly[recent_weekly['BLUE'] > 150]) >= 2:
                signals.append(f'周线BLUE>150 ({result["blue_weeks"]}周)')
            if result['gold_vol_count'] > 0:
                signals.append(f'黄金柱({result["gold_vol_count"]}次)')
            if result['double_vol_count'] > 0:
                signals.append(f'倍量柱({result["double_vol_count"]}次)')
            
            with print_lock:
                print(f"{symbol} {name} 发现三重信号: {', '.join(signals)}")
            
            return result
            
    except Exception as e:
        with print_lock:
            print(f"{symbol} {name} 处理出错: {e}")
    return None

def scan_signals_parallel(max_workers=20):
    """并行扫描A股信号"""
    print("正在获取A股列表...")
    stock_list = get_cn_tickers()[:100]
    
    if stock_list.empty:
        print("获取股票列表失败")
        return pd.DataFrame()
    
    print(f"\n开始并行扫描 {len(stock_list)} 只A股...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建进度条
        with tqdm(total=len(stock_list), desc="扫描进度") as pbar:
            # 提交所有任务
            future_to_stock = {executor.submit(process_single_stock, stock): stock 
                             for _, stock in stock_list.iterrows()}
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"{stock['code']} {stock['name']} 处理失败: {e}")
                finally:
                    pbar.update(1)
    
    return pd.DataFrame(results)

def update_realtime_data(results_df):
    """更新实时数据"""
    print("\n正在获取实时数据...")
    
    # 获取所有股票的实时数据
    try:
        df = ak.stock_zh_a_spot_em()
        
        # 更新每个股票的实时数据
        for idx, row in results_df.iterrows():
            code = row['symbol'][2:]  # 移除市场前缀
            stock_info = df[df['代码'] == code].iloc[0]
            
            # 更新实时数据
            results_df.at[idx, 'price'] = stock_info['最新价']
            results_df.at[idx, 'change'] = stock_info['涨跌幅']
            results_df.at[idx, 'Volume'] = stock_info['成交量']
            results_df.at[idx, 'turnover'] = stock_info['成交额']
            
        print("实时数据更新完成")
        return results_df
        
    except Exception as e:
        print(f"获取实时数据失败: {e}")
        return results_df

def main():
    """主函数"""
    start_time = time.time()
    
    # 并行扫描A股
    results = scan_signals_parallel(max_workers=20)  # 可以调整线程数
    
    if results is not None and not results.empty:
        # 更新实时数据
        results = update_realtime_data(results)
        
        # 显示结果
        print("\n发现信号的股票:")
        print("=" * 160)
        print(f"{'代码':<8} | {'名称':<8} | {'价格':>8} | {'涨跌幅':>8} | {'成交量':>12} | {'成交额':>12} | "
              f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | "
              f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | {'信号':<40}")
        print("-" * 160)
        
        for _, row in results.iterrows():
            signals = []
            if row['smile_long_daily'] == 1:
                signals.append('日PINK上穿10')
            if row['smile_short_daily'] == 1:
                signals.append('日PINK下穿94')
            if row['blue_days'] >= 3:
                signals.append(f'日BLUE>150({row["blue_days"]}天)')
            if row['smile_long_weekly'] == 1:
                signals.append('周PINK上穿10')
            if row['smile_short_weekly'] == 1:
                signals.append('周PINK下穿94')
            if row['blue_weeks'] >= 2:
                signals.append(f'周BLUE>150({row["blue_weeks"]}周)')
            
            signals_str = ', '.join(signals)
            print(f"{row['symbol']:<8} | {row['name']:<8} | {row['price']:8.2f} | {row['change']:8.2f}% | "
                  f"{row['Volume']:12.0f} | {row['turnover']:12.0f} | "
                  f"{row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                  f"{row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                  f"{signals_str:<40}")
        
        print("=" * 160)
        print(f"共发现 {len(results)} 只股票有信号")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 

