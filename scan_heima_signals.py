import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import concurrent.futures
import threading
import requests
from bs4 import BeautifulSoup

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher
from scan_signals import get_all_tickers
# 创建一个线程锁用于打印
print_lock = threading.Lock()

def process_single_stock(symbol):
    """处理单个股票"""
    try:
        # 1. 获取日线数据
        fetcher_daily = StockDataFetcher(symbol, source='polygon', interval='1d')
        data_daily = fetcher_daily.get_stock_data()
        
        if data_daily is None or data_daily.empty:
            with print_lock:
                print(f"{symbol} 无法获取日线数据")
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
        
        # 计算日线指标
        analysis_daily = StockAnalysis(data_daily)
        df_daily = analysis_daily.calculate_phantom_indicators()
        df_daily=analysis_daily.calculate_heatmap_volume()
        df_daily=analysis_daily.calculate_macd_signals()
        recent_daily = df_daily.tail(10)
        latest_daily = df_daily.iloc[-1]
        recent_5d = df_daily.tail(5)
        
        # 检查信号条件
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
        
        # 修改条件：必须有周线信号
        if has_weekly_signal and (has_daily_signal or has_volume_signal):
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'],
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
                'gold_vol_count': len(recent_5d[recent_5d['GOLD_VOL']]),
                'double_vol_count': len(recent_5d[recent_5d['DOUBLE_VOL']]),
                # 添加MACD相关指标
                'DIF': latest_daily['DIF'],
                'DEA': latest_daily['DEA'],
                'MACD': latest_daily['MACD'],
                'EMAMACD': latest_daily['EMAMACD'],
                'V1': latest_daily['V1'],
                'V2': latest_daily['V2'],
                'V3': latest_daily['V3'],
                'V4': latest_daily['V4'],
                '补血': 1 if latest_daily['MACD'] > df_daily['MACD'].shift(1).iloc[-1] else 0,
                '失血': 1 if latest_daily['MACD'] < df_daily['MACD'].shift(1).iloc[-1] else 0,
                '零轴下金叉': latest_daily['零轴下金叉'] if '零轴下金叉' in latest_daily else 0,
                '零轴上金叉': latest_daily['零轴上金叉'] if '零轴上金叉' in latest_daily else 0,
                '零轴上死叉': latest_daily['零轴上死叉'] if '零轴上死叉' in latest_daily else 0,
                '零轴下死叉': latest_daily['零轴下死叉'] if '零轴下死叉' in latest_daily else 0,
                '先机信号': latest_daily['先机信号'] if '先机信号' in latest_daily else 0,
                '底背离': latest_daily['底背离'] if '底背离' in latest_daily else 0,
                '顶背离': latest_daily['顶背离'] if '顶背离' in latest_daily else 0
            }
            return result
            
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
    return None

def scan_signals_parallel(max_workers=10):
    """并行扫描股票信号"""
    # 获取所有股票代码
    print("正在获取股票列表...")
    tickers = get_sp500_tickers()
    print(f"共获取到 {len(tickers)} 只股票")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_stock, symbol) for symbol in tickers]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def get_sp500_tickers():
    """获取标普500的股票代码列表"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    sp500_df = pd.read_html(str(table))[0]
    sp500_tickers = sp500_df['Symbol'].tolist()
    # 一些股票代码中可能包含点号，需要替换为减号以符合Yahoo Finance的格式
    sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]
    return sp500_tickers

def get_all_tickers():
    """获取所有股票代码列表"""
    key = "qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D"
    base_url = "https://api.polygon.io/v3/reference/tickers"
    
    params = {
        'market': 'stocks',
        'active': True,
        'sort': 'ticker',
        'order': 'asc',
        'limit': 1000,
        'apiKey': key
    }
    
    tickers = []
    
    try:
        while True:
            print(f"正在获取股票列表,当前已获取{len(tickers)}只...")
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if response.status_code != 200:
                print(f"获取股票列表失败: {data.get('error')}")
                break
                
            # 添加本页的股票代码
            for item in data['results']:
                if item.get('market') == 'stocks' and item.get('active'):
                    tickers.append(item['ticker'])
            
            # 检查是否还有下一页
            if 'next_url' not in data:
                break
                
            # 更新URL到下一页
            next_url = data['next_url']
            params['cursor'] = next_url.split('cursor=')[1]
            
            time.sleep(0.2)  # 避免请求过快
            
    except Exception as e:
        print(f"获取股票列表时出错: {e}")
    
    print(f"共获取到 {len(tickers)} 只股票")
    return tickers

def additional_sp_500():
    """额外添加的中概股列表"""
    additional_tickers = [
        'BILI', 'PDD', 'XPEV', 'NIO', 'BIDU', 'JD', 'NTES', 'TME', 'EDU', 'TAL',
        'HTHT', 'GDS', 'IQ', 'KC', 'ATHM', 'HUYA', 'VIPS', 'ZH', 'DADA', 'BGNE',
        'ZLAB', 'HTHT', 'YUMC', 'MNSO', 'API', 'TIGR', 'FUTU', 'UP'
    ]
    return list(set(additional_tickers))  # 去重

def main():
    """主函数"""
    start_time = time.time()
    
    # 并行扫描股票
    results = scan_signals_parallel(max_workers=20)  # 使用20个线程
    
    if not results.empty:
        # 保存v2格式的结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_v2 = f'signals_v2_{timestamp}.csv'
        
        # 创建v2格式的DataFrame
        df_v2 = pd.DataFrame({
            'symbol': results['symbol'],
            'signals': results.apply(lambda row: ', '.join([
                '日PINK上穿10' if row['smile_long_daily'] == 1 else '',
                '日PINK下穿94' if row['smile_short_daily'] == 1 else '',
                f'日BLUE>150({row["blue_days"]}天)' if row['blue_days'] >= 3 else '',
                '周PINK上穿10' if row['smile_long_weekly'] == 1 else '',
                '周PINK下穿94' if row['smile_short_weekly'] == 1 else '',
                f'周BLUE>150({row["blue_weeks"]}周)' if row['blue_weeks'] >= 2 else '',
                f'黄金柱({row["gold_vol_count"]}次)' if row["gold_vol_count"] > 0 else '',
                f'倍量柱({row["double_vol_count"]}次)' if row["double_vol_count"] > 0 else ''
            ]), axis=1),
            'DIF': results['DIF'] if 'DIF' in results else None,
            'DEA': results['DEA'] if 'DEA' in results else None,
            'MACD': results['MACD'] if 'MACD' in results else None,
            'EMAMACD': results['EMAMACD'] if 'EMAMACD' in results else None,
            'V1': results['V1'] if 'V1' in results else None,
            'V2': results['V2'] if 'V2' in results else None,
            'V3': results['V3'] if 'V3' in results else None,
            'V4': results['V4'] if 'V4' in results else None
        })
        
        # 保存v2格式文件
        df_v2.to_csv(filename_v2, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename_v2}")
        
        # 显示结果
        print("\n发现信号的股票:")
        print("=" * 220)  # 增加显示宽度
        print(f"{'代码':<8} | {'价格':>8} | {'成交量':>12} | {'成交额':>12} | "
              f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | "
              f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | "
              f"{'成交倍数':>8} | {'热力值':>4} | {'黄金柱':>4} | {'倍量柱':>4} | "
              f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'EMAMACD':>8} | {'信号':<40}")
        print("-" * 220)  # 增加显示宽度
        
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
            if row['gold_vol_count'] > 0:
                signals.append(f'黄金柱({row["gold_vol_count"]}次)')
            if row['double_vol_count'] > 0:
                signals.append(f'倍量柱({row["double_vol_count"]}次)')
            
            # 添加MACD相关信号
            if row.get('零轴下金叉', 0) == 1:
                signals.append('零轴下金叉')
            if row.get('零轴上金叉', 0) == 1:
                signals.append('零轴上金叉')
            if row.get('零轴上死叉', 0) == 1:
                signals.append('零轴上死叉')
            if row.get('零轴下死叉', 0) == 1:
                signals.append('零轴下死叉')
            if row.get('先机信号', 0) == 1:
                signals.append('先机信号')
            if row.get('底背离', 0) == 1:
                signals.append('底背离')
            if row.get('顶背离', 0) == 1:
                signals.append('顶背离')
            
            signals_str = ', '.join(signals)
            print(f"{row['symbol']:<8} | {row['price']:8.2f} | {row['Volume']:12.0f} | {row['turnover']:12.0f} | "
                  f"{row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                  f"{row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                  f"{row['vol_times']:8.1f} | {row['vol_color']:4.0f} | "
                  f"{row['gold_vol_count']:4d} | {row['double_vol_count']:4d} | "
                  f"{row.get('DIF', 0):8.2f} | {row.get('DEA', 0):8.2f} | {row.get('MACD', 0):8.2f} | "
                  f"{row.get('EMAMACD', 0):8.2f} | {signals_str:<40}")
        
        print("=" * 220)
        print(f"共发现 {len(results)} 只股票有信号")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 