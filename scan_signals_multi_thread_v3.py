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
import os
import json
from tqdm import tqdm
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher
from scan_signals import get_all_tickers

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局变量存储公司信息
COMPANY_INFO = {}

def init_company_info():
    """初始化公司信息"""
    global COMPANY_INFO
    print("\n正在初始化公司信息...")
    COMPANY_INFO = get_company_info()
    print(f"初始化完成，共加载 {len(COMPANY_INFO)} 家公司信息")

def process_single_stock(symbol):
    """处理单个股票，增加LIRED信号检测"""
    try:
        # 1. 获取日线数据
        fetcher_daily = StockDataFetcher(symbol, source='polygon', interval='1d')
        data_daily = fetcher_daily.get_stock_data()
        
        if data_daily is None or data_daily.empty:
            return None
        
        # 计算最近5天内最低价的涨幅
        if len(data_daily) >= 5:
            recent_data = data_daily.iloc[-5:]  # 最近5天数据
            min_price = recent_data['Low'].min()  # 5天内最低价
            current_price = data_daily['Close'].iloc[-1]  # 当前收盘价
            recent_change = (current_price / min_price - 1) * 100  # 相对最低价的涨幅
            
            if recent_change > 5:  # 涨幅超过5%的股票跳过
                return None
        
        # 重采样为周线数据，使用周一作为时间戳
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        # 2. 计算周线指标
        analysis_weekly = StockAnalysis(data_weekly)
        df_weekly = analysis_weekly.calculate_phantom_force()
        
        # 计算日线指标
        analysis_daily = StockAnalysis(data_daily)
        df_daily = analysis_daily.calculate_phantom_force()
        df_daily = analysis_daily.calculate_heatmap_volume()
        df_daily = analysis_daily.calculate_macd_signals()
        
        recent_weekly = df_weekly.tail(10)
        latest_weekly = df_weekly.iloc[-1]
        recent_daily = df_daily.tail(10)
        latest_daily = df_daily.iloc[-1]
        recent_5d = df_daily.tail(5)
        
        # 检查LIRED连续出现条件
        has_lired_signal = False
        
        # 确保LIRED列存在
        if 'phantom_lired' in df_daily.columns and 'phantom_lired' in df_weekly.columns:
            # 检查日线最近是否连续两天有LIRED值
            day_lired_count = sum(1 for i in range(len(recent_daily)-1, max(0, len(recent_daily)-3), -1) 
                                if recent_daily['phantom_lired'].iloc[i] < 0)
            
            # 检查周线最近是否连续两周有LIRED值
            week_lired_count = sum(1 for i in range(len(recent_weekly)-1, max(0, len(recent_weekly)-3), -1) 
                                 if recent_weekly['phantom_lired'].iloc[i] < 0)
            
            # 如果日线和周线都有连续两个或以上的LIRED，则认为有信号
            has_lired_signal = day_lired_count >= 2 and week_lired_count >= 2
        
        # 检查信号条件
        has_daily_signal = (
            latest_daily['phantom_buy'] or 
            latest_daily['phantom_sell'] or 
            len(recent_daily[recent_daily['phantom_blue'] > 150]) >= 3
        )
        
        has_weekly_signal = (
            latest_weekly['phantom_buy'] or 
            latest_weekly['phantom_sell'] or 
            len(recent_weekly[recent_weekly['phantom_blue'] > 150]) >= 2
        )
        
        has_volume_signal = (
            latest_daily['GOLD_VOL'] or 
            latest_daily['DOUBLE_VOL']
        )
        
        # 修改条件：必须有周线信号或LIRED信号
        if (has_weekly_signal and (has_daily_signal or has_volume_signal)) or has_lired_signal:
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'],
                'pink_daily': latest_daily['phantom_pink'],
                'blue_daily': latest_daily['phantom_blue'],
                'max_blue_daily': recent_daily['phantom_blue'].max(),
                'blue_days': len(recent_daily[recent_daily['phantom_blue'] > 150]),
                'pink_weekly': latest_weekly['phantom_pink'],
                'blue_weekly': latest_weekly['phantom_blue'],
                'max_blue_weekly': recent_weekly['phantom_blue'].max(),
                'blue_weeks': len(recent_weekly[recent_weekly['phantom_blue'] > 150]),
                'smile_long_daily': latest_daily['phantom_buy'],
                'smile_short_daily': latest_daily['phantom_sell'],
                'smile_long_weekly': latest_weekly['phantom_buy'],
                'smile_short_weekly': latest_weekly['phantom_sell'],
                'vol_times': latest_daily['VOL_TIMES'],
                'vol_color': latest_daily['HVOL_COLOR'],
                'gold_vol_count': len(recent_5d[recent_5d['GOLD_VOL']]),
                'double_vol_count': len(recent_5d[recent_5d['DOUBLE_VOL']]),
                # MACD相关指标
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
                '顶背离': latest_daily['顶背离'] if '顶背离' in latest_daily else 0,
                'has_lired_signal': has_lired_signal
            }
            
            # 添加LIRED相关字段，如果存在
            if 'phantom_lired' in df_daily.columns:
                result['lired_daily'] = latest_daily['phantom_lired']
                result['lired_days'] = day_lired_count if 'day_lired_count' in locals() else 0
            
            if 'phantom_lired' in df_weekly.columns:
                result['lired_weekly'] = latest_weekly['phantom_lired']
                result['lired_weeks'] = week_lired_count if 'week_lired_count' in locals() else 0
            
            with results_lock:
                return result
            
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
    
    return None

def scan_signals_parallel(max_workers=30, batch_size=100, cooldown=5):
    """并行扫描股票信号，使用批处理和进度条"""
    # 获取所有股票代码
    print("正在获取股票列表...")
    tickers = get_all_tickers()
    print(f"共获取到 {len(tickers)} 只股票")
    
    # 计算批次数
    all_results = []
    batch_count = (len(tickers) + batch_size - 1) // batch_size
    
    # 使用tqdm创建外层进度条来显示批次进度
    with tqdm(total=batch_count, desc="批次进度") as batch_pbar:
        # 按批次处理
        for i in range(0, len(tickers), batch_size):
            batch_start_time = time.time()
            batch_tickers = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"\n处理第 {batch_num}/{batch_count} 批 ({len(batch_tickers)} 只股票)")
            
            # 使用tqdm创建内层进度条来显示当前批次内的处理进度
            batch_results = []
            completed_count = 0
            
            # 定义回调函数处理完成的任务
            def process_result(future):
                nonlocal completed_count
                result = future.result()
                if result is not None:
                    with results_lock:
                        batch_results.append(result)
                
                # 更新进度条和计数
                with results_lock:
                    completed_count += 1
                    stock_pbar.update(1)
            
            # 使用进度条显示当前批次内的处理进度
            with tqdm(total=len(batch_tickers), desc="股票处理") as stock_pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    for symbol in batch_tickers:
                        future = executor.submit(process_single_stock, symbol)
                        future.add_done_callback(process_result)
                    
                    # 等待当前批次所有任务完成
                    while completed_count < len(batch_tickers):
                        time.sleep(0.1)
            
            # 添加结果到总结果列表
            if batch_results:
                all_results.extend(batch_results)
                print(f"批次 {batch_num} 发现 {len(batch_results)} 只有信号的股票")
            
            # 显示批次耗时
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            print(f"批次 {batch_num} 处理耗时: {batch_time:.2f} 秒")
            
            # 更新外层进度条
            batch_pbar.update(1)
            
            # 如果不是最后一批，等待一段时间再处理下一批
            if i + batch_size < len(tickers):
                print(f"休息 {cooldown} 秒后处理下一批...")
                time.sleep(cooldown)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

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

def get_company_info():
    """获取公司信息字典，优先使用缓存"""
    try:
        cache_file = 'company_info_cache.json'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(current_dir, cache_file)
        
        # 如果缓存文件存在，检查是否完整
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # 检查是否有足够多的公司信息（比如至少3000家）
                if len(existing_data) > 3000:
                    print(f"从缓存加载完整的公司信息: {len(existing_data)} 家公司")
                    return existing_data
                else:
                    print(f"缓存数据不完整，仅有 {len(existing_data)} 家公司，重新获取")
        

        
        def get_single_stock_info(ticker):
            """获取单个股票信息，带重试机制"""
            for attempt in range(3):  # 最多重试3次
                try:
                    # 首先尝试 yfinance
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if info.get('longName'):
                        return ticker, info['longName']
                    elif info.get('shortName'):
                        return ticker, info['shortName']
                    
                    # 如果 yfinance 失败，尝试 polygon
                    api_key = "6X6PDR2zxXXhGxCpBGKXzGOu_2dGYB0t"
                    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        if 'results' in data and data['results'].get('name'):
                            return ticker, data['results']['name']
                    
                    # 如果都失败了，使用网页抓取
                    url = f"https://finance.yahoo.com/quote/{ticker}"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        h1 = soup.find('h1')
                        if h1:
                            name = h1.text.split('(')[0].strip()
                            if name:
                                return ticker, name
                    
                    time.sleep(1)  # 失败后等待1秒再重试
                    
                except Exception as e:
                    print(f"获取 {ticker} 信息失败 (尝试 {attempt+1}/3): {e}")
                    time.sleep(2)  # 出错后等待2秒
            
            return ticker, f"{ticker} Stock"  # 所有尝试都失败后的默认值

        # 获取所有股票代码
        tickers = get_sp500_tickers()  # 使用你原有的函数获取股票列表
        
        # 加载现有缓存（如果存在）
        company_dict = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                company_dict = json.load(f)
        
        # 找出需要获取信息的股票
        missing_tickers = [t for t in tickers if t not in company_dict]
        if missing_tickers:
            print(f"\n需要获取 {len(missing_tickers)} 只股票的信息")
            
# 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(get_single_stock_info, ticker): ticker 
                                  for ticker in missing_tickers}
                
                for i, future in enumerate(as_completed(future_to_ticker), 1):
                    ticker = future_to_ticker[future]
                    try:
                        ticker, name = future.result()
                        company_dict[ticker] = name
                        
                        # 每获取100个公司就保存一次
                        if i % 100 == 0:
                            print(f"已获取 {i}/{len(missing_tickers)} 家公司信息")
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                json.dump(company_dict, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"处理 {ticker} 失败: {e}")
        
        print(f"共获取到 {len(company_dict)} 家公司信息")
        
        # 最终保存
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(company_dict, f, ensure_ascii=False, indent=2)
        print("公司信息已保存到缓存")
        
        return company_dict
        
    except Exception as e:
        print(f"获取公司信息失败: {e}")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

def load_history():
    """加载历史信号记录"""
    try:
        history_file = 'signals_history.json'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(current_dir, history_file)
        
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                # 添加出现次数统计
                for symbol in history:
                    if 'appear_count' not in history[symbol]:
                        history[symbol]['appear_count'] = 1
                return history
        return {}
    except Exception as e:
        print(f"加载历史记录失败: {e}")
        return {}

def save_history(history_dict):
    """保存历史信号记录"""
    try:
        history_file = 'signals_history.json'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(current_dir, history_file)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存历史记录失败: {e}")

def main():
    """主函数"""
    # 加载历史记录
    history_dict = load_history()
    
    # 程序开始时初始化公司信息
    init_company_info()
    
    start_time = time.time()
    print("\n开始扫描股票...")
    
    # 并行扫描股票，增加批处理和进度条
    results = scan_signals_parallel(max_workers=30, batch_size=700, cooldown=5)
    
    if not results.empty:
        # 使用全局的公司信息添加公司名称
        results['company_name'] = results['symbol'].map(lambda x: COMPANY_INFO.get(x, 'N/A'))
        
        # 创建当前信号的字典，用于比较
        current_signals = {}
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
            # 添加LIRED信号
            if row.get('has_lired_signal', False):
                signals.append('日周LIRED信号')
            
            # 保存当前信号
            current_signals[row['symbol']] = {
                'signals': signals,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
        
        # 找出新的信号
        new_signals = {}
        for symbol, info in current_signals.items():
            if symbol not in history_dict or set(info['signals']) != set(history_dict[symbol]['signals']):
                new_signals[symbol] = info
        
        # 更新历史记录
        history_dict.update(current_signals)
        save_history(history_dict)
        
        # 只显示新信号
        if new_signals:
            print("\n新发现的信号:")
            print("=" * 280)
            print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交量':>12} | {'成交额':>12} | "  # 添加公司名称列
                  f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | {'日LIRED':>8} | {'日LIRED数':>4} | "
                  f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | {'周LIRED':>8} | {'周LIRED数':>4} | "
                  f"{'成交倍数':>8} | {'热力值':>4} | {'黄金柱':>4} | {'倍量柱':>4} | "
                  f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'EMAMACD':>8} | {'信号':<40}")
            print("-" * 280)
            
            for symbol in new_signals.keys():
                row = results[results['symbol'] == symbol].iloc[0]
                signals = new_signals[symbol]['signals']
                signals_str = ', '.join(signals)
                
                # 获取LIRED相关值，如果不存在则使用默认值
                lired_daily = row.get('lired_daily', 0)
                lired_days = row.get('lired_days', 0)
                lired_weekly = row.get('lired_weekly', 0)
                lired_weeks = row.get('lired_weeks', 0)
                
                print(f"{symbol:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['Volume']:12.0f} | {row['turnover']:12.0f} | "  # 添加公司名称
                      f"{row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                      f"{lired_daily:8.2f} | {lired_days:4d} | "
                      f"{row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                      f"{lired_weekly:8.2f} | {lired_weeks:4d} | "
                      f"{row['vol_times']:8.1f} | {row['vol_color']:4.0f} | "
                      f"{row['gold_vol_count']:4d} | {row['double_vol_count']:4d} | "
                      f"{row.get('DIF', 0):8.2f} | {row.get('DEA', 0):8.2f} | {row.get('MACD', 0):8.2f} | "
                      f"{row.get('EMAMACD', 0):8.2f} | {signals_str:<40}")
            
            print("=" * 280)
            print(f"共发现 {len(new_signals)} 只新信号股票")
        else:
            print("\n未发现新的信号")
            
        # 保存完整的结果到CSV（包含所有信号）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_v2 = f'signals_v2_{timestamp}.csv'
        
        # 创建v2格式的DataFrame
        df_v2 = pd.DataFrame({
            'symbol': results['symbol'],
            'company_name': results['company_name'],  # 添加公司名称
            'signals': results.apply(lambda row: ', '.join([
                '日PINK上穿10' if row['smile_long_daily'] == 1 else '',
                '日PINK下穿94' if row['smile_short_daily'] == 1 else '',
                f'日BLUE>150({row["blue_days"]}天)' if row['blue_days'] >= 3 else '',
                '周PINK上穿10' if row['smile_long_weekly'] == 1 else '',
                '周PINK下穿94' if row['smile_short_weekly'] == 1 else '',
                f'周BLUE>150({row["blue_weeks"]}周)' if row['blue_weeks'] >= 2 else '',
                f'黄金柱({row["gold_vol_count"]}次)' if row["gold_vol_count"] > 0 else '',
                f'倍量柱({row["double_vol_count"]}次)' if row["double_vol_count"] > 0 else '',
                '日周LIRED信号' if row.get('has_lired_signal', False) else ''
            ]).replace(', ,', ',').replace(', , ,', ',').strip(', '), axis=1),
            'price': results['price'],
            'volume': results['Volume'],
            'turnover': results['turnover'],
            'pink_daily': results['pink_daily'],
            'blue_daily': results['blue_daily'],
            'pink_weekly': results['pink_weekly'],
            'blue_weekly': results['blue_weekly'],
            'lired_daily': results.get('lired_daily', 0),
            'lired_weekly': results.get('lired_weekly', 0),
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
        print("=" * 280)  # 增加显示宽度以适应公司名称和LIRED信息
        print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交量':>12} | {'成交额':>12} | "  # 添加公司名称列
              f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | {'日LIRED':>8} | {'日LIRED数':>4} | "
              f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | {'周LIRED':>8} | {'周LIRED数':>4} | "
              f"{'成交倍数':>8} | {'热力值':>4} | {'黄金柱':>4} | {'倍量柱':>4} | "
              f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'EMAMACD':>8} | {'信号':<40}")
        print("-" * 280)
        
        # 统计各类信号的数量
        signal_counts = {
            '日PINK上穿10': 0,
            '日PINK下穿94': 0,
            '日BLUE>150': 0,
            '周PINK上穿10': 0,
            '周PINK下穿94': 0,
            '周BLUE>150': 0,
            '黄金柱': 0,
            '倍量柱': 0,
            '日周LIRED信号': 0,
            '零轴下金叉': 0,
            '零轴上金叉': 0,
            '零轴上死叉': 0,
            '零轴下死叉': 0,
            '先机信号': 0,
            '底背离': 0,
            '顶背离': 0
        }
        
        for _, row in results.iterrows():
            signals = []
            if row['smile_long_daily'] == 1:
                signals.append('日PINK上穿10')
                signal_counts['日PINK上穿10'] += 1
            if row['smile_short_daily'] == 1:
                signals.append('日PINK下穿94')
                signal_counts['日PINK下穿94'] += 1
            if row['blue_days'] >= 3:
                signals.append(f'日BLUE>150({row["blue_days"]}天)')
                signal_counts['日BLUE>150'] += 1
            if row['smile_long_weekly'] == 1:
                signals.append('周PINK上穿10')
                signal_counts['周PINK上穿10'] += 1
            if row['smile_short_weekly'] == 1:
                signals.append('周PINK下穿94')
                signal_counts['周PINK下穿94'] += 1
            if row['blue_weeks'] >= 2:
                signals.append(f'周BLUE>150({row["blue_weeks"]}周)')
                signal_counts['周BLUE>150'] += 1
            if row['gold_vol_count'] > 0:
                signals.append(f'黄金柱({row["gold_vol_count"]}次)')
                signal_counts['黄金柱'] += 1
            if row['double_vol_count'] > 0:
                signals.append(f'倍量柱({row["double_vol_count"]}次)')
                signal_counts['倍量柱'] += 1
            
            # 添加LIRED信号
            if row.get('has_lired_signal', False):
                signals.append('日周LIRED信号')
                signal_counts['日周LIRED信号'] += 1
            
            # 添加MACD相关信号
            if row.get('零轴下金叉', 0) == 1:
                signals.append('零轴下金叉')
                signal_counts['零轴下金叉'] += 1
            if row.get('零轴上金叉', 0) == 1:
                signals.append('零轴上金叉')
                signal_counts['零轴上金叉'] += 1
            if row.get('零轴上死叉', 0) == 1:
                signals.append('零轴上死叉')
                signal_counts['零轴上死叉'] += 1
            if row.get('零轴下死叉', 0) == 1:
                signals.append('零轴下死叉')
                signal_counts['零轴下死叉'] += 1
            if row.get('先机信号', 0) == 1:
                signals.append('先机信号')
                signal_counts['先机信号'] += 1
            if row.get('底背离', 0) == 1:
                signals.append('底背离')
                signal_counts['底背离'] += 1
            if row.get('顶背离', 0) == 1:
                signals.append('顶背离')
                signal_counts['顶背离'] += 1
            
            signals_str = ', '.join(signals)
            
            # 获取LIRED相关值，如果不存在则使用默认值
            lired_daily = row.get('lired_daily', 0)
            lired_days = row.get('lired_days', 0)
            lired_weekly = row.get('lired_weekly', 0)
            lired_weeks = row.get('lired_weeks', 0)
            
            print(f"{row['symbol']:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['Volume']:12.0f} | {row['turnover']:12.0f} | "  # 添加公司名称
                  f"{row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                  f"{lired_daily:8.2f} | {lired_days:4d} | "
                  f"{row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                  f"{lired_weekly:8.2f} | {lired_weeks:4d} | "
                  f"{row['vol_times']:8.1f} | {row['vol_color']:4.0f} | "
                  f"{row['gold_vol_count']:4d} | {row['double_vol_count']:4d} | "
                  f"{row.get('DIF', 0):8.2f} | {row.get('DEA', 0):8.2f} | {row.get('MACD', 0):8.2f} | "
                  f"{row.get('EMAMACD', 0):8.2f} | {signals_str:<40}")
        
        print("=" * 280)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, count in signal_counts.items():
            if count > 0:  # 只显示有出现的信号
                print(f"{signal:<15}: {count:>5} 只")
        print("-" * 40)
        print(f"共发现 {len(results)} 只股票有信号")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()           #