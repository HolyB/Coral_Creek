import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time
import threading
import concurrent.futures
from tqdm import tqdm
import requests
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局变量存储公司信息
COMPANY_INFO = {}

# 定义富途函数（保持不变）
def REF(series, periods=1):
    return pd.Series(series).shift(periods).values

def EMA(series, periods):
    return pd.Series(series).ewm(span=periods, adjust=False).mean().values

def SMA(series, periods, weight=1):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def IF(condition, value_if_true, value_if_false):
    return np.where(condition, value_if_true, value_if_false)

def POW(series, power):
    return np.power(series, power)

def LLV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).min().values

def HHV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).max().values

def fetch_tickers_page(cursor=None):
    """单个线程获取一页股票数据"""
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
    if cursor:
        params['cursor'] = cursor
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            tickers = [item['ticker'] for item in data.get('results', []) if item.get('market') == 'stocks' and item.get('active')]
            next_cursor = data.get('next_url', '').split('cursor=')[1] if 'next_url' in data else None
            return tickers, next_cursor
        else:
            with print_lock:
                print(f"请求失败: {response.status_code} - {response.text}")
            return [], None
    except Exception as e:
        with print_lock:
            print(f"请求出错: {e}")
        return [], None

def get_all_tickers():
    """使用多线程并发获取所有股票代码，优化为付费账户"""
    ticker_cache_file = 'tickers_cache.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ticker_cache_path = os.path.join(current_dir, ticker_cache_file)
    
    if os.path.exists(ticker_cache_path):
        with open(ticker_cache_path, 'r', encoding='utf-8') as f:
            tickers = json.load(f)
            print(f"从缓存加载股票列表，共 {len(tickers)} 只股票")
    else:
        tickers = set()
        initial_cursor = None
        cursors = [initial_cursor]
        
        first_page_tickers, next_cursor = fetch_tickers_page()
        tickers.update(first_page_tickers)
        while next_cursor:
            cursors.append(next_cursor)
            _, next_cursor = fetch_tickers_page(next_cursor)
        
        max_workers = min(50, len(cursors))
        with print_lock:
            print(f"开始并发获取股票列表，总页数: {len(cursors)}，使用 {max_workers} 个线程")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cursor = {executor.submit(fetch_tickers_page, cursor): cursor for cursor in cursors[1:]}
            for future in tqdm(as_completed(future_to_cursor), total=len(cursors)-1, desc="Fetching Tickers"):
                page_tickers, _ = future.result()
                with results_lock:
                    tickers.update(page_tickers)
        
        chinese_stocks = additional_chinese_stocks()
        tickers.update(chinese_stocks)
        
        tickers = list(tickers)
        with open(ticker_cache_path, 'w', encoding='utf-8') as f:
            json.dump(tickers, f, ensure_ascii=False, indent=2)
        print(f"股票列表已缓存到 {ticker_cache_path}，共 {len(tickers)} 只股票")
    
    return tickers

def additional_chinese_stocks():
    """完整中概股列表（保持不变）"""
    chinese_tickers = [
        'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI', 'BILI', 'BIDU', 'NTES', 'TME', 
        'EDU', 'TAL', 'HTHT', 'GDS', 'IQ', 'KC', 'ATHM', 'HUYA', 'VIPS', 'ZH', 
        'DADA', 'BGNE', 'ZLAB', 'YUMC', 'MNSO', 'API', 'TIGR', 'FUTU', 'UP', 
        'QFIN', 'LU', 'BEKE', 'TCOM', 'ZTO', 'BZUN', 'WB', 'MOMO', 'YY', 'SOHU', 
        'NOAH', 'LX', 'FINV', 'GOTU', 'HOLI', 'NIU', 'TUYA', 'WBAI', 'JKS', 
        'DQ', 'CSIQ', 'RENN', 'LEJU', 'EH', 'CANG', 'UXIN', 'KNDI', 'CAAS', 
        'XNET', 'SOGO', 'WIMI', 'YRD', 'XYF', 'HUIZ', 'QTT', 'CCNC', 'CMCM', 
        'LIZI', 'TOUR', 'CTK', 'NCTY', 'ZJYL', 'AMBO', 'REDU', 'COE', 'ONE', 
        'DLNG', 'FENG', 'GLG', 'GRCL', 'JZ', 'TEDU', 'LKCO', 'AIHS', 'DTSS', 
        'XIN', 'SINO', 'QH', 'SEED', 'WAFU', 'WEI', 'CNTF', 'JRJC', 'BEDU', 
        'MOHO', 'RYB', 'SFUN', 'YIN', 'CNET', 'CCM', 'CLPS', 'DOGZ', 'HGSH', 
        'HLG', 'HX', 'NIU', 'OGEN', 'QSG', 'RLYB', 'SG', 'TC', 'UTME', 'ZCMD', 
        'PETZ', 'PHCF', 'RAAS', 'RCON', 'SDH', 'SNDA', 'SXTC', 'THTI', 'UCAR', 
        'XRS', 'YI', 'YJ', 'ZKIN'
    ]
    return list(set(chinese_tickers))

def fetch_company_info(ticker):
    """获取单个股票的公司信息（保持不变）"""
    api_key = "qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D"
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results'].get('name'):
                return ticker, data['results']['name']
        return ticker, f"{ticker} Stock"
    except Exception as e:
        with print_lock:
            print(f"获取 {ticker} 信息失败: {e}")
        return ticker, f"{ticker} Stock"

def get_company_info():
    """获取公司信息字典，仅使用Polygon API，优先使用缓存（保持不变）"""
    cache_file = 'company_info_cache.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, cache_file)
    
    tickers = get_all_tickers()
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            company_dict = json.load(f)
            if len(company_dict) >= len(tickers) * 0.9:
                print(f"从缓存加载公司信息: {len(company_dict)} 家公司")
                return company_dict
            else:
                print(f"缓存数据不完整，仅有 {len(company_dict)} 家公司，重新获取缺失部分")
    else:
        company_dict = {}

    missing_tickers = [t for t in tickers if t not in company_dict]
    if missing_tickers:
        print(f"\n需要获取 {len(missing_tickers)} 只股票的信息")
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_ticker = {executor.submit(fetch_company_info, ticker): ticker for ticker in missing_tickers}
            for i, future in enumerate(tqdm(as_completed(future_to_ticker), total=len(missing_tickers), desc="Fetching Company Info")):
                ticker, name = future.result()
                with results_lock:
                    company_dict[ticker] = name
                if (i + 1) % 100 == 0:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(company_dict, f, ensure_ascii=False, indent=2)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(company_dict, f, ensure_ascii=False, indent=2)
        print("公司信息已保存到缓存")
    
    return company_dict

def init_company_info():
    """初始化公司信息（保持不变）"""
    global COMPANY_INFO
    print("\n正在初始化公司信息...")
    COMPANY_INFO = get_company_info()
    print(f"初始化完成，共加载 {len(COMPANY_INFO)} 家公司信息")

def process_single_stock(symbol):
    """处理单个股票，仅关注BLUE和LIRED信号，调整为最近6天和5周"""
    try:
        fetcher_daily = StockDataFetcher(symbol, source='polygon', interval='1d')
        data_daily = fetcher_daily.get_stock_data()
        
        if data_daily is None or data_daily.empty:
            return None
            
        # 计算成交额（万元）并添加过滤
        latest_turnover = data_daily['Volume'].iloc[-1] * data_daily['Close'].iloc[-1] / 10000
        if latest_turnover < 100:  # 过滤成交额小于100万的股票
            return None
        
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        OPEN_D, HIGH_D, LOW_D, CLOSE_D = data_daily['Open'].values, data_daily['High'].values, data_daily['Low'].values, data_daily['Close'].values
        OPEN_W, HIGH_W, LOW_W, CLOSE_W = data_weekly['Open'].values, data_weekly['High'].values, data_weekly['Low'].values, data_weekly['Close'].values
        
        VAR1_D = REF((LOW_D + OPEN_D + CLOSE_D + HIGH_D) / 4, 1)
        VAR2_D = SMA(np.abs(LOW_D - VAR1_D), 13, 1) / SMA(np.maximum(LOW_D - VAR1_D, 0), 10, 1)
        VAR3_D = EMA(VAR2_D, 10)
        VAR4_D = LLV(LOW_D, 33)
        VAR5_D = EMA(IF(LOW_D <= VAR4_D, VAR3_D, 0), 3)
        VAR6_D = POW(np.abs(VAR5_D), 0.3) * np.sign(VAR5_D)
        
        VAR21_D = SMA(np.abs(HIGH_D - VAR1_D), 13, 1) / SMA(np.minimum(HIGH_D - VAR1_D, 0), 10, 1)
        VAR31_D = EMA(VAR21_D, 10)
        VAR41_D = HHV(HIGH_D, 33)
        VAR51_D = EMA(IF(HIGH_D >= VAR41_D, -VAR31_D, 0), 3)
        VAR61_D = POW(np.abs(VAR51_D), 0.3) * np.sign(VAR51_D)
        
        max_value_daily = np.nanmax(np.maximum(VAR6_D, np.abs(VAR61_D)))
        RADIO1_D = 458 / max_value_daily if max_value_daily > 0 else 1
        BLUE_D = IF(VAR5_D > REF(VAR5_D, 1), VAR6_D * RADIO1_D, 0)
        LIRED_D = IF(VAR51_D > REF(VAR51_D, 1), -VAR61_D * RADIO1_D, 0)
        
        VAR1_W = REF((LOW_W + OPEN_W + CLOSE_W + HIGH_W) / 4, 1)
        VAR2_W = SMA(np.abs(LOW_W - VAR1_W), 13, 1) / SMA(np.maximum(LOW_W - VAR1_W, 0), 10, 1)
        VAR3_W = EMA(VAR2_W, 10)
        VAR4_W = LLV(LOW_W, 33)
        VAR5_W = EMA(IF(LOW_W <= VAR4_W, VAR3_W, 0), 3)
        VAR6_W = POW(np.abs(VAR5_W), 0.3) * np.sign(VAR5_W)
        
        VAR21_W = SMA(np.abs(HIGH_W - VAR1_W), 13, 1) / SMA(np.minimum(HIGH_W - VAR1_W, 0), 10, 1)
        VAR31_W = EMA(VAR21_W, 10)
        VAR41_W = HHV(HIGH_W, 33)
        VAR51_W = EMA(IF(HIGH_W >= VAR41_W, -VAR31_W, 0), 3)
        VAR61_W = POW(np.abs(VAR51_W), 0.3) * np.sign(VAR51_W)
        
        max_value_weekly = np.nanmax(np.maximum(VAR6_W, np.abs(VAR61_W)))
        RADIO1_W = 350 / max_value_weekly if max_value_weekly > 0 else 1
        BLUE_W = IF(VAR5_W > REF(VAR5_W, 1), VAR6_W * RADIO1_W, 0)
        LIRED_W = IF(VAR51_W > REF(VAR51_W, 1), -VAR61_W * RADIO1_W, 0)
        
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'VAR5': VAR5_D, 'VAR6': VAR6_D, 'VAR51': VAR51_D,
            'BLUE': BLUE_D, 'LIRED': LIRED_D
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'VAR5': VAR5_W, 'VAR6': VAR6_W, 'VAR51': VAR51_W,
            'BLUE': BLUE_W, 'LIRED': LIRED_W
        }, index=data_weekly.index)
        
        if symbol in ['PLTR', 'TSLA']:
            recent_daily_20 = df_daily.tail(20)
            recent_weekly = df_weekly.tail(5)  # 调整为5周
            latest_daily = df_daily.iloc[-1]
            latest_weekly = df_weekly.iloc[-1]
            
            with print_lock:
                print(f"\n=== {symbol} 中间变量检查 ===")
                print(f"日线RADIO1: {RADIO1_D}")
                print(f"周线RADIO1: {RADIO1_W}")
                print(f"日线VAR5（最近20天）:\n{df_daily['VAR5'].tail(20).tolist()}")
                print(f"日线VAR6（最近20天）:\n{df_daily['VAR6'].tail(20).tolist()}")
                print(f"日线VAR51（最近20天）:\n{df_daily['VAR51'].tail(20).tolist()}")
                print(f"周线VAR5（最近5周）:\n{df_weekly['VAR5'].tail(5).tolist()}")
                print(f"周线VAR6（最近5周）:\n{df_weekly['VAR6'].tail(5).tolist()}")
                print(f"周线VAR51（最近5周）:\n{df_weekly['VAR51'].tail(5).tolist()}")
                
                print(f"\n=== 调试 {symbol} 的所有数值 ===")
                print(f"日线数据（最新一天）:\n{latest_daily.to_dict()}")
                print(f"周线数据（最新一周）:\n{latest_weekly.to_dict()}")
                print(f"最近20天日线BLUE:\n{recent_daily_20['BLUE'].tolist()}")
                print(f"最近5周周线BLUE:\n{recent_weekly['BLUE'].tolist()}")
                print(f"最近20天日线LIRED:\n{recent_daily_20['LIRED'].tolist()}")
                print(f"最近5周周线LIRED:\n{recent_weekly['LIRED'].tolist()}")
        
        # 调整为最近6天和5周
        recent_daily = df_daily.tail(6)
        recent_weekly = df_weekly.tail(5)
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        
        day_blue_count = len(recent_daily[recent_daily['BLUE'] > 150])
        week_blue_count = len(recent_weekly[recent_weekly['BLUE'] > 150])
        day_lired_count = len(recent_daily[recent_daily['LIRED'] < -170])
        week_lired_count = len(recent_weekly[recent_weekly['LIRED'] < -170])
        
        has_blue_signal = day_blue_count >= 3 or week_blue_count >= 2
        has_lired_signal = day_lired_count >= 3 or week_lired_count >= 2
        
        if has_blue_signal or has_lired_signal:
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'] / 10000,  # 单位：万
                'blue_daily': latest_daily['BLUE'],
                'max_blue_daily': recent_daily['BLUE'].max(),
                'blue_days': day_blue_count,
                'blue_weekly': latest_weekly['BLUE'],
                'max_blue_weekly': recent_weekly['BLUE'].max(),
                'blue_weeks': week_blue_count,
                'lired_daily': latest_daily['LIRED'],
                'lired_days': day_lired_count,
                'lired_weekly': latest_weekly['LIRED'],
                'lired_weeks': week_lired_count
            }
            return result
        
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
            import traceback
            traceback.print_exc()
    return None

def scan_signals_parallel(max_workers=30, batch_size=100, cooldown=5, limit=None):
    """并行扫描股票信号，使用批处理和进度条，可限制扫描数量（保持不变）"""
    tickers = get_all_tickers()
    
    for debug_symbol in ['PLTR', 'TSLA']:
        if debug_symbol not in tickers:
            tickers.insert(0, debug_symbol)
    
    if limit is not None:
        tickers = tickers[:limit]
        print(f"限制扫描前 {limit} 只股票，实际扫描 {len(tickers)} 只")
    else:
        print(f"扫描全部 {len(tickers)} 只股票")
    
    all_results = []
    batch_count = (len(tickers) + batch_size - 1) // batch_size
    
    with tqdm(total=batch_count, desc="Batch Progress") as batch_pbar:
        for i in range(0, len(tickers), batch_size):
            batch_start_time = time.time()
            batch_tickers = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            with print_lock:
                print(f"\nProcessing batch {batch_num}/{batch_count} ({len(batch_tickers)} stocks)")
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_stock, symbol): symbol for symbol in batch_tickers}
                for future in tqdm(as_completed(futures), total=len(batch_tickers), desc="Stock Progress"):
                    result = future.result()
                    if result is not None:
                        with results_lock:
                            batch_results.append(result)
            
            if batch_results:
                all_results.extend(batch_results)
                with print_lock:
                    print(f"Batch {batch_num} found {len(batch_results)} stocks with signals")
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            with print_lock:
                print(f"Batch {batch_num} processing time: {batch_time:.2f} seconds")
            batch_pbar.update(1)
            
            if i + batch_size < len(tickers):
                with print_lock:
                    print(f"Cooldown for {cooldown} seconds before next batch...")
                time.sleep(cooldown)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def main(limit=None):
    """主函数，支持限制扫描数量，调整为最近6天和5周"""
    init_company_info()
    
    start_time = time.time()
    print("\n开始扫描股票...")
    
    results = scan_signals_parallel(max_workers=30, batch_size=500, cooldown=10, limit=limit)
    
    if not results.empty:
        results['company_name'] = results['symbol'].map(lambda x: COMPANY_INFO.get(x, 'N/A'))
        
        print("\n发现信号的股票（仅BLUE和LIRED，成交额>100万）：")
        print("=" * 160)
        print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交额(万)':>12} | {'日BLUE':>8} | {'日BLUE天数':>4} | {'日LIRED':>8} | {'日LIRED数':>4} | "
              f"{'周BLUE':>8} | {'周BLUE周数':>4} | {'周LIRED':>8} | {'周LIRED数':>4} | {'信号':<20}")
        print("-" * 160)
        
        signal_counts = {
            '日BLUE>150': 0,
            '周BLUE>150': 0,
            '日LIRED<-170': 0,
            '周LIRED<-170': 0
        }
        
        count = 0
        for _, row in results.iterrows():
            signals = []
            if row['blue_days'] >= 3:
                signals.append(f'日BLUE>150({row["blue_days"]}天)')
                signal_counts['日BLUE>150'] += 1
            if row['blue_weeks'] >= 2:
                signals.append(f'周BLUE>150({row["blue_weeks"]}周)')
                signal_counts['周BLUE>150'] += 1
            if row['lired_days'] >= 3:
                signals.append(f'日LIRED<-170({row["lired_days"]}天)')
                signal_counts['日LIRED<-170'] += 1
            if row['lired_weeks'] >= 2:
                signals.append(f'周LIRED<-170({row["lired_weeks"]}周)')
                signal_counts['周LIRED<-170'] += 1
            
            signals_str = ', '.join(signals)
            if signals_str:
                count += 1
                print(f"{row['symbol']:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['turnover']:12.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                      f"{row['lired_daily']:8.2f} | {row['lired_days']:4d} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                      f"{row['lired_weekly']:8.2f} | {row['lired_weeks']:4d} | {signals_str:<20}")
        
        print("=" * 160)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, count in signal_counts.items():
            if count > 0:
                print(f"{signal:<15}: {count:>5} 只")
        print("-" * 40)
        print(f"共发现 {count} 只股票有信号")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'signals_blue_lired_{timestamp}.csv'
        results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main(limit=20000)