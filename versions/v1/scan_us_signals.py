# -*- coding: utf-8 -*-
"""
美股信号扫描器 (US Stock Signal Scanner)
基于 Polygon API 获取数据，扫描 BLUE 信号和黑马信号
"""
import os
import sys

# 必须在导入其他模块之前设置编码
if sys.platform == 'win32':
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # 设置控制台代码页为UTF-8
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    # 重新打开stdout和stderr以使用UTF-8编码
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # 忽略除零警告
import time
from datetime import datetime, timedelta
import requests
import threading
import concurrent.futures
from database_manager import StockDatabase
from tqdm import tqdm
from polygon import RESTClient
from notification_manager import NotificationManager

# 创建线程锁
results_lock = threading.Lock()
print_lock = threading.Lock()

# ==================== 技术指标函数 (与A股扫描器一致) ====================

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

def AVEDEV(series, periods):
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values

def MA(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def calculate_heima_signal(high, low, close, open_price):
    """计算黑马信号"""
    VAR1 = (high + low + close) / 3
    ma_var1 = MA(VAR1, 14)
    avedev_var1 = AVEDEV(VAR1, 14)
    avedev_var1 = np.where(avedev_var1 == 0, 0.0001, avedev_var1)
    VAR2 = (VAR1 - ma_var1) / (0.015 * avedev_var1)
    
    low_series = pd.Series(low)
    is_local_low = (low_series == low_series.rolling(window=16, min_periods=1, center=True).min())
    has_amplitude = (high - low) > 0.04
    VAR3 = np.where(is_local_low & has_amplitude, 80, 0)
    
    close_series = pd.Series(close)
    rolling_min = close_series.rolling(window=22, min_periods=1).min()
    rolling_max = close_series.rolling(window=22, min_periods=1).max()
    
    pct_change = close_series.pct_change()
    is_rising = pct_change > 0
    was_falling_1 = pct_change.shift(1) <= 0
    
    near_low = (close_series - rolling_min) / (rolling_max - rolling_min + 0.0001) < 0.2
    VAR4 = np.where(is_rising & was_falling_1 & near_low, 50, 0)
    
    heima_signal = (VAR2 < -110) & (VAR4 > 0)
    juedi_signal = (VAR2 < -110) & (VAR3 > 0)
    
    return heima_signal, juedi_signal, VAR2

def calculate_blue_signal(open_p, high, low, close):
    """计算BLUE信号"""
    VAR1 = REF((low + open_p + close + high) / 4, 1)
    VAR2 = SMA(np.abs(low - VAR1), 13, 1) / SMA(np.maximum(low - VAR1, 0), 10, 1)
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(low, 33)
    VAR5 = EMA(IF(low <= VAR4, VAR3, 0), 3)
    VAR6 = POW(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    VAR21 = SMA(np.abs(high - VAR1), 13, 1) / SMA(np.minimum(high - VAR1, 0), 10, 1)
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(high, 33)
    VAR51 = EMA(IF(high >= VAR41, -VAR31, 0), 3)
    VAR61 = POW(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    max_value = np.nanmax(np.maximum(VAR6, np.abs(VAR61)))
    RADIO1 = 200 / max_value if max_value > 0 else 1
    BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
    return BLUE

# ==================== 核心处理 ====================

def process_us_stock(ticker, api_key, max_retries=2):
    """处理单只美股，使用Polygon API，优化速度"""
    for attempt in range(max_retries):
        try:
            # 使用Polygon API获取数据
            client = RESTClient(api_key)
            
            # 优化：只获取过去1年数据（足够计算信号，速度更快）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 改为1年
            
            # 获取日线数据 - 优化：直接转换为list，避免多次迭代
            aggs = []
            try:
                # 一次性获取所有数据
                aggs_iter = client.list_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    limit=50000
                )
                # 转换为list（更快）
                aggs_list = list(aggs_iter)
                
                for a in aggs_list:
                    aggs.append({
                        'Date': pd.Timestamp.fromtimestamp(a.timestamp/1000),
                        'Open': a.open,
                        'High': a.high,
                        'Low': a.low,
                        'Close': a.close,
                        'Volume': a.volume,
                    })
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 减少重试延迟
                    continue
                else:
                    return None
            
            if not aggs:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(aggs)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            if df.empty or len(df) < 50:
                return None
            
            # Polygon数据列名: Open, High, Low, Close, Volume
            OPEN = df['Open'].values
            HIGH = df['High'].values
            LOW = df['Low'].values
            CLOSE = df['Close'].values
            VOLUME = df['Volume'].values
            
            # 1. 计算日线 BLUE
            BLUE_D = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
            
            # 2. 计算周线数据
            df_weekly = df.resample('W-MON').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            if df_weekly.empty:
                return None
                
            OPEN_W = df_weekly['Open'].values
            HIGH_W = df_weekly['High'].values
            LOW_W = df_weekly['Low'].values
            CLOSE_W = df_weekly['Close'].values
            
            BLUE_W = calculate_blue_signal(OPEN_W, HIGH_W, LOW_W, CLOSE_W)
            
            # 3. 计算黑马信号
            heima_d, juedi_d, _ = calculate_heima_signal(HIGH, LOW, CLOSE, OPEN)
            heima_w, juedi_w, _ = calculate_heima_signal(HIGH_W, LOW_W, CLOSE_W, OPEN_W)
            
            # 4. 分析结果
            # 获取最近6天的信号（与A股保持一致）
            recent_blue_d = BLUE_D[-6:]
            recent_blue_w = BLUE_W[-5:]  # 最近5周
            
            # 获取日期索引（用于记录信号日期和数值）
            recent_dates_d = df.index[-6:].strftime('%Y-%m-%d').tolist()
            recent_dates_w = df_weekly.index[-5:].strftime('%Y-%m-%d').tolist()
            
            # 记录每次信号出现的日期和对应的BLUE数值（格式：列表，每个元素是{"date": "2025-12-30", "value": 150.5}）
            day_blue_mask = recent_blue_d > 100
            week_blue_mask = recent_blue_w > 100
            day_blue_dates = [{"date": recent_dates_d[i], "value": float(recent_blue_d[i])} 
                             for i in range(len(recent_dates_d)) if day_blue_mask[i]]
            week_blue_dates = [{"date": recent_dates_w[i], "value": float(recent_blue_w[i])} 
                              for i in range(len(recent_dates_w)) if week_blue_mask[i]]
            
            blue_days = np.sum(day_blue_mask)
            blue_weeks = np.sum(week_blue_mask)
            
            # 使用与A股相同的阈值：日线需要3天，周线需要2周
            has_day_blue = blue_days >= 3  # 改为3天（原来是2天）
            has_week_blue = blue_weeks >= 2  # 改为2周（原来是1周，太宽松）
            
            # 黑马信号（保持原逻辑）
            recent_heima_d = heima_d[-6:] | juedi_d[-6:]
            recent_heima_w = heima_w[-5:] | juedi_w[-5:]
            
            has_heima = np.any(recent_heima_d) or np.any(recent_heima_w)
            
            # 记录黑马信号日期（只记录日期，不记录数值）
            heima_dates = []
            if np.any(recent_heima_d):
                heima_day_dates = [recent_dates_d[i] for i in range(len(recent_dates_d)) if recent_heima_d[i]]
                heima_dates.extend(heima_day_dates)
            if np.any(recent_heima_w):
                heima_week_dates = [recent_dates_w[i] for i in range(len(recent_dates_w)) if recent_heima_w[i]]
                heima_dates.extend(heima_week_dates)
            heima_dates = sorted(list(set(heima_dates)))  # 去重并排序
            
            # 只有在有信号时才返回（不包含黑马信号，因为黑马信号太多）
            # 只返回BLUE信号，黑马信号作为额外信息记录
            if has_day_blue or has_week_blue:
                last_price = CLOSE[-1]
                last_vol = VOLUME[-1]
                turnover = (last_price * last_vol) / 10000 # 万美元
                
                # 获取最近一次BLUE>100的值，而不是最近一周的值（可能为0）
                # 日线：最近一次BLUE>100的值
                day_blue_values = recent_blue_d[recent_blue_d > 100]
                latest_day_blue_value = float(day_blue_values[-1]) if len(day_blue_values) > 0 else float(recent_blue_d[-1])
                
                # 周线：最近一次BLUE>100的值
                week_blue_values = recent_blue_w[recent_blue_w > 100]
                latest_week_blue_value = float(week_blue_values[-1]) if len(week_blue_values) > 0 else float(recent_blue_w[-1])
                
                return {
                    'symbol': ticker,
                    'name': ticker, # 美股通常直接用代码
                    'price': last_price,
                    'turnover': turnover,
                    'blue_daily': latest_day_blue_value,  # 使用最近一次>100的值
                    'blue_days': int(blue_days),
                    'blue_weekly': latest_week_blue_value,  # 使用最近一次>100的值
                    'blue_weeks': int(blue_weeks),
                    'has_day_blue': bool(has_day_blue),
                    'has_week_blue': bool(has_week_blue),
                    'has_day_heima': bool(np.any(recent_heima_d)),
                    'has_week_heima': bool(np.any(recent_heima_w)),
                    'day_blue_dates': day_blue_dates,  # 日线BLUE信号日期列表
                    'week_blue_dates': week_blue_dates,  # 周线BLUE信号日期列表
                    'heima_dates': heima_dates  # 黑马信号日期列表
                }
            else:
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                # 最后一次尝试失败，静默失败（不打印，避免刷屏）
                return None
    return None

def get_all_us_tickers(api_key, limit=None):
    """从Polygon API获取所有美股ticker列表"""
    base_url = "https://api.polygon.io/v3/reference/tickers"
    all_tickers = []
    cursor = None
    
    print("Fetching all US stock tickers from Polygon API...")
    
    while True:
        params = {
            'market': 'stocks',
            'active': True,
            'sort': 'ticker',
            'order': 'asc',
            'limit': 1000,
            'apiKey': api_key
        }
        if cursor:
            params['cursor'] = cursor
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                # 只获取活跃的股票
                tickers = [item['ticker'] for item in results 
                          if item.get('market') == 'stocks' and item.get('active')]
                all_tickers.extend(tickers)
                
                print(f"  Fetched {len(all_tickers)} tickers...")
                
                # 检查是否有下一页
                next_url = data.get('next_url')
                if next_url and 'cursor=' in next_url:
                    cursor = next_url.split('cursor=')[1].split('&')[0]
                else:
                    break
                    
                # 如果设置了限制，达到后停止
                if limit and len(all_tickers) >= limit:
                    all_tickers = all_tickers[:limit]
                    break
                    
                time.sleep(0.1)  # 进一步减少延迟
            else:
                print(f"API request failed: {response.status_code}")
                break
        except Exception as e:
            print(f"Error fetching ticker list: {e}")
            break
    
    return list(set(all_tickers))  # 去重

def main():
    print("Starting US stock scan (using Polygon API)...")
    
    # Polygon API密钥
    api_key = os.getenv('POLYGON_API_KEY', 'qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D')
    
    # 获取所有美股列表
    use_file = False
    tickers_file = os.path.join(os.path.dirname(__file__), 'us_tickers.txt')
    
    # 检查是否有本地文件，如果有且用户想用文件，就用文件；否则从API获取
    if os.path.exists(tickers_file) and use_file:
        print(f"Reading ticker list from file: {tickers_file}")
        with open(tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        # 从Polygon API获取所有美股（限制数量）
        limit = 2000  # 限制扫描2000只股票
        tickers = get_all_us_tickers(api_key, limit=limit)
        if not tickers:
            print("Failed to get ticker list from API, trying local file...")
            if os.path.exists(tickers_file):
                with open(tickers_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
            else:
                print("Cannot get ticker list, exiting")
                return
        
    print(f"Loaded {len(tickers)} US stocks")
    
    results = []
    
    # 并行扫描参数
    max_workers = 20  # 并行线程数
    batch_size = 100  # 每批处理的股票数
    
    print(f"\nScanning {len(tickers)} stocks with {max_workers} parallel workers...")
    print("Using batch processing to optimize performance\n")
    
    completed_count = 0
    
    def process_result(future, ticker):
        nonlocal completed_count
        try:
            result = future.result(timeout=30)
            if result:
                with results_lock:
                    results.append(result)
                with print_lock:
                    print(f"[OK] {ticker}: Signal found")
        except:
            pass  # 静默失败
        finally:
            with results_lock:
                completed_count += 1
                pbar.update(1)
    
    # 分批并行处理
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(tickers), desc="Progress", ncols=100, file=sys.stdout,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        # 使用线程池并行处理当前批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_us_stock, ticker, api_key): ticker 
                for ticker in batch_tickers
            }
            
            # 添加完成回调
            for future, ticker in futures.items():
                future.add_done_callback(lambda f, t=ticker: process_result(f, t))
            
            # 等待当前批次完成（最多5分钟）
            concurrent.futures.wait(futures.keys(), timeout=300)
        
        # 批次之间短暂休息
        if batch_num < total_batches - 1:
            time.sleep(1)
    
    pbar.close()
            
    if results:
        df = pd.DataFrame(results)
        print(f"\nFound signals in {len(df)} stocks")
        
        # 保存到数据库
        try:
            db = StockDatabase()
            db.save_results_from_df(df, market='US')
            db.close()
            print(f"Saved {len(df)} records to database")
        except Exception as e:
            print(f"Database save failed: {e}")
            
        print("\nResults:")
        print(df[['symbol', 'price', 'blue_days', 'blue_weeks']].to_string())
        
        # 发送通知
        try:
            print("\nSending notification...")
            nm = NotificationManager()
            if nm.config.get('email_enabled'):
                # 准备数据
                total_scanned = len(tickers)
                blue_stocks = df[df['has_day_blue'] | df['has_week_blue']].to_dict('records')
                heima_stocks = df[df['has_day_heima'] | df['has_week_heima']].to_dict('records')
                
                # 检查自选股命中
                favorites_hits = []
                try:
                    favorites_df = db.get_all_favorites()
                    if not favorites_df.empty:
                        fav_symbols = set(favorites_df['symbol'].tolist())
                        favorites_hits = df[df['symbol'].isin(fav_symbols)].to_dict('records')
                except:
                    pass
                
                nm.send_scan_report(
                    market='US',
                    total_scanned=total_scanned,
                    blue_stocks=blue_stocks,
                    heima_stocks=heima_stocks,
                    favorites_hits=favorites_hits
                )
            else:
                print("Email notification disabled in config.")
        except Exception as e:
            print(f"Notification failed: {e}")
            
    else:
        print("No signals found")

if __name__ == "__main__":
    main()


