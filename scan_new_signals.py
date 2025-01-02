import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.stock_analysis import StockAnalysis
from tqdm import tqdm
import requests
import time

print_lock = threading.Lock()

def get_all_tickers():
    """获取所有股票代码列表"""
    key = "qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D"
    base_url = "https://api.polygon.io/v3/reference/tickers"
    
    params = {
        'market': 'stocks',
        'active': True,
        'sort': 'ticker',
        'order': 'asc',
        'limit': 100,  # 直接限制为100个
        'apiKey': key
    }
    
    tickers = []
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            print(f"获取股票列表失败: {data.get('error')}")
            return []
            
        # 只获取第一页的100个股票
        for item in data['results']:
            if item.get('market') == 'stocks' and item.get('active'):
                tickers.append(item['ticker'])
                
    except Exception as e:
        print(f"获取股票列表时出错: {e}")
    
    print(f"共获取到 {len(tickers)} 只股票")
    return tickers

def process_single_stock(symbol):
    """处理单个股票"""
    try:
        # 获取日线数据
        fetcher = StockDataFetcher(symbol, source='polygon', interval='1d')
        data = fetcher.get_stock_data()
        
        if data is None or data.empty:
            return None
            
        # 计算指标
        analysis = StockAnalysis(data)
        df = analysis.calculate_all_indicators()
        
        if df is None or df.empty:
            return None
        
        # 获取最新数据
        latest = df.iloc[-1]
        recent_daily = df.tail(10)
        recent_5d = df.tail(5)
        
        # 检查原有信号
        signals = {
            'BLUE信号_日线': len(recent_daily[recent_daily['BLUE'] > 150]) >= 3,
            'BLUE信号_周线': len(recent_daily[recent_daily['BLUE'] > 150]) >= 2,
            '笑脸信号_做多_日线': latest['笑脸信号_做多'] == 1,
            '笑脸信号_做空_日线': latest['笑脸信号_做空'] == 1,
            '成交量信号': len(recent_5d[recent_5d['DOUBLE_VOL'] | recent_5d['GOLD_VOL']]) > 0,
            # 新增掘底和黑马信号
            '掘底买点': latest.get('掘底买点', False),
            '黑马信号': latest.get('黑马信号', False)
        }
        
        # 检查四重条件
        has_heima = latest.get('黑马信号', False)
        has_judi = latest.get('掘底买点', False)
        has_blue_daily = len(recent_daily[recent_daily['BLUE'] > 150]) >= 3
        has_blue_weekly = len(recent_daily[recent_daily['BLUE'] > 150]) >= 2
        
        # 只有满足四重条件才返回结果
        if (has_heima or has_judi) and has_blue_daily and has_blue_weekly:
            result = {
                'symbol': symbol,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'turnover': latest['Volume'] * latest['Close'],
                **signals
            }
            
            # 打印信号
            with print_lock:
                print(f"\n{symbol} 发现四重信号:")
                signal_desc = []
                if has_heima:
                    signal_desc.append("黑马信号")
                if has_judi:
                    signal_desc.append("掘底买点")
                if has_blue_daily:
                    signal_desc.append("日线BLUE>150")
                if has_blue_weekly:
                    signal_desc.append("周线BLUE>150")
                print(f"信号组合: {', '.join(signal_desc)}")
                print(f"当前价格: {latest['Close']:.2f}")
                print(f"成交量: {latest['Volume']:.0f}")
            
            return result
            
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
    return None

def scan_new_signals(max_workers=10):
    """扫描股票的所有信号"""
    # 获取股票列表
    print("获取股票列表...")
    symbols = get_all_tickers()  # 已经限制为100个
    
    if not symbols:
        print("未能获取股票列表")
        return
    
    results = []
    print(f"\n开始扫描 {len(symbols)} 个股票...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_stock, symbol) for symbol in symbols]
        
        # 使用tqdm显示进度
        for future in tqdm(futures, total=len(symbols), desc="扫描进度"):
            result = future.result()
            if result:
                results.append(result)
    
    # 转换为DataFrame并排序
    if results:
        df_results = pd.DataFrame(results)
        
        # 计算信号强度（统计每个股票的信号数量）
        signal_columns = [col for col in df_results.columns if col not in ['symbol', 'price', 'volume', 'turnover']]
        df_results['signal_count'] = df_results[signal_columns].sum(axis=1)
        
        # 按信号数量和成交额排序
        df_results = df_results.sort_values(['signal_count', 'turnover'], ascending=[False, False])
        
        # 保存结果
        df_results.to_csv('new_signals.csv', index=False)
        
        print(f"\n共发现 {len(df_results)} 个股票有信号")
        print("\n前10个信号最强的股票：")
        display_cols = ['symbol', 'price', 'signal_count'] + [col for col in signal_columns if df_results[col].any()]
        print(df_results[display_cols].head(10))
    else:
        print("\n未发现任何信号")

if __name__ == "__main__":
    scan_new_signals() 