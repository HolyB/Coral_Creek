import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.stock_analysis import StockAnalysis

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


def scan_signals():
    """扫描股票信号"""
    print("正在获取股票列表...")
    tickers = get_all_tickers()[:2000]
    
    print("\n所有找到的股票代码:")
    print("=" * 80)
    # 每行打印10个股票代码
    for i in range(0, len(tickers), 10):
        row = tickers[i:i+10]
        print(" ".join(f"{ticker:<6}" for ticker in row))
    print("=" * 80)
    print(f"共找到 {len(tickers)} 只股票")
    
    print(f"\n开始扫描 {len(tickers)} 只股票...")
    results = []
    count = 0
    
    for symbol in tickers:
        count += 1
        print(f"\n处理股票 {symbol} ({count}/{len(tickers)})")
        
        try:
            # 获取日线数据
            fetcher_daily = StockDataFetcher(symbol, source='polygon', interval='1d')
            data_daily = fetcher_daily.get_stock_data()
            
            # 获取周线数据
            fetcher_weekly = StockDataFetcher(symbol, source='polygon', interval='1wk')
            data_weekly = fetcher_weekly.get_stock_data()
            
            if data_daily is None or data_daily.empty:
                print(f"{symbol} 无法获取日线数据")
                continue
                
            if data_weekly is None or data_weekly.empty:
                print(f"{symbol} 无法获取周线数据")
                continue
            
            print(f"{symbol} 获取到 {len(data_daily)} 天的日线数据")
            print(f"{symbol} 获取到 {len(data_weekly)} 周的周线数据")
            
            # 计算日线指标
            analysis_daily = StockAnalysis(data_daily)
            df_daily = analysis_daily.calculate_phantom_indicators()
            
            # 计算周线指标
            analysis_weekly = StockAnalysis(data_weekly)
            df_weekly = analysis_weekly.calculate_phantom_indicators()
            
            # 获取最近数据
            recent_daily = df_daily.tail(10)
            latest_daily = df_daily.iloc[-1]
            
            recent_weekly = df_weekly.tail(10)
            latest_weekly = df_weekly.iloc[-1]
            
            # 检查日线蓝线条件
            blue_days = len(recent_daily[recent_daily['BLUE'] > 150])
            
            # 检查周线蓝线条件
            blue_weeks = len(recent_weekly[recent_weekly['BLUE'] > 150])
            
            # 检查信号
            if ((blue_days >= 3) or  # 日线至少3天蓝线大于150
                latest_daily['笑脸信号_做多'] == 1 or 
                latest_daily['笑脸信号_做空'] == 1 or
                blue_weeks >= 2 or  # 周线至少2周蓝线大于150
                latest_weekly['笑脸信号_做多'] == 1 or 
                latest_weekly['笑脸信号_做空'] == 1):
                
                results.append({
                    'symbol': symbol,
                    'price': latest_daily['Close'],
                    'Volume': latest_daily['Volume'],
                    'pink_daily': latest_daily['PINK'],
                    'blue_daily': latest_daily['BLUE'],
                    'max_blue_daily': recent_daily['BLUE'].max(),
                    'blue_days': blue_days,
                    'pink_weekly': latest_weekly['PINK'],
                    'blue_weekly': latest_weekly['BLUE'],
                    'max_blue_weekly': recent_weekly['BLUE'].max(),
                    'blue_weeks': blue_weeks,
                    'smile_long_daily': latest_daily['笑脸信号_做多'],
                    'smile_short_daily': latest_daily['笑脸信号_做空'],
                    'smile_long_weekly': latest_weekly['笑脸信号_做多'],
                    'smile_short_weekly': latest_weekly['笑脸信号_做空']
                })
                
                # 打印具体的信号类型
                signals = []
                if latest_daily['笑脸信号_做多'] == 1:
                    signals.append('日线PINK上穿10')
                if latest_daily['笑脸信号_做空'] == 1:
                    signals.append('日线PINK下穿94')
                if blue_days >= 3:
                    signals.append(f'日线BLUE>150 ({blue_days}天)')
                if latest_weekly['笑脸信号_做多'] == 1:
                    signals.append('周PINK上穿10')
                if latest_weekly['笑脸信号_做空'] == 1:
                    signals.append('周PINK下穿94')
                if blue_weeks >= 2:
                    signals.append(f'周BLUE>150({blue_weeks}周)')
                
                print(f"{symbol} 发现信号: {', '.join(signals)}")
                
        except Exception as e:
            print(f"{symbol} 处理出错: {e}")
            continue
    
    return pd.DataFrame(results)

def main():
    """主函数"""
    start_time = time.time()
    
    # 扫描股票
    results = scan_signals()
    
    if results is not None and not results.empty:
        # 显示结果
        print("\n发现信号的股票:")
        print("=" * 160)
        print(f"{'代码':<6} | {'价格':>8} | {'成交量':>12} | "
              f"{'日PINK':>8} | {'日BLUE':>8} | {'日最大BLUE':>8} | {'日BLUE天数':>4} | "
              f"{'周PINK':>8} | {'周BLUE':>8} | {'周最大BLUE':>8} | {'周BLUE周数':>4} | {'信号':<40}")
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
            print(f"{row['symbol']:<6} | {row['price']:8.2f} | {row['Volume']:12.0f} | "
                  f"{row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | {row['max_blue_daily']:8.2f} | "
                  f"{row['blue_days']:4d} | "
                  f"{row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | {row['max_blue_weekly']:8.2f} | "
                  f"{row['blue_weeks']:4d} | {signals_str:<40}")
        
        print("=" * 160)
        print(f"共发现 {len(results)} 只股票有信号")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()