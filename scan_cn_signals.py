import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import akshare as ak

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

def get_realtime_price(symbol):
    """获取A股实时价格"""
    try:
        # 移除市场前缀
        code = symbol[2:]
        # 获取实时行情
        df = ak.stock_zh_a_spot_em()
        # 查找对应股票
        stock_info = df[df['代码'] == code].iloc[0]
        return {
            'current_price': stock_info['最新价'],
            'change_percent': stock_info['涨跌幅'],
            'volume': stock_info['成交量'],
            'turnover': stock_info['成交额']
        }
    except Exception as e:
        print(f"获取实时价格失败 ({symbol}): {e}")
        return None


def get_cn_tickers():
    """获取A股股票列表"""
    try:
        # 获取A股股票信息
        stock_info = ak.stock_info_a_code_name()
        
        # 添加市场前缀
        tickers = []
        for _, row in stock_info.iterrows():
            code = row['code']
            name = row['name']
            # 根据代码添加市场前缀
            if code.startswith('6'):
                tickers.append({'code': f'SH{code}', 'name': name})
            else:
                tickers.append({'code': f'SZ{code}', 'name': name})
        
        return pd.DataFrame(tickers)
        
    except Exception as e:
        print(f"获取A股列表失败: {e}")
        return pd.DataFrame()

def scan_signals():
    """扫描A股信号"""
    print("正在获取A股列表...")
    stock_list = get_cn_tickers()
    
    if stock_list.empty:
        print("获取股票列表失败")
        return
    
    print(f"\n开始扫描 {len(stock_list)} 只A股...")
    # 测试时只取前5个
    test_stocks = stock_list.head()
    print(f"测试股票:\n{test_stocks}")
    
    results = []
    count = 0
    
    for _, stock in test_stocks.iterrows():
        count += 1
        symbol = stock['code']
        name = stock['name']
        print(f"\n处理股票 {symbol} {name} ({count}/{len(test_stocks)})")
        
        try:
            # 获取实时价格
            realtime_data = get_realtime_price(symbol)
            if not realtime_data:
                print(f"{symbol} 无法获取实时价格")
                continue
                
            print(f"实时价格: {realtime_data['current_price']:.2f} 涨跌幅: {realtime_data['change_percent']:.2f}%")
            
            # 获取历史数据
            fetcher = StockDataFetcher(symbol, source='akshare', interval='1d')
            data = fetcher.get_stock_data()
            
            if data is None or data.empty:
                print(f"{symbol} 无法获取历史数据")
                continue
            
            print(f"{symbol} 获取到 {len(data)} 天的数据")
            
            # 计算指标
            analysis = StockAnalysis(data)
            df = analysis.calculate_phantom_force()
            
            # 获取最近数据
            recent = df.tail(10)
            latest = df.iloc[-1]
            
            # 打印调试信息
            print(f"{symbol} {name} 最新指标:")
            print(f"BLUE={latest['BLUE']:.2f}, PINK={latest['PINK']:.2f}")
            print(f"笑脸信号: 做多={latest['笑脸信号_做多']}, 做空={latest['笑脸信号_做空']}")
            
            # 检查信号
            if (latest['BLUE'] > 150 or 
                recent['BLUE'].max() > 150 or 
                latest['笑脸信号_做多'] == 1 or 
                latest['笑脸信号_做空'] == 1):
                
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'price': realtime_data['current_price'],
                    'change': realtime_data['change_percent'],
                    'Volume': realtime_data['volume'],
                    'turnover': realtime_data['turnover'],
                    'pink': latest['PINK'],
                    'blue': latest['BLUE'],
                    'max_blue': recent['BLUE'].max(),
                    'smile_long': latest['笑脸信号_做多'],
                    'smile_short': latest['笑脸信号_做空']
                })
                
        except Exception as e:
            print(f"{symbol} {name} 处理出错: {e}")
            continue
    
    return pd.DataFrame(results)

def main():
    """主函数"""
    start_time = time.time()
    
    # 扫描A股
    print("\n开始扫描A股市场...")
    results = scan_signals()
    
    if results is not None and not results.empty:
        # 显示结果
        print("\n发现信号的股票:")
        print("=" * 140)
        print(f"{'代码':<8} | {'名称':<8} | {'价格':>8} | {'涨跌幅':>8} | {'成交量':>12} | {'成交额':>12} | "
              f"{'PINK':>8} | {'BLUE':>8} | {'最大BLUE':>8} | {'信号':<20}")
        print("-" * 140)
        
        for _, row in results.iterrows():
            signals = []
            if row['smile_long'] == 1:
                signals.append('PINK上穿10')
            if row['smile_short'] == 1:
                signals.append('PINK下穿94')
            if row['max_blue'] > 150:
                signals.append(f'BLUE>{150}')
            
            signals_str = ', '.join(signals)
            print(f"{row['symbol']:<8} | {row['name']:<8} | {row['price']:8.2f} | {row['change']:8.2f}% | "
                  f"{row['Volume']:12.0f} | {row['turnover']:12.0f} | "
                  f"{row['pink']:8.2f} | {row['blue']:8.2f} | {row['max_blue']:8.2f} | {signals_str:<20}")
        
        print("=" * 140)
        print(f"共发现 {len(results)} 只股票有信号")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 