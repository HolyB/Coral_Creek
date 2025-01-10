import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import akshare as ak
import threading
import concurrent.futures
from tqdm import tqdm

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.MyTT import *

# 创建线程锁
print_lock = threading.Lock()

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

def process_single_stock(stock):
    """处理单个股票"""
    symbol = stock['code']
    name = stock['name']
    
    try:
        # 1. 获取日线数据
        fetcher = StockDataFetcher(symbol, source='akshare', interval='1d')
        df = fetcher.get_stock_data()
        
        if df is None or len(df) < 30:  # 确保有足够的数据
            return None
            
        CLOSE = df['Close'].values
        HIGH = df['High'].values
        LOW = df['Low'].values
        
        # 1. 计算涨停条件（近5天内有涨停）
        ZT = CLOSE > REF(CLOSE, 1) * 1.093  # 涨停判断
        recent_zt = ZT[-5:]  # 最近5天的涨停情况
        has_recent_zt = np.any(recent_zt)  # 是否有涨停
        
        # 2. 计算均线
        MA5 = MA(CLOSE, 5)
        MA10 = MA(CLOSE, 10)
        
        # 判断5日均线是否向上
        ma5_trend = MA5[-1] > MA5[-2]
        
        # 3. 计算MACD
        DIF, DEA, MACD = MACD(CLOSE)
        
        # 判断MACD是否金叉且在0轴上方
        macd_golden_cross = CROSS(DIF, DEA)[-1]  # 最新一天是否金叉
        above_zero = DIF[-1] > 0  # DIF在0轴上方
        
        # 计算DIF的角度（使用最近几天的数据计算趋势角度）
        dif_angle = np.arctan2(DIF[-1] - DIF[-5], 5) * 180 / np.pi
        
        # 4. 回踩判断
        price = CLOSE[-1]  # 最新价
        ma5_value = MA5[-1]  # 5日均线
        ma10_value = MA10[-1]  # 10日均线
        
        # 判断是否回踩到均线附近（允许3%的误差）
        near_ma5 = abs(price - ma5_value) / ma5_value < 0.03
        near_ma10 = abs(price - ma10_value) / ma10_value < 0.03
        
        # 返回所有条件的状态
        return {
            'symbol': symbol,
            'name': name,
            'price': price,
            'ma5': ma5_value,
            'ma10': ma10_value,
            'dif': DIF[-1],
            'dea': DEA[-1],
            'macd': MACD[-1],
            'dif_angle': dif_angle,
            'has_recent_zt': has_recent_zt,
            'ma5_trend': ma5_trend,
            'macd_golden_cross': macd_golden_cross,
            'above_zero': above_zero,
            'near_ma5': near_ma5,
            'near_ma10': near_ma10,
            'last_zt_days': np.where(recent_zt)[0][-1] + 1 if np.any(recent_zt) else None
        }
            
    except Exception as e:
        with print_lock:
            print(f"{symbol} {name} 处理失败: {e}")
    return None

def scan_cn_signals():
    """扫描A股信号"""
    print("正在获取A股列表...")
    stock_list = get_cn_tickers()
    
    if stock_list.empty:
        print("获取股票列表失败")
        return pd.DataFrame()
    
    print(f"\n开始并行扫描 {len(stock_list)} 只A股...")
    results = []
    
    # 创建线程锁用于打印和结果添加
    results_lock = threading.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
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
                        with results_lock:
                            results.append(result)
                except Exception as e:
                    with print_lock:
                        print(f"{stock['code']} {stock['name']} 处理失败: {e}")
                finally:
                    pbar.update(1)
    
    return pd.DataFrame(results)

def main():
    """主函数"""
    start_time = time.time()
    
    results = scan_cn_signals()
    
    if not results.empty:
        # 保存结果
        results.to_csv('jingji_dixin_debug.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存到 jingji_dixin_debug.csv")
        
        # 分析各个条件的股票数量
        print("\n条件统计:")
        print(f"总股票数: {len(results)}")
        print(f"近期有涨停: {results['has_recent_zt'].sum()}")
        print(f"5日均线向上: {results['ma5_trend'].sum()}")
        print(f"MACD金叉: {results['macd_golden_cross'].sum()}")
        print(f"DIF在0轴上方: {results['above_zero'].sum()}")
        print(f"回踩5日线: {results['near_ma5'].sum()}")
        print(f"回踩10日线: {results['near_ma10'].sum()}")
        print(f"DIF角度>30度: {(results['dif_angle'] > 30).sum()}")
        
        # 显示涨停股票的详细信息
        zt_stocks = results[results['has_recent_zt']]
        print("\n近期涨停股票详细信息:")
        print("=" * 140)
        print(f"{'代码':<8} | {'名称':<8} | {'价格':>8} | {'5日线':>8} | {'10日线':>8} | "
              f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'DIF角度':>8} | {'距涨停天数':>4} | "
              f"{'5日线向上':>6} | {'MACD金叉':>6} | {'零轴上方':>6} | {'回踩5日':>6} | {'回踩10日':>6}")
        print("-" * 140)
        
        for _, row in zt_stocks.iterrows():
            print(f"{row['symbol']:<8} | {row['name']:<8} | {row['price']:8.2f} | "
                  f"{row['ma5']:8.2f} | {row['ma10']:8.2f} | {row['dif']:8.2f} | "
                  f"{row['dea']:8.2f} | {row['macd']:8.2f} | {row['dif_angle']:8.2f} | "
                  f"{row['last_zt_days']:4d} | {'是' if row['ma5_trend'] else '否':>6} | "
                  f"{'是' if row['macd_golden_cross'] else '否':>6} | "
                  f"{'是' if row['above_zero'] else '否':>6} | "
                  f"{'是' if row['near_ma5'] else '否':>6} | "
                  f"{'是' if row['near_ma10'] else '否':>6}")
        
        print("=" * 140)
    else:
        print("\n未获取到股票数据")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 