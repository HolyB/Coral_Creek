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
    print_lock = threading.Lock()
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

def process_single_stock(stock):
    """处理单个股票"""
    symbol = stock['code']
    name = stock['name']
    
    try:
        # 1. 获取日线数据
        fetcher_daily = StockDataFetcher(symbol, source='akshare', interval='1d')
        data_daily = fetcher_daily.get_stock_data()
        
        if data_daily is None or data_daily.empty:
            with print_lock:
                print(f"{symbol} 无法获取历史数据")
            return None
            
        # 重采样为周线数据
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        # 2. 计算指标
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
                'name': name,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'],
                'pink_daily': latest_daily['PINK'],
                'blue_daily': latest_daily['BLUE'],
                'blue_days': len(recent_daily[recent_daily['BLUE'] > 150]),
                'pink_weekly': latest_weekly['PINK'],
                'blue_weekly': latest_weekly['BLUE'],
                'blue_weeks': len(recent_weekly[recent_weekly['BLUE'] > 150]),
                'smile_long_daily': latest_daily['笑脸信号_做多'],
                'smile_short_daily': latest_daily['笑脸信号_做空'],
                'smile_long_weekly': latest_weekly['笑脸信号_做多'],
                'smile_short_weekly': latest_weekly['笑脸信号_做空'],
                'vol_times': latest_daily['VOL_TIMES'],
                'vol_color': latest_daily['HVOL_COLOR'],
                'gold_vol_count': len(recent_5d[recent_5d['GOLD_VOL']]),
                'double_vol_count': len(recent_5d[recent_5d['DOUBLE_VOL']]),
                'DIF': latest_daily['DIF'],
                'DEA': latest_daily['DEA'],
                'MACD': latest_daily['MACD'],
                'EMAMACD': latest_daily['EMAMACD'],
                '零轴下金叉': latest_daily.get('零轴下金叉', 0),
                '零轴上金叉': latest_daily.get('零轴上金叉', 0),
                '零轴上死叉': latest_daily.get('零轴上死叉', 0),
                '零轴下死叉': latest_daily.get('零轴下死叉', 0),
                '先机信号': latest_daily.get('先机信号', 0),
                '底背离': latest_daily.get('底背离', 0),
                '顶背离': latest_daily.get('顶背离', 0)
            }
            return result
            
    except Exception as e:
        with print_lock:
            print(f"{symbol} {name} 处理失败: {e}")
        return None

def main():
    """主函数"""
    start_time = time.time()
    
    results = scan_cn_signals()[:200]
    
    if not results.empty:
        # 保存结果
        results.to_csv('cn_signals.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存到 cn_signals.csv")
        
        # 显示结果
        print("\n发现信号的股票:")
        print("=" * 220)
        print(f"{'代码':<8} | {'名称':<8} | {'价格':>8} | {'成交量':>12} | {'成交额':>12} | "
              f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | "
              f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | "
              f"{'成交倍数':>8} | {'热力值':>4} | {'黄金柱':>4} | {'倍量柱':>4} | "
              f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'EMAMACD':>8} | {'信号':<40}")
        print("-" * 220)
        
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
            
            # MACD相关信号
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
            print(f"{row['symbol']:<8} | {row['name']:<8} | {row['price']:8.2f} | {row['Volume']:12.0f} | "
                  f"{row['turnover']:12.0f} | {row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | "
                  f"{row['blue_days']:4d} | {row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | "
                  f"{row['blue_weeks']:4d} | {row['vol_times']:8.1f} | {row['vol_color']:4.0f} | "
                  f"{row['gold_vol_count']:4d} | {row['double_vol_count']:4d} | "
                  f"{row.get('DIF', 0):8.2f} | {row.get('DEA', 0):8.2f} | {row.get('MACD', 0):8.2f} | "
                  f"{row.get('EMAMACD', 0):8.2f} | {signals_str:<40}")
        
        print("=" * 220)
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
    main() 