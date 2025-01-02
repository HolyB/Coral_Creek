# import pandas as pd
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')
# from datetime import datetime, timedelta
# import time
# import concurrent.futures
# import threading
# from tqdm import tqdm
# import akshare as ak

# from Stock_utils.stock_analysis import StockAnalysis
# from Stock_utils.stock_data_fetcher import StockDataFetcher

# # 创建一个线程锁用于打印
# print_lock = threading.Lock()

# def process_single_stock(stock):
#     """处理单个股票"""
#     symbol = stock['code']
#     name = stock['name']
    
#     try:
#         # 1. 获取日线数据
#         fetcher_daily = StockDataFetcher(symbol, source='akshare', interval='1d')
#         data_daily = fetcher_daily.get_stock_data()
        
#         if data_daily is None or data_daily.empty:
#             return None
            
#         # 重采样为周线数据
#         data_weekly = data_daily.resample('W-MON').agg({
#             'Open': 'first',
#             'High': 'max',
#             'Low': 'min',
#             'Close': 'last',
#             'Volume': 'mean'
#         }).dropna()
        
#         # 2. 计算指标
#         analysis_weekly = StockAnalysis(data_weekly)
#         df_weekly = analysis_weekly.calculate_phantom_indicators()
        
#         analysis_daily = StockAnalysis(data_daily)
#         df_daily = analysis_daily.calculate_phantom_indicators()
#         df_daily = analysis_daily.calculate_heatmap_volume()
#         df_daily = analysis_daily.calculate_macd_signals()
        
#         recent_weekly = df_weekly.tail(10)
#         latest_weekly = df_weekly.iloc[-1]
#         recent_daily = df_daily.tail(10)
#         latest_daily = df_daily.iloc[-1]
#         recent_5d = df_daily.tail(5)
        
#         # 检查信号条件
#         has_daily_signal = (
#             latest_daily['笑脸信号_做多'] == 1 or 
#             latest_daily['笑脸信号_做空'] == 1 or 
#             len(recent_daily[recent_daily['BLUE'] > 150]) >= 3
#         )
        
#         has_weekly_signal = (
#             latest_weekly['笑脸信号_做多'] == 1 or 
#             latest_weekly['笑脸信号_做空'] == 1 or 
#             len(recent_weekly[recent_weekly['BLUE'] > 150]) >= 2
#         )
        
#         has_volume_signal = (
#             latest_daily['GOLD_VOL'] or 
#             latest_daily['DOUBLE_VOL']
#         )
        
#         # 修改条件：必须有周线信号
#         if has_weekly_signal and (has_daily_signal or has_volume_signal):
#             result = {
#                 'symbol': symbol,
#                 'name': name,
#                 'price': latest_daily['Close'],
#                 'Volume': latest_daily['Volume'],
#                 'turnover': latest_daily['Volume'] * latest_daily['Close'],
#                 'pink_daily': latest_daily['PINK'],
#                 'blue_daily': latest_daily['BLUE'],
#                 'max_blue_daily': recent_daily['BLUE'].max(),
#                 'blue_days': len(recent_daily[recent_daily['BLUE'] > 150]),
#                 'pink_weekly': latest_weekly['PINK'],
#                 'blue_weekly': latest_weekly['BLUE'],
#                 'max_blue_weekly': recent_weekly['BLUE'].max(),
#                 'blue_weeks': len(recent_weekly[recent_weekly['BLUE'] > 150]),
#                 'smile_long_daily': latest_daily['笑脸信号_做多'],
#                 'smile_short_daily': latest_daily['笑脸信号_做空'],
#                 'smile_long_weekly': latest_weekly['笑脸信号_做多'],
#                 'smile_short_weekly': latest_weekly['笑脸信号_做空'],
#                 'vol_times': latest_daily['VOL_TIMES'],
#                 'vol_color': latest_daily['HVOL_COLOR'],
#                 'gold_vol_count': len(recent_5d[recent_5d['GOLD_VOL']]),
#                 'double_vol_count': len(recent_5d[recent_5d['DOUBLE_VOL']]),
#                 'DIF': latest_daily['DIF'],
#                 'DEA': latest_daily['DEA'],
#                 'MACD': latest_daily['MACD'],
#                 'EMAMACD': latest_daily['EMAMACD'],
#                 'V1': latest_daily['V1'],
#                 'V2': latest_daily['V2'],
#                 'V3': latest_daily['V3'],
#                 'V4': latest_daily['V4'],
#                 '补血': 1 if latest_daily['MACD'] > df_daily['MACD'].shift(1).iloc[-1] else 0,
#                 '失血': 1 if latest_daily['MACD'] < df_daily['MACD'].shift(1).iloc[-1] else 0,
#                 '零轴下金叉': latest_daily['零轴下金叉'] if '零轴下金叉' in latest_daily else 0,
#                 '零轴上金叉': latest_daily['零轴上金叉'] if '零轴上金叉' in latest_daily else 0,
#                 '零轴上死叉': latest_daily['零轴上死叉'] if '零轴上死叉' in latest_daily else 0,
#                 '零轴下死叉': latest_daily['零轴下死叉'] if '零轴下死叉' in latest_daily else 0,
#                 '先机信号': latest_daily['先机信号'] if '先机信号' in latest_daily else 0,
#                 '底背离': latest_daily['底背离'] if '底背离' in latest_daily else 0,
#                 '顶背离': latest_daily['顶背离'] if '顶背离' in latest_daily else 0
#             }
#             return result
            
#     except Exception as e:
#         with print_lock:
#             print(f"{symbol} {name} 处理出错: {e}")
#     return None

# def get_cn_tickers():
#     """获取A股股票列表"""
#     try:
#         # 使用 akshare 获取A股列表
#         stock_list = ak.stock_zh_a_spot_em()
#         # 添加市场标识
#         stock_list['code'] = stock_list['代码'].apply(
#             lambda x: f"sh{x}" if x.startswith('6') else f"sz{x}"
#         )
#         # 重命名列
#         stock_list = stock_list.rename(columns={'名称': 'name'})
#         return stock_list[['code', 'name']]
#     except Exception as e:
#         print(f"获取股票列表失败: {e}")
#         return pd.DataFrame()

# def scan_signals_parallel(max_workers=20):
#     """并行扫描A股信号"""
#     print("正在获取A股列表...")
#     stock_list = get_cn_tickers()
    
#     if stock_list.empty:
#         print("获取股票列表失败")
#         return pd.DataFrame()
    
#     print(f"\n开始并行扫描 {len(stock_list)} 只A股...")
#     results = []
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         with tqdm(total=len(stock_list), desc="扫描进度") as pbar:
#             future_to_stock = {executor.submit(process_single_stock, stock): stock 
#                              for _, stock in stock_list.iterrows()}
            
#             for future in concurrent.futures.as_completed(future_to_stock):
#                 stock = future_to_stock[future]
#                 try:
#                     result = future.result()
#                     if result:
#                         results.append(result)
#                 except Exception as e:
#                     print(f"{stock['code']} {stock['name']} 处理失败: {e}")
#                 finally:
#                     pbar.update(1)
    
#     return pd.DataFrame(results)

# def main():
#     """主函数"""
#     start_time = time.time()
    
#     results = scan_signals_parallel(max_workers=20)
    
#     if not results.empty:
#         # 保存结果
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         filename_v2 = f'signals_v2_{timestamp}.csv'
        
#         # 创建v2格式的DataFrame
#         df_v2 = pd.DataFrame({
#             'symbol': results['symbol'],
#             'name': results['name'],
#             'signals': results.apply(lambda row: ', '.join([
#                 '日PINK上穿10' if row['smile_long_daily'] == 1 else '',
#                 '日PINK下穿94' if row['smile_short_daily'] == 1 else '',
#                 f'日BLUE>150({row["blue_days"]}天)' if row['blue_days'] >= 3 else '',
#                 '周PINK上穿10' if row['smile_long_weekly'] == 1 else '',
#                 '周PINK下穿94' if row['smile_short_weekly'] == 1 else '',
#                 f'周BLUE>150({row["blue_weeks"]}周)' if row['blue_weeks'] >= 2 else '',
#                 f'黄金柱({row["gold_vol_count"]}次)' if row["gold_vol_count"] > 0 else '',
#                 f'倍量柱({row["double_vol_count"]}次)' if row["double_vol_count"] > 0 else ''
#             ]), axis=1),
#             'DIF': results['DIF'],
#             'DEA': results['DEA'],
#             'MACD': results['MACD'],
#             'EMAMACD': results['EMAMACD'],
#             'V1': results['V1'],
#             'V2': results['V2'],
#             'V3': results['V3'],
#             'V4': results['V4']
#         })
        
#         df_v2.to_csv(filename_v2, index=False, encoding='utf-8-sig')
#         print(f"\n结果已保存到 {filename_v2}")
        
#         # 显示结果
#         print("\n发现信号的股票:")
#         print("=" * 220)
#         print(f"{'代码':<8} | {'名称':<8} | {'价格':>8} | {'成交量':>12} | {'成交额':>12} | "
#               f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | "
#               f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | "
#               f"{'成交倍数':>8} | {'热力值':>4} | {'黄金柱':>4} | {'倍量柱':>4} | "
#               f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'EMAMACD':>8} | {'信号':<40}")
#         print("-" * 220)
        
#         for _, row in results.iterrows():
#             signals = []
#             if row['smile_long_daily'] == 1:
#                 signals.append('日PINK上穿10')
#             if row['smile_short_daily'] == 1:
#                 signals.append('日PINK下穿94')
#             if row['blue_days'] >= 3:
#                 signals.append(f'日BLUE>150({row["blue_days"]}天)')
#             if row['smile_long_weekly'] == 1:
#                 signals.append('周PINK上穿10')
#             if row['smile_short_weekly'] == 1:
#                 signals.append('周PINK下穿94')
#             if row['blue_weeks'] >= 2:
#                 signals.append(f'周BLUE>150({row["blue_weeks"]}周)')
#             if row['gold_vol_count'] > 0:
#                 signals.append(f'黄金柱({row["gold_vol_count"]}次)')
#             if row['double_vol_count'] > 0:
#                 signals.append(f'倍量柱({row["double_vol_count"]}次)')
            
#             # 添加MACD相关信号
#             if row.get('零轴下金叉', 0) == 1:
#                 signals.append('零轴下金叉')
#             if row.get('零轴上金叉', 0) == 1:
#                 signals.append('零轴上金叉')
#             if row.get('零轴上死叉', 0) == 1:
#                 signals.append('零轴上死叉')
#             if row.get('零轴下死叉', 0) == 1:
#                 signals.append('零轴下死叉')
#             if row.get('先机信号', 0) == 1:
#                 signals.append('先机信号')
#             if row.get('底背离', 0) == 1:
#                 signals.append('底背离')
#             if row.get('顶背离', 0) == 1:
#                 signals.append('顶背离')
            
#             signals_str = ', '.join(signals)
#             print(f"{row['symbol']:<8} | {row['name']:<8} | {row['price']:8.2f} | {row['Volume']:12.0f} | "
#                   f"{row['turnover']:12.0f} | {row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | "
#                   f"{row['blue_days']:4d} | {row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | "
#                   f"{row['blue_weeks']:4d} | {row['vol_times']:8.1f} | {row['vol_color']:4.0f} | "
#                   f"{row['gold_vol_count']:4d} | {row['double_vol_count']:4d} | "
#                   f"{row.get('DIF', 0):8.2f} | {row.get('DEA', 0):8.2f} | {row.get('MACD', 0):8.2f} | "
#                   f"{row.get('EMAMACD', 0):8.2f} | {signals_str:<40}")
        
#         print("=" * 220)
#         print(f"共发现 {len(results)} 只股票有信号")
#     else:
#         print("\n未发现任何信号")
    
#     end_time = time.time()
#     print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

# if __name__ == "__main__":
#     main() 

import akshare as ak

stock_codes = ["sz000886", "sh600718", "sz002868", "sh600575"]  # 替换为你的实际股票代码
for code in stock_codes:
    try:
        print(f"正在获取股票数据: {code}")
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20220101", end_date="20230101", adjust="")
        print(df.head())
    except KeyError as e:
        print(f"从 AKShare 获取数据失败 ({code}): {e}")
    except Exception as e:
        print(f"其他错误 ({code}): {e}")
