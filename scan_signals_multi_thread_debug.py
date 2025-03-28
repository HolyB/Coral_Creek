import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time
import threading
import concurrent.futures
from tqdm import tqdm

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

# 定义富途函数
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

print_lock = threading.Lock()
results_lock = threading.Lock()
COMPANY_INFO = {}

def init_company_info():
    global COMPANY_INFO
    print("\n正在初始化公司信息...")
    COMPANY_INFO = {
        'PLTR': 'Palantir Technologies Inc.',
        'TSLA': 'Tesla, Inc.'
    }
    print(f"初始化完成，共加载 {len(COMPANY_INFO)} 家公司信息")

def process_single_stock(symbol):
    try:
        print(f"\n=== 开始处理 {symbol} ===")
        fetcher_daily = StockDataFetcher(symbol, source='polygon', interval='1d')
        print(f"{symbol} 创建 StockDataFetcher 成功")
        data_daily = fetcher_daily.get_stock_data()
        print(f"{symbol} 获取数据完成，data_daily 类型: {type(data_daily)}")
        
        if data_daily is None or data_daily.empty:
            print(f"{symbol} 数据为空或无效: data_daily = {data_daily}")
            return None
        
        print(f"{symbol} 日线数据行数: {len(data_daily)}")
        print(f"{symbol} 日线数据（最近5天）:\n{data_daily.tail(5)}")
        
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        print(f"{symbol} 周线数据行数: {len(data_weekly)}")
        
        # 提取数据
        OPEN_D, HIGH_D, LOW_D, CLOSE_D = data_daily['Open'].values, data_daily['High'].values, data_daily['Low'].values, data_daily['Close'].values
        OPEN_W, HIGH_W, LOW_W, CLOSE_W = data_weekly['Open'].values, data_weekly['High'].values, data_weekly['Low'].values, data_weekly['Close'].values
        
        # 日线海底捞月部分
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
        
        # 日线KDJ和PINK
        RSV1_D = (CLOSE_D - LLV(LOW_D, 39)) / (HHV(HIGH_D, 39) - LLV(LOW_D, 39)) * 100
        K_D = SMA(RSV1_D, 2, 1)
        D_D = SMA(K_D, 2, 1)
        J_D = 3 * K_D - 2 * D_D
        PINK_D = SMA(J_D, 2, 1)
        SMILE_LONG_D = IF(PINK_D > 10, 1, 0)
        SMILE_SHORT_D = IF(PINK_D < 94, 1, 0)
        
        # 周线海底捞月部分
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
        
        # 周线KDJ和PINK
        RSV1_W = (CLOSE_W - LLV(LOW_W, 39)) / (HHV(HIGH_W, 39) - LLV(LOW_W, 39)) * 100
        K_W = SMA(RSV1_W, 2, 1)
        D_W = SMA(K_W, 2, 1)
        J_W = 3 * K_W - 2 * D_W
        PINK_W = SMA(J_W, 2, 1)
        SMILE_LONG_W = IF(PINK_W > 10, 1, 0)
        SMILE_SHORT_W = IF(PINK_W < 94, 1, 0)
        
        # 初始化DataFrame
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'VAR5': VAR5_D, 'VAR51': VAR51_D, 'BLUE': BLUE_D, 'LIRED': LIRED_D,
            'PINK': PINK_D, '笑脸信号_做多': SMILE_LONG_D, '笑脸信号_做空': SMILE_SHORT_D
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'VAR5': VAR5_W, 'VAR51': VAR51_W, 'BLUE': BLUE_W, 'LIRED': LIRED_W,
            'PINK': PINK_W, '笑脸信号_做多': SMILE_LONG_W, '笑脸信号_做空': SMILE_SHORT_W
        }, index=data_weekly.index)
        
        # 计算其他指标并合并
        analysis_daily = StockAnalysis(data_daily)
        heatmap_df = analysis_daily.calculate_heatmap_volume()
        macd_df = analysis_daily.calculate_macd_signals()
        
        heatmap_df = heatmap_df if heatmap_df is not None else pd.DataFrame(index=data_daily.index)
        macd_df = macd_df if macd_df is not None else pd.DataFrame(index=data_daily.index)
        
        df_daily = pd.concat([df_daily, 
                              heatmap_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore'),
                              macd_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')], 
                             axis=1)
        
        recent_weekly = df_weekly.tail(10)
        recent_daily_20 = df_daily.tail(20)
        recent_daily = df_daily.tail(10)
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        recent_5d = df_daily.tail(5)
        
        # 调试输出
        print(f"\n=== {symbol} 中间变量检查 ===")
        print(f"日线RADIO1: {RADIO1_D}")
        print(f"周线RADIO1: {RADIO1_W}")
        print(f"日线VAR5（最近20天）:\n{df_daily['VAR5'].tail(20).tolist()}")
        print(f"日线VAR51（最近20天）:\n{df_daily['VAR51'].tail(20).tolist()}")
        print(f"周线VAR5（最近10周）:\n{df_weekly['VAR5'].tail(10).tolist()}")
        print(f"周线VAR51（最近10周）:\n{df_weekly['VAR51'].tail(10).tolist()}")
        
        print(f"\n=== 调试 {symbol} 的所有数值 ===")
        print(f"日线数据（最新一天）:\n{latest_daily.to_dict()}")
        print(f"周线数据（最新一周）:\n{latest_weekly.to_dict()}")
        print(f"最近20天日线BLUE:\n{recent_daily_20['BLUE'].tolist()}")
        print(f"最近10周周线BLUE:\n{recent_weekly['BLUE'].tolist()}")
        print(f"最近20天日线LIRED:\n{recent_daily_20['LIRED'].tolist()}")
        print(f"最近10周周线LIRED:\n{recent_weekly['LIRED'].tolist()}")
        print(f"日LIRED天数 (< -50): {sum(1 for x in recent_daily['LIRED'] if x < -50)}")
        print(f"周LIRED周数 (< -50): {sum(1 for x in recent_weekly['LIRED'] if x < -50)}")
        
        # 检查LIRED信号
        has_lired_signal = False
        day_lired_count = sum(1 for x in recent_daily['LIRED'] if x < -50)
        week_lired_count = sum(1 for x in recent_weekly['LIRED'] if x < -50)
        has_lired_signal = day_lired_count >= 2 and week_lired_count >= 2
        
        has_daily_signal = (
            latest_daily['笑脸信号_做多'] or 
            latest_daily['笑脸信号_做空'] or 
            len(recent_daily[recent_daily['BLUE'] > 50]) >= 3
        )
        has_weekly_signal = (
            latest_weekly['笑脸信号_做多'] or 
            latest_weekly['笑脸信号_做空'] or 
            len(recent_weekly[recent_weekly['BLUE'] > 50]) >= 2
        )
        has_volume_signal = (
            latest_daily['GOLD_VOL'] if 'GOLD_VOL' in latest_daily else False or 
            latest_daily['DOUBLE_VOL'] if 'DOUBLE_VOL' in latest_daily else False
        )
        
        print(f"有LIRED信号: {has_lired_signal}")
        print(f"有日线信号: {has_daily_signal}")
        print(f"有周线信号: {has_weekly_signal}")
        print(f"有成交量信号: {has_volume_signal}")
        print(f"最近5天GOLD_VOL: {recent_5d['GOLD_VOL'].tolist() if 'GOLD_VOL' in recent_5d else 'N/A'}")
        print(f"最近5天DOUBLE_VOL: {recent_5d['DOUBLE_VOL'].tolist() if 'DOUBLE_VOL' in recent_5d else 'N/A'}")
        
        if (has_weekly_signal and (has_daily_signal or has_volume_signal)) or has_lired_signal:
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'],
                'pink_daily': latest_daily['PINK'],
                'blue_daily': latest_daily['BLUE'],
                'max_blue_daily': recent_daily['BLUE'].max(),
                'blue_days': len(recent_daily[recent_daily['BLUE'] > 50]),
                'pink_weekly': latest_weekly['PINK'],
                'blue_weekly': latest_weekly['BLUE'],
                'max_blue_weekly': recent_weekly['BLUE'].max(),
                'blue_weeks': len(recent_weekly[recent_weekly['BLUE'] > 50]),
                'smile_long_daily': latest_daily['笑脸信号_做多'],
                'smile_short_daily': latest_daily['笑脸信号_做空'],
                'smile_long_weekly': latest_weekly['笑脸信号_做多'],
                'smile_short_weekly': latest_weekly['笑脸信号_做空'],
                'vol_times': latest_daily['VOL_TIMES'] if 'VOL_TIMES' in latest_daily else 0,
                'vol_color': latest_daily['HVOL_COLOR'] if 'HVOL_COLOR' in latest_daily else 0,
                'gold_vol_count': len(recent_5d[recent_5d['GOLD_VOL']]) if 'GOLD_VOL' in recent_5d else 0,
                'double_vol_count': len(recent_5d[recent_5d['DOUBLE_VOL']]) if 'DOUBLE_VOL' in recent_5d else 0,
                'DIF': latest_daily['DIF'] if 'DIF' in latest_daily else 0,
                'DEA': latest_daily['DEA'] if 'DEA' in latest_daily else 0,
                'MACD': latest_daily['MACD'] if 'MACD' in latest_daily else 0,
                'EMAMACD': latest_daily['EMAMACD'] if 'EMAMACD' in latest_daily else 0,
                '补血': 1 if 'MACD' in latest_daily and latest_daily['MACD'] > df_daily['MACD'].shift(1).iloc[-1] else 0,
                '失血': 1 if 'MACD' in latest_daily and latest_daily['MACD'] < df_daily['MACD'].shift(1).iloc[-1] else 0,
                '零轴下金叉': latest_daily['零轴下金叉'] if '零轴下金叉' in latest_daily else 0,
                '零轴上金叉': latest_daily['零轴上金叉'] if '零轴上金叉' in latest_daily else 0,
                '零轴上死叉': latest_daily['零轴上死叉'] if '零轴上死叉' in latest_daily else 0,
                '零轴下死叉': latest_daily['零轴下死叉'] if '零轴下死叉' in latest_daily else 0,
                '先机信号': latest_daily['先机信号'] if '先机信号' in latest_daily else 0,
                '底背离': latest_daily['底背离'] if '底背离' in latest_daily else 0,
                '顶背离': latest_daily['顶背离'] if '顶背离' in latest_daily else 0,
                'has_lired_signal': has_lired_signal,
                'lired_daily': latest_daily['LIRED'] if 'LIRED' in df_daily.columns else 0,
                'lired_days': day_lired_count,
                'lired_weekly': latest_weekly['LIRED'] if 'LIRED' in df_weekly.columns else 0,
                'lired_weeks': week_lired_count
            }
            print(f"{symbol} 满足信号条件，返回结果")
            with results_lock:
                return result
        else:
            print(f"{symbol} 未满足信号条件")
        
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
            import traceback
            traceback.print_exc()
    return None

def scan_signals_parallel(max_workers=1, batch_size=1, cooldown=0, tickers=None):
    if tickers is None:
        tickers = ['PLTR', 'TSLA']  # 添加TSLA
    print(f"Processing {len(tickers)} stocks")
    
    all_results = []
    batch_count = (len(tickers) + batch_size - 1) // batch_size
    
    with tqdm(total=batch_count, desc="Batch Progress") as batch_pbar:
        for i in range(0, len(tickers), batch_size):
            batch_start_time = time.time()
            batch_tickers = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"\nProcessing batch {batch_num}/{batch_count} ({len(batch_tickers)} stocks)")
            batch_results = []
            completed_count = 0
            
            def process_result(future):
                nonlocal completed_count
                result = future.result()
                if result is not None:
                    with results_lock:
                        batch_results.append(result)
                with results_lock:
                    completed_count += 1
                    stock_pbar.update(1)
            
            with tqdm(total=len(batch_tickers), desc="Stock Progress") as stock_pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for symbol in batch_tickers:
                        future = executor.submit(process_single_stock, symbol)
                        future.add_done_callback(process_result)
                    while completed_count < len(batch_tickers):
                        time.sleep(0.1)
            
            if batch_results:
                all_results.extend(batch_results)
                print(f"Batch {batch_num} found {len(batch_results)} stocks with signals")
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            print(f"Batch {batch_num} processing time: {batch_time:.2f} seconds")
            batch_pbar.update(1)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def main():
    init_company_info()
    start_time = time.time()
    print("\n开始扫描PLTR和TSLA...")
    
    tickers = ['PLTR', 'TSLA']
    results = scan_signals_parallel(tickers=tickers)
    
    if not results.empty:
        results['company_name'] = results['symbol'].map(lambda x: COMPANY_INFO.get(x, 'N/A'))
        print("\n信号结果:")
        print("=" * 280)
        print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交量':>12} | {'成交额':>12} | "
              f"{'日PINK':>8} | {'日BLUE':>8} | {'日BLUE天数':>4} | {'日LIRED':>8} | {'日LIRED数':>4} | "
              f"{'周PINK':>8} | {'周BLUE':>8} | {'周BLUE周数':>4} | {'周LIRED':>8} | {'周LIRED数':>4} | "
              f"{'成交倍数':>8} | {'热力值':>4} | {'黄金柱':>4} | {'倍量柱':>4} | "
              f"{'DIF':>8} | {'DEA':>8} | {'MACD':>8} | {'EMAMACD':>8} | {'信号':<40}")
        print("-" * 280)
        
        for _, row in results.iterrows():
            signals = []
            if row['smile_long_daily'] == 1:
                signals.append('日PINK上穿10')
            if row['smile_short_daily'] == 1:
                signals.append('日PINK下穿94')
            if row['blue_days'] >= 3:
                signals.append(f'日BLUE>50({row["blue_days"]}天)')
            if row['smile_long_weekly'] == 1:
                signals.append('周PINK上穿10')
            if row['smile_short_weekly'] == 1:
                signals.append('周PINK下穿94')
            if row['blue_weeks'] >= 2:
                signals.append(f'周BLUE>50({row["blue_weeks"]}周)')
            if row['gold_vol_count'] > 0:
                signals.append(f'黄金柱({row["gold_vol_count"]}次)')
            if row['double_vol_count'] > 0:
                signals.append(f'倍量柱({row["double_vol_count"]}次)')
            if row.get('has_lired_signal', False):
                signals.append('日周LIRED信号')
            
            signals_str = ', '.join(signals)
            
            print(f"{row['symbol']:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['Volume']:12.0f} | {row['turnover']:12.0f} | "
                  f"{row['pink_daily']:8.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                  f"{row['lired_daily']:8.2f} | {row['lired_days']:4d} | "
                  f"{row['pink_weekly']:8.2f} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                  f"{row['lired_weekly']:8.2f} | {row['lired_weeks']:4d} | "
                  f"{row['vol_times']:8.1f} | {row['vol_color']:4.0f} | "
                  f"{row['gold_vol_count']:4d} | {row['double_vol_count']:4d} | "
                  f"{row.get('DIF', 0):8.2f} | {row.get('DEA', 0):8.2f} | {row.get('MACD', 0):8.2f} | "
                  f"{row.get('EMAMACD', 0):8.2f} | {signals_str:<40}")
        print("=" * 280)
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()