import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time
import akshare as ak
import threading
import concurrent.futures
from tqdm import tqdm
import os
import traceback
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建全局线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

def get_cn_tickers():
    """获取A股股票列表，包括北交所股票"""
    try:
        stock_info = ak.stock_info_a_code_name()
        tickers = []
        for _, row in stock_info.iterrows():
            code = row['code']
            name = row['name']
            if code.startswith('688') or code.startswith('6'):
                tickers.append({'code': f'SH{code}', 'name': name})
            elif code.startswith('3') or code.startswith('0'):
                tickers.append({'code': f'SZ{code}', 'name': name})
            elif code.startswith('8') or code.startswith('4'):
                tickers.append({'code': f'BJ{code}', 'name': name})
            else:
                tickers.append({'code': f'SZ{code}', 'name': name})
        
        try:
            bj_stock_info = ak.stock_info_bj_name_code()
            for _, row in bj_stock_info.iterrows():
                if '证券代码' in row and '证券简称' in row:
                    code = row['证券代码']
                    name = row['证券简称']
                    if not any(item['code'] == f'BJ{code}' for item in tickers):
                        tickers.append({'code': f'BJ{code}', 'name': name})
        except Exception as e:
            with print_lock:
                logging.warning(f"获取北交所股票信息失败，将尝试继续使用主列表: {e}")
        
        return pd.DataFrame(tickers)
    except Exception as e:
        with print_lock:
            logging.error(f"获取A股列表失败: {e}")
        return pd.DataFrame()

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

def process_single_stock(stock):
    """处理单个股票，仅关注BLUE和LIRED信号，阈值为150"""
    symbol = stock['code']
    name = stock['name']
    
    try:
        logging.info(f"开始处理股票: {symbol} {name}")
        code = symbol[2:]  # 去掉前缀SH/SZ/BJ
        data_daily = ak.stock_zh_a_hist(symbol=code, period="daily")
        logging.info(f"获取 {symbol} 日线数据完成，行数: {len(data_daily) if data_daily is not None else 0}")
        
        if data_daily is None or data_daily.empty:
            with print_lock:
                logging.warning(f"获取数据出错 ({symbol} {name}): 数据为空")
            return None
        
        data_daily = data_daily.rename(columns={
            '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
            '收盘': 'Close', '成交量': 'Volume'
        }).set_index('Date')
        data_daily.index = pd.to_datetime(data_daily.index)
        
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        logging.info(f"{symbol} 周线数据重采样完成，行数: {len(data_weekly)}")
        
        OPEN_D, HIGH_D, LOW_D, CLOSE_D = data_daily['Open'].values, data_daily['High'].values, data_daily['Low'].values, data_daily['Close'].values
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
        
        OPEN_W, HIGH_W, LOW_W, CLOSE_W = data_weekly['Open'].values, data_weekly['High'].values, data_weekly['Low'].values, data_weekly['Close'].values
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
            'BLUE': BLUE_D, 'LIRED': LIRED_D
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'BLUE': BLUE_W, 'LIRED': LIRED_W
        }, index=data_weekly.index)
        
        recent_daily = df_daily.tail(6)
        recent_weekly = df_weekly.tail(5)
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        
        day_blue_count = len(recent_daily[recent_daily['BLUE'] > 150])
        week_blue_count = len(recent_weekly[recent_weekly['BLUE'] > 150])
        day_lired_count = len(recent_daily[recent_daily['LIRED'] < -150])
        week_lired_count = len(recent_weekly[recent_weekly['LIRED'] < -150])
        
        has_blue_signal = day_blue_count >= 3 or week_blue_count >= 2
        has_lired_signal = day_lired_count >= 3 or week_lired_count >= 2
        
        logging.info(f"{symbol} 信号检查: BLUE日={day_blue_count}, 周={week_blue_count}; LIRED日={day_lired_count}, 周={week_lired_count}")
        
        if has_blue_signal or has_lired_signal:
            result = {
                'symbol': symbol,
                'name': name,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'] / 10000,  # 单位：万
                'blue_daily': latest_daily['BLUE'],
                'blue_days': day_blue_count,
                'lired_daily': latest_daily['LIRED'],
                'lired_days': day_lired_count,
                'blue_weekly': latest_weekly['BLUE'],
                'blue_weeks': week_blue_count,
                'lired_weekly': latest_weekly['LIRED'],
                'lired_weeks': week_lired_count
            }
            logging.info(f"{symbol} 发现信号")
            return result
        return None
    
    except Exception as e:
        with print_lock:
            logging.error(f"获取数据出错 ({symbol} {name}): {str(e)}")
            traceback.print_exc()
        return None

def _scan_batch(batch, max_workers=5, max_wait_time=600):
    """扫描一批股票"""
    results = []
    problem_stocks = []
    completed_count = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def process_result(future, stock):
        nonlocal completed_count
        try:
            result = future.result(timeout=10)
            if result:
                with results_lock:
                    results.append(result)
        except concurrent.futures.TimeoutError:
            with print_lock:
                logging.warning(f"{stock['code']} {stock['name']} 处理超时")
            with results_lock:
                problem_stocks.append(stock)
        except Exception as e:
            with print_lock:
                logging.error(f"{stock['code']} {stock['name']} 处理失败: {e}")
            with results_lock:
                problem_stocks.append(stock)
        
        with results_lock:
            completed_count += 1
            pbar.update(1)
    
    with tqdm(total=len(batch), desc="批次扫描进度") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_stock, stock): stock for _, stock in batch.iterrows()}
            for future, stock in futures.items():
                future.add_done_callback(lambda f: process_result(f, stock))
            
            start_time = time.time()
            while completed_count < len(batch):
                if time.time() - start_time > max_wait_time:
                    with print_lock:
                        remaining = len(batch) - completed_count
                        logging.warning(f"已等待 {max_wait_time} 秒，仍有 {remaining} 只股票未完成。继续处理下一批...")
                    pbar.update(len(batch) - completed_count)
                    for future, stock in futures.items():
                        if not future.done():
                            with results_lock:
                                problem_stocks.append(stock)
                    break
                time.sleep(1)
    
    if problem_stocks:
        problem_df = pd.DataFrame(problem_stocks)
        problem_df.to_csv(f'problem_stocks_{timestamp}.csv', index=False, encoding='utf-8-sig')
        with print_lock:
            logging.info(f"本批次有 {len(problem_stocks)} 只问题股票，已保存到 problem_stocks_{timestamp}.csv")
    
    return pd.DataFrame(results)

def scan_in_batches(batch_size=500, cooldown=60, max_workers=5, start_batch=1, end_batch=None, max_wait_time=600):
    """分批扫描A股"""
    logging.info("正在获取A股列表...")
    stock_list = get_cn_tickers()
    
    if stock_list.empty:
        logging.error("获取股票列表失败")
        return pd.DataFrame()
    
    total_stocks = len(stock_list)
    logging.info(f"共获取到 {total_stocks} 只股票")
    
    batch_count = (total_stocks + batch_size - 1) // batch_size
    if end_batch is None:
        end_batch = batch_count
    
    start_batch = max(1, min(start_batch, batch_count))
    end_batch = max(start_batch, min(end_batch, batch_count))
    
    logging.info(f"将扫描第 {start_batch} 到 {end_batch} 批次，共 {end_batch-start_batch+1} 个批次")
    
    all_results = []
    
    for batch_num in range(start_batch, end_batch + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_stocks)
        batch = stock_list.iloc[start_idx:end_idx].reset_index(drop=True)
        
        logging.info(f"开始扫描第 {batch_num}/{batch_count} 批次 ({len(batch)} 只股票)...")
        
        batch_start_time = time.time()
        results_df = _scan_batch(batch, max_workers=max_workers, max_wait_time=max_wait_time)
        batch_end_time = time.time()
        
        if not results_df.empty:
            all_results.append(results_df)
            logging.info(f"批次 {batch_num} 发现 {len(results_df)} 只有信号的股票")
        else:
            logging.info(f"批次 {batch_num} 未发现信号")
        
        batch_time = batch_end_time - batch_start_time
        logging.info(f"批次 {batch_num} 处理耗时: {batch_time:.2f} 秒")
        
        if all_results:
            interim_results = pd.concat(all_results, ignore_index=True)
            interim_results.to_csv(f'cn_signals_interim_{batch_num}.csv', index=False, encoding='utf-8-sig')
            logging.info(f"已将中间结果保存到 cn_signals_interim_{batch_num}.csv")
        
        if batch_num < end_batch:
            logging.info(f"批次 {batch_num} 完成，休息 {cooldown} 秒...")
            time.sleep(cooldown)
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        return final_results
    else:
        return pd.DataFrame()

def main():
    """主函数"""
    start_time = time.time()
    
    os.makedirs("stock_cache", exist_ok=True)
    
    results = scan_in_batches(batch_size=500, cooldown=30, max_workers=30, start_batch=10, end_batch=None, max_wait_time=600)
    
    if not results.empty:
        print("\n发现信号的股票（仅BLUE和LIRED）：")
        print("=" * 160)
        print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交额(万)':>12} | {'日BLUE':>8} | {'日BLUE天数':>4} | {'日LIRED':>8} | {'日LIRED数':>4} | "
              f"{'周BLUE':>8} | {'周BLUE周数':>4} | {'周LIRED':>8} | {'周LIRED数':>4} | {'信号':<20}")
        print("-" * 160)
        
        signal_counts = {
            '日BLUE>150': 0,
            '周BLUE>150': 0,
            '日LIRED<-150': 0,
            '周LIRED<-150': 0
        }
        
        for _, row in results.iterrows():
            signals = []
            if row['blue_days'] >= 3 and row['blue_weeks'] >= 2:
                signals.append(f'日BLUE>150({row["blue_days"]}天)')
                signal_counts['日BLUE>150'] += 1
                signals.append(f'周BLUE>150({row["blue_weeks"]}周)')
                signal_counts['周BLUE>150'] += 1
            if row['lired_days'] >= 3 and row['lired_weeks'] >= 2:
                signals.append(f'日LIRED<-150({row["lired_days"]}天)')
                signal_counts['日LIRED<-150'] += 1
                signals.append(f'周LIRED<-150({row["lired_weeks"]}周)')
                signal_counts['周LIRED<-150'] += 1
            
            signals_str = ', '.join(signals)
            if signals_str:
                print(f"{row['symbol']:<8} | {row['name']:<40} | {row['price']:8.2f} | {row['turnover']:12.2f} | {row['blue_daily']:8.2f} | {row['blue_days']:4d} | "
                    f"{row['lired_daily']:8.2f} | {row['lired_days']:4d} | {row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | "
                    f"{row['lired_weekly']:8.2f} | {row['lired_weeks']:4d} | {signals_str:<20}")
        
        print("=" * 160)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, count in signal_counts.items():
            if count > 0:
                print(f"{signal:<15}: {count:>5} 只")
        print("-" * 40)
        print(f"共发现 {len(results)} 只股票有信号")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cn_signals_{timestamp}.csv'
        results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()