"""
信号回测系统 - 验证 BLUE + 黑马信号的买入效果

功能:
1. 回测历史信号的买入效果
2. 计算不同持有周期的收益率
3. 统计胜率、平均收益、最大回撤等指标
4. 支持 A 股和美股

用法示例:
    # 回测 A 股 BLUE 信号，持有 5 天
    python signal_backtest.py --market cn --signal blue --hold-days 5
    
    # 回测 A 股 BLUE+黑马 信号，持有 10 天
    python signal_backtest.py --market cn --signal blue --with-heima --hold-days 10
    
    # 回测美股 BLUE 信号，持有 5/10/20 天
    python signal_backtest.py --market us --signal blue --hold-days 5 10 20
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import argparse
import os
import logging
from tqdm import tqdm
import concurrent.futures
import threading

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()


# ==================== 数据获取 ====================

def get_cn_stock_data(ts_code, start_date, end_date, pro=None):
    """获取 A 股历史数据"""
    try:
        if pro is None:
            import tushare as ts
            ts.set_token('gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482')
            pro = ts.pro_api()
        
        # 使用与原扫描脚本相同的数据获取方式
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df is None or df.empty:
            return None
        
        df = df.rename(columns={
            'trade_date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        })
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df = df.set_index('Date').sort_index()
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        logging.error(f"获取 A 股数据失败 {ts_code}: {e}")
        return None


# 全局 tushare pro 实例
_tushare_pro = None

def get_tushare_pro():
    """获取全局 tushare pro 实例"""
    global _tushare_pro
    if _tushare_pro is None:
        import tushare as ts
        ts.set_token('gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482')
        _tushare_pro = ts.pro_api()
    return _tushare_pro


def get_us_stock_data(symbol, start_date, end_date):
    """获取美股历史数据"""
    try:
        from polygon import RESTClient
        api_key = os.getenv('POLYGON_API_KEY', 'qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D')
        client = RESTClient(api_key)
        
        aggs = []
        for a in client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append({
                'Date': pd.Timestamp.fromtimestamp(a.timestamp/1000),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
            })
        
        if not aggs:
            return None
        
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logging.error(f"获取美股数据失败 {symbol}: {e}")
        return None


# ==================== 技术指标计算 ====================

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

def MA(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def AVEDEV(series, periods):
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values


def calculate_signals(df):
    """计算 BLUE、LIRED 和黑马信号"""
    OPEN = df['Open'].values
    HIGH = df['High'].values
    LOW = df['Low'].values
    CLOSE = df['Close'].values
    
    # BLUE/LIRED 计算
    VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
    VAR2 = SMA(np.abs(LOW - VAR1), 13, 1) / SMA(np.maximum(LOW - VAR1, 0), 10, 1)
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(LOW, 33)
    VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
    VAR6 = POW(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    VAR21 = SMA(np.abs(HIGH - VAR1), 13, 1) / SMA(np.minimum(HIGH - VAR1, 0), 10, 1)
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(HIGH, 33)
    VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)
    VAR61 = POW(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    max_value = np.nanmax(np.maximum(VAR6, np.abs(VAR61)))
    RADIO1 = 200 / max_value if max_value > 0 else 1
    BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
    LIRED = IF(VAR51 > REF(VAR51, 1), -VAR61 * RADIO1, 0)
    
    # 黑马信号计算
    VAR1_H = (HIGH + LOW + CLOSE) / 3
    ma_var1 = MA(VAR1_H, 14)
    avedev_var1 = AVEDEV(VAR1_H, 14)
    avedev_var1 = np.where(avedev_var1 == 0, 0.0001, avedev_var1)
    CCI = (VAR1_H - ma_var1) / (0.015 * avedev_var1)
    
    # 简化的黑马信号
    close_series = pd.Series(CLOSE)
    rolling_min = close_series.rolling(window=22, min_periods=1).min()
    rolling_max = close_series.rolling(window=22, min_periods=1).max()
    pct_change = close_series.pct_change()
    is_rising = pct_change > 0
    was_falling_1 = pct_change.shift(1) <= 0
    was_falling_2 = pct_change.shift(2) <= 0
    was_falling_3 = pct_change.shift(3) <= 0
    near_low = (close_series - rolling_min) / (rolling_max - rolling_min + 0.0001) < 0.2
    
    heima_signal = (CCI < -110) & is_rising & was_falling_1 & was_falling_2 & was_falling_3 & near_low
    
    df['BLUE'] = BLUE
    df['LIRED'] = LIRED
    df['CCI'] = CCI
    df['heima'] = heima_signal
    
    return df


def find_signal_dates(df, signal_type='blue', blue_threshold=100, lired_threshold=-100, 
                      require_heima=False, lookback=6):
    """找出满足条件的信号日期"""
    signal_dates = []
    
    for i in range(lookback, len(df)):
        recent = df.iloc[i-lookback+1:i+1]
        current_date = df.index[i]
        
        has_signal = False
        
        if signal_type in ['blue', 'all']:
            blue_count = (recent['BLUE'] > blue_threshold).sum()
            if blue_count >= 3:
                has_signal = True
        
        if signal_type in ['red', 'all']:
            lired_count = (recent['LIRED'] < lired_threshold).sum()
            if lired_count >= 3:
                has_signal = True
        
        if require_heima:
            heima_count = recent['heima'].sum()
            if heima_count == 0:
                has_signal = False
        
        if has_signal:
            signal_dates.append({
                'date': current_date,
                'close': df.loc[current_date, 'Close'],
                'blue': df.loc[current_date, 'BLUE'],
                'lired': df.loc[current_date, 'LIRED'],
                'cci': df.loc[current_date, 'CCI'],
                'heima': df.loc[current_date, 'heima']
            })
    
    return signal_dates


def calculate_returns(df, signal_date, hold_days_list):
    """计算信号日期后不同持有天数的收益率"""
    try:
        signal_idx = df.index.get_loc(signal_date)
        buy_price = df.iloc[signal_idx]['Close']
        
        returns = {}
        for days in hold_days_list:
            target_idx = signal_idx + days
            if target_idx < len(df):
                sell_price = df.iloc[target_idx]['Close']
                ret = (sell_price - buy_price) / buy_price * 100
                returns[f'{days}d'] = ret
                returns[f'{days}d_price'] = sell_price
            else:
                returns[f'{days}d'] = np.nan
                returns[f'{days}d_price'] = np.nan
        
        # 计算最大回撤（持有期间）
        max_days = max(hold_days_list)
        end_idx = min(signal_idx + max_days, len(df))
        period_data = df.iloc[signal_idx:end_idx+1]
        
        if len(period_data) > 0:
            max_price = period_data['High'].max()
            min_price = period_data['Low'].min()
            returns['max_gain'] = (max_price - buy_price) / buy_price * 100
            returns['max_loss'] = (min_price - buy_price) / buy_price * 100
        
        return returns
    except Exception as e:
        return None


def backtest_single_stock(symbol, market, signal_type, hold_days_list, require_heima, 
                          start_date, end_date, thresholds, pro=None):
    """回测单只股票"""
    try:
        # 获取数据
        if market == 'cn':
            df = get_cn_stock_data(symbol, start_date, end_date, pro=pro)
        else:
            df = get_us_stock_data(symbol, start_date, end_date)
        
        if df is None or len(df) < 50:
            return None
        
        # 计算信号
        df = calculate_signals(df)
        
        # 找出信号日期
        signal_dates = find_signal_dates(
            df, signal_type, 
            thresholds['blue'], thresholds['lired'],
            require_heima, thresholds['lookback']
        )
        
        if not signal_dates:
            return None
        
        # 计算每个信号的收益
        results = []
        for signal in signal_dates:
            returns = calculate_returns(df, signal['date'], hold_days_list)
            if returns:
                result = {
                    'symbol': symbol,
                    'signal_date': signal['date'],
                    'buy_price': signal['close'],
                    'blue': signal['blue'],
                    'lired': signal['lired'],
                    'cci': signal['cci'],
                    'heima': signal['heima'],
                    **returns
                }
                results.append(result)
        
        return results
    except Exception as e:
        with print_lock:
            logging.error(f"回测 {symbol} 失败: {e}")
        return None


def run_backtest(symbols, market, signal_type, hold_days_list, require_heima, 
                 start_date, end_date, thresholds, max_workers=5, batch_size=100):
    """分批并行回测多只股票"""
    all_results = []
    
    # 获取全局 pro 实例
    pro = get_tushare_pro() if market == 'cn' else None
    
    # 分批处理以避免 API 限制
    total_batches = (len(symbols) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(symbols))
        batch_symbols = symbols[batch_start:batch_end]
        
        print(f"\n处理批次 {batch_idx + 1}/{total_batches} ({len(batch_symbols)} 只股票)...")
        
        with tqdm(total=len(batch_symbols), desc=f"批次 {batch_idx + 1}") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        backtest_single_stock, symbol, market, signal_type, 
                        hold_days_list, require_heima, start_date, end_date, thresholds, pro
                    ): symbol for symbol in batch_symbols
                }
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.extend(result)
                    pbar.update(1)
        
        # 批次间休息以避免 API 限制
        if batch_idx < total_batches - 1:
            print(f"批次 {batch_idx + 1} 完成，休息 30 秒...")
            time.sleep(30)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def analyze_results(df, hold_days_list):
    """分析回测结果"""
    print("\n" + "=" * 80)
    print("回测结果分析")
    print("=" * 80)
    
    print(f"\n总信号数: {len(df)}")
    print(f"涉及股票数: {df['symbol'].nunique()}")
    print(f"信号日期范围: {df['signal_date'].min()} ~ {df['signal_date'].max()}")
    
    print("\n" + "-" * 80)
    print("不同持有天数的收益统计")
    print("-" * 80)
    
    stats = []
    for days in hold_days_list:
        col = f'{days}d'
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                win_count = (valid_data > 0).sum()
                win_rate = win_count / len(valid_data) * 100
                
                stat = {
                    '持有天数': days,
                    '样本数': len(valid_data),
                    '平均收益率(%)': valid_data.mean(),
                    '中位数收益率(%)': valid_data.median(),
                    '胜率(%)': win_rate,
                    '最大收益(%)': valid_data.max(),
                    '最大亏损(%)': valid_data.min(),
                    '收益标准差(%)': valid_data.std()
                }
                stats.append(stat)
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # 按月份统计
    print("\n" + "-" * 80)
    print("按月份统计（以5天收益为例）")
    print("-" * 80)
    
    if '5d' in df.columns:
        df['month'] = df['signal_date'].dt.to_period('M')
        monthly_stats = df.groupby('month')['5d'].agg(['count', 'mean', lambda x: (x > 0).mean() * 100])
        monthly_stats.columns = ['信号数', '平均收益(%)', '胜率(%)']
        print(monthly_stats.tail(12).to_string())
    
    # 最佳和最差交易
    print("\n" + "-" * 80)
    print("最佳交易 TOP 10（5天收益）")
    print("-" * 80)
    
    if '5d' in df.columns:
        best = df.nlargest(10, '5d')[['symbol', 'signal_date', 'buy_price', '5d', 'blue', 'heima']]
        print(best.to_string(index=False))
    
    print("\n" + "-" * 80)
    print("最差交易 TOP 10（5天收益）")
    print("-" * 80)
    
    if '5d' in df.columns:
        worst = df.nsmallest(10, '5d')[['symbol', 'signal_date', 'buy_price', '5d', 'blue', 'heima']]
        print(worst.to_string(index=False))
    
    return stats_df


def get_stock_list(market, limit=None):
    """获取股票列表"""
    if market == 'cn':
        try:
            import tushare as ts
            ts.set_token('gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482')
            pro = ts.pro_api()
            stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
            symbols = stock_info['ts_code'].tolist()
            if limit:
                symbols = symbols[:limit]
            return symbols
        except Exception as e:
            logging.error(f"获取 A 股列表失败: {e}")
            return []
    else:
        # 美股使用缓存的列表
        cache_file = 'tickers_cache.json'
        if os.path.exists(cache_file):
            import json
            with open(cache_file, 'r') as f:
                symbols = json.load(f)
            if limit:
                symbols = symbols[:limit]
            return symbols
        else:
            logging.error("未找到美股缓存列表")
            return []


def main():
    parser = argparse.ArgumentParser(
        description='信号回测系统 - 验证 BLUE + 黑马信号的买入效果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
    # 回测 A 股 BLUE 信号，持有 5 天
    python signal_backtest.py --market cn --signal blue --hold-days 5
    
    # 回测 A 股 BLUE+黑马 信号，持有 5/10/20 天
    python signal_backtest.py --market cn --signal blue --with-heima --hold-days 5 10 20
    
    # 回测美股 BLUE 信号
    python signal_backtest.py --market us --signal blue --hold-days 5 10 20
    
    # 限制股票数量（用于快速测试）
    python signal_backtest.py --market cn --signal blue --limit 100 --hold-days 5
        """
    )
    
    parser.add_argument('--market', '-m', choices=['cn', 'us'], default='cn',
                        help='市场: cn=A股, us=美股')
    parser.add_argument('--signal', '-s', choices=['blue', 'red', 'all'], default='blue',
                        help='信号类型: blue=多头, red=空头, all=全部')
    parser.add_argument('--with-heima', action='store_true',
                        help='要求同时有黑马信号')
    parser.add_argument('--hold-days', '-d', type=int, nargs='+', default=[5, 10, 20],
                        help='持有天数（可指定多个）')
    parser.add_argument('--start-date', default=None,
                        help='回测开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', default=None,
                        help='回测结束日期 (YYYYMMDD)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='限制股票数量')
    parser.add_argument('--workers', '-w', type=int, default=5,
                        help='并发线程数')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='每批处理的股票数量')
    parser.add_argument('--blue-threshold', type=float, default=100,
                        help='BLUE 信号阈值')
    parser.add_argument('--lired-threshold', type=float, default=-100,
                        help='LIRED 信号阈值')
    
    args = parser.parse_args()
    
    # 设置日期
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y%m%d')
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    # 打印配置
    signal_desc = {'blue': 'BLUE(多头)', 'red': 'LIRED(空头)', 'all': '全部'}
    market_desc = {'cn': 'A股', 'us': '美股'}
    heima_desc = ' + 黑马信号' if args.with_heima else ''
    
    print("\n" + "=" * 60)
    print("信号回测系统")
    print("=" * 60)
    print(f"市场: {market_desc[args.market]}")
    print(f"信号类型: {signal_desc[args.signal]}{heima_desc}")
    print(f"持有天数: {args.hold_days}")
    print(f"回测区间: {args.start_date} ~ {args.end_date}")
    print(f"BLUE阈值: {args.blue_threshold}, LIRED阈值: {args.lired_threshold}")
    print("=" * 60)
    
    # 获取股票列表
    print("\n正在获取股票列表...")
    symbols = get_stock_list(args.market, args.limit)
    print(f"共 {len(symbols)} 只股票")
    
    if not symbols:
        print("未获取到股票列表")
        return
    
    # 设置阈值
    thresholds = {
        'blue': args.blue_threshold,
        'lired': args.lired_threshold,
        'lookback': 6
    }
    
    # 运行回测
    print("\n开始回测...")
    start_time = time.time()
    
    results = run_backtest(
        symbols, args.market, args.signal, args.hold_days,
        args.with_heima, args.start_date, args.end_date,
        thresholds, args.workers, args.batch_size
    )
    
    end_time = time.time()
    print(f"\n回测完成，耗时: {end_time - start_time:.2f} 秒")
    
    if results.empty:
        print("未找到任何信号")
        return
    
    # 分析结果
    stats = analyze_results(results, args.hold_days)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_suffix = args.signal
    heima_suffix = "_heima" if args.with_heima else ""
    
    results_file = f'backtest_{args.market}_{signal_suffix}{heima_suffix}_{timestamp}.csv'
    results.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {results_file}")
    
    stats_file = f'backtest_stats_{args.market}_{signal_suffix}{heima_suffix}_{timestamp}.csv'
    stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存到: {stats_file}")


if __name__ == "__main__":
    main()

