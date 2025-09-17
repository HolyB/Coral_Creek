#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版A股信号扫描脚本
使用多数据源获取股票列表，提供更好的容错机制
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import threading
import concurrent.futures
from tqdm import tqdm
import os
import traceback
import logging
import argparse
import json

# 导入我们的增强股票列表模块
from enhanced_stock_list import get_enhanced_cn_stock_list

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建全局线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局结果存储
all_results = []

def get_cn_tickers_enhanced(force_refresh=False):
    """
    使用增强版方法获取A股股票列表
    支持多数据源，自动容错
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 使用增强版方法获取A股列表...")
        
        # 使用增强的股票列表获取函数
        stock_df = get_enhanced_cn_stock_list(force_refresh=force_refresh)
        
        if stock_df.empty:
            logger.error("❌ 所有数据源都无法获取股票列表")
            return pd.DataFrame()
        
        # 转换为原脚本期望的格式
        tickers = []
        for _, row in stock_df.iterrows():
            tickers.append({
                'code': row['code'],  # 已经是tushare格式
                'name': row['name']
            })
        
        logger.info(f"✅ 增强版方法获取到 {len(tickers)} 只A股")
        
        # 按市场分类统计
        sh_count = len([t for t in tickers if t['code'].endswith('.SH')])
        sz_count = len([t for t in tickers if t['code'].endswith('.SZ')])
        bj_count = len([t for t in tickers if t['code'].endswith('.BJ')])
        
        logger.info(f"📊 市场分布: 沪市{sh_count}只, 深市{sz_count}只, 北交所{bj_count}只")
        
        return pd.DataFrame(tickers)
        
    except Exception as e:
        logger.error(f"❌ 增强版股票列表获取失败: {e}")
        return pd.DataFrame()

def get_stock_data_with_fallback(symbol, retries=3, delay=1):
    """
    带容错机制的股票数据获取
    """
    for attempt in range(retries):
        try:
            # 尝试从不同数据源获取数据
            if attempt == 0:
                # 第一次尝试：使用tushare
                return get_stock_data_tushare(symbol)
            elif attempt == 1:
                # 第二次尝试：使用akshare
                return get_stock_data_akshare(symbol)
            else:
                # 第三次尝试：使用其他API
                return get_stock_data_alternative(symbol)
                
        except Exception as e:
            logging.warning(f"⚠️ 获取{symbol}数据失败 (尝试{attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # 递增延迟
            continue
    
    return None

def get_stock_data_tushare(symbol):
    """使用Tushare获取股票数据"""
    try:
        import tushare as ts
        
        # tushare配置
        TUSHARE_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        # 获取日线数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            return None
        
        # 转换为标准格式
        df = df.rename(columns={
            'trade_date': 'date',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume',
            'amount': 'Amount'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        raise Exception(f"Tushare数据获取失败: {e}")

def get_stock_data_akshare(symbol):
    """使用AKShare获取股票数据"""
    try:
        import akshare as ak
        
        # 转换symbol格式
        if symbol.endswith('.SH'):
            ak_symbol = f"sh{symbol[:-3]}"
        elif symbol.endswith('.SZ'):
            ak_symbol = f"sz{symbol[:-3]}"
        elif symbol.endswith('.BJ'):
            ak_symbol = symbol[:-3]  # 北交所股票
        else:
            raise Exception(f"未知股票代码格式: {symbol}")
        
        # 获取历史数据
        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", adjust="qfq")
        
        if df.empty:
            return None
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low', 
            '收盘': 'Close',
            '成交量': 'Volume',
            '成交额': 'Amount'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        raise Exception(f"AKShare数据获取失败: {e}")

def get_stock_data_alternative(symbol):
    """使用备用API获取股票数据"""
    try:
        # 这里可以添加其他数据源，比如东方财富API等
        # 目前返回None，表示暂未实现
        raise Exception("备用数据源暂未实现")
        
    except Exception as e:
        raise Exception(f"备用数据源失败: {e}")

def REF(series, periods=1):
    """向前引用函数"""
    return pd.Series(series).shift(periods).values

def calculate_signals(df):
    """计算技术指标和信号"""
    try:
        if df is None or df.empty or len(df) < 50:
            return None
        
        # 计算基础指标
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 计算MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD_DIF'] = exp1 - exp2
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9).mean()
        df['MACD_BAR'] = (df['MACD_DIF'] - df['MACD_DEA']) * 2
        
        # 计算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算成交额
        df['turnover'] = df['Amount'] / 10000  # 转换为万元
        
        # 信号检测
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 多头信号
        bullish_signal = (
            latest['Close'] > latest['MA5'] and
            latest['MA5'] > latest['MA10'] and
            latest['MACD_DIF'] > latest['MACD_DEA'] and
            latest['RSI'] > 50
        )
        
        # 空头信号
        bearish_signal = (
            latest['Close'] < latest['MA5'] and
            latest['MA5'] < latest['MA10'] and
            latest['MACD_DIF'] < latest['MACD_DEA'] and
            latest['RSI'] < 50
        )
        
        return {
            'price': latest['Close'],
            'turnover': latest['turnover'],
            'ma5': latest['MA5'],
            'ma10': latest['MA10'],
            'ma20': latest['MA20'],
            'macd_dif': latest['MACD_DIF'],
            'macd_dea': latest['MACD_DEA'],
            'rsi': latest['RSI'],
            'bullish_signal': bullish_signal,
            'bearish_signal': bearish_signal,
            'volume': latest['Volume']
        }
        
    except Exception as e:
        logging.error(f"信号计算错误: {e}")
        return None

def process_single_stock(stock, min_turnover=200, signal_type='both'):
    """处理单只股票"""
    symbol = stock['code']
    name = stock['name']
    
    try:
        # 使用容错机制获取数据
        df = get_stock_data_with_fallback(symbol)
        
        if df is None or df.empty:
            return None
        
        # 计算信号
        signals = calculate_signals(df)
        
        if signals is None:
            return None
        
        # 过滤条件
        if signals['turnover'] < min_turnover:
            return None
        
        # 信号过滤
        has_signal = False
        if signal_type == 'both':
            has_signal = signals['bullish_signal'] or signals['bearish_signal']
        elif signal_type == 'bullish':
            has_signal = signals['bullish_signal']
        elif signal_type == 'bearish':
            has_signal = signals['bearish_signal']
        
        if not has_signal:
            return None
        
        # 准备结果
        result = {
            'symbol': symbol,
            'name': name,
            'price': signals['price'],
            'turnover': signals['turnover'],
            'signal_type': 'bullish' if signals['bullish_signal'] else 'bearish',
            'ma5': signals['ma5'],
            'ma10': signals['ma10'],
            'rsi': signals['rsi'],
            'macd_dif': signals['macd_dif'],
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with print_lock:
            print(f"✅ 发现信号: {symbol} {name} - {result['signal_type']}")
        
        return result
        
    except Exception as e:
        with print_lock:
            logging.warning(f"⚠️ 处理{symbol}失败: {e}")
        return None

def scan_signals_enhanced(batch_size=500, max_workers=20, min_turnover=200, 
                         signal_type='both', timing='', force_refresh=False):
    """增强版A股信号扫描"""
    global all_results
    
    print("=" * 80)
    print(f"🚀 增强版A股信号扫描系统启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 获取股票列表
    print("📋 获取A股股票列表...")
    tickers_df = get_cn_tickers_enhanced(force_refresh=force_refresh)
    
    if tickers_df.empty:
        print("❌ 无法获取股票列表，程序退出")
        return
    
    print(f"📊 共获取到 {len(tickers_df)} 只A股")
    
    # 批量处理
    total_stocks = len(tickers_df)
    tickers_list = tickers_df.to_dict('records')
    
    # 限制处理数量
    if batch_size > 0:
        tickers_list = tickers_list[:batch_size]
        print(f"🎯 本次扫描数量: {len(tickers_list)} 只股票")
    
    print(f"⚙️ 扫描参数: 线程数={max_workers}, 最小成交额={min_turnover}万, 信号类型={signal_type}")
    print("-" * 80)
    
    # 多线程处理
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_stock = {
            executor.submit(process_single_stock, stock, min_turnover, signal_type): stock 
            for stock in tickers_list
        }
        
        # 处理结果
        for future in tqdm(concurrent.futures.as_completed(future_to_stock), 
                          total=len(future_to_stock), desc="扫描进度"):
            try:
                result = future.result(timeout=30)
                if result:
                    with results_lock:
                        all_results.append(result)
                        
            except Exception as e:
                stock = future_to_stock[future]
                logging.warning(f"任务异常: {stock['code']} - {e}")
    
    # 输出结果
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🎉 扫描完成!")
    print("=" * 80)
    print(f"⏱️ 扫描耗时: {elapsed_time:.2f} 秒")
    print(f"📊 处理股票: {len(tickers_list)} 只")
    print(f"🎯 发现信号: {len(all_results)} 个")
    
    if all_results:
        # 保存结果
        output_file = f"cn_signals_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 结果已保存到: {output_file}")
        
        # 显示前10个结果
        print("\n📋 前10个信号:")
        print(results_df[['symbol', 'name', 'signal_type', 'price', 'turnover']].head(10).to_string(index=False))
        
        # 统计信息
        signal_counts = results_df['signal_type'].value_counts()
        print(f"\n📊 信号统计:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} 个")
    
    print("=" * 80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版A股信号扫描系统')
    parser.add_argument('--batch_size', type=int, default=500, help='批量处理数量 (0=全部)')
    parser.add_argument('--max_workers', type=int, default=20, help='最大线程数')
    parser.add_argument('--min_turnover', type=float, default=200, help='最小成交额(万元)')
    parser.add_argument('--signal_type', choices=['both', 'bullish', 'bearish'], 
                       default='both', help='信号类型')
    parser.add_argument('--timing', type=str, default='', help='时机标识')
    parser.add_argument('--force_refresh', action='store_true', help='强制刷新股票列表')
    
    args = parser.parse_args()
    
    try:
        scan_signals_enhanced(
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            min_turnover=args.min_turnover,
            signal_type=args.signal_type,
            timing=args.timing,
            force_refresh=args.force_refresh
        )
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断扫描")
    except Exception as e:
        print(f"❌ 扫描过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

