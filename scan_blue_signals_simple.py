#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版BLUE信号A股扫描脚本
只关注BLUE信号，使用简化的数据获取方法
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import time
import threading
import concurrent.futures
from tqdm import tqdm
import os
import traceback
import argparse

# 导入增强版股票列表
from enhanced_stock_list import get_enhanced_cn_stock_list

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 定义技术指标函数
def REF(series, periods=1):
    return pd.Series(series).shift(periods).values

def EMA(series, periods):
    return pd.Series(series).ewm(span=periods, adjust=False).mean().values

def SMA(series, periods, weight=1):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def IF(condition, value_if_true, value_if_false):
    return np.where(condition, value_if_true, value_if_false)

def LLV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).min().values

def HHV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).max().values

def get_stock_data_simple(symbol):
    """简化版股票数据获取"""
    try:
        import akshare as ak
        
        # 转换symbol格式（从tushare格式转换为akshare格式）
        if symbol.endswith('.SH'):
            ak_symbol = symbol[:-3]  # 去掉.SH
        elif symbol.endswith('.SZ'):
            ak_symbol = symbol[:-3]  # 去掉.SZ
        elif symbol.endswith('.BJ'):
            ak_symbol = symbol[:-3]  # 去掉.BJ
        else:
            ak_symbol = symbol
        
        # 获取历史数据
        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", adjust="qfq")
        
        if df.empty:
            return None
        
        # 重命名列
        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume',
            '成交额': 'Amount'
        }
        
        # 只重命名存在的列
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # 确保Date列是datetime类型
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        # 确保数据类型正确
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 按日期排序
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        with print_lock:
            print(f"⚠️ 获取{symbol}数据失败: {e}")
        return None

def convert_to_weekly(daily_df):
    """将日线数据转换为周线数据"""
    if daily_df is None or daily_df.empty:
        return None
    
    try:
        # 转换为周线数据
        weekly_df = daily_df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return weekly_df
        
    except Exception as e:
        print(f"周线转换失败: {e}")
        return None

def calculate_blue_signals(data_daily, data_weekly):
    """计算BLUE信号"""
    try:
        if data_daily is None or data_weekly is None:
            return None
        
        if len(data_daily) < 100 or len(data_weekly) < 20:
            return None
        
        # 日线BLUE计算
        OPEN_D = data_daily['Open'].values
        HIGH_D = data_daily['High'].values
        LOW_D = data_daily['Low'].values
        CLOSE_D = data_daily['Close'].values
        
        VAR1_D = REF((LOW_D + OPEN_D + CLOSE_D + HIGH_D) / 4, 1)
        VAR2_D = SMA(np.abs(LOW_D - VAR1_D), 13, 1) / SMA(np.maximum(LOW_D - VAR1_D, 0), 10, 1)
        VAR3_D = EMA(VAR2_D, 10)
        VAR4_D = LLV(LOW_D, 9)
        VAR5_D = HHV(VAR3_D, 30)
        VAR6_D = IF(LLV(LOW_D, 58) == VAR4_D, VAR3_D, 0)
        
        max_value_daily = np.nanmax(VAR6_D)
        RADIO1_D = 200 / max_value_daily if max_value_daily > 0 else 1
        BLUE_D = IF(VAR5_D > REF(VAR5_D, 1), VAR6_D * RADIO1_D, 0)
        
        # 周线BLUE计算
        OPEN_W = data_weekly['Open'].values
        HIGH_W = data_weekly['High'].values
        LOW_W = data_weekly['Low'].values
        CLOSE_W = data_weekly['Close'].values
        
        VAR1_W = REF((LOW_W + OPEN_W + CLOSE_W + HIGH_W) / 4, 1)
        VAR2_W = SMA(np.abs(LOW_W - VAR1_W), 13, 1) / SMA(np.maximum(LOW_W - VAR1_W, 0), 10, 1)
        VAR3_W = EMA(VAR2_W, 10)
        VAR4_W = LLV(LOW_W, 9)
        VAR5_W = HHV(VAR3_W, 30)
        VAR6_W = IF(LLV(LOW_W, 58) == VAR4_W, VAR3_W, 0)
        
        max_value_weekly = np.nanmax(VAR6_W)
        RADIO1_W = 200 / max_value_weekly if max_value_weekly > 0 else 1
        BLUE_W = IF(VAR5_W > REF(VAR5_W, 1), VAR6_W * RADIO1_W, 0)
        
        return {
            'daily_blue': BLUE_D,
            'weekly_blue': BLUE_W,
            'daily_close': CLOSE_D,
            'weekly_close': CLOSE_W,
            'daily_volume': data_daily['Volume'].values,
            'weekly_volume': data_weekly['Volume'].values
        }
        
    except Exception as e:
        print(f"BLUE信号计算失败: {e}")
        return None

def process_single_stock_simple(symbol, thresholds):
    """处理单个股票，计算BLUE信号"""
    try:
        # 获取股票数据
        data_daily = get_stock_data_simple(symbol)
        if data_daily is None:
            return None
        
        # 转换为周线数据
        data_weekly = convert_to_weekly(data_daily)
        if data_weekly is None:
            return None
        
        # 计算BLUE信号
        signals = calculate_blue_signals(data_daily, data_weekly)
        if signals is None:
            return None
        
        # 分析最近的信号
        recent_daily = signals['daily_blue'][-6:]  # 最近6天
        recent_weekly = signals['weekly_blue'][-5:]  # 最近5周
        
        # 查找满足BLUE条件的信号
        day_blue_signals = [x for x in recent_daily if x > thresholds['day_blue']]
        week_blue_signals = [x for x in recent_weekly if x > thresholds['week_blue']]
        
        day_blue_count = len(day_blue_signals)
        week_blue_count = len(week_blue_signals)
        
        # 判断是否满足条件
        has_day_blue = day_blue_count >= thresholds['day_blue_count']
        has_week_blue = week_blue_count >= thresholds['week_blue_count']
        
        if not (has_day_blue or has_week_blue):
            return None
        
        # 获取最新数据
        latest_price = signals['daily_close'][-1]
        latest_volume = signals['daily_volume'][-1]
        latest_day_blue = signals['daily_blue'][-1]
        latest_week_blue = signals['weekly_blue'][-1]
        
        # 计算成交额（万元）
        turnover = latest_volume * latest_price / 10000
        
        # 获取最近一次满足条件的信号值
        latest_day_blue_value = day_blue_signals[-1] if day_blue_signals else 0
        latest_week_blue_value = week_blue_signals[-1] if week_blue_signals else 0
        
        # 检查是否同时满足日线和周线条件
        has_day_week_blue = has_day_blue and has_week_blue
        
        result = {
            'symbol': symbol,
            'price': latest_price,
            'volume': latest_volume,
            'turnover': turnover,
            'blue_daily': latest_day_blue,
            'blue_weekly': latest_week_blue,
            'blue_days': day_blue_count,
            'blue_weeks': week_blue_count,
            'latest_day_blue_value': latest_day_blue_value,
            'latest_week_blue_value': latest_week_blue_value,
            'has_day_week_blue': has_day_week_blue,
            'timestamp': datetime.now()
        }
        
        with print_lock:
            signal_desc = []
            if has_day_blue:
                signal_desc.append(f"日BLUE:{day_blue_count}天({latest_day_blue_value:.1f})")
            if has_week_blue:
                signal_desc.append(f"周BLUE:{week_blue_count}周({latest_week_blue_value:.1f})")
            
            signal_str = ", ".join(signal_desc)
            dual_flag = " ⭐" if has_day_week_blue else ""
            
            print(f"✅ 发现BLUE信号: {symbol} - {signal_str}, 价格:{latest_price:.2f}, 成交额:{turnover:.0f}万{dual_flag}")
        
        return result
        
    except Exception as e:
        with print_lock:
            print(f"⚠️ 处理{symbol}失败: {e}")
        return None

def get_cn_tickers_simple():
    """获取A股股票列表"""
    try:
        # 使用增强版股票列表获取方法
        stock_df = get_enhanced_cn_stock_list(force_refresh=False)
        
        if stock_df.empty:
            print("❌ 无法获取股票列表")
            return []
        
        # 转换为列表格式
        tickers = []
        for _, row in stock_df.iterrows():
            tickers.append({
                'code': row['code'],
                'name': row['name']
            })
        
        print(f"✅ 获取到 {len(tickers)} 只A股")
        return tickers
        
    except Exception as e:
        print(f"❌ 获取A股列表失败: {e}")
        return []

def scan_blue_signals_simple(tickers, max_workers=10, min_turnover=200, thresholds=None):
    """简化版BLUE信号扫描"""
    
    # 设置默认阈值
    default_thresholds = {
        'day_blue': 100,
        'week_blue': 130,
        'day_blue_count': 3,
        'week_blue_count': 2
    }
    
    if thresholds:
        default_thresholds.update(thresholds)
    
    print(f"🔍 开始扫描BLUE信号...")
    print(f"📊 股票数量: {len(tickers)}")
    print(f"⚙️ 参数: 线程数={max_workers}, 最小成交额={min_turnover}万")
    print(f"🎯 BLUE阈值: 日线>{default_thresholds['day_blue']}, 周线>{default_thresholds['week_blue']}")
    print(f"📈 信号条件: 日线{default_thresholds['day_blue_count']}天, 周线{default_thresholds['week_blue_count']}周")
    print("-" * 80)
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_symbol = {
            executor.submit(process_single_stock_simple, ticker['code'], default_thresholds): ticker['code'] 
            for ticker in tickers
        }
        
        # 处理结果
        for future in tqdm(concurrent.futures.as_completed(future_to_symbol), 
                          total=len(future_to_symbol), desc="扫描进度"):
            try:
                result = future.result(timeout=30)
                if result and result['turnover'] >= min_turnover:
                    with results_lock:
                        results.append(result)
                        
            except Exception as e:
                symbol = future_to_symbol[future]
                print(f"任务异常: {symbol} - {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n🎉 BLUE信号扫描完成!")
    print(f"⏱️ 耗时: {elapsed_time:.2f} 秒")
    print(f"🎯 发现 {len(results)} 只股票满足BLUE信号条件")
    
    if results:
        # 按成交额排序
        results.sort(key=lambda x: x['turnover'], reverse=True)
        
        # 保存结果到CSV
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'blue_signals_simple_{timestamp}.csv'
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 结果已保存到: {filename}")
        
        # 显示前10个结果
        print(f"\n📋 前10个BLUE信号:")
        print("-" * 100)
        print(f"{'序号':<4} {'代码':<10} {'价格':<8} {'成交额(万)':<10} {'日BLUE':<15} {'周BLUE':<15} {'同时':<4}")
        print("-" * 100)
        
        for i, result in enumerate(results[:10], 1):
            day_blue_desc = f"{result['blue_days']}天({result['latest_day_blue_value']:.1f})" if result['blue_days'] >= default_thresholds['day_blue_count'] else "-"
            week_blue_desc = f"{result['blue_weeks']}周({result['latest_week_blue_value']:.1f})" if result['blue_weeks'] >= default_thresholds['week_blue_count'] else "-"
            dual_flag = "⭐" if result['has_day_week_blue'] else ""
            
            print(f"{i:<4} {result['symbol']:<10} {result['price']:<8.2f} {result['turnover']:<10.0f} {day_blue_desc:<15} {week_blue_desc:<15} {dual_flag:<4}")
        
        # 统计信息
        day_blue_count = len([r for r in results if r['blue_days'] >= default_thresholds['day_blue_count']])
        week_blue_count = len([r for r in results if r['blue_weeks'] >= default_thresholds['week_blue_count']])
        dual_blue_count = len([r for r in results if r['has_day_week_blue']])
        
        print(f"\n📊 BLUE信号统计:")
        print(f"   日线BLUE信号: {day_blue_count} 只")
        print(f"   周线BLUE信号: {week_blue_count} 只")
        print(f"   日周同时BLUE: {dual_blue_count} 只")
        
        # 显示日周同时BLUE的股票
        if dual_blue_count > 0:
            dual_stocks = [r for r in results if r['has_day_week_blue']]
            print(f"\n⭐ 日周同时BLUE的股票 ({dual_blue_count}只):")
            print("-" * 80)
            for stock in dual_stocks:
                print(f"   {stock['symbol']} - 价格:{stock['price']:.2f}, 成交额:{stock['turnover']:.0f}万")
                print(f"     日BLUE: {stock['blue_days']}天({stock['latest_day_blue_value']:.1f})")
                print(f"     周BLUE: {stock['blue_weeks']}周({stock['latest_week_blue_value']:.1f})")
                print()
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版A股BLUE信号扫描器')
    parser.add_argument('--batch_size', type=int, default=0, help='批量处理数量 (0=全部)')
    parser.add_argument('--max_workers', type=int, default=10, help='最大线程数')
    parser.add_argument('--min_turnover', type=float, default=200, help='最小成交额(万元)')
    parser.add_argument('--day_blue', type=float, default=100, help='日线BLUE阈值')
    parser.add_argument('--week_blue', type=float, default=130, help='周线BLUE阈值')
    parser.add_argument('--day_blue_count', type=int, default=3, help='日线BLUE出现次数')
    parser.add_argument('--week_blue_count', type=int, default=2, help='周线BLUE出现次数')
    parser.add_argument('--timing', type=str, default='', help='时机标识')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("🔵 简化版A股BLUE信号扫描系统")
        print("=" * 80)
        print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.timing:
            print(f"扫描时机: {args.timing}")
        print("=" * 80)
        
        # 获取股票列表
        tickers = get_cn_tickers_simple()
        if not tickers:
            print("❌ 无法获取股票列表")
            return
        
        # 限制批量大小
        if args.batch_size > 0:
            tickers = tickers[:args.batch_size]
            print(f"🎯 本次扫描: {len(tickers)} 只股票")
        
        # 设置阈值
        thresholds = {
            'day_blue': args.day_blue,
            'week_blue': args.week_blue,
            'day_blue_count': args.day_blue_count,
            'week_blue_count': args.week_blue_count
        }
        
        # 开始扫描
        results = scan_blue_signals_simple(
            tickers=tickers,
            max_workers=args.max_workers,
            min_turnover=args.min_turnover,
            thresholds=thresholds
        )
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断扫描")
    except Exception as e:
        print(f"❌ 扫描过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

