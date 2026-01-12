#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
信号追踪服务 - 计算历史信号的后续表现
"""
import os
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import (
    query_scan_results, 
    get_scanned_dates,
    get_connection
)
from data_fetcher import get_us_stock_data, get_cn_stock_data


def calculate_signal_returns(symbol: str, signal_date: str, market: str = 'US', 
                             days_list: list = [5, 10, 20]) -> dict:
    """
    计算单个信号的后续收益
    
    Args:
        symbol: 股票代码
        signal_date: 信号日期 (YYYY-MM-DD)
        market: 市场 (US/CN)
        days_list: 要计算的天数列表
    
    Returns:
        dict: 包含各时间点收益、最大涨幅、最大回撤
    """
    try:
        # 获取历史数据
        signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
        end_date = signal_dt + timedelta(days=max(days_list) + 30)
        
        if market == 'US':
            df = get_us_stock_data(symbol, days=max(days_list) + 60)
        else:
            # CN 市场
            df = get_cn_stock_data(symbol, days=max(days_list) + 60)
        
        if df is None or df.empty:
            return None
        
        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 找到信号日期之后的数据
        df_after = df[df.index >= signal_dt]
        
        if len(df_after) < 2:
            return None
        
        # 入场价格（信号日收盘价）
        entry_price = df_after['Close'].iloc[0]
        entry_date = df_after.index[0]
        
        result = {
            'symbol': symbol,
            'signal_date': signal_date,
            'entry_price': entry_price,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
        }
        
        # 计算各时间点收益
        for days in days_list:
            if days < len(df_after):
                exit_price = df_after['Close'].iloc[days]
                ret = (exit_price - entry_price) / entry_price * 100
                result[f'return_{days}d'] = round(ret, 2)
            else:
                result[f'return_{days}d'] = None
        
        # 计算追踪期间的最大涨幅和最大回撤
        track_data = df_after.head(max(days_list) + 1)
        if len(track_data) > 1:
            max_price = track_data['High'].max()
            min_price = track_data['Low'].min()
            result['max_gain'] = round((max_price - entry_price) / entry_price * 100, 2)
            result['max_drawdown'] = round((min_price - entry_price) / entry_price * 100, 2)
        
        # 当前价格和收益
        current_price = df['Close'].iloc[-1]
        result['current_price'] = round(current_price, 2)
        result['current_return'] = round((current_price - entry_price) / entry_price * 100, 2)
        
        return result
        
    except Exception as e:
        print(f"Error calculating returns for {symbol}: {e}")
        return None


def batch_calculate_returns(signals: list, market: str = 'US', 
                           max_workers: int = 10) -> list:
    """
    批量计算信号收益
    
    Args:
        signals: 信号列表 [{'symbol': 'AAPL', 'signal_date': '2025-12-01'}, ...]
        market: 市场
        max_workers: 并行线程数
    
    Returns:
        list: 收益计算结果列表
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sig in signals:
            future = executor.submit(
                calculate_signal_returns,
                sig['symbol'],
                sig['signal_date'],
                market
            )
            futures[future] = sig
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                # 合并原始信号信息
                sig = futures[future]
                result.update({
                    'day_blue': sig.get('day_blue', 0),
                    'week_blue': sig.get('week_blue', 0),
                    'name': sig.get('name', '')
                })
                results.append(result)
    
    return results


def get_signal_performance_summary(scan_date: str, market: str = 'US') -> dict:
    """
    获取某天扫描信号的整体表现统计
    
    Args:
        scan_date: 扫描日期
        market: 市场
    
    Returns:
        dict: 统计摘要
    """
    # 从数据库获取该天的扫描结果
    results = query_scan_results(scan_date=scan_date, market=market, limit=100)
    
    if not results:
        return None
    
    # 准备信号列表
    signals = [{
        'symbol': r['symbol'],
        'signal_date': scan_date,
        'day_blue': r.get('blue_daily', 0),
        'week_blue': r.get('blue_weekly', 0),
        'name': r.get('name', '')
    } for r in results]
    
    # 批量计算收益
    returns = batch_calculate_returns(signals, market, max_workers=15)
    
    if not returns:
        return None
    
    # 转换为 DataFrame 便于统计
    df = pd.DataFrame(returns)
    
    # 计算统计指标
    summary = {
        'scan_date': scan_date,
        'market': market,
        'total_signals': len(signals),
        'calculated_signals': len(returns),
    }
    
    # 各时间点统计
    for days in [5, 10, 20]:
        col = f'return_{days}d'
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                summary[f'avg_{days}d'] = round(valid.mean(), 2)
                summary[f'median_{days}d'] = round(valid.median(), 2)
                summary[f'win_rate_{days}d'] = round(len(valid[valid > 0]) / len(valid) * 100, 1)
                summary[f'big_win_{days}d'] = len(valid[valid > 10])  # 涨幅超10%
                summary[f'big_loss_{days}d'] = len(valid[valid < -10])  # 跌幅超10%
    
    # 分类信号
    if 'return_20d' in df.columns:
        df_valid = df.dropna(subset=['return_20d'])
        summary['excellent'] = len(df_valid[df_valid['return_20d'] > 10])  # >10%
        summary['good'] = len(df_valid[(df_valid['return_20d'] > 0) & (df_valid['return_20d'] <= 10)])  # 0-10%
        summary['poor'] = len(df_valid[df_valid['return_20d'] <= 0])  # <0%
    
    # 返回详细结果用于展示
    summary['details'] = returns
    
    return summary


def get_cached_performance(scan_date: str, market: str = 'US') -> list:
    """
    获取缓存的性能数据
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM signal_performance 
        WHERE scan_date = ? AND market = ?
    ''', (scan_date, market))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def save_performance_cache(performance_data: list):
    """
    保存性能计算结果到缓存
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    for p in performance_data:
        cursor.execute('''
            INSERT OR REPLACE INTO signal_performance 
            (symbol, scan_date, market, return_5d, return_10d, return_20d, 
             max_gain, max_drawdown, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            p.get('symbol'),
            p.get('signal_date'),
            p.get('market', 'US'),
            p.get('return_5d'),
            p.get('return_10d'),
            p.get('return_20d'),
            p.get('max_gain'),
            p.get('max_drawdown'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # 测试
    print("Testing signal tracker service...")
    
    # 测试单个股票
    result = calculate_signal_returns('AAPL', '2025-12-01', 'US')
    if result:
        print(f"AAPL returns: {result}")
    
    # 测试批量
    from db.database import init_db
    init_db()
    
    dates = get_scanned_dates(market='US')
    if dates:
        print(f"\nAvailable dates: {dates[:5]}")
        summary = get_signal_performance_summary(dates[0], 'US')
        if summary:
            print(f"\nSummary for {dates[0]}:")
            print(f"  Total signals: {summary['total_signals']}")
            print(f"  Win rate 20D: {summary.get('win_rate_20d', 'N/A')}%")
