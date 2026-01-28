#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Signal Performance - 计算信号前向收益

批量计算历史信号的前向收益并缓存到数据库
可以作为定时任务运行，也可以手动触发
"""
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data_fetcher import get_us_stock_data
from db.database import (
    get_signals_without_performance,
    upsert_signal_performance,
    get_performance_stats,
    init_db
)


def compute_forward_returns(symbol: str, signal_date: str, signal_price: float = None) -> Dict:
    """
    计算单个信号的前向收益
    
    Args:
        symbol: 股票代码
        signal_date: 信号日期 (YYYY-MM-DD)
        signal_price: 信号价格 (可选，如果没有会从数据获取)
    
    Returns:
        Dict with return_5d, return_10d, return_20d, max_gain, max_drawdown
    """
    try:
        # 获取股价数据
        df = get_us_stock_data(symbol, days=60)
        if df is None or df.empty:
            return {}
        
        import pandas as pd
        df.index = pd.to_datetime(df.index)
        signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
        
        # 找到信号日期或之后的第一个交易日
        valid_dates = df.index[df.index.date >= signal_dt.date()]
        if len(valid_dates) == 0:
            return {}
        
        base_date = valid_dates[0]
        base_idx = df.index.get_loc(base_date)
        base_price = signal_price if signal_price else df.loc[base_date, 'Close']
        
        result = {
            'symbol': symbol,
            'scan_date': signal_date,
        }
        
        # 计算 5/10/20 天收益
        for days in [5, 10, 20]:
            target_idx = base_idx + days
            if target_idx < len(df):
                future_price = df.iloc[target_idx]['Close']
                ret = (future_price - base_price) / base_price
                result[f'return_{days}d'] = round(float(ret), 4)
            else:
                result[f'return_{days}d'] = None
        
        # 计算最大涨幅和最大回撤 (20天内)
        if base_idx + 20 < len(df):
            future_prices = df.iloc[base_idx:base_idx + 20]['Close']
            returns = (future_prices - base_price) / base_price
            result['max_gain'] = round(float(returns.max()), 4)
            result['max_drawdown'] = round(float(returns.min()), 4)
        
        return result
        
    except Exception as e:
        print(f"Error computing returns for {symbol}: {e}")
        return {}


def batch_compute_performance(market: str = 'US', limit: int = 100, verbose: bool = True):
    """
    批量计算缺失的信号性能数据
    
    Args:
        market: 市场 (US/CN)
        limit: 最多处理的信号数量
        verbose: 是否打印进度
    
    Returns:
        (processed, success, failed) 计数
    """
    # 确保数据库初始化
    init_db()
    
    # 获取需要计算的信号
    signals = get_signals_without_performance(market=market, min_days_old=5, limit=limit)
    
    if not signals:
        if verbose:
            print(f"✅ 没有需要计算的 {market} 信号")
        return 0, 0, 0
    
    if verbose:
        print(f"📊 找到 {len(signals)} 个需要计算的 {market} 信号")
    
    processed = 0
    success = 0
    failed = 0
    
    for i, signal in enumerate(signals):
        symbol = signal['symbol']
        scan_date = signal['scan_date']
        price = signal.get('price')
        
        if verbose and i % 20 == 0:
            print(f"  处理进度: {i}/{len(signals)}...")
        
        # 计算前向收益
        result = compute_forward_returns(symbol, scan_date, price)
        
        if result and result.get('return_5d') is not None:
            # 保存到数据库
            upsert_signal_performance(
                symbol=symbol,
                scan_date=scan_date,
                market=market,
                return_5d=result.get('return_5d'),
                return_10d=result.get('return_10d'),
                return_20d=result.get('return_20d'),
                max_gain=result.get('max_gain'),
                max_drawdown=result.get('max_drawdown')
            )
            success += 1
        else:
            failed += 1
        
        processed += 1
    
    if verbose:
        print(f"✅ 处理完成: {processed} 个信号, {success} 成功, {failed} 失败")
        
        # 显示统计
        stats = get_performance_stats(market)
        if stats.get('total', 0) > 0:
            print(f"📈 {market} 市场性能缓存: 共 {stats['total']} 条")
            print(f"   平均收益: 5d={stats.get('avg_5d', 0):.2%}, 10d={stats.get('avg_10d', 0):.2%}, 20d={stats.get('avg_20d', 0):.2%}")
    
    return processed, success, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算信号前向收益')
    parser.add_argument('--market', type=str, default='US', help='市场 (US/CN)')
    parser.add_argument('--limit', type=int, default=100, help='最多处理数量')
    parser.add_argument('--all', action='store_true', help='处理所有市场')
    
    args = parser.parse_args()
    
    if args.all:
        print("=" * 50)
        print("🇺🇸 计算 US 市场信号...")
        batch_compute_performance('US', args.limit)
        
        print("\n" + "=" * 50)
        print("🇨🇳 计算 CN 市场信号...")
        batch_compute_performance('CN', args.limit)
    else:
        batch_compute_performance(args.market, args.limit)
