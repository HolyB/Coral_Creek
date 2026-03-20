#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略表现追踪器
追踪每个策略的历史表现
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


def get_supabase():
    """获取 Supabase 客户端"""
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key or not SUPABASE_AVAILABLE:
        return None
    try:
        return create_client(url, key)
    except:
        return None


def calculate_strategy_performance(strategy_name: str, days_back: int = 30, market: str = 'US') -> Dict:
    """计算策略历史表现
    
    基于历史信号计算策略如果被执行会有什么表现
    """
    from strategies.decision_system import get_strategy_manager
    from db.database import query_scan_results
    
    manager = get_strategy_manager()
    strategy = manager.get_strategy(strategy_name)
    
    if not strategy:
        return {'error': f'Strategy {strategy_name} not found'}
    
    # 获取历史数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 20)  # 多取20天用于计算收益
    
    results = query_scan_results(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        market=market,
        limit=5000
    )
    
    if not results:
        return {'error': 'No data available'}
    
    df = pd.DataFrame(results)
    
    # 按日期分组，模拟每天的选股
    dates = df['scan_date'].unique()
    dates = sorted(dates, reverse=True)[:days_back]
    
    total_picks = 0
    win_count = 0
    returns_5d = []
    returns_10d = []
    
    for pick_date in dates:
        day_df = df[df['scan_date'] == pick_date]
        picks = strategy.select(day_df, top_n=3)
        
        for pick in picks:
            total_picks += 1
            
            # 查找该股票后续表现 (简化: 使用下一个扫描日的价格)
            future_data = df[
                (df['symbol'] == pick.symbol) & 
                (df['scan_date'] > pick_date)
            ].sort_values('scan_date')
            
            if len(future_data) >= 1:
                # 计算5日后收益 (使用第一个可用日期近似)
                future_price = future_data.iloc[0]['price']
                if pick.entry_price > 0:
                    ret = (future_price - pick.entry_price) / pick.entry_price * 100
                    returns_5d.append(ret)
                    if ret > 0:
                        win_count += 1
    
    # 计算统计
    performance = {
        'strategy': strategy_name,
        'strategy_name': strategy.name,
        'total_picks': total_picks,
        'win_count': win_count,
        'win_rate': round(win_count / total_picks * 100, 1) if total_picks > 0 else 0,
        'avg_return': round(sum(returns_5d) / len(returns_5d), 2) if returns_5d else 0,
        'max_gain': round(max(returns_5d), 2) if returns_5d else 0,
        'max_loss': round(min(returns_5d), 2) if returns_5d else 0,
        'days_analyzed': len(dates),
        'market': market
    }
    
    return performance


def get_all_strategy_performance(days_back: int = 30, market: str = 'US') -> List[Dict]:
    """获取所有策略的表现"""
    from strategies.decision_system import get_strategy_manager
    
    manager = get_strategy_manager()
    strategies = manager.list_strategies()
    
    results = []
    for s in strategies:
        try:
            perf = calculate_strategy_performance(s['key'], days_back, market)
            perf['icon'] = s['icon']
            results.append(perf)
        except Exception as e:
            results.append({
                'strategy': s['key'],
                'strategy_name': s['name'],
                'icon': s['icon'],
                'error': str(e)
            })
    
    return results


def save_strategy_pick(strategy_name: str, symbol: str, pick_date: str, 
                       entry_price: float, stop_loss: float, take_profit: float,
                       market: str = 'US') -> bool:
    """保存策略选股记录到数据库 (用于后续追踪)"""
    supabase = get_supabase()
    if not supabase:
        return False
    
    try:
        # 需要先在 Supabase 创建 strategy_picks 表
        record = {
            'strategy': strategy_name,
            'symbol': symbol,
            'pick_date': pick_date,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market': market,
            'status': 'active'
        }
        supabase.table('strategy_picks').insert(record).execute()
        return True
    except Exception as e:
        print(f"Error saving pick: {e}")
        return False
