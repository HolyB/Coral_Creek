#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略回测追踪系统
追踪每个策略选股后的表现
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def get_stock_price_after_pick(symbol: str, pick_date: str, days: int = 10, market: str = 'US') -> Optional[pd.DataFrame]:
    """获取选股后N天的价格数据
    
    Returns:
        DataFrame with columns: date, close, change_pct, cumulative_pct
    """
    try:
        if market == 'US':
            from data_fetcher import get_us_stock_data
            df = get_us_stock_data(symbol, days=days + 30)  # 多取一些确保覆盖
        else:
            from data_fetcher import get_cn_stock_data
            df = get_cn_stock_data(symbol, days=days + 30)
        
        if df is None or df.empty:
            return None
        
        # 确保日期格式
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            df['date'] = pd.to_datetime(df.index)
        
        # 筛选 pick_date 之后的数据
        pick_dt = pd.to_datetime(pick_date)
        df = df[df['date'] >= pick_dt].copy()
        
        if len(df) < 2:
            return None
        
        # 取前 N 天
        df = df.head(days + 1)  # +1 包含买入日
        
        # 计算涨跌幅
        close_col = 'Close' if 'Close' in df.columns else 'close'
        entry_price = df[close_col].iloc[0]
        
        df['close'] = df[close_col]
        df['change_pct'] = df[close_col].pct_change() * 100
        df['cumulative_pct'] = (df[close_col] / entry_price - 1) * 100
        df['day'] = range(len(df))
        
        return df[['date', 'close', 'change_pct', 'cumulative_pct', 'day']].reset_index(drop=True)
        
    except Exception as e:
        print(f"Error getting price data for {symbol}: {e}")
        return None


def backtest_strategy_picks(picks: List[Dict], days: int = 10, market: str = 'US') -> Dict:
    """回测策略选股结果
    
    Args:
        picks: 选股列表 [{'symbol': 'AAPL', 'entry_price': 150, 'pick_date': '2024-01-15'}, ...]
        days: 追踪天数
        market: 市场
    
    Returns:
        回测结果统计
    """
    results = []
    
    for pick in picks:
        symbol = pick.get('symbol')
        pick_date = pick.get('pick_date') or pick.get('scan_date')
        entry_price = pick.get('entry_price') or pick.get('price')
        
        if not symbol or not pick_date:
            continue
        
        price_data = get_stock_price_after_pick(symbol, pick_date, days, market)
        
        if price_data is None or len(price_data) < 2:
            results.append({
                'symbol': symbol,
                'pick_date': pick_date,
                'entry_price': entry_price,
                'status': 'no_data',
                'daily_returns': [],
                'final_return': None
            })
            continue
        
        # 计算每日收益
        daily_returns = price_data['change_pct'].tolist()[1:]  # 跳过第一天
        cumulative = price_data['cumulative_pct'].tolist()
        final_return = cumulative[-1] if cumulative else 0
        
        # 判断盈亏
        max_gain = max(cumulative) if cumulative else 0
        max_loss = min(cumulative) if cumulative else 0
        
        results.append({
            'symbol': symbol,
            'pick_date': pick_date,
            'entry_price': entry_price,
            'status': 'win' if final_return > 0 else 'loss',
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative,
            'final_return': final_return,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'holding_days': len(daily_returns),
            'price_data': price_data
        })
    
    # 汇总统计
    valid_results = [r for r in results if r['status'] != 'no_data']
    
    if not valid_results:
        return {
            'picks': results,
            'summary': {
                'total': len(results),
                'valid': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            }
        }
    
    win_count = len([r for r in valid_results if r['final_return'] > 0])
    loss_count = len([r for r in valid_results if r['final_return'] <= 0])
    avg_return = sum([r['final_return'] for r in valid_results]) / len(valid_results)
    
    return {
        'picks': results,
        'summary': {
            'total': len(results),
            'valid': len(valid_results),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_count / len(valid_results) * 100 if valid_results else 0,
            'avg_return': avg_return,
            'total_return': sum([r['final_return'] for r in valid_results]),
            'max_gain': max([r['max_gain'] for r in valid_results]) if valid_results else 0,
            'max_loss': min([r['max_loss'] for r in valid_results]) if valid_results else 0
        }
    }


def get_historical_picks(strategy_name: str, days_back: int = 30, market: str = 'US') -> List[Dict]:
    """获取策略历史选股记录
    
    从数据库中模拟获取历史选股
    (实际使用时需要保存选股记录到数据库)
    """
    from db.database import query_scan_results, get_scanned_dates
    from strategies.decision_system import get_strategy_manager
    
    dates = get_scanned_dates(market=market)
    if not dates:
        return []
    
    # 取最近 days_back 天
    manager = get_strategy_manager()
    strategy = manager.strategies.get(strategy_name)
    
    if not strategy:
        return []
    
    picks = []
    
    for scan_date in dates[:days_back]:
        results = query_scan_results(scan_date=scan_date, market=market, limit=200)
        if not results:
            continue
        
        df = pd.DataFrame(results)
        strategy_picks = strategy.select(df, top_n=3)
        
        for p in strategy_picks:
            picks.append({
                'symbol': p.symbol,
                'pick_date': scan_date,
                'entry_price': p.entry_price,
                'strategy': strategy_name,
                'score': p.score
            })
    
    return picks


def create_performance_chart(backtest_result: Dict, symbol: str = None) -> dict:
    """创建回测表现图表数据
    
    Returns Plotly figure data
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    picks = backtest_result.get('picks', [])
    
    if symbol:
        picks = [p for p in picks if p['symbol'] == symbol]
    
    if not picks:
        return None
    
    pick = picks[0]
    price_data = pick.get('price_data')
    
    if price_data is None:
        return None
    
    # 创建图表
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'{pick["symbol"]} 走势', '累计收益率']
    )
    
    # 价格走势
    fig.add_trace(
        go.Scatter(
            x=price_data['date'],
            y=price_data['close'],
            mode='lines+markers',
            name='价格',
            line=dict(color='#58a6ff', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # 标注买入点
    fig.add_trace(
        go.Scatter(
            x=[price_data['date'].iloc[0]],
            y=[price_data['close'].iloc[0]],
            mode='markers+text',
            name='买入点',
            marker=dict(size=15, color='#3fb950', symbol='triangle-up'),
            text=['买入'],
            textposition='top center'
        ),
        row=1, col=1
    )
    
    # 累计收益率
    colors = ['#3fb950' if v >= 0 else '#f85149' for v in price_data['cumulative_pct']]
    fig.add_trace(
        go.Bar(
            x=price_data['date'],
            y=price_data['cumulative_pct'],
            name='累计收益',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def format_daily_returns_table(backtest_result: Dict) -> pd.DataFrame:
    """格式化每日收益表格"""
    rows = []
    
    for pick in backtest_result.get('picks', []):
        if pick['status'] == 'no_data':
            continue
        
        price_data = pick.get('price_data')
        if price_data is None:
            continue
        
        for _, row in price_data.iterrows():
            if row['day'] == 0:
                continue  # 跳过买入日
            
            rows.append({
                '股票': pick['symbol'],
                '日期': row['date'].strftime('%m-%d'),
                '收盘价': f"${row['close']:.2f}",
                '当日涨跌': f"{row['change_pct']:+.2f}%",
                '累计收益': f"{row['cumulative_pct']:+.2f}%",
                '持仓天数': int(row['day'])
            })
    
    return pd.DataFrame(rows)
