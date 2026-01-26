#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
博主推荐追踪服务层
计算推荐收益、博主业绩统计
"""
import os
from datetime import datetime, timedelta
from db.database import (
    get_recommendations, get_all_bloggers, get_blogger_stats,
    add_blogger, add_recommendation
)


def get_stock_price_on_date(ticker, target_date, market='CN'):
    """获取指定日期的股票价格"""
    from data_fetcher import get_cn_stock_data, get_us_stock_data
    
    try:
        if market == 'CN':
            df = get_cn_stock_data(ticker, days=30)
        else:
            df = get_us_stock_data(ticker, days=30)
        
        if df is None or df.empty:
            return None
        
        # 找到目标日期或最近的交易日
        df['Date'] = df.index if hasattr(df.index, 'date') else df.index
        df = df.reset_index(drop=True)
        
        target_dt = datetime.strptime(target_date, '%Y-%m-%d') if isinstance(target_date, str) else target_date
        
        for i, row in df.iterrows():
            row_date = row.name if hasattr(row.name, 'date') else row.get('Date')
            if row_date:
                if hasattr(row_date, 'date'):
                    row_date = row_date.date()
                if str(row_date) <= str(target_dt.date()):
                    return row['Close']
        
        return df.iloc[-1]['Close'] if len(df) > 0 else None
        
    except Exception as e:
        print(f"Error getting price for {ticker} on {target_date}: {e}")
        return None


def get_current_price(ticker, market='CN'):
    """获取当前价格"""
    from data_fetcher import get_cn_stock_data, get_us_stock_data
    
    try:
        if market == 'CN':
            df = get_cn_stock_data(ticker, days=5)
        else:
            df = get_us_stock_data(ticker, days=5)
        
        if df is None or df.empty:
            return None
        
        return df.iloc[-1]['Close']
        
    except Exception as e:
        print(f"Error getting current price for {ticker}: {e}")
        return None


def calculate_recommendation_return(rec):
    """计算单条推荐的收益率
    
    Args:
        rec: 推荐记录字典
    
    Returns:
        dict with return info
    """
    ticker = rec['ticker']
    market = rec.get('market', 'CN')
    rec_date = rec['rec_date']
    rec_price = rec.get('rec_price')
    
    # 如果没有推荐价格，尝试获取
    if not rec_price:
        rec_price = get_stock_price_on_date(ticker, rec_date, market)
    
    if not rec_price:
        return {'return_pct': None, 'current_price': None, 'days_held': None}
    
    current_price = get_current_price(ticker, market)
    if not current_price:
        return {'return_pct': None, 'current_price': None, 'days_held': None}
    
    return_pct = (current_price - rec_price) / rec_price * 100
    
    rec_dt = datetime.strptime(rec_date, '%Y-%m-%d') if isinstance(rec_date, str) else rec_date
    days_held = (datetime.now() - rec_dt).days
    
    return {
        'return_pct': round(return_pct, 2),
        'rec_price': rec_price,
        'current_price': round(current_price, 2),
        'days_held': days_held
    }


def get_recommendations_with_returns(blogger_id=None, market=None, limit=50):
    """获取带收益计算的推荐列表"""
    recs = get_recommendations(blogger_id=blogger_id, market=market, limit=limit)
    
    for rec in recs:
        returns = calculate_recommendation_return(rec)
        rec.update(returns)
    
    return recs


def get_blogger_performance(blogger_id=None):
    """获取博主业绩统计
    
    Returns:
        list of dicts with: name, rec_count, win_rate, avg_return, total_return
    """
    bloggers = get_all_bloggers()
    
    results = []
    for blogger in bloggers:
        if blogger_id and blogger['id'] != blogger_id:
            continue
        
        recs = get_recommendations(blogger_id=blogger['id'], limit=100)
        
        if not recs:
            results.append({
                'id': blogger['id'],
                'name': blogger['name'],
                'platform': blogger.get('platform'),
                'rec_count': 0,
                'win_count': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            })
            continue
        
        returns = []
        win_count = 0
        
        for rec in recs:
            ret_info = calculate_recommendation_return(rec)
            if ret_info['return_pct'] is not None:
                returns.append(ret_info['return_pct'])
                if ret_info['return_pct'] > 0:
                    win_count += 1
        
        avg_return = sum(returns) / len(returns) if returns else 0
        win_rate = win_count / len(returns) * 100 if returns else 0
        
        results.append({
            'id': blogger['id'],
            'name': blogger['name'],
            'platform': blogger.get('platform'),
            'rec_count': len(recs),
            'calculated_count': len(returns),
            'win_count': win_count,
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 2),
            'total_return': round(sum(returns), 2)
        })
    
    # 按平均收益排序
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    return results


if __name__ == "__main__":
    # 测试
    print("Testing blogger service...")
    
    from db.database import init_blogger_tables
    init_blogger_tables()
    
    # 添加测试博主
    # blogger_id = add_blogger("测试博主", platform="雪球", specialty="A股")
    # print(f"Added blogger: {blogger_id}")
    
    print("Blogger service ready!")
