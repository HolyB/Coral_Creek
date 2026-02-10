#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
博主推荐追踪服务层
计算推荐收益、博主业绩统计
"""
from datetime import datetime, timedelta
from db.database import (
    get_recommendations, get_all_bloggers
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


def _fetch_price_df(ticker, market='CN', days=240):
    """获取价格序列，统一成按日期升序的 DataFrame"""
    from data_fetcher import get_cn_stock_data, get_us_stock_data
    try:
        if market == 'CN':
            df = get_cn_stock_data(ticker, days=days)
        else:
            df = get_us_stock_data(ticker, days=days)
        if df is None or df.empty or 'Close' not in df.columns:
            return None
        out = df.copy()
        out = out.sort_index()
        return out
    except Exception as e:
        print(f"Error fetching df for {ticker}: {e}")
        return None


def _price_on_or_after(df, target_dt):
    if df is None or df.empty:
        return None
    for idx, row in df.iterrows():
        if hasattr(idx, 'to_pydatetime'):
            row_dt = idx.to_pydatetime()
        else:
            row_dt = idx
        if str(row_dt.date()) >= str(target_dt.date()):
            return float(row.get('Close'))
    return float(df.iloc[-1]['Close']) if len(df) > 0 else None


def _price_at_horizon(df, start_dt, horizon_days=10):
    target = start_dt + timedelta(days=int(horizon_days))
    return _price_on_or_after(df, target)


def calculate_recommendation_return(rec, horizon_days=10, price_cache=None):
    """计算单条推荐收益 + 买卖点有效性评估"""
    ticker = rec['ticker']
    market = rec.get('market', 'CN')
    rec_type = str(rec.get('rec_type', 'BUY') or 'BUY').upper()
    rec_date = rec['rec_date']
    rec_dt = datetime.strptime(rec_date, '%Y-%m-%d') if isinstance(rec_date, str) else rec_date

    cache_key = f"{market}:{ticker}"
    if isinstance(price_cache, dict) and cache_key in price_cache:
        df = price_cache[cache_key]
    else:
        df = _fetch_price_df(ticker, market=market, days=max(120, int(horizon_days) + 90))
        if isinstance(price_cache, dict):
            price_cache[cache_key] = df

    if df is None or df.empty:
        return {'return_pct': None, 'current_price': None, 'days_held': None, 'direction_ok': None}

    entry_price = rec.get('rec_price') or _price_on_or_after(df, rec_dt)
    if not entry_price:
        return {'return_pct': None, 'current_price': None, 'days_held': None, 'direction_ok': None}
    entry_price = float(entry_price)

    current_price = float(df.iloc[-1]['Close'])
    horizon_price = _price_at_horizon(df, rec_dt, horizon_days=horizon_days)
    if horizon_price is None:
        horizon_price = current_price
    horizon_price = float(horizon_price)

    # 从推荐日开始的路径表现
    path_df = df[df.index >= rec_dt]
    if path_df is None or path_df.empty:
        path_df = df.tail(1)
    max_close = float(path_df['Close'].max())
    min_close = float(path_df['Close'].min())

    raw_now_ret = (current_price - entry_price) / entry_price * 100.0
    raw_h_ret = (horizon_price - entry_price) / entry_price * 100.0

    if rec_type == 'SELL':
        directional_now_ret = -raw_now_ret
        directional_h_ret = -raw_h_ret
        mfe = (entry_price - min_close) / entry_price * 100.0
        mae = (max_close - entry_price) / entry_price * 100.0
        target_hit = bool(rec.get('target_price')) and min_close <= float(rec.get('target_price'))
        stop_hit = bool(rec.get('stop_loss')) and max_close >= float(rec.get('stop_loss'))
    else:
        # BUY/HOLD 默认按多头方向
        directional_now_ret = raw_now_ret
        directional_h_ret = raw_h_ret
        mfe = (max_close - entry_price) / entry_price * 100.0
        mae = (entry_price - min_close) / entry_price * 100.0
        target_hit = bool(rec.get('target_price')) and max_close >= float(rec.get('target_price'))
        stop_hit = bool(rec.get('stop_loss')) and min_close <= float(rec.get('stop_loss'))

    direction_ok = directional_h_ret > 0
    days_held = (datetime.now() - rec_dt).days

    return {
        'return_pct': round(raw_now_ret, 2),  # 保持原语义：当前价格相对推荐价收益
        'rec_price': round(entry_price, 4),
        'current_price': round(current_price, 2),
        'days_held': days_held,
        'horizon_days': int(horizon_days),
        'horizon_return_pct': round(raw_h_ret, 2),
        'directional_return_pct': round(directional_h_ret, 2),
        'direction_ok': bool(direction_ok),
        'mfe_pct': round(mfe, 2),
        'mae_pct': round(mae, 2),
        'target_hit': bool(target_hit),
        'stop_hit': bool(stop_hit),
    }


def get_recommendations_with_returns(blogger_id=None, market=None, portfolio_tag=None, limit=50, horizon_days=10):
    """获取带收益计算的推荐列表"""
    recs = get_recommendations(
        blogger_id=blogger_id,
        market=market,
        portfolio_tag=portfolio_tag,
        limit=limit,
    )
    price_cache = {}
    for rec in recs:
        returns = calculate_recommendation_return(rec, horizon_days=horizon_days, price_cache=price_cache)
        rec.update(returns)
    return recs


def get_blogger_performance(blogger_id=None, horizon_days=10, portfolio_tag=None):
    """获取博主业绩统计
    
    Returns:
        list of dicts with: name, rec_count, win_rate, avg_return, total_return
    """
    bloggers = get_all_bloggers()
    
    results = []
    for blogger in bloggers:
        if blogger_id and blogger['id'] != blogger_id:
            continue
        
        recs = get_recommendations(
            blogger_id=blogger['id'],
            portfolio_tag=portfolio_tag,
            limit=200,
        )
        
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
        
        raw_returns = []
        directional_returns = []
        win_count = 0
        price_cache = {}
        for rec in recs:
            ret_info = calculate_recommendation_return(rec, horizon_days=horizon_days, price_cache=price_cache)
            if ret_info['return_pct'] is not None and ret_info.get('directional_return_pct') is not None:
                raw_returns.append(ret_info['return_pct'])
                directional_returns.append(ret_info['directional_return_pct'])
                if ret_info['direction_ok']:
                    win_count += 1

        avg_return = sum(raw_returns) / len(raw_returns) if raw_returns else 0
        avg_directional = sum(directional_returns) / len(directional_returns) if directional_returns else 0
        win_rate = win_count / len(directional_returns) * 100 if directional_returns else 0

        results.append({
            'id': blogger['id'],
            'name': blogger['name'],
            'platform': blogger.get('platform'),
            'rec_count': len(recs),
            'calculated_count': len(directional_returns),
            'win_count': win_count,
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 2),
            'avg_directional_return': round(avg_directional, 2),
            'total_return': round(sum(raw_returns), 2)
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
