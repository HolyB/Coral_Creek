#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据获取模块 - 用于获取股票历史价格数据"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def get_us_stock_data(symbol, days=365):
    """获取美股历史数据（使用Polygon API）"""
    try:
        from polygon import RESTClient
        
        api_key = os.getenv('POLYGON_API_KEY', 'qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D')
        client = RESTClient(api_key)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        aggs = []
        try:
            aggs_iter = client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                limit=50000
            )
            
            for a in aggs_iter:
                aggs.append({
                    'Date': pd.Timestamp.fromtimestamp(a.timestamp/1000),
                    'Open': a.open,
                    'High': a.high,
                    'Low': a.low,
                    'Close': a.close,
                    'Volume': a.volume,
                })
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
        
        if not aggs:
            return None
        
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    except Exception as e:
        print(f"Error in get_us_stock_data: {e}")
        return None

def get_cn_stock_data(symbol, days=365):
    """获取A股历史数据（使用Tushare API）"""
    try:
        import tushare as ts
        
        pro = ts.pro_api()
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        # 转换symbol格式（如果是000001.SZ格式，需要提取代码）
        if '.' in symbol:
            ts_code = symbol
        else:
            # 需要判断是上海还是深圳
            if symbol.startswith('6'):
                ts_code = f"{symbol}.SH"
            elif symbol.startswith(('0', '3')):
                ts_code = f"{symbol}.SZ"
            elif symbol.startswith('8') or symbol.startswith('4'):
                ts_code = f"{symbol}.BJ"
            else:
                ts_code = symbol
        
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                return None
            
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.set_index('trade_date', inplace=True)
            df.sort_index(inplace=True)
            
            # 重命名列以匹配标准格式
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume'
            }, inplace=True)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"Error fetching Tushare data for {symbol}: {e}")
            return None
    except Exception as e:
        print(f"Error in get_cn_stock_data: {e}")
        return None

def get_stock_data(symbol, market='US', days=365):
    """通用函数：根据市场类型获取股票数据"""
    if market == 'US':
        return get_us_stock_data(symbol, days)
    elif market == 'CN':
        return get_cn_stock_data(symbol, days)
    else:
        return None


