#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据获取模块 - 用于获取股票历史价格数据"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    # 尝试从多个位置加载 .env
    env_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),  # versions/v2/.env
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # 项目根目录/.env
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def _get_polygon_api_key():
    """获取 Polygon API Key，如果未设置则抛出错误"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError(
            "POLYGON_API_KEY not set! Please either:\n"
            "1. Create a .env file with POLYGON_API_KEY=your_key\n"
            "2. Set the environment variable: export POLYGON_API_KEY=your_key\n"
            "Get your key at: https://polygon.io/dashboard/api-keys"
        )
    return api_key

def get_all_us_tickers():
    """从Polygon API获取所有活跃美股代码"""
    try:
        from polygon import RESTClient
        api_key = _get_polygon_api_key()
        client = RESTClient(api_key)
        
        print("Fetching all active US tickers from Polygon...")
        tickers = []
        
        # 只获取主要交易所的股票 (CS=Common Stock, ADR, ETF)
        # 排除 Warrant, Unit, Rights 等
        # Market = stocks
        
        # Polygon API 分页获取
        for t in client.list_tickers(market="stocks", active=True, limit=1000):
            # 简单过滤: 排除名字太长的(通常是权证)和带特殊符号的
            if len(t.ticker) <= 5: 
                tickers.append(t.ticker)
                
        print(f"Fetched {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        # Fallback to local cache if API fails
        cache_path = os.path.join(os.path.dirname(__file__), 'us_tickers.txt')
        if os.path.exists(cache_path):
            print("Fallback: Using local us_tickers.txt")
            with open(cache_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []

def get_all_cn_tickers():
    """从Tushare API获取所有A股股票代码"""
    try:
        import tushare as ts
        
        token = os.getenv('TUSHARE_TOKEN')
        if token:
            ts.set_token(token)
        
        pro = ts.pro_api()
        
        print("Fetching all active CN A-share tickers from Tushare...")
        
        # 获取上市状态为正常的股票
        df = pro.stock_basic(
            exchange='', 
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        
        if df is not None and not df.empty:
            tickers = df['ts_code'].tolist()
            print(f"Fetched {len(tickers)} CN tickers.")
            return tickers
        return []
    except Exception as e:
        print(f"Error fetching CN tickers: {e}")
        return []


# 缓存 A 股基本信息
_cn_stock_info_cache = {}

def get_cn_ticker_details(symbol):
    """获取A股股票详细信息（名称、行业等）"""
    global _cn_stock_info_cache
    
    # 如果缓存为空，加载所有股票基本信息
    if not _cn_stock_info_cache:
        try:
            import tushare as ts
            
            token = os.getenv('TUSHARE_TOKEN')
            if token:
                ts.set_token(token)
            
            pro = ts.pro_api()
            
            # 获取所有股票基本信息
            df = pro.stock_basic(
                exchange='', 
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    _cn_stock_info_cache[row['ts_code']] = {
                        'name': row.get('name', ''),
                        'industry': row.get('industry', ''),
                        'area': row.get('area', ''),
                        'market': row.get('market', ''),
                        'list_date': row.get('list_date', '')
                    }
        except Exception as e:
            print(f"Error loading CN stock info cache: {e}")
    
    # 从缓存返回
    if symbol in _cn_stock_info_cache:
        info = _cn_stock_info_cache[symbol]
        return {
            'market_cap': None,  # Tushare 基础版不提供市值
            'shares_outstanding': None,
            'name': info.get('name', symbol),
            'sic_description': info.get('industry', ''),
        }
    
    return None


def get_us_stock_data(symbol, days=365):
    """获取美股历史数据（使用Polygon API）"""
    try:
        from polygon import RESTClient
        
        api_key = _get_polygon_api_key()
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
            # print(f"Error fetching data for {symbol}: {e}")
            return None
        
        if not aggs:
            return None
        
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    except Exception as e:
        # print(f"Error in get_us_stock_data: {e}")
        return None

def get_ticker_details(symbol):
    """获取股票详细信息（市值、流通股本等）"""
    try:
        from polygon import RESTClient
        api_key = _get_polygon_api_key()
        client = RESTClient(api_key)
        
        # 调用 Ticker Details v3
        details = client.get_ticker_details(symbol)
        
        market_cap = getattr(details, 'market_cap', 0)
        shares_outstanding = getattr(details, 'weighted_shares_outstanding', 0)
        
        # 如果 market_cap 为 0 但有 shares_outstanding，可以估算
        # 但这里我们只返回原始数据
        
        return {
            'market_cap': market_cap,
            'shares_outstanding': shares_outstanding,
            'name': getattr(details, 'name', symbol),
            'sic_description': getattr(details, 'sic_description', ''),
            'ticker_root': getattr(details, 'ticker_root', symbol)
        }
    except Exception as e:
        # print(f"Error fetching details for {symbol}: {e}")
        return None

# Tushare API 调用计数器 (限流)
_tushare_call_count = 0
_tushare_call_limit = 450  # 每450次暂停一下 (Tushare 限制约500次/分钟)
_tushare_pause_seconds = 60

def get_cn_stock_data(symbol, days=365):
    """获取A股历史数据（使用Tushare API）"""
    global _tushare_call_count
    
    # 限流：每450次调用暂停60秒
    _tushare_call_count += 1
    if _tushare_call_count % _tushare_call_limit == 0:
        import time
        print(f"⏸️ Tushare rate limit: pausing for {_tushare_pause_seconds}s after {_tushare_call_count} calls...")
        time.sleep(_tushare_pause_seconds)
    
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


def get_cn_index_data(symbol, days=365):
    """获取A股指数历史数据（使用Tushare index_daily API）
    
    Args:
        symbol: 指数代码，如 '000001.SH' (上证指数), '399001.SZ' (深证成指)
        days: 获取天数
    """
    try:
        import tushare as ts
        
        token = os.getenv('TUSHARE_TOKEN')
        if token:
            ts.set_token(token)
        
        pro = ts.pro_api()
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        try:
            # 使用 index_daily API 获取指数数据
            df = pro.index_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
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
            print(f"Error fetching Tushare index data for {symbol}: {e}")
            return None
    except Exception as e:
        print(f"Error in get_cn_index_data: {e}")
        return None


def get_stock_data(symbol, market='US', days=365):
    """通用函数：根据市场类型获取股票数据"""
    if market == 'US':
        return get_us_stock_data(symbol, days)
    elif market == 'CN':
        return get_cn_stock_data(symbol, days)
    else:
        return None


# ==================== 板块/行业数据 ====================

def get_cn_sector_data():
    """获取A股行业板块涨跌数据
    
    通过聚合股票数据计算各行业表现
    
    Returns:
        DataFrame with columns: sector, name, change_pct, stock_count
    """
    import tushare as ts
    
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("TUSHARE_TOKEN not found")
        return None
    
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 获取所有股票的行业分类
        stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        if stocks is None or stocks.empty:
            print("Failed to get stock basic info")
            return None
        
        # 获取今日行情统计 (使用daily_basic获取更多股票)
        today = datetime.now()
        trade_date = today.strftime('%Y%m%d')
        
        # 尝试获取最近交易日的数据
        for days_back in range(5):
            try:
                check_date = (today - timedelta(days=days_back)).strftime('%Y%m%d')
                daily = pro.daily(trade_date=check_date, fields='ts_code,pct_chg,amount')
                if daily is not None and len(daily) > 100:
                    trade_date = check_date
                    break
            except:
                continue
        
        if daily is None or daily.empty:
            print("Failed to get daily data")
            return None
        
        # 合并行业信息
        merged = pd.merge(daily, stocks[['ts_code', 'industry']], on='ts_code', how='left')
        merged = merged.dropna(subset=['industry'])
        
        # 按行业聚合
        sector_stats = merged.groupby('industry').agg({
            'pct_chg': 'mean',  # 平均涨跌幅
            'amount': 'sum',    # 总成交额
            'ts_code': 'count'  # 股票数量
        }).reset_index()
        
        sector_stats.columns = ['name', 'change_pct', 'amount', 'stock_count']
        sector_stats['sector'] = sector_stats['name']  # 用名称作为code
        sector_stats['amount'] = sector_stats['amount'] / 100000  # 转为亿
        
        sector_stats = sector_stats.sort_values('change_pct', ascending=False)
        
        return sector_stats[['sector', 'name', 'change_pct', 'amount', 'stock_count']]
        
    except Exception as e:
        print(f"Error fetching CN sector data: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_us_sector_data():
    """获取美股行业板块涨跌数据
    
    通过聚合股票数据计算各行业表现
    
    Returns:
        DataFrame with columns: sector, name, change_pct, stock_count, top_stocks
    """
    # 使用 Polygon 的 grouped daily API 获取所有股票，然后按行业聚合
    api_key = _get_polygon_api_key()
    if not api_key:
        return None
    
    try:
        import requests
        
        # 获取昨日所有股票行情
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 使用 grouped daily API (需要更高级API权限)
        # 简化版：使用预定义的行业ETF代替
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
        
        sector_data = []
        
        for etf, sector_name in sector_etfs.items():
            try:
                # 获取ETF最近行情
                df = get_us_stock_data(etf, days=5)
                if df is not None and len(df) >= 2:
                    today_close = df.iloc[-1]['Close']
                    prev_close = df.iloc[-2]['Close']
                    change_pct = (today_close - prev_close) / prev_close * 100
                    
                    sector_data.append({
                        'sector': etf,
                        'name': sector_name,
                        'close': today_close,
                        'change_pct': change_pct,
                        'volume': df.iloc[-1].get('Volume', 0)
                    })
            except Exception as e:
                continue
        
        if not sector_data:
            return None
        
        result = pd.DataFrame(sector_data)
        result = result.sort_values('change_pct', ascending=False)
        return result
        
    except Exception as e:
        print(f"Error fetching US sector data: {e}")
        return None


def get_sector_data(market='US'):
    """通用函数：获取板块数据"""
    if market == 'US':
        return get_us_sector_data()
    elif market == 'CN':
        return get_cn_sector_data()
    else:
        return None
