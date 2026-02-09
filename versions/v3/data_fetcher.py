#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据获取模块 - 用于获取股票历史价格数据"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from collections import deque


# 最近数据抓取错误（用于UI诊断）
_RECENT_FETCH_ERRORS = deque(maxlen=50)

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


def _record_fetch_error(symbol: str, market: str, message: str):
    _RECENT_FETCH_ERRORS.appendleft({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": (symbol or "").upper(),
        "market": (market or "").upper(),
        "message": message[:220],
    })


def get_recent_fetch_errors(limit: int = 20):
    """获取最近行情抓取错误，供页面调试展示"""
    if limit <= 0:
        return []
    return list(_RECENT_FETCH_ERRORS)[:limit]

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

# A股市值缓存 (全局)
_cn_market_cap_cache = {}
_cn_market_cap_loaded = False

def _load_cn_market_cap_cache():
    """预加载A股市值数据 (使用 AkShare)"""
    global _cn_market_cap_cache, _cn_market_cap_loaded
    
    if _cn_market_cap_loaded:
        return
    
    try:
        import akshare as ak
        print("Loading A-share market cap data from AkShare...")
        spot_df = ak.stock_zh_a_spot_em()
        
        for _, row in spot_df.iterrows():
            code = str(row.get('代码', ''))
            _cn_market_cap_cache[code] = {
                'market_cap': row.get('总市值', 0) or 0,
                'name': row.get('名称', ''),
                'industry': '',  # AkShare 实时数据不包含行业
            }
        
        _cn_market_cap_loaded = True
        print(f"Loaded {len(_cn_market_cap_cache)} A-share market caps")
    except Exception as e:
        print(f"Failed to load A-share market cap: {e}")


def get_cn_ticker_details(symbol):
    """获取A股股票详细信息（名称、行业、市值等）"""
    global _cn_stock_info_cache, _cn_market_cap_cache
    
    # 如果基本信息缓存为空，加载所有股票基本信息
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
    
    # 提取股票代码 (去掉后缀)
    code = symbol.split('.')[0] if '.' in symbol else symbol
    
    # 获取基本信息
    info = _cn_stock_info_cache.get(symbol, {})
    name = info.get('name', symbol)
    industry = info.get('industry', '')
    
    # 获取市值 (优先从缓存，否则尝试加载)
    market_cap = 0
    if code in _cn_market_cap_cache:
        mc_info = _cn_market_cap_cache[code]
        market_cap = mc_info.get('market_cap', 0)
        if not name or name == symbol:
            name = mc_info.get('name', symbol)
    else:
        # 尝试加载市值缓存
        if not _cn_market_cap_loaded:
            _load_cn_market_cap_cache()
            if code in _cn_market_cap_cache:
                mc_info = _cn_market_cap_cache[code]
                market_cap = mc_info.get('market_cap', 0)
                if not name or name == symbol:
                    name = mc_info.get('name', symbol)
    
    return {
        'market_cap': market_cap,
        'shares_outstanding': None,
        'name': name,
        'sic_description': industry,
    }


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
            _record_fetch_error(symbol, "US", f"Polygon list_aggs failed: {str(e)}")
            return None
        
        if not aggs:
            _record_fetch_error(symbol, "US", "Polygon returned empty aggregates")
            return None
        
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    except Exception as e:
        _record_fetch_error(symbol, "US", f"get_us_stock_data exception: {str(e)}")
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
    """
    获取A股历史数据
    
    数据源优先级:
    1. Tushare (需要 Token)
    2. AkShare (免费，无需 Token) - 备选
    """
    # 尝试 Tushare
    df = _get_cn_stock_data_tushare(symbol, days)
    if df is not None and not df.empty:
        return df
    
    # Tushare 失败，尝试 AkShare
    print(f"⚠️ Tushare 获取 {symbol} 失败，尝试 AkShare...")
    df = _get_cn_stock_data_akshare(symbol, days)
    return df


def _get_cn_stock_data_tushare(symbol, days=365):
    """使用 Tushare 获取 A股数据"""
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
        print(f"Error in Tushare: {e}")
        return None


def _get_cn_stock_data_akshare(symbol, days=365):
    """
    使用 AkShare 获取 A股数据 (免费备选)
    
    参考: daily_stock_analysis 项目的 AkshareFetcher
    数据来源: 东方财富爬虫
    """
    try:
        import akshare as ak
        import time
        
        # 随机休眠防止被封
        time.sleep(0.5 + np.random.random())
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        # 提取纯代码
        code = symbol.split('.')[0] if '.' in symbol else symbol
        
        print(f"📡 AkShare 获取 {code} 历史数据...")
        
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )
        
        if df is None or df.empty:
            print(f"⚠️ AkShare 未找到 {code} 数据")
            return None
        
        # 转换列名（中文 -> 英文）
        df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume'
        }, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"✅ AkShare 获取 {code} 成功: {len(df)} 条记录")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except ImportError:
        print("⚠️ AkShare 未安装，请运行: pip install akshare")
        return None
    except Exception as e:
        print(f"❌ AkShare 获取 {symbol} 失败: {e}")
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


def get_cn_sector_data_period(period='1d'):
    """获取A股行业板块指定时间段的涨跌数据
    
    Args:
        period: '1d'=今日, '1w'=本周, '1m'=本月, 'ytd'=今年
    """
    import tushare as ts
    
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        return None
    
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 获取所有股票的行业分类
        stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        if stocks is None or stocks.empty:
            return None
        
        # 计算时间范围
        today = datetime.now()
        if period == '1d':
            days_back = 1
        elif period == '1w':
            days_back = 7
        elif period == '1m':
            days_back = 30
        elif period == 'ytd':
            # 从年初到现在
            days_back = (today - datetime(today.year, 1, 1)).days
        else:
            days_back = 1
        
        end_date = today.strftime('%Y%m%d')
        start_date = (today - timedelta(days=days_back + 5)).strftime('%Y%m%d')  # 多取几天防止节假日
        
        # 获取期间行情
        daily_data = []
        for days_ago in range(min(days_back + 5, 60)):
            check_date = (today - timedelta(days=days_ago)).strftime('%Y%m%d')
            try:
                df = pro.daily(trade_date=check_date, fields='ts_code,close,pct_chg')
                if df is not None and len(df) > 100:
                    df['date'] = check_date
                    daily_data.append(df)
                    if len(daily_data) >= 2 or (period == '1d' and len(daily_data) >= 1):
                        break
            except:
                continue
        
        if not daily_data:
            return get_cn_sector_data()  # 回退到单日数据
        
        if period == '1d' and len(daily_data) >= 1:
            # 单日直接返回
            return get_cn_sector_data()
        
        # 计算期间涨跌幅 (用最新和最早的收盘价)
        latest_df = daily_data[0]
        earliest_df = daily_data[-1] if len(daily_data) > 1 else daily_data[0]
        
        merged = pd.merge(
            latest_df[['ts_code', 'close']].rename(columns={'close': 'close_latest'}),
            earliest_df[['ts_code', 'close']].rename(columns={'close': 'close_earliest'}),
            on='ts_code', how='inner'
        )
        merged['pct_chg'] = (merged['close_latest'] - merged['close_earliest']) / merged['close_earliest'] * 100
        
        # 合并行业信息
        merged = pd.merge(merged, stocks[['ts_code', 'industry']], on='ts_code', how='left')
        merged = merged.dropna(subset=['industry'])
        
        # 按行业聚合
        sector_stats = merged.groupby('industry').agg({
            'pct_chg': 'mean',
            'ts_code': 'count'
        }).reset_index()
        
        sector_stats.columns = ['name', 'change_pct', 'stock_count']
        sector_stats['sector'] = sector_stats['name']
        sector_stats['amount'] = 0  # 期间成交额暂不计算
        
        sector_stats = sector_stats.sort_values('change_pct', ascending=False)
        return sector_stats[['sector', 'name', 'change_pct', 'amount', 'stock_count']]
        
    except Exception as e:
        print(f"Error in get_cn_sector_data_period: {e}")
        return get_cn_sector_data()  # 回退


def get_us_sector_data_period(period='1d'):
    """获取美股行业板块指定时间段的涨跌数据
    
    Args:
        period: '1d'=今日, '1w'=本周, '1m'=本月, 'ytd'=今年
    """
    api_key = _get_polygon_api_key()
    if not api_key:
        return None
    
    try:
        # 计算时间范围
        today = datetime.now()
        if period == '1d':
            days_back = 2
        elif period == '1w':
            days_back = 7
        elif period == '1m':
            days_back = 30
        elif period == 'ytd':
            days_back = (today - datetime(today.year, 1, 1)).days
        else:
            days_back = 2
        
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
                df = get_us_stock_data(etf, days=days_back + 10)
                if df is not None and len(df) >= 2:
                    latest_close = df.iloc[-1]['Close']
                    earliest_close = df.iloc[max(0, len(df) - days_back - 1)]['Close']
                    change_pct = (latest_close - earliest_close) / earliest_close * 100
                    
                    sector_data.append({
                        'sector': etf,
                        'name': sector_name,
                        'close': latest_close,
                        'change_pct': change_pct,
                        'volume': df.iloc[-1].get('Volume', 0)
                    })
            except:
                continue
        
        if not sector_data:
            return get_us_sector_data()
        
        result = pd.DataFrame(sector_data)
        result = result.sort_values('change_pct', ascending=False)
        return result
        
    except Exception as e:
        print(f"Error in get_us_sector_data_period: {e}")
        return get_us_sector_data()


def get_cn_sector_hot_stocks(sector_name, top_n=10):
    """获取A股指定板块的热门股票
    
    Args:
        sector_name: 行业名称，如 "电气设备", "半导体"
        top_n: 返回前N只股票
    
    Returns:
        DataFrame with columns: ts_code, name, pct_chg
    """
    import tushare as ts
    
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        return None
    
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 获取该行业的股票列表
        stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        sector_stocks = stocks[stocks['industry'] == sector_name]
        
        if sector_stocks is None or sector_stocks.empty:
            return None
        
        ticker_list = sector_stocks['ts_code'].tolist()
        
        # 获取最新交易日行情
        today = datetime.now()
        for days_back in range(5):
            check_date = (today - timedelta(days=days_back)).strftime('%Y%m%d')
            try:
                daily = pro.daily(trade_date=check_date, fields='ts_code,close,pct_chg')
                if daily is not None and len(daily) > 100:
                    break
            except:
                continue
        
        if daily is None or daily.empty:
            return None
        
        # 筛选该板块股票并合并名称
        sector_daily = daily[daily['ts_code'].isin(ticker_list)]
        sector_daily = pd.merge(sector_daily, sector_stocks[['ts_code', 'name']], on='ts_code', how='left')
        
        # 按涨跌幅排序
        sector_daily = sector_daily.sort_values('pct_chg', ascending=False)
        
        return sector_daily[['ts_code', 'name', 'pct_chg']].head(top_n)
        
    except Exception as e:
        print(f"Error in get_cn_sector_hot_stocks: {e}")
        return None


def get_us_sector_hot_stocks(sector_name, top_n=10):
    """获取美股指定板块的热门股票
    
    Args:
        sector_name: 板块名称，如 "Technology", "Financials"
        top_n: 返回前N只股票
    
    Returns:
        DataFrame with columns: symbol, name, pct_chg
    """
    # 每个行业的代表性股票
    sector_stocks = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ADBE', 'CRM', 'CSCO', 'ORCL', 'AMD', 'INTC', 'QCOM', 'TXN', 'IBM'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'CME'],
        'Healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'CVS', 'MDT', 'ISRG', 'GILD'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'PXD', 'DVN', 'HAL', 'BKR', 'KMI', 'WMB'],
        'Industrials': ['CAT', 'RTX', 'UNP', 'HON', 'BA', 'DE', 'LMT', 'UPS', 'GE', 'MMM', 'ETN', 'ITW', 'EMR', 'FDX', 'NSC'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'ORLY', 'MAR', 'YUM', 'DHI', 'GM'],
        'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'EL', 'KHC', 'GIS', 'SYY', 'STZ', 'K'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PCG', 'WEC', 'ED', 'ES', 'AWK', 'DTE', 'AEE'],
        'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'CTVA', 'DOW', 'DD', 'PPG', 'VMC', 'MLM', 'ALB', 'CF'],
        'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'WELL', 'SPG', 'DLR', 'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'UDR'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'CMCSA', 'NFLX', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'ATVI', 'WBD', 'OMC', 'IPG', 'TTWO']
    }
    
    if sector_name not in sector_stocks:
        return None
    
    tickers = sector_stocks[sector_name][:top_n + 5]  # 获取多几个以防有失败的
    
    try:
        stock_data = []
        
        for ticker in tickers:
            try:
                df = get_us_stock_data(ticker, days=5)
                if df is not None and len(df) >= 2:
                    latest_close = df.iloc[-1]['Close']
                    prev_close = df.iloc[-2]['Close']
                    pct_chg = (latest_close - prev_close) / prev_close * 100
                    
                    stock_data.append({
                        'symbol': ticker,
                        'name': ticker,  # 美股用代码作为名称
                        'pct_chg': round(pct_chg, 2)
                    })
            except:
                continue
            
            if len(stock_data) >= top_n:
                break
        
        if not stock_data:
            return None
        
        result = pd.DataFrame(stock_data)
        result = result.sort_values('pct_chg', ascending=False)
        return result.head(top_n)
        
    except Exception as e:
        print(f"Error in get_us_sector_hot_stocks: {e}")
        return None


# ==================== 增强板块分析 ====================

def get_cn_sector_enhanced():
    """获取A股行业板块增强数据
    
    返回:
        - 涨跌幅
        - 成交量放大倍数 (vs 5日均量)
        - 连续上涨天数
        - 资金流向 (主力净流入)
        - 综合热度评分
    """
    import tushare as ts
    
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        return None
    
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 获取申万行业指数列表
        index_list = pro.index_classify(level='L2', src='SW2021')
        if index_list is None or index_list.empty:
            return None
        
        # 获取最近10个交易日数据
        today = datetime.now()
        results = []
        
        for _, idx in index_list.head(50).iterrows():  # 限制数量避免超限
            try:
                ts_code = idx['index_code']
                name = idx['industry_name']
                
                # 获取日线数据
                daily = pro.sw_daily(ts_code=ts_code, 
                                     start_date=(today - timedelta(days=30)).strftime('%Y%m%d'),
                                     end_date=today.strftime('%Y%m%d'))
                
                if daily is None or len(daily) < 2:
                    continue
                
                daily = daily.sort_values('trade_date', ascending=True)
                
                # 1. 今日涨跌幅
                change_pct = daily.iloc[-1]['pct_change'] if 'pct_change' in daily.columns else 0
                
                # 2. 成交量放大 (vs 5日均量)
                if 'vol' in daily.columns and len(daily) >= 5:
                    vol_today = daily.iloc[-1]['vol']
                    vol_avg5 = daily.iloc[-6:-1]['vol'].mean() if len(daily) >= 6 else daily['vol'].mean()
                    volume_ratio = vol_today / vol_avg5 if vol_avg5 > 0 else 1
                else:
                    volume_ratio = 1
                
                # 3. 连续上涨天数
                consecutive_days = 0
                if 'pct_change' in daily.columns:
                    for i in range(len(daily) - 1, -1, -1):
                        if daily.iloc[i]['pct_change'] > 0:
                            consecutive_days += 1
                        else:
                            break
                
                # 4. 资金流向 (用成交额变化估算)
                if 'amount' in daily.columns and len(daily) >= 2:
                    amount_today = daily.iloc[-1]['amount']
                    amount_yesterday = daily.iloc[-2]['amount']
                    money_flow = amount_today - amount_yesterday
                else:
                    money_flow = 0
                
                # 5. 综合热度评分 (0-100)
                heat_score = 0
                # 涨幅贡献 (最高30分)
                heat_score += min(30, max(0, change_pct * 5))
                # 量比贡献 (最高25分)
                heat_score += min(25, max(0, (volume_ratio - 1) * 15))
                # 连涨贡献 (最高25分)
                heat_score += min(25, consecutive_days * 5)
                # 资金流入贡献 (最高20分)
                if money_flow > 0:
                    heat_score += min(20, 10)
                
                results.append({
                    'sector': ts_code,
                    'name': name,
                    'change_pct': round(change_pct, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'consecutive_days': consecutive_days,
                    'money_flow': round(money_flow / 100000000, 2),  # 亿元
                    'heat_score': round(heat_score, 1)
                })
                
            except Exception as e:
                continue
        
        if not results:
            return get_cn_sector_data()  # 回退
        
        df = pd.DataFrame(results)
        df = df.sort_values('heat_score', ascending=False)
        return df
        
    except Exception as e:
        print(f"Error in get_cn_sector_enhanced: {e}")
        return get_cn_sector_data()


def get_us_sector_enhanced():
    """获取美股行业板块增强数据
    
    返回:
        - 涨跌幅
        - 成交量放大倍数
        - 连续上涨天数
        - 综合热度评分
    """
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
    
    results = []
    
    for etf, name in sector_etfs.items():
        try:
            df = get_us_stock_data(etf, days=15)
            if df is None or len(df) < 2:
                continue
            
            # 1. 今日涨跌幅
            latest_close = df.iloc[-1]['Close']
            prev_close = df.iloc[-2]['Close']
            change_pct = (latest_close - prev_close) / prev_close * 100
            
            # 2. 成交量放大
            if 'Volume' in df.columns and len(df) >= 5:
                vol_today = df.iloc[-1]['Volume']
                vol_avg5 = df.iloc[-6:-1]['Volume'].mean() if len(df) >= 6 else df['Volume'].mean()
                volume_ratio = vol_today / vol_avg5 if vol_avg5 > 0 else 1
            else:
                volume_ratio = 1
            
            # 3. 连续上涨天数
            consecutive_days = 0
            for i in range(len(df) - 1, 0, -1):
                if df.iloc[i]['Close'] > df.iloc[i-1]['Close']:
                    consecutive_days += 1
                else:
                    break
            
            # 4. 综合热度评分
            heat_score = 0
            heat_score += min(30, max(0, change_pct * 5))
            heat_score += min(25, max(0, (volume_ratio - 1) * 15))
            heat_score += min(25, consecutive_days * 5)
            
            results.append({
                'sector': etf,
                'name': name,
                'change_pct': round(change_pct, 2),
                'volume_ratio': round(volume_ratio, 2),
                'consecutive_days': consecutive_days,
                'money_flow': 0,  # ETF暂无资金流数据
                'heat_score': round(heat_score, 1)
            })
            
        except Exception as e:
            continue
    
    if not results:
        return get_us_sector_data()
    
    df = pd.DataFrame(results)
    df = df.sort_values('heat_score', ascending=False)
    return df


# ==================== 北向资金数据 ====================

def get_north_money_flow(days: int = 30) -> pd.DataFrame:
    """
    获取北向资金（沪港通+深港通）历史流入流出数据
    
    数据来源: AkShare (东方财富)
    
    Args:
        days: 获取天数
    
    Returns:
        DataFrame with columns: date, sh_money, sz_money, north_money
    """
    try:
        import akshare as ak
        
        print("📡 获取北向资金历史数据...")
        
        # 分别获取沪股通和深股通历史数据
        df_sh = ak.stock_hsgt_hist_em(symbol='沪股通')
        df_sz = ak.stock_hsgt_hist_em(symbol='深股通')
        
        if df_sh is None or df_sz is None:
            print("⚠️ 北向资金数据为空")
            return pd.DataFrame()
        
        # 合并数据
        df_sh = df_sh[['日期', '当日成交净买额']].rename(columns={
            '日期': 'date',
            '当日成交净买额': 'sh_money'
        })
        df_sz = df_sz[['日期', '当日成交净买额']].rename(columns={
            '日期': 'date', 
            '当日成交净买额': 'sz_money'
        })
        
        df = pd.merge(df_sh, df_sz, on='date', how='outer')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)
        df['north_money'] = df['sh_money'].fillna(0) + df['sz_money'].fillna(0)
        df = df.head(days)
        
        print(f"✅ 获取北向资金历史数据成功: {len(df)} 条")
        return df
        
    except ImportError:
        print("⚠️ AkShare 未安装")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ 获取北向资金历史失败: {e}")
        return pd.DataFrame()


def get_north_money_today() -> dict:
    """
    获取今日北向资金流入情况
    
    数据来源: stock_hsgt_fund_flow_summary_em (东方财富实时)
    
    Returns:
        {
            'north_money': 净流入(亿),
            'sh_money': 沪股通(亿),
            'sz_money': 深股通(亿),
            'date': 日期
        }
    """
    try:
        import akshare as ak

        # 获取实时北向资金汇总
        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty:
            return {}

        def _pick_num_col(frame: pd.DataFrame, candidates) -> str:
            for c in candidates:
                if c in frame.columns:
                    return c
            return ""

        def _to_float(val) -> float:
            try:
                if isinstance(val, str):
                    val = val.replace(",", "").strip()
                return float(val)
            except Exception:
                return 0.0

        board_col = "板块" if "板块" in df.columns else ("类型名称" if "类型名称" in df.columns else "")
        dir_col = "资金方向" if "资金方向" in df.columns else ""
        date_col = "交易日" if "交易日" in df.columns else ("日期" if "日期" in df.columns else "")
        status_col = "交易状态" if "交易状态" in df.columns else ""
        money_col = _pick_num_col(df, ["成交净买额", "资金净流入", "当日成交净买额", "净流入额", "净买入额"])

        if not board_col or not money_col:
            return {}

        sh_mask = df[board_col].astype(str).str.contains("沪股通", na=False)
        sz_mask = df[board_col].astype(str).str.contains("深股通", na=False)
        if dir_col:
            north_mask = df[dir_col].astype(str).str.contains("北向", na=False)
            sh_mask = sh_mask & north_mask
            sz_mask = sz_mask & north_mask

        sh_row = df[sh_mask]
        sz_row = df[sz_mask]

        sh_money = _to_float(sh_row[money_col].iloc[0]) if len(sh_row) > 0 else 0.0
        sz_money = _to_float(sz_row[money_col].iloc[0]) if len(sz_row) > 0 else 0.0
        north_money = sh_money + sz_money

        date_str = str(df[date_col].iloc[0]) if date_col else ""
        trade_status = str(df[status_col].iloc[0]) if status_col and len(df) > 0 else ""

        return {
            "date": date_str,
            "north_money": float(north_money),
            "sh_money": float(sh_money),
            "sz_money": float(sz_money),
            "trade_status": trade_status,
            "money_col": money_col,
        }

    except Exception as e:
        print(f"❌ 获取今日北向资金失败: {e}")
        return {}


def get_north_holding_stocks(limit: int = 50) -> pd.DataFrame:
    """
    获取北向资金持股排行
    
    Returns:
        DataFrame with columns: code, name, hold_shares, hold_value, hold_ratio
    """
    try:
        import akshare as ak
        
        print("📡 获取北向资金持股排行...")
        
        df = ak.stock_hsgt_hold_stock_em()
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 重命名列
        col_map = {
            '代码': 'code',
            '名称': 'name',
            '持股数量': 'hold_shares',
            '持股市值': 'hold_value',
            '持股占比': 'hold_ratio',
            '今日增持股数': 'today_change'
        }
        
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # 按持股市值排序
        if 'hold_value' in df.columns:
            df = df.sort_values('hold_value', ascending=False)
        
        print(f"✅ 获取北向持股排行成功: {len(df)} 只")
        return df.head(limit)
        
    except Exception as e:
        print(f"❌ 获取北向持股排行失败: {e}")
        return pd.DataFrame()


# ==================== 港股数据 ====================

def get_hk_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    获取港股历史数据
    
    数据来源: YFinance (免费)
    
    Args:
        symbol: 港股代码，如 "00700" 或 "hk00700"
        days: 获取天数
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        import yfinance as yf
        
        # 标准化代码格式
        code = symbol.upper().replace('HK', '').replace('.HK', '')
        code = code.zfill(4)  # 补齐4位
        yf_symbol = f"{code}.HK"
        
        print(f"📡 YFinance 获取港股 {yf_symbol} 数据...")
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 获取数据
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            print(f"⚠️ 未找到 {yf_symbol} 数据")
            return None
        
        # 确保列名一致
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date'})
        
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # 只保留需要的列
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]
        
        print(f"✅ YFinance 获取 {yf_symbol} 成功: {len(df)} 条记录")
        return df
        
    except ImportError:
        print("⚠️ yfinance 未安装，请运行: pip install yfinance")
        return None
    except Exception as e:
        print(f"❌ 获取港股 {symbol} 失败: {e}")
        return None


def get_hk_stock_info(symbol: str) -> dict:
    """
    获取港股基本信息
    
    Args:
        symbol: 港股代码
    
    Returns:
        {'name': 公司名, 'market_cap': 市值, 'industry': 行业}
    """
    try:
        import yfinance as yf
        
        code = symbol.upper().replace('HK', '').replace('.HK', '').zfill(4)
        yf_symbol = f"{code}.HK"
        
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        
        return {
            'name': info.get('shortName', info.get('longName', '')),
            'market_cap': info.get('marketCap', 0),
            'industry': info.get('industry', ''),
            'sector': info.get('sector', ''),
            'currency': 'HKD'
        }
        
    except Exception as e:
        print(f"❌ 获取港股 {symbol} 信息失败: {e}")
        return {}


def get_stock_data(symbol: str, market: str = None, days: int = 365) -> pd.DataFrame:
    """
    统一的股票数据获取接口 - 支持美股/A股/港股
    
    Args:
        symbol: 股票代码
        market: 市场 ('US', 'CN', 'HK')，None 则自动识别
        days: 获取天数
    
    Returns:
        DataFrame with OHLCV data
    """
    # 自动识别市场
    if market is None:
        market = detect_market(symbol)
    
    market = market.upper()
    
    if market == 'US':
        return get_us_stock_data(symbol, days)
    elif market == 'CN':
        return get_cn_stock_data(symbol, days)
    elif market == 'HK':
        return get_hk_stock_data(symbol, days)
    else:
        print(f"⚠️ 未知市场: {market}")
        return None


def detect_market(symbol: str) -> str:
    """
    自动检测股票所属市场
    
    Rules:
    - 以 HK 开头或包含 .HK: 港股
    - 以数字开头且6位: A股
    - 其他: 美股
    """
    symbol = symbol.upper().strip()
    
    # 港股
    if symbol.startswith('HK') or '.HK' in symbol:
        return 'HK'
    
    # A股 (6位数字，或 .SH/.SZ 后缀)
    if symbol.endswith('.SH') or symbol.endswith('.SZ') or symbol.endswith('.BJ'):
        return 'CN'
    
    # 纯数字且6位
    code = symbol.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
    if code.isdigit() and len(code) == 6:
        return 'CN'
    
    # 默认美股
    return 'US'
