"""
批量拉取历史 K 线数据
Batch Fetch Historical Data

使用 yfinance 拉取数据并存储到本地数据库
"""

import pandas as pd
import numpy as np
import time
from datetime import date, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fetch_with_yfinance(symbol: str, days: int = 250, retry: int = 3) -> pd.DataFrame:
    """使用 yfinance 获取数据"""
    import yfinance as yf
    
    for attempt in range(retry):
        try:
            ticker = yf.Ticker(symbol)
            
            # 计算周期
            if days <= 60:
                period = '3mo'
            elif days <= 180:
                period = '6mo'
            elif days <= 365:
                period = '1y'
            else:
                period = '2y'
            
            df = ticker.history(period=period)
            
            if df.empty:
                return None
            
            # 重命名列
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # 只保留需要的列
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            return df
            
        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str:
                if attempt < retry - 1:
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f"   ⏳ 限流，等待 {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            return None
    
    return None


def fetch_with_polygon(symbol: str, days: int = 250) -> pd.DataFrame:
    """使用 Polygon API 获取数据"""
    try:
        from polygon import RESTClient
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            return None
        
        client = RESTClient(api_key)
        
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        aggs = list(client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            limit=50000
        ))
        
        if not aggs:
            return None
        
        df = pd.DataFrame([{
            'Date': pd.Timestamp.fromtimestamp(a.timestamp/1000),
            'Open': a.open,
            'High': a.high,
            'Low': a.low,
            'Close': a.close,
            'Volume': a.volume,
        } for a in aggs])
        
        return df
        
    except Exception as e:
        return None


def fetch_stock_data(symbol: str, days: int = 250) -> pd.DataFrame:
    """
    智能获取股票数据
    优先使用 Polygon，fallback 到 yfinance
    """
    # 先尝试 Polygon
    df = fetch_with_polygon(symbol, days)
    if df is not None and len(df) >= 60:
        return df
    
    # Fallback 到 yfinance
    df = fetch_with_yfinance(symbol, days)
    return df


def batch_fetch_and_store(market: str = 'US', 
                          max_symbols: int = 500,
                          delay: float = 0.2) -> dict:
    """
    批量拉取并存储历史数据
    
    Args:
        market: 市场
        max_symbols: 最大股票数
        delay: 每次请求后的延迟（秒）
    
    Returns:
        统计信息
    """
    from db.database import get_connection
    from db.stock_history import save_stock_history, get_history_stats, init_history_db
    
    init_history_db()
    
    print(f"\n{'='*60}")
    print(f"📥 批量拉取历史数据")
    print(f"   市场: {market}")
    print(f"   最大股票数: {max_symbols}")
    print(f"   请求延迟: {delay}s")
    print(f"{'='*60}\n")
    
    # 获取需要拉取的股票列表
    conn = get_connection()
    cursor = conn.cursor()
    
    # 获取信号最多的股票
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM scan_results
        WHERE market = ?
        GROUP BY symbol
        ORDER BY cnt DESC
        LIMIT ?
    """, (market, max_symbols))
    
    symbols = [row['symbol'] for row in cursor.fetchall()]
    conn.close()
    
    print(f"📋 待拉取股票: {len(symbols)}")
    
    # 检查已有数据
    stats_before = get_history_stats()
    existing_symbols = set()
    
    if stats_before['by_market'].get(market):
        # 获取已有股票列表
        import sqlite3
        from db.stock_history import get_history_db_path
        
        hist_conn = sqlite3.connect(get_history_db_path())
        hist_cursor = hist_conn.cursor()
        hist_cursor.execute("""
            SELECT DISTINCT symbol FROM stock_history 
            WHERE market = ?
        """, (market,))
        existing_symbols = {row[0] for row in hist_cursor.fetchall()}
        hist_conn.close()
    
    # 过滤掉已有数据的股票
    symbols_to_fetch = [s for s in symbols if s not in existing_symbols]
    print(f"   已有数据: {len(existing_symbols)}")
    print(f"   需要拉取: {len(symbols_to_fetch)}")
    
    if not symbols_to_fetch:
        print("✅ 所有股票数据已存在")
        return {'fetched': 0, 'failed': 0, 'skipped': len(existing_symbols)}
    
    # 开始拉取
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols_to_fetch):
        try:
            # 显示进度
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(symbols_to_fetch) - i - 1) / rate if rate > 0 else 0
                print(f"   [{i+1}/{len(symbols_to_fetch)}] {symbol}... (ETA: {eta/60:.1f}分钟)")
            
            # 拉取数据
            df = fetch_with_yfinance(symbol, days=250)
            
            if df is not None and len(df) >= 60:
                # 存储
                count = save_stock_history(symbol, market, df)
                success_count += 1
            else:
                fail_count += 1
            
            # 延迟避免限流
            time.sleep(delay)
            
            # 每 50 个休息更长时间
            if (i + 1) % 50 == 0:
                print(f"   💤 休息 5 秒...")
                time.sleep(5)
            
        except Exception as e:
            fail_count += 1
            if "Rate" in str(e) or "limit" in str(e).lower():
                print(f"   ⚠️ 限流，休息 30 秒...")
                time.sleep(30)
            continue
    
    # 统计
    elapsed = time.time() - start_time
    stats_after = get_history_stats()
    
    print(f"\n{'='*60}")
    print(f"✅ 拉取完成!")
    print(f"   耗时: {elapsed/60:.1f} 分钟")
    print(f"   成功: {success_count}")
    print(f"   失败: {fail_count}")
    print(f"   总股票数: {stats_after['total_symbols']}")
    print(f"   总记录数: {stats_after['total_records']}")
    print(f"{'='*60}")
    
    return {
        'fetched': success_count,
        'failed': fail_count,
        'skipped': len(existing_symbols),
        'total_symbols': stats_after['total_symbols'],
        'total_records': stats_after['total_records']
    }


def quick_fetch(symbols: list, market: str = 'US', delay: float = 0.5) -> int:
    """快速拉取指定股票列表"""
    from db.stock_history import save_stock_history, init_history_db
    
    init_history_db()
    
    success = 0
    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {symbol}...", end=" ")
        
        df = fetch_with_yfinance(symbol, days=250)
        
        if df is not None and len(df) >= 60:
            save_stock_history(symbol, market, df)
            print(f"✓ {len(df)} bars")
            success += 1
        else:
            print("✗")
        
        time.sleep(delay)
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量拉取历史数据')
    parser.add_argument('--market', default='US')
    parser.add_argument('--max', type=int, default=200, help='最大股票数')
    parser.add_argument('--delay', type=float, default=0.5, help='请求延迟(秒)')
    parser.add_argument('--quick', nargs='+', help='快速拉取指定股票')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_fetch(args.quick, args.market, args.delay)
    else:
        batch_fetch_and_store(args.market, args.max, args.delay)
