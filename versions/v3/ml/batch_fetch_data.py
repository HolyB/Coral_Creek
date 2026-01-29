"""
批量获取历史K线数据
Batch Fetch Historical Data

为更多股票获取历史数据，扩大训练样本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime, timedelta
from typing import List, Set
import pandas as pd

from db.database import get_connection
from db.stock_history import save_stock_history, get_history_stats, init_history_db
from data_fetcher import get_stock_data


def get_symbols_needing_data(market: str = 'US', min_days: int = 60) -> List[str]:
    """
    获取需要历史数据的股票列表
    
    优先级:
    1. 有扫描信号但没有历史数据的
    2. 历史数据不足的
    """
    conn = get_connection()
    
    # 获取所有有信号的股票
    query = """
        SELECT DISTINCT symbol 
        FROM scan_results 
        WHERE market = ?
    """
    signals_df = pd.read_sql_query(query, conn, params=(market,))
    signal_symbols = set(signals_df['symbol'].tolist())
    
    # 获取已有历史数据的股票
    query = """
        SELECT symbol, COUNT(*) as cnt
        FROM stock_history
        WHERE market = ?
        GROUP BY symbol
        HAVING cnt >= ?
    """
    
    # 检查 stock_history 表是否存在
    try:
        from db.stock_history import get_history_db_path
        import sqlite3
        history_conn = sqlite3.connect(get_history_db_path())
        history_df = pd.read_sql_query(query, history_conn, params=(market, min_days))
        history_symbols = set(history_df['symbol'].tolist())
        history_conn.close()
    except:
        history_symbols = set()
    
    conn.close()
    
    # 需要数据的股票 = 有信号但没历史数据的
    need_data = signal_symbols - history_symbols
    
    print(f"📊 {market} 市场统计:")
    print(f"   有信号的股票: {len(signal_symbols)}")
    print(f"   已有历史数据: {len(history_symbols)}")
    print(f"   需要获取数据: {len(need_data)}")
    
    return list(need_data)


def fetch_batch(symbols: List[str], 
                market: str = 'US',
                days: int = 365,
                batch_size: int = 50,
                delay: float = 0.5) -> dict:
    """
    批量获取历史数据
    
    Args:
        symbols: 股票列表
        market: 市场
        days: 获取天数
        batch_size: 每批数量
        delay: 请求间隔(秒)
    
    Returns:
        统计信息
    """
    init_history_db()
    
    total = len(symbols)
    success = 0
    failed = 0
    skipped = 0
    
    print(f"\n🚀 开始批量获取 {total} 只股票的历史数据")
    print(f"   每批: {batch_size}, 间隔: {delay}秒")
    print("="*50)
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        try:
            # 获取数据
            df = get_stock_data(symbol, days=days)
            
            if df is not None and len(df) >= 20:
                # 保存到数据库
                saved = save_stock_history(symbol, market, df)
                if saved > 0:
                    success += 1
                    if (success % 10) == 0:
                        print(f"   ✅ {success}/{total} - {symbol}: {saved} 条")
                else:
                    skipped += 1
            else:
                failed += 1
                if df is None:
                    print(f"   ❌ {symbol}: 无数据")
                else:
                    print(f"   ⚠️ {symbol}: 数据不足 ({len(df)} 条)")
            
            # 进度
            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"\n📈 进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                print(f"   成功: {success}, 失败: {failed}, 跳过: {skipped}")
                print(f"   预计剩余: {remaining/60:.1f} 分钟\n")
            
            # 延迟
            time.sleep(delay)
            
        except Exception as e:
            failed += 1
            print(f"   ❌ {symbol}: {e}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("📊 批量获取完成")
    print(f"   总数: {total}")
    print(f"   成功: {success}")
    print(f"   失败: {failed}")
    print(f"   跳过: {skipped}")
    print(f"   耗时: {elapsed/60:.1f} 分钟")
    print("="*50)
    
    # 更新后的统计
    stats = get_history_stats()
    print(f"\n📊 数据库当前状态:")
    print(f"   股票数: {stats.get('total_symbols', 0)}")
    print(f"   记录数: {stats.get('total_records', 0):,}")
    
    return {
        'total': total,
        'success': success,
        'failed': failed,
        'skipped': skipped,
        'elapsed_seconds': elapsed
    }


def run_fetch(market: str = 'US', 
              max_symbols: int = 500,
              days: int = 365,
              delay: float = 0.3) -> dict:
    """
    运行数据获取
    
    Args:
        market: 市场
        max_symbols: 最大获取数量
        days: 获取天数
        delay: 请求间隔
    """
    print("\n" + "="*60)
    print("📦 批量获取历史K线数据")
    print("="*60)
    
    # 获取需要数据的股票
    symbols = get_symbols_needing_data(market)
    
    if not symbols:
        print("✅ 所有股票都已有历史数据")
        return {'total': 0, 'success': 0}
    
    # 限制数量
    if len(symbols) > max_symbols:
        print(f"⚠️ 限制为前 {max_symbols} 只")
        symbols = symbols[:max_symbols]
    
    # 批量获取
    result = fetch_batch(symbols, market, days, delay=delay)
    
    return result


# === 命令行入口 ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量获取历史数据')
    parser.add_argument('--market', type=str, default='US', help='市场 (US/CN)')
    parser.add_argument('--max', type=int, default=500, help='最大获取数量')
    parser.add_argument('--days', type=int, default=365, help='获取天数')
    parser.add_argument('--delay', type=float, default=0.3, help='请求间隔(秒)')
    
    args = parser.parse_args()
    
    run_fetch(
        market=args.market,
        max_symbols=args.max,
        days=args.days,
        delay=args.delay
    )
