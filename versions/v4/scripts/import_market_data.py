#!/usr/bin/env python3
"""
将解压后的 CSV 市场数据导入 stock_history.db

数据来源:
  - A股: /Users/bertwang/Cursor/A股/1d_unzip/{month}/YYYYMMDD.csv
  - 美股: /Users/bertwang/Cursor/美股数据/1d_unzip/{month}/YYYYMMDD.csv

CSV 格式:
  A股: exchange,symbol,open,close,high,low,amount,volume,bob,eob,type,sequence
       exchange: SHSE -> .SH, SZSE -> .SZ
  美股: exchange,symbol,open,high,low,close,amount,volume,bob,eob,type,sequence
       (注意: 美股列顺序不同, close 在 high/low 之后)

用法:
    # 导入某月 A股+美股
    python scripts/import_market_data.py --month 202603

    # 只导入某天
    python scripts/import_market_data.py --date 20260313

    # 只导入特定市场
    python scripts/import_market_data.py --month 202603 --market CN

    # 完整流程: 解压 + 导入
    python scripts/import_market_data.py --month 202603 --unzip
"""

import os
import sys
import glob
import argparse
import sqlite3
import pandas as pd
from datetime import datetime

# 路径常量
CN_UNZIP_BASE = "/Users/bertwang/Cursor/A股/1d_unzip"
US_UNZIP_BASE = "/Users/bertwang/Cursor/美股数据/1d_unzip"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(V3_DIR, "db", "stock_history.db")

# Exchange code -> symbol suffix mapping
EXCHANGE_MAP_CN = {
    'SHSE': '.SH',
    'SZSE': '.SZ',
}


def init_db():
    """Initialize connection to existing stock_history table"""
    conn = sqlite3.connect(DB_PATH)
    # Ensure WAL mode for better concurrent performance
    conn.execute('PRAGMA journal_mode=WAL')
    # Indexes should already exist
    conn.commit()
    return conn


def parse_cn_csv(csv_path):
    """Parse A股 CSV -> list of (symbol, market, date, OHLCV) dicts"""
    if os.path.getsize(csv_path) == 0:
        return []
    df = pd.read_csv(csv_path)
    # Extract date from filename
    date_str = os.path.basename(csv_path).replace('.csv', '')
    trade_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    records = []
    for _, row in df.iterrows():
        exchange = row['exchange']
        suffix = EXCHANGE_MAP_CN.get(exchange)
        if not suffix:
            continue
        symbol = f"{row['symbol']}{suffix}"
        # Ensure symbol is zero-padded: 600000.SH not 600000.0.SH
        code = str(row['symbol']).split('.')[0]
        if exchange == 'SHSE' or exchange == 'SZSE':
            code = code.zfill(6)
        symbol = f"{code}{suffix}"

        records.append({
            'symbol': symbol,
            'market': 'CN',
            'trade_date': trade_date,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'turnover': float(row['amount']),
        })
    return records


def parse_us_csv(csv_path):
    """Parse 美股 CSV -> list of records"""
    if os.path.getsize(csv_path) == 0:
        return []
    df = pd.read_csv(csv_path)
    date_str = os.path.basename(csv_path).replace('.csv', '')
    trade_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    records = []
    for _, row in df.iterrows():
        symbol = str(row['symbol']).strip()
        if not symbol or symbol == 'nan':
            continue

        records.append({
            'symbol': symbol,
            'market': 'US',
            'trade_date': trade_date,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'turnover': float(row.get('amount', 0)),
        })
    return records


def import_records(conn, records):
    """Batch insert/replace records into stock_history"""
    if not records:
        return 0
    # Use INSERT OR REPLACE with the unique constraint on (symbol, market, trade_date)
    # First, check if unique index exists, create if not
    try:
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_sh_unique ON stock_history(symbol, market, trade_date)')
    except Exception:
        pass
    
    conn.executemany("""
        INSERT OR REPLACE INTO stock_history (symbol, market, trade_date, open, high, low, close, volume, turnover)
        VALUES (:symbol, :market, :trade_date, :open, :high, :low, :close, :volume, :turnover)
    """, records)
    conn.commit()
    return len(records)


def import_date(conn, date_str, markets=None):
    """Import data for a specific date (format: YYYYMMDD)"""
    if markets is None:
        markets = ['CN', 'US']

    # Determine which month folder
    month = date_str[:6]
    total = 0

    for market in markets:
        if market == 'CN':
            csv_path = os.path.join(CN_UNZIP_BASE, month, f"{date_str}.csv")
            parser = parse_cn_csv
        else:
            csv_path = os.path.join(US_UNZIP_BASE, month, f"{date_str}.csv")
            parser = parse_us_csv

        if not os.path.exists(csv_path):
            # Try year-level folder
            year = date_str[:4]
            if market == 'CN':
                csv_path = os.path.join(CN_UNZIP_BASE, year, f"{date_str}.csv")
            else:
                csv_path = os.path.join(US_UNZIP_BASE, year, f"{date_str}.csv")

        if not os.path.exists(csv_path):
            continue

        records = parser(csv_path)
        count = import_records(conn, records)
        total += count
        print(f"  {market} {date_str}: {count} records")

    return total


def import_month(conn, month, markets=None):
    """Import all dates in a month folder"""
    if markets is None:
        markets = ['CN', 'US']

    total = 0
    for market in markets:
        if market == 'CN':
            csv_dir = os.path.join(CN_UNZIP_BASE, month)
        else:
            csv_dir = os.path.join(US_UNZIP_BASE, month)

        if not os.path.exists(csv_dir):
            print(f"⚠️  {market} {month} 不存在: {csv_dir}")
            continue

        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        parser = parse_cn_csv if market == 'CN' else parse_us_csv
        label = "A股" if market == 'CN' else "美股"

        print(f"\n📥 导入 {label} {month} ({len(csv_files)} 天):")
        for csv_path in csv_files:
            date_str = os.path.basename(csv_path).replace('.csv', '')
            records = parser(csv_path)
            count = import_records(conn, records)
            total += count
            print(f"  {date_str}: {count} records")

    return total


def get_db_stats(conn):
    """Print current DB statistics"""
    for market in ['US', 'CN']:
        row = conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date), COUNT(DISTINCT symbol) FROM stock_history WHERE market=?",
            (market,)
        ).fetchone()
        print(f"  {market}: {row[0]} ~ {row[1]} | {row[2]} days | {row[3]} symbols")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导入市场数据到 stock_history.db")
    parser.add_argument("--month", help="按月份导入，如 202603")
    parser.add_argument("--date", help="按日期导入，如 20260313")
    parser.add_argument("--market", choices=['CN', 'US'], help="指定市场")
    parser.add_argument("--unzip", action="store_true", help="导入前先解压")
    parser.add_argument("--stats", action="store_true", help="只显示统计信息")

    args = parser.parse_args()

    conn = init_db()

    if args.stats:
        print("📊 stock_history.db 统计:")
        get_db_stats(conn)
        conn.close()
        sys.exit(0)

    if args.unzip and args.month:
        from scripts.unzip_market_data import unzip_month
        markets = [args.market] if args.market else None
        unzip_month(args.month, markets)

    markets = [args.market] if args.market else None

    if args.month:
        total = import_month(conn, args.month, markets)
    elif args.date:
        total = import_date(conn, args.date, markets)
    else:
        parser.print_help()
        conn.close()
        sys.exit(0)

    print(f"\n✅ 共导入 {total} 条记录")
    print("\n📊 当前数据库统计:")
    get_db_stats(conn)
    conn.close()
