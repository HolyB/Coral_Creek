#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backfill 后处理脚本：建立市场状态表 (Market Status)
================================================
功能：
1. 扫描 scan_results 表中出现过的所有日期
2. 拉取对应日期的 SPY (美股) / SH000001 (A股) 数据
3. 计算市场状态 (Regime): Bull/Bear/Neutral/Crash
4. 存入独立的 market_status 表，供策略回测时 JOIN 使用

使用方法:
    python scripts/post_backfill_regime.py --market US
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import get_db

def init_market_status_table(conn):
    """初始化 market_status 表"""
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_status (
        market TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        index_symbol TEXT,
        close_price REAL,
        ma20 REAL,
        ma50 REAL,
        ma200 REAL,
        regime TEXT,           -- Bull, Bear, Neutral
        trend TEXT,            -- Up, Down, Sideways
        is_crash BOOLEAN,      -- 是否暴跌日 (>2%)
        change_pct REAL,       -- 当日涨跌幅
        PRIMARY KEY (market, trade_date)
    )
    """)
    conn.commit()
    print("✅ market_status 表已就绪")

def get_scanned_dates(conn, market='US'):
    """从 scan_results 获取所有已扫描日期"""
    query = f"SELECT DISTINCT scan_date FROM scan_results WHERE market = '{market}' ORDER BY scan_date"
    df = pd.read_sql_query(query, conn)
    return pd.to_datetime(df['scan_date']).sort_values().tolist()

def fetch_spy_data(start_date, end_date):
    """从 Polygon 拉取 SPY 数据"""
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        # Try .env
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(parent_dir, ".env"))
            api_key = os.environ.get('POLYGON_API_KEY')
        except:
            pass
            
    if not api_key:
        print("❌ POLYGON_API_KEY 未找到")
        return pd.DataFrame()

    # 多拉一点数据算均线
    start_dt = start_date - timedelta(days=365)
    
    s_str = start_dt.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    print(f"📊 拉取 SPY 数据 (Polygon) {s_str} ~ {e_str}...")
    
    try:
        client = RESTClient(api_key)
        # SPY for US Market
        aggs = client.get_aggs("SPY", 1, "day", s_str, e_str, limit=50000)
        
        records = []
        for agg in aggs:
            dt = datetime.fromtimestamp(agg.timestamp / 1000)
            records.append({
                'Date': dt,
                'Close': float(agg.close),
                'Open': float(agg.open)
            })
            
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ Polygon API Error: {e}")
        return pd.DataFrame()

def calculate_market_status(df):
    """计算市场状态指标"""
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Change_Pct'] = df['Close'].pct_change() * 100
    
    def get_regime(row):
        if pd.isna(row['MA20']) or pd.isna(row['MA200']):
            return 'Neutral'
        if row['Close'] > row['MA20']:
            if row['Close'] > row['MA200']:
                return 'Bull'   # 站上短期和长期
            else:
                return 'Rebound' # 熊市反弹
        else:
            if row['Close'] < row['MA200']:
                return 'Bear'   #以此类推
            else:
                return 'Pullback' # 牛市回调
                
    def get_trend(row):
        if pd.isna(row['MA20']) or pd.isna(row['MA50']):
            return 'Sideways'
        if row['MA20'] > row['MA50']:
            return 'Up'
        return 'Down'

    df['Regime'] = df.apply(get_regime, axis=1)
    df['Trend'] = df.apply(get_trend, axis=1)
    df['Is_Crash'] = df['Change_Pct'] < -2.0
    
    return df

def save_market_status(conn, df, market='US'):
    """保存计算结果到数据库"""
    print(f"💾 保存 {len(df)} 条市场状态记录...")
    cursor = conn.cursor()
    
    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO market_status 
            (market, trade_date, index_symbol, close_price, ma20, ma50, ma200, regime, trend, is_crash, change_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market,
                row['Date'].strftime('%Y-%m-%d'),
                'SPY',
                row['Close'],
                row['MA20'],
                row['MA50'],
                row['MA200'],
                row['Regime'],
                row['Trend'],
                row['Is_Crash'],
                row['Change_Pct']
            ))
            count += 1
        except Exception as e:
            print(f"写入失败 {row['Date']}: {e}")
            
    conn.commit()
    print(f"✅ 成功写入 {count} 条记录")

def run_post_processing(market='US'):
    db_path = os.path.join(parent_dir, 'db', 'coral_creek.db')
    conn = sqlite3.connect(db_path)
    
    # 1. 初始化表
    init_market_status_table(conn)
    
    # 2. 获取日期范围
    dates = get_scanned_dates(conn, market)
    if not dates:
        print("⚠️ 没有已扫描的日期")
        return
        
    start_date = dates[0]
    end_date = dates[-1]
    print(f"📅 扫描范围: {start_date.date()} ~ {end_date.date()} ({len(dates)} 天)")
    
    # 3. 拉取大盘数据
    df = fetch_spy_data(start_date, end_date)
    if df.empty:
        print("❌ 无法获取大盘数据")
        return
        
    # 4. 计算指标
    df = calculate_market_status(df)
    
    # 5. 过滤出我们需要的时间段（但保留计算好的均线）
    df_save = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # 6. 保存
    save_market_status(conn, df_save, market)
    conn.close()

if __name__ == "__main__":
    run_post_processing()
