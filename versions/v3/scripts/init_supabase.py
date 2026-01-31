#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supabase 数据库初始化脚本
创建必要的表结构
"""
import os
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

def init_supabase_tables():
    """在 Supabase 中创建表"""
    from supabase import create_client
    
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    
    if not url or not key:
        print("❌ Missing SUPABASE_URL or SUPABASE_KEY")
        return False
    
    print(f"🔗 Connecting to Supabase: {url[:30]}...")
    supabase = create_client(url, key)
    
    # 测试连接
    try:
        # 尝试查询一个不存在的表 (会返回空或错误，但能确认连接)
        result = supabase.table('scan_results').select('*').limit(1).execute()
        print(f"✅ Connected! Found {len(result.data)} records in scan_results")
        return True
    except Exception as e:
        error_msg = str(e)
        if 'relation' in error_msg and 'does not exist' in error_msg:
            print("⚠️ Tables don't exist yet. Please create them in Supabase SQL Editor.")
            print("\n📋 Copy and run this SQL in Supabase SQL Editor:\n")
            print(get_create_tables_sql())
            return False
        else:
            print(f"❌ Connection error: {e}")
            return False


def get_create_tables_sql():
    """返回创建表的 SQL"""
    return """
-- 扫描结果表
CREATE TABLE IF NOT EXISTS scan_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    scan_date DATE NOT NULL,
    price REAL,
    turnover_m REAL,
    blue_daily REAL,
    blue_weekly REAL,
    blue_monthly REAL,
    adx REAL,
    volatility REAL,
    is_heima BOOLEAN,
    is_juedi BOOLEAN,
    strat_d_trend BOOLEAN,
    strat_c_resonance BOOLEAN,
    legacy_signal BOOLEAN,
    regime VARCHAR(50),
    adaptive_thresh REAL,
    vp_rating VARCHAR(20),
    profit_ratio REAL,
    wave_phase VARCHAR(50),
    wave_desc VARCHAR(100),
    chan_signal VARCHAR(50),
    chan_desc VARCHAR(100),
    market_cap REAL,
    cap_category VARCHAR(30),
    company_name VARCHAR(200),
    industry VARCHAR(200),
    stop_loss REAL,
    shares_rec INTEGER,
    risk_reward_score REAL,
    market VARCHAR(10) DEFAULT 'US',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, scan_date, market)
);

CREATE INDEX IF NOT EXISTS idx_scan_date ON scan_results(scan_date);
CREATE INDEX IF NOT EXISTS idx_symbol ON scan_results(symbol);
CREATE INDEX IF NOT EXISTS idx_blue_daily ON scan_results(blue_daily);
CREATE INDEX IF NOT EXISTS idx_market ON scan_results(market);

-- 信号性能缓存表
CREATE TABLE IF NOT EXISTS signal_performance (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    scan_date VARCHAR(20),
    market VARCHAR(10) DEFAULT 'US',
    return_5d REAL,
    return_10d REAL,
    return_20d REAL,
    max_gain REAL,
    max_drawdown REAL,
    calculated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, scan_date, market)
);

CREATE INDEX IF NOT EXISTS idx_perf_date ON signal_performance(scan_date);
CREATE INDEX IF NOT EXISTS idx_perf_market ON signal_performance(market);

-- 股票信息缓存表
CREATE TABLE IF NOT EXISTS stock_info (
    symbol VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200),
    industry VARCHAR(200),
    area VARCHAR(100),
    market VARCHAR(10),
    list_date VARCHAR(20),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 交易记录表
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    market VARCHAR(10) DEFAULT 'US',
    trade_type VARCHAR(10) NOT NULL,
    price REAL NOT NULL,
    shares INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 关注列表表
CREATE TABLE IF NOT EXISTS watchlist (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    market VARCHAR(10) DEFAULT 'US',
    entry_date DATE NOT NULL,
    entry_price REAL NOT NULL,
    shares INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'holding',
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, market, entry_date)
);

-- 博主表
CREATE TABLE IF NOT EXISTS bloggers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    platform VARCHAR(50),
    specialty VARCHAR(50),
    url VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 博主推荐表
CREATE TABLE IF NOT EXISTS blogger_recommendations (
    id SERIAL PRIMARY KEY,
    blogger_id INTEGER REFERENCES bloggers(id),
    symbol VARCHAR(20) NOT NULL,
    market VARCHAR(10) DEFAULT 'US',
    rec_date DATE NOT NULL,
    rec_price REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

SELECT 'All tables created successfully!' as status;
"""


if __name__ == "__main__":
    print("=" * 50)
    print("Supabase Database Initialization")
    print("=" * 50)
    
    init_supabase_tables()
