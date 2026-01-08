#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库连接和会话管理
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date

# 数据库文件路径
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "coral_creek.db")


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 返回字典格式
    return conn


@contextmanager
def get_db():
    """数据库连接上下文管理器"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """初始化数据库表"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 扫描结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, scan_date)
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_date ON scan_results(scan_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON scan_results(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blue_daily ON scan_results(blue_daily)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_date_blue ON scan_results(scan_date, blue_daily)")
        
        # 扫描任务表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date DATE NOT NULL UNIQUE,
                market VARCHAR(10) DEFAULT 'US',
                status VARCHAR(20) DEFAULT 'pending',
                total_stocks INTEGER DEFAULT 0,
                scanned_stocks INTEGER DEFAULT 0,
                signals_found INTEGER DEFAULT 0,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON scan_jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_date ON scan_jobs(scan_date)")
        
        print(f"✅ Database initialized at: {DB_PATH}")


def get_scanned_dates(start_date=None, end_date=None):
    """获取已扫描的日期列表"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT DISTINCT scan_date FROM scan_results"
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("scan_date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("scan_date <= ?")
                params.append(end_date)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY scan_date DESC"
        
        cursor.execute(query, params)
        return [row['scan_date'] for row in cursor.fetchall()]


def get_missing_dates(start_date, end_date):
    """获取缺失的交易日期 (排除周末)"""
    from datetime import timedelta
    
    scanned = set(get_scanned_dates(start_date, end_date))
    
    missing = []
    current = datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
    end = datetime.strptime(end_date, '%Y-%m-%d').date() if isinstance(end_date, str) else end_date
    
    while current <= end:
        # 跳过周末 (5=Saturday, 6=Sunday)
        if current.weekday() < 5:
            date_str = current.strftime('%Y-%m-%d')
            if date_str not in scanned:
                missing.append(date_str)
        current += timedelta(days=1)
    
    return missing


def insert_scan_result(result_dict):
    """插入或更新单条扫描结果"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # UPSERT: 如果存在则更新，不存在则插入
        cursor.execute("""
            INSERT INTO scan_results (
                symbol, scan_date, price, turnover_m, blue_daily, blue_weekly, blue_monthly,
                adx, volatility, is_heima, is_juedi, strat_d_trend, strat_c_resonance,
                legacy_signal, regime, adaptive_thresh, vp_rating, profit_ratio,
                wave_phase, wave_desc, chan_signal, chan_desc, market_cap, cap_category,
                company_name, industry, stop_loss, shares_rec, risk_reward_score, market, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, scan_date) DO UPDATE SET
                price = excluded.price,
                turnover_m = excluded.turnover_m,
                blue_daily = excluded.blue_daily,
                blue_weekly = excluded.blue_weekly,
                blue_monthly = excluded.blue_monthly,
                adx = excluded.adx,
                volatility = excluded.volatility,
                is_heima = excluded.is_heima,
                is_juedi = excluded.is_juedi,
                strat_d_trend = excluded.strat_d_trend,
                strat_c_resonance = excluded.strat_c_resonance,
                legacy_signal = excluded.legacy_signal,
                regime = excluded.regime,
                adaptive_thresh = excluded.adaptive_thresh,
                vp_rating = excluded.vp_rating,
                profit_ratio = excluded.profit_ratio,
                wave_phase = excluded.wave_phase,
                wave_desc = excluded.wave_desc,
                chan_signal = excluded.chan_signal,
                chan_desc = excluded.chan_desc,
                market_cap = excluded.market_cap,
                cap_category = excluded.cap_category,
                company_name = excluded.company_name,
                industry = excluded.industry,
                stop_loss = excluded.stop_loss,
                shares_rec = excluded.shares_rec,
                risk_reward_score = excluded.risk_reward_score,
                market = excluded.market,
                updated_at = CURRENT_TIMESTAMP
        """, (
            result_dict.get('Symbol'),
            result_dict.get('Date'),
            result_dict.get('Price'),
            result_dict.get('Turnover_M'),
            result_dict.get('Blue_Daily'),
            result_dict.get('Blue_Weekly'),
            result_dict.get('Blue_Monthly'),
            result_dict.get('ADX'),
            result_dict.get('Volatility'),
            result_dict.get('Is_Heima'),
            result_dict.get('Is_Juedi') if 'Is_Juedi' in result_dict else result_dict.get('Is_Heima'),
            result_dict.get('Strat_D_Trend'),
            result_dict.get('Strat_C_Resonance'),
            result_dict.get('Legacy_Signal'),
            result_dict.get('Regime'),
            result_dict.get('Adaptive_Thresh'),
            result_dict.get('VP_Rating'),
            result_dict.get('Profit_Ratio'),
            result_dict.get('Wave_Phase'),
            result_dict.get('Wave_Desc'),
            result_dict.get('Chan_Signal'),
            result_dict.get('Chan_Desc'),
            result_dict.get('Market_Cap'),
            result_dict.get('Cap_Category'),
            result_dict.get('Company_Name'),
            result_dict.get('Industry'),
            result_dict.get('Stop_Loss'),
            result_dict.get('Shares_Rec'),
            result_dict.get('Risk_Reward_Score'),
            result_dict.get('Market', 'US')  # 默认 US
        ))


def bulk_insert_scan_results(results_list):
    """批量插入扫描结果"""
    for result in results_list:
        insert_scan_result(result)


def query_scan_results(scan_date=None, start_date=None, end_date=None, 
                       min_blue=None, symbols=None, market=None, limit=None):
    """查询扫描结果"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM scan_results WHERE 1=1"
        params = []
        
        if scan_date:
            query += " AND scan_date = ?"
            params.append(scan_date)
        
        if start_date:
            query += " AND scan_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND scan_date <= ?"
            params.append(end_date)
        
        if min_blue is not None:
            query += " AND blue_daily >= ?"
            params.append(min_blue)
        
        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        
        if market:
            query += " AND market = ?"
            params.append(market)
        
        query += " ORDER BY scan_date DESC, blue_daily DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_stock_history(symbol, limit=30):
    """获取单只股票的历史扫描记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM scan_results 
            WHERE symbol = ? 
            ORDER BY scan_date DESC 
            LIMIT ?
        """, (symbol, limit))
        return [dict(row) for row in cursor.fetchall()]


def get_scan_job(scan_date):
    """获取扫描任务状态"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scan_jobs WHERE scan_date = ?", (scan_date,))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_scan_job(scan_date, market='US'):
    """创建扫描任务"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO scan_jobs (scan_date, market, status, created_at)
            VALUES (?, ?, 'pending', CURRENT_TIMESTAMP)
        """, (scan_date, market))


def update_scan_job(scan_date, **kwargs):
    """更新扫描任务状态"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        set_clauses = []
        params = []
        
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        
        params.append(scan_date)
        
        cursor.execute(f"""
            UPDATE scan_jobs SET {', '.join(set_clauses)}
            WHERE scan_date = ?
        """, params)


def get_db_stats():
    """获取数据库统计信息"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM scan_results")
        total_records = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) as total FROM scan_results")
        total_symbols = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(DISTINCT scan_date) as total FROM scan_results")
        total_dates = cursor.fetchone()['total']
        
        cursor.execute("SELECT MIN(scan_date) as min_date, MAX(scan_date) as max_date FROM scan_results")
        date_range = cursor.fetchone()
        
        return {
            'total_records': total_records,
            'total_symbols': total_symbols,
            'total_dates': total_dates,
            'min_date': date_range['min_date'],
            'max_date': date_range['max_date']
        }


if __name__ == "__main__":
    # 初始化数据库
    init_db()
    print(get_db_stats())


