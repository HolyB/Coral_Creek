"""
股票历史K线数据存储
Stock History Data Store

功能:
- 存储每只股票的历史 OHLCV 数据
- 支持增量更新
- 为 ML 模型提供数据
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Optional, List
import sqlite3
from pathlib import Path


def get_history_db_path() -> str:
    """获取历史数据数据库路径 (单独文件，避免主库过大)"""
    db_dir = Path(__file__).parent
    return str(db_dir / "stock_history.db")


def init_history_db():
    """初始化历史数据库"""
    conn = sqlite3.connect(get_history_db_path())
    cursor = conn.cursor()
    
    # K线历史表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(20) NOT NULL,
            market VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            turnover REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, market, trade_date)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_symbol ON stock_history(symbol, market)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_date ON stock_history(trade_date)")
    
    # 特征存储表 (每次扫描时的完整特征)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(20) NOT NULL,
            market VARCHAR(10) NOT NULL,
            scan_date DATE NOT NULL,
            
            -- 价格特征
            close REAL,
            open REAL,
            high REAL,
            low REAL,
            volume REAL,
            
            -- 均线
            ma_5 REAL,
            ma_10 REAL,
            ma_20 REAL,
            ma_60 REAL,
            ma_120 REAL,
            ma_250 REAL,
            
            -- 均线偏离
            ma_5_bias REAL,
            ma_20_bias REAL,
            ma_60_bias REAL,
            
            -- 动量
            return_1d REAL,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            return_60d REAL,
            
            -- 波动率
            volatility_5d REAL,
            volatility_20d REAL,
            volatility_60d REAL,
            atr_14 REAL,
            
            -- 技术指标
            rsi_6 REAL,
            rsi_14 REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            kdj_k REAL,
            kdj_d REAL,
            kdj_j REAL,
            
            -- 布林带
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            bb_width REAL,
            bb_position REAL,
            
            -- 成交量
            volume_ma_5 REAL,
            volume_ma_20 REAL,
            volume_ratio REAL,
            
            -- BLUE 信号
            blue_daily REAL,
            blue_weekly REAL,
            blue_monthly REAL,
            is_heima INTEGER,
            is_juedi INTEGER,
            
            -- 标签 (训练用，需要回填)
            return_1d_future REAL,
            return_5d_future REAL,
            return_10d_future REAL,
            return_30d_future REAL,
            return_60d_future REAL,
            max_drawdown_5d REAL,
            max_drawdown_30d REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, market, scan_date)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feature_symbol ON feature_store(symbol, market)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feature_date ON feature_store(scan_date)")
    
    conn.commit()
    conn.close()
    print("✅ 历史数据库初始化完成")


def save_stock_history(symbol: str, market: str, df: pd.DataFrame) -> int:
    """
    保存股票历史数据
    
    Args:
        symbol: 股票代码
        market: 市场
        df: DataFrame with Date, Open, High, Low, Close, Volume
    
    Returns:
        保存的记录数
    """
    if df.empty:
        return 0
    
    conn = sqlite3.connect(get_history_db_path())
    
    # 准备数据
    records = []
    for _, row in df.iterrows():
        trade_date = row.get('Date') or row.name
        if isinstance(trade_date, pd.Timestamp):
            trade_date = trade_date.strftime('%Y-%m-%d')
        elif isinstance(trade_date, str):
            pass
        else:
            trade_date = str(trade_date)
        
        records.append((
            symbol, market, trade_date,
            row.get('Open'), row.get('High'), row.get('Low'), 
            row.get('Close'), row.get('Volume'), row.get('Turnover')
        ))
    
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO stock_history 
        (symbol, market, trade_date, open, high, low, close, volume, turnover)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    inserted = cursor.rowcount
    conn.close()
    
    return len(records)


def get_stock_history(symbol: str, market: str, days: int = 250) -> pd.DataFrame:
    """
    获取股票历史数据
    
    Args:
        symbol: 股票代码
        market: 市场
        days: 获取天数
    
    Returns:
        DataFrame with OHLCV
    """
    conn = sqlite3.connect(get_history_db_path())
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days + 100)  # 多取一些以防节假日
    
    query = """
        SELECT trade_date as Date, open as Open, high as High, 
               low as Low, close as Close, volume as Volume
        FROM stock_history
        WHERE symbol = ? AND market = ?
          AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(
        symbol, market, start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d'), days
    ))
    
    conn.close()
    
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def get_history_stats() -> dict:
    """获取历史数据库统计"""
    conn = sqlite3.connect(get_history_db_path())
    cursor = conn.cursor()
    
    stats = {}
    
    # 股票数量
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stock_history")
    stats['total_symbols'] = cursor.fetchone()[0]
    
    # 记录数
    cursor.execute("SELECT COUNT(*) FROM stock_history")
    stats['total_records'] = cursor.fetchone()[0]
    
    # 日期范围
    cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM stock_history")
    row = cursor.fetchone()
    stats['date_range'] = (row[0], row[1])
    
    # 按市场统计
    cursor.execute("""
        SELECT market, COUNT(DISTINCT symbol), COUNT(*) 
        FROM stock_history GROUP BY market
    """)
    stats['by_market'] = {row[0]: {'symbols': row[1], 'records': row[2]} 
                         for row in cursor.fetchall()}
    
    conn.close()
    return stats


# === 初始化 ===
init_history_db()


if __name__ == "__main__":
    print("=== Stock History DB 测试 ===\n")
    
    stats = get_history_stats()
    print(f"股票数: {stats['total_symbols']}")
    print(f"记录数: {stats['total_records']}")
    print(f"日期范围: {stats['date_range']}")
    print(f"按市场: {stats['by_market']}")
