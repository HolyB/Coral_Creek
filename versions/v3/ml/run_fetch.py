#!/usr/bin/env python
"""快速批量拉取脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ml.fetch_history import fetch_stock_data
from db.stock_history import save_stock_history, init_history_db, get_history_stats
from db.database import get_connection
import time

def main():
    init_history_db()
    
    # 获取股票列表
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT symbol, COUNT(*) as cnt
        FROM scan_results
        WHERE market = "US"
        GROUP BY symbol
        ORDER BY cnt DESC
        LIMIT 200
    ''')
    symbols = [row['symbol'] for row in cursor.fetchall()]
    conn.close()
    
    print(f'待拉取: {len(symbols)} 只股票')
    
    success = 0
    fail = 0
    
    for i, symbol in enumerate(symbols):
        df = fetch_stock_data(symbol, days=250)
        
        if df is not None and len(df) >= 60:
            save_stock_history(symbol, 'US', df)
            success += 1
        else:
            fail += 1
        
        if (i + 1) % 20 == 0:
            print(f'[{i+1}/{len(symbols)}] 成功: {success}, 失败: {fail}')
        
        time.sleep(0.15)
    
    print(f'\n完成! 成功: {success}, 失败: {fail}')
    stats = get_history_stats()
    print(f'数据库: {stats["total_symbols"]} 股票, {stats["total_records"]} 条记录')

if __name__ == "__main__":
    main()
