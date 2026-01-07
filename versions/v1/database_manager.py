import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

class StockDatabase:
    def __init__(self, db_name="stock_signals.db"):
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_name)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        
        # 扫描结果表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            market TEXT NOT NULL,  -- 'CN' or 'US'
            price REAL,
            turnover REAL,
            blue_daily REAL,
            blue_days INTEGER,
            blue_weekly REAL,
            blue_weeks INTEGER,
            has_day_blue BOOLEAN,
            has_week_blue BOOLEAN,
            has_heima BOOLEAN,
            signals_summary TEXT,
            day_blue_dates TEXT,  -- JSON格式存储日线BLUE信号日期列表
            week_blue_dates TEXT,  -- JSON格式存储周线BLUE信号日期列表
            heima_dates TEXT,  -- JSON格式存储黑马信号日期列表
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(scan_date, symbol)
        )
        ''')
        
        # 添加新字段（如果表已存在，使用ALTER TABLE）
        try:
            cursor.execute('ALTER TABLE scan_results ADD COLUMN day_blue_dates TEXT')
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        try:
            cursor.execute('ALTER TABLE scan_results ADD COLUMN week_blue_dates TEXT')
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        try:
            cursor.execute('ALTER TABLE scan_results ADD COLUMN heima_dates TEXT')
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        # Create an index for faster querying by date
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_date ON scan_results (scan_date)')
        
        # 自选股表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_favorites (
            symbol TEXT PRIMARY KEY,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
        ''')
        
        self.conn.commit()

    def add_favorite(self, symbol, note=""):
        """添加自选股"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO user_favorites (symbol, note)
            VALUES (?, ?)
            ''', (symbol, note))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding favorite {symbol}: {e}")
            return False

    def remove_favorite(self, symbol):
        """移除自选股"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM user_favorites WHERE symbol = ?', (symbol,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error removing favorite {symbol}: {e}")
            return False

    def get_all_favorites(self):
        """获取所有自选股"""
        try:
            df = pd.read_sql_query("SELECT * FROM user_favorites", self.conn)
            return df
        except Exception as e:
            print(f"Error getting favorites: {e}")
            return pd.DataFrame()

    def is_favorite(self, symbol):
        """检查是否为自选股"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1 FROM user_favorites WHERE symbol = ?', (symbol,))
        return cursor.fetchone() is not None

    def save_results_from_df(self, df, market='CN'):
        """
        将 DataFrame 结果保存到数据库
        """
        if df.empty:
            return
            
        scan_date = datetime.now().strftime("%Y-%m-%d")
        cursor = self.conn.cursor()
        
        count = 0
        for _, row in df.iterrows():
            # Construct signals summary
            signals = []
            if row.get('has_day_blue'):
                signals.append(f"日BLUE({row.get('blue_days', 0)}天)")
            if row.get('has_week_blue'):
                signals.append(f"周BLUE({row.get('blue_weeks', 0)}周)")
            
            has_heima = False
            if row.get('has_day_heima') or row.get('has_week_heima'):
                has_heima = True
                signals.append("黑马信号")
                
            signals_str = ",".join(signals)

            # 获取信号日期列表（JSON格式）
            day_blue_dates = json.dumps(row.get('day_blue_dates', []), ensure_ascii=False) if row.get('day_blue_dates') else None
            week_blue_dates = json.dumps(row.get('week_blue_dates', []), ensure_ascii=False) if row.get('week_blue_dates') else None
            heima_dates = json.dumps(row.get('heima_dates', []), ensure_ascii=False) if row.get('heima_dates') else None
            
            try:
                cursor.execute('''
                INSERT OR REPLACE INTO scan_results (
                    scan_date, symbol, name, market, price, turnover,
                    blue_daily, blue_days, blue_weekly, blue_weeks, has_day_blue, has_week_blue, has_heima,
                    signals_summary, day_blue_dates, week_blue_dates, heima_dates
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    scan_date,
                    str(row['symbol']),
                    row.get('name', str(row['symbol'])),
                    market,
                    float(row['price']),
                    float(row.get('turnover', 0)),
                    float(row.get('blue_daily', 0)),
                    int(row.get('blue_days', 0)),
                    float(row.get('blue_weekly', 0)),
                    int(row.get('blue_weeks', 0)),
                    bool(row.get('has_day_blue', False)),
                    bool(row.get('has_week_blue', False)),
                    has_heima,
                    signals_str,
                    day_blue_dates,
                    week_blue_dates,
                    heima_dates
                ))
                count += 1
            except Exception as e:
                print(f"Error saving {row['symbol']} to DB: {e}")
                
        self.conn.commit()
        print(f"成功保存 {count} 条记录到数据库 ({self.db_path})")

    def get_results_by_date(self, date_str):
        """
        获取指定日期的结果
        """
        return pd.read_sql_query("SELECT * FROM scan_results WHERE scan_date = ?", self.conn, params=(date_str,))

    def get_available_dates(self):
        """
        获取所有有数据的日期
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT scan_date FROM scan_results ORDER BY scan_date DESC")
        return [row[0] for row in cursor.fetchall()]
    
    def get_stock_history(self, symbol, market=None):
        """
        获取某只股票的历史扫描记录
        """
        query = "SELECT * FROM scan_results WHERE symbol = ?"
        params = [symbol]
        
        if market:
            query += " AND market = ?"
            params.append(market)
        
        query += " ORDER BY scan_date DESC"
        return pd.read_sql_query(query, self.conn, params=params)
    
    def compare_dates(self, date1, date2, market=None):
        """
        对比两个日期的扫描结果
        返回：新增股票、消失股票、持续股票
        """
        df1 = self.get_results_by_date(date1)
        df2 = self.get_results_by_date(date2)
        
        if market:
            df1 = df1[df1['market'] == market]
            df2 = df2[df2['market'] == market]
        
        symbols1 = set(df1['symbol'].unique())
        symbols2 = set(df2['symbol'].unique())
        
        new_stocks = symbols2 - symbols1  # 新出现的股票
        disappeared_stocks = symbols1 - symbols2  # 消失的股票
        persistent_stocks = symbols1 & symbols2  # 持续存在的股票
        
        return {
            'new': new_stocks,
            'disappeared': disappeared_stocks,
            'persistent': persistent_stocks,
            'date1': date1,
            'date2': date2,
            'df1': df1,
            'df2': df2
        }

    def close(self):
        self.conn.close()


