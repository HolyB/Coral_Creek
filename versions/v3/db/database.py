#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库连接和会话管理
支持 SQLite (本地) 和 Supabase (云端)
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date

# 导入 Supabase 层
try:
    from db.supabase_db import (
        is_supabase_available, 
        query_scan_results_supabase,
        get_scanned_dates_supabase,
        get_db_stats_supabase,
        insert_scan_result_supabase
    )
    SUPABASE_LAYER_AVAILABLE = True
except ImportError:
    SUPABASE_LAYER_AVAILABLE = False

# 避免重复刷屏日志
_SUPABASE_LAG_WARNED = False

# 检查是否使用 Supabase
USE_SUPABASE = os.environ.get('SUPABASE_URL') is not None

# 数据库文件路径 (SQLite 备用)
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "coral_creek.db")


def _is_malformed_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("malformed" in msg) or ("disk image is malformed" in msg)


def _quarantine_corrupted_db() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bad_path = f"{DB_PATH}.corrupt.{ts}"
    try:
        if os.path.exists(DB_PATH):
            os.replace(DB_PATH, bad_path)
        print(f"⚠️ 检测到损坏数据库，已隔离: {bad_path}")
        return bad_path
    except Exception as e:
        print(f"⚠️ 隔离损坏数据库失败: {e}")
        return ""


def get_connection():
    """获取数据库连接"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 返回字典格式
        # 快速健康检查，提前发现损坏库
        cur = conn.cursor()
        cur.execute("PRAGMA quick_check")
        chk = cur.fetchone()
        chk_txt = str(chk[0]).lower() if chk and len(chk) > 0 else "ok"
        if chk_txt != "ok":
            raise sqlite3.DatabaseError(f"quick_check failed: {chk_txt}")
        return conn
    except sqlite3.DatabaseError as e:
        if not _is_malformed_error(e):
            raise
        _quarantine_corrupted_db()
        # 重建一个新库，保证应用可启动
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e2:
            print(f"⚠️ 重建本地库失败，降级到内存库: {e2}")
            mem = sqlite3.connect(":memory:")
            mem.row_factory = sqlite3.Row
            return mem


@contextmanager
def get_db():
    """数据库连接上下文管理器"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
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
                heima_daily BOOLEAN,
                heima_weekly BOOLEAN,
                heima_monthly BOOLEAN,
                juedi_daily BOOLEAN,
                juedi_weekly BOOLEAN,
                juedi_monthly BOOLEAN,
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
                duokongwang_buy BOOLEAN,
                duokongwang_sell BOOLEAN,
                market_cap REAL,
                cap_category VARCHAR(30),
                company_name VARCHAR(200),
                industry VARCHAR(200),
                day_high REAL,
                day_low REAL,
                day_close REAL,
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
        
        # 添加新的黑马字段 (如果不存在)
        new_heima_cols = [
            ('heima_daily', 'BOOLEAN'),
            ('heima_weekly', 'BOOLEAN'),
            ('heima_monthly', 'BOOLEAN'),
            ('juedi_daily', 'BOOLEAN'),
            ('juedi_weekly', 'BOOLEAN'),
            ('juedi_monthly', 'BOOLEAN'),
        ]
        for col_name, col_type in new_heima_cols:
            try:
                cursor.execute(f"ALTER TABLE scan_results ADD COLUMN {col_name} {col_type}")
            except:
                pass  # 列已存在
        # 日线OHLC补充字段（用于更精确的KDJ/背离评估）
        for col_name, col_type in [("day_high", "REAL"), ("day_low", "REAL"), ("day_close", "REAL")]:
            try:
                cursor.execute(f"ALTER TABLE scan_results ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
        # 多空王买卖点标记
        for col_name, col_type in [("duokongwang_buy", "BOOLEAN"), ("duokongwang_sell", "BOOLEAN")]:
            try:
                cursor.execute(f"ALTER TABLE scan_results ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
        
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
        
        # 股票基本信息表 (缓存所有股票的名称、行业等)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol VARCHAR(20) PRIMARY KEY,
                name VARCHAR(200),
                industry VARCHAR(200),
                area VARCHAR(100),
                market VARCHAR(10),
                list_date VARCHAR(20),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_market ON stock_info(market)")
        
        # 信号性能缓存表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20),
                scan_date VARCHAR(20),
                market VARCHAR(10) DEFAULT 'US',
                return_5d REAL,
                return_10d REAL,
                return_20d REAL,
                max_gain REAL,
                max_drawdown REAL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, scan_date, market)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_date ON signal_performance(scan_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_market ON signal_performance(market)")

        # 候选信号持续追踪表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidate_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                market VARCHAR(10) DEFAULT 'US',
                signal_date DATE NOT NULL,
                source VARCHAR(50) DEFAULT 'daily_scan',
                signal_price REAL NOT NULL,
                current_price REAL,
                pnl_pct REAL DEFAULT 0,
                days_since_signal INTEGER DEFAULT 0,
                first_positive_day INTEGER,
                first_nonpositive_after_positive_day INTEGER,
                max_up_pct REAL DEFAULT 0,
                max_drawdown_pct REAL DEFAULT 0,
                pnl_d1 REAL,
                pnl_d3 REAL,
                pnl_d5 REAL,
                pnl_d10 REAL,
                pnl_d20 REAL,
                cap_category VARCHAR(30),
                industry VARCHAR(200),
                signal_tags TEXT,
                blue_daily REAL,
                blue_weekly REAL,
                blue_monthly REAL,
                heima_daily BOOLEAN,
                heima_weekly BOOLEAN,
                heima_monthly BOOLEAN,
                juedi_daily BOOLEAN,
                juedi_weekly BOOLEAN,
                juedi_monthly BOOLEAN,
                vp_rating VARCHAR(20),
                profit_ratio REAL,
                status VARCHAR(20) DEFAULT 'tracking',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, market, signal_date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_date ON candidate_tracking(signal_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_market ON candidate_tracking(market)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_status ON candidate_tracking(status)")
        try:
            cursor.execute("SELECT first_nonpositive_after_positive_day FROM candidate_tracking LIMIT 1")
        except Exception:
            cursor.execute("ALTER TABLE candidate_tracking ADD COLUMN first_nonpositive_after_positive_day INTEGER")
        for col_name in ["juedi_daily", "juedi_weekly", "juedi_monthly"]:
            try:
                cursor.execute(f"ALTER TABLE candidate_tracking ADD COLUMN {col_name} BOOLEAN")
            except Exception:
                pass
        
        # Baseline 扫描结果表 (用于对比)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baseline_scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                scan_date DATE NOT NULL,
                scan_time VARCHAR(10) DEFAULT 'post',  -- pre/mid/post
                market VARCHAR(10) DEFAULT 'US',
                price REAL,
                turnover_m REAL,
                blue_daily REAL,
                blue_weekly REAL,
                blue_days INTEGER,
                blue_weeks INTEGER,
                latest_day_blue REAL,
                latest_week_blue REAL,
                has_day_week_blue BOOLEAN,
                company_name VARCHAR(200),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, scan_date, market, scan_time)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_baseline_date ON baseline_scan_results(scan_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_baseline_market ON baseline_scan_results(market)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_baseline_symbol ON baseline_scan_results(symbol)")
        
        # 交易记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                market VARCHAR(10) DEFAULT 'US',
                trade_type VARCHAR(10) NOT NULL,
                price REAL NOT NULL,
                shares INTEGER NOT NULL,
                trade_date DATE NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)")
        
        # 持仓/关注列表表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                market VARCHAR(10) DEFAULT 'US',
                entry_date DATE NOT NULL,
                entry_price REAL NOT NULL,
                shares INTEGER DEFAULT 0,
                status VARCHAR(20) DEFAULT 'holding',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, market, entry_date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_status ON watchlist(status)")

        # 应用设置表（用于持久化 UI 偏好）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key VARCHAR(100) PRIMARY KEY,
                setting_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 迁移: 如果 market 列不存在，添加它
        try:
            cursor.execute("SELECT market FROM scan_results LIMIT 1")
        except sqlite3.OperationalError:
            print("🔄 Adding market column to scan_results...")
            cursor.execute("ALTER TABLE scan_results ADD COLUMN market VARCHAR(10) DEFAULT 'US'")
        
        print(f"✅ Database initialized at: {DB_PATH}")


def get_app_setting(setting_key, default=None):
    """读取应用设置值（字符串）"""
    def _query_setting():
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT setting_value FROM app_settings WHERE setting_key = ?",
                (setting_key,),
            )
            row = cursor.fetchone()
            return row["setting_value"] if row else default

    try:
        return _query_setting()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            init_db()
            return _query_setting()
        raise


def set_app_setting(setting_key, setting_value):
    """写入应用设置值（字符串）"""
    def _upsert_setting():
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO app_settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(setting_key) DO UPDATE SET
                    setting_value = excluded.setting_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (setting_key, setting_value),
            )

    try:
        _upsert_setting()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            init_db()
            _upsert_setting()
        else:
            raise


def get_scanned_dates(start_date=None, end_date=None, market=None):
    """获取已扫描的日期列表 - 优先使用 Supabase"""
    def _sqlite_dates():
        def _query_dates():
            with get_db() as conn:
                cursor = conn.cursor()

                query = "SELECT DISTINCT scan_date FROM scan_results WHERE 1=1"
                params = []

                if start_date:
                    query += " AND scan_date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND scan_date <= ?"
                    params.append(end_date)
                if market:
                    query += " AND market = ?"
                    params.append(market)

                query += " ORDER BY scan_date DESC"
                cursor.execute(query, params)
                return [row['scan_date'] for row in cursor.fetchall()]

        try:
            return _query_dates()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                # 首次运行或空 DB：先初始化表结构，再返回空列表
                try:
                    init_db()
                    return _query_dates()
                except Exception as e2:
                    print(f"⚠️ SQLite 初始化后仍无法读取扫描日期: {e2}")
                    return []
            raise

    # 优先使用 Supabase
    if USE_SUPABASE and SUPABASE_LAYER_AVAILABLE:
        try:
            dates = get_scanned_dates_supabase(start_date, end_date, market)
            # 容错: 若 Supabase 日期落后于本地 SQLite，则优先使用本地最新日期
            local_dates = _sqlite_dates()
            if dates and local_dates:
                if str(dates[0]) >= str(local_dates[0]):
                    return dates
                global _SUPABASE_LAG_WARNED
                if not _SUPABASE_LAG_WARNED:
                    print(f"⚠️ Supabase 日期落后 (supabase={dates[0]}, sqlite={local_dates[0]}), 回退 SQLite")
                    _SUPABASE_LAG_WARNED = True
                return local_dates
            if dates:
                return dates
            if local_dates:
                return local_dates
        except Exception as e:
            print(f"⚠️ Supabase 日期查询失败: {e}")

    # SQLite 备用
    return _sqlite_dates()


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
    """插入或更新单条扫描结果 - 同时写入 SQLite 和 Supabase"""
    # 1. 写入 SQLite (本地备份)
    with get_db() as conn:
        cursor = conn.cursor()
        
        # UPSERT: 如果存在则更新，不存在则插入
        cursor.execute("""
            INSERT INTO scan_results (
                symbol, scan_date, price, turnover_m, blue_daily, blue_weekly, blue_monthly,
                adx, volatility, is_heima, is_juedi, 
                heima_daily, heima_weekly, heima_monthly, juedi_daily, juedi_weekly, juedi_monthly,
                strat_d_trend, strat_c_resonance,
                legacy_signal, regime, adaptive_thresh, vp_rating, profit_ratio,
                wave_phase, wave_desc, chan_signal, chan_desc, duokongwang_buy, duokongwang_sell,
                market_cap, cap_category, company_name, industry, day_high, day_low, day_close, stop_loss, shares_rec, risk_reward_score, market, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                heima_daily = excluded.heima_daily,
                heima_weekly = excluded.heima_weekly,
                heima_monthly = excluded.heima_monthly,
                juedi_daily = excluded.juedi_daily,
                juedi_weekly = excluded.juedi_weekly,
                juedi_monthly = excluded.juedi_monthly,
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
                duokongwang_buy = excluded.duokongwang_buy,
                duokongwang_sell = excluded.duokongwang_sell,
                market_cap = excluded.market_cap,
                cap_category = excluded.cap_category,
                company_name = excluded.company_name,
                industry = excluded.industry,
                day_high = excluded.day_high,
                day_low = excluded.day_low,
                day_close = excluded.day_close,
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
            result_dict.get('Heima_Daily'),
            result_dict.get('Heima_Weekly'),
            result_dict.get('Heima_Monthly'),
            result_dict.get('Juedi_Daily'),
            result_dict.get('Juedi_Weekly'),
            result_dict.get('Juedi_Monthly'),
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
            result_dict.get('Duokongwang_Buy'),
            result_dict.get('Duokongwang_Sell'),
            result_dict.get('Market_Cap'),
            result_dict.get('Cap_Category'),
            result_dict.get('Company_Name'),
            result_dict.get('Industry'),
            result_dict.get('Day_High'),
            result_dict.get('Day_Low'),
            result_dict.get('Day_Close') if result_dict.get('Day_Close') is not None else result_dict.get('Price'),
            result_dict.get('Stop_Loss'),
            result_dict.get('Shares_Rec'),
            result_dict.get('Risk_Reward_Score'),
            result_dict.get('Market', 'US')  # 默认 US
        ))
    
    # 2. 同步写入 Supabase (云端)
    if USE_SUPABASE and SUPABASE_LAYER_AVAILABLE:
        try:
            insert_scan_result_supabase(result_dict)
        except Exception as e:
            pass  # Supabase 失败不影响主流程


def bulk_insert_scan_results(results_list):
    """批量插入扫描结果"""
    for result in results_list:
        insert_scan_result(result)


def query_scan_results(scan_date=None, start_date=None, end_date=None, 
                       min_blue=None, symbols=None, market=None, limit=None):
    """查询扫描结果 - 优先使用 Supabase"""
    # 优先使用 Supabase
    if USE_SUPABASE and SUPABASE_LAYER_AVAILABLE:
        try:
            results = query_scan_results_supabase(
                scan_date=scan_date,
                start_date=start_date,
                end_date=end_date,
                min_blue=min_blue,
                market=market,
                limit=limit
            )
            if results:
                return results
        except Exception as e:
            print(f"⚠️ Supabase 查询失败，回退到 SQLite: {e}")
    
    # SQLite 备用
    def _sqlite_query():
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

    try:
        return _sqlite_query()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            try:
                init_db()
                return _sqlite_query()
            except Exception as e2:
                print(f"⚠️ SQLite 初始化后仍无法查询 scan_results: {e2}")
                return []
        raise


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


def get_first_scan_dates(symbols, market='US'):
    """批量获取股票首次出现在扫描结果中的日期
    
    Args:
        symbols: 股票代码列表
        market: 市场 (US/CN)
    
    Returns:
        dict: {symbol: first_scan_date} 
    """
    if not symbols:
        return {}
    
    # 优先使用 Supabase
    if USE_SUPABASE and SUPABASE_LAYER_AVAILABLE:
        try:
            from db.supabase_db import get_first_scan_dates_supabase
            result = get_first_scan_dates_supabase(symbols, market)
            if result:
                return result
        except Exception as e:
            print(f"⚠️ Supabase 获取首次日期失败，回退 SQLite: {e}")
    
    # SQLite 备用
    with get_db() as conn:
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in symbols])
        cursor.execute(f"""
            SELECT symbol, MIN(scan_date) as first_date
            FROM scan_results 
            WHERE symbol IN ({placeholders}) AND market = ?
            GROUP BY symbol
        """, symbols + [market])
        
        result = {}
        for row in cursor.fetchall():
            result[row['symbol']] = row['first_date']
        return result


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
    """获取数据库统计信息 - 优先使用 Supabase"""
    # 优先使用 Supabase
    if USE_SUPABASE and SUPABASE_LAYER_AVAILABLE:
        try:
            stats = get_db_stats_supabase()
            if stats and stats.get('total_records', 0) > 0:
                return stats
        except Exception as e:
            print(f"⚠️ Supabase stats 失败: {e}")
    
    # SQLite 备用
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


# ========== 股票信息缓存 ==========

def upsert_stock_info(symbol, name, industry=None, area=None, market='US', list_date=None):
    """插入或更新股票基本信息"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO stock_info (symbol, name, industry, area, market, list_date, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol) DO UPDATE SET
                name = excluded.name,
                industry = excluded.industry,
                area = excluded.area,
                market = excluded.market,
                list_date = excluded.list_date,
                updated_at = CURRENT_TIMESTAMP
        """, (symbol, name, industry, area, market, list_date))


def bulk_upsert_stock_info(stock_list):
    """批量插入股票信息 - stock_list: [{'symbol': '', 'name': '', 'industry': '', 'market': ''}, ...]"""
    with get_db() as conn:
        cursor = conn.cursor()
        for stock in stock_list:
            cursor.execute("""
                INSERT INTO stock_info (symbol, name, industry, area, market, list_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    name = excluded.name,
                    industry = excluded.industry,
                    area = excluded.area,
                    market = excluded.market,
                    list_date = excluded.list_date,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                stock.get('symbol'),
                stock.get('name'),
                stock.get('industry'),
                stock.get('area'),
                stock.get('market', 'US'),
                stock.get('list_date')
            ))


def get_stock_info(symbol):
    """获取单只股票的基本信息"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stock_info WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_stock_info_batch(symbols):
    """批量获取股票信息"""
    if not symbols:
        return {}
    with get_db() as conn:
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in symbols])
        cursor.execute(f"SELECT * FROM stock_info WHERE symbol IN ({placeholders})", symbols)
        return {row['symbol']: dict(row) for row in cursor.fetchall()}


def get_stock_info_count(market=None):
    """获取股票信息数量"""
    with get_db() as conn:
        cursor = conn.cursor()
        if market:
            cursor.execute("SELECT COUNT(*) as cnt FROM stock_info WHERE market = ?", (market,))
        else:
            cursor.execute("SELECT COUNT(*) as cnt FROM stock_info")
        return cursor.fetchone()['cnt']


# ==================== Baseline 扫描结果操作 ====================

def save_baseline_results(results, scan_date, market='US', scan_time='post'):
    """批量保存 baseline 扫描结果"""
    if not results:
        return 0
    
    with get_db() as conn:
        cursor = conn.cursor()
        saved = 0
        
        for r in results:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO baseline_scan_results (
                        symbol, scan_date, scan_time, market, price, turnover_m,
                        blue_daily, blue_weekly, blue_days, blue_weeks,
                        latest_day_blue, latest_week_blue, has_day_week_blue, company_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.get('symbol'),
                    scan_date,
                    scan_time,
                    market,
                    r.get('price'),
                    r.get('turnover'),
                    r.get('blue_daily'),
                    r.get('blue_weekly'),
                    r.get('blue_days'),
                    r.get('blue_weeks'),
                    r.get('latest_day_blue'),
                    r.get('latest_week_blue'),
                    r.get('has_day_week_blue', True),
                    r.get('name', '')
                ))
                saved += 1
            except Exception as e:
                print(f"Error saving {r.get('symbol')}: {e}")
        
        return saved


def query_baseline_results(scan_date=None, market=None, limit=100):
    """查询 baseline 扫描结果"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM baseline_scan_results WHERE 1=1"
        params = []
        
        if scan_date:
            query += " AND scan_date = ?"
            params.append(scan_date)
        if market:
            query += " AND market = ?"
            params.append(market)
        
        query += " ORDER BY latest_day_blue DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def compare_scan_results(scan_date, market='US'):
    """比较同一天的 baseline 和常规扫描结果"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 获取 baseline 结果的 symbols
        cursor.execute("""
            SELECT symbol FROM baseline_scan_results 
            WHERE scan_date = ? AND market = ?
        """, (scan_date, market))
        baseline_symbols = set(row['symbol'] for row in cursor.fetchall())
        
        # 获取常规扫描结果的 symbols
        cursor.execute("""
            SELECT symbol FROM scan_results 
            WHERE scan_date = ? AND market = ?
        """, (scan_date, market))
        regular_symbols = set(row['symbol'] for row in cursor.fetchall())
        
        return {
            'baseline_only': list(baseline_symbols - regular_symbols),
            'regular_only': list(regular_symbols - baseline_symbols),
            'both': list(baseline_symbols & regular_symbols),
            'baseline_count': len(baseline_symbols),
            'regular_count': len(regular_symbols),
        }


# ==================== Signal Performance Cache ====================

def upsert_signal_performance(symbol: str, scan_date: str, market: str = 'US',
                              return_5d: float = None, return_10d: float = None, 
                              return_20d: float = None, max_gain: float = None,
                              max_drawdown: float = None):
    """插入或更新信号性能缓存"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signal_performance 
            (symbol, scan_date, market, return_5d, return_10d, return_20d, max_gain, max_drawdown, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, scan_date, market) DO UPDATE SET
                return_5d = excluded.return_5d,
                return_10d = excluded.return_10d,
                return_20d = excluded.return_20d,
                max_gain = excluded.max_gain,
                max_drawdown = excluded.max_drawdown,
                calculated_at = CURRENT_TIMESTAMP
        """, (symbol, scan_date, market, return_5d, return_10d, return_20d, max_gain, max_drawdown))


def bulk_upsert_signal_performance(performance_list: list):
    """批量插入信号性能数据"""
    for p in performance_list:
        upsert_signal_performance(
            symbol=p.get('symbol'),
            scan_date=p.get('scan_date'),
            market=p.get('market', 'US'),
            return_5d=p.get('return_5d'),
            return_10d=p.get('return_10d'),
            return_20d=p.get('return_20d'),
            max_gain=p.get('max_gain'),
            max_drawdown=p.get('max_drawdown')
        )


def query_signal_performance(start_date: str = None, end_date: str = None, 
                             market: str = None, limit: int = 1000):
    """查询信号性能缓存"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT sp.*, sr.blue_daily, sr.price, sr.turnover_m 
            FROM signal_performance sp
            LEFT JOIN scan_results sr ON sp.symbol = sr.symbol AND sp.scan_date = sr.scan_date
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND sp.scan_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND sp.scan_date <= ?"
            params.append(end_date)
        if market:
            query += " AND sp.market = ?"
            params.append(market)
        
        query += " ORDER BY sp.scan_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_signals_without_performance(market: str = 'US', min_days_old: int = 5, limit: int = 500):
    """获取没有性能缓存的信号（用于计算前向收益）"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 只选择至少 min_days_old 天前的信号（确保有足够的前向数据）
        cursor.execute("""
            SELECT sr.symbol, sr.scan_date, sr.price, sr.blue_daily, sr.market
            FROM scan_results sr
            LEFT JOIN signal_performance sp 
                ON sr.symbol = sp.symbol AND sr.scan_date = sp.scan_date AND sr.market = sp.market
            WHERE sp.id IS NULL
            AND sr.market = ?
            AND DATE(sr.scan_date) <= DATE('now', ? || ' days')
            ORDER BY sr.scan_date DESC
            LIMIT ?
        """, (market, f'-{min_days_old}', limit))
        
        return [dict(row) for row in cursor.fetchall()]


def get_performance_stats(market: str = None):
    """获取性能缓存统计"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if market:
            cursor.execute("""
                SELECT COUNT(*) as total,
                       AVG(return_5d) as avg_5d,
                       AVG(return_10d) as avg_10d,
                       AVG(return_20d) as avg_20d
                FROM signal_performance WHERE market = ?
            """, (market,))
        else:
            cursor.execute("""
                SELECT COUNT(*) as total,
                       AVG(return_5d) as avg_5d,
                       AVG(return_10d) as avg_10d,
                       AVG(return_20d) as avg_20d
                FROM signal_performance
            """)
        
        row = cursor.fetchone()
        return dict(row) if row else {}


# ==================== 交易和持仓操作 ====================

def get_signal_history(symbol, market='US', limit=50):
    """获取股票的历史信号记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT scan_date, price, turnover_m, blue_daily, blue_weekly, blue_monthly,
                   is_heima, is_juedi, wave_phase, chan_signal
            FROM scan_results 
            WHERE symbol = ? AND market = ?
            ORDER BY scan_date DESC
            LIMIT ?
        """, (symbol, market, limit))
        return [dict(row) for row in cursor.fetchall()]


def add_trade(symbol, trade_type, price, shares, trade_date, market='US', notes=''):
    """添加交易记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (symbol, market, trade_type, price, shares, trade_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, market, trade_type.upper(), price, shares, trade_date, notes))
        return cursor.lastrowid


def add_to_watchlist(symbol, entry_price, shares=0, entry_date=None, market='US', status='holding', notes=''):
    """添加股票到持仓/关注列表"""
    if entry_date is None:
        entry_date = datetime.now().strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO watchlist (symbol, market, entry_date, entry_price, shares, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, market, entry_date, entry_price, shares, status, notes))
        return cursor.lastrowid


def get_portfolio(status='holding', market=None):
    """获取持仓列表"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM watchlist WHERE status = ?"
        params = [status]
        
        if market:
            query += " AND market = ?"
            params.append(market)
        
        query += " ORDER BY entry_date DESC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_trades(symbol=None, market=None, limit=100):
    """获取交易记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if market:
            query += " AND market = ?"
            params.append(market)
        
        query += " ORDER BY trade_date DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def update_watchlist_status(symbol, entry_date, new_status, market='US'):
    """更新持仓状态 (holding -> sold)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE watchlist SET status = ? 
            WHERE symbol = ? AND entry_date = ? AND market = ?
        """, (new_status, symbol, entry_date, market))
        return cursor.rowcount


def delete_from_watchlist(symbol, entry_date, market='US'):
    """从持仓列表删除"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM watchlist WHERE symbol = ? AND entry_date = ? AND market = ?
        """, (symbol, entry_date, market))
        return cursor.rowcount


# ==================== 博主推荐追踪 ====================

def init_blogger_tables():
    """初始化博主推荐相关表"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 博主表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bloggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                platform TEXT,
                specialty TEXT,
                url TEXT,
                avatar_url TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 推荐记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blogger_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                market TEXT DEFAULT 'CN',
                rec_date DATE NOT NULL,
                rec_price REAL,
                rec_type TEXT DEFAULT 'BUY',
                target_price REAL,
                stop_loss REAL,
                portfolio_tag TEXT,
                notes TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (blogger_id) REFERENCES bloggers(id)
            )
        """)

        # 兼容旧库：补 portfolio_tag 字段
        try:
            cursor.execute("SELECT portfolio_tag FROM recommendations LIMIT 1")
        except Exception:
            cursor.execute("ALTER TABLE recommendations ADD COLUMN portfolio_tag TEXT")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_blogger ON recommendations(blogger_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_date ON recommendations(rec_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker)")
        
        print("✅ Blogger tables initialized")


def add_blogger(name, platform=None, specialty=None, url=None, notes=None):
    """添加博主"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bloggers (name, platform, specialty, url, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (name, platform, specialty, url, notes))
        return cursor.lastrowid


def get_all_bloggers():
    """获取所有博主"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bloggers ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]


def get_blogger(blogger_id):
    """获取单个博主"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bloggers WHERE id = ?", (blogger_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_blogger(blogger_id, **kwargs):
    """更新博主信息"""
    with get_db() as conn:
        cursor = conn.cursor()
        set_clauses = [f"{k} = ?" for k in kwargs.keys()]
        params = list(kwargs.values()) + [blogger_id]
        cursor.execute(f"""
            UPDATE bloggers SET {', '.join(set_clauses)} WHERE id = ?
        """, params)
        return cursor.rowcount


def delete_blogger(blogger_id):
    """删除博主"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM recommendations WHERE blogger_id = ?", (blogger_id,))
        cursor.execute("DELETE FROM bloggers WHERE id = ?", (blogger_id,))
        return cursor.rowcount


def add_recommendation(blogger_id, ticker, rec_date, market='CN', rec_price=None,
                       rec_type='BUY', target_price=None, stop_loss=None,
                       portfolio_tag=None, notes=None, source_url=None):
    """添加推荐记录"""
    def _insert_once():
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recommendations (blogger_id, ticker, market, rec_date, rec_price, 
                                             rec_type, target_price, stop_loss, portfolio_tag, notes, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (blogger_id, ticker.upper(), market, rec_date, rec_price,
                  rec_type, target_price, stop_loss, portfolio_tag, notes, source_url))
            return cursor.lastrowid

    try:
        return _insert_once()
    except sqlite3.OperationalError as e:
        if "no such column" in str(e).lower() or "no such table" in str(e).lower():
            init_blogger_tables()
            return _insert_once()
        raise


def get_recommendations(blogger_id=None, ticker=None, market=None, start_date=None, end_date=None, portfolio_tag=None, limit=100):
    """查询推荐记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT r.*, b.name as blogger_name, b.platform
            FROM recommendations r
            JOIN bloggers b ON r.blogger_id = b.id
            WHERE 1=1
        """
        params = []
        
        if blogger_id:
            query += " AND r.blogger_id = ?"
            params.append(blogger_id)
        if ticker:
            query += " AND r.ticker = ?"
            params.append(ticker.upper())
        if market:
            query += " AND r.market = ?"
            params.append(market)
        if start_date:
            query += " AND r.rec_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND r.rec_date <= ?"
            params.append(end_date)
        if portfolio_tag:
            query += " AND r.portfolio_tag = ?"
            params.append(portfolio_tag)
        
        query += " ORDER BY r.rec_date DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor.execute(query, params)
        except sqlite3.OperationalError as e:
            if "no such column" in str(e).lower() or "no such table" in str(e).lower():
                init_blogger_tables()
                cursor.execute(query, params)
            else:
                raise
        return [dict(row) for row in cursor.fetchall()]


def delete_recommendation(rec_id):
    """删除推荐记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM recommendations WHERE id = ?", (rec_id,))
        return cursor.rowcount


def get_blogger_stats(blogger_id=None):
    """获取博主统计信息 (推荐数量、涵盖日期范围)"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if blogger_id:
            cursor.execute("""
                SELECT b.id, b.name, b.platform,
                       COUNT(r.id) as rec_count,
                       MIN(r.rec_date) as first_rec,
                       MAX(r.rec_date) as last_rec
                FROM bloggers b
                LEFT JOIN recommendations r ON b.id = r.blogger_id
                WHERE b.id = ?
                GROUP BY b.id
            """, (blogger_id,))
        else:
            cursor.execute("""
                SELECT b.id, b.name, b.platform,
                       COUNT(r.id) as rec_count,
                       MIN(r.rec_date) as first_rec,
                       MAX(r.rec_date) as last_rec
                FROM bloggers b
                LEFT JOIN recommendations r ON b.id = r.blogger_id
                GROUP BY b.id
                ORDER BY rec_count DESC
            """)
        
        return [dict(row) for row in cursor.fetchall()]


if __name__ == "__main__":
    # 初始化数据库
    init_db()
    init_blogger_tables()
    print(get_db_stats())
