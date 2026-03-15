#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多策略模拟盘交易系统 (Multi-Strategy Paper Trading)
====================================================
每个策略独立运行：独立账户 ($100k)、独立持仓、独立 NAV 跟踪。

6 个策略:
  - all_in:       买入所有 picks，5% 单笔上限
  - top1_daily:   每天只买预测最高的 1 只，10% 仓位
  - top3_daily:   每天只买预测最高的 3 只，10% 仓位
  - top10pct:     每天只买预测排名前 10% 的 picks
  - streak_only:  只买连续出现 ≥2 天的股票
  - large_cap:    只买中/大/超大市值的 picks

用法:
    PYTHONPATH=. python scripts/paper_trading.py --market US
    PYTHONPATH=. python scripts/paper_trading.py --market US --report
    PYTHONPATH=. python scripts/paper_trading.py --market US --compare
"""
import os, sys, sqlite3, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

DB_PATH = os.path.join(parent_dir, 'db', 'paper_trading.db')
INITIAL_CAPITAL = 100_000
HOLD_DAYS = 10

# Import strategies from portfolio tracker
from scripts.ml_portfolio_tracker import STRATEGIES


def init_db():
    """初始化模拟盘数据库 (v2: per-strategy)"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS paper_account_v2 (
            id INTEGER PRIMARY KEY,
            market TEXT,
            strategy_key TEXT,
            capital REAL,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(market, strategy_key)
        );

        CREATE TABLE IF NOT EXISTS paper_positions_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT,
            strategy_key TEXT,
            symbol TEXT,
            buy_date TEXT,
            buy_price REAL,
            shares REAL,
            amount REAL,
            pred_score REAL,
            status TEXT DEFAULT 'open',
            sell_date TEXT,
            sell_price REAL,
            pnl REAL,
            pnl_pct REAL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_daily_nav_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT,
            strategy_key TEXT,
            date TEXT,
            cash REAL,
            unrealized REAL,
            total_nav REAL,
            n_positions INTEGER,
            created_at TEXT,
            UNIQUE(market, strategy_key, date)
        );
    ''')
    conn.commit()
    return conn


def get_account(conn, market, strategy_key):
    """获取或创建策略账户"""
    row = conn.execute(
        'SELECT capital FROM paper_account_v2 WHERE market=? AND strategy_key=?',
        (market, strategy_key)
    ).fetchone()
    if row:
        return row[0]
    conn.execute(
        'INSERT INTO paper_account_v2 (market, strategy_key, capital, created_at, updated_at) VALUES (?,?,?,?,?)',
        (market, strategy_key, INITIAL_CAPITAL, datetime.now().isoformat(), datetime.now().isoformat())
    )
    conn.commit()
    return INITIAL_CAPITAL


def _get_price(symbol, market):
    """Get latest price for a symbol"""
    try:
        from db.stock_history import get_stock_history
        hist = get_stock_history(symbol, market, days=5)
        if hist is not None and not hist.empty:
            return float(hist.iloc[-1]['Close'])
    except:
        pass
    # Fallback: try partial_stock_history
    db_dir = os.path.join(parent_dir, 'db')
    for db_name in ['stock_history.db', 'partial_stock_history.db']:
        db_path = os.path.join(db_dir, db_name)
        if os.path.exists(db_path):
            try:
                c = sqlite3.connect(db_path)
                row = c.execute(
                    'SELECT close FROM stock_history WHERE symbol=? AND market=? ORDER BY trade_date DESC LIMIT 1',
                    (symbol, market)
                ).fetchone()
                c.close()
                if row:
                    return row[0]
            except:
                pass
    return None


def close_expired(conn, market, strategy_key, today):
    """结算到期仓位"""
    positions = conn.execute(
        'SELECT id, symbol, buy_date, buy_price, shares, amount FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status=?',
        (market, strategy_key, 'open')
    ).fetchall()

    closed = 0
    capital_return = 0

    for pid, symbol, buy_date, buy_price, shares, amount in positions:
        age = (pd.Timestamp(today) - pd.Timestamp(buy_date)).days
        if age < HOLD_DAYS:
            continue

        sell_price = _get_price(symbol, market) or buy_price
        pnl = (sell_price - buy_price) * shares
        pnl_pct = (sell_price / buy_price - 1) * 100

        conn.execute(
            '''UPDATE paper_positions_v2 SET status='closed', sell_date=?, sell_price=?, pnl=?, pnl_pct=? WHERE id=?''',
            (today, sell_price, pnl, pnl_pct, pid)
        )
        capital_return += amount + pnl
        closed += 1

    if closed > 0:
        conn.execute(
            'UPDATE paper_account_v2 SET capital=capital+?, updated_at=? WHERE market=? AND strategy_key=?',
            (capital_return, datetime.now().isoformat(), market, strategy_key)
        )
        conn.commit()

    return closed


def open_positions(conn, market, strategy_key, today):
    """按策略规则开仓"""
    from scripts.ml_daily_scorer import score_daily_signals

    strat = STRATEGIES[strategy_key]
    capital = get_account(conn, market, strategy_key)
    max_per_pos = strat['max_per_pos']
    daily_limit = strat.get('daily_pick_limit')
    filter_fn_name = strat.get('filter_fn')

    # Existing open symbols for this strategy
    open_syms = set(r[0] for r in conn.execute(
        'SELECT symbol FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status=?',
        (market, strategy_key, 'open')
    ).fetchall())

    # Score today's signals — returns {tier: [picks]}
    all_results = score_daily_signals(market=market, date=today, top_n=10)
    if not all_results:
        return 0

    # Flatten all picks from all tiers, sorted by score
    all_picks = []
    for tier, picks in all_results.items():
        if isinstance(picks, list):
            for p in picks:
                p['_tier'] = tier
                all_picks.append(p)
    all_picks.sort(key=lambda x: x.get('pred_10d', x.get('score', 0)), reverse=True)

    # Apply strategy filters
    if filter_fn_name == 'top10pct':
        n = max(1, len(all_picks) // 10)
        all_picks = all_picks[:n]
    elif filter_fn_name == 'streak_ge2':
        # Need streak info — check if pick appeared yesterday too
        from scripts.ml_portfolio_tracker import _get_all_picks, compute_signal_streaks
        try:
            picks_df = _get_all_picks(market)
            streaks = compute_signal_streaks(picks_df)
            all_picks = [p for p in all_picks if streaks.get((p['symbol'], today), 0) >= 2]
        except:
            pass  # If streak computation fails, allow all
    elif filter_fn_name == 'large_cap':
        from scripts.ml_daily_scorer import load_market_data
        mcap, _ = load_market_data(market)
        min_cap = 2e9 if market == 'US' else 1e10
        all_picks = [p for p in all_picks if mcap.get(p['symbol'], 0) >= min_cap]

    # Apply daily pick limit
    if daily_limit:
        all_picks = all_picks[:daily_limit]

    opened = 0
    for pick in all_picks:
        sym = pick['symbol']
        if sym in open_syms:
            continue

        pos_size = capital * max_per_pos
        if pos_size < 100 or capital < pos_size:
            break

        price = pick.get('price', 0)
        if price <= 0:
            continue
        shares = pos_size / price

        conn.execute(
            '''INSERT INTO paper_positions_v2
            (market, strategy_key, symbol, buy_date, buy_price, shares, amount, pred_score, status, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (market, strategy_key, sym, today, price, shares, pos_size,
             pick.get('pred_10d', pick.get('score', 0)), 'open', datetime.now().isoformat())
        )
        capital -= pos_size
        conn.execute(
            'UPDATE paper_account_v2 SET capital=?, updated_at=? WHERE market=? AND strategy_key=?',
            (capital, datetime.now().isoformat(), market, strategy_key)
        )
        open_syms.add(sym)
        opened += 1

    conn.commit()
    return opened


def record_nav(conn, market, strategy_key, today):
    """记录每日净值"""
    capital = get_account(conn, market, strategy_key)

    positions = conn.execute(
        'SELECT symbol, buy_price, shares, amount FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status=?',
        (market, strategy_key, 'open')
    ).fetchall()

    unrealized = 0
    for sym, buy_price, shares, amount in positions:
        price = _get_price(sym, market)
        if price:
            unrealized += price * shares
        else:
            unrealized += amount

    total_nav = capital + unrealized

    conn.execute(
        '''INSERT OR REPLACE INTO paper_daily_nav_v2
        (market, strategy_key, date, cash, unrealized, total_nav, n_positions, created_at)
        VALUES (?,?,?,?,?,?,?,?)''',
        (market, strategy_key, today, capital, unrealized, total_nav, len(positions),
         datetime.now().isoformat())
    )
    conn.commit()
    return total_nav


def run_daily(market='US'):
    """每日执行: 对所有策略执行 结算→开仓→记录净值"""
    from db.database import init_db as init_scan_db, get_scanned_dates
    init_scan_db()

    dates = get_scanned_dates(market=market)
    if not dates:
        print("❌ 无扫描数据")
        return

    today = dates[0]
    print(f"\n🤖 多策略模拟盘 ({market}): {today}")
    print(f"{'='*60}")

    conn = init_db()

    for sk, strat in STRATEGIES.items():
        print(f"\n  📋 {strat['name']} ({sk})")
        capital = get_account(conn, market, sk)
        print(f"     💰 Capital: ${capital:,.0f}")

        # 1. Close expired
        closed = close_expired(conn, market, sk, today)
        if closed:
            print(f"     📤 结算 {closed} 笔")

        # 2. Open new
        try:
            opened = open_positions(conn, market, sk, today)
            if opened:
                print(f"     📥 开仓 {opened} 笔")
        except Exception as e:
            print(f"     ⚠️ 开仓失败: {e}")
            opened = 0

        # 3. Record NAV
        nav = record_nav(conn, market, sk, today)
        ret = (nav / INITIAL_CAPITAL - 1) * 100
        print(f"     📊 NAV: ${nav:,.0f} ({ret:+.1f}%)")

    conn.close()
    print(f"\n{'='*60}")
    print("✅ 所有策略执行完毕")


def get_best_strategy(market='US'):
    """返回收益最高的策略 key"""
    if not os.path.exists(DB_PATH):
        return 'top3_daily'  # default
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            '''SELECT strategy_key, total_nav FROM paper_daily_nav_v2
               WHERE market=? AND date=(SELECT MAX(date) FROM paper_daily_nav_v2 WHERE market=?)
               ORDER BY total_nav DESC''',
            (market, market)
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return 'top3_daily'
    conn.close()
    if rows:
        return rows[0][0]
    return 'top3_daily'


def show_compare(market='US'):
    """显示策略对比表"""
    if not os.path.exists(DB_PATH):
        print("📊 暂无数据")
        return
    conn = sqlite3.connect(DB_PATH)

    print(f"\n{'='*70}")
    print(f"📊 策略对比 ({market})")
    print(f"{'='*70}")
    print(f"{'策略':<15} {'NAV':>12} {'收益':>8} {'胜率':>6} {'交易':>5} {'持仓':>5}")
    print(f"{'-'*70}")

    best_key = None
    best_nav = 0

    for sk in STRATEGIES:
        # Latest NAV
        nav_row = conn.execute(
            'SELECT total_nav, n_positions FROM paper_daily_nav_v2 WHERE market=? AND strategy_key=? ORDER BY date DESC LIMIT 1',
            (market, sk)
        ).fetchone()
        if not nav_row:
            continue

        nav, n_pos = nav_row
        ret = (nav / INITIAL_CAPITAL - 1) * 100

        if nav > best_nav:
            best_nav = nav
            best_key = sk

        # Trade stats
        trades = conn.execute(
            "SELECT COUNT(*), AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status='closed'",
            (market, sk)
        ).fetchone()
        n_trades = trades[0] or 0
        win_rate = (trades[1] or 0) * 100

        marker = ' ⭐' if sk == best_key else ''
        name = STRATEGIES[sk]['name']
        print(f"  {name:<13} ${nav:>10,.0f} {ret:>+7.1f}% {win_rate:>5.0f}% {n_trades:>5} {n_pos:>5}{marker}")

    conn.close()
    if best_key:
        print(f"\n  🏆 最优策略: {STRATEGIES[best_key]['name']} ({best_key})")


def show_report(market='US', strategy_key=None):
    """显示单个策略详细报告"""
    if not os.path.exists(DB_PATH):
        print("📊 暂无数据")
        return
    conn = sqlite3.connect(DB_PATH)

    keys = [strategy_key] if strategy_key else list(STRATEGIES.keys())

    for sk in keys:
        navs = pd.read_sql(
            'SELECT date, total_nav, n_positions FROM paper_daily_nav_v2 WHERE market=? AND strategy_key=? ORDER BY date',
            conn, params=[market, sk]
        )
        if navs.empty:
            continue

        latest_nav = navs.iloc[-1]['total_nav']
        total_ret = (latest_nav / INITIAL_CAPITAL - 1) * 100

        peak = navs['total_nav'].expanding().max()
        dd = (peak - navs['total_nav']) / peak * 100
        max_dd = dd.max()

        trades = pd.read_sql(
            "SELECT * FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status='closed'",
            conn, params=[market, sk]
        )

        name = STRATEGIES[sk]['name']
        print(f"\n  📊 {name} ({sk}): ${latest_nav:,.0f} ({total_ret:+.1f}%), MaxDD={max_dd:.1f}%, trades={len(trades)}")

        if not trades.empty:
            win_rate = (trades['pnl_pct'] > 0).mean() * 100
            avg_ret = trades['pnl_pct'].mean()
            print(f"     胜率: {win_rate:.0f}%, 平均收益: {avg_ret:+.1f}%")

    conn.close()


# ============================================================================
# Public API for Streamlit / Alpaca bridge
# ============================================================================

def get_paper_nav_history(market='US', strategy_key=None):
    """Get NAV history for display"""
    init_db()  # Ensure tables exist
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    if strategy_key:
        df = pd.read_sql(
            'SELECT date, total_nav, cash, unrealized, n_positions FROM paper_daily_nav_v2 WHERE market=? AND strategy_key=? ORDER BY date',
            conn, params=[market, strategy_key]
        )
    else:
        df = pd.read_sql(
            'SELECT strategy_key, date, total_nav, cash, unrealized, n_positions FROM paper_daily_nav_v2 WHERE market=? ORDER BY date',
            conn, params=[market]
        )
    conn.close()
    return df


def get_paper_positions(market='US', strategy_key=None, status='open'):
    """Get positions for display"""
    init_db()  # Ensure tables exist
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    if strategy_key:
        df = pd.read_sql(
            'SELECT * FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status=? ORDER BY buy_date DESC',
            conn, params=[market, strategy_key, status]
        )
    else:
        df = pd.read_sql(
            'SELECT * FROM paper_positions_v2 WHERE market=? AND status=? ORDER BY buy_date DESC',
            conn, params=[market, status]
        )
    conn.close()
    return df


def get_strategy_target_holdings(market='US', strategy_key=None):
    """Get target holdings for the best (or specified) strategy — used by Alpaca bridge"""
    if strategy_key is None:
        strategy_key = get_best_strategy(market)

    positions = get_paper_positions(market, strategy_key, 'open')
    if positions.empty:
        return strategy_key, {}

    # {symbol: shares}
    holdings = {}
    for _, row in positions.iterrows():
        holdings[row['symbol']] = {
            'shares': row['shares'],
            'buy_price': row['buy_price'],
            'buy_date': row['buy_date'],
        }
    return strategy_key, holdings


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Strategy Paper Trading')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--report', action='store_true', help='Show detailed report')
    parser.add_argument('--compare', action='store_true', help='Compare all strategies')
    parser.add_argument('--strategy', default=None, help='Specific strategy key')
    args = parser.parse_args()

    if args.compare:
        show_compare(args.market)
    elif args.report:
        show_report(args.market, args.strategy)
    else:
        run_daily(args.market)
        show_compare(args.market)
