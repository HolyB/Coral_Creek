#!/usr/bin/env python3
"""
ML Picks 虚拟组合追踪器
========================
模拟按 ML daily picks 实际投资的效果：
- 去重：同一只股票在持仓期内只买一次
- 等权：每笔交易等金额
- 到期自动卖出
- 计算 NAV 净值曲线

用法:
    PYTHONPATH=. python scripts/ml_portfolio_tracker.py --market US
    PYTHONPATH=. python scripts/ml_portfolio_tracker.py --market CN
    PYTHONPATH=. python scripts/ml_portfolio_tracker.py --market US --initial 100000
"""

import os
import sys
import sqlite3
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

DB_DIR = os.path.join(parent_dir, 'db')
PICKS_DB = os.path.join(DB_DIR, 'ml_daily_picks.db')
HIST_DB = os.path.join(DB_DIR, 'stock_history.db')


def _get_all_picks(market='US'):
    """Load all picks from DB, sorted by date"""
    conn = sqlite3.connect(PICKS_DB)
    df = pd.read_sql_query(
        '''SELECT date, symbol, price, primary_pred, holding_period, tier, segment, market
           FROM ml_picks_v2 WHERE market=? ORDER BY date, primary_pred DESC''',
        conn, params=(market,)
    )
    conn.close()
    return df


def _get_price_map(symbols, market, dates):
    """
    Build a price lookup: {symbol: {date_str: close_price}}
    """
    conn = sqlite3.connect(HIST_DB)
    min_date = min(dates) if dates else '2020-01-01'
    max_date = max(dates) if dates else '2030-01-01'

    placeholders = ','.join(['?' for _ in symbols])
    query = f'''
        SELECT symbol, trade_date, close 
        FROM stock_history 
        WHERE symbol IN ({placeholders}) AND market=?
          AND trade_date >= ? AND trade_date <= ?
        ORDER BY symbol, trade_date
    '''
    params = list(symbols) + [market, min_date, max_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    price_map = defaultdict(dict)
    for _, row in df.iterrows():
        price_map[row['symbol']][row['trade_date']] = row['close']

    return dict(price_map)


def _get_trading_days(market, start_date, end_date):
    """Get sorted list of trading dates from stock_history"""
    conn = sqlite3.connect(HIST_DB)
    rows = conn.execute(
        '''SELECT DISTINCT trade_date FROM stock_history 
           WHERE market=? AND trade_date >= ? AND trade_date <= ?
           ORDER BY trade_date''',
        (market, start_date, end_date)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def compute_signal_streaks(picks_df):
    """
    Compute consecutive appearance days for each (symbol, date).
    Returns dict: {(symbol, date): streak_count}
    
    If AEYE appears on 3/10, 3/11, 3/12, 3/13:
      - 3/10: streak=1, 3/11: streak=2, 3/12: streak=3, 3/13: streak=4
    """
    streaks = {}
    # Group by symbol, sort by date
    for symbol, grp in picks_df.groupby('symbol'):
        dates = sorted(grp['date'].unique())
        if len(dates) <= 1:
            for d in dates:
                streaks[(symbol, d)] = 1
            continue

        streak = 1
        streaks[(symbol, dates[0])] = 1

        for i in range(1, len(dates)):
            prev = pd.Timestamp(dates[i - 1])
            curr = pd.Timestamp(dates[i])
            gap = (curr - prev).days
            # Allow weekend gaps (1-3 calendar days = consecutive trading days)
            if gap <= 4:
                streak += 1
            else:
                streak = 1
            streaks[(symbol, dates[i])] = streak

    return streaks


def build_portfolio_history(market='US', initial_capital=100000.0):
    """
    Simulate a dedup equal-weight portfolio from historical ML picks.
    
    Returns:
        nav_df: DataFrame with columns [date, nav, cash, holdings_value, n_holdings, n_trades]
        trades: list of dicts with trade details
        current_holdings: list of dicts for currently open positions
    """
    picks_df = _get_all_picks(market)
    if picks_df.empty:
        return pd.DataFrame(), [], []

    pick_dates = sorted(picks_df['date'].unique())
    all_symbols = picks_df['symbol'].unique().tolist()

    # Determine date range
    first_date = pick_dates[0]
    last_date = pick_dates[-1]
    # Extend to cover holding periods
    hold_days = 10 if market == 'US' else 30
    extend_date = (pd.Timestamp(last_date) + pd.Timedelta(days=int(hold_days * 2))).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    end_date = min(extend_date, today)

    # Get all trading days and build price map
    trading_days = _get_trading_days(market, first_date, end_date)
    if not trading_days:
        return pd.DataFrame(), [], []

    price_map = _get_price_map(all_symbols, market, [first_date, end_date])

    # Signal streaks
    streaks = compute_signal_streaks(picks_df)

    # --- Portfolio simulation ---
    cash = initial_capital
    holdings = {}  # symbol -> {buy_date, buy_price, shares, sell_target_idx}
    trades = []
    nav_records = []

    # Index trading days
    td_to_idx = {d: i for i, d in enumerate(trading_days)}

    # Group picks by date
    picks_by_date = {}
    for d, grp in picks_df.groupby('date'):
        picks_by_date[d] = grp

    for td in trading_days:
        # 1. Sell expired holdings
        to_sell = []
        for sym, pos in holdings.items():
            if td_to_idx.get(td, 0) >= pos['sell_target_idx']:
                to_sell.append(sym)

        for sym in to_sell:
            pos = holdings.pop(sym)
            sell_price = price_map.get(sym, {}).get(td)
            if sell_price is None:
                # Try to find nearest available price
                sym_prices = price_map.get(sym, {})
                for fallback_d in trading_days[td_to_idx[td]::-1]:
                    if fallback_d in sym_prices:
                        sell_price = sym_prices[fallback_d]
                        break
                if sell_price is None:
                    sell_price = pos['buy_price']  # worst case

            proceeds = sell_price * pos['shares']
            pnl = (sell_price / pos['buy_price'] - 1) * 100
            cash += proceeds
            trades.append({
                'symbol': sym,
                'buy_date': pos['buy_date'],
                'buy_price': pos['buy_price'],
                'sell_date': td,
                'sell_price': sell_price,
                'shares': pos['shares'],
                'pnl_pct': round(pnl, 2),
                'pnl_abs': round(proceeds - pos['buy_price'] * pos['shares'], 2),
                'streak': pos.get('streak', 1),
            })

        # 2. Buy new picks (dedup: skip already held)
        if td in picks_by_date:
            day_picks = picks_by_date[td]
            new_picks = [r for _, r in day_picks.iterrows() if r['symbol'] not in holdings]

            if new_picks:
                # Equal-weight: allocate a portion of cash to new picks
                # Use a max allocation of 5% per position
                max_per_pos = cash * 0.05
                alloc_per_pick = min(max_per_pos, cash / max(len(new_picks), 1))

                for pick in new_picks:
                    sym = pick['symbol']
                    buy_price = pick['price']
                    if buy_price <= 0 or alloc_per_pick < 10:
                        continue

                    # Determine sell target
                    hp_str = pick.get('holding_period', '10d')
                    hp_days = int(hp_str.replace('d', '')) if isinstance(hp_str, str) else 10
                    buy_idx = td_to_idx.get(td, 0)
                    sell_target_idx = buy_idx + hp_days

                    shares = alloc_per_pick / buy_price
                    cost = shares * buy_price
                    cash -= cost

                    streak = streaks.get((sym, td), 1)

                    holdings[sym] = {
                        'buy_date': td,
                        'buy_price': buy_price,
                        'shares': shares,
                        'sell_target_idx': sell_target_idx,
                        'streak': streak,
                        'pred': pick.get('primary_pred', 0),
                    }

        # 3. Calculate NAV
        holdings_value = 0
        for sym, pos in holdings.items():
            current_price = price_map.get(sym, {}).get(td)
            if current_price is None:
                current_price = pos['buy_price']
            holdings_value += current_price * pos['shares']

        nav = cash + holdings_value
        nav_records.append({
            'date': td,
            'nav': round(nav, 2),
            'cash': round(cash, 2),
            'holdings_value': round(holdings_value, 2),
            'n_holdings': len(holdings),
            'n_trades': len([t for t in trades if t['sell_date'] == td]),
        })

    # Current holdings (still open)
    current_holdings = []
    last_td = trading_days[-1] if trading_days else None
    for sym, pos in holdings.items():
        latest_price = price_map.get(sym, {}).get(last_td, pos['buy_price'])
        pnl = (latest_price / pos['buy_price'] - 1) * 100
        current_holdings.append({
            'symbol': sym,
            'buy_date': pos['buy_date'],
            'buy_price': pos['buy_price'],
            'current_price': latest_price,
            'shares': pos['shares'],
            'pnl_pct': round(pnl, 2),
            'streak': pos.get('streak', 1),
            'pred': pos.get('pred', 0),
        })

    nav_df = pd.DataFrame(nav_records)
    return nav_df, trades, current_holdings


def compute_metrics(nav_df, trades):
    """Compute portfolio performance metrics"""
    if nav_df.empty:
        return {}

    first_nav = nav_df.iloc[0]['nav']
    last_nav = nav_df.iloc[-1]['nav']
    total_return = (last_nav / first_nav - 1) * 100

    # Annualized
    n_days = (pd.Timestamp(nav_df.iloc[-1]['date']) - pd.Timestamp(nav_df.iloc[0]['date'])).days
    ann_return = ((last_nav / first_nav) ** (365 / max(n_days, 1)) - 1) * 100 if n_days > 0 else 0

    # Max drawdown
    nav_series = nav_df['nav'].values
    peak = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - peak) / peak * 100
    max_dd = drawdowns.min()

    # Daily returns for Sharpe
    daily_rets = pd.Series(nav_series).pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

    # Trade stats
    if trades:
        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    return {
        'total_return': round(total_return, 2),
        'ann_return': round(ann_return, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe': round(sharpe, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
    }


# ===================== Main =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Portfolio Tracker')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--initial', type=float, default=100000, help='Initial capital')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"📊 ML Virtual Portfolio: {args.market}")
    print(f"{'='*60}")

    nav_df, trades, holdings = build_portfolio_history(args.market, args.initial)

    if nav_df.empty:
        print("❌ No data")
        sys.exit(0)

    metrics = compute_metrics(nav_df, trades)

    print(f"\n📈 Performance ({nav_df.iloc[0]['date']} → {nav_df.iloc[-1]['date']}):")
    print(f"   Total Return: {metrics['total_return']:+.2f}%")
    print(f"   Annualized:   {metrics['ann_return']:+.2f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"   Win Rate:     {metrics['win_rate']:.1f}% ({metrics['total_trades']} trades)")
    print(f"   Profit Factor:{metrics['profit_factor']:.2f}")

    if holdings:
        print(f"\n📋 Current Holdings ({len(holdings)}):")
        for h in sorted(holdings, key=lambda x: x['pnl_pct'], reverse=True):
            print(f"   {h['symbol']:8s} buy={h['buy_price']:.2f} now={h['current_price']:.2f} "
                  f"pnl={h['pnl_pct']:+.1f}% streak={h['streak']}d")

    if trades:
        print(f"\n📝 Last 10 Trades:")
        for t in trades[-10:]:
            print(f"   {t['symbol']:8s} {t['buy_date']}→{t['sell_date']} "
                  f"buy={t['buy_price']:.2f} sell={t['sell_price']:.2f} "
                  f"pnl={t['pnl_pct']:+.1f}% streak={t['streak']}d")

    print(f"\n📊 NAV: ${nav_df.iloc[0]['nav']:,.0f} → ${nav_df.iloc[-1]['nav']:,.0f}")
