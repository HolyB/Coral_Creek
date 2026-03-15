#!/usr/bin/env python3
"""
ML Picks 虚拟组合追踪器 (多策略版)
====================================
内置多种选股策略，模拟不同仓位管理方式的投资效果。

策略列表：
- all_in:       买入当天所有 picks（去重），5% 仓位上限
- top1_daily:   每天只买 pred 最高的 1 只（集中火力）
- top3_daily:   每天只买 pred 最高的 3 只
- top10pct:     每天只买 pred 排名前 10% 的
- streak_only:  只买连续出现 ≥2 天的股票（信号确认）
- large_cap:    只买大市值 (Mid+Large+Mega)

用法:
    PYTHONPATH=. python scripts/ml_portfolio_tracker.py --market US --strategy top1_daily
    PYTHONPATH=. python scripts/ml_portfolio_tracker.py --market US --compare
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
# Prefer full stock_history.db, fallback to partial
_full_hist = os.path.join(DB_DIR, 'stock_history.db')
_partial_hist = os.path.join(DB_DIR, 'partial_stock_history.db')
HIST_DB = _full_hist if os.path.exists(_full_hist) else _partial_hist

# ===================== Strategy Definitions =====================
STRATEGIES = {
    'all_in': {
        'name': '全仓买入',
        'desc': '买入所有 picks，5% 单笔上限',
        'max_per_pos': 0.05,
        'daily_pick_limit': None,
        'filter_fn': None,
    },
    'top1_daily': {
        'name': '每日 Top1',
        'desc': '每天只买预测最高的 1 只，10% 仓位',
        'max_per_pos': 0.10,
        'daily_pick_limit': 1,
        'filter_fn': None,
    },
    'top3_daily': {
        'name': '每日 Top3',
        'desc': '每天只买预测最高的 3 只，10% 仓位',
        'max_per_pos': 0.10,
        'daily_pick_limit': 3,
        'filter_fn': None,
    },
    'top10pct': {
        'name': '前10%精选',
        'desc': '每天只买预测排名前 10% 的 picks',
        'max_per_pos': 0.10,
        'daily_pick_limit': None,
        'filter_fn': 'top10pct',
    },
    'streak_only': {
        'name': '连续信号',
        'desc': '只买连续出现 ≥2 天的股票',
        'max_per_pos': 0.10,
        'daily_pick_limit': None,
        'filter_fn': 'streak_ge2',
    },
    'large_cap': {
        'name': '大盘精选',
        'desc': '只买中/大/超大市值的 picks',
        'max_per_pos': 0.10,
        'daily_pick_limit': 3,
        'filter_fn': 'large_cap',
    },
}


def _get_all_picks(market='US'):
    """Load all picks from DB, sorted by date"""
    conn = sqlite3.connect(PICKS_DB)
    df = pd.read_sql_query(
        '''SELECT date, symbol, price, primary_pred, holding_period, 
                  tier, segment, market, market_cap
           FROM ml_picks_v2 WHERE market=? ORDER BY date, primary_pred DESC''',
        conn, params=(market,)
    )
    conn.close()
    return df


def _get_price_map(symbols, market, dates):
    """Build a price lookup: {symbol: {date_str: close_price}}"""
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


def get_benchmark_returns(market, start_date, end_date):
    """
    Get benchmark index returns as {date: cumulative_return_%}.
    US: SPY, CN: HS300 (from cn_index_data.json)
    """
    import json
    
    if market == 'CN':
        idx_path = os.path.join(DB_DIR, 'cn_index_data.json')
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx_data = json.load(f)
            # Extract hs300_close for date range
            prices = {}
            for d, vals in idx_data.items():
                if start_date <= d <= end_date and 'hs300_close' in vals:
                    prices[d] = vals['hs300_close']
            if prices:
                dates_sorted = sorted(prices.keys())
                base = prices[dates_sorted[0]]
                return {d: (prices[d] / base - 1) * 100 for d in dates_sorted}, '沪深300'
    else:
        # US: try SPY from stock_history
        conn = sqlite3.connect(HIST_DB)
        rows = conn.execute(
            '''SELECT trade_date, close FROM stock_history
               WHERE symbol='SPY' AND market='US'
               AND trade_date >= ? AND trade_date <= ?
               ORDER BY trade_date''',
            (start_date, end_date)
        ).fetchall()
        conn.close()
        if rows:
            base = rows[0][1]
            return {r[0]: (r[1] / base - 1) * 100 for r in rows}, 'SPY'
    
    return {}, ''


def compute_signal_streaks(picks_df):
    """Compute consecutive appearance days for each (symbol, date)."""
    streaks = {}
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
            if gap <= 4:
                streak += 1
            else:
                streak = 1
            streaks[(symbol, dates[i])] = streak
    return streaks


def _filter_picks(day_picks, strategy_cfg, streaks, td):
    """Apply strategy filters to day's picks."""
    picks = list(day_picks.itertuples(index=False))
    # Already sorted by primary_pred desc
    
    filter_fn = strategy_cfg.get('filter_fn')
    
    if filter_fn == 'top10pct':
        n = max(1, len(picks) // 10)
        picks = picks[:n]
    elif filter_fn == 'streak_ge2':
        picks = [p for p in picks if streaks.get((p.symbol, td), 1) >= 2]
    elif filter_fn == 'large_cap':
        # Mid cap and above: US >= 2B, CN >= 10B
        min_mc = 2e9 if hasattr(picks[0], 'market') and picks[0].market == 'US' else 1e10
        picks = [p for p in picks if (getattr(p, 'market_cap', 0) or 0) >= min_mc]
    
    limit = strategy_cfg.get('daily_pick_limit')
    if limit:
        picks = picks[:limit]
    
    return picks


def build_portfolio_history(market='US', initial_capital=100000.0, strategy='all_in'):
    """
    Simulate a portfolio with the given strategy.
    
    Returns:
        nav_df, trades, current_holdings
    """
    strategy_cfg = STRATEGIES.get(strategy, STRATEGIES['all_in'])
    
    picks_df = _get_all_picks(market)
    if picks_df.empty:
        return pd.DataFrame(), [], []

    pick_dates = sorted(picks_df['date'].unique())
    all_symbols = picks_df['symbol'].unique().tolist()

    first_date = pick_dates[0]
    last_date = pick_dates[-1]
    hold_days = 10 if market == 'US' else 30
    extend_date = (pd.Timestamp(last_date) + pd.Timedelta(days=int(hold_days * 2))).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    end_date = min(extend_date, today)

    trading_days = _get_trading_days(market, first_date, end_date)
    if not trading_days:
        return pd.DataFrame(), [], []

    price_map = _get_price_map(all_symbols, market, [first_date, end_date])
    streaks = compute_signal_streaks(picks_df)

    cash = initial_capital
    holdings = {}
    trades = []
    nav_records = []
    td_to_idx = {d: i for i, d in enumerate(trading_days)}
    picks_by_date = {d: grp for d, grp in picks_df.groupby('date')}
    
    max_per_pos = strategy_cfg.get('max_per_pos', 0.05)

    for td in trading_days:
        # 1. Sell expired
        to_sell = [s for s, p in holdings.items() if td_to_idx.get(td, 0) >= p['sell_target_idx']]
        for sym in to_sell:
            pos = holdings.pop(sym)
            sell_price = price_map.get(sym, {}).get(td)
            if sell_price is None:
                sym_prices = price_map.get(sym, {})
                for fallback_d in trading_days[td_to_idx[td]::-1]:
                    if fallback_d in sym_prices:
                        sell_price = sym_prices[fallback_d]
                        break
                if sell_price is None:
                    sell_price = pos['buy_price']
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

        # 2. Buy new picks (dedup + strategy filter)
        if td in picks_by_date:
            day_picks_df = picks_by_date[td]
            filtered = _filter_picks(day_picks_df, strategy_cfg, streaks, td)
            new_picks = [p for p in filtered if p.symbol not in holdings]

            if new_picks:
                alloc_per_pick = min(cash * max_per_pos, cash / max(len(new_picks), 1))
                for pick in new_picks:
                    buy_price = pick.price
                    if buy_price <= 0 or alloc_per_pick < 10:
                        continue
                    hp_str = pick.holding_period if hasattr(pick, 'holding_period') and isinstance(pick.holding_period, str) else '10d'
                    hp_days = int(hp_str.replace('d', '')) if isinstance(hp_str, str) else 10
                    buy_idx = td_to_idx.get(td, 0)
                    shares = alloc_per_pick / buy_price
                    cash -= shares * buy_price
                    holdings[pick.symbol] = {
                        'buy_date': td,
                        'buy_price': buy_price,
                        'shares': shares,
                        'sell_target_idx': buy_idx + hp_days,
                        'streak': streaks.get((pick.symbol, td), 1),
                        'pred': pick.primary_pred if hasattr(pick, 'primary_pred') else 0,
                    }

        # 3. NAV
        hv = sum(price_map.get(s, {}).get(td, p['buy_price']) * p['shares'] for s, p in holdings.items())
        nav = cash + hv
        nav_records.append({
            'date': td, 'nav': round(nav, 2), 'cash': round(cash, 2),
            'holdings_value': round(hv, 2), 'n_holdings': len(holdings),
        })

    # Current holdings
    last_td = trading_days[-1] if trading_days else None
    current_holdings = []
    for sym, pos in holdings.items():
        cp = price_map.get(sym, {}).get(last_td, pos['buy_price'])
        current_holdings.append({
            'symbol': sym, 'buy_date': pos['buy_date'],
            'buy_price': pos['buy_price'], 'current_price': cp,
            'shares': pos['shares'],
            'pnl_pct': round((cp / pos['buy_price'] - 1) * 100, 2),
            'streak': pos.get('streak', 1), 'pred': pos.get('pred', 0),
        })

    return pd.DataFrame(nav_records), trades, current_holdings


def compute_metrics(nav_df, trades):
    """Compute portfolio performance metrics"""
    if nav_df.empty:
        return {}
    first_nav = nav_df.iloc[0]['nav']
    last_nav = nav_df.iloc[-1]['nav']
    total_return = (last_nav / first_nav - 1) * 100
    n_days = (pd.Timestamp(nav_df.iloc[-1]['date']) - pd.Timestamp(nav_df.iloc[0]['date'])).days
    ann_return = ((last_nav / first_nav) ** (365 / max(n_days, 1)) - 1) * 100 if n_days > 0 else 0
    nav_series = nav_df['nav'].values
    peak = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - peak) / peak * 100
    max_dd = drawdowns.min()
    daily_rets = pd.Series(nav_series).pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    if trades:
        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
    return {
        'total_return': round(total_return, 2), 'ann_return': round(ann_return, 2),
        'max_drawdown': round(max_dd, 2), 'sharpe': round(sharpe, 2),
        'total_trades': len(trades), 'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
    }


def compare_all_strategies(market='US', initial=100000.0):
    """Run all strategies and return comparison results."""
    results = {}
    for key, cfg in STRATEGIES.items():
        nav_df, trades, holdings = build_portfolio_history(market, initial, strategy=key)
        metrics = compute_metrics(nav_df, trades)
        results[key] = {
            'name': cfg['name'],
            'desc': cfg['desc'],
            'nav_df': nav_df,
            'trades': trades,
            'holdings': holdings,
            'metrics': metrics,
            'n_holdings': len(holdings),
        }
    return results


# ===================== Main =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Portfolio Tracker')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--initial', type=float, default=100000)
    parser.add_argument('--strategy', default='all_in', choices=list(STRATEGIES.keys()))
    parser.add_argument('--compare', action='store_true', help='Compare all strategies')

    args = parser.parse_args()

    if args.compare:
        print(f"\n{'='*80}")
        print(f"📊 Strategy Comparison: {args.market}")
        print(f"{'='*80}")
        results = compare_all_strategies(args.market, args.initial)
        print(f"\n{'Strategy':<18} {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Trades':>7} {'Hold':>5}")
        print('-' * 72)
        for key, r in results.items():
            m = r['metrics']
            if not m:
                continue
            print(f"{r['name']:<18} {m['total_return']:>+9.1f}% {m['sharpe']:>7.2f} "
                  f"{m['win_rate']:>7.1f}% {m['max_drawdown']:>7.1f}% {m['total_trades']:>6d} {r['n_holdings']:>5d}")
    else:
        nav_df, trades, holdings = build_portfolio_history(args.market, args.initial, args.strategy)
        if nav_df.empty:
            print("❌ No data")
            sys.exit(0)
        metrics = compute_metrics(nav_df, trades)
        cfg = STRATEGIES[args.strategy]
        print(f"\n{'='*60}")
        print(f"📊 {cfg['name']} ({args.market}): {cfg['desc']}")
        print(f"{'='*60}")
        print(f"  Return: {metrics['total_return']:+.2f}%  Sharpe: {metrics['sharpe']:.2f}  "
              f"WinRate: {metrics['win_rate']:.1f}%  MaxDD: {metrics['max_drawdown']:.1f}%")
        print(f"  Trades: {metrics['total_trades']}  Holdings: {len(holdings)}")
        print(f"  NAV: ${nav_df.iloc[0]['nav']:,.0f} → ${nav_df.iloc[-1]['nav']:,.0f}")
