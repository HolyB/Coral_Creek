#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-Forward Backtest v2 — 真实模拟
=====================================
- 价格过滤 (US>=$5, CN>=¥3)
- 手续费 + 滑点
- 真实仓位管理 (10%/仓, 资金约束)
- 逐日 NAV 追踪, 最大回撤, Sharpe
- 月度/市值档/板块分拆

PYTHONPATH=. python scripts/walk_forward_backtest.py --market CN
PYTHONPATH=. python scripts/walk_forward_backtest.py --market US
PYTHONPATH=. python scripts/walk_forward_backtest.py --market BOTH
"""
import os, sys, sqlite3, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

# ==================== Config ====================
HOLD_PERIODS = [1, 3, 5, 7, 10, 15, 20, 25, 30]
COMMISSION = 0.001      # 0.1% per trade (buy+sell)
SLIPPAGE = 0.001        # 0.1% per trade
POS_SIZE = 0.10         # 10% per position
INITIAL_CAP = 100000
MIN_PRICE = {'US': 5.0, 'CN': 3.0}
N_STOCKS = 500          # random sample size
TOP_PCT = 0.10          # top 10%


# ==================== Features ====================
def compute_features(df):
    """20 lightweight features"""
    df = df.sort_values('Date').copy()
    c, v = df['Close'], df['Volume'].replace(0, np.nan)
    F = pd.DataFrame(index=df.index)
    F['Date'] = df['Date']
    F['ret_1d'] = c.pct_change() * 100
    F['ret_5d'] = c.pct_change(5) * 100
    F['ret_10d'] = c.pct_change(10) * 100
    F['ret_20d'] = c.pct_change(20) * 100
    for w in [5, 10, 20, 60]:
        ma = c.rolling(w).mean()
        F[f'ma{w}_bias'] = (c - ma) / ma * 100
    F['vol_5d'] = c.pct_change().rolling(5).std() * 100
    F['vol_20d'] = c.pct_change().rolling(20).std() * 100
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    F['rsi_14'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    F['vol_ratio'] = v / v.rolling(20).mean()
    h20, l20 = c.rolling(20).max(), c.rolling(20).min()
    F['price_pos'] = (c - l20) / (h20 - l20).replace(0, np.nan) * 100
    F['body_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    F['range'] = (df['High'] - df['Low']) / df['Low'] * 100
    F['mom_accel'] = F['ret_5d'] - F['ret_5d'].shift(5)
    return F.dropna()


# ==================== Data Loading ====================
def load_data(market, start_date='2025-06-01'):
    hist_db = sqlite3.connect(str(V3_DIR / 'db' / 'stock_history.db'))
    all_syms = [r[0] for r in hist_db.execute(
        'SELECT DISTINCT symbol FROM stock_history WHERE market=?', (market,)
    ).fetchall()]
    np.random.seed(42)
    sample_syms = list(np.random.choice(all_syms, min(N_STOCKS, len(all_syms)), replace=False))
    ph = ','.join(['?'] * len(sample_syms))
    df_all = pd.read_sql_query(
        f"""SELECT symbol, trade_date as Date, open as Open, high as High,
            low as Low, close as Close, volume as Volume
        FROM stock_history WHERE market=? AND symbol IN ({ph})
        ORDER BY symbol, trade_date""",
        hist_db, params=[market] + sample_syms
    )
    hist_db.close()
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    # Price filter
    min_p = MIN_PRICE.get(market, 3)
    valid = df_all.groupby('symbol')['Close'].last()
    valid = valid[valid >= min_p].index.tolist()
    df_all = df_all[df_all['symbol'].isin(valid)]
    print(f"📥 {market}: {len(valid)}/{len(sample_syms)} stocks (price>={min_p}), {len(df_all):,} rows")

    # Market cap lookup
    mcap = {}
    try:
        pc = sqlite3.connect(str(V3_DIR / 'db' / 'ml_daily_picks.db'))
        for r in pc.execute('SELECT symbol,market_cap FROM ml_picks_v2 WHERE market=? AND market_cap>0 GROUP BY symbol', (market,)).fetchall():
            mcap[r[0]] = r[1]
        pc.close()
    except: pass

    # Build samples with all future prices for multi-horizon
    print("🧮 Computing features + price curves...")
    max_hold = max(HOLD_PERIODS)
    samples = []

    for sym, sdf in df_all.groupby('symbol'):
        if len(sdf) < 120: continue
        sdf = sdf.tail(1500).reset_index(drop=True)
        feats = compute_features(sdf)
        if len(feats) < 60: continue
        merged = feats.merge(sdf[['Date', 'Open', 'Close', 'High', 'Low']], on='Date', how='left')

        mc = mcap.get(sym, 0)
        if market == 'CN':
            board = '科创板' if sym.split('.')[0].startswith('688') else \
                    '创业板' if sym.split('.')[0][:3] in ('300','301') else '主板'
            tier = '>1000亿' if mc>=1e11 else '300-1000亿' if mc>=3e10 else \
                   '100-300亿' if mc>=1e10 else '50-100亿' if mc>=5e9 else '<50亿' if mc>0 else '未知'
        else:
            board = 'US'
            tier = 'Mega' if mc>=1e11 else 'Large' if mc>=1e10 else \
                   'Mid' if mc>=2e9 else 'Small' if mc>=3e8 else 'Micro' if mc>0 else '未知'

        for i in range(0, len(merged) - max_hold - 1, 5):
            row = merged.iloc[i]
            ep = row['Open']
            if pd.isna(ep) or ep <= 0: continue

            # Future prices for each horizon + daily close series
            labels = {}
            ok = True
            for h in HOLD_PERIODS:
                fi = min(i + h, len(merged) - 1)
                fp = merged.iloc[fi]['Close']
                if pd.isna(fp) or fp <= 0: ok = False; break
                labels[h] = (fp / ep - 1) * 100
            if not ok: continue

            fd = {c: row[c] for c in feats.columns if c != 'Date' and not pd.isna(row[c])}
            samples.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'month': row['Date'].strftime('%Y-%m'),
                'symbol': sym, 'board': board, 'tier': tier,
                'price': ep, 'features': fd, 'labels': labels,
            })

    print(f"   {len(samples):,} samples")
    return samples


# ==================== Walk-Forward Engine ====================
def walk_forward(market, samples, hold_days=5):
    import xgboost as xgb

    by_date = {}
    for s in samples:
        by_date.setdefault(s['date'], []).append(s)
    all_dates = sorted(by_date.keys())
    feat_names = sorted(samples[0]['features'].keys())
    eval_dates = [d for d in all_dates if d >= '2025-06-01']
    print(f"   {len(feat_names)} features, {len(eval_dates)} eval dates, hold={hold_days}d")

    # Portfolio simulation
    capital = INITIAL_CAP
    positions = []  # (symbol, buy_date, buy_price, amount, sell_date_target)
    nav_history = []  # (date, nav, n_pos)
    all_trades = []
    all_picks_info = []  # for breakdown

    cost_rate = COMMISSION + SLIPPAGE  # total per-trade cost

    for eval_date in eval_dates:
        # 1. Close expired positions
        new_pos = []
        for sym, bd, bp, amt, sell_target in positions:
            if eval_date >= sell_target:
                # Find actual sell price
                sell_row = None
                for s in by_date.get(eval_date, []):
                    if s['symbol'] == sym:
                        sell_row = s
                        break
                sp = sell_row['price'] if sell_row else bp  # use open price as proxy
                proceeds = amt * (sp / bp) * (1 - cost_rate)  # deduct sell cost
                pnl_pct = (sp / bp - 1) * 100 - cost_rate * 100
                capital += proceeds
                all_trades.append({
                    'symbol': sym, 'buy_date': bd, 'sell_date': eval_date,
                    'buy_price': bp, 'sell_price': sp, 'pnl_pct': pnl_pct,
                    'amount': amt, 'month': eval_date[:7],
                })
            else:
                new_pos.append((sym, bd, bp, amt, sell_target))
        positions = new_pos

        # 2. Train model (only on past data)
        train = [s for s in samples if s['date'] < eval_date]
        if len(train) < 500:
            # Record NAV
            unr = sum(a for _, _, _, a, _ in positions)
            nav_history.append((eval_date, capital + unr, len(positions)))
            continue
        if len(train) > 200000:
            train = train[-200000:]

        X_tr = np.nan_to_num(np.array([[s['features'].get(f, 0) for f in feat_names] for s in train]))
        y_tr = np.nan_to_num(np.array([s['labels'].get(5, 0) for s in train]))

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_tr, y_tr, verbose=False)

        # 3. Score today's candidates
        test = by_date.get(eval_date, [])
        if len(test) < 5:
            unr = sum(a for _, _, _, a, _ in positions)
            nav_history.append((eval_date, capital + unr, len(positions)))
            continue

        X_te = np.nan_to_num(np.array([[s['features'].get(f, 0) for f in feat_names] for s in test]))
        preds = model.predict(X_te)

        ranked = sorted(zip(preds, test), key=lambda x: -x[0])
        n_top = max(1, len(ranked) // 10)
        top_picks = ranked[:n_top]

        # 4. Open new positions
        open_syms = set(s for s, _, _, _, _ in positions)
        sell_target = eval_dates[min(eval_dates.index(eval_date) + hold_days, len(eval_dates) - 1)] \
            if eval_date in eval_dates else eval_date

        for pred, pick in top_picks:
            sym = pick['symbol']
            if sym in open_syms: continue
            pos_amount = capital * POS_SIZE
            if pos_amount < 100 or capital < pos_amount: continue

            buy_price = pick['price'] * (1 + cost_rate)  # add buy cost
            capital -= pos_amount
            positions.append((sym, eval_date, buy_price, pos_amount, sell_target))
            open_syms.add(sym)

            all_picks_info.append({
                'date': eval_date, 'month': eval_date[:7],
                'symbol': sym, 'board': pick['board'], 'tier': pick['tier'],
                'pred': pred, 'actual_5d': pick['labels'].get(hold_days, 0),
            })

        # 5. NAV (mark to market)
        unr_value = 0
        for sym, bd, bp, amt, st in positions:
            # Find current price
            current = None
            for s in test:
                if s['symbol'] == sym:
                    current = s['price']
                    break
            unr_value += amt * (current / bp) if current else amt
        nav = capital + unr_value
        nav_history.append((eval_date, nav, len(positions)))

    # Force close remaining positions at last known prices
    for sym, bd, bp, amt, st in positions:
        capital += amt  # approximate

    return nav_history, all_trades, all_picks_info


# ==================== Reporting ====================
def report(market, nav_history, trades, picks_info):
    nav_df = pd.DataFrame(nav_history, columns=['date', 'nav', 'n_pos'])
    if nav_df.empty:
        print("No results")
        return

    final_nav = nav_df['nav'].iloc[-1]
    total_ret = (final_nav / INITIAL_CAP - 1) * 100
    n_days = (pd.Timestamp(nav_df['date'].iloc[-1]) - pd.Timestamp(nav_df['date'].iloc[0])).days
    ann_ret = ((final_nav / INITIAL_CAP) ** (365 / max(n_days, 1)) - 1) * 100

    # Max drawdown
    peak = nav_df['nav'].expanding().max()
    dd = (nav_df['nav'] - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe
    daily_ret = nav_df['nav'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # Trade stats
    trade_df = pd.DataFrame(trades)
    if len(trade_df):
        wins = trade_df[trade_df['pnl_pct'] > 0]
        losses = trade_df[trade_df['pnl_pct'] <= 0]
        win_rate = len(wins) / len(trade_df) * 100
        avg_win = wins['pnl_pct'].mean() if len(wins) else 0
        avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) else 1
        pf = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        win_rate = avg_win = avg_loss = pf = 0

    print(f"\n{'='*65}")
    print(f"📊 Walk-Forward 真实模拟 ({market})")
    print(f"{'='*65}")
    print(f"  初始资金:    ${INITIAL_CAP:>12,}")
    print(f"  最终资金:    ${final_nav:>12,.0f}")
    print(f"  总收益:      {total_ret:>+11.1f}%")
    print(f"  年化收益:    {ann_ret:>+11.1f}%")
    print(f"  最大回撤:    {max_dd:>11.1f}%")
    print(f"  Sharpe:      {sharpe:>11.2f}")
    print(f"  总交易:      {len(trades):>11}")
    print(f"  胜率:        {win_rate:>10.1f}%")
    print(f"  平均盈利:    {avg_win:>+10.1f}%")
    print(f"  平均亏损:    {avg_loss:>10.1f}%")
    print(f"  盈亏比:      {pf:>11.2f}")
    print(f"  手续费+滑点: {(COMMISSION+SLIPPAGE)*100:.1f}% per trade")

    # Monthly breakdown
    if len(trade_df):
        print(f"\n📅 月度收益:")
        print(f"  {'月份':<10} {'交易':>5} {'胜率':>6} {'平均PnL':>8} {'总PnL':>9}")
        for mo, g in trade_df.groupby('month'):
            print(f"  {mo:<10} {len(g):>5} {(g['pnl_pct']>0).mean()*100:>5.0f}% {g['pnl_pct'].mean():>+7.1f}% {g['pnl_pct'].sum():>+8.0f}%")

    # Tier breakdown
    picks_df = pd.DataFrame(picks_info)
    if len(picks_df):
        print(f"\n📊 市值档 (Top10% picks):")
        print(f"  {'Tier':<12} {'n':>5} {'avg_pred':>9} {'actual':>8} {'胜率':>6}")
        for tier in sorted(picks_df['tier'].unique()):
            g = picks_df[picks_df['tier'] == tier]
            print(f"  {tier:<12} {len(g):>5} {g['pred'].mean():>+8.1f}% {g['actual_5d'].mean():>+7.1f}% {(g['actual_5d']>0).mean()*100:>5.0f}%")

    # Board breakdown (CN only)
    if market == 'CN' and len(picks_df):
        print(f"\n📊 板块:")
        for b in sorted(picks_df['board'].unique()):
            g = picks_df[picks_df['board'] == b]
            print(f"  {b:<8} {len(g):>5} picks, actual={g['actual_5d'].mean():+.1f}%, win={((g['actual_5d']>0).mean()*100):.0f}%")


# ==================== Main ====================
def run_all(market, hold_days=5):
    samples = load_data(market)
    nav_h, trades, picks = walk_forward(market, samples, hold_days)
    report(market, nav_h, trades, picks)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--market', default='CN', choices=['US', 'CN', 'BOTH'])
    p.add_argument('--hold', type=int, default=5)
    p.add_argument('--start', default='2025-06-01')
    args = p.parse_args()
    if args.market == 'BOTH':
        for m in ['CN', 'US']:
            run_all(m, args.hold)
            print("\n\n")
    else:
        run_all(args.market, args.hold)
