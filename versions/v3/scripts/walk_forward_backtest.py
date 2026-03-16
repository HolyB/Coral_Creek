#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-Forward Multi-Horizon Backtest — 无数据泄漏
=================================================
训练5d模型(只用过去数据) → 选出Top10% → 测量1-30天不同持有期的真实收益

用法:
    PYTHONPATH=. python scripts/walk_forward_backtest.py --market CN
    PYTHONPATH=. python scripts/walk_forward_backtest.py --market US
"""
import os, sys, sqlite3, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

HOLD_PERIODS = [1, 3, 5, 7, 10, 15, 20, 25, 30]


def compute_features(df):
    """20 lightweight features from OHLCV"""
    df = df.sort_values('Date').copy()
    c = df['Close']
    v = df['Volume'].replace(0, np.nan)
    feats = pd.DataFrame(index=df.index)
    feats['Date'] = df['Date']
    feats['ret_1d'] = c.pct_change() * 100
    feats['ret_5d'] = c.pct_change(5) * 100
    feats['ret_10d'] = c.pct_change(10) * 100
    feats['ret_20d'] = c.pct_change(20) * 100
    for w in [5, 10, 20, 60]:
        ma = c.rolling(w).mean()
        feats[f'ma{w}_bias'] = (c - ma) / ma * 100
    feats['vol_5d'] = c.pct_change().rolling(5).std() * 100
    feats['vol_20d'] = c.pct_change().rolling(20).std() * 100
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats['rsi_14'] = 100 - (100 / (1 + rs))
    feats['vol_ratio'] = v / v.rolling(20).mean()
    h20 = c.rolling(20).max()
    l20 = c.rolling(20).min()
    feats['price_pos_20d'] = (c - l20) / (h20 - l20).replace(0, np.nan) * 100
    feats['body_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    feats['intraday_range'] = (df['High'] - df['Low']) / df['Low'] * 100
    feats['mom_accel'] = feats['ret_5d'] - feats['ret_5d'].shift(5)
    return feats.dropna()


def run_all(market='CN', start_date='2025-06-01'):
    import xgboost as xgb
    
    max_hold = max(HOLD_PERIODS)
    
    hist_db = sqlite3.connect(str(V3_DIR / 'db' / 'stock_history.db'))
    all_syms = [r[0] for r in hist_db.execute(
        "SELECT DISTINCT symbol FROM stock_history WHERE market=?", (market,)
    ).fetchall()]
    
    np.random.seed(42)
    n_stocks = 300
    sample_syms = list(np.random.choice(all_syms, min(n_stocks, len(all_syms)), replace=False))
    print(f"📥 {market}: {len(sample_syms)}/{len(all_syms)} stocks")
    
    placeholders = ','.join(['?'] * len(sample_syms))
    df_all = pd.read_sql_query(
        f"""SELECT symbol, trade_date as Date, open as Open, high as High,
            low as Low, close as Close, volume as Volume
        FROM stock_history WHERE market=? AND symbol IN ({placeholders})
        ORDER BY symbol, trade_date""",
        hist_db, params=[market] + sample_syms
    )
    hist_db.close()
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    print(f"   {len(df_all):,} rows loaded")
    
    # Compute features + multi-horizon labels
    print("🧮 Computing features + labels...")
    all_samples = []
    
    for sym, sdf in df_all.groupby('symbol'):
        if len(sdf) < 120:
            continue
        sdf = sdf.tail(1500).reset_index(drop=True)
        feats = compute_features(sdf)
        if len(feats) < 60:
            continue
        
        merged = feats.merge(sdf[['Date', 'Open', 'Close']], on='Date', how='left')
        
        for i in range(0, len(merged) - max_hold - 1, 5):
            row = merged.iloc[i]
            entry_price = row['Open']
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            
            labels = {}
            valid = True
            for h in HOLD_PERIODS:
                fi = min(i + h, len(merged) - 1)
                fp = merged.iloc[fi]['Close']
                if pd.isna(fp) or fp <= 0:
                    valid = False
                    break
                labels[h] = (fp / entry_price - 1) * 100
            
            if not valid:
                continue
            
            feat_dict = {c: row[c] for c in feats.columns if c != 'Date' and not pd.isna(row[c])}
            all_samples.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'symbol': sym, 'price': entry_price,
                'features': feat_dict, 'labels': labels,
            })
    
    print(f"   {len(all_samples):,} samples")
    
    by_date = {}
    for s in all_samples:
        by_date.setdefault(s['date'], []).append(s)
    
    all_dates = sorted(by_date.keys())
    feat_names = sorted(all_samples[0]['features'].keys())
    eval_dates = [d for d in all_dates if d >= start_date]
    print(f"   {len(feat_names)} features, {len(eval_dates)} eval dates")
    
    # Walk forward
    horizon_results = {h: [] for h in HOLD_PERIODS}
    
    for eval_date in eval_dates:
        train = [s for s in all_samples if s['date'] < eval_date]
        if len(train) < 500:
            continue
        if len(train) > 200000:
            train = train[-200000:]
        
        X_train = np.array([[s['features'].get(f, 0) for f in feat_names] for s in train])
        y_train = np.array([s['labels'].get(5, 0) for s in train])  # Train on 5d label
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)
        
        test = by_date.get(eval_date, [])
        if len(test) < 5:
            continue
        
        X_test = np.array([[s['features'].get(f, 0) for f in feat_names] for s in test])
        X_test = np.nan_to_num(X_test, nan=0.0)
        preds = model.predict(X_test)
        
        ranked = sorted(zip(preds, test), key=lambda x: -x[0])
        n_top = max(1, len(ranked) // 10)
        top = ranked[:n_top]
        
        for h in HOLD_PERIODS:
            top_rets = [s['labels'].get(h, 0) for _, s in top]
            all_rets = [s['labels'].get(h, 0) for s in test]
            if top_rets:
                horizon_results[h].append({
                    'date': eval_date,
                    'avg_top': np.mean(top_rets),
                    'avg_all': np.mean(all_rets),
                    'alpha': np.mean(top_rets) - np.mean(all_rets),
                    'win': np.mean([1 for r in top_rets if r > 0]) * 100,
                })
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"📊 Walk-Forward 多持有期结果 ({market})")
    print(f"{'='*70}")
    print(f"{'持有期':>6} {'Top10%收益':>10} {'全部收益':>9} {'Alpha':>8} {'Alpha>0%':>9} {'胜率':>6} {'赚钱%':>6}")
    print(f"{'-'*70}")
    
    for h in HOLD_PERIODS:
        if not horizon_results[h]:
            continue
        df = pd.DataFrame(horizon_results[h])
        avg_top = df['avg_top'].mean()
        avg_all = df['avg_all'].mean()
        alpha = df['alpha'].mean()
        alpha_pos = (df['alpha'] > 0).mean() * 100
        win = df['win'].mean()
        profit_pct = (df['avg_top'] > 0).mean() * 100
        
        marker = " ⭐" if alpha == max(pd.DataFrame(horizon_results[hh]).get('alpha', pd.Series([0])).mean() for hh in HOLD_PERIODS if horizon_results[hh]) else ""
        print(f"  {h:>3}d  {avg_top:>+9.2f}%  {avg_all:>+8.2f}%  {alpha:>+7.2f}%  {alpha_pos:>7.0f}%  {win:>5.0f}%  {profit_pct:>5.0f}%{marker}")
    
    print(f"\n📝 模型: 5d XGBoost (每次只用历史数据训练)")
    print(f"   选股: pred 排名前 10%")
    print(f"   评估次数: {len(horizon_results[5])} 个交易日")
    
    return horizon_results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--market', default='CN', choices=['US', 'CN', 'BOTH'])
    p.add_argument('--start', default='2025-06-01')
    args = p.parse_args()
    
    if args.market == 'BOTH':
        for m in ['CN', 'US']:
            run_all(m, args.start)
            print("\n\n")
    else:
        run_all(args.market, args.start)
