#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回填 2026 年每日 ML Picks
=========================
遍历 feature_cache 中的日期，用 walk-forward 方式生成 Top picks，
保存到 ml_daily_picks.db。

用法:
    PYTHONPATH=. python scripts/backfill_daily_picks.py --market US
    PYTHONPATH=. python scripts/backfill_daily_picks.py --market CN
    PYTHONPATH=. python scripts/backfill_daily_picks.py --market CN --start 2026-01-01
"""
import os, sys, json, sqlite3, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from xgboost import XGBRegressor
from scripts.ml_daily_scorer import (
    CN_TIERS, US_TIERS, get_exchange, _init_picks_db, _save_picks
)


def backfill_picks(market='US', start_date='2026-01-01', end_date=None, top_n=3):
    """
    Walk-forward backfill daily picks from feature_cache.
    Uses 3-month training window, scores each day's stocks.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    cache_db = sqlite3.connect(os.path.join(parent_dir, 'db', 'ml_feature_cache.db'))
    
    # Get available dates
    dates = [r[0] for r in cache_db.execute(
        '''SELECT DISTINCT trade_date FROM feature_cache 
        WHERE market=? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date''',
        (market, start_date, end_date)
    ).fetchall()]
    
    print(f"\n{'='*60}")
    print(f"📊 Backfill ML Picks: {market}, {start_date} ~ {end_date}")
    print(f"   {len(dates)} trading days")
    print(f"{'='*60}")
    
    if not dates:
        print("❌ 无数据")
        cache_db.close()
        return
    
    # Load mcap data
    mcap_dict = {}
    mcap_file = 'cn_mcap_dict.json' if market == 'CN' else 'mcap_dict.json'
    mcap_path = os.path.join(parent_dir, 'db', mcap_file)
    if os.path.exists(mcap_path):
        with open(mcap_path) as f:
            mcap_dict = json.load(f)
    print(f"   Market cap data: {len(mcap_dict)} stocks")
    
    sector_dict = {}
    sec_file = 'cn_sector_dict.json' if market == 'CN' else 'sector_dict.json'
    sec_path = os.path.join(parent_dir, 'db', sec_file)
    if os.path.exists(sec_path):
        with open(sec_path) as f:
            sector_dict = json.load(f)
    
    tiers = CN_TIERS if market == 'CN' else US_TIERS
    min_price = 3 if market == 'CN' else 5
    # CN uses 30d as primary, US uses 10d
    train_label = 'label_30d' if market == 'CN' else 'label_10d'
    pred_horizon = '30d' if market == 'CN' else '10d'
    
    # Training window: CN has weekly dates so use fewer dates
    TRAIN_WINDOW = 8 if market == 'CN' else 60  # 8 weeks ≈ 2 months for CN
    MIN_TRAIN_SAMPLES = 50 if market == 'CN' else 100
    MIN_FEATURES = 20 if market == 'CN' else 50
    
    total_saved = 0
    t0 = time.time()
    
    for di, target_date in enumerate(dates):
        # Training data: previous TRAIN_WINDOW days
        train_end_idx = di
        train_start_idx = max(0, di - TRAIN_WINDOW)
        
        if train_end_idx - train_start_idx < 3:
            continue  # Not enough training data
        
        train_dates = dates[train_start_idx:train_end_idx]
        
        # Load training data
        placeholders = ','.join(['?'] * len(train_dates))
        train_rows = cache_db.execute(
            f'''SELECT features_json, {train_label} FROM feature_cache 
            WHERE market=? AND trade_date IN ({placeholders}) AND {train_label} IS NOT NULL''',
            [market] + train_dates
        ).fetchall()
        
        if len(train_rows) < MIN_TRAIN_SAMPLES:
            continue
        
        # Parse training features
        train_features = []
        train_labels = []
        for row in train_rows:
            feat = json.loads(row[0])
            price = feat.get('Close', feat.get('close', 0))
            if price < min_price:
                continue
            fv = {k: float(v) if v is not None else 0.0 for k, v in feat.items()
                  if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.','',1).replace('-','',1).isdigit())}
            train_features.append(fv)
            train_labels.append(float(row[1]) if row[1] else 0)
        
        if len(train_features) < MIN_FEATURES:
            continue
        
        train_df = pd.DataFrame(train_features)
        fn = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != 'Date']
        X_train = np.nan_to_num(train_df[fn].values.astype(np.float32), 0)
        y_train = np.clip(np.array(train_labels, dtype=np.float32), -50, 200)
        
        # Train model
        model = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Score target date stocks
        target_rows = cache_db.execute(
            '''SELECT symbol, features_json, label_5d, label_10d, label_20d, label_30d, label_60d
            FROM feature_cache WHERE market=? AND trade_date=?''',
            (market, target_date)
        ).fetchall()
        
        if not target_rows:
            continue
        
        scored = []
        for row in target_rows:
            symbol, fj, l5, l10, l20, l30, l60 = row
            feat = json.loads(fj)
            price = feat.get('Close', feat.get('close', 0))
            if price < min_price:
                continue
            
            fv = {k: float(v) if v is not None else 0.0 for k, v in feat.items()
                  if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.','',1).replace('-','',1).isdigit())}
            
            X = np.array([[float(fv.get(f, 0)) for f in fn]], dtype=np.float32)
            X = np.nan_to_num(X, 0)
            pred = float(model.predict(X)[0])
            
            mc = mcap_dict.get(symbol, 0)
            sec = sector_dict.get(symbol, {})
            exchange = get_exchange(symbol, market)
            
            scored.append({
                'symbol': symbol,
                'price': price,
                'exchange': exchange,
                'pred_5d': float(l5) if l5 else 0,  # actual for backfill
                'pred_10d': pred if market == 'US' else 0,
                'pred_20d': 0,
                'pred_30d': pred if market == 'CN' else 0,
                'pred_60d': 0,
                'primary_pred': pred,
                'market_cap': mc,
                'mcap_b': mc / 1e9 if mc > 0 else 0,
                'sector': sec.get('sic', sec.get('industry', ''))[:30] if isinstance(sec, dict) else str(sec)[:30],
                'holding_period': pred_horizon,
                # Store actuals for tracking
                '_actual_10d': float(l10) if l10 else None,
                '_actual_30d': float(l30) if l30 else None,
                '_actual_60d': float(l60) if l60 else None,
            })
        
        if not scored:
            continue
        
        scored_df = pd.DataFrame(scored).sort_values('primary_pred', ascending=False)
        
        # Build results dict
        results = {}
        if market == 'CN':
            for exchange in ['上证主板', '深证主板', '创业板', '科创板']:
                ex_df = scored_df[scored_df['exchange'] == exchange]
                if len(ex_df) == 0:
                    continue
                for tier_name, lo, hi in tiers:
                    tier_df = ex_df[(ex_df['market_cap'] >= lo) & (ex_df['market_cap'] < hi)]
                    if len(tier_df) == 0:
                        continue
                    key = f"{exchange} | {tier_name}"
                    results[key] = tier_df.head(top_n).to_dict('records')
                # Exchange-level
                key = f"{exchange} | 全部"
                results[key] = ex_df.head(top_n).to_dict('records')
        else:
            for tier_name, lo, hi in tiers:
                tier_df = scored_df[(scored_df['market_cap'] >= lo) & (scored_df['market_cap'] < hi)]
                if len(tier_df) == 0:
                    continue
                results[tier_name] = tier_df.head(top_n).to_dict('records')
        
        # Save
        _save_picks(target_date, market, results)
        total_saved += sum(len(v) for v in results.values())
        
        # Also backfill actual returns
        conn = _init_picks_db()
        for segment, picks in results.items():
            for p in picks:
                for act_col, db_col in [('_actual_10d', 'actual_10d'), ('_actual_30d', 'actual_30d'), ('_actual_60d', 'actual_60d')]:
                    if p.get(act_col) is not None:
                        conn.execute(f'''UPDATE ml_picks_v2 SET {db_col}=?
                            WHERE date=? AND symbol=? AND market=?''',
                            (p[act_col], target_date, p['symbol'], market))
        conn.commit()
        conn.close()
        
        if (di + 1) % 5 == 0 or di == len(dates) - 1:
            elapsed = time.time() - t0
            rate = (di + 1) / elapsed
            eta = (len(dates) - di - 1) / rate if rate > 0 else 0
            print(f"   📅 {target_date} [{di+1}/{len(dates)}] "
                  f"saved={total_saved}, {elapsed:.0f}s, ETA={eta:.0f}s", flush=True)
    
    cache_db.close()
    print(f"\n🎉 Backfill complete: {total_saved} picks saved across {len(dates)} days")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backfill Daily ML Picks')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--start', default='2026-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--top', type=int, default=3)
    
    args = parser.parse_args()
    backfill_picks(args.market, args.start, args.end, args.top)
