#!/usr/bin/env python3
"""回填 2026 YTD 所有交易日的选股记录到 ml_daily_picks.db
对 npz 中每个 2026 交易日，如果 DB 里没有记录，就跑一次预测并存进去。
Usage: python ml_backfill_picks.py --market US
       python ml_backfill_picks.py --market BOTH
"""
import sys, os, time, sqlite3, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
sys.path.insert(0, '/Users/bertwang/Cursor/Coral_Creek/versions/v3')

import numpy as np
from pathlib import Path

V3 = Path('/Users/bertwang/Cursor/Coral_Creek/versions/v3')

def backfill_market(market):
    print(f"\n{'='*60}")
    print(f"🔄 Backfill {market} YTD picks")
    print(f"{'='*60}")
    t0 = time.time()

    # Load npz
    npz_path = f'/tmp/{market.lower()}_daily_full.npz'
    d = np.load(npz_path, allow_pickle=True)
    X, dates, symbols = d['X'], d['dates'], d['symbols']
    fn = list(d['fn'])
    all_dates_2026 = sorted(set(dt for dt in dates if dt >= '2026-01-01'))
    print(f"  npz 2026 dates: {len(all_dates_2026)}")

    # Check DB
    db_path = V3 / 'db' / 'ml_daily_picks.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute('''CREATE TABLE IF NOT EXISTS mmoe_daily_picks (
        date TEXT, market TEXT, tier TEXT, symbol TEXT, name TEXT,
        price REAL, mcap REAL, blend REAL,
        pred_5d REAL, pred_10d REAL, pred_20d REAL,
        actual_5d REAL, actual_10d REAL, actual_20d REAL,
        PRIMARY KEY (date, market, tier))''')
    existing = set(r[0] for r in conn.execute(
        'SELECT DISTINCT date FROM mmoe_daily_picks WHERE market=?', (market,)).fetchall())
    conn.close()
    
    missing = [d for d in all_dates_2026 if d not in existing]
    print(f"  DB has: {len(existing)} dates | Missing: {len(missing)}")
    if not missing:
        print("  ✅ Already complete!"); return

    # Import pipeline components
    import torch
    from ml.models.mmoe import MMoEModel
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    DEVICE = 'cpu'
    BLEND_W = np.array([0.15, 0.30, 0.55])

    # Load market cap + names
    cc = sqlite3.connect(str(V3 / 'db' / 'coral_creek.db'))
    mcap_dict = {}
    names_dict = {}
    if market == 'US':
        for r in cc.execute('SELECT symbol, market_cap FROM stock_meta').fetchall():
            if r[1]: mcap_dict[r[0]] = r[1]
        for r in cc.execute('SELECT symbol, sic_desc FROM stock_meta').fetchall():
            if r[1]: names_dict[r[0]] = r[1]
    else:
        for r in cc.execute("SELECT symbol, name FROM stock_info WHERE market='CN'").fetchall():
            names_dict[r[0]] = (r[1] or r[0])[:10]
        # CN mcap from stock_meta_cn
        try:
            for r in cc.execute('SELECT symbol, total_mv FROM stock_meta_cn WHERE total_mv>0').fetchall():
                sym = r[0]
                mv = r[1]
                if sym.startswith('6'):
                    mcap_dict[sym + '.SH'] = mv
                else:
                    mcap_dict[sym + '.SZ'] = mv
        except Exception as e:
            print(f'  ⚠️ CN mcap load: {e}')
    cc.close()
    print(f'  mcap_dict: {len(mcap_dict)} entries')

    # Tier definitions
    if market == 'US':
        tiers = [
            ('Micro ($50-300M)', 5e7, 3e8),
            ('Small ($300M-2B)', 3e8, 2e9),
            ('Mid ($2-10B)', 2e9, 1e10),
            ('Large ($10-200B)', 1e10, 2e11),
            ('Mega (>$200B)', 2e11, 1e15),
        ]
    else:
        tiers = [
            ('小盘 (20-100亿)', 2e9, 1e10),
            ('中盘 (100-500亿)', 1e10, 5e10),
            ('大盘 (>500亿)', 5e10, 1e15),
        ]

    # Process each missing date
    conn = sqlite3.connect(str(db_path))
    for di, eval_date in enumerate(missing):
        print(f"\n  [{di+1}/{len(missing)}] {eval_date}...", end=' ', flush=True)

        # Get training data: 120 days before eval_date
        mask_before = dates < eval_date
        train_dates = sorted(set(dates[mask_before]))
        if len(train_dates) < 30:
            print("⏭ not enough history"); continue
        train_window = train_dates[-120:]
        train_mask = np.isin(dates, train_window)
        
        Xtrain = X[train_mask]
        y5_tr = d['y5'][train_mask]
        y10_tr = d['y10'][train_mask]
        y20_tr = d['y20'][train_mask]

        # Filter valid training samples
        valid = np.isfinite(y5_tr) & np.isfinite(y10_tr) & np.isfinite(y20_tr)
        valid &= np.all(np.isfinite(Xtrain), axis=1)
        if valid.sum() < 100:
            print("⏭ too few valid samples"); continue

        Xt = Xtrain[valid]; y5t = y5_tr[valid]; y10t = y10_tr[valid]; y20t = y20_tr[valid]

        # Predict date data
        pred_mask = dates == eval_date
        if pred_mask.sum() == 0:
            print("⏭ no data"); continue
        Xpred = X[pred_mask]
        pred_syms = symbols[pred_mask]

        # XGBoost leaf features
        Xfin = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0).astype(np.float32)
        Xpfin = np.nan_to_num(Xpred, nan=0, posinf=0, neginf=0).astype(np.float32)

        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.6, tree_method='hist',
            verbosity=0, n_jobs=-1
        )
        xgb_model.fit(Xfin, y20t)
        leaves_tr = xgb_model.apply(Xfin)
        leaves_pred = xgb_model.apply(Xpfin)

        # Combined features
        Xae_tr = np.hstack([Xfin, leaves_tr]).astype(np.float32)
        Xae_pred = np.hstack([Xpfin, leaves_pred]).astype(np.float32)

        scaler = StandardScaler()
        Xae_tr = scaler.fit_transform(Xae_tr)
        Xae_pred = scaler.transform(Xae_pred)
        Xae_tr = np.nan_to_num(Xae_tr, nan=0, posinf=0, neginf=0).astype(np.float32)
        Xae_pred = np.nan_to_num(Xae_pred, nan=0, posinf=0, neginf=0).astype(np.float32)

        # MMoE
        in_dim = Xae_tr.shape[1]
        mmoe = MMoEModel(in_dim, num_experts=4, expert_hidden=128, num_tasks=3).to(DEVICE)
        optimizer = torch.optim.Adam(mmoe.parameters(), lr=1e-3, weight_decay=1e-5)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(Xae_tr),
            torch.FloatTensor(y5t), torch.FloatTensor(y10t), torch.FloatTensor(y20t)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

        mmoe.train()
        for epoch in range(8):
            for bx, b5, b10, b20 in loader:
                bx = bx.to(DEVICE)
                preds = mmoe(bx)
                loss = (torch.nn.functional.mse_loss(preds[0].squeeze(), b5.to(DEVICE)) +
                        torch.nn.functional.mse_loss(preds[1].squeeze(), b10.to(DEVICE)) +
                        torch.nn.functional.mse_loss(preds[2].squeeze(), b20.to(DEVICE)))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Predict
        mmoe.eval()
        with torch.no_grad():
            preds = mmoe(torch.FloatTensor(Xae_pred).to(DEVICE))
            p5 = np.clip(preds[0].cpu().numpy().flatten(), -200, 200)
            p10 = np.clip(preds[1].cpu().numpy().flatten(), -200, 200)
            p20 = np.clip(preds[2].cpu().numpy().flatten(), -200, 200)
        blend = BLEND_W[0] * p5 + BLEND_W[1] * p10 + BLEND_W[2] * p20

        # Get price from npz features (use Close feature if available)
        close_idx = fn.index('Close') if 'Close' in fn else None

        # Per-tier top-1
        picks_saved = 0
        for tier_name, lo, hi in tiers:
            best_idx, best_blend, best_sym = None, -999, None
            for i, sym in enumerate(pred_syms):
                mc = mcap_dict.get(sym, 0)
                mc = mcap_dict.get(sym, 0)
                if mc < lo or mc >= hi:
                    continue
                if blend[i] > best_blend:
                    best_blend = blend[i]
                    best_idx = i
                    best_sym = sym

            if best_sym is not None:
                # Get real price from stock_history
                hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))
                pr = hconn.execute(
                    'SELECT close FROM stock_history WHERE symbol=? AND market=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1',
                    (best_sym, market, eval_date)).fetchone()
                hconn.close()
                price = pr[0] if pr and pr[0] > 0 else 1.0
                name = names_dict.get(best_sym, '')
                mc = mcap_dict.get(best_sym, 0)

                conn.execute('''INSERT OR REPLACE INTO mmoe_daily_picks
                    (date,market,tier,symbol,name,price,mcap,blend,pred_5d,pred_10d,pred_20d)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (eval_date, market, tier_name, best_sym, name, price, mc,
                     float(best_blend), float(p5[best_idx]), float(p10[best_idx]), float(p20[best_idx])))
                picks_saved += 1

        conn.commit()
        print(f"✅ {picks_saved} picks", flush=True)

    conn.close()
    print(f"\n🎉 {market} backfill done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='BOTH', choices=['US', 'CN', 'BOTH'])
    args = parser.parse_args()

    markets = ['US', 'CN'] if args.market == 'BOTH' else [args.market]
    for m in markets:
        backfill_market(m)

    # After backfill, update returns
    print("\n📈 Updating returns...")
    from scripts.ml_daily_pipeline import track_returns
    for m in markets:
        track_returns(m)
    print("✅ All done!")
