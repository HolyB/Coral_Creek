#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGB+MMoE 每日生产 Pipeline
===========================
每个交易日收盘后运行:
  1. 拉最新行情 (Polygon US / Tushare CN)
  2. 增量更新 npz 特征
  3. XGB+MMoE 训练+预测 → 每个市值 tier 各选 Top-1
  4. 存到本地 DB + Supabase 云端
  5. 追踪过去20天选股的实际收益
  6. 发邮件 + 推送报告

Usage:
  PYTHONPATH=. python scripts/ml_daily_pipeline.py --market US
  PYTHONPATH=. python scripts/ml_daily_pipeline.py --market CN
  PYTHONPATH=. python scripts/ml_daily_pipeline.py --market BOTH
"""
import os, sys, json, sqlite3, time, warnings, gc
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
from dotenv import load_dotenv
load_dotenv(V3 / '.env')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ===========================================================
# Config
# ===========================================================
WINDOW = 120          # training window (trading days)
BLEND_W = (0.2, 0.3, 0.5)  # blend weights for 5d/10d/20d
MIN_MCAP_US = 50e6;   MIN_MCAP_CN = 20e8
MIN_PRICE_US = 5.0;   MIN_PRICE_CN = 3.0

US_TIERS = [
    ('Mega (>$100B)',    100e9),
    ('Large ($10-100B)', 10e9),
    ('Mid ($2-10B)',     2e9),
    ('Small ($300M-2B)', 300e6),
    ('Micro ($50-300M)', 50e6),
]
CN_TIERS = [
    ('大盘 (>500亿)',    500e8),
    ('中盘 (100-500亿)', 100e8),
    ('小盘 (20-100亿)',  20e8),
]

def get_tier(mc, market):
    tiers = US_TIERS if market == 'US' else CN_TIERS
    for name, threshold in tiers:
        if mc >= threshold:
            return name
    return 'Nano' if market == 'US' else '微盘 (<20亿)'


# ===========================================================
# MMoE Model (same as backtest)
# ===========================================================
class MMoE(nn.Module):
    def __init__(self, dim, n_experts=4, hdim=128, n_tasks=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hdim), nn.BatchNorm1d(hdim), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hdim, hdim//2), nn.BatchNorm1d(hdim//2), nn.ReLU(),
        ) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, n_experts), nn.Softmax(dim=-1)
        ) for _ in range(n_tasks)])
        self.towers = nn.ModuleList([nn.Sequential(
            nn.Linear(hdim//2, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        ) for _ in range(n_tasks)])
    def forward(self, x):
        x = self.bn(x)
        eo = torch.stack([e(x) for e in self.experts], 1)
        return [self.towers[i]((eo * self.gates[i](x).unsqueeze(-1)).sum(1)).squeeze(-1) for i in range(3)]


def train_model(model, X, ys, epochs=8):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.FloatTensor(X), *[torch.FloatTensor(y) for y in ys])
    dl = DataLoader(ds, batch_size=4096, shuffle=True, drop_last=True)
    for _ in range(epochs):
        for batch in dl:
            x = batch[0].to(DEVICE); yy = [b.to(DEVICE) for b in batch[1:]]
            loss = sum(nn.HuberLoss(delta=10.0)(p, y) for p, y in zip(model(x), yy)) / len(yy)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


# ===========================================================
# Step 1: Update market data
# ===========================================================
def update_market_data(market):
    """Pull latest OHLCV via API"""
    print(f"  📥 Updating {market} market data...", flush=True)
    try:
        from db.stock_history import update_stock_history
        update_stock_history(market=market)
        print(f"  ✅ Market data updated", flush=True)
    except Exception as e:
        print(f"  ⚠️ Market data update skipped: {e}", flush=True)


# ===========================================================
# Step 2: Load npz + Train + Predict
# ===========================================================
def predict_today(market):
    """Load npz, train on recent data, predict on latest date, return per-tier top-1.
    Caches results: if today's picks already exist in DB, return cached version."""

    # Check cache first: if picks for today already exist, use them
    force = os.environ.get('ML_FORCE_PREDICT', '').lower() == 'true'
    db_path = V3 / 'db' / 'ml_daily_picks.db'
    if not force and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            today_picks_raw = conn.execute(
                "SELECT date, tier, symbol, name, price, blend, pred_20d, pred_5d, pred_10d, mcap "
                "FROM mmoe_daily_picks WHERE market=? ORDER BY date DESC LIMIT 10",
                (market,)
            ).fetchall()
            conn.close()
        except Exception:
            today_picks_raw = []
        if today_picks_raw:
            latest_date = today_picks_raw[0][0]
            # Get the eval_date from npz to compare
            npz_path = f'/tmp/{market.lower()}_daily_full.npz'
            if os.path.exists(npz_path):
                d = np.load(npz_path, allow_pickle=True)
                dates_all = d['dates']
                eval_date = str(np.unique(dates_all)[-1])
                if latest_date == eval_date:
                    print(f"  📦 Using cached picks for {eval_date}", flush=True)
                    picks = {}
                    top3 = {}
                    for r in today_picks_raw:
                        if r[0] != eval_date:
                            break
                        tier = r[1]
                        pick = {
                            'symbol': r[2], 'name': r[3], 'price': r[4],
                            'blend': r[5], 'pred_20d': r[6],
                            'pred_5d': r[7], 'pred_10d': r[8],
                            'mcap': r[9] or 0, 'tier': tier,
                        }
                        picks[tier] = pick
                        top3[tier] = [pick]  # Only top-1 from cache
                    if picks:
                        print(f"  ✅ {len(picks)} cached picks loaded", flush=True)
                        return eval_date, picks, top3

    npz_path = f'/tmp/{market.lower()}_daily_full.npz'
    if not os.path.exists(npz_path):
        print(f"  ❌ {npz_path} not found")
        return None, None, None

    print(f"  📦 Loading {npz_path}...", flush=True)
    d = np.load(npz_path, allow_pickle=True)
    X_all = d['X']; dates_all = d['dates']; symbols_all = d['symbols']
    y5, y10, y20 = d['y5'], d['y10'], d['y20']
    fn = list(d['fn'])

    unique_dates = sorted(set(dates_all))
    eval_date = unique_dates[-1]
    print(f"  📅 Predict date: {eval_date} | {X_all.shape[0]:,} × {X_all.shape[1]}", flush=True)

    # Train on recent WINDOW days (excluding eval_date)
    train_end_idx = unique_dates.index(eval_date)
    train_dates = unique_dates[max(0, train_end_idx - WINDOW):train_end_idx]
    tm = np.isin(dates_all, train_dates)
    X_tr = X_all[tm]
    ys_tr = [y5[tm], y10[tm], y20[tm]]
    v = ~np.isnan(ys_tr[2])
    for y in ys_tr: v &= ~np.isnan(y)
    X_tr = np.nan_to_num(X_tr[v], nan=0.0)
    ys_tr = [y[v] for y in ys_tr]
    print(f"  🏋️ Training on {len(X_tr):,} samples ({len(train_dates)} days)...", flush=True)

    # Fixed seed for reproducibility (based on date)
    seed = int(eval_date.replace('-', '')) % 2**31
    np.random.seed(seed)
    torch.manual_seed(seed)

    # XGBoost
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr).astype(np.float32)
    np.nan_to_num(Xs, copy=False, nan=0, posinf=0, neginf=0)
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        tree_method='hist', n_jobs=-1, verbosity=0, random_state=seed)
    xgb.fit(X_tr, ys_tr[-1])
    leaf_tr = xgb.apply(X_tr).astype(np.float32)
    Xa = np.hstack([Xs, leaf_tr])

    # MMoE
    mmoe = MMoE(Xa.shape[1], 4, 128, 3)
    mmoe = train_model(mmoe, Xa, ys_tr)
    del X_tr, Xs, leaf_tr, Xa; gc.collect()

    # Predict on eval_date
    em = dates_all == eval_date
    X_ev = np.nan_to_num(X_all[em], nan=0.0)
    syms_ev = symbols_all[em]
    print(f"  🔮 Predicting {len(X_ev):,} stocks...", flush=True)

    Xe = sc.transform(X_ev).astype(np.float32)
    np.nan_to_num(Xe, copy=False, nan=0, posinf=0, neginf=0)
    leaf_ev = xgb.apply(X_ev).astype(np.float32)
    Xae = np.hstack([Xe, leaf_ev])

    with torch.no_grad():
        preds = mmoe(torch.FloatTensor(Xae).to(DEVICE))
        p5 = np.clip(preds[0].cpu().numpy(), -200, 200)
        p10 = np.clip(preds[1].cpu().numpy(), -200, 200)
        p20 = np.clip(preds[2].cpu().numpy(), -200, 200)
    blend = BLEND_W[0] * p5 + BLEND_W[1] * p10 + BLEND_W[2] * p20

    # Load market cap + names
    conn = sqlite3.connect(str(V3 / 'db' / 'coral_creek.db'))
    mcap_dict = {}
    names_dict = {}
    if market == 'US':
        try:
            for r in conn.execute('SELECT symbol, market_cap FROM stock_meta'):
                mcap_dict[r[0]] = r[1] or 0
                names_dict[r[0]] = r[0]
        except Exception:
            # stock_meta may not exist — use stock_info fallback
            for r in conn.execute("SELECT symbol, name FROM stock_info WHERE market='US'"):
                names_dict[r[0]] = r[1] or r[0]
    else:
        for r in conn.execute("SELECT symbol, name FROM stock_info WHERE market='CN'"):
            names_dict[r[0]] = (r[1] or r[0])[:10]
        # CN mcap from stock_meta_cn
        try:
            for r in conn.execute('SELECT symbol, total_mv FROM stock_meta_cn'):
                sym = r[0]
                mv = r[1] or 0
                # Map pure digits to exchange suffix
                if sym.startswith('6'):
                    mcap_dict[sym + '.SH'] = mv
                else:
                    mcap_dict[sym + '.SZ'] = mv
        except:
            pass
    conn.close()

    # Price from stock_history
    hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))
    min_mcap = MIN_MCAP_US if market == 'US' else MIN_MCAP_CN
    min_price = MIN_PRICE_US if market == 'US' else MIN_PRICE_CN

    # Per-tier top-1
    tier_candidates = defaultdict(list)
    for i in range(len(syms_ev)):
        sym = syms_ev[i]
        mc = mcap_dict.get(sym, 0)
        if mcap_dict and mc < min_mcap: continue  # skip filter if no mcap data at all
        # Get latest close + today's OHLCV to check limit-up
        pr = hconn.execute(
            "SELECT trade_date, open, close, high, low, volume FROM stock_history WHERE symbol=? AND market=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 2",
            (sym, market, eval_date)
        ).fetchall()
        if not pr: continue
        price = pr[0][2]  # today's close as buy price
        if price is None or price < min_price: continue

        # Filter: 涨停板 / 一字板 / 低换手 (can't buy)
        if market == 'CN' and len(pr) >= 2:
            today_open, today_close, today_high, today_low = pr[0][1], pr[0][2], pr[0][3], pr[0][4]
            today_vol = pr[0][5] or 0
            prev_close = pr[1][2]
            if prev_close and prev_close > 0 and today_open:
                # Determine limit rate
                if sym.startswith('688') or sym.startswith('300'):
                    limit_pct = 0.20
                elif 'ST' in names_dict.get(sym, '').upper():
                    limit_pct = 0.05
                else:
                    limit_pct = 0.10
                upper_limit = round(prev_close * (1 + limit_pct), 2)
                # Case 1: 开盘涨停
                if today_open >= upper_limit:
                    continue
                # Case 2: 一字涨停 (open=close=high, near limit)
                if today_open == today_close == today_high and today_close >= upper_limit * 0.99:
                    continue
                # Case 3: 低换手封板 (volume < 50万股 且涨幅接近涨停)
                chg = (today_close / prev_close - 1)
                if today_vol < 500000 and chg >= limit_pct * 0.9:
                    continue
        elif market == 'US' and len(pr) >= 2:
            today_open = pr[0][1]
            prev_close = pr[1][2]
            if prev_close and prev_close > 0 and today_open:
                gap = (today_open / prev_close - 1)
                if gap > 0.15:  # US: skip >15% gap-up at open
                    continue

        tier = get_tier(mc, market)
        tier_candidates[tier].append({
            'symbol': sym,
            'name': names_dict.get(sym, sym),
            'price': price,
            'blend': float(blend[i]),
            'pred_5d': float(p5[i]),
            'pred_10d': float(p10[i]),
            'pred_20d': float(p20[i]),
            'mcap': mc,
            'tier': tier,
        })

    # Sort each tier and pick top-1 (also keep top-3 for multi-strategy)
    picks = {}
    top3 = {}
    for tier, cands in tier_candidates.items():
        cands.sort(key=lambda x: -x['blend'])
        picks[tier] = cands[0]  # Top-1
        top3[tier] = cands[:3]  # Top-3

    hconn.close()
    print(f"  ✅ {len(picks)} tier picks generated", flush=True)
    return eval_date, picks, top3


# ===========================================================
# Step 3: Save picks to DB (local + Supabase)
# ===========================================================
def save_picks(eval_date, market, picks):
    """Save to local SQLite + Supabase"""
    # Local
    db_path = V3 / 'db' / 'ml_daily_picks.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS mmoe_daily_picks (
        date TEXT, market TEXT, tier TEXT, symbol TEXT, name TEXT,
        price REAL, blend REAL, pred_5d REAL, pred_10d REAL, pred_20d REAL,
        mcap REAL, actual_5d REAL, actual_10d REAL, actual_20d REAL,
        created_at TEXT, PRIMARY KEY(date, market, tier)
    )""")
    for tier, p in picks.items():
        conn.execute("""INSERT OR REPLACE INTO mmoe_daily_picks
            (date,market,tier,symbol,name,price,blend,pred_5d,pred_10d,pred_20d,mcap,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (eval_date, market, tier, p['symbol'], p['name'], p['price'],
             p['blend'], p['pred_5d'], p['pred_10d'], p['pred_20d'],
             p['mcap'], datetime.now().isoformat()))
    conn.commit()
    print(f"  💾 Local: {len(picks)} picks saved to mmoe_daily_picks", flush=True)

    # Supabase (optional)
    try:
        from supabase import create_client
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        if url and key:
            sb = create_client(url, key)
            for tier, p in picks.items():
                sb.table('mmoe_daily_picks').upsert({
                    'date': eval_date, 'market': market, 'tier': tier,
                    'symbol': p['symbol'], 'name': p['name'],
                    'price': p['price'], 'blend': p['blend'],
                    'pred_5d': p['pred_5d'], 'pred_10d': p['pred_10d'],
                    'pred_20d': p['pred_20d'], 'mcap': p['mcap'],
                }).execute()
            print(f"  ☁️  Supabase: {len(picks)} picks synced", flush=True)
    except Exception as e:
        print(f"  ⚠️ Supabase sync skipped: {e}", flush=True)

    conn.close()


# ===========================================================
# Step 4: Track 20-day returns for past picks
# ===========================================================
def track_returns(market):
    """Backfill actual returns for picks from the last 20 days"""
    db_path = V3 / 'db' / 'ml_daily_picks.db'
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))

    # Get ALL picks missing any return value
    rows = conn.execute(
        """SELECT date, symbol, price, tier FROM mmoe_daily_picks
           WHERE market=? AND (actual_5d IS NULL OR actual_10d IS NULL OR actual_20d IS NULL)""",
        (market,)
    ).fetchall()

    updated = 0
    for pick_date, symbol, buy_price, tier in rows:
        if buy_price <= 0: continue
        # Get all trading days after pick_date for this symbol
        future = hconn.execute("""
            SELECT trade_date, close FROM stock_history
            WHERE symbol=? AND market=? AND trade_date>?
            ORDER BY trade_date
        """, (symbol, market, pick_date)).fetchall()

        for days, col in [(5, 'actual_5d'), (10, 'actual_10d'), (20, 'actual_20d')]:
            if len(future) >= days:
                close_price = future[days-1][1]
                ret = (close_price / buy_price - 1) * 100
                conn.execute(f"UPDATE mmoe_daily_picks SET {col}=? WHERE date=? AND market=? AND tier=?",
                    (ret, pick_date, market, tier))
                updated += 1
    conn.commit()
    conn.close(); hconn.close()
    print(f"  📊 Updated {updated} return values", flush=True)


# ===========================================================
# Step 4b: Check expiring picks and send sell reminders
# ===========================================================
def check_expiring_picks(market):
    """Check if any picks have reached their 5d/10d/20d holding period today"""
    db_path = V3 / 'db' / 'ml_daily_picks.db'
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))

    price_sym = "$" if market == 'US' else "¥"
    market_emoji = "🇺🇸" if market == 'US' else "🇨🇳"
    market_name = "美股" if market == 'US' else "A股"

    # Get all picks that might be expiring
    rows = conn.execute(
        """SELECT date, tier, symbol, name, price, pred_5d, pred_10d, pred_20d
           FROM mmoe_daily_picks WHERE market=? AND date >= date('now', '-25 days')""",
        (market,)
    ).fetchall()

    expiring = []
    for pick_date, tier, sym, name, buy_price, pred5, pred10, pred20 in rows:
        if buy_price <= 0: continue
        # Count trading days since pick
        trading_days = hconn.execute(
            """SELECT COUNT(*) FROM (
                SELECT DISTINCT trade_date FROM stock_history
                WHERE symbol=? AND market=? AND trade_date>?
                ORDER BY trade_date)""",
            (sym, market, pick_date)
        ).fetchone()[0]

        # Get current price
        cur = hconn.execute(
            "SELECT close FROM stock_history WHERE symbol=? AND market=? ORDER BY trade_date DESC LIMIT 1",
            (sym, market)).fetchone()
        cur_price = cur[0] if cur else buy_price
        cur_ret = (cur_price / buy_price - 1) * 100

        for horizon, pred, label in [(5, pred5, '5D'), (10, pred10, '10D'), (20, pred20, '20D')]:
            if trading_days == horizon:
                emoji = "🟢" if cur_ret > 0 else "🔴"
                expiring.append({
                    'horizon': label,
                    'pick_date': pick_date,
                    'tier': tier,
                    'symbol': sym,
                    'name': name or sym,
                    'buy_price': buy_price,
                    'cur_price': cur_price,
                    'return': cur_ret,
                    'predicted': pred or 0,
                    'emoji': emoji,
                })

    hconn.close()
    conn.close()

    if not expiring:
        print(f"  📭 No expiring picks today", flush=True)
        return

    # Build notification
    lines = [
        f"⏰ *{market_emoji} {market_name} 到期提醒*",
        f"📅 {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    for e in sorted(expiring, key=lambda x: x['horizon']):
        lines.append(
            f"  {e['emoji']} *{e['horizon']}到期* `{e['symbol']}` [{e['tier'][:8]}]\n"
            f"     买入: {price_sym}{e['buy_price']:.2f} → 现价: {price_sym}{e['cur_price']:.2f}\n"
            f"     实际: {e['return']:+.1f}% | 预测: {e['predicted']:+.1f}%\n"
            f"     📆 买入日: {e['pick_date']}"
        )

    lines.append("")
    lines.append("💡 *建议*: 到期持仓已达目标周期，请根据实际情况决定是否卖出")

    msg = "\n".join(lines)
    print(f"  ⏰ {len(expiring)} picks expiring today", flush=True)

    try:
        from services.notification import NotificationManager
        nm = NotificationManager()
        results = nm.send_all(f"{market_name} 到期提醒 ({len(expiring)}只)", msg)
        for ch, ok in results.items():
            print(f"    {'✅' if ok else '❌'} {ch}", flush=True)
    except Exception as e:
        print(f"  ⚠️ Expiry notification failed: {e}", flush=True)# ===========================================================
# Step 5: Generate report + send notifications
# ===========================================================
def send_report(eval_date, market, picks):
    """Send email + push notification with comprehensive report"""
    price_sym = "$" if market == 'US' else "¥"
    market_emoji = "🇺🇸" if market == 'US' else "🇨🇳"
    market_name = "美股" if market == 'US' else "A股"

    # Load YTD picks from DB
    db_path = V3 / 'db' / 'ml_daily_picks.db'
    recent_picks = []
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        recent_picks = conn.execute(
            """SELECT date,tier,symbol,name,price,blend,pred_20d,
                      actual_5d,actual_10d,actual_20d
               FROM mmoe_daily_picks WHERE market=? AND date>='2026-01-01'
               ORDER BY date DESC""",
            (market,)
        ).fetchall()
        conn.close()

    # Get current prices for recent picks
    hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))

    # ===== Section 1: Today's per-tier top-1 =====
    lines = [
        f"🤖 *{market_emoji} {market_name} XGB+MMoE 每日选股*",
        f"📅 {eval_date}",
        "",
        "━━━ 🎯 今日各市值 Top-1 ━━━",
    ]
    for tier in sorted(picks.keys()):
        p = picks[tier]
        lines.append(
            f"  📊 *{tier}*\n"
            f"     `{p['symbol']}` {p.get('name','')} {price_sym}{p['price']:.2f}\n"
            f"     5d={p['pred_5d']:+.1f}% 10d={p['pred_10d']:+.1f}% 20d={p['pred_20d']:+.1f}%"
        )

    # ===== Section 2: 最近60天选股记录 =====
    if recent_picks:
        lines.append("")
        lines.append(f"━━━ 📋 2026 YTD 选股记录 ({len(recent_picks)} 笔) ━━━")
        for r in recent_picks[:30]:  # Show latest 30
            dt, tier, sym, name, price, blend, pred, a5, a10, a20 = r
            cur = hconn.execute(
                "SELECT close FROM stock_history WHERE symbol=? AND market=? ORDER BY trade_date DESC LIMIT 1",
                (sym, market)).fetchone()
            cur_p = cur[0] if cur else price
            cur_ret = (cur_p / price - 1) * 100 if price > 0 else 0
            def _f(v): return f"{v:+.1f}%" if v is not None else "—"
            marker = "🟢" if cur_ret > 0 else "🔴"
            tier_short = tier[:8]
            lines.append(
                f"  {marker} {dt} `{sym}` {name or ''} {price_sym}{price:.2f}→{price_sym}{cur_p:.2f} "
                f"({cur_ret:+.1f}%) 5d:{_f(a5)} 10d:{_f(a10)} 20d:{_f(a20)} [{tier_short}]"
            )
        if len(recent_picks) > 30:
            lines.append(f"  ... 还有 {len(recent_picks)-30} 笔")

    # ===== Section 3: Per-tier cumulative stats =====
    if recent_picks:
        lines.append("")
        lines.append("━━━ 📊 各 Tier 累计表现 (YTD) ━━━")
        tier_data = defaultdict(list)
        for r in recent_picks:
            tier_data[r[1]].append(r)

        overall_rets = []
        for tier in sorted(tier_data.keys()):
            tpicks = tier_data[tier]
            rets = [r[9] for r in tpicks if r[9] is not None]
            if not rets:
                rets = [r[8] for r in tpicks if r[8] is not None]
            if rets:
                avg = np.mean(rets)
                wr = sum(1 for r in rets if r > 0) / len(rets) * 100
                overall_rets.extend(rets)
                emoji = "🟢" if avg > 0 else "🔴"
                lines.append(f"  {emoji} *{tier}*: {len(tpicks)}笔 avg={avg:+.1f}% WR={wr:.0f}%")
            else:
                lines.append(f"  ⏳ *{tier}*: {len(tpicks)}笔 (未结算)")

        if overall_rets:
            o_avg = np.mean(overall_rets)
            o_wr = sum(1 for r in overall_rets if r > 0) / len(overall_rets) * 100
            lines.append(f"\n  🏆 *总计*: {len(recent_picks)}笔 avg={o_avg:+.1f}% WR={o_wr:.0f}%")

    # ===== Section 4: Due for sale (策略到期提醒) =====
    if recent_picks:
        from datetime import timedelta
        hold_configs = [
            ('5天策略', 5),
            ('10天', 10),
            ('20天策略', 20),
        ]
        due_lines = []
        for r in recent_picks:
            dt, tier, sym, name, price, blend, pred, a5, a10, a20 = r
            days_held_row = hconn.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM stock_history WHERE symbol=? AND market=? AND trade_date>?",
                (sym, market, dt)
            ).fetchone()[0]
            cur = hconn.execute(
                "SELECT close FROM stock_history WHERE symbol=? AND market=? ORDER BY trade_date DESC LIMIT 1",
                (sym, market)).fetchone()
            cur_p = cur[0] if cur else price
            cur_ret = (cur_p / price - 1) * 100 if price > 0 else 0
            # Check against hold periods
            for config_name, hold_days in hold_configs:
                if days_held_row >= hold_days - 1 and days_held_row <= hold_days + 1:
                    emoji = '🟢' if cur_ret > 0 else '🔴'
                    due_lines.append(
                        f"  {emoji} `{sym}` {name or ''} [{config_name}] {days_held_row}d {price_sym}{price:.2f}→{price_sym}{cur_p:.2f} ({cur_ret:+.1f}%) [{tier[:8]}]"
                    )
                    break

        if due_lines:
            lines.append("")
            lines.append("━━━ ⏰ 到期可卖 ━━━")
            seen = set()
            for dl in due_lines:
                if dl not in seen:
                    lines.append(dl)
                    seen.add(dl)

    hconn.close()
    lines.append("")
    lines.append("⚠️ 仅供参考，不构成投资建议")
    lines.append("🔗 [在线查看](https://facaila.streamlit.app/)")
    msg = "\n".join(lines)

    # Push notifications
    try:
        from services.notification import NotificationManager
        nm = NotificationManager()
        results = nm.send_all(f"{market_name} XGB+MMoE 每日选股", msg)
        for ch, ok in results.items():
            print(f"  {'✅' if ok else '❌'} {ch}", flush=True)
    except Exception as e:
        print(f"  ⚠️ Push failed: {e}", flush=True)

    # Email (HTML) with strategy chart
    try:
        from scripts.ml_backtest_report import send_email_report
        from scripts.ml_strategy_chart import generate_strategy_chart
        chart_b64 = generate_strategy_chart(market, 365)
        html = _build_daily_html(eval_date, market, picks, recent_picks, chart_b64)
        send_email_report(html, market)
    except Exception as e:
        print(f"  ⚠️ Email skipped: {e}", flush=True)


def _build_daily_html(eval_date, market, picks, recent_picks=None, chart_b64=None):
    """Build premium dark-themed HTML with 4 sections: today + 60d history + tier stats + strategy chart"""
    price_sym = "$" if market == 'US' else "¥"
    market_name = "美股" if market == 'US' else "A股"
    market_emoji = "🇺🇸" if market == 'US' else "🇨🇳"

    # Tier color map (unified: 大=gold, 大中=blue, 中=purple, 小=pink, 微=cyan)
    if market == 'US':
        tier_colors = {
            'Mega (>$100B)': '#f59e0b',      # gold
            'Large ($10-100B)': '#3b82f6',    # blue
            'Mid ($2-10B)': '#8b5cf6',        # purple
            'Small ($300M-2B)': '#ec4899',    # pink
            'Micro ($50-300M)': '#06b6d4',    # cyan
        }
    else:
        tier_colors = {
            '大盘 (>500亿)': '#f59e0b',          # gold  (=Mega/Large)
            '中盘 (100-500亿)': '#8b5cf6',      # purple (=Mid)
            '小盘 (20-100亿)': '#ec4899',       # pink   (=Small)
        }
    def _tier_color(tier):
        return tier_colors.get(tier, '#6366f1')

    # ==== Section 1: Today's pick cards ====
    pick_cards = ""
    for tier in sorted(picks.keys()):
        p = picks[tier]
        mc = p['mcap']
        mcs = f"${mc/1e9:.1f}B" if market == 'US' and mc >= 1e9 else f"${mc/1e6:.0f}M" if market == 'US' else f"{mc/1e8:.0f}亿"
        def _vc(v): return '#22c55e' if v > 0 else '#ef4444' if v < 0 else '#ffffff'
        p5c = _vc(p['pred_5d'])
        p10c = _vc(p['pred_10d'])
        p20c = _vc(p['pred_20d'])
        tc = _tier_color(tier)
        pick_cards += f"""
        <div style="background:#1e293b;border-radius:10px;padding:16px;margin-bottom:10px;border-left:4px solid {tc}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div>
              <span style="font-size:11px;color:{tc};letter-spacing:1px;font-weight:700">● {tier}</span><br>
              <span style="font-size:20px;font-weight:800;color:#ffffff">{p['symbol']}</span>
              <span style="color:#cbd5e1;font-size:12px;margin-left:8px">{p['name']}</span>
            </div>
            <div style="text-align:right">
              <div style="font-size:18px;font-weight:700;color:#ffffff">{price_sym}{p['price']:.2f}</div>
              <div style="font-size:11px;color:#cbd5e1">{mcs}</div>
            </div>
          </div>
          <div style="display:flex;gap:8px">
            <div style="flex:1;background:#334155;border-radius:6px;padding:8px;text-align:center">
              <div style="font-size:16px;font-weight:700;color:{p5c}">{p['pred_5d']:+.1f}%</div>
              <div style="font-size:9px;color:#94a3b8">5D PRED</div>
            </div>
            <div style="flex:1;background:#334155;border-radius:6px;padding:8px;text-align:center">
              <div style="font-size:16px;font-weight:700;color:{p10c}">{p['pred_10d']:+.1f}%</div>
              <div style="font-size:9px;color:#94a3b8">10D PRED</div>
            </div>
            <div style="flex:1;background:#334155;border-radius:6px;padding:8px;text-align:center">
              <div style="font-size:16px;font-weight:700;color:{p20c}">{p['pred_20d']:+.1f}%</div>
              <div style="font-size:9px;color:#94a3b8">20D PRED</div>
            </div>
            <div style="flex:1;background:#334155;border-radius:6px;padding:8px;text-align:center">
              <div style="font-size:16px;font-weight:700;color:#a5b4fc">{p['blend']:+.1f}</div>
              <div style="font-size:9px;color:#94a3b8">BLEND</div>
            </div>
          </div>
        </div>"""

    # ==== Section 1b: Due for sale (到期可卖) in HTML ====
    due_html = ""
    if recent_picks:
        hconn_due = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))
        due_hold_configs = [('5天策略', 5), ('10天', 10), ('20天策略', 20)]
        due_items = []
        seen_due = set()
        for r in recent_picks:
            dt, tier, sym, name, price, blend, pred, a5, a10, a20 = r
            days_held = hconn_due.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM stock_history WHERE symbol=? AND market=? AND trade_date>?",
                (sym, market, dt)
            ).fetchone()[0]
            cur = hconn_due.execute(
                "SELECT close FROM stock_history WHERE symbol=? AND market=? ORDER BY trade_date DESC LIMIT 1",
                (sym, market)).fetchone()
            cur_p = cur[0] if cur else price
            cur_ret = (cur_p / price - 1) * 100 if price > 0 else 0
            for config_name, hold_days in due_hold_configs:
                if days_held >= hold_days - 1 and days_held <= hold_days + 1:
                    key = f"{sym}_{config_name}"
                    if key not in seen_due:
                        seen_due.add(key)
                        rc = '#22c55e' if cur_ret > 0 else '#ef4444'
                        due_items.append(f"""
                        <tr>
                          <td style="font-weight:700">{sym}<br><small style="color:#64748b">{name or ''}</small></td>
                          <td style="text-align:center;font-size:10px;color:{_tier_color(tier)}">{tier[:8]}</td>
                          <td style="text-align:center;color:#a5b4fc">{config_name}</td>
                          <td style="text-align:center">{days_held}d</td>
                          <td style="text-align:right">{price_sym}{price:.2f}</td>
                          <td style="text-align:right;color:{rc};font-weight:700">{price_sym}{cur_p:.2f}</td>
                          <td style="text-align:center;color:{rc};font-weight:700">{cur_ret:+.1f}%</td>
                        </tr>""")
                    break
        hconn_due.close()
        if due_items:
            due_html = f"""
            <div style="background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px;border-left:4px solid #f97316">
              <h3 style="color:#f97316;margin-top:0">⏰ 到期可卖</h3>
              <table style="width:100%;border-collapse:collapse;font-size:12px">
                <tr style="background:#334155">
                  <th>股票</th><th>Tier</th><th>策略</th><th>天数</th><th>买入</th><th>现价</th><th>涨跌</th>
                </tr>
                {''.join(due_items)}
              </table>
            </div>"""

    # ==== Section 2: Last 60 days history ====
    ytd_rows = ""
    if recent_picks:
        hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))
        for r in recent_picks:
            dt, tier, sym, name, price, blend, pred, a5, a10, a20 = r
            cur = hconn.execute(
                "SELECT close FROM stock_history WHERE symbol=? AND market=? ORDER BY trade_date DESC LIMIT 1",
                (sym, market)).fetchone()
            cur_p = cur[0] if cur else price
            cur_ret = (cur_p / price - 1) * 100 if price > 0 else 0
            cur_c = '#22c55e' if cur_ret > 0 else '#ef4444' if cur_ret < 0 else '#ffffff'
            # Calculate trading days held
            days_held = hconn.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM stock_history WHERE symbol=? AND market=? AND trade_date>?",
                (sym, market, dt)
            ).fetchone()[0]
            def _fc(v):
                if v is None: return '<td style="color:#64748b;text-align:center">—</td>'
                c = '#22c55e' if v > 0 else '#ef4444' if v < 0 else '#ffffff'
                return f'<td style="color:{c};font-weight:600;text-align:center">{v:+.1f}%</td>'
            marker = '✅' if cur_ret > 0 else '❌'
            tc = _tier_color(tier)
            ytd_rows += f"""<tr>
              <td style="color:#94a3b8;font-size:11px;white-space:nowrap">{dt}</td>
              <td style="font-size:10px;color:{tc};font-weight:700">● {tier[:12]}</td>
              <td><strong>{sym}</strong><br><small style="color:#64748b">{name or ''}</small></td>
              <td style="text-align:center;color:#94a3b8;font-size:12px">{days_held}d</td>
              <td style="text-align:right">{price_sym}{price:.2f}</td>
              <td style="text-align:right;color:{cur_c};font-weight:700">{price_sym}{cur_p:.2f}</td>
              <td style="text-align:center;color:{cur_c};font-weight:700">{cur_ret:+.1f}%</td>
              {_fc(a5)} {_fc(a10)} {_fc(a20)}
              <td style="text-align:center">{marker}</td>
            </tr>"""
        hconn.close()

    # ==== Section 3: Per-tier stats ====
    tier_stats_html = ""
    overall_stats_html = ""
    if recent_picks:
        tier_data = defaultdict(list)
        for r in recent_picks:
            tier_data[r[1]].append(r)
        all_rets = []
        for tier in sorted(tier_data.keys()):
            color = _tier_color(tier)
            tpicks = tier_data[tier]
            rets = [r[9] for r in tpicks if r[9] is not None]
            if not rets: rets = [r[8] for r in tpicks if r[8] is not None]
            if not rets: rets = [r[7] for r in tpicks if r[7] is not None]
            if rets:
                avg = np.mean(rets); wr = sum(1 for r in rets if r > 0) / len(rets) * 100
                all_rets.extend(rets)
                avg_c = '#22c55e' if avg >= 0 else '#ef4444'
                wr_c = '#22c55e' if wr >= 60 else '#ef4444'
            else:
                avg = 0; wr = 0; avg_c = '#475569'; wr_c = '#475569'
            tier_stats_html += f"""
            <div style="background:#0f172a;border-radius:8px;padding:14px;margin-bottom:8px;border-left:3px solid {color}">
              <div style="font-size:12px;color:{color};font-weight:700;margin-bottom:6px">{tier} ({len(tpicks)}笔)</div>
              <div style="display:flex;gap:10px">
                <div style="flex:1;text-align:center">
                  <div style="font-size:18px;font-weight:800;color:{avg_c}">{avg:+.1f}%</div>
                  <div style="font-size:9px;color:#64748b">AVG RET</div>
                </div>
                <div style="flex:1;text-align:center">
                  <div style="font-size:18px;font-weight:800;color:{wr_c}">{wr:.0f}%</div>
                  <div style="font-size:9px;color:#64748b">WIN RATE</div>
                </div>
                <div style="flex:1;text-align:center">
                  <div style="font-size:18px;font-weight:800;color:#818cf8">{len(rets)}/{len(tpicks)}</div>
                  <div style="font-size:9px;color:#64748b">SETTLED</div>
                </div>
              </div>
            </div>"""
        if all_rets:
            o_avg = np.mean(all_rets); o_wr = sum(1 for r in all_rets if r > 0) / len(all_rets) * 100
            o_avg_c = '#22c55e' if o_avg >= 0 else '#ef4444'
            o_wr_c = '#22c55e' if o_wr >= 60 else '#ef4444'
            overall_stats_html = f"""
            <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:10px;padding:16px;margin-top:12px;border:1px solid #6366f1">
              <div style="font-size:14px;font-weight:800;color:#fff;margin-bottom:10px">🏆 2026 YTD 总计 ({len(recent_picks)} 笔)</div>
              <div style="display:flex;gap:12px">
                <div style="flex:1;text-align:center">
                  <div style="font-size:28px;font-weight:900;color:{o_avg_c}">{o_avg:+.1f}%</div>
                  <div style="font-size:10px;color:#64748b">AVG RETURN</div>
                </div>
                <div style="flex:1;text-align:center">
                  <div style="font-size:28px;font-weight:900;color:{o_wr_c}">{o_wr:.0f}%</div>
                  <div style="font-size:10px;color:#64748b">WIN RATE</div>
                </div>
                <div style="flex:1;text-align:center">
                  <div style="font-size:28px;font-weight:900;color:#818cf8">{len(all_rets)}</div>
                  <div style="font-size:10px;color:#64748b">SETTLED</div>
                </div>
              </div>
            </div>"""

    # ==== Section 4: Strategy chart ====
    chart_section = ""
    if chart_b64:
        chart_section = f"""
<div class="section">
  <h2>📈 策略收益对比 (YTD)</h2>
  <img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:8px" alt="Strategy Chart">
</div>"""

    n_recent = len(recent_picks) if recent_picks else 0
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body {{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0;background:#0f172a;color:#e2e8f0}}
.container {{max-width:850px;margin:0 auto;padding:20px}}
.header {{background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#a855f7 100%);
  padding:30px 24px;border-radius:16px;margin-bottom:20px;text-align:center}}
.header h1 {{margin:0;font-size:24px;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.3)}}
.header p {{margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:13px}}
.section {{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:16px;border:1px solid #334155}}
.section h2 {{margin:0 0 14px;font-size:16px;color:#e2e8f0;border-bottom:2px solid #6366f1;padding-bottom:6px}}
table {{width:100%;border-collapse:collapse;font-size:11px}}
th {{background:#0f172a;padding:8px 5px;text-align:left;font-weight:600;color:#94a3b8;font-size:9px;text-transform:uppercase;letter-spacing:0.5px}}
td {{padding:6px 5px;border-bottom:1px solid #0f172a}}
tr:hover {{background:rgba(99,102,241,0.08)}}
.footer {{text-align:center;color:#64748b;font-size:11px;padding:16px;margin-top:16px}}
@media(max-width:600px){{table{{font-size:9px}}th,td{{padding:4px 2px}}.header h1{{font-size:18px}}}}
</style></head><body>
<div class="container">
<div class="header">
  <h1>{market_emoji} XGB+MMoE 每日选股</h1>
  <p>📅 {eval_date} | 🤖 XGBoost Leaf + MMoE (5d/10d/20d) | Per-Tier Top-1</p>
</div>
<div class="section">
  <h2>🎯 今日各市值 Top-1</h2>
  {pick_cards}
</div>
{due_html}
<div class="section">
  <h2>📋 2026 YTD 选股记录 ({n_recent} 笔)</h2>
  <table>
  <tr><th>日期</th><th>Tier</th><th>股票</th><th>天数</th><th>买入</th><th>现价</th><th>涨跌</th>
    <th>5D</th><th>10D</th><th>20D</th><th></th></tr>
  {ytd_rows}
  </table>
</div>
<div class="section">
  <h2>📊 各 Tier 累计表现</h2>
  {tier_stats_html}
  {overall_stats_html}
</div>
{chart_section}
<div class="footer">
  <p>📊 Coral Creek 量化平台 | XGB_leaf+MMoE Walk-Forward</p>
  <p>🔗 <a href="https://facaila.streamlit.app/" style="color:#818cf8">在线查看</a></p>
  <p>⚠️ 仅供参考，不构成投资建议</p>
</div>
</div></body></html>"""


# ===========================================================
# Main Pipeline
# ===========================================================
def run_pipeline(market):
    print(f"\n{'='*60}")
    print(f"🚀 {market} XGB+MMoE Daily Pipeline")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    t0 = time.time()

    # Step 1: Update market data (optional, skip if already updated)
    # update_market_data(market)

    # Step 2: Train + Predict
    print("\n📊 Step 2: Model Predict...", flush=True)
    eval_date, picks, top3 = predict_today(market)
    if not picks:
        print("  ❌ No picks generated")
        return

    print(f"\n🎯 {eval_date} Per-Tier Top-1:")
    price_sym = "$" if market == 'US' else "¥"
    for tier in sorted(picks.keys()):
        p = picks[tier]
        print(f"  {tier}: {p['symbol']} {price_sym}{p['price']:.2f} (blend={p['blend']:+.1f}, 20d={p['pred_20d']:+.1f}%)")

    # Step 3: Save to DB
    print("\n💾 Step 3: Save picks...", flush=True)
    save_picks(eval_date, market, picks)

    # Step 4: Track returns
    print("\n📈 Step 4: Track returns...", flush=True)
    track_returns(market)

    # Step 4b: Expiry reminders
    print("\n⏰ Step 4b: Check expiring picks...", flush=True)
    check_expiring_picks(market)

    # Step 5: Report + notify
    print("\n📧 Step 5: Send report...", flush=True)
    send_report(eval_date, market, picks)

    # Step 6: Auto-trade (Alpaca for US, CnPaperTrader for CN)
    if os.environ.get('ALPACA_TRADE', '').lower() == 'true':
        print(f"\n💰 Step 6: {'Alpaca' if market == 'US' else 'CN Paper'} auto-trade...", flush=True)
        execute_auto_trade(market, picks, top3)

    print(f"\n✅ {market} pipeline done in {time.time()-t0:.0f}s")


def _build_stats_html(strat_stats, strat_defs, colors):
    """Build strategy stats HTML table."""
    if not strat_stats:
        return '<p style="color:#64748b">暂无策略数据</p>'
    rows = ''
    for i, (sname, _, _, _, _) in enumerate(strat_defs):
        s = strat_stats.get(sname, {})
        n = s.get('n', 0)
        wr = s.get('wr', 0)
        avg = s.get('avg', 0)
        mdd = s.get('mdd', 0)
        nav = s.get('nav', 100)
        c = colors[i] if i < len(colors) else '#fff'
        wr_c = '#22c55e' if wr >= 50 else '#ef4444'
        avg_c = '#22c55e' if avg >= 0 else '#ef4444'
        nav_c = '#22c55e' if nav >= 100 else '#ef4444'
        rows += f"""<tr>
          <td style="font-weight:700;color:{c};padding:5px">{sname}</td>
          <td style="text-align:center">{n}</td>
          <td style="text-align:center;color:{wr_c};font-weight:700">{wr:.0f}%</td>
          <td style="text-align:center;color:{avg_c};font-weight:700">{avg:+.1f}%</td>
          <td style="text-align:center;color:#ef4444">{mdd:.1f}%</td>
          <td style="text-align:center;font-weight:700;color:{nav_c}">{nav:.1f}</td>
        </tr>"""
    return f"""<table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:20px">
        <tr style="background:#334155">
          <th style="padding:6px">策略</th><th>选股数</th><th>胜率</th><th>平均收益</th><th>最大回撤</th><th>NAV</th>
        </tr>{rows}</table>"""


def execute_auto_trade(market, picks, top3=None):
    """Execute trades for US (Alpaca) or CN (CnPaperTrader) with 6 strategies."""

    # Tier names per market
    if market == 'US':
        MID, LARGE, SMALL = 'Mid ($2-10B)', 'Large ($10-100B)', 'Small ($300M-2B)'
    else:
        MID, LARGE, SMALL = '中盘 (100-500亿)', '大盘 (>500亿)', '小盘 (20-100亿)'

    # Strategy definitions (same for both markets)
    STRATEGIES = [
        {'name': 'MID_10',   'prefix': 'MID',    'tiers': [MID],             'pct': 0.10, 'hold': 20, 'top_n': 1},
        {'name': 'LARGE_10', 'prefix': 'LARGE',  'tiers': [LARGE],           'pct': 0.10, 'hold': 20, 'top_n': 1},
        {'name': 'SMALL_10', 'prefix': 'SMALL',  'tiers': [SMALL],           'pct': 0.10, 'hold': 20, 'top_n': 1},
        {'name': 'ALL_3PCT', 'prefix': 'ALL3',   'tiers': [LARGE, MID, SMALL], 'pct': 0.03, 'hold': 20, 'top_n': 1},
        {'name': 'MID_TOP3', 'prefix': 'MIDTOP3','tiers': [MID],             'pct': 0.03, 'hold': 20, 'top_n': 3},
        {'name': 'MS_5DAY',  'prefix': 'MS5D',   'tiers': [MID, SMALL],      'pct': 0.10, 'hold': 5, 'top_n': 1},
    ]

    # Create traders per strategy
    def _get_trader(prefix):
        if market == 'US':
            try:
                from execution.alpaca_trader import AlpacaTrader, ALPACA_SDK_AVAILABLE
                if not ALPACA_SDK_AVAILABLE:
                    return None
            except Exception:
                return None
            default_key = os.environ.get('ALPACA_API_KEY', '')
            default_secret = os.environ.get('ALPACA_SECRET_KEY', '')
            api_key = os.environ.get(f'ALPACA_{prefix}_API_KEY', default_key if prefix == 'MID' else '')
            secret = os.environ.get(f'ALPACA_{prefix}_SECRET_KEY', default_secret if prefix == 'MID' else '')
            if not api_key or not secret:
                return None
            return AlpacaTrader(api_key=api_key, secret_key=secret, paper=True)
        else:
            from execution.cn_paper_trader import CnPaperTrader
            account_id = f"cn_{prefix.lower()}"
            return CnPaperTrader(account_id=account_id)

    trade_log = []  # Collect all trades for summary email
    strategy_traders = {}  # Keep trader instances for position query

    for strat in STRATEGIES:

        print(f"\n  {'='*50}", flush=True)
        print(f"  📊 Strategy: {strat['name']} ({strat['pct']*100:.0f}%, hold={strat['hold']}d, top{strat['top_n']})", flush=True)

        try:
            trader = _get_trader(strat['prefix'])
            if trader is None:
                continue
            strategy_traders[strat['name']] = trader
            account = trader.get_account()
            csym = '$' if market == 'US' else '¥'
            print(f"  💼 {csym}{account.equity:,.0f} (cash: {csym}{account.cash:,.0f})", flush=True)

            if not trader.is_market_open():
                print(f"  ⚠️ Market closed", flush=True); continue

            # Determine symbols to buy
            buy_targets = []
            for tier in strat['tiers']:
                if strat['top_n'] == 1:
                    if tier in picks:
                        buy_targets.append(picks[tier])
                else:
                    # Top-N from candidates
                    if top3 and tier in top3:
                        buy_targets.extend(top3[tier][:strat['top_n']])

            if not buy_targets:
                print(f"  ⚠️ No picks for this strategy", flush=True); continue

            # === SELL logic ===
            if strat['hold'] == 1:
                # Daily rotation: sell everything not in today's targets
                target_syms = {t['symbol'] for t in buy_targets}
                positions = trader.get_positions()
                for pos in positions:
                    if pos.symbol not in target_syms:
                        print(f"  📤 Sell {pos.symbol} ({pos.qty}股, {pos.unrealized_plpc:+.1f}%)", flush=True)
                        try:
                            trader.close_position(pos.symbol)
                            trade_log.append({'strategy': strat['name'], 'action': 'SELL', 'symbol': pos.symbol, 'qty': pos.qty, 'price': pos.current_price})
                        except Exception as e:
                            print(f"     ⚠️ {e}", flush=True)

            elif strat['hold'] >= 2:
                # Hold N days: sell positions older than N trading days
                db_path = V3 / 'db' / 'ml_daily_picks.db'
                hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))
                positions = trader.get_positions()
                for pos in positions:
                    # Find when we bought this
                    conn = sqlite3.connect(str(db_path))
                    bought = conn.execute(
                        "SELECT date FROM mmoe_daily_picks WHERE market=? AND symbol=? ORDER BY date DESC LIMIT 1",
                        (market, pos.symbol)
                    ).fetchone()
                    conn.close()
                    if bought:
                        days_held = hconn.execute(
                            "SELECT COUNT(DISTINCT trade_date) FROM stock_history WHERE symbol=? AND market=? AND trade_date>?",
                            (pos.symbol, market, bought[0])
                        ).fetchone()[0]
                        if days_held >= strat['hold']:
                            print(f"  📤 Sell {pos.symbol} (held {days_held}d >= {strat['hold']}d, {pos.unrealized_plpc:+.1f}%)", flush=True)
                            try:
                                trader.close_position(pos.symbol)
                                trade_log.append({'strategy': strat['name'], 'action': 'SELL', 'symbol': pos.symbol, 'qty': pos.qty, 'price': pos.current_price})
                            except Exception as e:
                                print(f"     ⚠️ {e}", flush=True)
                hconn.close()
            # hold == 0: never sell (accumulate)

            # === BUY logic ===
            equity = float(account.equity)
            for pick in buy_targets:
                sym = pick['symbol']
                # Skip if already holding
                existing = trader.get_position(sym)
                if existing and existing.qty > 0:
                    print(f"  ✅ Already holding {sym}", flush=True)
                    continue

                buy_amount = equity * strat['pct']
                price = trader.get_latest_price(sym) or pick['price']
                qty = int(buy_amount / price)
                if qty <= 0:
                    print(f"  ⚠️ Skip {sym}: insufficient funds", flush=True); continue

                print(f"  📥 Buy {sym}: {qty}股 @ ~{csym}{price:.2f} ({csym}{buy_amount:,.0f})", flush=True)
                try:
                    order = trader.buy_market(sym, qty)
                    print(f"     ✅ {order['id']}", flush=True)
                    trade_log.append({'strategy': strat['name'], 'action': 'BUY', 'symbol': sym, 'qty': qty, 'price': price})
                except Exception as e:
                    print(f"     ❌ {e}", flush=True)

        except Exception as e:
            print(f"  ❌ [{strat['name']}] Error: {e}", flush=True)
            trade_log.append({'strategy': strat['name'], 'action': 'ERROR', 'symbol': str(e), 'qty': 0, 'price': 0})

    # ============ Snapshot equity for CN accounts ============
    today = datetime.now().strftime('%Y-%m-%d')
    for sname, trader_inst in strategy_traders.items():
        if hasattr(trader_inst, 'snapshot_equity'):
            try:
                trader_inst.snapshot_equity(today)
            except Exception:
                pass

    # ============ Generate comprehensive strategy charts ============
    chart_b64 = ""
    strat_stats = {}
    STRAT_DEFS = [
        ('MID_10',   [MID],             0.10, 20, 1),
        ('LARGE_10', [LARGE],           0.10, 20, 1),
        ('SMALL_10', [SMALL],           0.10, 20, 1),
        ('ALL_3PCT', [LARGE, MID, SMALL], 0.03, 20, 1),
        ('MID_TOP3', [MID],             0.03, 20, 3),
        ('MS_5DAY',  [MID, SMALL],      0.10, 5,  1),
    ]
    colors = ['#60a5fa', '#22c55e', '#f97316', '#a78bfa', '#ec4899', '#facc15']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from io import BytesIO
        import base64

        # ---- Pull strategy picks from DB ----
        pconn = sqlite3.connect(str(V3 / 'db' / 'ml_daily_picks.db'))
        hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))

        # Get all picks with actual returns
        all_picks = pd.read_sql(
            "SELECT date, tier, symbol, price, actual_5d, actual_10d, actual_20d FROM mmoe_daily_picks WHERE market=? ORDER BY date",
            pconn, params=(market,)
        )
        pconn.close()

        # Get trading dates for NAV curve
        trade_dates = sorted(all_picks['date'].unique())

        # Simulate each strategy's NAV curve
        strat_navs = {}    # name -> [(date, nav)]
        strat_returns = {} # name -> [per-pick returns]
        strat_stats = {}   # name -> {wr, avg, maxdd, n}

        for sname, tiers, pct, hold, top_n in STRAT_DEFS:
            nav = 100.0
            nav_curve = []
            pick_rets = []

            # Filter picks for this strategy's tiers
            tier_picks = all_picks[all_picks['tier'].isin(tiers)].copy()

            # Group by date, take top_n per date
            for dt in trade_dates:
                day_picks = tier_picks[tier_picks['date'] == dt].head(top_n)
                if day_picks.empty:
                    nav_curve.append((dt, nav))
                    continue

                # Use actual return matching hold period
                ret_col = f'actual_{hold}d' if f'actual_{hold}d' in day_picks.columns else 'actual_20d'
                day_rets = day_picks[ret_col].dropna()

                if not day_rets.empty:
                    avg_ret = day_rets.mean()
                    # NAV change: pct * top_n positions * avg return
                    position_pct = min(pct * top_n, 0.30)  # cap at 30%
                    nav *= (1 + position_pct * avg_ret / 100)
                    pick_rets.extend(day_rets.tolist())

                nav_curve.append((dt, nav))

            strat_navs[sname] = nav_curve
            strat_returns[sname] = pick_rets

            # Stats
            if pick_rets:
                wins = sum(1 for r in pick_rets if r > 0)
                wr = wins / len(pick_rets) * 100
                avg = np.mean(pick_rets)
                # Max drawdown from NAV
                navs = [n[1] for n in nav_curve]
                peak = navs[0]
                mdd = 0
                for n in navs:
                    if n > peak: peak = n
                    dd = (peak - n) / peak * 100
                    if dd > mdd: mdd = dd
                strat_stats[sname] = {'wr': wr, 'avg': avg, 'mdd': mdd, 'n': len(pick_rets), 'nav': nav}
            else:
                strat_stats[sname] = {'wr': 0, 'avg': 0, 'mdd': 0, 'n': 0, 'nav': 100}

        hconn.close()

        # ---- Build 3-panel chart ----
        fig = plt.figure(figsize=(14, 12))
        fig.patch.set_facecolor('#0f172a')
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

        # Panel 1: NAV curves (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor('#1e293b')
        for i, (sname, _, _, _, _) in enumerate(STRAT_DEFS):
            if sname in strat_navs and len(strat_navs[sname]) > 1:
                dates = [n[0] for n in strat_navs[sname]]
                navs = [n[1] for n in strat_navs[sname]]
                final_ret = navs[-1] - 100
                ax1.plot(dates, navs, color=colors[i], linewidth=2,
                        label=f'{sname} ({final_ret:+.1f}%)', alpha=0.9)
        ax1.axhline(y=100, color='#64748b', linewidth=0.5, linestyle='--')
        ax1.set_title(f'{"🇺🇸 US" if market == "US" else "🇨🇳 CN"} 策略净值曲线 (模拟)', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('NAV (100=起始)', color='#94a3b8', fontsize=10)
        ax1.tick_params(colors='#94a3b8', labelsize=8)
        ax1.legend(loc='upper left', fontsize=8, facecolor='#334155', edgecolor='#475569', labelcolor='white', ncol=2)
        ax1.grid(True, alpha=0.1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # Show only every Nth date label
        if len(dates) > 20:
            ax1.set_xticks(ax1.get_xticks()[::max(1, len(ax1.get_xticks())//10)])

        # Panel 2: Return distribution (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor('#1e293b')
        all_rets_flat = []
        for i, (sname, _, _, _, _) in enumerate(STRAT_DEFS):
            rets = strat_returns.get(sname, [])
            if rets:
                ax2.hist(rets, bins=20, alpha=0.5, color=colors[i], label=sname, edgecolor='none')
                all_rets_flat.extend(rets)
        if all_rets_flat:
            ax2.axvline(x=0, color='#ef4444', linewidth=1, linestyle='--')
            ax2.axvline(x=np.mean(all_rets_flat), color='#22c55e', linewidth=1.5, linestyle='-',
                       label=f'Avg {np.mean(all_rets_flat):+.1f}%')
        ax2.set_title('收益分布', color='white', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Return %', color='#94a3b8', fontsize=9)
        ax2.set_ylabel('频次', color='#94a3b8', fontsize=9)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        ax2.legend(fontsize=6, facecolor='#334155', edgecolor='#475569', labelcolor='white', ncol=2)
        ax2.grid(True, alpha=0.1)

        # Panel 3: Win rate + stats (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor('#1e293b')
        snames = [s[0] for s in STRAT_DEFS]
        wrs = [strat_stats.get(s, {}).get('wr', 0) for s in snames]
        avgs = [strat_stats.get(s, {}).get('avg', 0) for s in snames]
        x = np.arange(len(snames))
        w = 0.35
        bars1 = ax3.bar(x - w/2, wrs, w, color='#60a5fa', alpha=0.8, label='Win Rate %')
        bars2 = ax3.bar(x + w/2, avgs, w, color=['#22c55e' if a >= 0 else '#ef4444' for a in avgs], alpha=0.8, label='Avg Ret %')
        ax3.axhline(y=50, color='#64748b', linewidth=0.5, linestyle='--')
        ax3.axhline(y=0, color='#64748b', linewidth=0.5, linestyle='-')
        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%', ha='center', va='bottom', color='#94a3b8', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h + (1 if h >= 0 else -3), f'{h:+.1f}%', ha='center', va='bottom', color='#94a3b8', fontsize=7)
        ax3.set_title('策略胜率 & 平均收益', color='white', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace('_', '\n') for s in snames], fontsize=7, color='#94a3b8')
        ax3.tick_params(colors='#94a3b8', labelsize=8)
        ax3.legend(fontsize=8, facecolor='#334155', edgecolor='#475569', labelcolor='white')
        ax3.grid(True, alpha=0.1, axis='y')

        # Save
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        print(f"  📈 Strategy charts generated ({len(trade_dates)} dates, {len(all_rets_flat)} picks)", flush=True)
    except Exception as e:
        print(f"  ⚠️ Chart generation failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    # ============ Send trade summary email ============
    if not trade_log and not strategy_traders:
        return

    csym = '$' if market == 'US' else '¥'
    mkt_label = '🇺🇸 US Alpaca' if market == 'US' else '🇨🇳 CN 虚拟盘'

    # Build symbol→name lookup
    names_map = {}
    try:
        nconn = sqlite3.connect(str(V3 / 'db' / 'ml_daily_picks.db'))
        for row in nconn.execute("SELECT DISTINCT symbol, name FROM mmoe_daily_picks WHERE name IS NOT NULL AND name!=''").fetchall():
            names_map[row[0]] = row[1]
        nconn.close()
    except Exception:
        pass

    # Build trades table rows
    trade_rows = ""
    for t in trade_log:
        ac = '#22c55e' if t['action'] == 'BUY' else '#ef4444' if t['action'] == 'SELL' else '#94a3b8'
        trade_rows += f"""<tr>
          <td style="font-weight:700">{t['strategy']}</td>
          <td style="color:{ac};font-weight:700">{t['action']}</td>
          <td>{t['symbol']}<br><small style="color:#64748b">{names_map.get(t['symbol'],'')}</small></td>
          <td style="text-align:right">{t['qty']:,.0f}</td>
          <td style="text-align:right">{csym}{t['price']:,.2f}</td>
          <td style="text-align:right">{csym}{t['qty']*t['price']:,.0f}</td>
        </tr>"""

    # Build positions table per strategy
    positions_html = ""
    total_equity = 0
    total_pl = 0
    for sname, trader_inst in strategy_traders.items():
        try:
            account = trader_inst.get_account()
            positions = trader_inst.get_positions()
            total_equity += account.equity

            strat_pl = sum(p.unrealized_pl for p in positions)
            total_pl += strat_pl
            plc = '#22c55e' if strat_pl > 0 else '#ef4444' if strat_pl < 0 else '#ffffff'

            pos_rows = ""
            # Query days held for each position
            try:
                _pconn = sqlite3.connect(str(V3 / 'db' / 'ml_daily_picks.db'))
                _hconn = sqlite3.connect(str(V3 / 'db' / 'stock_history.db'))
            except Exception:
                _pconn = _hconn = None
            for p in positions:
                pc = '#22c55e' if p.unrealized_plpc > 0 else '#ef4444'
                days_held = '—'
                if _pconn and _hconn:
                    try:
                        bought = _pconn.execute(
                            "SELECT date FROM mmoe_daily_picks WHERE market=? AND symbol=? ORDER BY date DESC LIMIT 1",
                            (market, p.symbol)).fetchone()
                        if bought:
                            dh = _hconn.execute(
                                "SELECT COUNT(DISTINCT trade_date) FROM stock_history WHERE symbol=? AND market=? AND trade_date>?",
                                (p.symbol, market, bought[0])).fetchone()[0]
                            days_held = f'{dh}d'
                    except Exception:
                        pass
                pos_rows += f"""<tr>
                  <td>{p.symbol}<br><small style="color:#64748b">{names_map.get(p.symbol,'')}</small></td>
                  <td style="text-align:center;color:#94a3b8">{days_held}</td>
                  <td style="text-align:right">{p.qty:,.0f}</td>
                  <td style="text-align:right">{csym}{p.avg_entry_price:.2f}</td>
                  <td style="text-align:right">{csym}{p.current_price:.2f}</td>
                  <td style="text-align:right;color:{pc};font-weight:700">{p.unrealized_plpc:+.1f}%</td>
                  <td style="text-align:right;color:{pc}">{csym}{p.unrealized_pl:+,.0f}</td>
                </tr>"""
            if _pconn: _pconn.close()
            if _hconn: _hconn.close()

            if not positions:
                pos_rows = '<tr><td colspan="7" style="color:#64748b;text-align:center">无持仓</td></tr>'

            positions_html += f"""
            <div style="margin-bottom:16px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-weight:700;color:#a5b4fc">{sname}</span>
                <span style="color:{plc};font-weight:700">{csym}{account.equity:,.0f} (P&L: {csym}{strat_pl:+,.0f})</span>
              </div>
              <table style="width:100%;border-collapse:collapse;font-size:12px">
                <tr style="background:#334155">
                  <th>股票</th><th>天数</th><th>数量</th><th>成本</th><th>现价</th><th>涨跌</th><th>盈亏</th>
                </tr>
                {pos_rows}
              </table>
            </div>"""
        except Exception:
            pass

    total_plc = '#22c55e' if total_pl > 0 else '#ef4444' if total_pl < 0 else '#ffffff'

    html = f"""
    <div style="font-family:system-ui;max-width:700px;margin:0 auto;background:#0f172a;color:#e2e8f0;padding:24px;border-radius:12px">
      <h1 style="color:#a5b4fc;margin-bottom:4px">{mkt_label} 交易报告</h1>
      <p style="color:#64748b;margin-top:0">{today}</p>

      <div style="display:flex;gap:16px;margin-bottom:20px">
        <div style="flex:1;background:#1e293b;border-radius:8px;padding:12px;text-align:center">
          <div style="font-size:24px;font-weight:800;color:#ffffff">{csym}{total_equity:,.0f}</div>
          <div style="font-size:11px;color:#94a3b8">总权益</div>
        </div>
        <div style="flex:1;background:#1e293b;border-radius:8px;padding:12px;text-align:center">
          <div style="font-size:24px;font-weight:800;color:{total_plc}">{csym}{total_pl:+,.0f}</div>
          <div style="font-size:11px;color:#94a3b8">总盈亏</div>
        </div>
        <div style="flex:1;background:#1e293b;border-radius:8px;padding:12px;text-align:center">
          <div style="font-size:24px;font-weight:800;color:#60a5fa">{len(trade_log)}</div>
          <div style="font-size:11px;color:#94a3b8">今日交易</div>
        </div>
      </div>

      <h2 style="color:#60a5fa;border-bottom:1px solid #334155;padding-bottom:8px">📋 今日交易</h2>
      <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:20px">
        <tr style="background:#334155">
          <th>策略</th><th>操作</th><th>股票</th><th>数量</th><th>价格</th><th>金额</th>
        </tr>
        {trade_rows}
      </table>

      <h2 style="color:#60a5fa;border-bottom:1px solid #334155;padding-bottom:8px">📈 策略收益对比 (YTD 模拟)</h2>
      {'<img src="data:image/png;base64,' + chart_b64 + '" style="width:100%;border-radius:8px;margin-bottom:20px">' if chart_b64 else '<p style="color:#64748b">数据积累中，需要更多交易日...</p>'}

      <h2 style="color:#60a5fa;border-bottom:1px solid #334155;padding-bottom:8px">📊 策略统计</h2>
      {_build_stats_html(strat_stats, STRAT_DEFS, colors)}

      <h2 style="color:#60a5fa;border-bottom:1px solid #334155;padding-bottom:8px">💼 持仓详情</h2>
      {positions_html}
    </div>
    """

    # Send email
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('SMTP_PORT', 587))
        sender = os.environ.get('SMTP_SENDER', '')
        password = os.environ.get('SMTP_PASSWORD', '')
        receivers = os.environ.get('EMAIL_RECEIVERS', sender).split(',')
        if sender and password:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"{'🇺🇸' if market=='US' else '🇨🇳'} 交易报告 {today} | {len(trade_log)} trades | PL {csym}{total_pl:+,.0f}"
            msg['From'] = sender
            msg['To'] = ', '.join(receivers)
            msg.attach(MIMEText(html, 'html'))
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.starttls()
                s.login(sender, password)
                s.sendmail(sender, receivers, msg.as_string())
            print(f"  📧 Trade report email sent", flush=True)
    except Exception as e:
        print(f"  ⚠️ Email failed: {e}", flush=True)

    # Also push summary to messaging
    try:
        from services.notification import NotificationManager
        nm = NotificationManager()
        syms = ', '.join(set(t['symbol'] for t in trade_log if t['action'] == 'BUY'))
        nm.send_all(
            f"{'🇺🇸' if market=='US' else '🇨🇳'} 交易完成",
            f"📊 {len(trade_log)} 笔交易\n💰 总权益: {csym}{total_equity:,.0f}\n📈 P&L: {csym}{total_pl:+,.0f}\n📥 买入: {syms}"
        )
    except Exception:
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='BOTH', choices=['US', 'CN', 'BOTH'])
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--trade', action='store_true', help='Enable Alpaca auto-trade for US')
    parser.add_argument('--force', action='store_true', help='Force re-prediction (ignore cache)')
    args = parser.parse_args()

    if args.trade:
        os.environ['ALPACA_TRADE'] = 'true'
    if args.force:
        os.environ['ML_FORCE_PREDICT'] = 'true'

    markets = ['US', 'CN'] if args.market == 'BOTH' else [args.market]
    for m in markets:
        run_pipeline(m)
    print("\n🎉 All done!")
