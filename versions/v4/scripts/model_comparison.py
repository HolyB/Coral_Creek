#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pointwise vs Pairwise vs Transformer Model Comparison (Walk-Forward)
====================================================================
Trains 6 model variants on a rolling 120-day window and evaluates
top-1 per-tier picks from 2026-01-02 onward, with actual
returns measured through the latest available data.

Models:
  1. XGB Pointwise (regression on y20)
  2. XGB+MMoE Pointwise (current pipeline)
  3. XGB Pairwise (rank:pairwise on y20 ranks within each day)
  4. XGB Pairwise + MMoE re-rank
  5. Transformer (FT-Transformer style, multi-task heads)
  6. XGB + Transformer (XGB leaf features → Transformer)

Each model is saved to /tmp/model_comparison/

Usage:
  cd versions/v4
  PYTHONPATH=. .venv/bin/python3 scripts/model_comparison.py --market US
"""
import os, sys, json, time, gc, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

V4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBRanker

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT_DIR = Path('/tmp/model_comparison')
OUT_DIR.mkdir(exist_ok=True)

WINDOW = 120
BLEND_W = (0.2, 0.3, 0.5)

# ===== Tier config =====
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

# ===== MMoE =====
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

def train_mmoe(model, X, ys, epochs=8):
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


# ===== FT-Transformer for Tabular Data =====
class TabTransformer(nn.Module):
    """Feature-Tokenizer Transformer for tabular data.
    Groups raw features into patches (tokens), applies self-attention,
    then multi-task prediction heads for 5d/10d/20d returns.
    """
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.15, n_tasks=3, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (n_features + patch_size - 1) // patch_size
        self.padded_dim = self.n_patches * patch_size

        # Feature tokenizer: project each patch to d_model
        self.input_bn = nn.BatchNorm1d(n_features)
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Multi-task prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x):
        B = x.shape[0]
        x = self.input_bn(x)

        # Pad to multiple of patch_size
        if x.shape[1] < self.padded_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - x.shape[1]))

        # Reshape into patches: (B, n_patches, patch_size)
        x = x.view(B, self.n_patches, self.patch_size)

        # Project patches to d_model
        x = self.patch_proj(x)  # (B, n_patches, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, d_model)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)

        # Use [CLS] token output for prediction
        cls_out = x[:, 0]  # (B, d_model)

        # Multi-task heads
        return [head(cls_out).squeeze(-1) for head in self.heads]


def train_transformer(model, X, ys, epochs=15, lr=5e-4, batch_size=2048):
    """Train transformer with cosine annealing and warmup."""
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    ds = TensorDataset(torch.FloatTensor(X), *[torch.FloatTensor(y) for y in ys])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dl:
            x = batch[0].to(DEVICE)
            yy = [b.to(DEVICE) for b in batch[1:]]
            preds = model(x)
            loss = sum(nn.HuberLoss(delta=10.0)(p, y) for p, y in zip(preds, yy)) / len(yy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()
    model.eval()
    return model

# ===== Load data =====
def load_npz(market):
    npz_path = f'/tmp/{market.lower()}_daily_full.npz'
    print(f"  Loading {npz_path}...", flush=True)
    d = np.load(npz_path, allow_pickle=True)
    return d['X'], d['dates'], d['symbols'], d['y5'], d['y10'], d['y20'], list(d['fn'])

def load_mcap(market):
    import sqlite3
    conn = sqlite3.connect(str(V4 / 'db' / 'coral_creek.db'))
    mcap = {}
    if market == 'US':
        try:
            for r in conn.execute('SELECT symbol, market_cap FROM stock_meta'):
                mcap[r[0]] = r[1] or 0
        except:
            pass
    else:
        try:
            for r in conn.execute('SELECT symbol, total_mv FROM stock_meta_cn'):
                sym = r[0]
                mv = r[1] or 0
                if sym.startswith('6'):
                    mcap[sym + '.SH'] = mv
                else:
                    mcap[sym + '.SZ'] = mv
        except:
            pass
    conn.close()
    return mcap

# ===== Core: walk-forward for one model type =====
def walk_forward(market, model_type, X_all, dates_all, symbols_all, y5, y10, y20, fn, mcap_dict,
                 eval_dates, all_unique_dates):
    """
    Walk-forward evaluation for one model type.
    Returns: list of {date, tier, symbol, blend, actual_returns...}
    """
    print(f"\n{'='*50}", flush=True)
    print(f"  Model: {model_type} | {market} | {len(eval_dates)} eval dates", flush=True)
    print(f"{'='*50}", flush=True)

    results = []
    min_mcap = 50e6 if market == 'US' else 20e8
    model_save_dir = OUT_DIR / f"{model_type}_{market}"
    model_save_dir.mkdir(exist_ok=True)

    for eval_i, eval_date in enumerate(eval_dates):
        t0 = time.time()
        
        # Training window
        ed_idx = all_unique_dates.index(eval_date)
        train_dates = all_unique_dates[max(0, ed_idx - WINDOW):ed_idx]
        if len(train_dates) < 30:
            continue

        tm = np.isin(dates_all, train_dates)
        X_tr = X_all[tm]
        ys_tr = [y5[tm], y10[tm], y20[tm]]
        v = ~np.isnan(ys_tr[2])
        for y in ys_tr: v &= ~np.isnan(y)
        X_tr = np.nan_to_num(X_tr[v], nan=0.0)
        ys_tr = [y[v] for y in ys_tr]

        if len(X_tr) < 500:
            continue

        seed = int(eval_date.replace('-', '')) % 2**31
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ====== Train ======
        sc = StandardScaler()
        Xs = sc.fit_transform(X_tr).astype(np.float32)
        np.nan_to_num(Xs, copy=False, nan=0, posinf=0, neginf=0)

        if model_type in ('xgb_pointwise', 'xgb_mmoe', 'xgb_transformer'):
            # XGB Pointwise (regression on y20)
            xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                tree_method='hist', n_jobs=-1, verbosity=0, random_state=seed)
            xgb.fit(X_tr, ys_tr[-1])  # predict y20
            leaf_tr = xgb.apply(X_tr).astype(np.float32)

            if model_type == 'xgb_mmoe':
                Xa = np.hstack([Xs, leaf_tr])
                mmoe = MMoE(Xa.shape[1], 4, 128, 3)
                mmoe = train_mmoe(mmoe, Xa, ys_tr)

            elif model_type == 'xgb_transformer':
                # XGB leaves → Transformer (full data, same as MMoE)
                Xa = np.hstack([Xs, leaf_tr])
                tfm = TabTransformer(Xa.shape[1], d_model=64, n_heads=4,
                                     n_layers=2, d_ff=128, dropout=0.15)
                tfm = train_transformer(tfm, Xa, ys_tr, epochs=8, batch_size=8192)

        elif model_type == 'transformer':
            # Pure Transformer (no XGB) — full data, same as MMoE
            tfm = TabTransformer(Xs.shape[1], d_model=64, n_heads=4,
                                 n_layers=2, d_ff=128, dropout=0.15)
            tfm = train_transformer(tfm, Xs, ys_tr, epochs=8, batch_size=8192)

        elif model_type in ('xgb_pairwise', 'xgb_pairwise_mmoe'):
            # XGB Pairwise: rank:pairwise with y20 as relevance
            # Need to build group (qid) by date
            train_dates_arr = dates_all[tm][v]
            unique_train_dates = sorted(set(train_dates_arr))
            
            # Convert y20 to ranks within each day (higher return = higher rank)
            y_rank = np.zeros_like(ys_tr[-1])
            qid = np.zeros(len(ys_tr[-1]), dtype=np.int32)
            
            for qi, td in enumerate(unique_train_dates):
                mask = train_dates_arr == td
                day_y = ys_tr[-1][mask]
                # Rank: higher return → higher score (0-based)
                ranks = np.argsort(np.argsort(day_y)).astype(np.float32)
                y_rank[mask] = ranks
                qid[mask] = qi

            # Sort by qid (XGBRanker requirement)
            sort_idx = np.argsort(qid)
            X_tr_sorted = X_tr[sort_idx]
            y_rank_sorted = y_rank[sort_idx]
            qid_sorted = qid[sort_idx]

            xgb_ranker = XGBRanker(
                tree_method='hist',
                objective='rank:pairwise',
                learning_rate=0.05,
                n_estimators=300,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                n_jobs=-1,
                verbosity=0,
                random_state=seed,
            )
            xgb_ranker.fit(X_tr_sorted, y_rank_sorted, qid=qid_sorted)

            if model_type == 'xgb_pairwise_mmoe':
                # Also train MMoE (using pointwise XGB leaves for features)
                xgb_pt = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                    tree_method='hist', n_jobs=-1, verbosity=0, random_state=seed)
                xgb_pt.fit(X_tr, ys_tr[-1])
                leaf_tr = xgb_pt.apply(X_tr).astype(np.float32)
                Xa = np.hstack([Xs, leaf_tr])
                mmoe = MMoE(Xa.shape[1], 4, 128, 3)
                mmoe = train_mmoe(mmoe, Xa, ys_tr)

        # ====== Predict on eval_date ======
        em = dates_all == eval_date
        X_ev = np.nan_to_num(X_all[em], nan=0.0)
        syms_ev = symbols_all[em]
        
        if len(X_ev) == 0:
            continue

        Xe = sc.transform(X_ev).astype(np.float32)
        np.nan_to_num(Xe, copy=False, nan=0, posinf=0, neginf=0)

        if model_type == 'xgb_pointwise':
            pred_20d = np.clip(xgb.predict(X_ev), -200, 200)
            blend = pred_20d  # Single task, just use y20

        elif model_type == 'xgb_mmoe':
            leaf_ev = xgb.apply(X_ev).astype(np.float32)
            Xae = np.hstack([Xe, leaf_ev])
            with torch.no_grad():
                preds = mmoe(torch.FloatTensor(Xae).to(DEVICE))
                p5 = np.clip(preds[0].cpu().numpy(), -200, 200)
                p10 = np.clip(preds[1].cpu().numpy(), -200, 200)
                p20 = np.clip(preds[2].cpu().numpy(), -200, 200)
            blend = BLEND_W[0] * p5 + BLEND_W[1] * p10 + BLEND_W[2] * p20

        elif model_type == 'transformer':
            with torch.no_grad():
                preds = tfm(torch.FloatTensor(Xe).to(DEVICE))
                p5 = np.clip(preds[0].cpu().numpy(), -200, 200)
                p10 = np.clip(preds[1].cpu().numpy(), -200, 200)
                p20 = np.clip(preds[2].cpu().numpy(), -200, 200)
            blend = BLEND_W[0] * p5 + BLEND_W[1] * p10 + BLEND_W[2] * p20

        elif model_type == 'xgb_transformer':
            leaf_ev = xgb.apply(X_ev).astype(np.float32)
            Xae = np.hstack([Xe, leaf_ev])
            with torch.no_grad():
                preds = tfm(torch.FloatTensor(Xae).to(DEVICE))
                p5 = np.clip(preds[0].cpu().numpy(), -200, 200)
                p10 = np.clip(preds[1].cpu().numpy(), -200, 200)
                p20 = np.clip(preds[2].cpu().numpy(), -200, 200)
            blend = BLEND_W[0] * p5 + BLEND_W[1] * p10 + BLEND_W[2] * p20

        elif model_type == 'xgb_pairwise':
            blend = xgb_ranker.predict(X_ev)  # ranking score (not % return)

        elif model_type == 'xgb_pairwise_mmoe':
            rank_score = xgb_ranker.predict(X_ev)
            leaf_ev = xgb_pt.apply(X_ev).astype(np.float32)
            Xae = np.hstack([Xe, leaf_ev])
            with torch.no_grad():
                preds = mmoe(torch.FloatTensor(Xae).to(DEVICE))
                p5 = np.clip(preds[0].cpu().numpy(), -200, 200)
                p10 = np.clip(preds[1].cpu().numpy(), -200, 200)
                p20 = np.clip(preds[2].cpu().numpy(), -200, 200)
            mmoe_blend = BLEND_W[0] * p5 + BLEND_W[1] * p10 + BLEND_W[2] * p20
            # Combine: 50% rank score (normalized) + 50% MMoE blend
            rank_norm = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-8)
            mmoe_norm = (mmoe_blend - mmoe_blend.mean()) / (mmoe_blend.std() + 1e-8)
            blend = 0.5 * rank_norm + 0.5 * mmoe_norm

        # ====== Per-tier top-1 picks ======
        tier_candidates = defaultdict(list)
        for i in range(len(syms_ev)):
            sym = syms_ev[i]
            mc = mcap_dict.get(sym, 0)
            if mc < min_mcap and mcap_dict:
                continue
            tier = get_tier(mc, market)
            tier_candidates[tier].append({
                'symbol': sym,
                'blend': float(blend[i]),
                'mcap': mc,
                'tier': tier,
            })

        # Get actual returns from y arrays
        for tier, cands in tier_candidates.items():
            cands.sort(key=lambda x: -x['blend'])
            pick = cands[0]
            sym = pick['symbol']
            
            # Get actual y20 for this symbol on this date
            idx = np.where((dates_all == eval_date) & (symbols_all == sym))[0]
            actual_5d = float(y5[idx[0]]) if len(idx) > 0 and not np.isnan(y5[idx[0]]) else None
            actual_10d = float(y10[idx[0]]) if len(idx) > 0 and not np.isnan(y10[idx[0]]) else None
            actual_20d = float(y20[idx[0]]) if len(idx) > 0 and not np.isnan(y20[idx[0]]) else None

            # Global rank
            all_blends = sorted([c['blend'] for t_cands in tier_candidates.values() for c in t_cands], reverse=True)
            global_rank = all_blends.index(pick['blend']) + 1 if pick['blend'] in all_blends else -1

            results.append({
                'date': eval_date,
                'model': model_type,
                'tier': tier,
                'symbol': sym,
                'blend': pick['blend'],
                'global_rank': global_rank,
                'total_candidates': len(all_blends),
                'actual_5d': actual_5d,
                'actual_10d': actual_10d,
                'actual_20d': actual_20d,
            })

        elapsed = time.time() - t0
        n_picks = len([r for r in results if r['date'] == eval_date])
        if eval_i % 5 == 0:
            print(f"  [{eval_i+1}/{len(eval_dates)}] {eval_date}: {n_picks} picks, {len(X_tr)} train, {elapsed:.1f}s", flush=True)

        del X_tr, Xs
        gc.collect()
        if DEVICE == 'mps':
            torch.mps.empty_cache()

    return results

# ===== Analysis =====
def analyze_results(all_results):
    """Compute summary statistics per model."""
    df = pd.DataFrame(all_results)
    if df.empty:
        return {}

    summary = {}
    for model_name in df['model'].unique():
        mdf = df[df['model'] == model_name]

        # Overall stats
        a20 = mdf['actual_20d'].dropna()
        a10 = mdf['actual_10d'].dropna()
        a5 = mdf['actual_5d'].dropna()

        stats = {
            'model': model_name,
            'total_picks': len(mdf),
            'unique_dates': mdf['date'].nunique(),
        }

        for label, series in [('5d', a5), ('10d', a10), ('20d', a20)]:
            if len(series) > 0:
                stats[f'{label}_avg'] = series.mean()
                stats[f'{label}_median'] = series.median()
                stats[f'{label}_wr'] = (series > 0).mean() * 100
                stats[f'{label}_best'] = series.max()
                stats[f'{label}_worst'] = series.min()
            else:
                stats[f'{label}_avg'] = 0
                stats[f'{label}_wr'] = 0

        # Per-tier stats
        tier_stats = {}
        for tier in mdf['tier'].unique():
            tdf = mdf[mdf['tier'] == tier]
            ta20 = tdf['actual_20d'].dropna()
            tier_stats[tier] = {
                'n': len(tdf),
                'avg_20d': ta20.mean() if len(ta20) > 0 else 0,
                'wr_20d': (ta20 > 0).mean() * 100 if len(ta20) > 0 else 0,
            }
        stats['tier_stats'] = tier_stats

        # Monthly stats
        mdf_copy = mdf.copy()
        mdf_copy['month'] = mdf_copy['date'].str[:7]
        monthly = {}
        for month, gdf in mdf_copy.groupby('month'):
            ma20 = gdf['actual_20d'].dropna()
            monthly[month] = {
                'n': len(gdf),
                'avg_20d': ma20.mean() if len(ma20) > 0 else 0,
                'wr_20d': (ma20 > 0).mean() * 100 if len(ma20) > 0 else 0,
            }
        stats['monthly'] = monthly

        summary[model_name] = stats
    return summary

# ===== HTML Report =====
def build_html_report(summary, all_results, market):
    df = pd.DataFrame(all_results)
    market_name = "美股" if market == 'US' else "A股"
    today = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Model comparison table
    model_rows = ""
    model_colors = {
        'xgb_pointwise': '#3b82f6',
        'xgb_mmoe': '#8b5cf6',
        'xgb_pairwise': '#f59e0b',
        'xgb_pairwise_mmoe': '#22c55e',
        'transformer': '#ef4444',
        'xgb_transformer': '#06b6d4',
    }
    for model_name, s in sorted(summary.items()):
        c = model_colors.get(model_name, '#fff')
        avg20 = s.get('20d_avg', 0)
        wr20 = s.get('20d_wr', 0)
        avg10 = s.get('10d_avg', 0)
        wr10 = s.get('10d_wr', 0)
        avg5 = s.get('5d_avg', 0)
        wr5 = s.get('5d_wr', 0)
        ac = '#22c55e' if avg20 > 0 else '#ef4444'
        wc = '#22c55e' if wr20 >= 50 else '#ef4444'
        model_rows += f"""<tr>
          <td style="font-weight:700;color:{c}">{model_name}</td>
          <td style="text-align:center">{s['total_picks']}</td>
          <td style="text-align:center;color:{'#22c55e' if avg5>0 else '#ef4444'};font-weight:700">{avg5:+.2f}%</td>
          <td style="text-align:center;color:{'#22c55e' if wr5>=50 else '#ef4444'}">{wr5:.0f}%</td>
          <td style="text-align:center;color:{'#22c55e' if avg10>0 else '#ef4444'};font-weight:700">{avg10:+.2f}%</td>
          <td style="text-align:center;color:{'#22c55e' if wr10>=50 else '#ef4444'}">{wr10:.0f}%</td>
          <td style="text-align:center;color:{ac};font-weight:700">{avg20:+.2f}%</td>
          <td style="text-align:center;color:{wc}">{wr20:.0f}%</td>
        </tr>"""

    # Per-tier comparison
    tier_section = ""
    all_tiers = sorted(set(r['tier'] for r in all_results))
    for tier in all_tiers:
        tier_rows = ""
        for model_name in sorted(summary.keys()):
            ts = summary[model_name].get('tier_stats', {}).get(tier, {})
            c = model_colors.get(model_name, '#fff')
            avg = ts.get('avg_20d', 0)
            wr = ts.get('wr_20d', 0)
            n = ts.get('n', 0)
            tier_rows += f"""<tr>
              <td style="color:{c};font-weight:600">{model_name}</td>
              <td style="text-align:center">{n}</td>
              <td style="text-align:center;color:{'#22c55e' if avg>0 else '#ef4444'};font-weight:700">{avg:+.2f}%</td>
              <td style="text-align:center;color:{'#22c55e' if wr>=50 else '#ef4444'}">{wr:.0f}%</td>
            </tr>"""
        tier_section += f"""<div style="margin-bottom:12px">
          <div style="font-weight:700;color:#a5b4fc;margin-bottom:4px">{tier}</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <tr style="background:#334155"><th>Model</th><th>Picks</th><th>Avg 20D</th><th>WR 20D</th></tr>
            {tier_rows}
          </table>
        </div>"""

    # Monthly comparison
    all_months = sorted(set(r['date'][:7] for r in all_results))
    monthly_rows = ""
    for month in all_months:
        cells = f'<td style="font-weight:600">{month}</td>'
        for model_name in sorted(summary.keys()):
            ms = summary[model_name].get('monthly', {}).get(month, {})
            avg = ms.get('avg_20d', 0)
            wr = ms.get('wr_20d', 0)
            n = ms.get('n', 0)
            ac = '#22c55e' if avg > 0 else '#ef4444'
            cells += f'<td style="text-align:center;color:{ac};font-weight:600">{avg:+.1f}% ({wr:.0f}%)</td>'
        monthly_rows += f"<tr>{cells}</tr>"
    
    monthly_header = '<th>Month</th>' + ''.join(
        f'<th style="text-align:center;color:{model_colors.get(m,"#fff")}">{m}</th>' 
        for m in sorted(summary.keys())
    )

    # Per-date picks table (last 30 days)
    daily_rows = ""
    recent_dates = sorted(set(r['date'] for r in all_results))[-30:]
    for dt in reversed(recent_dates):
        day_picks = [r for r in all_results if r['date'] == dt]
        for p in day_picks:
            c = model_colors.get(p['model'], '#fff')
            a20 = p.get('actual_20d')
            a20_str = f"{a20:+.1f}%" if a20 is not None else "—"
            a20_c = '#22c55e' if a20 and a20 > 0 else '#ef4444' if a20 and a20 < 0 else '#64748b'
            daily_rows += f"""<tr>
              <td style="color:#94a3b8;font-size:11px">{p['date']}</td>
              <td style="color:{c};font-weight:600;font-size:11px">{p['model'][:12]}</td>
              <td><strong>{p['symbol']}</strong></td>
              <td style="font-size:11px">{p['tier'][:15]}</td>
              <td style="text-align:center">{p.get('global_rank','?')}/{p.get('total_candidates','?')}</td>
              <td style="text-align:center;color:{a20_c};font-weight:700">{a20_str}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body {{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0;background:#0f172a;color:#e2e8f0}}
.container {{max-width:950px;margin:0 auto;padding:20px}}
.header {{background:linear-gradient(135deg,#6366f1 0%,#ec4899 100%);padding:40px 30px;border-radius:16px;margin-bottom:24px;text-align:center}}
.header h1 {{margin:0;font-size:26px;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.3)}}
.header p {{margin:8px 0 0;color:rgba(255,255,255,0.85);font-size:14px}}
.section {{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:24px;border:1px solid #334155}}
.section h2 {{margin:0 0 16px;font-size:18px;color:#e2e8f0;border-bottom:2px solid #6366f1;padding-bottom:8px}}
table {{width:100%;border-collapse:collapse;font-size:13px}}
th {{background:#0f172a;padding:10px 8px;text-align:left;font-weight:600;color:#94a3b8;font-size:11px;text-transform:uppercase}}
td {{padding:8px;border-bottom:1px solid #1e293b}}
tr:hover {{background:rgba(99,102,241,0.08)}}
.footer {{text-align:center;color:#64748b;font-size:12px;padding:20px;margin-top:24px}}
.legend {{display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap}}
.legend-item {{display:flex;align-items:center;gap:6px;font-size:12px}}
.dot {{width:12px;height:12px;border-radius:50%;display:inline-block}}
</style></head>
<body><div class="container">

<div class="header">
  <h1>🧪 Pointwise vs Pairwise 模型对比</h1>
  <p>📅 {today} | {market_name} YTD Walk-Forward 回测</p>
</div>

<div class="section">
  <h2>🏆 模型总览</h2>
  <div class="legend">
    <div class="legend-item"><span class="dot" style="background:#3b82f6"></span>XGB Pointwise</div>
    <div class="legend-item"><span class="dot" style="background:#8b5cf6"></span>XGB+MMoE</div>
    <div class="legend-item"><span class="dot" style="background:#f59e0b"></span>XGB Pairwise</div>
    <div class="legend-item"><span class="dot" style="background:#22c55e"></span>Pairwise+MMoE</div>
    <div class="legend-item"><span class="dot" style="background:#ef4444"></span>Transformer</div>
    <div class="legend-item"><span class="dot" style="background:#06b6d4"></span>XGB+Transformer</div>
  </div>
  <table>
    <tr style="background:#334155"><th>Model</th><th>Picks</th>
      <th style="text-align:center">Avg 5D</th><th style="text-align:center">WR 5D</th>
      <th style="text-align:center">Avg 10D</th><th style="text-align:center">WR 10D</th>
      <th style="text-align:center">Avg 20D</th><th style="text-align:center">WR 20D</th>
    </tr>
    {model_rows}
  </table>
</div>

<div class="section">
  <h2>📊 Per-Tier 对比</h2>
  {tier_section}
</div>

<div class="section">
  <h2>📅 月度趋势</h2>
  <table>
    <tr style="background:#334155">{monthly_header}</tr>
    {monthly_rows}
  </table>
</div>

<div class="section">
  <h2>📋 每日选股明细 (近30天)</h2>
  <table>
    <tr style="background:#334155"><th>Date</th><th>Model</th><th>Symbol</th><th>Tier</th><th>Rank</th><th>Actual 20D</th></tr>
    {daily_rows}
  </table>
</div>

<div class="footer">
  <p>⚠️ 仅供参考，不构成投资建议</p>
  <p style="font-size:9px;color:#475569">Models: XGB Pointwise / XGB+MMoE / XGBRanker Pairwise / Pairwise+MMoE</p>
</div>

</div></body></html>"""
    return html

# ===== Email =====
def send_email(html, market):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_SENDER') or os.getenv('SMTP_USER')
    smtp_pass = os.getenv('SMTP_PASSWORD')
    receivers = os.getenv('EMAIL_RECEIVERS', os.getenv('TO_EMAIL', ''))
    to_list = [e.strip() for e in receivers.split(',') if e.strip()]

    if not all([smtp_user, smtp_pass, to_list]):
        print("  ⚠️ Email not configured")
        return False

    market_name = "美股" if market == 'US' else "A股"
    subject = f"🧪 Pointwise vs Pairwise 模型对比 | {market_name} — {datetime.now().strftime('%Y-%m-%d')}"

    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = ', '.join(to_list)
    msg.attach(MIMEText(html, 'html', 'utf-8'))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"  ✅ Email sent to {', '.join(to_list)}")
        return True
    except Exception as e:
        print(f"  ❌ Email failed: {e}")
        return False

# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--start', default='2026-01-02')
    parser.add_argument('--end', default='2026-03-19')
    args = parser.parse_args()

    market = args.market
    t_total = time.time()

    # Load data
    print(f"\n🚀 Model Comparison: {market}", flush=True)
    X_all, dates_all, symbols_all, y5, y10, y20, fn = load_npz(market)
    mcap_dict = load_mcap(market)
    print(f"  Data: {X_all.shape[0]:,} x {X_all.shape[1]}, mcap: {len(mcap_dict):,}", flush=True)

    # Eval dates from 2026-01-02 to latest
    all_unique_dates = sorted(set(dates_all))
    eval_dates = [d for d in all_unique_dates if args.start <= d <= args.end]
    print(f"  Eval dates: {len(eval_dates)} ({eval_dates[0]}..{eval_dates[-1]})", flush=True)

    # Run all 4 models
    MODEL_TYPES = ['xgb_pointwise', 'xgb_mmoe', 'xgb_pairwise', 'xgb_pairwise_mmoe', 'transformer', 'xgb_transformer']
    all_results = []

    for model_type in MODEL_TYPES:
        results = walk_forward(
            market, model_type, X_all, dates_all, symbols_all,
            y5, y10, y20, fn, mcap_dict, eval_dates, all_unique_dates
        )
        all_results.extend(results)
        print(f"  ✅ {model_type}: {len(results)} picks", flush=True)

    # Analyze
    print(f"\n📊 Analyzing...", flush=True)
    summary = analyze_results(all_results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'Picks':>6} {'Avg5d':>8} {'WR5d':>6} {'Avg10d':>8} {'WR10d':>6} {'Avg20d':>8} {'WR20d':>6}")
    print(f"{'='*70}")
    for model_name, s in sorted(summary.items()):
        print(f"{model_name:<25} {s['total_picks']:>6} "
              f"{s.get('5d_avg',0):>+7.2f}% {s.get('5d_wr',0):>5.0f}% "
              f"{s.get('10d_avg',0):>+7.2f}% {s.get('10d_wr',0):>5.0f}% "
              f"{s.get('20d_avg',0):>+7.2f}% {s.get('20d_wr',0):>5.0f}%")
    print(f"{'='*70}")

    # Save results
    results_path = OUT_DIR / f'{market}_results.json'
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'picks': all_results}, f, default=str, indent=2)
    print(f"  💾 Results saved to {results_path}", flush=True)

    # Build HTML and send email
    print(f"\n📧 Building report & sending email...", flush=True)
    html = build_html_report(summary, all_results, market)
    html_path = OUT_DIR / f'{market}_model_comparison.html'
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  💾 HTML saved to {html_path}", flush=True)
    send_email(html, market)

    elapsed = time.time() - t_total
    print(f"\n✅ Done in {elapsed/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()
