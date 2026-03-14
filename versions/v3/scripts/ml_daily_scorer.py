#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML 每日选股评分器 (v2 — 全市场)
================================
- US: 按市值分层 (Micro/Small/Mid/Large/Mega)
- CN: 按板块(上证/深证/创业板/科创板) + 市值(10-50亿 ... >1000亿) 分层
- 多周期预测 (10d/30d/60d)
- 保存到 ml_daily_picks.db

用法:
    PYTHONPATH=. python scripts/ml_daily_scorer.py --market US
    PYTHONPATH=. python scripts/ml_daily_scorer.py --market CN
    PYTHONPATH=. python scripts/ml_daily_scorer.py --market US --date 2026-03-08
"""
import os, sys, json, sqlite3, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ===================== Market Cap Tiers =====================
US_TIERS = [
    ('Micro 50-300M',  5e7,  3e8),
    ('Small 300M-2B',  3e8,  2e9),
    ('Mid 2-10B',      2e9,  1e10),
    ('Large 10-100B',  1e10, 1e11),
    ('Mega >100B',     1e11, 1e15),
]

CN_TIERS = [
    ('10-50亿',    1e9,  5e9),
    ('50-100亿',   5e9,  1e10),
    ('100-300亿',  1e10, 3e10),
    ('300-1000亿', 3e10, 1e11),
    ('>1000亿',    1e11, 1e15),
]

# ===================== Exchange Classification =====================
def get_exchange(symbol, market='US'):
    """Classify stock by exchange/board"""
    if market == 'US':
        return 'US'
    # CN exchange classification
    code = symbol.split('.')[0] if '.' in symbol else symbol
    if code.startswith('688'):
        return '科创板'
    if code.startswith('300') or code.startswith('301'):
        return '创业板'
    if symbol.endswith('.SH'):
        return '上证主板'
    if symbol.endswith('.SZ'):
        return '深证主板'
    if symbol.endswith('.BJ'):
        return '北交所'
    return '其他'


# ===================== Model Loading =====================
def load_model(market='US'):
    """加载训练好的 ReturnPredictor"""
    from ml.models.return_predictor import ReturnPredictor
    model_dir = os.path.join(parent_dir, 'ml', 'saved_models', f'v2_{market.lower()}')
    predictor = ReturnPredictor()
    if predictor.load(model_dir):
        return predictor
    print(f"⚠️ 无法加载 {market} 模型: {model_dir}")
    return None


def load_market_data(market='US'):
    """加载市值和行业数据"""
    db_dir = os.path.join(parent_dir, 'db')
    mcap, sector = {}, {}
    
    if market == 'CN':
        mcap_path = os.path.join(db_dir, 'cn_mcap_dict.json')
        sector_path = os.path.join(db_dir, 'cn_sector_dict.json')
    else:
        mcap_path = os.path.join(db_dir, 'mcap_dict.json')
        sector_path = os.path.join(db_dir, 'sector_dict.json')
    
    if os.path.exists(mcap_path):
        with open(mcap_path) as f:
            mcap = json.load(f)
    
    if os.path.exists(sector_path):
        with open(sector_path) as f:
            sector = json.load(f)
    
    return mcap, sector


# ===================== Core Scoring =====================
def score_daily_signals(market='US', date=None, top_n=5):
    """
    对指定日期的 scan_results 用 ML 模型打分，按市值/板块分层输出 Top-N。
    
    Returns:
        dict: {
            'picks': {tier_or_exchange: [{'symbol':..., 'score':..., ...}]},
            'date': str,
            'market': str,
            'stats': {'total_scored': int, ...}
        }
    """
    from db.database import init_db, get_scanned_dates, query_scan_results
    
    init_db()
    
    # 1. Get target date
    if date is None:
        dates = get_scanned_dates(market=market)
        if not dates:
            print("❌ 无扫描数据")
            return {}
        date = dates[0]
    
    print(f"\n{'='*60}")
    print(f"📊 ML Daily Scorer: {date} ({market})")
    print(f"{'='*60}")
    
    # 2. Load model
    predictor = load_model(market)
    if predictor is None:
        return {}
    
    feature_names = predictor.feature_names
    print(f"   模型特征: {len(feature_names)}")
    
    # 3. Load scan results
    signals = query_scan_results(scan_date=date, market=market, limit=5000)
    if not signals:
        print(f"   ❌ {date} 无信号")
        return {}
    
    signals_df = pd.DataFrame(signals)
    print(f"   信号数: {len(signals_df)}")
    
    # 4. Load market data
    mcap_dict, sector_dict = load_market_data(market)
    
    # 5. Build features and score
    cache_db = sqlite3.connect(os.path.join(parent_dir, 'db', 'ml_feature_cache.db'))
    
    min_price = 3 if market == 'CN' else 5
    scored = []
    
    for _, row in signals_df.iterrows():
        symbol = row.get('symbol', '')
        price = float(row.get('price', 0) or 0)
        if price < min_price:
            continue
        
        # Get features from cache
        cache_row = cache_db.execute(
            'SELECT features_json FROM feature_cache WHERE symbol=? AND trade_date=? LIMIT 1',
            (symbol, date)
        ).fetchone()
        
        if cache_row is None:
            continue
        
        feat = json.loads(cache_row[0])
        
        # Build feature vector
        X = np.array([[float(feat.get(fn, 0) or 0) for fn in feature_names]], dtype=np.float32)
        X = np.nan_to_num(X, 0)
        
        # Score with model
        preds = predictor.predict(X)
        pred_5d = float(preds.get('5d', {0: 0})[0]) if '5d' in preds else 0
        pred_10d = float(preds.get('10d', preds.get('5d', {0: 0}))[0])
        pred_20d = float(preds.get('20d', {0: 0})[0]) if '20d' in preds else 0
        pred_30d = float(preds.get('30d', {0: 0})[0]) if '30d' in preds else pred_20d * 1.3
        pred_60d = float(preds.get('60d', {0: 0})[0]) if '60d' in preds else 0
        
        mc = mcap_dict.get(symbol, 0)
        sec = sector_dict.get(symbol, {})
        exchange = get_exchange(symbol, market)
        
        # Primary score: US uses 10d, CN uses 30d
        primary_pred = pred_10d if market == 'US' else pred_30d
        
        scored.append({
            'symbol': symbol,
            'price': price,
            'exchange': exchange,
            'pred_5d': pred_5d,
            'pred_10d': pred_10d,
            'pred_20d': pred_20d,
            'pred_30d': pred_30d,
            'pred_60d': pred_60d,
            'primary_pred': primary_pred,
            'market_cap': mc,
            'mcap_b': mc / 1e9 if mc > 0 else 0,
            'sector': sec.get('sic', sec.get('industry', ''))[:30] if isinstance(sec, dict) else str(sec)[:30],
            'blue_daily': float(row.get('blue_daily', 0) or 0),
            'holding_period': '10d' if market == 'US' else '30d',
        })
    
    cache_db.close()
    
    if not scored:
        print("   ❌ 无法评分（缓存中无匹配）")
        return {}
    
    scored_df = pd.DataFrame(scored).sort_values('primary_pred', ascending=False)
    print(f"   评分完成: {len(scored_df)} 只股票")
    
    # 6. Group by tier and exchange
    tiers = US_TIERS if market == 'US' else CN_TIERS
    results = {}
    
    if market == 'CN':
        # CN: Group by exchange first, then by market cap within each
        for exchange in ['上证主板', '深证主板', '创业板', '科创板']:
            ex_df = scored_df[scored_df['exchange'] == exchange]
            if len(ex_df) == 0:
                continue
            
            for tier_name, lo, hi in tiers:
                tier_df = ex_df[(ex_df['market_cap'] >= lo) & (ex_df['market_cap'] < hi)]
                if len(tier_df) == 0:
                    continue
                
                key = f"{exchange} | {tier_name}"
                top = tier_df.head(top_n)
                results[key] = top.to_dict('records')
            
            # Also add exchange-level top picks (all market caps)
            key = f"{exchange} | 全部"
            top = ex_df.head(top_n)
            results[key] = top.to_dict('records')
    else:
        # US: Group by market cap tier only
        for tier_name, lo, hi in tiers:
            tier_df = scored_df[(scored_df['market_cap'] >= lo) & (scored_df['market_cap'] < hi)]
            if len(tier_df) == 0:
                continue
            
            top = tier_df.head(top_n)
            results[tier_name] = top.to_dict('records')
    
    # Print results
    for key, picks in results.items():
        print(f"\n   🏷️ {key} (Top-{min(top_n, len(picks))}):")
        period_col = 'pred_30d' if market == 'CN' else 'pred_10d'
        period_label = '30d' if market == 'CN' else '10d'
        for i, p in enumerate(picks[:top_n]):
            mc = p['market_cap']
            if market == 'CN':
                mcap_str = f"¥{mc/1e8:.0f}亿" if mc >= 1e8 else f"¥{mc/1e6:.0f}M"
            else:
                mcap_str = f"${mc/1e9:.1f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
            print(f"      {i+1}. {p['symbol']:8s} {'$' if market=='US' else '¥'}{p['price']:.2f}  "
                  f"pred_{period_label}={p[period_col]:+.1f}%  {mcap_str}  {p['sector'][:15]}")
    
    # 7. Save to DB
    _save_picks(date, market, results)
    
    return {
        'picks': results,
        'date': date,
        'market': market,
        'stats': {
            'total_scored': len(scored_df),
            'total_picks': sum(len(v) for v in results.values()),
        }
    }


# ===================== Database =====================
def _init_picks_db():
    """Initialize or upgrade the picks database"""
    db_path = os.path.join(parent_dir, 'db', 'ml_daily_picks.db')
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS ml_picks_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        market TEXT NOT NULL,
        exchange TEXT DEFAULT '',
        tier TEXT DEFAULT '',
        segment TEXT DEFAULT '',
        rank INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        price REAL,
        pred_5d REAL,
        pred_10d REAL,
        pred_20d REAL,
        pred_30d REAL,
        pred_60d REAL,
        primary_pred REAL,
        holding_period TEXT DEFAULT '10d',
        market_cap REAL,
        sector TEXT DEFAULT '',
        actual_10d REAL,
        actual_30d REAL,
        actual_60d REAL,
        created_at TEXT,
        UNIQUE(date, market, segment, rank)
    )''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_v2_date ON ml_picks_v2(date, market)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_v2_symbol ON ml_picks_v2(symbol)")
    conn.commit()
    return conn


def _save_picks(date, market, results):
    """Save picks to SQLite for historical tracking"""
    conn = _init_picks_db()
    
    count = 0
    for segment, picks in results.items():
        # Parse exchange and tier from segment
        if '|' in segment:
            exchange, tier = [s.strip() for s in segment.split('|', 1)]
        else:
            exchange = market
            tier = segment
        
        for i, pick in enumerate(picks):
            conn.execute('''INSERT OR REPLACE INTO ml_picks_v2
                (date, market, exchange, tier, segment, rank, symbol, price,
                 pred_5d, pred_10d, pred_20d, pred_30d, pred_60d,
                 primary_pred, holding_period, market_cap, sector, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (date, market, exchange, tier, segment, i + 1,
                 pick['symbol'], pick['price'],
                 pick.get('pred_5d', 0), pick.get('pred_10d', 0),
                 pick.get('pred_20d', 0), pick.get('pred_30d', 0),
                 pick.get('pred_60d', 0), pick.get('primary_pred', 0),
                 pick.get('holding_period', '10d'),
                 pick.get('market_cap', 0),
                 pick.get('sector', ''),
                 datetime.now().isoformat()))
            count += 1
    
    conn.commit()
    conn.close()
    print(f"\n   💾 Saved {count} picks to ml_daily_picks.db (v2)")


# ===================== Formatting =====================
def format_picks_message(results_dict, date=None, market='US'):
    """Format picks for notification (Telegram/WeChat/Bark)"""
    if isinstance(results_dict, dict) and 'picks' in results_dict:
        results = results_dict['picks']
        date = results_dict.get('date', date)
    else:
        results = results_dict
    
    emoji = "🇺🇸" if market == 'US' else "🇨🇳"
    market_name = "美股" if market == 'US' else "A股"
    period = "10d" if market == 'US' else "30d"
    
    lines = [
        f"🤖 *ML 每日选股 | {emoji} {market_name}*",
        f"📅 {date or 'latest'}",
        "",
    ]
    
    for segment, picks in results.items():
        lines.append(f"🏷️ *{segment}*")
        for i, p in enumerate(picks[:3]):
            mcap = p.get('market_cap', 0)
            pred = p.get(f'pred_{period}', p.get('primary_pred', 0))
            if market == 'CN':
                mcap_str = f"¥{mcap/1e8:.0f}亿" if mcap >= 1e8 else ""
                lines.append(f"  {i+1}. `{p['symbol']}` ¥{p['price']:.2f} pred_{period}={pred:+.1f}% {mcap_str}")
            else:
                mcap_str = f"${mcap/1e9:.1f}B" if mcap >= 1e9 else f"${mcap/1e6:.0f}M"
                lines.append(f"  {i+1}. `{p['symbol']}` ${p['price']:.2f} pred_{period}={pred:+.1f}% {mcap_str}")
        lines.append("")
    
    lines.append("⚠️ ML模型预测，仅供参考，不构成投资建议")
    lines.append("🌐 [查看详情](https://facaila.streamlit.app/)")
    
    return "\n".join(lines)


# ===================== Historical Picks =====================
def get_historical_picks(market='US', days=30, segment=None):
    """Get historical ML picks for tracking"""
    db_path = os.path.join(parent_dir, 'db', 'ml_daily_picks.db')
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Try v2 table first
    try:
        query = f'''SELECT * FROM ml_picks_v2 
            WHERE market = ?'''
        params = [market]
        if segment:
            query += ' AND segment = ?'
            params.append(segment)
        query += f' ORDER BY date DESC, segment, rank LIMIT {days * 50}'
        df = pd.read_sql(query, conn, params=params)
    except:
        # Fallback to v1 table
        df = pd.read_sql(f'''SELECT * FROM ml_picks 
            WHERE market = ? ORDER BY date DESC, tier, rank LIMIT {days * 25}''',
            conn, params=[market])
    
    conn.close()
    return df


# ===================== Backfill Actual Returns =====================
def backfill_actual_returns(market='US'):
    """Fill in actual returns for historical picks"""
    db_path = os.path.join(parent_dir, 'db', 'ml_daily_picks.db')
    if not os.path.exists(db_path):
        return
    
    conn = sqlite3.connect(db_path)
    
    try:
        unfilled = conn.execute('''SELECT DISTINCT date, symbol FROM ml_picks_v2
            WHERE market=? AND actual_10d IS NULL
            ORDER BY date''', (market,)).fetchall()
    except:
        conn.close()
        return
    
    if not unfilled:
        print("   ✅ 所有 actual returns 已填充")
        conn.close()
        return
    
    from db.stock_history import get_stock_history
    
    filled = 0
    for pick_date, symbol in unfilled:
        try:
            hist = get_stock_history(symbol, days=90)
            if hist is None or len(hist) < 10:
                continue
            
            hist = hist.sort_index()
            pick_idx = hist.index.get_indexer([pick_date], method='nearest')[0]
            if pick_idx < 0:
                continue
            
            pick_price = hist.iloc[pick_idx]['Close']
            
            for days_ahead, col in [(10, 'actual_10d'), (30, 'actual_30d'), (60, 'actual_60d')]:
                future_idx = pick_idx + days_ahead
                if future_idx < len(hist):
                    future_price = hist.iloc[future_idx]['Close']
                    actual_ret = (future_price / pick_price - 1) * 100
                    conn.execute(f'''UPDATE ml_picks_v2 SET {col}=?
                        WHERE date=? AND symbol=? AND market=?''',
                        (actual_ret, pick_date, symbol, market))
                    filled += 1
        except:
            continue
    
    conn.commit()
    conn.close()
    print(f"   ✅ 回填 {filled} 个 actual returns")


# ===================== Main =====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ML Daily Scorer v2')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--date', default=None, help='YYYY-MM-DD')
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--backfill-returns', action='store_true', help='Backfill actual returns')
    
    args = parser.parse_args()
    
    if args.backfill_returns:
        backfill_actual_returns(args.market)
    else:
        result = score_daily_signals(args.market, args.date, args.top)
        
        if result and result.get('picks'):
            msg = format_picks_message(result, market=args.market)
            print(f"\n{'='*60}")
            print("📱 Notification Preview:")
            print(f"{'='*60}")
            print(msg)
