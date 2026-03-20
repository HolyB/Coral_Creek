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
    cache_db_path = os.path.join(parent_dir, 'db', 'ml_feature_cache.db')
    cache_db = None
    if os.path.exists(cache_db_path):
        cache_db = sqlite3.connect(cache_db_path)
    
    # Prepare live feature computation fallback
    hist_db = None
    fc = None
    try:
        from db.stock_history import get_history_db_path
        from ml.features.feature_calculator import FeatureCalculator
        hist_path = get_history_db_path()
        # Check if DB actually has data (init_history_db creates empty DB on import)
        has_data = False
        if os.path.exists(hist_path):
            tmp = sqlite3.connect(hist_path)
            cnt = tmp.execute('SELECT COUNT(*) FROM stock_history').fetchone()[0]
            tmp.close()
            has_data = cnt > 0
        # Fallback: try partial_stock_history.db (committed to repo, ~2MB)
        if not has_data:
            partial_path = os.path.join(parent_dir, 'db', 'partial_stock_history.db')
            if os.path.exists(partial_path):
                hist_path = partial_path
                has_data = True
        if has_data:
            hist_db = sqlite3.connect(hist_path)
            fc = FeatureCalculator()
    except:
        pass
    
    min_price = 3 if market == 'CN' else 5
    scored = []
    cache_hits = 0
    live_hits = 0
    
    for _, row in signals_df.iterrows():
        symbol = row.get('symbol', '')
        price = float(row.get('price', 0) or 0)
        if price < min_price:
            continue
        
        feat = None
        
        # Try 1: Get features from cache
        if cache_db:
            cache_row = cache_db.execute(
                'SELECT features_json FROM feature_cache WHERE symbol=? AND trade_date=? LIMIT 1',
                (symbol, date)
            ).fetchone()
            if cache_row:
                feat = json.loads(cache_row[0])
                cache_hits += 1
        
        # Try 2: Compute features live from stock_history
        if feat is None and hist_db and fc:
            try:
                df = pd.read_sql_query(
                    '''SELECT trade_date, open as Open, high as High, low as Low,
                       close as Close, volume as Volume
                    FROM stock_history WHERE symbol=? AND market=?
                    ORDER BY trade_date DESC LIMIT 150''',
                    hist_db, params=(symbol, market)
                )
                if not df.empty and len(df) >= 20:
                    df = df.sort_values('trade_date').reset_index(drop=True)
                    feat = fc.get_latest_features(df)
                    if feat:
                        # Add mcap features
                        mc = mcap_dict.get(symbol, 0)
                        feat['mcap_log'] = float(np.log10(mc)) if mc > 0 else 0
                        feat['mcap_bucket'] = (0 if mc <= 0 else 1 if mc < 1e9 else 2 if mc < 5e9
                            else 3 if mc < 1e10 else 4 if mc < 3e10 else 5 if mc < 1e11 else 6)
                        feat['sector_id'] = 0
                        feat['is_etf'] = 0
                        live_hits += 1
            except:
                pass
        
        if feat is None:
            continue
        
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
    
    if cache_db:
        cache_db.close()
    if hist_db:
        hist_db.close()
    
    if not scored:
        print(f"   ❌ 无法评分（cache={cache_hits}, live={live_hits}）")
        return {}
    
    scored_df = pd.DataFrame(scored).sort_values('primary_pred', ascending=False)
    print(f"   评分完成: {len(scored_df)} 只股票 (cache={cache_hits}, live={live_hits})")
    
    # 6. Group by tier and exchange
    tiers = US_TIERS if market == 'US' else CN_TIERS
    results = {}
    
    if market == 'CN':
        # CN: Group by board type (2 groups) × market cap tier
        # 主板 = 上证主板 + 深证主板, 中小创科 = 创业板 + 科创板
        board_groups = {
            '主板': ['上证主板', '深证主板'],
            '中小创科': ['创业板', '科创板'],
        }
        for group_name, exchanges in board_groups.items():
            ex_df = scored_df[scored_df['exchange'].isin(exchanges)]
            if len(ex_df) == 0:
                continue
            
            for tier_name, lo, hi in tiers:
                tier_df = ex_df[(ex_df['market_cap'] >= lo) & (ex_df['market_cap'] < hi)]
                if len(tier_df) == 0:
                    continue
                
                key = f"{group_name} | {tier_name}"
                top = tier_df.head(top_n)
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
    
    # Compute strategy markers for each symbol
    markers = {}  # {symbol: [tags]}
    try:
        # Flatten all picks to compute top10% threshold
        all_flat = []
        for seg, picks in results.items():
            if isinstance(picks, list):
                all_flat.extend(picks)
        
        if all_flat:
            preds = sorted([p.get('primary_pred', p.get(f'pred_{period}', 0)) for p in all_flat], reverse=True)
            top10_threshold = preds[max(0, len(preds) // 10 - 1)] if preds else 999
            
            # Large cap threshold
            min_large = 2e9 if market == 'US' else 1e10
            
            # Streak: check if symbol appeared in yesterday's picks
            streak_syms = set()
            try:
                from scripts.ml_portfolio_tracker import _get_all_picks, compute_signal_streaks
                picks_df = _get_all_picks(market)
                streaks = compute_signal_streaks(picks_df)
                for p in all_flat:
                    if streaks.get((p['symbol'], date), 0) >= 2:
                        streak_syms.add(p['symbol'])
            except:
                pass
            
            for p in all_flat:
                sym = p['symbol']
                tags = []
                pred_val = p.get('primary_pred', p.get(f'pred_{period}', 0))
                mcap = p.get('market_cap', 0)
                
                if pred_val >= top10_threshold:
                    tags.append('🔥')  # top 10%
                if sym in streak_syms:
                    tags.append('🔄')  # streak ≥ 2d
                if mcap >= min_large:
                    tags.append('💎')  # large cap
                
                if tags:
                    markers[sym] = ' '.join(tags)
    except:
        pass
    
    lines = [
        f"🤖 *ML 每日选股 | {emoji} {market_name}*",
        f"📅 {date or 'latest'}",
        f"🔥=Top10% 🔄=连续信号 💎=大盘",
        "",
    ]
    
    if market == 'CN':
        # CN: flatten all picks, show global top 15 sorted by pred
        all_flat = []
        for seg, picks in results.items():
            if isinstance(picks, list):
                for p in picks:
                    p['_segment'] = seg
                    all_flat.append(p)
        all_flat.sort(key=lambda x: x.get('primary_pred', x.get(f'pred_{period}', 0)), reverse=True)
        
        lines.append(f"📋 *全局 Top {min(15, len(all_flat))}*")
        for i, p in enumerate(all_flat[:15]):
            mcap = p.get('market_cap', 0)
            pred = p.get(f'pred_{period}', p.get('primary_pred', 0))
            sym = p['symbol']
            tag_str = f" {markers[sym]}" if sym in markers else ""
            mcap_str = f"¥{mcap/1e8:.0f}亿" if mcap >= 1e8 else ""
            seg_short = p.get('_segment', '')
            # Extract tier only (e.g. "100-300亿" from "主板 | 100-300亿")
            tier_part = seg_short.split('|')[-1].strip() if '|' in seg_short else seg_short
            lines.append(f"  {i+1:>2}. `{sym}` ¥{p['price']:.2f} {pred:+.1f}% {mcap_str}{tag_str}")
        lines.append("")
    else:
        # US: keep per-tier format (5 tiers × 3 = 15 picks)
        for segment, picks in results.items():
            lines.append(f"🏷️ *{segment}*")
            for i, p in enumerate(picks[:3]):
                mcap = p.get('market_cap', 0)
                pred = p.get(f'pred_{period}', p.get('primary_pred', 0))
                sym = p['symbol']
                tag_str = f" {markers[sym]}" if sym in markers else ""
                mcap_str = f"${mcap/1e9:.1f}B" if mcap >= 1e9 else f"${mcap/1e6:.0f}M"
                lines.append(f"  {i+1}. `{sym}` ${p['price']:.2f} {pred:+.1f}% {mcap_str}{tag_str}")
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
    today = pd.Timestamp.now().normalize()
    for pick_date, symbol in unfilled:
        try:
            hist = get_stock_history(symbol, market=market, days=90)
            if hist is None or len(hist) < 2:
                continue
            
            hist = hist.sort_values('Date').reset_index(drop=True)
            pick_ts = pd.Timestamp(pick_date)
            
            # Get pick day price
            pick_row = hist[hist['Date'] == pick_ts]
            if pick_row.empty:
                pick_row = hist[hist['Date'] <= pick_ts].tail(1)
            if pick_row.empty:
                continue
            
            pick_price = pick_row.iloc[0]['Close']
            if pick_price <= 0:
                continue
            
            # Get trading days AFTER pick_date
            future_rows = hist[hist['Date'] > pick_ts].sort_values('Date').reset_index(drop=True)
            if future_rows.empty:
                continue  # No data after buy date at all
            
            for days_ahead, col in [(10, 'actual_10d'), (30, 'actual_30d'), (60, 'actual_60d')]:
                if len(future_rows) >= days_ahead:
                    # Holding period elapsed: use the N-th trading day price
                    future_price = future_rows.iloc[days_ahead - 1]['Close']
                else:
                    # Not yet elapsed: use latest available price (floating return)
                    future_price = future_rows.iloc[-1]['Close']
                
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
