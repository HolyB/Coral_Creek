#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速 ML 训练 — 跳过逐天/逐股票加载，直接从 SQLite 批量读取
================================================================
用法:
    PYTHONPATH=. python scripts/fast_ml_train.py --market US --days 365
    PYTHONPATH=. python scripts/fast_ml_train.py --market US --days 9999 --all-tiers
"""

import os, sys, time, sqlite3, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

CACHE_DB = str(V3_DIR / "db" / "ml_feature_cache.db")


def _init_cache_db():
    """创建特征缓存表"""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_cache (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            market TEXT NOT NULL,
            features_json TEXT NOT NULL,
            label_1d REAL, label_5d REAL, label_10d REAL,
            label_20d REAL, label_30d REAL, label_60d REAL,
            group_id INTEGER,
            PRIMARY KEY (symbol, trade_date, market)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_market_date ON feature_cache(market, trade_date)")
    conn.commit()
    return conn


def _load_from_cache(market, cutoff, end_date, price_tier='standard'):
    """从缓存加载已计算的特征+标签 (分批加载，防止OOM)"""
    conn = sqlite3.connect(CACHE_DB)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM feature_cache WHERE market=? AND trade_date BETWEEN ? AND ?",
            (market, cutoff, end_date)
        ).fetchone()[0]
    except:
        conn.close()
        return None, None, None, None, 0
    
    if count == 0:
        conn.close()
        return None, None, None, None, 0
    
    # 如果数据太多，采样以避免OOM (上限50万)
    MAX_SAMPLES = 500_000
    sample_rate = 1
    if count > MAX_SAMPLES:
        sample_rate = count // MAX_SAMPLES + 1
        print(f"   📦 缓存命中: {count:,} 行 (采样1/{sample_rate}，目标~{count//sample_rate:,} 行)")
    else:
        print(f"   📦 缓存命中: {count:,} 行")
    
    import json as _json
    all_features = []
    all_returns = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
    all_groups = []
    
    # CN 市场价格下限是 ¥3
    tiers = {'standard': (3 if market == 'CN' else 5, 9999), 'penny': (0.01, 5)}
    lo, hi = tiers.get(price_tier, (0, 1e9))
    
    # 分批读取，每批50K
    BATCH = 50_000
    cursor = conn.execute(
        """SELECT features_json, label_1d, label_5d, label_10d, label_20d, label_30d, label_60d, group_id
        FROM feature_cache WHERE market=? AND trade_date BETWEEN ? AND ?""",
        (market, cutoff, end_date)
    )
    
    loaded = 0
    kept = 0
    row_idx = 0
    while True:
        batch = cursor.fetchmany(BATCH)
        if not batch:
            break
        
        for row in batch:
            row_idx += 1
            # 采样
            if sample_rate > 1 and row_idx % sample_rate != 0:
                continue
            
            feat = _json.loads(row[0])
            price = feat.get('Close', feat.get('close', 0))
            if not (lo <= price < hi):
                continue
            
            all_features.append(feat)
            all_returns['1d'].append(row[1])
            all_returns['5d'].append(row[2])
            all_returns['10d'].append(row[3])
            all_returns['20d'].append(row[4])
            all_returns['30d'].append(row[5])
            all_returns['60d'].append(row[6])
            all_groups.append(row[7])
            kept += 1
        
        loaded += len(batch)
        print(f"   📦 加载进度: {loaded:,}/{count:,} ({loaded*100//count}%), 保留 {kept:,} 行", flush=True)
    
    conn.close()
    print(f"   ✅ 缓存加载完成: {kept:,} 行")
    
    if not all_features:
        return None, None, None, None, 0
    
    print(f"   🧮 构建特征矩阵...", flush=True)
    feat_df = pd.DataFrame(all_features)
    del all_features  # 释放内存
    
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [c for c in numeric_cols if c != 'Date']
    
    X = feat_df[feature_names].values.astype(np.float32)  # float32 省一半内存
    del feat_df  # 释放内存
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    
    returns_dict = {k: np.array(v, dtype=np.float32) for k, v in all_returns.items()}
    groups = np.array(all_groups, dtype=np.int32)
    
    print(f"   ✅ 特征矩阵: {X.shape[0]:,} × {X.shape[1]} ({X.nbytes/1e6:.0f} MB)")
    return X, returns_dict, groups, feature_names, len(X)


def _save_to_cache(all_features, all_returns, all_groups, symbols_dates, market):
    """将计算好的特征+标签写入缓存"""
    import json as _json
    conn = _init_cache_db()
    
    batch = []
    for i, (feat, (symbol, trade_date)) in enumerate(zip(all_features, symbols_dates)):
        batch.append((
            symbol, trade_date, market,
            _json.dumps(feat, default=str),
            all_returns['1d'][i] if i < len(all_returns['1d']) else None,
            all_returns['5d'][i] if i < len(all_returns['5d']) else None,
            all_returns['10d'][i] if i < len(all_returns['10d']) else None,
            all_returns['20d'][i] if i < len(all_returns['20d']) else None,
            all_returns['30d'][i] if i < len(all_returns['30d']) else None,
            all_returns['60d'][i] if i < len(all_returns['60d']) else None,
            all_groups[i],
        ))
        
        if len(batch) >= 5000:
            conn.executemany(
                "INSERT OR REPLACE INTO feature_cache VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            batch = []
    
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO feature_cache VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()
    
    conn.close()
    print(f"   💾 已缓存 {len(all_features):,} 行到 ml_feature_cache.db")


def fast_prepare_dataset(market='US', days_back=365, price_tier='standard', max_samples=2000000, use_cache=True):
    """一次性 SQL 批量读取 scan_results + stock_history，极速准备数据集"""
    from db.stock_history import get_history_db_path
    from ml.features.feature_calculator import FeatureCalculator
    
    t0 = time.time()
    
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # 0. 尝试从缓存加载
    if use_cache:
        print(f"📦 检查特征缓存 ({cutoff} ~ {end_date})...")
        _init_cache_db()
        X, returns_dict, groups, feature_names, n = _load_from_cache(market, cutoff, end_date, price_tier)
        if n > 10000:  # 缓存太少不值得用
            print(f"✅ 从缓存加载 {n:,} 样本, {len(feature_names)} 特征 ({time.time()-t0:.1f}s)")
            return X, returns_dict, groups, feature_names, None
        elif n > 0:
            print(f"   缓存只有 {n:,} 样本，不够，重新计算...")
    
    # 1. 一次性读取所有 scan_results
    db_path = str(V3_DIR / "db" / "coral_creek.db")
    conn = sqlite3.connect(db_path)
    
    print(f"📊 批量加载 scan_results ({cutoff} ~ {end_date})...")
    signals_df = pd.read_sql_query(
        """SELECT symbol, scan_date, price, 
           COALESCE(blue_daily,0) as blue_daily, 
           COALESCE(blue_weekly,0) as blue_weekly,
           COALESCE(blue_monthly,0) as blue_monthly,
           COALESCE(is_heima,0) as is_heima,
           COALESCE(is_juedi,0) as is_juedi,
           COALESCE(profit_ratio,0) as profit_ratio
        FROM scan_results 
        WHERE market=? AND scan_date BETWEEN ? AND ?
        ORDER BY scan_date, symbol""",
        conn, params=(market, cutoff, end_date)
    )
    conn.close()
    
    print(f"   {len(signals_df):,} 行, {signals_df['scan_date'].nunique()} 天, "
          f"{signals_df['symbol'].nunique()} 只股票 ({time.time()-t0:.1f}s)")
    
    if signals_df.empty or signals_df['scan_date'].nunique() < 100:
        # Fallback: scan_results 太少时直接从 stock_history 取样
        reason = "空" if signals_df.empty else f"只有{signals_df['scan_date'].nunique()}天"
        print(f"   ⚠️ scan_results {reason}，从 stock_history 直接取样...")
        
        from db.stock_history import get_history_db_path
        hist_conn = sqlite3.connect(get_history_db_path())
        
        # 每只股票取每周五的数据（减少数据量）
        signals_df = pd.read_sql_query(
            """SELECT symbol, trade_date as scan_date, close as price,
                   0 as blue_daily, 0 as blue_weekly, 0 as blue_monthly,
                   0 as is_heima, 0 as is_juedi, 0 as profit_ratio
            FROM stock_history 
            WHERE market=? AND trade_date BETWEEN ? AND ?
            AND close > 0
            AND CAST(strftime('%w', trade_date) AS INTEGER) = 5
            ORDER BY trade_date, symbol""",
            hist_conn, params=(market, cutoff, end_date)
        )
        hist_conn.close()
        
        if signals_df.empty:
            print("   ❌ stock_history 也无数据")
            return None, None, None, None, None
        
        print(f"   📊 stock_history 取样: {len(signals_df):,} 行, "
              f"{signals_df['scan_date'].nunique()} 天, "
              f"{signals_df['symbol'].nunique()} 只股票")
    
    # 价格分层
    tiers = {'standard': (5, 9999), 'penny': (0.01, 5)}
    if market == 'CN':
        tiers['standard'] = (3, 9999)  # A股最低价 ¥3
    if price_tier in tiers:
        lo, hi = tiers[price_tier]
        before = len(signals_df)
        signals_df = signals_df[(signals_df['price'] >= lo) & (signals_df['price'] < hi)]
        print(f"   价格分层 [{price_tier}] ${lo}~${hi}: {before} → {len(signals_df)}")
    
    # 如果信号太多，按日期均匀采样
    if max_samples and len(signals_df) > max_samples:
        print(f"   ⚠️ 信号过多 ({len(signals_df):,})，采样至 {max_samples:,}")
        # 每只股票保留最近 max_per_stock 条
        n_symbols = signals_df['symbol'].nunique()
        max_per_stock = max(5, max_samples // n_symbols)
        signals_df = signals_df.groupby('symbol').tail(max_per_stock)
        # 如果还是太多，随机采样
        if len(signals_df) > max_samples:
            signals_df = signals_df.sample(n=max_samples, random_state=42)
        print(f"   采样后: {len(signals_df):,} 行")
    
    # 2. 一次性读取所有需要的 stock_history
    symbols = signals_df['symbol'].unique().tolist()
    print(f"\n📥 批量加载 stock_history ({len(symbols)} 只)...")
    
    t1 = time.time()
    hist_conn = sqlite3.connect(get_history_db_path())
    
    # 分批查询避免 SQLite 参数限制
    all_hist = []
    batch_size = 500
    for i in range(0, len(symbols), batch_size):
        batch_syms = symbols[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch_syms))
        df = pd.read_sql_query(
            f"""SELECT symbol, trade_date, open, high, low, close, volume
            FROM stock_history 
            WHERE market=? AND symbol IN ({placeholders})
            ORDER BY symbol, trade_date""",
            hist_conn,
            params=[market] + batch_syms
        )
        all_hist.append(df)
    
    hist_conn.close()
    hist_df = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    
    print(f"   {len(hist_df):,} 行 ({time.time()-t1:.1f}s)")
    
    if hist_df.empty:
        print("❌ 无历史数据")
        return None, None, None, None, None
    
    # 3. 构建 {symbol: DataFrame} 索引
    hist_groups = {}
    for symbol, group in hist_df.groupby('symbol'):
        sdf = group.copy()
        sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])
        sdf = sdf.set_index('trade_date').sort_index()
        sdf.columns = [c.capitalize() if c != 'trade_date' else c for c in sdf.columns]
        if len(sdf) >= 60:
            hist_groups[symbol] = sdf
    
    print(f"   有效股票: {len(hist_groups)}")
    
    # 释放原始 hist_df 节省内存
    del hist_df, all_hist
    import gc; gc.collect()
    
    # 4. 计算特征 + 标签 (向量化)
    print(f"\n🧮 计算特征和标签（含高级特征）...")
    t2 = time.time()
    
    calculator = FeatureCalculator()
    
    # 初始化高级特征计算器
    try:
        from ml.advanced_features import AdvancedFeatureEngineer
        adv_engineer = AdvancedFeatureEngineer()
        print("   ✅ AdvancedFeatureEngineer 已加载")
    except Exception as e:
        adv_engineer = None
        print(f"   ⚠️ AdvancedFeatureEngineer 跳过: {e}")
    
    try:
        from ml.alpha_factors import Alpha158Factors
        alpha158 = Alpha158Factors()
        print("   ✅ Alpha158Factors 已加载")
    except Exception as e:
        alpha158 = None
        print(f"   ⚠️ Alpha158Factors 跳过: {e}")
    
    has_caisen = False
    try:
        from ml.caisen_features import compute_caisen_features
        has_caisen = True
        print("   ✅ Caisen 筹码特征 已加载")
    except Exception as e:
        print(f"   ⚠️ Caisen 筹码特征 跳过: {e}")
    
    has_strategy = False
    try:
        from strategies.auto_backtester import generate_strategy_features
        has_strategy = True
        print("   ✅ 策略信号特征 已加载")
    except Exception as e:
        print(f"   ⚠️ 策略信号特征 跳过: {e}")
    
    # 加载大盘指标数据 (US=SPY, CN=沪深300)
    spy_feat_by_date = {}
    try:
        if market == 'CN':
            # CN: 从预计算的 cn_index_data.json 加载沪深300
            cn_idx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'cn_index_data.json')
            if os.path.exists(cn_idx_path):
                with open(cn_idx_path) as f:
                    cn_idx = json.load(f)
                # Rename hs300_* to spy_* for feature compatibility
                for dt, feats in cn_idx.items():
                    spy_feat_by_date[dt] = {
                        'spy_close': feats.get('hs300_close', 0),
                        'spy_rsi14': feats.get('hs300_rsi14', 0),
                        'spy_ret5': feats.get('hs300_ret5', 0),
                        'spy_ret20': feats.get('hs300_ret20', 0),
                        'spy_vol10': feats.get('hs300_vol10', 0),
                        'spy_above_ma20': feats.get('hs300_above_ma20', 0),
                        'spy_ma20_dist': feats.get('hs300_ma20_dist', 0),
                    }
                print(f"   ✅ 沪深300 大盘特征: {len(spy_feat_by_date)} 天")
            else:
                print("   ⚠️ cn_index_data.json 未找到，跳过大盘特征")
        else:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
            from polygon import RESTClient as PolyClient
            pclient = PolyClient(os.getenv('POLYGON_API_KEY'))
            spy_close = {}
            for bar in pclient.list_aggs('SPY', 1, 'day', '2024-01-01', '2026-12-31', limit=50000):
                dt = pd.Timestamp(bar.timestamp, unit='ms').strftime('%Y-%m-%d')
                spy_close[dt] = bar.close
            if spy_close:
                sdf = pd.DataFrame({'spy_close': spy_close}).sort_index()
                sdf['spy_rsi14'] = 100 - 100 / (1 + sdf['spy_close'].diff().clip(lower=0).rolling(14).mean() /
                                                    sdf['spy_close'].diff().clip(upper=0).abs().rolling(14).mean())
                sdf['spy_ret5'] = sdf['spy_close'].pct_change(5) * 100
                sdf['spy_ret20'] = sdf['spy_close'].pct_change(20) * 100
                sdf['spy_vol10'] = sdf['spy_close'].pct_change().rolling(10).std() * 100
                sdf['spy_ma20'] = sdf['spy_close'].rolling(20).mean()
                sdf['spy_above_ma20'] = (sdf['spy_close'] > sdf['spy_ma20']).astype(float)
                sdf['spy_ma20_dist'] = (sdf['spy_close'] / sdf['spy_ma20'] - 1) * 100
                mkt_cols = ['spy_close','spy_rsi14','spy_ret5','spy_ret20','spy_vol10','spy_above_ma20','spy_ma20_dist']
                spy_feat_by_date = sdf[mkt_cols].to_dict('index')
                print(f"   ✅ SPY 大盘特征: {len(spy_feat_by_date)} 天")
    except Exception as e:
        print(f"   ⚠️ 大盘特征跳过: {e}")
    
    # 加载 market_cap 数据
    db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db')
    mcap_dict = {}
    try:
        mcap_file = 'cn_mcap_dict.json' if market == 'CN' else 'mcap_dict.json'
        mcap_path = os.path.join(db_dir, mcap_file)
        if os.path.exists(mcap_path):
            with open(mcap_path) as f:
                mcap_dict = json.load(f)
            print(f"   ✅ Market Cap ({mcap_file}): {len(mcap_dict)} 只股票")
        else:
            # Fallback: from scan_results DB
            import sqlite3 as sq
            mdb = sq.connect(os.path.join(db_dir, 'coral_creek.db'))
            for sym, mc in mdb.execute('SELECT symbol, MAX(market_cap) FROM scan_results WHERE market_cap > 0 GROUP BY symbol'):
                mcap_dict[sym] = float(mc)
            if mcap_dict:
                with open(mcap_path, 'w') as f:
                    json.dump(mcap_dict, f)
            print(f"   ✅ Market Cap (from DB): {len(mcap_dict)} 只股票")
    except Exception as e:
        print(f"   ⚠️ Market Cap 跳过: {e}")
    
    # 加载 cross-asset ETF 数据 (TLT, GLD, USO, QQQ, IWM)
    crossasset_by_date = {}  # date -> {bonds_ret5, gold_ret5, ...}
    try:
        if market == 'CN':
            # CN: 预计算的 cn_crossasset_data.json (直接 date->features flat format)
            ca_path = os.path.join(db_dir, 'cn_crossasset_data.json')
            if os.path.exists(ca_path):
                with open(ca_path) as f:
                    cn_ca = json.load(f)
                # Rename CN ETF keys to match US feature names for compatibility
                rename_map = {
                    'hs300etf': 'nasdaq', 'zz500etf': 'smallcap', 'cyb_etf': 'oil',
                    'bond_etf': 'bonds', 'gold_etf': 'gold'
                }
                for dt, feats in cn_ca.items():
                    mapped = {}
                    for k, v in feats.items():
                        for cn_prefix, us_prefix in rename_map.items():
                            if k.startswith(cn_prefix):
                                new_key = k.replace(cn_prefix, us_prefix)
                                mapped[new_key] = float(v) if v is not None else 0.0
                                break
                    crossasset_by_date[dt] = mapped
                print(f"   ✅ CN Cross-Asset: {len(crossasset_by_date)} 天, 5 ETFs")
        else:
            ca_path = os.path.join(db_dir, 'crossasset_data.json')
            if os.path.exists(ca_path):
                with open(ca_path) as f:
                    ca_raw = json.load(f)
            else:
                # Auto-fetch from Polygon
                print("   📥 Fetching cross-asset ETFs from Polygon...")
                ca_raw = {}
                etf_map = {'TLT': 'bonds', 'GLD': 'gold', 'USO': 'oil', 'QQQ': 'nasdaq', 'IWM': 'smallcap'}
                for ticker, label in etf_map.items():
                    daily = {}
                    for bar in pclient.list_aggs(ticker, 1, 'day', '2024-01-01', '2026-12-31', limit=50000):
                        dt = pd.Timestamp(bar.timestamp, unit='ms').strftime('%Y-%m-%d')
                        daily[dt] = bar.close
                    df = pd.DataFrame({f'{label}_close': daily}).sort_index()
                    df[f'{label}_ret5'] = df[f'{label}_close'].pct_change(5) * 100
                    df[f'{label}_ret20'] = df[f'{label}_close'].pct_change(20) * 100
                    df[f'{label}_ma20_dist'] = (df[f'{label}_close'] / df[f'{label}_close'].rolling(20).mean() - 1) * 100
                    clean = {}
                    for d, vals in df.to_dict('index').items():
                        clean[d] = {k: (v if not pd.isna(v) else None) for k, v in vals.items()}
                    ca_raw[label] = clean
                with open(ca_path, 'w') as f:
                    json.dump(ca_raw, f)
            # Flatten: per date, all cross-asset features
            all_dates = set()
            for label_data in ca_raw.values():
                all_dates.update(label_data.keys())
            for dt in all_dates:
                row = {}
                for label, label_data in ca_raw.items():
                    if dt in label_data:
                        for k, v in label_data[dt].items():
                            row[k] = v if v is not None else 0.0
                crossasset_by_date[dt] = row
            print(f"   ✅ Cross-Asset: {len(crossasset_by_date)} 天, 5 ETFs")
    except Exception as e:
        print(f"   ⚠️ Cross-Asset 跳过: {e}")
    
    # 加载 sector/industry 数据
    sector_dict = {}
    sector_to_id = {}  # SIC description -> numeric ID
    try:
        sec_file = 'cn_sector_dict.json' if market == 'CN' else 'sector_dict.json'
        sec_path = os.path.join(db_dir, sec_file)
        if os.path.exists(sec_path):
            with open(sec_path) as f:
                sector_dict = json.load(f)
        elif market != 'CN':
            # Auto-fetch from Polygon (US only)
            print("   📥 Fetching sector/industry from Polygon...")
            if mcap_dict:
                import time as _t
                for sym in mcap_dict:
                    try:
                        d = pclient.get_ticker_details(sym)
                        sector_dict[sym] = {
                            'sic': getattr(d, 'sic_description', '') or '',
                            'type': getattr(d, 'type', '') or '',
                        }
                    except:
                        pass
                with open(sec_path, 'w') as f:
                    json.dump(sector_dict, f)
        # Build sector ID mapping (top 30 sectors get IDs, rest = 0)
        if sector_dict:
            from collections import Counter
            sic_counts = Counter(d.get('sic', '') for d in sector_dict.values() if d.get('sic'))
            for i, (sic, _) in enumerate(sic_counts.most_common(30)):
                sector_to_id[sic] = i + 1
        print(f"   ✅ Sector: {len(sector_dict)} 只股票, {len(sector_to_id)} 行业")
    except Exception as e:
        print(f"   ⚠️ Sector 跳过: {e}")
    
    all_features = []
    all_returns = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
    all_groups = []
    all_sym_dates = []  # (symbol, scan_date) for cache
    skipped = 0
    processed = 0
    
    # 按股票处理（每只股票只计算一次特征序列）
    for sym_idx, symbol in enumerate(hist_groups.keys()):
        hist = hist_groups[symbol]
        
        # 准备 history DataFrame for FeatureCalculator
        calc_hist = hist.reset_index()
        calc_hist = calc_hist.rename(columns={'trade_date': 'Date'})
        
        # 计算基础特征序列
        try:
            features_df = calculator.calculate_all(calc_hist)
        except Exception:
            skipped += 1
            continue
        
        if features_df.empty or len(features_df) < 60:
            skipped += 1
            continue
        
        # 计算高级特征（每只股票只算一次）
        adv_features_df = None
        alpha_df = None
        caisen_df = None
        strategy_df = None
        
        # 准备标准化 OHLCV for 高级特征
        hist_std = hist.copy()
        col_map = {}
        for c in hist_std.columns:
            cl = c.lower()
            if cl == 'close': col_map[c] = 'Close'
            elif cl == 'open': col_map[c] = 'Open'
            elif cl == 'high': col_map[c] = 'High'
            elif cl == 'low': col_map[c] = 'Low'
            elif cl == 'volume': col_map[c] = 'Volume'
        if col_map:
            hist_std = hist_std.rename(columns=col_map)
        
        # AdvancedFeatureEngineer
        if adv_engineer and all(c in hist_std.columns for c in ['Close', 'High', 'Low', 'Volume']):
            try:
                adv_features_df = adv_engineer.transform(hist_std).reset_index()
                if 'Date' not in adv_features_df.columns:
                    adv_features_df = adv_features_df.rename(columns={adv_features_df.columns[0]: 'Date'})
            except Exception:
                pass
        
        # Alpha158
        if alpha158 and all(c in hist_std.columns for c in ['Close', 'High', 'Low', 'Volume']):
            try:
                alpha_df = alpha158.compute(hist_std).reset_index()
                if 'Date' not in alpha_df.columns:
                    alpha_df = alpha_df.rename(columns={alpha_df.columns[0]: 'Date'})
            except Exception:
                pass
        
        # Caisen 筹码特征
        if has_caisen:
            try:
                caisen_df = compute_caisen_features(calc_hist)
                if caisen_df is not None and not caisen_df.empty:
                    if 'Date' not in caisen_df.columns and caisen_df.index.name:
                        caisen_df = caisen_df.reset_index()
                        if 'Date' not in caisen_df.columns:
                            caisen_df = caisen_df.rename(columns={caisen_df.columns[0]: 'Date'})
            except Exception:
                caisen_df = None
        
        # 策略信号特征
        if has_strategy:
            try:
                strategy_df = generate_strategy_features(calc_hist)
                if strategy_df is not None and not strategy_df.empty:
                    if not isinstance(strategy_df.index, pd.DatetimeIndex):
                        strategy_df.index = calc_hist['Date'].iloc[:len(strategy_df)]
            except Exception:
                strategy_df = None
        
        # 获取该股票的信号日期
        sym_signals = signals_df[signals_df['symbol'] == symbol]
        
        for _, signal in sym_signals.iterrows():
            signal_date = pd.to_datetime(signal['scan_date'])
            
            # 找到最近的特征行（防止未来泄漏）
            eligible_idx = features_df.index[features_df['Date'] <= signal_date]
            if len(eligible_idx) == 0:
                continue
            closest_idx = int(eligible_idx[-1])
            ref_date = features_df.loc[closest_idx, 'Date']
            if (signal_date - ref_date).days > 3:
                continue
            
            # 提取基础特征
            feature_row = features_df.loc[closest_idx]
            feature_dict = {col: feature_row.get(col, np.nan) 
                           for col in features_df.columns if col not in ['Date', 'date_diff']}
            feature_dict['blue_daily'] = signal.get('blue_daily', 0)
            feature_dict['blue_weekly'] = signal.get('blue_weekly', 0)
            feature_dict['blue_monthly'] = signal.get('blue_monthly', 0)
            feature_dict['is_heima'] = signal.get('is_heima', 0)
            feature_dict['profit_ratio'] = signal.get('profit_ratio', 0)
            
            # 加入高级特征
            skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'open', 'high', 'low', 'close', 'volume', 'Symbol'}
            
            if adv_features_df is not None and 'Date' in adv_features_df.columns:
                try:
                    adv_eligible = adv_features_df[adv_features_df['Date'] <= signal_date]
                    if len(adv_eligible) > 0:
                        adv_row = adv_eligible.iloc[-1]
                        for col in adv_row.index:
                            if col not in skip_cols and col not in feature_dict:
                                val = adv_row[col]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    feature_dict[f'adv_{col}'] = float(val)
                except Exception:
                    pass
            
            if alpha_df is not None and 'Date' in alpha_df.columns:
                try:
                    a_eligible = alpha_df[alpha_df['Date'] <= signal_date]
                    if len(a_eligible) > 0:
                        a_row = a_eligible.iloc[-1]
                        for col in a_row.index:
                            if col not in skip_cols and col not in feature_dict:
                                val = a_row[col]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    feature_dict[f'a158_{col}'] = float(val)
                except Exception:
                    pass
            
            if caisen_df is not None and 'Date' in caisen_df.columns:
                try:
                    cs_eligible = caisen_df[caisen_df['Date'] <= signal_date]
                    if len(cs_eligible) > 0:
                        cs_row = cs_eligible.iloc[-1]
                        for col in cs_row.index:
                            if col != 'Date' and col not in feature_dict:
                                val = cs_row[col]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    feature_dict[col] = float(val)
                except Exception:
                    pass
            
            if strategy_df is not None:
                try:
                    sf_eligible = strategy_df[strategy_df.index <= signal_date]
                    if len(sf_eligible) > 0:
                        sf_row = sf_eligible.iloc[-1]
                        for col in sf_row.index:
                            if col not in feature_dict:
                                val = sf_row[col]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    feature_dict[col] = float(val)
                except Exception:
                    pass
            
            # === 新增特征 (26个) ===
            # A) Market Cap
            mc = mcap_dict.get(symbol, 0)
            feature_dict['mcap_log'] = np.log10(max(mc, 1e6))
            feature_dict['mcap_bucket'] = 0 if mc < 3e8 else (1 if mc < 2e9 else (2 if mc < 1e10 else 3))
            
            # B) Calendar features
            feature_dict['day_of_week'] = signal_date.dayofweek
            feature_dict['month'] = signal_date.month
            feature_dict['is_month_start'] = 1.0 if signal_date.day <= 5 else 0.0
            feature_dict['is_month_end'] = 1.0 if signal_date.day >= 25 else 0.0
            feature_dict['is_quarter_end'] = 1.0 if signal_date.month in [3,6,9,12] and signal_date.day >= 15 else 0.0
            feature_dict['is_jan'] = 1.0 if signal_date.month == 1 else 0.0
            feature_dict['is_dec'] = 1.0 if signal_date.month == 12 else 0.0
            
            # C) Candle pattern features
            O = feature_dict.get('Open', 0)
            H = feature_dict.get('High', 0)
            L = feature_dict.get('Low', 0)
            C = feature_dict.get('Close', 0)
            rng = H - L if H > L else 0.01
            body = abs(C - O)
            feature_dict['candle_body_pct'] = body / rng * 100
            feature_dict['candle_upper_shadow'] = (H - max(O, C)) / rng * 100
            feature_dict['candle_lower_shadow'] = (min(O, C) - L) / rng * 100
            feature_dict['candle_is_doji'] = 1.0 if body / rng < 0.1 else 0.0
            feature_dict['candle_is_hammer'] = 1.0 if (min(O,C) - L) > 2 * body and body > 0 and (H - max(O,C)) < body else 0.0
            feature_dict['candle_is_shooting_star'] = 1.0 if (H - max(O,C)) > 2 * body and body > 0 and (min(O,C) - L) < body else 0.0
            feature_dict['candle_is_marubozu'] = 1.0 if body / rng > 0.9 else 0.0
            feature_dict['candle_is_long_body'] = 1.0 if body / rng > 0.7 else 0.0
            
            # D) Price features
            feature_dict['price_log'] = np.log10(max(C, 0.01))
            feature_dict['price_bucket'] = 0 if C < 10 else (1 if C < 30 else (2 if C < 100 else 3))
            
            # E) SPY market features
            dt_str = signal_date.strftime('%Y-%m-%d')
            if dt_str in spy_feat_by_date:
                for mk, mv in spy_feat_by_date[dt_str].items():
                    feature_dict[mk] = mv if not pd.isna(mv) else 0.0
            else:
                for mk in ['spy_close','spy_rsi14','spy_ret5','spy_ret20','spy_vol10','spy_above_ma20','spy_ma20_dist']:
                    feature_dict[mk] = 0.0
            
            # F) Cross-asset ETF features
            if dt_str in crossasset_by_date:
                for mk, mv in crossasset_by_date[dt_str].items():
                    feature_dict[mk] = float(mv) if mv is not None else 0.0
            
            # G) Sector/Industry features
            sec_info = sector_dict.get(symbol, {})
            sic = sec_info.get('sic', '')
            feature_dict['sector_id'] = sector_to_id.get(sic, 0)
            feature_dict['is_pharma'] = 1.0 if 'PHARMA' in sic.upper() else 0.0
            feature_dict['is_biotech'] = 1.0 if 'BIOLOG' in sic.upper() else 0.0
            feature_dict['is_bank'] = 1.0 if 'BANK' in sic.upper() else 0.0
            feature_dict['is_reit'] = 1.0 if 'REAL ESTATE' in sic.upper() else 0.0
            feature_dict['is_software'] = 1.0 if 'SOFTWARE' in sic.upper() else 0.0
            feature_dict['is_semiconductor'] = 1.0 if 'SEMICOND' in sic.upper() else 0.0
            feature_dict['is_etf'] = 1.0 if sec_info.get('type', '') in ('ETF', 'FUND') else 0.0
            
            # H) Relative strength vs SPY (stock return vs SPY return)
            if dt_str in spy_feat_by_date:
                spy_r5 = spy_feat_by_date[dt_str].get('spy_ret5', 0) or 0
                spy_r20 = spy_feat_by_date[dt_str].get('spy_ret20', 0) or 0
                stock_r5 = feature_dict.get('a158_roc_5', 0) or feature_dict.get('adv_momentum_5', 0) or 0
                stock_r20 = feature_dict.get('a158_roc_20', 0) or feature_dict.get('adv_momentum_20', 0) or 0
                feature_dict['rs_vs_spy_5d'] = stock_r5 - spy_r5
                feature_dict['rs_vs_spy_20d'] = stock_r20 - spy_r20
            else:
                feature_dict['rs_vs_spy_5d'] = 0.0
                feature_dict['rs_vs_spy_20d'] = 0.0
            
            # 计算标签（未来收益）
            entry_idx = closest_idx + 1
            if entry_idx >= len(features_df):
                continue
            entry_price = features_df.loc[entry_idx, 'Open']
            if pd.isna(entry_price) or float(entry_price) <= 0:
                continue
            
            for days in [1, 5, 10, 20, 30, 60]:
                future_idx = entry_idx + days
                if future_idx < len(features_df):
                    fp = features_df.loc[future_idx, 'Close']
                    if pd.isna(fp) or float(fp) <= 0:
                        all_returns[f'{days}d'].append(np.nan)
                    else:
                        ret = (fp - entry_price) / entry_price * 100
                        all_returns[f'{days}d'].append(ret)
                else:
                    all_returns[f'{days}d'].append(np.nan)
            
            all_features.append(feature_dict)
            all_groups.append(signal_date.toordinal())
            all_sym_dates.append((symbol, signal['scan_date']))
            processed += 1
        
        if (sym_idx + 1) % 500 == 0:
            elapsed = time.time() - t2
            n_feat = len(all_features[-1]) if all_features else 0
            print(f"   {sym_idx+1}/{len(hist_groups)} 股票, {processed} 样本, ~{n_feat} 特征 ({elapsed:.0f}s)")
    
    print(f"   完成: {processed} 样本, {skipped} 跳过 ({time.time()-t2:.0f}s)")
    
    if not all_features:
        return None, None, None, None, None
    
    # 5. 保存到缓存
    if use_cache:
        print(f"\n💾 保存特征缓存...")
        _save_to_cache(all_features, all_returns, all_groups, all_sym_dates, market)
    
    # 6. 转换为训练矩阵
    feat_df = pd.DataFrame(all_features)
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [c for c in numeric_cols if c != 'Date']
    
    X = feat_df[feature_names].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    
    returns_dict = {k: np.array(v, dtype=np.float64) for k, v in all_returns.items()}
    # Clip extreme returns to remove penny stock outliers (e.g. +1541%)
    for k in returns_dict:
        arr = returns_dict[k]
        n_extreme = np.sum(np.abs(np.nan_to_num(arr)) > 200)
        if n_extreme > 0:
            print(f"   ⚠️ Clipping {n_extreme} extreme values in {k} (|ret| > 200%)")
        returns_dict[k] = np.clip(arr, -100, 200)
    groups = np.array(all_groups)
    
    print(f"\n✅ 数据集准备完成:")
    print(f"   样本: {len(X):,}, 特征: {len(feature_names)}, 分组: {len(np.unique(groups))}")
    print(f"   总耗时: {time.time()-t0:.0f}s")
    
    return X, returns_dict, groups, feature_names, feat_df


def train_models(X, returns_dict, groups, feature_names, market='US', price_tier='standard'):
    """训练 XGBoost + Ranker"""
    from ml.models.return_predictor import ReturnPredictor
    from ml.models.signal_ranker import SignalRanker
    
    suffix = '_penny' if price_tier == 'penny' else ''
    model_dir = V3_DIR / "ml" / "saved_models" / f"v2_{market.lower()}{suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Drawdowns (简单估算)
    drawdowns_dict = {}
    for days_key in ['5d', '20d', '30d', '60d']:
        ret = returns_dict.get(days_key)
        if ret is not None:
            ret = np.array(ret, dtype=float)
            drawdowns_dict[days_key] = np.abs(np.minimum(np.nan_to_num(ret, nan=0.0), 0))
        else:
            drawdowns_dict[days_key] = np.zeros(len(X))
    
    # 1. ReturnPredictor
    print(f"\n{'='*60}")
    print(f"📈 训练 ReturnPredictor ({price_tier})...")
    rp = ReturnPredictor()
    rp_metrics = rp.train(X, returns_dict, feature_names, groups=groups)
    rp.save(str(model_dir))
    
    # 2. SignalRanker
    print(f"\n{'='*60}")
    print(f"🏆 训练 SignalRanker ({price_tier})...")
    sr = SignalRanker()
    sr_metrics = sr.train(X, returns_dict, drawdowns_dict, groups, feature_names)
    sr.save(str(model_dir))
    
    # 3. 保存元数据
    with open(model_dir / "feature_names.json", 'w') as f:
        json.dump(feature_names, f)
    
    meta = {
        'market': market,
        'price_tier': price_tier,
        'samples': len(X),
        'features': len(feature_names),
        'groups': len(np.unique(groups)),
        'trained_at': datetime.now().isoformat(),
        'return_predictor': rp_metrics,
    }
    with open(model_dir / "training_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"\n✅ 模型保存: {model_dir}")
    return meta


def run(market='US', days_back=365, all_tiers=False, max_samples=2000000):
    tiers = ['standard', 'penny'] if all_tiers else ['standard']
    
    for tier in tiers:
        print(f"\n{'#'*60}")
        print(f"## {market} {tier.upper()} (days={days_back})")
        print(f"{'#'*60}")
        
        X, returns_dict, groups, feature_names, _ = fast_prepare_dataset(
            market=market, days_back=days_back, price_tier=tier, max_samples=max_samples
        )
        
        if X is None:
            print("❌ 数据准备失败")
            continue
        
        # Ensure returns_dict values are float numpy arrays (convert None → NaN)
        for k in returns_dict:
            if returns_dict[k] is not None:
                returns_dict[k] = np.array(returns_dict[k], dtype=np.float64)
        
        train_models(X, returns_dict, groups, feature_names, market, tier)
    
    print(f"\n🎉 全部训练完成!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--all-tiers', action='store_true')
    parser.add_argument('--max-samples', type=int, default=2000000, help='最大样本数（防OOM）')
    args = parser.parse_args()
    
    run(args.market, args.days, args.all_tiers, args.max_samples)
