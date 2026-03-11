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
    """从缓存加载已计算的特征+标签"""
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
    
    print(f"   📦 缓存命中: {count:,} 行")
    
    rows = conn.execute(
        """SELECT features_json, label_1d, label_5d, label_10d, label_20d, label_30d, label_60d, group_id
        FROM feature_cache WHERE market=? AND trade_date BETWEEN ? AND ?""",
        (market, cutoff, end_date)
    ).fetchall()
    conn.close()
    
    import json as _json
    all_features = []
    all_returns = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
    all_groups = []
    
    for row in rows:
        feat = _json.loads(row[0])
        # 价格过滤
        price = feat.get('Close', feat.get('close', 0))
        tiers = {'standard': (5, 9999), 'penny': (0.01, 5)}
        if price_tier in tiers:
            lo, hi = tiers[price_tier]
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
    
    if not all_features:
        return None, None, None, None, 0
    
    feat_df = pd.DataFrame(all_features)
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [c for c in numeric_cols if c != 'Date']
    
    X = feat_df[feature_names].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    
    returns_dict = {k: np.array(v) for k, v in all_returns.items()}
    groups = np.array(all_groups)
    
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
    
    if signals_df.empty:
        return None, None, None, None, None
    
    # 价格分层
    tiers = {'standard': (5, 9999), 'penny': (0.01, 5)}
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
    
    returns_dict = {k: np.array(v) for k, v in all_returns.items()}
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
