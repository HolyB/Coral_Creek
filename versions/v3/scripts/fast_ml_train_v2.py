#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速 ML 训练 v2 — 直接从 stock_history 生成样本（不依赖 scan_results）
====================================================================
用法:
    PYTHONPATH=. python scripts/fast_ml_train_v2.py --market US --days 365
    PYTHONPATH=. python scripts/fast_ml_train_v2.py --market US --days 9999 --all-tiers
    
关键改进:
    - 从 stock_history 直接生成训练样本，不依赖 scan_results
    - 每只股票按采样间隔取 N 个时间点，每个时间点一个样本
    - 特征: 基础(88) + 高级(AdvancedFE) + Alpha158 + Caisen + 策略
    - 自动缓存到 ml_feature_cache.db
"""

import os, sys, time, sqlite3, json, gc, signal
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

CACHE_DB = str(V3_DIR / "db" / "ml_feature_cache.db")


def _init_cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_cache_v2 (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            market TEXT NOT NULL,
            price REAL,
            features_json TEXT NOT NULL,
            label_1d REAL, label_5d REAL, label_10d REAL,
            label_20d REAL, label_30d REAL, label_60d REAL,
            PRIMARY KEY (symbol, trade_date, market)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fcv2_mkt ON feature_cache_v2(market)")
    conn.commit()
    return conn


def _load_cache(market, min_date, max_date, price_tier='standard'):
    """从缓存加载"""
    conn = sqlite3.connect(CACHE_DB)
    try:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM feature_cache_v2 WHERE market=? AND trade_date BETWEEN ? AND ?",
            (market, min_date, max_date)
        ).fetchone()[0]
    except:
        conn.close()
        return None, None, None, None, 0

    if cnt < 1000000:  # 需要至少100万样本才用缓存
        conn.close()
        return None, None, None, None, cnt

    print(f"   📦 缓存命中: {cnt:,} 行")
    t0 = time.time()

    rows = conn.execute(
        """SELECT features_json, price, label_1d, label_5d, label_10d, label_20d, label_30d, label_60d, trade_date
        FROM feature_cache_v2 WHERE market=? AND trade_date BETWEEN ? AND ?""",
        (market, min_date, max_date)
    ).fetchall()
    conn.close()

    tiers = {'standard': (5, 9999), 'penny': (0.01, 5)}
    lo, hi = tiers.get(price_tier, (0, 99999))

    all_feats, all_rets, all_groups = [], {f'{d}d': [] for d in [1,5,10,20,30,60]}, []
    for row in rows:
        price = row[1] or 0
        if not (lo <= price < hi):
            continue
        feat = json.loads(row[0])
        all_feats.append(feat)
        all_rets['1d'].append(row[2] if row[2] is not None else np.nan)
        all_rets['5d'].append(row[3] if row[3] is not None else np.nan)
        all_rets['10d'].append(row[4] if row[4] is not None else np.nan)
        all_rets['20d'].append(row[5] if row[5] is not None else np.nan)
        all_rets['30d'].append(row[6] if row[6] is not None else np.nan)
        all_rets['60d'].append(row[7] if row[7] is not None else np.nan)
        all_groups.append(pd.to_datetime(row[8]).toordinal())

    if not all_feats:
        return None, None, None, None, 0

    feat_df = pd.DataFrame(all_feats)
    numeric_cols = [c for c in feat_df.select_dtypes(include=[np.number]).columns if c != 'Date']
    X = feat_df[numeric_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    print(f"   ✅ 从缓存加载 {len(X):,} 样本, {len(numeric_cols)} 特征 ({time.time()-t0:.1f}s)")
    return X, {k: np.array(v) for k, v in all_rets.items()}, np.array(all_groups), list(numeric_cols), len(X)


def _compute_benchmark_features(market, start_date, end_date, hist_conn):
    """
    计算大盘/基准特征 (SPY for US, HS300 for CN).
    Returns {date_str: {spy_close, spy_rsi14, spy_ret5, ...}}
    """
    result = {}

    if market == 'CN':
        # CN: 用 cn_index_data.json (HS300)
        idx_path = str(V3_DIR / 'db' / 'cn_index_data.json')
        if not os.path.exists(idx_path):
            return result
        with open(idx_path) as f:
            idx_data = json.load(f)
        for date_str, vals in idx_data.items():
            if start_date <= date_str <= end_date:
                result[date_str] = {
                    'spy_close': vals.get('hs300_close', 0),
                    'spy_rsi14': vals.get('hs300_rsi14', 50),
                    'spy_ret5': vals.get('hs300_ret5', 0),
                    'spy_ret20': vals.get('hs300_ret20', 0),
                    'spy_vol10': vals.get('hs300_vol10', 0),
                    'spy_above_ma20': vals.get('hs300_above_ma20', 0),
                    'spy_ma20_dist': vals.get('hs300_ma20_dist', 0),
                }
        return result

    # US: 用 SPY (从 yfinance 或 stock_history 获取)
    spy_df = None

    # Try stock_history first (SPY, then _MKT_AVG fallback)
    try:
        for bm_sym in ['SPY', '_MKT_AVG']:
            spy_df = pd.read_sql_query(
                f"""SELECT trade_date as Date, close as Close, volume as Volume
                   FROM stock_history WHERE symbol='{bm_sym}' AND market='US'
                   ORDER BY trade_date""",
                hist_conn
            )
            if len(spy_df) >= 60:
                print(f"   📊 US benchmark: {bm_sym} ({len(spy_df)} rows)")
                break
            spy_df = None
    except:
        spy_df = None

    # Fallback: yfinance
    if spy_df is None or spy_df.empty:
        try:
            import yfinance as yf
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            if not spy.empty:
                spy_df = spy[['Close', 'Volume']].reset_index()
                spy_df.columns = ['Date', 'Close', 'Volume']
                spy_df['Date'] = spy_df['Date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"   ⚠️ yfinance SPY: {e}")
            return result

    if spy_df is None or spy_df.empty:
        return result

    spy_df['Date'] = pd.to_datetime(spy_df['Date'])
    spy_df = spy_df.sort_values('Date').reset_index(drop=True)
    c = spy_df['Close'].astype(float)

    # Compute indicators
    ma20 = c.rolling(20).mean()
    rsi_delta = c.diff()
    gain = rsi_delta.clip(lower=0).rolling(14).mean()
    loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
    rsi14 = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ret5 = c.pct_change(5) * 100
    ret20 = c.pct_change(20) * 100
    vol10 = c.pct_change().rolling(10).std() * 100
    above_ma20 = (c > ma20).astype(float)
    ma20_dist = ((c - ma20) / ma20 * 100)

    for i in range(len(spy_df)):
        d = spy_df.iloc[i]['Date']
        ds = d.strftime('%Y-%m-%d')
        if ds < start_date or ds > end_date:
            continue
        result[ds] = {
            'spy_close': float(c.iloc[i]) if not pd.isna(c.iloc[i]) else 0,
            'spy_rsi14': float(rsi14.iloc[i]) if not pd.isna(rsi14.iloc[i]) else 50,
            'spy_ret5': float(ret5.iloc[i]) if not pd.isna(ret5.iloc[i]) else 0,
            'spy_ret20': float(ret20.iloc[i]) if not pd.isna(ret20.iloc[i]) else 0,
            'spy_vol10': float(vol10.iloc[i]) if not pd.isna(vol10.iloc[i]) else 0,
            'spy_above_ma20': float(above_ma20.iloc[i]) if not pd.isna(above_ma20.iloc[i]) else 0,
            'spy_ma20_dist': float(ma20_dist.iloc[i]) if not pd.isna(ma20_dist.iloc[i]) else 0,
        }

    # Cross-market ETF features (US only)
    cross_map = {
        'nasdaq': 'QQQ', 'gold': 'GLD', 'oil': 'USO',
        'bonds': 'TLT', 'smallcap': 'IWM',
    }
    for label, sym in cross_map.items():
        try:
            xdf = pd.read_sql_query(
                f"SELECT trade_date as Date, close as Close FROM stock_history "
                f"WHERE symbol='{sym}' AND market='US' ORDER BY trade_date",
                hist_conn)
            if len(xdf) < 60:
                continue
            xdf['Date'] = pd.to_datetime(xdf['Date'])
            xdf = xdf.sort_values('Date').reset_index(drop=True)
            xc = xdf['Close'].astype(float)
            xma20 = xc.rolling(20).mean()
            xret5 = xc.pct_change(5) * 100
            xret20 = xc.pct_change(20) * 100
            xma20_dist = ((xc - xma20) / xma20 * 100)
            for j in range(len(xdf)):
                ds = xdf.iloc[j]['Date'].strftime('%Y-%m-%d')
                if ds not in result:
                    continue
                result[ds][f'{label}_close'] = float(xc.iloc[j]) if not pd.isna(xc.iloc[j]) else 0
                result[ds][f'{label}_ret5'] = float(xret5.iloc[j]) if not pd.isna(xret5.iloc[j]) else 0
                result[ds][f'{label}_ret20'] = float(xret20.iloc[j]) if not pd.isna(xret20.iloc[j]) else 0
                result[ds][f'{label}_ma20_dist'] = float(xma20_dist.iloc[j]) if not pd.isna(xma20_dist.iloc[j]) else 0
            print(f"   ✅ {label}({sym}): {len(xdf)} days", flush=True)
        except Exception as e:
            print(f"   ⚠️ {label}({sym}): {e}", flush=True)

    return result

def prepare_dataset(market='US', days_back=365, price_tier='standard',
                    max_samples=2000000, sample_interval=5, use_cache=True):
    """
    直接从 stock_history 生成训练样本
    
    每只股票: 取最近 days_back 天的数据, 每隔 sample_interval 天取一个样本点
    """
    from db.stock_history import get_history_db_path
    from ml.features.feature_calculator import FeatureCalculator

    t0 = time.time()

    # 超时处理
    def _timeout_handler(signum, frame):
        raise TimeoutError("Stock feature computation timed out")
    signal.signal(signal.SIGALRM, _timeout_handler)

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

    # 0. 缓存
    if use_cache:
        print(f"📦 检查缓存 ({cutoff} ~ {end_date}, {price_tier})...")
        _init_cache()
        X, rets, groups, fnames, n = _load_cache(market, cutoff, end_date, price_tier)
        if X is not None and n >= 50000:
            return X, rets, groups, fnames, None
        elif n > 0:
            print(f"   缓存有 {n:,} 行但不足，从头计算...")

    # 1. 加载所有股票列表
    hist_db = get_history_db_path()
    conn = sqlite3.connect(hist_db)

    print(f"\n📥 加载 stock_history ({market})...")
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM stock_history WHERE market=?", (market,)
    ).fetchall()]
    print(f"   {len(symbols)} 只股票")

    # 价格过滤: 取最新价格判断分层
    tiers = {'standard': (5, 9999), 'penny': (0.01, 5)}
    lo, hi = tiers.get(price_tier, (0, 99999))

    # 2. 逐批加载 + 计算特征
    print(f"\n🧮 计算特征 + 标签 (interval={sample_interval}d)...")

    calculator = FeatureCalculator()

    # 初始化高级特征
    adv_eng, alpha158, has_caisen, has_strategy = None, None, False, False
    try:
        from ml.advanced_features import AdvancedFeatureEngineer
        adv_eng = AdvancedFeatureEngineer()
        print("   ✅ AdvancedFeatureEngineer")
    except Exception as e:
        print(f"   ⚠️ AdvancedFE 跳过: {e}")
    try:
        from ml.alpha_factors import Alpha158Factors
        alpha158 = Alpha158Factors()
        print("   ✅ Alpha158")
    except Exception as e:
        print(f"   ⚠️ Alpha158 跳过: {e}")
    try:
        from ml.caisen_features import compute_caisen_features
        has_caisen = True
        print("   ✅ Caisen")
    except:
        pass
    try:
        from strategies.auto_backtester import generate_strategy_features
        has_strategy = True
        print("   ✅ Strategy")
    except:
        pass

    # ---- 大盘/基准特征 (Benchmark Features) ----
    benchmark_features = {}  # {date_str: {spy_close, spy_rsi14, ...}}
    try:
        benchmark_features = _compute_benchmark_features(market, cutoff, end_date, conn)
        if benchmark_features:
            print(f"   ✅ 大盘特征: {len(benchmark_features)} 天 ({list(list(benchmark_features.values())[0].keys())[:3]}...)")
        else:
            print("   ⚠️ 大盘特征为空")
    except Exception as e:
        print(f"   ⚠️ 大盘特征失败: {e}")

    all_features = []
    all_returns = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
    all_groups = []
    all_cache_rows = []  # (symbol, date, price, feat_json, labels...)
    skipped = 0
    processed = 0
    batch_size = 200


    for batch_start in range(0, len(symbols), batch_size):
        batch_syms = symbols[batch_start:batch_start+batch_size]
        placeholders = ','.join(['?'] * len(batch_syms))

        # 批量读取
        df = pd.read_sql_query(
            f"""SELECT symbol, trade_date, open, high, low, close, volume
            FROM stock_history WHERE market=? AND symbol IN ({placeholders})
            ORDER BY symbol, trade_date""",
            conn, params=[market] + batch_syms
        )

        for symbol, sdf in df.groupby('symbol'):
            sdf = sdf.copy()
            sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])
            sdf = sdf.sort_values('trade_date').reset_index(drop=True)

            if len(sdf) < 120:  # 至少 120 天数据
                skipped += 1
                continue

            # 价格过滤
            latest_price = sdf['close'].iloc[-1]
            if not (lo <= latest_price < hi):
                skipped += 1
                continue

            # 限制历史长度，避免超大股票卡死（最多2000天）
            if len(sdf) > 2000:
                sdf = sdf.tail(2000).reset_index(drop=True)

            # 准备 OHLCV
            calc_df = sdf.rename(columns={
                'trade_date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })

            # 基础特征（带超时保护）
            try:
                signal.alarm(60)  # 60 秒超时
                feat_df = calculator.calculate_all(calc_df)
                signal.alarm(0)
            except:
                signal.alarm(0)
                skipped += 1
                continue
            if feat_df.empty or len(feat_df) < 120:
                skipped += 1
                continue

            # 高级特征 (每只股票算一次，带超时)
            try:
                signal.alarm(30)
                adv_df = _calc_adv(adv_eng, calc_df)
                a158_df = _calc_alpha158(alpha158, calc_df)
                caisen_df = _calc_caisen(has_caisen, calc_df)
                signal.alarm(0)
            except:
                signal.alarm(0)
                adv_df, a158_df, caisen_df = None, None, None

            # 确定采样日期范围
            cutoff_dt = pd.to_datetime(cutoff)
            end_dt = pd.to_datetime(end_date)
            valid_dates = feat_df[(feat_df['Date'] >= cutoff_dt) & (feat_df['Date'] <= end_dt)]

            if len(valid_dates) == 0:
                continue

            # 按间隔采样
            sample_indices = list(range(0, len(valid_dates), sample_interval))

            skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'open', 'high', 'low', 'close', 'volume', 'Symbol', 'date_diff'}

            for si in sample_indices:
                row_idx = valid_dates.index[si]
                feat_row = feat_df.loc[row_idx]
                sample_date = feat_row['Date']

                # 基础特征
                feat_dict = {}
                for col in feat_df.columns:
                    if col not in skip_cols:
                        v = feat_row.get(col, np.nan)
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            feat_dict[col] = float(v)

                # 高级特征
                _merge_adv(feat_dict, adv_df, sample_date, skip_cols)
                _merge_alpha(feat_dict, a158_df, sample_date, skip_cols)
                _merge_caisen(feat_dict, caisen_df, sample_date)

                # 大盘特征
                date_key = sample_date.strftime('%Y-%m-%d')
                bm = benchmark_features.get(date_key, {})
                for bk, bv in bm.items():
                    feat_dict[bk] = bv
                # Relative Strength vs Benchmark
                stock_ret5 = feat_dict.get('ret_5d', feat_dict.get('pct_5d', 0))
                stock_ret20 = feat_dict.get('ret_20d', feat_dict.get('pct_20d', 0))
                feat_dict['rs_vs_spy_5d'] = stock_ret5 - bm.get('spy_ret5', 0)
                feat_dict['rs_vs_spy_20d'] = stock_ret20 - bm.get('spy_ret20', 0)

                # 标签: 未来收益
                entry_idx = row_idx + 1
                if entry_idx >= len(feat_df):
                    continue
                entry_price = feat_df.loc[entry_idx, 'Open']
                if pd.isna(entry_price) or float(entry_price) <= 0:
                    continue

                labels = {}
                for days in [1, 5, 10, 20, 30, 60]:
                    future_idx = entry_idx + days
                    if future_idx < len(feat_df):
                        fp = feat_df.loc[future_idx, 'Close']
                        if pd.isna(fp) or float(fp) <= 0:
                            labels[f'{days}d'] = np.nan
                        else:
                            labels[f'{days}d'] = (fp - entry_price) / entry_price * 100
                    else:
                        labels[f'{days}d'] = np.nan

                all_features.append(feat_dict)
                for k in all_returns:
                    all_returns[k].append(labels.get(k, np.nan))
                all_groups.append(sample_date.toordinal())

                # 缓存行
                all_cache_rows.append((
                    symbol, sample_date.strftime('%Y-%m-%d'), market, float(latest_price),
                    json.dumps(feat_dict, default=str),
                    labels.get('1d'), labels.get('5d'), labels.get('10d'),
                    labels.get('20d'), labels.get('30d'), labels.get('60d'),
                ))
                processed += 1

                if max_samples and processed >= max_samples:
                    break
            if max_samples and processed >= max_samples:
                break

        # 进度 + 增量缓存
        done = min(batch_start + batch_size, len(symbols))
        n_feat = len(all_features[-1]) if all_features else 0
        elapsed = time.time() - t0
        print(f"   {done}/{len(symbols)} 股票, {processed:,} 样本, ~{n_feat} 特征 ({elapsed:.0f}s)")

        # 增量写缓存（每批写一次）
        if use_cache and all_cache_rows:
            cache_conn = _init_cache()
            cache_conn.executemany(
                "INSERT OR REPLACE INTO feature_cache_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                all_cache_rows
            )
            cache_conn.commit()
            cache_conn.close()
            all_cache_rows = []  # 清空已写入的

        if max_samples and processed >= max_samples:
            print(f"   ⚠️ 达到最大样本数 {max_samples:,}")
            break

    conn.close()
    print(f"   完成: {processed:,} 样本, {skipped} 跳过 ({time.time()-t0:.0f}s)")

    if not all_features:
        return None, None, None, None, None

    # 缓存
    if use_cache and all_cache_rows:
        print(f"\n💾 写入缓存...")
        cache_conn = _init_cache()
        for i in range(0, len(all_cache_rows), 5000):
            cache_conn.executemany(
                "INSERT OR REPLACE INTO feature_cache_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                all_cache_rows[i:i+5000]
            )
            cache_conn.commit()
        cache_conn.close()
        print(f"   💾 已缓存 {len(all_cache_rows):,} 行")

    # 构建矩阵
    feat_df = pd.DataFrame(all_features)
    numeric_cols = [c for c in feat_df.select_dtypes(include=[np.number]).columns if c != 'Date']
    X = feat_df[numeric_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    returns_dict = {k: np.array(v) for k, v in all_returns.items()}
    groups = np.array(all_groups)

    print(f"\n✅ 数据集: {len(X):,} 样本 × {len(numeric_cols)} 特征, {len(np.unique(groups))} 天")
    print(f"   总耗时: {time.time()-t0:.0f}s")

    return X, returns_dict, groups, list(numeric_cols), feat_df


# === 高级特征辅助函数 ===

def _calc_adv(adv_eng, df):
    if adv_eng is None:
        return None
    try:
        idx_df = df.set_index('Date')[['Open','High','Low','Close','Volume']]
        result = adv_eng.transform(idx_df).reset_index()
        result = result.rename(columns={result.columns[0]: 'Date'})
        return result
    except:
        return None

def _calc_alpha158(alpha158, df):
    if alpha158 is None:
        return None
    try:
        idx_df = df.set_index('Date')[['Open','High','Low','Close','Volume']]
        result = alpha158.compute(idx_df).reset_index()
        result = result.rename(columns={result.columns[0]: 'Date'})
        return result
    except:
        return None

def _calc_caisen(has_caisen, df):
    if not has_caisen:
        return None
    try:
        from ml.caisen_features import compute_caisen_features
        result = compute_caisen_features(df)
        if result is not None and not result.empty:
            if 'Date' not in result.columns:
                result = result.reset_index()
                if 'Date' not in result.columns:
                    result = result.rename(columns={result.columns[0]: 'Date'})
            return result
    except:
        pass
    return None

def _merge_adv(feat_dict, adv_df, date, skip_cols):
    if adv_df is None or 'Date' not in adv_df.columns:
        return
    try:
        eligible = adv_df[adv_df['Date'] <= date]
        if len(eligible) > 0:
            row = eligible.iloc[-1]
            for col in row.index:
                if col not in skip_cols and col not in feat_dict:
                    v = row[col]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        feat_dict[f'adv_{col}'] = float(v)
    except:
        pass

def _merge_alpha(feat_dict, alpha_df, date, skip_cols):
    if alpha_df is None or 'Date' not in alpha_df.columns:
        return
    try:
        eligible = alpha_df[alpha_df['Date'] <= date]
        if len(eligible) > 0:
            row = eligible.iloc[-1]
            for col in row.index:
                if col not in skip_cols and col not in feat_dict:
                    v = row[col]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        feat_dict[f'a158_{col}'] = float(v)
    except:
        pass

def _merge_caisen(feat_dict, caisen_df, date):
    if caisen_df is None or 'Date' not in caisen_df.columns:
        return
    try:
        eligible = caisen_df[caisen_df['Date'] <= date]
        if len(eligible) > 0:
            row = eligible.iloc[-1]
            for col in row.index:
                if col != 'Date' and col not in feat_dict:
                    v = row[col]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        feat_dict[col] = float(v)
    except:
        pass


# === 训练 ===

def train_models(X, returns_dict, groups, feature_names, market='US', price_tier='standard'):
    from ml.models.return_predictor import ReturnPredictor
    from ml.models.signal_ranker import SignalRanker

    suffix = '_penny' if price_tier == 'penny' else ''
    model_dir = V3_DIR / "ml" / "saved_models" / f"v2_{market.lower()}{suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)

    drawdowns_dict = {}
    for k in ['5d', '20d', '30d', '60d']:
        ret = returns_dict.get(k)
        if ret is not None:
            drawdowns_dict[k] = np.abs(np.minimum(np.nan_to_num(ret, nan=0.0), 0))
        else:
            drawdowns_dict[k] = np.zeros(len(X))

    print(f"\n{'='*60}")
    print(f"📈 训练 ReturnPredictor ({price_tier})...")
    rp = ReturnPredictor()
    rp_metrics = rp.train(X, returns_dict, feature_names, groups=groups)
    rp.save(str(model_dir))

    print(f"\n{'='*60}")
    print(f"🏆 训练 SignalRanker ({price_tier})...")
    sr = SignalRanker()
    sr_metrics = sr.train(X, returns_dict, drawdowns_dict, groups, feature_names)
    sr.save(str(model_dir))

    with open(model_dir / "feature_names.json", 'w') as f:
        json.dump(feature_names, f)

    meta = {
        'market': market, 'price_tier': price_tier,
        'samples': len(X), 'features': len(feature_names),
        'groups': int(len(np.unique(groups))),
        'trained_at': datetime.now().isoformat(),
    }
    with open(model_dir / "training_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n✅ 模型: {model_dir}")


def run(market='US', days_back=365, all_tiers=False, max_samples=2000000):
    tiers = ['standard', 'penny'] if all_tiers else ['standard']
    for tier in tiers:
        print(f"\n{'#'*60}")
        print(f"## {market} {tier.upper()} (days={days_back})")
        print(f"{'#'*60}")

        X, rets, groups, fnames, _ = prepare_dataset(
            market=market, days_back=days_back, price_tier=tier,
            max_samples=max_samples, sample_interval=5
        )
        if X is None:
            print("❌ 数据准备失败")
            continue

        train_models(X, rets, groups, fnames, market, tier)

    print(f"\n🎉 完成!")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--market', default='US', choices=['US', 'CN'])
    p.add_argument('--days', type=int, default=365)
    p.add_argument('--all-tiers', action='store_true')
    p.add_argument('--max-samples', type=int, default=2000000)
    args = p.parse_args()
    run(args.market, args.days, args.all_tiers, args.max_samples)
