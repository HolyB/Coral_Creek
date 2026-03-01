"""
ML 训练管道
ML Training Pipeline

完整流程:
1. 拉取历史 K 线数据
2. 计算技术特征
3. 计算标签 (未来收益/回撤)
4. 训练收益预测模型
5. 训练排序模型
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MLPipeline:
    """ML 训练管道"""
    
    # 价格分层阈值
    PRICE_TIERS = {
        'US': {'standard': (5.0, 9999), 'penny': (0.01, 5.0)},
        'CN': {'standard': (3.0, 9999), 'penny': (0.01, 3.0)},
    }

    def __init__(self,
                 market: str = 'US',
                 days_back: int = 180,
                 commission_bps: float = 5.0,
                 slippage_bps: float = 10.0,
                 use_fundamental_features: bool = True,
                 enable_fundamental_api: bool = False,
                 price_tier: str = 'standard'):
        """
        Args:
            price_tier: 'standard' (>=5), 'penny' (<5), or 'all' (不过滤)
        """
        self.market = market
        self.days_back = days_back
        self.price_tier = price_tier
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        # 双边成本: 开仓 + 平仓 (低价股滑点更大)
        if price_tier == 'penny':
            self.round_trip_cost_pct = 2.0 * (self.commission_bps + self.slippage_bps * 3) / 100.0
        else:
            self.round_trip_cost_pct = 2.0 * (self.commission_bps + self.slippage_bps) / 100.0
        # 基本面特征开关
        self.use_fundamental_features = bool(use_fundamental_features)
        self.enable_fundamental_api = bool(enable_fundamental_api)
        # 模型目录: standard → v2_us, penny → v2_us_penny
        tier_suffix = f"_{price_tier}" if price_tier != 'standard' else ''
        self.model_dir = Path(__file__).parent / "saved_models" / f"v2_{market.lower()}{tier_suffix}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.label_cost_profile: Dict[str, Dict] = {}
        self.feature_stability_report: Dict[str, Dict] = {}
        self.walk_forward_report: Dict[str, Dict] = {}
        # 统一目标口径: 中长线优先 + 超额收益 + 回撤惩罚
        self.objective_config = {
            "primary_horizons": ["20d", "60d"],
            "excess_baseline": "cross_sectional_median_by_scan_date",
            "risk_penalty_lambda": {
                "20d": 0.35,
                "60d": 0.45,
            },
        }

    @staticmethod
    def _compute_group_excess(returns: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        计算按 scan_date 分组的超额收益:
        excess = stock_return - group_median_return
        """
        excess = np.full_like(returns, np.nan, dtype=float)
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            vals = returns[mask]
            valid = ~np.isnan(vals)
            if valid.sum() == 0:
                continue
            median_val = np.nanmedian(vals)
            tmp = np.full(vals.shape, np.nan, dtype=float)
            tmp[valid] = vals[valid] - median_val
            excess[mask] = tmp
        return excess

    def _build_feature_stability_report(
        self,
        X_raw: np.ndarray,
        feature_names: List[str],
        groups: np.ndarray,
        returns_dict: Dict[str, np.ndarray],
    ) -> Dict:
        """生成特征稳定性报告（缺失率/漂移/IC）"""
        if X_raw is None or len(X_raw) == 0 or len(feature_names) == 0:
            return {}

        report_rows = []
        n = len(X_raw)
        split = int(n * 0.6)
        split = min(max(split, 1), n - 1) if n > 1 else 1

        y20 = returns_dict.get('20d', np.array([]))
        y60 = returns_dict.get('60d', np.array([]))

        for i, feat in enumerate(feature_names):
            col = X_raw[:, i]
            col = np.asarray(col, dtype=float)
            missing_rate = float(np.isnan(col).mean()) if len(col) else 1.0

            first = col[:split]
            last = col[split:]
            first_valid = first[~np.isnan(first)]
            last_valid = last[~np.isnan(last)]

            mean_first = float(np.mean(first_valid)) if len(first_valid) else 0.0
            mean_last = float(np.mean(last_valid)) if len(last_valid) else 0.0
            std_first = float(np.std(first_valid)) if len(first_valid) else 0.0
            std_last = float(np.std(last_valid)) if len(last_valid) else 0.0

            pooled_std = (std_first + std_last) / 2.0 if (std_first + std_last) > 0 else 1.0
            drift_score = abs(mean_last - mean_first) / pooled_std

            ic20 = np.nan
            if len(y20) == len(col):
                valid = ~np.isnan(col) & ~np.isnan(y20)
                if valid.sum() > 20:
                    ic20 = pd.Series(col[valid]).corr(pd.Series(y20[valid]), method='spearman')

            ic60 = np.nan
            if len(y60) == len(col):
                valid = ~np.isnan(col) & ~np.isnan(y60)
                if valid.sum() > 20:
                    ic60 = pd.Series(col[valid]).corr(pd.Series(y60[valid]), method='spearman')

            report_rows.append({
                'feature': feat,
                'missing_rate': float(missing_rate),
                'drift_score': float(drift_score),
                'ic_20d': float(ic20) if pd.notna(ic20) else np.nan,
                'ic_60d': float(ic60) if pd.notna(ic60) else np.nan,
            })

        df = pd.DataFrame(report_rows)
        if df.empty:
            return {}

        # 统一稳定性评分：低缺失、低漂移、高|IC|
        ic_abs = (df['ic_20d'].abs().fillna(0) + df['ic_60d'].abs().fillna(0)) / 2.0
        df['stability_score'] = (
            (1.0 - df['missing_rate'].clip(0, 1)) * 0.35
            + (1.0 / (1.0 + df['drift_score'].clip(lower=0))) * 0.30
            + (ic_abs.clip(0, 1)) * 0.35
        )

        top_stable = df.sort_values('stability_score', ascending=False).head(30)
        top_unstable = df.sort_values('stability_score', ascending=True).head(30)

        summary = {
            'feature_count': int(len(df)),
            'avg_missing_rate': float(df['missing_rate'].mean()),
            'avg_drift_score': float(df['drift_score'].mean()),
            'avg_abs_ic20': float(df['ic_20d'].abs().mean(skipna=True)),
            'avg_abs_ic60': float(df['ic_60d'].abs().mean(skipna=True)),
        }

        return {
            'summary': summary,
            'top_stable_features': top_stable.to_dict('records'),
            'top_unstable_features': top_unstable.to_dict('records'),
        }

    def _run_walk_forward_eval(
        self,
        X: np.ndarray,
        returns_dict: Dict[str, np.ndarray],
        groups: np.ndarray,
    ) -> Dict:
        """滚动训练-测试评估，验证中长线目标稳健性"""
        try:
            import xgboost as xgb
        except Exception:
            return {'status': 'skipped', 'reason': 'xgboost_not_available'}

        y = returns_dict.get('20d')
        if y is None or len(y) != len(X):
            return {'status': 'skipped', 'reason': 'missing_20d_target'}

        unique_groups = np.array(sorted(np.unique(groups)))
        if len(unique_groups) < 80:
            return {'status': 'skipped', 'reason': f'not_enough_groups:{len(unique_groups)}'}

        train_span = 50
        test_span = 10
        step = 10

        folds = []
        for start in range(0, len(unique_groups) - train_span - test_span + 1, step):
            tr_g = unique_groups[start:start + train_span]
            te_g = unique_groups[start + train_span:start + train_span + test_span]
            tr_mask = np.isin(groups, tr_g)
            te_mask = np.isin(groups, te_g)

            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te, y_te = X[te_mask], y[te_mask]
            g_te = groups[te_mask]

            valid_tr = ~np.isnan(y_tr)
            valid_te = ~np.isnan(y_te)
            X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
            X_te, y_te, g_te = X_te[valid_te], y_te[valid_te], g_te[valid_te]

            if len(X_tr) < 200 or len(X_te) < 50:
                continue

            model = xgb.XGBRegressor(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(X_tr, y_tr, verbose=False)
            pred = model.predict(X_te)

            ic = pd.Series(pred).corr(pd.Series(y_te), method='spearman')
            dir_acc = float(((pred > 0) == (y_te > 0)).mean())

            top_returns = []
            for g in np.unique(g_te):
                m = g_te == g
                if m.sum() < 5:
                    continue
                idx = np.argsort(-pred[m])
                k = max(1, int(len(idx) * 0.2))
                top_returns.append(float(np.mean(y_te[m][idx[:k]])))
            top_ret = float(np.mean(top_returns)) if top_returns else np.nan

            folds.append({
                'train_groups': int(len(tr_g)),
                'test_groups': int(len(te_g)),
                'spearman_ic': float(ic) if pd.notna(ic) else np.nan,
                'direction_acc': dir_acc,
                'top20_avg_return': top_ret,
            })

        if not folds:
            return {'status': 'skipped', 'reason': 'no_valid_folds'}

        df = pd.DataFrame(folds)
        return {
            'status': 'ok',
            'n_folds': int(len(df)),
            'avg_spearman_ic': float(df['spearman_ic'].mean(skipna=True)),
            'avg_direction_acc': float(df['direction_acc'].mean(skipna=True)),
            'avg_top20_return': float(df['top20_avg_return'].mean(skipna=True)),
            'folds': folds,
        }
    
    def fetch_and_store_history(self, symbols: List[str], 
                                 days: int = 365,
                                 batch_size: int = 50) -> int:
        """
        拉取并存储历史 K 线数据
        
        Args:
            symbols: 股票列表
            days: 拉取天数
            batch_size: 批量大小 (避免 API 限制)
        
        Returns:
            成功存储的股票数
        """
        from db.stock_history import save_stock_history
        from data_fetcher import get_stock_data
        
        print(f"\n📥 拉取 {len(symbols)} 只股票的历史数据...")
        
        success_count = 0
        
        for i, symbol in enumerate(symbols):
            try:
                # API 限流
                if i > 0 and i % batch_size == 0:
                    print(f"   进度: {i}/{len(symbols)}, 休息 5 秒...")
                    time.sleep(5)
                
                df = get_stock_data(symbol, market=self.market, days=days)
                
                if df is not None and len(df) > 60:
                    count = save_stock_history(symbol, self.market, df)
                    success_count += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"   ✓ {i+1}/{len(symbols)}: {symbol} ({len(df)} 天)")
                
            except Exception as e:
                print(f"   ✗ {symbol}: {e}")
                continue
        
        print(f"✅ 成功存储 {success_count}/{len(symbols)} 只股票")
        return success_count
    
    def prepare_dataset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, List[str], pd.DataFrame]:
        """
        准备完整数据集
        
        Returns:
            X: 特征矩阵
            returns_dict: 未来收益率字典
            drawdowns_dict: 未来最大回撤字典
            groups: 日期分组
            feature_names: 特征名称
            df: 原始数据
        """
        from db.database import query_scan_results, get_scanned_dates, init_db
        from db.stock_history import get_stock_history, save_stock_history
        from ml.features.feature_calculator import FeatureCalculator, FEATURE_COLUMNS
        from ml.features.fundamental_features import build_fundamental_features
        from ml.advanced_features import AdvancedFeatureEngineer
        from ml.alpha_factors import Alpha158Factors
        
        print(f"\n📊 准备数据集...")
        
        # 初始化数据库 (确保表存在)
        try:
            init_db()
        except:
            pass
        
        # 1. 获取有信号的股票 (自动选择 Supabase 或 SQLite)
        dates = get_scanned_dates(market=self.market)
        if not dates:
            print("❌ 无扫描日期数据")
            return None, None, None, None, None, None
        
        db_max_date = datetime.strptime(dates[0], '%Y-%m-%d').date()
        end_date = db_max_date - timedelta(days=5)  # 留5天给标签计算
        start_date = end_date - timedelta(days=self.days_back)
        
        print(f"   最新扫描: {dates[0]}, 查询范围: {start_date} ~ {end_date}")
        
        # 收集多天的扫描结果
        all_signals = []
        target_dates = [d for d in dates if start_date.strftime('%Y-%m-%d') <= d <= end_date.strftime('%Y-%m-%d')]
        print(f"   目标日期: {len(target_dates)} 天")
        
        for d in target_dates:
            results = query_scan_results(scan_date=d, market=self.market, limit=1000)
            for r in results:
                all_signals.append({
                    'symbol': r.get('symbol', ''),
                    'scan_date': d,
                    'price': float(r.get('price', 0) or 0),
                    'blue_daily': float(r.get('blue_daily', 0) or 0),
                    'blue_weekly': float(r.get('blue_weekly', 0) or 0),
                    'blue_monthly': float(r.get('blue_monthly', 0) or 0),
                    'is_heima': bool(r.get('is_heima', False) or r.get('heima_daily', False)),
                    # 基本面/分层字段（优先用扫描结果自带）
                    'market_cap': float(r.get('market_cap', 0) or 0),
                    'industry': str(r.get('industry', '') or ''),
                    'cap_category': str(r.get('cap_category', '') or ''),
                })
        
        signals_df = pd.DataFrame(all_signals)
        
        if signals_df.empty:
            print("❌ 无信号数据")
            return None, None, None, None, None, None
        
        print(f"   信号数: {len(signals_df)}")
        
        # 2. 为每个信号计算特征和标签
        calculator = FeatureCalculator()
        adv_engineer = AdvancedFeatureEngineer()
        alpha158 = Alpha158Factors()
        
        all_features = []
        all_returns_gross = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
        all_returns_net = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
        all_drawdowns = {f'{d}d': [] for d in [5, 20, 30, 60]}
        all_groups = []
        all_info = []
        
        # 按价格分层过滤
        if 'price' in signals_df.columns and self.price_tier != 'all':
            tiers = self.PRICE_TIERS.get(self.market, self.PRICE_TIERS['US'])
            lo, hi = tiers.get(self.price_tier, (0.01, 9999))
            before_filter = len(signals_df)
            signals_df = signals_df[(signals_df['price'] >= lo) & (signals_df['price'] < hi)]
            filtered_out = before_filter - len(signals_df)
            if filtered_out > 0:
                print(f"   价格分层 [{self.price_tier}] ${lo}~${hi}: 保留 {len(signals_df)}, 排除 {filtered_out}")

        symbols = signals_df['symbol'].unique()
        print(f"   股票数: {len(symbols)}")
        
        # 优先选有本地历史数据的股票，无限制；无历史的限制 API 调用量
        local_symbols = set()
        try:
            import sqlite3 as _sq3
            from db.stock_history import get_history_db_path
            _hconn = _sq3.connect(get_history_db_path())
            _hcur = _hconn.cursor()
            _ph = ",".join(["?"] * min(len(symbols), 5000))
            _hcur.execute(
                f"SELECT DISTINCT symbol FROM stock_history WHERE market = ? AND symbol IN ({_ph})",
                [self.market] + list(symbols)[:5000],
            )
            local_symbols = set(r[0] for r in _hcur.fetchall())
            _hconn.close()
        except Exception:
            pass

        # 有本地历史的全部用，没有的按信号频率取 top 200 (避免 API 超时)
        symbol_counts = signals_df['symbol'].value_counts()
        has_hist = [s for s in symbol_counts.index if s in local_symbols]
        no_hist = [s for s in symbol_counts.index if s not in local_symbols]
        max_api_symbols = 200
        symbols = has_hist + no_hist[:max_api_symbols]
        print(f"   有本地历史: {len(has_hist)}, 需 API: {min(len(no_hist), max_api_symbols)}, 总计: {len(symbols)}")
        
        # 按股票处理
        # 基本面外部 API 缓存（避免重复请求）
        fundamental_external_cache: Dict[str, Dict[str, float]] = {}
        processed = 0
        for i, symbol in enumerate(symbols):
            # 获取历史数据 (优先本地，否则 API)
            history = get_stock_history(symbol, self.market, days=250)
            
            # 如果本地没有，从 API 获取
            if history.empty or len(history) < 60:
                try:
                    from data_fetcher import get_stock_data
                    history = get_stock_data(symbol, market=self.market, days=250)
                    if history is not None and len(history) >= 60:
                        # 确保 Date 是列而不是 index
                        if history.index.name == 'Date':
                            history = history.reset_index()
                        if 'Date' not in history.columns and history.index.name:
                            history = history.reset_index()
                            history = history.rename(columns={history.columns[0]: 'Date'})
                        # 存储到本地
                        save_stock_history(symbol, self.market, history)
                except Exception as e:
                    continue
                
                # API 限流
                if (i + 1) % 5 == 0:
                    time.sleep(0.5)
            
            if history is None or history.empty or len(history) < 60:
                continue
            
            # 计算基础特征
            features_df = calculator.calculate_all(history)
            if features_df.empty:
                continue
            
            # 计算高级技术因子 (Phase 1: AdvancedFeatures)
            try:
                # 确保 history 有 OHLCV 列
                hist_for_adv = history.copy()
                if not isinstance(hist_for_adv.index, pd.DatetimeIndex):
                    if 'Date' in hist_for_adv.columns:
                        hist_for_adv = hist_for_adv.set_index('Date')
                    elif 'date' in hist_for_adv.columns:
                        hist_for_adv = hist_for_adv.set_index('date')
                    hist_for_adv.index = pd.to_datetime(hist_for_adv.index)
                
                # 标准化列名
                col_map = {}
                for c in hist_for_adv.columns:
                    cl = c.lower()
                    if cl == 'close': col_map[c] = 'Close'
                    elif cl == 'open': col_map[c] = 'Open'
                    elif cl == 'high': col_map[c] = 'High'
                    elif cl == 'low': col_map[c] = 'Low'
                    elif cl == 'volume': col_map[c] = 'Volume'
                if col_map:
                    hist_for_adv = hist_for_adv.rename(columns=col_map)
                
                if all(c in hist_for_adv.columns for c in ['Close', 'High', 'Low', 'Volume']):
                    adv_features_df = adv_engineer.transform(hist_for_adv)
                    # reset index 以匹配 features_df 的 Date 列
                    adv_features_df = adv_features_df.reset_index()
                    if 'Date' not in adv_features_df.columns and adv_features_df.columns[0] != 'Date':
                        adv_features_df = adv_features_df.rename(columns={adv_features_df.columns[0]: 'Date'})
                    
                    # Phase 2: Alpha158 因子
                    alpha_df = alpha158.compute(hist_for_adv)
                    alpha_df = alpha_df.reset_index()
                    if 'Date' not in alpha_df.columns and alpha_df.columns[0] != 'Date':
                        alpha_df = alpha_df.rename(columns={alpha_df.columns[0]: 'Date'})
                else:
                    adv_features_df = None
                    alpha_df = None
            except Exception:
                adv_features_df = None
                alpha_df = None
            
            # 获取该股票的信号
            symbol_signals = signals_df[signals_df['symbol'] == symbol]
            
            for _, signal in symbol_signals.iterrows():
                signal_date = pd.to_datetime(signal['scan_date'])
                
                # 防止未来泄漏: 仅使用 signal_date 当天或之前最近交易日
                eligible_idx = features_df.index[features_df['Date'] <= signal_date]
                if len(eligible_idx) == 0:
                    continue
                closest_idx = int(eligible_idx[-1])
                ref_date = features_df.loc[closest_idx, 'Date']
                if (signal_date - ref_date).days > 3:
                    continue
                
                # 提取特征
                feature_row = features_df.loc[closest_idx]
                
                # 添加 BLUE 信号特征
                feature_dict = {col: feature_row.get(col, np.nan) for col in features_df.columns 
                               if col not in ['Date', 'date_diff']}
                feature_dict['blue_daily'] = signal.get('blue_daily', 0)
                feature_dict['blue_weekly'] = signal.get('blue_weekly', 0)
                feature_dict['blue_monthly'] = signal.get('blue_monthly', 0)
                feature_dict['is_heima'] = signal.get('is_heima', 0)

                # 加入基本面特征（中文注释：这里是低频快照特征）
                if self.use_fundamental_features:
                    fund_feats = build_fundamental_features(
                        symbol=symbol,
                        market=self.market,
                        signal_row=signal.to_dict() if hasattr(signal, "to_dict") else dict(signal),
                        enable_external_api=self.enable_fundamental_api,
                        external_cache=fundamental_external_cache,
                    )
                    feature_dict.update(fund_feats)
                
                # 加入高级技术因子 (Phase 1)
                if adv_features_df is not None and 'Date' in adv_features_df.columns:
                    try:
                        adv_eligible = adv_features_df[adv_features_df['Date'] <= signal_date]
                        if len(adv_eligible) > 0:
                            adv_row = adv_eligible.iloc[-1]
                            # 只取数值列，排除 OHLCV 原始列和 Date
                            skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                                        'open', 'high', 'low', 'close', 'volume'}
                            for col in adv_row.index:
                                if col not in skip_cols and col not in feature_dict:
                                    val = adv_row[col]
                                    if isinstance(val, (int, float, np.integer, np.floating)):
                                        feature_dict[f'adv_{col}'] = float(val)
                    except Exception:
                        pass
                
                # 加入 Alpha158 因子 (Phase 2)
                if alpha_df is not None and 'Date' in alpha_df.columns:
                    try:
                        a_eligible = alpha_df[alpha_df['Date'] <= signal_date]
                        if len(a_eligible) > 0:
                            a_row = a_eligible.iloc[-1]
                            skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                            for col in a_row.index:
                                if col not in skip_cols and col not in feature_dict:
                                    val = a_row[col]
                                    if isinstance(val, (int, float, np.integer, np.floating)):
                                        feature_dict[f'a158_{col}'] = float(val)
                    except Exception:
                        pass
                
                # 加入蔡森图表特征 (Phase 3)
                try:
                    from ml.caisen_features import compute_caisen_features
                    cs_df = compute_caisen_features(history)
                    if cs_df is not None and not cs_df.empty and 'Date' not in cs_df.columns:
                        cs_df['Date'] = cs_df.index
                    if cs_df is not None and not cs_df.empty:
                        cs_eligible = cs_df[cs_df['Date'] <= signal_date] if 'Date' in cs_df.columns else cs_df[cs_df.index <= signal_date]
                        if len(cs_eligible) > 0:
                            cs_row = cs_eligible.iloc[-1]
                            for col in cs_row.index:
                                if col != 'Date' and col not in feature_dict:
                                    val = cs_row[col]
                                    if isinstance(val, (int, float, np.integer, np.floating)):
                                        feature_dict[col] = float(val)
                except Exception:
                    pass
                
                # 计算未来收益 (标签)
                # 入场口径: 信号后的下一交易日开盘价，更接近真实执行
                signal_idx = closest_idx
                entry_idx = signal_idx + 1
                if entry_idx >= len(features_df):
                    continue
                entry_price = features_df.loc[entry_idx, 'Open']
                if pd.isna(entry_price) or float(entry_price) <= 0:
                    continue
                
                for days in [1, 5, 10, 20, 30, 60]:
                    # 以入场日为 t0，持有 N 天后按收盘价离场
                    future_idx = entry_idx + days
                    if future_idx < len(features_df):
                        future_price = features_df.loc[future_idx, 'Close']
                        if pd.isna(future_price) or float(future_price) <= 0:
                            all_returns_gross[f'{days}d'].append(np.nan)
                            all_returns_net[f'{days}d'].append(np.nan)
                            continue
                        gross_return_pct = (future_price - entry_price) / entry_price * 100
                        net_return_pct = gross_return_pct - self.round_trip_cost_pct
                        all_returns_gross[f'{days}d'].append(gross_return_pct)
                        all_returns_net[f'{days}d'].append(net_return_pct)
                    else:
                        all_returns_gross[f'{days}d'].append(np.nan)
                        all_returns_net[f'{days}d'].append(np.nan)
                
                # 计算未来最大回撤
                for days in [5, 20, 30, 60]:
                    future_end = min(entry_idx + days, len(features_df) - 1)
                    if future_end > entry_idx:
                        future_prices = features_df.loc[entry_idx:future_end, 'Close'].values
                        future_prices = np.asarray(future_prices, dtype=float)
                        future_prices = future_prices[~np.isnan(future_prices)]
                        if len(future_prices) == 0:
                            all_drawdowns[f'{days}d'].append(np.nan)
                            continue
                        cummax = np.maximum.accumulate(future_prices)
                        drawdown = (cummax - future_prices) / cummax * 100
                        max_dd = np.max(drawdown)
                        all_drawdowns[f'{days}d'].append(max_dd)
                    else:
                        all_drawdowns[f'{days}d'].append(np.nan)
                
                all_features.append(feature_dict)
                all_groups.append(signal_date.toordinal())  # 同一天为一组
                all_info.append({
                    'symbol': symbol,
                    'scan_date': signal['scan_date'],
                    'price': entry_price
                })
        
        if not all_features:
            print("❌ 无有效特征")
            return None, None, None, None, None, None
        
        # 转换为数组
        features_df = pd.DataFrame(all_features)
        
        # 选择数值特征
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = [c for c in numeric_cols if c not in ['Date']]
        
        X_raw = features_df[feature_names].values.astype(float)
        X = X_raw.copy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 限制极端值
        X = np.clip(X, -1e6, 1e6)
        
        returns_dict = {k: np.array(v) for k, v in all_returns_net.items()}
        gross_returns_dict = {k: np.array(v) for k, v in all_returns_gross.items()}
        drawdowns_dict = {k: np.array(v) for k, v in all_drawdowns.items()}
        groups = np.array(all_groups)

        # 统一训练目标:
        # 1) 对 20d/60d 使用按日横截面超额收益
        # 2) 加入最大回撤惩罚，降低“高收益高波动”样本排名
        if len(groups) > 0:
            for horizon in ["20d", "60d"]:
                if horizon in returns_dict:
                    excess = self._compute_group_excess(returns_dict[horizon], groups)
                    dd = drawdowns_dict.get(horizon, np.full_like(excess, np.nan))
                    lam = float(self.objective_config["risk_penalty_lambda"].get(horizon, 0.0))
                    adj = excess - lam * np.nan_to_num(dd, nan=0.0)
                    returns_dict[horizon] = adj

        # 生成特征稳定性报告
        self.feature_stability_report = self._build_feature_stability_report(
            X_raw=X_raw,
            feature_names=feature_names,
            groups=groups,
            returns_dict=returns_dict,
        )
        
        info_df = pd.DataFrame(all_info)
        
        print(f"✅ 数据集准备完成:")
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {len(feature_names)}")
        print(f"   分组数: {len(np.unique(groups))}")
        print(
            "   基本面特征: {} (外部API:{})".format(
                "启用" if self.use_fundamental_features else "关闭",
                "启用" if self.enable_fundamental_api else "关闭",
            )
        )
        print(f"   训练标签: 净收益 (已扣双边成本 {self.round_trip_cost_pct:.2f}%)")

        # 保存毛/净收益对比画像，供 UI 展示
        profile = {
            'market': self.market,
            'commission_bps': self.commission_bps,
            'slippage_bps': self.slippage_bps,
            'round_trip_cost_pct': self.round_trip_cost_pct,
            'objective': self.objective_config,
            'horizons': {}
        }
        for k in gross_returns_dict.keys():
            gross = gross_returns_dict[k]
            net = returns_dict[k]
            valid_mask = ~np.isnan(gross) & ~np.isnan(net)
            if valid_mask.sum() == 0:
                continue
            gross_v = gross[valid_mask]
            net_v = net[valid_mask]
            profile['horizons'][k] = {
                'samples': int(valid_mask.sum()),
                'avg_gross_return_pct': float(np.mean(gross_v)),
                'avg_net_return_pct': float(np.mean(net_v)),
                'cost_drag_pct': float(np.mean(gross_v - net_v)),
                'gross_win_rate_pct': float((gross_v > 0).mean() * 100),
                'net_win_rate_pct': float((net_v > 0).mean() * 100),
            }
        self.label_cost_profile = profile
        
        return X, returns_dict, drawdowns_dict, groups, feature_names, info_df
    
    def train_all(self, upload: bool = False) -> Dict:
        """
        训练所有模型
        
        Args:
            upload: 是否上传到 HuggingFace Hub
        
        Returns:
            训练结果
        """
        from ml.models.return_predictor import ReturnPredictor
        from ml.models.signal_ranker import SignalRanker
        
        print(f"\n{'='*60}")
        print(f"🚀 Coral Creek ML 训练管道")
        print(f"   市场: {self.market}")
        print(f"   数据范围: 近 {self.days_back} 天")
        print(f"{'='*60}")
        
        # 1. 准备数据
        X, returns_dict, drawdowns_dict, groups, feature_names, info_df = self.prepare_dataset()
        
        if X is None:
            return {'status': 'failed', 'reason': '数据准备失败'}
        
        results = {'status': 'success', 'samples': len(X), 'features': len(feature_names)}
        
        # 2. 训练收益预测模型
        print("\n" + "="*60)
        return_predictor = ReturnPredictor()
        return_metrics = return_predictor.train(X, returns_dict, feature_names, groups=groups)
        return_predictor.save(str(self.model_dir))
        results['return_predictor'] = return_metrics
        results['label_cost_profile'] = self.label_cost_profile
        
        # 3. 训练排序模型
        print("\n" + "="*60)
        ranker = SignalRanker()
        ranker_metrics = ranker.train(X, returns_dict, drawdowns_dict, groups, feature_names)
        ranker.save(str(self.model_dir))
        results['signal_ranker'] = {h.value: m for h, m in ranker_metrics.items()}
        
        # 3.5 训练 LightGBM Ensemble (Phase 3)
        try:
            from ml.ensemble_predictor import LGBPredictor
            print("\n" + "="*60)
            print("🌳 LightGBM Ensemble 训练")
            lgb_pred = LGBPredictor(market=self.market)
            lgb_metrics = lgb_pred.train(X, returns_dict, feature_names, groups)
            lgb_pred.save()
            results['lgb_ensemble'] = lgb_metrics
        except Exception as e:
            print(f"⚠️ LGB 训练跳过: {e}")
        
        # 4. 保存特征名称
        import json
        with open(self.model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f)
        if self.label_cost_profile:
            with open(self.model_dir / "training_cost_profile.json", 'w') as f:
                json.dump(self.label_cost_profile, f, indent=2)
        with open(self.model_dir / "training_objective.json", 'w') as f:
            json.dump(self.objective_config, f, indent=2)
        if self.feature_stability_report:
            with open(self.model_dir / "feature_stability_report.json", 'w') as f:
                json.dump(self.feature_stability_report, f, indent=2)

        # Walk-forward 稳健性验证
        self.walk_forward_report = self._run_walk_forward_eval(X, returns_dict, groups)
        with open(self.model_dir / "walk_forward_report.json", 'w') as f:
            json.dump(self.walk_forward_report, f, indent=2)
        results['feature_stability'] = self.feature_stability_report
        results['walk_forward'] = self.walk_forward_report
        
        print(f"\n{'='*60}")
        print("✅ 训练完成!")
        print(f"   模型保存位置: {self.model_dir}")
        print(f"{'='*60}")
        
        # 5. 上传到 Hub (可选)
        if upload:
            try:
                from ml.model_registry import get_registry
                registry = get_registry()
                # TODO: 实现批量上传
                print("📤 上传功能待实现")
            except Exception as e:
                print(f"⚠️ 上传失败: {e}")
        
        return results


def train_pipeline(market: str = 'US',
                   days_back: int = 180,
                   upload: bool = False,
                   commission_bps: float = 5.0,
                   slippage_bps: float = 10.0,
                   price_tier: str = 'standard'):
    """便捷训练函数"""
    pipeline = MLPipeline(
        market=market,
        days_back=days_back,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        price_tier=price_tier,
    )
    return pipeline.train_all(upload=upload)


def train_all_tiers(market: str = 'US', days_back: int = 365, **kwargs):
    """训练所有价格分层模型"""
    all_results = {}
    for tier in ['standard', 'penny']:
        print(f"\n{'#'*60}")
        print(f"## 训练 {tier.upper()} 模型 ({market})")
        print(f"{'#'*60}")
        result = train_pipeline(market=market, days_back=days_back, price_tier=tier, **kwargs)
        all_results[tier] = result
    return all_results


# === 命令行入口 ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML 训练管道')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'])
    parser.add_argument('--days', type=int, default=180, help='数据天数')
    parser.add_argument('--upload', action='store_true', help='上传到 Hub')
    parser.add_argument('--fetch', action='store_true', help='先拉取历史数据')
    parser.add_argument('--commission-bps', type=float, default=5.0, help='单边手续费 (bps)')
    parser.add_argument('--slippage-bps', type=float, default=10.0, help='单边滑点 (bps)')
    parser.add_argument('--tier', type=str, default='standard', choices=['standard', 'penny', 'all'],
                        help='价格分层: standard(>=5), penny(<5), all(全部)')
    parser.add_argument('--all-tiers', action='store_true', help='训练所有分层模型')
    
    args = parser.parse_args()
    
    # 初始化数据库
    from db.database import init_db, query_scan_results, get_scanned_dates
    try:
        init_db()
        print("✅ 历史数据库初始化完成")
    except Exception as e:
        print(f"⚠️ 数据库初始化: {e}")
    
    if args.all_tiers:
        results = train_all_tiers(
            market=args.market, days_back=args.days,
            commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
        )
        for tier, r in results.items():
            print(f"\n{tier}: samples={r.get('samples')}, status={r.get('status')}")
    else:
        pipeline = MLPipeline(
            market=args.market,
            days_back=args.days,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            price_tier=args.tier,
        )
        
        if args.fetch:
            dates = get_scanned_dates(market=args.market)
            symbols = set()
            for d in dates[:30]:
                results = query_scan_results(scan_date=d, market=args.market, limit=1000)
                for r in results:
                    symbols.add(r.get('symbol', ''))
            symbols = sorted([s for s in symbols if s])
            print(f"   找到 {len(symbols)} 只股票")
            pipeline.fetch_and_store_history(symbols)
        
        results = pipeline.train_all(upload=args.upload)
        print(f"\n结果: {results}")
