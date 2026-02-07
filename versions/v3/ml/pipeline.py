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

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MLPipeline:
    """ML 训练管道"""
    
    def __init__(self,
                 market: str = 'US',
                 days_back: int = 180,
                 commission_bps: float = 5.0,
                 slippage_bps: float = 10.0):
        self.market = market
        self.days_back = days_back
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        # 双边成本: 开仓 + 平仓
        self.round_trip_cost_pct = 2.0 * (self.commission_bps + self.slippage_bps) / 100.0
        self.model_dir = Path(__file__).parent / "saved_models" / f"v2_{market.lower()}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.label_cost_profile: Dict[str, Dict] = {}
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
                })
        
        signals_df = pd.DataFrame(all_signals)
        
        if signals_df.empty:
            print("❌ 无信号数据")
            return None, None, None, None, None, None
        
        print(f"   信号数: {len(signals_df)}")
        
        # 2. 为每个信号计算特征和标签
        calculator = FeatureCalculator()
        
        all_features = []
        all_returns_gross = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
        all_returns_net = {f'{d}d': [] for d in [1, 5, 10, 20, 30, 60]}
        all_drawdowns = {f'{d}d': [] for d in [5, 20, 30, 60]}
        all_groups = []
        all_info = []
        
        symbols = signals_df['symbol'].unique()
        print(f"   股票数: {len(symbols)}")
        
        # 限制股票数量 (避免 API 超时)
        max_symbols = 200
        if len(symbols) > max_symbols:
            # 选择信号最多的股票
            symbol_counts = signals_df['symbol'].value_counts()
            symbols = symbol_counts.head(max_symbols).index.tolist()
            print(f"   限制为 Top {max_symbols} 股票")
        
        # 按股票处理
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
            
            # 计算特征
            features_df = calculator.calculate_all(history)
            if features_df.empty:
                continue
            
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
        
        X = features_df[feature_names].values
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
        
        info_df = pd.DataFrame(all_info)
        
        print(f"✅ 数据集准备完成:")
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {len(feature_names)}")
        print(f"   分组数: {len(np.unique(groups))}")
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
        
        # 4. 保存特征名称
        import json
        with open(self.model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f)
        if self.label_cost_profile:
            with open(self.model_dir / "training_cost_profile.json", 'w') as f:
                json.dump(self.label_cost_profile, f, indent=2)
        with open(self.model_dir / "training_objective.json", 'w') as f:
            json.dump(self.objective_config, f, indent=2)
        
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
                   slippage_bps: float = 10.0):
    """便捷训练函数"""
    pipeline = MLPipeline(
        market=market,
        days_back=days_back,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps
    )
    return pipeline.train_all(upload=upload)


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
    
    args = parser.parse_args()
    
    # 初始化数据库
    from db.database import init_db, query_scan_results, get_scanned_dates
    try:
        init_db()
        print("✅ 历史数据库初始化完成")
    except Exception as e:
        print(f"⚠️ 数据库初始化: {e}")
    
    pipeline = MLPipeline(
        market=args.market,
        days_back=args.days,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps
    )
    
    if args.fetch:
        # 获取信号股票列表 (从 Supabase 或 SQLite)
        dates = get_scanned_dates(market=args.market)
        symbols = set()
        for d in dates[:30]:  # 最近30天
            results = query_scan_results(scan_date=d, market=args.market, limit=1000)
            for r in results:
                symbols.add(r.get('symbol', ''))
        symbols = sorted([s for s in symbols if s])
        print(f"   找到 {len(symbols)} 只股票")
        
        pipeline.fetch_and_store_history(symbols)
    
    results = pipeline.train_all(upload=args.upload)
    print(f"\n结果: {results}")
