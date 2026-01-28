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
    
    def __init__(self, market: str = 'US', days_back: int = 180):
        self.market = market
        self.days_back = days_back
        self.model_dir = Path(__file__).parent / "saved_models" / f"v2_{market.lower()}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
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
        from db.database import get_connection
        from db.stock_history import get_stock_history, save_stock_history
        from ml.features.feature_calculator import FeatureCalculator, FEATURE_COLUMNS
        
        print(f"\n📊 准备数据集...")
        
        # 1. 获取有信号的股票
        conn = get_connection()
        
        # 使用数据库中的实际日期范围
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(scan_date) FROM scan_results WHERE market = ?", (self.market,))
        max_date_row = cursor.fetchone()
        if max_date_row and max_date_row[0]:
            db_max_date = datetime.strptime(max_date_row[0], '%Y-%m-%d').date()
            end_date = db_max_date - timedelta(days=5)  # 留5天给标签计算
        else:
            end_date = date.today() - timedelta(days=5)
        
        start_date = end_date - timedelta(days=self.days_back)
        
        print(f"   查询范围: {start_date} ~ {end_date}")
        
        query = """
            SELECT DISTINCT symbol, scan_date, price, blue_daily, blue_weekly, blue_monthly, is_heima
            FROM scan_results
            WHERE market = ? AND scan_date >= ? AND scan_date <= ?
            ORDER BY scan_date, symbol
        """
        signals_df = pd.read_sql_query(query, conn, params=(
            self.market, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        ))
        conn.close()
        
        if signals_df.empty:
            print("❌ 无信号数据")
            return None, None, None, None, None, None
        
        print(f"   信号数: {len(signals_df)}")
        
        # 2. 为每个信号计算特征和标签
        calculator = FeatureCalculator()
        
        all_features = []
        all_returns = {f'{d}d': [] for d in [1, 5, 10, 30, 60]}
        all_drawdowns = {f'{d}d': [] for d in [5, 30, 60]}
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
                
                # 找到信号日期在历史数据中的位置
                date_mask = features_df['Date'] == signal_date
                if not date_mask.any():
                    # 尝试找最近的日期
                    features_df['date_diff'] = abs(features_df['Date'] - signal_date)
                    closest_idx = features_df['date_diff'].idxmin()
                    if features_df.loc[closest_idx, 'date_diff'].days > 3:
                        continue
                else:
                    closest_idx = features_df[date_mask].index[0]
                
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
                entry_price = feature_row['Close']
                signal_idx = closest_idx
                
                for days in [1, 5, 10, 30, 60]:
                    future_idx = signal_idx + days
                    if future_idx < len(features_df):
                        future_price = features_df.loc[future_idx, 'Close']
                        return_pct = (future_price - entry_price) / entry_price * 100
                        all_returns[f'{days}d'].append(return_pct)
                    else:
                        all_returns[f'{days}d'].append(np.nan)
                
                # 计算未来最大回撤
                for days in [5, 30, 60]:
                    future_end = min(signal_idx + days, len(features_df) - 1)
                    if future_end > signal_idx:
                        future_prices = features_df.loc[signal_idx:future_end, 'Close'].values
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
        
        returns_dict = {k: np.array(v) for k, v in all_returns.items()}
        drawdowns_dict = {k: np.array(v) for k, v in all_drawdowns.items()}
        groups = np.array(all_groups)
        
        info_df = pd.DataFrame(all_info)
        
        print(f"✅ 数据集准备完成:")
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {len(feature_names)}")
        print(f"   分组数: {len(np.unique(groups))}")
        
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
        return_metrics = return_predictor.train(X, returns_dict, feature_names)
        return_predictor.save(str(self.model_dir))
        results['return_predictor'] = return_metrics
        
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


def train_pipeline(market: str = 'US', days_back: int = 180, upload: bool = False):
    """便捷训练函数"""
    pipeline = MLPipeline(market=market, days_back=days_back)
    return pipeline.train_all(upload=upload)


# === 命令行入口 ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML 训练管道')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'])
    parser.add_argument('--days', type=int, default=180, help='数据天数')
    parser.add_argument('--upload', action='store_true', help='上传到 Hub')
    parser.add_argument('--fetch', action='store_true', help='先拉取历史数据')
    
    args = parser.parse_args()
    
    pipeline = MLPipeline(market=args.market, days_back=args.days)
    
    if args.fetch:
        # 获取信号股票列表
        from db.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol FROM scan_results 
            WHERE market = ? 
            ORDER BY symbol
        """, (args.market,))
        symbols = [row['symbol'] for row in cursor.fetchall()]
        conn.close()
        
        pipeline.fetch_and_store_history(symbols)
    
    results = pipeline.train_all(upload=args.upload)
    print(f"\n结果: {results}")
