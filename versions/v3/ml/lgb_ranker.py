#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 LightGBM 排序模型训练
============================

不依赖 Qlib 复杂 API，直接使用:
- Coral Creek 的 FeatureCalculator
- LightGBM Ranker

这样更稳定，且可以完全控制特征
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ 请安装 LightGBM: pip install lightgbm")


class LightGBMRanker:
    """
    使用 LightGBM 的 Learning to Rank 模型
    
    适用于股票排序任务
    """
    
    def __init__(self, market: str = 'US'):
        self.market = market
        self.model = None
        self.feature_names = []
        self.model_dir = project_root / "ml" / "saved_models" / f"lgb_{market.lower()}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, 
                     symbols: List[str],
                     days: int = 365,
                     forward_days: int = 5) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        准备训练数据
        
        Returns:
            X: 特征 DataFrame
            y: 目标变量 (未来收益)
            groups: LightGBM 分组 (按日期)
        """
        from data_fetcher import get_stock_data
        from ml.features.feature_calculator import FeatureCalculator
        
        all_rows = []
        
        print(f"准备数据: {len(symbols)} 只股票, {days} 天历史")
        
        for i, symbol in enumerate(symbols):
            try:
                # 获取数据
                df = get_stock_data(symbol, self.market, days=days + 60)
                if df is None or len(df) < 100:
                    print(f"  跳过 {symbol}: 数据不足")
                    continue
                
                # 计算特征
                calc = FeatureCalculator()
                features_df = calc.calculate_all(df, {'daily': 100})  # 简化 BLUE 传入
                
                if features_df.empty or len(features_df) < forward_days + 10:
                    print(f"  跳过 {symbol}: 特征计算失败")
                    continue
                
                # 计算目标: 未来 N 天收益
                close_series = df['Close'].values
                future_returns = []
                for j in range(len(close_series) - forward_days):
                    ret = (close_series[j + forward_days] / close_series[j] - 1) * 100
                    future_returns.append(ret)
                
                # 截取数据
                features_df = features_df.iloc[:len(future_returns)].copy()
                features_df['_target'] = future_returns
                features_df['_symbol'] = symbol
                features_df['_date_idx'] = range(len(features_df))
                
                # 填充 NaN (用 0 或前向填充)
                # 只删除目标为 NaN 的行
                features_df = features_df[features_df['_target'].notna()]
                features_df = features_df.fillna(0)
                
                if len(features_df) >= 30:  # 降低阈值
                    all_rows.append(features_df)
                    print(f"  ✓ {symbol}: {len(features_df)} 样本")
                
            except Exception as e:
                print(f"  跳过 {symbol}: {e}")
                continue
        
        if not all_rows:
            return pd.DataFrame(), pd.Series(), np.array([])
        
        # 合并
        combined = pd.concat(all_rows, ignore_index=True)
        
        # 提取目标和元数据
        y = combined['_target']
        dates = combined['_date_idx'].values
        
        # 移除辅助列
        X = combined.drop(columns=['_target', '_symbol', '_date_idx'], errors='ignore')
        
        # 保存特征名
        self.feature_names = list(X.columns)
        
        # 计算分组 (按日期索引)
        # 对于排序任务，同一天的所有股票应该在一个组
        unique_dates = sorted(set(dates))
        date_to_group = {d: i for i, d in enumerate(unique_dates)}
        groups_array = np.array([date_to_group[d] for d in dates])
        
        # 计算每组的大小
        from collections import Counter
        group_counts = Counter(groups_array)
        group_sizes = [group_counts[i] for i in range(len(unique_dates))]
        
        print(f"\n数据准备完成: {len(X)} 样本, {len(self.feature_names)} 特征, {len(unique_dates)} 个时间点")
        
        return X, y, np.array(group_sizes)
    
    def train(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray):
        """训练 LightGBM Ranker"""
        if not LGB_AVAILABLE:
            print("❌ LightGBM 未安装")
            return None
        
        # 数据分割 (时序)
        n = len(X)
        train_size = int(n * 0.8)
        
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        # 创建数据集 (回归任务，不需要 group)
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 模型参数 (回归任务，预测收益率)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
        
        print("开始训练 LightGBM 回归模型...")
        
        # 训练
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        print("✅ 训练完成!")
        
        # 特征重要性
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = sorted(zip(self.feature_names, importance), key=lambda x: -x[1])[:10]
        print("\nTop 10 重要特征:")
        for name, imp in feature_imp:
            print(f"  {name}: {imp:.2f}")
        
        return self.model
    
    def save(self, path: Optional[str] = None):
        """保存模型"""
        if self.model is None:
            print("❌ 没有可保存的模型")
            return
        
        if path is None:
            path = self.model_dir
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_model(str(path / "ranker.txt"))
        
        # 保存特征名
        with open(path / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # 保存元数据
        metadata = {
            'market': self.market,
            'feature_count': len(self.feature_names),
            'created_at': datetime.now().isoformat(),
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 模型已保存到: {path}")
    
    def load(self, path: Optional[str] = None):
        """加载模型"""
        if path is None:
            path = self.model_dir
        else:
            path = Path(path)
        
        model_file = path / "ranker.txt"
        if not model_file.exists():
            print(f"❌ 模型文件不存在: {model_file}")
            return False
        
        self.model = lgb.Booster(model_file=str(model_file))
        
        feature_file = path / "feature_names.json"
        if feature_file.exists():
            with open(feature_file) as f:
                self.feature_names = json.load(f)
        
        print(f"✅ 模型已加载: {path}")
        return True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测排序分数"""
        if self.model is None:
            print("❌ 模型未加载")
            return np.zeros(len(X))
        
        # 确保特征对齐
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                for col in missing:
                    X[col] = 0
            X = X[self.feature_names]
        
        return self.model.predict(X)


def train_and_save(market: str = 'US', 
                   symbols: Optional[List[str]] = None,
                   days: int = 365):
    """训练并保存模型"""
    
    # 默认股票池
    if symbols is None:
        if market == 'US':
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
                'AMD', 'INTC', 'AVGO', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'QCOM'
            ]
        else:
            symbols = [
                '600000', '600036', '600519', '601318', '000001', '000002',
                '000333', '000651', '000858', '002415'
            ]
    
    print(f"\n{'='*60}")
    print(f"训练 LightGBM Ranker ({market})")
    print(f"{'='*60}")
    print(f"股票池: {len(symbols)} 只")
    print(f"历史天数: {days}")
    print(f"{'='*60}\n")
    
    ranker = LightGBMRanker(market=market)
    
    # 准备数据
    X, y, groups = ranker.prepare_data(symbols, days=days)
    
    if X.empty:
        print("❌ 数据准备失败")
        return None
    
    # 训练
    ranker.train(X, y, groups)
    
    # 保存
    ranker.save()
    
    return ranker


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练 LightGBM Ranker')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--days', type=int, default=365)
    
    args = parser.parse_args()
    
    train_and_save(market=args.market, days=args.days)
