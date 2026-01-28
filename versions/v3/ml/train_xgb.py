"""
XGBoost 信号预测模型训练
Train XGBoost Signal Predictor

功能:
- 从数据库加载历史信号数据
- 生成技术特征
- 训练 XGBoost 分类器预测信号盈利概率
- 保存模型到本地 / HuggingFace Hub

用法:
    python train_xgb.py --market US --days 180 --upload
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML 库
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("   运行: pip install xgboost scikit-learn")
    ML_AVAILABLE = False


class SignalDataset:
    """信号数据集生成器"""
    
    def __init__(self, market: str = 'US', holding_days: int = 5):
        self.market = market
        self.holding_days = holding_days
    
    def load_signals(self, days_back: int = 180) -> pd.DataFrame:
        """从数据库加载历史信号"""
        from db.database import get_connection
        
        conn = get_connection()
        
        end_date = date.today() - timedelta(days=self.holding_days)
        start_date = end_date - timedelta(days=days_back)
        
        query = """
            SELECT * FROM scan_results
            WHERE market = ? 
              AND scan_date >= ? 
              AND scan_date <= ?
            ORDER BY scan_date, symbol
        """
        
        df = pd.read_sql_query(query, conn, params=(
            self.market,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        ))
        
        conn.close()
        print(f"📊 加载 {len(df)} 条信号记录")
        return df
    
    def calculate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算标签 (N天后是否盈利)"""
        from db.database import get_connection
        
        if df.empty:
            return df
        
        conn = get_connection()
        results = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            signal_date = row['scan_date']
            entry_price = row.get('price', 0)
            
            if not entry_price or entry_price <= 0:
                continue
            
            # 查找 N 天后的价格
            exit_date = (pd.to_datetime(signal_date) + timedelta(days=self.holding_days)).strftime('%Y-%m-%d')
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT price FROM scan_results
                WHERE symbol = ? AND market = ? AND scan_date >= ?
                ORDER BY scan_date LIMIT 1
            """, (symbol, self.market, exit_date))
            
            exit_row = cursor.fetchone()
            
            if exit_row and exit_row['price']:
                exit_price = exit_row['price']
                return_pct = (exit_price - entry_price) / entry_price
                
                row_dict = row.to_dict()
                row_dict['exit_price'] = exit_price
                row_dict['return_pct'] = return_pct
                row_dict['is_win'] = 1 if return_pct > 0 else 0  # 二分类标签
                results.append(row_dict)
        
        conn.close()
        
        result_df = pd.DataFrame(results)
        print(f"📊 有效样本: {len(result_df)} (胜率: {result_df['is_win'].mean():.1%})")
        return result_df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """创建特征"""
        if df.empty:
            return pd.DataFrame(), []
        
        feature_cols = []
        
        # 1. BLUE 信号特征
        for col in ['blue_daily', 'blue_weekly', 'blue_monthly']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                feature_cols.append(col)
        
        # 2. 信号组合特征
        if 'blue_daily' in df.columns and 'blue_weekly' in df.columns:
            df['blue_daily_weekly_ratio'] = df['blue_daily'] / (df['blue_weekly'] + 1)
            df['blue_resonance'] = ((df['blue_daily'] >= 100) & (df['blue_weekly'] >= 100)).astype(int)
            feature_cols.extend(['blue_daily_weekly_ratio', 'blue_resonance'])
        
        # 3. 黑马/绝地信号
        for col in ['is_heima', 'is_juedi']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                feature_cols.append(col)
        
        # 4. 价格特征
        if 'price' in df.columns:
            df['log_price'] = np.log1p(df['price'])
            feature_cols.append('log_price')
        
        # 5. 星级评分
        if 'star_rating' in df.columns:
            df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce').fillna(0)
            feature_cols.append('star_rating')
        
        # 6. 时间特征
        df['scan_date'] = pd.to_datetime(df['scan_date'])
        df['day_of_week'] = df['scan_date'].dt.dayofweek
        df['month'] = df['scan_date'].dt.month
        feature_cols.extend(['day_of_week', 'month'])
        
        print(f"📊 特征数量: {len(feature_cols)}")
        return df, feature_cols
    
    def prepare_dataset(self, days_back: int = 180) -> Tuple[np.ndarray, np.ndarray, list, pd.DataFrame]:
        """
        准备完整数据集
        
        Returns:
            X: 特征矩阵
            y: 标签
            feature_names: 特征名称列表
            df: 原始数据
        """
        # 1. 加载数据
        df = self.load_signals(days_back)
        
        if df.empty:
            print("❌ 无数据")
            return None, None, None, None
        
        # 2. 计算标签
        df = self.calculate_labels(df)
        
        if df.empty or 'is_win' not in df.columns:
            print("❌ 无法计算标签")
            return None, None, None, None
        
        # 3. 创建特征
        df, feature_cols = self.create_features(df)
        
        # 4. 提取 X, y
        X = df[feature_cols].values
        y = df['is_win'].values
        
        # 处理 NaN
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y, feature_cols, df


class XGBSignalPredictor:
    """XGBoost 信号预测器"""
    
    def __init__(self, **params):
        self.params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': 1,  # 会在训练时动态计算
            'random_state': 42,
            **params
        }
        self.model = None
        self.feature_names = None
        self.metrics = {}
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: list = None,
              test_size: float = 0.2) -> Dict:
        """
        训练模型
        
        Returns:
            训练指标
        """
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]
        
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n🚀 开始训练...")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        print(f"   正样本比例: {y_train.mean():.1%}")
        
        # 处理不平衡数据：计算 scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        self.params['scale_pos_weight'] = scale_pos_weight
        print(f"   不平衡权重: {scale_pos_weight:.1f}")
        
        # 训练
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 评估
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_ratio': float(y_train.mean())
        }
        
        print(f"\n📈 模型性能:")
        print(f"   Accuracy:  {self.metrics['accuracy']:.3f}")
        print(f"   Precision: {self.metrics['precision']:.3f}")
        print(f"   Recall:    {self.metrics['recall']:.3f}")
        print(f"   F1 Score:  {self.metrics['f1']:.3f}")
        print(f"   AUC:       {self.metrics['auc']:.3f}")
        
        # 特征重要性
        print(f"\n🔍 特征重要性 (Top 10):")
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp[:10]:
            print(f"   {feat}: {imp:.3f}")
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.model is None or self.feature_names is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))


def train_and_save(market: str = 'US', 
                   days_back: int = 180,
                   holding_days: int = 5,
                   upload: bool = False) -> Optional[XGBSignalPredictor]:
    """
    完整训练流程
    
    Args:
        market: 市场 ('US' or 'CN')
        days_back: 训练数据天数
        holding_days: 持有天数
        upload: 是否上传到 HuggingFace Hub
    
    Returns:
        训练好的模型
    """
    if not ML_AVAILABLE:
        print("❌ ML 依赖未安装")
        return None
    
    print(f"=" * 50)
    print(f"🎯 XGBoost 信号预测模型训练")
    print(f"   市场: {market}")
    print(f"   数据: 近 {days_back} 天")
    print(f"   持有期: {holding_days} 天")
    print(f"=" * 50)
    
    # 1. 准备数据
    dataset = SignalDataset(market=market, holding_days=holding_days)
    X, y, feature_names, df = dataset.prepare_dataset(days_back)
    
    if X is None:
        print("❌ 数据准备失败")
        return None
    
    # 2. 训练模型
    predictor = XGBSignalPredictor()
    metrics = predictor.train(X, y, feature_names)
    
    # 3. 保存模型
    from ml.model_registry import save_model
    
    model_name = f"xgb_signal_{market.lower()}"
    metadata = {
        'market': market,
        'days_back': days_back,
        'holding_days': holding_days,
        'feature_names': feature_names,
        **metrics
    }
    
    save_model(predictor.model, model_name, metadata, upload=upload)
    
    # 同时保存特征名称（推理时需要）
    predictor_path = Path(__file__).parent / "saved_models" / model_name
    import json
    with open(predictor_path / "feature_names.json", 'w') as f:
        json.dump(feature_names, f)
    
    print(f"\n✅ 训练完成!")
    return predictor


# === 命令行入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练 XGBoost 信号预测模型')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'],
                       help='市场 (US/CN)')
    parser.add_argument('--days', type=int, default=180,
                       help='训练数据天数')
    parser.add_argument('--holding', type=int, default=5,
                       help='持有天数')
    parser.add_argument('--upload', action='store_true',
                       help='上传到 HuggingFace Hub')
    
    args = parser.parse_args()
    
    train_and_save(
        market=args.market,
        days_back=args.days,
        holding_days=args.holding,
        upload=args.upload
    )
