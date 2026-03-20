#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical ML Models - 统计机器学习模型

支持 XGBoost, LightGBM, Random Forest 等模型的训练和预测
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# 尝试导入 ML 库 (可选依赖)
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class SignalClassifier:
    """
    信号分类器 - 预测 BLUE 信号是否会盈利
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        初始化分类器
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = {}
        self.metrics = {}
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
    
    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量 (0/1)
            test_size: 测试集比例
        
        Returns:
            训练指标字典
        """
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 创建并训练模型
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算指标
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        return self.metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        交叉验证
        """
        if self.model is None:
            self.model = self._create_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist()
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """获取特征重要性 DataFrame"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {'Feature': k, 'Importance': v}
            for k, v in sorted(self.feature_importance.items(), key=lambda x: -x[1])
        ])


def get_available_models() -> List[str]:
    """获取可用的模型列表"""
    models = []
    
    if SKLEARN_AVAILABLE:
        models.extend(['random_forest', 'gradient_boosting'])
    
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    
    if LIGHTGBM_AVAILABLE:
        models.append('lightgbm')
    
    return models


def check_ml_dependencies() -> Dict[str, bool]:
    """检查 ML 依赖是否安装"""
    return {
        'sklearn': SKLEARN_AVAILABLE,
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE
    }
