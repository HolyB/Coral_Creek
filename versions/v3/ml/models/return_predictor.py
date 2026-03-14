"""
收益预测模型
Return Predictor

预测 1/5/10/30/60 天的收益率
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class ReturnPredictor:
    """多周期收益预测器"""
    
    # 统一目标: 以中长线为主（5/10/20/60）
    HORIZONS = {
        '5d': 5,    # 短线辅助
        '10d': 10,  # 中短线核心（回测最优）
        '20d': 20,  # 中线核心
        '60d': 60,  # 长线核心
    }
    
    # 默认参数
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    def __init__(self, custom_params: Dict[str, Dict] = None):
        """
        Args:
            custom_params: 自定义参数 {'1d': {...}, '5d': {...}}
        """
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Dict] = {}
        self.is_trained = False
        self.custom_params = custom_params or {}
    
    def _get_params(self, horizon: str) -> Dict:
        """获取指定周期的模型参数"""
        # 优先使用自定义参数
        if horizon in self.custom_params:
            params = self.DEFAULT_PARAMS.copy()
            params.update(self.custom_params[horizon])
            return params
        
        # 尝试从调优结果加载
        tuning_path = Path(__file__).parent.parent / 'tuning_results'
        for market in ['us', 'cn']:
            params_file = tuning_path / market / 'best_params.json'
            if params_file.exists():
                try:
                    with open(params_file) as f:
                        best_params = json.load(f)
                    key = f'regressor_{horizon}'
                    if key in best_params:
                        params = self.DEFAULT_PARAMS.copy()
                        params.update(best_params[key])
                        return params
                except:
                    pass
        
        return self.DEFAULT_PARAMS.copy()
    
    def train(self, 
              X: np.ndarray, 
              y_dict: Dict[str, np.ndarray],
              feature_names: List[str],
              groups: Optional[np.ndarray] = None,
              test_size: float = 0.2) -> Dict:
        """
        训练所有周期的模型
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y_dict: 标签字典 {'1d': array, '5d': array, ...}
            feature_names: 特征名称列表
            test_size: 测试集比例
        
        Returns:
            训练指标
        """
        if not ML_AVAILABLE:
            raise RuntimeError("XGBoost 未安装")
        
        self.feature_names = feature_names
        
        print(f"\n{'='*50}")
        print("🎯 收益预测模型训练")
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {len(feature_names)}")
        print(f"{'='*50}\n")
        
        for horizon_name, horizon_days in self.HORIZONS.items():
            if horizon_name not in y_dict:
                print(f"⚠️ 跳过 {horizon_name}: 无标签")
                continue
            
            y = y_dict[horizon_name]
            
            # 过滤无效样本
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            if len(X_valid) < 100:
                print(f"⚠️ 跳过 {horizon_name}: 样本不足 ({len(X_valid)})")
                continue
            
            # 划分训练/测试集 (优先使用时序切分，避免未来信息泄漏)
            if groups is not None:
                groups_valid = groups[valid_mask]
                unique_groups = np.unique(groups_valid)
                unique_groups = np.sort(unique_groups)

                if len(unique_groups) >= 10:
                    split_idx = max(1, int(len(unique_groups) * (1 - test_size)))
                    split_idx = min(split_idx, len(unique_groups) - 1)
                    train_groups = unique_groups[:split_idx]
                    test_groups = unique_groups[split_idx:]

                    train_mask = np.isin(groups_valid, train_groups)
                    test_mask = np.isin(groups_valid, test_groups)

                    X_train, X_test = X_valid[train_mask], X_valid[test_mask]
                    y_train, y_test = y_valid[train_mask], y_valid[test_mask]
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_valid, y_valid, test_size=test_size, random_state=42
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_valid, y_valid, test_size=test_size, random_state=42
                )
            
            # 获取参数 (优先用调优后的)
            params = self._get_params(horizon_name)
            
            print(f"📈 训练 {horizon_name} 模型...")
            print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")
            print(f"   参数: max_depth={params.get('max_depth')}, lr={params.get('learning_rate')}")
            
            # 训练模型
            model = xgb.XGBRegressor(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # 评估
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 方向准确率 (预测涨跌对不对)
            direction_acc = ((y_pred > 0) == (y_test > 0)).mean()
            
            self.models[horizon_name] = model
            self.metrics[horizon_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_acc,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            print(f"   RMSE: {np.sqrt(mse):.2f}%, MAE: {mae:.2f}%, R²: {r2:.3f}")
            print(f"   方向准确率: {direction_acc:.1%}")
            print()
        
        self.is_trained = len(self.models) > 0
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测所有周期的收益率
        
        Args:
            X: 特征矩阵
        
        Returns:
            {'1d': array, '5d': array, ...}
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        predictions = {}
        for horizon_name, model in self.models.items():
            predictions[horizon_name] = model.predict(X)
        
        return predictions
    
    def predict_single(self, features: Dict) -> Dict[str, float]:
        """预测单个样本"""
        if not self.is_trained:
            return {}
        
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0)
        
        predictions = self.predict(X)
        return {k: float(v[0]) for k, v in predictions.items()}
    
    def get_feature_importance(self, horizon: str = '5d') -> Dict[str, float]:
        """获取特征重要性"""
        if horizon not in self.models:
            return {}
        
        model = self.models[horizon]
        importance = dict(zip(self.feature_names, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存每个周期的模型
        for horizon_name, model in self.models.items():
            joblib.dump(model, save_dir / f"return_{horizon_name}.joblib")
        
        # 保存元数据
        metadata = {
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'horizons': list(self.models.keys())
        }
        with open(save_dir / "return_predictor_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 收益预测模型已保存: {path}")
    
    def load(self, path: str) -> bool:
        """加载模型"""
        import joblib
        
        save_dir = Path(path)
        meta_path = save_dir / "return_predictor_meta.json"
        
        if not meta_path.exists():
            return False
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.metrics = metadata['metrics']
        
        for horizon_name in metadata['horizons']:
            model_path = save_dir / f"return_{horizon_name}.joblib"
            if model_path.exists():
                self.models[horizon_name] = joblib.load(model_path)
        
        self.is_trained = len(self.models) > 0
        print(f"✅ 收益预测模型已加载: {list(self.models.keys())}")
        return self.is_trained


# === 测试 ===
if __name__ == "__main__":
    print("=== Return Predictor 测试 ===\n")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'f_{i}' for i in range(n_features)]
    
    # 模拟标签 (带噪声的线性组合)
    y_dict = {
        '1d': X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 2,
        '5d': X[:, 0] * 1.0 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 3,
        '10d': X[:, 0] * 1.5 + X[:, 3] * 0.8 + np.random.randn(n_samples) * 4,
    }
    
    predictor = ReturnPredictor()
    metrics = predictor.train(X, y_dict, feature_names)
    
    print("\n=== 特征重要性 (5d) ===")
    importance = predictor.get_feature_importance('5d')
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.3f}")
