"""
超参数调优模块
Hyperparameter Tuning

使用 GridSearch / RandomizedSearch 找最优参数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time

try:
    import xgboost as xgb
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, 
        cross_val_score, TimeSeriesSplit
    )
    from sklearn.metrics import make_scorer, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class HyperparameterTuner:
    """超参数调优器"""
    
    # XGBoost 回归参数搜索空间
    REGRESSOR_PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
    }
    
    # 快速搜索 (较小的空间)
    REGRESSOR_PARAM_GRID_FAST = {
        'n_estimators': [100, 200],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    
    # XGBoost Ranker 参数搜索空间
    RANKER_PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }
    
    def __init__(self):
        self.best_params: Dict[str, Dict] = {}
        self.cv_results: Dict[str, pd.DataFrame] = {}
        self.tuning_history: List[Dict] = []
        
    def tune_regressor(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       horizon: str = '5d',
                       method: str = 'random',
                       n_iter: int = 50,
                       cv: int = 5,
                       fast: bool = True) -> Dict:
        """
        调优收益预测模型
        
        Args:
            X: 特征矩阵
            y: 标签
            horizon: 预测周期
            method: 'grid' 或 'random'
            n_iter: RandomizedSearch 迭代次数
            cv: 交叉验证折数
            fast: 是否使用快速搜索空间
        
        Returns:
            最优参数和评估结果
        """
        if not ML_AVAILABLE:
            raise RuntimeError("sklearn/xgboost 未安装")
        
        print(f"\n{'='*50}")
        print(f"🔧 调优 ReturnPredictor ({horizon})")
        print(f"   样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"   方法: {method}, CV: {cv} 折")
        print(f"{'='*50}\n")
        
        # 过滤无效样本
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) < 100:
            print("❌ 样本不足")
            return {}
        
        # 基础模型
        base_model = xgb.XGBRegressor(random_state=42)
        
        # 选择参数空间
        param_grid = self.REGRESSOR_PARAM_GRID_FAST if fast else self.REGRESSOR_PARAM_GRID
        
        # 自定义评分函数 (方向准确率)
        def direction_accuracy(y_true, y_pred):
            return ((y_pred > 0) == (y_true > 0)).mean()
        
        direction_scorer = make_scorer(direction_accuracy)
        
        # 搜索
        start_time = time.time()
        
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring={
                    'r2': 'r2',
                    'neg_mse': 'neg_mean_squared_error',
                    'direction': direction_scorer
                },
                refit='direction',  # 以方向准确率为主
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring={
                    'r2': 'r2',
                    'neg_mse': 'neg_mean_squared_error',
                    'direction': direction_scorer
                },
                refit='direction',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        search.fit(X_valid, y_valid)
        
        elapsed = time.time() - start_time
        
        # 结果
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"\n✅ 调优完成 ({elapsed:.1f}秒)")
        print(f"\n📊 最优参数:")
        for k, v in best_params.items():
            print(f"   {k}: {v}")
        
        print(f"\n📈 最优得分 (方向准确率): {best_score:.3f}")
        
        # 获取 CV 结果
        cv_results = pd.DataFrame(search.cv_results_)
        
        # 保存
        self.best_params[f'regressor_{horizon}'] = best_params
        self.cv_results[f'regressor_{horizon}'] = cv_results
        
        # 与默认参数对比
        default_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        default_scores = cross_val_score(
            default_model, X_valid, y_valid, cv=cv, scoring=direction_scorer
        )
        
        improvement = (best_score - default_scores.mean()) * 100
        
        print(f"\n📊 对比默认参数:")
        print(f"   默认得分: {default_scores.mean():.3f} (±{default_scores.std():.3f})")
        print(f"   最优得分: {best_score:.3f}")
        print(f"   提升: {improvement:+.1f}%")
        
        result = {
            'horizon': horizon,
            'best_params': best_params,
            'best_score': best_score,
            'default_score': default_scores.mean(),
            'improvement': improvement,
            'elapsed_seconds': elapsed,
            'n_candidates': len(cv_results)
        }
        
        self.tuning_history.append(result)
        
        return result
    
    def tune_all_regressors(self, 
                            X: np.ndarray, 
                            y_dict: Dict[str, np.ndarray],
                            method: str = 'random',
                            n_iter: int = 30,
                            fast: bool = True) -> Dict[str, Dict]:
        """调优所有周期的收益预测模型"""
        
        results = {}
        
        for horizon, y in y_dict.items():
            try:
                result = self.tune_regressor(
                    X, y, horizon, method, n_iter, fast=fast
                )
                results[horizon] = result
            except Exception as e:
                print(f"❌ {horizon} 调优失败: {e}")
        
        return results
    
    def get_best_params(self, model_type: str = 'regressor', horizon: str = '5d') -> Dict:
        """获取最优参数"""
        key = f'{model_type}_{horizon}'
        return self.best_params.get(key, {})
    
    def save_results(self, path: str):
        """保存调优结果"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最优参数
        with open(save_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # 保存调优历史
        with open(save_dir / 'tuning_history.json', 'w') as f:
            json.dump(self.tuning_history, f, indent=2)
        
        # 保存 CV 结果
        for name, df in self.cv_results.items():
            df.to_csv(save_dir / f'cv_results_{name}.csv', index=False)
        
        print(f"✅ 调优结果已保存: {path}")
    
    def load_results(self, path: str) -> bool:
        """加载调优结果"""
        save_dir = Path(path)
        
        params_path = save_dir / 'best_params.json'
        if not params_path.exists():
            return False
        
        with open(params_path) as f:
            self.best_params = json.load(f)
        
        history_path = save_dir / 'tuning_history.json'
        if history_path.exists():
            with open(history_path) as f:
                self.tuning_history = json.load(f)
        
        print(f"✅ 已加载调优结果: {list(self.best_params.keys())}")
        return True


def run_tuning(market: str = 'US', fast: bool = True) -> Dict:
    """
    运行完整的超参数调优流程
    
    Args:
        market: 市场 (US/CN)
        fast: 是否快速模式
    
    Returns:
        调优结果
    """
    from ml.pipeline import MLPipeline
    
    print("\n" + "="*60)
    print("🔧 开始超参数调优")
    print("="*60)
    
    # 1. 准备数据
    print("\n📦 准备数据集...")
    pipeline = MLPipeline(market=market)
    X, returns_dict, drawdowns_dict, groups, feature_names, signals_df = pipeline.prepare_dataset()
    
    if X is None or len(X) == 0:
        print("❌ 无法准备数据")
        return {}
    
    print(f"   样本数: {len(X)}")
    print(f"   特征数: {len(feature_names)}")
    
    # 2. 调优
    tuner = HyperparameterTuner()
    
    results = tuner.tune_all_regressors(
        X, returns_dict,
        method='random',
        n_iter=30 if fast else 100,
        fast=fast
    )
    
    # 3. 保存结果
    save_path = Path(__file__).parent / 'tuning_results' / f'{market.lower()}'
    tuner.save_results(str(save_path))
    
    # 4. 汇总
    print("\n" + "="*60)
    print("📊 调优结果汇总")
    print("="*60)
    
    for horizon, result in results.items():
        if result:
            print(f"\n{horizon}:")
            print(f"  最优参数: {result['best_params']}")
            print(f"  方向准确率: {result['best_score']:.3f} (提升 {result['improvement']:+.1f}%)")
    
    return results


# === 命令行入口 ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='超参数调优')
    parser.add_argument('--market', type=str, default='US', help='市场')
    parser.add_argument('--fast', action='store_true', help='快速模式')
    
    args = parser.parse_args()
    
    run_tuning(market=args.market, fast=args.fast)
