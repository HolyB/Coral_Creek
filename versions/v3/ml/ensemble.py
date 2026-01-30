#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型集成器 - 多模型融合预测
Ensemble Predictor - Multi-model Fusion

功能:
- 融合多个模型预测结果
- 加权平均、投票等融合策略
- 自动选择最优权重
- 模型表现监控
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime


class EnsemblePredictor:
    """模型集成预测器"""
    
    def __init__(self, market: str = 'US'):
        self.market = market
        self.models = {}  # {name: (model, weight)}
        self.performance_history = []
        self.weights_method = 'equal'  # equal, performance, stacking
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """
        添加模型到集成
        
        Args:
            name: 模型名称
            model: 模型对象 (需有 predict_proba 方法)
            weight: 初始权重
        """
        self.models[name] = {
            'model': model,
            'weight': weight,
            'metrics': {}
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测概率
        
        Args:
            X: 特征矩阵
        
        Returns:
            融合后的概率
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        weights = []
        
        for name, info in self.models.items():
            model = info['model']
            weight = info['weight']
            
            try:
                # 获取模型预测
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    if prob.ndim > 1:
                        prob = prob[:, 1]  # 取正类概率
                else:
                    prob = model.predict(X)
                
                predictions.append(prob)
                weights.append(weight)
            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All models failed")
        
        # 加权平均
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化
        
        ensemble_prob = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_prob
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        集成预测类别
        
        Args:
            X: 特征矩阵
            threshold: 分类阈值
        
        Returns:
            预测类别
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def voting_predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        投票法预测
        
        Args:
            X: 特征矩阵
            threshold: 分类阈值
        
        Returns:
            投票结果
        """
        votes = []
        
        for name, info in self.models.items():
            model = info['model']
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    if prob.ndim > 1:
                        prob = prob[:, 1]
                    pred = (prob >= threshold).astype(int)
                else:
                    pred = model.predict(X)
                votes.append(pred)
            except:
                continue
        
        if not votes:
            raise ValueError("All models failed")
        
        # 多数投票
        votes = np.array(votes)
        majority = (votes.sum(axis=0) > len(votes) / 2).astype(int)
        
        return majority
    
    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        根据验证集表现更新权重
        
        Args:
            X_val: 验证特征
            y_val: 验证标签
        """
        accuracies = {}
        
        for name, info in self.models.items():
            model = info['model']
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_val)
                    if prob.ndim > 1:
                        prob = prob[:, 1]
                    pred = (prob >= 0.5).astype(int)
                else:
                    pred = model.predict(X_val)
                
                acc = (pred == y_val).mean()
                accuracies[name] = acc
                self.models[name]['metrics']['val_accuracy'] = acc
            except:
                accuracies[name] = 0.5  # 默认50%
        
        # 基于准确率计算权重
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            for name, acc in accuracies.items():
                self.models[name]['weight'] = acc / total_acc * len(accuracies)
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取各模型对预测的贡献
        
        Args:
            X: 特征矩阵
        
        Returns:
            各模型预测结果
        """
        contributions = {}
        
        for name, info in self.models.items():
            model = info['model']
            weight = info['weight']
            
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    if prob.ndim > 1:
                        prob = prob[:, 1]
                else:
                    prob = model.predict(X)
                
                contributions[name] = {
                    'predictions': prob,
                    'weight': weight,
                    'weighted': prob * weight
                }
            except:
                continue
        
        return contributions
    
    def save(self, path: str = None):
        """保存集成配置 (不含模型本身)"""
        if path is None:
            path = Path(__file__).parent / 'saved_models' / f'ensemble_{self.market}.json'
        
        config = {
            'market': self.market,
            'models': {
                name: {
                    'weight': info['weight'],
                    'metrics': info['metrics']
                }
                for name, info in self.models.items()
            },
            'weights_method': self.weights_method,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Ensemble config saved to {path}")
    
    def summary(self) -> pd.DataFrame:
        """获取集成摘要"""
        data = []
        for name, info in self.models.items():
            data.append({
                'Model': name,
                'Weight': round(info['weight'], 4),
                'Val Accuracy': info['metrics'].get('val_accuracy', '-')
            })
        return pd.DataFrame(data)


class AutoML:
    """自动化机器学习工具"""
    
    def __init__(self, market: str = 'US'):
        self.market = market
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        self.search_history = []
    
    def auto_train(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   model_types: List[str] = None,
                   cv_folds: int = 5,
                   scoring: str = 'roc_auc') -> Dict:
        """
        自动训练多种模型并选择最优
        
        Args:
            X: 特征矩阵
            y: 标签
            model_types: 要尝试的模型类型
            cv_folds: 交叉验证折数
            scoring: 评估指标
        
        Returns:
            训练结果
        """
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
        except ImportError:
            return {'error': 'sklearn not available'}
        
        if model_types is None:
            model_types = ['logistic', 'rf', 'gbm', 'xgb']
        
        results = []
        
        # 逻辑回归
        if 'logistic' in model_types:
            try:
                lr = LogisticRegression(max_iter=500, random_state=42)
                scores = cross_val_score(lr, X, y, cv=cv_folds, scoring=scoring)
                lr.fit(X, y)
                results.append({
                    'model_type': 'Logistic Regression',
                    'model': lr,
                    'cv_score_mean': scores.mean(),
                    'cv_score_std': scores.std()
                })
            except Exception as e:
                print(f"Logistic Regression failed: {e}")
        
        # 随机森林
        if 'rf' in model_types:
            try:
                rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                scores = cross_val_score(rf, X, y, cv=cv_folds, scoring=scoring)
                rf.fit(X, y)
                results.append({
                    'model_type': 'Random Forest',
                    'model': rf,
                    'cv_score_mean': scores.mean(),
                    'cv_score_std': scores.std()
                })
            except Exception as e:
                print(f"Random Forest failed: {e}")
        
        # Gradient Boosting
        if 'gbm' in model_types:
            try:
                gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
                scores = cross_val_score(gbm, X, y, cv=cv_folds, scoring=scoring)
                gbm.fit(X, y)
                results.append({
                    'model_type': 'Gradient Boosting',
                    'model': gbm,
                    'cv_score_mean': scores.mean(),
                    'cv_score_std': scores.std()
                })
            except Exception as e:
                print(f"GBM failed: {e}")
        
        # XGBoost
        if 'xgb' in model_types:
            try:
                import xgboost as xgb
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    use_label_encoder=False, eval_metric='logloss', random_state=42
                )
                scores = cross_val_score(xgb_model, X, y, cv=cv_folds, scoring=scoring)
                xgb_model.fit(X, y)
                results.append({
                    'model_type': 'XGBoost',
                    'model': xgb_model,
                    'cv_score_mean': scores.mean(),
                    'cv_score_std': scores.std()
                })
            except Exception as e:
                print(f"XGBoost failed: {e}")
        
        # 选择最优
        if results:
            best = max(results, key=lambda x: x['cv_score_mean'])
            self.best_model = best['model']
            self.best_params = {'model_type': best['model_type']}
            self.best_score = best['cv_score_mean']
            self.search_history = results
            
            return {
                'success': True,
                'best_model_type': best['model_type'],
                'best_cv_score': round(best['cv_score_mean'], 4),
                'all_results': [
                    {
                        'model': r['model_type'],
                        'score': round(r['cv_score_mean'], 4),
                        'std': round(r['cv_score_std'], 4)
                    }
                    for r in sorted(results, key=lambda x: -x['cv_score_mean'])
                ]
            }
        
        return {'error': 'No models trained successfully'}
    
    def get_best_model(self):
        """获取最优模型"""
        return self.best_model
    
    def create_ensemble(self) -> EnsemblePredictor:
        """
        使用训练过的模型创建集成
        
        Returns:
            EnsemblePredictor 实例
        """
        ensemble = EnsemblePredictor(market=self.market)
        
        for result in self.search_history:
            # 权重与 CV 分数成正比
            weight = result['cv_score_mean']
            ensemble.add_model(result['model_type'], result['model'], weight)
        
        return ensemble


if __name__ == "__main__":
    print("=== Ensemble Predictor 测试 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    X_train = np.random.randn(200, 10)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(50, 10)
    
    # 测试 AutoML
    automl = AutoML()
    result = automl.auto_train(X_train, y_train, model_types=['logistic', 'rf', 'gbm'])
    
    if 'error' not in result:
        print(f"Best Model: {result['best_model_type']}")
        print(f"Best CV Score: {result['best_cv_score']}")
        print("\nAll Results:")
        for r in result['all_results']:
            print(f"  {r['model']}: {r['score']} (±{r['std']})")
        
        # 创建集成
        ensemble = automl.create_ensemble()
        proba = ensemble.predict_proba(X_test)
        print(f"\nEnsemble predictions (first 5): {proba[:5]}")
        print(f"\nEnsemble Summary:")
        print(ensemble.summary())
    else:
        print(f"Error: {result['error']}")
