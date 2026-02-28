#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble Predictor — MMoE + LightGBM 双模型融合
================================================

Phase 3: 用两个互补模型做 ensemble
- MMoE: 深度多任务模型，擅长捕捉非线性交互
- LightGBM: 梯度提升树，擅长处理异构特征+可解释性强

融合策略:
- 方向概率: 加权平均 (可学习权重)
- 收益预测: 取两者中更保守的
- 排名分数: 几何平均
"""
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

MODEL_DIR = Path(__file__).parent / 'saved_models'


class LGBPredictor:
    """
    LightGBM 多目标预测器 (方向 + 收益 + 排名)
    
    训练 3 个 LightGBM 模型:
    1. direction: 二分类 (上涨/下跌)
    2. return_5d: 回归 (5日收益率)
    3. return_20d: 回归 (20日收益率)
    """
    
    def __init__(self, market: str = 'US'):
        self.market = market
        self.models = {}  # {'direction': model, 'return_5d': model, ...}
        self.feature_names = []
        self.model_dir = MODEL_DIR / f'v2_{market.lower()}_lgb'
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, X: np.ndarray, returns_dict: Dict[str, np.ndarray],
              feature_names: List[str], groups: np.ndarray) -> Dict:
        """训练 LightGBM 多目标模型"""
        if not LGB_AVAILABLE:
            return {'status': 'skipped', 'reason': 'lightgbm not available'}
        
        self.feature_names = feature_names
        results = {}
        
        # 有效样本 mask
        y5 = returns_dict.get('5d', np.array([]))
        y20 = returns_dict.get('20d', np.array([]))
        
        # 1. 方向分类 (5日)
        if len(y5) == len(X):
            valid = ~np.isnan(y5)
            X_v, y_v = X[valid], y5[valid]
            y_dir = (y_v > 0).astype(int)
            
            # 分训练/验证
            n = len(X_v)
            split = int(n * 0.8)
            
            dtrain = lgb.Dataset(X_v[:split], y_dir[:split], feature_name=feature_names)
            dval = lgb.Dataset(X_v[split:], y_dir[split:], feature_name=feature_names, reference=dtrain)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 30,
                'verbose': -1,
                'seed': 42,
            }
            
            model = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            self.models['direction'] = model
            
            # 评估
            pred = model.predict(X_v[split:])
            acc = ((pred > 0.5) == y_dir[split:]).mean()
            results['direction_accuracy'] = float(acc)
            print(f"  ✅ LGB Direction: {acc:.1%}")
        
        # 2. 5日收益回归
        if len(y5) == len(X):
            valid = ~np.isnan(y5)
            X_v, y_v = X[valid], y5[valid]
            n = len(X_v)
            split = int(n * 0.8)
            
            dtrain = lgb.Dataset(X_v[:split], y_v[:split], feature_name=feature_names)
            dval = lgb.Dataset(X_v[split:], y_v[split:], feature_name=feature_names, reference=dtrain)
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 30,
                'verbose': -1,
                'seed': 42,
            }
            
            model = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            self.models['return_5d'] = model
            
            pred = model.predict(X_v[split:])
            mae = np.mean(np.abs(pred - y_v[split:]))
            results['mae_5d'] = float(mae)
            print(f"  ✅ LGB Return 5d: MAE={mae:.2f}%")
        
        # 3. 20日收益回归
        if len(y20) == len(X):
            valid = ~np.isnan(y20)
            if valid.sum() > 200:
                X_v, y_v = X[valid], y20[valid]
                n = len(X_v)
                split = int(n * 0.8)
                
                dtrain = lgb.Dataset(X_v[:split], y_v[:split], feature_name=feature_names)
                dval = lgb.Dataset(X_v[split:], y_v[split:], feature_name=feature_names, reference=dtrain)
                
                params = {
                    'objective': 'regression',
                    'metric': 'mae',
                    'num_leaves': 63,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 30,
                    'verbose': -1,
                    'seed': 42,
                }
                
                model = lgb.train(
                    params, dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
                self.models['return_20d'] = model
                
                pred = model.predict(X_v[split:])
                mae = np.mean(np.abs(pred - y_v[split:]))
                results['mae_20d'] = float(mae)
                print(f"  ✅ LGB Return 20d: MAE={mae:.2f}%")
        
        # Feature importance
        if 'direction' in self.models:
            imp = self.models['direction'].feature_importance(importance_type='gain')
            top_idx = np.argsort(imp)[::-1][:20]
            results['top_features'] = [
                {'name': feature_names[i], 'importance': float(imp[i])}
                for i in top_idx
            ]
        
        return results
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """预测"""
        result = {}
        if 'direction' in self.models:
            result['direction'] = self.models['direction'].predict(X)
        if 'return_5d' in self.models:
            result['return_5d'] = self.models['return_5d'].predict(X)
        if 'return_20d' in self.models:
            result['return_20d'] = self.models['return_20d'].predict(X)
        return result
    
    def save(self):
        """保存所有模型"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(str(self.model_dir / f'{name}.txt'))
        
        meta = {
            'feature_names': self.feature_names,
            'models': list(self.models.keys()),
        }
        with open(self.model_dir / 'lgb_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"  💾 LGB models saved: {self.model_dir}")
    
    def load(self) -> bool:
        """加载模型"""
        if not LGB_AVAILABLE:
            return False
        
        meta_path = self.model_dir / 'lgb_meta.json'
        if not meta_path.exists():
            return False
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        self.feature_names = meta.get('feature_names', [])
        
        for name in meta.get('models', []):
            model_path = self.model_dir / f'{name}.txt'
            if model_path.exists():
                self.models[name] = lgb.Booster(model_file=str(model_path))
        
        return len(self.models) > 0


class EnsemblePredictor:
    """
    MMoE + LightGBM Ensemble 预测器
    
    融合策略:
    - 方向概率: w_mmoe * mmoe_dir + w_lgb * lgb_dir
    - 收益预测: 保守取 min(abs)
    - 最终分数: 综合加权
    """
    
    def __init__(self, 
                 mmoe_weight: float = 0.6,
                 lgb_weight: float = 0.4):
        """
        Args:
            mmoe_weight: MMoE 权重 (默认 0.6 — 多任务模型更强)
            lgb_weight: LightGBM 权重 (默认 0.4 — 树模型互补)
        """
        self.mmoe_weight = mmoe_weight
        self.lgb_weight = lgb_weight
    
    def ensemble_direction(self, 
                           mmoe_dir_prob: float, 
                           lgb_dir_prob: float) -> float:
        """融合方向概率"""
        return (self.mmoe_weight * mmoe_dir_prob + 
                self.lgb_weight * lgb_dir_prob)
    
    def ensemble_return(self,
                       mmoe_return: float,
                       lgb_return: float) -> float:
        """融合收益预测 — 保守策略"""
        # 同向取保守值，反向取较弱信号
        if mmoe_return * lgb_return > 0:
            # 同方向：取绝对值较小的
            return min(mmoe_return, lgb_return, key=abs)
        else:
            # 方向冲突：减半
            avg = (mmoe_return + lgb_return) / 2
            return avg * 0.5
    
    def ensemble_confidence(self,
                           mmoe_dir_prob: float,
                           lgb_dir_prob: float) -> float:
        """融合置信度"""
        # 两者一致时高置信，不一致时低置信
        agreement = 1 - abs(mmoe_dir_prob - lgb_dir_prob)
        avg_extremity = abs((mmoe_dir_prob + lgb_dir_prob) / 2 - 0.5) * 2
        return min(agreement * 0.5 + avg_extremity * 0.5 + 0.2, 0.95)
    
    def predict(self,
                mmoe_result: Dict,
                lgb_result: Dict) -> Dict:
        """
        融合 MMoE 和 LightGBM 的预测结果
        
        Args:
            mmoe_result: MMoE 预测 {'direction': [...], 'return_5d': [...], ...}
            lgb_result: LightGBM 预测 {'direction': [...], 'return_5d': [...], ...}
        
        Returns:
            融合后的预测结果
        """
        result = {}
        
        # 方向概率
        mmoe_dir = mmoe_result.get('direction', [0.5])
        lgb_dir = lgb_result.get('direction', [0.5])
        if len(mmoe_dir) > 0 and len(lgb_dir) > 0:
            result['direction'] = np.array([
                self.ensemble_direction(float(m), float(l))
                for m, l in zip(mmoe_dir, lgb_dir)
            ])
        
        # 5日收益
        mmoe_r5 = mmoe_result.get('return_5d', [0])
        lgb_r5 = lgb_result.get('return_5d', [0])
        if len(mmoe_r5) > 0 and len(lgb_r5) > 0:
            result['return_5d'] = np.array([
                self.ensemble_return(float(m), float(l))
                for m, l in zip(mmoe_r5, lgb_r5)
            ])
        
        # 20日收益
        mmoe_r20 = mmoe_result.get('return_20d', [0])
        lgb_r20 = lgb_result.get('return_20d', [0])
        if len(mmoe_r20) > 0 and len(lgb_r20) > 0:
            result['return_20d'] = np.array([
                self.ensemble_return(float(m), float(l))
                for m, l in zip(mmoe_r20, lgb_r20)
            ])
        
        # 置信度
        if 'direction' in result:
            result['confidence'] = np.array([
                self.ensemble_confidence(float(m), float(l))
                for m, l in zip(mmoe_dir, lgb_dir)
            ])
        
        # 其他 MMoE 独有的直接透传
        for key in ['max_dd', 'rank_score', 'volatility']:
            if key in mmoe_result and key not in result:
                result[key] = mmoe_result[key]
        
        return result
