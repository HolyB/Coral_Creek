#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost → MMoE Stacking 模型
============================

架构:
  原始特征 (110)
       │
  ┌────┴────┐
  │ XGBoost │ (Stage 1: 特征交互 + 叶子编码)
  │ 5d pred │
  │ 20d pred│
  │ leaf idx│ → one-hot → 降维
  └────┬────┘
       │ 拼接: 原始特征 + XGB 输出 (110 + K)
       │
  ┌────┴────┐
  │  MMoE   │ (Stage 2: 多任务学习)
  │ 6 tasks │
  └─────────┘

优势:
  - XGBoost 捕获高阶非线性特征交互
  - MMoE 做多任务的跨周期信息传递
  - 叶子编码 = 自动特征离散化 + 交叉
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional
from pathlib import Path
import json
import joblib

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

from ml.models.mmoe import MMoEPredictor, TASK_DEFS


class XGBMMoEStacker:
    """XGBoost → MMoE 两阶段 Stacking 模型"""

    def __init__(self,
                 # XGBoost 参数
                 xgb_max_depth: int = 4,
                 xgb_lr: float = 0.03,
                 xgb_n_estimators: int = 300,
                 xgb_leaf_dim: int = 16,  # 叶子编码降维维度
                 # MMoE 参数
                 num_experts: int = 4,
                 expert_hidden: int = 64,
                 expert_out: int = 32,
                 tower_hidden: int = 16,
                 dropout: float = 0.2,
                 mmoe_lr: float = 5e-4,
                 mmoe_epochs: int = 200,
                 mmoe_patience: int = 25,
                 mmoe_batch: int = 128):

        self.xgb_config = dict(
            max_depth=xgb_max_depth, lr=xgb_lr,
            n_estimators=xgb_n_estimators, leaf_dim=xgb_leaf_dim,
        )
        self.mmoe_config = dict(
            num_experts=num_experts, expert_hidden=expert_hidden,
            expert_out=expert_out, tower_hidden=tower_hidden,
            dropout=dropout, lr=mmoe_lr, epochs=mmoe_epochs,
            patience=mmoe_patience, batch_size=mmoe_batch,
        )

        self.xgb_5d = None
        self.xgb_20d = None
        self.leaf_pca = None  # PCA 降维叶子编码
        self.mmoe = None
        self.feature_names: List[str] = []
        self.stacked_feature_names: List[str] = []

    def _train_xgb_stage(self, X, returns_dict, groups, val_ratio=0.2):
        """Stage 1: 训练 XGBoost，提取叶子编码"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("pip install xgboost")

        from sklearn.decomposition import PCA

        y_5d = np.nan_to_num(returns_dict.get('5d', np.zeros(len(X))), 0.0)
        y_20d = np.nan_to_num(returns_dict.get('20d', np.zeros(len(X))), 0.0)

        # 时序分割 (用 groups)
        ugroups = sorted(set(groups))
        split = int(len(ugroups) * (1 - val_ratio))
        train_g = set(ugroups[:split])
        tmask = np.array([g in train_g for g in groups])

        X_tr, X_va = X[tmask], X[~tmask]
        y5_tr, y5_va = y_5d[tmask], y_5d[~tmask]
        y20_tr, y20_va = y_20d[tmask], y_20d[~tmask]

        print(f"\n📈 Stage 1: XGBoost 特征提取")
        print(f"   训练: {len(X_tr)}, 验证: {len(X_va)}")

        # 5d model
        self.xgb_5d = XGBRegressor(
            max_depth=self.xgb_config['max_depth'],
            learning_rate=self.xgb_config['lr'],
            n_estimators=self.xgb_config['n_estimators'],
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        self.xgb_5d.fit(X_tr, y5_tr, eval_set=[(X_va, y5_va)],
                        verbose=False)

        pred_5d = self.xgb_5d.predict(X)
        from sklearn.metrics import mean_absolute_error
        mae_5d = mean_absolute_error(y5_va, self.xgb_5d.predict(X_va))
        print(f"   XGB 5d MAE: {mae_5d:.2f}%")

        # 20d model
        self.xgb_20d = XGBRegressor(
            max_depth=self.xgb_config['max_depth'] + 1,
            learning_rate=self.xgb_config['lr'],
            n_estimators=self.xgb_config['n_estimators'],
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        self.xgb_20d.fit(X_tr, y20_tr, eval_set=[(X_va, y20_va)],
                         verbose=False)
        pred_20d = self.xgb_20d.predict(X)
        mae_20d = mean_absolute_error(y20_va, self.xgb_20d.predict(X_va))
        print(f"   XGB 20d MAE: {mae_20d:.2f}%")

        # 叶子编码
        leaf_5d = self.xgb_5d.apply(X)   # (n, n_trees) 每棵树的叶子节点 ID
        leaf_20d = self.xgb_20d.apply(X)
        leaf_all = np.hstack([leaf_5d, leaf_20d]).astype(float)

        # PCA 降维叶子编码 (从 600→leaf_dim)
        leaf_dim = self.xgb_config['leaf_dim']
        self.leaf_pca = PCA(n_components=min(leaf_dim, leaf_all.shape[1]))
        leaf_reduced = self.leaf_pca.fit_transform(leaf_all)
        print(f"   叶子编码: {leaf_all.shape[1]} → PCA → {leaf_reduced.shape[1]}")

        # 拼接: 原始特征 + XGB 预测 + 叶子编码
        xgb_features = np.column_stack([
            pred_5d.reshape(-1, 1),
            pred_20d.reshape(-1, 1),
            (pred_5d > 0).astype(float).reshape(-1, 1),  # XGB 方向
            np.abs(pred_5d).reshape(-1, 1),               # XGB 信心
            leaf_reduced,
        ])

        n_xgb = xgb_features.shape[1]
        xgb_feat_names = (
            ['xgb_pred_5d', 'xgb_pred_20d', 'xgb_dir_5d', 'xgb_confidence']
            + [f'xgb_leaf_{i}' for i in range(leaf_reduced.shape[1])]
        )

        X_stacked = np.hstack([X, xgb_features])
        print(f"   堆叠特征: {X.shape[1]} + {n_xgb} = {X_stacked.shape[1]}")

        return X_stacked, xgb_feat_names, {'mae_5d': mae_5d, 'mae_20d': mae_20d}

    def _extract_xgb_features(self, X):
        """对新数据提取 XGB 特征"""
        pred_5d = self.xgb_5d.predict(X)
        pred_20d = self.xgb_20d.predict(X)

        leaf_5d = self.xgb_5d.apply(X).astype(float)
        leaf_20d = self.xgb_20d.apply(X).astype(float)
        leaf_all = np.hstack([leaf_5d, leaf_20d])
        leaf_reduced = self.leaf_pca.transform(leaf_all)

        xgb_features = np.column_stack([
            pred_5d.reshape(-1, 1),
            pred_20d.reshape(-1, 1),
            (pred_5d > 0).astype(float).reshape(-1, 1),
            np.abs(pred_5d).reshape(-1, 1),
            leaf_reduced,
        ])
        return np.hstack([X, xgb_features])

    def train(self,
              X: np.ndarray,
              returns_dict: Dict[str, np.ndarray],
              feature_names: List[str],
              groups: np.ndarray,
              drawdowns_dict: Optional[Dict[str, np.ndarray]] = None,
              val_ratio: float = 0.2) -> Dict:
        """两阶段训练"""
        self.feature_names = feature_names

        print(f"\n{'='*60}")
        print(f"🏗️  XGBoost → MMoE Stacking")
        print(f"   样本: {X.shape[0]}, 原始特征: {X.shape[1]}")
        print(f"{'='*60}")

        # Stage 1: XGBoost
        X_stacked, xgb_feat_names, xgb_metrics = self._train_xgb_stage(
            X, returns_dict, groups, val_ratio)
        self.stacked_feature_names = feature_names + xgb_feat_names

        # Stage 2: MMoE
        print(f"\n🧠 Stage 2: MMoE 多任务学习")
        self.mmoe = MMoEPredictor(**self.mmoe_config)
        mmoe_results = self.mmoe.train(
            X_stacked, returns_dict, self.stacked_feature_names,
            groups, drawdowns_dict, val_ratio)

        results = {
            'stage1_xgb': xgb_metrics,
            'stage2_mmoe': mmoe_results,
            'total_features': X_stacked.shape[1],
            'original_features': X.shape[1],
            'xgb_added_features': X_stacked.shape[1] - X.shape[1],
        }

        return results

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """两阶段预测"""
        X_stacked = self._extract_xgb_features(X)
        return self.mmoe.predict(X_stacked)

    def save(self, path: str):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb_5d, p / 'xgb_5d.joblib')
        joblib.dump(self.xgb_20d, p / 'xgb_20d.joblib')
        joblib.dump(self.leaf_pca, p / 'leaf_pca.joblib')
        self.mmoe.save(str(p / 'mmoe'))
        meta = {
            'feature_names': self.feature_names,
            'stacked_feature_names': self.stacked_feature_names,
            'xgb_config': self.xgb_config,
            'mmoe_config': self.mmoe_config,
        }
        with open(p / 'stacker_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Stacking 模型已保存: {p}")

    def load(self, path: str):
        p = Path(path)
        with open(p / 'stacker_meta.json') as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']
        self.stacked_feature_names = meta['stacked_feature_names']
        self.xgb_config = meta['xgb_config']
        self.mmoe_config = meta['mmoe_config']
        self.xgb_5d = joblib.load(p / 'xgb_5d.joblib')
        self.xgb_20d = joblib.load(p / 'xgb_20d.joblib')
        self.leaf_pca = joblib.load(p / 'leaf_pca.joblib')
        self.mmoe = MMoEPredictor(**self.mmoe_config)
        self.mmoe.load(str(p / 'mmoe'))
        print(f"✅ Stacking 模型已加载: {p}")
