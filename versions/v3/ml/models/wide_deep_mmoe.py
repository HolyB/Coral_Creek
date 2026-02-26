#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wide&Deep MMoE — XGBoost 做 Wide, MMoE 做 Deep
================================================

架构:
  原始特征 (110)
       │
  ┌────┴────────────────────────┐
  │                              │
  │  DEEP (MMoE)                 │  WIDE (XGBoost)
  │  Expert1..K → Gate → Mix     │  pred_5d, pred_20d
  │       ↓                      │  leaf_pca (16d)
  │  expert_out (32)             │  → Linear → wide_out (16)
  │                              │
  └────────┬─────────────────────┘
           │ concat: expert_mix (32) + wide_out (16)
           │
      ┌────┴────┐
      │ Tower_i │  (任务特定)
      └────┬────┘
           │
       Task output
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
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    pass

from ml.models.mmoe import (
    ExpertNetwork, GateNetwork, TowerNetwork,
    MMoEPredictor, ALL_TASK_DEFS,
)


class WideDeepMMoEModel(nn.Module):
    """Wide&Deep + MMoE"""

    def __init__(self,
                 deep_dim: int,      # 原始特征维度
                 wide_dim: int,      # XGB 特征维度
                 num_experts: int = 4,
                 expert_hidden: int = 64,
                 expert_out: int = 32,
                 wide_out: int = 16,
                 tower_hidden: int = 32,
                 num_tasks: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks

        # Deep: MMoE experts + gates
        self.experts = nn.ModuleList([
            ExpertNetwork(deep_dim, expert_hidden, expert_out, dropout)
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            GateNetwork(deep_dim, num_experts)
            for _ in range(num_tasks)
        ])

        # Wide: XGB 特征直接线性变换
        self.wide_layer = nn.Sequential(
            nn.Linear(wide_dim, wide_out),
            nn.ReLU(),
        )

        # Towers: 输入 = expert_out + wide_out
        combined_dim = expert_out + wide_out
        self.towers = nn.ModuleList([
            TowerNetwork(combined_dim, tower_hidden, 1, dropout)
            for _ in range(num_tasks)
        ])

    def forward(self, x_deep, x_wide):
        # Deep path: MMoE
        expert_outs = torch.stack([e(x_deep) for e in self.experts], dim=1)
        # Wide path
        wide_out = self.wide_layer(x_wide)

        task_outputs = []
        for i in range(self.num_tasks):
            gw = self.gates[i](x_deep).unsqueeze(-1)
            mixed = (expert_outs * gw).sum(dim=1)  # (batch, expert_out)
            # Concat deep + wide
            combined = torch.cat([mixed, wide_out], dim=1)
            task_outputs.append(self.towers[i](combined).squeeze(-1))

        return task_outputs


class WideDeepMMoE:
    """Wide&Deep MMoE 训练器 — OOF XGBoost 做 Wide"""

    def __init__(self,
                 # XGB
                 xgb_n_estimators: int = 300,
                 xgb_max_depth: int = 4,
                 xgb_leaf_dim: int = 16,
                 n_folds: int = 5,
                 # MMoE
                 num_experts: int = 4,
                 expert_hidden: int = 64,
                 expert_out: int = 32,
                 wide_out: int = 16,
                 tower_hidden: int = 16,
                 dropout: float = 0.2,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-3,
                 epochs: int = 200,
                 batch_size: int = 128,
                 patience: int = 25,
                 task_defs: Optional[List] = None,
                 device: str = 'auto'):
        if not TORCH_AVAILABLE:
            raise ImportError("pip install torch")

        self.task_defs = task_defs or [t for t in ALL_TASK_DEFS if t['name'] != 'volatility']
        self.xgb_config = dict(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
                               leaf_dim=xgb_leaf_dim, n_folds=n_folds)
        self.mmoe_config = dict(
            num_experts=num_experts, expert_hidden=expert_hidden,
            expert_out=expert_out, wide_out=wide_out,
            tower_hidden=tower_hidden, dropout=dropout,
            lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size, patience=patience,
        )
        if device == 'auto':
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available()
                else 'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.xgb_models = []  # K-fold XGB models
        self.leaf_pcas = []
        self.model = None
        self.feature_names = []
        self.feature_mean = None
        self.feature_std = None
        self.wide_mean = None
        self.wide_std = None
        self.label_stats = {}

    def _build_oof_wide(self, X, y_5d, y_20d, groups):
        """OOF XGBoost → Wide 特征"""
        from xgboost import XGBRegressor
        from sklearn.decomposition import PCA

        unique_groups = sorted(set(groups))
        n_folds = self.xgb_config['n_folds']
        fold_size = max(1, len(unique_groups) // n_folds)

        oof_5d = np.zeros(len(X))
        oof_20d = np.zeros(len(X))
        leaf_dim = self.xgb_config['leaf_dim']
        oof_leaf = np.zeros((len(X), leaf_dim))

        self.xgb_models = []
        self.leaf_pcas = []

        for fold in range(n_folds):
            vs = fold * fold_size
            ve = (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_groups)
            val_g = set(unique_groups[vs:ve])
            vmask = np.array([g in val_g for g in groups])
            tmask = ~vmask

            if vmask.sum() == 0:
                continue

            xgb5 = XGBRegressor(
                max_depth=self.xgb_config['max_depth'], learning_rate=0.03,
                n_estimators=self.xgb_config['n_estimators'],
                subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
            xgb5.fit(X[tmask], y_5d[tmask], eval_set=[(X[vmask], y_5d[vmask])], verbose=False)

            xgb20 = XGBRegressor(
                max_depth=self.xgb_config['max_depth'] + 1, learning_rate=0.03,
                n_estimators=self.xgb_config['n_estimators'],
                subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
            xgb20.fit(X[tmask], y_20d[tmask], eval_set=[(X[vmask], y_20d[vmask])], verbose=False)

            oof_5d[vmask] = xgb5.predict(X[vmask])
            oof_20d[vmask] = xgb20.predict(X[vmask])

            # Leaf encoding
            l5 = xgb5.apply(X[vmask]).astype(float)
            l20 = xgb20.apply(X[vmask]).astype(float)
            lall = np.hstack([l5, l20])
            pca = PCA(n_components=min(leaf_dim, lall.shape[1]))
            pca.fit(np.hstack([xgb5.apply(X[tmask]).astype(float),
                               xgb20.apply(X[tmask]).astype(float)]))
            lr = pca.transform(lall)
            oof_leaf[vmask, :lr.shape[1]] = lr

            self.xgb_models.append((xgb5, xgb20))
            self.leaf_pcas.append(pca)

        # Wide features: [pred_5d, pred_20d, dir, confidence, leaf_pca...]
        wide = np.column_stack([
            oof_5d, oof_20d,
            (oof_5d > 0).astype(float),
            np.abs(oof_5d),
            oof_leaf,
        ])
        return wide

    def _extract_wide(self, X):
        """对新数据提取 Wide 特征 (用所有 fold 模型平均)"""
        from sklearn.decomposition import PCA
        n = len(X)
        pred_5d = np.zeros(n)
        pred_20d = np.zeros(n)
        leaf_dim = self.xgb_config['leaf_dim']
        leaf_acc = np.zeros((n, leaf_dim))

        for (xgb5, xgb20), pca in zip(self.xgb_models, self.leaf_pcas):
            pred_5d += xgb5.predict(X)
            pred_20d += xgb20.predict(X)
            l5 = xgb5.apply(X).astype(float)
            l20 = xgb20.apply(X).astype(float)
            lr = pca.transform(np.hstack([l5, l20]))
            leaf_acc[:, :lr.shape[1]] += lr

        k = len(self.xgb_models)
        pred_5d /= k
        pred_20d /= k
        leaf_acc /= k

        return np.column_stack([
            pred_5d, pred_20d,
            (pred_5d > 0).astype(float),
            np.abs(pred_5d),
            leaf_acc,
        ])

    def train(self, X, returns_dict, feature_names, groups,
              drawdowns_dict=None, val_ratio=0.2):
        self.feature_names = feature_names
        task_defs = self.task_defs
        num_tasks = len(task_defs)

        y_5d = np.nan_to_num(returns_dict.get('5d', np.zeros(len(X))), 0.0)
        y_20d = np.nan_to_num(returns_dict.get('20d', np.zeros(len(X))), 0.0)

        print(f"\n{'='*60}")
        print(f"🏗️  Wide&Deep MMoE ({num_tasks} 任务)")
        print(f"   样本: {X.shape[0]}, Deep特征: {X.shape[1]}, 设备: {self.device}")
        print(f"{'='*60}")

        # Stage 1: OOF XGBoost → Wide
        print(f"\n📈 Stage 1: OOF XGBoost (K={self.xgb_config['n_folds']})")
        wide = self._build_oof_wide(X, y_5d, y_20d, groups)
        wide_dim = wide.shape[1]
        print(f"   Wide 特征: {wide_dim} dims")

        # 标签
        labels = MMoEPredictor._build_labels(returns_dict, drawdowns_dict or {}, groups)
        valid = np.zeros(len(X), dtype=bool)
        for td in task_defs:
            if td['label_key'] in labels:
                valid |= ~np.isnan(labels[td['label_key']])
        X_v, wide_v, groups_v = X[valid], wide[valid], groups[valid]
        labels_v = {k: v[valid] for k, v in labels.items()}

        # 标准化
        self.feature_mean = np.nanmean(X_v, axis=0)
        self.feature_std = np.nanstd(X_v, axis=0) + 1e-8
        X_norm = np.nan_to_num((X_v - self.feature_mean) / self.feature_std, 0.0)

        self.wide_mean = np.nanmean(wide_v, axis=0)
        self.wide_std = np.nanstd(wide_v, axis=0) + 1e-8
        wide_norm = np.nan_to_num((wide_v - self.wide_mean) / self.wide_std, 0.0)

        # 标签标准化
        Y_list = []
        for td in task_defs:
            arr = np.nan_to_num(labels_v[td['label_key']].copy(), 0.0)
            if td['type'] in ('regression', 'ranking'):
                m, s = float(np.mean(arr)), float(np.std(arr) + 1e-8)
                self.label_stats[td['name']] = {'mean': m, 'std': s}
                arr = (arr - m) / s
            Y_list.append(arr)

        # Split
        ugroups = sorted(set(groups_v))
        split = int(len(ugroups) * (1 - val_ratio))
        train_g = set(ugroups[:split])
        tmask = np.array([g in train_g for g in groups_v])

        Xd_tr, Xd_va = X_norm[tmask], X_norm[~tmask]
        Xw_tr, Xw_va = wide_norm[tmask], wide_norm[~tmask]
        Y_tr = [y[tmask] for y in Y_list]
        Y_va = [y[~tmask] for y in Y_list]

        print(f"\n🧠 Stage 2: Wide&Deep MMoE")
        print(f"   训练: {len(Xd_tr)}, 验证: {len(Xd_va)}")
        print(f"   Deep: {Xd_tr.shape[1]} → Experts, Wide: {Xw_tr.shape[1]} → Direct")

        # DataLoader
        tr_t = [torch.FloatTensor(Xd_tr), torch.FloatTensor(Xw_tr)] + [torch.FloatTensor(y) for y in Y_tr]
        va_t = [torch.FloatTensor(Xd_va), torch.FloatTensor(Xw_va)] + [torch.FloatTensor(y) for y in Y_va]
        tr_loader = DataLoader(TensorDataset(*tr_t), batch_size=self.mmoe_config['batch_size'], shuffle=True)
        va_loader = DataLoader(TensorDataset(*va_t), batch_size=self.mmoe_config['batch_size'])

        # Model
        self.model = WideDeepMMoEModel(
            deep_dim=Xd_tr.shape[1], wide_dim=Xw_tr.shape[1],
            num_experts=self.mmoe_config['num_experts'],
            expert_hidden=self.mmoe_config['expert_hidden'],
            expert_out=self.mmoe_config['expert_out'],
            wide_out=self.mmoe_config['wide_out'],
            tower_hidden=self.mmoe_config['tower_hidden'],
            num_tasks=num_tasks, dropout=self.mmoe_config['dropout'],
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.mmoe_config['lr'],
                                weight_decay=self.mmoe_config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.mmoe_config['epochs'])
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        best_val = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(self.mmoe_config['epochs']):
            self.model.train()
            ep_loss = 0.0
            for batch in tr_loader:
                xd, xw = batch[0].to(self.device), batch[1].to(self.device)
                ybs = [batch[i+2].to(self.device) for i in range(num_tasks)]
                preds = self.model(xd, xw)
                loss = sum(td['weight'] * (bce if td['type'] == 'classification' else mse)(preds[i], ybs[i])
                           for i, td in enumerate(task_defs))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            scheduler.step()

            # Val
            self.model.eval()
            vloss = 0.0
            all_p = [[] for _ in range(num_tasks)]
            all_t = [[] for _ in range(num_tasks)]
            with torch.no_grad():
                for batch in va_loader:
                    xd, xw = batch[0].to(self.device), batch[1].to(self.device)
                    ybs = [batch[i+2].to(self.device) for i in range(num_tasks)]
                    preds = self.model(xd, xw)
                    bl = sum(td['weight'] * (bce if td['type'] == 'classification' else mse)(preds[i], ybs[i])
                             for i, td in enumerate(task_defs))
                    vloss += bl.item()
                    for i in range(num_tasks):
                        p = preds[i].cpu().numpy()
                        if task_defs[i]['type'] == 'classification':
                            p = 1 / (1 + np.exp(-p))
                        all_p[i].extend(p)
                        all_t[i].extend(ybs[i].cpu().numpy())

            avg_vl = vloss / max(len(va_loader), 1)

            dir_idx = next((i for i, t in enumerate(task_defs) if t['name'] == 'direction'), None)
            dir_acc = np.mean((np.array(all_p[dir_idx]) > 0.5) == np.array(all_t[dir_idx])) if dir_idx is not None else 0

            r5i = next((i for i, t in enumerate(task_defs) if t['name'] == 'return_5d'), None)
            if r5i is not None and 'return_5d' in self.label_stats:
                s = self.label_stats['return_5d']
                mae5 = np.mean(np.abs(np.array(all_p[r5i])*s['std']+s['mean'] - (np.array(all_t[r5i])*s['std']+s['mean'])))
            else:
                mae5 = 999

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}: val={avg_vl:.4f}, dir={dir_acc:.1%}, mae5={mae5:.2f}%")

            if avg_vl < best_val:
                best_val = avg_vl
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.mmoe_config['patience']:
                    print(f"   ⏹ Early stop @ epoch {epoch+1}")
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()

        results = {
            'epochs': epoch+1, 'best_val_loss': best_val,
            'dir_accuracy': dir_acc, 'mae_5d': mae5,
            'train_n': len(Xd_tr), 'val_n': len(Xd_va),
            'deep_features': Xd_tr.shape[1], 'wide_features': Xw_tr.shape[1],
        }
        print(f"\n✅ Wide&Deep MMoE: dir={dir_acc:.1%}, mae5={mae5:.2f}%")
        return results

    def predict(self, X):
        wide = self._extract_wide(X)
        Xn = np.nan_to_num((X - self.feature_mean) / self.feature_std, 0.0)
        wn = np.nan_to_num((wide - self.wide_mean) / self.wide_std, 0.0)
        xd = torch.FloatTensor(Xn).to(self.device)
        xw = torch.FloatTensor(wn).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(xd, xw)
        out = {}
        for i, td in enumerate(self.task_defs):
            p = preds[i].cpu().numpy()
            if td['type'] == 'classification':
                p = 1 / (1 + np.exp(-p))
            elif td['name'] in self.label_stats:
                s = self.label_stats[td['name']]
                p = p * s['std'] + s['mean']
            out[td['name']] = p
        return out

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / 'wide_deep_model.pt')
        for i, (xgb5, xgb20) in enumerate(self.xgb_models):
            joblib.dump(xgb5, p / f'xgb5_fold{i}.joblib')
            joblib.dump(xgb20, p / f'xgb20_fold{i}.joblib')
        for i, pca in enumerate(self.leaf_pcas):
            joblib.dump(pca, p / f'pca_fold{i}.joblib')
        meta = {
            'xgb_config': self.xgb_config, 'mmoe_config': self.mmoe_config,
            'task_defs': self.task_defs,
            'feature_names': self.feature_names,
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'wide_mean': self.wide_mean.tolist(),
            'wide_std': self.wide_std.tolist(),
            'label_stats': self.label_stats,
            'n_folds': len(self.xgb_models),
        }
        with open(p / 'wide_deep_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Wide&Deep 模型已保存: {p}")
