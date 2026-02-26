#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMoE (Multi-gate Mixture of Experts) 多任务预测模型
=================================================

6 个任务:
  1) 5d 收益预测 (回归)
  2) 20d 收益预测 (回归)
  3) 方向分类 (二分类: 5d 涨/跌)
  4) 最大回撤预测 (回归, 风控)
  5) 排序得分 (pairwise ranking)
  6) 波动率预测 (回归)

架构:
  Input → [Expert1, Expert2, ..., ExpertK] (共享)
       → Gate_i → Tower_i → Task_i 输出 (每任务独立)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    pass


# ====== 全部 6 个任务定义 ======
ALL_TASK_DEFS = [
    {'name': 'return_5d',   'type': 'regression',     'weight': 1.0,  'label_key': '5d'},
    {'name': 'return_20d',  'type': 'regression',     'weight': 1.5,  'label_key': '20d'},
    {'name': 'direction',   'type': 'classification', 'weight': 0.8,  'label_key': '_dir_5d'},
    {'name': 'max_dd',      'type': 'regression',     'weight': 1.0,  'label_key': '_max_dd'},
    {'name': 'rank_score',  'type': 'ranking',        'weight': 1.2,  'label_key': '_rank'},
    {'name': 'volatility',  'type': 'regression',     'weight': 0.6,  'label_key': '_vol'},
]

# 默认任务集 (向后兼容)
TASK_DEFS = ALL_TASK_DEFS


class ExpertNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class GateNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.gate(x)


class TowerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class MMoEModel(nn.Module):
    """Multi-gate Mixture of Experts — 6 任务"""

    def __init__(self,
                 input_dim: int,
                 num_experts: int = 6,
                 expert_hidden: int = 128,
                 expert_out: int = 64,
                 tower_hidden: int = 32,
                 num_tasks: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_hidden, expert_out, dropout)
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            GateNetwork(input_dim, num_experts)
            for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            TowerNetwork(expert_out, tower_hidden, 1, dropout)
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        task_outputs = []
        for i in range(self.num_tasks):
            gw = self.gates[i](x).unsqueeze(-1)
            mixed = (expert_outs * gw).sum(dim=1)
            task_outputs.append(self.towers[i](mixed).squeeze(-1))
        return task_outputs


class MMoEPredictor:
    """MMoE 训练 & 预测"""

    def __init__(self,
                 num_experts: int = 6,
                 expert_hidden: int = 128,
                 expert_out: int = 64,
                 tower_hidden: int = 32,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 epochs: int = 100,
                 batch_size: int = 256,
                 patience: int = 15,
                 device: str = 'auto',
                 task_defs: Optional[List] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("pip install torch")

        self.task_defs = task_defs or ALL_TASK_DEFS
        self.config = dict(
            num_experts=num_experts, expert_hidden=expert_hidden,
            expert_out=expert_out, tower_hidden=tower_hidden,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size, patience=patience,
        )
        if device == 'auto':
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available()
                else 'cuda' if torch.cuda.is_available()
                else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.feature_names: List[str] = []
        self.feature_mean = None
        self.feature_std = None
        self.label_stats: Dict = {}

    # --------------------------------------------------
    # 标签工程
    # --------------------------------------------------
    @staticmethod
    def _build_labels(returns_dict: Dict[str, np.ndarray],
                      drawdowns_dict: Dict[str, np.ndarray],
                      groups: np.ndarray) -> Dict[str, np.ndarray]:
        """从 pipeline 的 returns_dict / drawdowns_dict 构建 6 个标签"""
        n = len(groups)
        y = {}

        # 1) 5d / 20d 收益
        y['5d'] = returns_dict.get('5d', np.full(n, np.nan))
        y['20d'] = returns_dict.get('20d', np.full(n, np.nan))

        # 2) 方向 (5d)
        r5 = y['5d'].copy()
        y['_dir_5d'] = np.where(np.isnan(r5), np.nan, (r5 > 0).astype(float))

        # 3) 最大回撤 (取 20d drawdown，或从 5d 收益近似)
        dd20 = drawdowns_dict.get('20d', np.full(n, np.nan)) if drawdowns_dict else np.full(n, np.nan)
        if np.all(np.isnan(dd20)):
            # 用 5d 收益的负值近似
            dd20 = np.where(np.isnan(r5), np.nan, np.minimum(r5, 0))
        y['_max_dd'] = dd20

        # 4) 排序得分 (组内排名百分位)
        rank = np.full(n, np.nan)
        r_ref = y['20d'] if not np.all(np.isnan(y['20d'])) else y['5d']
        for g in set(groups):
            mask = groups == g
            vals = r_ref[mask]
            valid = ~np.isnan(vals)
            if valid.sum() >= 3:
                from scipy.stats import rankdata
                ranks = np.full(valid.sum(), np.nan)
                ranks = rankdata(vals[valid]) / valid.sum()  # 0~1 百分位
                idx = np.where(mask)[0][valid]
                rank[idx] = ranks
        y['_rank'] = rank

        # 5) 波动率 (用 5d 收益的绝对值近似)
        y['_vol'] = np.where(np.isnan(r5), np.nan, np.abs(r5))

        return y

    # --------------------------------------------------
    # 训练
    # --------------------------------------------------
    def train(self,
              X: np.ndarray,
              returns_dict: Dict[str, np.ndarray],
              feature_names: List[str],
              groups: np.ndarray,
              drawdowns_dict: Optional[Dict[str, np.ndarray]] = None,
              val_ratio: float = 0.2) -> Dict:

        self.feature_names = feature_names
        n_feat = X.shape[1]
        task_defs = self.task_defs
        num_tasks = len(task_defs)

        # 构建标签
        labels = self._build_labels(returns_dict, drawdowns_dict or {}, groups)

        # 有效样本
        valid = np.zeros(len(X), dtype=bool)
        for td in TASK_DEFS:
            lk = td['label_key']
            if lk in labels:
                valid |= ~np.isnan(labels[lk])
        X_v = X[valid]
        groups_v = groups[valid]
        labels_v = {k: v[valid] for k, v in labels.items()}

        print(f"\n{'='*50}")
        print(f"🧠 MMoE 多任务模型 ({num_tasks} 任务)")
        print(f"   样本: {len(X_v)}, 特征: {n_feat}, 专家: {self.config['num_experts']}")
        print(f"   设备: {self.device}")
        print(f"   任务: {', '.join(t['name'] for t in task_defs)}")
        print(f"{'='*50}")

        # 特征标准化
        self.feature_mean = np.nanmean(X_v, axis=0)
        self.feature_std = np.nanstd(X_v, axis=0) + 1e-8
        X_norm = (X_v - self.feature_mean) / self.feature_std
        X_norm = np.nan_to_num(X_norm, 0.0)

        # 标签标准化 (回归任务)
        Y_list = []
        for td in task_defs:
            arr = labels_v[td['label_key']].copy()
            arr = np.nan_to_num(arr, 0.0)
            if td['type'] in ('regression', 'ranking'):
                mean_v = float(np.mean(arr))
                std_v = float(np.std(arr) + 1e-8)
                self.label_stats[td['name']] = {'mean': mean_v, 'std': std_v}
                arr = (arr - mean_v) / std_v
            Y_list.append(arr)

        # Train/Val split (时序)
        ugroups = sorted(set(groups_v))
        split = int(len(ugroups) * (1 - val_ratio))
        train_g = set(ugroups[:split])
        tmask = np.array([g in train_g for g in groups_v])

        X_tr, X_va = X_norm[tmask], X_norm[~tmask]
        Y_tr = [y[tmask] for y in Y_list]
        Y_va = [y[~tmask] for y in Y_list]

        print(f"   训练: {len(X_tr)}, 验证: {len(X_va)}")

        # DataLoader
        tr_tensors = [torch.FloatTensor(X_tr)] + [torch.FloatTensor(y) for y in Y_tr]
        va_tensors = [torch.FloatTensor(X_va)] + [torch.FloatTensor(y) for y in Y_va]
        tr_loader = DataLoader(TensorDataset(*tr_tensors), batch_size=self.config['batch_size'], shuffle=True)
        va_loader = DataLoader(TensorDataset(*va_tensors), batch_size=self.config['batch_size'])

        # 模型
        self.model = MMoEModel(
            input_dim=n_feat, num_experts=self.config['num_experts'],
            expert_hidden=self.config['expert_hidden'],
            expert_out=self.config['expert_out'],
            tower_hidden=self.config['tower_hidden'],
            num_tasks=num_tasks, dropout=self.config['dropout'],
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])

        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        best_val = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(self.config['epochs']):
            # Train
            self.model.train()
            ep_loss = 0.0
            for batch in tr_loader:
                xb = batch[0].to(self.device)
                ybs = [batch[i+1].to(self.device) for i in range(num_tasks)]
                preds = self.model(xb)

                loss = torch.tensor(0.0, device=self.device)
                for i, td in enumerate(task_defs):
                    if td['type'] == 'classification':
                        loss += td['weight'] * bce(preds[i], ybs[i])
                    else:
                        loss += td['weight'] * mse(preds[i], ybs[i])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            scheduler.step()

            # Validate
            self.model.eval()
            vloss = 0.0
            all_preds = [[] for _ in range(num_tasks)]
            all_trues = [[] for _ in range(num_tasks)]

            with torch.no_grad():
                for batch in va_loader:
                    xb = batch[0].to(self.device)
                    ybs = [batch[i+1].to(self.device) for i in range(num_tasks)]
                    preds = self.model(xb)

                    bl = torch.tensor(0.0, device=self.device)
                    for i, td in enumerate(task_defs):
                        if td['type'] == 'classification':
                            bl += td['weight'] * bce(preds[i], ybs[i])
                        else:
                            bl += td['weight'] * mse(preds[i], ybs[i])
                    vloss += bl.item()

                    for i in range(num_tasks):
                        p = preds[i].cpu().numpy()
                        if task_defs[i]['type'] == 'classification':
                            p = 1 / (1 + np.exp(-p))  # sigmoid
                        all_preds[i].extend(p)
                        all_trues[i].extend(ybs[i].cpu().numpy())

            avg_vl = vloss / max(len(va_loader), 1)

            # direction accuracy (if direction task exists)
            dir_idx = next((i for i, t in enumerate(task_defs) if t['name'] == 'direction'), None)
            if dir_idx is not None:
                dir_pred = np.array(all_preds[dir_idx])
                dir_true = np.array(all_trues[dir_idx])
                dir_acc = np.mean((dir_pred > 0.5) == dir_true) if len(dir_true) > 0 else 0
            else:
                dir_acc = 0

            # 5d MAE (if return_5d task exists)
            ret5_idx = next((i for i, t in enumerate(task_defs) if t['name'] == 'return_5d'), None)
            if ret5_idx is not None and 'return_5d' in self.label_stats:
                p5 = np.array(all_preds[ret5_idx])
                t5 = np.array(all_trues[ret5_idx])
                s = self.label_stats['return_5d']
                p5r = p5 * s['std'] + s['mean']
                t5r = t5 * s['std'] + s['mean']
                mae5 = np.mean(np.abs(p5r - t5r))
            else:
                mae5 = 999

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}: val_loss={avg_vl:.4f}, "
                      f"dir_acc={dir_acc:.1%}, mae_5d={mae5:.2f}%")

            if avg_vl < best_val:
                best_val = avg_vl
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.config['patience']:
                    print(f"   ⏹ Early stop @ epoch {epoch+1}")
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()

        results = {
            'epochs': epoch + 1,
            'best_val_loss': best_val,
            'dir_accuracy': dir_acc,
            'mae_5d': mae5,
            'train_n': len(X_tr),
            'val_n': len(X_va),
            'device': str(self.device),
            'tasks': [t['name'] for t in task_defs],
        }
        print(f"\n✅ MMoE 训练完成: dir={dir_acc:.1%}, mae5={mae5:.2f}%")
        return results

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Not trained")
        Xn = np.nan_to_num((X - self.feature_mean) / self.feature_std, 0.0)
        xt = torch.FloatTensor(Xn).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(xt)

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

    def save(self, path: str):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / 'mmoe_model.pt')
        meta = {
            'config': self.config,
            'task_defs': self.task_defs,
            'feature_names': self.feature_names,
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'label_stats': self.label_stats,
        }
        with open(p / 'mmoe_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str):
        p = Path(path)
        with open(p / 'mmoe_meta.json') as f:
            meta = json.load(f)
        self.config = meta['config']
        self.task_defs = meta.get('task_defs', ALL_TASK_DEFS)
        self.feature_names = meta['feature_names']
        self.feature_mean = np.array(meta['feature_mean'])
        self.feature_std = np.array(meta['feature_std'])
        self.label_stats = meta['label_stats']
        self.model = MMoEModel(
            input_dim=len(self.feature_names),
            num_experts=self.config['num_experts'],
            expert_hidden=self.config['expert_hidden'],
            expert_out=self.config['expert_out'],
            tower_hidden=self.config['tower_hidden'],
            num_tasks=len(self.task_defs), dropout=0.0,
        ).to(self.device)
        sd = torch.load(p / 'mmoe_model.pt', map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.eval()
