#!/usr/bin/env python
"""
MMoE Head 消融实验 + OOF Stacking
==================================
测试不同 head 组合 + out-of-fold XGBoost stacking

用法: /Users/bertwang/miniconda3/bin/python ml/test_ablation.py
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.pipeline import MLPipeline
from ml.models.mmoe import MMoEPredictor, ALL_TASK_DEFS

# ============================
# 数据准备 (只做一次)
# ============================
print("📊 准备数据...")
t0 = time.time()
p = MLPipeline(market='US', days_back=9999)
X, ret, dd, grp, fn, info = p.prepare_dataset()
print(f"Data: {time.time()-t0:.0f}s, X={X.shape}\n")

# ============================
# 实验 1: Head 消融
# ============================
EXPERIMENTS = {
    'A_full_6':     ALL_TASK_DEFS,
    'B_core_3':     [t for t in ALL_TASK_DEFS if t['name'] in ('return_5d', 'return_20d', 'direction')],
    'C_returns_only': [t for t in ALL_TASK_DEFS if t['name'] in ('return_5d', 'return_20d')],
    'D_no_vol':     [t for t in ALL_TASK_DEFS if t['name'] != 'volatility'],
    'E_no_rank':    [t for t in ALL_TASK_DEFS if t['name'] != 'rank_score'],
    'F_no_dd':      [t for t in ALL_TASK_DEFS if t['name'] != 'max_dd'],
    'G_ret5_dir':   [t for t in ALL_TASK_DEFS if t['name'] in ('return_5d', 'direction')],
    'H_ret_dd_dir': [t for t in ALL_TASK_DEFS if t['name'] in ('return_5d', 'return_20d', 'direction', 'max_dd')],
}

mmoe_kwargs = dict(
    num_experts=4, expert_hidden=64, expert_out=32, tower_hidden=16,
    dropout=0.2, lr=5e-4, weight_decay=1e-3,
    epochs=200, batch_size=128, patience=25,
)

results_table = []

for name, tasks in EXPERIMENTS.items():
    print(f"\n{'='*50}")
    print(f"📋 实验 {name}: {[t['name'] for t in tasks]}")
    print(f"{'='*50}")

    t1 = time.time()
    mmoe = MMoEPredictor(**mmoe_kwargs, task_defs=tasks)
    r = mmoe.train(X, ret, fn, grp, dd)
    elapsed = time.time() - t1

    results_table.append({
        'name': name,
        'tasks': len(tasks),
        'dir_acc': r['dir_accuracy'],
        'mae_5d': r['mae_5d'],
        'epochs': r['epochs'],
        'time': elapsed,
    })

# ============================
# 实验 2: OOF Stacking
# ============================
print(f"\n\n{'#'*60}")
print(f"## OOF XGBoost → MMoE Stacking")
print(f"{'#'*60}")

from xgboost import XGBRegressor
from sklearn.decomposition import PCA

y_5d = np.nan_to_num(ret.get('5d', np.zeros(len(X))), 0.0)
y_20d = np.nan_to_num(ret.get('20d', np.zeros(len(X))), 0.0)

# K-Fold (时序分组)
unique_groups = sorted(set(grp))
n_folds = 5
fold_size = len(unique_groups) // n_folds

oof_pred_5d = np.zeros(len(X))
oof_pred_20d = np.zeros(len(X))
oof_leaf = np.zeros((len(X), 32))  # PCA 降维后

print(f"\nOOF: {n_folds} folds, {len(unique_groups)} groups")

for fold in range(n_folds):
    # 验证集分组
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_groups)
    val_groups = set(unique_groups[val_start:val_end])
    val_mask = np.array([g in val_groups for g in grp])
    train_mask = ~val_mask

    X_tr, X_va = X[train_mask], X[val_mask]
    y5_tr, y5_va = y_5d[train_mask], y_5d[val_mask]
    y20_tr, y20_va = y_20d[train_mask], y_20d[val_mask]

    # XGBoost 5d
    xgb5 = XGBRegressor(max_depth=4, learning_rate=0.03, n_estimators=300,
                         subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
    xgb5.fit(X_tr, y5_tr, eval_set=[(X_va, y5_va)], verbose=False)
    oof_pred_5d[val_mask] = xgb5.predict(X_va)

    # XGBoost 20d
    xgb20 = XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=300,
                          subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
    xgb20.fit(X_tr, y20_tr, eval_set=[(X_va, y20_va)], verbose=False)
    oof_pred_20d[val_mask] = xgb20.predict(X_va)

    # 叶子编码 (OOF)
    leaf5 = xgb5.apply(X_va).astype(float)
    leaf20 = xgb20.apply(X_va).astype(float)
    leaf_all = np.hstack([leaf5, leaf20])

    pca = PCA(n_components=min(16, leaf_all.shape[1]))
    pca.fit(np.hstack([xgb5.apply(X_tr).astype(float), xgb20.apply(X_tr).astype(float)]))
    leaf_reduced = pca.transform(leaf_all)

    oof_leaf[val_mask, :leaf_reduced.shape[1]] = leaf_reduced

    mae5 = np.mean(np.abs(oof_pred_5d[val_mask] - y5_va))
    print(f"  Fold {fold+1}: {val_mask.sum()} samples, XGB mae5={mae5:.2f}%")

# 拼接 OOF 特征
xgb_oof_features = np.column_stack([
    oof_pred_5d.reshape(-1, 1),
    oof_pred_20d.reshape(-1, 1),
    (oof_pred_5d > 0).astype(float).reshape(-1, 1),
    np.abs(oof_pred_5d).reshape(-1, 1),
    oof_leaf,
])

X_stacked = np.hstack([X, xgb_oof_features])
stacked_fn = fn + ['xgb_oof_5d', 'xgb_oof_20d', 'xgb_oof_dir', 'xgb_oof_conf'] + [f'xgb_leaf_{i}' for i in range(32)]

print(f"\nOOF features: {X.shape[1]} + {xgb_oof_features.shape[1]} = {X_stacked.shape[1]}")

# MMoE on OOF stacked features
print("\n🧠 MMoE on OOF-stacked features")
mmoe_oof = MMoEPredictor(**mmoe_kwargs)
r_oof = mmoe_oof.train(X_stacked, ret, stacked_fn, grp, dd)

results_table.append({
    'name': 'OOF_Stacking',
    'tasks': 6,
    'dir_acc': r_oof['dir_accuracy'],
    'mae_5d': r_oof['mae_5d'],
    'epochs': r_oof['epochs'],
    'time': 0,
})

# ============================
# 结果汇总
# ============================
print(f"\n\n{'='*70}")
print(f"📊 实验结果汇总")
print(f"{'='*70}")
print(f"{'实验':<16s} {'Tasks':>5s} {'Dir Acc':>8s} {'MAE 5d':>8s} {'Epochs':>7s}")
print(f"{'-'*50}")

for r in sorted(results_table, key=lambda x: -x['dir_acc']):
    print(f"{r['name']:<16s} {r['tasks']:>5d} {r['dir_acc']:>7.1%} {r['mae_5d']:>7.2f}% {r['epochs']:>7d}")

best = max(results_table, key=lambda x: x['dir_acc'])
print(f"\n🏆 最佳: {best['name']} (dir_acc={best['dir_acc']:.1%}, mae_5d={best['mae_5d']:.2f}%)")
