#!/usr/bin/env python
"""快速测试 XGB→MMoE Stacking — 先 kill 掉其他 python 再跑"""
import warnings; warnings.filterwarnings('ignore')
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

t0 = time.time()
from ml.pipeline import MLPipeline
from ml.models.xgb_mmoe_stacker import XGBMMoEStacker

print("准备数据...")
p = MLPipeline(market='US', days_back=9999)
X, ret, dd, grp, fn, info = p.prepare_dataset()
print(f'Data: {time.time()-t0:.0f}s, X={X.shape}')

print("\n训练 Stacking...")
stacker = XGBMMoEStacker(
    xgb_max_depth=4, xgb_lr=0.03, xgb_n_estimators=300, xgb_leaf_dim=16,
    num_experts=4, expert_hidden=64, expert_out=32, tower_hidden=16,
    dropout=0.2, mmoe_lr=5e-4, mmoe_epochs=200, mmoe_patience=25,
)
results = stacker.train(X, ret, fn, grp, dd)
stacker.save('ml/saved_models/v2_us_stacker')

s2 = results['stage2_mmoe']
print(f"\n{'='*50}")
print(f"Total: {time.time()-t0:.0f}s")
print(f"Features: {results['original_features']} + {results['xgb_added_features']} = {results['total_features']}")
print(f"XGB MAE5: {results['stage1_xgb']['mae_5d']:.2f}%")
print(f"Stacker dir_acc: {s2['dir_accuracy']:.1%}, MAE5: {s2['mae_5d']:.2f}%")
print(f"{'='*50}")
