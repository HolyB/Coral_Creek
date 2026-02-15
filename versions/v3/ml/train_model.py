"""
模型训练脚本 (Model Training Script)
=====================================
加载 Feature Pipeline 生成的数据集，训练 XGBoost Ranker 模型。
评估模型在不同时间范围内的排序能力 (NDCG@10)。

流程:
1. 加载 data/ml/dataset_v1.joblib
2. 初始化 SignalRanker (Learning to Rank)
3. 训练模型 (Short/Mid/Long Term horizons)
4. 输出评估指标 (NDCG, Top-K Return)
5. 保存模型到 versions/v3/ml/models/
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ml.models.signal_ranker import SignalRanker

# 配置
DATASET_PATH = os.path.join(parent_dir, 'data', 'ml', 'dataset_v1.joblib')
MODEL_DIR = os.path.join(parent_dir, 'ml', 'models', 'trained')

def train():
    print("🚀 Starting Model Training Pipeline...")
    
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}. Run train_data_pipeline.py first.")
        return
        
    print(f"📂 Loading dataset from {DATASET_PATH}...")
    data = joblib.load(DATASET_PATH)
    
    X = data['X']
    returns = data['returns']
    drawdowns = data['drawdowns']
    groups = data['groups']
    feature_names = data['feature_names']
    meta = data['meta']
    
    print(f"✅ Data loaded. Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Unique Groups (Dates): {len(groups)}")
    
    # 2. Add Meta info to X (Wait, SignalRanker doesn't use meta directly for training features, 
    #    but we might need date for TimeSeriesSplit inside train)
    #    Actually SignalRanker internal logic uses groups array to split, assuming groups are chronological.
    #    Since we sorted by Date in pipeline, groups are chronological. This is correct.
    
    # Create per-sample group IDs from counts
    # The ranker expects an array aligned with X, indicating which group each sample belongs to.
    group_counts = groups
    group_ids = []
    for i, count in enumerate(group_counts):
        group_ids.extend([i] * count)
    group_ids = np.array(group_ids)
    
    print(f"   Expanded Groups: {len(group_ids)} samples (aligned with X)")

    # Handle numeric instability
    print("🧹 Cleaning infinite values...")
    is_inf = np.isinf(X)
    if np.any(is_inf):
        print(f"   Found {np.sum(is_inf)} infinite values, replacing with NaN")
        X[is_inf] = np.nan

    # 3. Initialize Ranker
    ranker = SignalRanker()
    
    # 4. Train
    print("\n🧠 Training SignalRanker (XGBoost LTR)...")
    start_time = datetime.now()
    
    metrics = ranker.train(
        X=X,
        returns_dict=returns,
        drawdowns_dict=drawdowns,
        groups=group_ids,  # Pass per-sample IDs
        feature_names=feature_names
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Training completed in {duration:.1f}s")
    
    # 5. Show Metrics
    print("\n🏆 Model Performance (Start Date Split Validation):")
    for horizon, m in metrics.items():
        print(f"\n   Horizon: {horizon}")
        print(f"     NDCG@10:       {m.get('ndcg@10', 0):.4f} (Random guess is usually around 0.3-0.5 depending on distribution)")
        print(f"     Top-10 Return: {m.get('top10_avg_return', 0)*100:.2f}% (Avg return of top 10 predicted stocks)")
        
    # 6. Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'signal_ranker_v1')
    ranker.save(save_path)
    print(f"\n💾 Model saved to {save_path}")

if __name__ == "__main__":
    train()
