"""
训练数据构建流水线 (Training Data Pipeline)
===========================================
从 Parquet 缓存读取历史数据，应用 FeatureEngineer，生成训练集 (X, y, groups)。
输出格式为 joblib，可直接供 SignalRanker 训练。

流程:
1. 遍历 data/parquet/us/*.parquet
2. 并行处理：Feature Engineering + Label Generation
3. 合并数据，按日期排序
4. 生成 Query Groups (用于 Learning to Rank)
5. 划分 Train/Val/Test
6. 保存到 data/ml/dataset_v1.joblib
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ml.feature_engineering import FeatureEngineer

# 配置
PARQUET_DIR = os.path.join(parent_dir, 'data', 'parquet', 'us')
OUTPUT_DIR = os.path.join(parent_dir, 'data', 'ml')
MIN_DATE = '2022-01-01'
MAX_WORKERS = max(1, os.cpu_count() - 2)

def process_single_stock(file_path: str) -> pd.DataFrame:
    """
    处理单只股票：读取 -> 特征工程 -> 生成标签 -> 清洗
    """
    try:
        df = pd.read_parquet(file_path)
        if df.empty or len(df) < 100: # 至少要有足够得历史计算 MA100
            return None
            
        # 确保按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 过滤日期 (减少内存占用)
        # 注意: 我们需要保留 MIN_DATE 之前的数据用于计算 MA，
        # 所以先计算特征，再过滤日期。
        
        # 1. 特征工程
        fe = FeatureEngineer()
        # 这里暂时不传入 market_df，后续可以优化的点
        df = fe.transform(df)
        
        # 2. 生成标签 (Labels)
        # 预测未来 5天, 10天, 20天 收益率
        for h in [5, 10, 20]:
            df = fe.create_labels(df, horizon=h)
            
        # 3. 过滤无效数据
        # 去除 NaN (由于 rolling window 和 shift 产生的)
        df = df.dropna()
        
        # 4. 截取时间范围 (只保留 2022 之后的数据用于训练)
        df = df[df['date'] >= pd.Timestamp(MIN_DATE)]
        
        if df.empty:
            return None
            
        # 5. 内存优化: Float64 -> Float32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
            
        # 只保留需要的列
        # Features + Labels + Metadata
        # Metadata: date, symbol
        # Labels: ret_5d, dd_5d, ...
        # Features: feature_names
        
        return df
        
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        return None

def build_dataset():
    """构建完整数据集"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取 Parquet 文件列表
    files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    print(f"Found {len(files)} parquet files in {PARQUET_DIR}")
    
    if not files:
        print("❌ No data found! Run backfill first.")
        return
        
    all_dfs = []
    
    print(f"🚀 Processing stocks with {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_stock, f): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), unit="stock"):
            res = future.result()
            if res is not None:
                all_dfs.append(res)
                
    if not all_dfs:
        print("❌ No valid data generated.")
        return

    print("🔄 Concatenating dataframes...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"✅ Raw Dataset Shape: {full_df.shape}")
    
    # 按照 Date 排序 (对于 Learning to Rank 至关重要)
    # XGBoost Ranker 要求同一个 group (query) 的数据必须连续存放
    print("🔄 Sorting by Date...")
    full_df = full_df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    # 提取 Feature Names
    # 假设除了 date, symbol, ret_*, dd_* 之外的都是特征
    exclude_cols = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
    label_cols = {c for c in full_df.columns if c.startswith('ret_') or c.startswith('dd_')}
    feature_cols = [c for c in full_df.columns if c not in exclude_cols and c not in label_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols[:5]} ...")
    
    # 构建 X, y, groups
    print("📦 Building (X, y, groups)...")
    
    # X
    X = full_df[feature_cols].values
    
    # y (Dict of multiple horizons)
    returns_dict = {}
    drawdowns_dict = {}
    for h in [5, 10, 20]:
        if f'ret_{h}d' in full_df.columns:
            returns_dict[f'{h}d'] = full_df[f'ret_{h}d'].values
        if f'dd_{h}d' in full_df.columns:
            drawdowns_dict[f'{h}d'] = full_df[f'dd_{h}d'].values
            
    # Groups (每个日期有多少个股票)
    # 这种方法比 groupby 稍微快一点
    group_counts = full_df.groupby('date').size().values
    
    # 另外保留日期和股票代码，用于回测分析
    meta_df = full_df[['date', 'symbol']].copy()
    
    # 保存
    print("💾 Saving to disk...")
    save_path = os.path.join(OUTPUT_DIR, 'dataset_v1.joblib')
    joblib.dump({
        'X': X,
        'returns': returns_dict,
        'drawdowns': drawdowns_dict,
        'groups': group_counts,
        'feature_names': feature_cols,
        'meta': meta_df
    }, save_path)
    
    print(f"✅ Dataset saved to {save_path}")
    print(f"   X shape: {X.shape}")
    print(f"   Unique dates: {len(group_counts)}")

if __name__ == "__main__":
    build_dataset()
