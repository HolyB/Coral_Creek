"""
简化版 ML 训练
使用数据库已有的 scan_results 数据训练
不依赖外部 API
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_simple(market: str = 'US', days_back: int = 60):
    """
    简化版训练 - 只使用 scan_results 中的数据
    """
    from db.database import get_connection
    
    print(f"\n{'='*60}")
    print("🚀 简化版 ML 训练 (无外部 API)")
    print(f"   市场: {market}")
    print(f"   天数: {days_back}")
    print(f"{'='*60}")
    
    conn = get_connection()
    
    # 获取日期范围
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(scan_date) FROM scan_results WHERE market = ?", (market,))
    max_date = cursor.fetchone()[0]
    
    if not max_date:
        print("❌ 无数据")
        return None
    
    end_date = datetime.strptime(max_date, '%Y-%m-%d').date() - timedelta(days=5)
    start_date = end_date - timedelta(days=days_back)
    
    print(f"   日期范围: {start_date} ~ {end_date}")
    
    # 获取所有信号
    query = """
        SELECT symbol, scan_date, price, 
               COALESCE(blue_daily, 0) as blue_daily,
               COALESCE(blue_weekly, 0) as blue_weekly,
               COALESCE(blue_monthly, 0) as blue_monthly,
               COALESCE(is_heima, 0) as is_heima,
               COALESCE(is_juedi, 0) as is_juedi,
               COALESCE(volatility, 0) as volatility,
               COALESCE(adx, 0) as adx
        FROM scan_results
        WHERE market = ? AND scan_date >= ? AND scan_date <= ?
          AND price > 0
        ORDER BY symbol, scan_date
    """
    
    df = pd.read_sql_query(query, conn, params=(
        market, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    ))
    
    print(f"   原始记录: {len(df)}")
    
    if len(df) < 100:
        print("❌ 数据不足")
        conn.close()
        return None
    
    # 计算标签：5天后的收益
    # 对于每个信号，找同一股票5天后的价格
    results = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('scan_date')
        
        if len(symbol_df) < 2:
            continue
        
        for i in range(len(symbol_df) - 1):
            row = symbol_df.iloc[i]
            entry_price = row['price']
            entry_date = row['scan_date']
            
            # 找5天后的价格 (可能不精确，但用于演示)
            future_df = symbol_df[symbol_df['scan_date'] > entry_date].head(5)
            
            if len(future_df) > 0:
                exit_price = future_df.iloc[-1]['price']
                return_5d = (exit_price - entry_price) / entry_price * 100
                
                results.append({
                    'symbol': symbol,
                    'scan_date': entry_date,
                    'price': entry_price,
                    'blue_daily': row['blue_daily'],
                    'blue_weekly': row['blue_weekly'],
                    'blue_monthly': row['blue_monthly'],
                    'is_heima': row['is_heima'],
                    'is_juedi': row['is_juedi'],
                    'volatility': row['volatility'],
                    'adx': row['adx'],
                    'return_5d': return_5d,
                    'is_win': 1 if return_5d > 0 else 0
                })
    
    conn.close()
    
    if len(results) < 50:
        print("❌ 有效样本不足")
        return None
    
    result_df = pd.DataFrame(results)
    print(f"   有效样本: {len(result_df)}")
    print(f"   胜率: {result_df['is_win'].mean():.1%}")
    print(f"   平均收益: {result_df['return_5d'].mean():.2f}%")
    
    # 创建特征
    feature_cols = []
    
    # BLUE 特征
    result_df['blue_daily'] = pd.to_numeric(result_df['blue_daily'], errors='coerce').fillna(0)
    result_df['blue_weekly'] = pd.to_numeric(result_df['blue_weekly'], errors='coerce').fillna(0)
    result_df['blue_monthly'] = pd.to_numeric(result_df['blue_monthly'], errors='coerce').fillna(0)
    feature_cols.extend(['blue_daily', 'blue_weekly', 'blue_monthly'])
    
    # BLUE 衍生特征
    result_df['blue_dw_ratio'] = result_df['blue_daily'] / (result_df['blue_weekly'] + 1)
    result_df['blue_dw_resonance'] = ((result_df['blue_daily'] >= 100) & (result_df['blue_weekly'] >= 100)).astype(int)
    result_df['blue_dwm_resonance'] = ((result_df['blue_daily'] >= 100) & (result_df['blue_weekly'] >= 100) & (result_df['blue_monthly'] >= 100)).astype(int)
    feature_cols.extend(['blue_dw_ratio', 'blue_dw_resonance', 'blue_dwm_resonance'])
    
    # 信号特征
    result_df['is_heima'] = pd.to_numeric(result_df['is_heima'], errors='coerce').fillna(0).astype(int)
    result_df['is_juedi'] = pd.to_numeric(result_df['is_juedi'], errors='coerce').fillna(0).astype(int)
    feature_cols.extend(['is_heima', 'is_juedi'])
    
    # 其他特征
    result_df['log_price'] = np.log1p(result_df['price'])
    result_df['volatility'] = pd.to_numeric(result_df['volatility'], errors='coerce').fillna(0)
    result_df['adx'] = pd.to_numeric(result_df['adx'], errors='coerce').fillna(0)
    feature_cols.extend(['log_price', 'volatility', 'adx'])
    
    # 信号强度
    result_df['signal_strength'] = (
        (result_df['blue_daily'] >= 100).astype(int) +
        (result_df['blue_weekly'] >= 100).astype(int) +
        (result_df['blue_monthly'] >= 100).astype(int) +
        result_df['is_heima'] * 2
    )
    feature_cols.append('signal_strength')
    
    # 时间特征
    result_df['scan_date'] = pd.to_datetime(result_df['scan_date'])
    result_df['day_of_week'] = result_df['scan_date'].dt.dayofweek
    result_df['month'] = result_df['scan_date'].dt.month
    feature_cols.extend(['day_of_week', 'month'])
    
    print(f"   特征数: {len(feature_cols)}")
    
    # 准备训练数据
    X = result_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    y_class = result_df['is_win'].values
    y_reg = result_df['return_5d'].values
    
    # 训练分类模型
    print(f"\n{'='*60}")
    print("📊 训练分类模型 (预测涨跌)")
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # 处理不平衡
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")
        print(f"   正样本比例: {y_train.mean():.1%}")
        print(f"   不平衡权重: {scale_pos_weight:.1f}")
        
        model_class = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model_class.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = model_class.predict(X_test)
        y_prob = model_class.predict_proba(X_test)[:, 1]
        
        print(f"\n📈 分类模型性能:")
        print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
        print(f"   Precision: {precision_score(y_test, y_pred):.3f}")
        print(f"   Recall:    {recall_score(y_test, y_pred):.3f}")
        print(f"   F1:        {f1_score(y_test, y_pred):.3f}")
        print(f"   AUC:       {roc_auc_score(y_test, y_prob):.3f}")
        
        # 特征重要性
        print(f"\n🔍 特征重要性:")
        importance = dict(zip(feature_cols, model_class.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {feat}: {imp:.3f}")
        
        # 训练回归模型
        print(f"\n{'='*60}")
        print("📊 训练回归模型 (预测收益率)")
        
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        model_reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        model_reg.fit(X_train_r, y_train_r, eval_set=[(X_test_r, y_test_r)], verbose=False)
        
        y_pred_r = model_reg.predict(X_test_r)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        mae = mean_absolute_error(y_test_r, y_pred_r)
        r2 = r2_score(y_test_r, y_pred_r)
        direction_acc = ((y_pred_r > 0) == (y_test_r > 0)).mean()
        
        print(f"\n📈 回归模型性能:")
        print(f"   RMSE:      {rmse:.2f}%")
        print(f"   MAE:       {mae:.2f}%")
        print(f"   R²:        {r2:.3f}")
        print(f"   方向准确率: {direction_acc:.1%}")
        
        # 保存模型
        import joblib
        
        model_dir = Path(__file__).parent / "saved_models" / f"simple_{market.lower()}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model_class, model_dir / "classifier.joblib")
        joblib.dump(model_reg, model_dir / "regressor.joblib")
        
        import json
        with open(model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_cols, f)
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump({
                'market': market,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': feature_cols,
                'classifier_auc': float(roc_auc_score(y_test, y_prob)),
                'regressor_r2': float(r2),
                'direction_accuracy': float(direction_acc)
            }, f, indent=2)
        
        print(f"\n✅ 模型已保存到: {model_dir}")
        
        return {
            'status': 'success',
            'samples': len(result_df),
            'classifier_auc': roc_auc_score(y_test, y_prob),
            'regressor_r2': r2,
            'direction_accuracy': direction_acc
        }
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='US')
    parser.add_argument('--days', type=int, default=60)
    
    args = parser.parse_args()
    
    result = train_simple(args.market, args.days)
    print(f"\n结果: {result}")
