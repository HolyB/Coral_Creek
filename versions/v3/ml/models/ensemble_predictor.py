"""
集成预测器 - 串联模型
Ensemble Predictor

将 ReturnPredictor 的预测结果作为 SignalRanker 的输入特征
实现两阶段预测：预测收益 -> 综合排序
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .return_predictor import ReturnPredictor
from .signal_ranker import SignalRanker, TradingHorizon, HORIZON_CONFIGS


class EnsemblePredictor:
    """
    集成预测器 (串联模型)
    
    架构:
    Stage 1: ReturnPredictor 预测各周期收益
    Stage 2: SignalRanker 综合原始特征 + 预测收益进行排序
    
    优势:
    - Ranker 可以学习 "哪些预测更可信"
    - 结合收益预测和风险控制
    """
    
    def __init__(self):
        self.return_predictor = ReturnPredictor()
        self.signal_ranker = SignalRanker()
        self.is_trained = False
        self.metrics: Dict = {}
        
    def train(self,
              X: np.ndarray,
              returns_dict: Dict[str, np.ndarray],
              drawdowns_dict: Dict[str, np.ndarray],
              groups: np.ndarray,
              feature_names: List[str]) -> Dict:
        """
        两阶段训练
        
        Args:
            X: 原始特征矩阵
            returns_dict: 各周期实际收益
            drawdowns_dict: 各周期最大回撤
            groups: 分组信息
            feature_names: 特征名称
        
        Returns:
            训练指标
        """
        if not ML_AVAILABLE:
            raise RuntimeError("XGBoost 未安装")
        
        print("\n" + "="*60)
        print("🔗 集成模型训练 (串联模式)")
        print("="*60)
        
        # ========== Stage 1: 训练 ReturnPredictor ==========
        print("\n📊 Stage 1: 训练收益预测模型")
        
        stage1_metrics = self.return_predictor.train(
            X, returns_dict, feature_names
        )
        
        # ========== 生成预测特征 ==========
        print("\n🔧 生成预测特征...")
        
        # 用训练好的模型预测
        pred_returns = self.return_predictor.predict(X)
        
        # 构建增强特征矩阵
        # 原始特征 + 预测收益 + 预测特征交互
        enhanced_features = []
        enhanced_names = list(feature_names)
        
        for horizon, pred in pred_returns.items():
            enhanced_features.append(pred.reshape(-1, 1))
            enhanced_names.append(f'pred_return_{horizon}')
        
        # 添加预测收益的统计特征
        pred_array = np.column_stack(list(pred_returns.values()))
        
        # 预测收益均值
        pred_mean = np.nanmean(pred_array, axis=1).reshape(-1, 1)
        enhanced_features.append(pred_mean)
        enhanced_names.append('pred_return_mean')
        
        # 预测收益标准差 (不确定性)
        pred_std = np.nanstd(pred_array, axis=1).reshape(-1, 1)
        enhanced_features.append(pred_std)
        enhanced_names.append('pred_return_std')
        
        # 短期 vs 长期预测差异
        if '1d' in pred_returns and '30d' in pred_returns:
            momentum = (pred_returns['30d'] - pred_returns['1d']).reshape(-1, 1)
            enhanced_features.append(momentum)
            enhanced_names.append('pred_momentum')
        
        # 预测方向一致性
        directions = np.sign(pred_array)
        consistency = np.abs(np.mean(directions, axis=1)).reshape(-1, 1)
        enhanced_features.append(consistency)
        enhanced_names.append('pred_direction_consistency')
        
        # 合并特征
        X_enhanced = np.hstack([X] + enhanced_features)
        
        print(f"   原始特征: {X.shape[1]}")
        print(f"   增强特征: {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})")
        
        # ========== Stage 2: 训练 SignalRanker ==========
        print("\n📊 Stage 2: 训练排序模型 (使用增强特征)")
        
        stage2_metrics = self.signal_ranker.train(
            X_enhanced, returns_dict, drawdowns_dict, groups, enhanced_names
        )
        
        # 汇总指标
        self.metrics = {
            'stage1': stage1_metrics,
            'stage2': {h.value: m for h, m in stage2_metrics.items()} if stage2_metrics else {},
            'n_original_features': len(feature_names),
            'n_enhanced_features': len(enhanced_names),
            'added_features': [n for n in enhanced_names if n not in feature_names]
        }
        
        self.is_trained = self.return_predictor.is_trained and self.signal_ranker.is_trained
        
        print("\n" + "="*60)
        print("✅ 集成模型训练完成")
        print(f"   增强特征: {self.metrics['added_features']}")
        print("="*60)
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """预测收益 (Stage 1)"""
        return self.return_predictor.predict(X)
    
    def rank(self, X: np.ndarray, horizon: TradingHorizon = TradingHorizon.SHORT) -> np.ndarray:
        """
        排序 (Stage 1 + Stage 2)
        
        自动将原始特征增强后传给 Ranker
        """
        if not self.is_trained:
            return np.zeros(len(X))
        
        # 生成增强特征
        X_enhanced = self._enhance_features(X)
        
        return self.signal_ranker.rank(X_enhanced, horizon)
    
    def _enhance_features(self, X: np.ndarray) -> np.ndarray:
        """生成增强特征"""
        pred_returns = self.return_predictor.predict(X)
        
        enhanced = [X]
        
        # 预测收益
        for horizon, pred in pred_returns.items():
            enhanced.append(pred.reshape(-1, 1))
        
        # 统计特征
        pred_array = np.column_stack(list(pred_returns.values()))
        
        pred_mean = np.nanmean(pred_array, axis=1).reshape(-1, 1)
        enhanced.append(pred_mean)
        
        pred_std = np.nanstd(pred_array, axis=1).reshape(-1, 1)
        enhanced.append(pred_std)
        
        if '1d' in pred_returns and '30d' in pred_returns:
            momentum = (pred_returns['30d'] - pred_returns['1d']).reshape(-1, 1)
            enhanced.append(momentum)
        
        directions = np.sign(pred_array)
        consistency = np.abs(np.mean(directions, axis=1)).reshape(-1, 1)
        enhanced.append(consistency)
        
        return np.hstack(enhanced)
    
    def get_top_signals(self,
                        df: pd.DataFrame,
                        X: np.ndarray,
                        horizon: TradingHorizon,
                        top_n: int = 10) -> pd.DataFrame:
        """获取 Top N 信号"""
        X_enhanced = self._enhance_features(X)
        return self.signal_ranker.get_top_signals(df, X_enhanced, horizon, top_n)
    
    def save(self, path: str):
        """保存模型"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存两个子模型
        self.return_predictor.save(str(save_dir / 'return_predictor'))
        self.signal_ranker.save(str(save_dir / 'signal_ranker'))
        
        # 保存元数据
        with open(save_dir / 'ensemble_meta.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ 集成模型已保存: {path}")
    
    def load(self, path: str) -> bool:
        """加载模型"""
        save_dir = Path(path)
        
        if not (save_dir / 'ensemble_meta.json').exists():
            return False
        
        # 加载子模型
        self.return_predictor.load(str(save_dir / 'return_predictor'))
        self.signal_ranker.load(str(save_dir / 'signal_ranker'))
        
        # 加载元数据
        with open(save_dir / 'ensemble_meta.json') as f:
            self.metrics = json.load(f)
        
        self.is_trained = self.return_predictor.is_trained and self.signal_ranker.is_trained
        
        print(f"✅ 集成模型已加载")
        return self.is_trained


def compare_models(X: np.ndarray,
                   returns_dict: Dict[str, np.ndarray],
                   drawdowns_dict: Dict[str, np.ndarray],
                   groups: np.ndarray,
                   feature_names: List[str]) -> Dict:
    """
    对比独立模型 vs 串联模型
    
    Returns:
        对比结果
    """
    print("\n" + "="*70)
    print("📊 模型对比: 独立模式 vs 串联模式")
    print("="*70)
    
    results = {
        'independent': {},
        'ensemble': {}
    }
    
    # ========== 独立模型 ==========
    print("\n" + "-"*50)
    print("🔹 训练独立模型")
    print("-"*50)
    
    independent_predictor = ReturnPredictor()
    independent_ranker = SignalRanker()
    
    pred_metrics = independent_predictor.train(X, returns_dict, feature_names)
    rank_metrics = independent_ranker.train(X, returns_dict, drawdowns_dict, groups, feature_names)
    
    results['independent'] = {
        'predictor': pred_metrics,
        'ranker': {h.value: m for h, m in rank_metrics.items()} if rank_metrics else {}
    }
    
    # ========== 串联模型 ==========
    print("\n" + "-"*50)
    print("🔹 训练串联模型")
    print("-"*50)
    
    ensemble = EnsemblePredictor()
    ensemble_metrics = ensemble.train(X, returns_dict, drawdowns_dict, groups, feature_names)
    
    results['ensemble'] = ensemble_metrics
    
    # ========== 对比结果 ==========
    print("\n" + "="*70)
    print("📈 对比结果")
    print("="*70)
    
    # 对比排序模型的 NDCG
    print("\n排序模型 NDCG@10 对比:")
    print(f"{'周期':<15} {'独立模型':<15} {'串联模型':<15} {'提升':<10}")
    print("-" * 55)
    
    comparison = []
    for horizon in ['short', 'medium', 'long']:
        ind_ndcg = results['independent']['ranker'].get(horizon, {}).get('ndcg@10', 0)
        ens_ndcg = results['ensemble']['stage2'].get(horizon, {}).get('ndcg@10', 0)
        improvement = (ens_ndcg - ind_ndcg) * 100
        
        print(f"{horizon:<15} {ind_ndcg:<15.3f} {ens_ndcg:<15.3f} {improvement:+.1f}%")
        
        comparison.append({
            'horizon': horizon,
            'independent_ndcg': ind_ndcg,
            'ensemble_ndcg': ens_ndcg,
            'improvement': improvement
        })
    
    results['comparison'] = comparison
    
    return results


# === 测试 ===
if __name__ == "__main__":
    print("=== Ensemble Predictor 测试 ===\n")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    n_days = 20
    
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'f_{i}' for i in range(n_features)]
    
    returns_dict = {
        '1d': X[:, 0] * 0.5 + np.random.randn(n_samples) * 2,
        '5d': X[:, 0] * 1.0 + np.random.randn(n_samples) * 3,
        '10d': X[:, 0] * 1.5 + np.random.randn(n_samples) * 4,
        '30d': X[:, 0] * 2.0 + np.random.randn(n_samples) * 5,
    }
    
    drawdowns_dict = {
        '5d': np.abs(np.random.randn(n_samples) * 3),
        '30d': np.abs(np.random.randn(n_samples) * 5),
        '60d': np.abs(np.random.randn(n_samples) * 8),
    }
    
    groups = np.repeat(np.arange(n_days), n_samples // n_days)
    
    # 对比
    results = compare_models(X, returns_dict, drawdowns_dict, groups, feature_names)
