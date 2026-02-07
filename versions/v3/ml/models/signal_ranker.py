"""
信号排序模型
Signal Ranker

使用 Learning to Rank 对信号进行排序
找出最可能赚钱的股票
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

try:
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class TradingHorizon(Enum):
    """交易周期"""
    SHORT = "short"   # 短线: 1-5天
    MEDIUM = "medium" # 中线: 10-30天
    LONG = "long"     # 长线: 60天+


@dataclass
class HorizonConfig:
    """周期配置"""
    name: str
    days: List[int]           # 计算收益的天数
    target_return: float      # 目标收益率 (%)
    stop_loss: float          # 止损线 (%)
    weight_return: float      # 收益权重
    weight_risk: float        # 风险权重


# 专业交易者的周期配置
HORIZON_CONFIGS = {
    TradingHorizon.SHORT: HorizonConfig(
        name="短线 (1-5天)",
        days=[1, 3, 5],
        target_return=3.0,    # 3% 目标
        stop_loss=-3.0,       # 3% 止损
        weight_return=0.7,    # 更看重快速收益
        weight_risk=0.3
    ),
    TradingHorizon.MEDIUM: HorizonConfig(
        name="中线 (10-30天)",
        days=[10, 20, 30],
        target_return=10.0,   # 10% 目标
        stop_loss=-5.0,       # 5% 止损
        weight_return=0.5,
        weight_risk=0.5
    ),
    TradingHorizon.LONG: HorizonConfig(
        name="长线 (60天+)",
        days=[30, 60],
        target_return=25.0,   # 25% 目标
        stop_loss=-8.0,       # 8% 止损
        weight_return=0.4,
        weight_risk=0.6       # 更看重风险控制
    )
}


class SignalRanker:
    """信号排序器 (Learning to Rank)"""
    
    def __init__(self):
        self.models: Dict[TradingHorizon, xgb.XGBRanker] = {}
        self.feature_names: List[str] = []
        self.metrics: Dict[TradingHorizon, Dict] = {}
        self.is_trained = False
    
    def _create_ranking_labels(self, 
                               returns: np.ndarray, 
                               max_drawdowns: np.ndarray,
                               config: HorizonConfig) -> np.ndarray:
        """
        创建排序标签
        
        综合考虑收益和风险:
        score = w_return * 收益分数 + w_risk * 风险分数
        
        收益分数: 基于收益率的百分位
        风险分数: 基于最大回撤的百分位 (越小越好)
        """
        n = len(returns)
        
        # 处理 NaN
        returns = np.nan_to_num(returns, nan=0.0)
        max_drawdowns = np.nan_to_num(max_drawdowns, nan=0.0)
        
        # 收益分数 (0-100, 越高越好)
        return_rank = pd.Series(returns).rank(pct=True) * 100
        
        # 风险分数 (0-100, 回撤越小分数越高)
        risk_rank = (1 - pd.Series(max_drawdowns).rank(pct=True)) * 100
        
        # 综合分数
        score = config.weight_return * return_rank + config.weight_risk * risk_rank
        
        # 处理 NaN
        score = score.fillna(score.median())
        
        # 转为整数标签 (0-4 五档)
        try:
            labels = pd.qcut(score, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
            return labels.values.astype(int)
        except:
            # 如果分位数失败，使用简单分档
            labels = pd.cut(score, bins=5, labels=[0, 1, 2, 3, 4])
            return labels.fillna(2).values.astype(int)
    
    def train(self,
              X: np.ndarray,
              returns_dict: Dict[str, np.ndarray],
              drawdowns_dict: Dict[str, np.ndarray],
              groups: np.ndarray,
              feature_names: List[str]) -> Dict:
        """
        训练排序模型
        
        Args:
            X: 特征矩阵
            returns_dict: 各周期收益率 {'1d': array, '5d': array, ...}
            drawdowns_dict: 各周期最大回撤 {'5d': array, '30d': array, ...}
            groups: 分组 (同一天的信号为一组)
            feature_names: 特征名称
        
        Returns:
            训练指标
        """
        if not ML_AVAILABLE:
            raise RuntimeError("XGBoost 未安装")
        
        self.feature_names = feature_names
        
        print(f"\n{'='*50}")
        print("🎯 信号排序模型训练 (Learning to Rank)")
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {len(feature_names)}")
        print(f"   分组数: {len(np.unique(groups))}")
        print(f"{'='*50}\n")
        
        for horizon, config in HORIZON_CONFIGS.items():
            print(f"📊 训练 {config.name} 排序模型...")
            
            # 计算该周期的综合收益
            horizon_returns = []
            for day in config.days:
                key = f'{day}d'
                if key in returns_dict:
                    horizon_returns.append(returns_dict[key])
            
            if not horizon_returns:
                print(f"   ⚠️ 跳过: 无收益数据")
                continue
            
            # 平均收益
            avg_returns = np.nanmean(horizon_returns, axis=0)
            
            # 最大回撤 (取最长周期的)
            max_day = max(config.days)
            dd_key = f'{max_day}d'
            if dd_key in drawdowns_dict:
                max_dd = drawdowns_dict[dd_key]
            else:
                max_dd = np.zeros_like(avg_returns)
            
            # 创建排序标签
            labels = self._create_ranking_labels(avg_returns, max_dd, config)
            
            # 过滤无效样本
            valid_mask = ~np.isnan(avg_returns)
            X_valid = X[valid_mask]
            y_valid = labels[valid_mask]
            groups_valid = groups[valid_mask]
            
            if len(X_valid) < 100:
                print(f"   ⚠️ 跳过: 样本不足")
                continue

            # 时序切分 (按日期组)，先做 OOS 评估再训练最终模型
            unique_groups = np.unique(groups_valid)
            unique_groups = np.sort(unique_groups)
            if len(unique_groups) < 10:
                print(f"   ⚠️ 跳过: 分组不足 ({len(unique_groups)})")
                continue

            split_idx = max(1, int(len(unique_groups) * 0.8))
            split_idx = min(split_idx, len(unique_groups) - 1)
            train_groups = unique_groups[:split_idx]
            test_groups = unique_groups[split_idx:]

            train_mask = np.isin(groups_valid, train_groups)
            test_mask = np.isin(groups_valid, test_groups)

            X_train = X_valid[train_mask]
            y_train = y_valid[train_mask]
            groups_train = groups_valid[train_mask]

            X_test = X_valid[test_mask]
            y_test = y_valid[test_mask]
            groups_test = groups_valid[test_mask]
            returns_test = avg_returns[valid_mask][test_mask]

            if len(X_train) < 100 or len(X_test) < 30:
                print(f"   ⚠️ 跳过: 训练/测试样本不足")
                continue

            # 训练 XGBoost Ranker (仅训练窗口)
            model = xgb.XGBRanker(
                objective='rank:pairwise',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(
                X_train, y_train,
                group=self._get_group_sizes(groups_train),
                verbose=False
            )
            
            # OOS 评估: NDCG@10 + Top10 平均收益
            scores_test = model.predict(X_test)
            ndcg = self._calculate_ndcg(y_test, scores_test, groups_test, k=10)
            
            # OOS 评估: Top 10 平均收益
            top10_return = self._calculate_top_k_return(
                returns_test, scores_test, groups_test, k=10
            )

            # 最终模型: 用全部样本训练，用于线上推理
            final_model = xgb.XGBRanker(
                objective='rank:pairwise',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            final_model.fit(
                X_valid, y_valid,
                group=self._get_group_sizes(groups_valid),
                verbose=False
            )
            
            self.models[horizon] = final_model
            self.metrics[horizon] = {
                'ndcg@10': ndcg,
                'top10_avg_return': top10_return,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_train_groups': len(np.unique(groups_train)),
                'n_test_groups': len(np.unique(groups_test))
            }
            
            print(f"   NDCG@10: {ndcg:.3f}")
            print(f"   Top10 平均收益: {top10_return:.2f}%")
            print()
        
        self.is_trained = len(self.models) > 0
        return self.metrics
    
    def _get_group_sizes(self, groups: np.ndarray) -> List[int]:
        """获取每个组的大小"""
        unique_groups = np.unique(groups)
        return [np.sum(groups == g) for g in unique_groups]
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        groups: np.ndarray, k: int = 10) -> float:
        """计算 NDCG@k"""
        unique_groups = np.unique(groups)
        ndcgs = []
        
        for g in unique_groups:
            mask = groups == g
            if mask.sum() < k:
                continue
            
            true = y_true[mask]
            pred = y_pred[mask]
            
            # 按预测分数排序
            order = np.argsort(-pred)[:k]
            dcg = np.sum((2**true[order] - 1) / np.log2(np.arange(2, k + 2)))
            
            # 理想排序
            ideal_order = np.argsort(-true)[:k]
            idcg = np.sum((2**true[ideal_order] - 1) / np.log2(np.arange(2, k + 2)))
            
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs) if ndcgs else 0
    
    def _calculate_top_k_return(self, returns: np.ndarray, scores: np.ndarray,
                                groups: np.ndarray, k: int = 10) -> float:
        """计算 Top K 的平均收益"""
        unique_groups = np.unique(groups)
        top_returns = []
        
        for g in unique_groups:
            mask = groups == g
            if mask.sum() < k:
                continue
            
            ret = returns[mask]
            pred = scores[mask]
            
            # 选择 Top K
            top_idx = np.argsort(-pred)[:k]
            top_returns.extend(ret[top_idx])
        
        return np.mean(top_returns) if top_returns else 0
    
    def rank(self, X: np.ndarray, horizon: TradingHorizon = TradingHorizon.SHORT) -> np.ndarray:
        """
        对信号进行排序
        
        Args:
            X: 特征矩阵
            horizon: 交易周期
        
        Returns:
            排序分数 (越高越好)
        """
        if horizon not in self.models:
            # 返回默认分数
            return np.zeros(len(X))
        
        return self.models[horizon].predict(X)
    
    def get_top_signals(self, 
                        df: pd.DataFrame,
                        X: np.ndarray,
                        horizon: TradingHorizon,
                        top_n: int = 10) -> pd.DataFrame:
        """
        获取 Top N 信号
        
        Args:
            df: 原始信号 DataFrame
            X: 特征矩阵
            horizon: 交易周期
            top_n: 返回数量
        
        Returns:
            排序后的 Top N DataFrame
        """
        scores = self.rank(X, horizon)
        
        result = df.copy()
        result['rank_score'] = scores
        result['rank'] = result['rank_score'].rank(ascending=False, method='first').astype(int)
        
        return result.nsmallest(top_n, 'rank')
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for horizon, model in self.models.items():
            joblib.dump(model, save_dir / f"ranker_{horizon.value}.joblib")
        
        metadata = {
            'feature_names': self.feature_names,
            'metrics': {h.value: m for h, m in self.metrics.items()},
            'horizons': [h.value for h in self.models.keys()]
        }
        with open(save_dir / "ranker_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 排序模型已保存: {path}")
    
    def load(self, path: str) -> bool:
        """加载模型"""
        import joblib
        
        save_dir = Path(path)
        meta_path = save_dir / "ranker_meta.json"
        
        if not meta_path.exists():
            return False
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.metrics = {TradingHorizon(h): m for h, m in metadata['metrics'].items()}
        
        for horizon_str in metadata['horizons']:
            horizon = TradingHorizon(horizon_str)
            model_path = save_dir / f"ranker_{horizon_str}.joblib"
            if model_path.exists():
                self.models[horizon] = joblib.load(model_path)
        
        self.is_trained = len(self.models) > 0
        print(f"✅ 排序模型已加载: {[h.value for h in self.models.keys()]}")
        return self.is_trained


# === 测试 ===
if __name__ == "__main__":
    print("=== Signal Ranker 测试 ===\n")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    n_days = 20  # 20个交易日
    
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'f_{i}' for i in range(n_features)]
    
    # 模拟收益和回撤
    returns_dict = {
        '1d': X[:, 0] * 0.5 + np.random.randn(n_samples) * 2,
        '5d': X[:, 0] * 1.0 + np.random.randn(n_samples) * 3,
        '10d': X[:, 0] * 1.5 + np.random.randn(n_samples) * 4,
        '30d': X[:, 0] * 2.0 + np.random.randn(n_samples) * 5,
        '60d': X[:, 0] * 3.0 + np.random.randn(n_samples) * 6,
    }
    
    drawdowns_dict = {
        '5d': np.abs(np.random.randn(n_samples) * 3),
        '30d': np.abs(np.random.randn(n_samples) * 5),
        '60d': np.abs(np.random.randn(n_samples) * 8),
    }
    
    # 分组 (每天25个信号)
    groups = np.repeat(np.arange(n_days), n_samples // n_days)
    
    ranker = SignalRanker()
    metrics = ranker.train(X, returns_dict, drawdowns_dict, groups, feature_names)
    
    print("\n=== 排序测试 ===")
    test_X = np.random.randn(10, n_features)
    for horizon in TradingHorizon:
        if horizon in ranker.models:
            scores = ranker.rank(test_X, horizon)
            print(f"{horizon.value}: {scores[:5].round(2)}")
