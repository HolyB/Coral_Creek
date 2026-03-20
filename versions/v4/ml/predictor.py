"""
信号预测器 - 推理模块
Signal Predictor - Inference Module

功能:
- 加载训练好的模型
- 对新信号进行预测
- 返回盈利概率排序
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class SignalPredictor:
    """信号预测器"""
    
    def __init__(self, market: str = 'US'):
        self.market = market
        self.model = None
        self.feature_names = None
        self.metadata = None
        self._loaded = False
    
    def load(self, from_hub: bool = False) -> bool:
        """
        加载模型
        
        Args:
            from_hub: 是否从 HuggingFace Hub 下载
        
        Returns:
            是否加载成功
        """
        try:
            from ml.model_registry import load_model
            
            model_name = f"xgb_signal_{self.market.lower()}"
            self.model, self.metadata = load_model(model_name, from_hub=from_hub)
            
            # 加载特征名称
            model_dir = Path(__file__).parent / "saved_models" / model_name
            feature_file = model_dir / "feature_names.json"
            
            if feature_file.exists():
                with open(feature_file) as f:
                    self.feature_names = json.load(f)
            elif 'feature_names' in self.metadata:
                self.feature_names = self.metadata['feature_names']
            
            self._loaded = True
            print(f"✅ 模型已加载: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded and self.model is not None
    
    def _prepare_features(self, signals_df: pd.DataFrame) -> np.ndarray:
        """准备特征向量"""
        if self.feature_names is None:
            raise RuntimeError("特征名称未加载")
        
        df = signals_df.copy()
        
        # 创建与训练时相同的特征
        # 1. BLUE 信号
        for col in ['blue_daily', 'blue_weekly', 'blue_monthly']:
            if col not in df.columns:
                # 尝试从其他列名获取
                alt_names = {
                    'blue_daily': ['Day_BLUE', 'day_blue'],
                    'blue_weekly': ['Week_BLUE', 'week_blue'],
                    'blue_monthly': ['Month_BLUE', 'month_blue']
                }
                for alt in alt_names.get(col, []):
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
                else:
                    df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 2. 组合特征
        df['blue_daily_weekly_ratio'] = df['blue_daily'] / (df['blue_weekly'] + 1)
        df['blue_resonance'] = ((df['blue_daily'] >= 100) & (df['blue_weekly'] >= 100)).astype(int)
        
        # 3. 黑马/绝地
        for col in ['is_heima', 'is_juedi']:
            if col not in df.columns:
                alt_names = {'is_heima': ['Heima'], 'is_juedi': ['Juedi']}
                for alt in alt_names.get(col, []):
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
                else:
                    df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # 4. 价格
        price_col = 'price' if 'price' in df.columns else 'Close'
        if price_col in df.columns:
            df['log_price'] = np.log1p(pd.to_numeric(df[price_col], errors='coerce').fillna(0))
        else:
            df['log_price'] = 0
        
        # 5. 星级
        if 'star_rating' not in df.columns:
            df['star_rating'] = 0
        df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce').fillna(0)
        
        # 6. 时间特征
        if 'scan_date' in df.columns:
            df['scan_date'] = pd.to_datetime(df['scan_date'])
            df['day_of_week'] = df['scan_date'].dt.dayofweek
            df['month'] = df['scan_date'].dt.month
        else:
            from datetime import date
            today = date.today()
            df['day_of_week'] = today.weekday()
            df['month'] = today.month
        
        # 提取特征
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def predict(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        预测信号盈利概率
        
        Args:
            signals_df: 信号数据 DataFrame
        
        Returns:
            添加了 ml_prob 和 ml_rank 列的 DataFrame
        """
        if not self.is_loaded():
            if not self.load():
                # 返回原始数据，无预测
                signals_df['ml_prob'] = 0.5
                signals_df['ml_rank'] = range(1, len(signals_df) + 1)
                return signals_df
        
        try:
            X = self._prepare_features(signals_df)
            probs = self.model.predict_proba(X)[:, 1]
            
            result = signals_df.copy()
            result['ml_prob'] = probs
            result['ml_rank'] = result['ml_prob'].rank(ascending=False, method='first').astype(int)
            
            return result.sort_values('ml_prob', ascending=False)
            
        except Exception as e:
            print(f"⚠️ 预测失败: {e}")
            signals_df['ml_prob'] = 0.5
            signals_df['ml_rank'] = range(1, len(signals_df) + 1)
            return signals_df
    
    def get_top_signals(self, signals_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """获取 Top N 高概率信号"""
        result = self.predict(signals_df)
        return result.head(top_n)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.is_loaded():
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'market': self.market,
            'features': len(self.feature_names) if self.feature_names else 0,
            'metrics': {
                'accuracy': self.metadata.get('accuracy', 0),
                'precision': self.metadata.get('precision', 0),
                'recall': self.metadata.get('recall', 0),
                'f1': self.metadata.get('f1', 0),
                'auc': self.metadata.get('auc', 0),
            },
            'trained_on': self.metadata.get('saved_at', 'unknown'),
            'train_samples': self.metadata.get('train_samples', 0)
        }


# === 全局实例 ===
_predictors: Dict[str, SignalPredictor] = {}


def get_predictor(market: str = 'US') -> SignalPredictor:
    """获取预测器实例 (单例)"""
    global _predictors
    
    if market not in _predictors:
        _predictors[market] = SignalPredictor(market=market)
    
    return _predictors[market]


def predict_signals(signals_df: pd.DataFrame, market: str = 'US') -> pd.DataFrame:
    """便捷函数: 预测信号"""
    predictor = get_predictor(market)
    return predictor.predict(signals_df)


def get_top_signals(signals_df: pd.DataFrame, market: str = 'US', top_n: int = 10) -> pd.DataFrame:
    """便捷函数: 获取 Top N"""
    predictor = get_predictor(market)
    return predictor.get_top_signals(signals_df, top_n)


# === 测试 ===
if __name__ == "__main__":
    print("=== Signal Predictor 测试 ===\n")
    
    predictor = SignalPredictor(market='US')
    
    if predictor.load():
        info = predictor.get_model_info()
        print(f"模型状态: {info['status']}")
        print(f"特征数: {info['features']}")
        print(f"AUC: {info['metrics']['auc']:.3f}")
        
        # 测试预测
        test_df = pd.DataFrame([
            {'blue_daily': 120, 'blue_weekly': 80, 'price': 150},
            {'blue_daily': 150, 'blue_weekly': 130, 'price': 50},
            {'blue_daily': 90, 'blue_weekly': 60, 'price': 200},
        ])
        
        result = predictor.predict(test_df)
        print("\n测试预测结果:")
        print(result[['blue_daily', 'blue_weekly', 'ml_prob', 'ml_rank']])
    else:
        print("模型未训练，请先运行: python train_xgb.py")
