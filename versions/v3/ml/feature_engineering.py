#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Engineering Module - 特征工程模块

提供技术指标特征提取，用于机器学习模型训练
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class FeatureEngineer:
    """特征工程器 - 从 OHLCV 数据提取技术特征"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从 OHLCV 数据提取全部技术特征
        
        Args:
            df: 包含 Open, High, Low, Close, Volume 列的 DataFrame
        
        Returns:
            包含所有特征的 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        # 价格特征
        features = self._add_price_features(df, features)
        
        # 动量特征
        features = self._add_momentum_features(df, features)
        
        # 波动性特征
        features = self._add_volatility_features(df, features)
        
        # 成交量特征
        features = self._add_volume_features(df, features)
        
        # 趋势特征
        features = self._add_trend_features(df, features)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def _add_price_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """价格相关特征"""
        # 收益率
        features['return_1d'] = df['Close'].pct_change(1)
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_10d'] = df['Close'].pct_change(10)
        features['return_20d'] = df['Close'].pct_change(20)
        
        # 价格位置
        features['close_to_high'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)
        features['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        
        # 均线距离
        for period in [5, 10, 20, 50]:
            ma = df['Close'].rolling(period).mean()
            features[f'dist_ma{period}'] = (df['Close'] - ma) / ma
        
        return features
    
    def _add_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """动量特征"""
        # RSI
        for period in [6, 14, 28]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-9)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # 动量
        features['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        features['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        return features
    
    def _add_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """波动性特征"""
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = tr.rolling(period).mean() / df['Close']
        
        # 波动率
        for period in [10, 20, 60]:
            features[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std()
        
        # Bollinger Band 位置
        ma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        features['bb_position'] = (df['Close'] - ma20) / (2 * std20 + 1e-9)
        features['bb_width'] = (4 * std20) / ma20
        
        return features
    
    def _add_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """成交量特征"""
        # 成交量变化
        features['volume_change'] = df['Volume'].pct_change()
        features['volume_ma5_ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        features['volume_ma20_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # OBV (On-Balance Volume)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        features['obv_change'] = obv.pct_change(5)
        
        # 成交量加权价格
        features['vwap_dist'] = (df['Close'] - (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()) / df['Close']
        
        return features
    
    def _add_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """趋势特征"""
        # 均线排列
        ma5 = df['Close'].rolling(5).mean()
        ma10 = df['Close'].rolling(10).mean()
        ma20 = df['Close'].rolling(20).mean()
        
        features['ma_alignment'] = ((ma5 > ma10).astype(int) + (ma10 > ma20).astype(int)) / 2
        
        # ADX (简化版)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift()),
            np.abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        
        features['adx'] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        features['di_diff'] = plus_di - minus_di
        
        return features
    
    def get_feature_importance_display(self, importance_dict: Dict[str, float]) -> pd.DataFrame:
        """
        格式化特征重要性用于显示
        """
        df = pd.DataFrame([
            {'Feature': k, 'Importance': v}
            for k, v in sorted(importance_dict.items(), key=lambda x: -x[1])
        ])
        return df


def prepare_training_data(
    signals: List[Dict],
    forward_days: int = 10,
    target_type: str = 'binary',
    fetch_history: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    从信号数据准备 ML 训练数据
    
    Args:
        signals: 信号列表 (from backtest_service.run_signal_backtest)
        forward_days: 前向收益天数
        target_type: 'binary' (盈亏分类) 或 'regression' (收益回归)
        fetch_history: 是否获取历史数据以计算完整特征 (较慢但更准确)
    
    Returns:
        (X, y) 特征矩阵和目标变量
        
    Note:
        如果 fetch_history=False，仅使用信号中已有的特征。
        如果 fetch_history=True，会从数据库获取历史K线并计算 100+ 特征。
    """
    ret_col = f'return_{forward_days}d'
    
    # 过滤有效数据
    valid_signals = [s for s in signals if s.get(ret_col) is not None]
    
    if not valid_signals:
        return pd.DataFrame(), pd.Series()
    
    # === 构建特征矩阵 ===
    feature_rows = []
    
    for s in valid_signals:
        row = {
            # 核心 BLUE 特征
            'blue_daily': s.get('blue_daily', 0) or 0,
            'blue_weekly': s.get('blue_weekly', 0) or 0,
            'blue_monthly': s.get('blue_monthly', 0) or 0,
            'is_heima': 1 if s.get('is_heima') else 0,
            
            # 价格特征
            'price': s.get('price', 0) or 0,
            'price_log': np.log(max(s.get('price', 1), 1)),  # Log price for scaling
            
            # ADX/波动率 (如果可用)
            'adx': s.get('adx', 25) or 25,
            'volatility': s.get('volatility', 0.3) or 0.3,
            
            # 成交量特征 (如果可用)
            'volume_ratio': s.get('volume_ratio', 1.0) or 1.0,
            'turnover': s.get('turnover', 0) or 0,
            
            # 市值 (如果可用)
            'market_cap': s.get('market_cap', 0) or 0,
            'market_cap_log': np.log(max(s.get('market_cap', 1e6), 1e6)),
            
            # === BLUE 衍生特征 ===
            'blue_daily_level': _blue_level(s.get('blue_daily', 0) or 0),
            'blue_weekly_level': _blue_level(s.get('blue_weekly', 0) or 0),
            'blue_dw_resonance': int((s.get('blue_daily', 0) or 0) >= 100 and (s.get('blue_weekly', 0) or 0) >= 100),
            'blue_dwm_resonance': int(
                (s.get('blue_daily', 0) or 0) >= 100 and 
                (s.get('blue_weekly', 0) or 0) >= 100 and 
                (s.get('blue_monthly', 0) or 0) >= 100
            ),
            'blue_daily_deviation': (s.get('blue_daily', 0) or 0) - 100,
            'blue_overbought': int((s.get('blue_daily', 0) or 0) > 120),
            'blue_oversold': int((s.get('blue_daily', 0) or 0) < 20),
            'blue_golden_zone': int(80 <= (s.get('blue_daily', 0) or 0) <= 120),
            
            # 信号强度综合评分
            'signal_strength': _calculate_signal_strength(s),
        }
        feature_rows.append(row)
    
    X = pd.DataFrame(feature_rows)
    
    # 处理缺失值
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # === 构建目标变量 ===
    if target_type == 'binary':
        y = pd.Series([1 if s[ret_col] > 0 else 0 for s in valid_signals])
    else:
        y = pd.Series([s[ret_col] for s in valid_signals])
    
    return X, y


def _blue_level(blue_value: float) -> int:
    """BLUE 值分档: 0-4"""
    if blue_value >= 150:
        return 4  # 极强
    elif blue_value >= 120:
        return 3  # 强
    elif blue_value >= 100:
        return 2  # 中等
    elif blue_value >= 50:
        return 1  # 弱
    else:
        return 0  # 无信号


def _calculate_signal_strength(signal: Dict) -> int:
    """计算综合信号强度"""
    score = 0
    blue_d = signal.get('blue_daily', 0) or 0
    blue_w = signal.get('blue_weekly', 0) or 0
    blue_m = signal.get('blue_monthly', 0) or 0
    
    if blue_d >= 100: score += 1
    if blue_w >= 100: score += 1
    if blue_m >= 100: score += 1
    if signal.get('is_heima'): score += 2
    
    return min(score, 5)


if __name__ == "__main__":
    # 测试特征工程
    import sys
    sys.path.append('..')
    from data_fetcher import get_us_stock_data
    
    df = get_us_stock_data('AAPL', days=100)
    if df is not None:
        fe = FeatureEngineer()
        features = fe.extract_all_features(df)
        print(f"Extracted {len(fe.feature_names)} features")
        print(features.tail())
