#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征增强器 - 高级特征工程
Advanced Feature Engineering

新增特征:
1. 市场状态特征 (Market Regime)
2. 动量因子 (Momentum Factors)
3. 波动率特征 (Volatility Features)
4. 情绪指标 (Sentiment Indicators)
5. 交互特征 (Interaction Features)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class AdvancedFeatureEngineer:
    """高级特征工程器"""
    
    def __init__(self):
        self.feature_stats = {}
        self.feature_names = []
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建动量因子特征
        
        Args:
            df: 含有 Close, High, Low, Volume 的 DataFrame
        
        Returns:
            添加了动量特征的 DataFrame
        """
        df = df.copy()
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        
        # 相对强弱 (RS)
        df['rs_5_20'] = df['Close'].pct_change(5) / (df['Close'].pct_change(20) + 0.001)
        
        # 价格相对位置 (0-100)
        for period in [20, 60]:
            df[f'price_position_{period}'] = (
                (df['Close'] - df['Low'].rolling(period).min()) / 
                (df['High'].rolling(period).max() - df['Low'].rolling(period).min() + 0.001)
            ) * 100
        
        # 动量加速度
        momentum = df['Close'].pct_change(5)
        df['momentum_acceleration'] = momentum - momentum.shift(5)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建波动率特征
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            添加了波动率特征的 DataFrame
        """
        df = df.copy()
        
        # 历史波动率
        for period in [10, 20, 60]:
            df[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252) * 100
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / df['Close'] * 100
        
        # 波动率比率 (短期/长期)
        df['vol_ratio'] = df['volatility_10'] / (df['volatility_60'] + 0.001)
        
        # 真实波幅占比
        df['true_range_pct'] = tr / df['Close'] * 100
        
        # 内外盘波动
        df['upper_wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'] * 100
        df['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close'] * 100
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建成交量特征
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            添加了成交量特征的 DataFrame
        """
        df = df.copy()
        
        # 相对成交量
        for period in [5, 20]:
            df[f'relative_volume_{period}'] = df['Volume'] / (df['Volume'].rolling(period).mean() + 1)
        
        # 量价齐升标志
        df['vol_price_up'] = ((df['Close'] > df['Close'].shift(1)) & 
                              (df['Volume'] > df['Volume'].shift(1))).astype(int)
        
        # OBV (On-Balance Volume)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_slope_10'] = (obv - obv.shift(10)) / 10
        
        # 成交量MA偏离
        vol_ma_20 = df['Volume'].rolling(20).mean()
        df['vol_deviation'] = (df['Volume'] - vol_ma_20) / (vol_ma_20 + 1) * 100
        
        # 能量潮
        df['volume_momentum'] = df['Volume'].pct_change(5) * 100
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建趋势特征
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            添加了趋势特征的 DataFrame
        """
        df = df.copy()
        
        # 均线距离
        for period in [10, 20, 60]:
            ma = df['Close'].rolling(period).mean()
            df[f'ma_distance_{period}'] = (df['Close'] - ma) / ma * 100
        
        # 均线斜率
        for period in [10, 20]:
            ma = df['Close'].rolling(period).mean()
            df[f'ma_slope_{period}'] = (ma - ma.shift(5)) / ma.shift(5) * 100
        
        # 均线排列 (多头/空头)
        ma_5 = df['Close'].rolling(5).mean()
        ma_10 = df['Close'].rolling(10).mean()
        ma_20 = df['Close'].rolling(20).mean()
        df['ma_alignment'] = ((ma_5 > ma_10) & (ma_10 > ma_20)).astype(int) - \
                             ((ma_5 < ma_10) & (ma_10 < ma_20)).astype(int)
        
        # ADX 趋势强度
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        df['adx_14'] = dx.rolling(14).mean()
        
        # 趋势一致性
        df['trend_consistency'] = (
            (df['Close'] > df['Close'].shift(1)).astype(int) +
            (df['Close'] > df['Close'].shift(5)).astype(int) +
            (df['Close'] > df['Close'].shift(10)).astype(int)
        ) / 3 * 100
        
        return df
    
    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建形态特征
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            添加了形态特征的 DataFrame
        """
        df = df.copy()
        
        # 实体大小
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Close'] * 100
        
        # 实体方向
        df['body_direction'] = np.sign(df['Close'] - df['Open'])
        
        # 上下影线比率
        total_range = df['High'] - df['Low']
        body = abs(df['Close'] - df['Open'])
        df['shadow_ratio'] = (total_range - body) / (total_range + 0.001)
        
        # 连续阳/阴线
        up_days = (df['Close'] > df['Open']).astype(int)
        df['consecutive_up'] = up_days * (up_days.groupby((up_days != up_days.shift()).cumsum()).cumcount() + 1)
        
        down_days = (df['Close'] < df['Open']).astype(int)
        df['consecutive_down'] = down_days * (down_days.groupby((down_days != down_days.shift()).cumsum()).cumcount() + 1)
        
        # 价格突破
        df['above_20d_high'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
        df['below_20d_low'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            df: 含有已计算特征的 DataFrame
        
        Returns:
            添加了交互特征的 DataFrame
        """
        df = df.copy()
        
        # 动量 x 波动率
        if 'roc_10' in df.columns and 'volatility_20' in df.columns:
            df['momentum_volatility'] = df['roc_10'] / (df['volatility_20'] + 0.001)
        
        # 趋势 x 成交量
        if 'ma_distance_20' in df.columns and 'relative_volume_20' in df.columns:
            df['trend_volume_interaction'] = df['ma_distance_20'] * df['relative_volume_20']
        
        # ADX x 方向
        if 'adx_14' in df.columns and 'ma_alignment' in df.columns:
            df['trend_strength_direction'] = df['adx_14'] * df['ma_alignment']
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用所有特征工程
        
        Args:
            df: 原始 OHLCV DataFrame
        
        Returns:
            含有所有特征的 DataFrame
        """
        df = self.create_momentum_features(df)
        df = self.create_volatility_features(df)
        df = self.create_volume_features(df)
        df = self.create_trend_features(df)
        df = self.create_pattern_features(df)
        df = self.create_interaction_features(df)
        
        # 记录特征名
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_names = [col for col in df.columns if col not in original_cols]
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame, dropna: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        获取特征矩阵
        
        Args:
            df: 带有特征的 DataFrame
            dropna: 是否删除含 NaN 的行
        
        Returns:
            (特征矩阵, 特征名称列表)
        """
        feature_df = df[self.feature_names]
        
        if dropna:
            feature_df = feature_df.dropna()
        
        return feature_df.values, self.feature_names


if __name__ == "__main__":
    print("=== Advanced Feature Engineer 测试 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(100)),
        'High': 101 + np.cumsum(np.random.randn(100)),
        'Low': 99 + np.cumsum(np.random.randn(100)),
        'Close': 100 + np.cumsum(np.random.randn(100)),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # 确保 High >= max(Open, Close), Low <= min(Open, Close)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # 应用特征工程
    engineer = AdvancedFeatureEngineer()
    df_features = engineer.transform(df)
    
    print(f"原始列数: {5}")
    print(f"特征数: {len(engineer.feature_names)}")
    print(f"\n特征列表:")
    for i, name in enumerate(engineer.feature_names, 1):
        print(f"  {i}. {name}")
    
    # 获取特征矩阵
    X, names = engineer.get_feature_matrix(df_features)
    print(f"\n特征矩阵形状: {X.shape}")
