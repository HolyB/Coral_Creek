"""
特征工程流水线 (Feature Engineering Pipeline)
================================================
Feature Engineering for Coral Creek ML Models
- 将原始 OHLCV 转换为机器学习可用特征
- 计算技术指标 (Momentum, Volatility, Volume, Structure)
- 计算目标变量 (Label) 用于排序

依赖:
    - pandas
    - numpy
    - ta (可选, 推荐安装 pandas-ta)

用法:
    from ml.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_features = fe.transform(df_ohlcv)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算核心技术指标 (无需 ta-lib)"""
        df = df.copy()
        
        # 1. 均线距离 (Structure)
        for ma in [5, 10, 20, 50, 100, 200]:
            col_ma = f'ma_{ma}'
            df[col_ma] = df['close'].rolling(window=ma).mean()
            # Distance from MA (重要特征: 乖离率)
            df[f'dist_ma_{ma}'] = (df['close'] - df[col_ma]) / df[col_ma]
            
        # 2. 相对强弱 (RSI - 14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. 波动率 (ATR Ratio - 14)
        # TR = max(H-L, |H-Cp|, |L-Cp|)
        h, l, c_prev = df['high'], df['low'], df['close'].shift(1)
        tr = np.maximum(h - l, np.maximum(abs(h - c_prev), abs(l - c_prev)))
        atr = tr.rolling(window=14).mean()
        df['atr_ratio'] = atr / df['close']  # 归一化 ATR
        
        # 4. 前高/前低位置 (Position within range)
        for window in [20, 60, 250]:
            h_max = df['high'].rolling(window=window).max()
            l_min = df['low'].rolling(window=window).min()
            # 当前价格处于 N 天区间的百分位 (0~1)
            df[f'pos_in_{window}d'] = (df['close'] - l_min) / (h_max - l_min + 1e-6)
            
        # 5. 动量 (Momentum / ROC)
        for d in [1, 3, 5, 10, 20]:
            df[f'roc_{d}d'] = df['close'].pct_change(d)
        
        # 6. 量能 (Volume)
        # 量比 (Volume Ratio vs MA20)
        vol_ma20 = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / (vol_ma20 + 1e-6)
        # OBV diff (简单版)
        dir = np.sign(df['close'].diff())
        obv = (df['volume'] * dir).cumsum()
        df['obv_trend_5d'] = obv.pct_change(5) # OBV 5天斜率
        
        # 7. 波动带宽 (Bollinger Band Width)
        std20 = df['close'].rolling(20).std()
        df['bb_width'] = (4 * std20) / df['ma_20']
        
        return df
        
    def _add_market_features(self, df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """关联大盘特征 (Market Context)"""
        # 假设 df index 是 Date
        if market_df is None or market_df.empty:
            return df
            
        # Join market data on date
        # 这里需要 index 对齐
        common_idx = df.index.intersection(market_df.index)
        
        # Market Trend (SPY > MA20) -> 1 or 0
        mkt_trend = (market_df['close'] > market_df['ma_20']).astype(int)
        
        df.loc[common_idx, 'mkt_bull_flag'] = mkt_trend.loc[common_idx]
        df.loc[common_idx, 'mkt_rsi'] = market_df.loc[common_idx, 'rsi_14']
        
        # Beta (Rolling Correlation 60d)
        # df['beta_60d'] = df['pct_change'].rolling(60).corr(market_df['pct_change'])
        
        return df

    def transform(self, df: pd.DataFrame, 
                  market_df: Optional[pd.DataFrame] = None,
                  is_training: bool = False) -> pd.DataFrame:
        """
        主转换函数
        df 必须包含: open, high, low, close, volume (小写列名), index为日期
        """
        df = df.copy()
        
        # 基础指标计算
        df = self._calculate_technical_indicators(df)
        
        # 大盘关联
        if market_df is not None:
            df = self._add_market_features(df, market_df)
            
        # 去除前 N 行 (因为 rolling window 会产生 NaN)
        df = df.dropna()
        
        # 记录特征列名
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        self.feature_names = [c for c in df.columns if c not in exclude_cols]
        
        return df

    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        构造预测目标 (Label)
        Predict: Future N-day Return
        """
        # Close_(t+N) / Close_t - 1
        df[f'ret_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1.0
        
        # Future Max Drawdown (未来 N 天最大回撤)
        # 这是一个很有用的 Risk Label
        future_low = df['low'].rolling(window=horizon).min().shift(-horizon)
        df[f'dd_{horizon}d'] = (future_low - df['close']) / df['close']
        
        return df
