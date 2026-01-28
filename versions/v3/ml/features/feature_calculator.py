"""
特征计算器
Feature Calculator

从原始 K 线数据计算 100+ 技术特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureCalculator:
    """特征计算器"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征
        
        Args:
            df: DataFrame with Date, Open, High, Low, Close, Volume
        
        Returns:
            DataFrame with all features (最后一行是最新特征)
        """
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        
        result = df.copy()
        
        # 1. 均线
        result = self._add_ma_features(result)
        
        # 2. 动量
        result = self._add_momentum_features(result)
        
        # 3. 波动率
        result = self._add_volatility_features(result)
        
        # 4. RSI
        result = self._add_rsi_features(result)
        
        # 5. MACD
        result = self._add_macd_features(result)
        
        # 6. KDJ
        result = self._add_kdj_features(result)
        
        # 7. 布林带
        result = self._add_bollinger_features(result)
        
        # 8. 成交量
        result = self._add_volume_features(result)
        
        # 9. 价格形态
        result = self._add_pattern_features(result)
        
        return result
    
    def get_latest_features(self, df: pd.DataFrame) -> Dict:
        """获取最新一天的特征字典"""
        result = self.calculate_all(df)
        if result.empty:
            return {}
        
        latest = result.iloc[-1].to_dict()
        # 过滤掉 NaN
        return {k: v for k, v in latest.items() if pd.notna(v)}
    
    def _add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """均线特征"""
        close = df['Close']
        
        # 简单均线
        for period in [5, 10, 20, 60, 120, 250]:
            df[f'ma_{period}'] = close.rolling(period).mean()
        
        # 均线偏离度 (价格相对均线的位置)
        for period in [5, 20, 60]:
            ma = df[f'ma_{period}']
            df[f'ma_{period}_bias'] = (close - ma) / ma * 100
        
        # EMA
        for period in [12, 26]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # 均线多头排列 (短期在上)
        df['ma_bullish'] = (
            (df['ma_5'] > df['ma_10']) & 
            (df['ma_10'] > df['ma_20']) & 
            (df['ma_20'] > df['ma_60'])
        ).astype(int)
        
        # 金叉/死叉
        df['ma_5_10_cross'] = np.sign(df['ma_5'] - df['ma_10']).diff()
        df['ma_10_20_cross'] = np.sign(df['ma_10'] - df['ma_20']).diff()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量特征"""
        close = df['Close']
        
        # 历史收益率
        for period in [1, 5, 10, 20, 60]:
            df[f'return_{period}d'] = close.pct_change(period) * 100
        
        # 动量 (当前价格 / N天前价格)
        for period in [10, 20, 60]:
            df[f'momentum_{period}'] = close / close.shift(period)
        
        # ROC (Rate of Change)
        for period in [10, 20]:
            df[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
        
        # 加速度 (动量的变化)
        df['momentum_accel'] = df['momentum_10'].diff()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率特征"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # 标准差波动率
        for period in [5, 20, 60]:
            df[f'volatility_{period}d'] = close.pct_change().rolling(period).std() * np.sqrt(252) * 100
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_14_pct'] = df['atr_14'] / close * 100  # ATR 占价格百分比
        
        # 波动率变化
        df['volatility_change'] = df['volatility_20d'] / df['volatility_60d']
        
        # 日内波动
        df['intraday_range'] = (high - low) / close * 100
        df['intraday_range_ma'] = df['intraday_range'].rolling(10).mean()
        
        return df
    
    def _add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI 特征"""
        close = df['Close']
        delta = close.diff()
        
        for period in [6, 14]:
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI 超买超卖
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # RSI 背离 (价格创新高但RSI没有)
        df['rsi_divergence'] = (
            (close > close.rolling(20).max().shift(1)) & 
            (df['rsi_14'] < df['rsi_14'].rolling(20).max().shift(1))
        ).astype(int)
        
        return df
    
    def _add_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 特征"""
        close = df['Close']
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD 金叉/死叉
        df['macd_cross'] = np.sign(df['macd'] - df['macd_signal']).diff()
        
        # MACD 柱状图变化趋势
        df['macd_hist_trend'] = df['macd_hist'].diff()
        
        return df
    
    def _add_kdj_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """KDJ 特征"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        n = 9
        low_n = low.rolling(n).min()
        high_n = high.rolling(n).max()
        
        rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100
        
        df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # KDJ 金叉/死叉
        df['kdj_cross'] = np.sign(df['kdj_k'] - df['kdj_d']).diff()
        
        # KDJ 超买超卖
        df['kdj_overbought'] = (df['kdj_j'] > 100).astype(int)
        df['kdj_oversold'] = (df['kdj_j'] < 0).astype(int)
        
        return df
    
    def _add_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """布林带特征"""
        close = df['Close']
        
        period = 20
        std_mult = 2
        
        df['bb_middle'] = close.rolling(period).mean()
        bb_std = close.rolling(period).std()
        df['bb_upper'] = df['bb_middle'] + std_mult * bb_std
        df['bb_lower'] = df['bb_middle'] - std_mult * bb_std
        
        # 布林带宽度 (波动率指标)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        
        # 价格在布林带中的位置 (0=下轨, 1=上轨)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # 突破布林带
        df['bb_breakout_upper'] = (close > df['bb_upper']).astype(int)
        df['bb_breakout_lower'] = (close < df['bb_lower']).astype(int)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特征"""
        volume = df['Volume']
        close = df['Close']
        
        # 成交量均线
        for period in [5, 20, 60]:
            df[f'volume_ma_{period}'] = volume.rolling(period).mean()
        
        # 量比 (今日成交量 / 5日均量)
        df['volume_ratio'] = volume / (df['volume_ma_5'] + 1)
        
        # 成交量变化
        df['volume_change'] = volume.pct_change()
        
        # 放量 (成交量 > 2倍均量)
        df['volume_surge'] = (volume > 2 * df['volume_ma_20']).astype(int)
        
        # 缩量
        df['volume_shrink'] = (volume < 0.5 * df['volume_ma_20']).astype(int)
        
        # 量价配合 (价涨量增)
        price_up = close.diff() > 0
        volume_up = volume.diff() > 0
        df['volume_price_confirm'] = (price_up & volume_up).astype(int)
        
        # OBV (On Balance Volume)
        obv = (np.sign(close.diff()) * volume).cumsum()
        df['obv'] = obv
        df['obv_ma'] = obv.rolling(20).mean()
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格形态特征"""
        open_p = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        body = close - open_p
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
        
        # 实体大小 (相对于价格)
        df['body_size'] = abs(body) / close * 100
        
        # 上下影线
        df['upper_shadow_pct'] = upper_shadow / close * 100
        df['lower_shadow_pct'] = lower_shadow / close * 100
        
        # 十字星 (小实体，长影线)
        df['is_doji'] = (abs(body) < df['body_size'].rolling(20).mean() * 0.3).astype(int)
        
        # 锤子线 (下影线长，实体小)
        df['is_hammer'] = (
            (lower_shadow > abs(body) * 2) & 
            (upper_shadow < abs(body) * 0.5)
        ).astype(int)
        
        # 倒锤子线
        df['is_inv_hammer'] = (
            (upper_shadow > abs(body) * 2) & 
            (lower_shadow < abs(body) * 0.5)
        ).astype(int)
        
        # 连续上涨/下跌天数
        df['consecutive_up'] = (close > close.shift(1)).astype(int)
        df['consecutive_up'] = df['consecutive_up'].groupby(
            (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
        ).cumsum() * df['consecutive_up']
        
        df['consecutive_down'] = (close < close.shift(1)).astype(int)
        df['consecutive_down'] = df['consecutive_down'].groupby(
            (df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()
        ).cumsum() * df['consecutive_down']
        
        # 创新高/新低
        df['near_high_20'] = (close >= high.rolling(20).max() * 0.98).astype(int)
        df['near_low_20'] = (close <= low.rolling(20).min() * 1.02).astype(int)
        df['near_high_60'] = (close >= high.rolling(60).max() * 0.98).astype(int)
        df['near_low_60'] = (close <= low.rolling(60).min() * 1.02).astype(int)
        
        return df


# === 便捷函数 ===
_calculator = None

def get_calculator() -> FeatureCalculator:
    global _calculator
    if _calculator is None:
        _calculator = FeatureCalculator()
    return _calculator


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有特征"""
    return get_calculator().calculate_all(df)


def get_latest_features(df: pd.DataFrame) -> Dict:
    """获取最新特征"""
    return get_calculator().get_latest_features(df)


# === 特征列表 ===
FEATURE_COLUMNS = [
    # 价格
    'close', 'open', 'high', 'low', 'volume',
    
    # 均线
    'ma_5', 'ma_10', 'ma_20', 'ma_60', 'ma_120', 'ma_250',
    'ma_5_bias', 'ma_20_bias', 'ma_60_bias',
    'ema_12', 'ema_26',
    'ma_bullish', 'ma_5_10_cross', 'ma_10_20_cross',
    
    # 动量
    'return_1d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
    'momentum_10', 'momentum_20', 'momentum_60',
    'roc_10', 'roc_20', 'momentum_accel',
    
    # 波动率
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'atr_14', 'atr_14_pct', 'volatility_change',
    'intraday_range', 'intraday_range_ma',
    
    # RSI
    'rsi_6', 'rsi_14', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    
    # MACD
    'macd', 'macd_signal', 'macd_hist', 'macd_cross', 'macd_hist_trend',
    
    # KDJ
    'kdj_k', 'kdj_d', 'kdj_j', 'kdj_cross', 'kdj_overbought', 'kdj_oversold',
    
    # 布林带
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'bb_breakout_upper', 'bb_breakout_lower',
    
    # 成交量
    'volume_ma_5', 'volume_ma_20', 'volume_ma_60',
    'volume_ratio', 'volume_change', 'volume_surge', 'volume_shrink',
    'volume_price_confirm', 'obv', 'obv_ma',
    
    # 形态
    'body_size', 'upper_shadow_pct', 'lower_shadow_pct',
    'is_doji', 'is_hammer', 'is_inv_hammer',
    'consecutive_up', 'consecutive_down',
    'near_high_20', 'near_low_20', 'near_high_60', 'near_low_60',
]


if __name__ == "__main__":
    print("=== Feature Calculator 测试 ===\n")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.cumsum(np.random.randn(100) * 2),
        'High': 100 + np.cumsum(np.random.randn(100) * 2) + abs(np.random.randn(100)),
        'Low': 100 + np.cumsum(np.random.randn(100) * 2) - abs(np.random.randn(100)),
        'Close': 100 + np.cumsum(np.random.randn(100) * 2),
        'Volume': np.random.randint(1000000, 10000000, 100)
    })
    
    calc = FeatureCalculator()
    result = calc.calculate_all(test_df)
    
    print(f"输入: {len(test_df)} 行, {len(test_df.columns)} 列")
    print(f"输出: {len(result)} 行, {len(result.columns)} 列")
    print(f"\n特征数量: {len(result.columns) - len(test_df.columns)}")
    
    # 显示最新特征
    latest = calc.get_latest_features(test_df)
    print(f"\n最新特征 (Top 10):")
    for i, (k, v) in enumerate(list(latest.items())[:10]):
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
