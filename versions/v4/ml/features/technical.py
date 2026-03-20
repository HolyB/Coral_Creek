"""
技术因子库
Technical Factors Library

包含 100+ 技术因子，用于 ML 模型训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class TechnicalFeatures:
    """技术因子计算器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: OHLCV DataFrame，需要有 Open, High, Low, Close, Volume 列
        """
        self.df = df.copy()
        self.open = df['Open'].values
        self.high = df['High'].values
        self.low = df['Low'].values
        self.close = df['Close'].values
        self.volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))
        
        self.features = {}
    
    # ==================== 趋势类因子 ====================
    
    def add_ma_features(self, periods: List[int] = [5, 10, 20, 60, 120]):
        """移动平均线因子"""
        for p in periods:
            # MA
            self.features[f'ma_{p}'] = pd.Series(self.close).rolling(p).mean().values
            
            # MA 偏离度
            self.features[f'ma_{p}_bias'] = (self.close - self.features[f'ma_{p}']) / self.features[f'ma_{p}']
            
            # 价格在 MA 上方的天数占比
            above_ma = (self.close > self.features[f'ma_{p}']).astype(float)
            self.features[f'ma_{p}_above_ratio'] = pd.Series(above_ma).rolling(p).mean().values
        
        # MA 排列因子 (多头排列得分)
        if len(periods) >= 3:
            ma_cols = [self.features[f'ma_{p}'] for p in sorted(periods)[:3]]
            # 短期 > 中期 > 长期 = 多头
            self.features['ma_alignment'] = (
                (ma_cols[0] > ma_cols[1]).astype(float) + 
                (ma_cols[1] > ma_cols[2]).astype(float)
            ) / 2
        
        return self
    
    def add_ema_features(self, periods: List[int] = [12, 26, 50]):
        """指数移动平均线因子"""
        for p in periods:
            self.features[f'ema_{p}'] = pd.Series(self.close).ewm(span=p, adjust=False).mean().values
            self.features[f'ema_{p}_bias'] = (self.close - self.features[f'ema_{p}']) / self.features[f'ema_{p}']
        
        return self
    
    def add_macd_features(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD 因子"""
        ema_fast = pd.Series(self.close).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(self.close).ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        self.features['macd'] = macd.values
        self.features['macd_signal'] = signal_line.values
        self.features['macd_histogram'] = histogram.values
        
        # MACD 金叉/死叉
        self.features['macd_cross'] = np.sign(macd.values - signal_line.values)
        
        # MACD 加速度
        self.features['macd_acceleration'] = histogram.diff().values
        
        return self
    
    # ==================== 动量类因子 ====================
    
    def add_momentum_features(self, periods: List[int] = [5, 10, 20, 60]):
        """动量因子"""
        close_series = pd.Series(self.close)
        
        for p in periods:
            # 简单收益率
            self.features[f'return_{p}d'] = close_series.pct_change(p).values
            
            # 对数收益率
            self.features[f'log_return_{p}d'] = np.log(self.close / np.roll(self.close, p))
            self.features[f'log_return_{p}d'][:p] = np.nan
            
            # 动量 (价格变化)
            self.features[f'momentum_{p}d'] = (self.close - np.roll(self.close, p)) / np.roll(self.close, p)
            self.features[f'momentum_{p}d'][:p] = np.nan
        
        # 动量加速度
        if len(periods) >= 2:
            short_mom = self.features[f'momentum_{periods[0]}d']
            long_mom = self.features[f'momentum_{periods[1]}d']
            self.features['momentum_acceleration'] = short_mom - long_mom
        
        return self
    
    def add_rsi_features(self, periods: List[int] = [6, 12, 24]):
        """RSI 因子"""
        close_series = pd.Series(self.close)
        delta = close_series.diff()
        
        for p in periods:
            gain = delta.where(delta > 0, 0).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
            
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            self.features[f'rsi_{p}'] = rsi.values
            
            # RSI 超买超卖
            self.features[f'rsi_{p}_overbought'] = (rsi > 70).astype(float).values
            self.features[f'rsi_{p}_oversold'] = (rsi < 30).astype(float).values
        
        return self
    
    def add_kdj_features(self, n: int = 9, m1: int = 3, m2: int = 3):
        """KDJ 因子"""
        low_n = pd.Series(self.low).rolling(n).min()
        high_n = pd.Series(self.high).rolling(n).max()
        
        rsv = (pd.Series(self.close) - low_n) / (high_n - low_n + 1e-10) * 100
        
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        self.features['kdj_k'] = k.values
        self.features['kdj_d'] = d.values
        self.features['kdj_j'] = j.values
        
        # KDJ 金叉/死叉
        self.features['kdj_cross'] = np.sign(k.values - d.values)
        
        return self
    
    # ==================== 波动率因子 ====================
    
    def add_volatility_features(self, periods: List[int] = [5, 10, 20]):
        """波动率因子"""
        returns = pd.Series(self.close).pct_change()
        
        for p in periods:
            # 历史波动率
            self.features[f'volatility_{p}d'] = returns.rolling(p).std().values
            
            # 年化波动率
            self.features[f'volatility_{p}d_annual'] = returns.rolling(p).std().values * np.sqrt(252)
            
            # 波动率变化率
            vol = returns.rolling(p).std()
            self.features[f'volatility_{p}d_change'] = vol.pct_change().values
        
        return self
    
    def add_atr_features(self, periods: List[int] = [14, 20]):
        """ATR 因子"""
        high_series = pd.Series(self.high)
        low_series = pd.Series(self.low)
        close_series = pd.Series(self.close)
        
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift(1))
        tr3 = abs(low_series - close_series.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for p in periods:
            atr = tr.rolling(p).mean()
            self.features[f'atr_{p}'] = atr.values
            
            # ATR 相对价格
            self.features[f'atr_{p}_pct'] = (atr / close_series).values
            
            # ATR 变化
            self.features[f'atr_{p}_change'] = atr.pct_change().values
        
        return self
    
    def add_bollinger_features(self, period: int = 20, std_mult: float = 2.0):
        """布林带因子"""
        close_series = pd.Series(self.close)
        
        ma = close_series.rolling(period).mean()
        std = close_series.rolling(period).std()
        
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        
        self.features['bb_upper'] = upper.values
        self.features['bb_middle'] = ma.values
        self.features['bb_lower'] = lower.values
        
        # 布林带宽度
        self.features['bb_width'] = ((upper - lower) / ma).values
        
        # 布林带位置 (%B)
        self.features['bb_pct'] = ((close_series - lower) / (upper - lower + 1e-10)).values
        
        return self
    
    # ==================== 成交量因子 ====================
    
    def add_volume_features(self, periods: List[int] = [5, 10, 20]):
        """成交量因子"""
        vol_series = pd.Series(self.volume)
        close_series = pd.Series(self.close)
        
        for p in periods:
            # 成交量均线
            self.features[f'volume_ma_{p}'] = vol_series.rolling(p).mean().values
            
            # 成交量比 (相对均线)
            self.features[f'volume_ratio_{p}'] = (self.volume / vol_series.rolling(p).mean()).values
            
            # 成交额
            turnover = self.volume * self.close
            self.features[f'turnover_ma_{p}'] = pd.Series(turnover).rolling(p).mean().values
        
        # 量价背离 (价涨量缩 / 价跌量增)
        price_change = close_series.diff()
        vol_change = vol_series.diff()
        self.features['volume_price_divergence'] = np.sign(price_change.values) * np.sign(-vol_change.values)
        
        # OBV (能量潮)
        obv = (np.sign(price_change) * vol_series).cumsum()
        self.features['obv'] = obv.values
        self.features['obv_ma_10'] = obv.rolling(10).mean().values
        
        return self
    
    # ==================== 形态因子 ====================
    
    def add_candlestick_features(self):
        """K线形态因子"""
        body = self.close - self.open
        upper_shadow = self.high - np.maximum(self.close, self.open)
        lower_shadow = np.minimum(self.close, self.open) - self.low
        range_hl = self.high - self.low + 1e-10
        
        # 实体比例
        self.features['body_ratio'] = np.abs(body) / range_hl
        
        # 上影线比例
        self.features['upper_shadow_ratio'] = upper_shadow / range_hl
        
        # 下影线比例
        self.features['lower_shadow_ratio'] = lower_shadow / range_hl
        
        # 阴阳线
        self.features['candle_direction'] = np.sign(body)
        
        # 锤子线 (下影线长，实体小，上影线短)
        self.features['hammer'] = (
            (lower_shadow > 2 * np.abs(body)) & 
            (upper_shadow < 0.1 * range_hl)
        ).astype(float)
        
        # 十字星 (实体很小)
        self.features['doji'] = (np.abs(body) < 0.05 * range_hl).astype(float)
        
        return self
    
    def add_pattern_features(self):
        """价格形态因子"""
        close_series = pd.Series(self.close)
        high_series = pd.Series(self.high)
        low_series = pd.Series(self.low)
        
        # 新高新低
        for p in [5, 10, 20, 60]:
            self.features[f'new_high_{p}d'] = (self.close >= high_series.rolling(p).max()).astype(float)
            self.features[f'new_low_{p}d'] = (self.close <= low_series.rolling(p).min()).astype(float)
        
        # 距离高点/低点
        for p in [20, 60]:
            high_p = high_series.rolling(p).max()
            low_p = low_series.rolling(p).min()
            self.features[f'dist_from_high_{p}d'] = ((self.close - high_p) / high_p).values
            self.features[f'dist_from_low_{p}d'] = ((self.close - low_p) / low_p).values
        
        return self
    
    # ==================== 综合 ====================
    
    def add_all_features(self) -> 'TechnicalFeatures':
        """添加所有因子"""
        self.add_ma_features()
        self.add_ema_features()
        self.add_macd_features()
        self.add_momentum_features()
        self.add_rsi_features()
        self.add_kdj_features()
        self.add_volatility_features()
        self.add_atr_features()
        self.add_bollinger_features()
        self.add_volume_features()
        self.add_candlestick_features()
        self.add_pattern_features()
        
        return self
    
    def get_feature_df(self) -> pd.DataFrame:
        """获取特征 DataFrame"""
        feature_df = pd.DataFrame(self.features, index=self.df.index)
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """获取特征名列表"""
        return list(self.features.keys())


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有技术因子
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        特征 DataFrame
    """
    calculator = TechnicalFeatures(df)
    calculator.add_all_features()
    return calculator.get_feature_df()
