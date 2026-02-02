#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strategy Implementations
策略实现 - 将外部策略转化为可回测的信号

实现 TradingView 和社区热门策略的具体逻辑
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


@dataclass
class Signal:
    """交易信号"""
    date: datetime
    symbol: str
    action: str           # 'buy', 'sell', 'hold'
    strength: float       # 0-100
    price: float
    stop_loss: float = 0
    target: float = 0
    reason: str = ""


class IndicatorCalculator:
    """指标计算器"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """RSI 相对强弱指标"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """布林带"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        return pd.DataFrame({
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'width': (sma + std * std_dev - (sma - std * std_dev)) / sma
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR 真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """ADX 趋势强度"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = IndicatorCalculator.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """VWAP 成交量加权平均价"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """SuperTrend 指标"""
        atr = IndicatorCalculator.atr(high, low, close, period)
        
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(period, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                if direction.iloc[i-1] == 1:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = direction.iloc[i-1]
        
        return pd.DataFrame({
            'supertrend': supertrend,
            'direction': direction  # 1 = bullish, -1 = bearish
        })
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV 能量潮"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv


class StrategyImplementation:
    """策略实现基类"""
    
    name: str = "Base Strategy"
    description: str = ""
    indicators_used: List[str] = []
    
    def __init__(self):
        self.calc = IndicatorCalculator()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        
        Returns:
            DataFrame with added signal column (1=buy, -1=sell, 0=hold)
        """
        raise NotImplementedError


class EMACrossoverStrategy(StrategyImplementation):
    """EMA 交叉策略 (9/21)"""
    
    name = "EMA Crossover (9/21)"
    description = "Buy when EMA9 crosses above EMA21, sell when crosses below"
    indicators_used = ['EMA9', 'EMA21']
    
    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['ema_fast'] = self.calc.ema(df['Close'], self.fast_period)
        df['ema_slow'] = self.calc.ema(df['Close'], self.slow_period)
        
        df['signal'] = 0
        
        # 金叉买入
        df.loc[(df['ema_fast'] > df['ema_slow']) & 
               (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)), 'signal'] = 1
        
        # 死叉卖出
        df.loc[(df['ema_fast'] < df['ema_slow']) & 
               (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)), 'signal'] = -1
        
        return df


class RSIDivergenceStrategy(StrategyImplementation):
    """RSI 背离策略"""
    
    name = "RSI Divergence"
    description = "Buy on bullish RSI divergence, sell on bearish divergence"
    indicators_used = ['RSI']
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['rsi'] = self.calc.rsi(df['Close'], self.rsi_period)
        df['signal'] = 0
        
        # 简化版: RSI 超卖区反弹买入
        df.loc[(df['rsi'] < self.oversold) & (df['rsi'].shift(1) >= self.oversold), 'signal'] = 1
        df.loc[(df['rsi'] > self.overbought) & (df['rsi'].shift(1) <= self.overbought), 'signal'] = -1
        
        return df


class MACDHistogramStrategy(StrategyImplementation):
    """MACD 柱状图策略"""
    
    name = "MACD Histogram Reversal"
    description = "Buy when histogram turns positive, sell when turns negative"
    indicators_used = ['MACD', 'Signal Line', 'Histogram']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        macd_df = self.calc.macd(df['Close'])
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['histogram'] = macd_df['histogram']
        
        df['signal'] = 0
        
        # 柱状图由负转正 -> 买入
        df.loc[(df['histogram'] > 0) & (df['histogram'].shift(1) <= 0), 'signal'] = 1
        
        # 柱状图由正转负 -> 卖出
        df.loc[(df['histogram'] < 0) & (df['histogram'].shift(1) >= 0), 'signal'] = -1
        
        return df


class BollingerBandSqueezeStrategy(StrategyImplementation):
    """布林带收窄突破策略"""
    
    name = "Bollinger Band Squeeze"
    description = "Buy on breakout after squeeze"
    indicators_used = ['Bollinger Bands']
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, squeeze_threshold: float = 0.05):
        super().__init__()
        self.period = period
        self.std_dev = std_dev
        self.squeeze_threshold = squeeze_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        bb = self.calc.bollinger_bands(df['Close'], self.period, self.std_dev)
        df['bb_upper'] = bb['upper']
        df['bb_lower'] = bb['lower']
        df['bb_width'] = bb['width']
        
        df['signal'] = 0
        
        # 收窄后向上突破
        squeeze = df['bb_width'] < self.squeeze_threshold
        breakout_up = df['Close'] > df['bb_upper']
        
        df.loc[squeeze.shift(1) & breakout_up, 'signal'] = 1
        
        # 收窄后向下突破
        breakout_down = df['Close'] < df['bb_lower']
        df.loc[squeeze.shift(1) & breakout_down, 'signal'] = -1
        
        return df


class SuperTrendStrategy(StrategyImplementation):
    """SuperTrend 策略"""
    
    name = "SuperTrend Strategy"
    description = "Follow SuperTrend indicator direction"
    indicators_used = ['SuperTrend', 'ATR']
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__()
        self.period = period
        self.multiplier = multiplier
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        st = self.calc.supertrend(df['High'], df['Low'], df['Close'], 
                                   self.period, self.multiplier)
        df['supertrend'] = st['supertrend']
        df['st_direction'] = st['direction']
        
        df['signal'] = 0
        
        # 方向由 -1 变 1 -> 买入
        df.loc[(df['st_direction'] == 1) & (df['st_direction'].shift(1) == -1), 'signal'] = 1
        
        # 方向由 1 变 -1 -> 卖出
        df.loc[(df['st_direction'] == -1) & (df['st_direction'].shift(1) == 1), 'signal'] = -1
        
        return df


class ADXTrendStrategy(StrategyImplementation):
    """ADX 趋势策略"""
    
    name = "ADX Trend Trading"
    description = "Trade with strong trends using ADX"
    indicators_used = ['ADX', 'DI+', 'DI-']
    
    def __init__(self, adx_threshold: int = 25):
        super().__init__()
        self.adx_threshold = adx_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        adx_df = self.calc.adx(df['High'], df['Low'], df['Close'])
        df['adx'] = adx_df['adx']
        df['plus_di'] = adx_df['plus_di']
        df['minus_di'] = adx_df['minus_di']
        
        df['signal'] = 0
        
        # ADX > 阈值 且 DI+ > DI- -> 买入
        strong_trend = df['adx'] > self.adx_threshold
        bullish = df['plus_di'] > df['minus_di']
        was_bearish = df['plus_di'].shift(1) <= df['minus_di'].shift(1)
        
        df.loc[strong_trend & bullish & was_bearish, 'signal'] = 1
        
        # ADX > 阈值 且 DI- > DI+ -> 卖出
        bearish = df['minus_di'] > df['plus_di']
        was_bullish = df['minus_di'].shift(1) <= df['plus_di'].shift(1)
        
        df.loc[strong_trend & bearish & was_bullish, 'signal'] = -1
        
        return df


class VWAPBounceStrategy(StrategyImplementation):
    """VWAP 反弹策略"""
    
    name = "VWAP Bounce Strategy"
    description = "Buy on VWAP bounce with volume confirmation"
    indicators_used = ['VWAP', 'Volume']
    
    def __init__(self, bounce_threshold: float = 0.01):
        super().__init__()
        self.bounce_threshold = bounce_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['vwap'] = self.calc.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['vol_sma'] = df['Volume'].rolling(20).mean()
        
        df['signal'] = 0
        
        # 价格接近 VWAP 且有高成交量反弹
        near_vwap = abs(df['Low'] - df['vwap']) / df['vwap'] < self.bounce_threshold
        high_volume = df['Volume'] > df['vol_sma'] * 1.5
        price_up = df['Close'] > df['Open']
        
        df.loc[near_vwap & high_volume & price_up, 'signal'] = 1
        
        return df


class CombinedStrategy(StrategyImplementation):
    """组合策略 - 多策略共识"""
    
    name = "Combined Strategy"
    description = "Trade when multiple strategies agree"
    
    def __init__(self, strategies: List[StrategyImplementation] = None, min_agreement: int = 2):
        super().__init__()
        self.strategies = strategies or [
            EMACrossoverStrategy(),
            MACDHistogramStrategy(),
            RSIDivergenceStrategy()
        ]
        self.min_agreement = min_agreement
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 收集所有策略信号
        signals = []
        for strategy in self.strategies:
            result = strategy.generate_signals(df)
            signals.append(result['signal'])
        
        # 计算共识
        signal_df = pd.concat(signals, axis=1)
        buy_votes = (signal_df == 1).sum(axis=1)
        sell_votes = (signal_df == -1).sum(axis=1)
        
        df['signal'] = 0
        df.loc[buy_votes >= self.min_agreement, 'signal'] = 1
        df.loc[sell_votes >= self.min_agreement, 'signal'] = -1
        
        return df


# === 策略工厂 ===

STRATEGY_REGISTRY = {
    'ema_crossover': EMACrossoverStrategy,
    'rsi_divergence': RSIDivergenceStrategy,
    'macd_histogram': MACDHistogramStrategy,
    'bollinger_squeeze': BollingerBandSqueezeStrategy,
    'supertrend': SuperTrendStrategy,
    'adx_trend': ADXTrendStrategy,
    'vwap_bounce': VWAPBounceStrategy,
    'combined': CombinedStrategy
}


def get_strategy(name: str, **kwargs) -> StrategyImplementation:
    """获取策略实例"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[name](**kwargs)


def list_strategies() -> List[Dict]:
    """列出所有可用策略"""
    return [
        {
            'id': name,
            'name': cls.name,
            'description': cls.description,
            'indicators': cls.indicators_used
        }
        for name, cls in STRATEGY_REGISTRY.items()
    ]


def backtest_external_strategy(strategy_name: str, symbol: str, 
                               days: int = 365, market: str = 'US') -> Dict:
    """
    回测外部策略
    
    Args:
        strategy_name: 策略名称
        symbol: 股票代码
        days: 回测天数
        market: 市场
    
    Returns:
        回测结果
    """
    from data_fetcher import get_stock_data
    
    # 获取数据
    df = get_stock_data(symbol, market=market, days=days)
    if df is None or df.empty:
        return {'error': f'Failed to get data for {symbol}'}
    
    # 获取策略
    try:
        strategy = get_strategy(strategy_name)
    except ValueError as e:
        return {'error': str(e)}
    
    # 生成信号
    df = strategy.generate_signals(df)
    
    # 简单回测
    signals = df[df['signal'] != 0]
    
    if len(signals) == 0:
        return {
            'strategy': strategy.name,
            'symbol': symbol,
            'period': f'{days} days',
            'total_signals': 0,
            'message': 'No signals generated'
        }
    
    # 计算收益
    returns = []
    buy_price = None
    
    for idx, row in signals.iterrows():
        if row['signal'] == 1 and buy_price is None:
            buy_price = row['Close']
        elif row['signal'] == -1 and buy_price is not None:
            ret = (row['Close'] - buy_price) / buy_price
            returns.append(ret)
            buy_price = None
    
    if not returns:
        return {
            'strategy': strategy.name,
            'symbol': symbol,
            'period': f'{days} days',
            'total_signals': len(signals),
            'completed_trades': 0,
            'message': 'No completed trades'
        }
    
    returns = np.array(returns)
    
    return {
        'strategy': strategy.name,
        'symbol': symbol,
        'period': f'{days} days',
        'total_signals': len(signals),
        'completed_trades': len(returns),
        'win_rate': round((returns > 0).mean() * 100, 1),
        'avg_return': round(returns.mean() * 100, 2),
        'total_return': round((np.prod(1 + returns) - 1) * 100, 2),
        'max_gain': round(returns.max() * 100, 2),
        'max_loss': round(returns.min() * 100, 2),
        'sharpe': round(returns.mean() / returns.std(), 2) if returns.std() > 0 else 0
    }


if __name__ == "__main__":
    print("📊 Strategy Implementations Test")
    print("=" * 50)
    
    # 列出所有策略
    print("\n📋 Available Strategies:")
    for s in list_strategies():
        print(f"  - {s['id']}: {s['name']}")
    
    # 测试回测
    print("\n🧪 Testing backtest on NVDA:")
    result = backtest_external_strategy('ema_crossover', 'NVDA', days=365)
    print(f"  Strategy: {result.get('strategy')}")
    print(f"  Win Rate: {result.get('win_rate')}%")
    print(f"  Total Return: {result.get('total_return')}%")
