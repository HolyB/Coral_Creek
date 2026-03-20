#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全区域指标系统 (Safety Zone Indicator)
=========================================

基于通达信公式实现的综合趋势+安全区域指标

核心概念:
- 安全区域: 0-10 高安全, 10-20 安全, 20-50 粉区持币, 50-80 绿区持股, 80-90 风险, 90-100 高风险
- 趋势线: 基于 330 日低点和 210 日高点的长期趋势
- 多重买入信号: 强拉升, 加强拉升, 买半注, BBUY, BUY
- 卖出信号: 减仓, SOLD

使用:
    from strategies.safety_zone_indicator import SafetyZoneIndicator
    
    indicator = SafetyZoneIndicator()
    result = indicator.calculate(df)
    signals = indicator.get_signals(df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def SMA(series, n, m=1):
    """通达信 SMA: Y = (M*X + (N-M)*Y') / N"""
    return pd.Series(series).ewm(alpha=m/n, adjust=False).mean().values


def EMA(series, n):
    """指数移动平均"""
    return pd.Series(series).ewm(span=n, adjust=False).mean().values


def REF(series, n):
    """前 N 日值"""
    return pd.Series(series).shift(n).values


def LLV(series, n):
    """N 日内最低价"""
    return pd.Series(series).rolling(window=n, min_periods=1).min().values


def HHV(series, n):
    """N 日内最高价"""
    return pd.Series(series).rolling(window=n, min_periods=1).max().values


def MA(series, n):
    """简单移动平均"""
    return pd.Series(series).rolling(window=n, min_periods=1).mean().values


def CROSS(series1, series2):
    """series1 向上穿越 series2"""
    s1 = np.array(series1) if hasattr(series1, '__iter__') else np.full(len(series2) if hasattr(series2, '__len__') else 1, series1)
    s2 = np.array(series2) if hasattr(series2, '__iter__') else np.full(len(s1), series2)
    
    if len(s1) < 2:
        return np.array([False])
    
    prev_s1 = np.roll(s1, 1)
    prev_s1[0] = s1[0]
    
    result = (s1 > s2) & (prev_s1 <= s2)
    return result


def EXPMA(series, n):
    """等同于 EMA"""
    return EMA(series, n)


class SafetyZoneIndicator:
    """
    安全区域指标系统
    
    提供:
    1. 安全区域等级 (0-100)
    2. 趋势线方向
    3. 多重买卖信号
    """
    
    def __init__(self):
        self.cache = {}
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        计算所有指标
        
        Returns:
            dict: 包含所有指标值的字典
        """
        if df is None or len(df) < 50:
            return self._empty_result()
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # === 1. 顶底线 (Top-Bottom Line) ===
        rsva1 = (close - LLV(low, 9)) / (HHV(high, 9) - LLV(low, 9) + 1e-10) * 100
        rsva2 = 100 * (HHV(high, 9) - close) / (HHV(high, 9) - LLV(low, 9) + 1e-10)
        var21 = SMA(rsva2, 9, 1) + 100
        var11 = SMA(rsva1, 3, 1)
        var51 = SMA(var11, 3, 1) + 100
        top_bottom_line = var51 - var21 + 50  # 顶底线
        
        # === 2. 趋势线 (Trend Line) - 长期 ===
        var2 = LLV(low, min(330, len(df)))
        var3 = HHV(high, min(210, len(df)))
        with np.errstate(divide='ignore', invalid='ignore'):
            var4 = EMA((close - var2) / (var3 - var2 + 1e-10) * 100, 10) * -1 + 100
        trend_line = 100 - EMA(0.191 * REF(var4, 1) + 0.809 * var4, 1)
        trend_line = np.nan_to_num(trend_line, nan=50)
        
        # === 3. 强拉升信号 ===
        y1 = LLV(low, 17)
        y2 = SMA(np.abs(low - REF(low, 1)), 17, 1)
        y3 = SMA(np.maximum(low - REF(low, 1), 0), 17, 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            q = -(EMA(np.where(low <= y1, y2 / (y3 + 1e-10), -3), 1))
        strong_pullup = CROSS(q, 0)
        
        # === 4. 加强拉升信号 ===
        q1 = (close - MA(close, 40)) / (MA(close, 40) + 1e-10) * 100
        enhanced_pullup = CROSS(q1, -24)
        
        # === 5. 买半注信号 ===
        var22 = EXPMA(EXPMA(EXPMA((2*close + high + low) / 4, 4), 4), 4)
        tian = MA((var22 - REF(var22, 1)) / (REF(var22, 1) + 1e-10) * 100, 2)
        di = MA((var22 - REF(var22, 1)) / (REF(var22, 1) + 1e-10) * 100, 1)
        buy_half = (di > tian) & (di < 0)
        
        # === 6. TREND1 指标 ===
        var1b = (HHV(high, 9) - close) / (HHV(high, 9) - LLV(low, 9) + 1e-10) * 100 - 70
        var2b = SMA(var1b, 9, 1) + 100
        var3b = (close - LLV(low, 9)) / (HHV(high, 9) - LLV(low, 9) + 1e-10) * 100
        var4b = SMA(var3b, 3, 1)
        var5b = SMA(var4b, 3, 1) + 100
        var6b = var5b - var2b
        trend1 = np.where(var6b > 45, var6b - 45, 0)
        
        # === 7. 火焰山指标 ===
        var2q = REF(low, 1)
        var3q = SMA(np.abs(low - var2q), 3, 1) / (SMA(np.maximum(low - var2q, 0), 3, 1) + 1e-10) * 100
        var4q = EMA(np.where(close * 1.3, var3q * 10, var3q / 10), 3)
        var5q = LLV(low, 30)
        var6q = HHV(var4q, 30)
        var7q = np.where(MA(close, min(58, len(df))) > 0, 1, 0)
        var8q = EMA(np.where(low <= var5q, (var4q + var6q * 2) / 2, 0), 3) / 999 * var7q
        fire_mountain = np.clip(var8q, 0, 100)
        
        # === 8. BBUY 信号 (EMA 金叉) ===
        d1 = (close + low + high) / 3
        d2 = EMA(d1, 6)
        d3 = EMA(d2, 5)
        bbuy = CROSS(d2, d3)
        
        # === 9. 减仓信号 ===
        varr1 = SMA(np.maximum(close - REF(close, 1), 0), 6, 1) / (
            SMA(np.abs(close - REF(close, 1)), 6, 1) + 1e-10) * 100
        reduce_position = CROSS(80, varr1)
        
        # === 10. BUY/SOLD 信号 ===
        v1 = (close - LLV(low, 25)) / (HHV(high, 25) - LLV(low, 25) + 1e-10) * 100
        v2 = SMA(v1, 3, 1)
        trend = SMA(v2, 3, 1)
        powerline = SMA(trend, 3, 1)
        
        buy_signal = CROSS(trend, powerline) & (trend < 25)
        sell_signal = CROSS(powerline, trend) & (powerline > 80)
        
        # === 安全区域判断 ===
        safety_level = trend_line[-1] if len(trend_line) > 0 else 50
        
        if safety_level <= 10:
            zone = 'HIGH_SAFETY'
            zone_cn = '高安全区'
        elif safety_level <= 20:
            zone = 'SAFETY'
            zone_cn = '安全区'
        elif safety_level <= 50:
            zone = 'PINK_HOLD_CASH'
            zone_cn = '粉区持币'
        elif safety_level <= 80:
            zone = 'GREEN_HOLD_STOCK'
            zone_cn = '绿区持股'
        elif safety_level <= 90:
            zone = 'RISK'
            zone_cn = '风险区'
        else:
            zone = 'HIGH_RISK'
            zone_cn = '高风险区'
        
        return {
            # 核心指标
            'safety_level': float(safety_level),
            'zone': zone,
            'zone_cn': zone_cn,
            'trend_line': trend_line,
            'top_bottom_line': top_bottom_line,
            
            # 趋势判断
            'trend_up': bool(trend_line[-1] > REF(trend_line, 1)[-1]) if len(trend_line) > 1 else False,
            'top_bottom_up': bool(top_bottom_line[-1] > REF(top_bottom_line, 1)[-1]) if len(top_bottom_line) > 1 else False,
            
            # 买入信号
            'strong_pullup': bool(strong_pullup[-1]) if len(strong_pullup) > 0 else False,
            'enhanced_pullup': bool(enhanced_pullup[-1]) if len(enhanced_pullup) > 0 else False,
            'buy_half': bool(buy_half[-1]) if len(buy_half) > 0 else False,
            'bbuy': bool(bbuy[-1]) if len(bbuy) > 0 else False,
            'buy_signal': bool(buy_signal[-1]) if len(buy_signal) > 0 else False,
            
            # 卖出信号
            'reduce_position': bool(reduce_position[-1]) if len(reduce_position) > 0 else False,
            'sell_signal': bool(sell_signal[-1]) if len(sell_signal) > 0 else False,
            
            # 辅助指标
            'trend1': float(trend1[-1]) if len(trend1) > 0 else 0,
            'fire_mountain': float(fire_mountain[-1]) if len(fire_mountain) > 0 else 0,
        }
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'safety_level': 50,
            'zone': 'UNKNOWN',
            'zone_cn': '未知',
            'trend_line': np.array([50]),
            'top_bottom_line': np.array([50]),
            'trend_up': False,
            'top_bottom_up': False,
            'strong_pullup': False,
            'enhanced_pullup': False,
            'buy_half': False,
            'bbuy': False,
            'buy_signal': False,
            'reduce_position': False,
            'sell_signal': False,
            'trend1': 0,
            'fire_mountain': 0,
        }
    
    def get_signals(self, df: pd.DataFrame) -> Dict:
        """
        获取综合交易信号
        
        Returns:
            dict: {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'strength': 0-5,
                'signals': [...],
                'zone': {...}
            }
        """
        result = self.calculate(df)
        
        buy_signals = []
        sell_signals = []
        
        # 买入信号统计
        if result['strong_pullup']:
            buy_signals.append(('强拉升', 2))
        if result['enhanced_pullup']:
            buy_signals.append(('加强拉升', 2))
        if result['buy_half']:
            buy_signals.append(('买半注', 1))
        if result['bbuy']:
            buy_signals.append(('BBUY金叉', 1))
        if result['buy_signal']:
            buy_signals.append(('低位BUY', 2))
        
        # 区域加成
        if result['zone'] in ['HIGH_SAFETY', 'SAFETY']:
            buy_signals.append(('安全区域加成', 1))
        
        # 卖出信号统计
        if result['reduce_position']:
            sell_signals.append(('动量减仓', 2))
        if result['sell_signal']:
            sell_signals.append(('高位SOLD', 2))
        
        # 区域风险
        if result['zone'] in ['HIGH_RISK', 'RISK']:
            sell_signals.append(('风险区域', 1))
        
        # 计算综合得分
        buy_score = sum(s[1] for s in buy_signals)
        sell_score = sum(s[1] for s in sell_signals)
        
        # 决策
        if buy_score >= 3 and buy_score > sell_score:
            action = 'BUY'
            strength = min(5, buy_score)
        elif sell_score >= 3 and sell_score > buy_score:
            action = 'SELL'
            strength = min(5, sell_score)
        else:
            action = 'HOLD'
            strength = 0
        
        return {
            'action': action,
            'strength': strength,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'zone': {
                'level': result['safety_level'],
                'name': result['zone'],
                'name_cn': result['zone_cn'],
            },
            'trend_up': result['trend_up'],
            'fire_mountain': result['fire_mountain'],
        }
    
    def get_consensus_with_blue(self, df: pd.DataFrame, blue_value: float) -> Dict:
        """
        与 BLUE 信号进行共识分析
        
        Args:
            df: 价格数据
            blue_value: BLUE 信号值 (0-200)
            
        Returns:
            dict: 综合分析结果
        """
        zone_signals = self.get_signals(df)
        
        # BLUE 信号分析
        blue_strong = blue_value >= 100
        blue_medium = 50 <= blue_value < 100
        blue_weak = blue_value < 50
        
        # 安全区域分析
        in_safe_zone = zone_signals['zone']['name'] in ['HIGH_SAFETY', 'SAFETY', 'PINK_HOLD_CASH']
        in_risk_zone = zone_signals['zone']['name'] in ['HIGH_RISK', 'RISK']
        
        # 共识判断
        consensus = 'NEUTRAL'
        confidence = 0
        
        if blue_strong and in_safe_zone and zone_signals['action'] == 'BUY':
            consensus = 'STRONG_BUY'
            confidence = 95
        elif blue_medium and in_safe_zone:
            consensus = 'BUY'
            confidence = 75
        elif blue_weak and in_risk_zone:
            consensus = 'AVOID'
            confidence = 80
        elif zone_signals['action'] == 'SELL' and in_risk_zone:
            consensus = 'SELL'
            confidence = 85
        elif blue_strong and in_risk_zone:
            consensus = 'CAUTION'  # BLUE 强但在风险区
            confidence = 60
        
        return {
            'consensus': consensus,
            'confidence': confidence,
            'blue_value': blue_value,
            'zone': zone_signals['zone'],
            'signals': zone_signals,
            'reasoning': self._get_reasoning(consensus, blue_value, zone_signals),
        }
    
    def _get_reasoning(self, consensus: str, blue_value: float, signals: Dict) -> str:
        """生成推荐理由"""
        zone_cn = signals['zone']['name_cn']
        buy_count = len(signals['buy_signals'])
        sell_count = len(signals['sell_signals'])
        
        reasons = {
            'STRONG_BUY': f"BLUE信号强势({blue_value:.0f})，处于{zone_cn}，{buy_count}个买入信号共振",
            'BUY': f"BLUE信号适中({blue_value:.0f})，处于{zone_cn}，建议逢低布局",
            'AVOID': f"BLUE信号弱({blue_value:.0f})，处于{zone_cn}，建议观望",
            'SELL': f"处于{zone_cn}，有{sell_count}个卖出信号，建议减仓",
            'CAUTION': f"BLUE信号强({blue_value:.0f})但处于{zone_cn}，注意风险",
            'NEUTRAL': f"信号不明确，处于{zone_cn}，建议观望",
        }
        return reasons.get(consensus, "无明确信号")


def calculate_safety_zone(df: pd.DataFrame) -> Dict:
    """便捷函数: 计算安全区域指标"""
    indicator = SafetyZoneIndicator()
    return indicator.calculate(df)


def get_safety_zone_signals(df: pd.DataFrame) -> Dict:
    """便捷函数: 获取安全区域交易信号"""
    indicator = SafetyZoneIndicator()
    return indicator.get_signals(df)


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
    
    from data_fetcher import get_stock_data
    
    print("=== 安全区域指标测试 ===\n")
    
    for symbol in ['AAPL', 'NVDA', 'TSLA']:
        df = get_stock_data(symbol, 'US', days=400)
        if df is not None:
            indicator = SafetyZoneIndicator()
            result = indicator.calculate(df)
            signals = indicator.get_signals(df)
            
            print(f"{symbol}:")
            print(f"  安全度: {result['safety_level']:.1f} ({result['zone_cn']})")
            print(f"  趋势向上: {result['trend_up']}")
            print(f"  买入信号: {[s[0] for s in signals['buy_signals']]}")
            print(f"  卖出信号: {[s[0] for s in signals['sell_signals']]}")
            print(f"  综合建议: {signals['action']} (强度 {signals['strength']})")
            print()
