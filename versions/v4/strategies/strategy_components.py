#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略组件模块 - 可自由组合的买入/卖出条件
==============================================

使用方法:
    from strategies.strategy_components import StrategyBuilder
    
    # 创建策略
    strategy = StrategyBuilder()
    
    # 选择买入条件 (可多选，满足任一即可)
    strategy.add_buy_condition('blue_heima')        # BLUE>=100 + 黑马共振
    strategy.add_buy_condition('strong_blue')       # BLUE>=150 + 黑马
    strategy.add_buy_condition('bottom_peak')       # 底部筹码顶格峰
    
    # 选择卖出条件 (可多选，满足任一即可)
    strategy.add_sell_condition('kdj_overbought')   # KDJ J > 90
    strategy.add_sell_condition('chip_distribution') # 筹码顶部堆积
    strategy.add_sell_condition('ma_break')         # 跌破MA5
    
    # 使用策略
    should_buy, reason = strategy.check_buy(data, i, df)
    should_sell, reason = strategy.check_sell(data, i, df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field


# ============================================================================
# 买入条件函数
# ============================================================================

def buy_blue_heima(data: Dict, i: int, df: pd.DataFrame = None, 
                   blue_threshold: int = 100) -> Tuple[bool, str]:
    """
    买入条件: BLUE + 黑马共振
    
    条件:
    - 日BLUE >= threshold + (日黑马/掘地 OR 周黑马/掘地)
    - OR (日黑马/掘地 OR 周黑马/掘地) + (近5天BLUE >= threshold OR 周BLUE >= threshold)
    
    参数:
        blue_threshold: BLUE阈值，默认100
    """
    d_blue = data['blue'][i]
    d_heima = data['heima'][i]
    d_juedi = data['juedi'][i]
    w_blue = data['week_blue'][i]
    w_heima = data['week_heima'][i]
    w_juedi = data['week_juedi'][i]
    
    # 近5天BLUE检查
    start_idx = max(0, i - 4)
    recent_blues = data['blue'][start_idx:i+1]
    has_recent_blue = np.any(recent_blues >= blue_threshold)
    
    # 黑马上下文
    has_heima_context = d_heima or d_juedi or w_heima or w_juedi
    
    # 条件1: 日BLUE触发 + 黑马上下文
    if d_blue >= blue_threshold and has_heima_context:
        heima_type = '黑马' if d_heima else '掘地' if d_juedi else '周黑马' if w_heima else '周掘地'
        return True, f"BLUE{d_blue:.0f}+{heima_type}"
    
    # 条件2: 黑马触发 + BLUE上下文
    if has_heima_context and (has_recent_blue or w_blue >= blue_threshold):
        trigger = "日黑马" if d_heima else "日掘地" if d_juedi else "周黑马" if w_heima else "周掘地"
        blue_src = f"日BLUE{d_blue:.0f}" if has_recent_blue else f"周BLUE{w_blue:.0f}"
        return True, f"{trigger}+{blue_src}"
    
    return False, ""


def buy_strong_blue(data: Dict, i: int, df: pd.DataFrame = None,
                    blue_threshold: int = 150) -> Tuple[bool, str]:
    """
    买入条件: 强BLUE + 黑马共振
    
    条件: (日BLUE >= 150 OR 周BLUE >= 150) AND (黑马 OR 掘地)
    """
    d_blue = data['blue'][i]
    d_heima = data['heima'][i]
    d_juedi = data['juedi'][i]
    w_blue = data['week_blue'][i]
    w_heima = data['week_heima'][i]
    w_juedi = data['week_juedi'][i]
    
    has_strong_blue = d_blue >= blue_threshold or w_blue >= blue_threshold
    has_heima = d_heima or d_juedi or w_heima or w_juedi
    
    if has_strong_blue and has_heima:
        blue_src = f"日BLUE{d_blue:.0f}" if d_blue >= blue_threshold else f"周BLUE{w_blue:.0f}"
        heima_type = "日黑马" if d_heima else "日掘地" if d_juedi else "周黑马" if w_heima else "周掘地"
        return True, f"{blue_src}+{heima_type}"
    
    return False, ""


def buy_double_blue(data: Dict, i: int, df: pd.DataFrame = None,
                    blue_threshold: int = 150) -> Tuple[bool, str]:
    """
    买入条件: 日周BLUE双重共振
    
    条件: 日BLUE >= 150 AND 周BLUE >= 150 AND (黑马 OR 掘地)
    """
    d_blue = data['blue'][i]
    d_heima = data['heima'][i]
    d_juedi = data['juedi'][i]
    w_blue = data['week_blue'][i]
    w_heima = data['week_heima'][i]
    w_juedi = data['week_juedi'][i]
    
    has_heima = d_heima or d_juedi or w_heima or w_juedi
    
    if d_blue >= blue_threshold and w_blue >= blue_threshold and has_heima:
        heima_type = "日黑马" if d_heima else "日掘地" if d_juedi else "周黑马" if w_heima else "周掘地"
        return True, f"日BLUE{d_blue:.0f}+周BLUE{w_blue:.0f}+{heima_type}"
    
    return False, ""


def buy_bottom_peak(data: Dict, i: int, df: pd.DataFrame,
                    decay_factor: float = 0.97) -> Tuple[bool, str]:
    """
    买入条件: 底部筹码顶格峰
    
    条件:
    - POC位置 < 30% (筹码峰在底部)
    - 底部筹码占比 > 50%
    - 单峰最大占比 > 5%
    - 同时需要 BLUE >= 100 或 黑马/掘地 确认
    """
    if i < 50 or df is None:
        return False, ""
    
    d_blue = data['blue'][i]
    d_heima = data['heima'][i]
    d_juedi = data['juedi'][i]
    
    sub_df = df.iloc[:i+1]
    result = _analyze_bottom_peak(sub_df, decay_factor)
    
    if result and result['is_strong_bottom_peak']:
        # 需要BLUE或黑马确认
        if d_blue >= 100 or d_heima or d_juedi:
            confirm = f"BLUE{d_blue:.0f}" if d_blue >= 100 else "黑马" if d_heima else "掘地"
            return True, f"底部顶格峰+{confirm}"
    
    return False, ""


def buy_blue_only(data: Dict, i: int, df: pd.DataFrame = None,
                  blue_threshold: int = 200) -> Tuple[bool, str]:
    """
    买入条件: 纯BLUE信号 (无需黑马)
    
    条件: 日BLUE >= 200
    """
    d_blue = data['blue'][i]
    
    if d_blue >= blue_threshold:
        return True, f"超强BLUE{d_blue:.0f}"
    
    return False, ""


def buy_heima_only(data: Dict, i: int, df: pd.DataFrame = None) -> Tuple[bool, str]:
    """
    买入条件: 纯黑马/掘地信号
    
    条件: 日黑马 OR 日掘地
    """
    d_heima = data['heima'][i]
    d_juedi = data['juedi'][i]
    
    if d_heima:
        return True, "日黑马"
    if d_juedi:
        return True, "日掘地"
    
    return False, ""


def buy_safety_zone(data: Dict, i: int, df: pd.DataFrame,
                    zone_threshold: int = 30) -> Tuple[bool, str]:
    """
    买入条件: 安全区域低位买入
    
    条件: 安全区域 < 30 (深度低估)
    需要配合其他条件使用
    """
    if 'safety_zone' not in data or df is None:
        return False, ""
    
    zone = data['safety_zone'][i]
    if zone < zone_threshold:
        return True, f"安全区域{zone:.0f}"
    
    return False, ""


# ============================================================================
# 卖出条件函数
# ============================================================================

def sell_kdj_overbought(data: Dict, i: int, df: pd.DataFrame = None,
                        j_threshold: int = 90) -> Tuple[bool, str]:
    """
    卖出条件: KDJ超买
    
    条件: KDJ J > 90
    """
    kdj_j = data['kdj_j'][i]
    
    if kdj_j > j_threshold:
        return True, f"KDJ J={kdj_j:.0f}>{j_threshold}"
    
    return False, ""


def sell_ma_break(data: Dict, i: int, df: pd.DataFrame = None,
                  consecutive_days: int = 1) -> Tuple[bool, str]:
    """
    卖出条件: 跌破均线
    
    条件: 收盘价 < MA5 (可配置连续天数)
    
    参数:
        consecutive_days: 需要连续跌破的天数，默认1天
    """
    close = data['close'][i]
    ma5 = data['ma5'][i]
    
    if np.isnan(ma5):
        return False, ""
    
    if consecutive_days == 1:
        if close < ma5:
            return True, f"跌破MA5({ma5:.2f})"
    else:
        # 检查连续天数
        for j in range(consecutive_days):
            idx = i - j
            if idx < 0:
                return False, ""
            if data['close'][idx] >= data['ma5'][idx]:
                return False, ""
        return True, f"连续{consecutive_days}天跌破MA5"
    
    return False, ""


def sell_chip_distribution(data: Dict, i: int, df: pd.DataFrame,
                           lookback_days: int = 20,
                           top_increase_threshold: float = 3.0,
                           bottom_decrease_threshold: float = 3.0) -> Tuple[bool, str]:
    """
    卖出条件: 筹码分布异常
    
    条件: 满足以下2个或以上
    - 顶部筹码增加 > 3%
    - 底部筹码减少 > 3%
    - 平均成本上涨 > 2%
    """
    if i < 50 or df is None:
        return False, ""
    
    sub_df = df.iloc[:i+1]
    result = _analyze_chip_distribution(sub_df, lookback_days)
    
    if result and result['should_sell']:
        return True, f"筹码出货({result['description']})"
    
    return False, ""


def sell_chip_with_ma(data: Dict, i: int, df: pd.DataFrame,
                      lookback_days: int = 20) -> Tuple[bool, str]:
    """
    卖出条件: 跌破MA5 + 筹码异常
    
    条件: 收盘价 < MA5 AND 筹码有异常信号
    """
    close = data['close'][i]
    ma5 = data['ma5'][i]
    
    if np.isnan(ma5) or close >= ma5:
        return False, ""
    
    if i < 50 or df is None:
        return False, ""
    
    sub_df = df.iloc[:i+1]
    result = _analyze_chip_distribution(sub_df, lookback_days)
    
    if result and result['sell_score'] >= 1:
        return True, f"跌破MA5+{result['description']}"
    
    return False, ""


def sell_safety_zone_high(data: Dict, i: int, df: pd.DataFrame = None,
                          zone_threshold: int = 90) -> Tuple[bool, str]:
    """
    卖出条件: 安全区域高风险
    
    条件: 安全区域 > 90
    """
    if 'safety_zone' not in data:
        return False, ""
    
    zone = data['safety_zone'][i]
    if zone > zone_threshold:
        return True, f"高风险区{zone:.0f}"
    
    return False, ""


def sell_profit_target(data: Dict, i: int, df: pd.DataFrame = None,
                       target_pct: float = 20.0,
                       entry_price: float = None) -> Tuple[bool, str]:
    """
    卖出条件: 止盈
    
    条件: 收益 >= target_pct%
    
    参数:
        target_pct: 目标收益率，默认20%
        entry_price: 买入价格 (需要从外部传入)
    """
    if entry_price is None or entry_price <= 0:
        return False, ""
    
    close = data['close'][i]
    profit_pct = (close / entry_price - 1) * 100
    
    if profit_pct >= target_pct:
        return True, f"止盈{profit_pct:.1f}%"
    
    return False, ""


def sell_stop_loss(data: Dict, i: int, df: pd.DataFrame = None,
                   stop_pct: float = -8.0,
                   entry_price: float = None) -> Tuple[bool, str]:
    """
    卖出条件: 止损
    
    条件: 亏损 <= stop_pct%
    
    参数:
        stop_pct: 止损线，默认-8%
        entry_price: 买入价格 (需要从外部传入)
    """
    if entry_price is None or entry_price <= 0:
        return False, ""
    
    close = data['close'][i]
    profit_pct = (close / entry_price - 1) * 100
    
    if profit_pct <= stop_pct:
        return True, f"止损{profit_pct:.1f}%"
    
    return False, ""


def sell_trailing_stop(data: Dict, i: int, df: pd.DataFrame = None,
                       trail_pct: float = 10.0,
                       peak_price: float = None) -> Tuple[bool, str]:
    """
    卖出条件: 移动止损
    
    条件: 从最高点回撤 >= trail_pct%
    
    参数:
        trail_pct: 回撤止损线，默认10%
        peak_price: 持仓期间最高价 (需要从外部传入)
    """
    if peak_price is None or peak_price <= 0:
        return False, ""
    
    close = data['close'][i]
    drawdown = (peak_price - close) / peak_price * 100
    
    if drawdown >= trail_pct:
        return True, f"回撤止损{drawdown:.1f}%"
    
    return False, ""


# ============================================================================
# 辅助函数
# ============================================================================

def _analyze_chip_distribution(df: pd.DataFrame, lookback_days: int = 20,
                               decay_factor: float = 0.97) -> Optional[Dict]:
    """分析筹码分布"""
    if len(df) < lookback_days + 30:
        return None
    
    def calc_chip_distribution(data, price_min, price_max, bins=70):
        bin_size = (price_max - price_min) / bins if price_max > price_min else 1
        volume_profile = np.zeros(bins)
        price_bins = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        total_days = len(data)
        for i, (idx, row) in enumerate(data.iterrows()):
            days_ago = total_days - 1 - i
            time_weight = decay_factor ** days_ago
            weighted_vol = row['Volume'] * time_weight
            
            day_high, day_low, day_close = row['High'], row['Low'], row['Close']
            
            if day_high == day_low or bin_size == 0:
                bin_idx = int((day_close - price_min) / bin_size)
                bin_idx = min(max(bin_idx, 0), bins - 1)
                volume_profile[bin_idx] += weighted_vol
            else:
                start_bin = max(int((day_low - price_min) / bin_size), 0)
                end_bin = min(int((day_high - price_min) / bin_size), bins - 1)
                close_bin = min(max(int((day_close - price_min) / bin_size), start_bin), end_bin)
                
                for b in range(start_bin, end_bin + 1):
                    dist_to_close = abs(b - close_bin)
                    max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                    weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                    volume_profile[b] += weighted_vol * weight
        
        return volume_profile, bin_centers
    
    price_min, price_max = df['Low'].min(), df['High'].max()
    bins = 70
    
    df_past = df.iloc[:-lookback_days]
    past_profile, bin_centers = calc_chip_distribution(df_past, price_min, price_max, bins)
    current_profile, _ = calc_chip_distribution(df, price_min, price_max, bins)
    
    past_total = np.sum(past_profile)
    current_total = np.sum(current_profile)
    if past_total > 0:
        past_profile = past_profile / past_total
    if current_total > 0:
        current_profile = current_profile / current_total
    
    current_close = df['Close'].iloc[-1]
    
    # 顶部筹码
    top_threshold = current_close * 1.05
    top_bins = bin_centers > top_threshold
    top_increase = np.sum(current_profile[top_bins]) * 100 - np.sum(past_profile[top_bins]) * 100
    
    # 底部筹码
    bottom_threshold = current_close * 0.85
    bottom_bins = bin_centers < bottom_threshold
    bottom_decrease = np.sum(past_profile[bottom_bins]) * 100 - np.sum(current_profile[bottom_bins]) * 100
    
    # 成本变化
    past_avg_cost = np.sum(bin_centers * past_profile) if past_total > 0 else current_close
    current_avg_cost = np.sum(bin_centers * current_profile) if current_total > 0 else current_close
    cost_change_pct = (current_avg_cost - past_avg_cost) / past_avg_cost * 100 if past_avg_cost > 0 else 0
    
    is_top_heavy = top_increase > 3
    is_bottom_light = bottom_decrease > 3
    is_cost_rising = cost_change_pct > 2
    
    sell_score = sum([is_top_heavy, is_bottom_light, is_cost_rising])
    should_sell = sell_score >= 2
    
    parts = []
    if is_top_heavy:
        parts.append(f"顶部+{top_increase:.1f}%")
    if is_bottom_light:
        parts.append(f"底部-{bottom_decrease:.1f}%")
    if is_cost_rising:
        parts.append(f"成本+{cost_change_pct:.1f}%")
    
    return {
        'should_sell': should_sell,
        'sell_score': sell_score,
        'top_increase': top_increase,
        'bottom_decrease': bottom_decrease,
        'cost_change_pct': cost_change_pct,
        'description': ", ".join(parts) if parts else "筹码正常"
    }


def _analyze_bottom_peak(df: pd.DataFrame, decay_factor: float = 0.97) -> Optional[Dict]:
    """分析底部筹码峰"""
    if len(df) < 50:
        return None
    
    price_min, price_max = df['Low'].min(), df['High'].max()
    bins = 70
    bin_size = (price_max - price_min) / bins if price_max > price_min else 1
    
    volume_profile = np.zeros(bins)
    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    total_days = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        days_ago = total_days - 1 - i
        time_weight = decay_factor ** days_ago
        weighted_vol = row['Volume'] * time_weight
        
        day_high, day_low, day_close = row['High'], row['Low'], row['Close']
        
        if day_high == day_low or bin_size == 0:
            bin_idx = int((day_close - price_min) / bin_size)
            bin_idx = min(max(bin_idx, 0), bins - 1)
            volume_profile[bin_idx] += weighted_vol
        else:
            start_bin = max(int((day_low - price_min) / bin_size), 0)
            end_bin = min(int((day_high - price_min) / bin_size), bins - 1)
            close_bin = min(max(int((day_close - price_min) / bin_size), start_bin), end_bin)
            
            for b in range(start_bin, end_bin + 1):
                dist_to_close = abs(b - close_bin)
                max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                volume_profile[b] += weighted_vol * weight
    
    total_vol = np.sum(volume_profile)
    if total_vol == 0:
        return None
    
    poc_idx = np.argmax(volume_profile)
    poc_price = bin_centers[poc_idx]
    poc_position = (poc_price - price_min) / (price_max - price_min) * 100
    
    max_chip_pct = np.max(volume_profile) / total_vol * 100
    
    bottom_30_price = price_min + (price_max - price_min) * 0.30
    bottom_chip_ratio = sum(volume_profile[i] for i, p in enumerate(bin_centers) if p <= bottom_30_price)
    bottom_chip_ratio = bottom_chip_ratio / total_vol * 100
    
    is_strong_bottom_peak = (poc_position < 30) and (bottom_chip_ratio > 50) and (max_chip_pct > 5)
    is_bottom_peak = (poc_position < 35) and (bottom_chip_ratio > 35)
    
    return {
        'is_strong_bottom_peak': is_strong_bottom_peak,
        'is_bottom_peak': is_bottom_peak,
        'poc_position': poc_position,
        'bottom_chip_ratio': bottom_chip_ratio,
        'max_chip_pct': max_chip_pct
    }


# ============================================================================
# 策略注册表
# ============================================================================

@dataclass
class StrategyCondition:
    """策略条件"""
    name: str
    description: str
    func: Callable
    params: Dict = field(default_factory=dict)


# 买入条件注册表
BUY_CONDITIONS = {
    'blue_heima': StrategyCondition(
        name='blue_heima',
        description='BLUE>=100 + 黑马/掘地共振',
        func=buy_blue_heima,
        params={'blue_threshold': 100}
    ),
    'strong_blue': StrategyCondition(
        name='strong_blue', 
        description='(日BLUE>=150 OR 周BLUE>=150) + 黑马/掘地',
        func=buy_strong_blue,
        params={'blue_threshold': 150}
    ),
    'double_blue': StrategyCondition(
        name='double_blue',
        description='日BLUE>=150 AND 周BLUE>=150 + 黑马/掘地',
        func=buy_double_blue,
        params={'blue_threshold': 150}
    ),
    'bottom_peak': StrategyCondition(
        name='bottom_peak',
        description='底部筹码顶格峰 + BLUE/黑马确认',
        func=buy_bottom_peak,
        params={'decay_factor': 0.97}
    ),
    'blue_only': StrategyCondition(
        name='blue_only',
        description='超强BLUE>=200 (无需黑马)',
        func=buy_blue_only,
        params={'blue_threshold': 200}
    ),
    'heima_only': StrategyCondition(
        name='heima_only',
        description='日黑马 OR 日掘地',
        func=buy_heima_only,
        params={}
    ),
    'safety_zone_low': StrategyCondition(
        name='safety_zone_low',
        description='安全区域 < 30 (深度低估)',
        func=buy_safety_zone,
        params={'zone_threshold': 30}
    ),
}

# 卖出条件注册表
SELL_CONDITIONS = {
    'kdj_overbought': StrategyCondition(
        name='kdj_overbought',
        description='KDJ J > 90 超买',
        func=sell_kdj_overbought,
        params={'j_threshold': 90}
    ),
    'ma_break': StrategyCondition(
        name='ma_break',
        description='跌破MA5',
        func=sell_ma_break,
        params={'consecutive_days': 1}
    ),
    'ma_break_2day': StrategyCondition(
        name='ma_break_2day',
        description='连续2天跌破MA5',
        func=sell_ma_break,
        params={'consecutive_days': 2}
    ),
    'chip_distribution': StrategyCondition(
        name='chip_distribution',
        description='筹码顶部堆积 + 底部减少',
        func=sell_chip_distribution,
        params={'lookback_days': 20}
    ),
    'chip_with_ma': StrategyCondition(
        name='chip_with_ma',
        description='跌破MA5 + 筹码异常',
        func=sell_chip_with_ma,
        params={'lookback_days': 20}
    ),
    'safety_zone_high': StrategyCondition(
        name='safety_zone_high',
        description='安全区域 > 90 高风险',
        func=sell_safety_zone_high,
        params={'zone_threshold': 90}
    ),
    'profit_target_20': StrategyCondition(
        name='profit_target_20',
        description='止盈20%',
        func=sell_profit_target,
        params={'target_pct': 20.0}
    ),
    'stop_loss_8': StrategyCondition(
        name='stop_loss_8',
        description='止损-8%',
        func=sell_stop_loss,
        params={'stop_pct': -8.0}
    ),
    'trailing_stop_10': StrategyCondition(
        name='trailing_stop_10',
        description='回撤10%止损',
        func=sell_trailing_stop,
        params={'trail_pct': 10.0}
    ),
}


# ============================================================================
# 策略构建器
# ============================================================================

class StrategyBuilder:
    """
    策略构建器 - 自由组合买入/卖出条件
    
    使用示例:
        strategy = StrategyBuilder("我的策略")
        
        # 添加买入条件 (满足任一即可触发)
        strategy.add_buy_condition('blue_heima')
        strategy.add_buy_condition('bottom_peak')
        
        # 添加卖出条件 (满足任一即可触发)
        strategy.add_sell_condition('kdj_overbought')
        strategy.add_sell_condition('chip_distribution')
        
        # 检查信号
        should_buy, reason = strategy.check_buy(data, i, df)
        should_sell, reason = strategy.check_sell(data, i, df)
    """
    
    def __init__(self, name: str = "自定义策略"):
        self.name = name
        self.buy_conditions: List[StrategyCondition] = []
        self.sell_conditions: List[StrategyCondition] = []
        self.entry_price: float = None
        self.peak_price: float = None
    
    def add_buy_condition(self, condition_name: str, **custom_params) -> 'StrategyBuilder':
        """添加买入条件"""
        if condition_name not in BUY_CONDITIONS:
            raise ValueError(f"未知买入条件: {condition_name}. 可用: {list(BUY_CONDITIONS.keys())}")
        
        condition = BUY_CONDITIONS[condition_name]
        # 合并自定义参数
        params = {**condition.params, **custom_params}
        self.buy_conditions.append(StrategyCondition(
            name=condition.name,
            description=condition.description,
            func=condition.func,
            params=params
        ))
        return self
    
    def add_sell_condition(self, condition_name: str, **custom_params) -> 'StrategyBuilder':
        """添加卖出条件"""
        if condition_name not in SELL_CONDITIONS:
            raise ValueError(f"未知卖出条件: {condition_name}. 可用: {list(SELL_CONDITIONS.keys())}")
        
        condition = SELL_CONDITIONS[condition_name]
        params = {**condition.params, **custom_params}
        self.sell_conditions.append(StrategyCondition(
            name=condition.name,
            description=condition.description,
            func=condition.func,
            params=params
        ))
        return self
    
    def set_entry_price(self, price: float):
        """设置买入价格 (用于止盈止损计算)"""
        self.entry_price = price
        self.peak_price = price
    
    def update_peak_price(self, current_price: float):
        """更新最高价 (用于移动止损)"""
        if self.peak_price is None or current_price > self.peak_price:
            self.peak_price = current_price
    
    def reset_position(self):
        """重置持仓状态"""
        self.entry_price = None
        self.peak_price = None
    
    def check_buy(self, data: Dict, i: int, df: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        检查买入条件
        
        Returns:
            (should_buy, reason): 是否买入及原因
        """
        for condition in self.buy_conditions:
            params = dict(condition.params)
            # 添加特殊参数
            if 'entry_price' in params:
                params['entry_price'] = self.entry_price
            
            try:
                result, reason = condition.func(data, i, df, **params)
                if result:
                    return True, reason
            except Exception as e:
                continue
        
        return False, ""
    
    def check_sell(self, data: Dict, i: int, df: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        检查卖出条件
        
        Returns:
            (should_sell, reason): 是否卖出及原因
        """
        for condition in self.sell_conditions:
            params = dict(condition.params)
            # 添加特殊参数
            if condition.name in ['profit_target_20', 'stop_loss_8']:
                params['entry_price'] = self.entry_price
            if condition.name == 'trailing_stop_10':
                params['peak_price'] = self.peak_price
            
            try:
                result, reason = condition.func(data, i, df, **params)
                if result:
                    return True, reason
            except Exception as e:
                continue
        
        return False, ""
    
    def describe(self) -> str:
        """描述策略"""
        lines = [f"策略: {self.name}", ""]
        
        lines.append("买入条件 (满足任一):")
        for c in self.buy_conditions:
            lines.append(f"  - {c.description}")
        
        lines.append("")
        lines.append("卖出条件 (满足任一):")
        for c in self.sell_conditions:
            lines.append(f"  - {c.description}")
        
        return "\n".join(lines)


# ============================================================================
# 预设策略模板
# ============================================================================

def create_original_strategy() -> StrategyBuilder:
    """创建原有系统策略"""
    return (StrategyBuilder("原有系统")
            .add_buy_condition('blue_heima')
            .add_sell_condition('kdj_overbought')
            .add_sell_condition('ma_break_2day'))


def create_chip_strategy() -> StrategyBuilder:
    """创建筹码增强策略"""
    return (StrategyBuilder("筹码增强策略")
            .add_buy_condition('blue_heima')
            .add_sell_condition('kdj_overbought')
            .add_sell_condition('chip_distribution')
            .add_sell_condition('chip_with_ma'))


def create_enhanced_strategy() -> StrategyBuilder:
    """创建增强策略"""
    return (StrategyBuilder("增强策略")
            .add_buy_condition('strong_blue')
            .add_buy_condition('bottom_peak')
            .add_sell_condition('kdj_overbought')
            .add_sell_condition('chip_distribution'))


def create_aggressive_strategy() -> StrategyBuilder:
    """创建激进策略"""
    return (StrategyBuilder("激进策略")
            .add_buy_condition('heima_only')
            .add_sell_condition('profit_target_20')
            .add_sell_condition('stop_loss_8')
            .add_sell_condition('trailing_stop_10'))


def create_conservative_strategy() -> StrategyBuilder:
    """创建保守策略"""
    return (StrategyBuilder("保守策略")
            .add_buy_condition('double_blue')
            .add_buy_condition('bottom_peak')
            .add_sell_condition('kdj_overbought')
            .add_sell_condition('chip_distribution')
            .add_sell_condition('safety_zone_high'))


# ============================================================================
# 工具函数
# ============================================================================

def list_all_conditions():
    """列出所有可用条件"""
    print("=" * 60)
    print("可用买入条件")
    print("=" * 60)
    for name, cond in BUY_CONDITIONS.items():
        print(f"  {name:20} - {cond.description}")
    
    print()
    print("=" * 60)
    print("可用卖出条件")
    print("=" * 60)
    for name, cond in SELL_CONDITIONS.items():
        print(f"  {name:20} - {cond.description}")


if __name__ == "__main__":
    # 列出所有可用条件
    list_all_conditions()
    
    print("\n" + "=" * 60)
    print("预设策略示例")
    print("=" * 60)
    
    # 展示预设策略
    for create_func in [create_original_strategy, create_chip_strategy, 
                        create_enhanced_strategy, create_conservative_strategy]:
        strategy = create_func()
        print()
        print(strategy.describe())
