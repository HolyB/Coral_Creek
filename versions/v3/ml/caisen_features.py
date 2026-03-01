#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蔡森指标特征工程 — Caisen Chart Features
==========================================
将蔡森图表中的所有曲线值转为 ML 特征

特征列表:
1. 海底捞月 (BLUE)
   - blue_value: BLUE 柱值 (0-200)
   - blue_slope_3d: BLUE 3日斜率
   - blue_days_active: 连续有 BLUE 天数
   - blue_disappear: BLUE 刚消失 (买点, 0/1)

2. 负向海底捞月 (LIRED)
   - lired_value: LIRED 柱值
   - lired_disappear: LIRED 刚消失 (卖点, 0/1)

3. 资金力度 (红/黄/绿)
   - fund_red: 超大单流入
   - fund_yellow: 大单流入  
   - fund_green: 资金流出
   - fund_net: 净资金 = red + yellow + green
   - fund_flow_line: 资金流量线 (lightblue)
   - fund_flow_slope_5d: 资金流量5日斜率

4. PINK 主力线 (KDJ变体)
   - pink_value: PINK 值 (0-100)
   - pink_below_10: PINK < 10 (底部, 0/1)
   - pink_above_90: PINK > 90 (顶部, 0/1)
   - pink_slope_3d: PINK 3日斜率
   - pink_cross_up_10: PINK 上穿 10 (买, 0/1)

5. 黑马信号系统
   - cci_value: CCI 值 (归一化)
   - kdj_k/d/j: KDJ 值
   - heima_signal: 黑马 (0/1)
   - juedi_signal: 掘地 (0/1)
   - golden_bottom: 黄金底 (0/1)
   - main_force: 主力进场 (0/1)
   - washing: 洗盘 (0/1)
   - var50_value: 海底捞月原始值
   - var50_rising: var50 递增 (主力进场)
   - bot_golden_cross: 底部金叉 (0/1)
   - two_golden_cross: 二次金叉 (0/1)
   - top_divergence: 顶背离 (0/1)

6. 复合特征
   - signal_strength: 信号综合强度
   - bull_bear_ratio: 多空力量比
   - caisen_regime: 蔡森状态 (底部/上升/顶部/下降)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_caisen_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    从 OHLCV 历史数据计算所有蔡森图表特征
    
    Args:
        hist: DataFrame with Open, High, Low, Close, Volume columns
              index should be DatetimeIndex
    
    Returns:
        DataFrame with all caisen features, same index as input
    """
    from indicator_utils import (
        calculate_blue_signal_series,
        calculate_heima_full,
        calculate_phantom_indicator,
    )
    
    # 标准化列名
    for need in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if need not in hist.columns:
            for c in hist.columns:
                if c.lower() == need.lower():
                    hist = hist.rename(columns={c: need})
    
    required = ['Open', 'High', 'Low', 'Close']
    if not all(c in hist.columns for c in required):
        return pd.DataFrame(index=hist.index)
    
    o = hist['Open'].values.astype(float)
    h = hist['High'].values.astype(float)
    l = hist['Low'].values.astype(float)
    c = hist['Close'].values.astype(float)
    v = hist['Volume'].values.astype(float) if 'Volume' in hist.columns else np.ones(len(hist))
    n = len(c)
    
    features = pd.DataFrame(index=hist.index)
    
    # === 1. BLUE 海底捞月 ===
    blue = calculate_blue_signal_series(o, h, l, c)
    features['cs_blue_value'] = blue
    features['cs_blue_slope_3d'] = pd.Series(blue).diff(3).values / 3
    features['cs_blue_active'] = _consecutive_count(blue > 0)
    features['cs_blue_disappear'] = _signal_disappear(blue > 0)
    features['cs_blue_pct'] = blue / 200.0  # 归一化 0-1
    
    # === 2. 幻影主力指标 ===
    phantom = calculate_phantom_indicator(o, h, l, c, v)
    
    # LIRED (逃顶)
    features['cs_lired_value'] = phantom['lired']
    features['cs_lired_disappear'] = phantom['lired_disappear'].astype(float)
    
    # 资金力度
    features['cs_fund_red'] = phantom['red']        # 超大单流入
    features['cs_fund_yellow'] = phantom['yellow']   # 大单流入
    features['cs_fund_green'] = phantom['green']     # 资金流出 (负值)
    features['cs_fund_net'] = phantom['red'] + phantom['yellow'] + phantom['green']
    features['cs_fund_flow'] = phantom['lightblue']  # 资金流量线
    features['cs_fund_flow_slope'] = pd.Series(phantom['lightblue']).diff(5).values / 5
    
    # 资金力度归一化
    fund_abs = np.abs(features['cs_fund_net'].values)
    fund_max = np.percentile(fund_abs[fund_abs > 0], 95) if np.any(fund_abs > 0) else 1
    features['cs_fund_net_norm'] = features['cs_fund_net'] / (fund_max + 1e-8)
    
    # PINK 主力资金线
    features['cs_pink_value'] = phantom['pink']
    features['cs_pink_norm'] = phantom['pink'] / 100.0  # 0-1
    features['cs_pink_below_10'] = (phantom['pink'] < 10).astype(float)
    features['cs_pink_above_90'] = (phantom['pink'] > 90).astype(float)
    features['cs_pink_slope_3d'] = pd.Series(phantom['pink']).diff(3).values / 3
    features['cs_pink_buy'] = phantom['buy_signal'].astype(float)   # PINK 上穿 10
    features['cs_pink_sell'] = phantom['sell_signal'].astype(float)  # PINK 下穿 90
    
    # === 3. 黑马指标系统 ===
    heima = calculate_heima_full(h, l, c, o)
    
    # KDJ
    features['cs_kdj_k'] = heima['K'] / 100.0  # 归一化 0-1
    features['cs_kdj_d'] = heima['D'] / 100.0
    features['cs_kdj_j'] = heima['J'] / 100.0
    features['cs_kdj_cross_up'] = heima['bot_golden_cross'].astype(float)
    features['cs_kdj_two_cross'] = heima['two_golden_cross'].astype(float)
    
    # CCI
    cci = heima['CCI']
    features['cs_cci_value'] = cci / 200.0  # 归一化 ~(-1, 1)
    features['cs_cci_oversold'] = (cci < -100).astype(float)
    features['cs_cci_overbought'] = (cci > 100).astype(float)
    features['cs_cci_extreme_oversold'] = (cci < -200).astype(float)
    
    # 信号
    features['cs_heima'] = heima['heima'].astype(float)
    features['cs_juedi'] = heima['juedi'].astype(float)
    features['cs_golden_bottom'] = heima['golden_bottom'].astype(float)
    features['cs_top_divergence'] = heima['top_divergence'].astype(float)
    
    # 主力进场/洗盘
    features['cs_main_force'] = heima['main_force_enter'].astype(float)
    features['cs_washing'] = heima['washing'].astype(float)
    features['cs_var50'] = heima['var50']
    features['cs_var50_slope'] = pd.Series(heima['var50']).diff(3).values / 3
    
    # === 4. 复合特征 ===
    
    # 信号综合强度 (越大越看多)
    signal_strength = np.zeros(n, dtype=float)
    signal_strength += features['cs_blue_pct'].values * 3       # BLUE 最重要
    signal_strength -= features['cs_lired_value'].values * 2    # LIRED 看空
    signal_strength += features['cs_fund_net_norm'].values       # 资金净流入
    signal_strength += features['cs_main_force'].values * 2     # 主力进场
    signal_strength -= features['cs_washing'].values             # 洗盘扣分
    signal_strength += features['cs_heima'].values * 3           # 黑马加分
    signal_strength += features['cs_juedi'].values * 3           # 掘地加分
    signal_strength += features['cs_golden_bottom'].values * 4   # 黄金底最强
    signal_strength -= features['cs_top_divergence'].values * 3  # 顶背离减分
    signal_strength += features['cs_pink_buy'].values * 2        # PINK 买点
    signal_strength -= features['cs_pink_sell'].values * 2       # PINK 卖点
    features['cs_signal_strength'] = signal_strength
    
    # 多空力量比
    bull = (features['cs_fund_red'].values + features['cs_fund_yellow'].values)
    bear = np.abs(features['cs_fund_green'].values)
    features['cs_bull_bear_ratio'] = bull / (bear + 1e-8) - 1  # >0 多头, <0 空头
    features['cs_bull_bear_ratio'] = features['cs_bull_bear_ratio'].clip(-5, 5)
    
    # 蔡森状态判断 (0=底部, 1=上升, 2=顶部, 3=下降)
    regime = np.zeros(n, dtype=float)
    for i in range(n):
        p = features['cs_pink_norm'].values[i]
        b = features['cs_blue_pct'].values[i]
        lr = features['cs_lired_value'].values[i]
        mf = features['cs_main_force'].values[i]
        
        if b > 0.1 or p < 0.15:          # BLUE 有信号 或 PINK 低位
            regime[i] = 0  # 底部
        elif mf > 0 and p < 0.7:
            regime[i] = 1  # 上升
        elif p > 0.85 or lr > 0:
            regime[i] = 2  # 顶部
        else:
            regime[i] = 3  # 中性/下降
    
    features['cs_regime'] = regime
    features['cs_regime_bottom'] = (regime == 0).astype(float)
    features['cs_regime_rising'] = (regime == 1).astype(float)
    features['cs_regime_top'] = (regime == 2).astype(float)
    
    # NaN 处理
    features = features.fillna(0)
    
    return features


def _consecutive_count(condition):
    """计算连续满足条件的天数"""
    n = len(condition)
    result = np.zeros(n, dtype=float)
    count = 0
    for i in range(n):
        if condition[i]:
            count += 1
        else:
            count = 0
        result[i] = count
    return result


def _signal_disappear(condition):
    """信号从有到无 (消失点 = 买卖点)"""
    n = len(condition)
    result = np.zeros(n, dtype=float)
    for i in range(1, n):
        if condition[i - 1] and not condition[i]:
            result[i] = 1.0
    return result


# 所有蔡森特征列名
CAISEN_FEATURE_NAMES = [
    # BLUE
    'cs_blue_value', 'cs_blue_slope_3d', 'cs_blue_active', 
    'cs_blue_disappear', 'cs_blue_pct',
    # LIRED
    'cs_lired_value', 'cs_lired_disappear',
    # 资金力度
    'cs_fund_red', 'cs_fund_yellow', 'cs_fund_green',
    'cs_fund_net', 'cs_fund_flow', 'cs_fund_flow_slope', 'cs_fund_net_norm',
    # PINK
    'cs_pink_value', 'cs_pink_norm', 'cs_pink_below_10',
    'cs_pink_above_90', 'cs_pink_slope_3d', 'cs_pink_buy', 'cs_pink_sell',
    # KDJ
    'cs_kdj_k', 'cs_kdj_d', 'cs_kdj_j',
    'cs_kdj_cross_up', 'cs_kdj_two_cross',
    # CCI
    'cs_cci_value', 'cs_cci_oversold', 'cs_cci_overbought', 'cs_cci_extreme_oversold',
    # 信号
    'cs_heima', 'cs_juedi', 'cs_golden_bottom', 'cs_top_divergence',
    # 主力
    'cs_main_force', 'cs_washing', 'cs_var50', 'cs_var50_slope',
    # 复合
    'cs_signal_strength', 'cs_bull_bear_ratio',
    'cs_regime', 'cs_regime_bottom', 'cs_regime_rising', 'cs_regime_top',
]
