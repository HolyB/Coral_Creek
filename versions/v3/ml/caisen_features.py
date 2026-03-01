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
    
    # === 5. 蔡森16量价图形 + 萧明道12结构 ===
    # 逐日滚动检测（不只看最后一天）
    _add_pattern_features(features, hist)
    
    # === 6. TA-Lib 61 K线形态识别 ===
    _add_talib_patterns(features, o, h, l, c)
    
    # NaN 处理
    features = features.fillna(0)
    
    return features


def _add_pattern_features(features: pd.DataFrame, hist: pd.DataFrame):
    """
    逐日计算蔡森16量价形态 + 萧明道12结构特征
    每个形态 → 一个 0/1 特征列
    """
    n = len(hist)
    close = hist['Close'].values.astype(float)
    high = hist['High'].values.astype(float)
    low = hist['Low'].values.astype(float)
    opn = hist['Open'].values.astype(float)
    vol = hist['Volume'].values.astype(float) if 'Volume' in hist.columns else np.ones(n)
    
    # 预计算滚动指标
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    vma5 = pd.Series(vol).rolling(5).mean().values
    vma20 = pd.Series(vol).rolling(20).mean().values
    
    # 初始化所有图形列
    p_codes = [f'cs_P{i:02d}' for i in range(1, 17)]
    x_codes = [f'cs_X{i:02d}' for i in range(1, 13)]
    for code in p_codes + x_codes:
        features[code] = 0.0
    
    # 汇总列
    features['cs_pattern_bull_count'] = 0.0
    features['cs_pattern_bear_count'] = 0.0
    features['cs_pattern_score'] = 0.0  # 多-空
    
    min_lookback = 25
    
    for i in range(min_lookback, n):
        vr = vol[i] / max(vma20[i], 1e-9) if not np.isnan(vma20[i]) else 1.0
        ret1 = (close[i] / close[i-1] - 1) * 100 if close[i-1] > 0 else 0
        ret5 = (close[i] / close[max(0, i-5)] - 1) * 100 if close[max(0, i-5)] > 0 else 0
        prev20_high = np.max(high[max(0, i-20):i]) if i > 1 else high[i]
        prev20_low = np.min(low[max(0, i-20):i]) if i > 1 else low[i]
        
        upper_shadow = high[i] - max(close[i], opn[i])
        body = max(abs(close[i] - opn[i]), 1e-9)
        usr = upper_shadow / body
        
        bull = 0
        bear = 0
        
        # --- 蔡森 P01-P16 ---
        # P01 放量突破
        if close[i] > prev20_high and vr >= 1.5:
            features.iloc[i, features.columns.get_loc('cs_P01')] = 1; bull += 1
        # P02 缩量回踩
        if not np.isnan(ma20[i]) and abs(close[i] - ma20[i]) / max(ma20[i], 1e-9) <= 0.02 and vr <= 0.75 and close[i] >= close[i-1]:
            features.iloc[i, features.columns.get_loc('cs_P02')] = 1; bull += 1
        # P03 底部堆量
        if i >= 6 and sum(vol[i-2:i+1]) > sum(vol[i-5:i-2]) * 1.4 and ret5 < -3:
            features.iloc[i, features.columns.get_loc('cs_P03')] = 1; bull += 1
        # P04 平台突破
        if i >= 21:
            box = (np.max(high[i-20:i]) - np.min(low[i-20:i])) / max(close[i], 1e-9)
            if box < 0.12 and close[i] > np.max(high[i-20:i]) and vr > 1.3:
                features.iloc[i, features.columns.get_loc('cs_P04')] = 1; bull += 1
        # P05 量价齐升
        if ret1 > 2.0 and vr > 1.3:
            features.iloc[i, features.columns.get_loc('cs_P05')] = 1; bull += 1
        # P06 缩量新高
        if i >= 60 and close[i] >= np.max(close[i-60:i]) and vr < 0.85:
            features.iloc[i, features.columns.get_loc('cs_P06')] = 1  # 中性
        # P07 放量滞涨
        if abs(ret1) < 1.0 and vr > 1.8:
            features.iloc[i, features.columns.get_loc('cs_P07')] = 1; bear += 1
        # P08 巨量阴线
        if ret1 < -3.0 and vr > 2.0:
            features.iloc[i, features.columns.get_loc('cs_P08')] = 1; bear += 1
        # P09 价涨量缩背离
        if ret5 > 5.0 and not np.isnan(vma5[i]) and not np.isnan(vma20[i]) and vma5[i] < vma20[i] * 0.9:
            features.iloc[i, features.columns.get_loc('cs_P09')] = 1; bear += 1
        # P10 放量长上影
        if usr > 1.5 and vr > 1.5:
            features.iloc[i, features.columns.get_loc('cs_P10')] = 1; bear += 1
        # P11 跌破均线放量
        if not np.isnan(ma20[i]) and close[i] < ma20[i] <= close[i-1] and vr > 1.3:
            features.iloc[i, features.columns.get_loc('cs_P11')] = 1; bear += 1
        # P12 缩量止跌
        if ret5 < -5.0 and ret1 > 0 and vr < 0.85:
            features.iloc[i, features.columns.get_loc('cs_P12')] = 1  # 中性
        # P13 周线突破 (用日线近 27 天代替)
        if i >= 27 and close[i] > np.max(high[i-27:i]) and vr > 1.2:
            features.iloc[i, features.columns.get_loc('cs_P13')] = 1; bull += 1
        # P14 月线转强 (日线 120 天)
        if i >= 120:
            ma120 = np.mean(close[i-120:i])
            if close[i] > ma120 and close[i-1] <= ma120:
                features.iloc[i, features.columns.get_loc('cs_P14')] = 1; bull += 1
        # P15 多周期共振
        if not np.isnan(ma20[i]) and not np.isnan(ma60[i]) and close[i] > ma20[i] > ma60[i] and ret5 > 0:
            features.iloc[i, features.columns.get_loc('cs_P15')] = 1; bull += 1
        # P16 下跌量能衰竭
        if ret5 < -8.0 and vr < 0.75 and close[i] > prev20_low:
            features.iloc[i, features.columns.get_loc('cs_P16')] = 1  # 中性
        
        # --- 萧明道 X01-X12 ---
        # X01 上升结构完整
        if not np.isnan(ma20[i]) and not np.isnan(ma60[i]) and close[i] > ma20[i] > ma60[i]:
            features.iloc[i, features.columns.get_loc('cs_X01')] = 1; bull += 1
        # X02 缩量回踩不破
        if not np.isnan(ma20[i]) and abs(close[i] - ma20[i]) / max(ma20[i], 1e-9) <= 0.02 and vr <= 0.8 and close[i] >= close[i-1]:
            features.iloc[i, features.columns.get_loc('cs_X02')] = 1; bull += 1
        # X03 平台突破确认
        if i >= 21:
            box = (np.max(high[i-20:i]) - np.min(low[i-20:i])) / max(close[i], 1e-9)
            if box < 0.12 and close[i] > np.max(high[i-20:i]) and vr > 1.3:
                features.iloc[i, features.columns.get_loc('cs_X03')] = 1; bull += 1
        # X04 黄金坑反包
        if i >= 5 and ret1 > 2.0 and close[i] > opn[i] and close[i] > close[i-1]:
            low3 = np.min(low[i-3:i])
            if low3 < close[i-4]:
                features.iloc[i, features.columns.get_loc('cs_X04')] = 1; bull += 1
        # X05 多头排列共振
        if not np.isnan(ma5[i]) and not np.isnan(ma20[i]) and not np.isnan(ma60[i]) and ma5[i] > ma20[i] > ma60[i]:
            features.iloc[i, features.columns.get_loc('cs_X05')] = 1; bull += 1
        # X06 高位量价背离
        if i >= 60 and close[i] >= np.max(close[i-60:i]) and not np.isnan(vma5[i]) and not np.isnan(vma20[i]) and vma5[i] < vma20[i] * 0.9:
            features.iloc[i, features.columns.get_loc('cs_X06')] = 1; bear += 1
        # X07 巨量滞涨
        if abs(ret1) < 1.0 and vr > 1.8:
            features.iloc[i, features.columns.get_loc('cs_X07')] = 1; bear += 1
        # X08 关键支撑失守
        if not np.isnan(ma20[i]) and close[i] < ma20[i] <= close[i-1] and vr > 1.2:
            features.iloc[i, features.columns.get_loc('cs_X08')] = 1; bear += 1
        # X09 反弹无量
        if ret5 < -5.0 and ret1 > 0 and vr < 0.9:
            features.iloc[i, features.columns.get_loc('cs_X09')] = 1; bear += 1
        # X10 结构中性整理
        if i >= 21:
            box = (np.max(high[i-20:i]) - np.min(low[i-20:i])) / max(close[i], 1e-9)
            if box <= 0.15 and abs(ret5) < 4.0:
                features.iloc[i, features.columns.get_loc('cs_X10')] = 1  # 中性
        # X11 下跌结构衰竭
        if ret5 < -8.0 and vr < 0.8 and close[i] > prev20_low:
            features.iloc[i, features.columns.get_loc('cs_X11')] = 1  # 中性
        # X12 趋势反转确认
        if not np.isnan(ma20[i]) and close[i-1] <= ma20[i] < close[i] and close[i] > prev20_high:
            features.iloc[i, features.columns.get_loc('cs_X12')] = 1; bull += 1
        
        features.iloc[i, features.columns.get_loc('cs_pattern_bull_count')] = bull
        features.iloc[i, features.columns.get_loc('cs_pattern_bear_count')] = bear
        features.iloc[i, features.columns.get_loc('cs_pattern_score')] = bull - bear


def _add_talib_patterns(features: pd.DataFrame, o, h, l, c):
    """
    用 TA-Lib 识别 61 种经典 K 线形态
    
    每个形态输出:
      +100 = 看涨, -100 = 看跌, 0 = 无信号
    我们归一化为 -1/0/+1
    """
    try:
        import talib
    except ImportError:
        return
    
    cdl_funcs = [f for f in dir(talib) if f.startswith('CDL')]
    
    bull_sum = np.zeros(len(c), dtype=float)
    bear_sum = np.zeros(len(c), dtype=float)
    cdl_data = {}
    
    for func_name in cdl_funcs:
        try:
            func = getattr(talib, func_name)
            result = func(o, h, l, c)
            # 归一化到 -1/0/+1
            normalized = np.sign(result).astype(float)
            col_name = f'cdl_{func_name[3:].lower()}'
            cdl_data[col_name] = normalized
            
            bull_sum += (normalized > 0).astype(float)
            bear_sum += (normalized < 0).astype(float)
        except Exception:
            pass
    
    # 一次性添加所有列 (避免 fragmentation 警告)
    cdl_data['cdl_bull_count'] = bull_sum
    cdl_data['cdl_bear_count'] = bear_sum
    cdl_data['cdl_net_score'] = bull_sum - bear_sum
    
    cdl_df = pd.DataFrame(cdl_data, index=features.index)
    for col in cdl_df.columns:
        features[col] = cdl_df[col].values


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


# TA-Lib K线形态名称
TALIB_CDL_NAMES = [
    'cdl_2crows', 'cdl_3blackcrows', 'cdl_3inside', 'cdl_3linestrike',
    'cdl_3outside', 'cdl_3starsinsouth', 'cdl_3whitesoldiers',
    'cdl_abandonedbaby', 'cdl_advanceblock', 'cdl_belthold',
    'cdl_breakaway', 'cdl_closingmarubozu', 'cdl_concealbabyswall',
    'cdl_counterattack', 'cdl_darkcloudcover', 'cdl_doji',
    'cdl_dojistar', 'cdl_dragonflydoji', 'cdl_engulfing',
    'cdl_eveningdojistar', 'cdl_eveningstar', 'cdl_gapsidesidewhite',
    'cdl_gravestonedoji', 'cdl_hammer', 'cdl_hangingman',
    'cdl_harami', 'cdl_haramicross', 'cdl_highwave',
    'cdl_hikkake', 'cdl_hikkakemod', 'cdl_homingpigeon',
    'cdl_identical3crows', 'cdl_inneck', 'cdl_invertedhammer',
    'cdl_kicking', 'cdl_kickingbylength', 'cdl_ladderbottom',
    'cdl_longleggeddoji', 'cdl_longline', 'cdl_marubozu',
    'cdl_matchinglow', 'cdl_mathold', 'cdl_morningdojistar',
    'cdl_morningstar', 'cdl_onneck', 'cdl_piercing',
    'cdl_rickshawman', 'cdl_risefall3methods', 'cdl_separatinglines',
    'cdl_shootingstar', 'cdl_shortline', 'cdl_spinningtop',
    'cdl_stalledpattern', 'cdl_sticksandwich', 'cdl_takuri',
    'cdl_tasukigap', 'cdl_thrusting', 'cdl_tristar',
    'cdl_unique3river', 'cdl_upsidegap2crows', 'cdl_xsidegap3methods',
    # 复合
    'cdl_bull_count', 'cdl_bear_count', 'cdl_net_score',
]

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
    # 蔡森16量价图形
    'cs_P01', 'cs_P02', 'cs_P03', 'cs_P04', 'cs_P05', 'cs_P06',
    'cs_P07', 'cs_P08', 'cs_P09', 'cs_P10', 'cs_P11', 'cs_P12',
    'cs_P13', 'cs_P14', 'cs_P15', 'cs_P16',
    # 萧明道12结构
    'cs_X01', 'cs_X02', 'cs_X03', 'cs_X04', 'cs_X05', 'cs_X06',
    'cs_X07', 'cs_X08', 'cs_X09', 'cs_X10', 'cs_X11', 'cs_X12',
    # 图形汇总
    'cs_pattern_bull_count', 'cs_pattern_bear_count', 'cs_pattern_score',
] + TALIB_CDL_NAMES

