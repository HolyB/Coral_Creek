#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib Alpha158 风格因子 (纯 pandas/numpy 实现)
=============================================

参考 Microsoft Qlib Alpha158 因子集，无需安装 Qlib 库。
计算 ~60 个精选因子 (从 158 个中选出 IC 高/稳定性好的)。

分类:
1. 价格动量 (Price Momentum)
2. 量价异动 (Volume-Price Anomaly)
3. 波动率形态 (Volatility Shape)
4. 技术微结构 (Microstructure)
5. 时序统计 (Time-Series Stats)

所有因子名以 'qf_' 前缀，避免和 feature_calculator 的特征冲突。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


QLIB_FEATURE_NAMES: List[str] = []  # 在模块加载时填充


def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    """滚动排名百分位 (0~1)"""
    return s.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def _ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """滚动相关系数"""
    return x.rolling(window).corr(y)


def _ts_delta(s: pd.Series, d: int) -> pd.Series:
    """差分"""
    return s - s.shift(d)


def _ts_decay_linear(s: pd.Series, window: int) -> pd.Series:
    """线性衰减加权移动平均"""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return s.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


def calculate_qlib_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 Qlib 风格的 Alpha 因子。

    Args:
        df: OHLCV DataFrame，需要列: Open, High, Low, Close, Volume
            (index 为 datetime 或有 Date 列)

    Returns:
        原 DataFrame 加上 ~60 个 qf_ 前缀的因子列
    """
    df = df.copy()
    
    # 统一列名
    col_map = {}
    for target, candidates in {
        'Close': ['Close', 'close', 'CLOSE'],
        'Open': ['Open', 'open', 'OPEN'],
        'High': ['High', 'high', 'HIGH'],
        'Low': ['Low', 'low', 'LOW'],
        'Volume': ['Volume', 'volume', 'VOLUME', 'vol'],
    }.items():
        for c in candidates:
            if c in df.columns:
                col_map[c] = target
                break
    if col_map:
        df = df.rename(columns=col_map)

    C = df['Close']
    O = df['Open']
    H = df['High']
    L = df['Low']
    V = df['Volume'].astype(float)

    # 避免除 0
    eps = 1e-10

    # ============================
    # 1. 价格动量 (Price Momentum)
    # ============================
    # 多周期 ROC
    for d in [1, 2, 3, 5, 10, 20, 30, 60]:
        df[f'qf_roc_{d}'] = C / C.shift(d) - 1

    # 多周期 BIAS (乖离率)
    for w in [5, 10, 20, 60]:
        ma = C.rolling(w).mean()
        df[f'qf_bias_{w}'] = (C - ma) / (ma + eps)

    # 动量加速度 (ROC 的 ROC)
    roc5 = C / C.shift(5) - 1
    df['qf_roc_accel'] = roc5 - roc5.shift(5)

    # 价格通道位置: (Close - N日Low) / (N日High - N日Low)
    for w in [5, 10, 20, 60]:
        hh = H.rolling(w).max()
        ll = L.rolling(w).min()
        df[f'qf_channel_pos_{w}'] = (C - ll) / (hh - ll + eps)

    # ============================
    # 2. 量价异动 (Volume-Price)
    # ============================
    # 量比
    for w in [5, 10, 20]:
        df[f'qf_vratio_{w}'] = V / (V.rolling(w).mean() + eps)

    # 量价相关 (rolling corr)
    for w in [5, 10, 20]:
        df[f'qf_vp_corr_{w}'] = _ts_corr(C.pct_change(), V.pct_change(), w)

    # VWAP 偏离
    vwap = (C * V).rolling(20).sum() / (V.rolling(20).sum() + eps)
    df['qf_vwap_bias'] = (C - vwap) / (vwap + eps)

    # 量能趋势 (Volume momentum)
    for d in [5, 10, 20]:
        df[f'qf_vmom_{d}'] = V / (V.shift(d) + eps) - 1

    # ============================
    # 3. 波动率形态 (Volatility)
    # ============================
    log_ret = np.log(C / C.shift(1))

    # 已实现波动率 (不同窗口)
    for w in [5, 10, 20, 60]:
        df[f'qf_rvol_{w}'] = log_ret.rolling(w).std() * np.sqrt(252)

    # 波动率比 (短期/长期)
    df['qf_vol_ratio_5_20'] = (
        log_ret.rolling(5).std() / (log_ret.rolling(20).std() + eps)
    )
    df['qf_vol_ratio_10_60'] = (
        log_ret.rolling(10).std() / (log_ret.rolling(60).std() + eps)
    )

    # 日内振幅 / ATR
    tr = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['qf_intraday_range'] = (H - L) / (C + eps)
    df['qf_atr_ratio'] = atr14 / (C + eps)

    # 高低点偏度
    df['qf_hl_skew'] = ((H - C) - (C - L)) / (H - L + eps)

    # ============================
    # 4. 技术微结构
    # ============================
    # 加权收盘价
    wclose = (2 * C + H + L) / 4
    df['qf_wclose_bias'] = (C - wclose) / (wclose + eps)

    # 缺口 (Gap)
    df['qf_gap'] = O / C.shift(1) - 1

    # 上影线 / 下影线比
    body = (C - O).abs()
    upper_shadow = H - pd.concat([C, O], axis=1).max(axis=1)
    lower_shadow = pd.concat([C, O], axis=1).min(axis=1) - L
    df['qf_upper_shadow_ratio'] = upper_shadow / (body + eps)
    df['qf_lower_shadow_ratio'] = lower_shadow / (body + eps)

    # 实体占比
    df['qf_body_ratio'] = body / (H - L + eps)

    # ============================
    # 5. 时序统计 (Qlib 特色)
    # ============================
    # 滚动偏度/峰度
    for w in [20, 60]:
        df[f'qf_skew_{w}'] = log_ret.rolling(w).skew()
        df[f'qf_kurt_{w}'] = log_ret.rolling(w).kurt()

    # 滚动排名百分位
    df['qf_rank_close_20'] = _ts_rank(C, 20)
    df['qf_rank_vol_20'] = _ts_rank(V, 20)

    # 线性衰减加权均价 vs 简单均价
    df['qf_decay_bias_10'] = (
        _ts_decay_linear(C, 10) / (C.rolling(10).mean() + eps) - 1
    )
    df['qf_decay_bias_20'] = (
        _ts_decay_linear(C, 20) / (C.rolling(20).mean() + eps) - 1
    )

    # 最大回撤 (滚动)
    for w in [10, 20, 60]:
        cummax = C.rolling(w).max()
        df[f'qf_max_drawdown_{w}'] = (C - cummax) / (cummax + eps)

    # 涨跌天数比
    up = (C > C.shift(1)).astype(float)
    for w in [5, 10, 20]:
        df[f'qf_up_ratio_{w}'] = up.rolling(w).mean()

    # 收益率自相关 (Mean Reversion / Momentum signal)
    for lag in [1, 5]:
        df[f'qf_ret_autocorr_{lag}'] = log_ret.rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag), raw=False
        )

    # 更新全局特征名列表
    global QLIB_FEATURE_NAMES
    QLIB_FEATURE_NAMES = sorted([c for c in df.columns if c.startswith('qf_')])

    return df


def get_qlib_feature_names() -> List[str]:
    """返回所有 qf_ 因子名列表"""
    if not QLIB_FEATURE_NAMES:
        # 用空 DataFrame 触发一次计算以填充名称
        dummy = pd.DataFrame({
            'Open': [1]*100, 'High': [1]*100, 'Low': [1]*100,
            'Close': [1]*100, 'Volume': [1]*100,
        })
        calculate_qlib_alpha_features(dummy)
    return QLIB_FEATURE_NAMES


def get_qlib_latest_features(df: pd.DataFrame) -> Dict[str, float]:
    """计算 Qlib 因子并返回最后一行字典"""
    result = calculate_qlib_alpha_features(df)
    if result.empty:
        return {}
    qf_cols = [c for c in result.columns if c.startswith('qf_')]
    latest = result[qf_cols].iloc[-1]
    return {k: float(v) if pd.notna(v) else 0.0 for k, v in latest.items()}
