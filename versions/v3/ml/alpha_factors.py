#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha158 因子库 — 纯 pandas 实现 (不依赖 Qlib)
================================================

实现 Qlib Alpha158 中最有价值的量化因子，补充 AdvancedFeatureEngineer 中未覆盖的部分。

因子分类:
  1. BIAS 均线偏离 (5/10/20/60)
  2. RSI 相对强弱 (6/12/24)
  3. MACD 及衍生
  4. KDJ 随机指标
  5. 布林带衍生
  6. VWAP 偏离
  7. 收益率统计 (skew/kurtosis)
  8. 波动率偏斜 (上涨日 vs 下跌日)
  9. 跳空/缺口
  10. 换手率变化
  11. William %R
  12. CCI (商品通道指标)
  13. MFI (资金流量)
  14. 高低价比
"""
import numpy as np
import pandas as pd
from typing import List


class Alpha158Factors:
    """
    Alpha158 量化因子计算器
    
    用法:
        alpha = Alpha158Factors()
        df_with_factors = alpha.compute(ohlcv_df)
    """
    
    def __init__(self):
        self.factor_names: List[str] = []
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有 Alpha158 因子
        
        Args:
            df: 含 Open, High, Low, Close, Volume 列的 DataFrame
                 (index 为 DatetimeIndex 或含 Date 列)
        Returns:
            原 df 加上所有因子列
        """
        df = df.copy()
        
        # 确保有需要的列
        required = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        C = df['Close'].astype(float)
        O = df['Open'].astype(float)
        H = df['High'].astype(float)
        L = df['Low'].astype(float)
        V = df['Volume'].astype(float)
        
        # ===== 1. BIAS (均线偏离) =====
        for n in [5, 10, 20, 60]:
            ma = C.rolling(n, min_periods=1).mean()
            df[f'bias_{n}'] = (C - ma) / (ma + 1e-8) * 100
        
        # ===== 2. RSI (相对强弱) =====
        for n in [6, 12, 24]:
            delta = C.diff()
            gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{n}'] = 100 - 100 / (1 + rs)
        
        # ===== 3. MACD 及衍生 =====
        ema12 = C.ewm(span=12, adjust=False).mean()
        ema26 = C.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)
        df['macd_dif'] = dif
        df['macd_dea'] = dea
        df['macd_hist'] = macd_hist
        df['macd_cross'] = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).astype(int) - \
                           ((dif < dea) & (dif.shift(1) >= dea.shift(1))).astype(int)
        # MACD 归一化
        df['macd_dif_pct'] = dif / (C + 1e-8) * 100
        
        # ===== 4. KDJ =====
        for n in [9, 14]:
            low_n = L.rolling(n, min_periods=1).min()
            high_n = H.rolling(n, min_periods=1).max()
            rsv = (C - low_n) / (high_n - low_n + 1e-8) * 100
            k = rsv.ewm(com=2, adjust=False).mean()
            d = k.ewm(com=2, adjust=False).mean()
            j = 3 * k - 2 * d
            df[f'kdj_k_{n}'] = k
            df[f'kdj_d_{n}'] = d
            df[f'kdj_j_{n}'] = j
        
        # ===== 5. 布林带衍生 =====
        for n in [20]:
            ma = C.rolling(n, min_periods=1).mean()
            std = C.rolling(n, min_periods=1).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            df[f'bband_width_{n}'] = (upper - lower) / (ma + 1e-8) * 100
            df[f'bband_pctb_{n}'] = (C - lower) / (upper - lower + 1e-8)
            # 距上轨/下轨距离
            df[f'bband_upper_dist_{n}'] = (upper - C) / (C + 1e-8) * 100
            df[f'bband_lower_dist_{n}'] = (C - lower) / (C + 1e-8) * 100
        
        # ===== 6. VWAP 偏离 =====
        typical = (H + L + C) / 3
        vwap = (typical * V).rolling(20, min_periods=1).sum() / (V.rolling(20, min_periods=1).sum() + 1e-8)
        df['vwap_bias'] = (C - vwap) / (vwap + 1e-8) * 100
        
        # ===== 7. 收益率统计 =====
        ret = C.pct_change()
        for n in [20, 60]:
            df[f'return_skew_{n}'] = ret.rolling(n, min_periods=10).skew()
            df[f'return_kurt_{n}'] = ret.rolling(n, min_periods=10).kurt()
            df[f'return_mean_{n}'] = ret.rolling(n, min_periods=5).mean() * 100
            df[f'return_std_{n}'] = ret.rolling(n, min_periods=5).std() * 100
        
        # ===== 8. 波动率偏斜 =====
        up_ret = ret.clip(lower=0)
        down_ret = (-ret.clip(upper=0))
        for n in [20]:
            up_vol = up_ret.rolling(n, min_periods=5).std()
            down_vol = down_ret.rolling(n, min_periods=5).std()
            df[f'vol_skew_{n}'] = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
            # 上涨日占比
            df[f'up_ratio_{n}'] = (ret > 0).rolling(n, min_periods=5).mean() * 100
        
        # ===== 9. 跳空/缺口 =====
        df['gap'] = (O - C.shift(1)) / (C.shift(1) + 1e-8) * 100
        df['gap_abs'] = df['gap'].abs()
        # 累积跳空
        df['gap_sum_5'] = df['gap'].rolling(5, min_periods=1).sum()
        
        # ===== 10. 换手率变化 =====
        vol_ma5 = V.rolling(5, min_periods=1).mean()
        vol_ma20 = V.rolling(20, min_periods=1).mean()
        df['turnover_chg_5'] = V / (vol_ma5 + 1) - 1
        df['turnover_chg_20'] = V / (vol_ma20 + 1) - 1
        df['vol_ma5_ma20_ratio'] = vol_ma5 / (vol_ma20 + 1)
        
        # ===== 11. William %R =====
        for n in [14, 28]:
            hh = H.rolling(n, min_periods=1).max()
            ll = L.rolling(n, min_periods=1).min()
            df[f'willr_{n}'] = (hh - C) / (hh - ll + 1e-8) * -100
        
        # ===== 12. CCI (商品通道指标) =====
        for n in [14, 20]:
            tp = (H + L + C) / 3
            tp_ma = tp.rolling(n, min_periods=1).mean()
            tp_md = tp.rolling(n, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            df[f'cci_{n}'] = (tp - tp_ma) / (0.015 * tp_md + 1e-8)
        
        # ===== 13. MFI (资金流量指标) =====
        tp = (H + L + C) / 3
        mf = tp * V
        pos_mf = (tp > tp.shift(1)).astype(float) * mf
        neg_mf = (tp < tp.shift(1)).astype(float) * mf
        for n in [14]:
            pos_sum = pos_mf.rolling(n, min_periods=1).sum()
            neg_sum = neg_mf.rolling(n, min_periods=1).sum()
            mfr = pos_sum / (neg_sum + 1e-8)
            df[f'mfi_{n}'] = 100 - 100 / (1 + mfr)
        
        # ===== 14. 高低价比 / 价格效率 =====
        for n in [5, 20]:
            df[f'high_low_ratio_{n}'] = (H.rolling(n).max() / (L.rolling(n).min() + 1e-8) - 1) * 100
            # 价格效率 = |收盘变化| / 路径长度
            close_chg = (C - C.shift(n)).abs()
            path_len = (C.diff().abs()).rolling(n, min_periods=1).sum()
            df[f'price_efficiency_{n}'] = close_chg / (path_len + 1e-8)
        
        # ===== 15. 量价相关性 =====
        for n in [10, 20]:
            df[f'corr_close_vol_{n}'] = C.rolling(n, min_periods=5).corr(V)
            df[f'corr_ret_vol_{n}'] = ret.rolling(n, min_periods=5).corr(V)
        
        # ===== 16. 强势/弱势天数比 =====
        df['strong_day_ratio_10'] = (
            ((C > O) & (V > vol_ma5)).rolling(10, min_periods=3).mean() * 100
        )
        
        # ===== 17. Chaikin Volatility =====
        hl_ema = (H - L).ewm(span=10, adjust=False).mean()
        df['chaikin_vol'] = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-8) * 100
        
        # 收集因子名称
        base_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Adj Close'}
        self.factor_names = [c for c in df.columns if c not in base_cols]
        
        return df
    
    def get_factor_names(self) -> List[str]:
        """返回最近一次 compute 生成的因子名称"""
        return self.factor_names


# ============================================================================
# 便捷函数
# ============================================================================

def compute_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """一键计算 Alpha158 因子"""
    alpha = Alpha158Factors()
    return alpha.compute(df)


if __name__ == '__main__':
    print("=== Alpha158 Factors 测试 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = 100 * (1 + np.random.randn(n) * 0.02).cumprod()
    df = pd.DataFrame({
        'Open': close * (1 + np.random.randn(n) * 0.005),
        'High': close * (1 + abs(np.random.randn(n) * 0.01)),
        'Low': close * (1 - abs(np.random.randn(n) * 0.01)),
        'Close': close,
        'Volume': np.random.randint(100000, 1000000, n).astype(float),
    }, index=dates)
    
    alpha = Alpha158Factors()
    result = alpha.compute(df)
    
    factors = alpha.get_factor_names()
    print(f"因子数: {len(factors)}")
    print(f"\n因子列表:")
    for i, name in enumerate(factors, 1):
        val = result[name].iloc[-1]
        nan_pct = result[name].isna().mean() * 100
        print(f"  {i:2d}. {name:30s}  last={val:10.4f}  nan%={nan_pct:.0f}%")
    
    print(f"\n非 NaN 行数: {result.dropna().shape[0]} / {len(result)}")
