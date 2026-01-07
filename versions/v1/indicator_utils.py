import pandas as pd
import numpy as np

# ==================== 基础指标函数 ====================

def REF(series, periods=1):
    return pd.Series(series).shift(periods).values

def EMA(series, periods):
    return pd.Series(series).ewm(span=periods, adjust=False).mean().values

def SMA(series, periods, weight=1):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def IF(condition, value_if_true, value_if_false):
    return np.where(condition, value_if_true, value_if_false)

def POW(series, power):
    return np.power(series, power)

def LLV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).min().values

def HHV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).max().values

def MA(series, periods):
    """移动平均"""
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def AVEDEV(series, periods):
    """平均绝对偏差"""
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values

# ==================== 复合信号计算 ====================

def calculate_blue_signal_series(open_p, high, low, close):
    """
    计算 BLUE 信号序列
    返回: BLUE值序列 (numpy array)
    """
    VAR1 = REF((low + open_p + close + high) / 4, 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom1 = SMA(np.maximum(low - VAR1, 0), 10, 1)
        VAR2 = np.where(denom1 != 0, SMA(np.abs(low - VAR1), 13, 1) / denom1, 0)
        
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(low, 33)
    VAR5 = EMA(IF(low <= VAR4, VAR3, 0), 3)
    VAR6 = POW(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    # 为了计算RADIO1，需要计算LIRED相关变量（但不使用LIRED信号）
    with np.errstate(divide='ignore', invalid='ignore'):
        denom2 = SMA(np.minimum(high - VAR1, 0), 10, 1)
        # 注意: denom2 可能为负数或0，这里只要非0即可除
        VAR21 = np.where(denom2 != 0, SMA(np.abs(high - VAR1), 13, 1) / denom2, 0)
        
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(high, 33)
    VAR51 = EMA(IF(high >= VAR41, -VAR31, 0), 3)
    VAR61 = POW(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    max_value = np.maximum(VAR6, np.abs(VAR61))
    
    # 避免除以零或无效值，使用 numpy 向量化操作
    # 如果 max_value > 0, RADIO1 = 200 / max_value, 否则为 1
    with np.errstate(divide='ignore', invalid='ignore'):
        RADIO1 = np.where(max_value > 0, 200 / max_value, 1)
        # 处理可能出现的 inf
        RADIO1 = np.where(np.isinf(RADIO1), 1, RADIO1)
        
    BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
    
    # 清理 NaN
    BLUE = np.nan_to_num(BLUE, nan=0)
    
    return BLUE

def calculate_heima_signal_series(high, low, close, open_p):
    """
    计算黑马信号序列
    返回: heima_signal (bool array), juedi_signal (bool array)
    """
    # VAR1 = (HIGH + LOW + CLOSE) / 3
    VAR1 = (high + low + close) / 3
    
    # VAR2 = (VAR1 - MA(VAR1, 14)) / (0.015 * AVEDEV(VAR1, 14))
    ma_var1 = MA(VAR1, 14)
    avedev_var1 = AVEDEV(VAR1, 14)
    avedev_var1 = np.where(avedev_var1 == 0, 0.0001, avedev_var1)
    VAR2 = (VAR1 - ma_var1) / (0.015 * avedev_var1)
    
    # VAR3: 检测局部低点且有一定振幅
    low_series = pd.Series(low)
    is_local_low = (low_series == low_series.rolling(window=16, min_periods=1, center=True).min())
    has_amplitude = (high - low) > 0.04
    VAR3 = np.where(is_local_low & has_amplitude, 80, 0)
    
    # VAR4: 检测价格从下降转为上升的拐点
    close_series = pd.Series(close)
    rolling_min = close_series.rolling(window=22, min_periods=1).min()
    rolling_max = close_series.rolling(window=22, min_periods=1).max()
    
    pct_change = close_series.pct_change()
    is_rising = pct_change > 0
    was_falling_1 = pct_change.shift(1) <= 0
    was_falling_2 = pct_change.shift(2) <= 0
    was_falling_3 = pct_change.shift(3) <= 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = rolling_max - rolling_min
        denom = np.where(denom == 0, 0.0001, denom)
        near_low = (close_series - rolling_min) / denom < 0.2
        
    VAR4 = np.where(is_rising & was_falling_1 & was_falling_2 & was_falling_3 & near_low, 50, 0)
    
    # 黑马信号: VAR2 < -110 AND VAR4 > 0
    heima_signal = (VAR2 < -110) & (VAR4 > 0)
    
    # 掘底买点: VAR2 < -110 AND VAR3 > 0
    juedi_signal = (VAR2 < -110) & (VAR3 > 0)
    
    return heima_signal, juedi_signal

def calculate_kdj_series(high, low, close, N=9, M1=3, M2=3):
    """
    计算 KDJ 指标
    返回: K, D, J (numpy arrays)
    """
    # 转换输入为 Series 以利用 pandas 函数，如果已经是 Series 则不变
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    lowest_low = low_s.rolling(window=N, min_periods=1).min()
    highest_high = high_s.rolling(window=N, min_periods=1).max()
    
    # 避免除以零
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan) # 将0替换为NaN以避免除零错误
    
    rsv = (close_s - lowest_low) / denom * 100
    rsv = rsv.fillna(50) # 如果分母为0（最高价=最低价），RSV取50
    
    # K = 2/3 * PreK + 1/3 * RSV  =>  EMA(RSV, alpha=1/3)
    K = rsv.ewm(alpha=1/M1, adjust=False).mean()
    
    # D = 2/3 * PreD + 1/3 * K    =>  EMA(K, alpha=1/3)
    D = K.ewm(alpha=1/M2, adjust=False).mean()
    
    J = 3 * K - 2 * D
    
    return K.values, D.values, J.values

def calculate_volume_profile_metrics(price_series, volume_series, current_price, lookback_days=120, bins=50):
    """
    计算特定时刻的筹码分布特征
    
    Args:
        price_series: 历史价格序列 (截至当前时刻)
        volume_series: 历史成交量序列
        current_price: 当前价格
        lookback_days: 回溯天数 (默认半年约120交易日)
        bins: 价格分箱数
        
    Returns:
        dict: 包含筹码分布特征的字典
            - profit_ratio: 获利盘比例 (0-1)
            - poc_price: 筹码峰价格 (Point of Control)
            - price_pos: 当前价格相对于POC的位置 ('Above', 'Below', 'At')
            - concentration: 筹码集中度 (VA 宽度 / POC)
    """
    if len(price_series) < 10:
        return {'profit_ratio': 0, 'poc_price': current_price, 'price_pos': 'Unknown', 'concentration': 0}
        
    # 截取最近 N 天的数据
    recent_prices = price_series[-lookback_days:]
    recent_volumes = volume_series[-lookback_days:]
    
    if len(recent_prices) == 0:
        return {'profit_ratio': 0, 'poc_price': current_price, 'price_pos': 'Unknown', 'concentration': 0}
        
    # 计算价格区间
    min_p = np.min(recent_prices)
    max_p = np.max(recent_prices)
    
    if min_p == max_p:
        return {'profit_ratio': 0.5, 'poc_price': min_p, 'price_pos': 'At', 'concentration': 1}
        
    # 创建直方图 bins
    hist, bin_edges = np.histogram(recent_prices, bins=bins, weights=recent_volumes)
    
    # 1. 获利盘比例 (Profit Ratio)
    # 计算当前价格以下的成交量总和 / 总成交量
    # 这里用简单的近似：统计低于 current_price 的 bins 的 volume
    
    total_volume = np.sum(hist)
    if total_volume == 0:
         return {'profit_ratio': 0, 'poc_price': current_price, 'price_pos': 'Unknown', 'concentration': 0}

    # 找到 current_price 所在的 bin index
    # np.digitize 返回的是索引 (1-based), 所以要减 1
    current_idx = np.digitize(current_price, bin_edges) - 1
    current_idx = max(0, min(current_idx, bins - 1))
    
    # 获利盘 = 价格低于当前价格的所有 bins 的成交量之和
    profit_volume = np.sum(hist[:current_idx])
    # 加上当前 bin 的一半 (假设均匀分布)
    profit_volume += hist[current_idx] * 0.5
    
    profit_ratio = profit_volume / total_volume
    
    # 2. POC (Point of Control) - 筹码峰
    max_vol_idx = np.argmax(hist)
    poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
    
    # 3. 相对位置
    if current_price > poc_price * 1.02:
        price_pos = 'Above' # 上方 (支撑)
    elif current_price < poc_price * 0.98:
        price_pos = 'Below' # 下方 (压力)
    else:
        price_pos = 'At'    # 附近 (震荡)
        
    # 4. 筹码集中度 (Concentration)
    # 计算 70% 筹码分布的价格范围宽度
    # 这是一个简化版本
    sorted_indices = np.argsort(hist)[::-1] # 按成交量降序排列
    cum_vol = 0
    va_indices = []
    for idx in sorted_indices:
        cum_vol += hist[idx]
        va_indices.append(idx)
        if cum_vol >= total_volume * 0.7:
            break
            
    va_min_idx = min(va_indices)
    va_max_idx = max(va_indices)
    va_low = bin_edges[va_min_idx]
    va_high = bin_edges[va_max_idx+1]
    
    concentration = (va_high - va_low) / poc_price
    
    return {
        'profit_ratio': round(profit_ratio, 4),
        'poc_price': round(poc_price, 2),
        'price_pos': price_pos,
        'concentration': round(concentration, 4),
        'va_low': round(va_low, 2),
        'va_high': round(va_high, 2)
    }

def calculate_volatility(close_series, window=252):
    """
    计算年化波动率
    Args:
        close_series: 收盘价序列 (pd.Series or np.array)
        window: 滚动窗口大小 (默认252天，即1年)
    Returns:
        float: 最近的年化波动率
    """
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series(close_series)
        
    # 计算对数收益率
    log_ret = np.log(close_series / close_series.shift(1))
    
    # 计算滚动标准差 * sqrt(252)
    vol = log_ret.rolling(window=window).std() * np.sqrt(252)
    
    # 返回最后一个有效值
    if len(vol) > 0 and pd.notna(vol.iloc[-1]):
        return vol.iloc[-1]
    else:
        # 如果数据不足一年，就用整体标准差
        return log_ret.std() * np.sqrt(252)

def calculate_volatility(close_series, days=252):
    """
    计算年化波动率
    """
    if len(close_series) < 2:
        return 0
        
    # 计算对数收益率
    log_returns = np.log(pd.Series(close_series) / pd.Series(close_series).shift(1))
    
    # 计算最近 N 天的波动率 (标准差) 并年化
    volatility = log_returns.tail(days).std() * np.sqrt(252)
    
    return volatility if not np.isnan(volatility) else 0
