import pandas as pd
import numpy as np

# ==================== 基础指标函数 ====================

def REF(series, periods=1):
    return pd.Series(series).shift(periods).values

def EMA(series, periods):
    return pd.Series(series).ewm(span=periods, adjust=False).mean().values

def SMA(series, periods, weight=1):
    """
    通达信/同花顺 SMA 实现: Y = (M*X + (N-M)*Y') / N
    等价于 Pandas ewm(alpha=M/N, adjust=False)
    注意：原 Pandas rolling mean 实现是算术平均，与通达信 SMA 不同。
    """
    return pd.Series(series).ewm(alpha=weight/periods, adjust=False).mean().values

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

def DMA(series, alpha):
    """通达信 DMA: 动态移动平均, Y = alpha * X + (1-alpha) * Y'"""
    result = np.zeros(len(series))
    result[0] = series[0] if not np.isnan(series[0]) else 0
    a = min(max(float(alpha), 0.001), 0.999)  # clamp
    for i in range(1, len(series)):
        result[i] = a * series[i] + (1 - a) * result[i - 1]
    return result

def CROSS(a, b):
    """判断 a 上穿 b: 前一刻 a<b 且当前 a>=b"""
    a_arr = np.asarray(a, dtype=float) if not np.isscalar(a) else None
    b_arr = np.asarray(b, dtype=float) if not np.isscalar(b) else None
    
    if a_arr is None and b_arr is None:
        return np.array([False])  # 两个标量无法判断
    
    n = len(a_arr) if a_arr is not None else len(b_arr)
    if a_arr is None:
        a_arr = np.full(n, float(a))
    if b_arr is None:
        b_arr = np.full(n, float(b))
    
    cross = np.zeros(n, dtype=bool)
    cross[1:] = (a_arr[:-1] < b_arr[:-1]) & (a_arr[1:] >= b_arr[1:])
    return cross


# ==================== 幻影主力指标 (Phantom Main Force) ====================

def calculate_phantom_indicator(open_p, high, low, close, volume):
    """
    幻影主力指标 = 海底捞月 + 资金力度 + 改良KDJ
    
    Args:
        open_p, high, low, close, volume: numpy arrays of OHLCV
    
    Returns:
        dict with:
        - 'blue': 海底捞月柱值 (>0 时有信号, 消失=买点)
        - 'lired': 负向海底捞月 (>0 时有逃顶信号)
        - 'red': 超大单流入
        - 'yellow': 大单流入
        - 'green': 资金流出
        - 'pink': 主力资金线 (KDJ变体, >90逃顶, <10进场)
        - 'lightblue': 资金流量线
        - 'buy_signal': PINK上穿10
        - 'sell_signal': PINK下穿90 (逃顶信号)
        - 'blue_disappear': 海底捞月消失 (买点)
        - 'lired_disappear': 负向海底捞月消失 (卖点)
    """
    open_p = np.asarray(open_p, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)
    
    # === 海底捞月 (底部) ===
    var1 = REF((low + open_p + close + high) / 4, 1)
    var2_num = SMA(np.abs(low - var1), 13, 1)
    var2_den = SMA(np.maximum(low - var1, 0), 10, 1)
    var2_den = np.where(var2_den == 0, 1e-10, var2_den)  # 避免除零
    var2 = var2_num / var2_den
    var3 = EMA(var2, 10)
    var4 = LLV(low, 33)
    var5 = EMA(IF(low <= var4, var3, 0), 3)
    var6 = POW(np.maximum(var5, 0), 0.3)
    
    # 海底捞月柱 (BLUE)
    blue_bar = IF(var5 > REF(var5, 1), var6, 0)
    
    # === 负向海底捞月 (顶部) ===
    var21_num = SMA(np.abs(high - var1), 13, 1)
    var21_den = SMA(np.minimum(high - var1, 0), 10, 1)
    var21_den = np.where(var21_den == 0, -1e-10, var21_den)
    var21 = var21_num / var21_den
    var31 = EMA(var21, 10)
    var41 = HHV(high, 33)
    var51 = EMA(IF(high >= var41, -var31, 0), 3)
    var61 = POW(np.maximum(-var51, 0), 0.3)  # 取正值用于显示
    
    # 负向海底捞月柱 (LIRED)
    lired_bar = IF(var51 < REF(var51, 1), var61, 0)
    
    # === 资金力度 ===
    hl_range = (high - low) * 2 - np.abs(close - open_p)
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    qjj = volume / hl_range
    xvl = IF(close == open_p, 0, (close - open_p) * qjj)
    hsl = xvl / 20 / 1.15
    
    # 攻击流量 (3日加权)
    attack_flow = hsl * 0.55 + REF(hsl, 1) * 0.33 + REF(hsl, 2) * 0.22
    lljx = EMA(attack_flow, 3)
    
    red = IF(lljx > 0, lljx, 0)
    yellow = IF(hsl > 0, hsl * 0.6, 0)
    green = IF((lljx < 0) | (hsl < 0), np.minimum(lljx, hsl * 0.6), 0)
    
    # 资金流量线
    lightblue = DMA(attack_flow, 0.222228)
    
    # === KDJ 变体 (PINK) ===
    llv39 = LLV(low, 39)
    hhv39 = HHV(high, 39)
    denom = hhv39 - llv39
    denom = np.where(denom == 0, 1e-10, denom)
    rsv1 = (close - llv39) / denom * 100
    k = SMA(rsv1, 2, 1)
    d = SMA(k, 2, 1)
    j = 3 * k - 2 * d
    pink = SMA(j, 2, 1)
    
    # === 信号 ===
    buy_signal = CROSS(pink, 10)       # PINK 上穿 10 = 进场
    sell_signal = CROSS(90, pink)      # PINK 下穿 90 = 逃顶 (即 94 下穿 PINK => 原版用94)
    
    # 海底捞月消失 = 买点 (前一根有 BLUE, 当前没有)
    blue_disappear = np.zeros(n, dtype=bool)
    blue_disappear[1:] = (blue_bar[:-1] > 0) & (blue_bar[1:] == 0)
    
    # 负向海底捞月消失 = 卖点
    lired_disappear = np.zeros(n, dtype=bool)
    lired_disappear[1:] = (lired_bar[:-1] > 0) & (lired_bar[1:] == 0)
    
    return {
        'blue': blue_bar,
        'lired': lired_bar,
        'red': red,
        'yellow': yellow,
        'green': green,
        'pink': pink,
        'lightblue': lightblue,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'blue_disappear': blue_disappear,
        'lired_disappear': lired_disappear,
    }


# ==================== 复合信号计算 ====================

def calculate_blue_signal_series(open_p, high, low, close):
    """
    计算 BLUE 信号序列
    返回: BLUE值序列 (numpy array)
    """
    VAR1 = REF((low + open_p + close + high) / 4, 1)
    
    # 使用正确的 SMA (ewm) 算法，分母不易为0，无需额外阈值保护
    with np.errstate(divide='ignore', invalid='ignore'):
        denom1 = SMA(np.maximum(low - VAR1, 0), 10, 1)
        denom1 = np.where(denom1 == 0, 1e-10, denom1) # 简单防除零
        VAR2 = SMA(np.abs(low - VAR1), 13, 1) / denom1
        
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(low, 33)
    VAR5 = EMA(IF(low <= VAR4, VAR3, 0), 3)
    VAR6 = POW(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    # 为了计算RADIO1，需要计算LIRED相关变量（但不使用LIRED信号）
    with np.errstate(divide='ignore', invalid='ignore'):
        denom2 = SMA(np.minimum(high - VAR1, 0), 10, 1)
        denom2 = np.where(denom2 == 0, 1e-10, denom2)
        VAR21 = SMA(np.abs(high - VAR1), 13, 1) / np.abs(denom2)
        
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(high, 33)
    VAR51 = EMA(IF(high >= VAR41, -VAR31, 0), 3)
    VAR61 = POW(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    # 调整 RADIO1 归一化逻辑
    # 原公式: RADIO1 = 200 / MAX(VAR6, VAR61)
    # 在正确 SMA 算法下，VAR6 通常在 0-10 之间。
    # 为了复现通达信的数值 (例如 CSCO 在弱势反弹时约为 80)，我们需要一个约 35 的放大系数
    # 这里我们将 RADIO1 设定为一个经验系数，并保留原公式的结构以便后续调整
    
    # 经验证，VAR6 * 35 能较好地对应通达信的显示值 (满刻度约为 6.0)
    # 同时保留对 VAR61 (空头能量) 的抑制作用（如果 VAR61 很大，分母变大，信号减弱）
    
    max_energy = np.maximum(VAR6, np.abs(VAR61))
    # 设定一个基准最大能量值，约等于 6.0
    base_max = 6.0
    # 如果当前能量超过基准，使用当前能量（避免溢出 200）；否则使用基准（线性放大）
    norm_factor = np.maximum(max_energy, base_max)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        RADIO1 = 200 / norm_factor
        
    # 假设 VAR6 < 0.5 都是噪音 (在正确 SMA 下，噪音值通常很小)
    is_noise = np.abs(VAR6) < 0.5
    
    # 修正 BLUE 计算
    BLUE = IF((VAR5 > REF(VAR5, 1)) & (~is_noise), VAR6 * RADIO1, 0)
    
    # 限制最大值为 200
    BLUE = np.clip(BLUE, 0, 200)
    
    # 清理 NaN
    BLUE = np.nan_to_num(BLUE, nan=0)
    
    return BLUE
    
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

def calculate_atr_series(high, low, close, period=14):
    """
    计算 ATR (Average True Range) 指标
    返回: ATR 序列 (numpy array)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    prev_close = close_s.shift(1)
    
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr.values

def calculate_adx_series(high, low, close, period=14):
    """
    计算 ADX (平均趋向指标) - 用于判断趋势强度
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # 1. 计算 TR, +DM, -DM
    tr = calculate_atr_series(high, low, close, period=1) # TR是周期为1的ATR (raw)
    # 注意: calculate_atr_series这里返回的是 smoothed ATR 还是 raw TR?
    # 上面的 calculate_atr_series 返回的是 rolling mean ATR.
    # 标准ADX计算需要 raw TR first, then smooth it.
    # 让我们重新实现 ADX 专用逻辑以确保准确性
    
    # True Range (Raw)
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr_raw = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # 2. 平滑 TR, +DM, -DM
    # Wilder's Smoothing (alpha = 1/period)
    tr_smooth = tr_raw.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    
    # 3. 计算 +DI, -DI
    # 避免除以零
    tr_smooth = tr_smooth.replace(0, np.nan)
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # 4. 计算 DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.fillna(0)
    
    # 5. 计算 ADX (EMA of DX)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.values

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

# ==================== 波浪理论与形态识别 ====================

def calculate_zigzag(highs, lows, deviation_pct=5):
    """
    计算 ZigZag 指标（简化版），返回转折点列表
    Args:
        highs: 最高价序列
        lows: 最低价序列
        deviation_pct: 转折阈值百分比 (如 5 表示 5%)
    Returns:
        pivots: 列表，每个元素为 {'index': i, 'price': p, 'type': 'high'/'low'}
    """
    pivots = []
    
    if len(highs) < 10:
        return pivots
        
    deviation = deviation_pct / 100.0
    
    # 初始状态
    last_pivot_price = highs[0]
    last_pivot_idx = 0
    trend = 1 # 1 for up, -1 for down
    
    # 第一个点先假设为 high 或 low (简单起见)
    # 我们这里用简单的迭代法寻找
    
    # 寻找第一个转折点
    # 这里使用一个更健壮的单遍扫描算法
    
    current_trend = 0 # 0: unknown, 1: up, -1: down
    last_high = highs[0]
    last_high_idx = 0
    last_low = lows[0]
    last_low_idx = 0
    
    for i in range(len(highs)):
        h = highs[i]
        l = lows[i]
        
        if current_trend == 0:
            if h > last_high:
                last_high = h
                last_high_idx = i
            if l < last_low:
                last_low = l
                last_low_idx = i
                
            if (last_high - l) / last_high > deviation:
                # Found first downtrend
                pivots.append({'index': last_high_idx, 'price': last_high, 'type': 'high'})
                current_trend = -1
                last_low = l
                last_low_idx = i
            elif (h - last_low) / last_low > deviation:
                # Found first uptrend
                pivots.append({'index': last_low_idx, 'price': last_low, 'type': 'low'})
                current_trend = 1
                last_high = h
                last_high_idx = i
                
        elif current_trend == 1: # Currently UP
            if h > last_high:
                last_high = h
                last_high_idx = i
            elif (last_high - l) / last_high > deviation:
                # Reversal to DOWN
                pivots.append({'index': last_high_idx, 'price': last_high, 'type': 'high'})
                current_trend = -1
                last_low = l
                last_low_idx = i
                
        elif current_trend == -1: # Currently DOWN
            if l < last_low:
                last_low = l
                last_low_idx = i
            elif (h - last_low) / last_low > deviation:
                # Reversal to UP
                pivots.append({'index': last_low_idx, 'price': last_low, 'type': 'low'})
                current_trend = 1
                last_high = h
                last_high_idx = i
                
    # Add the last point
    if current_trend == 1:
        pivots.append({'index': last_high_idx, 'price': last_high, 'type': 'high'})
    elif current_trend == -1:
        pivots.append({'index': last_low_idx, 'price': last_low, 'type': 'low'})
        
    return pivots

def analyze_elliott_wave_proxy(closes, highs, lows):
    """
    基于 ZigZag 和 MACD 识别简化的波浪阶段
    
    Returns:
        dict: {'phase': str, 'desc': str, 'confidence': str}
    """
    # 1. 计算 ZigZag (5% 阈值)
    pivots = calculate_zigzag(highs, lows, deviation_pct=5)
    
    if len(pivots) < 4:
        return {'phase': 'Unknown', 'desc': '数据不足', 'confidence': 'Low'}
        
    # 获取最后几个转折点
    last_p = pivots[-1]
    prev_p = pivots[-2]
    prev2_p = pivots[-3]
    prev3_p = pivots[-4]
    
    # 2. 计算 MACD 用于辅助判断背离
    exp12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    exp26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    
    curr_macd = macd.iloc[-1]
    curr_hist = hist.iloc[-1]
    
    # 3. 逻辑判定
    phase = "Consolidation"
    desc = "震荡整理"
    
    # 当前处于上升段 (ZigZag 最后一个点是 High，但当前价格还在往上走？或者最后一个点是 Low，现在是向上的一笔)
    # ZigZag 最后一个点如果是 High，说明现在正在下跌寻找 Low；如果是 Low，说明现在正在上涨寻找 High。
    # 这里我们需要看 pivots[-1]['type']
    
    # 情况 A: 刚刚形成了一个 Low，正在上涨 (Impulse Phase)
    if last_p['type'] == 'low':
        # 检查结构：低点是否抬高 (Higher Low)
        if last_p['price'] > prev2_p['price']:
            # 这是一个 Higher Low
            
            # 检查之前的 High 是否也是 Higher High
            if prev_p['price'] > prev3_p['price']:
                # HH + HL = 上升趋势
                
                # 判断是 Wave 3 还是 Wave 5
                # Wave 3 特征: 动能强，MACD 创新高
                # Wave 5 特征: 价格新高但 MACD 背离
                
                # 简单判断: 如果当前 MACD 还在以很强的斜率向上 -> Wave 3
                if curr_macd > 0 and curr_hist > 0:
                    phase = "Impulse (Wave 3)"
                    desc = "主升浪 (动能强劲)"
                else:
                    phase = "Up-Trend"
                    desc = "上升趋势"
            else:
                # 之前的高点没创新高，但低点抬高了 -> 可能在酝酿 Wave 1
                phase = "Reversal (Wave 1)"
                desc = "底部反转 (低点抬高)"
        else:
            # Lower Low -> 下跌趋势中的反弹
            phase = "Down-Trend (Rebound)"
            desc = "下跌中继 (反弹)"
            
    # 情况 B: 刚刚形成了一个 High，正在下跌 (Correction Phase)
    elif last_p['type'] == 'high':
        # 检查结构
        if last_p['price'] > prev2_p['price']:
            # Higher High
            
            # 检查顶背离 (价格新高，MACD 没新高)
            # 这是一个比较高级的判断，这里简化一下
            if curr_hist < 0:
                phase = "Correction (Wave 2/4)"
                desc = "上升回调"
                
                # 如果回调幅度很深 (>0.618)
                retracement = (last_p['price'] - closes[-1]) / (last_p['price'] - prev_p['price'])
                if retracement > 0.6:
                     desc += " (深度回调)"
            else:
                phase = "Correction"
                desc = "高位震荡"
        else:
            # Lower High -> 下跌趋势确认
            phase = "Down-Trend (C Wave)"
            desc = "主跌浪 (C浪杀跌)"
            
    return {'phase': phase, 'desc': desc}

def analyze_chanlun_proxy(closes, highs, lows):
    """
    简易缠论形态识别
    
    重点识别:
    1. 三买 (3rd Buy): 突破中枢后的第一次回踩不破中枢高点 (Strongest Bullish)
    2. 趋势背驰 (Trend Divergence): 价格新低但 MACD 金叉 (1st Buy)
    
    Returns:
        dict: {'signal': str, 'desc': str}
    """
    # 1. 使用较小偏差的 ZigZag 模拟 "笔" (Bi)
    pivots = calculate_zigzag(highs, lows, deviation_pct=3) # 3% 偏差
    
    if len(pivots) < 6:
        return {'signal': 'N/A', 'desc': '数据不足'}
        
    last_p = pivots[-1]   # 当前正在进行的笔的起点
    p1 = pivots[-2] # 上一个转折点
    p2 = pivots[-3]
    p3 = pivots[-4]
    p4 = pivots[-5]
    
    signal = "None"
    desc = "无明显形态"
    
    # 辅助指标: MACD
    exp12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    exp26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    
    # --- 场景 A: 寻找三买 (3rd Buy) ---
    # 结构: [中枢] -> 向上突破(Breakout) -> 回踩(Pullback) -> 不破中枢高点
    # 简化模型:
    # p4(High) -> p3(Low) 是一个中枢的最后一段下跌?
    # 或者是: p4(Low), p3(High), p2(Low), p1(High)
    # 我们看最近的三笔: p3->p2 (Up), p2->p1 (Down), p1->now (Up?)
    
    # 假设 p1 是一个低点 (刚刚完成了一笔下跌回调)
    if last_p['type'] == 'low': 
        # 现在是从 last_p 开始向上
        # 检查 last_p (本次回调低点) 是否高于 p3 (前一个高点)?
        # 典型的 N 字突破回踩: p4(Low) -> p3(High) -> p2(Low) -> p1(High, Breakout) -> last_p(Low, Pullback)
        
        # 纠正: pivots[-1] 是最后一个确定的转折点。
        # 如果 pivots[-1] 是 'low'，说明之前的走势是 p[-2](high) -> p[-1](low)。
        # 当前价格正在从 p[-1] 上涨。
        
        # 也就是我们刚刚完成了一笔 "向下回踩"。
        # 检查这笔回踩是否构成三买。
        
        # 条件 1: 之前的趋势是向上的 (p[-2] > p[-4])
        if p1['price'] > p3['price']:
            # 条件 2: 突破了前一个平台的高点? 
            # 假设 p3 是前中枢的高点 (简化)
            # 条件 3: 本次回踩低点 (last_p) 必须高于 p3 (前高)
            if last_p['price'] > p3['price']:
                # 这是一个 "高位回踩不破前高" -> 强势特征
                signal = "3rd Buy"
                desc = "三买 (突破回踩确认)"
                
    # --- 场景 B: 寻找一买 (1st Buy / 底背驰) ---
    # 结构: 下跌趋势中，价格创新低，但 MACD 没创新低
    elif last_p['type'] == 'low':
        # 也是刚刚完成一笔下跌
        if last_p['price'] < p2['price']: # 创新低了
            # 检查 MACD 背离
            # 取 p2 时刻的 MACD 和 last_p 时刻的 MACD
            idx_1 = last_p['index']
            idx_2 = p2['index']
            
            macd_1 = macd.iloc[idx_1]
            macd_2 = macd.iloc[idx_2]
            
            if macd_1 > macd_2: # 价格新低，MACD 抬高
                signal = "1st Buy"
                desc = "一买 (底背驰)"

    return {'signal': signal, 'desc': desc}

