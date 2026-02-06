#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整信号系统回测 - 原有系统 vs 原有系统+安全区域
=====================================================

原有系统买入条件 (策略C):
- 日BLUE >= 100 + (日黑马/掘地 OR 周黑马/掘地)
- OR (日黑马/掘地 OR 周黑马/掘地) + (近5天日BLUE >= 100 OR 周BLUE >= 100)

原有系统卖出条件:
1. KDJ J > 90 (超买)
2. 跌破5日均线
3. 止损 (可选)

增强系统:
- 买入时: 原有条件 + 安全区域 < 80 (不在风险区才买)
- 卖出时: 原有条件 + 安全区域 > 90 直接卖
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_stock_data
from indicator_utils import (
    calculate_blue_signal_series, 
    calculate_heima_signal_series, 
    calculate_kdj_series
)
from strategies.safety_zone_indicator import SafetyZoneIndicator


def analyze_chip_distribution(df, lookback_days=20, decay_factor=0.97):
    """
    分析筹码分布，检测顶部堆积和底部减少
    
    Returns:
        dict: {
            'is_top_heavy': bool,  # 顶部筹码堆积
            'is_bottom_light': bool,  # 底部筹码减少
            'should_sell': bool,  # 是否应该卖出
            'top_chip_ratio': float,  # 顶部筹码占比
            'bottom_chip_ratio': float,  # 底部筹码占比
            'cost_change_pct': float,  # 成本变化百分比
            'description': str  # 描述
        }
    """
    if len(df) < lookback_days + 30:
        return None
    
    # 计算筹码分布
    def calc_chip_distribution(data, price_min, price_max, bins=70):
        bin_size = (price_max - price_min) / bins if price_max > price_min else 1
        volume_profile = np.zeros(bins)
        price_bins = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        total_days = len(data)
        for i, (idx, row) in enumerate(data.iterrows()):
            day_high = row['High']
            day_low = row['Low']
            day_close = row['Close']
            day_vol = row['Volume']
            
            days_ago = total_days - 1 - i
            time_weight = decay_factor ** days_ago
            weighted_vol = day_vol * time_weight
            
            if day_high == day_low or bin_size == 0:
                bin_idx = int((day_close - price_min) / bin_size)
                bin_idx = min(max(bin_idx, 0), bins - 1)
                volume_profile[bin_idx] += weighted_vol
            else:
                start_bin = int((day_low - price_min) / bin_size)
                end_bin = int((day_high - price_min) / bin_size)
                start_bin = max(start_bin, 0)
                end_bin = min(end_bin, bins - 1)
                close_bin = int((day_close - price_min) / bin_size)
                close_bin = min(max(close_bin, start_bin), end_bin)
                
                if start_bin == end_bin:
                    volume_profile[start_bin] += weighted_vol
                else:
                    for b in range(start_bin, end_bin + 1):
                        dist_to_close = abs(b - close_bin)
                        max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                        weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                        volume_profile[b] += weighted_vol * weight
        
        return volume_profile, bin_centers
    
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bins = 70
    
    # 计算过去和当前筹码分布
    df_past = df.iloc[:-lookback_days]
    df_current = df
    
    past_profile, bin_centers = calc_chip_distribution(df_past, price_min, price_max, bins)
    current_profile, _ = calc_chip_distribution(df_current, price_min, price_max, bins)
    
    # 归一化
    past_total = np.sum(past_profile)
    current_total = np.sum(current_profile)
    if past_total > 0:
        past_profile = past_profile / past_total
    if current_total > 0:
        current_profile = current_profile / current_total
    
    current_close = df['Close'].iloc[-1]
    
    # 计算顶部筹码 (当前价格上方 20%)
    top_threshold = current_close * 1.05
    top_bins = bin_centers > top_threshold
    current_top_ratio = np.sum(current_profile[top_bins]) * 100
    past_top_ratio = np.sum(past_profile[top_bins]) * 100
    top_increase = current_top_ratio - past_top_ratio
    
    # 计算底部筹码 (当前价格下方 20%)
    bottom_threshold = current_close * 0.85
    bottom_bins = bin_centers < bottom_threshold
    current_bottom_ratio = np.sum(current_profile[bottom_bins]) * 100
    past_bottom_ratio = np.sum(past_profile[bottom_bins]) * 100
    bottom_decrease = past_bottom_ratio - current_bottom_ratio
    
    # 计算成本变化
    past_avg_cost = np.sum(bin_centers * past_profile) if past_total > 0 else current_close
    current_avg_cost = np.sum(bin_centers * current_profile) if current_total > 0 else current_close
    cost_change_pct = (current_avg_cost - past_avg_cost) / past_avg_cost * 100 if past_avg_cost > 0 else 0
    
    # 判断卖出信号
    # 顶部堆积: 顶部筹码增加 > 3%
    is_top_heavy = top_increase > 3
    # 底部减少: 底部筹码减少 > 3%  
    is_bottom_light = bottom_decrease > 3
    # 成本上移: 平均成本上涨 > 2%
    is_cost_rising = cost_change_pct > 2
    
    # 综合判断
    sell_score = 0
    if is_top_heavy:
        sell_score += 1
    if is_bottom_light:
        sell_score += 1
    if is_cost_rising:
        sell_score += 1
    
    should_sell = sell_score >= 2
    
    # 生成描述
    parts = []
    if is_top_heavy:
        parts.append(f"顶部筹码+{top_increase:.1f}%")
    if is_bottom_light:
        parts.append(f"底部筹码-{bottom_decrease:.1f}%")
    if is_cost_rising:
        parts.append(f"成本+{cost_change_pct:.1f}%")
    description = ", ".join(parts) if parts else "筹码正常"
    
    return {
        'is_top_heavy': is_top_heavy,
        'is_bottom_light': is_bottom_light,
        'is_cost_rising': is_cost_rising,
        'should_sell': should_sell,
        'sell_score': sell_score,
        'top_chip_ratio': current_top_ratio,
        'bottom_chip_ratio': current_bottom_ratio,
        'top_increase': top_increase,
        'bottom_decrease': bottom_decrease,
        'cost_change_pct': cost_change_pct,
        'description': description
    }


def analyze_bottom_peak(df, decay_factor=0.97):
    """
    检测底部筹码顶格峰 (买入信号)
    
    顶格峰条件:
    1. POC 位置 < 30% (筹码峰在底部)
    2. 底部筹码占比 > 50%
    3. 单峰最大占比 > 5%
    
    Returns:
        dict: {
            'is_strong_bottom_peak': bool,  # 强势底部顶格峰
            'is_bottom_peak': bool,  # 普通底部密集
            'poc_position': float,  # POC位置 0-100%
            'bottom_chip_ratio': float,  # 底部筹码占比
            'max_chip_pct': float,  # 单峰最大占比
            'description': str
        }
    """
    if len(df) < 50:
        return None
    
    # 计算筹码分布
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bins = 70
    bin_size = (price_max - price_min) / bins if price_max > price_min else 1
    
    volume_profile = np.zeros(bins)
    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    total_days = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        day_high = row['High']
        day_low = row['Low']
        day_close = row['Close']
        day_vol = row['Volume']
        
        days_ago = total_days - 1 - i
        time_weight = decay_factor ** days_ago
        weighted_vol = day_vol * time_weight
        
        if day_high == day_low or bin_size == 0:
            bin_idx = int((day_close - price_min) / bin_size)
            bin_idx = min(max(bin_idx, 0), bins - 1)
            volume_profile[bin_idx] += weighted_vol
        else:
            start_bin = int((day_low - price_min) / bin_size)
            end_bin = int((day_high - price_min) / bin_size)
            start_bin = max(start_bin, 0)
            end_bin = min(end_bin, bins - 1)
            close_bin = int((day_close - price_min) / bin_size)
            close_bin = min(max(close_bin, start_bin), end_bin)
            
            if start_bin == end_bin:
                volume_profile[start_bin] += weighted_vol
            else:
                for b in range(start_bin, end_bin + 1):
                    dist_to_close = abs(b - close_bin)
                    max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                    weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                    volume_profile[b] += weighted_vol * weight
    
    total_vol = np.sum(volume_profile)
    if total_vol == 0:
        return None
    
    # POC 位置
    poc_idx = np.argmax(volume_profile)
    poc_price = bin_centers[poc_idx]
    poc_position = (poc_price - price_min) / (price_max - price_min) * 100
    
    # 单峰最大占比
    max_chip_pct = np.max(volume_profile) / total_vol * 100
    
    # 底部筹码占比 (底部30%价格区间)
    bottom_30_price = price_min + (price_max - price_min) * 0.30
    bottom_chip_ratio = 0
    for i, p in enumerate(bin_centers):
        if p <= bottom_30_price:
            bottom_chip_ratio += volume_profile[i]
    bottom_chip_ratio = bottom_chip_ratio / total_vol * 100
    
    # 判定规则
    # 强信号: POC < 30% + 底部筹码 > 50% + 单峰 > 5%
    is_strong_bottom_peak = (poc_position < 30) and (bottom_chip_ratio > 50) and (max_chip_pct > 5)
    
    # 普通信号: POC < 35% + 底部筹码 > 35%
    is_bottom_peak = (poc_position < 35) and (bottom_chip_ratio > 35)
    
    # 描述
    if is_strong_bottom_peak:
        description = f"🔥底部顶格峰(POC:{poc_position:.0f}%底部:{bottom_chip_ratio:.0f}%)"
    elif is_bottom_peak:
        description = f"📍底部密集(POC:{poc_position:.0f}%底部:{bottom_chip_ratio:.0f}%)"
    else:
        description = f"普通(POC:{poc_position:.0f}%)"
    
    return {
        'is_strong_bottom_peak': is_strong_bottom_peak,
        'is_bottom_peak': is_bottom_peak,
        'poc_position': poc_position,
        'bottom_chip_ratio': bottom_chip_ratio,
        'max_chip_pct': max_chip_pct,
        'description': description
    }


class FullSystemBacktester:
    """完整信号系统回测器"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.indicator = SafetyZoneIndicator()
        self.blue_threshold = 100
    
    def prepare_data(self, df_daily: pd.DataFrame):
        """准备所有指标数据"""
        # 日线指标
        blue = calculate_blue_signal_series(
            df_daily['Open'].values, df_daily['High'].values,
            df_daily['Low'].values, df_daily['Close'].values
        )
        heima, juedi = calculate_heima_signal_series(
            df_daily['High'].values, df_daily['Low'].values,
            df_daily['Close'].values, df_daily['Open'].values
        )
        _, _, j = calculate_kdj_series(
            df_daily['High'].values, df_daily['Low'].values, 
            df_daily['Close'].values
        )
        
        # 周线数据
        df_weekly = df_daily.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(df_weekly) >= 5:
            week_blue = calculate_blue_signal_series(
                df_weekly['Open'].values, df_weekly['High'].values,
                df_weekly['Low'].values, df_weekly['Close'].values
            )
            week_heima, week_juedi = calculate_heima_signal_series(
                df_weekly['High'].values, df_weekly['Low'].values,
                df_weekly['Close'].values, df_weekly['Open'].values
            )
            # 映射到日线
            df_weekly['Week_BLUE'] = week_blue
            df_weekly['Week_Heima'] = week_heima
            df_weekly['Week_Juedi'] = week_juedi
            
            week_blue_ref = df_weekly['Week_BLUE'].shift(1).reindex(
                df_daily.index, method='ffill'
            ).fillna(0).values
            week_heima_ref = df_weekly['Week_Heima'].shift(1).reindex(
                df_daily.index, method='ffill'
            ).fillna(False).values
            week_juedi_ref = df_weekly['Week_Juedi'].shift(1).reindex(
                df_daily.index, method='ffill'
            ).fillna(False).values
        else:
            week_blue_ref = np.zeros(len(df_daily))
            week_heima_ref = np.zeros(len(df_daily), dtype=bool)
            week_juedi_ref = np.zeros(len(df_daily), dtype=bool)
        
        # 5日均线
        ma5 = pd.Series(df_daily['Close'].values).rolling(5).mean().values
        
        return {
            'blue': blue,
            'heima': heima,
            'juedi': juedi,
            'kdj_j': j,
            'week_blue': week_blue_ref,
            'week_heima': week_heima_ref,
            'week_juedi': week_juedi_ref,
            'ma5': ma5,
            'close': df_daily['Close'].values,
            'low': df_daily['Low'].values,
        }
    
    def check_buy_signal_original(self, data: Dict, i: int) -> Tuple[bool, str]:
        """检查原有系统买入信号"""
        d_blue = data['blue'][i]
        d_heima = data['heima'][i]
        d_juedi = data['juedi'][i]
        w_blue = data['week_blue'][i]
        w_heima = data['week_heima'][i]
        w_juedi = data['week_juedi'][i]
        
        # 近5天日BLUE是否 > threshold
        start_idx = max(0, i - 4)
        recent_blues = data['blue'][start_idx:i+1]
        has_recent_blue = np.any(recent_blues >= self.blue_threshold)
        
        # 黑马上下文
        has_heima_context = d_heima or d_juedi or w_heima or w_juedi
        
        # 策略C逻辑
        # 条件1: 日BLUE触发 + 黑马上下文
        if d_blue >= self.blue_threshold and has_heima_context:
            return True, f"BLUE{d_blue:.0f}+{'黑马' if d_heima else '掘地' if d_juedi else '周黑马'}"
        
        # 条件2: 黑马触发 + BLUE上下文
        if (d_heima or d_juedi or w_heima or w_juedi):
            if has_recent_blue or w_blue >= self.blue_threshold:
                trigger = "日黑马" if d_heima else "日掘地" if d_juedi else "周黑马" if w_heima else "周掘地"
                blue_src = f"日BLUE{d_blue:.0f}" if has_recent_blue else f"周BLUE{w_blue:.0f}"
                return True, f"{trigger}+{blue_src}"
        
        return False, ""
    
    def check_buy_signal_enhanced(self, data: Dict, i: int, df_daily: pd.DataFrame) -> Tuple[bool, str]:
        """
        增强版买入信号检测
        
        买入条件:
        1. (日BLUE >= 150 OR 周BLUE >= 150) AND (黑马 OR 掘地)
        2. OR 底部筹码顶格峰 + BLUE/黑马确认
        """
        d_blue = data['blue'][i]
        d_heima = data['heima'][i]
        d_juedi = data['juedi'][i]
        w_blue = data['week_blue'][i]
        w_heima = data['week_heima'][i]
        w_juedi = data['week_juedi'][i]
        
        # 条件1: (日BLUE >= 150 OR 周BLUE >= 150) AND (黑马 OR 掘地)
        has_strong_blue = d_blue >= 150 or w_blue >= 150
        has_heima_signal = d_heima or d_juedi or w_heima or w_juedi
        
        if has_strong_blue and has_heima_signal:
            blue_src = f"日BLUE{d_blue:.0f}" if d_blue >= 150 else f"周BLUE{w_blue:.0f}"
            heima_type = "日黑马" if d_heima else "日掘地" if d_juedi else "周黑马" if w_heima else "周掘地"
            return True, f"{blue_src}+{heima_type}"
        
        # 条件2: 底部筹码顶格峰
        if i >= 50:
            sub_df = df_daily.iloc[:i+1]
            peak_result = analyze_bottom_peak(sub_df)
            if peak_result and peak_result['is_strong_bottom_peak']:
                # 同时需要有BLUE或黑马信号作为确认
                if d_blue >= 100 or d_heima or d_juedi:
                    confirm = f"BLUE{d_blue:.0f}" if d_blue >= 100 else "黑马" if d_heima else "掘地"
                    return True, f"{peak_result['description']}+{confirm}"
        
        return False, ""
    
    def check_sell_signal_original(self, data: Dict, i: int) -> Tuple[bool, str]:
        """检查原有系统卖出信号 (改进版)"""
        kdj_j = data['kdj_j'][i]
        close = data['close'][i]
        ma5 = data['ma5'][i]
        
        # 条件1: KDJ J > 90 (超买)
        if kdj_j > 90:
            return True, f"KDJ J={kdj_j:.0f}>90"
        
        # 条件2: 跌破5日均线 (需要连续2天)
        # 检查前一天也是否跌破
        if i >= 1:
            prev_close = data['close'][i-1]
            prev_ma5 = data['ma5'][i-1]
            if not np.isnan(ma5) and not np.isnan(prev_ma5):
                if close < ma5 and prev_close < prev_ma5:
                    return True, f"连续跌破MA5"
        
        return False, ""
    
    def check_sell_signal_with_chips(self, data: Dict, i: int, df_daily: pd.DataFrame) -> Tuple[bool, str]:
        """检查增强版卖出信号 (包含筹码分布)"""
        kdj_j = data['kdj_j'][i]
        close = data['close'][i]
        ma5 = data['ma5'][i]
        
        # 条件1: KDJ J > 90 (超买)
        if kdj_j > 90:
            return True, f"KDJ J={kdj_j:.0f}>90"
        
        # 条件2: 筹码分布显示顶部堆积
        if i >= 50:
            sub_df = df_daily.iloc[:i+1]
            chip_result = analyze_chip_distribution(sub_df)
            if chip_result and chip_result['should_sell']:
                return True, f"筹码出货({chip_result['description']})"
        
        # 条件3: 跌破5日均线 + 筹码信号
        if not np.isnan(ma5) and close < ma5:
            if i >= 50:
                sub_df = df_daily.iloc[:i+1]
                chip_result = analyze_chip_distribution(sub_df)
                if chip_result and chip_result['sell_score'] >= 1:
                    return True, f"跌破MA5+{chip_result['description']}"
        
        return False, ""
    
    def backtest_original(self, df_daily: pd.DataFrame) -> Dict:
        """回测原有系统"""
        data = self.prepare_data(df_daily)
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(20, len(df_daily) - 1):  # 留一天用于次日开盘买入
            close = data['close'][i]
            next_open = df_daily['Open'].iloc[i + 1]
            
            # 卖出检查
            if position == 1:
                should_sell, reason = self.check_sell_signal_original(data, i)
                if should_sell:
                    # 收盘卖出
                    revenue = shares * close * (1 - self.commission)
                    pnl = revenue - trades[-1]['cost']
                    cash += revenue
                    trades.append({
                        'type': 'SELL', 'price': close, 'shares': shares,
                        'pnl': pnl, 'reason': reason
                    })
                    shares = 0
                    position = 0
            
            # 买入检查
            elif position == 0:
                should_buy, reason = self.check_buy_signal_original(data, i)
                if should_buy and cash > 0:
                    # 次日开盘买入
                    shares = int(cash * (1 - self.commission) / next_open)
                    if shares > 0:
                        cost = shares * next_open * (1 + self.commission)
                        cash -= cost
                        position = 1
                        trades.append({
                            'type': 'BUY', 'price': next_open, 'shares': shares,
                            'cost': cost, 'reason': reason
                        })
            
            # 记录权益
            equity = cash + shares * close
            equity_curve.append(equity)
        
        # 最后一天
        equity_curve.append(cash + shares * data['close'][-1])
        
        return self._calc_metrics(equity_curve, trades, len(df_daily))
    
    def backtest_with_zone(self, df_daily: pd.DataFrame) -> Dict:
        """回测原有系统 + 安全区域"""
        data = self.prepare_data(df_daily)
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(50, len(df_daily) - 1):
            close = data['close'][i]
            next_open = df_daily['Open'].iloc[i + 1]
            
            # 计算安全区域
            sub_df = df_daily.iloc[:i+1]
            zone_result = self.indicator.calculate(sub_df)
            zone_level = zone_result.get('safety_level', 50)
            
            # 卖出检查 (增强版)
            if position == 1:
                # 增强条件: 安全区域 > 90 直接卖
                if zone_level > 90:
                    revenue = shares * close * (1 - self.commission)
                    pnl = revenue - trades[-1]['cost']
                    cash += revenue
                    trades.append({
                        'type': 'SELL', 'price': close, 'shares': shares,
                        'pnl': pnl, 'reason': f"高风险区{zone_level:.0f}"
                    })
                    shares = 0
                    position = 0
                else:
                    should_sell, reason = self.check_sell_signal_original(data, i)
                    if should_sell:
                        revenue = shares * close * (1 - self.commission)
                        pnl = revenue - trades[-1]['cost']
                        cash += revenue
                        trades.append({
                            'type': 'SELL', 'price': close, 'shares': shares,
                            'pnl': pnl, 'reason': reason
                        })
                        shares = 0
                        position = 0
            
            # 买入检查 (增强版)
            elif position == 0:
                should_buy, reason = self.check_buy_signal_original(data, i)
                # 增强条件: 安全区域 < 80 才买入
                if should_buy and zone_level < 80 and cash > 0:
                    shares = int(cash * (1 - self.commission) / next_open)
                    if shares > 0:
                        cost = shares * next_open * (1 + self.commission)
                        cash -= cost
                        position = 1
                        trades.append({
                            'type': 'BUY', 'price': next_open, 'shares': shares,
                            'cost': cost, 'reason': f"{reason}|区域{zone_level:.0f}"
                        })
            
            equity = cash + shares * close
            equity_curve.append(equity)
        
        equity_curve.append(cash + shares * data['close'][-1])
        
        return self._calc_metrics(equity_curve, trades, len(df_daily))
    
    def backtest_buy_hold(self, df_daily: pd.DataFrame) -> Dict:
        """Buy & Hold 基准"""
        start = float(df_daily['Close'].iloc[0])
        end = float(df_daily['Close'].iloc[-1])
        total_return = (end / start - 1) * 100
        days = len(df_daily)
        annual_return = ((end / start) ** (252 / days) - 1) * 100
        
        equity = df_daily['Close'] / start * self.initial_capital
        peak = equity.cummax()
        drawdown = (peak - equity) / peak * 100
        
        returns = df_daily['Close'].pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': drawdown.max(),
            'win_rate': 100.0 if total_return > 0 else 0.0,
            'sharpe': sharpe,
            'trades': 1,
            'buy_trades': 1,
            'sell_trades': 0,
        }
    
    def _calc_metrics(self, equity_curve: List, trades: List, days: int) -> Dict:
        equity_curve = np.array(equity_curve)
        final_equity = equity_curve[-1]
        
        total_return = (final_equity / self.initial_capital - 1) * 100
        annual_return = ((final_equity / self.initial_capital) ** (252 / days) - 1) * 100
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        winning = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        win_rate = (winning / len(sell_trades) * 100) if sell_trades else 0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'trades': len(trades),
            'buy_trades': len([t for t in trades if t['type'] == 'BUY']),
            'sell_trades': len(sell_trades),
        }
    
    def backtest_with_chips(self, df_daily: pd.DataFrame) -> Dict:
        """回测原有系统 + 筹码分布卖出"""
        data = self.prepare_data(df_daily)
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(50, len(df_daily) - 1):
            close = data['close'][i]
            next_open = df_daily['Open'].iloc[i + 1]
            
            # 卖出检查 (使用筹码分布)
            if position == 1:
                should_sell, reason = self.check_sell_signal_with_chips(data, i, df_daily)
                if should_sell:
                    revenue = shares * close * (1 - self.commission)
                    pnl = revenue - trades[-1]['cost']
                    cash += revenue
                    trades.append({
                        'type': 'SELL', 'price': close, 'shares': shares,
                        'pnl': pnl, 'reason': reason
                    })
                    shares = 0
                    position = 0
            
            # 买入检查 (原有逻辑)
            elif position == 0:
                should_buy, reason = self.check_buy_signal_original(data, i)
                if should_buy and cash > 0:
                    shares = int(cash * (1 - self.commission) / next_open)
                    if shares > 0:
                        cost = shares * next_open * (1 + self.commission)
                        cash -= cost
                        position = 1
                        trades.append({
                            'type': 'BUY', 'price': next_open, 'shares': shares,
                            'cost': cost, 'reason': reason
                        })
            
            equity = cash + shares * close
            equity_curve.append(equity)
        
        equity_curve.append(cash + shares * data['close'][-1])
        
        return self._calc_metrics(equity_curve, trades, len(df_daily))
    
    def backtest_enhanced(self, df_daily: pd.DataFrame) -> Dict:
        """
        回测增强版策略
        
        买入: 日BLUE>=150 + 周BLUE>=150 + 黑马/掘地
              OR 底部筹码顶格峰 + BLUE/黑马确认
        卖出: KDJ J>90 OR 筹码顶部堆积 OR 跌破MA5+筹码异常
        """
        data = self.prepare_data(df_daily)
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(50, len(df_daily) - 1):
            close = data['close'][i]
            next_open = df_daily['Open'].iloc[i + 1]
            
            # 卖出检查 (使用筹码分布)
            if position == 1:
                should_sell, reason = self.check_sell_signal_with_chips(data, i, df_daily)
                if should_sell:
                    revenue = shares * close * (1 - self.commission)
                    pnl = revenue - trades[-1]['cost']
                    cash += revenue
                    trades.append({
                        'type': 'SELL', 'price': close, 'shares': shares,
                        'pnl': pnl, 'reason': reason
                    })
                    shares = 0
                    position = 0
            
            # 买入检查 (增强版)
            elif position == 0:
                should_buy, reason = self.check_buy_signal_enhanced(data, i, df_daily)
                if should_buy and cash > 0:
                    shares = int(cash * (1 - self.commission) / next_open)
                    if shares > 0:
                        cost = shares * next_open * (1 + self.commission)
                        cash -= cost
                        position = 1
                        trades.append({
                            'type': 'BUY', 'price': next_open, 'shares': shares,
                            'cost': cost, 'reason': reason
                        })
            
            equity = cash + shares * close
            equity_curve.append(equity)
        
        equity_curve.append(cash + shares * data['close'][-1])
        
        return self._calc_metrics(equity_curve, trades, len(df_daily))


def run_backtest(symbols: List[str], market: str = 'US', days: int = 730):
    """运行回测"""
    all_results = {
        'Buy & Hold': [],
        '原有系统': [],
        '原有+筹码卖': [],
        '增强系统': [],
    }
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"回测: {symbol}")
        print(f"{'='*60}")
        
        df = get_stock_data(symbol, market, days=days)
        if df is None or len(df) < 100:
            print(f"  跳过: 数据不足")
            continue
        
        bt = FullSystemBacktester()
        
        results = {}
        print("  运行 Buy & Hold...")
        results['Buy & Hold'] = bt.backtest_buy_hold(df)
        
        print("  运行 原有系统 (BLUE>=100+黑马共振, J>90/连跌MA5卖)...")
        results['原有系统'] = bt.backtest_original(df)
        
        print("  运行 原有+筹码卖 (原有买入, 筹码分布卖出)...")
        results['原有+筹码卖'] = bt.backtest_with_chips(df)
        
        print("  运行 增强系统 (BLUE>=150共振+顶格峰买, 筹码卖)...")
        results['增强系统'] = bt.backtest_enhanced(df)
        
        print(f"\n  {'策略':<14} {'年化%':<10} {'回撤%':<10} {'胜率%':<10} {'夏普':<8} {'买入':<6} {'卖出':<6}")
        print("  " + "-" * 68)
        for name, r in results.items():
            print(f"  {name:<14} {r['annual_return']:>8.1f}% {r['max_drawdown']:>8.1f}% "
                  f"{r['win_rate']:>8.1f}% {r['sharpe']:>7.2f} {r['buy_trades']:>5} {r['sell_trades']:>5}")
            all_results[name].append(r)
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"综合汇总 ({len(symbols)} 只股票)")
    print(f"{'='*70}")
    
    print(f"\n{'策略':<14} {'平均年化%':<12} {'平均回撤%':<12} {'平均胜率%':<12} {'平均夏普':<10}")
    print("-" * 60)
    
    summary = []
    for name, results in all_results.items():
        if results:
            avg_annual = np.mean([r['annual_return'] for r in results])
            avg_dd = np.mean([r['max_drawdown'] for r in results])
            avg_wr = np.mean([r['win_rate'] for r in results])
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            
            print(f"{name:<14} {avg_annual:>10.1f}% {avg_dd:>10.1f}% "
                  f"{avg_wr:>10.1f}% {avg_sharpe:>9.2f}")
            
            summary.append({
                'strategy': name,
                'avg_annual': avg_annual,
                'avg_drawdown': avg_dd,
                'avg_win_rate': avg_wr,
                'avg_sharpe': avg_sharpe,
            })
    
    # 对比分析
    print("\n" + "="*70)
    print("对比分析")
    print("="*70)
    
    if len(summary) >= 2:
        original = next((s for s in summary if s['strategy'] == '原有系统'), None)
        chips_sell = next((s for s in summary if s['strategy'] == '原有+筹码卖'), None)
        enhanced = next((s for s in summary if s['strategy'] == '增强系统'), None)
        buyhold = next((s for s in summary if s['strategy'] == 'Buy & Hold'), None)
        
        if original and chips_sell:
            print(f"\n原有系统 vs Buy&Hold:")
            print(f"  年化收益: {'+' if original['avg_annual'] > buyhold['avg_annual'] else ''}"
                  f"{original['avg_annual'] - buyhold['avg_annual']:.1f}%")
            print(f"  回撤改善: {'+' if buyhold['avg_drawdown'] > original['avg_drawdown'] else ''}"
                  f"{buyhold['avg_drawdown'] - original['avg_drawdown']:.1f}%")
            
            print(f"\n原有+筹码卖 vs 原有系统:")
            return_diff = chips_sell['avg_annual'] - original['avg_annual']
            wr_diff = chips_sell['avg_win_rate'] - original['avg_win_rate']
            sharpe_diff = chips_sell['avg_sharpe'] - original['avg_sharpe']
            print(f"  年化收益: {'+' if return_diff > 0 else ''}{return_diff:.1f}%")
            print(f"  胜率提升: {'+' if wr_diff > 0 else ''}{wr_diff:.1f}%")
            print(f"  夏普提升: {'+' if sharpe_diff > 0 else ''}{sharpe_diff:.2f}")
        
        if enhanced:
            print(f"\n增强系统 vs 原有+筹码卖:")
            return_diff = enhanced['avg_annual'] - chips_sell['avg_annual']
            wr_diff = enhanced['avg_win_rate'] - chips_sell['avg_win_rate']
            sharpe_diff = enhanced['avg_sharpe'] - chips_sell['avg_sharpe']
            print(f"  年化收益: {'+' if return_diff > 0 else ''}{return_diff:.1f}%")
            print(f"  胜率提升: {'+' if wr_diff > 0 else ''}{wr_diff:.1f}%")
            print(f"  夏普提升: {'+' if sharpe_diff > 0 else ''}{sharpe_diff:.2f}")
            
            if enhanced['avg_sharpe'] > chips_sell['avg_sharpe'] and enhanced['avg_sharpe'] > original['avg_sharpe']:
                print(f"\n✅ 结论: 增强系统表现最佳!")
            elif chips_sell['avg_sharpe'] > original['avg_sharpe']:
                print(f"\n✅ 结论: 筹码卖出信号有效提升表现!")
            else:
                print(f"\n⚠️ 结论: 需要更多数据验证")
    
    return summary


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         完整信号系统回测                                    ║
║                                                                           ║
║  原有系统:                                                                ║
║    买入: 日BLUE>=100 + 黑马/掘地共振                                       ║
║    卖出: KDJ J>90 OR 连续2天跌破MA5                                        ║
║                                                                           ║
║  原有+筹码卖出:                                                           ║
║    买入: 同原有                                                           ║
║    卖出: J>90 OR 筹码顶部堆积+底部减少 OR 跌破MA5+筹码异常                   ║
║                                                                           ║
║  增强系统:                                                                ║
║    买入: 日BLUE>=150 + 周BLUE>=150 + 黑马/掘地                             ║
║          OR 底部筹码顶格峰 + BLUE/黑马确认                                  ║
║    卖出: 同上筹码卖出                                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
    run_backtest(symbols, 'US', days=730)

