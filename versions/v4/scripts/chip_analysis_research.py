#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筹码分布分析研究 - 寻找最优的底部顶格峰检测参数

分析真实股票数据，找出合理的过滤规则
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data_fetcher import get_us_stock_data


def calculate_chip_distribution(df, decay_factor=0.97):
    """计算筹码分布"""
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_range = price_max - price_min
    
    bins = 70
    bin_size = price_range / bins if price_range > 0 else 1
    volume_profile = np.zeros(bins)
    bin_centers = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
    
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
    if total_vol > 0:
        volume_profile = volume_profile / total_vol
    
    return volume_profile, bin_centers, price_min, price_max


def analyze_chip_metrics(symbol, days=100):
    """分析单只股票的筹码分布指标"""
    df = get_us_stock_data(symbol, days=days)
    if df is None or len(df) < 30:
        return None
    
    try:
        profile, centers, price_min, price_max = calculate_chip_distribution(df)
        total_vol = np.sum(profile)
        current_close = df['Close'].iloc[-1]
        
        # 1. POC (最大筹码峰)
        poc_idx = np.argmax(profile)
        poc_price = centers[poc_idx]
        poc_pct = profile[poc_idx] * 100  # 最大单峰占比
        
        # 2. 底部区域定义 (价格区间的底部 30%)
        bottom_30_price = price_min + (price_max - price_min) * 0.30
        bottom_chip_pct = sum(profile[centers <= bottom_30_price]) * 100
        
        # 3. POC 位置 (0-100%, 0=最底, 100=最顶)
        poc_position = (poc_price - price_min) / (price_max - price_min) * 100 if price_max > price_min else 50
        
        # 4. 获利盘
        profit_pct = sum(profile[centers < current_close]) * 100
        
        # 5. 当前价格位置 (0-100%)
        price_position = (current_close - price_min) / (price_max - price_min) * 100 if price_max > price_min else 50
        
        # 6. 筹码集中度 (POC ±10% 区间)
        near_poc = sum(profile[(centers >= poc_price * 0.9) & (centers <= poc_price * 1.1)]) * 100
        
        # 7. 价格距 POC 距离
        dist_to_poc = (current_close - poc_price) / poc_price * 100 if poc_price > 0 else 0
        
        return {
            'symbol': symbol,
            'current_price': current_close,
            'poc_price': poc_price,
            'poc_single_bar_pct': poc_pct,  # 单峰最大占比
            'poc_position_pct': poc_position,  # POC 在价格区间的位置
            'bottom_30_chip_pct': bottom_chip_pct,  # 底部 30% 区域的筹码占比
            'profit_pct': profit_pct,  # 获利盘占比
            'price_position_pct': price_position,  # 当前价格在区间的位置
            'concentration_pct': near_poc,  # 筹码集中度
            'dist_to_poc_pct': dist_to_poc,  # 价格距 POC 距离
            'price_min': price_min,
            'price_max': price_max,
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None


def analyze_multiple_stocks(symbols, days=100):
    """分析多只股票"""
    results = []
    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Analyzing {sym}...")
        result = analyze_chip_metrics(sym, days)
        if result:
            results.append(result)
    return pd.DataFrame(results)


if __name__ == "__main__":
    # 分析一批代表性股票
    test_symbols = [
        # 大盘科技股
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA',
        # 中盘成长股
        'CRM', 'ADBE', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'ROKU',
        # 小盘/波动大的股票
        'PLTR', 'COIN', 'MARA', 'RIOT', 'SOFI', 'HOOD',
        # 传统行业
        'JPM', 'BAC', 'XOM', 'CVX', 'WMT', 'KO',
        # 随机选一些
        'AMD', 'INTC', 'MU', 'QCOM', 'AVGO', 'AMAT'
    ]
    
    print("=" * 60)
    print("筹码分布参数分析 - 确定合理的过滤阈值")
    print("=" * 60)
    
    df = analyze_multiple_stocks(test_symbols, days=100)
    
    if len(df) > 0:
        print("\n" + "=" * 60)
        print("📊 整体统计")
        print("=" * 60)
        
        print("\n🔹 单峰最大占比 (poc_single_bar_pct):")
        print(f"   最小: {df['poc_single_bar_pct'].min():.1f}%")
        print(f"   最大: {df['poc_single_bar_pct'].max():.1f}%")
        print(f"   中位数: {df['poc_single_bar_pct'].median():.1f}%")
        print(f"   平均: {df['poc_single_bar_pct'].mean():.1f}%")
        print(f"   >10% 的股票数: {len(df[df['poc_single_bar_pct'] > 10])}")
        print(f"   >15% 的股票数: {len(df[df['poc_single_bar_pct'] > 15])}")
        print(f"   >20% 的股票数: {len(df[df['poc_single_bar_pct'] > 20])}")
        
        print("\n🔹 底部 30% 区域筹码占比 (bottom_30_chip_pct):")
        print(f"   最小: {df['bottom_30_chip_pct'].min():.1f}%")
        print(f"   最大: {df['bottom_30_chip_pct'].max():.1f}%")
        print(f"   中位数: {df['bottom_30_chip_pct'].median():.1f}%")
        print(f"   >30% 的股票数: {len(df[df['bottom_30_chip_pct'] > 30])}")
        print(f"   >40% 的股票数: {len(df[df['bottom_30_chip_pct'] > 40])}")
        print(f"   >50% 的股票数: {len(df[df['bottom_30_chip_pct'] > 50])}")
        
        print("\n🔹 POC 位置 (poc_position_pct, 0=最底, 100=最顶):")
        print(f"   最小: {df['poc_position_pct'].min():.1f}%")
        print(f"   最大: {df['poc_position_pct'].max():.1f}%")
        print(f"   中位数: {df['poc_position_pct'].median():.1f}%")
        print(f"   <30% (底部) 的股票数: {len(df[df['poc_position_pct'] < 30])}")
        
        print("\n🔹 筹码集中度 (concentration_pct):")
        print(f"   最小: {df['concentration_pct'].min():.1f}%")
        print(f"   最大: {df['concentration_pct'].max():.1f}%")
        print(f"   中位数: {df['concentration_pct'].median():.1f}%")
        
        # 识别候选的底部顶格峰
        print("\n" + "=" * 60)
        print("🔥 底部顶格峰候选股票 (POC在底部30% + 单峰>10% + 底部堆积>30%)")
        print("=" * 60)
        
        candidates = df[
            (df['poc_position_pct'] < 30) &  # POC 在底部
            (df['poc_single_bar_pct'] > 10) &  # 有明显单峰
            (df['bottom_30_chip_pct'] > 30)  # 底部筹码密集
        ]
        
        if len(candidates) > 0:
            for _, row in candidates.iterrows():
                print(f"\n   {row['symbol']}: ${row['current_price']:.2f}")
                print(f"      POC: ${row['poc_price']:.2f} (位置: {row['poc_position_pct']:.0f}%)")
                print(f"      单峰: {row['poc_single_bar_pct']:.1f}% | 底部堆积: {row['bottom_30_chip_pct']:.1f}%")
                print(f"      获利盘: {row['profit_pct']:.1f}% | 集中度: {row['concentration_pct']:.1f}%")
        else:
            print("   没有符合条件的候选")
        
        # 保存详细数据
        df.to_csv('chip_analysis_results.csv', index=False)
        print("\n✅ 详细数据已保存到 chip_analysis_results.csv")
