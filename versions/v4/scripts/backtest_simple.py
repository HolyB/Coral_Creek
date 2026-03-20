#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化信号回测 - 日线 BLUE 主导
================================

策略:
- 原有: 日BLUE >= 50 买入, 日BLUE = 0 且连续3天 卖出
- 增强: 原有 + 安全区域过滤 (高于80不买, 低于50加仓信心)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_stock_data
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series
from strategies.safety_zone_indicator import SafetyZoneIndicator


class SimpleBacktester:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.indicator = SafetyZoneIndicator()
    
    def backtest_blue_only(self, df, buy_threshold=50, sell_threshold=20) -> Dict:
        """纯日线 BLUE 策略"""
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values
        )
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        zero_count = 0  # 连续为0的天数
        
        for i in range(1, len(df)):
            price = float(df['Close'].iloc[i])
            blue_val = blue[i]
            
            # 买入: BLUE >= threshold
            if blue_val >= buy_threshold and position == 0 and cash > 0:
                shares = int(cash * (1 - self.commission) / price)
                if shares > 0:
                    cash -= shares * price * (1 + self.commission)
                    position = 1
                    trades.append({'type': 'BUY', 'price': price, 'shares': shares, 'blue': blue_val})
                    zero_count = 0
            
            # 卖出: BLUE < threshold 且持仓
            elif position == 1:
                if blue_val < sell_threshold:
                    zero_count += 1
                else:
                    zero_count = 0
                
                # 连续3天低于阈值则卖出
                if zero_count >= 3 or blue_val == 0:
                    revenue = shares * price * (1 - self.commission)
                    pnl = revenue - trades[-1]['price'] * trades[-1]['shares']
                    cash += revenue
                    trades.append({'type': 'SELL', 'price': price, 'shares': shares, 'pnl': pnl})
                    shares = 0
                    position = 0
                    zero_count = 0
            
            equity_curve.append(cash + shares * price)
        
        return self._calc_metrics(equity_curve, trades, df)
    
    def backtest_blue_with_zone(self, df, buy_threshold=50, sell_threshold=20) -> Dict:
        """BLUE + 安全区域过滤"""
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values
        )
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        zero_count = 0
        
        for i in range(50, len(df)):
            price = float(df['Close'].iloc[i])
            blue_val = blue[i]
            
            # 计算安全区域
            sub_df = df.iloc[:i+1]
            zone_result = self.indicator.calculate(sub_df)
            zone_level = zone_result.get('safety_level', 50)
            
            # 买入: BLUE >= threshold AND 安全区域 < 80 (不在风险区)
            if blue_val >= buy_threshold and position == 0 and cash > 0:
                if zone_level < 80:  # 安全区域过滤
                    shares = int(cash * (1 - self.commission) / price)
                    if shares > 0:
                        cash -= shares * price * (1 + self.commission)
                        position = 1
                        trades.append({'type': 'BUY', 'price': price, 'shares': shares, 
                                      'blue': blue_val, 'zone': zone_level})
                        zero_count = 0
            
            # 卖出逻辑
            elif position == 1:
                # 强制卖出: 安全区域 > 90 高风险
                if zone_level > 90:
                    revenue = shares * price * (1 - self.commission)
                    pnl = revenue - trades[-1]['price'] * trades[-1]['shares']
                    cash += revenue
                    trades.append({'type': 'SELL', 'price': price, 'shares': shares, 
                                  'pnl': pnl, 'reason': 'high_risk'})
                    shares = 0
                    position = 0
                    zero_count = 0
                else:
                    # 普通卖出逻辑
                    if blue_val < sell_threshold:
                        zero_count += 1
                    else:
                        zero_count = 0
                    
                    if zero_count >= 3 or blue_val == 0:
                        revenue = shares * price * (1 - self.commission)
                        pnl = revenue - trades[-1]['price'] * trades[-1]['shares']
                        cash += revenue
                        trades.append({'type': 'SELL', 'price': price, 'shares': shares, 
                                      'pnl': pnl, 'reason': 'blue_weak'})
                        shares = 0
                        position = 0
                        zero_count = 0
            
            equity_curve.append(cash + shares * price)
        
        return self._calc_metrics(equity_curve, trades, df)
    
    def backtest_full_system(self, df) -> Dict:
        """完整系统: BLUE + 黑马 + 安全区域"""
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values
        )
        heima, juedi = calculate_heima_signal_series(
            df['High'].values, df['Low'].values,
            df['Close'].values, df['Open'].values
        )
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(50, len(df)):
            price = float(df['Close'].iloc[i])
            blue_val = blue[i]
            heima_val = heima[i]
            juedi_val = juedi[i]
            
            # 计算安全区域
            sub_df = df.iloc[:i+1]
            zone_result = self.indicator.calculate(sub_df)
            zone_level = zone_result.get('safety_level', 50)
            
            # 计算综合得分
            score = 0
            reasons = []
            
            if blue_val >= 100:
                score += 20
                reasons.append(f"BLUE{blue_val:.0f}")
            elif blue_val >= 50:
                score += 12
                reasons.append(f"BLUE{blue_val:.0f}")
            elif blue_val > 0:
                score += 5
            
            if heima_val:
                score += 10
                reasons.append("黑马🐴")
            if juedi_val:
                score += 8
                reasons.append("掘地⛏️")
            
            # 安全区域调整
            if zone_level <= 30:
                score += 8
                reasons.append(f"安全区{zone_level:.0f}")
            elif zone_level <= 50:
                score += 4
            elif zone_level >= 90:
                score -= 15
                reasons.append(f"高危{zone_level:.0f}")
            elif zone_level >= 80:
                score -= 8
                reasons.append(f"风险{zone_level:.0f}")
            
            # 买入: 综合得分 >= 20
            if score >= 20 and position == 0 and cash > 0:
                shares = int(cash * (1 - self.commission) / price)
                if shares > 0:
                    cash -= shares * price * (1 + self.commission)
                    position = 1
                    trades.append({'type': 'BUY', 'price': price, 'shares': shares, 
                                  'score': score, 'reasons': reasons})
            
            # 卖出: 得分 < 5 或 安全区域 > 90
            elif position == 1 and (score < 5 or zone_level > 90):
                revenue = shares * price * (1 - self.commission)
                pnl = revenue - trades[-1]['price'] * trades[-1]['shares']
                cash += revenue
                trades.append({'type': 'SELL', 'price': price, 'shares': shares, 
                              'pnl': pnl, 'score': score})
                shares = 0
                position = 0
            
            equity_curve.append(cash + shares * price)
        
        return self._calc_metrics(equity_curve, trades, df)
    
    def backtest_buy_hold(self, df) -> Dict:
        start = float(df['Close'].iloc[0])
        end = float(df['Close'].iloc[-1])
        total_return = (end / start - 1) * 100
        days = len(df)
        annual_return = ((end / start) ** (252 / days) - 1) * 100
        
        equity = df['Close'] / start * self.initial_capital
        peak = equity.cummax()
        drawdown = (peak - equity) / peak * 100
        
        returns = df['Close'].pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': drawdown.max(),
            'win_rate': 100.0 if total_return > 0 else 0.0,
            'sharpe': sharpe,
            'trades': 1,
        }
    
    def _calc_metrics(self, equity_curve, trades, df) -> Dict:
        equity_curve = np.array(equity_curve)
        final_equity = equity_curve[-1]
        
        total_return = (final_equity / self.initial_capital - 1) * 100
        days = len(df)
        annual_return = ((final_equity / self.initial_capital) ** (252 / days) - 1) * 100
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        winning = len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        total_sells = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = (winning / total_sells * 100) if total_sells > 0 else 0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'trades': len(trades),
        }


def run_test(symbols: List[str], market: str = 'US', days: int = 730):
    all_results = {
        'Buy & Hold': [],
        '纯BLUE': [],
        'BLUE+安全区': [],
        '完整系统': [],
    }
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"测试 {symbol}")
        
        df = get_stock_data(symbol, market, days=days)
        if df is None or len(df) < 100:
            print(f"  跳过: 数据不足")
            continue
        
        bt = SimpleBacktester()
        
        results = {}
        results['Buy & Hold'] = bt.backtest_buy_hold(df)
        results['纯BLUE'] = bt.backtest_blue_only(df)
        results['BLUE+安全区'] = bt.backtest_blue_with_zone(df)
        results['完整系统'] = bt.backtest_full_system(df)
        
        print(f"  {'策略':<12} {'年化%':<10} {'回撤%':<10} {'胜率%':<10} {'夏普':<8} {'交易':<6}")
        print("  " + "-" * 60)
        for name, r in results.items():
            print(f"  {name:<12} {r['annual_return']:>8.1f}% {r['max_drawdown']:>8.1f}% "
                  f"{r['win_rate']:>8.1f}% {r['sharpe']:>7.2f} {r['trades']:>5}")
            all_results[name].append(r)
    
    # 汇总
    print(f"\n{'='*60}")
    print("综合平均结果")
    print(f"{'='*60}")
    
    print(f"\n{'策略':<12} {'平均年化%':<12} {'平均回撤%':<12} {'平均胜率%':<12} {'平均夏普':<10}")
    print("-" * 60)
    
    for name, results in all_results.items():
        if results:
            print(f"{name:<12} {np.mean([r['annual_return'] for r in results]):>10.1f}% "
                  f"{np.mean([r['max_drawdown'] for r in results]):>10.1f}% "
                  f"{np.mean([r['win_rate'] for r in results]):>10.1f}% "
                  f"{np.mean([r['sharpe'] for r in results]):>9.2f}")
    
    # 对比
    print("\n" + "="*60)
    print("对比分析")
    print("="*60)
    
    blue_only = np.mean([r['annual_return'] for r in all_results['纯BLUE']]) if all_results['纯BLUE'] else 0
    blue_zone = np.mean([r['annual_return'] for r in all_results['BLUE+安全区']]) if all_results['BLUE+安全区'] else 0
    full = np.mean([r['annual_return'] for r in all_results['完整系统']]) if all_results['完整系统'] else 0
    
    print(f"\n纯BLUE vs Buy&Hold: {'+' if blue_only > np.mean([r['annual_return'] for r in all_results['Buy & Hold']]) else ''}{blue_only - np.mean([r['annual_return'] for r in all_results['Buy & Hold']]):.1f}%")
    print(f"BLUE+安全区 vs 纯BLUE: {'+' if blue_zone > blue_only else ''}{blue_zone - blue_only:.1f}%")
    print(f"完整系统 vs 纯BLUE: {'+' if full > blue_only else ''}{full - blue_only:.1f}%")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║              简化信号回测                                   ║
║                                                            ║
║  对比:                                                     ║
║  1. Buy & Hold                                             ║
║  2. 纯 BLUE (日线 >= 50 买入)                              ║ 
║  3. BLUE + 安全区域 (风险区不买)                           ║
║  4. 完整系统 (BLUE + 黑马 + 安全区域)                      ║
╚════════════════════════════════════════════════════════════╝
""")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
    run_test(symbols, 'US', days=730)
