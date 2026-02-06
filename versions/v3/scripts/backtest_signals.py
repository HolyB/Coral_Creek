#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
信号组合回测对比
================

对比以下策略的回测效果:
1. 纯 BLUE 策略
2. 纯安全区域策略
3. BLUE + 安全区域共识策略
4. Buy & Hold 基准

输出:
- 年化收益率
- 最大回撤
- 胜率
- 夏普比率
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_stock_data
from indicator_utils import calculate_blue_signal_series
from strategies.safety_zone_indicator import SafetyZoneIndicator


class SignalBacktester:
    """信号回测器"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.indicator = SafetyZoneIndicator()
    
    def backtest_pure_blue(self, df: pd.DataFrame, blue_threshold: float = 100) -> Dict:
        """
        纯 BLUE 策略回测
        买入: BLUE >= threshold
        卖出: BLUE < threshold * 0.5
        """
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values
        )
        
        signals = []
        for i in range(len(blue)):
            if blue[i] >= blue_threshold:
                signals.append('BUY')
            elif blue[i] < blue_threshold * 0.5:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return self._simulate_trades(df, signals, "BLUE策略")
    
    def backtest_pure_zone(self, df: pd.DataFrame) -> Dict:
        """
        纯安全区域策略回测
        买入: 安全区(0-50) + 买入信号
        卖出: 风险区(80-100) + 卖出信号
        """
        signals = []
        
        for i in range(50, len(df)):
            sub_df = df.iloc[:i+1]
            result = self.indicator.get_signals(sub_df)
            
            zone_level = result['zone']['level']
            
            if zone_level <= 50 and result['buy_score'] >= 2:
                signals.append('BUY')
            elif zone_level >= 80 and result['sell_score'] >= 1:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        # 前 50 天填充 HOLD
        signals = ['HOLD'] * 50 + signals
        
        return self._simulate_trades(df, signals, "安全区域策略")
    
    def backtest_consensus(self, df: pd.DataFrame) -> Dict:
        """
        BLUE + 安全区域共识策略
        买入: BLUE 强 + 安全区域低
        卖出: BLUE 弱 + 风险区域高
        """
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values
        )
        
        signals = []
        
        for i in range(50, len(df)):
            sub_df = df.iloc[:i+1]
            blue_value = blue[i]
            
            consensus = self.indicator.get_consensus_with_blue(sub_df, blue_value)
            
            if consensus['consensus'] == 'STRONG_BUY':
                signals.append('BUY')
            elif consensus['consensus'] == 'BUY':
                signals.append('BUY')
            elif consensus['consensus'] == 'SELL':
                signals.append('SELL')
            elif consensus['consensus'] == 'AVOID':
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        signals = ['HOLD'] * 50 + signals
        
        return self._simulate_trades(df, signals, "共识策略")
    
    def backtest_buy_hold(self, df: pd.DataFrame) -> Dict:
        """Buy & Hold 基准 - 第一天买入，一直持有"""
        # 简单计算 Buy & Hold
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        total_return = (end_price / start_price - 1) * 100
        days = len(df)
        annual_return = ((end_price / start_price) ** (252 / days) - 1) * 100
        
        # 计算最大回撤
        equity = df['Close'] / start_price * self.initial_capital
        peak = equity.cummax()
        drawdown = (peak - equity) / peak * 100
        max_drawdown = drawdown.max()
        
        # 夏普比率
        returns = df['Close'].pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        return {
            'strategy': 'Buy & Hold',
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': 100.0 if total_return > 0 else 0.0,
            'trades': 1,
            'sharpe': sharpe,
            'final_equity': self.initial_capital * (1 + total_return / 100),
        }
    
    def _simulate_trades(self, df: pd.DataFrame, signals: List[str], strategy_name: str) -> Dict:
        """模拟交易"""
        cash = self.initial_capital
        shares = 0
        position = 0  # 0: 空仓, 1: 持仓
        
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            signal = signals[i] if i < len(signals) else 'HOLD'
            
            # 买入
            if signal == 'BUY' and position == 0 and cash > 0:
                shares = int(cash * (1 - self.commission) / price)
                if shares > 0:
                    cost = shares * price * (1 + self.commission)
                    cash -= cost
                    position = 1
                    trades.append({
                        'type': 'BUY',
                        'date': df.index[i],
                        'price': price,
                        'shares': shares
                    })
            
            # 卖出
            elif signal == 'SELL' and position == 1 and shares > 0:
                revenue = shares * price * (1 - self.commission)
                cash += revenue
                trades.append({
                    'type': 'SELL',
                    'date': df.index[i],
                    'price': price,
                    'shares': shares,
                    'pnl': revenue - trades[-1]['price'] * trades[-1]['shares'] if trades else 0
                })
                shares = 0
                position = 0
            
            # 计算权益
            equity = cash + shares * price
            equity_curve.append(equity)
        
        # 计算结果
        equity_curve = np.array(equity_curve)
        
        # 如果还持有，计算最终价值
        final_equity = cash + shares * df['Close'].iloc[-1]
        
        # 收益率
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # 年化收益
        days = len(df)
        annual_return = ((final_equity / self.initial_capital) ** (252 / days) - 1) * 100
        
        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # 胜率
        winning_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        total_sells = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = (winning_trades / total_sells * 100) if total_sells > 0 else 0
        
        # 夏普比率
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        return {
            'strategy': strategy_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': len(trades),
            'sharpe': sharpe,
            'final_equity': final_equity,
        }


def run_comparison(symbol: str, market: str = 'US', days: int = 730):
    """运行对比回测"""
    print(f"\n{'='*60}")
    print(f"回测对比: {symbol} ({days}天)")
    print(f"{'='*60}")
    
    df = get_stock_data(symbol, market, days=days)
    if df is None or len(df) < 100:
        print(f"❌ 数据不足: {symbol}")
        return None
    
    backtester = SignalBacktester()
    
    results = []
    
    # 1. Buy & Hold
    print("  运行 Buy & Hold...")
    results.append(backtester.backtest_buy_hold(df))
    
    # 2. 纯 BLUE
    print("  运行 BLUE 策略...")
    results.append(backtester.backtest_pure_blue(df))
    
    # 3. 纯安全区域
    print("  运行 安全区域策略...")
    results.append(backtester.backtest_pure_zone(df))
    
    # 4. 共识策略
    print("  运行 共识策略...")
    results.append(backtester.backtest_consensus(df))
    
    # 打印结果
    print(f"\n{'策略':<15} {'总收益%':<10} {'年化%':<10} {'最大回撤%':<10} {'胜率%':<10} {'夏普':<8} {'交易次数':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['strategy']:<15} {r['total_return']:>8.1f}% {r['annual_return']:>8.1f}% {r['max_drawdown']:>8.1f}% {r['win_rate']:>8.1f}% {r['sharpe']:>7.2f} {r['trades']:>6}")
    
    return results


def run_multi_stock_comparison(symbols: List[str], market: str = 'US', days: int = 730):
    """多股票对比"""
    all_results = {
        'Buy & Hold': [],
        'BLUE策略': [],
        '安全区域策略': [],
        '共识策略': [],
    }
    
    for symbol in symbols:
        results = run_comparison(symbol, market, days)
        if results:
            for r in results:
                all_results[r['strategy']].append(r)
    
    # 汇总平均
    print(f"\n{'='*60}")
    print(f"综合平均结果 ({len(symbols)} 只股票)")
    print(f"{'='*60}")
    
    print(f"\n{'策略':<15} {'平均年化%':<12} {'平均回撤%':<12} {'平均胜率%':<12} {'平均夏普':<10}")
    print("-" * 65)
    
    summary = []
    for strategy, results in all_results.items():
        if results:
            avg_annual = np.mean([r['annual_return'] for r in results])
            avg_dd = np.mean([r['max_drawdown'] for r in results])
            avg_wr = np.mean([r['win_rate'] for r in results])
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            
            print(f"{strategy:<15} {avg_annual:>10.1f}% {avg_dd:>10.1f}% {avg_wr:>10.1f}% {avg_sharpe:>9.2f}")
            
            summary.append({
                'strategy': strategy,
                'avg_annual_return': avg_annual,
                'avg_max_drawdown': avg_dd,
                'avg_win_rate': avg_wr,
                'avg_sharpe': avg_sharpe,
            })
    
    return summary


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           信号组合回测对比                                    ║
║                                                              ║
║  对比策略:                                                   ║
║  1. Buy & Hold (基准)                                        ║
║  2. BLUE 策略 (BLUE >= 100 买入)                            ║
║  3. 安全区域策略 (安全区+买入信号)                           ║
║  4. 共识策略 (BLUE + 安全区域)                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 测试多只股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
    
    summary = run_multi_stock_comparison(symbols, 'US', days=730)
    
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    
    # 找出最佳策略
    if summary:
        best_by_return = max(summary, key=lambda x: x['avg_annual_return'])
        best_by_sharpe = max(summary, key=lambda x: x['avg_sharpe'])
        
        print(f"  最高年化收益: {best_by_return['strategy']} ({best_by_return['avg_annual_return']:.1f}%)")
        print(f"  最佳风险调整: {best_by_sharpe['strategy']} (夏普 {best_by_sharpe['avg_sharpe']:.2f})")
