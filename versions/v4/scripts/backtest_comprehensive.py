#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整信号系统回测
================

综合回测所有信号系统:
1. 现有系统: BLUE 200 (日/周/月) + 黑马 + 掘地
2. 现有系统 + 安全区域
3. 对比增强效果

评分体系 (模拟实际 _compute_verdict):
- BLUE日线 >= 200: +15
- BLUE周线 >= 100: +15  
- BLUE月线 >= 100: +10
- 黑马信号: +5
- 掘地信号: +3
- 安全区域: ±15
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
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series
from strategies.safety_zone_indicator import SafetyZoneIndicator


class ComprehensiveBacktester:
    """综合信号回测器"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.indicator = SafetyZoneIndicator()
    
    def compute_score_original(self, df_daily, df_weekly, df_monthly, i) -> Tuple[int, List[str]]:
        """
        计算原有系统得分 (BLUE + 黑马 + 掘地)
        
        实际 BLUE 分布:
        - >= 200: 非常罕见 (约 1-2%)
        - >= 100: 罕见 (约 2-5%)
        - >= 50: 较少 (约 10%)
        - > 0: 较常见 (约 20%)
        """
        score = 0
        reasons = []
        
        # 计算 BLUE
        if len(df_daily) > i:
            sub_daily = df_daily.iloc[:i+1]
            blue_daily = self._calc_blue(sub_daily)
            
            # 使用更合理的阈值
            if blue_daily >= 150:
                score += 20
                reasons.append(f"日BLUE {blue_daily:.0f} 极强")
            elif blue_daily >= 100:
                score += 15
                reasons.append(f"日BLUE {blue_daily:.0f} 强")
            elif blue_daily >= 50:
                score += 10
                reasons.append(f"日BLUE {blue_daily:.0f} 中等")
            elif blue_daily > 0:
                score += 5
        else:
            blue_daily = 0
        
        # 周线 BLUE
        if len(df_weekly) >= 3:
            blue_weekly = self._calc_blue(df_weekly)
            if blue_weekly >= 100:
                score += 15
                reasons.append(f"周BLUE {blue_weekly:.0f}")
            elif blue_weekly >= 50:
                score += 10
            elif blue_weekly > 0:
                score += 5
        
        # 月线 BLUE
        if len(df_monthly) >= 3:
            blue_monthly = self._calc_blue(df_monthly)
            if blue_monthly >= 100:
                score += 12
                reasons.append(f"月BLUE {blue_monthly:.0f}")
            elif blue_monthly >= 50:
                score += 8
            elif blue_monthly > 0:
                score += 4
        
        # 黑马/掘地
        if len(df_daily) > i:
            sub_daily = df_daily.iloc[:i+1]
            heima, juedi = self._calc_heima(sub_daily)
            if heima:
                score += 8
                reasons.append("黑马信号🐴")
            if juedi:
                score += 5
                reasons.append("掘地信号⛏️")
        
        return score, reasons
    
    def compute_score_with_zone(self, df_daily, df_weekly, df_monthly, i) -> Tuple[int, List[str]]:
        """
        计算增强系统得分 (原有 + 安全区域)
        """
        # 先计算原有得分
        score, reasons = self.compute_score_original(df_daily, df_weekly, df_monthly, i)
        
        # 添加安全区域
        if len(df_daily) > i and i >= 50:
            sub_daily = df_daily.iloc[:i+1]
            zone_result = self.indicator.calculate(sub_daily)
            zone_level = zone_result.get('safety_level', 50)
            zone_name = zone_result.get('zone_cn', '未知')
            
            if zone_level <= 20:
                score += 12
                reasons.append(f"安全区{zone_name}({zone_level:.0f})")
            elif zone_level <= 50:
                score += 6
                reasons.append(f"粉区{zone_name}({zone_level:.0f})")
            elif zone_level <= 80:
                pass  # 绿区持股不加减分
            elif zone_level <= 90:
                score -= 5
                reasons.append(f"风险区{zone_name}({zone_level:.0f})")
            else:
                score -= 10
                reasons.append(f"高风险{zone_name}({zone_level:.0f})")
        
        return score, reasons
    
    def _calc_blue(self, df) -> float:
        try:
            blue = calculate_blue_signal_series(
                df['Open'].values, df['High'].values,
                df['Low'].values, df['Close'].values
            )
            return float(blue[-1]) if len(blue) > 0 else 0
        except:
            return 0
    
    def _calc_heima(self, df) -> Tuple[bool, bool]:
        try:
            heima, juedi = calculate_heima_signal_series(
                df['High'].values, df['Low'].values,
                df['Close'].values, df['Open'].values
            )
            return bool(heima[-1]) if len(heima) > 0 else False, \
                   bool(juedi[-1]) if len(juedi) > 0 else False
        except:
            return False, False
    
    def backtest_strategy(self, df_daily, df_weekly, df_monthly, 
                          use_zone: bool = False,
                          buy_threshold: int = 30,
                          sell_threshold: int = 15) -> Dict:
        """
        回测策略
        
        Args:
            use_zone: 是否使用安全区域
            buy_threshold: 买入阈值分数
            sell_threshold: 卖出阈值分数
        """
        cash = self.initial_capital
        shares = 0
        position = 0
        
        trades = []
        equity_curve = [self.initial_capital]
        score_history = []
        
        for i in range(60, len(df_daily)):
            price = float(df_daily['Close'].iloc[i])
            
            # 计算得分
            if use_zone:
                score, reasons = self.compute_score_with_zone(df_daily, df_weekly, df_monthly, i)
            else:
                score, reasons = self.compute_score_original(df_daily, df_weekly, df_monthly, i)
            
            score_history.append(score)
            
            # 交易逻辑
            if score >= buy_threshold and position == 0 and cash > 0:
                # 买入
                shares = int(cash * (1 - self.commission) / price)
                if shares > 0:
                    cost = shares * price * (1 + self.commission)
                    cash -= cost
                    position = 1
                    trades.append({
                        'type': 'BUY',
                        'date': df_daily.index[i] if hasattr(df_daily.index[i], 'strftime') else i,
                        'price': price,
                        'shares': shares,
                        'score': score,
                        'reasons': reasons
                    })
            
            elif score < sell_threshold and position == 1 and shares > 0:
                # 卖出
                revenue = shares * price * (1 - self.commission)
                pnl = revenue - trades[-1]['price'] * trades[-1]['shares'] if trades else 0
                cash += revenue
                trades.append({
                    'type': 'SELL',
                    'date': df_daily.index[i] if hasattr(df_daily.index[i], 'strftime') else i,
                    'price': price,
                    'shares': shares,
                    'pnl': pnl,
                    'score': score
                })
                shares = 0
                position = 0
            
            # 计算权益
            equity = cash + shares * price
            equity_curve.append(equity)
        
        # 计算结果
        equity_curve = np.array(equity_curve)
        final_equity = cash + shares * float(df_daily['Close'].iloc[-1])
        
        total_return = (final_equity / self.initial_capital - 1) * 100
        days = len(df_daily)
        annual_return = ((final_equity / self.initial_capital) ** (252 / days) - 1) * 100
        
        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # 胜率
        winning_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        total_sells = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = (winning_trades / total_sells * 100) if total_sells > 0 else 0
        
        # 夏普
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'trades': len(trades),
            'final_equity': final_equity,
            'avg_score': np.mean(score_history) if score_history else 0,
        }
    
    def backtest_buy_hold(self, df_daily) -> Dict:
        """Buy & Hold 基准"""
        start_price = float(df_daily['Close'].iloc[0])
        end_price = float(df_daily['Close'].iloc[-1])
        
        total_return = (end_price / start_price - 1) * 100
        days = len(df_daily)
        annual_return = ((end_price / start_price) ** (252 / days) - 1) * 100
        
        equity = df_daily['Close'] / start_price * self.initial_capital
        peak = equity.cummax()
        drawdown = (peak - equity) / peak * 100
        max_drawdown = drawdown.max()
        
        returns = df_daily['Close'].pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': 100.0 if total_return > 0 else 0.0,
            'sharpe': sharpe,
            'trades': 1,
            'final_equity': self.initial_capital * (1 + total_return / 100),
            'avg_score': 0,
        }


def run_comprehensive_backtest(symbol: str, market: str = 'US', days: int = 730):
    """对单只股票进行综合回测"""
    print(f"\n{'='*60}")
    print(f"综合回测: {symbol}")
    print(f"{'='*60}")
    
    df = get_stock_data(symbol, market, days=days)
    if df is None or len(df) < 100:
        print(f"❌ 数据不足: {symbol}")
        return None
    
    # 转换周/月线
    df_daily = df.copy()
    df_weekly = df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_monthly = df.resample('M').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    backtester = ComprehensiveBacktester()
    
    results = {}
    
    # 1. Buy & Hold
    print("  测试 Buy & Hold...")
    results['Buy & Hold'] = backtester.backtest_buy_hold(df_daily)
    
    # 2. 原有系统 (BLUE 200 + 黑马 + 掘地)
    print("  测试 原有系统 (BLUE+黑马+掘地)...")
    results['原有系统'] = backtester.backtest_strategy(
        df_daily, df_weekly, df_monthly, 
        use_zone=False, buy_threshold=20, sell_threshold=10
    )
    
    # 3. 原有 + 安全区域
    print("  测试 综合系统 (原有+安全区域)...")
    results['综合系统'] = backtester.backtest_strategy(
        df_daily, df_weekly, df_monthly,
        use_zone=True, buy_threshold=25, sell_threshold=12
    )
    
    # 4. 更激进的综合系统
    print("  测试 激进综合 (阈值更低)...")
    results['激进综合'] = backtester.backtest_strategy(
        df_daily, df_weekly, df_monthly,
        use_zone=True, buy_threshold=18, sell_threshold=8
    )
    
    # 打印结果
    print(f"\n{'策略':<12} {'总收益%':<10} {'年化%':<10} {'回撤%':<10} {'胜率%':<10} {'夏普':<8} {'交易':<6}")
    print("-" * 75)
    
    for name, r in results.items():
        print(f"{name:<12} {r['total_return']:>8.1f}% {r['annual_return']:>8.1f}% "
              f"{r['max_drawdown']:>8.1f}% {r['win_rate']:>8.1f}% {r['sharpe']:>7.2f} {r['trades']:>5}")
    
    return results


def run_multi_stock_backtest(symbols: List[str], market: str = 'US', days: int = 730):
    """多股票综合回测"""
    all_results = {
        'Buy & Hold': [],
        '原有系统': [],
        '综合系统': [],
        '激进综合': [],
    }
    
    for symbol in symbols:
        results = run_comprehensive_backtest(symbol, market, days)
        if results:
            for name, r in results.items():
                if name in all_results:
                    all_results[name].append(r)
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"综合汇总 ({len(symbols)} 只股票)")
    print(f"{'='*70}")
    
    print(f"\n{'策略':<12} {'平均年化%':<12} {'平均回撤%':<12} {'平均胜率%':<12} {'平均夏普':<10} {'平均交易':<8}")
    print("-" * 75)
    
    summary = []
    for name, results in all_results.items():
        if results:
            avg_annual = np.mean([r['annual_return'] for r in results])
            avg_dd = np.mean([r['max_drawdown'] for r in results])
            avg_wr = np.mean([r['win_rate'] for r in results])
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            avg_trades = np.mean([r['trades'] for r in results])
            
            print(f"{name:<12} {avg_annual:>10.1f}% {avg_dd:>10.1f}% {avg_wr:>10.1f}% "
                  f"{avg_sharpe:>9.2f} {avg_trades:>7.1f}")
            
            summary.append({
                'strategy': name,
                'avg_annual': avg_annual,
                'avg_drawdown': avg_dd,
                'avg_win_rate': avg_wr,
                'avg_sharpe': avg_sharpe,
            })
    
    # 对比分析
    print("\n" + "=" * 70)
    print("对比分析:")
    print("=" * 70)
    
    if len(summary) >= 3:
        original = next((s for s in summary if s['strategy'] == '原有系统'), None)
        combined = next((s for s in summary if s['strategy'] == '综合系统'), None)
        
        if original and combined:
            return_diff = combined['avg_annual'] - original['avg_annual']
            dd_diff = original['avg_drawdown'] - combined['avg_drawdown']
            wr_diff = combined['avg_win_rate'] - original['avg_win_rate']
            sharpe_diff = combined['avg_sharpe'] - original['avg_sharpe']
            
            print(f"\n  综合系统 vs 原有系统:")
            print(f"    年化收益: {'+' if return_diff > 0 else ''}{return_diff:.1f}%")
            print(f"    回撤改善: {'+' if dd_diff > 0 else ''}{dd_diff:.1f}%")
            print(f"    胜率提升: {'+' if wr_diff > 0 else ''}{wr_diff:.1f}%")
            print(f"    夏普提升: {'+' if sharpe_diff > 0 else ''}{sharpe_diff:.2f}")
            
            if return_diff > 0 and sharpe_diff > 0:
                print(f"\n  ✅ 结论: 安全区域显著增强了信号系统!")
            elif sharpe_diff > 0:
                print(f"\n  ✅ 结论: 安全区域改善了风险调整收益!")
            else:
                print(f"\n  ⚠️ 结论: 安全区域效果需进一步验证")
    
    return summary


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           综合信号系统回测                                        ║
║                                                                  ║
║  对比策略:                                                       ║
║  1. Buy & Hold (基准)                                            ║
║  2. 原有系统: BLUE 200 (日/周/月) + 黑马 + 掘地                   ║
║  3. 综合系统: 原有 + 安全区域                                    ║
║  4. 激进综合: 更低买入阈值                                       ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # 测试股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
    
    run_multi_stock_backtest(symbols, 'US', days=730)
