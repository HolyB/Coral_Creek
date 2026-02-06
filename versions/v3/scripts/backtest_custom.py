#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略组合回测器 - 支持自由组合买卖条件
==========================================

使用方法:
    python scripts/backtest_custom.py --buy blue_heima bottom_peak --sell kdj_overbought chip_distribution

或直接编辑 CUSTOM_STRATEGY 配置
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_stock_data
from indicator_utils import (
    calculate_blue_signal_series, 
    calculate_heima_signal_series, 
    calculate_kdj_series
)
from strategies.strategy_components import (
    StrategyBuilder,
    BUY_CONDITIONS,
    SELL_CONDITIONS,
    list_all_conditions,
    create_original_strategy,
    create_chip_strategy,
    create_enhanced_strategy,
    create_conservative_strategy,
)


# ============================================================================
# 自定义策略配置 - 在这里定义您的策略
# ============================================================================

CUSTOM_STRATEGIES = {
    # 策略1: 原有系统
    "原有系统": {
        "buy": ["blue_heima"],
        "sell": ["kdj_overbought", "ma_break_2day"],
    },
    
    # 策略2: 筹码卖出
    "原有+筹码卖": {
        "buy": ["blue_heima"],
        "sell": ["kdj_overbought", "chip_distribution", "chip_with_ma"],
    },
    
    # 策略3: 您的新策略 - BLUE>=150 + 黑马 + 底部峰
    "增强策略": {
        "buy": ["strong_blue", "bottom_peak"],
        "sell": ["kdj_overbought", "chip_distribution"],
    },
    
    # 策略4: 保守策略 - 日周双BLUE + 底部峰
    "保守策略": {
        "buy": ["double_blue", "bottom_peak"],
        "sell": ["kdj_overbought", "chip_distribution", "safety_zone_high"],
    },
    
    # 策略5: 纯BLUE策略
    "纯BLUE策略": {
        "buy": ["blue_only"],
        "sell": ["kdj_overbought", "chip_distribution"],
    },
    
    # === 在这里添加更多自定义策略 ===
}


class CustomBacktester:
    """自定义策略回测器"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def prepare_data(self, df_daily: pd.DataFrame) -> Dict:
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
    
    def backtest_strategy(self, df_daily: pd.DataFrame, 
                          strategy: StrategyBuilder,
                          verbose: bool = False) -> Dict:
        """回测单个策略"""
        data = self.prepare_data(df_daily)
        
        cash = self.initial_capital
        shares = 0
        position = 0
        trades = []
        equity_curve = [self.initial_capital]
        
        for i in range(50, len(df_daily) - 1):
            close = data['close'][i]
            next_open = df_daily['Open'].iloc[i + 1]
            
            # 更新最高价 (用于移动止损)
            if position == 1:
                strategy.update_peak_price(close)
            
            # 卖出检查
            if position == 1:
                should_sell, reason = strategy.check_sell(data, i, df_daily)
                if should_sell:
                    revenue = shares * close * (1 - self.commission)
                    pnl = revenue - trades[-1]['cost']
                    cash += revenue
                    trades.append({
                        'type': 'SELL', 'price': close, 'shares': shares,
                        'pnl': pnl, 'reason': reason,
                        'date': df_daily.index[i].strftime('%Y-%m-%d')
                    })
                    if verbose:
                        print(f"    SELL: {df_daily.index[i].strftime('%Y-%m-%d')} ${close:.2f} | {reason} | PnL: ${pnl:.2f}")
                    shares = 0
                    position = 0
                    strategy.reset_position()
            
            # 买入检查
            elif position == 0:
                should_buy, reason = strategy.check_buy(data, i, df_daily)
                if should_buy and cash > 0:
                    shares = int(cash * (1 - self.commission) / next_open)
                    if shares > 0:
                        cost = shares * next_open * (1 + self.commission)
                        cash -= cost
                        position = 1
                        strategy.set_entry_price(next_open)
                        trades.append({
                            'type': 'BUY', 'price': next_open, 'shares': shares,
                            'cost': cost, 'reason': reason,
                            'date': df_daily.index[i+1].strftime('%Y-%m-%d')
                        })
                        if verbose:
                            print(f"    BUY:  {df_daily.index[i+1].strftime('%Y-%m-%d')} ${next_open:.2f} | {reason}")
            
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


def build_strategy_from_config(name: str, config: Dict) -> StrategyBuilder:
    """从配置构建策略"""
    strategy = StrategyBuilder(name)
    
    for buy_cond in config.get('buy', []):
        strategy.add_buy_condition(buy_cond)
    
    for sell_cond in config.get('sell', []):
        strategy.add_sell_condition(sell_cond)
    
    return strategy


def run_comparison(symbols: List[str], market: str = 'US', days: int = 730,
                   strategies_config: Dict = None, verbose: bool = False):
    """运行策略对比"""
    if strategies_config is None:
        strategies_config = CUSTOM_STRATEGIES
    
    all_results = {'Buy & Hold': []}
    for name in strategies_config:
        all_results[name] = []
    
    bt = CustomBacktester()
    
    for symbol in symbols:
        print(f"\n{'='*65}")
        print(f"回测: {symbol}")
        print(f"{'='*65}")
        
        df = get_stock_data(symbol, market, days=days)
        if df is None or len(df) < 100:
            print(f"  跳过: 数据不足")
            continue
        
        results = {}
        
        # Buy & Hold
        results['Buy & Hold'] = bt.backtest_buy_hold(df)
        
        # 各策略
        for name, config in strategies_config.items():
            strategy = build_strategy_from_config(name, config)
            if verbose:
                print(f"\n  {name}:")
            results[name] = bt.backtest_strategy(df, strategy, verbose=verbose)
        
        # 打印结果
        print(f"\n  {'策略':<16} {'年化%':<10} {'回撤%':<10} {'胜率%':<10} {'夏普':<8} {'买入':<6} {'卖出':<6}")
        print("  " + "-" * 72)
        for name, r in results.items():
            print(f"  {name:<16} {r['annual_return']:>8.1f}% {r['max_drawdown']:>8.1f}% "
                  f"{r['win_rate']:>8.1f}% {r['sharpe']:>7.2f} {r['buy_trades']:>5} {r['sell_trades']:>5}")
            all_results[name].append(r)
    
    # 汇总
    print(f"\n{'='*75}")
    print(f"综合汇总 ({len(symbols)} 只股票)")
    print(f"{'='*75}")
    
    print(f"\n{'策略':<16} {'平均年化%':<12} {'平均回撤%':<12} {'平均胜率%':<12} {'平均夏普':<10}")
    print("-" * 65)
    
    for name, results in all_results.items():
        if results:
            avg_annual = np.mean([r['annual_return'] for r in results])
            avg_dd = np.mean([r['max_drawdown'] for r in results])
            avg_wr = np.mean([r['win_rate'] for r in results])
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            
            print(f"{name:<16} {avg_annual:>10.1f}% {avg_dd:>10.1f}% "
                  f"{avg_wr:>10.1f}% {avg_sharpe:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description='策略组合回测器')
    parser.add_argument('--list', action='store_true', help='列出所有可用条件')
    parser.add_argument('--buy', nargs='+', help='买入条件列表')
    parser.add_argument('--sell', nargs='+', help='卖出条件列表')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META'],
                        help='股票列表')
    parser.add_argument('--days', type=int, default=730, help='回测天数')
    parser.add_argument('--verbose', action='store_true', help='显示详细交易记录')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_conditions()
        return
    
    # 如果指定了自定义条件
    if args.buy and args.sell:
        custom_config = {
            "自定义策略": {
                "buy": args.buy,
                "sell": args.sell,
            }
        }
        print("\n" + "="*60)
        print("自定义策略配置")
        print("="*60)
        print(f"买入条件: {', '.join(args.buy)}")
        print(f"卖出条件: {', '.join(args.sell)}")
        
        run_comparison(args.symbols, 'US', args.days, custom_config, verbose=args.verbose)
    else:
        # 使用预设策略
        print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         策略组合回测系统                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  买入条件:                                                                 ║
║    blue_heima    - BLUE>=100 + 黑马/掘地共振                               ║
║    strong_blue   - (日BLUE>=150 OR 周BLUE>=150) + 黑马/掘地                ║
║    double_blue   - 日BLUE>=150 AND 周BLUE>=150 + 黑马/掘地                 ║
║    bottom_peak   - 底部筹码顶格峰 + BLUE/黑马确认                          ║
║    blue_only     - 超强BLUE>=200 (无需黑马)                                ║
║    heima_only    - 日黑马 OR 日掘地                                        ║
║                                                                           ║
║  卖出条件:                                                                 ║
║    kdj_overbought    - KDJ J > 90 超买                                    ║
║    ma_break          - 跌破MA5                                            ║
║    ma_break_2day     - 连续2天跌破MA5                                      ║
║    chip_distribution - 筹码顶部堆积                                        ║
║    chip_with_ma      - 跌破MA5 + 筹码异常                                  ║
║    profit_target_20  - 止盈20%                                            ║
║    stop_loss_8       - 止损-8%                                            ║
║    trailing_stop_10  - 回撤10%止损                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
        run_comparison(args.symbols, 'US', args.days, verbose=args.verbose)


if __name__ == "__main__":
    main()
