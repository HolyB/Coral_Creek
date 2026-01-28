#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测引擎 - 验证交易信号的历史表现
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys
import os

# 添加父目录以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Backtester:
    """回测引擎核心类"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 交易佣金率 (双边)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = []
        self.results = {}
    
    def run_signal_backtest(self, signals_df: pd.DataFrame, 
                            holding_days: int = 10,
                            market: str = 'US') -> Dict:
        """
        对扫描信号进行固定持有期回测
        
        Args:
            signals_df: 信号数据，需包含 symbol, scan_date, price
            holding_days: 持有天数
            market: 市场 (US/CN)
        
        Returns:
            回测结果字典
        """
        from data_fetcher import get_us_stock_data, get_cn_stock_data
        
        self.trades = []
        
        for _, signal in signals_df.iterrows():
            symbol = signal.get('symbol') or signal.get('Symbol')
            entry_date = signal.get('scan_date') or signal.get('Date')
            entry_price = signal.get('price') or signal.get('Price')
            
            if not all([symbol, entry_date, entry_price]):
                continue
            
            try:
                # 获取后续价格数据
                if market == 'CN':
                    df = get_cn_stock_data(symbol, days=holding_days + 10)
                else:
                    df = get_us_stock_data(symbol, days=holding_days + 10)
                
                if df is None or len(df) < 2:
                    continue
                
                # 找到入场日期后的第N个交易日价格
                df = df.reset_index()
                entry_dt = pd.to_datetime(entry_date)
                
                # 找到入场日期的索引
                mask = df.index >= 0  # 简化处理，取最近数据
                if len(df) > holding_days:
                    exit_price = df.iloc[-1]['Close']
                    
                    # 计算收益
                    pnl = (exit_price - entry_price) / entry_price * 100
                    
                    self.trades.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'holding_days': holding_days,
                        'pnl_pct': round(pnl, 2),
                        'win': pnl > 0
                    })
                    
            except Exception as e:
                continue
        
        self.results = self._calculate_stats()
        return self.results
    
    def run_historical_backtest(self, 
                                 strategy_func,
                                 symbols: List[str],
                                 start_date: str,
                                 end_date: str,
                                 market: str = 'US') -> Dict:
        """
        运行历史回测 (完整时间序列)
        
        Args:
            strategy_func: 策略函数，接收价格数据返回信号
            symbols: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            market: 市场
        
        Returns:
            回测结果
        """
        from data_fetcher import get_us_stock_data, get_cn_stock_data
        
        portfolio_value = self.initial_capital
        self.equity_curve = [{'date': start_date, 'value': portfolio_value}]
        
        # 简化版：遍历每只股票
        for symbol in symbols:
            try:
                if market == 'CN':
                    df = get_cn_stock_data(symbol, days=365)
                else:
                    df = get_us_stock_data(symbol, days=365)
                
                if df is None or len(df) < 20:
                    continue
                
                # 应用策略函数获取信号
                signals = strategy_func(df)
                
                # 根据信号计算收益
                # (简化实现)
                
            except Exception as e:
                continue
        
        self.results = self._calculate_stats()
        return self.results
    
    def _calculate_stats(self) -> Dict:
        """计算回测统计指标"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基础统计
        total_trades = len(trades_df)
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] <= 0]
        
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        avg_return = trades_df['pnl_pct'].mean()
        total_return = trades_df['pnl_pct'].sum()
        
        # 标准差和夏普比率
        returns = trades_df['pnl_pct'].values
        std_return = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        # 盈亏比
        avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 最大回撤 (近似)
        cumulative = (1 + trades_df['pnl_pct'] / 100).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak * 100
        max_drawdown = abs(drawdown.min())
        
        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'avg_return': round(avg_return, 2),
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'trades': self.trades
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame(self.trades)
    
    def compare_with_benchmark(self, benchmark: str = 'SPY', period_days: int = 30) -> Dict:
        """
        与基准对比
        
        Args:
            benchmark: 基准代码 (SPY/QQQ)
            period_days: 对比周期
        
        Returns:
            对比结果
        """
        from data_fetcher import get_us_stock_data
        
        try:
            df = get_us_stock_data(benchmark, days=period_days + 5)
            if df is not None and len(df) >= 2:
                benchmark_return = (df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close'] * 100
            else:
                benchmark_return = 0
        except:
            benchmark_return = 0
        
        strategy_return = self.results.get('total_return', 0)
        
        return {
            'strategy_return': strategy_return,
            'benchmark_return': round(benchmark_return, 2),
            'alpha': round(strategy_return - benchmark_return, 2),
            'outperformed': strategy_return > benchmark_return
        }


def backtest_blue_signals(min_blue: float = 0.6, 
                          holding_days: int = 10, 
                          market: str = 'US',
                          limit: int = 50) -> Dict:
    """
    回测 BLUE 信号的历史表现
    
    Args:
        min_blue: 最小蓝色值过滤
        holding_days: 持有天数
        market: 市场
        limit: 信号数量限制
    
    Returns:
        回测结果
    """
    from db.database import query_scan_results
    
    # 获取历史信号
    results = query_scan_results(min_blue=min_blue, market=market, limit=limit)
    
    if not results:
        return {'error': 'No signals found'}
    
    signals_df = pd.DataFrame(results)
    
    # 运行回测
    bt = Backtester()
    stats = bt.run_signal_backtest(signals_df, holding_days=holding_days, market=market)
    
    # 添加基准对比
    benchmark = bt.compare_with_benchmark(
        benchmark='SPY' if market == 'US' else '000001.SS',
        period_days=30
    )
    stats['benchmark'] = benchmark
    
    return stats


if __name__ == "__main__":
    # 测试回测
    print("Testing Backtester...")
    
    result = backtest_blue_signals(min_blue=0.5, holding_days=10, market='US', limit=20)
    print(f"Results: {result}")
