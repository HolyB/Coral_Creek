#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动回测调度器 (Auto Backtest Scheduler)
========================================

功能:
1. 基于每日扫描信号自动执行 Paper Trading
2. 跟踪持仓表现
3. 生成回测报告
4. 发送通知

使用:
    python scripts/auto_backtest.py --mode paper    # Paper Trading 自动交易
    python scripts/auto_backtest.py --mode backtest # 历史回测
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(parent_dir, '.env'))


class AutoBacktester:
    """
    自动回测器
    
    基于信号自动执行交易并跟踪表现
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_positions: int = 10,
                 position_size_pct: float = 0.10,
                 stop_loss_pct: float = 0.08,
                 take_profit_pct: float = 0.20,
                 min_blue_score: float = 100,
                 use_paper_trading: bool = True):
        """
        初始化
        
        Args:
            initial_capital: 初始资金
            max_positions: 最大持仓数量
            position_size_pct: 单只股票仓位比例
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
            min_blue_score: 最低 BLUE 分数要求
            use_paper_trading: 是否使用 Alpaca Paper Trading
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_blue_score = min_blue_score
        self.use_paper_trading = use_paper_trading
        
        # Alpaca 连接
        self.trader = None
        self.signal_trader = None
        
        if use_paper_trading:
            self._init_alpaca()
        
        # 交易记录
        self.trades: List[Dict] = []
        self.daily_equity: List[Dict] = []
        
    def _init_alpaca(self):
        """初始化 Alpaca 连接"""
        try:
            from execution.alpaca_trader import AlpacaTrader, SignalTrader, ALPACA_SDK_AVAILABLE
            
            if not ALPACA_SDK_AVAILABLE:
                print("❌ 请安装 alpaca-py: pip install alpaca-py")
                return
            
            api_key = os.environ.get('ALPACA_API_KEY')
            secret_key = os.environ.get('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                print("❌ 请设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY 环境变量")
                return
            
            self.trader = AlpacaTrader(paper=True)
            self.signal_trader = SignalTrader(
                trader=self.trader,
                max_position_pct=self.position_size_pct,
                stop_loss_pct=self.stop_loss_pct
            )
            
            account = self.trader.get_account()
            print(f"✅ Alpaca Paper Trading 连接成功!")
            print(f"   账户余额: ${account.equity:,.2f}")
            print(f"   可用资金: ${account.buying_power:,.2f}")
            
        except Exception as e:
            print(f"❌ Alpaca 连接失败: {e}")
            self.trader = None
    
    def get_today_signals(self, min_turnover: float = 10.0) -> List[Dict]:
        """
        获取今日扫描信号
        
        Args:
            min_turnover: 最低成交额 (百万美元)，过滤低流动性股票
        """
        try:
            from db.database import query_scan_results, get_scanned_dates
            
            # 获取最新扫描日期
            dates = get_scanned_dates(market='US')
            if not dates:
                print("⚠️ 没有找到扫描数据")
                return []
            
            latest_date = dates[0]
            print(f"📅 使用扫描日期: {latest_date}")
            
            # 查询信号
            results = query_scan_results(
                scan_date=latest_date,
                market='US',
                min_blue=self.min_blue_score
            )
            
            if not results:
                print(f"⚠️ 没有满足条件的信号 (BLUE >= {self.min_blue_score})")
                return []
            
            # 过滤低流动性股票和验证 Alpaca 支持
            filtered = []
            for r in results:
                turnover = r.get('turnover_m') or 0
                market_cap = r.get('market_cap') or 0
                symbol = r.get('symbol', '')
                
                # 过滤条件: 成交额 >= $10M, 市值 >= $100M, 非特殊符号
                if (turnover >= min_turnover and 
                    market_cap >= 100_000_000 and
                    len(symbol) <= 5 and  # 排除特殊后缀
                    not any(c in symbol for c in ['-', '.', '/'])):
                    
                    # 验证 Alpaca 能获取价格
                    if self.trader:
                        try:
                            price = self.trader.get_latest_price(symbol)
                            if price > 0:
                                r['current_price'] = price
                                filtered.append(r)
                        except:
                            pass
                    else:
                        filtered.append(r)
                    
                    # 最多验证 20 只
                    if len(filtered) >= 20:
                        break
            
            print(f"📊 过滤后: {len(filtered)}/{len(results)} (成交额 >= ${min_turnover}M, 市值 >= $100M)")
            
            # 按 BLUE 分数排序
            filtered.sort(key=lambda x: x.get('blue_daily', 0) or 0, reverse=True)
            
            return filtered
            
        except Exception as e:
            print(f"❌ 获取信号失败: {e}")
            return []
    
    def execute_signals(self, signals: List[Dict]) -> Dict:
        """
        执行信号交易
        
        Args:
            signals: 信号列表
            
        Returns:
            执行结果
        """
        if not self.trader:
            return {'success': False, 'message': 'Alpaca 未连接'}
        
        results = {
            'executed': [],
            'skipped': [],
            'errors': []
        }
        
        # 获取当前持仓
        positions = self.trader.get_positions()
        current_symbols = {p.symbol for p in positions}
        
        available_slots = self.max_positions - len(current_symbols)
        print(f"📊 当前持仓: {len(current_symbols)}, 可用槽位: {available_slots}")
        
        if available_slots <= 0:
            results['message'] = '持仓已满'
            return results
        
        # 执行买入
        for signal in signals[:available_slots]:
            symbol = signal.get('symbol')
            
            if symbol in current_symbols:
                results['skipped'].append({
                    'symbol': symbol,
                    'reason': '已持仓'
                })
                continue
            
            blue_score = signal.get('blue_daily', 0)
            reason = f"BLUE={blue_score:.0f}"
            
            print(f"🔄 执行买入: {symbol} ({reason})")
            
            result = self.signal_trader.execute_buy_signal(symbol, reason)
            
            if result['success']:
                results['executed'].append(result)
                print(f"   ✅ {result['message']}")
            else:
                results['errors'].append(result)
                print(f"   ❌ {result['message']}")
        
        return results
    
    def check_stop_conditions(self) -> Dict:
        """
        检查止盈止损条件，执行卖出
        """
        if not self.trader:
            return {'success': False}
        
        positions = self.trader.get_positions()
        results = {'sold': [], 'kept': []}
        
        for pos in positions:
            pnl_pct = pos.unrealized_plpc / 100  # 转换为小数
            
            # 止盈
            if pnl_pct >= self.take_profit_pct:
                print(f"🎯 止盈卖出: {pos.symbol} (+{pos.unrealized_plpc:.2f}%)")
                result = self.signal_trader.execute_sell_signal(
                    pos.symbol, 
                    f"止盈: +{pos.unrealized_plpc:.2f}%"
                )
                results['sold'].append(result)
                
            # 止损 (通过止损单自动执行，这里只记录)
            elif pnl_pct <= -self.stop_loss_pct:
                print(f"🛑 止损触发: {pos.symbol} ({pos.unrealized_plpc:.2f}%)")
                results['sold'].append({
                    'symbol': pos.symbol,
                    'reason': '止损触发'
                })
            else:
                results['kept'].append({
                    'symbol': pos.symbol,
                    'pnl_pct': pos.unrealized_plpc
                })
        
        return results
    
    def get_portfolio_status(self) -> Dict:
        """获取当前投资组合状态"""
        if not self.trader:
            return {}
        
        return self.signal_trader.get_portfolio_summary()
    
    def run_daily_routine(self) -> Dict:
        """
        执行每日例行程序
        
        1. 检查止盈止损
        2. 获取今日信号
        3. 执行新的买入
        """
        print("\n" + "="*60)
        print(f"📅 每日自动交易 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        results = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'stop_check': None,
            'signals': None,
            'executions': None,
            'portfolio': None
        }
        
        # 1. 检查止盈止损
        print("\n🔍 Step 1: 检查止盈止损条件...")
        results['stop_check'] = self.check_stop_conditions()
        
        # 2. 获取今日信号
        print("\n📡 Step 2: 获取交易信号...")
        signals = self.get_today_signals()
        results['signals'] = len(signals)
        
        # 3. 执行买入
        if signals:
            print(f"\n💰 Step 3: 执行买入 (Top {min(self.max_positions, len(signals))} 信号)...")
            results['executions'] = self.execute_signals(signals)
        
        # 4. 获取当前组合状态
        print("\n📊 Step 4: 获取投资组合状态...")
        results['portfolio'] = self.get_portfolio_status()
        
        # 记录每日权益
        if results['portfolio']:
            self.daily_equity.append({
                'date': results['date'],
                'equity': results['portfolio']['account']['equity'],
                'cash': results['portfolio']['account']['cash'],
                'positions': results['portfolio']['position_count']
            })
        
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """打印汇总"""
        print("\n" + "="*60)
        print("📋 汇总报告")
        print("="*60)
        
        portfolio = results.get('portfolio', {})
        if portfolio:
            account = portfolio.get('account', {})
            print(f"\n💰 账户状态:")
            print(f"   总权益: ${account.get('equity', 0):,.2f}")
            print(f"   现金: ${account.get('cash', 0):,.2f}")
            print(f"   持仓数: {portfolio.get('position_count', 0)}")
            print(f"   总盈亏: ${portfolio.get('total_pnl', 0):,.2f}")
            
            positions = portfolio.get('positions', [])
            if positions:
                print(f"\n📊 持仓详情:")
                for p in positions:
                    pnl_color = "🟢" if p['pnl'] >= 0 else "🔴"
                    print(f"   {pnl_color} {p['symbol']}: {p['qty']}股 @ ${p['avg_entry']:.2f} "
                          f"-> ${p['current_price']:.2f} ({p['pnl_pct']:+.2f}%)")
        
        executions = results.get('executions', {})
        if executions:
            executed = executions.get('executed', [])
            if executed:
                print(f"\n✅ 今日买入: {len(executed)} 笔")
                for e in executed:
                    print(f"   {e['symbol']}: {e['qty']}股 @ ${e['price']:.2f}")


def run_historical_backtest(symbols: List[str] = None, days: int = 365):
    """
    运行历史回测 (不使用 Alpaca，纯历史数据回测)
    """
    from backtester import SimpleBacktester
    
    if not symbols:
        # 获取最近信号作为回测标的
        try:
            from db.database import query_scan_results, get_scanned_dates
            dates = get_scanned_dates(market='US')
            if dates:
                results = query_scan_results(scan_date=dates[0], market='US', min_blue=100)
                symbols = [r['symbol'] for r in results[:10]]
        except:
            symbols = ['NVDA', 'AAPL', 'MSFT', 'META', 'GOOGL']
    
    print(f"\n📊 历史回测: {len(symbols)} 只股票, {days} 天")
    print("="*60)
    
    all_results = []
    
    for symbol in symbols:
        try:
            bt = SimpleBacktester(
                symbol=symbol,
                market='US',
                days=days,
                blue_threshold=100
            )
            bt.load_data()
            bt.calculate_signals()
            results = bt.run_backtest()
            
            print(f"✅ {symbol}: 收益 {results.get('total_return', 0):.2f}%, "
                  f"胜率 {results.get('win_rate', 0):.0f}%")
            
            all_results.append({
                'symbol': symbol,
                **results
            })
            
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    # 汇总
    if all_results:
        avg_return = sum(r.get('total_return', 0) for r in all_results) / len(all_results)
        avg_win_rate = sum(r.get('win_rate', 0) for r in all_results) / len(all_results)
        
        print(f"\n📈 汇总: 平均收益 {avg_return:.2f}%, 平均胜率 {avg_win_rate:.0f}%")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='自动回测调度器')
    parser.add_argument('--mode', choices=['paper', 'backtest', 'status'], 
                        default='status', help='运行模式')
    parser.add_argument('--days', type=int, default=365, help='回测天数')
    parser.add_argument('--max-positions', type=int, default=5, help='最大持仓数')
    parser.add_argument('--min-blue', type=float, default=100, help='最低BLUE分数')
    
    args = parser.parse_args()
    
    if args.mode == 'paper':
        # Paper Trading 模式
        backtester = AutoBacktester(
            max_positions=args.max_positions,
            min_blue_score=args.min_blue,
            use_paper_trading=True
        )
        backtester.run_daily_routine()
        
    elif args.mode == 'backtest':
        # 历史回测模式
        run_historical_backtest(days=args.days)
        
    elif args.mode == 'status':
        # 只显示状态
        backtester = AutoBacktester(use_paper_trading=True)
        status = backtester.get_portfolio_status()
        
        if status:
            account = status.get('account', {})
            print(f"\n💰 账户状态:")
            print(f"   类型: {'模拟盘' if account.get('is_paper') else '实盘'}")
            print(f"   总权益: ${account.get('equity', 0):,.2f}")
            print(f"   可用资金: ${account.get('buying_power', 0):,.2f}")
            
            positions = status.get('positions', [])
            print(f"\n📊 持仓 ({len(positions)} 只):")
            for p in positions:
                pnl_color = "🟢" if p['pnl'] >= 0 else "🔴"
                print(f"   {pnl_color} {p['symbol']}: {p['qty']}股, "
                      f"${p['market_value']:.2f}, {p['pnl_pct']:+.2f}%")


if __name__ == "__main__":
    main()
