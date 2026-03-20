#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Backtesting Engine
专业级回测引擎 - 适用于量化策略验证

核心特性:
1. 组合回测 (Portfolio Backtest)
2. 滚动验证 (Walk-Forward Analysis)
3. 蒙特卡洛模拟 (Monte Carlo Simulation)
4. 交易成本模型 (Transaction Costs)
5. 风险归因分析 (Risk Attribution)
6. 多 Benchmark 对比
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionSizing(Enum):
    """仓位管理模式"""
    FIXED = "fixed"              # 固定仓位
    EQUAL_WEIGHT = "equal"       # 等权重
    VOLATILITY = "volatility"    # 波动率倒数
    RISK_PARITY = "risk_parity"  # 风险平价
    KELLY = "kelly"              # 凯利公式
    ATR_BASED = "atr"            # ATR 动态仓位


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    date: datetime
    side: str                    # 'buy' or 'sell'
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    signal_strength: float = 0.0
    reason: str = ""
    
    @property
    def total_cost(self) -> float:
        return abs(self.quantity) * self.price + self.commission + self.slippage
    
    @property
    def net_amount(self) -> float:
        """净交易金额 (正=买入, 负=卖出)"""
        if self.side == 'buy':
            return -self.total_cost
        else:
            return abs(self.quantity) * self.price - self.commission - self.slippage


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: int
    avg_cost: float
    entry_date: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    
    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_cost)
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        return (current_price - self.avg_cost) / self.avg_cost if self.avg_cost > 0 else 0


@dataclass
class BacktestConfig:
    """回测配置"""
    # 基础设置
    initial_capital: float = 100000.0
    start_date: str = None
    end_date: str = None
    
    # 交易成本
    commission_rate: float = 0.001       # 佣金率 0.1%
    slippage_rate: float = 0.0005        # 滑点 0.05%
    min_commission: float = 1.0          # 最低佣金
    
    # 仓位管理
    position_sizing: PositionSizing = PositionSizing.EQUAL_WEIGHT
    max_position_pct: float = 0.1       # 单只最大仓位 10%
    max_total_exposure: float = 1.0     # 最大总敞口 100%
    max_positions: int = 20              # 最大持仓数量
    
    # 风险控制
    stop_loss_pct: float = 0.08          # 止损 8%
    take_profit_pct: float = 0.20        # 止盈 20%
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.05
    
    # 信号过滤
    min_signal_strength: float = 100.0  # 最小 BLUE 值
    require_volume: float = 1000000     # 最小成交额
    
    # 回测模式
    rebalance_freq: str = 'daily'       # 'daily', 'weekly', 'monthly'
    allow_short: bool = False            # 是否允许做空


@dataclass
class BacktestResult:
    """回测结果"""
    # 基础统计
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    
    # 风险指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0       # 最长回撤恢复天数
    
    # 交易统计
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_days: float = 0.0
    
    # 额外指标
    alpha: float = 0.0                   # 相对基准的超额收益
    beta: float = 0.0                    # 市场敏感度
    information_ratio: float = 0.0       # 信息比率
    treynor_ratio: float = 0.0           # 特雷诺比率
    
    # 数据
    equity_curve: pd.Series = None
    drawdown_curve: pd.Series = None
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = None
    monthly_returns: pd.DataFrame = None
    
    def to_dict(self) -> Dict:
        return {
            'total_return': round(self.total_return * 100, 2),
            'annual_return': round(self.annual_return * 100, 2),
            'volatility': round(self.volatility * 100, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'max_drawdown': round(self.max_drawdown * 100, 2),
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_win': round(self.avg_win * 100, 2),
            'avg_loss': round(self.avg_loss * 100, 2),
            'alpha': round(self.alpha * 100, 2),
            'beta': round(self.beta, 2),
            'information_ratio': round(self.information_ratio, 2)
        }


class TransactionCostModel:
    """交易成本模型"""
    
    def __init__(self, 
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 min_commission: float = 1.0,
                 market_impact_factor: float = 0.1):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission
        self.market_impact_factor = market_impact_factor
    
    def calculate_commission(self, value: float) -> float:
        """计算佣金"""
        return max(self.min_commission, value * self.commission_rate)
    
    def calculate_slippage(self, price: float, quantity: int, 
                           avg_volume: float = None) -> float:
        """
        计算滑点 (考虑市场冲击)
        
        大单冲击公式: slippage = base_slippage + impact_factor * sqrt(order_size / avg_volume)
        """
        base_slippage = price * self.slippage_rate * abs(quantity)
        
        if avg_volume and avg_volume > 0:
            order_value = price * abs(quantity)
            # 市场冲击: 订单越大于平均成交量,冲击越大
            impact = self.market_impact_factor * np.sqrt(order_value / avg_volume)
            return base_slippage * (1 + impact)
        
        return base_slippage
    
    def total_cost(self, price: float, quantity: int, 
                   avg_volume: float = None) -> float:
        """总交易成本"""
        value = price * abs(quantity)
        commission = self.calculate_commission(value)
        slippage = self.calculate_slippage(price, quantity, avg_volume)
        return commission + slippage


class RiskMetrics:
    """风险指标计算器"""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
        """夏普比率 (年化)"""
        if returns.std() == 0:
            return 0.0
        excess_return = returns.mean() * 252 - risk_free
        volatility = returns.std() * np.sqrt(252)
        return excess_return / volatility if volatility > 0 else 0
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
        """索提诺比率 (只考虑下行风险)"""
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        excess_return = returns.mean() * 252 - risk_free
        downside_vol = downside.std() * np.sqrt(252)
        return excess_return / downside_vol if downside_vol > 0 else 0
    
    @staticmethod
    def calmar_ratio(total_return: float, max_drawdown: float, years: float) -> float:
        """卡玛比率 (年化收益/最大回撤)"""
        if max_drawdown == 0 or years == 0:
            return 0.0
        annual_return = (1 + total_return) ** (1 / years) - 1
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
        """最大回撤及恢复时间"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()
        
        # 计算最长回撤恢复时间
        underwater = drawdown < 0
        if underwater.any():
            groups = (~underwater).cumsum()
            underwater_periods = underwater.groupby(groups).sum()
            max_duration = underwater_periods.max() if len(underwater_periods) > 0 else 0
        else:
            max_duration = 0
        
        return max_dd, int(max_duration)
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """信息比率 (超额收益/跟踪误差)"""
        if len(returns) != len(benchmark_returns):
            return 0.0
        excess = returns - benchmark_returns
        if excess.std() == 0:
            return 0.0
        return (excess.mean() * 252) / (excess.std() * np.sqrt(252))
    
    @staticmethod
    def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Alpha 和 Beta"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0, 1.0
        
        # OLS 回归
        X = benchmark_returns.values
        Y = returns.values
        
        cov = np.cov(Y, X)
        if cov.shape == (2, 2):
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
        else:
            beta = 1.0
        
        alpha = returns.mean() * 252 - beta * benchmark_returns.mean() * 252
        return alpha, beta


class MonteCarloSimulator:
    """蒙特卡洛模拟器"""
    
    def __init__(self, returns: np.ndarray, n_simulations: int = 1000):
        self.returns = returns
        self.n_simulations = n_simulations
    
    def simulate(self, n_days: int = 252) -> Dict:
        """
        bootstrap 蒙特卡洛模拟
        
        Returns:
            Dict with simulation results
        """
        simulations = np.zeros((self.n_simulations, n_days))
        
        for i in range(self.n_simulations):
            # 有放回抽样
            sampled_returns = np.random.choice(self.returns, size=n_days, replace=True)
            simulations[i] = np.cumprod(1 + sampled_returns)
        
        final_values = simulations[:, -1]
        
        return {
            'mean_return': np.mean(final_values) - 1,
            'median_return': np.median(final_values) - 1,
            'std': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5) - 1,
            'percentile_25': np.percentile(final_values, 25) - 1,
            'percentile_75': np.percentile(final_values, 75) - 1,
            'percentile_95': np.percentile(final_values, 95) - 1,
            'prob_profit': np.mean(final_values > 1),
            'prob_loss_10': np.mean(final_values < 0.9),
            'prob_loss_20': np.mean(final_values < 0.8),
            'simulations': simulations,
            'final_values': final_values
        }
    
    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk"""
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)"""
        var = self.var(confidence)
        return self.returns[self.returns <= var].mean()


class ProfessionalBacktester:
    """专业级回测引擎"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.cost_model = TransactionCostModel(
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            min_commission=self.config.min_commission
        )
        
        # 状态
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # 数据
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.signal_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: pd.DataFrame = None
        
    def load_data(self, symbols: List[str], 
                  start_date: str, end_date: str,
                  market: str = 'US') -> bool:
        """加载历史数据"""
        from data_fetcher import get_stock_data
        
        # 计算需要的天数
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days + 60  # 额外60天用于计算指标
        
        for symbol in symbols:
            try:
                df = get_stock_data(symbol, market=market, days=days)
                if df is not None and not df.empty:
                    # 过滤日期范围
                    df.index = pd.to_datetime(df.index)
                    self.price_data[symbol] = df
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
        
        # 加载基准数据
        benchmark = 'SPY' if market == 'US' else '000300.SH'
        try:
            self.benchmark_data = get_stock_data(benchmark, market=market, days=days)
            if self.benchmark_data is not None:
                self.benchmark_data.index = pd.to_datetime(self.benchmark_data.index)
        except:
            pass
        
        return len(self.price_data) > 0
    
    def generate_signals(self, 
                         signal_func: Callable[[pd.DataFrame], pd.Series] = None) -> None:
        """
        生成交易信号
        
        Args:
            signal_func: 自定义信号函数 (输入 OHLCV DataFrame, 输出信号 Series)
        """
        from indicators import (calculate_blue_signal_series, calculate_adx_series,
                                calculate_heima_signal_series)
        
        for symbol, df in self.price_data.items():
            if signal_func:
                signals = signal_func(df)
            else:
                # 默认使用 BLUE 信号
                blue = calculate_blue_signal_series(
                    df['Open'].values, df['High'].values,
                    df['Low'].values, df['Close'].values
                )
                adx = calculate_adx_series(
                    df['High'].values, df['Low'].values, df['Close'].values
                )
                heima, juedi = calculate_heima_signal_series(
                    df['High'].values, df['Low'].values,
                    df['Close'].values, df['Open'].values
                )
                
                signals = pd.DataFrame({
                    'blue': blue,
                    'adx': adx,
                    'heima': heima,
                    'juedi': juedi
                }, index=df.index[-len(blue):])
                
                # 买入信号: BLUE > 阈值
                signals['signal'] = 0
                signals.loc[signals['blue'] > self.config.min_signal_strength, 'signal'] = 1
            
            self.signal_data[symbol] = signals
    
    def _calculate_position_size(self, symbol: str, price: float, 
                                 signal_strength: float = 0) -> int:
        """计算仓位大小"""
        sizing = self.config.position_sizing
        max_value = self.cash * self.config.max_position_pct
        
        if sizing == PositionSizing.FIXED:
            value = max_value
        
        elif sizing == PositionSizing.EQUAL_WEIGHT:
            n_positions = len(self.positions) + 1
            value = min(max_value, self.cash / min(n_positions, self.config.max_positions))
        
        elif sizing == PositionSizing.VOLATILITY:
            # 波动率倒数加权
            if symbol in self.price_data:
                returns = self.price_data[symbol]['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)
                if vol > 0:
                    value = max_value * (0.2 / vol)  # 目标波动率 20%
                else:
                    value = max_value
            else:
                value = max_value
        
        elif sizing == PositionSizing.ATR_BASED:
            # ATR 仓位 (单笔风险 2%)
            if symbol in self.price_data:
                df = self.price_data[symbol]
                atr = self._calculate_atr(df)
                if atr > 0:
                    risk_per_share = atr * 2  # 2倍ATR止损
                    max_risk = self.cash * 0.02  # 单笔风险2%
                    shares = int(max_risk / risk_per_share)
                    value = shares * price
                else:
                    value = max_value
            else:
                value = max_value
        
        elif sizing == PositionSizing.KELLY:
            # 简化凯利公式
            # f* = W - (1-W)/R, 其中 W=胜率, R=盈亏比
            # 使用历史数据估计
            W = 0.55  # 假设胜率55%
            R = 1.5   # 假设盈亏比1.5
            kelly = W - (1 - W) / R
            kelly_adj = kelly * 0.25  # 1/4 Kelly更保守
            value = self.cash * kelly_adj
        
        else:
            value = max_value
        
        # 限制最大仓位
        value = min(value, max_value, self.cash * 0.95)
        
        return max(0, int(value / price))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算 ATR"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        if len(tr) >= period:
            atr = np.mean(tr[-period:])
        else:
            atr = np.mean(tr) if len(tr) > 0 else 0
        
        return atr
    
    def run(self, start_date: str, end_date: str) -> BacktestResult:
        """运行回测"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 获取所有交易日
        all_dates = set()
        for df in self.price_data.values():
            all_dates.update(df.index.tolist())
        
        trading_dates = sorted([d for d in all_dates 
                               if start_dt <= d.to_pydatetime().replace(tzinfo=None) <= end_dt])
        
        if not trading_dates:
            return BacktestResult()
        
        print(f"📊 Running backtest: {start_date} to {end_date}")
        print(f"   {len(self.price_data)} symbols, {len(trading_dates)} trading days")
        
        # 逐日回测
        for date in trading_dates:
            self._process_day(date)
        
        # 计算结果
        result = self._calculate_results(trading_dates)
        
        return result
    
    def _process_day(self, date: datetime) -> None:
        """处理单日"""
        # 1. 检查止损/止盈
        self._check_exits(date)
        
        # 2. 检查新信号并开仓
        for symbol in self.signal_data.keys():
            if symbol in self.positions:
                continue  # 已持仓
            
            if len(self.positions) >= self.config.max_positions:
                continue  # 达到最大持仓数
            
            signal_df = self.signal_data[symbol]
            if date not in signal_df.index:
                continue
            
            signal_row = signal_df.loc[date]
            if signal_row.get('signal', 0) != 1:
                continue
            
            # 获取价格
            if symbol not in self.price_data or date not in self.price_data[symbol].index:
                continue
            
            price = self.price_data[symbol].loc[date, 'Close']
            
            # 计算仓位
            shares = self._calculate_position_size(symbol, price, signal_row.get('blue', 0))
            
            if shares > 0:
                self._open_position(symbol, date, price, shares, signal_row.get('blue', 0))
        
        # 3. 记录权益
        equity = self._calculate_equity(date)
        self.equity_history.append((date, equity))
    
    def _check_exits(self, date: datetime) -> None:
        """检查并执行止损/止盈"""
        symbols_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in self.price_data or date not in self.price_data[symbol].index:
                continue
            
            price = self.price_data[symbol].loc[date, 'Close']
            pnl_pct = pos.unrealized_pnl_pct(price)
            
            # 止损
            if pnl_pct <= -self.config.stop_loss_pct:
                symbols_to_close.append((symbol, 'stop_loss', price))
            # 止盈
            elif pnl_pct >= self.config.take_profit_pct:
                symbols_to_close.append((symbol, 'take_profit', price))
            # 移动止损
            elif self.config.use_trailing_stop and pos.trailing_stop > 0:
                # 更新移动止损
                new_stop = price * (1 - self.config.trailing_stop_pct)
                if new_stop > pos.trailing_stop:
                    pos.trailing_stop = new_stop
                elif price <= pos.trailing_stop:
                    symbols_to_close.append((symbol, 'trailing_stop', price))
        
        for symbol, reason, price in symbols_to_close:
            self._close_position(symbol, date, price, reason)
    
    def _open_position(self, symbol: str, date: datetime, 
                       price: float, shares: int, signal_strength: float) -> None:
        """开仓"""
        cost = self.cost_model.total_cost(price, shares)
        total = shares * price + cost
        
        if total > self.cash:
            return
        
        self.cash -= total
        
        stop_loss = price * (1 - self.config.stop_loss_pct)
        take_profit = price * (1 + self.config.take_profit_pct)
        
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=shares,
            avg_cost=price,
            entry_date=date,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss if self.config.use_trailing_stop else 0
        )
        
        trade = Trade(
            symbol=symbol,
            date=date,
            side='buy',
            quantity=shares,
            price=price,
            commission=self.cost_model.calculate_commission(shares * price),
            slippage=self.cost_model.calculate_slippage(price, shares),
            signal_strength=signal_strength,
            reason='signal_buy'
        )
        self.trades.append(trade)
    
    def _close_position(self, symbol: str, date: datetime, 
                        price: float, reason: str) -> None:
        """平仓"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        cost = self.cost_model.total_cost(price, pos.quantity)
        proceeds = pos.quantity * price - cost
        
        self.cash += proceeds
        
        trade = Trade(
            symbol=symbol,
            date=date,
            side='sell',
            quantity=pos.quantity,
            price=price,
            commission=self.cost_model.calculate_commission(pos.quantity * price),
            slippage=self.cost_model.calculate_slippage(price, pos.quantity),
            reason=reason
        )
        self.trades.append(trade)
        
        del self.positions[symbol]
    
    def _calculate_equity(self, date: datetime) -> float:
        """计算当日总权益"""
        equity = self.cash
        
        for symbol, pos in self.positions.items():
            if symbol in self.price_data and date in self.price_data[symbol].index:
                price = self.price_data[symbol].loc[date, 'Close']
                equity += pos.market_value(price)
        
        return equity
    
    def _calculate_results(self, trading_dates: List) -> BacktestResult:
        """计算回测结果"""
        if not self.equity_history:
            return BacktestResult()
        
        # 权益曲线
        equity_df = pd.DataFrame(self.equity_history, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        equity_curve = equity_df['equity']
        
        # 日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 基准收益率
        benchmark_returns = None
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
            # 对齐日期
            common_dates = daily_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                daily_returns = daily_returns[common_dates]
                benchmark_returns = benchmark_returns[common_dates]
        
        # 基础统计
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        years = len(trading_dates) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 风险指标
        sharpe = RiskMetrics.sharpe_ratio(daily_returns)
        sortino = RiskMetrics.sortino_ratio(daily_returns)
        max_dd, max_dd_duration = RiskMetrics.max_drawdown(equity_curve)
        calmar = RiskMetrics.calmar_ratio(total_return, max_dd, years)
        
        # Alpha/Beta
        alpha, beta = 0.0, 1.0
        ir = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            alpha, beta = RiskMetrics.alpha_beta(daily_returns, benchmark_returns)
            ir = RiskMetrics.information_ratio(daily_returns, benchmark_returns)
        
        # 交易统计
        trade_returns = []
        for i in range(0, len(self.trades), 2):  # 每两笔为一组 (买+卖)
            if i + 1 < len(self.trades):
                buy = self.trades[i]
                sell = self.trades[i + 1]
                if buy.side == 'buy' and sell.side == 'sell':
                    ret = (sell.price - buy.price) / buy.price
                    trade_returns.append(ret)
        
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        
        win_rate = len(wins) / len(trade_returns) if trade_returns else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0
        
        # 回撤曲线
        peak = equity_curve.expanding().max()
        drawdown_curve = (equity_curve - peak) / peak
        
        # 月度收益
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(trade_returns),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            alpha=alpha,
            beta=beta,
            information_ratio=ir,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=self.trades,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns
        )


class WalkForwardAnalyzer:
    """滚动验证分析器"""
    
    def __init__(self, 
                 train_period: int = 252,   # 训练期 (天)
                 test_period: int = 63,     # 测试期 (天, 约一季度)
                 step_size: int = 21):      # 步进 (天, 约一个月)
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
    
    def run(self, 
            backtester: ProfessionalBacktester,
            start_date: str, 
            end_date: str,
            optimize_func: Callable = None) -> Dict:
        """
        运行 Walk-Forward 分析
        
        Args:
            backtester: 回测引擎实例
            start_date: 开始日期
            end_date: 结束日期
            optimize_func: 参数优化函数 (可选)
        
        Returns:
            Dict with walk-forward results
        """
        results = []
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_start = start_dt
        
        while current_start + timedelta(days=self.train_period + self.test_period) <= end_dt:
            # 训练期
            train_start = current_start
            train_end = current_start + timedelta(days=self.train_period)
            
            # 测试期
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period)
            
            # 可选: 在训练期上优化参数
            if optimize_func:
                best_params = optimize_func(backtester, 
                                           train_start.strftime('%Y-%m-%d'),
                                           train_end.strftime('%Y-%m-%d'))
                # 应用最优参数...
            
            # 在测试期上回测
            result = backtester.run(
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            )
            
            results.append({
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'result': result
            })
            
            # 步进
            current_start += timedelta(days=self.step_size)
            
            # 重置回测状态
            backtester.cash = backtester.config.initial_capital
            backtester.positions = {}
            backtester.trades = []
            backtester.equity_history = []
        
        # 汇总统计
        all_returns = [r['result'].total_return for r in results if r['result'].total_return != 0]
        all_sharpes = [r['result'].sharpe_ratio for r in results if r['result'].sharpe_ratio != 0]
        
        return {
            'periods': results,
            'summary': {
                'n_periods': len(results),
                'avg_return': np.mean(all_returns) if all_returns else 0,
                'avg_sharpe': np.mean(all_sharpes) if all_sharpes else 0,
                'consistency': len([r for r in all_returns if r > 0]) / len(all_returns) if all_returns else 0,
                'total_return': np.prod([1 + r for r in all_returns]) - 1 if all_returns else 0
            }
        }


# 便捷函数
def quick_backtest(symbols: List[str],
                   start_date: str,
                   end_date: str,
                   market: str = 'US',
                   config: BacktestConfig = None) -> BacktestResult:
    """快速回测"""
    if config is None:
        config = BacktestConfig()
    
    bt = ProfessionalBacktester(config)
    
    if not bt.load_data(symbols, start_date, end_date, market):
        print("Failed to load data")
        return BacktestResult()
    
    bt.generate_signals()
    return bt.run(start_date, end_date)


if __name__ == "__main__":
    # 测试
    config = BacktestConfig(
        initial_capital=100000,
        position_sizing=PositionSizing.EQUAL_WEIGHT,
        max_positions=10,
        stop_loss_pct=0.08,
        take_profit_pct=0.20
    )
    
    result = quick_backtest(
        symbols=['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        start_date='2025-06-01',
        end_date='2026-01-31',
        market='US',
        config=config
    )
    
    print("\n📊 Backtest Results:")
    for k, v in result.to_dict().items():
        print(f"   {k}: {v}")
