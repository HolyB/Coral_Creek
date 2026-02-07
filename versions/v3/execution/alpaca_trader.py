#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpaca Paper Trading Integration - 模拟盘交易集成
==================================================

功能:
- 模拟盘交易 (Paper Trading)
- 实盘交易 (Live Trading) - 需要真实资金
- 订单管理 (下单/撤单/查询)
- 持仓管理
- 账户信息

使用前需要:
1. 注册 Alpaca 账号: https://alpaca.markets/
2. 获取 API Key 和 Secret
3. 配置环境变量:
   ALPACA_API_KEY=your_api_key
   ALPACA_SECRET_KEY=your_secret_key
   ALPACA_PAPER=true  (模拟盘) 或 false (实盘)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# 尝试导入 alpaca-trade-api
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, 
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import (
        OrderSide, 
        TimeInForce, 
        OrderStatus,
        QueryOrderStatus
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_SDK_AVAILABLE = True
except ImportError:
    ALPACA_SDK_AVAILABLE = False
    TradingClient = None
    OrderSide = None


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradeOrder:
    """交易订单"""
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = "market"
    limit_price: float = None
    stop_price: float = None
    time_in_force: str = "day"  # day, gtc, ioc, fok


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str


@dataclass
class AccountInfo:
    """账户信息"""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str
    is_paper: bool


class AlpacaTrader:
    """
    Alpaca 交易客户端
    
    支持模拟盘和实盘交易
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True,
                 enable_hard_risk_guards: bool = True,
                 max_single_position_pct: float = 0.20,
                 max_daily_loss_pct: float = 0.03,
                 max_portfolio_drawdown_pct: float = 0.15):
        """
        初始化交易客户端
        
        Args:
            api_key: Alpaca API Key (或从环境变量 ALPACA_API_KEY 获取)
            secret_key: Alpaca Secret Key (或从环境变量 ALPACA_SECRET_KEY 获取)
            paper: 是否使用模拟盘 (默认 True)
        """
        if not ALPACA_SDK_AVAILABLE:
            raise ImportError(
                "请安装 alpaca-py: pip install alpaca-py\n"
                "文档: https://alpaca.markets/docs/python-sdk/"
            )
        
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')
        self.paper = paper if paper is not None else os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'

        # 风控开关（可由环境变量覆盖）
        self.enable_hard_risk_guards = (
            os.environ.get('ALPACA_ENABLE_HARD_RISK_GUARDS', str(enable_hard_risk_guards)).lower() == 'true'
        )
        self.max_single_position_pct = float(
            os.environ.get('ALPACA_MAX_SINGLE_POSITION_PCT', max_single_position_pct)
        )
        self.max_daily_loss_pct = float(
            os.environ.get('ALPACA_MAX_DAILY_LOSS_PCT', max_daily_loss_pct)
        )
        self.max_portfolio_drawdown_pct = float(
            os.environ.get('ALPACA_MAX_PORTFOLIO_DRAWDOWN_PCT', max_portfolio_drawdown_pct)
        )

        # 运行态峰值净值（用于回撤风控）
        self._peak_equity = None
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "请设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY 环境变量\n"
                "或在初始化时传入 api_key 和 secret_key"
            )
        
        # 初始化客户端
        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )
        
        # 数据客户端 (用于获取实时价格)
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

    def _update_peak_equity(self, equity: float) -> None:
        """更新会话内峰值净值"""
        if equity <= 0:
            return
        if self._peak_equity is None:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)

    def _validate_buy_order(self, symbol: str, qty: float, ref_price: Optional[float] = None) -> None:
        """
        买单硬风控校验。
        触发风险限制时抛出 ValueError，调用方直接显示错误即可。
        """
        if not self.enable_hard_risk_guards:
            return

        if qty <= 0:
            raise ValueError("风控拦截：下单数量必须大于 0")

        account = self.client.get_account()
        equity = float(account.equity)
        if equity <= 0:
            raise ValueError("风控拦截：账户净值异常，禁止开新仓")

        self._update_peak_equity(equity)

        # 1) 组合最大回撤限制（基于会话内峰值）
        if self._peak_equity and self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.max_portfolio_drawdown_pct:
                raise ValueError(
                    f"风控拦截：组合回撤 {drawdown:.2%} 超过阈值 {self.max_portfolio_drawdown_pct:.2%}，禁止开新仓"
                )

        # 2) 当日亏损限制（使用 Alpaca last_equity）
        last_equity_raw = getattr(account, 'last_equity', None)
        try:
            last_equity = float(last_equity_raw) if last_equity_raw is not None else 0.0
        except (TypeError, ValueError):
            last_equity = 0.0

        if last_equity > 0:
            daily_loss = (last_equity - equity) / last_equity
            if daily_loss >= self.max_daily_loss_pct:
                raise ValueError(
                    f"风控拦截：当日亏损 {daily_loss:.2%} 超过阈值 {self.max_daily_loss_pct:.2%}，禁止开新仓"
                )

        # 3) 单票最大仓位限制
        # 行情接口偶发抖动时，跳过该项校验，但保留日亏/回撤等硬风控，避免误伤可交易时段。
        price = ref_price if ref_price and ref_price > 0 else self.get_latest_price(symbol)
        if price <= 0:
            return

        new_order_value = float(qty) * float(price)
        current_position_value = 0.0
        try:
            pos = self.client.get_open_position(symbol)
            current_position_value = float(pos.market_value)
        except Exception:
            current_position_value = 0.0

        post_trade_position_value = current_position_value + new_order_value
        position_limit_value = equity * self.max_single_position_pct
        if post_trade_position_value > position_limit_value:
            raise ValueError(
                f"风控拦截：{symbol} 下单后仓位 ${post_trade_position_value:,.2f} 超过上限 ${position_limit_value:,.2f}"
            )
    
    # ============================================================================
    # 账户信息
    # ============================================================================
    
    def get_account(self) -> AccountInfo:
        """获取账户信息"""
        account = self.client.get_account()
        
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            status=account.status.value,
            is_paper=self.paper
        )
    
    def get_positions(self) -> List[Position]:
        """获取所有持仓"""
        positions = self.client.get_all_positions()
        
        result = []
        for pos in positions:
            result.append(Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_plpc=float(pos.unrealized_plpc) * 100,
                side=pos.side.value
            ))
        
        return result
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取单个持仓"""
        try:
            pos = self.client.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_plpc=float(pos.unrealized_plpc) * 100,
                side=pos.side.value
            )
        except Exception:
            return None
    
    # ============================================================================
    # 下单
    # ============================================================================
    
    def buy_market(self, symbol: str, qty: float, time_in_force: str = "day") -> Dict:
        """
        市价买入
        
        Args:
            symbol: 股票代码
            qty: 买入数量
            time_in_force: 有效期 (day, gtc, ioc, fok)
        
        Returns:
            订单信息
        """
        self._validate_buy_order(symbol, qty)
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)
        
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=tif
        )
        
        order = self.client.submit_order(order_request)
        return self._order_to_dict(order)
    
    def sell_market(self, symbol: str, qty: float, time_in_force: str = "day") -> Dict:
        """市价卖出"""
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)
        
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=tif
        )
        
        order = self.client.submit_order(order_request)
        return self._order_to_dict(order)
    
    def buy_limit(self, symbol: str, qty: float, limit_price: float, 
                  time_in_force: str = "day") -> Dict:
        """限价买入"""
        self._validate_buy_order(symbol, qty, ref_price=limit_price)
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)
        
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=tif,
            limit_price=limit_price
        )
        
        order = self.client.submit_order(order_request)
        return self._order_to_dict(order)
    
    def sell_limit(self, symbol: str, qty: float, limit_price: float,
                   time_in_force: str = "day") -> Dict:
        """限价卖出"""
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)
        
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=tif,
            limit_price=limit_price
        )
        
        order = self.client.submit_order(order_request)
        return self._order_to_dict(order)
    
    def buy_stop(self, symbol: str, qty: float, stop_price: float,
                 time_in_force: str = "day") -> Dict:
        """止损买入 (突破买入)"""
        self._validate_buy_order(symbol, qty, ref_price=stop_price)
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)
        
        order_request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=tif,
            stop_price=stop_price
        )
        
        order = self.client.submit_order(order_request)
        return self._order_to_dict(order)
    
    def sell_stop(self, symbol: str, qty: float, stop_price: float,
                  time_in_force: str = "day") -> Dict:
        """止损卖出"""
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)
        
        order_request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=tif,
            stop_price=stop_price
        )
        
        order = self.client.submit_order(order_request)
        return self._order_to_dict(order)
    
    def close_position(self, symbol: str) -> Dict:
        """平仓 (卖出全部持仓)"""
        order = self.client.close_position(symbol)
        return self._order_to_dict(order)
    
    def close_all_positions(self) -> List[Dict]:
        """清仓 (卖出所有持仓)"""
        orders = self.client.close_all_positions(cancel_orders=True)
        return [self._order_to_dict(o) for o in orders]
    
    # ============================================================================
    # 订单管理
    # ============================================================================
    
    def get_orders(self, status: str = "open") -> List[Dict]:
        """
        获取订单列表
        
        Args:
            status: 订单状态 (open, closed, all)
        """
        if status == "open":
            query_status = QueryOrderStatus.OPEN
        elif status == "closed":
            query_status = QueryOrderStatus.CLOSED
        else:
            query_status = QueryOrderStatus.ALL
        
        request = GetOrdersRequest(status=query_status)
        orders = self.client.get_orders(request)
        
        return [self._order_to_dict(o) for o in orders]
    
    def get_order(self, order_id: str) -> Dict:
        """获取单个订单"""
        order = self.client.get_order_by_id(order_id)
        return self._order_to_dict(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False
    
    def cancel_all_orders(self) -> bool:
        """撤销所有订单"""
        try:
            self.client.cancel_orders()
            return True
        except Exception:
            return False
    
    # ============================================================================
    # 市场数据
    # ============================================================================
    
    def get_latest_price(self, symbol: str) -> float:
        """获取最新价格"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(days=1)
            )
            bars = self.data_client.get_stock_bars(request)
            if symbol in bars and len(bars[symbol]) > 0:
                return float(bars[symbol][-1].close)
        except Exception:
            pass

        # 兜底: 分钟K线拿不到时尝试 latest trade
        try:
            latest_trade_req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            latest_trade = self.data_client.get_stock_latest_trade(latest_trade_req)

            trade_obj = None
            if isinstance(latest_trade, dict):
                trade_obj = latest_trade.get(symbol)
            else:
                data = getattr(latest_trade, "data", None)
                if isinstance(data, dict):
                    trade_obj = data.get(symbol)
                if trade_obj is None:
                    try:
                        trade_obj = latest_trade[symbol]
                    except Exception:
                        trade_obj = None

            if trade_obj is not None:
                price = float(getattr(trade_obj, "price", 0.0) or 0.0)
                if price > 0:
                    return price
        except Exception:
            pass

        return 0.0
    
    def is_market_open(self) -> bool:
        """检查市场是否开盘"""
        clock = self.client.get_clock()
        return clock.is_open
    
    def get_market_hours(self) -> Dict:
        """获取市场开闭盘时间"""
        clock = self.client.get_clock()
        return {
            'is_open': clock.is_open,
            'next_open': clock.next_open.isoformat() if clock.next_open else None,
            'next_close': clock.next_close.isoformat() if clock.next_close else None,
        }
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _order_to_dict(self, order) -> Dict:
        """将订单对象转换为字典"""
        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'side': order.side.value,
            'qty': float(order.qty) if order.qty else None,
            'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
            'type': order.type.value,
            'status': order.status.value,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'created_at': order.created_at.isoformat() if order.created_at else None,
            'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
            'filled_at': order.filled_at.isoformat() if order.filled_at else None,
        }


# ============================================================================
# 信号自动交易
# ============================================================================

class SignalTrader:
    """
    信号自动交易器
    
    将系统信号转换为实际交易
    """
    
    def __init__(self, trader: AlpacaTrader, 
                 max_position_pct: float = 0.1,
                 stop_loss_pct: float = 0.08):
        """
        Args:
            trader: AlpacaTrader 实例
            max_position_pct: 单只股票最大仓位比例 (默认 10%)
            stop_loss_pct: 止损比例 (默认 8%)
        """
        self.trader = trader
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.trade_log = []
    
    def execute_buy_signal(self, symbol: str, signal_reason: str = "") -> Dict:
        """
        执行买入信号
        
        Args:
            symbol: 股票代码
            signal_reason: 信号原因
        
        Returns:
            执行结果
        """
        account = self.trader.get_account()
        
        # 检查是否已有持仓
        position = self.trader.get_position(symbol)
        if position and position.qty > 0:
            return {
                'success': False,
                'message': f'{symbol} 已持仓 {position.qty} 股',
                'symbol': symbol
            }
        
        # 计算买入数量
        max_amount = account.buying_power * self.max_position_pct
        current_price = self.trader.get_latest_price(symbol)
        
        if current_price <= 0:
            return {
                'success': False,
                'message': f'无法获取 {symbol} 价格',
                'symbol': symbol
            }
        
        qty = int(max_amount / current_price)
        if qty <= 0:
            return {
                'success': False,
                'message': f'资金不足购买 {symbol}',
                'symbol': symbol
            }
        
        # 下单
        try:
            order = self.trader.buy_market(symbol, qty)
            
            # 设置止损单
            stop_price = round(current_price * (1 - self.stop_loss_pct), 2)
            self.trader.sell_stop(symbol, qty, stop_price, "gtc")
            
            result = {
                'success': True,
                'message': f'买入 {symbol} {qty} 股 @ ${current_price:.2f}',
                'symbol': symbol,
                'qty': qty,
                'price': current_price,
                'order_id': order['id'],
                'stop_price': stop_price,
                'reason': signal_reason,
                'timestamp': datetime.now().isoformat()
            }
            
            self.trade_log.append(result)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f'下单失败: {str(e)}',
                'symbol': symbol
            }
    
    def execute_sell_signal(self, symbol: str, signal_reason: str = "") -> Dict:
        """执行卖出信号"""
        position = self.trader.get_position(symbol)
        
        if not position or position.qty <= 0:
            return {
                'success': False,
                'message': f'未持有 {symbol}',
                'symbol': symbol
            }
        
        try:
            # 先撤销该股票的所有挂单
            orders = self.trader.get_orders("open")
            for order in orders:
                if order['symbol'] == symbol:
                    self.trader.cancel_order(order['id'])
            
            # 平仓
            order = self.trader.close_position(symbol)
            
            result = {
                'success': True,
                'message': f'卖出 {symbol} {position.qty} 股',
                'symbol': symbol,
                'qty': position.qty,
                'avg_entry': position.avg_entry_price,
                'current_price': position.current_price,
                'pnl': position.unrealized_pl,
                'pnl_pct': position.unrealized_plpc,
                'order_id': order['id'],
                'reason': signal_reason,
                'timestamp': datetime.now().isoformat()
            }
            
            self.trade_log.append(result)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f'平仓失败: {str(e)}',
                'symbol': symbol
            }
    
    def get_portfolio_summary(self) -> Dict:
        """获取持仓汇总"""
        account = self.trader.get_account()
        positions = self.trader.get_positions()
        
        return {
            'account': {
                'equity': account.equity,
                'cash': account.cash,
                'buying_power': account.buying_power,
                'is_paper': account.is_paper,
            },
            'positions': [
                {
                    'symbol': p.symbol,
                    'qty': p.qty,
                    'avg_entry': p.avg_entry_price,
                    'current_price': p.current_price,
                    'market_value': p.market_value,
                    'pnl': p.unrealized_pl,
                    'pnl_pct': p.unrealized_plpc,
                }
                for p in positions
            ],
            'total_pnl': sum(p.unrealized_pl for p in positions),
            'position_count': len(positions)
        }
    
    def get_trade_history(self) -> List[Dict]:
        """获取交易历史"""
        return self.trade_log


# ============================================================================
# 测试和演示
# ============================================================================

def check_alpaca_available() -> bool:
    """检查 Alpaca SDK 是否可用"""
    return ALPACA_SDK_AVAILABLE


def setup_instructions() -> str:
    """返回设置说明"""
    return """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    Alpaca 模拟盘交易设置指南                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  1. 注册 Alpaca 账号 (免费):                                              ║
║     https://alpaca.markets/                                               ║
║                                                                           ║
║  2. 获取 API Keys:                                                        ║
║     登录后 -> Paper Trading -> Your API Keys                              ║
║                                                                           ║
║  3. 安装 SDK:                                                             ║
║     pip install alpaca-py                                                 ║
║                                                                           ║
║  4. 配置环境变量 (在 .env 文件中添加):                                     ║
║     ALPACA_API_KEY=your_api_key_here                                      ║
║     ALPACA_SECRET_KEY=your_secret_key_here                                ║
║     ALPACA_PAPER=true                                                     ║
║                                                                           ║
║  5. 测试连接:                                                             ║
║     python execution/alpaca_trader.py                                     ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  ⚠️ 注意: Paper Trading 使用虚拟资金, 不涉及真实交易                        ║
║          初始虚拟资金: $100,000                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(setup_instructions())
    
    if not ALPACA_SDK_AVAILABLE:
        print("❌ 请先安装: pip install alpaca-py")
        sys.exit(1)
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ 请设置环境变量 ALPACA_API_KEY 和 ALPACA_SECRET_KEY")
        sys.exit(1)
    
    print("\n🔌 连接 Alpaca Paper Trading...")
    
    try:
        trader = AlpacaTrader(paper=True)
        
        # 账户信息
        account = trader.get_account()
        print(f"\n✅ 连接成功!")
        print(f"   账户类型: {'模拟盘' if account.is_paper else '实盘'}")
        print(f"   账户余额: ${account.equity:,.2f}")
        print(f"   可用资金: ${account.cash:,.2f}")
        print(f"   购买力: ${account.buying_power:,.2f}")
        
        # 持仓
        positions = trader.get_positions()
        print(f"\n📊 当前持仓: {len(positions)} 只")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.qty}股 @ ${pos.avg_entry_price:.2f} "
                  f"-> ${pos.current_price:.2f} ({pos.unrealized_plpc:+.2f}%)")
        
        # 市场状态
        market = trader.get_market_hours()
        print(f"\n🕐 市场状态: {'开盘中' if market['is_open'] else '已休市'}")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
