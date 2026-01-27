#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风控模块 - 仓位管理、止损、风险监控
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class RiskManager:
    """风险管理核心类"""
    
    def __init__(self, 
                 total_capital: float = 100000,
                 max_position_pct: float = 0.1,  # 单仓最大10%
                 max_drawdown_pct: float = 0.2,  # 最大回撤20%
                 risk_per_trade_pct: float = 0.02):  # 每笔风险2%
        """
        初始化风控参数
        
        Args:
            total_capital: 总资金
            max_position_pct: 单只股票最大仓位比例
            max_drawdown_pct: 最大允许回撤
            risk_per_trade_pct: 每笔交易风险比例
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        
        self.current_equity = total_capital
        self.peak_equity = total_capital
        self.positions = {}
    
    def calc_position_size_fixed(self, price: float, pct: float = None) -> Dict:
        """
        固定比例仓位计算
        
        Args:
            price: 股票价格
            pct: 仓位比例 (默认使用max_position_pct)
        
        Returns:
            仓位信息
        """
        pct = pct or self.max_position_pct
        position_value = self.current_equity * pct
        shares = int(position_value / price)
        
        return {
            'shares': shares,
            'position_value': round(shares * price, 2),
            'position_pct': round(shares * price / self.current_equity * 100, 2),
            'method': 'fixed_pct'
        }
    
    def calc_position_size_kelly(self, 
                                  price: float, 
                                  win_rate: float,
                                  avg_win: float,
                                  avg_loss: float,
                                  kelly_fraction: float = 0.25) -> Dict:
        """
        凯利公式仓位计算
        
        Kelly% = W - (1-W)/R
        W = 胜率
        R = 盈亏比 (平均盈利/平均亏损)
        
        Args:
            price: 股票价格
            win_rate: 历史胜率 (0-1)
            avg_win: 平均盈利比例
            avg_loss: 平均亏损比例
            kelly_fraction: Kelly 分数 (建议使用1/4或1/2)
        
        Returns:
            仓位信息
        """
        if avg_loss == 0:
            return self.calc_position_size_fixed(price)
        
        R = avg_win / abs(avg_loss)
        kelly_pct = win_rate - (1 - win_rate) / R
        
        # 限制Kelly比例
        kelly_pct = max(0, min(kelly_pct, self.max_position_pct))
        
        # 使用分数Kelly减少风险
        adjusted_pct = kelly_pct * kelly_fraction
        
        position_value = self.current_equity * adjusted_pct
        shares = int(position_value / price)
        
        return {
            'shares': shares,
            'position_value': round(shares * price, 2),
            'position_pct': round(adjusted_pct * 100, 2),
            'kelly_raw': round(kelly_pct * 100, 2),
            'kelly_adjusted': round(adjusted_pct * 100, 2),
            'method': 'kelly'
        }
    
    def calc_position_size_volatility(self, 
                                       price: float,
                                       atr: float,
                                       atr_multiplier: float = 2) -> Dict:
        """
        基于波动率的仓位计算 (ATR方法)
        
        仓位 = 风险资金 / (ATR * 乘数)
        
        Args:
            price: 股票价格
            atr: 平均真实波幅
            atr_multiplier: ATR乘数 (止损距离)
        
        Returns:
            仓位信息
        """
        risk_per_share = atr * atr_multiplier
        risk_capital = self.current_equity * self.risk_per_trade_pct
        
        if risk_per_share == 0:
            return self.calc_position_size_fixed(price)
        
        shares = int(risk_capital / risk_per_share)
        position_value = shares * price
        
        # 确保不超过最大仓位
        max_value = self.current_equity * self.max_position_pct
        if position_value > max_value:
            shares = int(max_value / price)
            position_value = shares * price
        
        return {
            'shares': shares,
            'position_value': round(position_value, 2),
            'position_pct': round(position_value / self.current_equity * 100, 2),
            'stop_loss_price': round(price - atr * atr_multiplier, 2),
            'method': 'atr_volatility'
        }
    
    def set_stop_loss(self, 
                      entry_price: float, 
                      method: str = 'percent',
                      **kwargs) -> Dict:
        """
        计算止损价
        
        Args:
            entry_price: 入场价格
            method: 止损方法 (percent/atr/support)
            **kwargs: 方法参数
        
        Returns:
            止损信息
        """
        if method == 'percent':
            pct = kwargs.get('pct', 0.05)  # 默认5%
            stop_price = entry_price * (1 - pct)
            
        elif method == 'atr':
            atr = kwargs.get('atr', entry_price * 0.02)
            multiplier = kwargs.get('multiplier', 2)
            stop_price = entry_price - atr * multiplier
            
        elif method == 'support':
            support = kwargs.get('support', entry_price * 0.95)
            buffer = kwargs.get('buffer', 0.01)
            stop_price = support * (1 - buffer)
            
        else:
            stop_price = entry_price * 0.95
        
        return {
            'entry_price': entry_price,
            'stop_price': round(stop_price, 2),
            'stop_pct': round((entry_price - stop_price) / entry_price * 100, 2),
            'method': method
        }
    
    def set_take_profit(self,
                        entry_price: float,
                        stop_price: float,
                        risk_reward_ratio: float = 2) -> Dict:
        """
        计算止盈价 (基于风险回报比)
        
        Args:
            entry_price: 入场价格
            stop_price: 止损价格
            risk_reward_ratio: 风险回报比
        
        Returns:
            止盈信息
        """
        risk_per_share = entry_price - stop_price
        reward = risk_per_share * risk_reward_ratio
        take_profit = entry_price + reward
        
        return {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'take_profit': round(take_profit, 2),
            'risk': round(risk_per_share, 2),
            'reward': round(reward, 2),
            'rr_ratio': risk_reward_ratio
        }
    
    def check_drawdown(self, current_value: float) -> Dict:
        """
        检查回撤状态
        
        Args:
            current_value: 当前权益
        
        Returns:
            回撤状态
        """
        self.current_equity = current_value
        self.peak_equity = max(self.peak_equity, current_value)
        
        drawdown = (self.peak_equity - current_value) / self.peak_equity
        
        return {
            'current_equity': current_value,
            'peak_equity': self.peak_equity,
            'drawdown_pct': round(drawdown * 100, 2),
            'max_allowed_pct': self.max_drawdown_pct * 100,
            'is_exceeded': drawdown > self.max_drawdown_pct,
            'action': 'STOP_TRADING' if drawdown > self.max_drawdown_pct else 'CONTINUE'
        }
    
    def portfolio_risk_check(self, positions: List[Dict]) -> Dict:
        """
        检查组合风险
        
        Args:
            positions: 持仓列表 [{'symbol': '', 'value': '', 'sector': ''}, ...]
        
        Returns:
            风险检查结果
        """
        if not positions:
            return {'status': 'OK', 'warnings': []}
        
        warnings = []
        
        total_value = sum(p.get('value', 0) for p in positions)
        
        # 检查单仓集中度
        for pos in positions:
            pos_pct = pos.get('value', 0) / total_value if total_value > 0 else 0
            if pos_pct > self.max_position_pct:
                warnings.append(f"{pos.get('symbol')}: 仓位 {pos_pct*100:.1f}% 超过限制 {self.max_position_pct*100}%")
        
        # 检查行业集中度
        sector_exposure = {}
        for pos in positions:
            sector = pos.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get('value', 0)
        
        for sector, value in sector_exposure.items():
            sector_pct = value / total_value if total_value > 0 else 0
            if sector_pct > 0.3:  # 行业集中度超过30%
                warnings.append(f"行业 {sector}: 集中度 {sector_pct*100:.1f}% 过高")
        
        return {
            'status': 'WARNING' if warnings else 'OK',
            'total_exposure': round(total_value, 2),
            'num_positions': len(positions),
            'warnings': warnings
        }
    
    def recommend_position(self, 
                           symbol: str,
                           price: float,
                           win_rate: float = 0.5,
                           avg_win: float = 5,
                           avg_loss: float = 3,
                           atr: float = None) -> Dict:
        """
        综合推荐仓位和止损
        
        Args:
            symbol: 股票代码
            price: 当前价格
            win_rate: 历史胜率
            avg_win: 平均盈利%
            avg_loss: 平均亏损%
            atr: ATR (可选)
        
        Returns:
            完整交易建议
        """
        # 计算仓位 (使用Kelly)
        position = self.calc_position_size_kelly(price, win_rate, avg_win, avg_loss)
        
        # 设置止损
        if atr:
            stop = self.set_stop_loss(price, method='atr', atr=atr)
        else:
            stop = self.set_stop_loss(price, method='percent', pct=avg_loss/100)
        
        # 设置止盈
        tp = self.set_take_profit(price, stop['stop_price'], risk_reward_ratio=2)
        
        return {
            'symbol': symbol,
            'entry_price': price,
            'shares': position['shares'],
            'position_value': position['position_value'],
            'position_pct': position['position_pct'],
            'stop_loss': stop['stop_price'],
            'take_profit': tp['take_profit'],
            'risk_reward': tp['rr_ratio'],
            'method': position['method']
        }


if __name__ == "__main__":
    # 测试风控模块
    rm = RiskManager(total_capital=100000)
    
    # 测试仓位计算
    pos = rm.calc_position_size_kelly(
        price=50, 
        win_rate=0.55, 
        avg_win=8, 
        avg_loss=4
    )
    print(f"Kelly Position: {pos}")
    
    # 测试止损
    stop = rm.set_stop_loss(entry_price=50, method='percent', pct=0.05)
    print(f"Stop Loss: {stop}")
    
    # 测试完整建议
    rec = rm.recommend_position(
        symbol='AAPL',
        price=180,
        win_rate=0.55,
        avg_win=8,
        avg_loss=4
    )
    print(f"Recommendation: {rec}")
