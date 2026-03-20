"""
仓位管理模块
Position Sizing & Risk Budgeting

包含：
- 固定比例仓位
- 凯利公式
- 风险预算
- 波动率目标
- 最大回撤控制
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PositionLimit:
    """仓位限制配置"""
    max_single_position: float = 0.20      # 单只股票最大仓位
    max_sector_exposure: float = 0.40      # 单一行业最大敞口
    max_correlation_pair: float = 0.80     # 高相关性资产对的限制
    min_positions: int = 5                  # 最少持仓数量
    max_positions: int = 20                 # 最多持仓数量


class PositionSizer:
    """仓位计算器"""
    
    def __init__(self, 
                 total_capital: float,
                 risk_per_trade: float = 0.02,
                 limits: Optional[PositionLimit] = None):
        """
        Args:
            total_capital: 总资金
            risk_per_trade: 每笔交易风险比例 (默认 2%)
            limits: 仓位限制配置
        """
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade
        self.limits = limits or PositionLimit()
    
    def fixed_fractional(self, 
                         entry_price: float, 
                         stop_loss: float) -> Dict:
        """
        固定比例仓位法
        基于止损距离计算仓位
        
        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
        
        Returns:
            {'shares': 股数, 'position_value': 仓位金额, 'risk_amount': 风险金额}
        """
        risk_amount = self.total_capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {'shares': 0, 'position_value': 0, 'risk_amount': 0, 'error': '止损价格等于入场价'}
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # 检查是否超过单一持仓限制
        max_position = self.total_capital * self.limits.max_single_position
        if position_value > max_position:
            shares = int(max_position / entry_price)
            position_value = shares * entry_price
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_pct': position_value / self.total_capital,
            'risk_amount': shares * risk_per_share,
            'risk_pct': (shares * risk_per_share) / self.total_capital
        }
    
    def kelly_criterion(self, 
                        win_rate: float, 
                        avg_win: float, 
                        avg_loss: float,
                        fraction: float = 0.5) -> float:
        """
        凯利公式计算最优仓位
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损 (正值)
            fraction: 凯利比例的保守系数 (0.5 = 半凯利)
        
        Returns:
            建议仓位比例
        """
        if avg_loss == 0:
            return 0
        
        # 凯利公式: f = (bp - q) / b
        # b = avg_win / avg_loss (赔率)
        # p = win_rate
        # q = 1 - win_rate
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # 应用保守系数
        kelly = max(0, kelly * fraction)
        
        # 不超过单一持仓限制
        kelly = min(kelly, self.limits.max_single_position)
        
        return kelly
    
    def volatility_target(self,
                          asset_volatility: float,
                          target_volatility: float = 0.15) -> float:
        """
        波动率目标仓位
        根据资产波动率调整仓位，使组合波动率接近目标
        
        Args:
            asset_volatility: 资产年化波动率
            target_volatility: 目标年化波动率
        
        Returns:
            建议仓位比例
        """
        if asset_volatility == 0:
            return 0
        
        position = target_volatility / asset_volatility
        
        # 限制在合理范围
        position = min(position, self.limits.max_single_position * 2)  # 允许 2x 杠杆
        position = max(position, 0)
        
        return position
    
    def risk_parity(self, volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        风险平价配置
        使每个资产的风险贡献相等
        
        Args:
            volatilities: 各资产波动率字典
        
        Returns:
            各资产权重字典
        """
        assets = list(volatilities.keys())
        vols = np.array([volatilities[a] for a in assets])
        
        # 风险平价: 权重与波动率成反比
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        return dict(zip(assets, weights))
    
    def max_drawdown_control(self,
                             current_drawdown: float,
                             max_allowed_dd: float = 0.10,
                             base_position: float = 1.0) -> float:
        """
        最大回撤控制
        当回撤接近阈值时自动减仓
        
        Args:
            current_drawdown: 当前回撤 (负值)
            max_allowed_dd: 最大允许回撤
            base_position: 基础仓位比例
        
        Returns:
            调整后的仓位比例
        """
        current_dd = abs(current_drawdown)
        max_dd = abs(max_allowed_dd)
        
        if current_dd >= max_dd:
            # 达到最大回撤，清仓
            return 0
        
        # 线性减仓
        # 回撤 0% -> 100% 仓位
        # 回撤 max_dd -> 0% 仓位
        reduction_ratio = 1 - (current_dd / max_dd)
        
        return base_position * reduction_ratio


class RiskBudget:
    """风险预算管理"""
    
    def __init__(self, 
                 total_risk_budget: float = 0.10,
                 max_single_risk: float = 0.02):
        """
        Args:
            total_risk_budget: 总风险预算 (组合最大损失)
            max_single_risk: 单笔最大风险
        """
        self.total_risk_budget = total_risk_budget
        self.max_single_risk = max_single_risk
        self.used_risk = 0.0
        self.positions = {}
    
    def available_risk(self) -> float:
        """剩余可用风险额度"""
        return max(0, self.total_risk_budget - self.used_risk)
    
    def can_add_position(self, risk_amount: float) -> bool:
        """检查是否可以新增仓位"""
        if risk_amount > self.max_single_risk:
            return False
        if risk_amount > self.available_risk():
            return False
        return True
    
    def add_position(self, symbol: str, risk_amount: float) -> bool:
        """
        添加仓位
        
        Returns:
            是否成功
        """
        if not self.can_add_position(risk_amount):
            return False
        
        self.positions[symbol] = risk_amount
        self.used_risk += risk_amount
        return True
    
    def remove_position(self, symbol: str):
        """移除仓位"""
        if symbol in self.positions:
            self.used_risk -= self.positions[symbol]
            del self.positions[symbol]
    
    def summary(self) -> Dict:
        """风险预算摘要"""
        return {
            'total_budget': self.total_risk_budget,
            'used_risk': self.used_risk,
            'available_risk': self.available_risk(),
            'utilization': self.used_risk / self.total_risk_budget if self.total_risk_budget > 0 else 0,
            'position_count': len(self.positions),
            'positions': self.positions.copy()
        }
