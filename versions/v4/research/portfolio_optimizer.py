#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组合优化模块 - Markowitz 均值-方差优化

功能:
- 均值-方差优化 (Mean-Variance)
- 最大夏普比率组合
- 风险平价 (Risk Parity)
- 有效前沿
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PortfolioOptimizer:
    """组合优化器"""
    
    def __init__(self, returns_df: pd.DataFrame = None):
        """
        初始化优化器
        
        Args:
            returns_df: 收益率 DataFrame，index=日期，columns=股票代码
        """
        self.returns_df = returns_df
        self.mean_returns = None
        self.cov_matrix = None
        self.n_assets = 0
        
        if returns_df is not None:
            self._calculate_stats()
    
    def _calculate_stats(self):
        """计算收益统计"""
        if self.returns_df is None:
            return
        
        self.mean_returns = self.returns_df.mean() * 252  # 年化
        self.cov_matrix = self.returns_df.cov() * 252  # 年化
        self.n_assets = len(self.returns_df.columns)
    
    def set_returns(self, returns_df: pd.DataFrame):
        """设置收益数据"""
        self.returns_df = returns_df
        self._calculate_stats()
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """计算组合预期收益"""
        return np.dot(weights, self.mean_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """计算组合波动率"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算组合夏普比率"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - risk_free_rate) / vol if vol > 0 else 0
    
    def optimize_max_sharpe(self, risk_free_rate: float = 0.02) -> Dict:
        """
        优化最大夏普比率组合
        
        Args:
            risk_free_rate: 无风险利率
        
        Returns:
            优化结果
        """
        if self.n_assets == 0:
            return {'error': 'No data'}
        
        # 目标函数 (负夏普，因为minimize)
        def neg_sharpe(weights):
            return -self.portfolio_sharpe(weights, risk_free_rate)
        
        # 约束条件
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(neg_sharpe, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        return {
            'method': 'max_sharpe',
            'weights': dict(zip(self.returns_df.columns, np.round(weights, 4))),
            'expected_return': round(self.portfolio_return(weights) * 100, 2),
            'volatility': round(self.portfolio_volatility(weights) * 100, 2),
            'sharpe_ratio': round(self.portfolio_sharpe(weights, risk_free_rate), 2)
        }
    
    def optimize_min_volatility(self) -> Dict:
        """
        优化最小波动率组合
        
        Returns:
            优化结果
        """
        if self.n_assets == 0:
            return {'error': 'No data'}
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(self.portfolio_volatility, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        return {
            'method': 'min_volatility',
            'weights': dict(zip(self.returns_df.columns, np.round(weights, 4))),
            'expected_return': round(self.portfolio_return(weights) * 100, 2),
            'volatility': round(self.portfolio_volatility(weights) * 100, 2),
            'sharpe_ratio': round(self.portfolio_sharpe(weights), 2)
        }
    
    def optimize_target_return(self, target_return: float) -> Dict:
        """
        给定目标收益的最小风险组合
        
        Args:
            target_return: 目标年化收益率 (如 0.15 表示 15%)
        
        Returns:
            优化结果
        """
        if self.n_assets == 0:
            return {'error': 'No data'}
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(self.portfolio_volatility, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        return {
            'method': 'target_return',
            'target': f'{target_return*100:.1f}%',
            'weights': dict(zip(self.returns_df.columns, np.round(weights, 4))),
            'expected_return': round(self.portfolio_return(weights) * 100, 2),
            'volatility': round(self.portfolio_volatility(weights) * 100, 2),
            'sharpe_ratio': round(self.portfolio_sharpe(weights), 2)
        }
    
    def risk_parity(self) -> Dict:
        """
        风险平价配置 - 每个资产对组合风险贡献相等
        
        Returns:
            风险平价权重
        """
        if self.n_assets == 0:
            return {'error': 'No data'}
        
        # 风险贡献函数
        def risk_contribution(weights):
            vol = self.portfolio_volatility(weights)
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / vol
            return risk_contrib
        
        # 目标：使风险贡献相等
        def objective(weights):
            rc = risk_contribution(weights)
            target = np.mean(rc)
            return np.sum((rc - target) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.01, 1) for _ in range(self.n_assets))  # 最小1%
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        rc = risk_contribution(weights)
        
        return {
            'method': 'risk_parity',
            'weights': dict(zip(self.returns_df.columns, np.round(weights, 4))),
            'risk_contributions': dict(zip(self.returns_df.columns, np.round(rc, 4))),
            'expected_return': round(self.portfolio_return(weights) * 100, 2),
            'volatility': round(self.portfolio_volatility(weights) * 100, 2),
            'sharpe_ratio': round(self.portfolio_sharpe(weights), 2)
        }
    
    def efficient_frontier(self, n_points: int = 20) -> pd.DataFrame:
        """
        计算有效前沿
        
        Args:
            n_points: 采样点数
        
        Returns:
            有效前沿 DataFrame
        """
        if self.n_assets == 0:
            return pd.DataFrame()
        
        # 获取收益范围
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        for target in target_returns:
            try:
                result = self.optimize_target_return(target)
                if 'error' not in result:
                    frontier.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe': result['sharpe_ratio']
                    })
            except:
                continue
        
        return pd.DataFrame(frontier)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """获取相关性矩阵"""
        if self.returns_df is None:
            return pd.DataFrame()
        return self.returns_df.corr()


def optimize_portfolio_from_symbols(symbols: List[str], 
                                     market: str = 'US',
                                     days: int = 252) -> Dict:
    """
    从股票代码列表优化组合
    
    Args:
        symbols: 股票代码列表
        market: 市场
        days: 历史天数
    
    Returns:
        优化结果
    """
    from data_fetcher import get_us_stock_data, get_cn_stock_data
    
    returns_dict = {}
    
    for symbol in symbols[:10]:  # 最多10只
        try:
            if market == 'CN':
                df = get_cn_stock_data(symbol, days=days)
            else:
                df = get_us_stock_data(symbol, days=days)
            
            if df is not None and len(df) > 20:
                returns = df['Close'].pct_change().dropna()
                returns_dict[symbol] = returns
        except:
            continue
    
    if len(returns_dict) < 2:
        return {'error': 'Need at least 2 stocks with data'}
    
    # 对齐数据
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    if len(returns_df) < 20:
        return {'error': 'Insufficient historical data'}
    
    optimizer = PortfolioOptimizer(returns_df)
    
    return {
        'max_sharpe': optimizer.optimize_max_sharpe(),
        'min_vol': optimizer.optimize_min_volatility(),
        'risk_parity': optimizer.risk_parity(),
        'correlation': optimizer.get_correlation_matrix().to_dict()
    }


if __name__ == "__main__":
    print("Testing Portfolio Optimizer...")
    
    # 模拟数据测试
    np.random.seed(42)
    n_days = 252
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    returns = pd.DataFrame({
        sym: np.random.randn(n_days) * 0.02 + 0.0005
        for sym in symbols
    })
    
    optimizer = PortfolioOptimizer(returns)
    
    print("\n=== Max Sharpe Portfolio ===")
    result = optimizer.optimize_max_sharpe()
    print(f"Weights: {result['weights']}")
    print(f"Expected Return: {result['expected_return']}%")
    print(f"Volatility: {result['volatility']}%")
    print(f"Sharpe: {result['sharpe_ratio']}")
    
    print("\n=== Risk Parity Portfolio ===")
    rp = optimizer.risk_parity()
    print(f"Weights: {rp['weights']}")
    print(f"Sharpe: {rp['sharpe_ratio']}")
