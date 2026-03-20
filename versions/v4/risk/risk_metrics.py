"""
风险指标计算模块
Risk Metrics Calculator

包含：
- VaR (Value at Risk)
- CVaR (Conditional VaR)
- 最大回撤
- 波动率
- Sharpe/Sortino 比率
- 相关性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class RiskMetrics:
    """风险指标计算器"""
    
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.02):
        """
        Args:
            returns: 收益率序列 (日收益率)
            risk_free_rate: 无风险利率 (年化)
        """
        self.returns = returns.dropna()
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
    
    def var(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        计算 Value at Risk
        
        Args:
            confidence: 置信度 (0.95 = 95%)
            method: 'historical', 'parametric', 'cornish_fisher'
        
        Returns:
            VaR (负值表示损失)
        """
        if method == 'historical':
            return np.percentile(self.returns, (1 - confidence) * 100)
        
        elif method == 'parametric':
            # 假设正态分布
            mu = self.returns.mean()
            sigma = self.returns.std()
            return stats.norm.ppf(1 - confidence, mu, sigma)
        
        elif method == 'cornish_fisher':
            # 考虑偏度和峰度的修正
            mu = self.returns.mean()
            sigma = self.returns.std()
            skew = self.returns.skew()
            kurt = self.returns.kurtosis()
            
            z = stats.norm.ppf(1 - confidence)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                    (z**3 - 3*z) * (kurt - 3) / 24 - 
                    (2*z**3 - 5*z) * skew**2 / 36)
            
            return mu + sigma * z_cf
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def cvar(self, confidence: float = 0.95) -> float:
        """
        计算 Conditional VaR (Expected Shortfall)
        超过 VaR 的平均损失
        """
        var = self.var(confidence, method='historical')
        return self.returns[self.returns <= var].mean()
    
    def max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        计算最大回撤
        
        Returns:
            (最大回撤值, 峰值日期, 谷底日期)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = cumulative[:trough_date].idxmax()
        
        return max_dd, peak_date, trough_date
    
    def volatility(self, annualize: bool = True) -> float:
        """计算波动率"""
        vol = self.returns.std()
        if annualize:
            vol *= np.sqrt(252)
        return vol
    
    def sharpe_ratio(self) -> float:
        """计算 Sharpe 比率 (年化)"""
        excess_returns = self.returns - self.daily_rf
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def sortino_ratio(self) -> float:
        """计算 Sortino 比率 (只考虑下行风险)"""
        excess_returns = self.returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0
        
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calmar_ratio(self) -> float:
        """计算 Calmar 比率 (年化收益 / 最大回撤)"""
        annual_return = self.returns.mean() * 252
        max_dd, _, _ = self.max_drawdown()
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / abs(max_dd)
    
    def summary(self) -> Dict:
        """生成风险指标摘要"""
        max_dd, peak, trough = self.max_drawdown()
        
        return {
            'total_return': (1 + self.returns).prod() - 1,
            'annual_return': self.returns.mean() * 252,
            'volatility': self.volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'max_drawdown': max_dd,
            'max_dd_peak': peak,
            'max_dd_trough': trough,
            'var_95': self.var(0.95),
            'cvar_95': self.cvar(0.95),
            'var_99': self.var(0.99),
            'skewness': self.returns.skew(),
            'kurtosis': self.returns.kurtosis(),
            'win_rate': (self.returns > 0).mean(),
            'profit_factor': abs(self.returns[self.returns > 0].sum() / 
                               self.returns[self.returns < 0].sum()) if (self.returns < 0).any() else np.inf
        }


class PortfolioRisk:
    """组合风险分析"""
    
    def __init__(self, returns_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            returns_df: 多资产收益率 DataFrame，columns 为资产名
            weights: 权重字典，如 {'AAPL': 0.3, 'NVDA': 0.7}
        """
        self.returns_df = returns_df.dropna()
        
        if weights is None:
            # 等权重
            n = len(returns_df.columns)
            weights = {col: 1/n for col in returns_df.columns}
        
        self.weights = weights
        self.weight_array = np.array([weights.get(col, 0) for col in returns_df.columns])
    
    def correlation_matrix(self) -> pd.DataFrame:
        """计算相关性矩阵"""
        return self.returns_df.corr()
    
    def covariance_matrix(self, annualize: bool = True) -> pd.DataFrame:
        """计算协方差矩阵"""
        cov = self.returns_df.cov()
        if annualize:
            cov *= 252
        return cov
    
    def portfolio_return(self) -> float:
        """计算组合预期收益 (年化)"""
        return (self.returns_df.mean() * self.weight_array).sum() * 252
    
    def portfolio_volatility(self) -> float:
        """计算组合波动率 (年化)"""
        cov = self.covariance_matrix(annualize=True)
        port_var = np.dot(self.weight_array, np.dot(cov, self.weight_array))
        return np.sqrt(port_var)
    
    def marginal_risk_contribution(self) -> pd.Series:
        """计算边际风险贡献"""
        cov = self.covariance_matrix(annualize=True)
        port_vol = self.portfolio_volatility()
        
        mrc = np.dot(cov, self.weight_array) / port_vol
        return pd.Series(mrc, index=self.returns_df.columns)
    
    def risk_contribution(self) -> pd.Series:
        """计算各资产的风险贡献"""
        mrc = self.marginal_risk_contribution()
        rc = self.weight_array * mrc
        return pd.Series(rc, index=self.returns_df.columns)
    
    def risk_contribution_pct(self) -> pd.Series:
        """计算各资产的风险贡献百分比"""
        rc = self.risk_contribution()
        return rc / rc.sum()
    
    def concentration_risk(self) -> Dict:
        """计算集中度风险"""
        weights_series = pd.Series(self.weights)
        
        # 最大单一持仓
        max_weight = weights_series.max()
        max_weight_asset = weights_series.idxmax()
        
        # HHI 指数 (赫芬达尔指数)
        hhi = (weights_series ** 2).sum()
        
        # 有效资产数 (1/HHI)
        effective_n = 1 / hhi if hhi > 0 else len(weights_series)
        
        return {
            'max_weight': max_weight,
            'max_weight_asset': max_weight_asset,
            'hhi': hhi,
            'effective_n': effective_n,
            'top3_weight': weights_series.nlargest(3).sum()
        }
    
    def diversification_ratio(self) -> float:
        """
        计算分散化比率
        = 加权平均波动率 / 组合波动率
        比率越高，分散化效果越好
        """
        individual_vols = self.returns_df.std() * np.sqrt(252)
        weighted_avg_vol = (individual_vols * self.weight_array).sum()
        port_vol = self.portfolio_volatility()
        
        return weighted_avg_vol / port_vol if port_vol > 0 else 1
    
    def summary(self) -> Dict:
        """生成组合风险摘要"""
        concentration = self.concentration_risk()
        
        # 计算组合收益序列
        portfolio_returns = (self.returns_df * self.weight_array).sum(axis=1)
        port_metrics = RiskMetrics(portfolio_returns)
        
        return {
            'portfolio_return': self.portfolio_return(),
            'portfolio_volatility': self.portfolio_volatility(),
            'sharpe_ratio': port_metrics.sharpe_ratio(),
            'max_drawdown': port_metrics.max_drawdown()[0],
            'var_95': port_metrics.var(0.95),
            'diversification_ratio': self.diversification_ratio(),
            **concentration
        }


def calculate_rolling_metrics(returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    计算滚动风险指标
    
    Args:
        returns: 收益率序列
        window: 滚动窗口 (交易日)
    
    Returns:
        DataFrame with rolling metrics
    """
    results = []
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        metrics = RiskMetrics(window_returns)
        
        results.append({
            'date': returns.index[i],
            'rolling_sharpe': metrics.sharpe_ratio(),
            'rolling_vol': metrics.volatility(),
            'rolling_var': metrics.var(0.95),
            'rolling_max_dd': metrics.max_drawdown()[0]
        })
    
    return pd.DataFrame(results).set_index('date')


def stress_test(returns: pd.Series, scenarios: Dict[str, float]) -> pd.DataFrame:
    """
    压力测试
    
    Args:
        returns: 收益率序列
        scenarios: 场景字典，如 {'市场暴跌': -0.10, '黑天鹅': -0.20}
    
    Returns:
        压力测试结果
    """
    current_value = 1.0
    results = []
    
    for name, shock in scenarios.items():
        stressed_value = current_value * (1 + shock)
        results.append({
            'scenario': name,
            'shock': shock,
            'value_after': stressed_value,
            'loss': current_value - stressed_value,
            'loss_pct': -shock
        })
    
    return pd.DataFrame(results)
