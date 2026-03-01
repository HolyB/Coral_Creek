#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组合优化模块 — 基于 Riskfolio-Lib
=====================================
- 从等权 → 优化权重
- 支持 MVO, HRP, CVaR, Black-Litterman
- 集成 BLUE/MMoE 信号作为 views
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = 'HRP',
    risk_measure: str = 'MV',
    views: Optional[Dict[str, float]] = None,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> Dict:
    """
    组合优化
    
    Args:
        returns: DataFrame, index=dates, columns=symbols, values=daily returns
        method: 优化方法
            - 'MVO': Mean-Variance (经典 Markowitz)
            - 'HRP': Hierarchical Risk Parity (不需要预测收益)
            - 'CVaR': Conditional Value-at-Risk (控制尾部风险)
            - 'MaxSharpe': 最大 Sharpe Ratio
            - 'MinVol': 最小波动率
            - 'EqualWeight': 等权 (baseline)
            - 'BL': Black-Litterman (需要 views)
        risk_measure: 风险度量 ('MV'=variance, 'CVaR', 'MAD', 'MSV')
        views: BLUE/MMoE 信号作为 views {symbol: expected_return}
        max_weight: 单只股票最大权重
        min_weight: 单只股票最小权重
    
    Returns:
        dict with:
        - 'weights': {symbol: weight}
        - 'method': 使用的方法
        - 'stats': 组合统计
    """
    try:
        import riskfolio as rp
    except ImportError:
        print("⚠️ riskfolio-lib not installed. Run: pip install riskfolio-lib")
        return _equal_weight(returns)
    
    n = len(returns.columns)
    
    if method == 'EqualWeight':
        return _equal_weight(returns)
    
    if method == 'HRP':
        return _hrp_optimize(returns, max_weight)
    
    # 构建 Portfolio 对象
    port = rp.Portfolio(returns=returns)
    
    # 估计参数
    port.assets_stats(method_mu='hist', method_cov='hist')
    
    # 权重约束
    port.upperlng = max_weight  # 单只最大
    if min_weight > 0:
        port.lowerlng = min_weight
    
    # Black-Litterman views
    if method == 'BL' and views:
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        for i, (sym, expected_ret) in enumerate(views.items()):
            if sym in returns.columns:
                j = list(returns.columns).index(sym)
                P[i, j] = 1
                Q[i] = expected_ret
        
        port.blacklitterman_stats(
            P=P, Q=Q,
            delta=2.5,  # risk aversion
            rf=0.04 / 252,  # risk-free rate
        )
    
    # 优化
    try:
        if method in ['MVO', 'MaxSharpe', 'BL']:
            w = port.optimization(
                model='Classic',
                rm=risk_measure,
                obj='Sharpe',  # maximize Sharpe
                hist=True,
            )
        elif method == 'MinVol':
            w = port.optimization(
                model='Classic',
                rm=risk_measure,
                obj='MinRisk',
                hist=True,
            )
        elif method == 'CVaR':
            w = port.optimization(
                model='Classic',
                rm='CVaR',
                obj='Sharpe',
                hist=True,
            )
        else:
            return _equal_weight(returns)
        
        if w is None or w.empty:
            print(f"⚠️ {method} optimization failed, falling back to HRP")
            return _hrp_optimize(returns, max_weight)
        
        weights = {sym: float(w.loc[sym, 'weights']) for sym in w.index}
        
    except Exception as e:
        print(f"⚠️ {method} error: {e}, falling back to HRP")
        return _hrp_optimize(returns, max_weight)
    
    # 组合统计
    stats = _calc_portfolio_stats(returns, weights)
    
    return {
        'weights': weights,
        'method': method,
        'stats': stats,
        'n_assets': sum(1 for v in weights.values() if v > 0.01),
    }


def _hrp_optimize(returns: pd.DataFrame, max_weight: float = 0.25) -> Dict:
    """Hierarchical Risk Parity — 最稳健的方法"""
    try:
        import riskfolio as rp
        
        port = rp.HCPortfolio(returns=returns)
        w = port.optimization(
            model='HRP',
            codependence='pearson',
            rm='MV',
            rf=0.04 / 252,
            linkage='single',
            leaf_order=True,
        )
        
        if w is None or w.empty:
            return _equal_weight(returns)
        
        weights = {sym: float(w.loc[sym, 'weights']) for sym in w.index}
        
        # 截断超过 max_weight 的
        total = sum(weights.values())
        weights = {k: min(v / total, max_weight) for k, v in weights.items()}
        remaining = 1 - sum(weights.values())
        if remaining > 0.01:
            non_capped = {k: v for k, v in weights.items() if v < max_weight}
            if non_capped:
                bonus = remaining / len(non_capped)
                for k in non_capped:
                    weights[k] += bonus
        
        stats = _calc_portfolio_stats(returns, weights)
        
        return {
            'weights': weights,
            'method': 'HRP',
            'stats': stats,
            'n_assets': sum(1 for v in weights.values() if v > 0.01),
        }
    except Exception as e:
        print(f"⚠️ HRP error: {e}")
        return _equal_weight(returns)


def _equal_weight(returns: pd.DataFrame) -> Dict:
    """等权基准"""
    n = len(returns.columns)
    weights = {sym: 1.0 / n for sym in returns.columns}
    stats = _calc_portfolio_stats(returns, weights)
    return {
        'weights': weights,
        'method': 'EqualWeight',
        'stats': stats,
        'n_assets': n,
    }


def _calc_portfolio_stats(returns: pd.DataFrame, weights: Dict) -> Dict:
    """计算组合统计"""
    w = np.array([weights.get(sym, 0) for sym in returns.columns])
    w = w / (w.sum() + 1e-8)
    
    port_returns = (returns.values @ w)
    
    total_return = float((1 + pd.Series(port_returns)).prod() - 1)
    annual_return = float(np.mean(port_returns) * 252)
    annual_vol = float(np.std(port_returns) * np.sqrt(252))
    sharpe = annual_return / (annual_vol + 1e-8)
    
    # Max drawdown
    cumulative = (1 + pd.Series(port_returns)).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak - 1)
    max_dd = float(drawdown.min())
    
    # CVaR
    sorted_rets = np.sort(port_returns)
    cvar_5 = float(np.mean(sorted_rets[:max(1, int(len(sorted_rets) * 0.05))]))
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'cvar_5pct': cvar_5,
        'win_rate': float((port_returns > 0).mean()),
        'n_days': len(port_returns),
    }


def compare_methods(
    returns: pd.DataFrame,
    views: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    对比所有优化方法
    
    Args:
        returns: 收益矩阵
        views: BLUE/MMoE views
    
    Returns:
        对比 DataFrame
    """
    methods = ['EqualWeight', 'HRP', 'MinVol', 'MaxSharpe', 'CVaR']
    if views:
        methods.append('BL')
    
    results = []
    for m in methods:
        try:
            r = optimize_portfolio(returns, method=m, views=views)
            stats = r['stats']
            results.append({
                'method': m,
                'return': stats['annual_return'],
                'volatility': stats['annual_volatility'],
                'sharpe': stats['sharpe'],
                'max_dd': stats['max_drawdown'],
                'cvar_5': stats['cvar_5pct'],
                'win_rate': stats['win_rate'],
                'n_assets': r['n_assets'],
                'max_weight': max(r['weights'].values()) if r['weights'] else 0,
            })
        except Exception as e:
            results.append({'method': m, 'error': str(e)})
    
    return pd.DataFrame(results)


def optimize_blue_portfolio(
    market: str = 'US',
    days_back: int = 60,
    min_blue: float = 50,
    top_n: int = 20,
) -> Dict:
    """
    对 BLUE 选出的股票做组合优化
    
    1. 从 scan_results 获取最近 BLUE >= min_blue 的股票
    2. 获取历史价格
    3. 用 HRP + MaxSharpe + CVaR 对比
    4. 返回最优权重
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    
    init_db()
    
    # 获取最近的 BLUE 股票
    dates = get_scanned_dates(market=market)
    
    from collections import Counter
    candidates = Counter()
    blue_scores = {}
    
    for d in dates[:days_back]:
        results = query_scan_results(scan_date=d, market=market, limit=500)
        for r in results:
            sym = r.get('symbol', '')
            blue = float(r.get('blue_daily', 0) or 0)
            if sym and blue >= min_blue:
                candidates[sym] += 1
                if sym not in blue_scores or blue > blue_scores[sym]:
                    blue_scores[sym] = blue
    
    # 按出现次数 + BLUE 值排序
    ranked = sorted(candidates.keys(), 
                    key=lambda s: (candidates[s], blue_scores.get(s, 0)), 
                    reverse=True)[:top_n]
    
    if len(ranked) < 3:
        print(f"⚠️ Not enough BLUE stocks (found {len(ranked)})")
        return {}
    
    print(f"📊 {market} BLUE Portfolio: {len(ranked)} stocks")
    
    # 获取价格数据
    price_data = {}
    for sym in ranked:
        hist = get_stock_history(sym, market, days=days_back + 30)
        if hist is None or hist.empty:
            continue
        
        if not isinstance(hist.index, pd.DatetimeIndex):
            for col in ['Date', 'date']:
                if col in hist.columns:
                    hist = hist.set_index(col)
                    break
            hist.index = pd.to_datetime(hist.index)
        
        close_col = 'Close' if 'Close' in hist.columns else 'close'
        if close_col in hist.columns:
            price_data[sym] = hist[close_col]
    
    if len(price_data) < 3:
        print(f"⚠️ Not enough price data (found {len(price_data)})")
        return {}
    
    prices = pd.DataFrame(price_data).dropna(how='all')
    prices = prices.ffill().dropna(axis=1)
    
    if prices.shape[1] < 3 or prices.shape[0] < 20:
        return {}
    
    returns = prices.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0).clip(-0.5, 0.5)
    
    # 构建 BLUE views
    views = {}
    for sym in prices.columns:
        b = blue_scores.get(sym, 50)
        # BLUE 越高 → 预期日收益越高
        views[sym] = (b / 200) * 0.001  # 100 BLUE → 0.05% 日收益
    
    # 对比所有方法
    print(f"\n🔬 Comparing optimization methods...")
    comparison = compare_methods(returns, views=views)
    
    # 选最佳 (by Sharpe)
    valid = comparison.dropna(subset=['sharpe'])
    if valid.empty:
        return {}
    
    best_method = valid.loc[valid['sharpe'].idxmax(), 'method']
    best = optimize_portfolio(returns, method=best_method, views=views)
    
    print(f"\n📊 Method comparison:")
    for _, row in comparison.iterrows():
        if 'error' not in row or pd.isna(row.get('error')):
            print(f"  {row['method']:<15} Sharpe={row.get('sharpe', 0):.2f}  "
                  f"Return={row.get('return', 0):.1%}  "
                  f"MaxDD={row.get('max_dd', 0):.1%}  "
                  f"Assets={row.get('n_assets', 0):.0f}")
    
    print(f"\n🏆 Best: {best_method} (Sharpe={best['stats']['sharpe']:.2f})")
    
    # Top weights
    sorted_w = sorted(best['weights'].items(), key=lambda x: -x[1])
    print(f"\n📋 Top allocations:")
    for sym, w in sorted_w[:10]:
        if w > 0.01:
            b = blue_scores.get(sym, 0)
            print(f"  {sym:<8} {w:.1%}  (BLUE={b:.0f})")
    
    return {
        'best_method': best_method,
        'weights': best['weights'],
        'stats': best['stats'],
        'comparison': comparison,
        'blue_scores': blue_scores,
    }
