#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子评估模块 — 基于 alphalens-reloaded
=========================================
- 计算每个因子的 IC (信息系数)
- 分层收益分析
- 因子换手率
- 自动筛除无效因子
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

def evaluate_factors(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    periods: Tuple[int, ...] = (1, 5, 10, 20),
    quantiles: int = 5,
    min_ic: float = 0.02,
) -> Dict:
    """
    评估所有因子的有效性
    
    Args:
        factor_df: DataFrame, index=DatetimeIndex, columns=[symbol, factor1, factor2, ...]
                   或 MultiIndex (date, asset) 的 Series/DataFrame
        prices: DataFrame, index=DatetimeIndex, columns=symbols, values=close price
        periods: 持有期 (天)
        quantiles: 分位数
        min_ic: 最低 IC 阈值（低于此值的因子被标记为无效）
    
    Returns:
        dict with:
        - 'factor_scores': 每个因子的 IC/IR/收益汇总
        - 'valid_factors': IC >= min_ic 的因子列表
        - 'invalid_factors': IC < min_ic 的因子列表
        - 'details': 每个因子的详细 alphalens 结果
    """
    try:
        import alphalens
        from alphalens.utils import get_clean_factor_and_forward_returns
        from alphalens.performance import (
            factor_information_coefficient,
            mean_return_by_quantile,
            factor_alpha_beta,
        )
        from alphalens.tears import create_full_tear_sheet
    except ImportError:
        print("⚠️ alphalens-reloaded not installed. Run: pip install alphalens-reloaded")
        return {}
    
    results = {}
    factor_scores = []
    
    # 确定因子列
    if isinstance(factor_df, pd.DataFrame) and 'symbol' in factor_df.columns:
        factor_cols = [c for c in factor_df.columns if c not in ['symbol', 'date', 'Date']]
    elif isinstance(factor_df, pd.DataFrame):
        factor_cols = list(factor_df.columns)
    else:
        factor_cols = [factor_df.name or 'factor']
    
    for fname in factor_cols:
        try:
            # 构建 alphalens 所需的 factor 格式
            # MultiIndex: (date, asset), value = factor value
            if 'symbol' in factor_df.columns:
                df = factor_df[['symbol', fname]].copy()
                df = df.dropna()
                if df.empty:
                    continue
                
                # 构建 MultiIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    continue
                
                factor_data = df.set_index('symbol', append=True)[fname]
                factor_data.index.names = ['date', 'asset']
            else:
                continue
            
            # 对齐 prices
            common_assets = set(factor_data.index.get_level_values('asset')) & set(prices.columns)
            if len(common_assets) < 10:
                continue
            
            factor_filtered = factor_data[
                factor_data.index.get_level_values('asset').isin(common_assets)
            ]
            prices_filtered = prices[list(common_assets)]
            
            # alphalens 清洗
            merged = get_clean_factor_and_forward_returns(
                factor_filtered,
                prices_filtered,
                quantiles=quantiles,
                periods=periods,
                max_loss=0.5,
            )
            
            # IC 分析
            ic = factor_information_coefficient(merged)
            mean_ic = ic.mean()
            ic_std = ic.std()
            ir = mean_ic / (ic_std + 1e-8)  # Information Ratio
            
            # 分层收益
            mean_ret, _ = mean_return_by_quantile(merged)
            
            # 多空收益 (top - bottom quantile)
            if len(mean_ret) >= 2:
                long_short = {}
                for p in periods:
                    col = f'{p}D'
                    if col in mean_ret.columns:
                        ls = float(mean_ret[col].iloc[-1] - mean_ret[col].iloc[0])
                        long_short[f'ls_{p}d'] = ls * 252 / p  # 年化
            else:
                long_short = {}
            
            # 汇总
            best_period = max(periods, key=lambda p: abs(float(mean_ic.get(f'{p}D', 0))))
            best_ic = float(mean_ic.get(f'{best_period}D', 0))
            best_ir = float(ir.get(f'{best_period}D', 0))
            
            score = {
                'factor': fname,
                'ic_1d': float(mean_ic.get('1D', 0)),
                'ic_5d': float(mean_ic.get('5D', 0)),
                'ic_10d': float(mean_ic.get('10D', 0)),
                'ic_20d': float(mean_ic.get('20D', 0)),
                'best_ic': best_ic,
                'best_period': f'{best_period}D',
                'ir': best_ir,
                'valid': abs(best_ic) >= min_ic,
                **long_short,
            }
            factor_scores.append(score)
            
            results[fname] = {
                'ic': ic,
                'mean_returns': mean_ret,
                'merged_data': merged,
            }
            
        except Exception as e:
            factor_scores.append({
                'factor': fname,
                'error': str(e),
                'valid': False,
            })
    
    # 排序
    scores_df = pd.DataFrame(factor_scores)
    if not scores_df.empty and 'best_ic' in scores_df.columns:
        scores_df = scores_df.sort_values('best_ic', ascending=False, key=abs)
    
    valid = scores_df[scores_df.get('valid', False) == True]['factor'].tolist() if not scores_df.empty else []
    invalid = scores_df[scores_df.get('valid', False) == False]['factor'].tolist() if not scores_df.empty else []
    
    return {
        'factor_scores': scores_df,
        'valid_factors': valid,
        'invalid_factors': invalid,
        'details': results,
    }


def build_factor_data_from_db(market: str = 'US', days_back: int = 180, 
                                max_stocks: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从数据库构建 alphalens 所需的数据格式
    
    Returns:
        factor_df: DataFrame with DatetimeIndex, columns=[symbol, factor1, factor2, ...]
        prices_df: DataFrame with DatetimeIndex, columns=symbols
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    from ml.advanced_features import AdvancedFeatureEngineer
    from ml.alpha_factors import Alpha158Factors as Alpha158
    
    init_db()
    
    # 获取活跃股票
    dates = get_scanned_dates(market=market)
    from collections import Counter
    sym_counts = Counter()
    for d in dates[:60]:  # 最近 60 天
        results = query_scan_results(scan_date=d, market=market, limit=1000)
        for r in results:
            sym = r.get('symbol', '')
            if sym:
                sym_counts[sym] += 1
    
    # 选出最活跃的
    top_symbols = [s for s, _ in sym_counts.most_common(max_stocks)]
    
    # 构建价格矩阵和因子矩阵
    all_prices = {}
    all_factors = []
    
    adv = AdvancedFeatureEngineer()
    alpha = Alpha158()
    
    for sym in top_symbols:
        hist = get_stock_history(sym, market, days=days_back + 50)
        if hist is None or hist.empty or len(hist) < 50:
            continue
        
        # 标准化
        if not isinstance(hist.index, pd.DatetimeIndex):
            for col in ['Date', 'date']:
                if col in hist.columns:
                    hist = hist.set_index(col)
                    break
            hist.index = pd.to_datetime(hist.index)
        
        for need in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if need not in hist.columns:
                for c in hist.columns:
                    if c.lower() == need.lower():
                        hist = hist.rename(columns={c: need})
        
        if 'Close' not in hist.columns:
            continue
        
        # 价格
        all_prices[sym] = hist['Close']
        
        # 因子
        try:
            df_adv = adv.transform(hist)
            df_alpha = alpha.compute(hist)
            
            # 合并
            factors = pd.DataFrame(index=hist.index)
            for col in df_adv.columns:
                factors[col] = df_adv[col]
            for col in df_alpha.columns:
                if col not in factors.columns:
                    factors[col] = df_alpha[col]
            
            factors['symbol'] = sym
            all_factors.append(factors)
        except Exception:
            continue
    
    # 合并
    if not all_factors:
        return pd.DataFrame(), pd.DataFrame()
    
    factor_df = pd.concat(all_factors, axis=0)
    prices_df = pd.DataFrame(all_prices)
    
    # 只保留共同日期
    common_dates = factor_df.index.intersection(prices_df.index)
    factor_df = factor_df.loc[factor_df.index.isin(common_dates)]
    prices_df = prices_df.loc[common_dates]
    
    return factor_df, prices_df


def quick_factor_report(market: str = 'US', top_n: int = 30) -> pd.DataFrame:
    """
    快速生成因子评估报告
    
    Returns:
        DataFrame with factor scores sorted by IC
    """
    print(f"📊 Building factor data for {market}...")
    factor_df, prices_df = build_factor_data_from_db(market=market, max_stocks=100)
    
    if factor_df.empty:
        print("❌ No factor data available")
        return pd.DataFrame()
    
    print(f"  Factors: {len([c for c in factor_df.columns if c != 'symbol'])}")
    print(f"  Stocks: {factor_df['symbol'].nunique()}")
    print(f"  Dates: {factor_df.index.nunique()}")
    
    print(f"\n🔬 Evaluating factors...")
    result = evaluate_factors(factor_df, prices_df, periods=(1, 5, 10), min_ic=0.02)
    
    if not result:
        return pd.DataFrame()
    
    scores = result['factor_scores']
    print(f"\n✅ Valid factors ({len(result['valid_factors'])}): {result['valid_factors'][:top_n]}")
    print(f"❌ Invalid factors ({len(result['invalid_factors'])}): {result['invalid_factors'][:10]}")
    
    return scores
