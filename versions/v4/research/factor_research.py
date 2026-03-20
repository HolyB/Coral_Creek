#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子研究框架 - IC分析、因子收益、因子相关性
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FactorResearch:
    """因子研究核心类"""
    
    def __init__(self):
        self.factor_data = None
        self.return_data = None
        self.ic_results = {}
    
    def calculate_ic(self, 
                     factor_values: pd.Series, 
                     forward_returns: pd.Series,
                     method: str = 'spearman') -> float:
        """
        计算因子 IC (Information Coefficient)
        
        IC = Correlation(Factor, Forward_Return)
        
        Args:
            factor_values: 因子值 (截面)
            forward_returns: 前向收益率
            method: 相关系数方法 (spearman/pearson)
        
        Returns:
            IC 值 (-1 到 1)
        """
        # 对齐数据
        aligned = pd.DataFrame({
            'factor': factor_values,
            'return': forward_returns
        }).dropna()
        
        if len(aligned) < 5:
            return np.nan
        
        if method == 'spearman':
            ic, _ = stats.spearmanr(aligned['factor'], aligned['return'])
        else:
            ic, _ = stats.pearsonr(aligned['factor'], aligned['return'])
        
        return ic
    
    def calculate_ic_series(self, 
                            factor_df: pd.DataFrame,
                            return_df: pd.DataFrame,
                            factor_col: str = 'factor',
                            forward_days: int = 5) -> pd.DataFrame:
        """
        计算因子 IC 时间序列
        
        Args:
            factor_df: 因子数据 (需要 date, symbol, factor 列)
            return_df: 收益数据 (需要 date, symbol, return 列)
            factor_col: 因子列名
            forward_days: 前向收益天数
        
        Returns:
            IC 时间序列 DataFrame
        """
        dates = factor_df['date'].unique()
        ic_series = []
        
        for date in sorted(dates):
            # 获取当日因子值
            day_factors = factor_df[factor_df['date'] == date].set_index('symbol')[factor_col]
            
            # 获取N天后收益
            future_date = pd.to_datetime(date) + timedelta(days=forward_days)
            day_returns = return_df[return_df['date'] >= str(future_date)[:10]]
            
            if len(day_returns) > 0:
                day_returns = day_returns.groupby('symbol')['return'].first()
                ic = self.calculate_ic(day_factors, day_returns)
                
                ic_series.append({
                    'date': date,
                    'ic': ic,
                    'n_stocks': len(day_factors)
                })
        
        return pd.DataFrame(ic_series)
    
    def calculate_ic_stats(self, ic_series: pd.Series) -> Dict:
        """
        计算 IC 统计指标
        
        Args:
            ic_series: IC 时间序列
        
        Returns:
            IC 统计字典
        """
        ic_clean = ic_series.dropna()
        
        if len(ic_clean) == 0:
            return {'mean_ic': 0, 'ic_ir': 0, 'ic_positive_rate': 0}
        
        mean_ic = ic_clean.mean()
        std_ic = ic_clean.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0  # IC信息比率
        positive_rate = (ic_clean > 0).mean() * 100
        
        return {
            'mean_ic': round(mean_ic, 4),
            'std_ic': round(std_ic, 4),
            'ic_ir': round(ic_ir, 4),  # IC_IR > 0.5 为好因子
            'ic_positive_rate': round(positive_rate, 2),
            'n_periods': len(ic_clean)
        }
    
    def calculate_factor_return(self,
                                factor_df: pd.DataFrame,
                                return_df: pd.DataFrame,
                                factor_col: str = 'factor',
                                n_quantiles: int = 5) -> pd.DataFrame:
        """
        计算分组收益 (因子分层回测)
        
        Args:
            factor_df: 因子数据
            return_df: 收益数据
            factor_col: 因子列名
            n_quantiles: 分组数量
        
        Returns:
            分组收益 DataFrame
        """
        # 合并数据
        merged = pd.merge(
            factor_df[['date', 'symbol', factor_col]],
            return_df[['date', 'symbol', 'return']],
            on=['date', 'symbol'],
            how='inner'
        )
        
        if len(merged) == 0:
            return pd.DataFrame()
        
        # 按日期分组计算分位数
        def assign_quantile(group):
            group['quantile'] = pd.qcut(
                group[factor_col], 
                q=n_quantiles, 
                labels=range(1, n_quantiles + 1),
                duplicates='drop'
            )
            return group
        
        merged = merged.groupby('date').apply(assign_quantile).reset_index(drop=True)
        
        # 计算分组平均收益
        quantile_returns = merged.groupby(['date', 'quantile'])['return'].mean().unstack()
        
        # 计算累计收益
        cum_returns = (1 + quantile_returns / 100).cumprod()
        
        return {
            'quantile_returns': quantile_returns,
            'cumulative_returns': cum_returns,
            'long_short_return': quantile_returns[n_quantiles].mean() - quantile_returns[1].mean()
        }
    
    def calculate_factor_correlation(self, factor_df: pd.DataFrame, 
                                      factor_cols: List[str]) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        
        Args:
            factor_df: 包含多个因子列的 DataFrame
            factor_cols: 因子列名列表
        
        Returns:
            相关性矩阵
        """
        return factor_df[factor_cols].corr()
    
    def analyze_blue_factor(self, market: str = 'US', days: int = 60) -> Dict:
        """
        分析 BLUE 因子的预测能力
        
        Args:
            market: 市场
            days: 分析天数
        
        Returns:
            BLUE 因子分析结果
        """
        from db.database import query_scan_results
        
        # 获取历史扫描结果
        results = query_scan_results(market=market, limit=500)
        
        if not results:
            return {'error': 'No data'}
        
        df = pd.DataFrame(results)
        
        # 准备数据
        factor_data = df[['scan_date', 'symbol', 'blue_daily']].copy()
        factor_data.columns = ['date', 'symbol', 'factor']
        
        # 模拟收益 (简化：用价格变化)
        return_data = df[['scan_date', 'symbol', 'price']].copy()
        return_data.columns = ['date', 'symbol', 'return']
        
        # 计算 IC (简化版)
        ic_values = []
        for date in factor_data['date'].unique()[:20]:  # 前20天
            day_factors = factor_data[factor_data['date'] == date]['factor']
            day_returns = return_data[return_data['date'] == date]['return']
            
            if len(day_factors) > 5 and len(day_returns) > 5:
                # 模拟前向收益
                ic = self.calculate_ic(day_factors, day_returns)
                if not np.isnan(ic):
                    ic_values.append(ic)
        
        if ic_values:
            stats = self.calculate_ic_stats(pd.Series(ic_values))
        else:
            stats = {'mean_ic': 0, 'ic_ir': 0}
        
        return {
            'factor': 'BLUE_Daily',
            'market': market,
            'n_records': len(df),
            'stats': stats
        }


def analyze_factors_from_scan(market: str = 'US') -> Dict:
    """
    快速分析扫描数据中的因子
    """
    fr = FactorResearch()
    return fr.analyze_blue_factor(market=market)


if __name__ == "__main__":
    print("Testing Factor Research...")
    result = analyze_factors_from_scan('US')
    print(f"Result: {result}")
