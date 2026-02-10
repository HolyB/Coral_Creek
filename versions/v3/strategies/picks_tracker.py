#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Daily Picks Performance Tracker
每日机会历史表现追踪系统

功能:
1. 记录每日扫描出的机会及其特征
2. 追踪后续 5/10/20 天表现
3. 分析哪些特征与成功相关
4. 生成策略有效性报告
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import json

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


@dataclass
class DailyPick:
    """每日机会记录"""
    symbol: str
    pick_date: str
    strategy: str                      # 策略名称
    
    # 入选时的特征
    price: float = 0.0
    blue_daily: float = 0.0
    blue_weekly: float = 0.0
    blue_monthly: float = 0.0
    adx: float = 0.0
    turnover: float = 0.0
    market_cap: float = 0.0
    
    # 标签特征
    is_heima: bool = False
    is_juedi: bool = False
    is_new_discovery: bool = False      # 新发现
    chip_pattern: str = ""              # 筹码形态
    news_sentiment: str = ""            # 新闻情绪
    
    # 后续表现 (需要后续更新)
    return_d1: float = None
    return_d3: float = None
    return_d5: float = None
    return_d10: float = None
    return_d20: float = None
    max_gain: float = None              # 期间最高涨幅
    max_loss: float = None              # 期间最大回撤
    
    # 元数据
    market: str = 'US'
    created_at: str = None
    updated_at: str = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'pick_date': self.pick_date,
            'strategy': self.strategy,
            'price': self.price,
            'blue_daily': self.blue_daily,
            'blue_weekly': self.blue_weekly,
            'blue_monthly': self.blue_monthly,
            'adx': self.adx,
            'turnover': self.turnover,
            'market_cap': self.market_cap,
            'is_heima': self.is_heima,
            'is_juedi': self.is_juedi,
            'is_new_discovery': self.is_new_discovery,
            'chip_pattern': self.chip_pattern,
            'news_sentiment': self.news_sentiment,
            'return_d1': self.return_d1,
            'return_d3': self.return_d3,
            'return_d5': self.return_d5,
            'return_d10': self.return_d10,
            'return_d20': self.return_d20,
            'max_gain': self.max_gain,
            'max_loss': self.max_loss,
            'market': self.market,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DailyPick':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PicksPerformanceTracker:
    """机会表现追踪器"""
    
    def __init__(self, use_supabase: bool = True):
        self.use_supabase = use_supabase
        self._supabase = None
        self.table_name = 'daily_picks_performance'
    
    def _get_supabase(self):
        """获取 Supabase 客户端"""
        if self._supabase is None:
            try:
                from supabase import create_client
                url = os.getenv('SUPABASE_URL')
                key = os.getenv('SUPABASE_KEY')
                if url and key:
                    self._supabase = create_client(url, key)
            except Exception as e:
                print(f"Supabase not available: {e}")
        return self._supabase
    
    def record_pick(self, pick: DailyPick) -> bool:
        """记录一个机会"""
        pick.created_at = datetime.now().isoformat()
        
        supabase = self._get_supabase()
        if supabase:
            try:
                supabase.table(self.table_name).upsert(
                    pick.to_dict(),
                    on_conflict='symbol,pick_date,strategy'
                ).execute()
                return True
            except Exception as e:
                print(f"Failed to record pick: {e}")
                return False
        return False
    
    def record_picks_batch(self, picks: List[DailyPick]) -> int:
        """批量记录机会"""
        success = 0
        for pick in picks:
            if self.record_pick(pick):
                success += 1
        return success
    
    def update_returns(self, symbol: str, pick_date: str, 
                       returns: Dict[str, float]) -> bool:
        """更新后续收益"""
        supabase = self._get_supabase()
        if supabase:
            try:
                update_data = {'updated_at': datetime.now().isoformat()}
                
                if 'return_d1' in returns:
                    update_data['return_d1'] = returns['return_d1']
                if 'return_d3' in returns:
                    update_data['return_d3'] = returns['return_d3']
                if 'return_d5' in returns:
                    update_data['return_d5'] = returns['return_d5']
                if 'return_d10' in returns:
                    update_data['return_d10'] = returns['return_d10']
                if 'return_d20' in returns:
                    update_data['return_d20'] = returns['return_d20']
                if 'max_gain' in returns:
                    update_data['max_gain'] = returns['max_gain']
                if 'max_loss' in returns:
                    update_data['max_loss'] = returns['max_loss']
                
                supabase.table(self.table_name).update(update_data).match({
                    'symbol': symbol,
                    'pick_date': pick_date
                }).execute()
                return True
            except Exception as e:
                print(f"Failed to update returns: {e}")
        return False
    
    def get_picks_needing_update(self, days_old: int = 5) -> List[Dict]:
        """获取需要更新收益的记录"""
        supabase = self._get_supabase()
        if not supabase:
            return []
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')
            
            # 查询 return_d5 == null 且 pick_date <= cutoff
            response = supabase.table(self.table_name).select('*').is_(
                'return_d5', 'null'
            ).lte('pick_date', cutoff_date).execute()
            
            return response.data or []
        except Exception as e:
            print(f"Failed to get picks needing update: {e}")
            return []
    
    def calculate_forward_returns(self, symbol: str, pick_date: str, 
                                  market: str = 'US') -> Dict[str, float]:
        """计算前向收益"""
        try:
            from data_fetcher import get_stock_data
            
            pick_dt = datetime.strptime(pick_date, '%Y-%m-%d')
            
            # 获取足够的历史数据
            df = get_stock_data(symbol, market=market, days=60)
            if df is None or df.empty:
                return {}
            
            df.index = pd.to_datetime(df.index)
            
            # 找到入选日期的价格
            pick_mask = df.index.date == pick_dt.date()
            if not pick_mask.any():
                # 取最近的交易日
                valid_dates = df.index[df.index >= pick_dt]
                if len(valid_dates) == 0:
                    return {}
                base_idx = df.index.get_loc(valid_dates[0])
                base_price = df.iloc[base_idx]['Close']
            else:
                base_idx = df.index.get_loc(df.index[pick_mask][0])
                base_price = df.loc[pick_mask, 'Close'].iloc[0]
            
            returns = {}
            
            # 各周期收益
            for days, key in [(1, 'return_d1'), (3, 'return_d3'), 
                              (5, 'return_d5'), (10, 'return_d10'), (20, 'return_d20')]:
                target_idx = base_idx + days
                if target_idx < len(df):
                    future_price = df.iloc[target_idx]['Close']
                    returns[key] = round((future_price - base_price) / base_price * 100, 2)
            
            # 期间最高/最低
            if base_idx + 20 < len(df):
                period_data = df.iloc[base_idx:base_idx + 21]
                max_price = period_data['High'].max()
                min_price = period_data['Low'].min()
                returns['max_gain'] = round((max_price - base_price) / base_price * 100, 2)
                returns['max_loss'] = round((min_price - base_price) / base_price * 100, 2)
            
            return returns
            
        except Exception as e:
            print(f"Error calculating returns for {symbol}: {e}")
            return {}
    
    def batch_update_returns(self, limit: int = 50) -> Dict:
        """批量更新收益"""
        picks = self.get_picks_needing_update()[:limit]
        
        updated = 0
        errors = 0
        
        for pick in picks:
            returns = self.calculate_forward_returns(
                pick['symbol'], 
                pick['pick_date'],
                pick.get('market', 'US')
            )
            
            if returns:
                if self.update_returns(pick['symbol'], pick['pick_date'], returns):
                    updated += 1
                else:
                    errors += 1
            else:
                errors += 1
        
        return {
            'total': len(picks),
            'updated': updated,
            'errors': errors
        }
    
    def get_performance_summary(self, 
                                 start_date: str = None,
                                 end_date: str = None,
                                 strategy: str = None,
                                 market: str = None) -> Dict:
        """获取表现汇总"""
        supabase = self._get_supabase()
        if not supabase:
            return {}
        
        try:
            query = supabase.table(self.table_name).select('*')
            
            if start_date:
                query = query.gte('pick_date', start_date)
            if end_date:
                query = query.lte('pick_date', end_date)
            if strategy:
                query = query.eq('strategy', strategy)
            if market:
                query = query.eq('market', market)
            
            response = query.not_.is_('return_d5', 'null').execute()
            data = response.data or []
            
            if not data:
                return {'total_picks': 0}
            
            df = pd.DataFrame(data)
            
            # 计算汇总统计
            return {
                'total_picks': len(df),
                'avg_return_d5': round(df['return_d5'].mean(), 2),
                'avg_return_d10': round(df['return_d10'].mean(), 2) if 'return_d10' in df.columns else None,
                'win_rate_d5': round((df['return_d5'] > 0).mean() * 100, 1),
                'avg_max_gain': round(df['max_gain'].mean(), 2) if 'max_gain' in df.columns and df['max_gain'].notna().any() else None,
                'avg_max_loss': round(df['max_loss'].mean(), 2) if 'max_loss' in df.columns and df['max_loss'].notna().any() else None,
                'best_pick': df.loc[df['return_d5'].idxmax()].to_dict() if len(df) > 0 else None,
                'worst_pick': df.loc[df['return_d5'].idxmin()].to_dict() if len(df) > 0 else None
            }
            
        except Exception as e:
            if "daily_picks_performance" in str(e) and "PGRST205" in str(e):
                # 云端未建该表时静默降级，避免刷屏
                return {}
            print(f"Failed to get performance summary: {e}")
            return {}


class FeatureAnalyzer:
    """特征分析器 - 分析哪些特征与成功相关"""
    
    def __init__(self, tracker: PicksPerformanceTracker):
        self.tracker = tracker
    
    def get_data(self, min_samples: int = 100) -> pd.DataFrame:
        """获取分析数据"""
        supabase = self.tracker._get_supabase()
        if not supabase:
            return pd.DataFrame()
        
        try:
            response = supabase.table(self.tracker.table_name).select('*').not_.is_(
                'return_d5', 'null'
            ).execute()
            
            data = response.data or []
            if len(data) < min_samples:
                print(f"Not enough samples: {len(data)} < {min_samples}")
            
            return pd.DataFrame(data)
        except Exception as e:
            if "daily_picks_performance" in str(e) and "PGRST205" in str(e):
                return pd.DataFrame()
            print(f"Failed to get data: {e}")
            return pd.DataFrame()
    
    def feature_importance(self, target_col: str = 'return_d5') -> Dict:
        """
        计算特征重要性 (相关性分析)
        
        Returns:
            Dict with feature correlations to target
        """
        df = self.get_data()
        if df.empty or target_col not in df.columns:
            return {}
        
        # 数值特征
        numeric_features = ['blue_daily', 'blue_weekly', 'blue_monthly', 
                           'adx', 'turnover', 'market_cap']
        
        correlations = {}
        for feat in numeric_features:
            if feat in df.columns:
                # 过滤 NaN
                valid = df[[feat, target_col]].dropna()
                if len(valid) > 10:
                    corr = valid[feat].corr(valid[target_col])
                    correlations[feat] = round(corr, 3)
        
        # 分类特征 (计算平均收益)
        categorical_analysis = {}
        
        # 黑马信号
        if 'is_heima' in df.columns:
            heima_returns = df[df['is_heima'] == True][target_col].mean()
            non_heima_returns = df[df['is_heima'] == False][target_col].mean()
            categorical_analysis['heima_effect'] = {
                'heima_avg': round(heima_returns, 2) if not np.isnan(heima_returns) else None,
                'non_heima_avg': round(non_heima_returns, 2) if not np.isnan(non_heima_returns) else None,
                'lift': round(heima_returns - non_heima_returns, 2) if not np.isnan(heima_returns) and not np.isnan(non_heima_returns) else None
            }
        
        # 新发现
        if 'is_new_discovery' in df.columns:
            new_returns = df[df['is_new_discovery'] == True][target_col].mean()
            old_returns = df[df['is_new_discovery'] == False][target_col].mean()
            categorical_analysis['new_discovery_effect'] = {
                'new_avg': round(new_returns, 2) if not np.isnan(new_returns) else None,
                'old_avg': round(old_returns, 2) if not np.isnan(old_returns) else None,
                'lift': round(new_returns - old_returns, 2) if not np.isnan(new_returns) and not np.isnan(old_returns) else None
            }
        
        # 筹码形态
        if 'chip_pattern' in df.columns:
            chip_analysis = df.groupby('chip_pattern')[target_col].agg(['mean', 'count']).round(2)
            categorical_analysis['chip_pattern'] = chip_analysis.to_dict()
        
        return {
            'correlations': correlations,
            'categorical_analysis': categorical_analysis,
            'n_samples': len(df)
        }
    
    def strategy_effectiveness(self) -> Dict:
        """分析各策略有效性"""
        df = self.get_data()
        if df.empty or 'strategy' not in df.columns:
            return {}
        
        results = {}
        
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            
            if len(strategy_data) < 5:
                continue
            
            returns_d5 = strategy_data['return_d5'].dropna()
            
            results[strategy] = {
                'total_picks': len(strategy_data),
                'avg_return_d5': round(returns_d5.mean(), 2),
                'win_rate': round((returns_d5 > 0).mean() * 100, 1),
                'std_return': round(returns_d5.std(), 2),
                'sharpe_like': round(returns_d5.mean() / returns_d5.std(), 2) if returns_d5.std() > 0 else 0,
                'best': round(returns_d5.max(), 2),
                'worst': round(returns_d5.min(), 2)
            }
        
        # 按平均收益排序
        results = dict(sorted(results.items(), 
                             key=lambda x: x[1]['avg_return_d5'], 
                             reverse=True))
        
        return results
    
    def optimal_parameters(self) -> Dict:
        """找出最优参数范围"""
        df = self.get_data()
        if df.empty:
            return {}
        
        target = 'return_d5'
        
        # 对数值特征进行分箱分析
        optimal = {}
        
        for feature in ['blue_daily', 'blue_weekly', 'adx', 'turnover']:
            if feature not in df.columns:
                continue
            
            valid = df[[feature, target]].dropna()
            if len(valid) < 20:
                continue
            
            # 四分位数分箱
            try:
                valid['bin'] = pd.qcut(valid[feature], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                bin_returns = valid.groupby('bin')[target].mean()
                
                best_bin = bin_returns.idxmax()
                best_return = bin_returns.max()
                
                # 获取最优区间
                bin_ranges = valid.groupby('bin')[feature].agg(['min', 'max'])
                
                optimal[feature] = {
                    'best_quartile': str(best_bin),
                    'best_avg_return': round(best_return, 2),
                    'optimal_range': [
                        round(bin_ranges.loc[best_bin, 'min'], 1),
                        round(bin_ranges.loc[best_bin, 'max'], 1)
                    ],
                    'all_quartile_returns': bin_returns.round(2).to_dict()
                }
            except Exception:
                continue
        
        return optimal
    
    def generate_report(self) -> str:
        """生成分析报告 (Markdown)"""
        importance = self.feature_importance()
        strategies = self.strategy_effectiveness()
        optimal = self.optimal_parameters()
        
        report = "# 📊 每日机会表现分析报告\n\n"
        report += f"**分析样本数**: {importance.get('n_samples', 0)}\n\n"
        
        # 特征相关性
        report += "## 📈 特征重要性 (与5日收益相关性)\n\n"
        report += "| 特征 | 相关系数 | 解读 |\n"
        report += "|------|----------|------|\n"
        
        for feat, corr in importance.get('correlations', {}).items():
            if corr > 0.1:
                interpretation = "✅ 正相关"
            elif corr < -0.1:
                interpretation = "❌ 负相关"
            else:
                interpretation = "➖ 弱相关"
            report += f"| {feat} | {corr} | {interpretation} |\n"
        
        # 分类特征
        report += "\n## 🏷️ 分类特征分析\n\n"
        
        cat_analysis = importance.get('categorical_analysis', {})
        if 'heima_effect' in cat_analysis:
            he = cat_analysis['heima_effect']
            report += f"**黑马信号**: 有黑马 {he.get('heima_avg')}% vs 无黑马 {he.get('non_heima_avg')}% "
            report += f"(提升 {he.get('lift')}%)\n\n"
        
        if 'new_discovery_effect' in cat_analysis:
            ne = cat_analysis['new_discovery_effect']
            report += f"**新发现**: 新发现 {ne.get('new_avg')}% vs 老股 {ne.get('old_avg')}% "
            report += f"(提升 {ne.get('lift')}%)\n\n"
        
        # 策略有效性
        report += "## 🎯 策略有效性排名\n\n"
        report += "| 策略 | 选股数 | 平均收益 | 胜率 | Sharpe-like |\n"
        report += "|------|--------|----------|------|-------------|\n"
        
        for strategy, stats in strategies.items():
            report += f"| {strategy} | {stats['total_picks']} | {stats['avg_return_d5']}% | {stats['win_rate']}% | {stats['sharpe_like']} |\n"
        
        # 最优参数
        report += "\n## ⚙️ 最优参数区间\n\n"
        
        for feat, opt in optimal.items():
            report += f"**{feat}**: 最优区间 {opt['optimal_range'][0]} - {opt['optimal_range'][1]}"
            report += f" (平均收益 {opt['best_avg_return']}%)\n"
        
        report += "\n---\n"
        report += f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        
        return report


# 便捷函数
def record_todays_picks(picks_df: pd.DataFrame, 
                        strategy: str = 'BLUE',
                        market: str = 'US') -> int:
    """
    记录今日机会
    
    Args:
        picks_df: 包含机会数据的 DataFrame
        strategy: 策略名称
        market: 市场
    
    Returns:
        成功记录的数量
    """
    tracker = PicksPerformanceTracker()
    today = datetime.now().strftime('%Y-%m-%d')
    
    picks = []
    for _, row in picks_df.iterrows():
        pick = DailyPick(
            symbol=row.get('Ticker', row.get('symbol', '')),
            pick_date=today,
            strategy=strategy,
            price=row.get('Price', row.get('price', 0)),
            blue_daily=row.get('Day BLUE', row.get('blue_daily', 0)),
            blue_weekly=row.get('Week BLUE', row.get('blue_weekly', 0)),
            blue_monthly=row.get('Month BLUE', row.get('blue_monthly', 0)),
            adx=row.get('ADX', row.get('adx', 0)),
            turnover=row.get('Turnover', row.get('turnover_m', 0)),
            market_cap=row.get('Mkt Cap', row.get('market_cap', 0)),
            is_heima=row.get('黑马', row.get('is_heima', False)),
            is_juedi=row.get('掘地', row.get('is_juedi', False)),
            is_new_discovery='🆕' in str(row.get('新发现', '')),
            chip_pattern=row.get('筹码形态', ''),
            market=market
        )
        picks.append(pick)
    
    return tracker.record_picks_batch(picks)


def get_feature_analysis() -> Dict:
    """获取特征分析结果"""
    tracker = PicksPerformanceTracker()
    analyzer = FeatureAnalyzer(tracker)
    return analyzer.feature_importance()


def get_strategy_report() -> str:
    """获取策略报告"""
    tracker = PicksPerformanceTracker()
    analyzer = FeatureAnalyzer(tracker)
    return analyzer.generate_report()


if __name__ == "__main__":
    # 测试
    tracker = PicksPerformanceTracker()
    
    # 记录测试数据
    test_pick = DailyPick(
        symbol='NVDA',
        pick_date='2026-01-20',
        strategy='BLUE_强势',
        price=125.0,
        blue_daily=185,
        blue_weekly=120,
        adx=35,
        market='US'
    )
    
    print("Recording test pick...")
    tracker.record_pick(test_pick)
    
    # 更新收益
    print("Updating returns...")
    result = tracker.batch_update_returns(limit=10)
    print(f"Updated: {result}")
    
    # 获取汇总
    print("\nPerformance Summary:")
    summary = tracker.get_performance_summary()
    print(json.dumps(summary, indent=2, default=str))
