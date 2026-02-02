#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strategy Aggregator
策略聚合器 - 追踪博主和社区策略

功能:
1. TradingView 热门策略追踪
2. 博主/KOL 选股追踪
3. 社区策略表现验证
4. 策略复制与回测
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import requests

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class StrategySource(Enum):
    """策略来源"""
    TRADINGVIEW = "tradingview"
    TWITTER = "twitter"
    REDDIT = "reddit"
    DISCORD = "discord"
    WEIBO = "weibo"
    XUEQIU = "xueqiu"       # 雪球
    BLOGGER = "blogger"     # 独立博主
    MANUAL = "manual"       # 手动添加


class StrategyCategory(Enum):
    """策略类别"""
    MOMENTUM = "momentum"        # 动量策略
    REVERSAL = "reversal"        # 反转策略
    BREAKOUT = "breakout"        # 突破策略
    TREND_FOLLOWING = "trend"    # 趋势跟踪
    MEAN_REVERSION = "reversion" # 均值回归
    VOLATILITY = "volatility"    # 波动率策略
    FUNDAMENTAL = "fundamental"  # 基本面
    TECHNICAL = "technical"      # 技术分析
    QUANT = "quant"             # 量化策略
    OTHER = "other"


@dataclass
class StrategyAuthor:
    """策略作者/博主"""
    id: str
    name: str
    platform: StrategySource
    profile_url: str = ""
    followers: int = 0
    description: str = ""
    specialty: str = ""              # 擅长领域
    track_record: str = ""           # 历史战绩
    verified: bool = False
    created_at: str = ""
    
    # 表现统计 (自动计算)
    total_picks: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'platform': self.platform.value if isinstance(self.platform, StrategySource) else self.platform,
            'profile_url': self.profile_url,
            'followers': self.followers,
            'description': self.description,
            'specialty': self.specialty,
            'track_record': self.track_record,
            'verified': self.verified,
            'total_picks': self.total_picks,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return
        }


@dataclass 
class ExternalStrategy:
    """外部策略"""
    id: str
    name: str
    author: str
    source: StrategySource
    category: StrategyCategory
    
    description: str = ""
    url: str = ""
    
    # 策略参数
    entry_rules: str = ""            # 入场规则描述
    exit_rules: str = ""             # 出场规则描述
    indicators: List[str] = field(default_factory=list)  # 使用的指标
    timeframe: str = "daily"         # 时间框架
    
    # 原作者声称的表现
    claimed_win_rate: float = 0.0
    claimed_return: float = 0.0
    claimed_sharpe: float = 0.0
    
    # 我们验证的表现
    verified_win_rate: float = None
    verified_return: float = None
    verified_sharpe: float = None
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    last_updated: str = ""
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'author': self.author,
            'source': self.source.value if isinstance(self.source, StrategySource) else self.source,
            'category': self.category.value if isinstance(self.category, StrategyCategory) else self.category,
            'description': self.description,
            'url': self.url,
            'entry_rules': self.entry_rules,
            'exit_rules': self.exit_rules,
            'indicators': self.indicators,
            'timeframe': self.timeframe,
            'claimed_win_rate': self.claimed_win_rate,
            'claimed_return': self.claimed_return,
            'verified_win_rate': self.verified_win_rate,
            'verified_return': self.verified_return,
            'tags': self.tags,
            'is_active': self.is_active
        }


@dataclass
class AuthorPick:
    """博主选股记录"""
    author_id: str
    symbol: str
    pick_date: str
    action: str              # 'buy', 'sell', 'watch'
    
    price_at_pick: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    reasoning: str = ""
    
    # 后续表现
    return_d1: float = None
    return_d5: float = None
    return_d10: float = None
    return_d20: float = None
    hit_target: bool = None
    hit_stop: bool = None
    
    source_url: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'author_id': self.author_id,
            'symbol': self.symbol,
            'pick_date': self.pick_date,
            'action': self.action,
            'price_at_pick': self.price_at_pick,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'reasoning': self.reasoning,
            'return_d1': self.return_d1,
            'return_d5': self.return_d5,
            'return_d10': self.return_d10,
            'return_d20': self.return_d20,
            'hit_target': self.hit_target,
            'hit_stop': self.hit_stop,
            'source_url': self.source_url
        }


class TradingViewScraper:
    """TradingView 策略抓取器"""
    
    BASE_URL = "https://www.tradingview.com"
    
    # 热门策略列表 (预定义，因为 TV 没有公开 API)
    POPULAR_STRATEGIES = [
        {
            'name': 'RSI Divergence Strategy',
            'category': StrategyCategory.REVERSAL,
            'indicators': ['RSI', 'Divergence'],
            'entry_rules': 'Buy on bullish RSI divergence when price makes lower low but RSI makes higher low',
            'exit_rules': 'Sell when RSI > 70 or price hits resistance',
            'claimed_win_rate': 65.0,
            'url': 'https://www.tradingview.com/script/xxxxx/'
        },
        {
            'name': 'EMA Crossover (9/21)',
            'category': StrategyCategory.TREND_FOLLOWING,
            'indicators': ['EMA9', 'EMA21'],
            'entry_rules': 'Buy when EMA9 crosses above EMA21',
            'exit_rules': 'Sell when EMA9 crosses below EMA21',
            'claimed_win_rate': 55.0
        },
        {
            'name': 'MACD Histogram Reversal',
            'category': StrategyCategory.MOMENTUM,
            'indicators': ['MACD', 'Histogram'],
            'entry_rules': 'Buy when MACD histogram turns positive after being negative',
            'exit_rules': 'Sell when histogram turns negative',
            'claimed_win_rate': 58.0
        },
        {
            'name': 'Bollinger Band Squeeze',
            'category': StrategyCategory.VOLATILITY,
            'indicators': ['Bollinger Bands', 'Keltner Channel'],
            'entry_rules': 'Buy when bands squeeze and price breaks above upper band',
            'exit_rules': 'Sell when price reaches 2x ATR target',
            'claimed_win_rate': 52.0
        },
        {
            'name': 'Volume Price Analysis',
            'category': StrategyCategory.TECHNICAL,
            'indicators': ['Volume', 'VWAP', 'OBV'],
            'entry_rules': 'Buy on high volume breakout above VWAP',
            'exit_rules': 'Sell when price falls below VWAP on high volume',
            'claimed_win_rate': 60.0
        },
        {
            'name': 'SuperTrend Strategy',
            'category': StrategyCategory.TREND_FOLLOWING,
            'indicators': ['SuperTrend', 'ATR'],
            'entry_rules': 'Buy when SuperTrend turns green (price above indicator)',
            'exit_rules': 'Sell when SuperTrend turns red',
            'claimed_win_rate': 53.0
        },
        {
            'name': 'Ichimoku Cloud Strategy',
            'category': StrategyCategory.TREND_FOLLOWING,
            'indicators': ['Ichimoku Cloud', 'Tenkan', 'Kijun'],
            'entry_rules': 'Buy when price above cloud and Tenkan crosses above Kijun',
            'exit_rules': 'Sell when price enters cloud or Tenkan crosses below Kijun',
            'claimed_win_rate': 58.0
        },
        {
            'name': 'ADX Trend Trading',
            'category': StrategyCategory.TREND_FOLLOWING,
            'indicators': ['ADX', 'DI+', 'DI-'],
            'entry_rules': 'Buy when ADX > 25 and DI+ > DI-',
            'exit_rules': 'Sell when ADX < 20 or DI- > DI+',
            'claimed_win_rate': 56.0
        },
        {
            'name': 'Fibonacci Retracement',
            'category': StrategyCategory.REVERSAL,
            'indicators': ['Fibonacci', 'Support/Resistance'],
            'entry_rules': 'Buy at 61.8% retracement level with confirmation',
            'exit_rules': 'Sell at previous high or 161.8% extension',
            'claimed_win_rate': 62.0
        },
        {
            'name': 'VWAP Bounce Strategy',
            'category': StrategyCategory.MEAN_REVERSION,
            'indicators': ['VWAP', 'Volume'],
            'entry_rules': 'Buy when price bounces off VWAP with increasing volume',
            'exit_rules': 'Sell at upper standard deviation band',
            'claimed_win_rate': 57.0
        }
    ]
    
    def get_popular_strategies(self) -> List[ExternalStrategy]:
        """获取热门策略"""
        strategies = []
        
        for i, s in enumerate(self.POPULAR_STRATEGIES):
            strategy = ExternalStrategy(
                id=f"tv_{i+1}",
                name=s['name'],
                author="TradingView Community",
                source=StrategySource.TRADINGVIEW,
                category=s['category'],
                description=s.get('description', ''),
                url=s.get('url', ''),
                entry_rules=s['entry_rules'],
                exit_rules=s['exit_rules'],
                indicators=s['indicators'],
                claimed_win_rate=s['claimed_win_rate'],
                tags=['tradingview', 'popular']
            )
            strategies.append(strategy)
        
        return strategies


class XueqiuScraper:
    """雪球博主抓取器"""
    
    # 知名雪球博主 (预定义)
    FAMOUS_AUTHORS = [
        {
            'id': 'xq_001',
            'name': '股市药丸',
            'specialty': 'A股价值投资',
            'followers': 500000,
            'description': '专注大盘蓝筹和消费龙头'
        },
        {
            'id': 'xq_002', 
            'name': '唐朝',
            'specialty': '价值投资',
            'followers': 800000,
            'description': '巴菲特投资理念践行者'
        },
        {
            'id': 'xq_003',
            'name': '小小辛巴',
            'specialty': '逆向投资',
            'followers': 300000,
            'description': '擅长在恐慌中寻找机会'
        },
        {
            'id': 'xq_004',
            'name': '释老毛',
            'specialty': '技术分析',
            'followers': 200000,
            'description': '趋势交易者'
        }
    ]
    
    def get_famous_authors(self) -> List[StrategyAuthor]:
        """获取知名博主"""
        authors = []
        
        for a in self.FAMOUS_AUTHORS:
            author = StrategyAuthor(
                id=a['id'],
                name=a['name'],
                platform=StrategySource.XUEQIU,
                specialty=a['specialty'],
                followers=a['followers'],
                description=a['description'],
                profile_url=f"https://xueqiu.com/u/{a['id']}"
            )
            authors.append(author)
        
        return authors


class StrategyAggregator:
    """策略聚合器 - 主类"""
    
    def __init__(self):
        self.tv_scraper = TradingViewScraper()
        self.xq_scraper = XueqiuScraper()
        self._supabase = None
    
    def _get_supabase(self):
        if self._supabase is None:
            try:
                from supabase import create_client
                url = os.getenv('SUPABASE_URL')
                key = os.getenv('SUPABASE_KEY')
                if url and key:
                    self._supabase = create_client(url, key)
            except:
                pass
        return self._supabase
    
    # === 策略管理 ===
    
    def get_all_strategies(self) -> List[ExternalStrategy]:
        """获取所有策略"""
        strategies = []
        
        # TradingView 策略
        strategies.extend(self.tv_scraper.get_popular_strategies())
        
        # 从数据库加载自定义策略
        supabase = self._get_supabase()
        if supabase:
            try:
                response = supabase.table('external_strategies').select('*').execute()
                for row in (response.data or []):
                    strategy = ExternalStrategy(
                        id=row['id'],
                        name=row['name'],
                        author=row['author'],
                        source=StrategySource(row['source']),
                        category=StrategyCategory(row['category']),
                        description=row.get('description', ''),
                        url=row.get('url', ''),
                        entry_rules=row.get('entry_rules', ''),
                        exit_rules=row.get('exit_rules', ''),
                        indicators=row.get('indicators', []),
                        claimed_win_rate=row.get('claimed_win_rate', 0),
                        verified_win_rate=row.get('verified_win_rate'),
                        verified_return=row.get('verified_return'),
                        tags=row.get('tags', [])
                    )
                    strategies.append(strategy)
            except:
                pass
        
        return strategies
    
    def add_strategy(self, strategy: ExternalStrategy) -> bool:
        """添加策略"""
        supabase = self._get_supabase()
        if not supabase:
            return False
        
        try:
            strategy.created_at = datetime.now().isoformat()
            supabase.table('external_strategies').upsert(
                strategy.to_dict(),
                on_conflict='id'
            ).execute()
            return True
        except Exception as e:
            print(f"Failed to add strategy: {e}")
            return False
    
    # === 博主管理 ===
    
    def get_all_authors(self) -> List[StrategyAuthor]:
        """获取所有博主"""
        authors = []
        
        # 雪球博主
        authors.extend(self.xq_scraper.get_famous_authors())
        
        # 从数据库加载
        supabase = self._get_supabase()
        if supabase:
            try:
                response = supabase.table('strategy_authors').select('*').execute()
                for row in (response.data or []):
                    author = StrategyAuthor(
                        id=row['id'],
                        name=row['name'],
                        platform=StrategySource(row['platform']),
                        profile_url=row.get('profile_url', ''),
                        followers=row.get('followers', 0),
                        description=row.get('description', ''),
                        specialty=row.get('specialty', ''),
                        total_picks=row.get('total_picks', 0),
                        win_rate=row.get('win_rate', 0),
                        avg_return=row.get('avg_return', 0)
                    )
                    authors.append(author)
            except:
                pass
        
        return authors
    
    def add_author(self, author: StrategyAuthor) -> bool:
        """添加博主"""
        supabase = self._get_supabase()
        if not supabase:
            return False
        
        try:
            author.created_at = datetime.now().isoformat()
            supabase.table('strategy_authors').upsert(
                author.to_dict(),
                on_conflict='id'
            ).execute()
            return True
        except Exception as e:
            print(f"Failed to add author: {e}")
            return False
    
    # === 选股追踪 ===
    
    def record_author_pick(self, pick: AuthorPick) -> bool:
        """记录博主选股"""
        supabase = self._get_supabase()
        if not supabase:
            return False
        
        try:
            pick.created_at = datetime.now().isoformat()
            supabase.table('author_picks').upsert(
                pick.to_dict(),
                on_conflict='author_id,symbol,pick_date'
            ).execute()
            return True
        except Exception as e:
            print(f"Failed to record pick: {e}")
            return False
    
    def get_author_picks(self, author_id: str = None, 
                         days: int = 30) -> List[Dict]:
        """获取博主选股记录"""
        supabase = self._get_supabase()
        if not supabase:
            return []
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = supabase.table('author_picks').select('*').gte('pick_date', start_date)
            
            if author_id:
                query = query.eq('author_id', author_id)
            
            response = query.order('pick_date', desc=True).execute()
            return response.data or []
        except:
            return []
    
    def update_pick_returns(self, author_id: str, symbol: str, 
                            pick_date: str, returns: Dict) -> bool:
        """更新选股收益"""
        supabase = self._get_supabase()
        if not supabase:
            return False
        
        try:
            supabase.table('author_picks').update(returns).match({
                'author_id': author_id,
                'symbol': symbol,
                'pick_date': pick_date
            }).execute()
            return True
        except:
            return False
    
    # === 统计分析 ===
    
    def get_author_performance(self, author_id: str) -> Dict:
        """获取博主表现统计"""
        picks = self.get_author_picks(author_id, days=365)
        
        if not picks:
            return {}
        
        df = pd.DataFrame(picks)
        
        # 过滤有收益数据的
        df_valid = df[df['return_d5'].notna()]
        
        if df_valid.empty:
            return {'total_picks': len(picks), 'tracked_picks': 0}
        
        returns = df_valid['return_d5']
        
        return {
            'total_picks': len(picks),
            'tracked_picks': len(df_valid),
            'avg_return_d5': round(returns.mean(), 2),
            'win_rate': round((returns > 0).mean() * 100, 1),
            'best_pick': df_valid.loc[returns.idxmax()].to_dict() if len(returns) > 0 else None,
            'worst_pick': df_valid.loc[returns.idxmin()].to_dict() if len(returns) > 0 else None,
            'total_return': round(returns.sum(), 2)
        }
    
    def get_author_leaderboard(self, top_n: int = 10) -> List[Dict]:
        """博主排行榜"""
        authors = self.get_all_authors()
        
        leaderboard = []
        
        for author in authors:
            perf = self.get_author_performance(author.id)
            if perf.get('tracked_picks', 0) >= 5:  # 至少5个有效选股
                leaderboard.append({
                    'author': author.name,
                    'platform': author.platform.value if isinstance(author.platform, StrategySource) else author.platform,
                    'specialty': author.specialty,
                    'tracked_picks': perf['tracked_picks'],
                    'win_rate': perf.get('win_rate', 0),
                    'avg_return': perf.get('avg_return_d5', 0)
                })
        
        # 按胜率排序
        leaderboard.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return leaderboard[:top_n]
    
    def get_strategy_comparison(self) -> pd.DataFrame:
        """策略对比表"""
        strategies = self.get_all_strategies()
        
        data = []
        for s in strategies:
            data.append({
                '策略名称': s.name,
                '来源': s.source.value if isinstance(s.source, StrategySource) else s.source,
                '类别': s.category.value if isinstance(s.category, StrategyCategory) else s.category,
                '声称胜率': f"{s.claimed_win_rate}%",
                '验证胜率': f"{s.verified_win_rate}%" if s.verified_win_rate else "未验证",
                '主要指标': ', '.join(s.indicators[:3]) if s.indicators else ''
            })
        
        return pd.DataFrame(data)
    
    # === 策略验证 ===
    
    def backtest_strategy(self, strategy: ExternalStrategy, 
                          symbols: List[str] = None,
                          days: int = 365) -> Dict:
        """
        回测外部策略
        
        注意: 这是简化版回测,主要验证策略概念是否有效
        """
        # 这里需要根据 strategy.entry_rules 和 indicators 
        # 实现具体的回测逻辑
        # 目前返回占位结果
        
        return {
            'strategy': strategy.name,
            'period': f"{days} days",
            'status': 'pending',
            'message': '完整回测需要实现具体指标逻辑'
        }


# === 便捷函数 ===

def get_tradingview_strategies() -> List[Dict]:
    """获取 TradingView 热门策略"""
    aggregator = StrategyAggregator()
    strategies = aggregator.tv_scraper.get_popular_strategies()
    return [s.to_dict() for s in strategies]


def get_author_leaderboard(top_n: int = 10) -> List[Dict]:
    """获取博主排行榜"""
    aggregator = StrategyAggregator()
    return aggregator.get_author_leaderboard(top_n)


def add_manual_author(name: str, platform: str, specialty: str = "") -> bool:
    """手动添加博主"""
    aggregator = StrategyAggregator()
    
    author = StrategyAuthor(
        id=f"manual_{name.lower().replace(' ', '_')}",
        name=name,
        platform=StrategySource(platform) if platform in [s.value for s in StrategySource] else StrategySource.MANUAL,
        specialty=specialty
    )
    
    return aggregator.add_author(author)


def record_blogger_pick(author_name: str, symbol: str, 
                        action: str = 'buy', reasoning: str = "") -> bool:
    """记录博主选股"""
    aggregator = StrategyAggregator()
    
    # 查找或创建作者
    authors = aggregator.get_all_authors()
    author_id = None
    
    for a in authors:
        if a.name == author_name:
            author_id = a.id
            break
    
    if not author_id:
        author_id = f"manual_{author_name.lower().replace(' ', '_')}"
    
    pick = AuthorPick(
        author_id=author_id,
        symbol=symbol.upper(),
        pick_date=datetime.now().strftime('%Y-%m-%d'),
        action=action,
        reasoning=reasoning
    )
    
    return aggregator.record_author_pick(pick)


if __name__ == "__main__":
    print("🎯 Strategy Aggregator Test")
    
    aggregator = StrategyAggregator()
    
    # 获取 TradingView 策略
    print("\n📊 TradingView Popular Strategies:")
    strategies = aggregator.tv_scraper.get_popular_strategies()
    for s in strategies[:5]:
        print(f"  - {s.name} ({s.category.value}): {s.claimed_win_rate}% win rate")
    
    # 获取博主
    print("\n👤 Famous Authors:")
    authors = aggregator.get_all_authors()
    for a in authors[:5]:
        platform = a.platform.value if isinstance(a.platform, StrategySource) else a.platform
        print(f"  - {a.name} ({platform}): {a.specialty}")
