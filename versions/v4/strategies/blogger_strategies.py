#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知名博主策略库
内置多位知名交易者的选股策略，可回测对比
"""
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from strategies.decision_system import BaseStrategy, StrategyPick


@dataclass
class BloggerInfo:
    """博主信息"""
    name: str
    name_cn: str       # 中文名
    platform: str      # 雪球/Twitter/公众号
    style: str         # 短线/波段/价值
    description: str
    language: str = "en"  # en/cn
    url: str = ""
    twitter: str = ""
    weibo: str = ""
    xueqiu: str = ""   # 雪球
    youtube: str = ""
    books: List[str] = None  # 著作

    def __post_init__(self):
        if self.books is None:
            self.books = []


class MarkMinerviniStrategy(BaseStrategy):
    """马克·米纳维尼 (Mark Minervini)
    
    美股冠军交易员，《股票魔法师》作者
    VCP模式: 波动收缩形态 + 突破买入
    
    核心规则:
    1. 股价在200日均线以上
    2. 52周高点附近 (在10%以内)
    3. 成交量收缩后放量突破
    4. 相对强度高
    """
    
    def __init__(self):
        super().__init__(
            name="米纳维尼VCP",
            description="波动收缩突破，美股冠军策略",
            icon="🏆"
        )
        self.blogger = BloggerInfo(
            name="Mark Minervini",
            name_cn="马克·米纳维尼",
            platform="Twitter/Website",
            style="趋势突破",
            description="美股交易冠军，VCP模式创始人，连续多年USIC冠军",
            language="en",
            url="https://www.minervini.com/",
            twitter="https://twitter.com/markminervini",
            youtube="https://www.youtube.com/@MarkMinervini",
            books=["Trade Like a Stock Market Wizard", "Think & Trade Like a Champion"]
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        
        # VCP 条件: 强势 + 波动收缩
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 75]
        if vol_col in filtered.columns:
            filtered = filtered[filtered[vol_col] <= 0.25]  # 波动收缩
        if adx_col in filtered.columns:
            filtered = filtered[filtered[adx_col] >= 20]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 5]
        
        if filtered.empty:
            return picks
        
        # 评分
        filtered['score'] = 0
        if blue_col in filtered.columns:
            filtered['score'] += filtered[blue_col].fillna(0) * 0.5
        if adx_col in filtered.columns:
            filtered['score'] += filtered[adx_col].fillna(0) * 0.3
        if vol_col in filtered.columns:
            filtered['score'] += (1 - filtered[vol_col].fillna(0.5)) * 20
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.02)
            stop_loss = self.calculate_stop_loss(price, vol * 0.8)
            take_profit = self.calculate_take_profit(price, 3.0, stop_loss)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"VCP模式 BLUE={row.get(blue_col,0):.0f}",
                confidence="高" if row['score'] > 80 else "中"
            ))
        
        return picks


class WilliamONeilStrategy(BaseStrategy):
    """威廉·欧奈尔 (William O'Neil)
    
    《笑傲股市》作者，CANSLIM系统创始人
    IBD创始人
    
    核心规则:
    1. C - 当季每股收益增长
    2. A - 年度收益增长
    3. N - 新产品/新高
    4. S - 供需关系 (股本小)
    5. L - 领涨股
    6. I - 机构持股
    7. M - 市场方向
    """
    
    def __init__(self):
        super().__init__(
            name="欧奈尔CANSLIM",
            description="成长股投资，杯柄形态",
            icon="📈"
        )
        self.blogger = BloggerInfo(
            name="William O'Neil",
            name_cn="威廉·欧奈尔",
            platform="IBD (Investor's Business Daily)",
            style="成长投资",
            description="CANSLIM系统创始人，IBD创始人",
            language="en",
            url="https://www.investors.com/",
            twitter="https://twitter.com/IBDinvestors",
            youtube="https://www.youtube.com/@investorsbusinessdaily",
            books=["How to Make Money in Stocks", "笑傲股市"]
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        blue_weekly = 'blue_weekly' if 'blue_weekly' in df.columns else 'Blue_Weekly'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # CANSLIM 简化: 强势领涨 + 多周期确认
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 80]
        if blue_weekly in filtered.columns:
            filtered = filtered[filtered[blue_weekly] >= 50]
        if adx_col in filtered.columns:
            filtered = filtered[filtered[adx_col] >= 25]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 10]  # 机构关注
        
        if filtered.empty:
            return picks
        
        filtered['score'] = 0
        if blue_col in filtered.columns:
            filtered['score'] += filtered[blue_col].fillna(0) * 0.4
        if blue_weekly in filtered.columns:
            filtered['score'] += filtered[blue_weekly].fillna(0) * 0.3
        if adx_col in filtered.columns:
            filtered['score'] += filtered[adx_col].fillna(0) * 0.3
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.02)
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 2.5, stop_loss)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"CANSLIM 日={row.get(blue_col,0):.0f} 周={row.get(blue_weekly,0):.0f}",
                confidence="高"
            ))
        
        return picks


class JesseLivermoreStrategy(BaseStrategy):
    """杰西·利弗莫尔 (Jesse Livermore)
    
    传奇交易员，《股票大作手回忆录》主角
    趋势跟踪 + 关键点突破
    
    核心规则:
    1. 顺势交易，不抄底
    2. 等待关键点突破
    3. 金字塔加仓
    4. 严格止损
    """
    
    def __init__(self):
        super().__init__(
            name="利弗莫尔关键点",
            description="趋势跟踪，关键点突破",
            icon="📜"
        )
        self.blogger = BloggerInfo(
            name="Jesse Livermore",
            name_cn="杰西·利弗莫尔",
            platform="历史传奇",
            style="趋势跟踪",
            description="华尔街传奇交易员，曾多次做空市场获利",
            language="en",
            url="https://en.wikipedia.org/wiki/Jesse_Livermore",
            books=["Reminerta of a Stock Operator (股票大作手回忆录)", "How to Trade in Stocks"]
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # 关键点: 极强信号
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 90]  # 极强
        if adx_col in filtered.columns:
            filtered = filtered[filtered[adx_col] >= 30]  # 强趋势
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 3]
        
        if filtered.empty:
            # 放宽条件
            filtered = df.copy()
            if blue_col in filtered.columns:
                filtered = filtered[filtered[blue_col] >= 85]
        
        if filtered.empty:
            return picks
        
        filtered['score'] = 0
        if blue_col in filtered.columns:
            filtered['score'] = filtered[blue_col].fillna(0)
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.02)
            stop_loss = self.calculate_stop_loss(price, vol * 0.7)  # 严格止损
            take_profit = self.calculate_take_profit(price, 4.0, stop_loss)  # 让利润奔跑
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"关键点突破 BLUE={row.get(blue_col,0):.0f}",
                confidence="高" if row.get(adx_col, 0) > 35 else "中"
            ))
        
        return picks


class TaoBoStrategy(BaseStrategy):
    """陶博士 (淘股吧/雪球知名博主)
    
    A股短线游资风格
    龙头战法 + 打板模式
    
    核心规则:
    1. 抓龙头股，拒绝跟风
    2. 涨停板模式
    3. 高换手 + 高关注
    4. 快进快出
    """
    
    def __init__(self):
        super().__init__(
            name="龙头战法",
            description="A股游资风格，抓龙头",
            icon="🐉"
        )
        self.blogger = BloggerInfo(
            name="Longtou School",
            name_cn="龙头战法派",
            platform="淘股吧/雪球",
            style="短线游资",
            description="A股龙头战法，专注涨停板和强势股",
            language="cn",
            url="https://www.taoguba.com.cn/",
            xueqiu="https://xueqiu.com/",
            books=["龙头战法", "涨停板战法"]
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # 龙头条件: 极强 + 高换手
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 85]
        if turnover_col in filtered.columns:
            # 取成交额前20%
            threshold = df[turnover_col].quantile(0.8) if turnover_col in df.columns else 0
            filtered = filtered[filtered[turnover_col] >= threshold]
        
        if filtered.empty:
            return picks
        
        # 评分: BLUE 为主
        filtered['score'] = 0
        if blue_col in filtered.columns:
            filtered['score'] = filtered[blue_col].fillna(0)
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.03)
            stop_loss = self.calculate_stop_loss(price, vol * 0.6)  # 快速止损
            take_profit = self.calculate_take_profit(price, 2.0, stop_loss)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"龙头 BLUE={row.get(blue_col,0):.0f} 换手高",
                confidence="中"  # 短线风险较高
            ))
        
        return picks


class BuffettValueStrategy(BaseStrategy):
    """巴菲特价值投资 (Warren Buffett)
    
    价值投资之父
    买入并持有优质公司
    
    核心规则:
    1. 护城河/竞争优势
    2. 优秀管理层
    3. 价格低于内在价值
    4. 长期持有
    """
    
    def __init__(self):
        super().__init__(
            name="巴菲特价值",
            description="价值投资，长期持有",
            icon="🦅"
        )
        self.blogger = BloggerInfo(
            name="Warren Buffett",
            name_cn="沃伦·巴菲特",
            platform="Berkshire Hathaway",
            style="价值投资",
            description="价值投资之父，伯克希尔·哈撒韦CEO",
            language="en",
            url="https://www.berkshirehathaway.com/",
            twitter="https://twitter.com/WarrenBuffett",
            books=["巴菲特致股东的信", "滚雪球"]
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        
        # 价值条件: 稳定 + 高流动性
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 20]  # 大盘股
        if vol_col in filtered.columns:
            filtered = filtered[filtered[vol_col] <= 0.2]  # 低波动
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 50]  # 有信号
        
        if filtered.empty:
            return picks
        
        # 评分: 稳定性优先
        filtered['score'] = 0
        if turnover_col in filtered.columns:
            max_t = filtered[turnover_col].max()
            if max_t > 0:
                filtered['score'] += filtered[turnover_col] / max_t * 50
        if vol_col in filtered.columns:
            filtered['score'] += (1 - filtered[vol_col]) * 30
        if blue_col in filtered.columns:
            filtered['score'] += filtered[blue_col].fillna(0) * 0.2
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.015)
            stop_loss = self.calculate_stop_loss(price, vol * 2)  # 宽松止损
            take_profit = self.calculate_take_profit(price, 1.5, stop_loss)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"价值蓝筹 成交={row.get(turnover_col,0):.0f}M",
                confidence="高"
            ))
        
        return picks


# 博主策略注册表
BLOGGER_STRATEGIES = {
    'minervini': MarkMinerviniStrategy,
    'oneil': WilliamONeilStrategy,
    'livermore': JesseLivermoreStrategy,
    'taobo': TaoBoStrategy,
    'buffett': BuffettValueStrategy,
}

# 知名博主/交易员链接库 (不提供策略，仅供学习)
FAMOUS_TRADERS = [
    # === 英文博主 ===
    {
        "name": "Mark Minervini",
        "name_cn": "马克·米纳维尼",
        "platform": "Twitter/Website",
        "language": "en",
        "style": "趋势突破/VCP",
        "links": {
            "twitter": "https://twitter.com/markminervini",
            "website": "https://www.minervini.com/",
            "youtube": "https://www.youtube.com/@MarkMinervini"
        },
        "books": ["Trade Like a Stock Market Wizard", "Think & Trade Like a Champion"]
    },
    {
        "name": "William O'Neil",
        "name_cn": "威廉·欧奈尔",
        "platform": "IBD",
        "language": "en",
        "style": "CANSLIM成长股",
        "links": {
            "website": "https://www.investors.com/",
            "twitter": "https://twitter.com/IBDinvestors"
        },
        "books": ["How to Make Money in Stocks (笑傲股市)"]
    },
    {
        "name": "Dan Zanger",
        "name_cn": "丹·赞格",
        "platform": "Twitter/Website",
        "language": "en",
        "style": "图表形态",
        "links": {
            "twitter": "https://twitter.com/DanZanger",
            "website": "https://www.chartpattern.com/"
        },
        "books": []
    },
    {
        "name": "Qullamaggie (Kristjan Kullamägi)",
        "name_cn": "库拉玛吉",
        "platform": "Twitter/YouTube",
        "language": "en",
        "style": "动量交易",
        "links": {
            "twitter": "https://twitter.com/qaboratory",
            "youtube": "https://www.youtube.com/@Qullamaggie"
        },
        "books": []
    },
    {
        "name": "Oliver Kell",
        "name_cn": "奥利弗·凯尔",
        "platform": "Twitter",
        "language": "en",
        "style": "USIC冠军/趋势",
        "links": {
            "twitter": "https://twitter.com/Oliver_Kell"
        },
        "books": []
    },
    {
        "name": "Richard Moglen",
        "name_cn": "理查德·莫格伦",
        "platform": "Twitter",
        "language": "en",
        "style": "成长股/动量",
        "links": {
            "twitter": "https://twitter.com/RichardMoglen"
        },
        "books": []
    },
    
    # === 中文博主 ===
    {
        "name": "缠中说禅",
        "name_cn": "缠中说禅",
        "platform": "博客/雪球",
        "language": "cn",
        "style": "缠论",
        "links": {
            "blog": "http://blog.sina.com.cn/chzhshch"
        },
        "books": ["缠论108课"]
    },
    {
        "name": "淘股吧养家心法",
        "name_cn": "养家心法",
        "platform": "淘股吧",
        "language": "cn",
        "style": "短线游资",
        "links": {
            "taoguba": "https://www.taoguba.com.cn/"
        },
        "books": ["养家心法"]
    },
    {
        "name": "职业炒手",
        "name_cn": "职业炒手",
        "platform": "淘股吧",
        "language": "cn",
        "style": "短线接力",
        "links": {
            "taoguba": "https://www.taoguba.com.cn/"
        },
        "books": []
    },
    {
        "name": "雪球-不明真相的群众",
        "name_cn": "方三文",
        "platform": "雪球",
        "language": "cn",
        "style": "价值投资",
        "links": {
            "xueqiu": "https://xueqiu.com/u/4471316972"
        },
        "books": ["老二非死不可"]
    },
    {
        "name": "雪球-释老毛",
        "name_cn": "释老毛",
        "platform": "雪球",
        "language": "cn",
        "style": "价值投资",
        "links": {
            "xueqiu": "https://xueqiu.com/u/1460392750"
        },
        "books": []
    },
    {
        "name": "雪球-DAVID自由之路",
        "name_cn": "DAVID自由之路",
        "platform": "雪球",
        "language": "cn",
        "style": "可转债/套利",
        "links": {
            "xueqiu": "https://xueqiu.com/u/5819606767"
        },
        "books": ["可转债投资魔法书"]
    },
    {
        "name": "雪球-持有封基",
        "name_cn": "持有封基",
        "platform": "雪球",
        "language": "cn",
        "style": "封基/套利",
        "links": {
            "xueqiu": "https://xueqiu.com/u/2552425919"
        },
        "books": []
    },
    {
        "name": "微博-天地侠影",
        "name_cn": "天地侠影",
        "platform": "微博/公众号",
        "language": "cn",
        "style": "周期股",
        "links": {
            "weibo": "https://weibo.com/"
        },
        "books": []
    },
]


def get_blogger_strategy(name: str) -> Optional[BaseStrategy]:
    """获取博主策略实例"""
    cls = BLOGGER_STRATEGIES.get(name.lower())
    if cls:
        return cls()
    return None


def list_blogger_strategies() -> List[Dict]:
    """列出所有博主策略"""
    result = []
    for key, cls in BLOGGER_STRATEGIES.items():
        instance = cls()
        blogger_dict = {}
        if hasattr(instance, 'blogger'):
            b = instance.blogger
            blogger_dict = {
                'name': b.name,
                'name_cn': b.name_cn,
                'platform': b.platform,
                'style': b.style,
                'description': b.description,
                'language': b.language,
                'url': b.url,
                'twitter': b.twitter,
                'youtube': b.youtube,
                'xueqiu': b.xueqiu,
                'books': b.books
            }
        result.append({
            'key': key,
            'name': instance.name,
            'icon': instance.icon,
            'description': instance.description,
            'blogger': blogger_dict
        })
    return result


def get_famous_traders(language: str = None) -> List[Dict]:
    """获取知名交易员列表
    
    Args:
        language: 'en' for English, 'cn' for Chinese, None for all
    """
    if language:
        return [t for t in FAMOUS_TRADERS if t['language'] == language]
    return FAMOUS_TRADERS

