#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻数据模型
定义新闻事件、影响评分等核心数据结构
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict


class EventType(Enum):
    """新闻事件类型"""
    EARNINGS = "earnings"           # 财报/业绩
    GUIDANCE = "guidance"           # 业绩指引
    ANALYST = "analyst"             # 分析师评级
    INSIDER = "insider"             # 内部交易
    M_AND_A = "m_and_a"             # 并购重组
    PRODUCT = "product"             # 产品/服务
    LEGAL = "legal"                 # 法律监管
    MACRO = "macro"                 # 宏观政策
    TECHNICAL = "technical"         # 技术突破
    PARTNERSHIP = "partnership"     # 合作/协议
    DIVIDEND = "dividend"           # 分红派息
    STOCK_SPLIT = "stock_split"     # 股票拆分
    OFFERING = "offering"           # 增发/配股
    BANKRUPTCY = "bankruptcy"       # 破产/重组
    EXECUTIVE = "executive"         # 高管变动
    OTHER = "other"                 # 其他

    @property
    def chinese_name(self) -> str:
        names = {
            EventType.EARNINGS: "📊 财报业绩",
            EventType.GUIDANCE: "🎯 业绩指引",
            EventType.ANALYST: "📈 分析师评级",
            EventType.INSIDER: "👔 内部交易",
            EventType.M_AND_A: "🤝 并购重组",
            EventType.PRODUCT: "📦 产品服务",
            EventType.LEGAL: "⚖️ 法律监管",
            EventType.MACRO: "🌍 宏观政策",
            EventType.TECHNICAL: "🔬 技术突破",
            EventType.PARTNERSHIP: "🤝 合作协议",
            EventType.DIVIDEND: "💰 分红派息",
            EventType.STOCK_SPLIT: "✂️ 股票拆分",
            EventType.OFFERING: "📤 增发配股",
            EventType.BANKRUPTCY: "⚠️ 破产重组",
            EventType.EXECUTIVE: "👤 高管变动",
            EventType.OTHER: "📰 其他"
        }
        return names.get(self, "📰 其他")


class Sentiment(Enum):
    """新闻情感"""
    VERY_BULLISH = "very_bullish"   # 强烈利好
    BULLISH = "bullish"             # 利好
    NEUTRAL = "neutral"             # 中性
    BEARISH = "bearish"             # 利空
    VERY_BEARISH = "very_bearish"   # 强烈利空
    
    @property
    def score(self) -> float:
        """情感分数 -1.0 到 +1.0"""
        scores = {
            Sentiment.VERY_BULLISH: 1.0,
            Sentiment.BULLISH: 0.5,
            Sentiment.NEUTRAL: 0.0,
            Sentiment.BEARISH: -0.5,
            Sentiment.VERY_BEARISH: -1.0
        }
        return scores.get(self, 0.0)
    
    @property
    def emoji(self) -> str:
        emojis = {
            Sentiment.VERY_BULLISH: "🔥",
            Sentiment.BULLISH: "📈",
            Sentiment.NEUTRAL: "➖",
            Sentiment.BEARISH: "📉",
            Sentiment.VERY_BEARISH: "💥"
        }
        return emojis.get(self, "➖")


class TimeHorizon(Enum):
    """影响时间范围"""
    IMMEDIATE = "immediate"   # 即时 (盘中)
    SHORT = "short"           # 短期 (1-5天)
    MEDIUM = "medium"         # 中期 (1-4周)
    LONG = "long"             # 长期 (1个月+)


@dataclass
class NewsEvent:
    """新闻事件"""
    id: str                                 # 唯一ID
    symbol: str                             # 股票代码
    title: str                              # 新闻标题
    source: str                             # 来源
    url: str                                # 链接
    published_at: datetime                  # 发布时间
    
    # 分类结果
    event_type: EventType = EventType.OTHER
    sentiment: Sentiment = Sentiment.NEUTRAL
    
    # 元数据
    summary: str = ""                       # 摘要
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # 实体识别
    
    # 处理状态
    is_classified: bool = False
    classified_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'title': self.title,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'event_type': self.event_type.value,
            'sentiment': self.sentiment.value,
            'summary': self.summary,
            'keywords': self.keywords,
            'is_classified': self.is_classified
        }


@dataclass
class NewsImpact:
    """新闻影响评估"""
    news_id: str                            # 关联的新闻ID
    symbol: str                             # 股票代码
    
    # 预测
    expected_impact_pct: float = 0.0        # 预期价格影响 (%)
    confidence: float = 0.0                 # 置信度 0-100
    time_horizon: TimeHorizon = TimeHorizon.SHORT
    urgency: int = 1                        # 紧急程度 1-5
    
    # 来源质量
    source_credibility: float = 0.5         # 来源可信度 0-1
    
    # 信号影响
    signal_multiplier: float = 1.0          # 信号加权因子
    should_alert: bool = False              # 是否需要推送提醒
    alert_priority: int = 3                 # 提醒优先级 1-5
    
    # 追踪
    created_at: datetime = field(default_factory=datetime.now)
    
    # 实际表现 (用于回测)
    actual_d1_return: Optional[float] = None
    actual_d3_return: Optional[float] = None
    actual_d5_return: Optional[float] = None
    actual_d10_return: Optional[float] = None
    
    def prediction_accuracy(self) -> Optional[float]:
        """计算预测准确度"""
        if self.actual_d5_return is None:
            return None
        
        # 方向是否正确
        direction_correct = (self.expected_impact_pct > 0) == (self.actual_d5_return > 0)
        
        # 幅度误差
        if abs(self.expected_impact_pct) > 0:
            magnitude_error = abs(self.actual_d5_return - self.expected_impact_pct) / abs(self.expected_impact_pct)
            magnitude_score = max(0, 1 - magnitude_error)
        else:
            magnitude_score = 1.0 if abs(self.actual_d5_return) < 1 else 0.5
        
        return (0.6 if direction_correct else 0.0) + (0.4 * magnitude_score)
    
    def to_dict(self) -> Dict:
        return {
            'news_id': self.news_id,
            'symbol': self.symbol,
            'expected_impact_pct': self.expected_impact_pct,
            'confidence': self.confidence,
            'time_horizon': self.time_horizon.value,
            'urgency': self.urgency,
            'source_credibility': self.source_credibility,
            'signal_multiplier': self.signal_multiplier,
            'should_alert': self.should_alert,
            'created_at': self.created_at.isoformat(),
            'actual_d1_return': self.actual_d1_return,
            'actual_d5_return': self.actual_d5_return
        }


@dataclass
class NewsDigest:
    """新闻摘要 - 用于信号增强"""
    symbol: str
    period: str                             # 'today' / 'week'
    
    total_news_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    
    dominant_sentiment: Sentiment = Sentiment.NEUTRAL
    avg_expected_impact: float = 0.0
    
    key_events: List[str] = field(default_factory=list)
    signal_adjustment: float = 1.0          # 信号调整因子
    
    def sentiment_ratio(self) -> float:
        """情感比率 (-1 到 +1)"""
        total = self.bullish_count + self.bearish_count
        if total == 0:
            return 0.0
        return (self.bullish_count - self.bearish_count) / total
