#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
影响评分器
评估新闻对股价的潜在影响
"""
import logging
from datetime import datetime
from typing import Dict, Optional, List

from .models import (
    NewsEvent, NewsImpact, EventType, Sentiment, TimeHorizon, NewsDigest
)

logger = logging.getLogger(__name__)


# 事件类型的基础影响系数
EVENT_IMPACT_BASE = {
    EventType.EARNINGS: {
        'base_impact': 5.0,      # 基础影响 %
        'time_horizon': TimeHorizon.IMMEDIATE,
        'urgency': 5,
        'volatility_mult': 1.5   # 波动放大系数
    },
    EventType.GUIDANCE: {
        'base_impact': 4.0,
        'time_horizon': TimeHorizon.IMMEDIATE,
        'urgency': 5,
        'volatility_mult': 1.3
    },
    EventType.ANALYST: {
        'base_impact': 2.5,
        'time_horizon': TimeHorizon.SHORT,
        'urgency': 3,
        'volatility_mult': 1.0
    },
    EventType.M_AND_A: {
        'base_impact': 10.0,
        'time_horizon': TimeHorizon.IMMEDIATE,
        'urgency': 5,
        'volatility_mult': 2.0
    },
    EventType.PRODUCT: {
        'base_impact': 3.0,
        'time_horizon': TimeHorizon.MEDIUM,
        'urgency': 3,
        'volatility_mult': 1.2
    },
    EventType.LEGAL: {
        'base_impact': 4.0,
        'time_horizon': TimeHorizon.SHORT,
        'urgency': 4,
        'volatility_mult': 1.5
    },
    EventType.INSIDER: {
        'base_impact': 2.0,
        'time_horizon': TimeHorizon.MEDIUM,
        'urgency': 2,
        'volatility_mult': 0.8
    },
    EventType.MACRO: {
        'base_impact': 3.0,
        'time_horizon': TimeHorizon.MEDIUM,
        'urgency': 3,
        'volatility_mult': 1.0
    },
    EventType.PARTNERSHIP: {
        'base_impact': 2.5,
        'time_horizon': TimeHorizon.SHORT,
        'urgency': 3,
        'volatility_mult': 1.0
    },
    EventType.DIVIDEND: {
        'base_impact': 1.5,
        'time_horizon': TimeHorizon.SHORT,
        'urgency': 2,
        'volatility_mult': 0.7
    },
    EventType.OFFERING: {
        'base_impact': 3.5,
        'time_horizon': TimeHorizon.IMMEDIATE,
        'urgency': 4,
        'volatility_mult': 1.3
    },
    EventType.BANKRUPTCY: {
        'base_impact': 20.0,
        'time_horizon': TimeHorizon.IMMEDIATE,
        'urgency': 5,
        'volatility_mult': 3.0
    },
    EventType.OTHER: {
        'base_impact': 1.0,
        'time_horizon': TimeHorizon.MEDIUM,
        'urgency': 1,
        'volatility_mult': 0.5
    }
}

# 来源可信度
SOURCE_CREDIBILITY = {
    # 高可信度
    'Reuters': 0.95,
    'Bloomberg': 0.95,
    'WSJ': 0.90,
    'Wall Street Journal': 0.90,
    'CNBC': 0.85,
    'Financial Times': 0.90,
    'SEC': 0.99,
    'PR Newswire': 0.80,
    'Business Wire': 0.80,
    '新华社': 0.95,
    '财新': 0.85,
    '证券时报': 0.80,
    '上海证券报': 0.80,
    
    # 中等可信度
    'Seeking Alpha': 0.60,
    'Yahoo Finance': 0.65,
    'MarketWatch': 0.70,
    'Benzinga': 0.60,
    '东方财富': 0.65,
    '同花顺': 0.60,
    
    # 默认
    'default': 0.50
}


class ImpactScorer:
    """影响评分器"""
    
    def __init__(self, alert_threshold: float = 3.0):
        self.alert_threshold = alert_threshold  # 触发提醒的影响阈值
    
    def score(self, event: NewsEvent, llm_prediction: Dict = None) -> NewsImpact:
        """评估新闻影响"""
        
        # 1. 获取事件类型的基础参数
        event_params = EVENT_IMPACT_BASE.get(
            event.event_type, 
            EVENT_IMPACT_BASE[EventType.OTHER]
        )
        
        base_impact = event_params['base_impact']
        time_horizon = event_params['time_horizon']
        urgency = event_params['urgency']
        volatility_mult = event_params['volatility_mult']
        
        # 2. 情感调整
        sentiment_mult = event.sentiment.score  # -1.0 到 +1.0
        
        # 3. 计算预期影响
        expected_impact = base_impact * sentiment_mult * volatility_mult
        
        # 4. 如果有 LLM 预测，使用其结果
        if llm_prediction and 'expected_impact_pct' in llm_prediction:
            try:
                llm_impact = float(llm_prediction['expected_impact_pct'])
                # 加权平均: 60% LLM + 40% 规则
                expected_impact = 0.6 * llm_impact + 0.4 * expected_impact
            except:
                pass
        
        # 5. 置信度计算
        confidence = 50.0  # 基础置信度
        
        # 来源可信度加成
        source_cred = self._get_source_credibility(event.source)
        confidence += source_cred * 30
        
        # 事件类型加成 (财报/并购等更容易预测)
        if event.event_type in [EventType.EARNINGS, EventType.M_AND_A, EventType.DIVIDEND]:
            confidence += 15
        
        # LLM 预测加成
        if llm_prediction and 'confidence' in llm_prediction:
            try:
                llm_conf = float(llm_prediction['confidence'])
                confidence = 0.5 * confidence + 0.5 * llm_conf
            except:
                pass
        
        confidence = min(95, max(10, confidence))
        
        # 6. 信号乘数 (用于增强/减弱交易信号)
        signal_multiplier = 1.0
        if abs(expected_impact) > 5:
            signal_multiplier = 1.0 + (abs(expected_impact) - 5) * 0.05
        if sentiment_mult < 0:
            signal_multiplier = 2 - signal_multiplier  # 反转
        
        # 7. 是否触发提醒
        should_alert = abs(expected_impact) >= self.alert_threshold
        alert_priority = min(5, max(1, int(abs(expected_impact) / 2)))
        
        return NewsImpact(
            news_id=event.id,
            symbol=event.symbol,
            expected_impact_pct=round(expected_impact, 2),
            confidence=round(confidence, 1),
            time_horizon=time_horizon,
            urgency=urgency,
            source_credibility=source_cred,
            signal_multiplier=round(signal_multiplier, 2),
            should_alert=should_alert,
            alert_priority=alert_priority
        )
    
    def _get_source_credibility(self, source: str) -> float:
        """获取来源可信度"""
        for key, cred in SOURCE_CREDIBILITY.items():
            if key.lower() in source.lower():
                return cred
        return SOURCE_CREDIBILITY['default']
    
    def create_digest(self, events: List[NewsEvent], 
                      impacts: List[NewsImpact], 
                      symbol: str) -> NewsDigest:
        """创建新闻摘要"""
        if not events:
            return NewsDigest(symbol=symbol, period='today')
        
        # 统计情感
        bullish = sum(1 for e in events if e.sentiment.score > 0)
        bearish = sum(1 for e in events if e.sentiment.score < 0)
        neutral = len(events) - bullish - bearish
        
        # 主导情感
        if bullish > bearish and bullish > neutral:
            dominant = Sentiment.BULLISH
        elif bearish > bullish and bearish > neutral:
            dominant = Sentiment.BEARISH
        else:
            dominant = Sentiment.NEUTRAL
        
        # 平均影响
        if impacts:
            avg_impact = sum(i.expected_impact_pct for i in impacts) / len(impacts)
        else:
            avg_impact = 0
        
        # 关键事件
        key_events = []
        for e in events[:3]:
            key_events.append(f"{e.event_type.chinese_name}: {e.title[:30]}")
        
        # 信号调整因子
        signal_adjustment = 1.0
        if impacts:
            avg_mult = sum(i.signal_multiplier for i in impacts) / len(impacts)
            signal_adjustment = avg_mult
        
        return NewsDigest(
            symbol=symbol,
            period='today',
            total_news_count=len(events),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            dominant_sentiment=dominant,
            avg_expected_impact=round(avg_impact, 2),
            key_events=key_events,
            signal_adjustment=round(signal_adjustment, 2)
        )


# 全局单例
_impact_scorer = None

def get_impact_scorer() -> ImpactScorer:
    global _impact_scorer
    if _impact_scorer is None:
        _impact_scorer = ImpactScorer()
    return _impact_scorer
