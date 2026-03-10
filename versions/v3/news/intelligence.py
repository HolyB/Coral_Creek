#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻智能系统 - 统一接口
整合抓取、分类、评分的完整流程
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from .models import (
    NewsEvent, NewsImpact, NewsDigest, EventType, Sentiment
)
from .crawler import NewsCrawler, get_news_crawler
from .classifier import EventClassifier, get_event_classifier
from .scorer import ImpactScorer, get_impact_scorer

logger = logging.getLogger(__name__)


class NewsIntelligence:
    """新闻智能系统 - 主接口"""
    
    def __init__(self, use_llm: bool = True):
        self.crawler = get_news_crawler()
        self.classifier = get_event_classifier(use_llm=use_llm)
        self.scorer = get_impact_scorer()
    
    def analyze_symbol(self, symbol: str, company_name: str = "", 
                       market: str = 'US') -> Tuple[List[NewsEvent], List[NewsImpact], NewsDigest]:
        """分析单只股票的新闻
        
        Returns:
            - events: 分类后的新闻列表
            - impacts: 影响评估列表
            - digest: 新闻摘要
        """
        logger.info(f"Analyzing news for {symbol}...")
        
        # 1. 抓取新闻
        raw_events = self.crawler.crawl_all(
            symbol, company_name, market, max_per_source=5
        )
        logger.info(f"Crawled {len(raw_events)} news for {symbol}")
        
        if not raw_events:
            return [], [], NewsDigest(symbol=symbol, period='today')
        
        # 2. 分类
        classified_events = self.classifier.classify_batch(raw_events)
        
        # 3. 评分
        impacts = []
        for event in classified_events:
            impact = self.scorer.score(event)
            impacts.append(impact)
        
        # 4. 生成摘要
        digest = self.scorer.create_digest(classified_events, impacts, symbol)
        
        logger.info(f"Analysis complete: {digest.total_news_count} news, "
                    f"sentiment ratio: {digest.sentiment_ratio():.2f}")
        
        return classified_events, impacts, digest
    
    def analyze_portfolio(self, symbols: List[str], 
                          market: str = 'US') -> Dict[str, Tuple[List[NewsEvent], NewsDigest]]:
        """分析投资组合的所有新闻"""
        results = {}
        
        for symbol in symbols:
            try:
                events, impacts, digest = self.analyze_symbol(symbol, market=market)
                results[symbol] = (events, digest)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = ([], NewsDigest(symbol=symbol, period='today'))
        
        return results
    
    def get_alerts(self, symbols: List[str], 
                   market: str = 'US') -> List[Dict]:
        """获取需要提醒的重要新闻"""
        alerts = []
        
        for symbol in symbols:
            try:
                events, impacts, digest = self.analyze_symbol(symbol, market=market)
                
                for event, impact in zip(events, impacts):
                    if impact.should_alert:
                        alerts.append({
                            'symbol': symbol,
                            'title': event.title,
                            'event_type': event.event_type.chinese_name,
                            'sentiment': event.sentiment.emoji,
                            'expected_impact': impact.expected_impact_pct,
                            'confidence': impact.confidence,
                            'urgency': impact.urgency,
                            'priority': impact.alert_priority,
                            'source': event.source,
                            'url': event.url,
                            'published_at': event.published_at.isoformat()
                        })
            except Exception as e:
                logger.error(f"Error getting alerts for {symbol}: {e}")
        
        # 按优先级排序
        alerts.sort(key=lambda x: (-x['priority'], -abs(x['expected_impact'])))
        
        return alerts
    
    def enhance_signal(self, symbol: str, base_confidence: float,
                       market: str = 'US') -> Tuple[float, str, NewsDigest]:
        """增强交易信号
        
        Args:
            symbol: 股票代码
            base_confidence: 基础信心度 (来自技术分析)
            market: 市场
        
        Returns:
            - adjusted_confidence: 调整后的信心度
            - adjustment_reason: 调整原因
            - digest: 新闻摘要
        """
        events, impacts, digest = self.analyze_symbol(symbol, market=market)
        
        if not events:
            return base_confidence, "无重大新闻", digest
        
        # 计算调整
        adjusted = base_confidence * digest.signal_adjustment
        
        # 生成原因
        reasons = []
        
        if digest.bullish_count > digest.bearish_count:
            reasons.append(f"利好新闻 {digest.bullish_count} 条")
        elif digest.bearish_count > digest.bullish_count:
            reasons.append(f"利空新闻 {digest.bearish_count} 条")
        
        if digest.avg_expected_impact > 3:
            reasons.append(f"预期强势影响 +{digest.avg_expected_impact:.1f}%")
        elif digest.avg_expected_impact < -3:
            reasons.append(f"预期利空影响 {digest.avg_expected_impact:.1f}%")
        
        if digest.key_events:
            reasons.append(digest.key_events[0])
        
        reason = "; ".join(reasons) if reasons else "新闻影响中性"
        
        return round(adjusted, 1), reason, digest
    
    def format_telegram_alert(self, alerts: List[Dict]) -> str:
        """格式化 Telegram 提醒消息"""
        if not alerts:
            return ""
        
        msg = "🚨 *新闻提醒*\n"
        msg += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        msg += "━" * 20 + "\n\n"
        
        for alert in alerts[:5]:  # 最多5条
            impact = alert['expected_impact']
            impact_str = f"+{impact:.1f}%" if impact > 0 else f"{impact:.1f}%"
            
            msg += f"📰 *{alert['symbol']}*\n"
            msg += f"   {alert['title'][:40]}...\n"
            msg += f"   {alert['event_type']} | {alert['sentiment']}\n"
            msg += f"   预期影响: {impact_str} (置信度 {alert['confidence']:.0f}%)\n"
            msg += f"   来源: {alert['source']}\n\n"
        
        msg += "━" * 20 + "\n"
        msg += "🔗 [详情](https://facaila.streamlit.app/)"
        
        return msg
    
    def format_digest_text(self, digest: NewsDigest) -> str:
        """格式化新闻摘要为文本"""
        if digest.total_news_count == 0:
            return f"📰 {digest.symbol}: 暂无重大新闻"
        
        sentiment_emoji = digest.dominant_sentiment.emoji if hasattr(digest.dominant_sentiment, 'emoji') else "➖"
        
        text = f"📰 *{digest.symbol} 新闻摘要*\n"
        text += f"   共 {digest.total_news_count} 条新闻\n"
        text += f"   情绪: {sentiment_emoji} 利好{digest.bullish_count}/利空{digest.bearish_count}\n"
        
        if abs(digest.avg_expected_impact) > 0.5:
            impact_str = f"+{digest.avg_expected_impact:.1f}%" if digest.avg_expected_impact > 0 else f"{digest.avg_expected_impact:.1f}%"
            text += f"   预期影响: {impact_str}\n"
        
        if digest.key_events:
            text += f"   关键: {digest.key_events[0]}\n"
        
        return text


# 全局单例
_news_intelligence = None

def get_news_intelligence(use_llm: bool = True) -> NewsIntelligence:
    global _news_intelligence
    if _news_intelligence is None:
        _news_intelligence = NewsIntelligence(use_llm=use_llm)
    return _news_intelligence
