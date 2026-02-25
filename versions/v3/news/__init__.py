#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
News Intelligence System (v2)
事件驱动的新闻智能分析系统
多源聚合: Google News · yfinance · StockTwits · Reddit · Finnhub
"""

from .models import NewsEvent, NewsImpact, EventType, Sentiment
from .crawler import (
    NewsCrawler, GoogleNewsCrawler, YFinanceNewsCrawler,
    StockTwitsCrawler, ApeWisdomCrawler, FinnhubCrawler,
    get_news_crawler
)
from .classifier import EventClassifier, get_event_classifier
from .scorer import ImpactScorer, get_impact_scorer
from .intelligence import NewsIntelligence, get_news_intelligence

__all__ = [
    'NewsEvent',
    'NewsImpact',
    'EventType',
    'Sentiment',
    'NewsCrawler',
    'GoogleNewsCrawler',
    'YFinanceNewsCrawler',
    'StockTwitsCrawler',
    'ApeWisdomCrawler',
    'FinnhubCrawler',
    'EventClassifier',
    'ImpactScorer',
    'NewsIntelligence',
    'get_news_intelligence',
    'get_news_crawler',
]
