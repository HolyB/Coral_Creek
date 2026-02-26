#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻情绪特征 (News Sentiment Features)
========================================

为 ML 特征工程提供新闻 + 社交媒体情绪特征。
设计原则:
  - 轻量级: 只算数值特征，不保存原始文本
  - 可缓存: 10分钟缓存避免重复 API 调用
  - 降级安全: 新闻不可用时返回全 0 特征

特征列:
  news_count          — 新闻条数
  news_bullish_ratio  — 利好占比 (0-1)
  news_bearish_ratio  — 利空占比 (0-1)
  news_sentiment_net  — 净情绪 (bullish - bearish) / total
  social_buzz_score   — 社交热度综合分
  reddit_mentions     — Reddit/WSB 提及数
  reddit_rank         — WSB 排名 (越小越热, 0=未上榜)
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 简易内存缓存 (10分钟)
_cache: Dict[str, Dict] = {}  # key: f"{symbol}_{market}_{time_bucket}" -> features
_CACHE_MINUTES = 10


def _cache_key(symbol: str, market: str) -> str:
    bucket = datetime.now().strftime("%Y%m%d%H") + str(datetime.now().minute // _CACHE_MINUTES)
    return f"{symbol}_{market}_{bucket}"


def get_news_sentiment_features(
    symbol: str,
    market: str = 'US',
    company_name: str = '',
) -> Dict[str, float]:
    """获取新闻情绪特征字典。
    
    Returns:
        {
            'news_count': int,
            'news_bullish_ratio': float (0-1),
            'news_bearish_ratio': float (0-1),
            'news_sentiment_net': float (-1 to 1),
            'social_buzz_score': float,
            'reddit_mentions': int,
            'reddit_rank': int (0 = not ranked),
        }
    """
    key = _cache_key(symbol, market)
    if key in _cache:
        return _cache[key]
    
    features = _default_features()
    
    try:
        features = _compute_features(symbol, market, company_name)
    except Exception as e:
        logger.warning(f"News sentiment features failed for {symbol}: {e}")
    
    _cache[key] = features
    return features


def _default_features() -> Dict[str, float]:
    """返回全 0 默认特征 (新闻不可用时)"""
    return {
        'news_count': 0,
        'news_bullish_ratio': 0.0,
        'news_bearish_ratio': 0.0,
        'news_sentiment_net': 0.0,
        'social_buzz_score': 0.0,
        'reddit_mentions': 0,
        'reddit_rank': 0,
    }


def _compute_features(symbol: str, market: str, company_name: str) -> Dict[str, float]:
    """实际计算新闻情绪特征"""
    features = _default_features()
    
    # --- 新闻情绪 ---
    try:
        from news.crawler import get_news_crawler
        from news.classifier import get_event_classifier
        
        crawler = get_news_crawler()
        events = crawler.crawl_all(symbol, company_name, market, max_per_source=3)
        
        if events:
            # 分类 (优先用 Gemini)
            classifier = get_event_classifier(use_llm=True)
            events = classifier.classify_batch(events)
            
            total = len(events)
            bullish = sum(1 for e in events if e.sentiment.score > 0)
            bearish = sum(1 for e in events if e.sentiment.score < 0)
            
            features['news_count'] = total
            features['news_bullish_ratio'] = bullish / total if total > 0 else 0
            features['news_bearish_ratio'] = bearish / total if total > 0 else 0
            features['news_sentiment_net'] = (
                (bullish - bearish) / total if total > 0 else 0
            )
    except ImportError:
        pass  # 新闻模块不可用
    except Exception as e:
        logger.debug(f"News crawl failed for {symbol}: {e}")
    
    # --- 社交热度 ---
    try:
        from news.crawler import get_news_crawler
        crawler = get_news_crawler()
        buzz = crawler.get_social_buzz(symbol, market)
        
        features['social_buzz_score'] = buzz.get('total_buzz_score', 0)
        features['reddit_mentions'] = buzz.get('reddit_mentions', 0)
        features['reddit_rank'] = buzz.get('reddit_rank', 0) or 0
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Social buzz failed for {symbol}: {e}")
    
    return features


def get_batch_news_features(
    symbols: list,
    market: str = 'US',
) -> Dict[str, Dict[str, float]]:
    """批量获取新闻特征 (用于训练数据集)"""
    results = {}
    for sym in symbols:
        results[sym] = get_news_sentiment_features(sym, market)
    return results


# 特征名列表 (ML 模型需要知道列名)
NEWS_FEATURE_NAMES = [
    'news_count',
    'news_bullish_ratio',
    'news_bearish_ratio',
    'news_sentiment_net',
    'social_buzz_score',
    'reddit_mentions',
    'reddit_rank',
]
