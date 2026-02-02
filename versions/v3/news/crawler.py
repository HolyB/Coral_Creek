#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻抓取器
支持多来源：Google News RSS, Polygon.io, Finnhub 等
"""
import os
import re
import html
import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from urllib.parse import quote

from .models import NewsEvent, EventType, Sentiment

logger = logging.getLogger(__name__)


class GoogleNewsCrawler:
    """Google News RSS 抓取器 (免费、稳定)"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    
    def crawl(self, symbol: str, company_name: str = "", 
              market: str = 'US', max_results: int = 10) -> List[NewsEvent]:
        """抓取新闻"""
        events = []
        
        try:
            if market == 'CN':
                query = f"{company_name} {symbol} 股票" if company_name else f"{symbol} 股票"
                lang, gl = 'zh-CN', 'CN'
            else:
                query = f"{symbol} stock news"
                lang, gl = 'en-US', 'US'
            
            encoded_query = quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang}&gl={gl}&ceid={gl}:{lang}"
            
            resp = requests.get(url, headers=self.headers, timeout=10)
            
            if resp.status_code == 200:
                content = resp.text
                items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                
                for item_str in items[:max_results]:
                    event = self._parse_item(item_str, symbol)
                    if event:
                        events.append(event)
                        
        except Exception as e:
            logger.error(f"Google News crawl error for {symbol}: {e}")
        
        return events
    
    def _parse_item(self, item_str: str, symbol: str) -> Optional[NewsEvent]:
        """解析 RSS item"""
        try:
            title_match = re.search(r'<title>(.*?)</title>', item_str)
            link_match = re.search(r'<link>(.*?)</link>', item_str)
            pub_match = re.search(r'<pubDate>(.*?)</pubDate>', item_str)
            source_match = re.search(r'<source url=".*?">(.*?)</source>', item_str)
            
            if not title_match:
                return None
            
            title = html.unescape(title_match.group(1))
            source = source_match.group(1) if source_match else "Google News"
            
            # 解析标题中的来源
            if " - " in title and not source_match:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]
            
            # 解析发布时间
            published_at = datetime.now()
            if pub_match:
                try:
                    pub_str = pub_match.group(1)
                    # RFC 2822 format: "Sat, 01 Feb 2026 10:30:00 GMT"
                    from email.utils import parsedate_to_datetime
                    published_at = parsedate_to_datetime(pub_str)
                except:
                    pass
            
            # 生成唯一ID
            content_hash = hashlib.md5(f"{symbol}{title}{link_match.group(1) if link_match else ''}".encode()).hexdigest()[:12]
            
            return NewsEvent(
                id=f"gn_{content_hash}",
                symbol=symbol,
                title=title,
                source=source,
                url=link_match.group(1) if link_match else "",
                published_at=published_at,
                summary=title  # RSS 没有摘要
            )
            
        except Exception as e:
            logger.error(f"Parse RSS item error: {e}")
            return None


class PolygonNewsCrawler:
    """Polygon.io 新闻抓取器 (需要 API Key)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.base_url = "https://api.polygon.io"
    
    def crawl(self, symbol: str, max_results: int = 10, 
              days_back: int = 7) -> List[NewsEvent]:
        """抓取新闻"""
        events = []
        
        if not self.api_key:
            logger.warning("Polygon API key not set")
            return events
        
        try:
            url = f"{self.base_url}/v2/reference/news"
            params = {
                'ticker': symbol,
                'limit': max_results,
                'order': 'desc',
                'apiKey': self.api_key
            }
            
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('results', []):
                    event = self._parse_item(item, symbol)
                    if event:
                        events.append(event)
                        
        except Exception as e:
            logger.error(f"Polygon news crawl error for {symbol}: {e}")
        
        return events
    
    def _parse_item(self, item: Dict, symbol: str) -> Optional[NewsEvent]:
        """解析 Polygon 新闻"""
        try:
            # 解析时间
            pub_str = item.get('published_utc', '')
            try:
                published_at = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
            
            # 生成ID
            article_id = item.get('id', '')
            if not article_id:
                article_id = hashlib.md5(f"{symbol}{item.get('title', '')}".encode()).hexdigest()[:12]
            
            return NewsEvent(
                id=f"pg_{article_id}",
                symbol=symbol,
                title=item.get('title', ''),
                source=item.get('publisher', {}).get('name', 'Polygon'),
                url=item.get('article_url', ''),
                published_at=published_at,
                summary=item.get('description', ''),
                keywords=item.get('keywords', [])
            )
            
        except Exception as e:
            logger.error(f"Parse Polygon item error: {e}")
            return None


class NewsCrawler:
    """统一新闻抓取接口"""
    
    def __init__(self):
        self.google_crawler = GoogleNewsCrawler()
        self.polygon_crawler = PolygonNewsCrawler()
    
    def crawl_all(self, symbol: str, company_name: str = "", 
                  market: str = 'US', max_per_source: int = 5) -> List[NewsEvent]:
        """从所有来源抓取新闻"""
        all_events = []
        seen_titles = set()
        
        # 1. Google News (免费、最全)
        google_events = self.google_crawler.crawl(
            symbol, company_name, market, max_per_source
        )
        for event in google_events:
            if event.title not in seen_titles:
                all_events.append(event)
                seen_titles.add(event.title)
        
        # 2. Polygon (美股、有 API Key 时)
        if market == 'US':
            polygon_events = self.polygon_crawler.crawl(symbol, max_per_source)
            for event in polygon_events:
                if event.title not in seen_titles:
                    all_events.append(event)
                    seen_titles.add(event.title)
        
        # 按时间排序
        all_events.sort(key=lambda x: x.published_at, reverse=True)
        
        logger.info(f"Crawled {len(all_events)} unique news for {symbol}")
        return all_events
    
    def crawl_batch(self, symbols: List[str], market: str = 'US', 
                    max_per_symbol: int = 3) -> Dict[str, List[NewsEvent]]:
        """批量抓取"""
        results = {}
        
        for symbol in symbols:
            events = self.crawl_all(symbol, market=market, max_per_source=max_per_symbol)
            results[symbol] = events
        
        return results


# 全局单例
_news_crawler = None

def get_news_crawler() -> NewsCrawler:
    global _news_crawler
    if _news_crawler is None:
        _news_crawler = NewsCrawler()
    return _news_crawler
