#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻 & 社交抓取器 (v2)
=======================

多源数据聚合:
  L1 (零成本): Google News RSS, yfinance, StockTwits, ApeWisdom
  L2 (免费Key): Finnhub
  L3 (爬虫):    东方财富股吧
"""
import os
import re
import html
import json
import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from urllib.parse import quote

from .models import NewsEvent, EventType, Sentiment

logger = logging.getLogger(__name__)

# =========================================================
# L1: Google News RSS (免费, 已有)
# =========================================================

class GoogleNewsCrawler:
    """Google News RSS 抓取器 (免费、稳定)"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    
    def crawl(self, symbol: str, company_name: str = "", 
              market: str = 'US', max_results: int = 10) -> List[NewsEvent]:
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
        try:
            title_match = re.search(r'<title>(.*?)</title>', item_str)
            link_match = re.search(r'<link>(.*?)</link>', item_str)
            pub_match = re.search(r'<pubDate>(.*?)</pubDate>', item_str)
            source_match = re.search(r'<source url=".*?">(.*?)</source>', item_str)
            
            if not title_match:
                return None
            
            title = html.unescape(title_match.group(1))
            source = source_match.group(1) if source_match else "Google News"
            
            if " - " in title and not source_match:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]
            
            published_at = datetime.now()
            if pub_match:
                try:
                    from email.utils import parsedate_to_datetime
                    published_at = parsedate_to_datetime(pub_match.group(1))
                except:
                    pass
            
            content_hash = hashlib.md5(
                f"{symbol}{title}{link_match.group(1) if link_match else ''}".encode()
            ).hexdigest()[:12]
            
            return NewsEvent(
                id=f"gn_{content_hash}",
                symbol=symbol,
                title=title,
                source=source,
                url=link_match.group(1) if link_match else "",
                published_at=published_at,
                summary=title
            )
        except Exception as e:
            logger.error(f"Parse RSS item error: {e}")
            return None


# =========================================================
# L1: yfinance News (零成本, 已安装)
# =========================================================

class YFinanceNewsCrawler:
    """yfinance 内置新闻 — 有 title + summary + 直达链接"""
    
    def crawl(self, symbol: str, max_results: int = 10) -> List[NewsEvent]:
        events = []
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                return events
            
            for item in news[:max_results]:
                event = self._parse_item(item, symbol)
                if event:
                    events.append(event)
        except Exception as e:
            logger.error(f"yfinance news error for {symbol}: {e}")
        return events
    
    def _parse_item(self, item: Dict, symbol: str) -> Optional[NewsEvent]:
        try:
            content = item.get('content', item)
            
            title = content.get('title', '')
            if not title:
                return None
            
            summary = content.get('summary', '') or ''
            pub_str = content.get('pubDate', '')
            source_name = ''
            url = ''
            
            # 解析 provider
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                source_name = provider.get('displayName', 'Yahoo Finance')
                url = provider.get('url', '')
            
            # 解析 clickThroughUrl 或 canonicalUrl
            click_url = content.get('clickThroughUrl', {})
            if isinstance(click_url, dict):
                url = click_url.get('url', url)
            canonical = content.get('canonicalUrl', {})
            if isinstance(canonical, dict):
                url = canonical.get('url', url) or url
            
            # 解析发布时间
            published_at = datetime.now()
            if pub_str:
                try:
                    published_at = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                except:
                    pass
            
            content_id = content.get('id', hashlib.md5(title.encode()).hexdigest()[:12])
            
            return NewsEvent(
                id=f"yf_{content_id[:12]}",
                symbol=symbol,
                title=title,
                source=source_name or "Yahoo Finance",
                url=url,
                published_at=published_at,
                summary=summary[:300] if summary else title,
            )
        except Exception as e:
            logger.error(f"Parse yfinance item error: {e}")
            return None


# =========================================================
# L1: StockTwits (免费, 美股交易员最活跃)
# =========================================================

class StockTwitsCrawler:
    """StockTwits API — 实时交易员讨论和情绪"""
    
    BASE_URL = "https://api.stocktwits.com/api/2"
    _available = None  # None = untested, True/False after first try
    
    def crawl(self, symbol: str, max_results: int = 15) -> List[NewsEvent]:
        events = []
        if StockTwitsCrawler._available is False:
            return events  # 已知被 Cloudflare 阻止
        try:
            url = f"{self.BASE_URL}/streams/symbol/{symbol}.json"
            resp = requests.get(url, timeout=8, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
            })
            
            if resp.status_code == 403:
                StockTwitsCrawler._available = False
                logger.info("StockTwits blocked by Cloudflare — disabled for this session")
                return events
            elif resp.status_code != 200:
                logger.warning(f"StockTwits API returned {resp.status_code} for {symbol}")
                return events
            
            StockTwitsCrawler._available = True
            
            data = resp.json()
            messages = data.get('messages', [])
            
            for msg in messages[:max_results]:
                event = self._parse_message(msg, symbol)
                if event:
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"StockTwits crawl error for {symbol}: {e}")
        return events
    
    def _parse_message(self, msg: Dict, symbol: str) -> Optional[NewsEvent]:
        try:
            body = msg.get('body', '')
            if not body or len(body) < 10:
                return None
            
            msg_id = str(msg.get('id', ''))
            user = msg.get('user', {})
            username = user.get('username', 'unknown')
            
            # 解析情绪
            entities = msg.get('entities', {})
            sentiment_data = entities.get('sentiment', {})
            st_sentiment = sentiment_data.get('basic', 'Neutral') if sentiment_data else 'Neutral'
            
            # 映射情绪
            sentiment_map = {
                'Bullish': Sentiment.BULLISH,
                'Bearish': Sentiment.BEARISH,
                'Neutral': Sentiment.NEUTRAL,
            }
            sentiment = sentiment_map.get(st_sentiment, Sentiment.NEUTRAL)
            
            # 解析时间
            created_at = msg.get('created_at', '')
            published_at = datetime.now()
            if created_at:
                try:
                    published_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    try:
                        from email.utils import parsedate_to_datetime
                        published_at = parsedate_to_datetime(created_at)
                    except:
                        pass
            
            return NewsEvent(
                id=f"st_{msg_id}",
                symbol=symbol,
                title=body[:120],
                source=f"StockTwits @{username}",
                url=f"https://stocktwits.com/symbol/{symbol}",
                published_at=published_at,
                summary=body[:300],
                sentiment=sentiment,
                is_classified=True,  # StockTwits 自带情绪
                event_type=EventType.OTHER,
            )
        except Exception as e:
            logger.error(f"Parse StockTwits msg error: {e}")
            return None
    
    def get_trending(self) -> List[Dict]:
        """获取 StockTwits 热门股票"""
        if StockTwitsCrawler._available is False:
            return []
        try:
            url = f"{self.BASE_URL}/trending/symbols.json"
            resp = requests.get(url, timeout=8, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
            })
            if resp.status_code == 403:
                StockTwitsCrawler._available = False
                return []
            if resp.status_code == 200:
                StockTwitsCrawler._available = True
                data = resp.json()
                symbols = data.get('symbols', [])
                return [
                    {'symbol': s.get('symbol', ''), 'title': s.get('title', ''), 
                     'watchlist_count': s.get('watchlist_count', 0)}
                    for s in symbols[:20]
                ]
        except Exception as e:
            logger.error(f"StockTwits trending error: {e}")
        return []


# =========================================================
# L1: ApeWisdom (免费, Reddit/WSB 热度追踪)
# =========================================================

class ApeWisdomCrawler:
    """ApeWisdom API — Reddit WallStreetBets 热门股票"""
    
    BASE_URL = "https://apewisdom.io/api/v1.0"
    
    def get_trending(self, filter_type: str = "all-stocks", limit: int = 20) -> List[Dict]:
        """获取 WSB/Reddit 热门股票
        
        filter_type: all-stocks, wallstreetbets, stocks, ...
        """
        try:
            url = f"{self.BASE_URL}/filter/{filter_type}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])
                return [
                    {
                        'symbol': r.get('ticker', ''),
                        'name': r.get('name', ''),
                        'mentions': r.get('mentions', 0),
                        'upvotes': r.get('upvotes', 0),
                        'rank': r.get('rank', 0),
                        'mentions_24h_ago': r.get('mentions_24h_ago', 0),
                    }
                    for r in results[:limit]
                ]
        except Exception as e:
            logger.error(f"ApeWisdom trending error: {e}")
        return []
    
    def get_symbol_mentions(self, symbol: str) -> Optional[Dict]:
        """获取某股票的 Reddit 讨论热度"""
        trending = self.get_trending(limit=50)
        for item in trending:
            if item['symbol'].upper() == symbol.upper():
                return item
        return None


# =========================================================
# L2: Finnhub News (免费 API Key)
# =========================================================

class FinnhubCrawler:
    """Finnhub 新闻抓取器 — 需要免费 API key"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
        self.base_url = "https://finnhub.io/api/v1"
    
    @property
    def is_available(self):
        return bool(self.api_key)
    
    def crawl(self, symbol: str, days_back: int = 7, 
              max_results: int = 20) -> List[NewsEvent]:
        events = []
        if not self.api_key:
            return events
        
        try:
            now = datetime.now()
            from_date = (now - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = now.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': self.api_key
            }
            
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data[:max_results]:
                    event = self._parse_item(item, symbol)
                    if event:
                        events.append(event)
            elif resp.status_code == 429:
                logger.warning("Finnhub rate limit reached")
            else:
                logger.warning(f"Finnhub returned {resp.status_code}")
                
        except Exception as e:
            logger.error(f"Finnhub crawl error for {symbol}: {e}")
        return events
    
    def _parse_item(self, item: Dict, symbol: str) -> Optional[NewsEvent]:
        try:
            title = item.get('headline', '')
            if not title:
                return None
            
            timestamp = item.get('datetime', 0)
            published_at = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            
            news_id = str(item.get('id', hashlib.md5(title.encode()).hexdigest()[:12]))
            
            return NewsEvent(
                id=f"fh_{news_id[:12]}",
                symbol=symbol,
                title=title,
                source=item.get('source', 'Finnhub'),
                url=item.get('url', ''),
                published_at=published_at,
                summary=item.get('summary', title)[:300],
            )
        except Exception as e:
            logger.error(f"Parse Finnhub item error: {e}")
            return None
    
    def get_social_sentiment(self, symbol: str) -> Optional[Dict]:
        """获取 Finnhub 社交媒体情绪 (Reddit + Twitter)"""
        if not self.api_key:
            return None
        try:
            url = f"{self.base_url}/stock/social-sentiment"
            params = {'symbol': symbol, 'token': self.api_key}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                reddit = data.get('reddit', [])
                twitter = data.get('twitter', [])
                
                reddit_mentions = sum(r.get('mention', 0) for r in reddit[-7:])
                reddit_sentiment = (
                    sum(r.get('positiveMention', 0) for r in reddit[-7:]) -
                    sum(r.get('negativeMention', 0) for r in reddit[-7:])
                )
                twitter_mentions = sum(t.get('mention', 0) for t in twitter[-7:])
                twitter_sentiment = (
                    sum(t.get('positiveMention', 0) for t in twitter[-7:]) -
                    sum(t.get('negativeMention', 0) for t in twitter[-7:])
                )
                
                return {
                    'reddit_mentions_7d': reddit_mentions,
                    'reddit_sentiment_7d': reddit_sentiment,
                    'twitter_mentions_7d': twitter_mentions,
                    'twitter_sentiment_7d': twitter_sentiment,
                    'total_mentions': reddit_mentions + twitter_mentions,
                    'total_sentiment': reddit_sentiment + twitter_sentiment,
                }
        except Exception as e:
            logger.error(f"Finnhub social sentiment error: {e}")
        return None


# =========================================================
# L2: Polygon News (需要 API Key)
# =========================================================

class PolygonNewsCrawler:
    """Polygon.io 新闻抓取器"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY', '')
        self.base_url = "https://api.polygon.io"
    
    @property
    def is_available(self):
        return bool(self.api_key)
    
    def crawl(self, symbol: str, max_results: int = 10) -> List[NewsEvent]:
        events = []
        if not self.api_key:
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
        try:
            pub_str = item.get('published_utc', '')
            try:
                published_at = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
            
            article_id = item.get('id', '')
            if not article_id:
                article_id = hashlib.md5(
                    f"{symbol}{item.get('title', '')}".encode()
                ).hexdigest()[:12]
            
            return NewsEvent(
                id=f"pg_{article_id[:12]}",
                symbol=symbol,
                title=item.get('title', ''),
                source=item.get('publisher', {}).get('name', 'Polygon'),
                url=item.get('article_url', ''),
                published_at=published_at,
                summary=item.get('description', '')[:300],
                keywords=item.get('keywords', [])
            )
        except Exception as e:
            logger.error(f"Parse Polygon item error: {e}")
            return None


# =========================================================
# 统一新闻抓取接口
# =========================================================

class NewsCrawler:
    """统一新闻抓取接口 — 多源聚合"""
    
    def __init__(self):
        self.google = GoogleNewsCrawler()
        self.yfinance = YFinanceNewsCrawler()
        self.stocktwits = StockTwitsCrawler()
        self.apewisdom = ApeWisdomCrawler()
        self.finnhub = FinnhubCrawler()
        self.polygon = PolygonNewsCrawler()
    
    def crawl_all(self, symbol: str, company_name: str = "", 
                  market: str = 'US', max_per_source: int = 5) -> List[NewsEvent]:
        """从所有来源抓取新闻"""
        all_events = []
        seen_titles = set()
        sources_used = []
        
        def _add_events(events: List[NewsEvent], source_name: str):
            count = 0
            for event in events:
                # 去重：标题前40字符相同视为重复
                title_key = event.title[:40].lower().strip()
                if title_key not in seen_titles:
                    all_events.append(event)
                    seen_titles.add(title_key)
                    count += 1
            if count > 0:
                sources_used.append(f"{source_name}({count})")
        
        # L1: yfinance (美股优先，最结构化)
        if market == 'US':
            _add_events(
                self.yfinance.crawl(symbol, max_per_source),
                "yfinance"
            )
        
        # L1: Google News (全覆盖)
        _add_events(
            self.google.crawl(symbol, company_name, market, max_per_source),
            "Google"
        )
        
        # L1: StockTwits (美股社交)
        if market == 'US':
            _add_events(
                self.stocktwits.crawl(symbol, max_per_source),
                "StockTwits"
            )
        
        # L2: Finnhub (有 key 时)
        if market == 'US' and self.finnhub.is_available:
            _add_events(
                self.finnhub.crawl(symbol, max_results=max_per_source),
                "Finnhub"
            )
        
        # L2: Polygon (有 key 时)
        if market == 'US' and self.polygon.is_available:
            _add_events(
                self.polygon.crawl(symbol, max_per_source),
                "Polygon"
            )
        
        # 按时间排序
        all_events.sort(key=lambda x: x.published_at, reverse=True)
        
        logger.info(f"Crawled {len(all_events)} news for {symbol} from: {', '.join(sources_used)}")
        return all_events
    
    def crawl_batch(self, symbols: List[str], market: str = 'US', 
                    max_per_symbol: int = 3) -> Dict[str, List[NewsEvent]]:
        """批量抓取"""
        results = {}
        for symbol in symbols:
            events = self.crawl_all(symbol, market=market, max_per_source=max_per_symbol)
            results[symbol] = events
        return results
    
    def get_social_buzz(self, symbol: str, market: str = 'US') -> Dict:
        """获取社交媒体综合热度"""
        buzz = {
            'symbol': symbol,
            'stocktwits_sentiment': None,
            'reddit_rank': None,
            'reddit_mentions': 0,
            'finnhub_social': None,
            'total_buzz_score': 0,
        }
        
        # StockTwits
        if market == 'US':
            try:
                st_events = self.stocktwits.crawl(symbol, max_results=20)
                if st_events:
                    bull = sum(1 for e in st_events if e.sentiment == Sentiment.BULLISH)
                    bear = sum(1 for e in st_events if e.sentiment == Sentiment.BEARISH)
                    total = bull + bear
                    buzz['stocktwits_sentiment'] = {
                        'total': len(st_events),
                        'bullish': bull,
                        'bearish': bear,
                        'ratio': (bull - bear) / total if total > 0 else 0,
                    }
                    buzz['total_buzz_score'] += len(st_events) * 2
            except:
                pass
        
        # ApeWisdom (Reddit)
        if market == 'US':
            try:
                ape = self.apewisdom.get_symbol_mentions(symbol)
                if ape:
                    buzz['reddit_rank'] = ape.get('rank')
                    buzz['reddit_mentions'] = ape.get('mentions', 0)
                    buzz['total_buzz_score'] += ape.get('mentions', 0)
            except:
                pass
        
        # Finnhub social
        if market == 'US' and self.finnhub.is_available:
            try:
                fh_social = self.finnhub.get_social_sentiment(symbol)
                if fh_social:
                    buzz['finnhub_social'] = fh_social
                    buzz['total_buzz_score'] += fh_social.get('total_mentions', 0)
            except:
                pass
        
        return buzz


# 全局单例
_news_crawler = None

def get_news_crawler() -> NewsCrawler:
    global _news_crawler
    if _news_crawler is None:
        _news_crawler = NewsCrawler()
    return _news_crawler
