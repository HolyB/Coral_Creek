
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    platform: str  # Reddit, Twitter, etc.
    title: str
    snippet: str
    url: str
    date: str
    sentiment: str = "Neutral"  # Bullish, Bearish, Neutral

class SocialMonitorService:
    """社交媒体舆情监控服务"""
    
    def __init__(self):
        self.ddgs = DDGS() if HAS_DDGS else None
        
        # 简单情绪词典
        self.bullish_keywords = [
            'buy', 'long', 'bull', 'moon', 'rocket', 'call', 'yolo', 'undervalued', 
            'breakout', 'support', 'hold', 'diamond hands', 'bought', 'growth',
            '买入', '看多', '起飞', '低估', '突破', '支撑', '拿住', '加仓'
        ]
        self.bearish_keywords = [
            'sell', 'short', 'bear', 'crash', 'tank', 'put', 'dump', 'overvalued',
            'resistance', 'drop', 'sold', 'bubble', 'scam',
            '卖出', '看空', '暴跌', '崩盘', '高估', '阻力', '泡沫', '离场'
        ]

    def _analyze_sentiment(self, text: str) -> str:
        """简单的基于规则的情绪分析"""
        text = text.lower()
        bull_score = sum(1 for w in self.bullish_keywords if w in text)
        bear_score = sum(1 for w in self.bearish_keywords if w in text)
        
        if bull_score > bear_score:
            return "Bullish"
        elif bear_score > bull_score:
            return "Bearish"
        else:
            return "Neutral"

    def search_reddit(self, symbol: str, limit: int = 10) -> List[SocialPost]:
        """搜索 Reddit 讨论"""
        if not self.ddgs:
            return []
            
        posts = []
        try:
            # 搜索查询：股票代码 + site:reddit.com
            # 排除一些非股票板块可能更好，但先全局搜
            query = f"{symbol} stock site:reddit.com"
            
            # 使用 text 搜索 (v4+ API)
            results = self.ddgs.text(query, max_results=limit)
            
            if results:
                for res in results:
                    title = res.get('title', '')
                    body = res.get('body', '')
                    
                    # 尝试从 body 提取日期 (DDG 有时不返回日期)
                    # 简单处理：假设都是近期的
                    
                    posts.append(SocialPost(
                        platform="Reddit",
                        title=title,
                        snippet=body,
                        url=res.get('href', ''),
                        date="Recent", # DDG 不一定返回准确日期
                        sentiment=self._analyze_sentiment(title + " " + body)
                    ))
                    
        except Exception as e:
            logger.error(f"Reddit search error for {symbol}: {e}")
            
        return posts

    def search_twitter(self, symbol: str, limit: int = 10) -> List[SocialPost]:
        """搜索 Twitter/X 讨论"""
        if not self.ddgs:
            return []
            
        posts = []
        try:
            # 搜索查询
            query = f"${symbol} stock site:twitter.com"
            
            results = self.ddgs.text(query, max_results=limit)
            
            if results:
                for res in results:
                    title = res.get('title', '')
                    # Twitter 标题通常是 "Name (@handle) on X: '...'"
                    # 清洗标题
                    if " on X: " in title:
                        parts = title.split(" on X: ", 1)
                        author = parts[0]
                        content = parts[1].strip('"')
                        display_title = f"{author}: {content[:50]}..."
                    else:
                        display_title = title
                        
                    body = res.get('body', '')
                    
                    posts.append(SocialPost(
                        platform="Twitter",
                        title=display_title,
                        snippet=body,
                        url=res.get('href', ''),
                        date="Recent",
                        sentiment=self._analyze_sentiment(body)
                    ))
                    
        except Exception as e:
            logger.error(f"Twitter search error for {symbol}: {e}")
            
        return posts
    
    def search_guba(self, symbol: str, limit: int = 10) -> List[SocialPost]:
        """搜索东方财富股吧 (针对 A股)"""
        if not self.ddgs:
            return []
        
        posts = []
        try:
            # 去掉后缀，如 000001.SZ -> 000001
            code = symbol.split('.')[0]
            query = f"{code} 股吧 site:guba.eastmoney.com"
            
            results = self.ddgs.text(query, max_results=limit)
            
            if results:
                for res in results:
                    title = res.get('title', '')
                    body = res.get('body', '')
                    
                    # 过滤掉非帖子页面
                    if "股吧" not in title and "东方财富" not in title:
                        continue
                        
                    posts.append(SocialPost(
                        platform="股吧",
                        title=title,
                        snippet=body,
                        url=res.get('href', ''),
                        date="Recent",
                        sentiment=self._analyze_sentiment(title + " " + body)
                    ))
        except Exception as e:
            logger.error(f"Guba search error for {symbol}: {e}")
            
        return posts

    def get_social_report(self, symbol: str, market: str = 'US') -> Dict:
        """生成综合舆情报告"""
        logger.info(f"Generating social report for {symbol} ({market})")
        
        all_posts = []
        
        if market == 'US':
            all_posts.extend(self.search_reddit(symbol, limit=8))
            all_posts.extend(self.search_twitter(symbol, limit=8))
        else:
            all_posts.extend(self.search_guba(symbol, limit=10))
            # 也可以搜雪球 site:xueqiu.com
            try:
                if self.ddgs:
                    xq_results = self.ddgs.text(f"{symbol} site:xueqiu.com", max_results=5)
                    for res in xq_results:
                        all_posts.append(SocialPost(
                            platform="雪球",
                            title=res.get('title', ''),
                            snippet=res.get('body', ''),
                            url=res.get('href', ''),
                            date="Recent",
                            sentiment=self._analyze_sentiment(res.get('title', '') + " " + res.get('body', ''))
                        ))
            except:
                pass
        
        # 统计情绪
        bullish = sum(1 for p in all_posts if p.sentiment == 'Bullish')
        bearish = sum(1 for p in all_posts if p.sentiment == 'Bearish')
        neutral = len(all_posts) - bullish - bearish
        
        # 热门话题 (简单词频统计或取前几个标题)
        hot_topics = [p.title for p in all_posts[:5]]
        
        return {
            'symbol': symbol,
            'total_posts': len(all_posts),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'posts': all_posts,
            'sentiment_score': (bullish - bearish) / len(all_posts) if all_posts else 0
        }

# 全局实例
_social_service = None

def get_social_service():
    global _social_service
    if _social_service is None:
        _social_service = SocialMonitorService()
    return _social_service

if __name__ == "__main__":
    # 测试
    svc = get_social_service()
    report = svc.get_social_report("NVDA", market="US")
    print(f"NVDA Report: {report['bullish_count']} Bull / {report['bearish_count']} Bear")
    for p in report['posts'][:3]:
        print(f"[{p.platform}] {p.sentiment}: {p.title}")
