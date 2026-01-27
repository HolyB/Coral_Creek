
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    snippet: str
    url: str
    source: str
    published_date: Optional[str] = None
    
    def to_text(self) -> str:
        return f"【{self.source}】{self.title}\n{self.snippet}"

class DuckDuckGoSearchService:
    """
    DuckDuckGo 免费搜索服务
    """
    
    def __init__(self):
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.available = True
        except ImportError:
            logger.error("duckduckgo_search not installed. Run: pip install duckduckgo-search")
            self.available = False

    def search_news(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """搜索新闻"""
        if not self.available:
            return []
            
        results = []
        try:
            # 使用 DDGS news 搜索
            news_results = self.ddgs.news(keywords=query, max_results=max_results)
            if news_results:
                for item in news_results:
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        snippet=item.get('body', ''),
                        url=item.get('url', ''),
                        source=item.get('source', 'Unknown'),
                        published_date=item.get('date', '')
                    ))
            
            # 如果新闻搜索为空，尝试普通搜索
            if not results:
                text_results = self.ddgs.text(keywords=query, max_results=max_results)
                if text_results:
                    for item in text_results:
                        results.append(SearchResult(
                            title=item.get('title', ''),
                            snippet=item.get('body', ''),
                            url=item.get('href', ''),
                            source='DuckDuckGo',
                            published_date=None
                        ))
                        
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def get_stock_news(self, symbol: str, stock_name: str = "") -> str:
        """获取股票相关新闻并格式化为文本"""
        query = f"{stock_name} {symbol} 股票 最新消息" if stock_name else f"{symbol} stock news"
        
        # 增加关键词以获取更有价值的信息
        # 针对中文股票（如A股）
        if any('\u4e00' <= char <= '\u9fff' for char in stock_name):
            query = f"{stock_name} {symbol} 利好 利空 公告"
            
        results = self.search_news(query, max_results=5)
        
        if not results:
            return "未找到相关新闻。"
            
        formatted_text = f"【{stock_name or symbol} 最新情报】\n"
        for i, res in enumerate(results, 1):
            formatted_text += f"{i}. {res.title}\n   摘要: {res.snippet[:200]}...\n   来源: {res.source} ({res.published_date or '未知日期'})\n\n"
            
        return formatted_text

# 全局单例
_search_service = None

def get_search_service():
    global _search_service
    if _search_service is None:
        _search_service = DuckDuckGoSearchService()
    return _search_service
