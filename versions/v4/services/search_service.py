
import logging
import requests
import re
import html
from typing import List, Optional
from dataclasses import dataclass
from urllib.parse import quote

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
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"【{self.source}】{self.title}{date_str}\n{self.snippet}"

class GoogleNewsProvider:
    """Google News RSS (稳定、由于无需Key、多语言支持)"""
    
    def search(self, query: str, lang: str = 'zh-CN', gl: str = 'CN') -> List[SearchResult]:
        results = []
        try:
            # 构建 RSS URL
            encoded_query = quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang}&gl={gl}&ceid={gl}:{lang}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                # 简单正则解析 XML，比 xml.etree 更容错
                content = resp.text
                items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                
                for item_str in items[:5]:
                    title_match = re.search(r'<title>(.*?)</title>', item_str)
                    link_match = re.search(r'<link>(.*?)</link>', item_str)
                    pub_match = re.search(r'<pubDate>(.*?)</pubDate>', item_str)
                    source_match = re.search(r'<source url=".*?">(.*?)</source>', item_str)
                    
                    if title_match:
                        title = html.unescape(title_match.group(1))
                        # Google News 标题通常包含来源，如 "Title - Source"
                        source = source_match.group(1) if source_match else "Google News"
                        if " - " in title and not source_match:
                            parts = title.rsplit(" - ", 1)
                            title = parts[0]
                            source = parts[1]
                            
                        results.append(SearchResult(
                            title=title,
                            snippet="点击链接查看详情",  # RSS title 即摘要
                            url=link_match.group(1) if link_match else "",
                            source=source,
                            published_date=pub_match.group(1) if pub_match else ""
                        ))
        except Exception as e:
            logger.error(f"Google News search error: {e}")
            
        return results

class SearchService:
    """
    搜索服务 (Google News RSS)
    """
    
    def __init__(self):
        self.provider = GoogleNewsProvider()

    def get_stock_news(self, symbol: str, stock_name: str = "") -> str:
        """获取股票相关新闻"""
        
        logger.info(f"[SearchService] Starting news search for symbol={symbol}, name={stock_name}")
        
        # 判断是 A股 还是 美股
        is_cn = symbol.endswith(('.SH', '.SZ')) or (stock_name and any('\u4e00' <= char <= '\u9fff' for char in stock_name))
        
        try:
            if is_cn:
                # A股查询策略
                query = f"{stock_name} {symbol} 股票"
                logger.info(f"[SearchService] CN stock query: {query}")
                results = self.provider.search(query, lang='zh-CN', gl='CN')
            else:
                # 美股查询策略
                query = f"{symbol} stock news"
                logger.info(f"[SearchService] US stock query: {query}")
                results = self.provider.search(query, lang='en-US', gl='US')
            
            logger.info(f"[SearchService] Found {len(results)} results")
            
            if not results:
                return f"暂无 {stock_name or symbol} 相关新闻。可能是网络问题，请稍后重试。"
                
            # 格式化输出
            formatted_text = f"【{stock_name or symbol} 最新情报】\n"
            for i, res in enumerate(results, 1):
                formatted_text += f"{i}. {res.title}\n   来源: {res.source} ({res.published_date})\n\n"
                
            return formatted_text
            
        except Exception as e:
            logger.error(f"[SearchService] Error fetching news: {e}")
            return f"新闻搜索出错: {str(e)[:100]}。将仅基于技术面分析。"

# 全局单例
_search_service = None

def get_search_service():
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
