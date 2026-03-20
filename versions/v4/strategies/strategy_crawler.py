#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略爬虫 — Strategy Crawler
===============================
每日自动爬取各平台的量化策略/公式/帖子，提取交易逻辑并自动回测

数据源:
  1. TradingView (Pine Script ideas, 社区策略)
  2. 通达信公式 (tdx formula forums)
  3. 经传多赢 (JingZhuan community)
  4. 聚宽/RiceQuant (量化社区)
  5. Reddit r/algotrading
  6. GitHub trending (quant repos)
"""
import os
import re
import json
import time
import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "db"
STRATEGY_DB = DB_DIR / "strategies.db"


@dataclass
class CrawledStrategy:
    """爬取到的策略"""
    source: str              # 来源平台
    title: str               # 标题
    url: str                 # 链接
    author: str              # 作者
    content: str             # 原始内容 (公式/描述)
    strategy_type: str       # buy_sell / indicator / screener / risk
    market: str              # US / CN / crypto / all
    tags: List[str] = field(default_factory=list)
    likes: int = 0           # 点赞/收藏数
    crawled_at: str = ""
    content_hash: str = ""   # 去重用
    
    # 解析后的交易规则
    buy_rules: List[str] = field(default_factory=list)
    sell_rules: List[str] = field(default_factory=list)
    parsed_logic: str = ""   # 解析后的交易逻辑 (Python 伪代码)
    
    # 回测结果
    backtest_done: bool = False
    backtest_result: Dict = field(default_factory=dict)


def _init_strategy_db():
    """初始化策略数据库"""
    conn = sqlite3.connect(str(STRATEGY_DB))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS crawled_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            url TEXT,
            author TEXT,
            content TEXT,
            strategy_type TEXT,
            market TEXT,
            tags TEXT,
            likes INTEGER DEFAULT 0,
            crawled_at TEXT,
            content_hash TEXT UNIQUE,
            buy_rules TEXT,
            sell_rules TEXT,
            parsed_logic TEXT,
            backtest_done BOOLEAN DEFAULT 0,
            backtest_sharpe REAL,
            backtest_return REAL,
            backtest_winrate REAL,
            backtest_maxdd REAL,
            backtest_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_strat_source ON crawled_strategies(source)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_strat_hash ON crawled_strategies(content_hash)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_strat_backtest ON crawled_strategies(backtest_done)")
    conn.commit()
    return conn


def _save_strategy(conn, strat: CrawledStrategy):
    """保存爬取的策略 (去重)"""
    if not strat.content_hash:
        strat.content_hash = hashlib.md5(
            (strat.title + strat.content[:500]).encode()
        ).hexdigest()
    
    try:
        conn.execute("""
            INSERT OR IGNORE INTO crawled_strategies
            (source, title, url, author, content, strategy_type, market,
             tags, likes, crawled_at, content_hash, buy_rules, sell_rules,
             parsed_logic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strat.source, strat.title, strat.url, strat.author,
            strat.content, strat.strategy_type, strat.market,
            json.dumps(strat.tags, ensure_ascii=False),
            strat.likes, strat.crawled_at or datetime.now().isoformat(),
            strat.content_hash,
            json.dumps(strat.buy_rules, ensure_ascii=False),
            json.dumps(strat.sell_rules, ensure_ascii=False),
            strat.parsed_logic,
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Save strategy error: {e}")
        return False


# =============================================================
# 1. TradingView Community Scripts
# =============================================================
class TradingViewCrawler:
    """TradingView 公开策略爬取 (通过 RSS / 搜索)"""
    
    BASE = "https://www.tradingview.com"
    
    def crawl(self, max_results: int = 20) -> List[CrawledStrategy]:
        strategies = []
        
        # TradingView 没有公开 API，用 Google 搜索替代
        queries = [
            "site:tradingview.com/script/ RSI MACD strategy",
            "site:tradingview.com/script/ buy sell signal",
            "site:tradingview.com/script/ moving average crossover",
            "site:tradingview.com/script/ volume breakout",
            "site:tradingview.com/script/ momentum strategy 2024 2025",
        ]
        
        for query in queries[:3]:
            try:
                results = self._google_search(query, max_results=5)
                for r in results:
                    strat = CrawledStrategy(
                        source="TradingView",
                        title=r.get('title', ''),
                        url=r.get('url', ''),
                        author=r.get('author', 'unknown'),
                        content=r.get('snippet', ''),
                        strategy_type='buy_sell',
                        market='all',
                        tags=['pine_script', 'community'],
                        crawled_at=datetime.now().isoformat(),
                    )
                    strat.content_hash = hashlib.md5(strat.url.encode()).hexdigest()
                    strategies.append(strat)
                    if len(strategies) >= max_results:
                        break
            except Exception as e:
                logger.warning(f"TradingView crawl error: {e}")
            
            time.sleep(1)  # 礼貌爬取
        
        return strategies
    
    def _google_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """通过 Google Custom Search 或直接搜索"""
        if requests is None:
            return []
        
        # 尝试 Google Custom Search API
        api_key = os.getenv('GOOGLE_API_KEY', '')
        cx = os.getenv('GOOGLE_CX', '')
        
        if api_key and cx:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {'key': api_key, 'cx': cx, 'q': query, 'num': max_results}
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    items = resp.json().get('items', [])
                    return [{'title': i['title'], 'url': i['link'], 
                            'snippet': i.get('snippet', '')} for i in items]
            except Exception:
                pass
        
        # Fallback: 直接请求 (可能被限流)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        try:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            resp = requests.get(url, headers=headers, timeout=10)
            results = []
            # 简单解析
            for match in re.finditer(r'<a[^>]*href="(https://www\.tradingview\.com/script/[^"]+)"[^>]*>([^<]+)', resp.text):
                results.append({
                    'url': match.group(1),
                    'title': match.group(2).strip(),
                    'snippet': '',
                })
                if len(results) >= max_results:
                    break
            return results
        except Exception:
            return []


# =============================================================
# 2. 通达信 / 经传 公式论坛
# =============================================================
class TDXFormulaCrawler:
    """通达信公式论坛爬取"""
    
    FORUMS = [
        # 理想论坛
        ("https://www.55188.com/forum-8-1.html", "理想论坛"),
        # CSDN 通达信
        ("https://blog.csdn.net/", "CSDN"),
    ]
    
    def crawl(self, max_results: int = 20) -> List[CrawledStrategy]:
        strategies = []
        
        if requests is None:
            return strategies
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0'
        }
        
        # 搜索通达信公式策略
        search_queries = [
            "通达信 选股 公式 买入 信号 2024",
            "通达信 量价突破 公式",
            "通达信 底部放量 选股",
            "经传多赢 指标 公式",
        ]
        
        for query in search_queries[:2]:
            try:
                url = f"https://html.duckduckgo.com/html/?q={query}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                
                # 提取链接和标题
                for match in re.finditer(
                    r'<a[^>]*href="([^"]+)"[^>]*class="result__a"[^>]*>(.+?)</a>',
                    resp.text, re.DOTALL
                ):
                    link = match.group(1)
                    title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                    
                    if any(kw in title for kw in ['公式', '选股', '指标', '策略', '信号']):
                        strat = CrawledStrategy(
                            source="TDX_Forum",
                            title=title[:100],
                            url=link,
                            author="forum_user",
                            content=title,
                            strategy_type='screener',
                            market='CN',
                            tags=['tdx', 'formula', 'cn_stock'],
                            crawled_at=datetime.now().isoformat(),
                        )
                        strat.content_hash = hashlib.md5(link.encode()).hexdigest()
                        strategies.append(strat)
                
                if len(strategies) >= max_results:
                    break
            except Exception as e:
                logger.warning(f"TDX crawl error: {e}")
            
            time.sleep(2)
        
        return strategies[:max_results]


# =============================================================
# 3. 聚宽/米筐 量化社区
# =============================================================
class QuantCommunityCrawler:
    """量化社区策略爬取 (聚宽、米筐等)"""
    
    def crawl(self, max_results: int = 20) -> List[CrawledStrategy]:
        strategies = []
        
        if requests is None:
            return strategies
        
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        
        queries = [
            "聚宽 量化策略 多因子 alpha",
            "米筐 ricequant 策略 回测 2024",
            "python 量化 选股策略 机器学习",
            "backtrader strategy python stock",
        ]
        
        for query in queries[:2]:
            try:
                url = f"https://html.duckduckgo.com/html/?q={query}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                
                for match in re.finditer(
                    r'<a[^>]*href="([^"]+)"[^>]*class="result__a"[^>]*>(.+?)</a>',
                    resp.text, re.DOTALL
                ):
                    link = match.group(1)
                    title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                    
                    if any(kw in title.lower() for kw in ['策略', 'strategy', '因子', 'alpha', '选股', '回测']):
                        strat = CrawledStrategy(
                            source="Quant_Community",
                            title=title[:100],
                            url=link,
                            author="community",
                            content=title,
                            strategy_type='buy_sell',
                            market='CN' if any(kw in title for kw in ['聚宽', '米筐', 'A股']) else 'US',
                            tags=['quant', 'community'],
                            crawled_at=datetime.now().isoformat(),
                        )
                        strat.content_hash = hashlib.md5(link.encode()).hexdigest()
                        strategies.append(strat)
                
                if len(strategies) >= max_results:
                    break
            except Exception as e:
                logger.warning(f"Quant community crawl error: {e}")
            
            time.sleep(2)
        
        return strategies[:max_results]


# =============================================================
# 4. Reddit r/algotrading
# =============================================================
class RedditAlgoTradingCrawler:
    """Reddit r/algotrading 策略爬取"""
    
    def crawl(self, max_results: int = 20) -> List[CrawledStrategy]:
        strategies = []
        
        if requests is None:
            return strategies
        
        headers = {
            'User-Agent': 'Coral_Creek_Bot/1.0 (Strategy Research)',
            'Accept': 'application/json',
        }
        
        # Reddit JSON API (不需要 OAuth)
        subreddits = ['algotrading', 'quantfinance', 'SystemTrading']
        
        for sub in subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit=25"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                for post in data.get('data', {}).get('children', []):
                    d = post.get('data', {})
                    title = d.get('title', '')
                    selftext = d.get('selftext', '')
                    score = d.get('score', 0)
                    
                    # 过滤低质量帖子
                    if score < 10:
                        continue
                    
                    # 检查是否与策略相关
                    strategy_keywords = [
                        'strategy', 'backtest', 'signal', 'indicator',
                        'moving average', 'rsi', 'macd', 'momentum',
                        'alpha', 'edge', 'profitable', 'returns'
                    ]
                    
                    text = (title + ' ' + selftext).lower()
                    if not any(kw in text for kw in strategy_keywords):
                        continue
                    
                    strat = CrawledStrategy(
                        source="Reddit",
                        title=title[:100],
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        author=d.get('author', 'anon'),
                        content=selftext[:2000],
                        strategy_type='buy_sell',
                        market='US',
                        tags=['reddit', sub],
                        likes=score,
                        crawled_at=datetime.now().isoformat(),
                    )
                    strat.content_hash = hashlib.md5(
                        d.get('permalink', '').encode()
                    ).hexdigest()
                    strategies.append(strat)
                    
                    if len(strategies) >= max_results:
                        break
                    
            except Exception as e:
                logger.warning(f"Reddit crawl error: {e}")
            
            time.sleep(2)
        
        return strategies[:max_results]


# =============================================================
# 5. GitHub Trending Quant Repos
# =============================================================
class GitHubQuantCrawler:
    """GitHub 量化策略仓库爬取"""
    
    def crawl(self, max_results: int = 10) -> List[CrawledStrategy]:
        strategies = []
        
        if requests is None:
            return strategies
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Coral_Creek_Strategy_Crawler',
        }
        
        token = os.getenv('GITHUB_TOKEN', '')
        if token:
            headers['Authorization'] = f'token {token}'
        
        # 搜索量化策略仓库
        queries = [
            'trading strategy python stars:>50 pushed:>2025-01-01',
            'stock screener alpha python stars:>20 pushed:>2025-01-01',
            'quantitative trading backtest stars:>100',
        ]
        
        for query in queries[:2]:
            try:
                url = "https://api.github.com/search/repositories"
                params = {
                    'q': query,
                    'sort': 'updated',
                    'order': 'desc',
                    'per_page': 10,
                }
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code != 200:
                    continue
                
                for repo in resp.json().get('items', []):
                    strat = CrawledStrategy(
                        source="GitHub",
                        title=repo.get('full_name', ''),
                        url=repo.get('html_url', ''),
                        author=repo.get('owner', {}).get('login', ''),
                        content=repo.get('description', '') or '',
                        strategy_type='buy_sell',
                        market='all',
                        tags=['github', 'open_source'] + (repo.get('topics', []) or []),
                        likes=repo.get('stargazers_count', 0),
                        crawled_at=datetime.now().isoformat(),
                    )
                    strat.content_hash = hashlib.md5(
                        repo.get('html_url', '').encode()
                    ).hexdigest()
                    strategies.append(strat)
                    
                    if len(strategies) >= max_results:
                        break
                        
            except Exception as e:
                logger.warning(f"GitHub crawl error: {e}")
            
            time.sleep(2)
        
        return strategies[:max_results]


# =============================================================
# 聚合爬虫
# =============================================================
class StrategyCrawler:
    """统一策略爬虫"""
    
    def __init__(self):
        self.crawlers = {
            'tradingview': TradingViewCrawler(),
            'tdx': TDXFormulaCrawler(),
            'quant': QuantCommunityCrawler(),
            'reddit': RedditAlgoTradingCrawler(),
            'github': GitHubQuantCrawler(),
        }
    
    def crawl_all(self, max_per_source: int = 10) -> List[CrawledStrategy]:
        """爬取所有来源"""
        all_strategies = []
        
        for name, crawler in self.crawlers.items():
            try:
                logger.info(f"Crawling {name}...")
                results = crawler.crawl(max_results=max_per_source)
                logger.info(f"  {name}: {len(results)} strategies")
                all_strategies.extend(results)
            except Exception as e:
                logger.warning(f"Crawler {name} failed: {e}")
        
        return all_strategies
    
    def crawl_and_save(self, max_per_source: int = 10) -> int:
        """爬取并存入数据库"""
        conn = _init_strategy_db()
        strategies = self.crawl_all(max_per_source)
        
        saved = 0
        for s in strategies:
            if _save_strategy(conn, s):
                saved += 1
        
        conn.close()
        logger.info(f"Saved {saved}/{len(strategies)} strategies")
        return saved
    
    def get_stats(self) -> Dict:
        """获取爬取统计"""
        conn = _init_strategy_db()
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM crawled_strategies")
        total = c.fetchone()[0]
        
        c.execute("SELECT source, COUNT(*) FROM crawled_strategies GROUP BY source")
        by_source = {r[0]: r[1] for r in c.fetchall()}
        
        c.execute("SELECT COUNT(*) FROM crawled_strategies WHERE backtest_done = 1")
        backtested = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM crawled_strategies WHERE backtest_sharpe > 1.0")
        good = c.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total,
            'by_source': by_source,
            'backtested': backtested,
            'good_strategies': good,
        }


# =============================================================
# 策略解析器 — 从文本提取交易规则
# =============================================================
def parse_strategy_rules(text: str) -> Dict:
    """
    从策略描述中提取买卖规则
    
    支持格式:
    - Pine Script 片段
    - 通达信公式
    - 自然语言描述
    """
    rules = {
        'buy_conditions': [],
        'sell_conditions': [],
        'indicators_used': [],
        'timeframe': 'daily',
    }
    
    # 检测指标
    indicator_patterns = {
        'RSI': r'(?i)\brsi\b',
        'MACD': r'(?i)\bmacd\b',
        'MA': r'(?i)\b(sma|ema|ma\d+|移动平均)\b',
        'Volume': r'(?i)\b(volume|vol|成交量|放量|缩量)\b',
        'KDJ': r'(?i)\b(kdj|随机)\b',
        'CCI': r'(?i)\b(cci)\b',
        'Bollinger': r'(?i)\b(bband|bollinger|布林)\b',
        'ATR': r'(?i)\b(atr|真实波幅)\b',
        'ADX': r'(?i)\b(adx|趋势强度)\b',
    }
    
    for name, pattern in indicator_patterns.items():
        if re.search(pattern, text):
            rules['indicators_used'].append(name)
    
    # 提取买入条件
    buy_patterns = [
        r'(?:买入|buy|long|做多).*?(?:当|if|when)?\s*(.+?)(?:\n|;|。)',
        r'(?:entry|进场|开仓).*?(?:条件|condition).*?(.+?)(?:\n|;|。)',
        r'(?:golden cross|金叉|上穿|突破)(.+?)(?:\n|;|。)',
    ]
    
    for pattern in buy_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        rules['buy_conditions'].extend([m.strip()[:100] for m in matches])
    
    # 提取卖出条件
    sell_patterns = [
        r'(?:卖出|sell|short|做空).*?(?:当|if|when)?\s*(.+?)(?:\n|;|。)',
        r'(?:exit|离场|平仓).*?(?:条件|condition).*?(.+?)(?:\n|;|。)',
        r'(?:dead cross|死叉|下穿|跌破)(.+?)(?:\n|;|。)',
    ]
    
    for pattern in sell_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        rules['sell_conditions'].extend([m.strip()[:100] for m in matches])
    
    return rules


# =============================================================
# 运行入口
# =============================================================
def daily_crawl():
    """每日爬取入口 (GitHub Action 调用)"""
    print(f"🕷️ Strategy Crawler — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)
    
    crawler = StrategyCrawler()
    saved = crawler.crawl_and_save(max_per_source=15)
    
    stats = crawler.get_stats()
    print(f"\n📊 Database stats:")
    print(f"  Total strategies: {stats['total']}")
    print(f"  By source: {stats['by_source']}")
    print(f"  Backtested: {stats['backtested']}")
    print(f"  Good (Sharpe>1): {stats['good_strategies']}")
    
    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    daily_crawl()
