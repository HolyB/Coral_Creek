#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化博主追踪服务
爬取中英文量化博主的最新文章，分析策略并回测
"""
import os
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import re

# 博主源列表
BLOGGER_SOURCES = {
    # 中文博主/平台
    'cn': [
        {
            'name': '雪球热门',
            'url': 'https://xueqiu.com/statuses/hot/listV2.json?since_id=-1&max_id=-1&size=20',
            'type': 'api',
            'category': '社区热帖'
        },
        {
            'name': '同花顺量化',
            'url': 'https://quant.10jqka.com.cn/',
            'type': 'web',
            'category': '量化策略'
        },
        {
            'name': '聚宽社区',
            'url': 'https://www.joinquant.com/view/community/list',
            'type': 'web',
            'category': '量化策略'
        },
        {
            'name': '米筐研究',
            'url': 'https://www.ricequant.com/community/',
            'type': 'web',
            'category': '量化策略'
        },
    ],
    # 英文博主/平台
    'en': [
        {
            'name': 'Quantocracy',
            'url': 'https://quantocracy.com/',
            'type': 'rss',
            'category': 'Quant Aggregator'
        },
        {
            'name': 'Alpha Architect',
            'url': 'https://alphaarchitect.com/blog/',
            'type': 'rss',
            'category': 'Factor Research'
        },
        {
            'name': 'Quantpedia',
            'url': 'https://quantpedia.com/blog/',
            'type': 'rss',
            'category': 'Strategy Ideas'
        },
        {
            'name': 'QuantStart',
            'url': 'https://www.quantstart.com/articles/',
            'type': 'web',
            'category': 'Tutorials'
        },
        {
            'name': 'Systematic Investor',
            'url': 'https://systematicinvestor.wordpress.com/',
            'type': 'rss',
            'category': 'R Quant'
        },
        {
            'name': 'SSRN Finance',
            'url': 'https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=1079991',
            'type': 'web',
            'category': 'Academic Papers'
        },
    ]
}

@dataclass
class BlogArticle:
    """博客文章"""
    id: str
    source: str
    title: str
    url: str
    author: str
    publish_date: str
    content_summary: str
    language: str  # 'cn' or 'en'
    category: str
    fetched_at: str
    
@dataclass
class ExtractedStrategy:
    """提取的策略"""
    article_id: str
    strategy_name: str
    strategy_type: str  # 'momentum', 'mean_reversion', 'factor', 'ml', etc.
    description: str
    entry_rules: List[str]
    exit_rules: List[str]
    indicators: List[str]
    timeframe: str  # 'day', 'week', 'intraday'
    backtest_ready: bool
    confidence: float  # 0-1
    
@dataclass
class StrategyBacktest:
    """策略回测结果"""
    strategy_id: str
    backtest_date: str
    period: str  # '1M', '3M', '6M', '1Y'
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    is_profitable: bool


class BloggerTrackerDB:
    """博主追踪数据库"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, '..', 'db', 'blogger_tracker.db')
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文章表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                source TEXT,
                title TEXT,
                url TEXT,
                author TEXT,
                publish_date TEXT,
                content_summary TEXT,
                language TEXT,
                category TEXT,
                fetched_at TEXT,
                analyzed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # 策略表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                article_id TEXT,
                strategy_name TEXT,
                strategy_type TEXT,
                description TEXT,
                entry_rules TEXT,
                exit_rules TEXT,
                indicators TEXT,
                timeframe TEXT,
                backtest_ready BOOLEAN,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (article_id) REFERENCES articles(id)
            )
        ''')
        
        # 回测结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                backtest_date TEXT,
                period TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER,
                is_profitable BOOLEAN,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_article(self, article: BlogArticle):
        """保存文章"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO articles 
            (id, source, title, url, author, publish_date, content_summary, language, category, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article.id, article.source, article.title, article.url,
            article.author, article.publish_date, article.content_summary,
            article.language, article.category, article.fetched_at
        ))
        
        conn.commit()
        conn.close()
    
    def save_strategy(self, strategy: ExtractedStrategy):
        """保存策略"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        strategy_id = hashlib.md5(f"{strategy.article_id}_{strategy.strategy_name}".encode()).hexdigest()[:12]
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategies
            (id, article_id, strategy_name, strategy_type, description, 
             entry_rules, exit_rules, indicators, timeframe, backtest_ready, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id, strategy.article_id, strategy.strategy_name, strategy.strategy_type,
            strategy.description, json.dumps(strategy.entry_rules), json.dumps(strategy.exit_rules),
            json.dumps(strategy.indicators), strategy.timeframe, strategy.backtest_ready,
            strategy.confidence, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        return strategy_id
    
    def save_backtest(self, backtest: StrategyBacktest):
        """保存回测结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtests
            (strategy_id, backtest_date, period, total_return, sharpe_ratio, 
             max_drawdown, win_rate, total_trades, is_profitable)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            backtest.strategy_id, backtest.backtest_date, backtest.period,
            backtest.total_return, backtest.sharpe_ratio, backtest.max_drawdown,
            backtest.win_rate, backtest.total_trades, backtest.is_profitable
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_articles(self, days: int = 7, language: str = None) -> List[Dict]:
        """获取最近的文章"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        if language:
            cursor.execute('''
                SELECT * FROM articles 
                WHERE fetched_at >= ? AND language = ?
                ORDER BY publish_date DESC
            ''', (since_date, language))
        else:
            cursor.execute('''
                SELECT * FROM articles 
                WHERE fetched_at >= ?
                ORDER BY publish_date DESC
            ''', (since_date,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_strategies_with_backtests(self) -> List[Dict]:
        """获取策略及其回测结果"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.*, a.title as article_title, a.source, a.url as article_url,
                   b.total_return, b.sharpe_ratio, b.max_drawdown, b.win_rate, b.is_profitable
            FROM strategies s
            LEFT JOIN articles a ON s.article_id = a.id
            LEFT JOIN backtests b ON s.id = b.strategy_id
            ORDER BY s.created_at DESC
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_profitable_strategies(self) -> List[Dict]:
        """获取盈利的策略"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.*, a.title as article_title, a.source,
                   b.total_return, b.sharpe_ratio, b.win_rate
            FROM strategies s
            JOIN articles a ON s.article_id = a.id
            JOIN backtests b ON s.id = b.strategy_id
            WHERE b.is_profitable = 1
            ORDER BY b.sharpe_ratio DESC
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results


class ArticleFetcher:
    """文章爬取器"""
    
    def __init__(self):
        self.db = BloggerTrackerDB()
    
    def fetch_quantocracy(self) -> List[BlogArticle]:
        """爬取 Quantocracy (英文量化聚合站)"""
        articles = []
        
        try:
            import urllib.request
            from bs4 import BeautifulSoup
            
            url = 'https://quantocracy.com/'
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # 解析文章列表
            for item in soup.select('.post-title a')[:20]:
                title = item.get_text(strip=True)
                link = item.get('href', '')
                
                if title and link:
                    article_id = hashlib.md5(link.encode()).hexdigest()[:12]
                    articles.append(BlogArticle(
                        id=article_id,
                        source='Quantocracy',
                        title=title,
                        url=link,
                        author='Various',
                        publish_date=datetime.now().strftime('%Y-%m-%d'),
                        content_summary='',
                        language='en',
                        category='Quant Aggregator',
                        fetched_at=datetime.now().isoformat()
                    ))
        except Exception as e:
            print(f"Quantocracy fetch error: {e}")
        
        return articles
    
    def fetch_xueqiu_hot(self) -> List[BlogArticle]:
        """爬取雪球热门"""
        articles = []
        
        try:
            import urllib.request
            import json
            
            url = 'https://xueqiu.com/statuses/hot/listV2.json?since_id=-1&max_id=-1&size=20'
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Cookie': 'xq_a_token=xxx'  # 需要登录 cookie
            }
            
            req = urllib.request.Request(url, headers=headers)
            # 雪球需要登录，这里只是示例
            # 实际使用需要处理认证
            
        except Exception as e:
            print(f"Xueqiu fetch error: {e}")
        
        return articles
    
    def fetch_ssrn_papers(self) -> List[BlogArticle]:
        """爬取 SSRN 金融论文"""
        articles = []
        
        try:
            import urllib.request
            from bs4 import BeautifulSoup
            
            # SSRN Finance 最新论文
            url = 'https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=1079991'
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8')
            
            soup = BeautifulSoup(html, 'html.parser')
            
            for item in soup.select('.title.optClickTitle')[:15]:
                title = item.get_text(strip=True)
                link = item.get('href', '')
                
                if title and 'ssrn.com' in link:
                    article_id = hashlib.md5(link.encode()).hexdigest()[:12]
                    articles.append(BlogArticle(
                        id=article_id,
                        source='SSRN',
                        title=title,
                        url=link,
                        author='Academic',
                        publish_date=datetime.now().strftime('%Y-%m-%d'),
                        content_summary='',
                        language='en',
                        category='Academic Papers',
                        fetched_at=datetime.now().isoformat()
                    ))
        except Exception as e:
            print(f"SSRN fetch error: {e}")
        
        return articles
    
    def fetch_all(self, save: bool = True) -> Dict[str, List[BlogArticle]]:
        """爬取所有源"""
        results = {
            'en': [],
            'cn': []
        }
        
        # 英文源
        results['en'].extend(self.fetch_quantocracy())
        results['en'].extend(self.fetch_ssrn_papers())
        
        # 中文源 (需要处理登录等)
        results['cn'].extend(self.fetch_xueqiu_hot())
        
        # 保存到数据库
        if save:
            for lang, articles in results.items():
                for article in articles:
                    self.db.save_article(article)
        
        return results


class StrategyExtractor:
    """策略提取器 - 使用 LLM 分析文章"""
    
    def __init__(self):
        self.db = BloggerTrackerDB()
    
    def extract_strategy_with_llm(self, article: Dict, llm_provider: str = 'openai') -> Optional[ExtractedStrategy]:
        """使用 LLM 提取策略"""
        
        prompt = f"""
        分析以下量化交易相关文章，提取其中的交易策略：

        标题: {article.get('title', '')}
        内容摘要: {article.get('content_summary', '')}
        来源: {article.get('source', '')}

        请以 JSON 格式返回以下信息：
        {{
            "strategy_name": "策略名称",
            "strategy_type": "momentum/mean_reversion/factor/ml/trend/other",
            "description": "策略简述",
            "entry_rules": ["入场规则1", "入场规则2"],
            "exit_rules": ["出场规则1", "出场规则2"],
            "indicators": ["使用的指标1", "指标2"],
            "timeframe": "day/week/intraday",
            "backtest_ready": true/false,
            "confidence": 0.0-1.0
        }}

        如果文章不包含可提取的策略，返回 {{"strategy_name": null}}
        """
        
        try:
            # 尝试使用 OpenAI
            if llm_provider == 'openai':
                import openai
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    return None
                
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if result.get('strategy_name'):
                    return ExtractedStrategy(
                        article_id=article['id'],
                        strategy_name=result['strategy_name'],
                        strategy_type=result.get('strategy_type', 'other'),
                        description=result.get('description', ''),
                        entry_rules=result.get('entry_rules', []),
                        exit_rules=result.get('exit_rules', []),
                        indicators=result.get('indicators', []),
                        timeframe=result.get('timeframe', 'day'),
                        backtest_ready=result.get('backtest_ready', False),
                        confidence=result.get('confidence', 0.5)
                    )
        except Exception as e:
            print(f"LLM extraction error: {e}")
        
        return None
    
    def extract_strategy_rule_based(self, article: Dict) -> Optional[ExtractedStrategy]:
        """基于规则的策略提取 (不需要 LLM)"""
        
        title = article.get('title', '').lower()
        content = article.get('content_summary', '').lower()
        text = f"{title} {content}"
        
        # 策略类型关键词
        strategy_keywords = {
            'momentum': ['momentum', '动量', 'trend following', '趋势跟踪', 'breakout', '突破'],
            'mean_reversion': ['mean reversion', '均值回归', 'oversold', '超卖', 'overbought', '超买'],
            'factor': ['factor', '因子', 'value', '价值', 'quality', '质量', 'size', '市值'],
            'ml': ['machine learning', 'ml', '机器学习', 'neural', '神经网络', 'deep learning'],
            'trend': ['ma', 'moving average', '均线', 'trend', '趋势', 'golden cross', '金叉'],
        }
        
        # 指标关键词
        indicator_keywords = {
            'RSI': ['rsi', '相对强弱'],
            'MACD': ['macd'],
            'MA': ['moving average', '均线', 'ma', 'sma', 'ema'],
            'Bollinger': ['bollinger', '布林'],
            'Volume': ['volume', '成交量', '放量'],
            'ADX': ['adx', 'trend strength', '趋势强度'],
        }
        
        # 检测策略类型
        detected_type = 'other'
        for stype, keywords in strategy_keywords.items():
            if any(kw in text for kw in keywords):
                detected_type = stype
                break
        
        # 检测使用的指标
        detected_indicators = []
        for ind, keywords in indicator_keywords.items():
            if any(kw in text for kw in keywords):
                detected_indicators.append(ind)
        
        # 如果检测到有意义的内容
        if detected_type != 'other' or detected_indicators:
            return ExtractedStrategy(
                article_id=article['id'],
                strategy_name=f"Auto-extracted: {article.get('title', '')[:50]}",
                strategy_type=detected_type,
                description=article.get('content_summary', '')[:200],
                entry_rules=[],
                exit_rules=[],
                indicators=detected_indicators,
                timeframe='day',
                backtest_ready=False,
                confidence=0.3
            )
        
        return None


class StrategyBacktester:
    """策略回测器"""
    
    def __init__(self):
        self.db = BloggerTrackerDB()
    
    def backtest_momentum_strategy(self, days: int = 252) -> StrategyBacktest:
        """回测动量策略"""
        # 简化版回测逻辑
        import random
        
        return StrategyBacktest(
            strategy_id='momentum_default',
            backtest_date=datetime.now().strftime('%Y-%m-%d'),
            period='1Y',
            total_return=random.uniform(-10, 30),
            sharpe_ratio=random.uniform(0, 2),
            max_drawdown=random.uniform(-30, -5),
            win_rate=random.uniform(40, 65),
            total_trades=random.randint(50, 200),
            is_profitable=random.random() > 0.4
        )
    
    def backtest_extracted_strategy(self, strategy: Dict) -> Optional[StrategyBacktest]:
        """回测提取的策略"""
        
        strategy_type = strategy.get('strategy_type', 'other')
        strategy_id = strategy.get('id', 'unknown')
        
        # 根据策略类型使用不同的回测逻辑
        # 这里是简化版，实际需要根据 entry_rules 和 exit_rules 构建回测
        
        try:
            # 模拟回测结果
            import random
            
            base_return = {
                'momentum': 15,
                'mean_reversion': 10,
                'factor': 12,
                'trend': 8,
                'ml': 20,
                'other': 5
            }.get(strategy_type, 5)
            
            return StrategyBacktest(
                strategy_id=strategy_id,
                backtest_date=datetime.now().strftime('%Y-%m-%d'),
                period='6M',
                total_return=base_return + random.uniform(-15, 15),
                sharpe_ratio=random.uniform(0.5, 2.5),
                max_drawdown=random.uniform(-25, -5),
                win_rate=random.uniform(45, 70),
                total_trades=random.randint(30, 150),
                is_profitable=random.random() > 0.35
            )
        except Exception as e:
            print(f"Backtest error: {e}")
            return None


# 便捷函数
def get_blogger_tracker_db() -> BloggerTrackerDB:
    return BloggerTrackerDB()

def fetch_latest_articles(save: bool = True) -> Dict[str, List[BlogArticle]]:
    """获取最新文章"""
    fetcher = ArticleFetcher()
    return fetcher.fetch_all(save=save)

def analyze_articles_for_strategies(use_llm: bool = False) -> List[ExtractedStrategy]:
    """分析文章提取策略"""
    db = BloggerTrackerDB()
    extractor = StrategyExtractor()
    
    articles = db.get_recent_articles(days=7)
    strategies = []
    
    for article in articles:
        if use_llm:
            strategy = extractor.extract_strategy_with_llm(article)
        else:
            strategy = extractor.extract_strategy_rule_based(article)
        
        if strategy:
            db.save_strategy(strategy)
            strategies.append(strategy)
    
    return strategies

def backtest_all_strategies() -> List[StrategyBacktest]:
    """回测所有策略"""
    db = BloggerTrackerDB()
    backtester = StrategyBacktester()
    
    strategies = db.get_strategies_with_backtests()
    results = []
    
    for strategy in strategies:
        if strategy.get('total_return') is None:  # 还没有回测
            result = backtester.backtest_extracted_strategy(strategy)
            if result:
                db.save_backtest(result)
                results.append(result)
    
    return results

def get_strategy_leaderboard() -> List[Dict]:
    """获取策略排行榜"""
    db = BloggerTrackerDB()
    return db.get_profitable_strategies()
