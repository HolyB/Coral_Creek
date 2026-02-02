#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件分类器
使用 LLM 对新闻进行结构化解析
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from .models import NewsEvent, EventType, Sentiment

logger = logging.getLogger(__name__)


# 事件类型关键词映射 (用于快速预分类)
EVENT_KEYWORDS = {
    EventType.EARNINGS: [
        'earnings', 'revenue', 'profit', 'loss', 'EPS', 'quarterly', 
        'fiscal', 'beat', 'miss', '财报', '业绩', '利润', '营收', '净利'
    ],
    EventType.GUIDANCE: [
        'guidance', 'outlook', 'forecast', 'expect', 'raise', 'lower',
        '指引', '预期', '展望', '上调', '下调'
    ],
    EventType.ANALYST: [
        'upgrade', 'downgrade', 'rating', 'target', 'analyst', 'price target',
        '评级', '目标价', '分析师', '买入', '卖出', '持有'
    ],
    EventType.INSIDER: [
        'insider', 'executive', 'CEO', 'CFO', 'buy', 'sell', 'stock purchase',
        '增持', '减持', '回购', '高管'
    ],
    EventType.M_AND_A: [
        'acquire', 'merger', 'acquisition', 'takeover', 'buyout', 'spin-off',
        '并购', '收购', '合并', '重组', '拆分'
    ],
    EventType.PRODUCT: [
        'launch', 'product', 'release', 'announce', 'new', 'FDA', 'approval',
        '发布', '产品', '新品', '批准', '认证'
    ],
    EventType.LEGAL: [
        'lawsuit', 'SEC', 'investigation', 'fine', 'penalty', 'court',
        '诉讼', '调查', '罚款', '监管', '违规'
    ],
    EventType.MACRO: [
        'Fed', 'interest rate', 'tariff', 'policy', 'regulation', 'tax',
        '利率', '关税', '政策', '监管', '税收'
    ],
    EventType.PARTNERSHIP: [
        'partnership', 'deal', 'contract', 'agreement', 'collaboration',
        '合作', '协议', '签约', '订单'
    ],
    EventType.DIVIDEND: [
        'dividend', 'payout', 'distribution', '分红', '派息', '股息'
    ],
    EventType.OFFERING: [
        'offering', 'issuance', 'secondary', 'dilution', '增发', '配股', '定增'
    ]
}

# 情感关键词
SENTIMENT_KEYWORDS = {
    'very_bullish': [
        'surge', 'soar', 'skyrocket', 'breakthrough', 'record high', 
        'blowout', 'massive', 'exceptional', '暴涨', '大涨', '突破', '创新高'
    ],
    'bullish': [
        'rise', 'gain', 'up', 'beat', 'exceed', 'positive', 'growth', 
        'strong', 'improve', '上涨', '增长', '利好', '超预期'
    ],
    'bearish': [
        'fall', 'drop', 'decline', 'miss', 'weak', 'concern', 'slow',
        '下跌', '下滑', '低于预期', '利空'
    ],
    'very_bearish': [
        'crash', 'plunge', 'collapse', 'crisis', 'disaster', 'bankruptcy',
        '暴跌', '崩盘', '危机', '爆雷', '破产'
    ]
}


class RuleBasedClassifier:
    """规则分类器 (快速、无需 API)"""
    
    def classify(self, event: NewsEvent) -> Tuple[EventType, Sentiment]:
        """基于关键词分类"""
        text = f"{event.title} {event.summary}".lower()
        
        # 1. 事件类型
        event_type = EventType.OTHER
        max_matches = 0
        
        for etype, keywords in EVENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw.lower() in text)
            if matches > max_matches:
                max_matches = matches
                event_type = etype
        
        # 2. 情感分析
        sentiment = Sentiment.NEUTRAL
        
        very_bullish = sum(1 for kw in SENTIMENT_KEYWORDS['very_bullish'] if kw.lower() in text)
        bullish = sum(1 for kw in SENTIMENT_KEYWORDS['bullish'] if kw.lower() in text)
        bearish = sum(1 for kw in SENTIMENT_KEYWORDS['bearish'] if kw.lower() in text)
        very_bearish = sum(1 for kw in SENTIMENT_KEYWORDS['very_bearish'] if kw.lower() in text)
        
        if very_bearish > 0:
            sentiment = Sentiment.VERY_BEARISH
        elif very_bullish > 0:
            sentiment = Sentiment.VERY_BULLISH
        elif bearish > bullish:
            sentiment = Sentiment.BEARISH
        elif bullish > bearish:
            sentiment = Sentiment.BULLISH
        
        return event_type, sentiment


class LLMClassifier:
    """LLM 分类器 (更准确)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self._client = None
    
    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                logger.error(f"Failed to init Gemini: {e}")
        return self._client
    
    def classify(self, event: NewsEvent) -> Tuple[EventType, Sentiment, Dict]:
        """使用 LLM 分类新闻"""
        client = self._get_client()
        if not client:
            # 回退到规则分类
            rule_classifier = RuleBasedClassifier()
            event_type, sentiment = rule_classifier.classify(event)
            return event_type, sentiment, {}
        
        prompt = f"""分析以下股票新闻，返回 JSON 格式结果。

新闻标题: {event.title}
股票代码: {event.symbol}
来源: {event.source}

返回格式:
{{
    "event_type": "财报/评级/产品/法律/并购/合作/宏观/其他 (选一个)",
    "sentiment": "强烈利好/利好/中性/利空/强烈利空 (选一个)",
    "expected_impact_pct": 数字 (预估对股价影响的百分比, 如 +3.5 或 -2.0),
    "confidence": 数字 (置信度 0-100),
    "summary": "一句话总结 (20字以内)",
    "key_entities": ["实体1", "实体2"],
    "time_horizon": "immediate/short/medium/long"
}}

只返回 JSON，不要其他内容。"""

        try:
            response = client.generate_content(prompt)
            text = response.text.strip()
            
            # 清理可能的 markdown 包装
            if text.startswith('```'):
                text = text.split('\n', 1)[1]
            if text.endswith('```'):
                text = text.rsplit('\n', 1)[0]
            if text.startswith('json'):
                text = text[4:].strip()
            
            result = json.loads(text)
            
            # 解析事件类型
            event_type_map = {
                '财报': EventType.EARNINGS,
                '业绩': EventType.EARNINGS,
                '评级': EventType.ANALYST,
                '产品': EventType.PRODUCT,
                '法律': EventType.LEGAL,
                '并购': EventType.M_AND_A,
                '合作': EventType.PARTNERSHIP,
                '宏观': EventType.MACRO,
                '分红': EventType.DIVIDEND,
            }
            event_type_str = result.get('event_type', '其他')
            event_type = EventType.OTHER
            for key, etype in event_type_map.items():
                if key in event_type_str:
                    event_type = etype
                    break
            
            # 解析情感
            sentiment_map = {
                '强烈利好': Sentiment.VERY_BULLISH,
                '利好': Sentiment.BULLISH,
                '中性': Sentiment.NEUTRAL,
                '利空': Sentiment.BEARISH,
                '强烈利空': Sentiment.VERY_BEARISH,
            }
            sentiment_str = result.get('sentiment', '中性')
            sentiment = sentiment_map.get(sentiment_str, Sentiment.NEUTRAL)
            
            return event_type, sentiment, result
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            rule_classifier = RuleBasedClassifier()
            event_type, sentiment = rule_classifier.classify(event)
            return event_type, sentiment, {}


class EventClassifier:
    """统一事件分类接口"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier() if use_llm else None
    
    def classify(self, event: NewsEvent) -> NewsEvent:
        """分类单条新闻"""
        if self.use_llm and self.llm_classifier:
            event_type, sentiment, extra = self.llm_classifier.classify(event)
            
            # 更新事件
            event.event_type = event_type
            event.sentiment = sentiment
            if extra:
                event.summary = extra.get('summary', event.summary)
                event.keywords = extra.get('key_entities', [])
        else:
            event_type, sentiment = self.rule_classifier.classify(event)
            event.event_type = event_type
            event.sentiment = sentiment
        
        event.is_classified = True
        event.classified_at = datetime.now()
        
        return event
    
    def classify_batch(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """批量分类"""
        classified = []
        
        for event in events:
            try:
                classified_event = self.classify(event)
                classified.append(classified_event)
            except Exception as e:
                logger.error(f"Classification error for {event.title[:30]}: {e}")
                event.is_classified = False
                classified.append(event)
        
        return classified


# 全局单例
_event_classifier = None

def get_event_classifier(use_llm: bool = True) -> EventClassifier:
    global _event_classifier
    if _event_classifier is None:
        _event_classifier = EventClassifier(use_llm=use_llm)
    return _event_classifier
