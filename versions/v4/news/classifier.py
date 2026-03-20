#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件分类器 (v2)
================
- 规则分类器 (快速, 无需API)
- Gemini 批量分类器 (5-10条一批, 更准确)
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from .models import NewsEvent, EventType, Sentiment

logger = logging.getLogger(__name__)


# 事件类型关键词映射
EVENT_KEYWORDS = {
    EventType.EARNINGS: [
        'earnings', 'revenue', 'profit', 'loss', 'EPS', 'quarterly results',
        'fiscal', 'beat expectations', 'miss expectations', 'guidance',
        '财报', '业绩', '利润', '营收', '净利', '季报', '年报'
    ],
    EventType.GUIDANCE: [
        'guidance', 'outlook', 'forecast', 'raises forecast', 'lowers forecast',
        '指引', '预期', '展望', '上调', '下调'
    ],
    EventType.ANALYST: [
        'upgrade', 'downgrade', 'rating', 'price target', 'analyst',
        'overweight', 'underweight', 'outperform', 'buy rating',
        '评级', '目标价', '分析师', '买入评级', '卖出评级'
    ],
    EventType.INSIDER: [
        'insider', 'executive buy', 'executive sell', 'stock purchase',
        'buyback', 'repurchase',
        '增持', '减持', '回购', '高管交易'
    ],
    EventType.M_AND_A: [
        'acquire', 'merger', 'acquisition', 'takeover', 'buyout', 
        'spin-off', 'split', 'restructure',
        '并购', '收购', '合并', '重组', '拆分'
    ],
    EventType.PRODUCT: [
        'launch', 'new product', 'release', 'FDA approval', 'patent',
        'innovation', 'technology',
        '发布', '新产品', '新品', '批准', '专利'
    ],
    EventType.LEGAL: [
        'lawsuit', 'SEC', 'investigation', 'fine', 'penalty', 'court',
        'settlement', 'regulation',
        '诉讼', '调查', '罚款', '监管', '违规'
    ],
    EventType.MACRO: [
        'Fed', 'interest rate', 'tariff', 'policy', 'regulation', 'tax',
        'inflation', 'GDP', 'jobs report',
        '利率', '关税', '政策', '监管', '税收', '通胀'
    ],
    EventType.PARTNERSHIP: [
        'partnership', 'deal', 'contract', 'agreement', 'collaboration',
        '合作', '协议', '签约', '订单'
    ],
    EventType.DIVIDEND: [
        'dividend', 'payout', 'distribution', 'yield',
        '分红', '派息', '股息'
    ],
}

# 情感关键词 (更精确)
SENTIMENT_KEYWORDS = {
    'very_bullish': [
        'surge', 'soar', 'skyrocket', 'breakthrough', 'record high',
        'blowout', 'massive beat', 'exceptional', 'all-time high',
        '暴涨', '大涨', '突破', '创新高', '历史新高'
    ],
    'bullish': [
        'rises', 'gains', 'beat', 'exceeds', 'positive', 'growth',
        'strong', 'improves', 'higher', 'upgrades', 'rally', 'buy point',
        '上涨', '增长', '利好', '超预期', '上调'
    ],
    'bearish': [
        'falls', 'drops', 'declines', 'misses', 'weak', 'concern',
        'slows', 'lower', 'downgrades', 'questions', 'pressure', 'risk',
        '下跌', '下滑', '低于预期', '利空', '下调'
    ],
    'very_bearish': [
        'crash', 'plunge', 'collapse', 'crisis', 'disaster', 'bankruptcy',
        'investigation', 'fraud', 'massive loss',
        '暴跌', '崩盘', '危机', '爆雷', '破产'
    ]
}


class RuleBasedClassifier:
    """规则分类器 (快速、无需 API)"""
    
    def classify(self, event: NewsEvent) -> Tuple[EventType, Sentiment]:
        text = f"{event.title} {event.summary}".lower()
        
        # 事件类型 (加权匹配)
        event_type = EventType.OTHER
        max_score = 0
        
        for etype, keywords in EVENT_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw.lower() in text:
                    # 更长的关键词权重更大
                    score += len(kw.split())
            if score > max_score:
                max_score = score
                event_type = etype
        
        # 情感分析 (加权)
        sentiment = Sentiment.NEUTRAL
        
        vb = sum(2 for kw in SENTIMENT_KEYWORDS['very_bullish'] if kw.lower() in text)
        b = sum(1 for kw in SENTIMENT_KEYWORDS['bullish'] if kw.lower() in text)
        be = sum(1 for kw in SENTIMENT_KEYWORDS['bearish'] if kw.lower() in text)
        vbe = sum(2 for kw in SENTIMENT_KEYWORDS['very_bearish'] if kw.lower() in text)
        
        bull_score = vb + b
        bear_score = vbe + be
        
        if vbe > 0 and bear_score > bull_score:
            sentiment = Sentiment.VERY_BEARISH
        elif vb > 0 and bull_score > bear_score:
            sentiment = Sentiment.VERY_BULLISH
        elif bear_score > bull_score + 1:
            sentiment = Sentiment.BEARISH
        elif bull_score > bear_score + 1:
            sentiment = Sentiment.BULLISH
        
        return event_type, sentiment


class GeminiBatchClassifier:
    """Gemini 批量分类器 — 多条新闻一次性分类"""
    
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        if self._client is not None:
            return self._client
        
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                try:
                    import streamlit as st
                    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                        api_key = st.secrets['GEMINI_API_KEY']
                except:
                    pass
            
            if not api_key:
                return None
            
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel('gemini-2.5-flash')
            return self._client
        except Exception as e:
            logger.error(f"Gemini init error: {e}")
            return None
    
    def classify_batch(self, events: List[NewsEvent]) -> List[Tuple[EventType, Sentiment, Dict]]:
        """批量分类 (最多10条一批)"""
        client = self._get_client()
        if not client or not events:
            return [(EventType.OTHER, Sentiment.NEUTRAL, {}) for _ in events]
        
        # 构建批量 prompt
        news_list = ""
        for i, e in enumerate(events[:10]):
            news_list += f"{i+1}. [{e.symbol}] {e.title}\n"
            if e.summary and e.summary != e.title:
                news_list += f"   摘要: {e.summary[:80]}\n"
        
        prompt = f"""分析以下{len(events[:10])}条股票新闻，返回 JSON 数组。

{news_list}

对每条新闻返回:
[
  {{
    "index": 1,
    "event_type": "财报/评级/产品/法律/并购/合作/宏观/社交讨论/其他",
    "sentiment": "强烈利好/利好/中性/利空/强烈利空",
    "impact": 预估影响百分比(如+3.5或-2.0),
    "summary": "一句话中文摘要(15字)"
  }},
  ...
]

规则:
- 社交媒体帖子(StockTwits等)归类为"社交讨论"
- 只输出JSON数组，不要其他内容"""
        
        try:
            response = client.generate_content(prompt)
            text = response.text.strip()
            
            # 清理 markdown
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            if text.startswith('json'):
                text = text[4:].strip()
            
            results = json.loads(text)
            
            # 解析结果
            type_map = {
                '财报': EventType.EARNINGS, '业绩': EventType.EARNINGS,
                '评级': EventType.ANALYST, '产品': EventType.PRODUCT,
                '法律': EventType.LEGAL, '并购': EventType.M_AND_A,
                '合作': EventType.PARTNERSHIP, '宏观': EventType.MACRO,
                '分红': EventType.DIVIDEND, '社交': EventType.OTHER,
            }
            sent_map = {
                '强烈利好': Sentiment.VERY_BULLISH,
                '利好': Sentiment.BULLISH,
                '中性': Sentiment.NEUTRAL,
                '利空': Sentiment.BEARISH,
                '强烈利空': Sentiment.VERY_BEARISH,
            }
            
            classified = []
            for item in results:
                # 事件类型
                et_str = item.get('event_type', '其他')
                et = EventType.OTHER
                for k, v in type_map.items():
                    if k in et_str:
                        et = v
                        break
                
                # 情感
                s_str = item.get('sentiment', '中性')
                s = sent_map.get(s_str, Sentiment.NEUTRAL)
                
                classified.append((et, s, item))
            
            # 补齐
            while len(classified) < len(events):
                classified.append((EventType.OTHER, Sentiment.NEUTRAL, {}))
            
            return classified[:len(events)]
            
        except Exception as e:
            logger.error(f"Gemini batch classify error: {e}")
            return [(EventType.OTHER, Sentiment.NEUTRAL, {}) for _ in events]


class EventClassifier:
    """统一事件分类接口"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.rule_classifier = RuleBasedClassifier()
        self.gemini_classifier = GeminiBatchClassifier() if use_llm else None
    
    def classify(self, event: NewsEvent) -> NewsEvent:
        """分类单条新闻"""
        if event.is_classified:
            return event  # 跳过已分类的 (如 StockTwits 自带情绪)
        
        event_type, sentiment = self.rule_classifier.classify(event)
        event.event_type = event_type
        event.sentiment = sentiment
        event.is_classified = True
        event.classified_at = datetime.now()
        return event
    
    def classify_batch(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """批量分类 — 已分类的跳过，未分类的用 Gemini 批量处理"""
        # 分离已分类 vs 未分类
        need_classify = []
        need_classify_indices = []
        
        for i, event in enumerate(events):
            if not event.is_classified:
                need_classify.append(event)
                need_classify_indices.append(i)
        
        if not need_classify:
            return events
        
        # 尝试 Gemini 批量
        if self.use_llm and self.gemini_classifier:
            try:
                results = self.gemini_classifier.classify_batch(need_classify)
                for j, (et, s, extra) in enumerate(results):
                    idx = need_classify_indices[j]
                    events[idx].event_type = et
                    events[idx].sentiment = s
                    events[idx].is_classified = True
                    events[idx].classified_at = datetime.now()
                    if extra:
                        summary = extra.get('summary', '')
                        if summary:
                            events[idx].summary = summary
                return events
            except Exception as e:
                logger.error(f"Gemini batch failed, falling back to rules: {e}")
        
        # 回退到规则分类
        for idx in need_classify_indices:
            events[idx] = self.classify(events[idx])
        
        return events


# 全局单例 (支持重建)
_event_classifier = None
_classifier_use_llm = None

def get_event_classifier(use_llm: bool = True) -> EventClassifier:
    global _event_classifier, _classifier_use_llm
    if _event_classifier is None or _classifier_use_llm != use_llm:
        _event_classifier = EventClassifier(use_llm=use_llm)
        _classifier_use_llm = use_llm
    return _event_classifier
