#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Intelligence Module - 大语言模型智能分析

功能:
- 新闻情感分析
- 自然语言查询
- 市场报告生成
"""
import os
import sys
from typing import Dict, List, Optional

# 尝试导入 OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 尝试导入 Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def check_llm_available() -> Dict[str, bool]:
    """检查 LLM 库是否可用"""
    return {
        'openai': OPENAI_AVAILABLE,
        'anthropic': ANTHROPIC_AVAILABLE
    }


def get_openai_client() -> Optional['OpenAI']:
    """获取 OpenAI 客户端"""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None
    
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> Optional['anthropic.Anthropic']:
    """获取 Anthropic 客户端"""
    if not ANTHROPIC_AVAILABLE:
        return None
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    
    return anthropic.Anthropic(api_key=api_key)


class LLMAnalyzer:
    """LLM 分析器"""
    
    def __init__(self, provider: str = 'openai'):
        """
        初始化分析器
        
        Args:
            provider: 'openai' 或 'anthropic'
        """
        self.provider = provider
        self.client = None
        
        if provider == 'openai':
            self.client = get_openai_client()
            self.model = 'gpt-4o-mini'
        elif provider == 'anthropic':
            self.client = get_anthropic_client()
            self.model = 'claude-3-haiku-20240307'
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self.client is not None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        分析文本情感
        
        Args:
            text: 新闻或评论文本
        
        Returns:
            Dict with sentiment, score, reasoning
        """
        if not self.is_available():
            return {'error': 'LLM client not available'}
        
        prompt = f"""分析以下财经文本的市场情感。

文本:
{text}

请返回JSON格式:
{{
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "key_points": ["要点1", "要点2"],
    "reasoning": "分析原因"
}}
"""
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的金融分析师。"},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                import json
                return json.loads(response.choices[0].message.content)
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                import json
                return json.loads(response.content[0].text)
        
        except Exception as e:
            return {'error': str(e)}
    
    def natural_query(self, query: str, context: str = "") -> str:
        """
        自然语言查询
        
        Args:
            query: 用户问题 (如 "找出超卖的科技股")
            context: 当前市场上下文
        
        Returns:
            回答文本
        """
        if not self.is_available():
            return "LLM client not available"
        
        system_prompt = """你是 Coral Creek 智能量化系统的 AI 助手。
你可以帮助用户:
1. 解释技术指标 (BLUE 信号, ADX, RSI 等)
2. 分析市场趋势
3. 回答量化交易相关问题

当前系统支持:
- BLUE 信号: 综合超卖指标 (>100 为买入信号)
- 黑马/掘底: 特殊反转信号
- 周线/月线共振: 多周期确认

请用简洁专业的语言回答。"""
        
        user_prompt = query
        if context:
            user_prompt = f"当前市场数据:\n{context}\n\n用户问题: {query}"
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_market_report(self, signals: List[Dict]) -> str:
        """
        生成市场报告
        
        Args:
            signals: 当日信号列表
        
        Returns:
            Markdown 格式的市场报告
        """
        if not self.is_available():
            return "LLM client not available"
        
        # 构建信号摘要
        if not signals:
            signal_summary = "今日无触发信号"
        else:
            signal_summary = f"今日共有 {len(signals)} 个 BLUE 信号:\n"
            for s in signals[:10]:  # 最多展示 10 个
                signal_summary += f"- {s.get('symbol', 'N/A')}: BLUE={s.get('blue_daily', 0):.1f}, 价格=${s.get('price', 0):.2f}\n"
        
        prompt = f"""基于以下信号数据，生成一份简洁的每日市场报告。

信号摘要:
{signal_summary}

请生成 Markdown 格式报告，包含:
1. 📊 市场概览 (2-3句话)
2. 🔥 热门信号 (如果有)
3. ⚠️ 风险提示
4. 💡 操作建议

保持简洁专业。"""
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的量化分析师，负责撰写每日市场报告。"},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        
        except Exception as e:
            return f"Error generating report: {str(e)}"


def quick_sentiment_check(text: str, provider: str = 'openai') -> Dict:
    """快速情感分析"""
    analyzer = LLMAnalyzer(provider)
    return analyzer.analyze_sentiment(text)


def ask_ai(question: str, provider: str = 'openai') -> str:
    """快速 AI 问答"""
    analyzer = LLMAnalyzer(provider)
    return analyzer.natural_query(question)


if __name__ == "__main__":
    print("LLM Module Status:")
    status = check_llm_available()
    print(f"  OpenAI: {'✅' if status['openai'] else '❌'}")
    print(f"  Anthropic: {'✅' if status['anthropic'] else '❌'}")
    
    # 测试
    if status['openai'] and os.environ.get('OPENAI_API_KEY'):
        print("\nTesting OpenAI...")
        result = ask_ai("什么是 BLUE 指标？")
        print(f"Response: {result[:200]}...")
