#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Intelligence Module - 大语言模型智能分析

功能:
- 新闻情感分析
- 自然语言查询
- 市场报告生成
- AI 决策仪表盘 (新增)
"""
import os
import sys
import json
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

# 尝试导入 Google Generative AI (Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def check_llm_available() -> Dict[str, bool]:
    """检查 LLM 库是否可用"""
    return {
        'openai': OPENAI_AVAILABLE,
        'anthropic': ANTHROPIC_AVAILABLE,
        'gemini': GEMINI_AVAILABLE
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


def get_gemini_model():
    """获取 Gemini 模型"""
    if not GEMINI_AVAILABLE:
        return None
    
    # 优先从 Streamlit secrets 读取 (Streamlit Cloud)
    api_key = None
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            api_key = st.secrets['GEMINI_API_KEY']
    except:
        pass
    
    # 回退到环境变量 (本地开发)
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


class LLMAnalyzer:
    """LLM 分析器"""
    
    def __init__(self, provider: str = 'gemini'):
        """
        初始化分析器
        
        Args:
            provider: 'openai', 'anthropic', 或 'gemini'
        """
        self.provider = provider
        self.client = None
        
        if provider == 'openai':
            self.client = get_openai_client()
            self.model = 'gpt-4o-mini'
        elif provider == 'anthropic':
            self.client = get_anthropic_client()
            self.model = 'claude-3-haiku-20240307'
        elif provider == 'gemini':
            self.client = get_gemini_model()
            self.model = 'gemini-2.5-flash'
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self.client is not None
    
    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """统一的 LLM 调用接口"""
        if not self.is_available():
            return ""
        
        try:
            if self.provider == 'openai':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=system_prompt if system_prompt else "",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.provider == 'gemini':
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.client.generate_content(full_prompt)
                return response.text
        
        except Exception as e:
            return f"Error: {str(e)}"
        
        return ""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """分析文本情感"""
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
}}"""
        
        result = self._call_llm(prompt, "你是一位专业的金融分析师。只返回JSON。")
        try:
            # 尝试提取 JSON
            if '{' in result:
                json_str = result[result.find('{'):result.rfind('}')+1]
                return json.loads(json_str)
        except:
            pass
        return {'error': 'Parse failed', 'raw': result}
    
    def natural_query(self, query: str, context: str = "") -> str:
        """自然语言查询"""
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
        
        return self._call_llm(user_prompt, system_prompt)
    
    def generate_market_report(self, signals: List[Dict]) -> str:
        """生成市场报告"""
        if not self.is_available():
            return "LLM client not available"
        
        if not signals:
            signal_summary = "今日无触发信号"
        else:
            signal_summary = f"今日共有 {len(signals)} 个 BLUE 信号:\n"
            for s in signals[:10]:
                blue_val = float(s.get('blue_daily', 0) or 0)
                price_val = float(s.get('price', 0) or 0)
                signal_summary += f"- {s.get('symbol', 'N/A')}: BLUE={blue_val:.1f}, 价格=${price_val:.2f}\n"
        
        prompt = f"""基于以下信号数据，生成一份简洁的每日市场报告。

信号摘要:
{signal_summary}

请生成 Markdown 格式报告，包含:
1. 📊 市场概览 (2-3句话)
2. 🔥 热门信号 (如果有)
3. ⚠️ 风险提示
4. 💡 操作建议

保持简洁专业。"""
        
        return self._call_llm(prompt, "你是一位专业的量化分析师，负责撰写每日市场报告。")
    
    def generate_decision_dashboard(self, stock_data: Dict, news_context: str = "") -> Dict:
        """
        生成 AI 决策仪表盘 (类似 daily_stock_analysis)
        
        Args:
            stock_data: 股票数据
            news_context: 新闻上下文 (新增)
        
        Returns:
            决策仪表盘 Dict
        """
        if not self.is_available():
            return {'error': 'LLM client not available'}
        
        symbol = stock_data.get('symbol', 'N/A')
        # Convert to float safely to avoid format errors
        def safe_float(val, default=0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        price = safe_float(stock_data.get('price'), 0)
        blue = safe_float(stock_data.get('blue_daily'), 0)
        blue_w = safe_float(stock_data.get('blue_weekly'), 0)
        ma5 = safe_float(stock_data.get('ma5'), 0)
        ma10 = safe_float(stock_data.get('ma10'), 0)
        ma20 = safe_float(stock_data.get('ma20'), 0)
        rsi = safe_float(stock_data.get('rsi'), 50)
        vol_ratio = safe_float(stock_data.get('volume_ratio'), 1)
        
        # 计算乖离率
        bias = ((price - ma5) / ma5 * 100) if ma5 > 0 else 0
        bias_status = "安全" if abs(bias) < 2 else ("警戒" if abs(bias) < 5 else "危险")
        
        prompt = f"""你是一位专注于趋势交易的专业量化分析师，负责生成【决策仪表盘】。

## 核心交易理念（必须严格遵守）

### 1. 严禁追高
- 乖离率 = (现价 - MA5) / MA5 × 100%
- 乖离率 < 2%：最佳买点 ✅
- 乖离率 2-5%：可小仓介入 ⚠️
- 乖离率 > 5%：严禁追高！直接判定为"观望" ❌

### 2. 趋势交易
- 多头排列：MA5 > MA10 > MA20 ✅
- 空头排列坚决不碰 ❌

### 3. BLUE 信号系统
- BLUE > 100：超卖反弹信号 ✅
- BLUE 50-100：观望区域 ⚠️
- BLUE < 50：弱势 ❌

========== 股票数据 ==========
股票代码: {symbol}
当前价格: ${price:.2f}
BLUE信号(日): {blue:.1f}
BLUE信号(周): {blue_w:.1f}
MA5: ${ma5:.2f}
MA10: ${ma10:.2f}
MA20: ${ma20:.2f}
乖离率(MA5): {bias:.1f}% ({bias_status})
RSI: {rsi:.1f}
量比: {vol_ratio:.2f}

========== 近期情报 ==========
{news_context if news_context else "暂无新闻"}
==============================

请生成JSON格式的决策仪表盘:
{{
    "verdict": "一句话核心结论（30字以内，直接告诉用户该买该卖）",
    "signal": "BUY" | "HOLD" | "SELL",
    "confidence": 0-100,
    "entry_price": 建议买入价（在MA5附近）,
    "stop_loss": 止损价（跌破MA20或X%）,
    "target_price": 目标价,
    "news_summary": "舆情分析：是否有减持/业绩雷/利好 (1-2句话)",
    "checklist": [
        {{"item": "BLUE信号", "status": "✅" | "⚠️" | "❌", "detail": "BLUE=XX (超卖区/观望区/弱势)"}},
        {{"item": "均线排列", "status": "✅" | "⚠️" | "❌", "detail": "多头排列/空头排列/缠绕"}},
        {{"item": "乖离率", "status": "✅" | "⚠️" | "❌", "detail": "X.X% (安全/警戒/危险)"}},
        {{"item": "量价配合", "status": "✅" | "⚠️" | "❌", "detail": "量比=X.X (放量/缩量/正常)"}},
        {{"item": "趋势强度", "status": "✅" | "⚠️" | "❌", "detail": "RSI=XX (超买/中性/超卖)"}},
        {{"item": "舆情风控", "status": "✅" | "⚠️" | "❌", "detail": "利好/无风险/减持风险/业绩雷"}}
    ],
    "position_advice": {{
        "no_position": "空仓者建议：具体操作",
        "has_position": "持仓者建议：具体操作"
    }},
    "risk_warning": "风险提示"
}}

## 评分标准
- 80-100分（买入）：多头排列 + BLUE>100 + 乖离率<2% + 量能配合
- 60-79分（观望偏多）：允许一项不满足
- 40-59分（观望）：乖离率>5% 或 均线缠绕
- 0-39分（卖出）：空头排列 或 跌破MA20 或 重大利空"""
        
        
        result = self._call_llm(prompt)
        
        # 尝试解析 LLM 响应
        try:
            if result and '{' in result and 'Error' not in result:
                json_str = result[result.find('{'):result.rfind('}')+1]
                parsed = json.loads(json_str)
                if 'verdict' in parsed:
                    return parsed
        except Exception as e:
            pass
        
        # LLM 失败时，使用本地算法分析
        # 计算乖离率
        bias = (price - ma5) / ma5 * 100 if ma5 > 0 else 0
        
        # 判断信号
        if blue > 100 and ma5 > ma10 > ma20 and abs(bias) < 5:
            signal = 'BUY'
            verdict = '✅ 多头排列 + BLUE超卖信号，可低吸'
            confidence = min(80, int(blue / 2))
        elif blue > 80 and vol_ratio > 1.2:
            signal = 'BUY'
            verdict = '📈 有企稳迹象，关注突破'
            confidence = 60
        elif bias > 5:
            signal = 'SELL'
            verdict = '⚠️ 乖离率过高，严禁追高'
            confidence = 70
        elif ma5 < ma10 < ma20:
            signal = 'SELL'
            verdict = '📉 空头排列，建议观望'
            confidence = 65
        else:
            signal = 'HOLD'
            verdict = '🔄 趋势不明，建议观望等待'
            confidence = 40
        
        # 构建检查清单 (5项)
        checklist = [
            {
                'item': 'BLUE信号',
                'status': '✅' if blue > 100 else ('⚠️' if blue > 50 else '❌'),
                'detail': f'BLUE={blue:.0f}' + (' (超卖区)' if blue > 100 else (' (观望区)' if blue > 50 else ' (弱势)'))
            },
            {
                'item': '均线排列',
                'status': '✅' if ma5 > ma10 > ma20 else ('⚠️' if ma5 > ma10 else '❌'),
                'detail': '多头排列' if ma5 > ma10 > ma20 else ('弱势多头' if ma5 > ma10 else '空头/缠绕')
            },
            {
                'item': '乖离率',
                'status': '✅' if abs(bias) < 2 else ('⚠️' if abs(bias) < 5 else '❌'),
                'detail': f'{bias:+.1f}% ' + ('安全' if abs(bias) < 2 else ('警戒' if abs(bias) < 5 else '❌严禁追高'))
            },
            {
                'item': '量价配合',
                'status': '✅' if vol_ratio > 1.5 else ('⚠️' if vol_ratio > 0.8 else '❌'),
                'detail': f'量比={vol_ratio:.1f}x ' + ('放量' if vol_ratio > 1.5 else ('正常' if vol_ratio > 0.8 else '缩量'))
            },
            {
                'item': '趋势强度',
                'status': '✅' if 30 < rsi < 70 else ('⚠️' if rsi > 70 else '❌'),
                'detail': f'RSI={rsi:.0f} ' + ('中性' if 30 < rsi < 70 else ('超买' if rsi > 70 else '超卖'))
            }
        ]
        
        # 生成持仓建议
        if signal == 'BUY':
            position_advice = {
                'no_position': f'空仓者：可在${ma5:.2f}附近低吸建仓',
                'has_position': '持仓者：继续持有，适当加仓'
            }
        elif signal == 'SELL':
            position_advice = {
                'no_position': '空仓者：暂时观望，不要追入',
                'has_position': '持仓者：考虑减仓或设止损'
            }
        else:
            position_advice = {
                'no_position': '空仓者：等待更好的买点',
                'has_position': '持仓者：持股观望'
            }
        
        return {
            'verdict': verdict,
            'signal': signal,
            'confidence': confidence,
            'entry_price': round(ma5, 2),  # 建议在MA5附近买入
            'stop_loss': round(ma20 * 0.97, 2),  # 止损在MA20下方3%
            'target_price': round(price * 1.15, 2),  # 目标15%收益
            'checklist': checklist,
            'position_advice': position_advice,
            'risk_warning': '⚠️ 本地算法分析，建议结合AI和人工判断',
            'analysis_mode': 'local'  # 标记为本地分析
        }


def quick_sentiment_check(text: str, provider: str = 'gemini') -> Dict:
    """快速情感分析"""
    analyzer = LLMAnalyzer(provider)
    return analyzer.analyze_sentiment(text)


def ask_ai(question: str, provider: str = 'gemini') -> str:
    """快速 AI 问答"""
    analyzer = LLMAnalyzer(provider)
    return analyzer.natural_query(question)


def generate_stock_decision(stock_data: Dict, provider: str = 'gemini') -> Dict:
    """生成股票决策仪表盘"""
    analyzer = LLMAnalyzer(provider)
    return analyzer.generate_decision_dashboard(stock_data)


if __name__ == "__main__":
    print("LLM Module Status:")
    status = check_llm_available()
    print(f"  OpenAI: {'✅' if status['openai'] else '❌'}")
    print(f"  Anthropic: {'✅' if status['anthropic'] else '❌'}")
    print(f"  Gemini: {'✅' if status['gemini'] else '❌'}")
    
    # 测试 Gemini
    if status['gemini'] and os.environ.get('GEMINI_API_KEY'):
        print("\nTesting Gemini Decision Dashboard...")
        test_data = {
            'symbol': 'NVDA',
            'price': 135.50,
            'blue_daily': 120,
            'blue_weekly': 85,
            'ma5': 134,
            'ma10': 132,
            'ma20': 128,
            'rsi': 35,
            'volume_ratio': 1.8
        }
        result = generate_stock_decision(test_data)
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
