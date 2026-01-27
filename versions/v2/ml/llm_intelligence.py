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
    
    def generate_decision_dashboard(self, stock_data: Dict) -> Dict:
        """
        生成 AI 决策仪表盘 (类似 daily_stock_analysis)
        
        Args:
            stock_data: 股票数据，包含:
                - symbol: 股票代码
                - price: 当前价格
                - blue_daily: BLUE日线值
                - blue_weekly: BLUE周线值 (可选)
                - ma5, ma10, ma20: 均线
                - rsi: RSI值 (可选)
                - volume_ratio: 量比 (可选)
                - sector: 行业 (可选)
        
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
        
        prompt = f"""你是一位专业的量化交易分析师。基于以下股票数据，生成决策仪表盘。

股票: {symbol}
当前价格: ${price:.2f}
BLUE信号(日): {blue:.1f} (>100为超卖买入信号)
BLUE信号(周): {blue_w:.1f}
MA5: ${ma5:.2f}
MA10: ${ma10:.2f}  
MA20: ${ma20:.2f}
RSI: {rsi:.1f}
量比: {vol_ratio:.2f}

请生成JSON格式的决策仪表盘:
{{
    "verdict": "一句话核心结论 (如: 强烈买入/观望/回避)",
    "signal": "BUY" | "HOLD" | "SELL",
    "confidence": 0-100,
    "entry_price": 建议买入价,
    "stop_loss": 止损价,
    "target_price": 目标价,
    "checklist": [
        {{"item": "BLUE信号", "status": "✅" | "⚠️" | "❌", "detail": "说明"}},
        {{"item": "均线排列", "status": "✅" | "⚠️" | "❌", "detail": "说明"}},
        {{"item": "量价配合", "status": "✅" | "⚠️" | "❌", "detail": "说明"}},
        {{"item": "趋势判断", "status": "✅" | "⚠️" | "❌", "detail": "说明"}}
    ],
    "risk_warning": "风险提示"
}}

规则:
- BLUE > 100 为超卖买入区
- MA5 > MA10 > MA20 为多头排列 ✅
- 量比 > 1.5 为放量
- 严禁追高：乖离率 > 5% 标记危险"""
        
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
        
        # 构建检查清单
        checklist = [
            {
                'item': 'BLUE信号',
                'status': '✅' if blue > 100 else ('⚠️' if blue > 50 else '❌'),
                'detail': f'BLUE={blue:.0f}' + (' (超卖区)' if blue > 100 else '')
            },
            {
                'item': '均线排列',
                'status': '✅' if ma5 > ma10 > ma20 else '❌',
                'detail': '多头排列' if ma5 > ma10 > ma20 else '空头/缠绕'
            },
            {
                'item': '乖离率',
                'status': '✅' if abs(bias) < 2 else ('⚠️' if abs(bias) < 5 else '❌'),
                'detail': f'{bias:+.1f}%' + (' ⚠️追高风险' if bias > 5 else '')
            },
            {
                'item': '量价配合',
                'status': '✅' if vol_ratio > 1.2 else '⚠️',
                'detail': f'量比={vol_ratio:.1f}x'
            }
        ]
        
        return {
            'verdict': verdict,
            'signal': signal,
            'confidence': confidence,
            'entry_price': round(ma5, 2),  # 建议在MA5附近买入
            'stop_loss': round(ma20 * 0.97, 2),  # 止损在MA20下方3%
            'target_price': round(price * 1.15, 2),  # 目标15%收益
            'checklist': checklist,
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
