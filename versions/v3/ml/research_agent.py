#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Research Agent (Dexter-Style)
======================================

自主金融研究 Agent，灵感来自 Dexter。
用 Gemini 做 LLM 推理，结合 Coral Creek 的指标体系，
自动分步执行深度个股研究。

流程:
1. 任务规划 (Plan) — 把研究问题拆解为步骤
2. 数据获取 (Fetch) — 拉取技术面、基本面、财务、行业数据
3. 分析推理 (Analyze) — 用 LLM 对每个维度做分析
4. 交叉验证 (Validate) — 检查各维度结论是否一致
5. 综合报告 (Report) — 生成结构化研究报告
"""

import os
import json
import time
import traceback
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


@dataclass
class ResearchStep:
    """研究步骤"""
    name: str
    description: str
    status: str = "pending"  # pending, running, done, error
    result: str = ""
    data: Dict = field(default_factory=dict)
    duration: float = 0.0


@dataclass
class ResearchReport:
    """研究报告"""
    symbol: str
    company_name: str = ""
    market: str = "US"
    signal: str = "HOLD"  # BUY / SELL / HOLD
    confidence: int = 50
    verdict: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_reward: float = 0.0
    sections: Dict[str, str] = field(default_factory=dict)
    checklist: List[Dict] = field(default_factory=list)
    steps: List[ResearchStep] = field(default_factory=list)
    total_duration: float = 0.0
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'market': self.market,
            'signal': self.signal,
            'confidence': self.confidence,
            'verdict': self.verdict,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'risk_reward': self.risk_reward,
            'sections': self.sections,
            'checklist': self.checklist,
            'total_duration': self.total_duration,
            'created_at': self.created_at,
        }


class ResearchAgent:
    """
    深度研究 Agent — 类似 Dexter 的自主金融研究
    
    用法:
        agent = ResearchAgent(market='US')
        report = agent.research('AAPL', price=185.0, blue_daily=120, ...)
    """
    
    def __init__(self, market: str = 'US', provider: str = 'gemini'):
        self.market = market
        self.provider = provider
        self._llm = None
        self._progress_callback: Optional[Callable] = None
    
    def _get_llm(self):
        """延迟加载 LLM"""
        if self._llm is None:
            from ml.llm_intelligence import LLMAnalyzer
            self._llm = LLMAnalyzer(provider=self.provider)
        return self._llm
    
    def _call_llm(self, prompt: str, system: str = "") -> str:
        """调用 LLM"""
        llm = self._get_llm()
        if not llm.is_available():
            return "[LLM 不可用，请检查 GEMINI_API_KEY]"
        return llm._call_llm(prompt, system)
    
    def _notify(self, step_name: str, status: str, detail: str = ""):
        """通知进度"""
        if self._progress_callback:
            self._progress_callback(step_name, status, detail)
    
    # =========================================================
    # 数据获取工具 (类似 Dexter 的 tools)
    # =========================================================
    
    def _fetch_price_history(self, symbol: str, days: int = 365) -> Optional[Dict]:
        """获取价格历史"""
        try:
            from data_fetcher import get_stock_data
            df = get_stock_data(symbol, market=self.market, days=days)
            if df is None or df.empty:
                return None
            
            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values
            volumes = df['Volume'].values
            
            # 基础统计
            current = float(closes[-1])
            high_52w = float(max(highs[-252:])) if len(highs) >= 252 else float(max(highs))
            low_52w = float(min(lows[-252:])) if len(lows) >= 252 else float(min(lows))
            
            # 均线
            ma5 = float(closes[-5:].mean()) if len(closes) >= 5 else current
            ma10 = float(closes[-10:].mean()) if len(closes) >= 10 else current
            ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else current
            ma60 = float(closes[-60:].mean()) if len(closes) >= 60 else current
            ma120 = float(closes[-120:].mean()) if len(closes) >= 120 else current
            ma250 = float(closes[-250:].mean()) if len(closes) >= 250 else current
            
            # 涨跌幅
            ret_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            ret_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
            ret_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
            ret_60d = (closes[-1] / closes[-61] - 1) * 100 if len(closes) >= 61 else 0
            
            # 成交量
            vol_avg_20 = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes[-1])
            vol_ratio = float(volumes[-1]) / vol_avg_20 if vol_avg_20 > 0 else 1.0
            
            # RSI
            import numpy as np
            if len(closes) >= 15:
                deltas = np.diff(closes[-15:])
                gains = np.where(deltas > 0, deltas, 0).mean()
                losses = np.where(deltas < 0, -deltas, 0).mean()
                rsi = 100 - 100 / (1 + gains / losses) if losses > 0 else 100
            else:
                rsi = 50
            
            return {
                'current_price': current,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'pct_from_high': (current / high_52w - 1) * 100,
                'pct_from_low': (current / low_52w - 1) * 100,
                'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
                'ma60': ma60, 'ma120': ma120, 'ma250': ma250,
                'above_ma20': current > ma20,
                'above_ma60': current > ma60,
                'above_ma250': current > ma250,
                'ret_1d': round(ret_1d, 2),
                'ret_5d': round(ret_5d, 2),
                'ret_20d': round(ret_20d, 2),
                'ret_60d': round(ret_60d, 2),
                'volume_ratio': round(vol_ratio, 2),
                'rsi_14': round(float(rsi), 1),
                'data_points': len(closes),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _fetch_fundamentals(self, symbol: str) -> Optional[Dict]:
        """获取基本面数据 (yfinance)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            
            return {
                'company_name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', '未知'),
                'industry': info.get('industry', '未知'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_trailing': info.get('trailingPE'),
                'pe_forward': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'dividend_yield': info.get('dividendYield'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'free_cashflow': info.get('freeCashflow'),
                'revenue': info.get('totalRevenue'),
                'beta': info.get('beta'),
                'target_high': info.get('targetHighPrice'),
                'target_low': info.get('targetLowPrice'),
                'target_mean': info.get('targetMeanPrice'),
                'analyst_rating': info.get('recommendationKey'),
                'num_analysts': info.get('numberOfAnalystOpinions'),
                'business_summary': (info.get('longBusinessSummary', '') or '')[:800],
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _fetch_financials(self, symbol: str) -> Optional[Dict]:
        """获取财务报表数据"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            result = {}
            
            # 收入报表
            income = ticker.quarterly_income_stmt
            if income is not None and not income.empty:
                latest = income.iloc[:, 0]
                prev = income.iloc[:, 1] if income.shape[1] > 1 else None
                
                revenue = latest.get('Total Revenue', 0)
                net_income = latest.get('Net Income', 0)
                
                result['last_quarter_revenue'] = float(revenue) if revenue else 0
                result['last_quarter_net_income'] = float(net_income) if net_income else 0
                
                if prev is not None:
                    prev_rev = prev.get('Total Revenue', 0)
                    if prev_rev and float(prev_rev) > 0:
                        result['revenue_qoq'] = round((float(revenue) / float(prev_rev) - 1) * 100, 1)
            
            # 资产负债表
            balance = ticker.quarterly_balance_sheet
            if balance is not None and not balance.empty:
                latest_b = balance.iloc[:, 0]
                result['total_assets'] = float(latest_b.get('Total Assets', 0) or 0)
                result['total_debt'] = float(latest_b.get('Total Debt', 0) or 0)
                result['total_equity'] = float(latest_b.get('Stockholders Equity', 0) or 0)
                result['cash'] = float(latest_b.get('Cash And Cash Equivalents', 0) or 0)
            
            # 现金流
            cashflow = ticker.quarterly_cashflow
            if cashflow is not None and not cashflow.empty:
                latest_cf = cashflow.iloc[:, 0]
                result['operating_cashflow'] = float(latest_cf.get('Operating Cash Flow', 0) or 0)
                result['capex'] = float(latest_cf.get('Capital Expenditure', 0) or 0)
                result['fcf'] = float(latest_cf.get('Free Cash Flow', 0) or 0)
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    # =========================================================
    # 研究流程
    # =========================================================
    
    def research(self, symbol: str,
                 price: float = 0,
                 blue_daily: float = 0,
                 blue_weekly: float = 0,
                 blue_monthly: float = 0,
                 adx: float = 0,
                 is_heima: bool = False,
                 is_juedi: bool = False,
                 progress_callback: Callable = None,
                 ) -> ResearchReport:
        """
        执行完整深度研究
        
        Args:
            symbol: 股票代码
            price: 当前价格
            blue_daily/weekly/monthly: BLUE 信号值
            adx: ADX 趋势强度
            is_heima: 是否有黑马信号
            is_juedi: 是否有掘地信号
            progress_callback: 进度回调 (step_name, status, detail)
        
        Returns:
            ResearchReport
        """
        self._progress_callback = progress_callback
        report = ResearchReport(
            symbol=symbol,
            market=self.market,
            created_at=datetime.now().isoformat(),
        )
        
        start_time = time.time()
        
        # 预定义研究步骤
        steps = [
            ResearchStep("📊 技术面分析", "分析价格趋势、均线、动量指标"),
            ResearchStep("🏢 基本面研究", "获取公司信息、估值指标、分析师评级"),
            ResearchStep("💰 财务报表分析", "分析营收、利润、现金流、负债"),
            ResearchStep("🎯 Coral Creek 信号", "整合 BLUE/黑马/ADX 等独有指标"),
            ResearchStep("🔬 综合诊断", "交叉验证 + 生成最终结论"),
        ]
        report.steps = steps
        
        # Coral Creek 信号数据 (已有)
        cc_signals = {
            'blue_daily': blue_daily,
            'blue_weekly': blue_weekly,
            'blue_monthly': blue_monthly,
            'adx': adx,
            'is_heima': is_heima,
            'is_juedi': is_juedi,
        }
        
        # --- Step 1: 技术面分析 ---
        self._run_step(steps[0], self._step_technical, symbol, price, report)
        
        # --- Step 2: 基本面研究 ---
        self._run_step(steps[1], self._step_fundamentals, symbol, report)
        
        # --- Step 3: 财务报表 ---
        self._run_step(steps[2], self._step_financials, symbol, report)
        
        # --- Step 4: Coral Creek 信号整合 ---
        self._run_step(steps[3], self._step_coral_creek_signals, symbol, cc_signals, report)
        
        # --- Step 5: 综合诊断 ---
        self._run_step(steps[4], self._step_final_diagnosis, symbol, report)
        
        report.total_duration = round(time.time() - start_time, 1)
        return report
    
    def _run_step(self, step: ResearchStep, func, *args):
        """执行单个步骤"""
        step.status = "running"
        self._notify(step.name, "running")
        t0 = time.time()
        try:
            func(*args)
            step.status = "done"
            step.duration = round(time.time() - t0, 1)
            self._notify(step.name, "done", f"{step.duration}s")
        except Exception as e:
            step.status = "error"
            step.result = str(e)
            step.duration = round(time.time() - t0, 1)
            self._notify(step.name, "error", str(e))
    
    # =========================================================
    # 各步骤实现
    # =========================================================
    
    def _step_technical(self, symbol: str, price: float, report: ResearchReport):
        """Step 1: 技术面分析"""
        self._notify("📊 技术面分析", "running", "正在获取价格历史...")
        
        data = self._fetch_price_history(symbol, days=365)
        if not data or 'error' in data:
            report.sections['technical'] = f"⚠️ 技术数据获取失败: {data.get('error', '未知错误')}"
            return
        
        if price <= 0:
            price = data['current_price']
        report.entry_price = price
        
        self._notify("📊 技术面分析", "running", "AI 正在分析技术面...")
        
        prompt = f"""你是专业的技术分析师。分析以下 {symbol} 的技术数据，给出简洁结论。

技术数据:
- 当前价格: ${data['current_price']:.2f}
- 52周高点: ${data['high_52w']:.2f} (距高点 {data['pct_from_high']:.1f}%)
- 52周低点: ${data['low_52w']:.2f} (距低点 {data['pct_from_low']:.1f}%)
- MA5: ${data['ma5']:.2f} | MA20: ${data['ma20']:.2f} | MA60: ${data['ma60']:.2f} | MA250: ${data['ma250']:.2f}
- 在MA20之上: {'是' if data['above_ma20'] else '否'} | 在MA60之上: {'是' if data['above_ma60'] else '否'} | 在MA250之上: {'是' if data['above_ma250'] else '否'}
- 涨跌幅: 1日 {data['ret_1d']:+.1f}% | 5日 {data['ret_5d']:+.1f}% | 20日 {data['ret_20d']:+.1f}% | 60日 {data['ret_60d']:+.1f}%
- RSI(14): {data['rsi_14']:.1f}
- 量比: {data['volume_ratio']:.2f}

请用中文回答，包含:
1. 趋势判断 (上升/下降/震荡)
2. 支撑位和压力位
3. 动量状态 (超买/超卖/中性)
4. 技术面结论 (看多/看空/中性)

限200字以内。"""
        
        result = self._call_llm(prompt, "你是专业股票技术分析师，回答简洁精准。")
        report.sections['technical'] = result
        report.steps[0].data = data
    
    def _step_fundamentals(self, symbol: str, report: ResearchReport):
        """Step 2: 基本面研究"""
        self._notify("🏢 基本面研究", "running", "正在获取公司信息...")
        
        data = self._fetch_fundamentals(symbol)
        if not data or 'error' in data:
            report.sections['fundamentals'] = f"⚠️ 基本面数据获取失败: {data.get('error', '未知错误')}"
            return
        
        report.company_name = data.get('company_name', symbol)
        
        self._notify("🏢 基本面研究", "running", "AI 正在分析基本面...")
        
        def fmt_cap(v):
            if not v: return "N/A"
            if v >= 1e12: return f"${v/1e12:.2f}T"
            if v >= 1e9: return f"${v/1e9:.2f}B"
            return f"${v/1e6:.0f}M"
        
        def fmt_pct(v):
            return f"{v*100:.1f}%" if v else "N/A"
        
        prompt = f"""分析 {symbol} ({data['company_name']}) 的基本面:

公司信息:
- 行业: {data['sector']} / {data['industry']}
- 市值: {fmt_cap(data['market_cap'])}
- Beta: {data.get('beta', 'N/A')}

估值指标:
- PE(TTM): {data.get('pe_trailing', 'N/A')} | PE(FWD): {data.get('pe_forward', 'N/A')}
- PEG: {data.get('peg_ratio', 'N/A')} | PB: {data.get('pb_ratio', 'N/A')}
- PS: {data.get('ps_ratio', 'N/A')} | EV/EBITDA: {data.get('ev_ebitda', 'N/A')}

盈利能力:
- 利润率: {fmt_pct(data.get('profit_margin'))}
- 营业利润率: {fmt_pct(data.get('operating_margin'))}
- ROE: {fmt_pct(data.get('roe'))} | ROA: {fmt_pct(data.get('roa'))}

成长性:
- 营收增长: {fmt_pct(data.get('revenue_growth'))}
- 盈利增长: {fmt_pct(data.get('earnings_growth'))}

分析师:
- 评级: {data.get('analyst_rating', 'N/A')} | 分析师数: {data.get('num_analysts', 'N/A')}
- 目标价: ${data.get('target_low', 0):.2f} ~ ${data.get('target_high', 0):.2f} (均值 ${data.get('target_mean', 0):.2f})

公司简介: {data.get('business_summary', '')[:300]}

请用中文回答，包含:
1. 估值水平 (便宜/合理/偏贵)
2. 盈利质量
3. 成长前景
4. 基本面结论

限200字以内。"""
        
        result = self._call_llm(prompt, "你是资深基本面分析师。")
        report.sections['fundamentals'] = result
        report.steps[1].data = data
    
    def _step_financials(self, symbol: str, report: ResearchReport):
        """Step 3: 财务报表分析"""
        self._notify("💰 财务报表", "running", "正在获取财务数据...")
        
        data = self._fetch_financials(symbol)
        if not data or 'error' in data:
            report.sections['financials'] = f"⚠️ 财务数据获取失败 (A股可能不支持): {data.get('error', '')}"
            return
        
        self._notify("💰 财务报表", "running", "AI 正在分析财报...")
        
        def fmt_b(v):
            if not v: return "N/A"
            if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
            if abs(v) >= 1e6: return f"${v/1e6:.0f}M"
            return f"${v:,.0f}"
        
        prompt = f"""分析 {symbol} 的最新季度财务报表:

收入:
- 营收: {fmt_b(data.get('last_quarter_revenue'))}
- 净利润: {fmt_b(data.get('last_quarter_net_income'))}
- 环比营收变化: {data.get('revenue_qoq', 'N/A')}%

资产负债:
- 总资产: {fmt_b(data.get('total_assets'))}
- 总负债: {fmt_b(data.get('total_debt'))}
- 股东权益: {fmt_b(data.get('total_equity'))}
- 现金: {fmt_b(data.get('cash'))}

现金流:
- 经营现金流: {fmt_b(data.get('operating_cashflow'))}
- 资本支出: {fmt_b(data.get('capex'))}
- 自由现金流: {fmt_b(data.get('fcf'))}

请用中文分析:
1. 盈利状况 (是否赚钱)
2. 资产质量 (负债率、现金充裕度)
3. 现金流质量
4. 财务风险警示

限150字以内。"""
        
        result = self._call_llm(prompt, "你是财务分析师，关注数据异常和风险。")
        report.sections['financials'] = result
        report.steps[2].data = data
    
    def _step_coral_creek_signals(self, symbol: str, cc_signals: Dict, report: ResearchReport):
        """Step 4: Coral Creek 独有信号分析"""
        self._notify("🎯 Coral Creek 信号", "running", "整合信号数据...")
        
        blue_d = cc_signals.get('blue_daily', 0)
        blue_w = cc_signals.get('blue_weekly', 0)
        blue_m = cc_signals.get('blue_monthly', 0)
        adx = cc_signals.get('adx', 0)
        is_heima = cc_signals.get('is_heima', False)
        is_juedi = cc_signals.get('is_juedi', False)
        
        # 信号解读
        signals = []
        if blue_d > 100: signals.append(f"🔵 日线 BLUE={blue_d:.0f} (强抄底)")
        elif blue_d > 50: signals.append(f"🔵 日线 BLUE={blue_d:.0f} (弱信号)")
        else: signals.append(f"⬜ 日线 BLUE={blue_d:.0f} (无信号)")
        
        if blue_w > 100: signals.append(f"🔵 周线 BLUE={blue_w:.0f} (中期底)")
        if blue_m > 100: signals.append(f"🔵 月线 BLUE={blue_m:.0f} (大级别底)")
        
        if adx > 40: signals.append(f"📈 ADX={adx:.0f} (极强趋势)")
        elif adx > 25: signals.append(f"📈 ADX={adx:.0f} (中等趋势)")
        else: signals.append(f"📊 ADX={adx:.0f} (弱趋势/震荡)")
        
        if is_heima: signals.append("🐴 有黑马信号 (爆发潜力)")
        if is_juedi: signals.append("⛏️ 有掘地信号 (底部挖掘)")
        
        # 信号评分
        score = 0
        if blue_d > 100: score += 30
        if blue_w > 100: score += 20
        if blue_m > 100: score += 15
        if adx > 25: score += 10
        if is_heima: score += 15
        if is_juedi: score += 10
        
        analysis = f"""**Coral Creek 信号面板:**

{'  ⁃  '.join(signals)}

🏆 信号综合评分: **{score}/100**

{'✅ 多周期共振 — BLUE 在多个周期同时触发，是高置信度底部信号' if (blue_d > 100 and blue_w > 100) else ''}
{'⚠️ 仅日线触发 — 建议等待周线确认' if (blue_d > 100 and blue_w <= 100) else ''}
{'🚫 当前无 BLUE 信号' if blue_d <= 0 else ''}
"""
        report.sections['coral_creek'] = analysis
        report.steps[3].data = {'signals': cc_signals, 'score': score}
    
    def _step_final_diagnosis(self, symbol: str, report: ResearchReport):
        """Step 5: 综合诊断 — 交叉验证 + 最终结论"""
        self._notify("🔬 综合诊断", "running", "AI 正在交叉验证各维度...")
        
        # 汇总所有之前的分析
        all_analyses = ""
        for key, section in report.sections.items():
            all_analyses += f"\n--- {key} ---\n{section}\n"
        
        prompt = f"""你是首席投资策略师。基于以下多维度分析，给出 {symbol} 的最终投资诊断。

{all_analyses}

请严格按以下 JSON 格式回答 (不要加其他文字):
{{
    "signal": "BUY或SELL或HOLD",
    "confidence": 0到100的整数,
    "verdict": "一句话结论（30字以内）",
    "entry_price": 建议买入价（数字）,
    "stop_loss": 建议止损价（数字）,
    "target_price": 建议目标价（数字）,
    "bull_case": "看多理由（50字以内）",
    "bear_case": "看空理由（50字以内）",
    "checklist": [
        {{"item": "检查项", "status": "pass或fail或warn", "note": "说明"}},
        {{"item": "检查项2", "status": "pass或fail或warn", "note": "说明"}}
    ]
}}

注意:
- 只输出 JSON，不要有任何其他文字
- 如果数据不足，confidence 不超过 50
- entry_price/stop_loss/target_price 用数字，不要带 $ 符号
- checklist 至少 5 项，涵盖趋势、估值、财务、信号、风险
"""
        
        response = self._call_llm(prompt, "你是严谨的首席投资策略师。只输出 JSON。")
        
        # 解析 JSON
        try:
            # 去掉可能的 markdown 代码块包裹
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            
            result = json.loads(cleaned)
            
            report.signal = result.get('signal', 'HOLD')
            report.confidence = int(result.get('confidence', 50))
            report.verdict = result.get('verdict', '')
            
            # 价格
            curr = report.entry_price or 1
            report.entry_price = float(result.get('entry_price', curr))
            report.stop_loss = float(result.get('stop_loss', curr * 0.92))
            report.target_price = float(result.get('target_price', curr * 1.15))
            
            risk = report.entry_price - report.stop_loss
            reward = report.target_price - report.entry_price
            report.risk_reward = round(reward / risk, 2) if risk > 0 else 0
            
            # 看多/看空理由
            report.sections['bull_case'] = result.get('bull_case', '')
            report.sections['bear_case'] = result.get('bear_case', '')
            
            # 检查清单
            report.checklist = result.get('checklist', [])
            
        except (json.JSONDecodeError, Exception) as e:
            # 如果 JSON 解析失败，把原始回复保存
            report.sections['diagnosis'] = response
            report.signal = 'HOLD'
            report.confidence = 30
            report.verdict = f'AI分析完成（JSON解析失败: {str(e)[:50]}）'
