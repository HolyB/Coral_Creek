#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一股票详情组件 - 整合所有页面的个股分析功能

功能包括:
1. K线图表 (日/周/月线切换, 日期滑动条, 筹码分布)
2. 筹码分析 (获利盘/套牢盘/主力动向)
3. 技术指标 (BLUE/ADX/黑马/交易计划)
4. AI诊断 (决策仪表盘+大师分析)
5. 问AI (yfinance数据+自由对话)
6. 新闻舆情
7. 操作区 (加入观察/模拟买入)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

# 导入必要的模块
import sys
import os

# 确保 versions/v3 在 sys.path 中 (Streamlit Cloud 兼容)
# 使用 realpath 确保路径被正确解析
_component_dir = os.path.realpath(os.path.dirname(__file__))  # .../components
_v3_dir = os.path.realpath(os.path.join(_component_dir, '..'))  # .../versions/v3

# 强制添加到 sys.path 最前面
sys.path.insert(0, _v3_dir)

# 同时尝试添加可能的 Streamlit Cloud 路径
_possible_paths = [
    '/mount/src/coral_creek/versions/v3',
    os.path.join(os.getcwd(), 'versions', 'v3'),
    os.getcwd(),  # 如果 cwd 已经是 v3
]
for p in _possible_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def render_unified_stock_detail(
    symbol: str,
    market: str = 'US',
    # 可选的预加载数据 (避免重复获取)
    hist_data: pd.DataFrame = None,
    stock_info: Dict = None,
    scan_row: Dict = None,
    # 显示配置
    show_charts: bool = True,
    show_chips: bool = True,
    show_indicators: bool = True,
    show_ai: bool = True,
    show_ask_ai: bool = True,
    show_news: bool = True,
    show_actions: bool = True,
    # 唯一key前缀 (避免组件冲突)
    key_prefix: str = ""
):
    """
    渲染统一的股票详情面板
    
    Args:
        symbol: 股票代码
        market: 市场 ('US' / 'CN')
        hist_data: 预加载的历史数据 (可选)
        stock_info: 预加载的股票信息 (可选)
        scan_row: 扫描结果中的行数据 (可选)
        show_*: 各模块显示开关
        key_prefix: Streamlit组件key前缀
    """
    from data_fetcher import get_stock_data
    from indicator_utils import calculate_blue_signal_series, calculate_adx_series, calculate_heima_signal_series
    
    price_symbol = "¥" if market == "CN" else "$"
    unique_key = f"{key_prefix}_{symbol}" if key_prefix else symbol
    
    # === 1. 获取数据 ===
    with st.spinner(f"正在加载 {symbol} 数据..."):
        # 历史数据 (带缓存)
        if hist_data is None:
            hist_data = _cached_get_stock_data(symbol, market=market, days=3650)  # 10年
        
        if hist_data is None or hist_data.empty:
            st.error(f"❌ 无法获取 {symbol} 的数据")
            return
        
        # 获取yfinance信息 (公司基本面, 带缓存)
        yf_info = _cached_yfinance_info(symbol) if show_ask_ai or show_indicators else {}
        
        # 计算各周期数据
        df_daily = hist_data.copy()
        df_weekly = hist_data.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        df_monthly = hist_data.resample('ME').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        # 计算BLUE信号
        blue_daily = _calc_blue(df_daily)
        blue_weekly = _calc_blue(df_weekly) if len(df_weekly) >= 10 else 0
        blue_monthly = _calc_blue(df_monthly) if len(df_monthly) >= 6 else 0
        
        # 计算ADX
        adx_val = _calc_adx(df_daily)
        
        # 计算黑马/掘地
        heima_daily, juedi_daily = _calc_heima(df_daily)
        heima_weekly, juedi_weekly = _calc_heima(df_weekly) if len(df_weekly) >= 10 else (False, False)
        heima_monthly, juedi_monthly = _calc_heima(df_monthly) if len(df_monthly) >= 6 else (False, False)
        
        # 计算幻影主力
        phantom = _calc_phantom(df_daily)
        
        # 当前价格
        current_price = float(df_daily['Close'].iloc[-1])
        
        # 公司名称 - A股优先从数据库获取
        company_name = symbol
        if market == 'CN':
            # 尝试从数据库获取A股名称
            try:
                from db.database import get_connection
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT company_name FROM scan_results 
                    WHERE symbol = ? AND company_name IS NOT NULL AND company_name != ''
                    ORDER BY scan_date DESC LIMIT 1
                ''', (symbol,))
                row = cursor.fetchone()
                conn.close()
                if row and row[0]:
                    company_name = row[0]
            except:
                pass
        
        # 如果还没有名称，尝试yfinance
        if company_name == symbol:
            company_name = yf_info.get('shortName', yf_info.get('longName', symbol))
    
    # === 2. 综合判断面板 ===
    verdict = _compute_verdict(
        blue_daily, blue_weekly, blue_monthly, adx_val,
        heima_daily, heima_weekly, heima_monthly,
        juedi_daily, juedi_weekly, juedi_monthly,
        df_daily, current_price, phantom=phantom
    )
    
    # 判断面板 + 指标
    verdict_col, metrics_col = st.columns([1, 2])
    
    with verdict_col:
        v_color = verdict['color']
        v_bg = verdict['bg']
        v_action = verdict['action']
        v_score = verdict['score']
        v_label = verdict['label']
        
        st.markdown(f"""
        <div style="background: {v_bg}; border-left: 5px solid {v_color}; 
                    border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 4px;">{company_name}</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {v_color}; line-height: 1.1;">
                {v_action}
            </div>
            <div style="font-size: 1.1rem; color: {v_color}; margin: 4px 0;">
                {v_label} ({v_score}/100)
            </div>
            <div style="font-size: 1.4rem; font-weight: 600; color: #c9d1d9; margin-top: 8px;">
                {price_symbol}{current_price:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 关键理由
        for reason in verdict['reasons'][:3]:
            st.caption(reason)
    
    with metrics_col:
        # 指标卡片 (2 行 x 4 列)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            st.metric("日BLUE", f"{blue_daily:.0f}", delta="信号" if blue_daily > 100 else None)
        with r1c2:
            st.metric("周BLUE", f"{blue_weekly:.0f}", delta="信号" if blue_weekly > 100 else None)
        with r1c3:
            st.metric("月BLUE", f"{blue_monthly:.0f}", delta="信号" if blue_monthly > 100 else None)
        with r1c4:
            st.metric("ADX", f"{adx_val:.1f}", delta="强趋势" if adx_val > 25 else None)
        
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            heima_list = []
            if heima_daily: heima_list.append("日🐴")
            if heima_weekly: heima_list.append("周🐴")
            if heima_monthly: heima_list.append("月🐴")
            st.metric("黑马", " ".join(heima_list) if heima_list else "无")
        with r2c2:
            juedi_list = []
            if juedi_daily: juedi_list.append("日⛏️")
            if juedi_weekly: juedi_list.append("周⛏️")
            if juedi_monthly: juedi_list.append("月⛏️")
            st.metric("掘地", " ".join(juedi_list) if juedi_list else "无")
        with r2c3:
            # 近5日涨跌
            if len(df_daily) > 5:
                chg5 = (current_price / float(df_daily['Close'].iloc[-6]) - 1) * 100
                st.metric("5日涨跌", f"{chg5:+.1f}%")
            else:
                st.metric("5日涨跌", "N/A")
        with r2c4:
            # 量比
            if len(df_daily) > 20:
                vol_today = float(df_daily['Volume'].iloc[-1])
                vol_avg = float(df_daily['Volume'].iloc[-20:].mean())
                vol_ratio = vol_today / vol_avg if vol_avg > 0 else 0
                st.metric("量比", f"{vol_ratio:.1f}x")
            else:
                st.metric("量比", "N/A")
    
    st.divider()
    
    # === 3. 主要内容标签页 ===
    tabs = []
    tab_names = []
    
    # ML预测放在最前面 (重要)
    tab_names.append("🎯 ML预测")
    
    if show_charts:
        tab_names.append("📈 K线图表")
    if phantom:
        tab_names.append("👻 幻影主力")
    if show_chips:
        tab_names.append("📊 筹码分析")
    if show_indicators:
        tab_names.append("🔍 技术指标")
    if show_ai:
        tab_names.append("🤖 AI诊断")
    if show_ask_ai:
        tab_names.append("🗣️ 问AI")
    if show_news:
        tab_names.append("📰 新闻舆情")
    
    if tab_names:
        tabs = st.tabs(tab_names)
        tab_idx = 0
        
        # === Tab: ML预测 (新增) ===
        with tabs[tab_idx]:
            _render_ml_prediction_tab(
                symbol=symbol,
                market=market,
                hist_data=df_daily,
                blue_daily=blue_daily,
                blue_weekly=blue_weekly,
                blue_monthly=blue_monthly,
                is_heima=heima_daily,
                current_price=current_price,
                price_symbol=price_symbol,
                unique_key=unique_key
            )
        tab_idx += 1
        
        # === Tab: K线图表 ===
        if show_charts:
            with tabs[tab_idx]:
                _render_chart_tab(
                    symbol, df_daily, df_weekly, df_monthly,
                    price_symbol, unique_key, market
                )
            tab_idx += 1
        
        # === Tab: 幻影主力 ===
        if phantom:
            with tabs[tab_idx]:
                _render_phantom_tab(
                    symbol, df_daily, phantom, adx_val,
                    price_symbol, unique_key
                )
            tab_idx += 1
        
        # === Tab: 筹码分析 ===
        if show_chips:
            with tabs[tab_idx]:
                _render_chips_tab(symbol, df_daily, unique_key)
            tab_idx += 1
        
        # === Tab: 技术指标 ===
        if show_indicators:
            with tabs[tab_idx]:
                _render_indicators_tab(
                    symbol, current_price, price_symbol,
                    blue_daily, blue_weekly, blue_monthly, adx_val,
                    heima_daily, heima_weekly, heima_monthly,
                    juedi_daily, juedi_weekly, juedi_monthly,
                    yf_info, unique_key
                )
            tab_idx += 1
        
        # === Tab: AI诊断 ===
        if show_ai:
            with tabs[tab_idx]:
                _render_ai_diagnosis_tab(
                    symbol, current_price, price_symbol,
                    blue_daily, blue_weekly, blue_monthly, adx_val,
                    market, unique_key
                )
            tab_idx += 1
        
        # === Tab: 问AI ===
        if show_ask_ai:
            with tabs[tab_idx]:
                _render_ask_ai_tab(
                    symbol, company_name, current_price, price_symbol,
                    blue_daily, blue_weekly,
                    yf_info, market, unique_key
                )
            tab_idx += 1
        
        # === Tab: 新闻舆情 ===
        if show_news:
            with tabs[tab_idx]:
                _render_news_tab(symbol, company_name, market, unique_key)
            tab_idx += 1
    
    # === 4. 操作区 ===
    if show_actions:
        st.divider()
        _render_actions(
            symbol, current_price, price_symbol,
            blue_daily, blue_weekly,
            market, unique_key
        )


# ==================== 辅助函数 ====================

@st.cache_data(ttl=600, show_spinner=False)
def _cached_get_stock_data(symbol: str, market: str = 'US', days: int = 3650):
    """缓存股票历史数据 (10分钟TTL)"""
    from data_fetcher import get_stock_data
    return get_stock_data(symbol, market=market, days=days)

def _compute_verdict(
    blue_daily, blue_weekly, blue_monthly, adx,
    heima_daily, heima_weekly, heima_monthly,
    juedi_daily, juedi_weekly, juedi_monthly,
    df_daily, current_price, phantom: Dict = None
) -> Dict:
    """
    综合所有信号计算买卖判断
    
    评分体系 (0-100):
    - BLUE信号 (0-35): 日线+周线+月线共振
    - 趋势确认 (0-18): ADX趋势强度
    - 特殊信号 (0-17): 黑马+掘地加分
    - 价量形态 (0-18): 量价配合、均线支撑
    - 幻影主力 (±12): PINK/资金流向/海底捞月 (可加可减)
    
    Returns:
        {'action': '买入', 'score': 75, 'label': '看多', 'color': '#00C853',
         'bg': 'rgba(0,200,83,0.1)', 'reasons': ['日BLUE>100...', ...]}
    """
    score = 0
    reasons = []
    
    # === 1. BLUE 信号评分 (0-35) ===
    blue_score = 0
    
    # 日线 (0-15)
    if blue_daily >= 150:
        blue_score += 15
        reasons.append(f"✅ 日BLUE {blue_daily:.0f} 极强信号")
    elif blue_daily >= 100:
        blue_score += 12
        reasons.append(f"✅ 日BLUE {blue_daily:.0f} 强信号")
    elif blue_daily >= 50:
        blue_score += 7
        reasons.append(f"🟡 日BLUE {blue_daily:.0f} 中等信号")
    elif blue_daily > 0:
        blue_score += 3
    
    # 周线 (0-15) - 共振加分更多
    if blue_weekly >= 100:
        blue_score += 15
        reasons.append(f"✅ 周BLUE {blue_weekly:.0f} 周线共振确认")
    elif blue_weekly >= 50:
        blue_score += 10
    elif blue_weekly > 0:
        blue_score += 4
    
    # 月线 (0-10) - 大级别
    if blue_monthly >= 100:
        blue_score += 10
        reasons.append(f"✅ 月BLUE {blue_monthly:.0f} 月线大级别底部")
    elif blue_monthly >= 50:
        blue_score += 6
    elif blue_monthly > 0:
        blue_score += 2
    
    score += min(blue_score, 35)
    
    # === 2. 趋势确认 (0-18) ===
    if adx >= 40:
        score += 18
        reasons.append(f"✅ ADX {adx:.0f} 强趋势")
    elif adx >= 25:
        score += 13
        reasons.append(f"✅ ADX {adx:.0f} 趋势确认")
    elif adx >= 15:
        score += 7
    else:
        score += 3
        reasons.append(f"⚠️ ADX {adx:.0f} 趋势不明")
    
    # === 3. 特殊信号 (0-17) ===
    special_score = 0
    special_signals = []
    
    if heima_daily:
        special_score += 5
        special_signals.append("日黑马🐴")
    if heima_weekly:
        special_score += 5
        special_signals.append("周黑马🐴")
    if heima_monthly:
        special_score += 4
        special_signals.append("月黑马🐴")
    if juedi_daily:
        special_score += 3
        special_signals.append("日掘地⛏️")
    if juedi_weekly:
        special_score += 3
        special_signals.append("周掘地⛏️")
    if juedi_monthly:
        special_score += 2
        special_signals.append("月掘地⛏️")
    
    if special_signals:
        reasons.append(f"✅ 特殊信号: {' '.join(special_signals)}")
    
    score += min(special_score, 17)
    
    # === 4. 价量形态 (0-18) ===
    volume_score = 0
    try:
        if len(df_daily) >= 20:
            # 均线支撑
            sma5 = float(df_daily['Close'].rolling(5).mean().iloc[-1])
            sma20 = float(df_daily['Close'].rolling(20).mean().iloc[-1])
            
            if current_price > sma5 > sma20:
                volume_score += 8
                reasons.append("✅ 价格在5日/20日均线上方，多头排列")
            elif current_price > sma20:
                volume_score += 4
            elif current_price < sma20:
                volume_score += 0
                reasons.append("⚠️ 价格低于20日均线")
            
            # 量价配合
            vol_today = float(df_daily['Volume'].iloc[-1])
            vol_avg20 = float(df_daily['Volume'].iloc[-20:].mean())
            vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1
            
            if vol_ratio > 1.5:
                volume_score += 6
                reasons.append(f"✅ 放量 {vol_ratio:.1f}x (量价配合)")
            elif vol_ratio > 0.8:
                volume_score += 4
            else:
                volume_score += 1
                reasons.append(f"⚠️ 缩量 {vol_ratio:.1f}x")
            
            # 近期走势
            chg5 = (current_price / float(df_daily['Close'].iloc[-6]) - 1) * 100 if len(df_daily) > 5 else 0
            if chg5 > 5:
                volume_score += 6
            elif chg5 > 0:
                volume_score += 4
            elif chg5 > -5:
                volume_score += 2
            else:
                reasons.append(f"⚠️ 近5日跌 {chg5:.1f}%")
    except:
        volume_score = 5  # 数据异常给个基础分
    
    score += min(volume_score, 18)
    
    # === 5. 幻影主力 (±12) ===
    phantom_score = 0
    if phantom and isinstance(phantom, dict) and 'pink' in phantom:
        pink = phantom['pink']
        red = phantom['red']
        green = phantom['green']
        blue_bar = phantom['blue']
        lired = phantom['lired']
        buy_sig = phantom['buy_signal']
        sell_sig = phantom['sell_signal']
        blue_dis = phantom['blue_disappear']
        
        pink_val = float(pink[-1]) if len(pink) > 0 else 50
        red_val = float(red[-1]) if len(red) > 0 else 0
        green_val = float(green[-1]) if len(green) > 0 else 0
        has_blue_bar = float(blue_bar[-1]) > 0 if len(blue_bar) > 0 else False
        has_lired = float(lired[-1]) > 0 if len(lired) > 0 else False
        is_buy = bool(buy_sig[-1]) if len(buy_sig) > 0 else False
        is_sell = bool(sell_sig[-1]) if len(sell_sig) > 0 else False
        is_blue_dis = bool(blue_dis[-1]) if len(blue_dis) > 0 else False
        
        # 资金流向
        if red_val > 0:
            phantom_score += 3
            reasons.append(f"✅ 幻影: 主力资金流入")
        elif green_val < 0:
            phantom_score -= 3
            reasons.append(f"⚠️ 幻影: 资金流出")
        
        # BLUE消失 + 趋势中 = 强买入信号 (回测61%胜率)
        if is_blue_dis and adx >= 25:
            phantom_score += 5
            reasons.append(f"✅ 幻影: 海底捞月消失 + 趋势回调买入 (61%)")
        elif is_blue_dis:
            phantom_score += 2
        
        # LIRED (顶部压力)
        if has_lired:
            phantom_score -= 2
            reasons.append(f"⚠️ 幻影: 顶部压力出现")
        
        # PINK进场/逃顶
        if is_buy and pink_val < 15:
            phantom_score += 3
            reasons.append(f"✅ 幻影: PINK超卖进场信号")
        elif is_sell and green_val < 0 and adx < 30:
            phantom_score -= 4
            reasons.append(f"🚨 幻影: 逃顶信号 (PINK跌破90+资金流出)")
        elif is_sell:
            phantom_score -= 1  # 弱信号
        
        # PINK极值区域 (仅作辅助)
        if pink_val > 95:
            phantom_score -= 1
        elif pink_val < 5:
            phantom_score += 1
    
    score += max(min(phantom_score, 12), -12)
    
    # === 生成判断 ===
    score = min(score, 100)
    
    if score >= 80:
        action, label = "强烈买入", "极度看多"
        color, bg = "#00E676", "rgba(0,230,118,0.12)"
    elif score >= 65:
        action, label = "买入", "看多"
        color, bg = "#00C853", "rgba(0,200,83,0.10)"
    elif score >= 50:
        action, label = "偏多观望", "中性偏多"
        color, bg = "#FFD600", "rgba(255,214,0,0.10)"
    elif score >= 35:
        action, label = "观望", "中性"
        color, bg = "#8b949e", "rgba(139,148,158,0.10)"
    elif score >= 20:
        action, label = "偏空", "谨慎"
        color, bg = "#FF6D00", "rgba(255,109,0,0.10)"
    else:
        action, label = "回避", "看空"
        color, bg = "#FF1744", "rgba(255,23,68,0.10)"
    
    # 只保留最相关的理由
    reasons = [r for r in reasons if r.startswith("✅") or r.startswith("⚠️")][:5]
    if not reasons:
        reasons = ["ℹ️ 暂无明确信号"]
    
    return {
        'action': action,
        'label': label,
        'score': score,
        'color': color,
        'bg': bg,
        'reasons': reasons
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_yfinance_info(symbol: str) -> Dict:
    """缓存yfinance股票信息 (1小时TTL)"""
    return _get_yfinance_info(symbol)

def _get_yfinance_info(symbol: str) -> Dict:
    """获取yfinance股票信息"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return ticker.info or {}
    except:
        return {}


def _calc_blue(df: pd.DataFrame) -> float:
    """计算BLUE信号"""
    try:
        from indicator_utils import calculate_blue_signal_series
        blue = calculate_blue_signal_series(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values
        )
        return float(blue[-1]) if len(blue) > 0 else 0
    except:
        return 0


def _calc_adx(df: pd.DataFrame) -> float:
    """计算ADX"""
    try:
        from indicator_utils import calculate_adx_series
        adx = calculate_adx_series(
            df['High'].values, df['Low'].values, df['Close'].values
        )
        return float(adx[-1]) if len(adx) > 0 else 0
    except:
        return 0


def _calc_heima(df: pd.DataFrame) -> tuple:
    """计算黑马/掘地信号"""
    try:
        from indicator_utils import calculate_heima_signal_series
        heima, juedi = calculate_heima_signal_series(
            df['High'].values, df['Low'].values,
            df['Close'].values, df['Open'].values
        )
        return (bool(heima[-1]) if len(heima) > 0 else False,
                bool(juedi[-1]) if len(juedi) > 0 else False)
    except:
        return (False, False)


def _calc_phantom(df: pd.DataFrame) -> Dict:
    """计算幻影主力指标"""
    try:
        from indicator_utils import calculate_phantom_indicator
        if len(df) < 50:
            return {}
        return calculate_phantom_indicator(
            df['Open'].values, df['High'].values,
            df['Low'].values, df['Close'].values,
            df['Volume'].values
        )
    except Exception:
        return {}


# ==================== 各Tab渲染函数 ====================

def _render_chart_tab(symbol, df_daily, df_weekly, df_monthly, price_symbol, unique_key, market):
    """渲染K线图表标签页"""
    import plotly.graph_objects as go
    
    # 周期选择
    period_options = {"📅 日线": "daily", "📆 周线": "weekly", "🗓️ 月线": "monthly"}
    selected_period_label = st.radio(
        "选择周期",
        options=list(period_options.keys()),
        horizontal=True,
        index=0,
        key=f"period_{unique_key}"
    )
    selected_period = period_options[selected_period_label]
    
    # 选择数据
    if selected_period == 'weekly':
        display_data = df_weekly
        chart_title = f"{symbol} - 周线图"
    elif selected_period == 'monthly':
        display_data = df_monthly
        chart_title = f"{symbol} - 月线图"
    else:
        display_data = df_daily.tail(365)
        chart_title = f"{symbol} - 日线图"
    
    if len(display_data) < 10:
        st.warning("数据不足，无法显示图表")
        return
    
    # 日期滑动条
    date_list = display_data.index.tolist()
    default_idx = len(date_list) - 1
    
    selected_date_idx = st.slider(
        "📅 拖动选择日期 (筹码分布会动态变化)",
        min_value=10,
        max_value=len(date_list) - 1,
        value=default_idx,
        format="",
        key=f"slider_{unique_key}_{selected_period}"
    )
    
    selected_date = date_list[selected_date_idx]
    st.caption(f"🎯 选中日期: **{selected_date.strftime('%Y-%m-%d')}** | 收盘价: **{price_symbol}{display_data.loc[selected_date, 'Close']:.2f}**")
    
    # 创建K线图
    chart_data = display_data.iloc[:selected_date_idx + 1].copy()
    
    try:
        # 尝试使用高级图表函数
        from chart_utils import create_candlestick_chart_dynamic
        fig = create_candlestick_chart_dynamic(
            display_data, chart_data, symbol, chart_title,
            period=selected_period, show_volume_profile=True,
            highlight_date=selected_date
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_key}_{selected_period}")
        
        # 显示筹码分析
        if hasattr(fig, '_chip_analysis'):
            chip = fig._chip_analysis
            st.markdown(f"### 📊 筹码快照 {chip.get('buy_signal_strength', '')}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("🟢 获利盘", f"{chip.get('profit_ratio', 0)*100:.1f}%")
            with c2:
                st.metric("🔴 套牢盘", f"{chip.get('trapped_ratio', 0)*100:.1f}%")
            with c3:
                st.metric("📍 集中度", f"{chip.get('concentration', 0)*100:.1f}%")
            with c4:
                st.metric("💰 平均成本", f"{price_symbol}{chip.get('avg_cost', 0):.2f}")
    except Exception as e:
        # 回退到简单图表
        fig = go.Figure(data=[go.Candlestick(
            x=display_data.index,
            open=display_data['Open'],
            high=display_data['High'],
            low=display_data['Low'],
            close=display_data['Close']
        )])
        
        # 添加均线
        ma5 = display_data['Close'].rolling(5).mean()
        ma20 = display_data['Close'].rolling(20).mean()
        ma60 = display_data['Close'].rolling(60).mean()
        
        fig.add_trace(go.Scatter(x=display_data.index, y=ma5, name='MA5', line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=display_data.index, y=ma20, name='MA20', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=display_data.index, y=ma60, name='MA60', line=dict(color='purple', width=1)))
        
        fig.update_layout(
            title=chart_title,
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_simple_{unique_key}")


def _render_phantom_tab(symbol, df_daily, phantom, adx_val, price_symbol, unique_key):
    """渲染幻影主力指标标签页"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("### 👻 幻影主力指标")
    st.caption("海底捞月 + 资金力度 + 改良KDJ (39周期)")
    
    pink = phantom['pink']
    blue_bar = phantom['blue']
    lired = phantom['lired']
    red = phantom['red']
    yellow = phantom['yellow']
    green = phantom['green']
    lightblue = phantom['lightblue']
    buy_sig = phantom['buy_signal']
    sell_sig = phantom['sell_signal']
    blue_dis = phantom['blue_disappear']
    lired_dis = phantom['lired_disappear']
    
    close = df_daily['Close'].values
    n = len(close)
    
    # === 当前状态面板 ===
    pink_val = float(pink[-1])
    red_val = float(red[-1])
    green_val = float(green[-1])
    has_blue = float(blue_bar[-1]) > 0
    has_lired = float(lired[-1]) > 0
    is_buy = bool(buy_sig[-1]) if n > 0 else False
    is_sell = bool(sell_sig[-1]) if n > 0 else False
    is_blue_dis = bool(blue_dis[-1]) if n > 0 else False
    
    # 状态判断
    if is_sell and green_val < 0 and adx_val < 30:
        status_emoji, status_text, status_color = "🚨", "逃顶预警 (多重确认)", "#FF1744"
    elif is_sell:
        status_emoji, status_text, status_color = "⚠️", "PINK逃顶 (需确认)", "#FF6D00"
    elif is_blue_dis and adx_val >= 25:
        status_emoji, status_text, status_color = "🎯", "趋势回调买入 (BLUE消失+趋势)", "#00E676"
    elif is_buy:
        status_emoji, status_text, status_color = "💚", "PINK超卖进场", "#00C853"
    elif has_blue:
        status_emoji, status_text, status_color = "🔵", "海底捞月中 (等待消失=买点)", "#448AFF"
    elif has_lired:
        status_emoji, status_text, status_color = "🔴", "顶部压力出现", "#FF6D00"
    elif pink_val > 90:
        status_emoji, status_text, status_color = "🟡", "超买区域 (注意风险)", "#FFD600"
    elif pink_val < 10:
        status_emoji, status_text, status_color = "🟢", "超卖区域 (关注进场)", "#00C853"
    else:
        status_emoji, status_text, status_color = "⚪", "中性观望", "#8b949e"
    
    # 资金状态
    if red_val > 0:
        flow_text = "主力流入 🔴"
        flow_color = "#FF4444"
    elif green_val < 0:
        flow_text = "资金流出 🟢"
        flow_color = "#00CC00"
    else:
        flow_text = "中性"
        flow_color = "#8b949e"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(88,28,135,0.15), rgba(30,64,175,0.12));
                border: 1px solid rgba(139,92,246,0.3); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 1.8rem;">{status_emoji}</span>
            <div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {status_color};">{status_text}</div>
                <div style="font-size: 0.8rem; color: #8b949e;">PINK: {pink_val:.1f} | 资金: {flow_text}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pk_delta = "超买" if pink_val > 90 else "超卖" if pink_val < 10 else "中性"
        st.metric("PINK (KDJ)", f"{pink_val:.1f}", pk_delta)
    with c2:
        st.metric("海底捞月", "有 🔵" if has_blue else "无", "消失=买点" if has_blue else None)
    with c3:
        st.metric("顶部压力", "有 🔴" if has_lired else "无")
    with c4:
        st.metric("资金方向", flow_text)
    
    # === 信号统计 ===
    lookback = min(120, n)
    recent_buys = int(buy_sig[-lookback:].sum()) if n >= lookback else 0
    recent_sells = int(sell_sig[-lookback:].sum()) if n >= lookback else 0
    recent_blue_dis = int(blue_dis[-lookback:].sum()) if n >= lookback else 0
    
    st.markdown(f"**近{lookback}天信号**: 进场信号 **{recent_buys}**次 | 逃顶信号 **{recent_sells}**次 | BLUE消失 **{recent_blue_dis}**次")
    
    # === Plotly 图表 ===
    # 只显示最近 N 天
    show_days = min(200, n)
    idx_start = n - show_days
    dates = df_daily.index[idx_start:]
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.25, 0.2, 0.2],
        subplot_titles=[f"{symbol} 价格", "海底捞月 (BLUE/LIRED)", "资金力度", "PINK线 (KDJ)"]
    )
    
    # Row 1: K线
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df_daily['Open'].values[idx_start:],
        high=df_daily['High'].values[idx_start:],
        low=df_daily['Low'].values[idx_start:],
        close=close[idx_start:],
        name='K线',
        showlegend=False,
    ), row=1, col=1)
    
    # 买卖信号标记
    buy_idx = np.where(buy_sig[idx_start:])[0]
    sell_idx = np.where(sell_sig[idx_start:])[0]
    blue_dis_idx = np.where(blue_dis[idx_start:])[0]
    
    if len(buy_idx) > 0:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in buy_idx],
            y=[close[idx_start + i] * 0.97 for i in buy_idx],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#00E676'),
            name='进场 (PINK↑10)',
        ), row=1, col=1)
    
    if len(sell_idx) > 0:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in sell_idx],
            y=[close[idx_start + i] * 1.03 for i in sell_idx],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='#FF1744'),
            name='逃顶 (PINK↓90)',
        ), row=1, col=1)
    
    if len(blue_dis_idx) > 0:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in blue_dis_idx],
            y=[close[idx_start + i] * 0.95 for i in blue_dis_idx],
            mode='markers',
            marker=dict(symbol='star', size=10, color='#448AFF'),
            name='BLUE消失 (买点)',
        ), row=1, col=1)
    
    # Row 2: 海底捞月
    bb = blue_bar[idx_start:]
    lr = lired[idx_start:]
    fig.add_trace(go.Bar(
        x=dates, y=bb, name='BLUE (底部)',
        marker_color='#0066FF', opacity=0.8,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=[-v for v in lr], name='LIRED (顶部)',
        marker_color='#FF4444', opacity=0.8,
    ), row=2, col=1)
    
    # Row 3: 资金力度
    r_vals = red[idx_start:]
    y_vals = yellow[idx_start:]
    g_vals = green[idx_start:]
    lb_vals = lightblue[idx_start:]
    
    fig.add_trace(go.Bar(x=dates, y=r_vals, name='超大单流入', marker_color='#FF0000', opacity=0.8), row=3, col=1)
    fig.add_trace(go.Bar(x=dates, y=y_vals, name='大单流入', marker_color='#FFFF00', opacity=0.6), row=3, col=1)
    fig.add_trace(go.Bar(x=dates, y=g_vals, name='资金流出', marker_color='#00FF00', opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lb_vals, name='资金流量线', line=dict(color='#00FFFF', width=1.5)), row=3, col=1)
    
    # Row 4: PINK线
    pk = pink[idx_start:]
    fig.add_trace(go.Scatter(
        x=dates, y=pk, name='PINK (KDJ)',
        line=dict(color='#FF00FF', width=2)
    ), row=4, col=1)
    fig.add_hline(y=90, line_dash="dot", line_color="#FF4444", annotation_text="逃顶线 90", row=4, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="#00CC00", annotation_text="进场线 10", row=4, col=1)
    
    # 超买超卖区域填充
    fig.add_hrect(y0=90, y1=110, fillcolor="rgba(255,23,68,0.08)", line_width=0, row=4, col=1)
    fig.add_hrect(y0=-10, y1=10, fillcolor="rgba(0,200,83,0.08)", line_width=0, row=4, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis4_rangeslider_visible=False,
        xaxis_rangeslider_visible=False,
        barmode='overlay',
    )
    fig.update_xaxes(rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True, key=f"phantom_{unique_key}")
    
    # === 信号解读 ===
    with st.expander("📖 指标解读 & 使用指南", expanded=False):
        st.markdown("""
**信号有效性 (实测回测结果):**

| 信号 | 用法 | 胜率 | 适用场景 |
|---|---|---|---|
| 🔵 BLUE消失 + ADX>25 | **趋势回调买入** | **61%** | 强趋势中的回调 |
| 🔴 PINK↓90 + 资金流出 + ADX<30 | **逃顶预警** | **55%** | 非强趋势行情 |
| 💚 PINK↑10 | **超卖反弹** | ~50% | 需配合其他确认 |
| 红柱/绿柱 | **情绪辅助** | N/A | 辅助判断多空 |

**重要提醒:**
- 🔵 BLUE柱出现 = 正在触底中，**不是买点**。BLUE柱**消失** = 买点
- 逃顶信号在强趋势股(NVDA等)中**容易误报**，需要资金流出确认
- PINK线用的是39周期KDJ (比标准9周期更慢)，更适合中线判断
- 资金力度是量价推算值，**不代表真实主力资金**
        """)
    
    # === 回测统计 ===
    with st.expander("📊 该股信号历史回测", expanded=False):
        # 逃顶回测
        sell_indices = np.where(sell_sig)[0]
        if len(sell_indices) > 0:
            st.markdown("**逃顶信号回测 (PINK↓90):**")
            records = []
            for idx in sell_indices:
                if idx + 5 < n:
                    ret5 = (close[idx + 5] / close[idx] - 1) * 100
                    date_str = str(df_daily.index[idx])[:10]
                    records.append({
                        '日期': date_str,
                        '价格': f"{price_symbol}{close[idx]:.2f}",
                        '5日收益': f"{ret5:+.1f}%",
                        '判断': '✅正确' if ret5 < 0 else '❌错误'
                    })
            if records:
                df_bt = pd.DataFrame(records[-10:])  # 最近10条
                wins = sum(1 for r in records if '✅' in r['判断'])
                st.markdown(f"总{len(records)}次, 胜率 **{wins}/{len(records)} = {wins/len(records)*100:.0f}%**")
                st.dataframe(df_bt, use_container_width=True, hide_index=True, key=f"phantom_bt_sell_{unique_key}")
        
        # BLUE消失回测
        bd_indices = np.where(blue_dis)[0]
        if len(bd_indices) > 0:
            st.markdown("**BLUE消失 (抄底) 回测:**")
            records = []
            for idx in bd_indices:
                if idx + 5 < n:
                    ret5 = (close[idx + 5] / close[idx] - 1) * 100
                    date_str = str(df_daily.index[idx])[:10]
                    in_trend = "✅趋势中" if adx_val >= 25 else "⚠️非趋势"
                    records.append({
                        '日期': date_str,
                        '价格': f"{price_symbol}{close[idx]:.2f}",
                        '5日收益': f"{ret5:+.1f}%",
                        '趋势': in_trend,
                        '判断': '✅正确' if ret5 > 0 else '❌错误'
                    })
            if records:
                df_bt = pd.DataFrame(records[-10:])
                wins = sum(1 for r in records if '✅正确' in r['判断'])
                st.markdown(f"总{len(records)}次, 胜率 **{wins}/{len(records)} = {wins/len(records)*100:.0f}%**")
                st.dataframe(df_bt, use_container_width=True, hide_index=True, key=f"phantom_bt_blue_{unique_key}")


def _render_chips_tab(symbol, df_daily, unique_key):
    """渲染筹码分析标签页"""
    st.markdown("### 📊 筹码分布分析")
    
    try:
        from chart_utils import analyze_chip_flow, create_chip_flow_chart, create_chip_change_chart
        
        # 对比周期选择
        lookback_options = {"5天": 5, "10天": 10, "20天": 20, "30天": 30, "60天": 60}
        selected_lookback = st.select_slider(
            "对比周期",
            options=list(lookback_options.keys()),
            value="20天",
            key=f"chips_lookback_{unique_key}"
        )
        lookback_days = lookback_options[selected_lookback]
        
        chip_flow = analyze_chip_flow(df_daily.tail(365), lookback_days=lookback_days)
        
        if chip_flow:
            st.markdown(f"## {chip_flow['action_emoji']} **{chip_flow['action']}**")
            st.caption(chip_flow['action_desc'])
            
            cf1, cf2, cf3 = st.columns(3)
            with cf1:
                st.metric("低位筹码变化", f"{chip_flow['low_chip_increase']:+.1f}%")
            with cf2:
                st.metric("高位筹码流出", f"{chip_flow['high_chip_decrease']:+.1f}%")
            with cf3:
                st.metric("平均成本变化", f"{chip_flow['cost_change_pct']:+.1f}%")
            
            # 筹码流动图
            with st.expander("📊 筹码流动对比图", expanded=False):
                flow_fig = create_chip_flow_chart(chip_flow, symbol)
                if flow_fig:
                    st.plotly_chart(flow_fig, use_container_width=True, key=f"flow_{unique_key}")
                
                change_fig = create_chip_change_chart(chip_flow)
                if change_fig:
                    st.plotly_chart(change_fig, use_container_width=True, key=f"change_{unique_key}")
        else:
            st.warning("数据不足，无法分析筹码流动")
            
    except Exception as e:
        st.error(f"筹码分析暂不可用: {e}")


def _render_indicators_tab(symbol, current_price, price_symbol,
                           blue_daily, blue_weekly, blue_monthly, adx_val,
                           heima_daily, heima_weekly, heima_monthly,
                           juedi_daily, juedi_weekly, juedi_monthly,
                           yf_info, unique_key):
    """渲染技术指标标签页"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟦 BLUE 信号")
        
        b1, b2, b3 = st.columns(3)
        with b1:
            color = "🟢" if blue_daily > 100 else "⚪"
            st.metric(f"{color} 日线", f"{blue_daily:.0f}")
        with b2:
            color = "🟢" if blue_weekly > 100 else "⚪"
            st.metric(f"{color} 周线", f"{blue_weekly:.0f}")
        with b3:
            color = "🟢" if blue_monthly > 100 else "⚪"
            st.metric(f"{color} 月线", f"{blue_monthly:.0f}")
        
        # 信号解读
        signals = []
        if blue_daily > 100: signals.append("日线抄底")
        if blue_weekly > 100: signals.append("周线抄底")
        if blue_monthly > 100: signals.append("月线抄底")
        
        if signals:
            st.success(f"**当前信号**: {', '.join(signals)}")
        else:
            st.info("当前无BLUE买入信号")
        
        st.divider()
        
        st.markdown("### 🐴 黑马/掘地信号")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("日线", "🐴" if heima_daily else ("⛏️" if juedi_daily else "-"))
        with h2:
            st.metric("周线", "🐴" if heima_weekly else ("⛏️" if juedi_weekly else "-"))
        with h3:
            st.metric("月线", "🐴" if heima_monthly else ("⛏️" if juedi_monthly else "-"))
    
    with col2:
        st.markdown("### 📈 趋势分析")
        st.metric("ADX 趋势强度", f"{adx_val:.1f}")
        
        if adx_val > 40:
            st.success("**极强趋势** - 顺势操作")
        elif adx_val > 25:
            st.info("**中等趋势** - 可考虑入场")
        else:
            st.warning("**弱趋势/震荡** - 谨慎操作")
        
        st.divider()
        
        st.markdown("### 📋 交易计划")
        stop_loss = current_price * 0.92
        target = current_price * 1.15
        rr_ratio = (target - current_price) / (current_price - stop_loss) if current_price > stop_loss else 0
        
        st.metric("🎯 目标价 (+15%)", f"{price_symbol}{target:.2f}")
        st.metric("🛑 止损价 (-8%)", f"{price_symbol}{stop_loss:.2f}")
        st.metric("📊 风险收益比", f"1:{rr_ratio:.1f}")
    
    # 公司基本面 (如果有yfinance数据)
    if yf_info:
        st.divider()
        st.markdown("### 🏢 公司基本面")
        
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            market_cap = yf_info.get('marketCap', 0)
            if market_cap >= 1e12:
                cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                cap_str = f"${market_cap/1e9:.2f}B"
            else:
                cap_str = f"${market_cap/1e6:.2f}M" if market_cap else "N/A"
            st.metric("市值", cap_str)
        with f2:
            pe = yf_info.get('trailingPE', yf_info.get('forwardPE', 'N/A'))
            st.metric("PE", f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A")
        with f3:
            profit = yf_info.get('profitMargins', 0)
            st.metric("利润率", f"{profit*100:.1f}%" if profit else "N/A")
        with f4:
            growth = yf_info.get('revenueGrowth', 0)
            st.metric("营收增长", f"{growth*100:.1f}%" if growth else "N/A")


def _render_ai_diagnosis_tab(symbol, current_price, price_symbol,
                             blue_daily, blue_weekly, blue_monthly, adx_val,
                             market, unique_key):
    """渲染AI诊断标签页"""
    
    st.markdown("### 🤖 AI 智能诊断")
    
    ai_col1, ai_col2 = st.columns([1, 3])
    with ai_col1:
        do_ai_diag = st.button("🚀 启动诊断", key=f"ai_diag_{unique_key}", type="primary", use_container_width=True)
    with ai_col2:
        st.caption("综合技术面、基本面、舆情进行AI分析")
    
    if do_ai_diag:
        with st.spinner("🤖 AI 正在分析..."):
            try:
                from ml.llm_intelligence import LLMAnalyzer
                
                stock_data = {
                    'symbol': symbol,
                    'price': current_price,
                    'blue_daily': blue_daily,
                    'blue_weekly': blue_weekly,
                    'ma5': current_price * 0.98,
                    'ma20': current_price * 0.94,
                    'rsi': 50,
                    'volume_ratio': 1.2
                }
                
                analyzer = LLMAnalyzer(provider='gemini')
                result = analyzer.generate_decision_dashboard(stock_data, "")
                
                # 显示结果
                signal = result.get('signal', 'HOLD')
                confidence = result.get('confidence', 0)
                verdict = result.get('verdict', '分析中...')
                
                signal_colors = {
                    "BUY": ("#00C853", "🟢", "买入"),
                    "SELL": ("#FF1744", "🔴", "卖出"),
                    "HOLD": ("#FFD600", "🟡", "观望")
                }
                color, icon, label = signal_colors.get(signal, ("#FFD600", "🟡", "观望"))
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                            border-left: 4px solid {color}; 
                            padding: 16px; border-radius: 8px;">
                    <h2 style="margin: 0; color: {color};">{icon} {label} | {symbol}</h2>
                    <p style="margin: 8px 0 0 0; font-size: 1.1em;">📌 {verdict}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 价位建议
                st.markdown("**💰 狙击价位**")
                p1, p2, p3 = st.columns(3)
                entry = result.get('entry_price', current_price)
                stop = result.get('stop_loss', current_price * 0.92)
                target = result.get('target_price', current_price * 1.15)
                
                p1.metric("🎯 买入价", f"{price_symbol}{entry:.2f}")
                p2.metric("🛑 止损价", f"{price_symbol}{stop:.2f}")
                p3.metric("🚀 目标价", f"{price_symbol}{target:.2f}")
                
                st.caption(f"📊 置信度: {confidence}%")
                
            except Exception as e:
                st.error(f"AI诊断失败: {e}")
    
    # 大师分析
    st.divider()
    st.markdown("### 🎓 大师量化分析")
    
    if st.button("🤖 咨询5位大师", key=f"master_{unique_key}"):
        with st.spinner("正在咨询大师..."):
            try:
                from strategies.master_strategies import analyze_stock_for_master
                from data_fetcher import get_stock_data
                
                h_df = get_stock_data(symbol, market=market, days=60)
                if h_df is not None and not h_df.empty:
                    cc = h_df['Close'].values
                    td_cnt = 0
                    if len(cc) > 13 and cc[-1] > cc[-5]:
                        for k in range(1, 10):
                            if cc[-k] > cc[-k-4]: td_cnt += 1
                            else: break
                    
                    v_avg = h_df['Volume'].rolling(5).mean().iloc[-1]
                    v_now = h_df['Volume'].iloc[-1]
                    
                    ans = analyze_stock_for_master(
                        symbol=symbol,
                        blue_daily=blue_daily,
                        blue_weekly=blue_weekly,
                        blue_monthly=blue_monthly,
                        adx=adx_val,
                        vol_ratio=v_now/v_avg if v_avg > 0 else 1.0,
                        change_pct=(cc[-1]/cc[-2]-1)*100 if len(cc) > 1 else 0,
                        price=current_price,
                        sma5=h_df['Close'].rolling(5).mean().iloc[-1],
                        sma20=h_df['Close'].rolling(20).mean().iloc[-1],
                        td_count=td_cnt,
                        is_heima=False
                    )
                    
                    # 展示
                    m_cols = st.columns(3)
                    strats = [
                        ('cai_sen', '蔡森(量价)', '📈'),
                        ('td_sequential', 'DeMark(拐点)', '🔄'),
                        ('xiao_mingdao', '萧明道(均线)', '📏'),
                        ('heima', '黑马(爆点)', '🐎'),
                        ('blue', 'BLUE(趋势)', '🌊')
                    ]
                    
                    for i, (k, n, ic) in enumerate(strats):
                        r = ans.get(k)
                        if not r: continue
                        
                        sig = getattr(r, 'signal', None)
                        conf = getattr(r, 'confidence', 0)
                        reason = getattr(r, 'reason', '')
                        
                        with m_cols[i % 3]:
                            st.markdown(f"**{ic} {n}**")
                            if sig == 'BUY':
                                st.success(f"✅ 买入 ({conf}%)")
                            elif sig == 'SELL':
                                st.error(f"❌ 卖出")
                            else:
                                st.info("⚪ 观望")
                            st.caption(str(reason)[:50])
                else:
                    st.warning("数据不足")
            except Exception as e:
                st.error(f"大师分析失败: {e}")


def _render_ask_ai_tab(symbol, company_name, current_price, price_symbol,
                       blue_daily, blue_weekly, yf_info, market, unique_key):
    """渲染问AI标签页"""
    
    st.markdown("### 🗣️ 向AI询问关于这只股票的任何问题")
    
    # 预设问题
    st.markdown("**💡 常见问题:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    preset_question = None
    with q_col1:
        if st.button("📊 公司基本面", key=f"q1_{unique_key}"):
            preset_question = f"{symbol}的基本面如何？主营业务、市值、PE、PB是多少？"
        if st.button("📈 技术面分析", key=f"q4_{unique_key}"):
            preset_question = f"{symbol}的技术形态如何？支撑位和压力位在哪里？"
    
    with q_col2:
        if st.button("💰 财务状况", key=f"q2_{unique_key}"):
            preset_question = f"{symbol}的财务状况如何？营收增长率、利润率、负债率是多少？"
        if st.button("🎯 买卖建议", key=f"q5_{unique_key}"):
            preset_question = f"现在是买入{symbol}的好时机吗？应该设置什么止损和止盈？"
    
    with q_col3:
        if st.button("📰 最近新闻", key=f"q3_{unique_key}"):
            preset_question = f"{symbol}最近有什么重大新闻或事件？对股价有什么影响？"
        if st.button("⚠️ 风险分析", key=f"q6_{unique_key}"):
            preset_question = f"投资{symbol}有哪些主要风险？需要注意什么？"
    
    st.divider()
    
    # 问题输入
    user_question = st.text_input(
        "或输入你的问题:",
        value=preset_question if preset_question else "",
        placeholder=f"例如: {symbol}的竞争对手有哪些？",
        key=f"user_q_{unique_key}"
    )
    
    if st.button("🚀 提问", key=f"ask_{unique_key}", type="primary"):
        if user_question:
            with st.spinner("🤖 AI 正在获取数据并分析..."):
                try:
                    from ml.llm_intelligence import LLMAnalyzer
                    
                    # 构建上下文
                    def format_cap(cap):
                        if cap >= 1e12: return f"${cap/1e12:.2f}万亿"
                        elif cap >= 1e9: return f"${cap/1e9:.2f}亿"
                        else: return f"${cap/1e6:.2f}百万" if cap >= 1e6 else "N/A"
                    
                    market_cap = yf_info.get('marketCap', 0)
                    pe_ratio = yf_info.get('trailingPE', 'N/A')
                    profit_margin = yf_info.get('profitMargins', 0)
                    revenue_growth = yf_info.get('revenueGrowth', 0)
                    business_summary = yf_info.get('longBusinessSummary', '')[:500] if yf_info.get('longBusinessSummary') else ''
                    
                    context = f"""
=== 股票信息 ===
代码: {symbol}
公司名称: {company_name}
行业: {yf_info.get('industry', '未知')}
板块: {yf_info.get('sector', '未知')}

=== 估值指标 ===
市值: {format_cap(market_cap)}
PE: {pe_ratio if pe_ratio != 'N/A' else 'N/A'}
利润率: {f'{profit_margin*100:.1f}%' if profit_margin else 'N/A'}
营收增长: {f'{revenue_growth*100:.1f}%' if revenue_growth else 'N/A'}

=== 技术指标 ===
当前价格: {price_symbol}{current_price:.2f}
日线BLUE: {blue_daily:.0f}
周线BLUE: {blue_weekly:.0f}

=== 公司简介 ===
{business_summary}
"""
                    
                    analyzer = LLMAnalyzer(provider='gemini')
                    response = analyzer.natural_query(f"基于以上{symbol}的数据回答: {user_question}", context)
                    
                    # 显示数据摘要
                    st.markdown("### 📊 已获取数据:")
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.caption(f"**{company_name}**")
                        st.caption(f"行业: {yf_info.get('industry', '未知')}")
                    with d2:
                        st.caption(f"市值: {format_cap(market_cap)}")
                        st.caption(f"PE: {pe_ratio}")
                    with d3:
                        st.caption(f"营收增长: {f'{revenue_growth*100:.1f}%' if revenue_growth else 'N/A'}")
                        st.caption(f"利润率: {f'{profit_margin*100:.1f}%' if profit_margin else 'N/A'}")
                    
                    st.divider()
                    
                    # 显示回答
                    st.markdown("### 🤖 AI 回答:")
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"AI回答失败: {e}")
        else:
            st.warning("请输入问题")


def _render_news_tab(symbol, company_name, market, unique_key):
    """渲染新闻舆情标签页"""
    
    news_col, social_col = st.columns(2)
    
    with news_col:
        st.markdown("### 📰 新闻分析")
        
        if st.button("🔍 获取新闻", key=f"news_{unique_key}"):
            with st.spinner("正在分析新闻..."):
                try:
                    from news import get_news_intelligence
                    
                    intel = get_news_intelligence(use_llm=False)
                    events, impacts, digest = intel.analyze_symbol(symbol, company_name, market=market)
                    
                    if events:
                        sentiment_ratio = digest.sentiment_ratio()
                        sentiment_emoji = "🟢" if sentiment_ratio > 0.3 else ("🔴" if sentiment_ratio < -0.3 else "⚪")
                        
                        st.metric(f"{sentiment_emoji} 市场情绪", f"利好{digest.bullish_count}/利空{digest.bearish_count}")
                        st.metric("📊 预期影响", f"{digest.avg_expected_impact:+.2f}%")
                        
                        st.markdown("**最新新闻:**")
                        for e in events[:5]:
                            sentiment_icon = e.sentiment.emoji if hasattr(e.sentiment, 'emoji') else "➖"
                            st.markdown(f"- {sentiment_icon} [{e.title[:50]}...]({e.url})")
                    else:
                        st.info("📭 暂无相关新闻")
                except Exception as e:
                    st.warning(f"新闻分析暂不可用: {e}")
    
    with social_col:
        st.markdown("### 🗣️ 社区舆情")
        
        if st.button("🔍 分析舆情", key=f"social_{unique_key}"):
            with st.spinner("扫描社区讨论..."):
                try:
                    from services.social_monitor import get_social_service
                    svc = get_social_service()
                    report = svc.get_social_report(symbol, market=market)
                    
                    s1, s2, s3 = st.columns(3)
                    s1.metric("🐂 看多", report['bullish_count'])
                    s2.metric("🐻 看空", report['bearish_count'])
                    s3.metric("😶 中性", report['neutral_count'])
                    
                    if report['posts']:
                        st.markdown("**热门讨论:**")
                        for p in report['posts'][:3]:
                            icon = "🐦" if p.platform == "Twitter" else "🤖"
                            sent = "🟢" if p.sentiment == "Bullish" else "🔴" if p.sentiment == "Bearish" else "⚪"
                            st.markdown(f"- {icon}{sent} {p.title[:40]}...")
                    else:
                        st.info("暂无讨论")
                except Exception as e:
                    st.warning(f"舆情分析暂不可用: {e}")


def _render_actions(symbol, current_price, price_symbol, blue_daily, blue_weekly, market, unique_key):
    """渲染操作区 - 快速交易工具"""
    
    st.markdown("### 💰 快速操作")
    
    tab_buy, tab_calc, tab_watch = st.tabs(["🛒 模拟买入", "📐 仓位/P&L 计算器", "📋 观察列表"])
    
    with tab_buy:
        col_buy1, col_buy2 = st.columns([2, 1])
        
        with col_buy1:
            suggested_shares = max(1, int(1000 / current_price)) if current_price > 0 else 10
            shares = st.number_input("买入股数", min_value=1, value=suggested_shares, key=f"shares_{unique_key}")
            
            buy_cost = shares * current_price
            
            # 快速股数选择
            quick_cols = st.columns(4)
            amounts = [1000, 5000, 10000, 50000]
            for i, amt in enumerate(amounts):
                with quick_cols[i]:
                    quick_shares = max(1, int(amt / current_price)) if current_price > 0 else 1
                    if st.button(f"{price_symbol}{amt:,}", key=f"quick_{amt}_{unique_key}", use_container_width=True):
                        st.session_state[f"shares_{unique_key}"] = quick_shares
                        st.rerun()
        
        with col_buy2:
            st.metric("买入成本", f"{price_symbol}{buy_cost:,.2f}")
            stop_price = current_price * 0.92
            target_price = current_price * 1.15
            st.caption(f"🛑 建议止损: {price_symbol}{stop_price:.2f} (-8%)")
            st.caption(f"🎯 建议目标: {price_symbol}{target_price:.2f} (+15%)")
            max_loss = shares * (current_price - stop_price)
            st.caption(f"⚠️ 最大亏损: {price_symbol}{max_loss:.2f}")
        
        if st.button("✅ 确认模拟买入", key=f"buy_{unique_key}", type="primary", use_container_width=True):
            try:
                from services.portfolio_service import paper_buy
                result = paper_buy(symbol, shares, current_price, market)
                if result.get('success'):
                    st.success(f"✅ 买入成功! {symbol} x {shares}股 @ {price_symbol}{current_price:.2f}")
                    st.caption(f"佣金: {price_symbol}{result.get('commission', 0):.2f} | 余额: {price_symbol}{result.get('new_balance', 0):,.2f}")
                    st.balloons()
                else:
                    st.error(f"❌ {result.get('error', '未知错误')}")
            except Exception as e:
                st.error(f"❌ 买入异常: {e}")
    
    with tab_calc:
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            st.markdown("**📐 仓位计算器**")
            account_size = st.number_input("账户总资金", min_value=1000, value=100000, step=10000, 
                                           key=f"acct_{unique_key}", format="%d")
            risk_pct = st.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5, key=f"risk_{unique_key}")
            stop_pct = st.slider("止损幅度 (%)", 2.0, 15.0, 8.0, 1.0, key=f"stop_{unique_key}")
            
            risk_amount = account_size * risk_pct / 100
            stop_distance = current_price * stop_pct / 100
            calc_shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            position_size = calc_shares * current_price
            position_pct = position_size / account_size * 100 if account_size > 0 else 0
            
            st.success(f"**建议买入: {calc_shares} 股**")
            st.caption(f"仓位金额: {price_symbol}{position_size:,.0f} ({position_pct:.1f}%)")
            st.caption(f"风险金额: {price_symbol}{risk_amount:,.0f}")
            st.caption(f"止损价: {price_symbol}{current_price * (1 - stop_pct/100):.2f}")
        
        with calc_col2:
            st.markdown("**💹 P&L 计算器**")
            entry_p = st.number_input("买入价", value=round(current_price, 2), step=0.01, 
                                      key=f"entry_{unique_key}", format="%.2f")
            exit_p = st.number_input("卖出价", value=round(current_price * 1.10, 2), step=0.01, 
                                     key=f"exit_{unique_key}", format="%.2f")
            pl_shares = st.number_input("股数", min_value=1, value=100, key=f"pl_shares_{unique_key}")
            
            profit = (exit_p - entry_p) * pl_shares
            profit_pct = (exit_p / entry_p - 1) * 100 if entry_p > 0 else 0
            
            if profit >= 0:
                st.success(f"**盈利: {price_symbol}{profit:,.2f} (+{profit_pct:.1f}%)**")
            else:
                st.error(f"**亏损: {price_symbol}{profit:,.2f} ({profit_pct:.1f}%)**")
            
            # 风险回报比
            rr_stop = current_price * 0.92
            rr_target = exit_p
            risk = entry_p - rr_stop
            reward = rr_target - entry_p
            rr_ratio = reward / risk if risk > 0 else 0
            st.caption(f"风险回报比: **{rr_ratio:.1f}:1**" + (" ✅" if rr_ratio >= 2 else " ⚠️"))
    
    with tab_watch:
        st.markdown("**📋 加入观察列表**")
        
        watch_cols = st.columns([2, 1, 1])
        with watch_cols[0]:
            watch_note = st.text_input("备注", value=f"日BLUE:{blue_daily:.0f} 周BLUE:{blue_weekly:.0f}", 
                                       key=f"watch_note_{unique_key}")
        with watch_cols[1]:
            watch_target = st.number_input("目标价", value=round(current_price * 1.15, 2), 
                                           key=f"watch_target_{unique_key}", format="%.2f")
        with watch_cols[2]:
            watch_stop = st.number_input("止损价", value=round(current_price * 0.92, 2), 
                                         key=f"watch_stop_{unique_key}", format="%.2f")
        
        if st.button("➕ 加入观察", key=f"watch_{unique_key}", use_container_width=True, type="primary"):
            try:
                from services.signal_tracker import add_to_watchlist
                add_to_watchlist(
                    symbol=symbol,
                    market=market,
                    entry_price=current_price,
                    target_price=watch_target,
                    stop_loss=watch_stop,
                    signal_type='manual',
                    signal_score=blue_daily,
                    notes=watch_note
                )
                st.success(f"✅ {symbol} 已加入观察列表")
            except Exception as e:
                st.error(f"添加失败: {e}")


def _render_ml_prediction_tab(
    symbol: str,
    market: str,
    hist_data: pd.DataFrame,
    blue_daily: float,
    blue_weekly: float,
    blue_monthly: float,
    is_heima: bool,
    current_price: float,
    price_symbol: str,
    unique_key: str
):
    """
    渲染 ML 预测标签页
    
    显示:
    1. 收益预测 (ReturnPredictor)
    2. 排序得分 (SignalRanker - Learning to Rank)
    3. 交易建议 (止损/目标/仓位)
    """
    st.markdown("### 🎯 AI 智能预测")
    
    try:
        from ml.smart_picker import SmartPicker, StockPick
        
        # 构造信号数据
        signal_data = pd.Series({
            'symbol': symbol,
            'price': current_price,
            'blue_daily': blue_daily,
            'blue_weekly': blue_weekly,
            'blue_monthly': blue_monthly,
            'is_heima': 1 if is_heima else 0,
            'company_name': ''
        })
        
        # 分析三个周期 (skip_prefilter=True: 用户已主动选择该股票，不需要预过滤)
        results = {}
        for horizon in ['short', 'medium', 'long']:
            picker = SmartPicker(market=market, horizon=horizon)
            pick = picker._analyze_stock(signal_data, hist_data, skip_prefilter=True)
            if pick:
                results[horizon] = pick
        
        if not results:
            st.warning("⚠️ 无法生成预测 (模型未加载或数据异常)")
            st.caption("可能原因: 1) ML 依赖未安装 2) 模型文件缺失 3) 价格数据异常")
            return
        
        # === 选择默认周期 ===
        horizon_labels = {"short": "短线 (1-5天)", "medium": "中线 (10-30天)", "long": "长线 (60天+)"}
        selected_horizon = st.radio(
            "选择预测周期",
            options=list(results.keys()),
            format_func=lambda x: horizon_labels.get(x, x),
            horizontal=True,
            key=f"ml_horizon_{unique_key}"
        )
        
        pick = results.get(selected_horizon)
        if not pick:
            st.warning("该周期无预测数据")
            return
        
        st.divider()
        
        # === 核心指标卡片 ===
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            stars = "⭐" * pick.star_rating + "☆" * (5 - pick.star_rating)
            st.metric("综合评分", f"{pick.overall_score:.0f}/100")
            st.caption(stars)
        
        with m2:
            color = "green" if pick.pred_return_5d > 0 else "red"
            st.metric(
                "预测收益", 
                f"{pick.pred_return_5d:+.1f}%",
                delta=f"上涨概率 {pick.pred_direction_prob:.0%}"
            )
        
        with m3:
            # 获取对应周期的排名分
            rank_score = pick.rank_score_short
            if selected_horizon == 'medium':
                rank_score = pick.rank_score_medium
            elif selected_horizon == 'long':
                rank_score = pick.rank_score_long
            st.metric("🏆 排序得分", f"{rank_score:.1f}")
            st.caption("Learning to Rank")
        
        with m4:
            st.metric("风险收益比", f"1:{pick.risk_reward_ratio:.1f}")
            st.caption(f"建议仓位: {pick.suggested_position_pct:.0f}%")
        
        st.divider()
        
        # === 交易计划 ===
        st.markdown("### 📋 交易计划")
        
        plan_cols = st.columns(3)
        
        with plan_cols[0]:
            st.markdown(f"""
            **🎯 入场价**
            
            <div style="font-size: 1.5em; font-weight: bold; color: #2196F3;">
                {price_symbol}{current_price:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        with plan_cols[1]:
            st.markdown(f"""
            **🛑 止损价**
            
            <div style="font-size: 1.5em; font-weight: bold; color: #FF5252;">
                {price_symbol}{pick.stop_loss_price:.2f}
            </div>
            <div style="color: #FF5252;">({pick.stop_loss_pct:+.1f}%)</div>
            """, unsafe_allow_html=True)
        
        with plan_cols[2]:
            st.markdown(f"""
            **🎯 目标价**
            
            <div style="font-size: 1.5em; font-weight: bold; color: #00C853;">
                {price_symbol}{pick.target_price:.2f}
            </div>
            <div style="color: #00C853;">(+{pick.target_pct:.1f}%)</div>
            """, unsafe_allow_html=True)
        
        # === 专业仓位计算 ===
        st.markdown("### 💰 仓位计算器")
        st.caption("基于固定比例仓位法 (风险管理最佳实践)")
        
        pos_cols = st.columns([1, 1, 2])
        
        with pos_cols[0]:
            total_capital = st.number_input(
                "总资金 ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000,
                key=f"ml_capital_{unique_key}"
            )
        
        with pos_cols[1]:
            risk_per_trade = st.slider(
                "单笔风险 (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key=f"ml_risk_{unique_key}"
            ) / 100
        
        with pos_cols[2]:
            # 使用 PositionSizer 计算
            try:
                from risk.position_sizer import PositionSizer
                
                sizer = PositionSizer(total_capital=total_capital, risk_per_trade=risk_per_trade)
                result = sizer.fixed_fractional(
                    entry_price=current_price,
                    stop_loss=pick.stop_loss_price
                )
                
                shares = result.get('shares', 0)
                position_value = result.get('position_value', 0)
                position_pct = result.get('position_pct', 0)
                risk_amount = result.get('risk_amount', 0)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            padding: 16px; border-radius: 10px; border-left: 4px solid #00C853;">
                    <div style="font-size: 1.8em; font-weight: bold; color: #00C853;">
                        买入 {shares} 股
                    </div>
                    <div style="margin-top: 8px;">
                        📊 仓位金额: {price_symbol}{position_value:,.0f} ({position_pct:.1%})
                    </div>
                    <div>
                        ⚠️ 最大亏损: {price_symbol}{risk_amount:,.0f} ({risk_per_trade:.1%}本金)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                # 回退到简单计算
                risk_amount = total_capital * risk_per_trade
                stop_distance = current_price - pick.stop_loss_price
                shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
                position_value = shares * current_price
                
                st.metric("建议买入", f"{shares} 股")
                st.caption(f"仓位: {price_symbol}{position_value:,.0f}")
        
        # 凯利公式建议 (可选展开)
        with st.expander("📈 凯利公式建议 (进阶)", expanded=False):
            st.markdown("""
            **凯利公式** 是数学家 John Kelly 提出的最优仓位公式:
            
            `f* = (bp - q) / b`
            
            其中:
            - b = 赔率 (平均盈利 / 平均亏损)
            - p = 胜率
            - q = 1 - p
            """)
            
            kelly_col1, kelly_col2 = st.columns(2)
            with kelly_col1:
                win_rate = st.slider("历史胜率 (%)", 30, 80, 55, key=f"kelly_wr_{unique_key}") / 100
                avg_win = st.number_input("平均盈利 (%)", 1.0, 50.0, 8.0, key=f"kelly_win_{unique_key}")
            with kelly_col2:
                avg_loss = st.number_input("平均亏损 (%)", 1.0, 20.0, 5.0, key=f"kelly_loss_{unique_key}")
            
            try:
                from risk.position_sizer import PositionSizer
                sizer = PositionSizer(total_capital=total_capital)
                kelly_fraction = sizer.kelly_criterion(
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    fraction=0.5  # 半凯利 (更保守)
                )
                
                st.metric(
                    "半凯利建议仓位", 
                    f"{kelly_fraction:.1%}",
                    delta=f"约 {price_symbol}{total_capital * kelly_fraction:,.0f}"
                )
                
                if kelly_fraction <= 0:
                    st.warning("⚠️ 凯利公式建议不开仓 (期望值为负)")
                elif kelly_fraction > 0.25:
                    st.info("💡 凯利建议仓位较高，建议使用半凯利或更保守的比例")
                    
            except Exception as e:
                st.warning(f"凯利计算失败: {e}")
        
        st.divider()
        
        # === 信号验证 ===
        st.markdown("### ✓ 信号验证")
        
        sig_cols = st.columns(2)
        
        with sig_cols[0]:
            st.markdown("**确认信号:**")
            if pick.signals_confirmed:
                for sig in pick.signals_confirmed:
                    st.markdown(f"<span style='color: #00C853;'>{sig}</span>", unsafe_allow_html=True)
            else:
                st.caption("无确认信号")
        
        with sig_cols[1]:
            st.markdown("**风险提示:**")
            if pick.signals_warning:
                for warn in pick.signals_warning:
                    st.markdown(f"<span style='color: #FFD600;'>{warn}</span>", unsafe_allow_html=True)
            else:
                st.caption("暂无风险提示")
        
        # === 指标徽章 ===
        st.markdown(f"""
        <div style="display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap;">
            <span style="background: #E91E6333; padding: 6px 12px; border-radius: 12px; font-weight: bold;">
                🏆 排名分 {rank_score:.0f}
            </span>
            <span style="background: #00C85333; padding: 6px 12px; border-radius: 12px;">
                日B {pick.blue_daily:.0f}
            </span>
            <span style="background: #FFD60033; padding: 6px 12px; border-radius: 12px;">
                周B {pick.blue_weekly:.0f}
            </span>
            <span style="background: #2196F333; padding: 6px 12px; border-radius: 12px;">
                月B {pick.blue_monthly:.0f}
            </span>
            <span style="background: #9C27B033; padding: 6px 12px; border-radius: 12px;">
                RSI {pick.rsi:.0f}
            </span>
            <span style="background: #FF572233; padding: 6px 12px; border-radius: 12px;">
                量比 {pick.volume_ratio:.1f}x
            </span>
            <span style="background: #60606033; padding: 6px 12px; border-radius: 12px;">
                信号分 {pick.signal_score}/5
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # === 模型状态 ===
        with st.expander("📊 模型详情", expanded=False):
            from pathlib import Path
            model_dir = Path(__file__).parent.parent / "ml" / "saved_models" / f"v2_{market.lower()}"
            
            status_cols = st.columns(2)
            with status_cols[0]:
                return_exists = (model_dir / "return_5d.joblib").exists()
                if return_exists:
                    st.success("✓ 收益预测模型已加载")
                else:
                    st.warning("⚠ 收益预测模型未训练 (使用规则引擎)")
            
            with status_cols[1]:
                ranker_exists = (model_dir / f"ranker_{selected_horizon}.joblib").exists()
                if ranker_exists:
                    st.success(f"✓ 排序模型 ({selected_horizon}) 已加载")
                else:
                    st.warning(f"⚠ 排序模型 ({selected_horizon}) 未训练 (使用规则引擎)")
            
            st.markdown("""
            **评分构成:**
            - 排序模型分 (25%): Learning to Rank 输出
            - 收益预测分 (20%): 预测收益 × 置信度
            - 信号验证分 (25%): BLUE/MACD/成交量确认
            - 方向概率分 (15%): 上涨概率
            - 风险收益分 (15%): 风险收益比
            """)
        
        # === 免责声明 ===
        st.caption("⚠️ 以上预测基于历史数据和技术分析，仅供参考，不构成投资建议。请严格执行止损。")
        
    except Exception as e:
        st.error(f"ML 预测失败: {e}")
        import traceback
        with st.expander("错误详情"):
            st.code(traceback.format_exc())
