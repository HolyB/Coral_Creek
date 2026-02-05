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
        # 历史数据
        if hist_data is None:
            hist_data = get_stock_data(symbol, market=market, days=3650)  # 10年
        
        if hist_data is None or hist_data.empty:
            st.error(f"❌ 无法获取 {symbol} 的数据")
            return
        
        # 获取yfinance信息 (公司基本面)
        yf_info = _get_yfinance_info(symbol) if show_ask_ai or show_indicators else {}
        
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
    
    # === 2. 顶部概览 ===
    st.subheader(f"🔍 {symbol} - {company_name}")
    
    # 指标卡片
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("当前价格", f"{price_symbol}{current_price:.2f}")
    with m2:
        st.metric("日BLUE", f"{blue_daily:.0f}", delta="信号" if blue_daily > 100 else None)
    with m3:
        st.metric("周BLUE", f"{blue_weekly:.0f}", delta="信号" if blue_weekly > 100 else None)
    with m4:
        st.metric("月BLUE", f"{blue_monthly:.0f}", delta="信号" if blue_monthly > 100 else None)
    with m5:
        st.metric("ADX", f"{adx_val:.1f}", delta="强趋势" if adx_val > 25 else None)
    with m6:
        signals = []
        if heima_daily: signals.append("日🐴")
        if heima_weekly: signals.append("周🐴")
        if heima_monthly: signals.append("月🐴")
        st.metric("黑马信号", " ".join(signals) if signals else "无")
    
    st.divider()
    
    # === 3. 主要内容标签页 ===
    tabs = []
    tab_names = []
    
    # ML预测放在最前面 (重要)
    tab_names.append("🎯 ML预测")
    
    if show_charts:
        tab_names.append("📈 K线图表")
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
    """渲染操作区"""
    
    st.markdown("### 💰 操作")
    
    act_col1, act_col2 = st.columns(2)
    
    with act_col1:
        st.markdown("**📋 加入观察列表**")
        if st.button("➕ 加入观察", key=f"watch_{unique_key}", use_container_width=True):
            try:
                from services.signal_tracker import add_to_watchlist
                add_to_watchlist(
                    symbol=symbol,
                    market=market,
                    entry_price=current_price,
                    target_price=current_price * 1.15,
                    stop_loss=current_price * 0.92,
                    signal_type='manual',
                    signal_score=blue_daily,
                    notes=f"手动添加 | 日BLUE:{blue_daily:.0f} 周BLUE:{blue_weekly:.0f}"
                )
                st.success(f"✅ {symbol} 已加入观察列表")
            except Exception as e:
                st.error(f"添加失败: {e}")
    
    with act_col2:
        st.markdown("**💰 模拟买入**")
        
        suggested_shares = max(1, int(1000 / current_price)) if current_price > 0 else 10
        shares = st.number_input("买入股数", min_value=1, value=suggested_shares, key=f"shares_{unique_key}")
        
        buy_cost = shares * current_price
        st.caption(f"预计花费: {price_symbol}{buy_cost:,.2f}")
        
        if st.button("✅ 确认买入", key=f"buy_{unique_key}", type="primary", use_container_width=True):
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
        
        # 分析三个周期
        results = {}
        for horizon in ['short', 'medium', 'long']:
            picker = SmartPicker(market=market, horizon=horizon)
            pick = picker._analyze_stock(signal_data, hist_data)
            if pick:
                results[horizon] = pick
        
        if not results:
            st.warning("⚠️ 无法生成预测 (数据不足或模型未训练)")
            st.info("💡 请确保已训练 ML 模型，或数据至少有 60 天历史")
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
