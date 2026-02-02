#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
📰 新闻中心页面
事件驱动的智能新闻分析
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

# 导入新闻智能系统
try:
    from news import NewsIntelligence, get_news_intelligence
    from news.models import EventType, Sentiment, NewsDigest
    NEWS_AVAILABLE = True
except ImportError as e:
    NEWS_AVAILABLE = False
    print(f"News module not available: {e}")


def render_news_center_page():
    """渲染新闻中心页面"""
    st.title("📰 新闻智能中心")
    st.caption("事件驱动的新闻分析与信号增强系统")
    
    if not NEWS_AVAILABLE:
        st.error("❌ 新闻模块未正确加载")
        return
    
    # 创建 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 个股分析", 
        "📊 持仓新闻", 
        "🚨 重要提醒",
        "📈 新闻表现"
    ])
    
    with tab1:
        render_single_stock_analysis()
    
    with tab2:
        render_portfolio_news()
    
    with tab3:
        render_news_alerts()
    
    with tab4:
        render_news_performance()


def render_single_stock_analysis():
    """个股新闻分析"""
    st.subheader("🔍 个股新闻分析")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="NVDA", placeholder="如 AAPL, TSLA, 600519.SH")
    
    with col2:
        market = st.selectbox("市场", ["US", "CN"], index=0)
    
    with col3:
        use_llm = st.checkbox("使用 AI 分析", value=False, help="使用 Gemini 进行深度分析")
    
    if st.button("🔎 分析新闻", type="primary", use_container_width=True):
        if not symbol:
            st.warning("请输入股票代码")
            return
        
        with st.spinner(f"正在分析 {symbol} 的新闻..."):
            try:
                intel = get_news_intelligence(use_llm=use_llm)
                events, impacts, digest = intel.analyze_symbol(symbol.upper(), market=market)
                
                if not events:
                    st.info(f"📭 暂无 {symbol} 相关新闻")
                    return
                
                # 显示摘要卡片
                render_digest_card(digest, symbol)
                
                st.divider()
                
                # 显示新闻列表
                st.subheader(f"📋 新闻详情 ({len(events)} 条)")
                
                for i, (event, impact) in enumerate(zip(events, impacts)):
                    render_news_card(event, impact, i)
                    
            except Exception as e:
                st.error(f"分析失败: {e}")


def render_digest_card(digest: NewsDigest, symbol: str):
    """渲染新闻摘要卡片"""
    # 情绪指示器
    sentiment_ratio = digest.sentiment_ratio()
    if sentiment_ratio > 0.3:
        sentiment_color = "green"
        sentiment_text = "看涨"
        sentiment_emoji = "🟢"
    elif sentiment_ratio < -0.3:
        sentiment_color = "red"
        sentiment_text = "看跌"
        sentiment_emoji = "🔴"
    else:
        sentiment_color = "gray"
        sentiment_text = "中性"
        sentiment_emoji = "⚪"
    
    # 卡片布局
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📰 新闻数量",
            digest.total_news_count,
            delta=None
        )
    
    with col2:
        delta_text = f"+{digest.bullish_count}" if digest.bullish_count > digest.bearish_count else f"-{digest.bearish_count}"
        st.metric(
            f"{sentiment_emoji} 市场情绪",
            sentiment_text,
            delta=f"利好{digest.bullish_count}/利空{digest.bearish_count}"
        )
    
    with col3:
        impact = digest.avg_expected_impact
        st.metric(
            "📊 预期影响",
            f"{impact:+.2f}%",
            delta="强势" if abs(impact) > 3 else "温和"
        )
    
    with col4:
        st.metric(
            "🎯 信号调整",
            f"{digest.signal_adjustment:.2f}x",
            delta="增强" if digest.signal_adjustment > 1 else ("减弱" if digest.signal_adjustment < 1 else "不变")
        )
    
    # 关键事件
    if digest.key_events:
        st.info("🔑 **关键事件:** " + " | ".join(digest.key_events[:3]))


def render_news_card(event, impact, index: int):
    """渲染单条新闻卡片"""
    # 情感颜色
    sentiment_colors = {
        Sentiment.VERY_BULLISH: "🟢🟢",
        Sentiment.BULLISH: "🟢",
        Sentiment.NEUTRAL: "⚪",
        Sentiment.BEARISH: "🔴",
        Sentiment.VERY_BEARISH: "🔴🔴"
    }
    
    sentiment_emoji = sentiment_colors.get(event.sentiment, "⚪")
    
    # 使用 expander 展示详情
    with st.expander(
        f"{sentiment_emoji} **{event.title[:60]}{'...' if len(event.title) > 60 else ''}**",
        expanded=(index == 0)  # 第一条默认展开
    ):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            **📌 事件类型:** {event.event_type.chinese_name}
            
            **📅 发布时间:** {event.published_at.strftime('%Y-%m-%d %H:%M') if event.published_at else 'N/A'}
            
            **📰 来源:** {event.source}
            
            **🔗 链接:** [{event.source}]({event.url})
            """)
        
        with col2:
            st.markdown(f"""
            **预期影响**
            ### {impact.expected_impact_pct:+.1f}%
            
            置信度: {impact.confidence:.0f}%
            
            紧急度: {'🔥' * impact.urgency}
            """)
        
        # 关键词标签
        if event.keywords:
            st.markdown("**🏷️ 关键词:** " + " ".join([f"`{kw}`" for kw in event.keywords[:5]]))


def render_portfolio_news():
    """持仓新闻分析"""
    st.subheader("📊 持仓相关新闻")
    
    # 输入持仓列表
    default_portfolio = "NVDA, AAPL, MSFT, GOOGL, TSLA"
    portfolio_input = st.text_input(
        "输入持仓代码 (逗号分隔)",
        value=default_portfolio,
        help="输入你的持仓股票代码"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        market = st.selectbox("市场", ["US", "CN"], index=0, key="portfolio_market")
    with col2:
        use_llm = st.checkbox("使用 AI 分析", value=False, key="portfolio_llm")
    
    if st.button("📊 分析持仓新闻", type="primary", use_container_width=True):
        symbols = [s.strip().upper() for s in portfolio_input.split(",") if s.strip()]
        
        if not symbols:
            st.warning("请输入股票代码")
            return
        
        intel = get_news_intelligence(use_llm=use_llm)
        
        progress = st.progress(0)
        all_digests = []
        all_alerts = []
        
        for i, symbol in enumerate(symbols):
            progress.progress((i + 1) / len(symbols))
            
            try:
                events, impacts, digest = intel.analyze_symbol(symbol, market=market)
                all_digests.append((symbol, digest))
                
                # 收集提醒
                for event, impact in zip(events, impacts):
                    if impact.should_alert:
                        all_alerts.append({
                            'symbol': symbol,
                            'title': event.title,
                            'event_type': event.event_type.chinese_name,
                            'sentiment': event.sentiment.emoji,
                            'expected_impact': impact.expected_impact_pct
                        })
            except Exception as e:
                st.warning(f"⚠️ {symbol} 分析失败: {e}")
        
        progress.empty()
        
        # 显示摘要表格
        if all_digests:
            st.subheader("📋 持仓新闻摘要")
            
            df_data = []
            for symbol, digest in all_digests:
                sentiment_ratio = digest.sentiment_ratio()
                sentiment_emoji = "🟢" if sentiment_ratio > 0.3 else ("🔴" if sentiment_ratio < -0.3 else "⚪")
                
                df_data.append({
                    '股票': symbol,
                    '新闻数': digest.total_news_count,
                    '情绪': sentiment_emoji,
                    '利好': digest.bullish_count,
                    '利空': digest.bearish_count,
                    '预期影响': f"{digest.avg_expected_impact:+.2f}%",
                    '信号调整': f"{digest.signal_adjustment:.2f}x"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 显示需要关注的提醒
        if all_alerts:
            st.subheader(f"🚨 需要关注 ({len(all_alerts)} 条)")
            
            for alert in all_alerts[:10]:
                impact = alert['expected_impact']
                color = "green" if impact > 0 else "red"
                st.markdown(f"""
                **{alert['symbol']}** - {alert['event_type']} {alert['sentiment']}
                
                {alert['title'][:50]}... ({impact:+.1f}%)
                """)
                st.divider()


def render_news_alerts():
    """重要新闻提醒"""
    st.subheader("🚨 重要新闻提醒")
    
    st.info("""
    💡 **提示**: 此功能会自动监控你的持仓，当有重大新闻时推送提醒到 Telegram。
    
    **触发条件:**
    - 财报发布 (业绩超预期/暴雷)
    - 分析师评级变化
    - 重大并购/拆分
    - 法律/监管事件
    - 预期影响 > 3%
    """)
    
    # 配置提醒
    st.subheader("⚙️ 提醒配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("监控列表", value="NVDA, AAPL, TSLA", help="逗号分隔")
        impact_threshold = st.slider("影响阈值 (%)", 1, 10, 3, help="预期影响超过此值才提醒")
    
    with col2:
        st.multiselect(
            "事件类型过滤",
            options=["财报", "评级", "并购", "产品", "法律", "全部"],
            default=["财报", "评级", "并购"]
        )
        st.checkbox("开启 Telegram 推送", value=True)
    
    st.markdown("---")
    
    # 最近提醒历史 (占位)
    st.subheader("📜 最近提醒")
    
    st.markdown("""
    | 时间 | 股票 | 事件 | 影响 | 状态 |
    |------|------|------|------|------|
    | 02-01 14:30 | NVDA | 📊 财报超预期 | +5.2% | ✅ 已推送 |
    | 02-01 10:15 | AAPL | 📈 分析师上调 | +2.1% | ✅ 已推送 |
    | 01-31 16:00 | TSLA | ⚠️ 产能问题 | -3.5% | ✅ 已推送 |
    """)


def render_news_performance():
    """新闻预测表现追踪"""
    st.subheader("📈 新闻预测表现")
    
    st.info("""
    💡 **说明**: 追踪新闻预测的准确性，帮助优化模型。
    
    **计算方法:**
    - 方向准确率: 预测涨跌方向正确的比例
    - 幅度准确率: 预测幅度与实际幅度的接近程度
    """)
    
    # 模拟数据展示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 总预测数", "156", delta="+23 本周")
    
    with col2:
        st.metric("🎯 方向准确率", "68%", delta="+5%")
    
    with col3:
        st.metric("📏 幅度准确率", "54%", delta="+2%")
    
    with col4:
        st.metric("💰 累计价值", "+$12,340", delta="基于预测的虚拟收益")
    
    st.divider()
    
    # 按事件类型的表现
    st.subheader("📊 各事件类型表现")
    
    performance_data = {
        '事件类型': ['📊 财报', '📈 评级', '🤝 并购', '📦 产品', '⚖️ 法律', '🌍 宏观'],
        '预测数': [45, 38, 12, 28, 8, 25],
        '方向准确率': ['78%', '65%', '83%', '60%', '71%', '52%'],
        '平均影响': ['+4.2%', '+2.1%', '+8.5%', '+1.8%', '-3.2%', '+1.5%'],
        '可信度': ['⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐']
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **💡 洞察:**
    - 并购事件预测最准确 (83%)，但样本量较小
    - 财报事件影响最大 (+4.2%)，可信度高
    - 宏观事件最难预测 (52%)，建议降低权重
    """)


# 导出页面函数
def get_news_center_page():
    """获取页面渲染函数"""
    return render_news_center_page
