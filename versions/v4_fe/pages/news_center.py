#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
📰 新闻智能中心 (v2)
====================
多源新闻 + 社交媒体 + AI 分析
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

# 导入新闻系统
try:
    from news import NewsIntelligence, get_news_intelligence
    from news.models import EventType, Sentiment, NewsDigest
    from news.crawler import (
        get_news_crawler, StockTwitsCrawler, ApeWisdomCrawler
    )
    NEWS_AVAILABLE = True
except ImportError as e:
    NEWS_AVAILABLE = False
    print(f"News module not available: {e}")


def render_news_center_page():
    """渲染新闻中心页面"""
    st.title("📰 新闻智能中心")
    st.caption("多源新闻聚合 + 社交媒体情绪 + AI 分析 | Google News · yfinance · StockTwits · Reddit")
    
    if not NEWS_AVAILABLE:
        st.error("❌ 新闻模块未正确加载")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 个股新闻",
        "🔥 社交热度",
        "📊 持仓新闻",
        "📈 趋势发现",
    ])
    
    with tab1:
        _render_single_stock_tab()
    with tab2:
        _render_social_buzz_tab()
    with tab3:
        _render_portfolio_news_tab()
    with tab4:
        _render_trending_tab()


def _render_single_stock_tab():
    """个股新闻分析 — 多源 + AI 分类"""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.text_input("股票代码", value="NVDA", placeholder="如 AAPL, TSLA, 600519.SH",
                               key="news_symbol")
    with col2:
        market = st.selectbox("市场", ["US", "CN"], index=0, key="news_market")
    with col3:
        use_llm = st.checkbox("🧠 AI 分类", value=True, help="用 Gemini 批量分类",
                              key="news_llm")
    
    if st.button("🔎 分析新闻", type="primary", use_container_width=True, key="news_analyze"):
        if not symbol:
            st.warning("请输入股票代码")
            return
        
        with st.spinner(f"正在从多个来源抓取 {symbol.upper()} 的新闻..."):
            try:
                intel = get_news_intelligence(use_llm=use_llm)
                events, impacts, digest = intel.analyze_symbol(
                    symbol.upper(), market=market
                )
                
                if not events:
                    st.info(f"📭 暂无 {symbol} 相关新闻")
                    return
                
                # 缓存结果
                st.session_state['news_events'] = events
                st.session_state['news_impacts'] = impacts
                st.session_state['news_digest'] = digest
                st.session_state['news_current_symbol'] = symbol.upper()
                
            except Exception as e:
                st.error(f"分析失败: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # 显示缓存的结果
    events = st.session_state.get('news_events')
    impacts = st.session_state.get('news_impacts')
    digest = st.session_state.get('news_digest')
    current_symbol = st.session_state.get('news_current_symbol', '')
    
    if events and digest:
        _render_digest_card(digest, current_symbol)
        st.divider()
        
        # 按来源统计
        sources = {}
        for e in events:
            src = e.source.split('@')[0].strip() if '@' in e.source else e.source
            sources[src] = sources.get(src, 0) + 1
        
        source_str = " · ".join([f"`{k}` ×{v}" for k, v in sorted(
            sources.items(), key=lambda x: -x[1]
        )[:5]])
        st.caption(f"📡 数据来源: {source_str}")
        
        # 新闻列表
        st.subheader(f"📋 新闻详情 ({len(events)} 条)")
        
        # 过滤器
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            event_types = list(set(e.event_type.chinese_name for e in events))
            filter_type = st.multiselect("事件类型", event_types, default=event_types,
                                         key="news_filter_type")
        with fcol2:
            sentiments = ["全部", "🐂 利好", "🐻 利空", "➖ 中性"]
            filter_sent = st.selectbox("情绪", sentiments, key="news_filter_sent")
        
        for i, (event, impact) in enumerate(zip(events, impacts)):
            # 过滤
            if event.event_type.chinese_name not in filter_type:
                continue
            if filter_sent == "🐂 利好" and event.sentiment.score <= 0:
                continue
            if filter_sent == "🐻 利空" and event.sentiment.score >= 0:
                continue
            if filter_sent == "➖ 中性" and event.sentiment.score != 0:
                continue
            
            _render_news_card(event, impact, i)


def _render_digest_card(digest, symbol: str):
    """渲染新闻摘要卡片"""
    sentiment_ratio = digest.sentiment_ratio()
    
    if sentiment_ratio > 0.3:
        color, emoji, text = "#00C853", "🟢", "看涨"
    elif sentiment_ratio < -0.3:
        color, emoji, text = "#FF1744", "🔴", "看跌"
    else:
        color, emoji, text = "#FFD600", "⚪", "中性"
    
    # 主卡片
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}15, {color}08);
                border-left: 4px solid {color}; padding: 16px; border-radius: 10px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div>
                <span style="font-size: 2em;">{emoji}</span>
            </div>
            <div style="flex: 1;">
                <h3 style="margin: 0; color: {color};">{symbol} — {text}</h3>
                <span style="color: #b0b0b0;">
                    📰 {digest.total_news_count} 条新闻 · 
                    🐂 {digest.bullish_count} 利好 · 
                    🐻 {digest.bearish_count} 利空 · 
                    ➖ {digest.neutral_count} 中性
                </span>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: {color};">
                    {digest.avg_expected_impact:+.1f}%
                </div>
                <div style="font-size: 0.8em; color: #8b949e;">预期影响</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if digest.key_events:
        st.info("🔑 **关键事件:** " + " | ".join(digest.key_events[:3]))


def _render_news_card(event, impact, index: int):
    """渲染单条新闻卡片"""
    
    sentiment_icons = {
        Sentiment.VERY_BULLISH: "🔥",
        Sentiment.BULLISH: "📈",
        Sentiment.NEUTRAL: "➖",
        Sentiment.BEARISH: "📉",
        Sentiment.VERY_BEARISH: "💥",
    }
    icon = sentiment_icons.get(event.sentiment, "➖")
    
    # 来源标记
    source_badges = {
        'StockTwits': '💬',
        'Yahoo': '📰',
        'Google': '🔍',
        'Finnhub': '📊',
        'Polygon': '🔷',
    }
    src_badge = "📰"
    for key, badge in source_badges.items():
        if key.lower() in event.source.lower():
            src_badge = badge
            break
    
    title_display = event.title[:80] + ('...' if len(event.title) > 80 else '')
    
    with st.expander(
        f"{icon} {src_badge} **{title_display}**",
        expanded=(index == 0)
    ):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
**📌 类型:** {event.event_type.chinese_name}

**📅 时间:** {event.published_at.strftime('%Y-%m-%d %H:%M') if event.published_at else 'N/A'}

**📰 来源:** {event.source}
            """)
            
            if event.summary and event.summary != event.title:
                st.caption(f"📝 {event.summary[:200]}")
            
            if event.url:
                st.markdown(f"🔗 [查看原文]({event.url})")
        
        with col2:
            impact_color = "#00C853" if impact.expected_impact_pct > 0 else (
                "#FF1744" if impact.expected_impact_pct < 0 else "#FFD600"
            )
            st.markdown(f"""
<div style="text-align: center; background: {impact_color}15; 
            padding: 12px; border-radius: 8px;">
    <div style="font-size: 1.5em; font-weight: bold; color: {impact_color};">
        {impact.expected_impact_pct:+.1f}%
    </div>
    <div style="font-size: 0.8em; color: #8b949e;">预期影响</div>
    <div style="margin-top: 4px;">置信度: {impact.confidence:.0f}%</div>
    <div>紧急度: {'🔥' * impact.urgency}</div>
</div>
            """, unsafe_allow_html=True)
        
        if event.keywords:
            st.markdown("**🏷️ 关键词:** " + " ".join([f"`{kw}`" for kw in event.keywords[:5]]))


def _render_social_buzz_tab():
    """社交媒体热度 — StockTwits + Reddit"""
    
    st.subheader("🔥 社交媒体热度")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("查询股票", value="NVDA", key="social_symbol")
    with col2:
        st.caption("")
        analyze = st.button("📊 分析社交热度", type="primary", key="social_btn",
                           use_container_width=True)
    
    if analyze and symbol:
        with st.spinner(f"正在分析 {symbol.upper()} 的社交媒体..."):
            try:
                crawler = get_news_crawler()
                buzz = crawler.get_social_buzz(symbol.upper(), market='US')
                
                st.session_state['social_buzz'] = buzz
            except Exception as e:
                st.error(f"分析失败: {e}")
    
    buzz = st.session_state.get('social_buzz')
    if buzz:
        _render_buzz_card(buzz)
    
    # StockTwits 趋势 (直接显示)
    st.divider()
    st.subheader("📈 StockTwits 热门")
    
    if st.button("🔄 刷新热门", key="st_trending"):
        with st.spinner("获取 StockTwits 热门..."):
            try:
                st_crawler = StockTwitsCrawler()
                trending = st_crawler.get_trending()
                st.session_state['st_trending'] = trending
            except Exception as e:
                st.error(f"获取失败: {e}")
    
    trending = st.session_state.get('st_trending')
    if trending and isinstance(trending, list):
        df = pd.DataFrame(trending)
        if not df.empty:
            st.dataframe(
                df[['symbol', 'title', 'watchlist_count']].rename(columns={
                    'symbol': '代码', 'title': '名称', 'watchlist_count': '关注数'
                }),
                use_container_width=True, hide_index=True
            )
    
    # Reddit/WSB 趋势
    st.divider()
    st.subheader("🦍 Reddit WallStreetBets 热门")
    
    if st.button("🔄 刷新 WSB", key="wsb_trending"):
        with st.spinner("获取 Reddit 热门..."):
            try:
                ape = ApeWisdomCrawler()
                wsb = ape.get_trending(filter_type="all-stocks", limit=15)
                st.session_state['wsb_trending'] = wsb
            except Exception as e:
                st.error(f"获取失败: {e}")
    
    wsb = st.session_state.get('wsb_trending')
    if wsb and isinstance(wsb, list):
        df = pd.DataFrame(wsb)
        if not df.empty:
            cols = ['rank', 'symbol', 'name', 'mentions', 'upvotes']
            cols = [c for c in cols if c in df.columns]
            st.dataframe(
                df[cols].rename(columns={
                    'rank': '排名', 'symbol': '代码', 'name': '名称',
                    'mentions': '提及次数', 'upvotes': '点赞数'
                }),
                use_container_width=True, hide_index=True
            )


def _render_buzz_card(buzz: Dict):
    """渲染社交热度卡片"""
    
    symbol = buzz.get('symbol', '')
    total_score = buzz.get('total_buzz_score', 0)
    
    # 热度等级
    if total_score > 100:
        heat = "🔥🔥🔥 极高"
        heat_color = "#FF1744"
    elif total_score > 50:
        heat = "🔥🔥 高"
        heat_color = "#FF9100"
    elif total_score > 20:
        heat = "🔥 中等"
        heat_color = "#FFD600"
    else:
        heat = "❄️ 低"
        heat_color = "#4FC3F7"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {heat_color}15, {heat_color}08);
                border-left: 4px solid {heat_color}; padding: 16px; border-radius: 10px;">
        <h3 style="margin: 0; color: {heat_color};">
            {symbol} 社交热度: {heat}
        </h3>
        <span style="color: #8b949e;">综合评分: {total_score}</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # StockTwits
    st_data = buzz.get('stocktwits_sentiment')
    with col1:
        st.markdown("**💬 StockTwits**")
        if st_data:
            ratio = st_data.get('ratio', 0)
            ratio_color = "#00C853" if ratio > 0 else ("#FF1744" if ratio < 0 else "#FFD600")
            st.metric("讨论数", st_data.get('total', 0))
            st.metric("🐂 看多", st_data.get('bullish', 0))
            st.metric("🐻 看空", st_data.get('bearish', 0))
        else:
            st.caption("暂无数据")
    
    # Reddit
    with col2:
        st.markdown("**🦍 Reddit**")
        rank = buzz.get('reddit_rank')
        mentions = buzz.get('reddit_mentions', 0)
        if rank:
            st.metric("WSB 排名", f"#{rank}")
            st.metric("提及次数", mentions)
        else:
            st.caption("未进入热榜")
    
    # Finnhub
    fh = buzz.get('finnhub_social')
    with col3:
        st.markdown("**📊 Finnhub 社交**")
        if fh:
            st.metric("Reddit 提及 (7天)", fh.get('reddit_mentions_7d', 0))
            st.metric("Twitter 提及 (7天)", fh.get('twitter_mentions_7d', 0))
        else:
            st.caption("需要 Finnhub API key")


def _render_portfolio_news_tab():
    """持仓新闻分析"""
    
    st.subheader("📊 持仓新闻分析")
    
    default_portfolio = "NVDA, AAPL, MSFT, GOOGL, TSLA"
    portfolio_input = st.text_input(
        "输入持仓代码 (逗号分隔)", value=default_portfolio,
        key="portfolio_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="portfolio_market")
    with col2:
        use_llm = st.checkbox("🧠 AI 分类", value=True, key="portfolio_llm")
    
    if st.button("📊 分析持仓", type="primary", use_container_width=True, key="portfolio_btn"):
        symbols = [s.strip().upper() for s in portfolio_input.split(",") if s.strip()]
        
        if not symbols:
            st.warning("请输入股票代码")
            return
        
        intel = get_news_intelligence(use_llm=use_llm)
        progress = st.progress(0)
        all_digests = []
        all_alerts = []
        
        for i, symbol in enumerate(symbols):
            progress.progress((i + 1) / len(symbols), text=f"分析 {symbol}...")
            try:
                events, impacts, digest = intel.analyze_symbol(symbol, market=market)
                all_digests.append((symbol, digest))
                
                for event, impact in zip(events, impacts):
                    if impact.should_alert:
                        all_alerts.append({
                            'symbol': symbol,
                            'title': event.title[:60],
                            'type': event.event_type.chinese_name,
                            'sentiment': event.sentiment.emoji,
                            'impact': impact.expected_impact_pct,
                        })
            except Exception as e:
                st.warning(f"⚠️ {symbol}: {e}")
        
        progress.empty()
        
        if all_digests:
            st.subheader("📋 持仓新闻摘要")
            
            rows = []
            for symbol, digest in all_digests:
                ratio = digest.sentiment_ratio()
                emoji = "🟢" if ratio > 0.3 else ("🔴" if ratio < -0.3 else "⚪")
                rows.append({
                    '股票': symbol,
                    '情绪': emoji,
                    '新闻数': digest.total_news_count,
                    '利好': digest.bullish_count,
                    '利空': digest.bearish_count,
                    '预期影响': f"{digest.avg_expected_impact:+.1f}%",
                    '信号调整': f"{digest.signal_adjustment:.2f}x",
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        if all_alerts:
            st.subheader(f"🚨 需要关注 ({len(all_alerts)} 条)")
            for alert in all_alerts[:10]:
                impact = alert['impact']
                icon = "📈" if impact > 0 else "📉"
                st.markdown(
                    f"**{alert['symbol']}** {icon} {alert['type']} {alert['sentiment']} "
                    f"— {alert['title']} ({impact:+.1f}%)"
                )


def _render_trending_tab():
    """趋势发现 — 数据源状态"""
    
    st.subheader("📈 趋势发现")
    
    # 数据源状态
    st.markdown("### 📡 数据源状态")
    
    import os
    sources = {
        'Google News RSS': ('✅ 可用', '免费', '全球'),
        'yfinance News': ('✅ 可用', '免费', '美股'),
        'StockTwits API': ('✅ 可用', '免费', '美股'),
        'ApeWisdom (Reddit/WSB)': ('✅ 可用', '免费', '美股'),
        'Finnhub': (
            '✅ 可用' if os.getenv('FINNHUB_API_KEY') else '⚠️ 需要 API Key',
            '免费 (60次/分)', '全球'
        ),
        'Polygon': (
            '✅ 可用' if os.getenv('POLYGON_API_KEY') else '⚠️ 需要 API Key',
            '免费 (5次/分)', '美股'
        ),
        'Gemini AI 分类': (
            '✅ 可用' if (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')) else '⚠️ 需要 API Key',
            '免费额度', '全球'
        ),
    }
    
    rows = []
    for name, (status, cost, market) in sources.items():
        rows.append({
            '数据源': name,
            '状态': status,
            '费用': cost,
            '覆盖': market,
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # API Key 配置提示
    st.markdown("### 🔑 配置提示")
    st.info("""
**可选 API Key 配置 (免费注册)：**
- `FINNHUB_API_KEY` — [注册 Finnhub](https://finnhub.io/) → 60 次/分钟免费额度
- `POLYGON_API_KEY` — [注册 Polygon](https://polygon.io/) → 5 次/分钟免费额度
- `GEMINI_API_KEY` — [注册 Google AI](https://aistudio.google.com/) → 免费额度
    
设置为环境变量或 Streamlit secrets 即可。
    """)
    
    # 快速新闻测试
    st.divider()
    st.markdown("### 🧪 快速测试")
    test_symbol = st.text_input("测试代码", value="AAPL", key="test_symbol")
    if st.button("🧪 测试所有源", key="test_sources"):
        with st.spinner("测试中..."):
            crawler = get_news_crawler()
            
            # Google
            gn = crawler.google.crawl(test_symbol, max_results=2)
            st.write(f"Google News: {len(gn)} 条")
            
            # yfinance
            yf_n = crawler.yfinance.crawl(test_symbol, max_results=2)
            st.write(f"yfinance: {len(yf_n)} 条")
            
            # StockTwits
            st_n = crawler.stocktwits.crawl(test_symbol, max_results=2)
            st.write(f"StockTwits: {len(st_n)} 条")
            
            # ApeWisdom
            ape = crawler.apewisdom.get_symbol_mentions(test_symbol)
            st.write(f"ApeWisdom: {'找到' if ape else '未在热榜'}")
            
            # Finnhub
            if crawler.finnhub.is_available:
                fh = crawler.finnhub.crawl(test_symbol, max_results=2)
                st.write(f"Finnhub: {len(fh)} 条")
            else:
                st.write("Finnhub: ⚠️ 无 key")
            
            # Polygon
            if crawler.polygon.is_available:
                pg = crawler.polygon.crawl(test_symbol, max_results=2)
                st.write(f"Polygon: {len(pg)} 条")
            else:
                st.write("Polygon: ⚠️ 无 key")


# 导出
def get_news_center_page():
    return render_news_center_page
