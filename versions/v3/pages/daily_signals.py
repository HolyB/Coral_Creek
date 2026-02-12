#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日买卖点推荐
生成并显示今日的买入/卖出信号
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def render_daily_signals_page():
    """每日买卖信号页面"""
    st.header("🎯 每日买卖点 (Daily Signals)")
    st.caption("基于多策略分析的每日买入/卖出建议")
    
    # 导入信号系统
    try:
        from strategies.signal_system import get_signal_manager
        from db.database import query_scan_results, get_scanned_dates, get_stock_info_batch
    except ImportError as e:
        st.error(f"模块导入失败: {e}")
        return
    
    manager = get_signal_manager()
    
    # 侧边栏设置
    try:
        with st.sidebar:
            st.subheader("🎯 买卖点设置")
            
            market_choice = st.radio("市场", ["🇺🇸 美股", "🇨🇳 A股"], horizontal=True, key="daily_sig_market")
            market = "US" if "美股" in market_choice else "CN"
            
            min_confidence = st.slider("最低信心度", 30, 90, 50, help="过滤信心度低的信号")
            
            signal_type = st.selectbox("信号类型", ["全部", "仅买入", "仅卖出"])
            
            if st.button("🔄 刷新信号", type="primary", width='stretch'):
                with st.spinner("生成交易信号..."):
                    result = manager.generate_daily_signals(market=market)
                    if 'error' not in result:
                        st.success(f"✅ 已生成 {result.get('buy_signals', 0)} 个买入, {result.get('sell_signals', 0)} 个卖出信号")
                    else:
                        st.error(result['error'])
        
        # 获取信号
        today = datetime.now().strftime('%Y-%m-%d')
        signals = manager.get_todays_signals(market=market)
        display_date = today

        # 如果今天没信号，尝试获取最近一次的信号（防止时区问题导致看不到刚才生成的）
        if not signals:
            hist_signals = manager.get_historical_signals(days=5, market=market)
            if hist_signals:
                # 取最近一天的
                latest_date = hist_signals[0]['generated_at']
                signals = [s for s in hist_signals if s['generated_at'] == latest_date]
                display_date = latest_date
                st.warning(f"⚠️ 未找到 {today} 的信号，显示最近一次 ({latest_date}) 的记录")
        
        # 过滤
        if signals:
            total_count = len(signals)
            signals = [s for s in signals if s.get('confidence', 0) >= min_confidence]
            
            if signal_type == "仅买入":
                signals = [s for s in signals if s['signal_type'] == '买入']
            elif signal_type == "仅卖出":
                signals = [s for s in signals if s['signal_type'] in ['卖出', '止损', '止盈']]
            
            st.caption(f"📅 信号日期: {display_date} | 🔍 找到 {total_count} 个信号 (过滤后: {len(signals)}个)")

        if not signals:
            st.info(f"📅 {display_date} 暂无符合条件的信号 (信心度>={min_confidence}%)")
            st.info("💡 请点击侧边栏「🔄 刷新信号」按钮生成今日最新信号")
            
            with st.expander("📖 信号说明"):
                st.markdown("""
                ### 买入信号条件
                
                | 信号名称 | 条件 | 信心度 |
                |----------|------|--------|
                | BLUE突破 | BLUE > 180 | 70-95% |
                | 趋势确认 | BLUE 150-180 + ADX > 25 | 50-70% |
                | 黑马形态 | is_heima = True | 55% |
                | 绝地反击 | is_juedi = True | 45% |
                | 量价齐升 | 成交 > 50M + BLUE 120-160 | 55-75% |
                
                ### 卖出信号条件
                
                | 信号名称 | 条件 |
                |----------|------|
                | 止损 | 价格 <= 止损位 |
                | 止盈 | 价格 >= 目标价 |
                | 超时 | 持仓 > 10天且盈利 < 5% |
                """)
            return
        
        # 分类显示
        buy_signals = [s for s in signals if s['signal_type'] == '买入']
        sell_signals = [s for s in signals if s['signal_type'] != '买入']
        
        # 优化：只获取要显示的股票的信息 (避免一次性查询上千个 symbol 导致 SQLite 报错)
        display_buy = buy_signals[:10]
        display_sell = sell_signals[:10]
        display_symbols = list(set([s['symbol'] for s in display_buy + display_sell]))
        stock_info = get_stock_info_batch(display_symbols) if display_symbols else {}

        price_sym = "¥" if market == "CN" else "$"
        
        # === 买入信号 ===
        st.subheader(f"🟢 买入信号 ({len(buy_signals)}个)")
        
        if buy_signals:
            for i, sig in enumerate(buy_signals[:10]):
                info = stock_info.get(sig['symbol'], {})
                name = info.get('name', '')[:15] if info else ''
                
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                    
                    with col1:
                        strength_icon = "🔥" if sig['strength'] == '强烈' else "⚡" if sig['strength'] == '中等' else "💧"
                        st.markdown(f"**{sig['symbol']}** {name}")
                        st.caption(f"{strength_icon} {sig['strength']} | {sig['strategy']}")
                    
                    with col2:
                        st.metric("价格", f"{price_sym}{sig['price']:.2f}")
                    
                    with col3:
                        st.metric("信心", f"{sig['confidence']:.0f}%")
                    
                    with col4:
                        target = sig.get('target_price', 0)
                        stop = sig.get('stop_loss', 0)
                        if target and stop:
                            upside = (target / sig['price'] - 1) * 100
                            downside = (1 - stop / sig['price']) * 100
                            st.markdown(f"🎯 目标: {price_sym}{target:.2f} (+{upside:.0f}%)")
                            st.markdown(f"🛑 止损: {price_sym}{stop:.2f} (-{downside:.0f}%)")
                    
                    st.caption(f"💡 {sig['reason']}")
                    st.divider()
        else:
            st.info("暂无买入信号")
        
        # === 卖出信号 ===
        st.subheader(f"🔴 卖出/止损信号 ({len(sell_signals)}个)")
        
        if sell_signals:
            for sig in sell_signals[:10]:
                info = stock_info.get(sig['symbol'], {})
                name = info.get('name', '')[:15] if info else ''
                
                signal_icon = {
                    '止损': '🛑',
                    '止盈': '🎯',
                    '卖出': '🔴'
                }.get(sig['signal_type'], '📊')
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 3])
                    
                    with col1:
                        st.markdown(f"**{sig['symbol']}** {name}")
                        st.caption(f"{signal_icon} {sig['signal_type']}")
                    
                    with col2:
                        st.metric("价格", f"{price_sym}{sig['price']:.2f}")
                    
                    with col3:
                        st.markdown(f"⚠️ {sig['reason']}")
                    
                    st.divider()
        else:
            st.info("暂无卖出信号")
        
        # === 统计摘要 ===
        st.subheader("📊 今日信号统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("买入信号", len(buy_signals))
        
        with col2:
            st.metric("卖出信号", len(sell_signals))
        
        with col3:
            avg_conf = sum([s['confidence'] for s in buy_signals]) / len(buy_signals) if buy_signals else 0
            st.metric("平均信心度", f"{avg_conf:.0f}%")
        
        with col4:
            strong_count = len([s for s in buy_signals if s['strength'] == '强烈'])
            st.metric("强烈信号", strong_count)
            
    except Exception as e:
        import traceback
        st.error(f"❌ 运行错误: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    render_daily_signals_page()
