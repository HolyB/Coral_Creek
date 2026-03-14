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
            
            if st.button("🔄 刷新信号", type="primary", use_container_width=True):
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
            
        # === ML 模型推荐 ===
        st.subheader("🤖 ML 每日机会")
        
        try:
            from scripts.ml_daily_scorer import get_historical_picks
            
            # Load historical picks from v2 table
            hist_df = get_historical_picks(market=market, days=60)
            
            if hist_df.empty:
                st.info("📦 暂无 ML 推荐数据。请运行回填: `PYTHONPATH=. python scripts/backfill_daily_picks.py`")
            else:
                # Get latest date's picks
                latest_date = hist_df['date'].max()
                latest_picks = hist_df[hist_df['date'] == latest_date]
                
                st.caption(f"📅 最新推荐日期: {latest_date} | 市场: {'🇺🇸 美股' if market == 'US' else '🇨🇳 A股'}")
                
                if market == 'CN':
                    _render_cn_picks(latest_picks, hist_df)
                else:
                    _render_us_picks(latest_picks, hist_df)
                
                # Historical Performance Tracking
                _render_historical_performance(hist_df, market)
        
        except Exception as e:
            st.warning(f"ML 模型推荐暂不可用: {e}")
            import traceback
            st.code(traceback.format_exc())
        
    except Exception as e:
        import traceback
        st.error(f"❌ 运行错误: {str(e)}")
        st.code(traceback.format_exc())


def _render_cn_picks(latest_picks, hist_df):
    """Render CN picks with exchange tabs"""
    exchanges = ['上证主板', '深证主板', '创业板', '科创板']
    available = [ex for ex in exchanges if any(latest_picks['exchange'] == ex)]
    
    if not available:
        available = [ex for ex in exchanges if any(latest_picks['segment'].str.contains(ex))]
    
    if not available:
        st.info("暂无 A 股推荐")
        return
    
    ex_tabs = st.tabs([f"{'📊' if i==0 else '📈'} {ex}" for i, ex in enumerate(available)])
    
    for tab, exchange in zip(ex_tabs, available):
        with tab:
            # Filter picks for this exchange
            ex_picks = latest_picks[latest_picks['segment'].str.contains(exchange)]
            
            if ex_picks.empty:
                st.info(f"{exchange} 暂无推荐")
                continue
            
            # Separate "全部" picks and tier picks
            all_picks = ex_picks[ex_picks['segment'].str.contains('全部')]
            tier_picks = ex_picks[~ex_picks['segment'].str.contains('全部')]
            
            # Show overall top picks for this exchange
            if not all_picks.empty:
                st.markdown(f"**🏆 {exchange} Top Picks**")
                _render_pick_table(all_picks, market='CN')
            
            # Show by tier
            if not tier_picks.empty:
                tiers = tier_picks['tier'].unique()
                tier_tabs = st.tabs([f"💰 {t}" for t in tiers])
                for ttab, tier in zip(tier_tabs, tiers):
                    with ttab:
                        t_df = tier_picks[tier_picks['tier'] == tier]
                        _render_pick_table(t_df, market='CN')


def _render_us_picks(latest_picks, hist_df):
    """Render US picks with market cap tabs"""
    tiers = latest_picks['tier'].unique() if 'tier' in latest_picks.columns else latest_picks['segment'].unique()
    
    if len(tiers) == 0:
        st.info("暂无美股推荐")
        return
    
    tier_tabs = st.tabs([f"💰 {t}" for t in tiers])
    for tab, tier in zip(tier_tabs, tiers):
        with tab:
            if 'tier' in latest_picks.columns:
                t_df = latest_picks[latest_picks['tier'] == tier]
            else:
                t_df = latest_picks[latest_picks['segment'] == tier]
            _render_pick_table(t_df, market='US')


def _render_pick_table(picks_df, market='US'):
    """Render a styled picks table"""
    if picks_df.empty:
        st.info("暂无数据")
        return
    
    price_sym = "¥" if market == 'CN' else "$"
    pred_col = 'pred_30d' if market == 'CN' else 'pred_10d'
    pred_label = '预测30d' if market == 'CN' else '预测10d'
    actual_col = 'actual_30d' if market == 'CN' else 'actual_10d'
    
    display_df = picks_df.copy()
    
    # Format columns
    display_df['排名'] = display_df['rank'].astype(int)
    display_df['代码'] = display_df['symbol']
    display_df['价格'] = display_df['price'].apply(lambda x: f"{price_sym}{x:.2f}")
    
    if pred_col in display_df.columns:
        display_df[pred_label] = display_df[pred_col].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    
    if 'market_cap' in display_df.columns:
        if market == 'CN':
            display_df['市值'] = display_df['market_cap'].apply(
                lambda x: f"¥{x/1e8:.0f}亿" if x >= 1e8 else "—")
        else:
            display_df['市值'] = display_df['market_cap'].apply(
                lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M" if x > 0 else "—")
    
    if actual_col in display_df.columns and display_df[actual_col].notna().any():
        display_df['实际收益'] = display_df[actual_col].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "⏳")
        display_df['结果'] = display_df[actual_col].apply(
            lambda x: "✅" if pd.notna(x) and x > 0 else ("❌" if pd.notna(x) else "⏳"))
    
    show_cols = ['排名', '代码', '价格', pred_label]
    if '市值' in display_df.columns:
        show_cols.append('市值')
    if '实际收益' in display_df.columns:
        show_cols.extend(['实际收益', '结果'])
    
    show_df = display_df[[c for c in show_cols if c in display_df.columns]]
    show_df = show_df.reset_index(drop=True)
    
    st.dataframe(show_df, use_container_width=True, hide_index=True)


def _render_historical_performance(hist_df, market):
    """Render historical performance tracking"""
    with st.expander("📈 历史推荐表现追踪"):
        actual_col = 'actual_30d' if market == 'CN' else 'actual_10d'
        pred_col = 'pred_30d' if market == 'CN' else 'pred_10d'
        period = '30d' if market == 'CN' else '10d'
        
        if actual_col not in hist_df.columns or hist_df[actual_col].isna().all():
            st.info("⏳ 暂无已验证的推荐（需要等待持仓期结束）")
            # Still show recent picks
            recent = hist_df.sort_values('date', ascending=False).head(30)
            show_cols = ['date', 'segment', 'rank', 'symbol', 'price', pred_col]
            show_cols = [c for c in show_cols if c in recent.columns]
            st.dataframe(recent[show_cols], use_container_width=True, hide_index=True)
            return
        
        verified = hist_df[hist_df[actual_col].notna()].copy()
        if verified.empty:
            st.info("⏳ 暂无已验证的推荐")
            return
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total = len(verified)
            wins = (verified[actual_col] > 0).sum()
            st.metric("总推荐数", total)
        with col2:
            win_rate = wins / total * 100 if total > 0 else 0
            st.metric("胜率", f"{win_rate:.0f}%")
        with col3:
            avg_ret = verified[actual_col].mean()
            st.metric(f"平均{period}收益", f"{avg_ret:+.1f}%")
        with col4:
            median_ret = verified[actual_col].median()
            st.metric(f"中位{period}收益", f"{median_ret:+.1f}%")
        
        # Top-1 vs Top-3 performance
        if 'rank' in verified.columns:
            st.markdown("**📊 按排名表现**")
            for rank_filter, label in [(1, 'Top-1'), (3, 'Top-3')]:
                rank_df = verified[verified['rank'] <= rank_filter]
                if not rank_df.empty:
                    wr = (rank_df[actual_col] > 0).sum() / len(rank_df) * 100
                    avg = rank_df[actual_col].mean()
                    st.caption(f"{label}: 胜率 {wr:.0f}%, 平均收益 {avg:+.1f}% (n={len(rank_df)})")
        
        # By segment performance
        if 'segment' in verified.columns:
            st.markdown("**📊 按板块/市值表现**")
            seg_stats = verified.groupby('segment').agg({
                actual_col: ['count', 'mean', lambda x: (x > 0).mean() * 100]
            }).round(1)
            seg_stats.columns = ['推荐次数', f'平均{period}收益%', '胜率%']
            seg_stats = seg_stats.sort_values(f'平均{period}收益%', ascending=False)
            st.dataframe(seg_stats, use_container_width=True)


if __name__ == "__main__":
    render_daily_signals_page()

