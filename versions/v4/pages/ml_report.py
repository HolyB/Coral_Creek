#!/usr/bin/env python3
"""Coral Creek Way — ML 每日选股报告页面"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

V4 = Path(__file__).resolve().parent.parent
DB_PICKS = V4 / 'db' / 'ml_daily_picks.db'
DB_HIST  = V4 / 'db' / 'stock_history.db'

# Tier definitions
US_TIERS = {
    'Mega (>$100B)':       {'color': '#f59e0b', 'order': 0},
    'Large ($10-100B)':    {'color': '#3b82f6', 'order': 1},
    'Mid ($2-10B)':        {'color': '#8b5cf6', 'order': 2},
    'Small ($300M-2B)':    {'color': '#ec4899', 'order': 3},
    'Micro ($50-300M)':    {'color': '#06b6d4', 'order': 4},
}
CN_TIERS = {
    '大盘 (>500亿)':   {'color': '#f59e0b', 'order': 0},
    '中盘 (100-500亿)': {'color': '#8b5cf6', 'order': 1},
    '小盘 (20-100亿)':  {'color': '#ec4899', 'order': 2},
}

STRAT_DEFS = [
    ('MID_10',   ['Mid ($2-10B)'],                                     0.10, 20, 1),
    ('LARGE_10', ['Large ($10-100B)'],                                0.10, 20, 1),
    ('SMALL_10', ['Small ($300M-2B)'],                                0.10, 20, 1),
    ('ALL_3PCT', ['Large ($10-100B)', 'Mid ($2-10B)', 'Small ($300M-2B)'], 0.03, 20, 1),
    ('MID_TOP3', ['Mid ($2-10B)'],                                     0.03, 20, 3),
    ('MS_5DAY',  ['Mid ($2-10B)', 'Small ($300M-2B)'],                0.10, 5,  1),
]
STRAT_COLORS = ['#60a5fa', '#22c55e', '#f97316', '#a78bfa', '#ec4899', '#facc15']


@st.cache_data(ttl=300, show_spinner=False)
def load_picks(market):
    conn = sqlite3.connect(str(DB_PICKS))
    df = pd.read_sql(
        "SELECT * FROM mmoe_daily_picks WHERE market=? ORDER BY date DESC",
        conn, params=(market,)
    )
    conn.close()
    return df


def render_ml_report_page():
    st.header("📊 Coral Creek Way — 选股报告")

    col1, col2 = st.columns([1, 3])
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="report_market")

    df = load_picks(market)
    if df.empty:
        st.warning("暂无 ML 选股数据")
        return

    tiers_map = US_TIERS if market == 'US' else CN_TIERS
    csym = '$' if market == 'US' else '¥'
    dates = sorted(df['date'].unique(), reverse=True)

    # ===== KPI cards =====
    latest_date = dates[0]
    latest = df[df['date'] == latest_date]
    total_picks = len(df)
    ret_cols = ['actual_5d', 'actual_10d', 'actual_20d']
    overall_wr = 0
    for rc in ret_cols:
        valid = df[rc].dropna()
        if len(valid) > 0:
            overall_wr = (valid > 0).mean() * 100
            break

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 交易日数", f"{len(dates)}")
    c2.metric("📋 总选股数", f"{total_picks}")
    c3.metric("🎯 胜率 (20d)", f"{overall_wr:.0f}%")
    avg20 = df['actual_20d'].dropna().mean()
    c4.metric("📈 平均收益 (20d)", f"{avg20:+.1f}%" if not pd.isna(avg20) else "—")

    # ===== Tabs =====
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 今日选股", "📈 策略对比", "📊 Tier 分析", "📋 历史记录"])

    # --- Tab 1: Today's picks ---
    with tab1:
        st.subheader(f"📅 {latest_date} 选股")
        for _, row in latest.iterrows():
            tier = row.get('tier', '')
            tc = tiers_map.get(tier, {}).get('color', '#818cf8')
            r5  = row.get('actual_5d')
            r10 = row.get('actual_10d')
            r20 = row.get('actual_20d')

            with st.container():
                cols = st.columns([2, 1, 1, 1, 1])
                name = row.get('name', '') or ''
                sym_display = f"{row['symbol']} {name}" if name else row['symbol']
                cols[0].markdown(f"**<span style='color:{tc}'>{sym_display}</span>** — {tier}", unsafe_allow_html=True)
                cols[1].metric("买入价", f"{csym}{row.get('price', 0):.2f}")
                cols[2].metric("5d", f"{r5:+.1f}%" if pd.notna(r5) else "⏳", delta_color="normal")
                cols[3].metric("10d", f"{r10:+.1f}%" if pd.notna(r10) else "⏳", delta_color="normal")
                cols[4].metric("20d", f"{r20:+.1f}%" if pd.notna(r20) else "⏳", delta_color="normal")
            st.divider()

    # --- Tab 2: Strategy comparison ---
    with tab2:
        st.subheader("📈 策略净值模拟 (YTD)")
        st.caption("💡 收益上限 ±100% 截断，防止低价股异常值扭曲")

        strat_navs = {}
        strat_stats = {}

        trade_dates = sorted(df['date'].unique())
        for sname, tiers, pct, hold, top_n in STRAT_DEFS:
            # Simulate realistic position management:
            # - Each day open position(s) using pct of INITIAL capital (not compounding)
            # - Position closes after `hold` trading days with actual return
            # - Track cumulative P&L as additive, not multiplicative
            initial_cap = 100.0
            cumulative_pl = 0.0
            curve = []
            rets = []
            tier_df = df[df['tier'].isin(tiers)]

            for dt in trade_dates:
                day = tier_df[tier_df['date'] == dt].head(top_n)
                if not day.empty:
                    ret_col = f'actual_{hold}d' if f'actual_{hold}d' in day.columns else 'actual_20d'
                    day_rets = day[ret_col].dropna()
                    if not day_rets.empty:
                        # Each pick gets pct allocation of initial capital
                        for r in day_rets.tolist():
                            r_capped = max(-100, min(100, r))  # Cap at ±100%
                            position_size = initial_cap * pct
                            cumulative_pl += position_size * r_capped / 100
                            rets.append(r_capped)

                nav = initial_cap + cumulative_pl
                curve.append((dt, nav))

            strat_navs[sname] = curve
            if rets:
                wins = sum(1 for r in rets if r > 0)
                navs = [n[1] for n in curve]
                peak = navs[0]
                mdd = 0
                for n in navs:
                    if n > peak: peak = n
                    dd = (peak - n) / peak * 100
                    if dd > mdd: mdd = dd
                strat_stats[sname] = {
                    'wr': wins / len(rets) * 100, 'avg': np.mean(rets),
                    'mdd': mdd, 'n': len(rets), 'nav': nav
                }
            else:
                strat_stats[sname] = {'wr': 0, 'avg': 0, 'mdd': 0, 'n': 0, 'nav': 100}

        # NAV Chart
        fig = go.Figure()
        for i, (sname, _, _, _, _) in enumerate(STRAT_DEFS):
            if sname in strat_navs:
                d = [n[0] for n in strat_navs[sname]]
                v = [n[1] for n in strat_navs[sname]]
                ret = v[-1] - 100 if v else 0
                fig.add_trace(go.Scatter(
                    x=d, y=v, name=f"{sname} ({ret:+.1f}%)",
                    line=dict(color=STRAT_COLORS[i], width=2.5),
                    hovertemplate="%{x}<br>NAV: %{y:.1f}<extra></extra>"
                ))
        fig.add_hline(y=100, line_dash="dash", line_color="#64748b", opacity=0.5)
        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="策略净值曲线 (100=起始)",
            yaxis_title="NAV",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        stats_df = pd.DataFrame([
            {
                '策略': sname,
                '选股数': strat_stats.get(sname, {}).get('n', 0),
                '胜率': f"{strat_stats.get(sname, {}).get('wr', 0):.0f}%",
                '平均收益': f"{strat_stats.get(sname, {}).get('avg', 0):+.1f}%",
                '最大回撤': f"{strat_stats.get(sname, {}).get('mdd', 0):.1f}%",
                'NAV': f"{strat_stats.get(sname, {}).get('nav', 100):.1f}",
            }
            for sname, _, _, _, _ in STRAT_DEFS
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Return distribution
        st.subheader("📊 收益分布")
        fig2 = go.Figure()
        for i, (sname, _, _, _, _) in enumerate(STRAT_DEFS):
            rets = []
            tier_df = df[df['tier'].isin(STRAT_DEFS[i][1])]
            hold = STRAT_DEFS[i][3]
            rc = f'actual_{hold}d'
            if rc in tier_df.columns:
                rets = tier_df[rc].dropna().tolist()
            if rets:
                fig2.add_trace(go.Histogram(
                    x=rets, name=sname, opacity=0.6,
                    marker_color=STRAT_COLORS[i], nbinsx=25
                ))
        fig2.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig2.update_layout(
            template="plotly_dark", height=350, barmode='overlay',
            title="选股收益分布 (按持有期)", xaxis_title="Return %",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 3: Per-tier analysis ---
    with tab3:
        st.subheader("📊 市值分层表现")
        for tier, info in sorted(tiers_map.items(), key=lambda x: x[1]['order']):
            tier_df = df[df['tier'] == tier]
            if tier_df.empty:
                continue
            r20 = tier_df['actual_20d'].dropna()
            wr = (r20 > 0).mean() * 100 if len(r20) > 0 else 0
            avg = r20.mean() if len(r20) > 0 else 0

            with st.expander(f"**{tier}** — {len(tier_df)} picks | WR {wr:.0f}% | Avg {avg:+.1f}%", expanded=False):
                tier_view = tier_df.copy()
                tier_view['sym_name'] = tier_view.apply(lambda r: f"{r['symbol']} {r.get('name','') or ''}" if r.get('name') else r['symbol'], axis=1)
                display = tier_view[['date', 'sym_name', 'price', 'actual_5d', 'actual_10d', 'actual_20d']].copy()
                display.columns = ['日期', '股票', '买入价', '5d%', '10d%', '20d%']
                st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Tab 4: Full history ---
    with tab4:
        st.subheader("📋 全部选股记录")
        hist_view = df.copy()
        hist_view['sym_name'] = hist_view.apply(lambda r: f"{r['symbol']} {r.get('name','') or ''}" if r.get('name') else r['symbol'], axis=1)
        display_df = hist_view[['date', 'tier', 'sym_name', 'price', 'blend', 'actual_5d', 'actual_10d', 'actual_20d']].copy()
        display_df.columns = ['日期', 'Tier', '股票', '买入价', '综合评分', '5d%', '10d%', '20d%']
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
