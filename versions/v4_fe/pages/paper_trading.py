"""
多策略模拟盘总览 - Multi-Strategy Paper Trading Dashboard
=========================================================
显示 6 个策略的模拟盘盈亏状态 + 最优策略 + Alpaca 同步状态
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    if hasattr(st, "secrets"):
        for key in st.secrets:
            value = st.secrets[key]
            if isinstance(value, str) and (key not in os.environ or not os.environ[key]):
                os.environ[key] = value
except:
    pass

st.set_page_config(page_title="策略模拟盘", page_icon="🏆", layout="wide")

from scripts.paper_trading import (
    STRATEGIES, get_paper_nav_history, get_paper_positions,
    get_best_strategy, INITIAL_CAPITAL, DB_PATH
)

st.title("🏆 多策略模拟盘")

# ============================================================================
# 市场选择
# ============================================================================
market = st.radio("", ["US", "CN"], horizontal=True, label_visibility="collapsed")

# ============================================================================
# 策略对比概览
# ============================================================================
st.subheader("📊 策略盈亏对比")

best_key = get_best_strategy(market)
has_data = False

# Build comparison data
import sqlite3
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)

    rows = []
    for sk, strat in STRATEGIES.items():
        try:
            nav_row = conn.execute(
                'SELECT total_nav, n_positions, date FROM paper_daily_nav_v2 WHERE market=? AND strategy_key=? ORDER BY date DESC LIMIT 1',
                (market, sk)
            ).fetchone()
        except:
            nav_row = None

        if nav_row:
            has_data = True
            nav, n_pos, last_date = nav_row
            ret = (nav / INITIAL_CAPITAL - 1) * 100

            # Trade stats
            try:
                trades = conn.execute(
                    "SELECT COUNT(*), AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END), SUM(pnl) FROM paper_positions_v2 WHERE market=? AND strategy_key=? AND status='closed'",
                    (market, sk)
                ).fetchone()
                n_trades = trades[0] or 0
                win_rate = (trades[1] or 0) * 100
                total_pnl = trades[2] or 0
            except:
                n_trades, win_rate, total_pnl = 0, 0, 0

            # Max drawdown
            try:
                navs = pd.read_sql(
                    'SELECT total_nav FROM paper_daily_nav_v2 WHERE market=? AND strategy_key=? ORDER BY date',
                    conn, params=[market, sk])
                if not navs.empty:
                    peak = navs['total_nav'].expanding().max()
                    dd = (peak - navs['total_nav']) / peak * 100
                    max_dd = dd.max()
                else:
                    max_dd = 0
            except:
                max_dd = 0

            rows.append({
                'key': sk,
                '策略': strat['name'],
                'NAV': nav,
                '收益率': ret,
                '最大回撤': max_dd,
                '胜率': win_rate,
                '交易数': n_trades,
                '持仓数': n_pos,
                '最后更新': last_date,
                'is_best': sk == best_key,
            })

    conn.close()

if not has_data:
    st.info("📋 暂无模拟盘数据。策略会在每日 Actions 运行后自动生成数据。")
    st.caption("6 个策略将独立运行：全仓买入、每日Top1、每日Top3、前10%精选、连续信号、大盘精选")
else:
    # Strategy metrics cards
    cols = st.columns(3)
    for i, row in enumerate(sorted(rows, key=lambda x: x['收益率'], reverse=True)):
        with cols[i % 3]:
            is_best = row['is_best']
            ret = row['收益率']
            color = "🟢" if ret >= 0 else "🔴"
            badge = " ⭐ Alpaca" if is_best else ""

            st.markdown(f"""
            <div style="
                background: {'linear-gradient(135deg, #1a3a2a, #0d1f17)' if ret >= 0 else 'linear-gradient(135deg, #3a1a1a, #1f0d0d)'};
                border: 1px solid {'#2d5a3d' if ret >= 0 else '#5a2d2d'};
                border-radius: 12px; padding: 16px; margin-bottom: 12px;
                {'box-shadow: 0 0 12px rgba(255,215,0,0.3); border-color: gold;' if is_best else ''}
            ">
                <div style="font-size: 14px; color: #aaa;">
                    {color} {row['策略']}{badge}
                </div>
                <div style="font-size: 28px; font-weight: bold; color: {'#4ade80' if ret >= 0 else '#f87171'};">
                    {ret:+.1f}%
                </div>
                <div style="font-size: 13px; color: #888;">
                    NAV ${row['NAV']:,.0f} | DD {row['最大回撤']:.1f}% | WR {row['胜率']:.0f}% | {row['交易数']}笔
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================================
    # NAV 曲线对比
    # ============================================================================
    st.subheader("📈 策略净值曲线")

    all_nav = get_paper_nav_history(market)
    if not all_nav.empty and 'strategy_key' in all_nav.columns:
        fig = go.Figure()

        colors = {
            'all_in': '#8b5cf6',
            'top1_daily': '#f59e0b',
            'top3_daily': '#10b981',
            'top10pct': '#3b82f6',
            'streak_only': '#ef4444',
            'large_cap': '#ec4899',
        }

        for sk in STRATEGIES:
            sk_data = all_nav[all_nav['strategy_key'] == sk]
            if sk_data.empty:
                continue
            name = STRATEGIES[sk]['name']
            is_best = sk == best_key
            fig.add_trace(go.Scatter(
                x=sk_data['date'],
                y=sk_data['total_nav'],
                name=f"{'⭐ ' if is_best else ''}{name}",
                line=dict(
                    color=colors.get(sk, '#888'),
                    width=3 if is_best else 1.5,
                    dash='solid' if is_best else None,
                ),
            ))

        # Baseline
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray",
                      annotation_text="$100K", annotation_position="right")

        fig.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation='h', y=-0.15),
            yaxis_title='NAV ($)',
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # 策略详情 (展开式)
    # ============================================================================
    st.subheader("📋 策略详情")

    selected_key = st.selectbox(
        "选择策略查看详情",
        list(STRATEGIES.keys()),
        format_func=lambda x: f"{'⭐ ' if x == best_key else ''}{STRATEGIES[x]['name']} — {STRATEGIES[x]['desc']}",
        index=list(STRATEGIES.keys()).index(best_key) if best_key in STRATEGIES else 0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📦 当前持仓**")
        open_pos = get_paper_positions(market, selected_key, 'open')
        if open_pos.empty:
            st.caption("暂无持仓")
        else:
            display = open_pos[['symbol', 'buy_date', 'buy_price', 'shares', 'amount']].copy()
            display.columns = ['股票', '买入日', '买入价', '股数', '金额']
            display['买入价'] = display['买入价'].apply(lambda x: f"${x:.2f}")
            display['股数'] = display['股数'].apply(lambda x: f"{x:.1f}")
            display['金额'] = display['金额'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**📝 最近交易**")
        closed_pos = get_paper_positions(market, selected_key, 'closed')
        if closed_pos.empty:
            st.caption("暂无已平仓记录")
        else:
            recent = closed_pos.head(10)
            display = recent[['symbol', 'buy_date', 'sell_date', 'buy_price', 'sell_price', 'pnl_pct']].copy()
            display.columns = ['股票', '买入', '卖出', '买价', '卖价', '盈亏%']
            display['买价'] = display['买价'].apply(lambda x: f"${x:.2f}")
            display['卖价'] = display['卖价'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
            display['盈亏%'] = display['盈亏%'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
            st.dataframe(display, use_container_width=True, hide_index=True)

# ============================================================================
# Alpaca 同步状态
# ============================================================================
st.markdown("---")
st.subheader("🔗 Alpaca 模拟盘")

try:
    from execution.alpaca_trader import AlpacaTrader, ALPACA_SDK_AVAILABLE

    def _resolve_keys():
        api = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if api and secret:
            return api, secret
        try:
            if hasattr(st, "secrets"):
                api = api or st.secrets.get("ALPACA_API_KEY")
                secret = secret or st.secrets.get("ALPACA_SECRET_KEY")
                g = st.secrets.get("alpaca")
                if isinstance(g, dict):
                    api = api or g.get("api_key")
                    secret = secret or g.get("secret_key")
        except:
            pass
        return api, secret

    if ALPACA_SDK_AVAILABLE:
        api_key, secret_key = _resolve_keys()
        if api_key and secret_key:
            trader = AlpacaTrader(api_key=api_key, secret_key=secret_key, paper=True)
            account = trader.get_account()
            positions = trader.get_positions()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Alpaca 净值", f"${account.equity:,.0f}")
            c2.metric("现金", f"${account.cash:,.0f}")
            ret_pct = (account.equity / 100_000 - 1) * 100
            c3.metric("总收益", f"{ret_pct:+.1f}%")
            c4.metric("持仓数", f"{len(positions)}")

            if positions:
                pos_data = [{
                    '股票': p.symbol,
                    '数量': int(p.qty),
                    '成本': f"${p.avg_entry_price:.2f}",
                    '现价': f"${p.current_price:.2f}",
                    '盈亏': f"${p.unrealized_pl:+,.0f}",
                    '盈亏%': f"{p.unrealized_plpc:+.1f}%"
                } for p in positions]
                st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

            st.caption(f"⭐ 当前由 **{STRATEGIES.get(best_key, {}).get('name', best_key)}** 策略驱动 | "
                       f"{'🟢 开盘中' if trader.get_market_hours()['is_open'] else '🔴 休市'}")
        else:
            st.caption("⚠️ 未配置 Alpaca API Keys (ALPACA_API_KEY / ALPACA_SECRET_KEY)")
    else:
        st.caption("⚠️ 未安装 alpaca-py (`pip install alpaca-py`)")
except ImportError:
    st.caption("⚠️ Alpaca SDK 未安装")

st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
