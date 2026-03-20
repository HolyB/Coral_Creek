#!/usr/bin/env python3
"""Coral Creek Way — 模拟盘仪表板 (US Alpaca + CN Virtual)"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

V4 = Path(__file__).resolve().parent.parent

STRAT_NAMES = ['MID_10', 'LARGE_10', 'SMALL_10', 'ALL_3PCT', 'MID_TOP3', 'MS_5DAY']
STRAT_COLORS = ['#60a5fa', '#22c55e', '#f97316', '#a78bfa', '#ec4899', '#facc15']
STRAT_KEY_MAP = {
    'MID_10':   ('ALPACA_API_KEY',         'ALPACA_SECRET_KEY'),
    'LARGE_10': ('ALPACA_LARGE_API_KEY',   'ALPACA_LARGE_SECRET_KEY'),
    'SMALL_10': ('ALPACA_SMALL_API_KEY',   'ALPACA_SMALL_SECRET_KEY'),
    'ALL_3PCT': ('ALPACA_ALL3_API_KEY',    'ALPACA_ALL3_SECRET_KEY'),
    'MID_TOP3': ('ALPACA_MIDTOP3_API_KEY', 'ALPACA_MIDTOP3_SECRET_KEY'),
    'MS_5DAY':  ('ALPACA_MS5D_API_KEY',    'ALPACA_MS5D_SECRET_KEY'),
}


@st.cache_data(ttl=120, show_spinner="加载 Alpaca 账户...")
def get_alpaca_accounts():
    """Fetch all 6 Alpaca paper accounts"""
    import requests
    results = {}
    for sname, (key_var, secret_var) in STRAT_KEY_MAP.items():
        key = os.environ.get(key_var, '')
        secret = os.environ.get(secret_var, '')
        if not key or not secret:
            results[sname] = {'error': 'missing key'}
            continue
        try:
            headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
            acct = requests.get('https://paper-api.alpaca.markets/v2/account', headers=headers, timeout=10).json()
            positions = requests.get('https://paper-api.alpaca.markets/v2/positions', headers=headers, timeout=10).json()
            results[sname] = {
                'equity': float(acct.get('equity', 0)),
                'cash': float(acct.get('cash', 0)),
                'pl': float(acct.get('equity', 0)) - 100000,
                'positions': positions if isinstance(positions, list) else [],
                'status': acct.get('status', 'unknown'),
            }
        except Exception as e:
            results[sname] = {'error': str(e)}
    return results


@st.cache_data(ttl=120, show_spinner="加载 CN 虚拟盘...")
def get_cn_accounts():
    """Fetch CN paper trading accounts from cn_paper_trading.db"""
    db_path = V4 / 'db' / 'cn_paper_trading.db'
    if not db_path.exists():
        return {'_error': f'DB not found: {db_path}'}
    try:
        conn = sqlite3.connect(str(db_path))
        accounts = pd.read_sql("SELECT account_id, cash, created_at FROM accounts", conn)
        results = {}
        # Get latest stock prices for position valuation
        hist_db = V4 / 'db' / 'stock_history.db'
        hconn = sqlite3.connect(str(hist_db)) if hist_db.exists() else None

        for _, acc in accounts.iterrows():
            acc_id = acc['account_id']
            cash = float(acc['cash'])

            # Positions
            positions = pd.read_sql(
                "SELECT symbol, qty, avg_entry_price, opened_at FROM positions WHERE account_id=? AND qty>0",
                conn, params=(acc_id,)
            )
            pos_list = []
            pos_value = 0
            for _, p in positions.iterrows():
                sym = p['symbol']
                qty = float(p['qty'])
                entry = float(p['avg_entry_price'])
                cur_price = entry  # fallback
                if hconn:
                    row = hconn.execute(
                        "SELECT close FROM stock_history WHERE symbol=? AND market='CN' ORDER BY trade_date DESC LIMIT 1",
                        (sym,)
                    ).fetchone()
                    if row:
                        cur_price = row[0]
                pnl_pct = (cur_price / entry - 1) * 100 if entry > 0 else 0
                mv = qty * cur_price
                pos_value += mv
                pos_list.append({
                    'symbol': sym, 'qty': int(qty),
                    'avg_entry_price': entry, 'current_price': cur_price,
                    'pnl_pct': pnl_pct, 'market_value': mv,
                })

            equity = cash + pos_value

            # Equity history
            equity_hist = pd.read_sql(
                "SELECT date, equity, cash, positions_value FROM equity_history WHERE account_id=? ORDER BY date",
                conn, params=(acc_id,)
            )

            results[acc_id] = {
                'equity': equity,
                'cash': cash,
                'pl': equity - 100000,
                'positions': pos_list,
                'equity_history': equity_hist.to_dict('records') if not equity_hist.empty else [],
            }
        if hconn:
            hconn.close()
        conn.close()
        return results
    except Exception as e:
        return {'_error': str(e)}


def render_paper_trading_page():
    st.header("💰 模拟盘仪表板")

    tab_us, tab_cn = st.tabs(["🇺🇸 US Alpaca", "🇨🇳 CN 虚拟盘"])

    # ==================== US Tab ====================
    with tab_us:
        st.subheader("🇺🇸 Alpaca Paper Trading — 6 策略")

        accounts = get_alpaca_accounts()

        # Summary KPIs
        total_equity = sum(a.get('equity', 0) for a in accounts.values() if 'error' not in a)
        total_pl = sum(a.get('pl', 0) for a in accounts.values() if 'error' not in a)
        total_positions = sum(len(a.get('positions', [])) for a in accounts.values() if 'error' not in a)
        active = sum(1 for a in accounts.values() if 'error' not in a)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 总权益", f"${total_equity:,.0f}")
        c2.metric("📈 总盈亏", f"${total_pl:+,.0f}", delta_color="normal")
        c3.metric("📊 总持仓", f"{total_positions}")
        c4.metric("✅ 活跃账户", f"{active}/6")

        st.divider()

        # Per-strategy cards
        for i, sname in enumerate(STRAT_NAMES):
            data = accounts.get(sname, {})
            if 'error' in data:
                st.error(f"**{sname}**: {data['error']}")
                continue

            equity = data.get('equity', 0)
            cash = data.get('cash', 0)
            pl = data.get('pl', 0)
            positions = data.get('positions', [])
            pl_color = "🟢" if pl >= 0 else "🔴"

            with st.expander(
                f"**{sname}** {pl_color} ${equity:,.0f} | P&L: ${pl:+,.0f} | {len(positions)} 持仓",
                expanded=len(positions) > 0
            ):
                if positions:
                    rows = []
                    for p in positions:
                        unrealized = float(p.get('unrealized_pl', 0))
                        pct = float(p.get('unrealized_plpc', 0)) * 100
                        rows.append({
                            '股票': p.get('symbol', ''),
                            '数量': int(float(p.get('qty', 0))),
                            '成本': f"${float(p.get('avg_entry_price', 0)):.2f}",
                            '现价': f"${float(p.get('current_price', 0)):.2f}",
                            '涨跌': f"{pct:+.1f}%",
                            '盈亏': f"${unrealized:+,.0f}",
                            '市值': f"${float(p.get('market_value', 0)):,.0f}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("无持仓")

                col_a, col_b = st.columns(2)
                col_a.metric("权益", f"${equity:,.0f}")
                col_b.metric("现金", f"${cash:,.0f}")

        # Allocation pie chart
        st.subheader("📊 策略配置")
        equities = []
        labels = []
        for sname in STRAT_NAMES:
            data = accounts.get(sname, {})
            if 'error' not in data:
                equities.append(data.get('equity', 0))
                labels.append(sname)

        if equities:
            fig = go.Figure(go.Pie(
                labels=labels, values=equities,
                marker=dict(colors=STRAT_COLORS[:len(labels)]),
                textinfo='label+percent', textposition='inside',
                hole=0.4
            ))
            fig.update_layout(
                template="plotly_dark", height=350,
                title="策略权益分布",
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==================== CN Tab ====================
    with tab_cn:
        st.subheader("🇨🇳 CN 虚拟盘")
        cn_data = get_cn_accounts()

        if not cn_data or '_error' in cn_data:
            err = cn_data.get('_error', '未知错误') if cn_data else '无数据'
            st.warning(f"CN 虚拟盘: {err}")
            return

        # CN summary KPIs
        cn_equity = sum(d.get('equity', 0) for d in cn_data.values())
        cn_pl = sum(d.get('pl', 0) for d in cn_data.values())
        cn_pos = sum(len(d.get('positions', [])) for d in cn_data.values())
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("💰 总权益", f"¥{cn_equity:,.0f}")
        cc2.metric("📈 总盈亏", f"¥{cn_pl:+,.0f}")
        cc3.metric("📊 总持仓", f"{cn_pos}")
        st.divider()

        for acc_name, data in cn_data.items():
            equity = data.get('equity', 0)
            pl = data.get('pl', 0)
            positions = data.get('positions', [])
            equity_hist = data.get('equity_history', [])
            pl_color = "🟢" if pl >= 0 else "🔴"

            with st.expander(
                f"**{acc_name}** {pl_color} ¥{equity:,.0f} | P&L: ¥{pl:+,.0f} | {len(positions)} 持仓",
                expanded=len(positions) > 0
            ):
                if positions:
                    rows = []
                    for p in positions:
                        rows.append({
                            '股票': p.get('symbol', ''),
                            '数量': p.get('qty', 0),
                            '成本': f"¥{p.get('avg_entry_price', 0):.2f}",
                            '现价': f"¥{p.get('current_price', 0):.2f}",
                            '涨跌': f"{p.get('pnl_pct', 0):+.1f}%",
                            '市值': f"¥{p.get('market_value', 0):,.0f}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("无持仓")

                # Equity curve
                if equity_hist:
                    edf = pd.DataFrame(equity_hist)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=edf['date'], y=edf['equity'],
                        fill='tozeroy', fillcolor='rgba(99,102,241,0.1)',
                        line=dict(color='#6366f1', width=2),
                        name='权益'
                    ))
                    fig.update_layout(
                        template="plotly_dark", height=250,
                        title=f"{acc_name} 权益曲线",
                        yaxis_title="¥", margin=dict(l=40, r=20, t=40, b=30),
                    )
                    st.plotly_chart(fig, use_container_width=True)
