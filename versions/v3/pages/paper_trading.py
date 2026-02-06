"""
模拟盘交易页面 - Paper Trading Dashboard
==========================================
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="模拟盘交易",
    page_icon="💰",
    layout="wide"
)

# 检查 Alpaca SDK
try:
    from execution.alpaca_trader import (
        AlpacaTrader, 
        SignalTrader,
        check_alpaca_available,
        setup_instructions,
        ALPACA_SDK_AVAILABLE
    )
except ImportError:
    ALPACA_SDK_AVAILABLE = False

st.title("💰 模拟盘交易")

if not ALPACA_SDK_AVAILABLE:
    st.error("❌ 请安装 Alpaca SDK: `pip install alpaca-py`")
    st.code("pip install alpaca-py", language="bash")
    st.stop()

# 检查 API Keys
import os
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

if not api_key or not secret_key:
    st.warning("⚠️ 未配置 Alpaca API Keys")
    
    with st.expander("📖 配置指南", expanded=True):
        st.markdown("""
        ### 设置步骤
        
        1. **注册 Alpaca 账号** (免费): [https://alpaca.markets/](https://alpaca.markets/)
        
        2. **获取 API Keys**:
           - 登录后点击 "Paper Trading"
           - 点击 "Your API Keys"
           - 复制 API Key 和 Secret Key
        
        3. **配置环境变量**:
           在 `.env` 文件中添加:
           ```
           ALPACA_API_KEY=your_api_key_here
           ALPACA_SECRET_KEY=your_secret_key_here
           ALPACA_PAPER=true
           ```
        
        4. **重启应用**
        """)
    
    # 手动输入
    st.markdown("---")
    st.subheader("🔑 临时输入 API Keys")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_api_key = st.text_input("API Key", type="password")
    with col2:
        temp_secret_key = st.text_input("Secret Key", type="password")
    
    if temp_api_key and temp_secret_key:
        api_key = temp_api_key
        secret_key = temp_secret_key
        st.success("✅ 已输入临时 API Keys")
    else:
        st.stop()

# 初始化 Trader
@st.cache_resource
def get_trader(api_key: str, secret_key: str):
    return AlpacaTrader(api_key=api_key, secret_key=secret_key, paper=True)

try:
    trader = get_trader(api_key, secret_key)
    account = trader.get_account()
except Exception as e:
    st.error(f"❌ 连接失败: {e}")
    st.stop()

# ============================================================================
# 主界面
# ============================================================================

# 账户信息
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "账户净值",
        f"${account.equity:,.2f}",
        help="总资产价值"
    )

with col2:
    st.metric(
        "可用现金",
        f"${account.cash:,.2f}",
        help="未投资的现金"
    )

with col3:
    st.metric(
        "购买力",
        f"${account.buying_power:,.2f}",
        help="可用于购买股票的资金"
    )

with col4:
    market = trader.get_market_hours()
    status = "🟢 开盘中" if market['is_open'] else "🔴 休市"
    st.metric("市场状态", status)

# 标签页
tab1, tab2, tab3, tab4 = st.tabs(["📊 持仓", "📝 下单", "📋 订单", "🤖 自动交易"])

# ============================================================================
# Tab 1: 持仓
# ============================================================================
with tab1:
    st.subheader("当前持仓")
    
    positions = trader.get_positions()
    
    if not positions:
        st.info("暂无持仓")
    else:
        pos_data = []
        total_pnl = 0
        
        for pos in positions:
            total_pnl += pos.unrealized_pl
            pos_data.append({
                '股票': pos.symbol,
                '数量': int(pos.qty),
                '成本价': f"${pos.avg_entry_price:.2f}",
                '现价': f"${pos.current_price:.2f}",
                '市值': f"${pos.market_value:,.2f}",
                '盈亏': f"${pos.unrealized_pl:+,.2f}",
                '盈亏%': f"{pos.unrealized_plpc:+.2f}%"
            })
        
        df = pd.DataFrame(pos_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 总盈亏
        color = "green" if total_pnl >= 0 else "red"
        st.markdown(f"**总浮动盈亏:** <span style='color:{color}'>${total_pnl:+,.2f}</span>", 
                    unsafe_allow_html=True)
        
        # 平仓操作
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            close_symbol = st.selectbox("选择股票平仓", [p.symbol for p in positions])
        
        with col2:
            if st.button("平仓", type="secondary"):
                result = trader.close_position(close_symbol)
                st.success(f"✅ 已提交平仓订单: {result['id']}")
                st.rerun()
        
        if st.button("🚨 全部清仓", type="primary"):
            if st.checkbox("确认清仓所有持仓"):
                trader.close_all_positions()
                st.success("✅ 已清仓所有持仓")
                st.rerun()

# ============================================================================
# Tab 2: 下单
# ============================================================================
with tab2:
    st.subheader("手动下单")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("股票代码", "AAPL").upper()
        qty = st.number_input("数量", min_value=1, value=10)
        order_type = st.selectbox("订单类型", ["市价单", "限价单", "止损单"])
    
    with col2:
        side = st.radio("方向", ["买入", "卖出"], horizontal=True)
        
        if order_type == "限价单":
            price = st.number_input("限价", min_value=0.01, value=100.0, step=0.01)
        elif order_type == "止损单":
            stop_price = st.number_input("止损价", min_value=0.01, value=100.0, step=0.01)
        
        tif = st.selectbox("有效期", ["day", "gtc", "ioc"], 
                           format_func=lambda x: {"day": "当日有效", "gtc": "撤销前有效", "ioc": "立即成交或取消"}[x])
    
    st.markdown("---")
    
    if st.button("📤 提交订单", type="primary"):
        try:
            if side == "买入":
                if order_type == "市价单":
                    result = trader.buy_market(symbol, qty, tif)
                elif order_type == "限价单":
                    result = trader.buy_limit(symbol, qty, price, tif)
                else:
                    result = trader.buy_stop(symbol, qty, stop_price, tif)
            else:
                if order_type == "市价单":
                    result = trader.sell_market(symbol, qty, tif)
                elif order_type == "限价单":
                    result = trader.sell_limit(symbol, qty, price, tif)
                else:
                    result = trader.sell_stop(symbol, qty, stop_price, tif)
            
            st.success(f"✅ 订单已提交!")
            st.json(result)
            
        except Exception as e:
            st.error(f"❌ 下单失败: {e}")

# ============================================================================
# Tab 3: 订单
# ============================================================================
with tab3:
    st.subheader("订单管理")
    
    order_status = st.radio("订单状态", ["open", "closed", "all"], 
                            format_func=lambda x: {"open": "待成交", "closed": "已成交", "all": "全部"}[x],
                            horizontal=True)
    
    orders = trader.get_orders(order_status)
    
    if not orders:
        status_map = {"open": "待成交", "closed": "已成交", "all": ""}
        st.info(f"暂无{status_map[order_status]}订单")
    else:
        order_data = []
        for order in orders:
            order_data.append({
                '订单ID': order['id'][:8] + "...",
                '股票': order['symbol'],
                '方向': '买入' if order['side'] == 'buy' else '卖出',
                '类型': order['type'],
                '数量': order['qty'],
                '已成交': order['filled_qty'],
                '状态': order['status'],
                '创建时间': order['created_at'][:19] if order['created_at'] else ""
            })
        
        df = pd.DataFrame(order_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 撤单操作
        if order_status == "open" and orders:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                cancel_order = st.selectbox(
                    "选择订单撤销", 
                    orders,
                    format_func=lambda x: f"{x['symbol']} - {x['side']} {x['qty']}股"
                )
            
            with col2:
                if st.button("撤销订单"):
                    if trader.cancel_order(cancel_order['id']):
                        st.success("✅ 订单已撤销")
                        st.rerun()
                    else:
                        st.error("❌ 撤销失败")
            
            if st.button("撤销所有订单"):
                trader.cancel_all_orders()
                st.success("✅ 所有订单已撤销")
                st.rerun()

# ============================================================================
# Tab 4: 自动交易
# ============================================================================
with tab4:
    st.subheader("🤖 信号自动交易")
    
    st.markdown("""
    根据系统信号自动执行交易。当检测到买入/卖出信号时，自动下单。
    """)
    
    # 配置
    col1, col2 = st.columns(2)
    
    with col1:
        max_position = st.slider("单只股票最大仓位", 5, 30, 10, 5)
        st.caption(f"每只股票最多使用 {max_position}% 资金")
    
    with col2:
        stop_loss = st.slider("止损比例", 3, 15, 8)
        st.caption(f"自动设置 {stop_loss}% 止损单")
    
    st.markdown("---")
    
    # 信号执行
    st.subheader("执行买入信号")
    
    buy_symbol = st.text_input("股票代码 (买入)", "").upper()
    buy_reason = st.text_input("信号原因", "BLUE + 黑马共振")
    
    if st.button("🟢 执行买入", type="primary") and buy_symbol:
        signal_trader = SignalTrader(trader, max_position/100, stop_loss/100)
        result = signal_trader.execute_buy_signal(buy_symbol, buy_reason)
        
        if result['success']:
            st.success(f"✅ {result['message']}")
            st.info(f"止损价: ${result.get('stop_price', 0):.2f}")
        else:
            st.error(f"❌ {result['message']}")
    
    st.markdown("---")
    
    st.subheader("执行卖出信号")
    
    positions = trader.get_positions()
    if positions:
        sell_symbol = st.selectbox("选择持仓卖出", [p.symbol for p in positions])
        sell_reason = st.text_input("卖出原因", "KDJ J > 90")
        
        if st.button("🔴 执行卖出", type="secondary"):
            signal_trader = SignalTrader(trader)
            result = signal_trader.execute_sell_signal(sell_symbol, sell_reason)
            
            if result['success']:
                st.success(f"✅ {result['message']}")
                if result.get('pnl'):
                    color = "green" if result['pnl'] >= 0 else "red"
                    st.markdown(f"盈亏: <span style='color:{color}'>${result['pnl']:+,.2f} ({result['pnl_pct']:+.2f}%)</span>",
                               unsafe_allow_html=True)
            else:
                st.error(f"❌ {result['message']}")
    else:
        st.info("暂无持仓可卖出")

# 页脚
st.markdown("---")
st.caption(f"🔌 已连接 Alpaca {'模拟盘' if account.is_paper else '实盘'} | 最后更新: {datetime.now().strftime('%H:%M:%S')}")
