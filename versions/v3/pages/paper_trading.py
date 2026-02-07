"""
模拟盘交易页面 - Paper Trading Dashboard
==========================================
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
env_file = project_root / ".env"


def _env_float(name: str, default: float) -> float:
    """安全读取环境变量中的浮点数"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _upsert_env_values(file_path: Path, values: dict) -> None:
    """写入或更新 .env 中指定键值"""
    lines = []
    if file_path.exists():
        lines = file_path.read_text(encoding="utf-8").splitlines()

    updated = {k: False for k in values}
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue

        key = stripped.split("=", 1)[0].strip()
        if key in values:
            new_lines.append(f"{key}={values[key]}")
            updated[key] = True
        else:
            new_lines.append(line)

    for key, done in updated.items():
        if not done:
            new_lines.append(f"{key}={values[key]}")

    file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _show_trade_error(err: Exception) -> None:
    """统一展示交易错误"""
    msg = str(err)
    if "风控拦截" in msg:
        st.warning(f"🛡️ {msg}")
    else:
        st.error(f"❌ 下单失败: {msg}")


def _resolve_alpaca_keys():
    """优先读取环境变量，其次读取 Streamlit secrets（含 [alpaca] 分组）"""
    api = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if api and secret:
        return api, secret

    try:
        if hasattr(st, "secrets"):
            api = api or st.secrets.get("ALPACA_API_KEY") or st.secrets.get("alpaca_api_key")
            secret = secret or st.secrets.get("ALPACA_SECRET_KEY") or st.secrets.get("alpaca_secret_key")

            alpaca_group = st.secrets.get("alpaca")
            if isinstance(alpaca_group, dict):
                api = api or alpaca_group.get("api_key") or alpaca_group.get("ALPACA_API_KEY")
                secret = secret or alpaca_group.get("secret_key") or alpaca_group.get("ALPACA_SECRET_KEY")
    except Exception:
        pass

    return api, secret

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
api_key, secret_key = _resolve_alpaca_keys()

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
def get_trader(api_key: str, secret_key: str,
               enable_hard_risk_guards: bool,
               max_single_position_pct: float,
               max_daily_loss_pct: float,
               max_portfolio_drawdown_pct: float):
    return AlpacaTrader(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
        enable_hard_risk_guards=enable_hard_risk_guards,
        max_single_position_pct=max_single_position_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        max_portfolio_drawdown_pct=max_portfolio_drawdown_pct
    )


enable_hard_risk_guards = os.environ.get("ALPACA_ENABLE_HARD_RISK_GUARDS", "true").lower() == "true"
max_single_position_pct = _env_float("ALPACA_MAX_SINGLE_POSITION_PCT", 0.20)
max_daily_loss_pct = _env_float("ALPACA_MAX_DAILY_LOSS_PCT", 0.03)
max_portfolio_drawdown_pct = _env_float("ALPACA_MAX_PORTFOLIO_DRAWDOWN_PCT", 0.15)

try:
    trader = get_trader(
        api_key,
        secret_key,
        enable_hard_risk_guards,
        max_single_position_pct,
        max_daily_loss_pct,
        max_portfolio_drawdown_pct
    )
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

st.caption("当前生效风控参数")
r1, r2, r3, r4 = st.columns(4)
r1.metric("硬风控", "开启" if enable_hard_risk_guards else "关闭")
r2.metric("单票上限", f"{max_single_position_pct * 100:.1f}%")
r3.metric("日亏损上限", f"{max_daily_loss_pct * 100:.1f}%")
r4.metric("回撤上限", f"{max_portfolio_drawdown_pct * 100:.1f}%")

with st.expander("🛡️ 风控参数（执行层）", expanded=False):
    st.caption("修改后会写入 versions/v3/.env，并立即生效。")
    risk_enable = st.checkbox("启用硬风控", value=enable_hard_risk_guards)
    risk_single = st.slider(
        "单票最大仓位 (%)", min_value=5, max_value=50,
        value=int(round(max_single_position_pct * 100))
    )
    risk_daily = st.slider(
        "当日最大亏损 (%)", min_value=1, max_value=20,
        value=int(round(max_daily_loss_pct * 100))
    )
    risk_dd = st.slider(
        "组合最大回撤 (%)", min_value=5, max_value=50,
        value=int(round(max_portfolio_drawdown_pct * 100))
    )

    if st.button("💾 保存风控参数", type="secondary"):
        updates = {
            "ALPACA_ENABLE_HARD_RISK_GUARDS": str(risk_enable).lower(),
            "ALPACA_MAX_SINGLE_POSITION_PCT": f"{risk_single / 100:.4f}",
            "ALPACA_MAX_DAILY_LOSS_PCT": f"{risk_daily / 100:.4f}",
            "ALPACA_MAX_PORTFOLIO_DRAWDOWN_PCT": f"{risk_dd / 100:.4f}",
        }
        _upsert_env_values(env_file, updates)
        for k, v in updates.items():
            os.environ[k] = v
        get_trader.clear()
        st.success("✅ 风控参数已保存并应用")
        st.rerun()

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
            _show_trade_error(e)

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
# Tab 4: 自动交易 (增强版)
# ============================================================================
with tab4:
    st.subheader("🤖 信号自动交易")
    
    # 子标签页
    auto_tab1, auto_tab2 = st.tabs(["📡 批量信号交易", "🔧 手动信号交易"])
    
    # -------------------- 批量信号交易 --------------------
    with auto_tab1:
        st.markdown("根据最新扫描信号，自动批量买入符合条件的股票")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_max_positions = st.slider("最大持仓数", 1, 10, 5, key="auto_max")
            auto_min_blue = st.slider("最低 BLUE 分数", 80, 200, 100, key="auto_blue")
            auto_position_pct = st.slider("单股仓位 %", 5, 25, 10, key="auto_pct")
        
        with col2:
            auto_stop_loss = st.slider("止损 %", 3, 15, 8, key="auto_sl")
            auto_take_profit = st.slider("止盈 %", 10, 50, 20, key="auto_tp")
            auto_min_turnover = st.slider("最低成交额 (M)", 1, 50, 10, key="auto_turnover")
        
        st.markdown("---")
        
        # 信号预览
        if st.button("🔍 预览可交易信号", key="preview_signals"):
            with st.spinner("获取信号..."):
                try:
                    from db.database import query_scan_results, get_scanned_dates
                    
                    dates = get_scanned_dates(market='US')
                    if dates:
                        results = query_scan_results(scan_date=dates[0], market='US', min_blue=auto_min_blue)
                        
                        # 过滤
                        filtered = []
                        for r in results:
                            turnover = r.get('turnover_m') or 0
                            cap = r.get('market_cap') or 0
                            symbol = r.get('symbol', '')
                            
                            if (turnover >= auto_min_turnover and 
                                cap >= 100_000_000 and 
                                len(symbol) <= 5):
                                filtered.append(r)
                                if len(filtered) >= 20:
                                    break
                        
                        if filtered:
                            st.success(f"✅ 找到 {len(filtered)} 个符合条件的信号 (扫描日期: {dates[0]})")
                            
                            df_preview = pd.DataFrame([{
                                '股票': r.get('symbol'),
                                '名称': r.get('name', '')[:15],
                                '价格': f"${r.get('price', 0):.2f}",
                                'BLUE日': f"{r.get('blue_daily', 0):.0f}",
                                'BLUE周': f"{r.get('blue_weekly', 0):.0f}",
                                '成交额': f"${r.get('turnover_m', 0):.1f}M",
                                '黑马': '🐴' if r.get('heima_daily') else ''
                            } for r in filtered[:10]])
                            
                            st.dataframe(df_preview, use_container_width=True, hide_index=True)
                        else:
                            st.warning(f"没有符合条件的信号 (BLUE >= {auto_min_blue}, 成交额 >= ${auto_min_turnover}M)")
                    else:
                        st.error("没有扫描数据")
                        
                except Exception as e:
                    st.error(f"获取信号失败: {e}")
        
        st.markdown("---")
        
        # 执行批量交易
        if st.button("🚀 执行批量买入", type="primary", key="batch_buy"):
            with st.spinner("执行信号交易..."):
                try:
                    from db.database import query_scan_results, get_scanned_dates
                    
                    signal_trader = SignalTrader(
                        trader=trader,
                        max_position_pct=auto_position_pct/100,
                        stop_loss_pct=auto_stop_loss/100
                    )
                    
                    # 获取当前持仓
                    positions = trader.get_positions()
                    current_symbols = {p.symbol for p in positions}
                    available_slots = auto_max_positions - len(current_symbols)
                    
                    if available_slots <= 0:
                        st.warning(f"⚠️ 持仓已满 ({len(current_symbols)}/{auto_max_positions})")
                    else:
                        # 获取信号
                        dates = get_scanned_dates(market='US')
                        if dates:
                            results = query_scan_results(scan_date=dates[0], market='US', min_blue=auto_min_blue)
                            
                            # 过滤并验证
                            filtered = []
                            for r in results:
                                turnover = r.get('turnover_m') or 0
                                cap = r.get('market_cap') or 0
                                symbol = r.get('symbol', '')
                                
                                if (turnover >= auto_min_turnover and 
                                    cap >= 100_000_000 and 
                                    len(symbol) <= 5 and
                                    symbol not in current_symbols):
                                    
                                    # 验证价格
                                    try:
                                        price = trader.get_latest_price(symbol)
                                        if price > 0:
                                            r['current_price'] = price
                                            filtered.append(r)
                                    except:
                                        pass
                                    
                                    if len(filtered) >= available_slots:
                                        break
                            
                            if filtered:
                                # 按 BLUE 排序
                                filtered.sort(key=lambda x: x.get('blue_daily', 0) or 0, reverse=True)
                                
                                executed = []
                                for signal in filtered[:available_slots]:
                                    symbol = signal['symbol']
                                    blue_score = signal.get('blue_daily', 0)
                                    result = signal_trader.execute_buy_signal(symbol, f"BLUE={blue_score:.0f}")
                                    
                                    if result['success']:
                                        executed.append(result)
                                        st.success(f"✅ {result['message']}")
                                    else:
                                        st.warning(f"⚠️ {symbol}: {result['message']}")
                                
                                if executed:
                                    st.balloons()
                                    st.success(f"🎉 成功买入 {len(executed)} 只股票!")
                            else:
                                st.warning("没有找到可交易的股票")
                        else:
                            st.error("没有扫描数据")
                            
                except Exception as e:
                    st.error(f"执行失败: {e}")
    
    # -------------------- 手动信号交易 --------------------
    with auto_tab2:
        st.markdown("手动输入股票代码执行信号交易")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_position = st.slider("单只股票最大仓位", 5, 30, 10, 5, key="manual_max")
            st.caption(f"每只股票最多使用 {max_position}% 资金")
        
        with col2:
            stop_loss = st.slider("止损比例", 3, 15, 8, key="manual_sl")
            st.caption(f"自动设置 {stop_loss}% 止损单")
        
        st.markdown("---")
        
        # 买入
        st.markdown("##### 🟢 买入信号")
        buy_symbol = st.text_input("股票代码", "", key="manual_buy_symbol").upper()
        buy_reason = st.text_input("信号原因", "BLUE + 黑马共振", key="manual_buy_reason")
        
        if st.button("执行买入", type="primary", key="manual_buy_btn") and buy_symbol:
            signal_trader = SignalTrader(trader, max_position/100, stop_loss/100)
            result = signal_trader.execute_buy_signal(buy_symbol, buy_reason)
            
            if result['success']:
                st.success(f"✅ {result['message']}")
                st.info(f"止损价: ${result.get('stop_price', 0):.2f}")
            else:
                st.error(f"❌ {result['message']}")
        
        st.markdown("---")
        
        # 卖出
        st.markdown("##### 🔴 卖出信号")
        positions = trader.get_positions()
        if positions:
            sell_symbol = st.selectbox("选择持仓卖出", [p.symbol for p in positions], key="manual_sell_symbol")
            sell_reason = st.text_input("卖出原因", "KDJ J > 90", key="manual_sell_reason")
            
            if st.button("执行卖出", type="secondary", key="manual_sell_btn"):
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
