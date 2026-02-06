"""
Alpaca 持仓小部件
================
可嵌入侧边栏或页面任何位置，显示实时 Alpaca 持仓
"""
import streamlit as st
import pandas as pd
import os


def _show_trade_error(err: Exception):
    """统一展示交易错误"""
    msg = str(err)
    if "风控拦截" in msg:
        st.warning(f"🛡️ {msg}")
    else:
        st.error(f"❌ 交易失败: {msg}")


def get_alpaca_trader():
    """获取 Alpaca Trader 实例 (缓存)"""
    try:
        from execution.alpaca_trader import AlpacaTrader, ALPACA_SDK_AVAILABLE
        
        if not ALPACA_SDK_AVAILABLE:
            return None
        
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            return None
        
        return AlpacaTrader(api_key=api_key, secret_key=secret_key, paper=True)
    except Exception:
        return None


def render_alpaca_sidebar_widget():
    """
    在侧边栏底部显示 Alpaca 持仓摘要
    """
    trader = get_alpaca_trader()
    
    if not trader:
        st.caption("💰 Alpaca 未连接")
        if st.button("⚙️ 配置 API", key="sidebar_alpaca_config"):
            st.session_state['show_alpaca_config'] = True
        return
    
    try:
        account = trader.get_account()
        positions = trader.get_positions()
        market = trader.get_market_hours()
        
        # 计算总盈亏
        total_pnl = sum(p.unrealized_pl for p in positions)
        total_pnl_pct = (total_pnl / float(account.equity)) * 100 if float(account.equity) > 0 else 0
        
        # 市场状态
        status_icon = "🟢" if market['is_open'] else "🔴"
        
        # 显示摘要卡片
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                    border-radius: 12px; padding: 12px; margin-top: 10px;
                    border: 1px solid #2a3a5e;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: #fff;">💰 Alpaca</span>
                <span style="font-size: 0.75em; color: #888;">{status_icon} Paper</span>
            </div>
            <div style="font-size: 1.3em; font-weight: bold; color: #00D4AA; margin: 6px 0;">
                ${float(account.equity):,.0f}
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85em;">
                <span style="color: {'#00C853' if total_pnl >= 0 else '#FF5252'};">
                    {'+' if total_pnl >= 0 else ''}${total_pnl:,.0f} ({total_pnl_pct:+.2f}%)
                </span>
                <span style="color: #888;">{len(positions)} 持仓</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 展开按钮
        if st.button("📊 管理持仓", key="sidebar_alpaca_manage", use_container_width=True):
            st.session_state['show_alpaca_panel'] = True
            
    except Exception as e:
        st.caption(f"⚠️ {str(e)[:30]}")


def render_alpaca_floating_bar():
    """
    在页面底部显示浮动持仓栏
    """
    trader = get_alpaca_trader()
    
    if not trader:
        return  # 未连接时不显示
    
    try:
        account = trader.get_account()
        positions = trader.get_positions()
        market = trader.get_market_hours()
        
        if not positions:
            return  # 无持仓时不显示
        
        # 计算总盈亏
        total_pnl = sum(p.unrealized_pl for p in positions)
        total_pnl_pct = (total_pnl / float(account.equity)) * 100 if float(account.equity) > 0 else 0
        
        # 显示浮动栏
        st.markdown("---")
        
        # 标题行
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            status_icon = "🟢" if market['is_open'] else "🔴"
            st.markdown(f"### 💼 Alpaca 持仓 {status_icon}")
        
        with col2:
            pnl_color = "green" if total_pnl >= 0 else "red"
            st.markdown(f"""
            <div style="text-align: right; padding-top: 8px;">
                <span style="color: {pnl_color}; font-weight: bold; font-size: 1.1em;">
                    {'+' if total_pnl >= 0 else ''}${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)
                </span>
                <span style="color: #888; margin-left: 12px;">净值: ${float(account.equity):,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            expand = st.checkbox("展开详情", value=False, key="alpaca_bar_expand")
        
        # 持仓卡片 (简略)
        if not expand:
            cols = st.columns(min(len(positions), 6) + 1)
            for i, pos in enumerate(positions[:6]):
                with cols[i]:
                    pnl_pct = pos.unrealized_plpc
                    emoji = "🟢" if pnl_pct >= 0 else "🔴"
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 8px; 
                                padding: 8px; text-align: center;">
                        <div style="font-weight: bold;">{pos.symbol}</div>
                        <div style="color: {'#00C853' if pnl_pct >= 0 else '#FF5252'}; font-size: 0.9em;">
                            {pnl_pct:+.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(positions) > 6:
                with cols[6]:
                    st.markdown(f"<div style='padding-top: 12px; color: #888;'>+{len(positions)-6} 更多</div>", 
                               unsafe_allow_html=True)
        
        # 展开详情
        else:
            # 持仓表格
            pos_data = []
            for pos in positions:
                pnl_emoji = "🟢" if pos.unrealized_pl >= 0 else "🔴"
                pos_data.append({
                    '': pnl_emoji,
                    '股票': pos.symbol,
                    '数量': int(pos.qty),
                    '成本': f"${pos.avg_entry_price:.2f}",
                    '现价': f"${pos.current_price:.2f}",
                    '市值': f"${pos.market_value:,.2f}",
                    '盈亏': f"${pos.unrealized_pl:+,.2f}",
                    '盈亏%': f"{pos.unrealized_plpc:+.2f}%"
                })
            
            df = pd.DataFrame(pos_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 操作按钮
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                sell_symbol = st.selectbox(
                    "选择平仓",
                    options=[p.symbol for p in positions],
                    key="floating_sell_select",
                    label_visibility="collapsed"
                )
            
            with col2:
                if st.button("📤 平仓", key="floating_sell_btn"):
                    try:
                        trader.close_position(sell_symbol)
                        st.success(f"✅ {sell_symbol} 已平仓")
                        st.rerun()
                    except Exception as e:
                        st.error(f"平仓失败: {e}")
            
            with col3:
                if st.button("🚨 全部清仓", key="floating_close_all"):
                    try:
                        trader.close_all_positions()
                        st.success("✅ 所有持仓已清仓")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清仓失败: {e}")
            
            with col4:
                if st.button("🔄 刷新", key="floating_refresh"):
                    st.rerun()
                    
    except Exception as e:
        st.caption(f"⚠️ Alpaca 连接异常: {e}")


def render_alpaca_quick_trade(symbol: str = None, suggested_price: float = None):
    """
    快速交易组件 - 可嵌入股票详情页
    
    Args:
        symbol: 预填股票代码
        suggested_price: 建议价格
    """
    trader = get_alpaca_trader()
    
    if not trader:
        st.warning("⚠️ 请配置 Alpaca API 后使用快速交易")
        return
    
    try:
        account = trader.get_account()
        buying_power = float(account.buying_power)
        
        st.markdown("#### 🚀 Alpaca 快速交易")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            trade_symbol = st.text_input(
                "股票代码",
                value=symbol or "",
                key=f"quick_trade_symbol_{symbol or 'default'}",
                placeholder="NVDA"
            ).upper()
        
        with col2:
            # 获取当前价格
            if trade_symbol:
                try:
                    current_price = trader.get_latest_price(trade_symbol)
                except:
                    current_price = suggested_price or 100
            else:
                current_price = suggested_price or 100
            
            # 默认数量: 约占 10% 仓位
            default_qty = max(1, int(buying_power * 0.1 / current_price)) if current_price > 0 else 10
            trade_qty = st.number_input("数量", min_value=1, value=default_qty, 
                                        key=f"quick_trade_qty_{symbol or 'default'}")
        
        with col3:
            trade_side = st.selectbox("方向", ["买入", "卖出"], 
                                      key=f"quick_trade_side_{symbol or 'default'}")
        
        with col4:
            st.write("")  # 占位
            st.write("")
            if st.button("🚀 执行", type="primary", key=f"quick_trade_exec_{symbol or 'default'}"):
                if trade_symbol:
                    with st.spinner("执行中..."):
                        try:
                            if trade_side == "买入":
                                order = trader.buy_market(trade_symbol, trade_qty)
                            else:
                                order = trader.sell_market(trade_symbol, trade_qty)
                            
                            st.success(f"✅ 订单已提交: {order['id'][:8]}...")
                            st.rerun()
                        except Exception as e:
                            _show_trade_error(e)
                else:
                    st.warning("请输入股票代码")
        
        # 预估信息
        if trade_symbol and trade_qty > 0:
            try:
                price = trader.get_latest_price(trade_symbol)
                total = price * trade_qty
                pct = (total / buying_power) * 100
                st.caption(f"💰 预估: ${total:,.2f} (占可用资金 {pct:.1f}%) | 可用: ${buying_power:,.0f}")
            except:
                pass
                
    except Exception as e:
        st.error(f"交易组件加载失败: {e}")


def render_inline_backtest(symbol: str, market: str = 'US', days: int = 365):
    """
    内联快速回测 - 显示该股票的策略历史表现
    
    Args:
        symbol: 股票代码
        market: 市场
        days: 回测天数
    """
    try:
        from data_fetcher import get_stock_data
        from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series
        import numpy as np
        
        # 获取数据
        df = get_stock_data(symbol, market=market, days=days)
        
        if df is None or len(df) < 100:
            st.caption("📊 历史数据不足，无法回测")
            return
        
        # 计算指标
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        blue = calculate_blue_signal_series(opens, highs, lows, closes)
        heima, _ = calculate_heima_signal_series(highs, lows, closes, opens)
        
        # 简单回测: BLUE > 100 买入
        strategies = [
            {'name': 'BLUE>100', 'signal': blue > 100},
            {'name': 'BLUE>150', 'signal': blue > 150},
            {'name': 'BLUE+黑马', 'signal': (blue > 100) & heima}
        ]
        
        results = []
        hold_days = 10
        
        for strat in strategies:
            signal = strat['signal']
            
            trades = []
            i = 0
            while i < len(df) - hold_days:
                if signal[i]:
                    entry = closes[i]
                    exit_price = closes[min(i + hold_days, len(closes) - 1)]
                    pnl = (exit_price - entry) / entry
                    trades.append(pnl)
                    i += hold_days
                else:
                    i += 1
            
            if trades:
                total_return = (1 + sum(trades)) - 1
                win_rate = sum(1 for t in trades if t > 0) / len(trades)
                results.append({
                    '策略': strat['name'],
                    '收益': f"{total_return*100:+.0f}%",
                    '胜率': f"{win_rate*100:.0f}%",
                    '交易': len(trades)
                })
        
        if results:
            st.markdown("**📈 快速回测** (持有10天)")
            
            # 使用紧凑的卡片显示
            cols = st.columns(len(results))
            for i, r in enumerate(results):
                with cols[i]:
                    color = "#00C853" if "+" in r['收益'] else "#FF5252"
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border-radius: 8px; 
                                padding: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                        <div style="font-size: 0.8em; color: #888;">{r['策略']}</div>
                        <div style="font-weight: bold; color: {color}; font-size: 1.1em;">{r['收益']}</div>
                        <div style="font-size: 0.75em; color: #888;">胜率{r['胜率']} {r['交易']}笔</div>
                    </div>
                    """, unsafe_allow_html=True)
        
    except Exception as e:
        st.caption(f"回测失败: {e}")
