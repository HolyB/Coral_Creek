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


def render_alpaca_sidebar_widget(enabled: bool = True, current_market: str = "US"):
    """
    在侧边栏底部显示 Alpaca 持仓摘要 + Paper Trading 子账户状态
    """
    trader = get_alpaca_trader()
    
    # 获取当前选中的 Paper 子账户
    paper_account_name = st.session_state.get('global_paper_account_name', 'default')
    
    # Paper Trading 子账户信息
    try:
        from services.portfolio_service import get_paper_account, get_paper_account_config
        paper_account = get_paper_account(paper_account_name)
        paper_config = get_paper_account_config(paper_account_name)
        paper_available = True
    except Exception:
        paper_account = None
        paper_config = {}
        paper_available = False
    
    # Alpaca 信息
    if not enabled:
        st.caption(f"💰 Alpaca 仅支持美股（当前: {current_market}）")
    elif trader:
        try:
            account = trader.get_account()
            positions = trader.get_positions()
            market = trader.get_market_hours()
            
            # 计算总盈亏
            total_pnl = sum(p.unrealized_pl for p in positions)
            total_pnl_pct = (total_pnl / float(account.equity)) * 100 if float(account.equity) > 0 else 0
            
            # 市场状态
            status_icon = "🟢" if market['is_open'] else "🔴"
            
            # Alpaca 摘要卡片
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
    else:
        st.caption("💰 Alpaca 未连接")
        if st.button("⚙️ 配置 API", key="sidebar_alpaca_config"):
            st.session_state['show_alpaca_config'] = True
    
    # Paper Trading 子账户卡片
    if paper_available and paper_account:
        paper_equity = paper_account.get('total_equity', 0)
        paper_pnl = paper_account.get('total_pnl', 0)
        paper_pnl_pct = paper_account.get('total_pnl_pct', 0)
        strategy_note = paper_config.get('strategy_note', '')[:30]
        max_pos = float(paper_config.get('max_single_position_pct', 0.30)) * 100
        max_dd = float(paper_config.get('max_drawdown_pct', 0.20)) * 100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e1e3f, #2a1a3e); 
                    border-radius: 12px; padding: 10px; margin-top: 8px;
                    border: 1px solid #3a2a5e;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: #fff;">🎮 模拟盘</span>
                <span style="font-size: 0.7em; color: #9C27B0;">{paper_account_name}</span>
            </div>
            <div style="font-size: 1.1em; font-weight: bold; color: #CE93D8; margin: 4px 0;">
                ${paper_equity:,.0f}
            </div>
            <div style="font-size: 0.8em; color: {'#00C853' if paper_pnl >= 0 else '#FF5252'};">
                {'+' if paper_pnl >= 0 else ''}${paper_pnl:,.0f} ({paper_pnl_pct:+.1f}%)
            </div>
            <div style="font-size: 0.65em; color: #888; margin-top: 4px;">
                🛡️ 单票≤{max_pos:.0f}% | 回撤≤{max_dd:.0f}%
            </div>
            {f'<div style="font-size: 0.6em; color: #666; margin-top: 2px;">📝 {strategy_note}...</div>' if strategy_note else ''}
        </div>
        """, unsafe_allow_html=True)



def render_alpaca_floating_bar(enabled: bool = True, market: str = "US"):
    """
    在页面底部显示浮动持仓栏
    """
    if not enabled:
        return

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
        
        # 持仓卡片 (简略 - 但带快速卖出)
        if not expand:
            # 初始化卖出确认状态
            if 'floating_confirm_sell' not in st.session_state:
                st.session_state['floating_confirm_sell'] = None
            
            cols = st.columns(min(len(positions), 5) + 1)
            for i, pos in enumerate(positions[:5]):
                with cols[i]:
                    pnl_pct = pos.unrealized_plpc
                    pnl_color = '#00C853' if pnl_pct >= 0 else '#FF5252'
                    
                    # 检查是否正在确认卖出这只股票
                    confirming = st.session_state.get('floating_confirm_sell') == pos.symbol
                    
                    if confirming:
                        # 确认卖出模式
                        st.markdown(f"""
                        <div style="background: rgba(255,82,82,0.15); border-radius: 8px; 
                                    padding: 8px; text-align: center; border: 1px solid #FF5252;">
                            <div style="font-weight: bold; color: #FF5252;">{pos.symbol}</div>
                            <div style="font-size: 0.75em; color: #888;">确认平仓?</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("✅", key=f"confirm_yes_{pos.symbol}", help="确认"):
                                try:
                                    trader.close_position(pos.symbol)
                                    st.session_state['floating_confirm_sell'] = None
                                    st.success(f"✅ {pos.symbol} 已平仓")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ {e}")
                        with btn_col2:
                            if st.button("❌", key=f"confirm_no_{pos.symbol}", help="取消"):
                                st.session_state['floating_confirm_sell'] = None
                                st.rerun()
                    else:
                        # 正常显示模式
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); border-radius: 8px; 
                                    padding: 8px; text-align: center;">
                            <div style="font-weight: bold;">{pos.symbol}</div>
                            <div style="color: {pnl_color}; font-size: 0.9em;">
                                {pnl_pct:+.1f}%
                            </div>
                            <div style="color: #888; font-size: 0.7em;">${pos.market_value:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 快速卖出按钮
                        if st.button("📤", key=f"quick_sell_{pos.symbol}", help=f"平仓 {pos.symbol}"):
                            st.session_state['floating_confirm_sell'] = pos.symbol
                            st.rerun()
            
            if len(positions) > 5:
                with cols[5]:
                    st.markdown(f"<div style='padding-top: 12px; color: #888;'>+{len(positions)-5} 更多</div>", 
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


def render_alpaca_quick_trade(symbol: str = None, suggested_price: float = None, market: str = "US"):
    """
    快速交易组件 - 可嵌入股票详情页
    
    Args:
        symbol: 预填股票代码
        suggested_price: 建议价格
    """
    if market != "US":
        st.info(f"ℹ️ 当前为 {market} 市场，Alpaca 仅支持美股。请使用下方“模拟买入”。")
        return

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
        
        # 持仓周期选择
        hold_options = [5, 10, 20]
        col_title, col_period = st.columns([2, 1])
        with col_title:
            st.markdown("**📈 快速回测**")
        with col_period:
            hold_days = st.selectbox(
                "持有天数",
                hold_options,
                index=1,
                key=f"hold_days_{symbol}_{market}",
                label_visibility="collapsed"
            )

        # 缓存键包含持仓周期，切换 5/10/20 天会触发重算
        cache_key = f"backtest_{symbol}_{market}_{days}_{hold_days}"

        # 检查缓存 (session_state)
        if cache_key in st.session_state:
            cached = st.session_state[cache_key]
            _render_backtest_results(cached['results'], hold_days)
            return
        
        # 获取数据
        with st.spinner(f"加载 {symbol} 历史数据..."):
            df = get_stock_data(symbol, market=market, days=days)
        
        if df is None or len(df) < 100:
            st.caption("📊 历史数据不足，无法回测")
            return
        
        # 计算指标
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        volumes = df['Volume'].values
        
        blue = calculate_blue_signal_series(opens, highs, lows, closes)
        heima, juedi = calculate_heima_signal_series(highs, lows, closes, opens)
        
        # 计算额外指标
        vol_ma20 = np.convolve(volumes, np.ones(20)/20, mode='same')
        vol_ratio = volumes / (vol_ma20 + 1e-10)
        
        # RSI
        delta = np.diff(closes, prepend=closes[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
        avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
        
        # MA
        ma5 = np.convolve(closes, np.ones(5)/5, mode='same')
        ma20 = np.convolve(closes, np.ones(20)/20, mode='same')
        ma_cross = (ma5 > ma20) & (np.roll(ma5, 1) <= np.roll(ma20, 1))
        
        # 扩展策略列表 (6个策略)
        strategies = [
            {'name': 'BLUE>100', 'signal': blue > 100, 'color': '#2196F3'},
            {'name': 'BLUE>150', 'signal': blue > 150, 'color': '#4CAF50'},
            {'name': 'BLUE+黑马', 'signal': (blue > 100) & heima, 'color': '#FF9800'},
            {'name': '日周共振', 'signal': (blue > 120) & (np.roll(blue, 5) > 100), 'color': '#9C27B0'},
            {'name': 'RSI超卖', 'signal': (rsi < 30) & (blue > 80), 'color': '#00BCD4'},
            {'name': '量价齐升', 'signal': (blue > 100) & (vol_ratio > 1.5), 'color': '#E91E63'},
        ]
        
        # 执行回测
        results = []
        
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
                total_return = sum(trades)  # 简单收益累加
                win_rate = sum(1 for t in trades if t > 0) / len(trades)
                avg_return = np.mean(trades)
                max_dd = min(trades) if trades else 0
                
                results.append({
                    '策略': strat['name'],
                    'color': strat['color'],
                    '收益': total_return * 100,
                    '胜率': win_rate * 100,
                    '交易': len(trades),
                    '平均': avg_return * 100,
                    '最大亏': max_dd * 100
                })
        
        # 缓存结果
        st.session_state[cache_key] = {
            'results': results,
        }
        
        _render_backtest_results(results, hold_days)
        
    except Exception as e:
        st.caption(f"回测失败: {e}")


def _render_backtest_results(results: list, hold_days: int):
    """渲染回测结果卡片"""
    if not results:
        st.caption("无有效回测结果")
        return
    
    # 按收益排序
    results = sorted(results, key=lambda x: x['收益'], reverse=True)
    
    # 使用紧凑的卡片显示 (3列2行)
    cols_per_row = 3
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, r in enumerate(results[row_start:row_start + cols_per_row]):
            with cols[i]:
                ret = r['收益']
                color = r.get('color', '#00C853' if ret > 0 else '#FF5252')
                border_color = color if ret > 5 else 'rgba(255,255,255,0.1)'
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border-radius: 8px; 
                            padding: 10px; text-align: center; border: 1px solid {border_color};
                            margin-bottom: 8px;">
                    <div style="font-size: 0.75em; color: {color}; font-weight: bold;">{r['策略']}</div>
                    <div style="font-weight: bold; color: {'#00C853' if ret > 0 else '#FF5252'}; font-size: 1.2em;">
                        {ret:+.0f}%
                    </div>
                    <div style="font-size: 0.7em; color: #888;">
                        胜率 {r['胜率']:.0f}% | {r['交易']}笔
                    </div>
                    <div style="font-size: 0.65em; color: #666; margin-top: 2px;">
                        平均 {r['平均']:+.1f}% | 最大亏 {r['最大亏']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.caption(f"📅 回测期间: 过去1年 | 持有 {hold_days} 天")
