"""
策略回测页面 - 自由组合买卖条件
========================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_stock_data
from indicator_utils import (
    calculate_blue_signal_series, 
    calculate_heima_signal_series, 
    calculate_kdj_series
)
from strategies.strategy_components import (
    StrategyBuilder,
    BUY_CONDITIONS,
    SELL_CONDITIONS,
)

st.set_page_config(
    page_title="策略回测",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# 回测器
# ============================================================================

def prepare_data(df_daily: pd.DataFrame) -> dict:
    """准备所有指标数据"""
    blue = calculate_blue_signal_series(
        df_daily['Open'].values, df_daily['High'].values,
        df_daily['Low'].values, df_daily['Close'].values
    )
    heima, juedi = calculate_heima_signal_series(
        df_daily['High'].values, df_daily['Low'].values,
        df_daily['Close'].values, df_daily['Open'].values
    )
    _, _, j = calculate_kdj_series(
        df_daily['High'].values, df_daily['Low'].values, 
        df_daily['Close'].values
    )
    
    # 周线数据
    df_weekly = df_daily.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    if len(df_weekly) >= 5:
        week_blue = calculate_blue_signal_series(
            df_weekly['Open'].values, df_weekly['High'].values,
            df_weekly['Low'].values, df_weekly['Close'].values
        )
        week_heima, week_juedi = calculate_heima_signal_series(
            df_weekly['High'].values, df_weekly['Low'].values,
            df_weekly['Close'].values, df_weekly['Open'].values
        )
        df_weekly['Week_BLUE'] = week_blue
        df_weekly['Week_Heima'] = week_heima
        df_weekly['Week_Juedi'] = week_juedi
        
        week_blue_ref = df_weekly['Week_BLUE'].shift(1).reindex(
            df_daily.index, method='ffill'
        ).fillna(0).values
        week_heima_ref = df_weekly['Week_Heima'].shift(1).reindex(
            df_daily.index, method='ffill'
        ).fillna(False).values
        week_juedi_ref = df_weekly['Week_Juedi'].shift(1).reindex(
            df_daily.index, method='ffill'
        ).fillna(False).values
    else:
        week_blue_ref = np.zeros(len(df_daily))
        week_heima_ref = np.zeros(len(df_daily), dtype=bool)
        week_juedi_ref = np.zeros(len(df_daily), dtype=bool)
    
    ma5 = pd.Series(df_daily['Close'].values).rolling(5).mean().values
    
    return {
        'blue': blue,
        'heima': heima,
        'juedi': juedi,
        'kdj_j': j,
        'week_blue': week_blue_ref,
        'week_heima': week_heima_ref,
        'week_juedi': week_juedi_ref,
        'ma5': ma5,
        'close': df_daily['Close'].values,
        'low': df_daily['Low'].values,
    }


def run_backtest(df_daily: pd.DataFrame, strategy: StrategyBuilder, 
                 initial_capital: float = 100000, commission: float = 0.001) -> dict:
    """运行回测"""
    data = prepare_data(df_daily)
    
    cash = initial_capital
    shares = 0
    position = 0
    trades = []
    equity_curve = [initial_capital]
    dates = [df_daily.index[49]]
    
    for i in range(50, len(df_daily) - 1):
        close = data['close'][i]
        next_open = df_daily['Open'].iloc[i + 1]
        
        if position == 1:
            strategy.update_peak_price(close)
        
        # 卖出检查
        if position == 1:
            should_sell, reason = strategy.check_sell(data, i, df_daily)
            if should_sell:
                revenue = shares * close * (1 - commission)
                pnl = revenue - trades[-1]['cost']
                cash += revenue
                trades.append({
                    'type': 'SELL', 'price': close, 'shares': shares,
                    'pnl': pnl, 'reason': reason,
                    'date': df_daily.index[i]
                })
                shares = 0
                position = 0
                strategy.reset_position()
        
        # 买入检查
        elif position == 0:
            should_buy, reason = strategy.check_buy(data, i, df_daily)
            if should_buy and cash > 0:
                shares = int(cash * (1 - commission) / next_open)
                if shares > 0:
                    cost = shares * next_open * (1 + commission)
                    cash -= cost
                    position = 1
                    strategy.set_entry_price(next_open)
                    trades.append({
                        'type': 'BUY', 'price': next_open, 'shares': shares,
                        'cost': cost, 'reason': reason,
                        'date': df_daily.index[i+1]
                    })
        
        equity = cash + shares * close
        equity_curve.append(equity)
        dates.append(df_daily.index[i])
    
    equity_curve.append(cash + shares * data['close'][-1])
    dates.append(df_daily.index[-1])
    
    # 计算指标
    equity_curve = np.array(equity_curve)
    final_equity = equity_curve[-1]
    days = len(df_daily)
    
    total_return = (final_equity / initial_capital - 1) * 100
    annual_return = ((final_equity / initial_capital) ** (252 / days) - 1) * 100
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_drawdown = np.max(drawdown)
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winning = len([t for t in sell_trades if t.get('pnl', 0) > 0])
    win_rate = (winning / len(sell_trades) * 100) if sell_trades else 0
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades': trades,
        'equity_curve': equity_curve,
        'dates': dates,
        'final_equity': final_equity,
    }


def create_equity_chart(dates, equity_curve, trades, df_daily):
    """创建权益曲线图"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("策略权益曲线", "股价走势")
    )
    
    # 权益曲线
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_curve,
        mode='lines',
        name='策略权益',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)'
    ), row=1, col=1)
    
    # 初始资金线
    fig.add_hline(y=100000, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 股价
    fig.add_trace(go.Candlestick(
        x=df_daily.index,
        open=df_daily['Open'],
        high=df_daily['High'],
        low=df_daily['Low'],
        close=df_daily['Close'],
        name='股价',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=2, col=1)
    
    # 买卖点标记
    for trade in trades:
        if trade['type'] == 'BUY':
            fig.add_annotation(
                x=trade['date'],
                y=trade['price'],
                text="B",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#4CAF50",
                font=dict(color="#4CAF50", size=12),
                row=2, col=1
            )
        else:
            color = "#4CAF50" if trade.get('pnl', 0) > 0 else "#F44336"
            fig.add_annotation(
                x=trade['date'],
                y=trade['price'],
                text="S",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                font=dict(color=color, size=12),
                row=2, col=1
            )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    
    return fig


# ============================================================================
# 页面UI
# ============================================================================

st.title("📊 策略回测系统")
st.markdown("自由组合买入/卖出条件，测试策略表现")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 策略配置")
    
    # 股票选择
    st.subheader("📈 股票选择")
    market = st.radio("市场", ["US", "CN"], horizontal=True)
    
    if market == "US":
        default_symbols = "AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META"
    else:
        default_symbols = "600519, 000858, 002594"
    
    symbols_input = st.text_area("股票代码 (逗号分隔)", default_symbols)
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    
    days = st.slider("回测天数", 180, 1095, 730)
    
    st.markdown("---")
    
    # 买入条件
    st.subheader("🟢 买入条件")
    st.caption("满足任一条件即可买入")
    
    buy_options = {
        'blue_heima': 'BLUE≥100 + 黑马共振',
        'strong_blue': '强BLUE≥150 + 黑马',
        'double_blue': '日周双BLUE≥150',
        'bottom_peak': '底部筹码顶格峰',
        'blue_only': '超强BLUE≥200',
        'heima_only': '纯黑马/掘地',
    }
    
    selected_buy = []
    for key, label in buy_options.items():
        if st.checkbox(label, value=(key == 'blue_heima'), key=f"buy_{key}"):
            selected_buy.append(key)
    
    st.markdown("---")
    
    # 卖出条件
    st.subheader("🔴 卖出条件")
    st.caption("满足任一条件即可卖出")
    
    sell_options = {
        'kdj_overbought': 'KDJ J>90 超买',
        'chip_distribution': '筹码顶部堆积',
        'chip_with_ma': '跌破MA5+筹码异常',
        'ma_break': '跌破MA5',
        'ma_break_2day': '连续2天跌破MA5',
        'profit_target_20': '止盈20%',
        'stop_loss_8': '止损-8%',
        'trailing_stop_10': '回撤10%止损',
    }
    
    selected_sell = []
    for key, label in sell_options.items():
        default = key in ['kdj_overbought', 'chip_distribution']
        if st.checkbox(label, value=default, key=f"sell_{key}"):
            selected_sell.append(key)
    
    st.markdown("---")
    
    run_button = st.button("🚀 运行回测", type="primary", use_container_width=True)

# 主区域
if not selected_buy:
    st.warning("⚠️ 请至少选择一个买入条件")
elif not selected_sell:
    st.warning("⚠️ 请至少选择一个卖出条件")
elif run_button:
    # 显示策略配置
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**买入条件**: {', '.join([buy_options[k] for k in selected_buy])}")
    with col2:
        st.info(f"**卖出条件**: {', '.join([sell_options[k] for k in selected_sell])}")
    
    # 构建策略
    strategy = StrategyBuilder("自定义策略")
    for cond in selected_buy:
        strategy.add_buy_condition(cond)
    for cond in selected_sell:
        strategy.add_sell_condition(cond)
    
    # 运行回测
    all_results = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.text(f"正在回测 {symbol}...")
        progress.progress((idx + 1) / len(symbols))
        
        try:
            df = get_stock_data(symbol, market, days=days)
            if df is None or len(df) < 100:
                st.warning(f"⚠️ {symbol}: 数据不足，跳过")
                continue
            
            result = run_backtest(df, strategy)
            result['symbol'] = symbol
            result['df'] = df
            all_results.append(result)
        except Exception as e:
            st.error(f"❌ {symbol}: {str(e)}")
    
    progress.empty()
    status.empty()
    
    if not all_results:
        st.error("没有可用的回测结果")
    else:
        # 汇总表格
        st.subheader("📊 回测结果汇总")
        
        summary_data = []
        for r in all_results:
            summary_data.append({
                '股票': r['symbol'],
                '年化收益%': f"{r['annual_return']:.1f}%",
                '最大回撤%': f"{r['max_drawdown']:.1f}%",
                '胜率%': f"{r['win_rate']:.1f}%",
                '夏普比率': f"{r['sharpe']:.2f}",
                '交易次数': len([t for t in r['trades'] if t['type'] == 'BUY']),
                '最终权益': f"${r['final_equity']:,.0f}",
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # 平均指标
        avg_annual = np.mean([r['annual_return'] for r in all_results])
        avg_dd = np.mean([r['max_drawdown'] for r in all_results])
        avg_wr = np.mean([r['win_rate'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("平均年化", f"{avg_annual:.1f}%")
        col2.metric("平均回撤", f"{avg_dd:.1f}%")
        col3.metric("平均胜率", f"{avg_wr:.1f}%")
        col4.metric("平均夏普", f"{avg_sharpe:.2f}")
        
        st.markdown("---")
        
        # 单股详情
        st.subheader("📈 单股详情")
        
        tabs = st.tabs([r['symbol'] for r in all_results])
        
        for tab, result in zip(tabs, all_results):
            with tab:
                # 权益曲线
                fig = create_equity_chart(
                    result['dates'], 
                    result['equity_curve'], 
                    result['trades'],
                    result['df']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 交易记录
                if result['trades']:
                    st.subheader("📝 交易记录")
                    trade_data = []
                    for t in result['trades']:
                        trade_data.append({
                            '日期': t['date'].strftime('%Y-%m-%d'),
                            '类型': '买入' if t['type'] == 'BUY' else '卖出',
                            '价格': f"${t['price']:.2f}",
                            '原因': t['reason'],
                            '盈亏': f"${t.get('pnl', 0):,.2f}" if t['type'] == 'SELL' else '-'
                        })
                    st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)

else:
    # 默认显示说明
    st.markdown("""
    ### 📖 使用说明
    
    1. **选择股票**: 在左侧输入股票代码，用逗号分隔
    2. **选择买入条件**: 勾选一个或多个买入条件（满足任一即可买入）
    3. **选择卖出条件**: 勾选一个或多个卖出条件（满足任一即可卖出）
    4. **点击运行**: 点击"运行回测"按钮开始测试
    
    ---
    
    ### 🎯 推荐策略组合
    
    | 策略类型 | 买入条件 | 卖出条件 |
    |---------|---------|---------|
    | **稳健型** | BLUE≥100+黑马 | KDJ超买 + 筹码顶部堆积 |
    | **激进型** | 强BLUE≥150+黑马, 底部顶格峰 | KDJ超买 + 筹码 + 止损8% |
    | **保守型** | 日周双BLUE + 底部顶格峰 | KDJ超买 + 筹码 + 回撤止损 |
    
    ---
    
    ### 📋 买入条件说明
    
    | 条件 | 说明 |
    |------|------|
    | **BLUE≥100+黑马** | 日BLUE≥100 配合 黑马/掘地 信号 |
    | **强BLUE≥150+黑马** | 日/周BLUE≥150 配合 黑马/掘地 |
    | **日周双BLUE** | 日BLUE≥150 且 周BLUE≥150 |
    | **底部顶格峰** | 筹码密集在底部30%价格区间 |
    | **超强BLUE≥200** | 纯BLUE信号，无需黑马确认 |
    
    ### 📋 卖出条件说明
    
    | 条件 | 说明 |
    |------|------|
    | **KDJ J>90** | 技术超买信号 |
    | **筹码顶部堆积** | 顶部筹码增加+底部筹码减少 |
    | **跌破MA5+筹码** | 跌破均线配合筹码异常 |
    | **止盈20%** | 盈利达20%自动止盈 |
    | **止损-8%** | 亏损达8%自动止损 |
    | **回撤10%止损** | 从最高点回撤10%止损 |
    """)
