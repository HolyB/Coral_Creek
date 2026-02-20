"""
策略对比回测页面
================
比较不同交易策略的历史表现
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="策略对比",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 策略对比回测")
st.info("对比不同买入策略的历史表现，找出最优组合")

# ============================================================================
# 策略定义
# ============================================================================

STRATEGIES = {
    'pure_blue_100': {
        'name': 'BLUE > 100',
        'desc': '日线 BLUE 突破 100 买入',
        'color': '#2196F3',
        'buy_condition': lambda df: df['blue'] > 100,
        'params': {'blue_threshold': 100}
    },
    'pure_blue_150': {
        'name': 'BLUE > 150',
        'desc': '日线 BLUE 突破 150 买入 (保守)',
        'color': '#1976D2',
        'buy_condition': lambda df: df['blue'] > 150,
        'params': {'blue_threshold': 150}
    },
    'blue_heima': {
        'name': 'BLUE + 黑马',
        'desc': 'BLUE > 100 且有黑马信号',
        'color': '#4CAF50',
        'buy_condition': lambda df: (df['blue'] > 100) & (df['heima']),
        'params': {'require_heima': True}
    },
    'blue_week': {
        'name': '日周共振',
        'desc': '日线 + 周线同时 BLUE > 100',
        'color': '#FF9800',
        'buy_condition': lambda df: (df['blue'] > 100) & (df.get('week_blue', 0) > 100),
        'params': {'require_week_blue': True}
    },
    'blue_kdj': {
        'name': 'BLUE + KDJ',
        'desc': 'BLUE > 100 且 J < 20 (超卖)',
        'color': '#9C27B0',
        'buy_condition': lambda df: (df['blue'] > 100) & (df.get('kdj_j', 50) < 20),
        'params': {'require_kdj': True}
    }
}

# ============================================================================
# 回测核心逻辑
# ============================================================================

def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    from indicator_utils import (
        calculate_blue_signal_series,
        calculate_heima_signal_series,
        calculate_kdj_series
    )
    
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # BLUE
    df['blue'] = calculate_blue_signal_series(opens, highs, lows, closes)
    
    # 黑马
    heima, juedi = calculate_heima_signal_series(highs, lows, closes, opens)
    df['heima'] = heima
    
    # KDJ
    k, d, j = calculate_kdj_series(highs, lows, closes)
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j
    
    # 周线 BLUE (简化版: 用5日 BLUE 均值代替)
    df['week_blue'] = pd.Series(df['blue']).rolling(5).mean().values
    
    return df


def run_single_backtest(df: pd.DataFrame, strategy_key: str, 
                        hold_days: int = 10, stop_loss: float = 0.08) -> dict:
    """运行单个策略回测"""
    
    strategy = STRATEGIES[strategy_key]
    
    try:
        buy_condition = strategy['buy_condition'](df)
    except:
        buy_condition = df['blue'] > strategy['params'].get('blue_threshold', 100)
    
    trades = []
    equity = [1.0]  # 初始净值
    
    i = 0
    while i < len(df) - hold_days:
        if buy_condition.iloc[i]:
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
            
            # 寻找出场点
            exit_idx = i + hold_days
            exit_price = entry_price
            exit_reason = 'time'
            
            for j in range(i + 1, min(i + hold_days + 1, len(df))):
                current_price = df['Close'].iloc[j]
                
                # 止损
                if current_price < entry_price * (1 - stop_loss):
                    exit_idx = j
                    exit_price = current_price
                    exit_reason = 'stop_loss'
                    break
                
                # 止盈 (BLUE 下降到 50 以下)
                if df['blue'].iloc[j] < 50 and j > i + 3:
                    exit_idx = j
                    exit_price = current_price
                    exit_reason = 'signal_exit'
                    break
                    
                exit_price = current_price
            
            pnl_pct = (exit_price - entry_price) / entry_price
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[exit_idx] if exit_idx < len(df) else df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason
            })
            
            # 更新净值
            new_equity = equity[-1] * (1 + pnl_pct)
            equity.append(new_equity)
            
            # 跳过持有期
            i = exit_idx + 1
        else:
            i += 1
    
    # 计算统计
    if not trades:
        return {
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'sharpe': 0,
            'win_rate': 0,
            'total_trades': 0,
            'equity_curve': [1.0]
        }
    
    total_return = (equity[-1] - 1) * 100
    years = len(df) / 252
    annual_return = ((equity[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # 最大回撤
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # 夏普比率 (简化)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else [0]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # 胜率
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'equity_curve': equity,
        'trades': trades
    }


# ============================================================================
# UI
# ============================================================================

# 参数设置
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    symbol = st.text_input("股票代码", value="NVDA", help="输入美股代码").upper()

with col2:
    days = st.slider("回测天数", 90, 1095, 365, 30)

with col3:
    hold_days = st.slider("持有天数", 5, 30, 10)

# 策略选择
st.markdown("---")
st.markdown("### 选择对比策略")

selected = []
cols = st.columns(len(STRATEGIES))
for i, (key, strategy) in enumerate(STRATEGIES.items()):
    with cols[i]:
        if st.checkbox(strategy['name'], value=(key in ['pure_blue_100', 'blue_heima']), 
                       help=strategy['desc'], key=f"strat_{key}"):
            selected.append(key)

# 运行按钮
st.markdown("---")
run_btn = st.button("🚀 运行策略对比", type="primary", use_container_width=True)

if run_btn and symbol and selected:
    with st.spinner(f"正在对 {symbol} 运行 {len(selected)} 个策略回测..."):
        try:
            from data_fetcher import get_us_stock_data
            
            # 获取数据
            df = get_us_stock_data(symbol, days=days)
            
            if df is None or len(df) < 100:
                st.error(f"无法获取 {symbol} 的数据，请确认代码正确")
            else:
                # 计算信号
                df = calculate_signals(df)
                
                # 运行回测
                results = {}
                for key in selected:
                    results[key] = run_single_backtest(df, key, hold_days=hold_days)
                
                # ==================== 显示结果 ====================
                
                st.markdown("---")
                st.markdown("### 📊 回测结果对比")
                
                # 对比表格
                comparison_data = []
                for key in selected:
                    r = results[key]
                    comparison_data.append({
                        '策略': STRATEGIES[key]['name'],
                        '总收益': f"{r['total_return']:+.1f}%",
                        '年化收益': f"{r['annual_return']:+.1f}%",
                        '最大回撤': f"{r['max_drawdown']:.1f}%",
                        '夏普比率': f"{r['sharpe']:.2f}",
                        '胜率': f"{r['win_rate']:.0f}%",
                        '交易次数': r['total_trades']
                    })
                
                df_compare = pd.DataFrame(comparison_data)
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
                # 找出最佳
                if results:
                    best_return = max(results.items(), key=lambda x: x[1]['total_return'])
                    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"📈 **最高收益**: {STRATEGIES[best_return[0]]['name']} ({best_return[1]['total_return']:+.1f}%)")
                    with col2:
                        st.info(f"⚖️ **最佳风险调整**: {STRATEGIES[best_sharpe[0]]['name']} (Sharpe: {best_sharpe[1]['sharpe']:.2f})")
                
                # 权益曲线
                st.markdown("---")
                st.markdown("### 📈 权益曲线对比")
                
                fig = go.Figure()
                
                for key in selected:
                    equity = results[key]['equity_curve']
                    fig.add_trace(go.Scatter(
                        y=equity,
                        mode='lines',
                        name=STRATEGIES[key]['name'],
                        line=dict(color=STRATEGIES[key]['color'], width=2)
                    ))
                
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                             annotation_text="起始净值")
                
                fig.update_layout(
                    title=f"{symbol} 策略对比 ({days}天)",
                    xaxis_title="交易次数",
                    yaxis_title="账户净值",
                    height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 交易明细
                st.markdown("---")
                st.markdown("### 📋 交易明细")
                
                detail_strategy = st.selectbox(
                    "选择策略查看明细",
                    options=selected,
                    format_func=lambda x: STRATEGIES[x]['name']
                )
                
                if detail_strategy and results[detail_strategy].get('trades'):
                    trades = results[detail_strategy]['trades']
                    
                    trade_data = []
                    for t in trades[-20:]:  # 最近20笔
                        pnl_emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
                        trade_data.append({
                            '': pnl_emoji,
                            '入场日期': str(t['entry_date'])[:10],
                            '出场日期': str(t['exit_date'])[:10],
                            '入场价': f"${t['entry_price']:.2f}",
                            '出场价': f"${t['exit_price']:.2f}",
                            '盈亏': f"{t['pnl_pct']*100:+.2f}%",
                            '出场原因': {'time': '持有到期', 'stop_loss': '止损', 'signal_exit': '信号'}[t['exit_reason']]
                        })
                    
                    st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
                else:
                    st.info("该策略没有产生交易")
                    
        except Exception as e:
            st.error(f"回测失败: {e}")
            import traceback
            st.code(traceback.format_exc())

elif run_btn and not selected:
    st.warning("请至少选择一个策略")

# 页脚
st.markdown("---")
st.caption("💡 提示: 回测结果仅供参考，过去表现不代表未来收益")
