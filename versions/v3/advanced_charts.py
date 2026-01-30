#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级图表工具 - 扩展分析可视化

新增图表:
1. 多周期共振热力图
2. 行业资金流向图
3. 信号强度雷达图
4. 收益归因分析图
5. 相关性矩阵热力图
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


def create_multi_timeframe_heatmap(symbols_data: Dict[str, Dict]) -> go.Figure:
    """
    创建多周期共振热力图
    
    Args:
        symbols_data: {symbol: {'day_blue': 120, 'week_blue': 80, 'month_blue': 60, 'adx': 35}}
    
    Returns:
        Plotly Figure
    """
    if not symbols_data:
        return None
    
    # 准备数据
    symbols = list(symbols_data.keys())
    metrics = ['Day BLUE', 'Week BLUE', 'Month BLUE', 'ADX']
    
    values = []
    for symbol in symbols:
        data = symbols_data[symbol]
        row = [
            data.get('day_blue', 0),
            data.get('week_blue', 0),
            data.get('month_blue', 0),
            data.get('adx', 0)
        ]
        values.append(row)
    
    values = np.array(values)
    
    # 标准化到0-100
    normalized = np.zeros_like(values, dtype=float)
    for i, col in enumerate(values.T):
        max_val = max(col.max(), 1)
        normalized[:, i] = col / max_val * 100
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=normalized,
        x=metrics,
        y=symbols,
        colorscale=[
            [0, '#1a1a2e'],
            [0.3, '#16213e'],
            [0.5, '#0f3460'],
            [0.7, '#e94560'],
            [1, '#ff6b6b']
        ],
        text=values.astype(str),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="强度%")
    ))
    
    fig.update_layout(
        title="📊 多周期共振热力图",
        xaxis_title="时间周期",
        yaxis_title="股票",
        height=max(400, len(symbols) * 25 + 100)
    )
    
    return fig


def create_sector_flow_chart(sector_data: List[Dict]) -> go.Figure:
    """
    创建行业资金流向图
    
    Args:
        sector_data: [{'sector': 'Technology', 'inflow': 1.2, 'outflow': -0.8, 'net': 0.4}]
    
    Returns:
        Plotly Figure
    """
    if not sector_data:
        return None
    
    df = pd.DataFrame(sector_data)
    df = df.sort_values('net', ascending=True)
    
    fig = go.Figure()
    
    # 净流入条形图
    colors = ['#3fb950' if x > 0 else '#f85149' for x in df['net']]
    
    fig.add_trace(go.Bar(
        y=df['sector'],
        x=df['net'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}B" for x in df['net']],
        textposition='outside',
        name='净流入'
    ))
    
    fig.update_layout(
        title="🏭 行业资金净流向",
        xaxis_title="净流入 (Billion $)",
        yaxis_title="",
        height=max(400, len(df) * 30 + 100),
        showlegend=False
    )
    
    # 添加零线
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    
    return fig


def create_signal_radar_chart(signal_data: Dict) -> go.Figure:
    """
    创建信号强度雷达图
    
    Args:
        signal_data: {
            'blue_strength': 85,
            'trend_strength': 70,
            'volume_strength': 60,
            'chip_strength': 75,
            'momentum_strength': 80
        }
    
    Returns:
        Plotly Figure
    """
    categories = ['BLUE信号', '趋势强度', '成交量', '筹码形态', '动量']
    values = [
        signal_data.get('blue_strength', 0),
        signal_data.get('trend_strength', 0),
        signal_data.get('volume_strength', 0),
        signal_data.get('chip_strength', 0),
        signal_data.get('momentum_strength', 0)
    ]
    
    # 闭合雷达图
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(79, 195, 247, 0.3)',
        line=dict(color='#4fc3f7', width=2),
        name='信号强度'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title="🎯 信号强度雷达图",
        height=400
    )
    
    return fig


def create_return_attribution_chart(attribution_data: Dict) -> go.Figure:
    """
    创建收益归因分析图
    
    Args:
        attribution_data: {
            'total_return': 15.5,
            'market_return': 8.2,
            'sector_return': 3.1,
            'stock_selection': 4.2
        }
    
    Returns:
        Plotly Figure
    """
    categories = ['市场贡献', '行业贡献', '选股贡献', '总收益']
    values = [
        attribution_data.get('market_return', 0),
        attribution_data.get('sector_return', 0),
        attribution_data.get('stock_selection', 0),
        attribution_data.get('total_return', 0)
    ]
    
    colors = ['#4fc3f7', '#81c784', '#ffb74d', '#ba68c8']
    
    fig = go.Figure()
    
    # 瀑布图效果
    fig.add_trace(go.Waterfall(
        name="收益归因",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=categories,
        y=values[:-1] + [None],  # 最后一个由 measure='total' 自动计算
        textposition="outside",
        text=[f"{v:+.2f}%" for v in values[:-1]] + [f"{values[-1]:+.2f}%"],
        connector={"line": {"color": "rgba(255,255,255,0.3)"}},
        increasing={"marker": {"color": "#3fb950"}},
        decreasing={"marker": {"color": "#f85149"}},
        totals={"marker": {"color": "#ba68c8"}}
    ))
    
    fig.update_layout(
        title="📈 收益归因分析",
        yaxis_title="收益率 (%)",
        height=400,
        showlegend=False
    )
    
    return fig


def create_correlation_matrix(returns_df: pd.DataFrame, 
                               symbols: List[str] = None) -> go.Figure:
    """
    创建相关性矩阵热力图
    
    Args:
        returns_df: 收益率DataFrame (columns=symbols, rows=dates)
        symbols: 要显示的股票列表
    
    Returns:
        Plotly Figure
    """
    if symbols:
        returns_df = returns_df[symbols]
    
    # 计算相关性矩阵
    corr_matrix = returns_df.corr()
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0, '#f85149'],
            [0.5, '#21262d'],
            [1, '#3fb950']
        ],
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2).astype(str),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title="相关系数")
    ))
    
    fig.update_layout(
        title="🔗 股票相关性矩阵",
        height=max(400, len(corr_matrix) * 30 + 100),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_drawdown_chart(equity_curve: List[float], dates: List = None) -> go.Figure:
    """
    创建回撤曲线图
    
    Args:
        equity_curve: 资金曲线
        dates: 日期列表
    
    Returns:
        Plotly Figure
    """
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    
    x_axis = dates if dates else list(range(len(equity)))
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.1)
    
    # 资金曲线
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=equity,
        mode='lines',
        name='资金曲线',
        line=dict(color='#4fc3f7', width=2)
    ), row=1, col=1)
    
    # 峰值线
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=peak,
        mode='lines',
        name='峰值',
        line=dict(color='#81c784', width=1, dash='dash')
    ), row=1, col=1)
    
    # 回撤区域
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=drawdown,
        fill='tozeroy',
        name='回撤',
        line=dict(color='#f85149', width=1),
        fillcolor='rgba(248, 81, 73, 0.3)'
    ), row=2, col=1)
    
    # 最大回撤标记
    max_dd_idx = np.argmin(drawdown)
    fig.add_trace(go.Scatter(
        x=[x_axis[max_dd_idx]],
        y=[drawdown[max_dd_idx]],
        mode='markers+text',
        name='最大回撤',
        marker=dict(size=10, color='#f85149'),
        text=[f"{drawdown[max_dd_idx]:.1f}%"],
        textposition='top center'
    ), row=2, col=1)
    
    fig.update_layout(
        title="📉 资金曲线与回撤分析",
        height=500,
        legend=dict(orientation='h', y=1.1)
    )
    
    fig.update_yaxes(title_text="资金 ($)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    
    return fig


def create_performance_comparison_chart(strategies: Dict[str, List[float]],
                                         dates: List = None) -> go.Figure:
    """
    创建策略对比图
    
    Args:
        strategies: {'策略A': [1, 1.02, 1.05, ...], '策略B': [...]}
        dates: 日期列表
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    colors = ['#4fc3f7', '#81c784', '#ffb74d', '#ba68c8', '#f85149']
    
    for i, (name, values) in enumerate(strategies.items()):
        x_axis = dates if dates else list(range(len(values)))
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=values,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # 基准线
    fig.add_hline(y=1, line_dash="dash", line_color="white", opacity=0.5,
                  annotation_text="基准 (1.0)")
    
    fig.update_layout(
        title="📊 策略对比",
        xaxis_title="时间",
        yaxis_title="累计收益倍数",
        height=400,
        legend=dict(orientation='h', y=-0.15)
    )
    
    return fig


def create_volume_price_divergence_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    创建量价背离分析图
    
    Args:
        df: OHLCV DataFrame
        symbol: 股票代码
    
    Returns:
        Plotly Figure
    """
    if df is None or len(df) < 20:
        return None
    
    df = df.copy()
    
    # 计算指标
    df['price_ma20'] = df['Close'].rolling(20).mean()
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['price_trend'] = (df['Close'] - df['price_ma20']) / df['price_ma20'] * 100
    df['vol_trend'] = (df['Volume'] - df['vol_ma20']) / df['vol_ma20'] * 100
    
    # 检测背离
    df['divergence'] = 0
    df.loc[(df['price_trend'] > 0) & (df['vol_trend'] < -20), 'divergence'] = -1  # 价升量减
    df.loc[(df['price_trend'] < 0) & (df['vol_trend'] > 20), 'divergence'] = 1   # 价跌量增
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05)
    
    # K线图
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K线'
    ), row=1, col=1)
    
    # MA20
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price_ma20'],
        mode='lines',
        name='MA20',
        line=dict(color='#ffb74d', width=1)
    ), row=1, col=1)
    
    # 成交量
    colors = ['#3fb950' if c >= o else '#f85149' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color=colors,
        name='成交量',
        opacity=0.7
    ), row=2, col=1)
    
    # 量价背离指标
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['divergence'],
        mode='lines',
        name='背离信号',
        line=dict(color='#ba68c8', width=2),
        fill='tozeroy',
        fillcolor='rgba(186, 104, 200, 0.3)'
    ), row=3, col=1)
    
    fig.update_layout(
        title=f"📊 {symbol} 量价背离分析",
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation='h', y=1.05)
    )
    
    return fig


if __name__ == "__main__":
    # 测试
    print("Testing advanced chart utils...")
    
    # 测试多周期热力图
    test_data = {
        'AAPL': {'day_blue': 120, 'week_blue': 80, 'month_blue': 60, 'adx': 35},
        'NVDA': {'day_blue': 150, 'week_blue': 130, 'month_blue': 90, 'adx': 45},
        'TSLA': {'day_blue': 80, 'week_blue': 60, 'month_blue': 40, 'adx': 25}
    }
    fig = create_multi_timeframe_heatmap(test_data)
    print(f"Heatmap created: {fig is not None}")
