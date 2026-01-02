#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图表工具模块 - 用于创建K线图和信号图表"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def calculate_blue_signal(open_p, high, low, close):
    """计算BLUE信号（与扫描器中的函数一致）"""
    def REF(x, n):
        return np.roll(x, n)
    
    def EMA(x, n):
        return pd.Series(x).ewm(span=n, adjust=False).mean().values
    
    def SMA(x, n):
        return pd.Series(x).rolling(window=n).mean().values
    
    def LLV(x, n):
        return pd.Series(x).rolling(window=n).min().values
    
    def HHV(x, n):
        return pd.Series(x).rolling(window=n).max().values
    
    def IF(condition, true_value, false_value):
        return np.where(condition, true_value, false_value)
    
    OPEN = open_p
    HIGH = high
    LOW = low
    CLOSE = close
    
    VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
    VAR2 = SMA(np.abs(LOW - VAR1), 13) / SMA(np.maximum(LOW - VAR1, 0), 10)
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(LOW, 33)
    VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
    VAR6 = np.power(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    VAR31 = EMA(IF(HIGH >= VAR1, -VAR2, 0), 3)
    VAR41 = EMA(IF(HIGH >= VAR1, -VAR3, 0), 3)
    VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)
    VAR61 = np.power(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    max_value = np.nanmax(np.maximum(VAR6, np.abs(VAR61)))
    RADIO1 = 200 / max_value if max_value > 0 else 1
    BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
    return BLUE

def create_candlestick_chart(df, symbol, name, period='daily', day_blue_dates=None, week_blue_dates=None, heima_dates=None, show_volume_profile=False):
    """创建K线图，标注信号位置
    
    Args:
        df: 日线数据DataFrame
        symbol: 股票代码
        name: 股票名称
        period: 信号周期 ('daily', 'weekly', 'monthly')
        day_blue_dates: 日线BLUE信号日期
        week_blue_dates: 周线BLUE信号日期
        heima_dates: 黑马信号日期
        show_volume_profile: 是否显示筹码分布图
    """
    if show_volume_profile:
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            column_widths=[0.8, 0.2],
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}, {"rowspan": 2}],
                   [{"secondary_y": False}, None]],
            subplot_titles=(f'{symbol} ({name}) - 价格走势', '筹码分布', f'BLUE信号趋势 ({period}周期)')
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} ({name}) - 价格走势', f'BLUE信号趋势 ({period}周期)')
        )
    
    # 根据周期选择数据
    if period == 'daily':
        # 使用日线数据
        chart_df = df.copy()
        OPEN = df['Open'].values
        HIGH = df['High'].values
        LOW = df['Low'].values
        CLOSE = df['Close'].values
        BLUE = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
        signal_dates = day_blue_dates
        signal_name = '日线BLUE'
        signal_color = 'blue'
        signal_symbol = 'triangle-up'
    elif period == 'weekly':
        # 转换为周线数据
        chart_df = df.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        OPEN = chart_df['Open'].values
        HIGH = chart_df['High'].values
        LOW = chart_df['Low'].values
        CLOSE = chart_df['Close'].values
        BLUE = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
        signal_dates = week_blue_dates
        signal_name = '周线BLUE'
        signal_color = 'green'
        signal_symbol = 'square'
    elif period == 'monthly':
        # 转换为月线数据
        chart_df = df.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        OPEN = chart_df['Open'].values
        HIGH = chart_df['High'].values
        LOW = chart_df['Low'].values
        CLOSE = chart_df['Close'].values
        BLUE = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
        signal_dates = week_blue_dates  # 月线暂时用周线数据
        signal_name = '月线BLUE'
        signal_color = 'purple'
        signal_symbol = 'diamond'
    else:
        chart_df = df.copy()
        OPEN = df['Open'].values
        HIGH = df['High'].values
        LOW = df['Low'].values
        CLOSE = df['Close'].values
        BLUE = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
        signal_dates = day_blue_dates
        signal_name = '日线BLUE'
        signal_color = 'blue'
        signal_symbol = 'triangle-up'
    
    BLUE_D = BLUE
    
    # 1. K线图
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='价格'
        ),
        row=1, col=1
    )
    
    # 添加筹码分布图 (Volume Profile)
    if show_volume_profile and not chart_df.empty:
        try:
            # 计算价格区间
            price_min = chart_df['Low'].min()
            price_max = chart_df['High'].max()
            price_range = price_max - price_min
            
            # 创建价格分箱 (50个区间)
            bins = 50
            bin_size = price_range / bins
            
            # 初始化每个分箱的成交量
            volume_profile = np.zeros(bins)
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # 简单算法：将每天的成交量分配到该天的价格区间内
            # 改进算法：假设成交量均匀分布在当日(High-Low)区间内
            for idx, row in chart_df.iterrows():
                day_high = row['High']
                day_low = row['Low']
                day_vol = row['Volume']
                
                if day_high == day_low:
                    # 只有单一价格，直接归入对应bin
                    bin_idx = int((day_high - price_min) / bin_size)
                    bin_idx = min(bin_idx, bins - 1)
                    volume_profile[bin_idx] += day_vol
                else:
                    # 找出当日价格覆盖了哪些bin
                    start_bin = int((day_low - price_min) / bin_size)
                    end_bin = int((day_high - price_min) / bin_size)
                    end_bin = min(end_bin, bins - 1)
                    
                    # 简单均匀分配
                    if start_bin == end_bin:
                         volume_profile[start_bin] += day_vol
                    else:
                        vol_per_bin = day_vol / (end_bin - start_bin + 1)
                        for b in range(start_bin, end_bin + 1):
                            volume_profile[b] += vol_per_bin
            
            # 绘制横向柱状图
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            
            fig.add_trace(
                go.Bar(
                    x=volume_profile,
                    y=bin_centers,
                    orientation='h',
                    name='筹码分布',
                    marker=dict(
                        color='rgba(0, 0, 255, 0.3)',  # 加深颜色，用半透明蓝色
                        line=dict(width=1, color='rgba(0, 0, 255, 0.5)') # 增加边框
                    ),
                    showlegend=False,
                    hoverinfo='y+x'
                ),
                row=1, col=2
            )
        except Exception as e:
            print(f"Error calculating volume profile: {e}")

    # 标注信号（根据选择的周期）
    if signal_dates:
        try:
            if isinstance(signal_dates, str):
                dates_data = json.loads(signal_dates)
            else:
                dates_data = signal_dates
            
            if dates_data and len(dates_data) > 0:
                # 提取日期列表
                if isinstance(dates_data[0], dict):
                    signal_dates_list = [item['date'] for item in dates_data]
                else:
                    signal_dates_list = dates_data
                
                # 在K线图上标注
                for date_str in signal_dates_list:
                    try:
                        date = pd.to_datetime(date_str)
                        # 根据周期调整日期匹配
                        if period == 'weekly':
                            # 找到该日期所在的周
                            week_start = date - pd.Timedelta(days=date.weekday())
                            if week_start in chart_df.index:
                                price = chart_df.loc[week_start, 'Close']
                                date = week_start
                            elif date in chart_df.index:
                                price = chart_df.loc[date, 'Close']
                            else:
                                continue
                        elif period == 'monthly':
                            # 找到该日期所在的月
                            month_start = date.replace(day=1)
                            if month_start in chart_df.index:
                                price = chart_df.loc[month_start, 'Close']
                                date = month_start
                            elif date in chart_df.index:
                                price = chart_df.loc[date, 'Close']
                            else:
                                continue
                        else:
                            # 日线
                            if date in chart_df.index:
                                price = chart_df.loc[date, 'Close']
                            else:
                                continue
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[date],
                                y=[price],
                                mode='markers',
                                marker=dict(
                                    symbol=signal_symbol,
                                    size=15,
                                    color=signal_color,
                                    line=dict(width=2, color=f'dark{signal_color}')
                                ),
                                name=signal_name,
                                showlegend=False,
                                hovertemplate=f'{signal_name}信号<br>{date_str}<br>价格: {price:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    except Exception as e:
                        pass
        except:
            pass
    
    # 2. BLUE信号趋势图
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=BLUE_D,
            mode='lines',
            name=f'BLUE信号({period})',
            line=dict(color=signal_color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba(0, 100, 255, 0.2)'
        ),
        row=2, col=1
    )
    
    # 添加BLUE=100的参考线
    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                  annotation_text="BLUE=100", row=2, col=1)
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text=f"{symbol} ({name}) - 价格与信号分析",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 更新y轴标签
    fig.update_yaxes(title_text="价格", row=1, col=1)
    if show_volume_profile:
        fig.update_xaxes(title_text="成交量", showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2) # 隐藏右侧Y轴刻度，因为和左侧对齐
    fig.update_yaxes(title_text="BLUE信号值", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    
    return fig

