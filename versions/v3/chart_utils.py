#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图表工具模块 - 用于创建K线图和信号图表"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# 统一使用 indicator_utils 的 BLUE 计算
from indicator_utils import calculate_blue_signal_series

def calculate_blue_signal(open_p, high, low, close):
    """计算BLUE信号（统一使用 indicator_utils 版本）"""
    return calculate_blue_signal_series(open_p, high, low, close)


def create_candlestick_chart_dynamic(df_full, df_for_vp, symbol, name, period='daily', 
                                     day_blue_dates=None, week_blue_dates=None, heima_dates=None, 
                                     show_volume_profile=False, stop_loss_price=None, highlight_date=None):
    """创建带动态筹码分布的K线图
    
    Args:
        df_full: 完整数据用于K线显示
        df_for_vp: 截止选中日期的数据用于筹码分布计算
        symbol: 股票代码
        name: 股票名称
        period: 信号周期 ('daily', 'weekly', 'monthly')
        show_volume_profile: 是否显示筹码分布图
        stop_loss_price: 止损价格（可选）
        highlight_date: 高亮显示的日期
    """
    if show_volume_profile:
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.12,  # 增加垂直间距
            horizontal_spacing=0.03,  # 减少水平间距
            column_widths=[0.78, 0.22],  # 调整列宽
            row_heights=[0.72, 0.28],
            specs=[[{"secondary_y": False}, {"rowspan": 2}],
                   [{"secondary_y": False}, None]],
            subplot_titles=('', '', '')  # 移除默认标题，手动添加
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.72, 0.28],
            subplot_titles=('', '')
        )
    
    chart_df = df_full
    
    # 计算 BLUE 信号
    OPEN = chart_df['Open'].values
    HIGH = chart_df['High'].values
    LOW = chart_df['Low'].values
    CLOSE = chart_df['Close'].values
    BLUE = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
    
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
    
    # 高亮选中的日期 (用 scatter 标记代替 vline)
    if highlight_date is not None:
        try:
            highlight_price = df_full.loc[highlight_date, 'High'] if highlight_date in df_full.index else df_full['High'].iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[highlight_date],
                    y=[highlight_price * 1.02],  # 稍微高于最高价
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=12, color='orange'),
                    text=['📍'],
                    textposition='top center',
                    name='选中日期',
                    showlegend=False,
                    hovertemplate=f'选中日期<extra></extra>'
                ),
                row=1, col=1
            )
        except:
            pass  # 忽略高亮错误
    
    # 添加筹码分布图 (Volume Profile) - 使用截止选中日期的数据
    # 算法改进：加入时间衰减 + 三角分布（更接近通达信/同花顺）
    if show_volume_profile and not df_for_vp.empty:
        try:
            # 计算价格区间 (基于完整数据，保持Y轴一致)
            price_min = df_full['Low'].min()
            price_max = df_full['High'].max()
            price_range = price_max - price_min
            
            # 创建价格分箱 (70个区间，更精细)
            bins = 70
            bin_size = price_range / bins if price_range > 0 else 1
            
            # 初始化每个分箱的成交量
            volume_profile = np.zeros(bins)
            price_bins = np.linspace(price_min, price_max, bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            
            # === 核心改进：时间衰减 + 三角分布 ===
            total_days = len(df_for_vp)
            decay_factor = 0.97  # 每天衰减 3%，约 60 天后权重降为 ~16%
            
            for i, (idx, row) in enumerate(df_for_vp.iterrows()):
                day_high = row['High']
                day_low = row['Low']
                day_close = row['Close']
                day_vol = row['Volume']
                
                # 时间衰减：越近的日期权重越高
                days_ago = total_days - 1 - i
                time_weight = decay_factor ** days_ago
                weighted_vol = day_vol * time_weight
                
                if day_high == day_low or bin_size == 0:
                    bin_idx = int((day_close - price_min) / bin_size)
                    bin_idx = min(max(bin_idx, 0), bins - 1)
                    volume_profile[bin_idx] += weighted_vol
                else:
                    # 三角分布：成交量集中在收盘价附近
                    start_bin = int((day_low - price_min) / bin_size)
                    end_bin = int((day_high - price_min) / bin_size)
                    start_bin = max(start_bin, 0)
                    end_bin = min(end_bin, bins - 1)
                    close_bin = int((day_close - price_min) / bin_size)
                    close_bin = min(max(close_bin, start_bin), end_bin)
                    
                    if start_bin == end_bin:
                        volume_profile[start_bin] += weighted_vol
                    else:
                        # 三角分布权重：离收盘价越近权重越高
                        for b in range(start_bin, end_bin + 1):
                            dist_to_close = abs(b - close_bin)
                            max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                            # 线性衰减：收盘价处权重=1，边缘权重=0.2
                            weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                            volume_profile[b] += weighted_vol * weight
            
            # 归一化（因为三角分布会改变总量）
            if np.sum(volume_profile) > 0:
                volume_profile = volume_profile / np.sum(volume_profile) * np.sum([r['Volume'] for _, r in df_for_vp.iterrows()])
            
            # 计算总成交量和每个价位的占比
            total_volume = np.sum(volume_profile)
            volume_pct = (volume_profile / total_volume * 100) if total_volume > 0 else np.zeros(bins)
            
            # 计算累计百分比 (该价位及以下所有筹码的占比)
            cumulative_pct = np.cumsum(volume_profile) / total_volume * 100 if total_volume > 0 else np.zeros(bins)
            
            # 寻找 POC (Point of Control)
            max_vol_idx = np.argmax(volume_profile)
            poc_price = bin_centers[max_vol_idx]
            
            # 颜色编码: 使用选中日期的收盘价
            current_close = df_for_vp['Close'].iloc[-1] if not df_for_vp.empty else df_full['Close'].iloc[-1]
            bar_colors = []
            
            for price in bin_centers:
                if abs(price - poc_price) < (bin_size / 2):
                    bar_colors.append('rgba(255, 69, 0, 0.8)')  # 橙红色 (POC)
                elif price < current_close:
                    bar_colors.append('rgba(50, 205, 50, 0.6)')  # 绿色 (获利盘)
                else:
                    bar_colors.append('rgba(220, 20, 60, 0.6)')  # 红色 (套牢盘)

            # 组合 customdata: [占比, 累计占比]
            custom_data = np.column_stack([volume_pct, cumulative_pct])
            
            fig.add_trace(
                go.Bar(
                    x=volume_profile,
                    y=bin_centers,
                    orientation='h',
                    name='筹码分布',
                    marker=dict(
                        color=bar_colors,
                        line=dict(width=0.5, color='rgba(0,0,0,0.1)')
                    ),
                    showlegend=False,
                    customdata=custom_data,
                    hovertemplate='价格: $%{y:.2f}<br>堆积量: %{x:,.0f}<br>占比: %{customdata[0]:.2f}%<br>下方筹码: %{customdata[1]:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 在主图画出 POC 延伸线
            fig.add_hline(
                y=poc_price, 
                line_dash="dot", 
                line_color="rgba(255, 69, 0, 0.6)", 
                line_width=1.5,
                annotation_text=f"POC", 
                annotation_position="left",
                annotation=dict(font_size=9, font_color="rgba(255, 69, 0, 0.8)"),
                row=1, col=1
            )
            
            # === 计算关键筹码指标 ===
            total_vol = np.sum(volume_profile)
            
            # 1. 获利盘 / 套牢盘
            profit_vol = 0
            trapped_vol = 0
            for i, p in enumerate(bin_centers):
                if p < current_close:
                    profit_vol += volume_profile[i]
                else:
                    trapped_vol += volume_profile[i]
            
            profit_ratio = profit_vol / total_vol if total_vol > 0 else 0
            trapped_ratio = trapped_vol / total_vol if total_vol > 0 else 0
            
            # 2. 筹码集中度 (POC ±10% 区间)
            near_poc_vol = 0
            poc_range_low = poc_price * 0.9
            poc_range_high = poc_price * 1.1
            for i, p in enumerate(bin_centers):
                if poc_range_low <= p <= poc_range_high:
                    near_poc_vol += volume_profile[i]
            concentration = near_poc_vol / total_vol if total_vol > 0 else 0
            
            # 3. 加权平均成本
            avg_cost = np.sum(bin_centers * volume_profile) / total_vol if total_vol > 0 else current_close
            
            # 4. 90% 成本区间 (去掉最高5%和最低5%的筹码)
            cumsum = np.cumsum(volume_profile)
            cumsum_pct = cumsum / total_vol if total_vol > 0 else cumsum
            
            low_5_idx = np.searchsorted(cumsum_pct, 0.05)
            high_95_idx = np.searchsorted(cumsum_pct, 0.95)
            low_5_idx = max(0, min(low_5_idx, bins - 1))
            high_95_idx = max(0, min(high_95_idx, bins - 1))
            
            cost_90_low = bin_centers[low_5_idx]
            cost_90_high = bin_centers[high_95_idx]
            cost_90_range = cost_90_high - cost_90_low
            
            # 5. 距离 POC 的百分比
            dist_to_poc_pct = (current_close - poc_price) / poc_price * 100 if poc_price > 0 else 0
            
            # 6. 压力位和支撑位 (找次高峰)
            # 在当前价格上方找最大的筹码堆积 = 压力位
            # 在当前价格下方找最大的筹码堆积 = 支撑位
            current_bin = int((current_close - price_min) / bin_size)
            current_bin = max(0, min(current_bin, bins - 1))
            
            # 上方压力
            resistance_price = None
            if current_bin < bins - 1:
                above_profile = volume_profile[current_bin + 1:]
                if len(above_profile) > 0 and np.max(above_profile) > 0:
                    above_max_idx = np.argmax(above_profile)
                    resistance_price = bin_centers[current_bin + 1 + above_max_idx]
            
            # 下方支撑
            support_price = None
            if current_bin > 0:
                below_profile = volume_profile[:current_bin]
                if len(below_profile) > 0 and np.max(below_profile) > 0:
                    below_max_idx = np.argmax(below_profile)
                    support_price = bin_centers[below_max_idx]
            
            # 形态判断
            pattern_desc = "普通分布"
            if concentration > 0.6:
                pattern_desc = "单峰密集 (主力控盘)"
            elif concentration > 0.4:
                pattern_desc = "相对集中"
            elif concentration < 0.25:
                pattern_desc = "多峰发散 (筹码分散)"
            
            # === 底部顶格峰检测 (数据驱动版) ===
            # 基于 32 只股票的真实数据分析确定阈值
            # - 单峰最大占比范围: 2.3% - 8.0% (中位数 4.2%)
            # - 底部堆积 >50% 的股票约 19%
            
            # 1. 计算最大单根筹码占比
            max_chip_pct = np.max(volume_profile) / total_vol * 100 if total_vol > 0 else 0
            
            # 2. 底部区域筹码占比 (底部 30% 价格区间的筹码)
            bottom_30_price = price_min + (price_max - price_min) * 0.30
            bottom_chip_ratio = 0
            for i, p in enumerate(bin_centers):
                if p <= bottom_30_price:
                    bottom_chip_ratio += volume_profile[i]
            bottom_chip_ratio = bottom_chip_ratio / total_vol if total_vol > 0 else 0
            
            # 3. POC 位置 (0-100%, 0=最底, 100=最顶)
            poc_position = (poc_price - price_min) / (price_max - price_min) * 100 if price_max > price_min else 50
            
            # === 判定规则 (V2 严格版, 数据验证) ===
            # 强信号: POC 位置 <30% + 底部筹码 >50% + 单峰 >5%
            is_strong_bottom_peak = (poc_position < 30) and (bottom_chip_ratio > 0.50) and (max_chip_pct > 5)
            
            # 普通信号: POC 位置 <35% + 底部筹码 >35%
            is_bottom_peak = (poc_position < 35) and (bottom_chip_ratio > 0.35)
            
            # 更新形态描述
            if is_strong_bottom_peak:
                pattern_desc = f"🔥 底部顶格峰 (POC:{poc_position:.0f}% 底部:{bottom_chip_ratio*100:.0f}%)"
            elif is_bottom_peak:
                pattern_desc = f"📍 底部密集 (POC:{poc_position:.0f}% 底部:{bottom_chip_ratio*100:.0f}%)"
            
            # 买点评估 (数据驱动版)
            buy_signal_strength = ""
            if is_strong_bottom_peak:
                buy_signal_strength = "🔥 强势买点 (底部顶格峰)"
            elif is_bottom_peak:
                buy_signal_strength = "🟡 底部吸筹 (可关注)"
            elif profit_ratio > 0.90 and concentration > 0.5:
                buy_signal_strength = "🟢 极佳买点"
            elif profit_ratio > 0.80 and concentration > 0.4:
                buy_signal_strength = "🟡 较好买点"
            elif profit_ratio < 0.30:
                buy_signal_strength = "🔴 谨慎 (套牢盘重)"
            else:
                buy_signal_strength = "⚪ 中性"
            
            # 将筹码指标存储在 fig 的 layout 中供外部使用
            fig._chip_analysis = {
                'profit_ratio': profit_ratio,
                'trapped_ratio': trapped_ratio,
                'concentration': concentration,
                'avg_cost': avg_cost,
                'poc_price': poc_price,
                'cost_90_low': cost_90_low,
                'cost_90_high': cost_90_high,
                'cost_90_range': cost_90_range,
                'dist_to_poc_pct': dist_to_poc_pct,
                'support_price': support_price,
                'resistance_price': resistance_price,
                'pattern_desc': pattern_desc,
                'buy_signal_strength': buy_signal_strength,
                'current_close': current_close,
                # 底部顶格峰指标 (数据驱动版)
                'is_bottom_peak': is_bottom_peak,
                'is_strong_bottom_peak': is_strong_bottom_peak,
                'bottom_chip_ratio': bottom_chip_ratio,
                'poc_position': poc_position,  # POC 位置 (0-100%)
                'max_chip_pct': max_chip_pct,  # 单峰最大占比
            }
            
        except Exception as e:
            print(f"Error calculating volume profile: {e}")

    # 2. BLUE信号趋势图
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=BLUE,
            mode='lines',
            name=f'BLUE信号({period})',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ),
        row=2, col=1
    )
    
    # 添加BLUE=100的参考线 (简化标注)
    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                  annotation_text="100", 
                  annotation_position="left",
                   annotation=dict(font_size=10),
                  row=2, col=1)
    
    # 添加止损线 (如果提供)
    if stop_loss_price is not None and period == 'daily':
        fig.add_hline(y=stop_loss_price, line_dash="dot", line_color="red", line_width=2,
                     annotation_text=f"SL ${stop_loss_price:.2f}", 
                     annotation_position="left",
                     annotation=dict(font_size=9),
                     row=1, col=1)

    # === 添加黑马信号标记 ===
    try:
        # 计算黑马信号
        from indicator_utils import calculate_heima_signal_series
        heima_signal, juedi_signal = calculate_heima_signal_series(HIGH, LOW, CLOSE, OPEN)
        
        # 找出黑马信号的日期
        heima_dates_calc = chart_df.index[heima_signal].tolist()
        
        if len(heima_dates_calc) > 0:
            # 获取黑马信号日期对应的价格 (标记在最低价下方)
            heima_prices = [chart_df.loc[d, 'Low'] for d in heima_dates_calc if d in chart_df.index]
            heima_dates_valid = [d for d in heima_dates_calc if d in chart_df.index]
            
            if heima_dates_valid:
                fig.add_trace(
                    go.Scatter(
                        x=heima_dates_valid,
                        y=[p * 0.98 for p in heima_prices],  # 稍微低于最低价
                        mode='markers+text',
                        marker=dict(symbol='triangle-up', size=12, color='#a371f7'),
                        text=['🐴'] * len(heima_dates_valid),
                        textposition='bottom center',
                        name='黑马信号',
                        showlegend=True,
                        hovertemplate='黑马信号<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
    except Exception as e:
        pass  # 忽略黑马信号计算错误

    # === 优化布局 - 改进鼠标联动 ===
    fig.update_layout(
        height=750,
        title=dict(
            text=f"<b>{symbol}</b> - {name}",
            font=dict(size=16),
            x=0.02,
            xanchor='left'
        ),
        xaxis_rangeslider_visible=False,
        # 改用 closest 模式，让每个子图独立响应
        hovermode='closest',
        # 添加十字准线
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12,
            font_family="monospace"
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.01, 
            xanchor="right", 
            x=0.75,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        font=dict(size=11)
    )
    
    # === 添加十字准线 (spike lines) ===
    # 主图 Y轴 - 添加水平准线
    fig.update_yaxes(
        title_text="", 
        tickfont=dict(size=10),
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='rgba(128, 128, 128, 0.5)',
        spikedash='dot',
        row=1, col=1
    )
    
    # 主图 X轴 - 添加垂直准线
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='rgba(128, 128, 128, 0.5)',
        spikedash='dot',
        tickfont=dict(size=9),
        row=1, col=1
    )
    
    # 筹码分布 - 匹配 Y 轴范围
    if show_volume_profile:
        # 获取主图 Y 轴范围
        y_min = df_full['Low'].min() * 0.98
        y_max = df_full['High'].max() * 1.02
        
        fig.update_xaxes(
            title_text="", 
            showticklabels=False, 
            row=1, col=2
        )
        fig.update_yaxes(
            showticklabels=False,
            # 关键：匹配主图的 Y 轴范围，实现联动
            matches='y',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikecolor='rgba(128, 128, 128, 0.5)',
            spikedash='dot',
            row=1, col=2
        )
    
    # BLUE 信号图
    fig.update_yaxes(
        title_text="BLUE", 
        title_font=dict(size=10),
        tickfont=dict(size=9),
        row=2, col=1
    )
    fig.update_xaxes(
        title_text="", 
        tickfont=dict(size=9),
        row=2, col=1
    )
    
    # 主图 X轴
    fig.update_xaxes(
        tickfont=dict(size=9),
        row=1, col=1
    )
    
    return fig


def analyze_chip_flow(df, lookback_days=20, decay_factor=0.97):
    """
    分析筹码流动，检测主力建仓/出货
    
    Args:
        df: 包含 OHLCV 的 DataFrame
        lookback_days: 对比的天数 (默认20天)
        decay_factor: 时间衰减因子
    
    Returns:
        dict: 筹码流动分析结果
    """
    if len(df) < lookback_days + 30:
        return None
    
    # 计算筹码分布的辅助函数
    def calc_chip_distribution(data, price_min, price_max, bins=70):
        bin_size = (price_max - price_min) / bins if price_max > price_min else 1
        volume_profile = np.zeros(bins)
        price_bins = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        total_days = len(data)
        for i, (idx, row) in enumerate(data.iterrows()):
            day_high = row['High']
            day_low = row['Low']
            day_close = row['Close']
            day_vol = row['Volume']
            
            days_ago = total_days - 1 - i
            time_weight = decay_factor ** days_ago
            weighted_vol = day_vol * time_weight
            
            if day_high == day_low or bin_size == 0:
                bin_idx = int((day_close - price_min) / bin_size)
                bin_idx = min(max(bin_idx, 0), bins - 1)
                volume_profile[bin_idx] += weighted_vol
            else:
                start_bin = int((day_low - price_min) / bin_size)
                end_bin = int((day_high - price_min) / bin_size)
                start_bin = max(start_bin, 0)
                end_bin = min(end_bin, bins - 1)
                close_bin = int((day_close - price_min) / bin_size)
                close_bin = min(max(close_bin, start_bin), end_bin)
                
                if start_bin == end_bin:
                    volume_profile[start_bin] += weighted_vol
                else:
                    for b in range(start_bin, end_bin + 1):
                        dist_to_close = abs(b - close_bin)
                        max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                        weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                        volume_profile[b] += weighted_vol * weight
        
        return volume_profile, bin_centers
    
    # 使用统一的价格区间
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bins = 70
    bin_centers = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
    
    # 计算 N 天前的筹码分布
    df_past = df.iloc[:-lookback_days]
    past_profile, _ = calc_chip_distribution(df_past, price_min, price_max, bins)
    
    # 计算当前的筹码分布
    df_current = df
    current_profile, _ = calc_chip_distribution(df_current, price_min, price_max, bins)
    
    # 归一化
    past_total = np.sum(past_profile)
    current_total = np.sum(current_profile)
    if past_total > 0:
        past_profile = past_profile / past_total
    if current_total > 0:
        current_profile = current_profile / current_total
    
    # 计算变化
    chip_change = current_profile - past_profile
    
    # 当前价格
    current_close = df['Close'].iloc[-1]
    current_bin = int((current_close - price_min) / ((price_max - price_min) / bins))
    current_bin = max(0, min(current_bin, bins - 1))
    
    # 分析指标
    # 1. 低位筹码增加量 (当前价格下方 20% 区间)
    low_threshold = current_close * 0.8
    low_bins = bin_centers < low_threshold
    low_chip_increase = np.sum(chip_change[low_bins]) * 100  # 百分比
    
    # 2. 高位筹码减少量 (当前价格上方 20% 区间)
    high_threshold = current_close * 1.2
    high_bins = bin_centers > high_threshold
    high_chip_decrease = -np.sum(chip_change[high_bins]) * 100  # 正数表示减少
    
    # 3. 当前价格附近筹码变化 (±10%)
    near_low = current_close * 0.9
    near_high = current_close * 1.1
    near_bins = (bin_centers >= near_low) & (bin_centers <= near_high)
    near_chip_change = np.sum(chip_change[near_bins]) * 100
    
    # 4. 平均成本变化
    past_avg_cost = np.sum(bin_centers * past_profile) if past_total > 0 else current_close
    current_avg_cost = np.sum(bin_centers * current_profile) if current_total > 0 else current_close
    cost_change = current_avg_cost - past_avg_cost
    cost_change_pct = cost_change / past_avg_cost * 100 if past_avg_cost > 0 else 0
    
    # 5. 集中度变化
    def calc_concentration(profile, centers, ref_price):
        total = np.sum(profile)
        if total == 0:
            return 0
        poc_idx = np.argmax(profile)
        poc_price = centers[poc_idx]
        near_poc = 0
        for i, p in enumerate(centers):
            if poc_price * 0.9 <= p <= poc_price * 1.1:
                near_poc += profile[i]
        return near_poc / total
    
    past_concentration = calc_concentration(past_profile, bin_centers, current_close)
    current_concentration = calc_concentration(current_profile, bin_centers, current_close)
    concentration_change = (current_concentration - past_concentration) * 100
    
    # 判断主力行为
    action = "观望"
    action_emoji = "⚪"
    action_desc = ""
    
    # 建仓特征: 低位筹码增加 + 高位筹码减少 + 成本下移 + 集中度上升
    building_score = 0
    if low_chip_increase > 2:
        building_score += 1
    if high_chip_decrease > 2:
        building_score += 1
    if cost_change_pct < -1:
        building_score += 1
    if concentration_change > 3:
        building_score += 1
    if near_chip_change > 5:
        building_score += 1
    
    # 出货特征: 高位筹码增加 + 低位筹码减少 + 成本上移
    distributing_score = 0
    if low_chip_increase < -2:
        distributing_score += 1
    if high_chip_decrease < -2:
        distributing_score += 1
    if cost_change_pct > 2:
        distributing_score += 1
    if concentration_change < -3:
        distributing_score += 1
    
    if building_score >= 3:
        action = "主力建仓"
        action_emoji = "🟢"
        action_desc = f"低位筹码增加{low_chip_increase:.1f}%，成本下移{cost_change_pct:.1f}%"
    elif building_score >= 2:
        action = "疑似建仓"
        action_emoji = "🟡"
        action_desc = f"低位筹码变化{low_chip_increase:+.1f}%，集中度变化{concentration_change:+.1f}%"
    elif distributing_score >= 3:
        action = "主力出货"
        action_emoji = "🔴"
        action_desc = f"高位筹码增加，成本上移{cost_change_pct:.1f}%"
    elif distributing_score >= 2:
        action = "疑似出货"
        action_emoji = "🟠"
        action_desc = f"筹码向上转移，集中度下降{-concentration_change:.1f}%"
    else:
        action = "震荡整理"
        action_emoji = "⚪"
        action_desc = "筹码变化不明显"
    
    return {
        'lookback_days': lookback_days,
        'low_chip_increase': low_chip_increase,
        'high_chip_decrease': high_chip_decrease,
        'near_chip_change': near_chip_change,
        'cost_change': cost_change,
        'cost_change_pct': cost_change_pct,
        'past_avg_cost': past_avg_cost,
        'current_avg_cost': current_avg_cost,
        'concentration_change': concentration_change,
        'past_concentration': past_concentration,
        'current_concentration': current_concentration,
        'action': action,
        'action_emoji': action_emoji,
        'action_desc': action_desc,
        'building_score': building_score,
        'distributing_score': distributing_score,
        'chip_change': chip_change,
        'bin_centers': bin_centers,
        'past_profile': past_profile,
        'current_profile': current_profile
    }


def create_chip_flow_chart(chip_flow_data, symbol):
    """创建筹码流动对比图"""
    if chip_flow_data is None:
        return None
    
    bin_centers = chip_flow_data['bin_centers']
    past_profile = chip_flow_data['past_profile'] * 100
    current_profile = chip_flow_data['current_profile'] * 100
    chip_change = chip_flow_data['chip_change'] * 100
    
    # 计算当前价格和POC
    current_price = chip_flow_data.get('current_close', bin_centers[len(bin_centers)//2])
    poc_idx = np.argmax(chip_flow_data['current_profile'])
    poc_price = bin_centers[poc_idx]
    
    # 根据价格位置生成渐变颜色 (获利=绿色, 套牢=红色)
    current_colors = []
    for i, price in enumerate(bin_centers):
        if price < current_price * 0.95:  # 获利区
            intensity = min(current_profile[i] / max(current_profile.max(), 1) * 0.8 + 0.2, 1)
            current_colors.append(f'rgba(50, 205, 50, {intensity})')
        elif price > current_price * 1.05:  # 套牢区
            intensity = min(current_profile[i] / max(current_profile.max(), 1) * 0.8 + 0.2, 1)
            current_colors.append(f'rgba(220, 50, 50, {intensity})')
        else:  # 成本区
            intensity = min(current_profile[i] / max(current_profile.max(), 1) * 0.8 + 0.2, 1)
            current_colors.append(f'rgba(255, 165, 0, {intensity})')
    
    fig = go.Figure()
    
    # 过去筹码分布 (灰色)
    fig.add_trace(
        go.Bar(
            y=bin_centers,
            x=-past_profile,
            orientation='h',
            name=f'{chip_flow_data["lookback_days"]}天前',
            marker_color='rgba(120, 120, 120, 0.5)',
            hovertemplate='$%{y:.2f}: %{customdata:.1f}%<extra>过去</extra>',
            customdata=past_profile
        )
    )
    
    # 当前筹码分布 (渐变色)
    fig.add_trace(
        go.Bar(
            y=bin_centers,
            x=current_profile,
            orientation='h',
            name='现在',
            marker_color=current_colors,
            hovertemplate='$%{y:.2f}: %{x:.1f}%<extra>现在</extra>'
        )
    )
    
    # 添加零线
    fig.add_vline(x=0, line_color="white", line_width=2)
    
    # POC 标记线
    fig.add_hline(
        y=poc_price, 
        line_dash="dot", 
        line_color="rgba(255, 69, 0, 0.8)", 
        line_width=2,
        annotation_text=f"POC ${poc_price:.2f}",
        annotation_position="right",
        annotation=dict(font_size=10, font_color="orange")
    )
    
    # 当前价格标记
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="rgba(0, 191, 255, 0.9)",
        line_width=2,
        annotation_text=f"现价 ${current_price:.2f}",
        annotation_position="left",
        annotation=dict(font_size=10, font_color="deepskyblue")
    )
    
    # 布局 - 暗色主题
    fig.update_layout(
        height=450,
        barmode='overlay',
        hovermode='y unified',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.8)',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=70, r=70, t=40, b=50),
        xaxis=dict(
            title=dict(text="← 过去 | 筹码占比 (%) | 现在 →", font=dict(size=11)),
            tickfont=dict(size=10),
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='white',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title=dict(text="价格 ($)", font=dict(size=11)),
            tickfont=dict(size=10),
            gridcolor='rgba(255,255,255,0.1)'
        )
    )
    
    return fig


def create_chip_change_chart(chip_flow_data):
    """创建筹码变化图 (单独的图表)"""
    if chip_flow_data is None:
        return None
    
    bin_centers = chip_flow_data['bin_centers']
    chip_change = chip_flow_data['chip_change'] * 100
    
    # 颜色: 红增绿减
    colors = ['rgba(220, 50, 50, 0.8)' if c > 0 else 'rgba(50, 180, 50, 0.8)' for c in chip_change]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=bin_centers,
            x=chip_change,
            orientation='h',
            marker_color=colors,
            hovertemplate='$%{y:.2f}: %{x:+.2f}%<extra></extra>'
        )
    )
    
    # 添加零线
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        height=350,
        showlegend=False,
        hovermode='y',  # 横向 hover
        margin=dict(l=60, r=30, t=30, b=50),
        xaxis=dict(
            title=dict(text="筹码变化 (%) | 红=增加 绿=减少", font=dict(size=11)),
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=dict(text="价格 ($)", font=dict(size=11)),
            tickfont=dict(size=10)
        )
    )
    
    return fig


def create_candlestick_chart(df, symbol, name, period='daily', day_blue_dates=None, week_blue_dates=None, heima_dates=None, show_volume_profile=False, stop_loss_price=None):
    """创建K线图，标注信号位置和止损线
    
    Args:
        df: 日线数据DataFrame
        symbol: 股票代码
        name: 股票名称
        period: 信号周期 ('daily', 'weekly', 'monthly')
        day_blue_dates: 日线BLUE信号日期
        week_blue_dates: 周线BLUE信号日期
        heima_dates: 黑马信号日期
        show_volume_profile: 是否显示筹码分布图
        stop_loss_price: 止损价格（可选）
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
            
            # --- 增强版筹码逻辑 ---
            # 1. 寻找 POC (Point of Control)
            max_vol_idx = np.argmax(volume_profile)
            poc_price = bin_centers[max_vol_idx]
            
            # 2. 颜色编码: 获利盘 vs 套牢盘 vs POC
            current_close = chart_df['Close'].iloc[-1]
            bar_colors = []
            
            for price in bin_centers:
                # POC 判定 (最长筹码峰)
                if abs(price - poc_price) < (bin_size / 2):
                    bar_colors.append('rgba(255, 69, 0, 0.8)') # 橙红色 (POC)
                # 获利盘 (Profit Chips, 低于现价)
                elif price < current_close:
                    bar_colors.append('rgba(255, 215, 0, 0.5)') # 金色 (支撑)
                # 套牢盘 (Trapped Chips, 高于现价)
                else:
                    bar_colors.append('rgba(0, 191, 255, 0.5)') # 这种蓝更亮一些 (压力)

            fig.add_trace(
                go.Bar(
                    x=volume_profile,
                    y=bin_centers,
                    orientation='h',
                    name='筹码分布',
                    marker=dict(
                        color=bar_colors, # 使用动态颜色
                        line=dict(width=0.5, color='rgba(0,0,0,0.1)')
                    ),
                    showlegend=False,
                    hoverinfo='y+x',
                    hovertemplate='价格: %{y:.2f}<br>堆积量: %{x:.0f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. 在主图画出 POC 延伸线 (作为强支撑/压力参考)
            fig.add_hline(
                y=poc_price, 
                line_dash="dot", 
                line_color="rgba(255, 69, 0, 0.6)", 
                line_width=1.5,
                annotation_text="POC (筹码峰)", 
                annotation_position="right",
                row=1, col=1
            )
            
            # 4. 计算并标注筹码集中度 (形态识别)
            total_vol = np.sum(volume_profile)
            # 计算 POC 附近 20% 价格区间内的筹码占比
            near_poc_vol = 0
            poc_range_low = poc_price * 0.9
            poc_range_high = poc_price * 1.1
            
            for i, p in enumerate(bin_centers):
                if poc_range_low <= p <= poc_range_high:
                    near_poc_vol += volume_profile[i]
            
            concentration = near_poc_vol / total_vol if total_vol > 0 else 0
            
            pattern_desc = "普通分布"
            if concentration > 0.6:
                pattern_desc = "单峰密集 (强支撑)"
            elif concentration < 0.3:
                pattern_desc = "多峰发散 (震荡)"
                
            # 在图表标题或子标题显示形态
            fig.layout.annotations[1].text = f"筹码分布<br>({pattern_desc})"
            
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
    
    # 添加止损线 (如果提供)
    if stop_loss_price is not None and period == 'daily':
        fig.add_hline(y=stop_loss_price, line_dash="dot", line_color="red", line_width=2,
                     annotation_text=f"Stop Loss: {stop_loss_price:.2f}", 
                     annotation_position="bottom right",
                     row=1, col=1)

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


def quick_chip_analysis(df, decay_factor=0.97):
    """
    快速计算筹码分布指标（不生成图表）
    用于在表格中显示底部顶格峰指标
    
    Args:
        df: 包含 OHLCV 的 DataFrame (日线数据)
        decay_factor: 时间衰减因子
    
    Returns:
        dict: 筹码分析结果，包含 is_bottom_peak, is_strong_bottom_peak 等
    """
    if df is None or len(df) < 30:
        return None
    
    try:
        price_min = df['Low'].min()
        price_max = df['High'].max()
        price_range = price_max - price_min
        
        if price_range <= 0:
            return None
        
        bins = 70
        bin_size = price_range / bins
        volume_profile = np.zeros(bins)
        bin_centers = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
        
        total_days = len(df)
        for i, (idx, row) in enumerate(df.iterrows()):
            day_high = row['High']
            day_low = row['Low']
            day_close = row['Close']
            day_vol = row['Volume']
            
            days_ago = total_days - 1 - i
            time_weight = decay_factor ** days_ago
            weighted_vol = day_vol * time_weight
            
            if day_high == day_low or bin_size == 0:
                bin_idx = int((day_close - price_min) / bin_size)
                bin_idx = min(max(bin_idx, 0), bins - 1)
                volume_profile[bin_idx] += weighted_vol
            else:
                start_bin = int((day_low - price_min) / bin_size)
                end_bin = int((day_high - price_min) / bin_size)
                start_bin = max(start_bin, 0)
                end_bin = min(end_bin, bins - 1)
                close_bin = int((day_close - price_min) / bin_size)
                close_bin = min(max(close_bin, start_bin), end_bin)
                
                if start_bin == end_bin:
                    volume_profile[start_bin] += weighted_vol
                else:
                    for b in range(start_bin, end_bin + 1):
                        dist_to_close = abs(b - close_bin)
                        max_dist = max(close_bin - start_bin, end_bin - close_bin, 1)
                        weight = 1.0 - 0.8 * (dist_to_close / max_dist)
                        volume_profile[b] += weighted_vol * weight
        
        total_vol = np.sum(volume_profile)
        if total_vol == 0:
            return None
        
        # POC
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_centers[poc_idx]
        
        # 最大单峰占比
        max_chip_pct = np.max(volume_profile) / total_vol * 100
        
        # POC 位置 (0-100%)
        poc_position = (poc_price - price_min) / price_range * 100
        
        # 底部筹码占比
        bottom_30_price = price_min + price_range * 0.30
        bottom_chip_ratio = sum(volume_profile[bin_centers <= bottom_30_price]) / total_vol
        
        # 判定规则
        is_strong_bottom_peak = (poc_position < 30) and (bottom_chip_ratio > 0.50) and (max_chip_pct > 5)
        is_bottom_peak = (poc_position < 35) and (bottom_chip_ratio > 0.35)
        
        # 标签
        if is_strong_bottom_peak:
            label = "🔥"
        elif is_bottom_peak:
            label = "📍"
        else:
            label = ""
        
        return {
            'is_bottom_peak': is_bottom_peak,
            'is_strong_bottom_peak': is_strong_bottom_peak,
            'bottom_chip_ratio': bottom_chip_ratio,
            'poc_position': poc_position,
            'max_chip_pct': max_chip_pct,
            'label': label
        }
    except Exception as e:
        return None
