#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蒙特卡洛模拟 - 通过随机抽样评估策略风险

功能:
- 随机抽样历史交易进行N次模拟
- 计算置信区间和破产概率
- 可视化收益分布
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def monte_carlo_simulation(trades: List[Dict], 
                           num_simulations: int = 1000,
                           initial_capital: float = 100000,
                           trades_per_sim: int = 50,
                           bankruptcy_threshold: float = 0.5) -> Dict:
    """
    蒙特卡洛模拟
    
    Args:
        trades: 历史交易列表 [{'pnl_pct': 5.2}, ...]
        num_simulations: 模拟次数
        initial_capital: 初始资金
        trades_per_sim: 每次模拟的交易数
        bankruptcy_threshold: 破产阈值 (资金低于初始的多少比例视为破产)
    
    Returns:
        模拟结果
    """
    if not trades or len(trades) < 10:
        return {'error': 'Insufficient trade history (need at least 10 trades)'}
    
    # 提取收益率
    returns = np.array([t.get('pnl_pct', 0) / 100 for t in trades])
    
    # 模拟
    final_values = []
    max_drawdowns = []
    bankruptcy_count = 0
    all_curves = []
    
    for _ in range(num_simulations):
        # 随机抽样
        sampled_returns = np.random.choice(returns, size=trades_per_sim, replace=True)
        
        # 计算资金曲线
        equity = initial_capital
        equity_curve = [equity]
        peak = equity
        max_dd = 0
        
        for ret in sampled_returns:
            equity = equity * (1 + ret)
            equity_curve.append(equity)
            
            # 更新最大回撤
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_dd:
                max_dd = drawdown
            
            # 检查是否破产
            if equity < initial_capital * bankruptcy_threshold:
                bankruptcy_count += 1
                break
        
        final_values.append(equity)
        max_drawdowns.append(max_dd)
        all_curves.append(equity_curve)
    
    final_values = np.array(final_values)
    max_drawdowns = np.array(max_drawdowns)
    
    # 计算统计量
    mean_final = np.mean(final_values)
    median_final = np.median(final_values)
    std_final = np.std(final_values)
    
    # 置信区间
    ci_5 = np.percentile(final_values, 5)
    ci_25 = np.percentile(final_values, 25)
    ci_75 = np.percentile(final_values, 75)
    ci_95 = np.percentile(final_values, 95)
    
    # 总收益率
    mean_return = (mean_final - initial_capital) / initial_capital * 100
    
    return {
        'num_simulations': num_simulations,
        'trades_per_sim': trades_per_sim,
        'initial_capital': initial_capital,
        
        # 终值统计
        'mean_final_value': round(mean_final, 2),
        'median_final_value': round(median_final, 2),
        'std_final_value': round(std_final, 2),
        'mean_return_pct': round(mean_return, 2),
        
        # 置信区间
        'ci_5': round(ci_5, 2),
        'ci_25': round(ci_25, 2),
        'ci_75': round(ci_75, 2),
        'ci_95': round(ci_95, 2),
        
        # 风险指标
        'mean_max_drawdown': round(np.mean(max_drawdowns) * 100, 2),
        'bankruptcy_probability': round(bankruptcy_count / num_simulations * 100, 2),
        'profit_probability': round(np.sum(final_values > initial_capital) / num_simulations * 100, 2),
        
        # 原始数据 (用于绘图)
        'final_values': final_values.tolist(),
        'sample_curves': all_curves[:100]  # 只保留100条用于绘图
    }


def create_monte_carlo_charts(mc_result: Dict) -> Dict[str, go.Figure]:
    """
    创建蒙特卡洛模拟图表
    
    Returns:
        {'distribution': fig1, 'curves': fig2, 'summary': fig3}
    """
    figures = {}
    
    # 1. 终值分布直方图
    final_values = mc_result.get('final_values', [])
    if final_values:
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            name='终值分布',
            marker_color='#4CAF50',
            opacity=0.7
        ))
        
        # 添加关键线
        initial = mc_result['initial_capital']
        fig_dist.add_vline(x=initial, line_dash="dash", line_color="red",
                          annotation_text="初始资金")
        fig_dist.add_vline(x=mc_result['mean_final_value'], line_dash="solid", 
                          line_color="blue", annotation_text="均值")
        fig_dist.add_vline(x=mc_result['ci_5'], line_dash="dot", line_color="orange",
                          annotation_text="5%分位")
        fig_dist.add_vline(x=mc_result['ci_95'], line_dash="dot", line_color="green",
                          annotation_text="95%分位")
        
        fig_dist.update_layout(
            title="蒙特卡洛模拟 - 终值分布",
            xaxis_title="最终资金 ($)",
            yaxis_title="频次",
            height=400
        )
        
        figures['distribution'] = fig_dist
    
    # 2. 资金曲线图 (抽样显示)
    sample_curves = mc_result.get('sample_curves', [])
    if sample_curves:
        fig_curves = go.Figure()
        
        # 显示前50条曲线
        for i, curve in enumerate(sample_curves[:50]):
            color = '#3fb950' if curve[-1] > mc_result['initial_capital'] else '#f85149'
            fig_curves.add_trace(go.Scatter(
                y=curve,
                mode='lines',
                line=dict(color=color, width=0.5),
                opacity=0.3,
                showlegend=False
            ))
        
        # 添加初始资金线
        fig_curves.add_hline(y=mc_result['initial_capital'], line_dash="dash",
                            line_color="white", annotation_text="初始资金")
        
        fig_curves.update_layout(
            title="资金曲线模拟 (50条样本)",
            xaxis_title="交易序号",
            yaxis_title="资金 ($)",
            height=400,
            template="plotly_dark"
        )
        
        figures['curves'] = fig_curves
    
    # 3. 风险指标仪表盘
    fig_gauge = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # 盈利概率
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number",
        value=mc_result.get('profit_probability', 0),
        title={'text': "盈利概率 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 50], 'color': '#f85149'},
                {'range': [50, 70], 'color': '#d29922'},
                {'range': [70, 100], 'color': '#3fb950'}
            ]
        }
    ), row=1, col=1)
    
    # 破产概率
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number",
        value=mc_result.get('bankruptcy_probability', 0),
        title={'text': "破产概率 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#f85149"},
            'steps': [
                {'range': [0, 10], 'color': '#3fb950'},
                {'range': [10, 30], 'color': '#d29922'},
                {'range': [30, 100], 'color': '#f85149'}
            ]
        }
    ), row=1, col=2)
    
    # 平均回撤
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number",
        value=mc_result.get('mean_max_drawdown', 0),
        title={'text': "平均最大回撤 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2196F3"},
            'steps': [
                {'range': [0, 20], 'color': '#3fb950'},
                {'range': [20, 40], 'color': '#d29922'},
                {'range': [40, 100], 'color': '#f85149'}
            ]
        }
    ), row=1, col=3)
    
    fig_gauge.update_layout(height=300)
    figures['gauges'] = fig_gauge
    
    return figures


if __name__ == "__main__":
    # 测试
    test_trades = [{'pnl_pct': np.random.normal(2, 8)} for _ in range(100)]
    
    result = monte_carlo_simulation(test_trades, num_simulations=500)
    
    print(f"Mean Return: {result['mean_return_pct']}%")
    print(f"Profit Probability: {result['profit_probability']}%")
    print(f"Bankruptcy Probability: {result['bankruptcy_probability']}%")
    print(f"90% CI: ${result['ci_5']:,.0f} - ${result['ci_95']:,.0f}")
