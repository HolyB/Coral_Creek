#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal Backtest Service - 信号回测验证服务

提供 BLUE 信号历史表现验证功能:
- 计算信号触发后的前向收益
- 与 SPY 基准对比
- 计算风险指标 (Sharpe, Sortino, Max Drawdown)
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data_fetcher import get_us_stock_data
from db.database import query_scan_results, get_scanned_dates


def get_forward_returns(symbol: str, signal_date: str, 
                        forward_days: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    计算信号日期后的前向收益率
    
    Args:
        symbol: 股票代码
        signal_date: 信号日期 (YYYY-MM-DD)
        forward_days: 前向天数列表
    
    Returns:
        Dict with returns for each period, e.g. {'5d': 0.05, '10d': 0.08, '20d': 0.12}
    """
    try:
        # 获取信号日期后的股价数据
        signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
        end_dt = signal_dt + timedelta(days=max(forward_days) + 10)  # 额外天数处理节假日
        
        df = get_us_stock_data(symbol, days=max(forward_days) + 30)
        if df is None or df.empty:
            return {}
        
        # 找到信号日期对应的价格
        df.index = pd.to_datetime(df.index)
        signal_mask = df.index.date == signal_dt.date()
        
        if not signal_mask.any():
            # 信号日期没有数据，取最近的交易日
            valid_dates = df.index[df.index >= signal_dt]
            if len(valid_dates) == 0:
                return {}
            signal_price = df.loc[valid_dates[0], 'Close']
            base_idx = df.index.get_loc(valid_dates[0])
        else:
            signal_price = df.loc[signal_mask, 'Close'].iloc[0]
            base_idx = df.index.get_loc(df.index[signal_mask][0])
        
        returns = {}
        for days in forward_days:
            target_idx = base_idx + days
            if target_idx < len(df):
                future_price = df.iloc[target_idx]['Close']
                ret = (future_price - signal_price) / signal_price
                returns[f'{days}d'] = round(ret, 4)
            else:
                returns[f'{days}d'] = None
        
        return returns
        
    except Exception as e:
        print(f"Error getting forward returns for {symbol}: {e}")
        return {}


def get_spy_returns(start_date: str, forward_days: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    获取 SPY 在同期的收益率作为基准
    """
    return get_forward_returns('SPY', start_date, forward_days)


def calculate_backtest_metrics(returns_list: List[float]) -> Dict[str, float]:
    """
    计算回测核心指标
    
    Args:
        returns_list: 收益率列表 (小数格式, 如 0.05 = 5%)
    
    Returns:
        Dict with metrics: win_rate, avg_return, sharpe, sortino, max_drawdown, profit_factor
    """
    if not returns_list or len(returns_list) == 0:
        return {
            'win_rate': 0,
            'avg_return': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0
        }
    
    returns = np.array([r for r in returns_list if r is not None])
    
    if len(returns) == 0:
        return {
            'win_rate': 0,
            'avg_return': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0
        }
    
    # 基础统计
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_return = np.mean(returns)
    
    # Sharpe Ratio (假设无风险利率 = 0，简化计算)
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    # Sortino Ratio (只考虑下行风险)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino = np.mean(returns) / downside_std if downside_std > 0 else 0
    
    # 最大回撤 (累积收益序列的最大回撤)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    # Profit Factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    return {
        'win_rate': round(win_rate * 100, 2),  # 百分比
        'avg_return': round(avg_return * 100, 2),  # 百分比
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_drawdown * 100, 2),  # 百分比
        'profit_factor': round(profit_factor, 2),
        'total_signals': len(returns),
        'winning_signals': len(wins),
        'losing_signals': len(losses)
    }


def run_signal_backtest(
    start_date: str = None,
    end_date: str = None,
    market: str = 'US',
    min_blue: float = 100,
    forward_days: int = 10,
    limit: int = 500,
    forward_days_list: Optional[List[int]] = None,
    cap_filter: str = "all"
) -> Dict:
    """
    运行完整的信号回测
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        market: 市场 (US/CN)
        min_blue: 最低 BLUE 阈值
        forward_days: 前向收益天数 (5/10/20)
        limit: 最多分析的信号数量
    
    Returns:
        Dict with backtest results
    """
    # 默认回测最近90天
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"📊 Running backtest: {start_date} to {end_date}, market={market}, min_blue={min_blue}")
    
    horizons = sorted(set([int(forward_days)] + [int(x) for x in (forward_days_list or []) if int(x) > 0]))
    if not horizons:
        horizons = [10]
    primary_horizon = horizons[0]

    # 获取历史扫描结果
    signals_all = query_scan_results(
        start_date=start_date,
        end_date=end_date,
        market=market,
        min_blue=min_blue,
        limit=limit
    )
    signals = [s for s in (signals_all or []) if _cap_filter_match(s, cap_filter)]
    
    if not signals:
        return {
            'metrics': calculate_backtest_metrics([]),
            'signals': [],
            'spy_comparison': {},
            'params': {
                'start_date': start_date,
                'end_date': end_date,
                'market': market,
                'min_blue': min_blue,
                'forward_days': primary_horizon,
                'horizons': horizons,
                'cap_filter': cap_filter
            }
        }
    
    print(f"📈 Found {len(signals)} signals to analyze")
    
    # 计算每个信号的前向收益
    signal_results = []
    returns_by_h = {h: [] for h in horizons}
    spy_returns_cache = {}
    
    for i, signal in enumerate(signals):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(signals)}...")
        
        symbol = signal['symbol']
        signal_date = signal['scan_date']
        
        # 获取前向收益
        fwd_returns = get_forward_returns(symbol, signal_date, horizons)
        
        # 获取 SPY 同期收益 (缓存)
        if signal_date not in spy_returns_cache:
            spy_returns_cache[signal_date] = get_forward_returns('SPY', signal_date, horizons)
        
        row = {
            'symbol': symbol,
            'signal_date': signal_date,
            'blue_daily': signal.get('blue_daily', 0),
            'price': signal.get('price', 0),
            'market_cap': signal.get('market_cap'),
            'cap_category': signal.get('cap_category'),
        }
        for h in horizons:
            ret_key = f'{h}d'
            ret = fwd_returns.get(ret_key)
            spy_ret = spy_returns_cache[signal_date].get(ret_key)
            row[f'return_{h}d'] = ret
            row[f'spy_return_{h}d'] = spy_ret
            row[f'alpha_{h}d'] = (ret - spy_ret) if ret is not None and spy_ret is not None else None
            if ret is not None:
                returns_by_h[h].append(ret)
        signal_results.append(row)
    
    metrics_by_h = {}
    for h in horizons:
        metrics_by_h[f'{h}d'] = calculate_backtest_metrics(returns_by_h.get(h, []))
    metrics = metrics_by_h.get(f'{primary_horizon}d', calculate_backtest_metrics([]))
    
    # SPY 基准表现
    spy_returns = [r for r in spy_returns_cache.values() if r.get(f'{primary_horizon}d') is not None]
    spy_returns_flat = [r[f'{primary_horizon}d'] for r in spy_returns]
    spy_metrics = calculate_backtest_metrics(spy_returns_flat) if spy_returns_flat else {}
    
    print(f"✅ Backtest complete. Win rate: {metrics['win_rate']}%, Avg return: {metrics['avg_return']}%")
    
    return {
        'metrics': metrics,
        'metrics_by_horizon': metrics_by_h,
        'spy_metrics': spy_metrics,
        'signals': signal_results,
        'cap_segment_metrics': _build_cap_segment_metrics(signal_results, primary_horizon),
        'params': {
            'start_date': start_date,
            'end_date': end_date,
            'market': market,
            'min_blue': min_blue,
            'forward_days': primary_horizon,
            'horizons': horizons,
            'cap_filter': cap_filter,
            'total_analyzed': len(signal_results)
        }
    }


def run_full_signal_backtest(
    start_date: str = None,
    end_date: str = None,
    market: str = 'US',
    min_blue: float = 100,
    holding_days: int = 20,
    limit: int = 500,
    cap_filter: str = "all",
    initial_capital: float = 100000.0,
    max_positions: int = 10,
    position_size_pct: float = 0.1,
    commission: float = 0.0005,
    slippage: float = 0.001,
    run_walk_forward: bool = True
) -> Dict:
    """完整回测模式：组合回测 + 可选walk-forward"""
    from backtest.backtester import Backtester

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    signals_all = query_scan_results(
        start_date=start_date,
        end_date=end_date,
        market=market,
        min_blue=min_blue,
        limit=limit
    )
    signals = [s for s in (signals_all or []) if _cap_filter_match(s, cap_filter)]
    if not signals:
        return {
            'mode': 'full',
            'portfolio_metrics': {},
            'walk_forward': {'status': 'skipped', 'reason': 'no_signals'},
            'signals_count': 0,
            'params': {
                'start_date': start_date, 'end_date': end_date, 'market': market,
                'min_blue': min_blue, 'holding_days': holding_days, 'cap_filter': cap_filter
            }
        }

    signals_df = pd.DataFrame(signals)
    bt = Backtester(initial_capital=initial_capital, commission=commission, slippage=slippage)
    portfolio_metrics = bt.run_portfolio_backtest(
        signals_df=signals_df,
        holding_days=holding_days,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        market=market
    )

    walk = {'status': 'skipped', 'reason': 'disabled'}
    if run_walk_forward:
        walk_raw = bt.walk_forward_backtest(
            signals_df=signals_df,
            train_days=60,
            test_days=20,
            holding_days=holding_days,
            market=market
        )
        walk = walk_raw if isinstance(walk_raw, dict) else {'status': 'skipped', 'reason': 'unknown'}

    return {
        'mode': 'full',
        'portfolio_metrics': portfolio_metrics,
        'walk_forward': walk,
        'trades': portfolio_metrics.get('trades', []),
        'signals_count': len(signals_df),
        'params': {
            'start_date': start_date,
            'end_date': end_date,
            'market': market,
            'min_blue': min_blue,
            'holding_days': holding_days,
            'cap_filter': cap_filter,
            'initial_capital': initial_capital,
            'max_positions': max_positions,
            'position_size_pct': position_size_pct,
            'commission': commission,
            'slippage': slippage,
        }
    }


def get_backtest_summary_table(backtest_result: Dict) -> pd.DataFrame:
    """
    生成回测摘要表格
    """
    metrics = backtest_result.get('metrics', {})
    spy_metrics = backtest_result.get('spy_metrics', {})
    
    data = {
        'Metric': [
            'Win Rate (%)',
            'Avg Return (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown (%)',
            'Profit Factor',
            'Total Signals'
        ],
        'BLUE Signals': [
            metrics.get('win_rate', 0),
            metrics.get('avg_return', 0),
            metrics.get('sharpe', 0),
            metrics.get('sortino', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('profit_factor', 0),
            metrics.get('total_signals', 0)
        ],
        'SPY Benchmark': [
            spy_metrics.get('win_rate', 0),
            spy_metrics.get('avg_return', 0),
            spy_metrics.get('sharpe', 0),
            spy_metrics.get('sortino', 0),
            spy_metrics.get('max_drawdown', 0),
            spy_metrics.get('profit_factor', 0),
            spy_metrics.get('total_signals', 0)
        ]
    }
    
    return pd.DataFrame(data)


def create_cumulative_returns_chart(backtest_result: Dict) -> 'go.Figure':
    """
    创建累积收益曲线图 (BLUE Signals vs SPY)
    
    Args:
        backtest_result: run_signal_backtest 的返回结果
    
    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    signals = backtest_result.get('signals', [])
    params = backtest_result.get('params', {})
    forward_days = params.get('forward_days', 10)
    ret_col = f'return_{forward_days}d'
    spy_ret_col = f'spy_return_{forward_days}d'
    
    if not signals:
        # 返回空图表
        fig = go.Figure()
        fig.add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#8b949e")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # 按日期排序
    df = pd.DataFrame(signals)
    df = df.sort_values('signal_date').reset_index(drop=True)
    
    # 过滤有效收益数据
    df_valid = df[df[ret_col].notna()].copy()
    
    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient forward data for returns calculation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#8b949e")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # 计算累积收益
    df_valid['blue_cumulative'] = (1 + df_valid[ret_col]).cumprod() - 1
    
    if spy_ret_col in df_valid.columns and df_valid[spy_ret_col].notna().any():
        df_valid['spy_cumulative'] = (1 + df_valid[spy_ret_col].fillna(0)).cumprod() - 1
    else:
        df_valid['spy_cumulative'] = 0
    
    # 计算 Alpha
    df_valid['alpha_cumulative'] = df_valid['blue_cumulative'] - df_valid['spy_cumulative']
    
    # 计算回撤
    blue_peak = (1 + df_valid['blue_cumulative']).cummax()
    df_valid['drawdown'] = ((1 + df_valid['blue_cumulative']) - blue_peak) / blue_peak * 100
    
    # 创建双 Y 轴图表
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=('Cumulative Returns', 'Drawdown')
    )
    
    # 主图: 累积收益曲线
    # BLUE 信号收益
    fig.add_trace(
        go.Scatter(
            x=df_valid['signal_date'],
            y=df_valid['blue_cumulative'] * 100,
            mode='lines',
            name='BLUE Signals',
            line=dict(color='#58a6ff', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.1)',
            hovertemplate='<b>Date</b>: %{x}<br><b>Return</b>: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # SPY 基准收益
    fig.add_trace(
        go.Scatter(
            x=df_valid['signal_date'],
            y=df_valid['spy_cumulative'] * 100,
            mode='lines',
            name='SPY Benchmark',
            line=dict(color='#f0883e', width=2, dash='dot'),
            hovertemplate='<b>Date</b>: %{x}<br><b>SPY Return</b>: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Alpha
    fig.add_trace(
        go.Scatter(
            x=df_valid['signal_date'],
            y=df_valid['alpha_cumulative'] * 100,
            mode='lines',
            name='Alpha (vs SPY)',
            line=dict(color='#3fb950', width=1.5),
            hovertemplate='<b>Date</b>: %{x}<br><b>Alpha</b>: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 子图: 回撤
    fig.add_trace(
        go.Scatter(
            x=df_valid['signal_date'],
            y=df_valid['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#f85149', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(248, 81, 73, 0.2)',
            hovertemplate='<b>Date</b>: %{x}<br><b>Drawdown</b>: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        hovermode='x unified'
    )
    
    # 更新坐标轴
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(48, 54, 61, 0.5)',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(48, 54, 61, 0.5)',
        zeroline=True,
        zerolinecolor='rgba(139, 148, 158, 0.3)',
        ticksuffix='%'
    )
    
    # 子图标题样式
    fig.update_annotations(font=dict(size=12, color='#8b949e'))
    
    return fig


if __name__ == "__main__":
    # 测试回测
    result = run_signal_backtest(
        start_date='2025-12-01',
        end_date='2026-01-15',
        market='US',
        min_blue=100,
        forward_days=10,
        limit=100
    )
    
    print("\n📊 Backtest Summary:")
    print(get_backtest_summary_table(result).to_string(index=False))
def _normalize_cap_category(raw: Optional[str], market_cap: Optional[float] = None) -> str:
    """标准化市值类别"""
    txt = (raw or "").lower()
    if "mega" in txt or "超大盘" in txt:
        return "mega"
    if "large" in txt or "大盘" in txt:
        return "large"
    if "mid" in txt or "中盘" in txt:
        return "mid"
    if "small" in txt or "小盘" in txt:
        return "small"
    if "micro" in txt or "微盘" in txt:
        return "micro"

    try:
        mc = float(market_cap or 0)
        if mc >= 2e11:
            return "mega"
        if mc >= 1e10:
            return "large"
        if mc >= 2e9:
            return "mid"
        if mc >= 3e8:
            return "small"
        if mc > 0:
            return "micro"
    except Exception:
        pass
    return "unknown"


def _cap_filter_match(signal: Dict, cap_filter: str) -> bool:
    if cap_filter == "all":
        return True
    cap = _normalize_cap_category(signal.get("cap_category"), signal.get("market_cap"))
    if cap_filter == "mega_large":
        return cap in {"mega", "large"}
    if cap_filter == "mid":
        return cap == "mid"
    if cap_filter == "small_micro":
        return cap in {"small", "micro"}
    return True


def _build_cap_segment_metrics(signal_results: List[Dict], horizon: int) -> List[Dict]:
    ret_key = f"return_{horizon}d"
    segments = {
        "mega_large": {"label": "Mega/Large", "vals": []},
        "mid": {"label": "Mid", "vals": []},
        "small_micro": {"label": "Small/Micro", "vals": []},
    }

    for r in signal_results:
        ret = r.get(ret_key)
        if ret is None:
            continue
        cap = _normalize_cap_category(r.get("cap_category"), r.get("market_cap"))
        if cap in {"mega", "large"}:
            segments["mega_large"]["vals"].append(ret)
        elif cap == "mid":
            segments["mid"]["vals"].append(ret)
        elif cap in {"small", "micro"}:
            segments["small_micro"]["vals"].append(ret)

    out = []
    for k, v in segments.items():
        vals = v["vals"]
        m = calculate_backtest_metrics(vals)
        out.append({
            "segment": k,
            "label": v["label"],
            "signals": m.get("total_signals", 0),
            "win_rate": m.get("win_rate", 0),
            "avg_return": m.get("avg_return", 0),
            "sharpe": m.get("sharpe", 0),
            "max_drawdown": m.get("max_drawdown", 0),
        })
    return out
