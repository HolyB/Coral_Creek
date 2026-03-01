"""
📊 QuantStats 策略回测报告
========================================
在线查看所有策略的 QuantStats 专业回测报告
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys, json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="QuantStats 回测",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# 收益序列构建
# ============================================================================

@st.cache_data(ttl=3600)
def build_strategy_returns(market: str, days_back: int, min_blue: float,
                           holding_days: int, strategy_name: str) -> pd.Series:
    """从 scan_results 构建策略日收益"""
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    init_db()

    dates = get_scanned_dates(market=market)
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    dates = [d for d in dates if d >= cutoff]
    dates.sort()

    price_range = (5, 500) if market == 'US' else (3, 200)
    all_trades = []

    for scan_date in dates:
        results = query_scan_results(scan_date=scan_date, market=market, limit=500)
        for r in results:
            sym = r.get('symbol', '')
            blue = float(r.get('blue_daily', 0) or 0)
            price = float(r.get('price', 0) or 0)

            if not sym or blue < min_blue:
                continue
            if not (price_range[0] < price < price_range[1]):
                continue

            hist = get_stock_history(sym, market, days=holding_days + 30)
            if hist is None or hist.empty:
                continue

            # 标准化
            if 'Close' not in hist.columns:
                for c in hist.columns:
                    if c.lower() == 'close':
                        hist = hist.rename(columns={c: 'Close'})
                        break
            if 'Close' not in hist.columns:
                continue

            if not isinstance(hist.index, pd.DatetimeIndex):
                if 'Date' in hist.columns:
                    hist = hist.set_index('Date')
                elif 'date' in hist.columns:
                    hist = hist.set_index('date')
                hist.index = pd.to_datetime(hist.index)

            entry_date = pd.Timestamp(scan_date)
            mask = hist.index >= entry_date
            future = hist.loc[mask].head(holding_days + 1)
            if len(future) < 2:
                continue

            for i in range(1, len(future)):
                prev_close = float(future.iloc[i - 1]['Close'])
                if prev_close <= 0:
                    continue
                day_ret = float(future.iloc[i]['Close']) / prev_close - 1
                all_trades.append({
                    'date': future.index[i],
                    'return': day_ret,
                })

    if not all_trades:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_trades)
    daily = df.groupby('date')['return'].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def compute_metrics(returns: pd.Series) -> dict:
    """计算关键回测指标"""
    if returns.empty or len(returns) < 5:
        return {}

    returns = returns.replace([np.inf, -np.inf], 0).fillna(0).clip(-0.5, 0.5)

    try:
        import quantstats as qs
        metrics = {
            'total_return': float(qs.stats.comp(returns) * 100),
            'cagr': float(qs.stats.cagr(returns) * 100),
            'sharpe': float(qs.stats.sharpe(returns)),
            'sortino': float(qs.stats.sortino(returns)),
            'max_drawdown': float(qs.stats.max_drawdown(returns) * 100),
            'calmar': float(qs.stats.calmar(returns)),
            'volatility': float(qs.stats.volatility(returns) * 100),
            'win_rate': float(qs.stats.win_rate(returns) * 100),
            'avg_win': float(qs.stats.avg_win(returns) * 100),
            'avg_loss': float(qs.stats.avg_loss(returns) * 100),
            'profit_factor': float(qs.stats.profit_factor(returns)),
            'best_day': float(returns.max() * 100),
            'worst_day': float(returns.min() * 100),
            'trading_days': len(returns),
        }
        # 处理 inf/nan
        for k, v in metrics.items():
            if not np.isfinite(v):
                metrics[k] = 0.0
        return metrics
    except Exception as e:
        return {'error': str(e)}


def create_equity_chart(returns: pd.Series, name: str) -> go.Figure:
    """创建权益曲线"""
    equity = (1 + returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode='lines', name=name,
        line=dict(width=2.5),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.08)'
    ))

    # 基准线
    fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                  annotation_text='初始=1.0')

    # 最高点
    peak_idx = equity.idxmax()
    fig.add_annotation(
        x=peak_idx, y=equity[peak_idx],
        text=f"峰值 {equity[peak_idx]:.3f}",
        showarrow=True, arrowhead=2, arrowcolor='#4CAF50',
        font=dict(color='#4CAF50', size=11)
    )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title='', yaxis_title='净值',
        template='plotly_dark',
        font=dict(size=12),
    )
    return fig


def create_drawdown_chart(returns: pd.Series) -> go.Figure:
    """回撤曲线"""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode='lines', name='回撤',
        line=dict(color='#F44336', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(244, 67, 54, 0.15)'
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title='', yaxis_title='回撤 %',
        template='plotly_dark',
        font=dict(size=11),
    )
    return fig


def create_monthly_heatmap(returns: pd.Series) -> go.Figure:
    """月度收益热力图"""
    if returns.empty:
        return go.Figure()

    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    df = pd.DataFrame({'return': monthly})
    df['year'] = df.index.year
    df['month'] = df.index.month

    pivot = df.pivot_table(values='return', index='year', columns='month', aggfunc='mean')
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[month_names[i - 1] for i in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=[[0, '#F44336'], [0.5, '#1E1E2E'], [1, '#4CAF50']],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont=dict(size=11),
        hovertemplate='%{y} %{x}: %{z:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        height=max(150, 50 * len(pivot)),
        margin=dict(l=10, r=10, t=10, b=10),
        template='plotly_dark',
        font=dict(size=11),
    )
    return fig


def create_comparison_chart(all_returns: dict) -> go.Figure:
    """多策略对比"""
    fig = go.Figure()
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

    for i, (name, rets) in enumerate(all_returns.items()):
        if rets.empty:
            continue
        equity = (1 + rets).cumprod()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode='lines', name=name,
            line=dict(width=2.5, color=colors[i % len(colors)])
        ))

    fig.add_hline(y=1.0, line_dash='dash', line_color='gray')
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title='', yaxis_title='净值',
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(size=12),
    )
    return fig


# ============================================================================
# HTML 报告下载
# ============================================================================

def generate_html_report(returns: pd.Series, name: str, market: str) -> str:
    """生成 QuantStats HTML 报告并返回路径"""
    try:
        import quantstats as qs
        report_dir = project_root / 'reports' / 'backtest'
        report_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime('%Y%m%d')
        path = report_dir / f'{name}_{market}_{today}.html'

        benchmark = 'SPY' if market == 'US' else None
        qs.reports.html(returns, benchmark=benchmark,
                        title=f'{name} - {market}',
                        output=str(path))
        return str(path)
    except:
        return ''


# ============================================================================
# 页面 UI
# ============================================================================

# CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-label {
        color: #8892b0;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
    }
    .positive { color: #4CAF50; }
    .negative { color: #F44336; }
    .neutral { color: #FFC107; }
    .strategy-header {
        background: linear-gradient(90deg, #0d47a1, #1565c0, #1976d2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 36px;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="strategy-header">📊 QuantStats 回测报告</div>', unsafe_allow_html=True)
st.caption("专业级策略回测分析 — 实时生成 | Sharpe · Sortino · Calmar · MaxDD")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")

    market = st.radio("市场", ["US", "CN"], horizontal=True)
    days_back = st.selectbox("回测周期",
                             [30, 60, 90, 180, 365],
                             index=2,
                             format_func=lambda x: f"{x} 天")

    st.subheader("📋 策略")
    strategies = {}
    if st.checkbox("BLUE ≥ 50 信号", value=True):
        strategies['BLUE_50'] = {'min_blue': 50, 'holding_days': 10}
    if st.checkbox("BLUE ≥ 80 高质量", value=True):
        strategies['BLUE_80'] = {'min_blue': 80, 'holding_days': 10}
    if st.checkbox("BLUE ≥ 100 强信号", value=True):
        strategies['BLUE_100'] = {'min_blue': 100, 'holding_days': 10}
    if st.checkbox("BLUE ≥ 150 超强", value=False):
        strategies['BLUE_150'] = {'min_blue': 150, 'holding_days': 10}
    if st.checkbox("短线 (持仓5天)", value=False):
        strategies['Short_5d'] = {'min_blue': 80, 'holding_days': 5}
    if st.checkbox("中线 (持仓20天)", value=False):
        strategies['Mid_20d'] = {'min_blue': 80, 'holding_days': 20}

    st.markdown("---")
    run = st.button("🚀 生成报告", type="primary", use_container_width=True)

# 主区域
if not strategies:
    st.warning("⚠️ 请至少选择一个策略")
elif run:
    all_returns = {}
    all_metrics = {}

    progress = st.progress(0.0)
    status = st.empty()

    for i, (name, params) in enumerate(strategies.items()):
        status.text(f"计算 {name}...")
        progress.progress((i + 0.5) / len(strategies))

        rets = build_strategy_returns(
            market=market,
            days_back=days_back,
            min_blue=params['min_blue'],
            holding_days=params['holding_days'],
            strategy_name=name,
        )

        if not rets.empty:
            all_returns[name] = rets
            all_metrics[name] = compute_metrics(rets)

        progress.progress((i + 1) / len(strategies))

    progress.empty()
    status.empty()

    if not all_returns:
        st.error("❌ 没有生成任何有效数据")
    else:
        # ============================================
        # 1. 策略对比
        # ============================================
        st.subheader("📈 策略净值对比")
        fig = create_comparison_chart(all_returns)
        st.plotly_chart(fig, use_container_width=True)

        # ============================================
        # 2. 指标汇总表
        # ============================================
        st.subheader("📋 关键指标")

        summary_data = []
        for name, m in all_metrics.items():
            if 'error' in m:
                continue
            summary_data.append({
                '策略': name,
                '总收益': f"{m.get('total_return', 0):.1f}%",
                'CAGR': f"{m.get('cagr', 0):.1f}%",
                'Sharpe': f"{m.get('sharpe', 0):.2f}",
                'Sortino': f"{m.get('sortino', 0):.2f}",
                '最大回撤': f"{m.get('max_drawdown', 0):.1f}%",
                'Calmar': f"{m.get('calmar', 0):.2f}",
                '波动率': f"{m.get('volatility', 0):.1f}%",
                '胜率': f"{m.get('win_rate', 0):.1f}%",
                '盈亏比': f"{m.get('profit_factor', 0):.2f}",
                '最好单日': f"{m.get('best_day', 0):.2f}%",
                '最差单日': f"{m.get('worst_day', 0):.2f}%",
                '交易天': m.get('trading_days', 0),
            })

        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # ============================================
        # 3. 单策略详情
        # ============================================
        st.markdown("---")
        st.subheader("🔍 策略详情")

        tabs = st.tabs(list(all_returns.keys()))

        for tab, (name, rets) in zip(tabs, all_returns.items()):
            with tab:
                m = all_metrics.get(name, {})

                # 核心指标卡片
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                total_ret = m.get('total_return', 0)
                ret_color = 'positive' if total_ret > 0 else 'negative'
                c1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">总收益</div>
                    <div class="metric-value {ret_color}">{total_ret:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                sharpe = m.get('sharpe', 0)
                s_color = 'positive' if sharpe > 0.5 else ('negative' if sharpe < 0 else 'neutral')
                c2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sharpe</div>
                    <div class="metric-value {s_color}">{sharpe:.2f}</div>
                </div>""", unsafe_allow_html=True)

                sortino = m.get('sortino', 0)
                c3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sortino</div>
                    <div class="metric-value {'positive' if sortino > 0 else 'negative'}">{sortino:.2f}</div>
                </div>""", unsafe_allow_html=True)

                dd = m.get('max_drawdown', 0)
                c4.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">最大回撤</div>
                    <div class="metric-value negative">{dd:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                wr = m.get('win_rate', 0)
                c5.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">胜率</div>
                    <div class="metric-value {'positive' if wr > 50 else 'negative'}">{wr:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                vol = m.get('volatility', 0)
                c6.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">波动率</div>
                    <div class="metric-value neutral">{vol:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                # 权益曲线 + 回撤
                st.plotly_chart(create_equity_chart(rets, name), use_container_width=True)
                st.plotly_chart(create_drawdown_chart(rets), use_container_width=True)

                # 月度热力图
                st.markdown("##### 📅 月度收益热力图")
                st.plotly_chart(create_monthly_heatmap(rets), use_container_width=True)

                # 下载 HTML 报告
                col_dl1, col_dl2 = st.columns([1, 3])
                with col_dl1:
                    if st.button(f"📥 下载 HTML 报告", key=f"dl_{name}"):
                        with st.spinner("生成中..."):
                            path = generate_html_report(rets, name, market)
                            if path:
                                with open(path, 'r') as f:
                                    html_content = f.read()
                                st.download_button(
                                    "💾 保存报告",
                                    html_content,
                                    file_name=f"{name}_{market}_{datetime.now().strftime('%Y%m%d')}.html",
                                    mime="text/html",
                                    key=f"save_{name}",
                                )
                            else:
                                st.error("生成失败")

        # ============================================
        # 4. RL Agent 元数据 (如果有)
        # ============================================
        rl_dir = project_root / 'ml' / 'saved_models' / f'rl_agent_{market.lower()}'
        rl_meta_file = rl_dir / 'rl_meta.json'
        if rl_meta_file.exists():
            st.markdown("---")
            st.subheader("🤖 RL Agent 状态")

            with open(rl_meta_file) as f:
                rl_meta = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            oos = rl_meta.get('out_of_sample', {})
            c1.metric("OOS 收益", f"{oos.get('avg_return_pct', 0):.1f}%")
            c2.metric("OOS 胜率", f"{oos.get('win_rate_pct', 0):.0f}%")
            c3.metric("训练股票", rl_meta.get('n_total', '?'))
            c4.metric("过拟合差距", f"{rl_meta.get('overfit_gap', 0):.1f}%")

else:
    # 默认说明
    st.markdown("""
    ### 📖 使用说明

    1. **选择市场**: US 美股 / CN A 股
    2. **选择回测周期**: 30~365 天
    3. **勾选策略**: BLUE 信号强度 / 持仓天数
    4. **点击生成**: 自动计算 & 展示专业报告

    ---

    ### 📋 包含指标

    | 指标 | 说明 |
    |------|------|
    | **Sharpe Ratio** | 风险调整后收益，> 1 优秀 |
    | **Sortino Ratio** | 只惩罚下行波动 |
    | **Calmar Ratio** | 年化收益 / 最大回撤 |
    | **Max Drawdown** | 从峰值到最低点的最大亏损 |
    | **Win Rate** | 盈利天占比 |
    | **Profit Factor** | 总盈利 / 总亏损 |

    ---

    ### 🔄 对比已有回测

    本页面与 `strategy_backtest.py` 互补:
    - **策略回测页**: 单股，自定义买卖条件
    - **QuantStats 报告**: 全市场扫描信号统计，专业金融指标

    """)

    # 检查已有报告
    report_dir = project_root / 'reports' / 'backtest'
    if report_dir.exists():
        reports = list(report_dir.glob('*.html'))
        if reports:
            st.markdown(f"### 📁 已有报告 ({len(reports)} 份)")
            for rp in sorted(reports, reverse=True)[:10]:
                col1, col2 = st.columns([3, 1])
                col1.text(rp.name)
                with open(rp, 'r') as f:
                    col2.download_button(
                        "📥", f.read(),
                        file_name=rp.name,
                        mime='text/html',
                        key=f"existing_{rp.stem}",
                    )
