"""
股票信号可视化界面 - 查看扫描结果的股票指标和图表

功能:
1. 加载扫描结果CSV文件
2. 显示股票列表和信号统计
3. 交互式K线图、成交量、BLUE/LIRED指标
4. 黑马信号标注

用法:
    streamlit run stock_signal_viewer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="股票信号可视化",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stock-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .signal-blue {
        color: #00d4ff;
        font-weight: 600;
    }
    .signal-red {
        color: #ff4757;
        font-weight: 600;
    }
    .signal-heima {
        color: #ffd700;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 技术指标函数 ====================

def REF(series, periods=1):
    return pd.Series(series).shift(periods).values

def EMA(series, periods):
    return pd.Series(series).ewm(span=periods, adjust=False).mean().values

def SMA(series, periods, weight=1):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def IF(condition, value_if_true, value_if_false):
    return np.where(condition, value_if_true, value_if_false)

def POW(series, power):
    return np.power(series, power)

def LLV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).min().values

def HHV(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).max().values

def MA(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def AVEDEV(series, periods):
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values


def calculate_indicators(df):
    """计算所有技术指标"""
    OPEN = df['Open'].values
    HIGH = df['High'].values
    LOW = df['Low'].values
    CLOSE = df['Close'].values
    
    # BLUE/LIRED 计算
    VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
    VAR2 = SMA(np.abs(LOW - VAR1), 13, 1) / SMA(np.maximum(LOW - VAR1, 0), 10, 1)
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(LOW, 33)
    VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
    VAR6 = POW(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    VAR21 = SMA(np.abs(HIGH - VAR1), 13, 1) / SMA(np.minimum(HIGH - VAR1, 0), 10, 1)
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(HIGH, 33)
    VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)
    VAR61 = POW(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    max_value = np.nanmax(np.maximum(VAR6, np.abs(VAR61)))
    RADIO1 = 200 / max_value if max_value > 0 else 1
    BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
    LIRED = IF(VAR51 > REF(VAR51, 1), -VAR61 * RADIO1, 0)
    
    # 黑马信号计算
    VAR1_H = (HIGH + LOW + CLOSE) / 3
    ma_var1 = MA(VAR1_H, 14)
    avedev_var1 = AVEDEV(VAR1_H, 14)
    avedev_var1 = np.where(avedev_var1 == 0, 0.0001, avedev_var1)
    CCI = (VAR1_H - ma_var1) / (0.015 * avedev_var1)
    
    # 黑马信号逻辑
    close_series = pd.Series(CLOSE)
    rolling_min = close_series.rolling(window=22, min_periods=1).min()
    rolling_max = close_series.rolling(window=22, min_periods=1).max()
    pct_change = close_series.pct_change()
    is_rising = pct_change > 0
    was_falling_1 = pct_change.shift(1) <= 0
    was_falling_2 = pct_change.shift(2) <= 0
    was_falling_3 = pct_change.shift(3) <= 0
    near_low = (close_series - rolling_min) / (rolling_max - rolling_min + 0.0001) < 0.2
    heima_signal = (CCI < -110) & is_rising & was_falling_1 & was_falling_2 & was_falling_3 & near_low
    
    # MA均线
    MA5 = MA(CLOSE, 5)
    MA10 = MA(CLOSE, 10)
    MA20 = MA(CLOSE, 20)
    MA60 = MA(CLOSE, 60)
    
    df['BLUE'] = BLUE
    df['LIRED'] = LIRED
    df['CCI'] = CCI
    df['HeimaSignal'] = heima_signal
    df['MA5'] = MA5
    df['MA10'] = MA10
    df['MA20'] = MA20
    df['MA60'] = MA60
    
    return df


# ==================== 数据获取函数 ====================

@st.cache_data(ttl=300)
def get_cn_stock_data(ts_code, days=365):
    """获取A股历史数据"""
    try:
        import tushare as ts
        ts.set_token('gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482')
        pro = ts.pro_api()
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        
        df = df.rename(columns={
            'trade_date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume',
            'amount': 'Amount'
        })
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df = df.set_index('Date').sort_index()
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None


@st.cache_data(ttl=300)
def get_us_stock_data(symbol, days=365):
    """获取美股历史数据"""
    try:
        from polygon import RESTClient
        api_key = os.getenv('POLYGON_API_KEY', 'qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D')
        client = RESTClient(api_key)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        aggs = []
        for a in client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append({
                'Date': pd.Timestamp.fromtimestamp(a.timestamp/1000),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
            })
        
        if not aggs:
            return None
        
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None


def load_signal_files():
    """加载扫描结果CSV文件列表"""
    patterns = [
        'cn_signals_*.csv',
        'signals_*.csv',
        'backtest_*.csv'
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    # 按修改时间排序
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files


def create_stock_chart(df, symbol, name=""):
    """创建交互式股票图表"""
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.2, 0.15],
        subplot_titles=(f'{symbol} {name} K线图', '成交量', 'BLUE/LIRED 指标', 'CCI')
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K线',
            increasing_line_color='#ff4757',
            decreasing_line_color='#2ed573'
        ),
        row=1, col=1
    )
    
    # 均线
    colors = {'MA5': '#ffda79', 'MA10': '#ff6b81', 'MA20': '#70a1ff', 'MA60': '#a29bfe'}
    for ma, color in colors.items():
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color, width=1)),
                row=1, col=1
            )
    
    # 黑马信号标注
    heima_dates = df[df['HeimaSignal'] == True].index
    if len(heima_dates) > 0:
        heima_prices = df.loc[heima_dates, 'Low'] * 0.98
        fig.add_trace(
            go.Scatter(
                x=heima_dates,
                y=heima_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='#ffd700'),
                name='黑马信号'
            ),
            row=1, col=1
        )
    
    # 成交量
    colors_vol = ['#ff4757' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#2ed573' 
                  for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors_vol),
        row=2, col=1
    )
    
    # BLUE 指标
    blue_values = df['BLUE'].fillna(0)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=blue_values.clip(lower=0),
            name='BLUE',
            marker_color='#00d4ff'
        ),
        row=3, col=1
    )
    
    # LIRED 指标
    lired_values = df['LIRED'].fillna(0)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=lired_values.clip(upper=0),
            name='LIRED',
            marker_color='#ff4757'
        ),
        row=3, col=1
    )
    
    # BLUE/LIRED 阈值线
    fig.add_hline(y=100, line_dash="dash", line_color="cyan", opacity=0.5, row=3, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    
    # CCI 指标
    fig.add_trace(
        go.Scatter(x=df.index, y=df['CCI'], name='CCI', line=dict(color='#ffa502', width=1.5)),
        row=4, col=1
    )
    fig.add_hline(y=-110, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.3, row=4, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="red", opacity=0.3, row=4, col=1)
    
    # 图表样式
    fig.update_layout(
        height=900,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


# ==================== 主应用 ====================

def main():
    st.markdown('<h1 class="main-header">📈 股票信号可视化</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 设置")
        
        # 市场选择
        market = st.selectbox("选择市场", ["A股", "美股"], index=0)
        
        # 数据来源
        data_source = st.radio("数据来源", ["从扫描结果加载", "手动输入代码"])
        
        if data_source == "从扫描结果加载":
            # 加载CSV文件
            csv_files = load_signal_files()
            if csv_files:
                selected_file = st.selectbox("选择扫描结果文件", csv_files)
                
                if selected_file:
                    try:
                        signal_df = pd.read_csv(selected_file, encoding='utf-8-sig')
                        st.success(f"已加载 {len(signal_df)} 只股票")
                        
                        # 信号统计
                        st.subheader("信号统计")
                        cols = st.columns(2)
                        
                        if 'has_day_blue' in signal_df.columns:
                            with cols[0]:
                                day_blue = signal_df['has_day_blue'].sum() if 'has_day_blue' in signal_df.columns else 0
                                st.metric("日BLUE", f"{day_blue} 只")
                        
                        if 'has_week_blue' in signal_df.columns:
                            with cols[1]:
                                week_blue = signal_df['has_week_blue'].sum() if 'has_week_blue' in signal_df.columns else 0
                                st.metric("周BLUE", f"{week_blue} 只")
                        
                        if 'has_day_heima' in signal_df.columns:
                            cols2 = st.columns(2)
                            with cols2[0]:
                                day_heima = signal_df['has_day_heima'].sum() if 'has_day_heima' in signal_df.columns else 0
                                st.metric("日黑马", f"{day_heima} 只")
                            with cols2[1]:
                                week_heima = signal_df['has_week_heima'].sum() if 'has_week_heima' in signal_df.columns else 0
                                st.metric("周黑马", f"{week_heima} 只")
                        
                    except Exception as e:
                        st.error(f"加载文件失败: {e}")
                        signal_df = None
            else:
                st.warning("未找到扫描结果文件")
                signal_df = None
        else:
            signal_df = None
        
        # 时间范围
        st.subheader("时间范围")
        days = st.slider("显示天数", 60, 365, 180)
    
    # 主内容区
    if data_source == "从扫描结果加载" and signal_df is not None:
        # 显示股票列表
        st.subheader("📋 股票列表")
        
        # 筛选条件
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_day_blue = st.checkbox("仅显示日BLUE", value=False)
        with col2:
            filter_week_blue = st.checkbox("仅显示周BLUE", value=False)
        with col3:
            filter_heima = st.checkbox("仅显示黑马", value=False)
        
        filtered_df = signal_df.copy()
        if filter_day_blue and 'has_day_blue' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['has_day_blue'] == True]
        if filter_week_blue and 'has_week_blue' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['has_week_blue'] == True]
        if filter_heima:
            if 'has_day_heima' in filtered_df.columns and 'has_week_heima' in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df['has_day_heima'] == True) | (filtered_df['has_week_heima'] == True)]
        
        # 显示表格
        display_cols = ['symbol', 'name', 'price', 'turnover']
        if 'has_day_blue' in filtered_df.columns:
            display_cols.extend(['blue_days', 'blue_weeks'])
        if 'has_day_heima' in filtered_df.columns:
            display_cols.extend(['day_heima_count', 'week_heima_count'])
        
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols].head(50),
            use_container_width=True,
            height=300
        )
        
        # 股票选择
        st.subheader("📈 股票图表")
        
        symbol_col = 'symbol' if 'symbol' in filtered_df.columns else 'ts_code'
        name_col = 'name' if 'name' in filtered_df.columns else None
        
        if symbol_col in filtered_df.columns:
            stock_options = filtered_df[symbol_col].tolist()
            
            if name_col and name_col in filtered_df.columns:
                stock_labels = [f"{row[symbol_col]} - {row[name_col]}" for _, row in filtered_df.iterrows()]
                selected_label = st.selectbox("选择股票", stock_labels)
                selected_idx = stock_labels.index(selected_label)
                selected_stock = stock_options[selected_idx]
                selected_name = filtered_df.iloc[selected_idx][name_col] if name_col else ""
            else:
                selected_stock = st.selectbox("选择股票", stock_options)
                selected_name = ""
            
            if selected_stock:
                with st.spinner(f"正在加载 {selected_stock} 数据..."):
                    if market == "A股":
                        df = get_cn_stock_data(selected_stock, days)
                    else:
                        df = get_us_stock_data(selected_stock, days)
                    
                    if df is not None and len(df) > 0:
                        # 显示图表
                        fig = create_stock_chart(df, selected_stock, selected_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示最新数据
                        st.subheader("📊 最新数据")
                        cols = st.columns(5)
                        latest = df.iloc[-1]
                        
                        with cols[0]:
                            st.metric("收盘价", f"{latest['Close']:.2f}")
                        with cols[1]:
                            pct_change = (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(df) > 1 else 0
                            st.metric("涨跌幅", f"{pct_change:.2f}%", delta=f"{pct_change:.2f}%")
                        with cols[2]:
                            st.metric("BLUE", f"{latest['BLUE']:.1f}")
                        with cols[3]:
                            st.metric("LIRED", f"{latest['LIRED']:.1f}")
                        with cols[4]:
                            st.metric("CCI", f"{latest['CCI']:.1f}")
                        
                        # 信号状态
                        st.subheader("🚦 信号状态")
                        
                        # 检查最近6天的信号
                        recent = df.tail(6)
                        blue_days = (recent['BLUE'] > 100).sum()
                        lired_days = (recent['LIRED'] < -100).sum()
                        heima_days = recent['HeimaSignal'].sum()
                        
                        signal_cols = st.columns(3)
                        with signal_cols[0]:
                            if blue_days >= 3:
                                st.success(f"✅ BLUE信号 ({blue_days}天)")
                            else:
                                st.info(f"BLUE: {blue_days}天")
                        
                        with signal_cols[1]:
                            if lired_days >= 3:
                                st.error(f"⚠️ LIRED信号 ({lired_days}天)")
                            else:
                                st.info(f"LIRED: {lired_days}天")
                        
                        with signal_cols[2]:
                            if heima_days > 0:
                                st.warning(f"🐎 黑马信号 ({heima_days}天)")
                            else:
                                st.info("无黑马信号")
                    else:
                        st.error("获取数据失败")
    
    else:
        # 手动输入模式
        st.subheader("🔍 手动查询")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if market == "A股":
                stock_code = st.text_input("输入股票代码", placeholder="例如: 000001.SZ, 600519.SH")
            else:
                stock_code = st.text_input("输入股票代码", placeholder="例如: AAPL, MSFT")
        
        with col2:
            stock_name = st.text_input("股票名称 (可选)", placeholder="例如: 平安银行")
        
        if stock_code:
            if st.button("🔍 查询", type="primary"):
                with st.spinner(f"正在加载 {stock_code} 数据..."):
                    if market == "A股":
                        df = get_cn_stock_data(stock_code, days)
                    else:
                        df = get_us_stock_data(stock_code, days)
                    
                    if df is not None and len(df) > 0:
                        fig = create_stock_chart(df, stock_code, stock_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示最新数据
                        st.subheader("📊 最新数据")
                        cols = st.columns(5)
                        latest = df.iloc[-1]
                        
                        with cols[0]:
                            st.metric("收盘价", f"{latest['Close']:.2f}")
                        with cols[1]:
                            pct_change = (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(df) > 1 else 0
                            st.metric("涨跌幅", f"{pct_change:.2f}%", delta=f"{pct_change:.2f}%")
                        with cols[2]:
                            st.metric("BLUE", f"{latest['BLUE']:.1f}")
                        with cols[3]:
                            st.metric("LIRED", f"{latest['LIRED']:.1f}")
                        with cols[4]:
                            st.metric("CCI", f"{latest['CCI']:.1f}")
                    else:
                        st.error("获取数据失败，请检查股票代码")


if __name__ == "__main__":
    main()

