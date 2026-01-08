import streamlit as st
import pandas as pd
import glob
import os
import sys
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 添加当前目录到路径，以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from chart_utils import create_candlestick_chart, create_candlestick_chart_dynamic, analyze_chip_flow, create_chip_flow_chart, create_chip_change_chart
from data_fetcher import get_us_stock_data as fetch_data_from_polygon, get_ticker_details
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_adx_series
from backtester import SimpleBacktester
from db.database import (
    query_scan_results, get_scanned_dates, get_db_stats, 
    get_stock_history, init_db, get_scan_job
)

# 设置页面配置
st.set_page_config(
    page_title="Coral Creek V2.0 - 智能量化系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函数 ---

def format_large_number(num):
    """格式化大数字 (B/M/K)"""
    if not num or pd.isna(num):
        return "N/A"
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"

def load_scan_results_from_db(scan_date=None, market=None):
    """从数据库加载扫描结果"""
    try:
        # 如果没有指定日期，获取最新日期
        if scan_date is None:
            dates = get_scanned_dates()
            if not dates:
                return None, None
            scan_date = dates[0]  # 最新日期
        
        # 查询数据 - 传入 market 参数
        results = query_scan_results(scan_date=scan_date, market=market)
        if not results:
            return None, scan_date
        
        df = pd.DataFrame(results)
        
        # --- 数据标准化与列名映射 ---
        col_map = {
            'symbol': 'Ticker',
            'blue_daily': 'Day BLUE',
            'blue_weekly': 'Week BLUE',
            'blue_monthly': 'Month BLUE',
            'stop_loss': 'Stop Loss',
            'shares_rec': 'Shares Rec',
            'vp_rating': 'Vol Profile',
            'market_cap': 'Mkt Cap Raw',
            'company_name': 'Name',
            'industry': 'Industry',
            'turnover_m': 'Turnover',
            'price': 'Price',
            'adx': 'ADX',
            'volatility': 'Volatility',
            'is_heima': 'Is_Heima',
            'strat_d_trend': 'Strat_D_Trend',
            'strat_c_resonance': 'Strat_C_Resonance',
            'legacy_signal': 'Legacy_Signal',
            'regime': 'Regime',
            'adaptive_thresh': 'Adaptive_Thresh',
            'profit_ratio': 'Profit_Ratio',
            'wave_phase': 'Wave_Phase',
            'wave_desc': 'Wave_Desc',
            'chan_signal': 'Chan_Signal',
            'chan_desc': 'Chan_Desc',
            'cap_category': 'Cap_Category',
            'risk_reward_score': 'Risk_Reward_Score',
            'scan_date': 'Date'
        }
        df.rename(columns=col_map, inplace=True)
        
        # 格式化市值
        if 'Mkt Cap Raw' in df.columns:
            df['Mkt Cap'] = pd.to_numeric(df['Mkt Cap Raw'], errors='coerce').fillna(0) / 1_000_000_000
        else:
            df['Mkt Cap'] = 0.0
        
        # 合成 Strategy 列
        def get_strategy_label(row):
            strategies = []
            if row.get('Strat_D_Trend', False):
                strategies.append('Trend-D')
            if row.get('Strat_C_Resonance', False):
                strategies.append('Resonance-C')
            if not strategies and row.get('Legacy_Signal', False):
                strategies.append('Legacy')
            return " | ".join(strategies) if strategies else "N/A"
            
        df['Strategy'] = df.apply(get_strategy_label, axis=1)
        
        # 合成 Score 列
        def calculate_score(row):
            score = 0
            blue = row.get('Day BLUE', 0) or 0
            score += min(blue / 200, 1.0) * 40
            adx = row.get('ADX', 0) or 0
            score += min(adx / 60, 1.0) * 30
            pr = row.get('Profit_Ratio', 0.5) or 0.5
            score += pr * 30
            return int(score)
            
        df['Score'] = df.apply(calculate_score, axis=1)
        
        # 类型转换
        for col in ['Price', 'Day BLUE', 'Week BLUE', 'Month BLUE', 'Stop Loss', 'ADX', 'Turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, scan_date
    except Exception as e:
        st.error(f"数据库读取失败: {e}")
        return None, None


def load_latest_scan_results():
    """加载最新的扫描结果 - 优先从数据库，回退到 CSV"""
    # 首先尝试从数据库加载
    try:
        init_db()  # 确保数据库已初始化
        stats = get_db_stats()
        if stats and stats['total_records'] > 0:
            # 数据库有数据，使用数据库
            return load_scan_results_from_db()
    except:
        pass
    
    # 回退到 CSV 文件
    files = glob.glob(os.path.join(current_dir, "enhanced_scan_results_*.csv"))
    if not files:
        return None, None
    
    latest_file = max(files, key=os.path.getsize)
    
    try:
        df = pd.read_csv(latest_file)
        
        col_map = {
            'Symbol': 'Ticker',
            'Blue_Daily': 'Day BLUE',
            'Blue_Weekly': 'Week BLUE',
            'Blue_Monthly': 'Month BLUE',
            'Stop_Loss': 'Stop Loss',
            'Shares_Rec': 'Shares Rec',
            'VP_Rating': 'Vol Profile',
            'Market_Cap': 'Mkt Cap Raw',
            'Company_Name': 'Name',
            'Industry': 'Industry',
            'Turnover_M': 'Turnover'
        }
        df.rename(columns=col_map, inplace=True)
        
        if 'Mkt Cap Raw' in df.columns:
            df['Mkt Cap'] = pd.to_numeric(df['Mkt Cap Raw'], errors='coerce').fillna(0) / 1_000_000_000
        else:
            df['Mkt Cap'] = 0.0
            
        def get_strategy_label(row):
            strategies = []
            if row.get('Strat_D_Trend', False):
                strategies.append('Trend-D')
            if row.get('Strat_C_Resonance', False):
                strategies.append('Resonance-C')
            if not strategies and row.get('Legacy_Signal', False):
                strategies.append('Legacy')
            return " | ".join(strategies) if strategies else "N/A"
            
        df['Strategy'] = df.apply(get_strategy_label, axis=1)
        
        def calculate_score(row):
            score = 0
            blue = row.get('Day BLUE', 0)
            score += min(blue / 200, 1.0) * 40
            adx = row.get('ADX', 0)
            score += min(adx / 60, 1.0) * 30
            pr = row.get('Profit_Ratio', 0.5)
            score += pr * 30
            return int(score)
            
        if 'Score' not in df.columns:
            df['Score'] = df.apply(calculate_score, axis=1)

        for col in ['Price', 'Day BLUE', 'Week BLUE', 'Stop Loss', 'Score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, os.path.basename(latest_file)
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None, None

def get_market_mood(df):
    """根据扫描结果判断市场情绪"""
    if df is None or df.empty:
        return "未知", "gray"
    
    high_score_count = len(df[df['Score'] >= 85])
    total_count = len(df)
    ratio = high_score_count / total_count if total_count > 0 else 0
    
    if total_count > 50: 
        if ratio > 0.3:
            return "🔥 极度火热 (FOMO)", "red"
        elif ratio > 0.15:
            return "☀️ 积极做多", "orange"
        elif ratio > 0.05:
            return "☁️ 震荡分化", "blue"
        else:
            return "❄️ 冰点/观望", "lightblue"
    else:
        return f"扫描样本数: {total_count}", "gray"

# --- 页面逻辑 ---

def render_scan_page():
    st.header("🌊 每日机会扫描 (Opportunity Scanner)")
    
    # 侧边栏：数据源选择
    with st.sidebar:
        st.divider()
        st.header("📂 数据源")
        
        # === 市场选择器 ===
        st.subheader("🌍 市场选择")
        market_options = {"🇺🇸 美股 (US)": "US", "🇨🇳 A股 (CN)": "CN"}
        selected_market_label = st.radio(
            "选择市场",
            options=list(market_options.keys()),
            horizontal=True,
            index=0,
            help="切换美股/A股扫描结果"
        )
        selected_market = market_options[selected_market_label]
        
        st.divider()
        
        # 检查数据库状态
        try:
            init_db()
            stats = get_db_stats()
            use_db = stats and stats['total_records'] > 0
        except:
            use_db = False
            stats = None
        
        if use_db:
            st.success("✅ 数据库模式")
            st.caption(f"📊 总记录: {stats['total_records']:,}")
            st.caption(f"📅 日期范围: {stats['min_date']} ~ {stats['max_date']}")
            
            # 日期选择器 - 按所选市场过滤
            available_dates = get_scanned_dates(market=selected_market)
            if available_dates:
                # 转换为 datetime 对象用于 selectbox
                date_options = available_dates[:30]  # 最近30天
                selected_date = st.selectbox(
                    "📅 选择日期",
                    options=date_options,
                    index=0,
                    help=f"选择要查看的 {selected_market} 扫描日期"
                )
                
                # 显示该日期的扫描状态
                job = get_scan_job(selected_date)
                if job:
                    st.caption(f"⏱️ 扫描于: {job.get('finished_at', 'N/A')}")
                    st.caption(f"📈 发现信号: {job.get('signals_found', 'N/A')} 只")
            else:
                selected_date = None
                st.warning(f"暂无 {selected_market} 扫描数据")
        else:
            st.info("📁 CSV 文件模式")
            selected_date = None
        
        if st.button("🔄 刷新数据"):
            st.rerun()
    
    # 加载数据 - 按所选市场过滤
    if use_db and selected_date:
        df, data_source = load_scan_results_from_db(selected_date, market=selected_market)
        if data_source:
            data_source = f"📅 {data_source} ({selected_market})"
    else:
        df, data_source = load_latest_scan_results()
        if data_source and not data_source.startswith("📅"):
            data_source = f"📁 {data_source}"

    if df is None or df.empty:
        st.warning("⚠️ 未找到扫描结果。")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **方式一**: 运行每日扫描\n```bash\ncd versions/v2\npython scripts/run_daily_scan.py\n```")
        with col2:
            st.info("💡 **方式二**: 批量回填历史数据\n```bash\ncd versions/v2\npython scripts/backfill.py --start 2025-12-01 --end 2026-01-07\n```")
        return
            
    # 侧边栏：继续筛选器
    with st.sidebar:
        st.divider()
        st.header("🎛️ 多维筛选")
        st.caption("根据您的偏好自由组合过滤条件")
        
        # === 1. 流动性筛选 (最重要!) ===
        st.subheader("💧 流动性")
        
        # 日均成交额 (Turnover) - 使用 Turnover_M 列 (百万美元)
        if 'Turnover' in df.columns:
            turnover_col = 'Turnover'
        elif 'Turnover_M' in df.columns:
            df['Turnover'] = df['Turnover_M']  # 统一列名
            turnover_col = 'Turnover'
        else:
            turnover_col = None
            
        if turnover_col and turnover_col in df.columns:
            max_turnover = float(df[turnover_col].max()) if df[turnover_col].max() > 0 else 1000
            min_turnover_val = st.slider(
                "最低日成交额 ($M)", 
                min_value=0.0, 
                max_value=min(max_turnover, 500.0),  # 上限500M，避免slider太长
                value=0.0,  # 默认0 (显示所有)
                step=0.5,
                help="过滤成交额过低的股票，避免流动性风险"
            )
            df = df[df[turnover_col] >= min_turnover_val]
        
        # === 2. 信号强度筛选 ===
        st.subheader("📊 信号强度")
        
        # BLUE 信号
        if 'Day BLUE' in df.columns:
            blue_range = st.slider(
                "Day BLUE 范围",
                min_value=0.0,
                max_value=200.0,
                value=(0.0, 200.0),  # 默认 0-200 (显示所有)
                step=10.0,
                help="BLUE 越高代表抄底信号越强"
            )
            df = df[(df['Day BLUE'] >= blue_range[0]) & (df['Day BLUE'] <= blue_range[1])]
        
        # ADX 趋势强度
        if 'ADX' in df.columns:
            adx_min = st.slider(
                "最低 ADX (趋势强度)",
                min_value=0.0,
                max_value=80.0,
                value=0.0,  # 默认 0 (显示所有)
                step=5.0,
                help="ADX > 25 表示趋势明确，ADX > 40 表示强趋势"
            )
            df = df[df['ADX'] >= adx_min]
        
        # === 3. 市值与价格筛选 ===
        st.subheader("💰 市值 & 价格")
        
        # 市值规模 (Multi-Select)
        if 'Cap_Category' in df.columns:
            all_caps = df['Cap_Category'].unique().tolist()
            # 排序：按市值从大到小
            cap_order = ['Mega-Cap (巨头)', 'Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap', 'Unknown']
            sorted_caps = [c for c in cap_order if c in all_caps] + [c for c in all_caps if c not in cap_order]
            selected_caps = st.multiselect(
                "市值规模", 
                sorted_caps, 
                default=sorted_caps,
                help="Mega > $200B, Large > $10B, Mid > $2B, Small > $300M, Micro < $300M"
            )
            if selected_caps:
                df = df[df['Cap_Category'].isin(selected_caps)]
        
        # 价格区间
        if 'Price' in df.columns:
            price_range = st.slider(
                "价格区间 ($)",
                min_value=0.0,
                max_value=min(float(df['Price'].max()), 5000.0),
                value=(1.0, 1000.0),  # 默认 $1-$1000
                step=1.0,
                help="过滤仙股 (<$1) 和超高价股"
            )
            df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]
        
        # === 4. 策略类型筛选 ===
        st.subheader("🎯 策略类型")
        
        if 'Strategy' in df.columns:
            all_strategies = df['Strategy'].unique().tolist()
            selected_strategies = st.multiselect(
                "策略标签", 
                all_strategies, 
                default=all_strategies,
                help="Trend-D: 趋势跟随, Resonance-C: 多周期共振"
            )
            if selected_strategies:
                df = df[df['Strategy'].isin(selected_strategies)]
        
        # === 5. 高级筛选 (折叠) ===
        with st.expander("🔬 高级筛选", expanded=False):
            # 获利盘比例
            if 'Profit_Ratio' in df.columns:
                pr_range = st.slider(
                    "获利盘比例 (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),  # 默认不限制
                    step=5,
                    help="获利盘高 = 筹码结构好，但可能已经涨过；获利盘低 = 套牢盘多，反弹空间大但风险也大"
                )
                df = df[(df['Profit_Ratio'] * 100 >= pr_range[0]) & (df['Profit_Ratio'] * 100 <= pr_range[1])]
            
            # 波浪形态筛选
            if 'Wave_Phase' in df.columns:
                all_waves = df['Wave_Phase'].unique().tolist()
                selected_waves = st.multiselect("波浪形态", all_waves, default=all_waves)
                if selected_waves:
                    df = df[df['Wave_Phase'].isin(selected_waves)]
            
            # 缠论信号筛选
            if 'Chan_Signal' in df.columns:
                all_chans = df['Chan_Signal'].unique().tolist()
                selected_chans = st.multiselect("缠论信号", all_chans, default=all_chans)
                if selected_chans:
                    df = df[df['Chan_Signal'].isin(selected_chans)]
        
        # 显示筛选结果统计
        st.divider()
        st.metric("筛选后结果", f"{len(df)} 只", help="符合所有筛选条件的股票数量")

    # 2. 顶部仪表盘
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("筛选后机会", f"{len(df)} 只", help="符合当前筛选条件的股票数量")

    with col2:
        # 强信号：BLUE > 150
        strong_signals = len(df[df['Day BLUE'] > 150]) if 'Day BLUE' in df.columns else 0
        st.metric("🔥 强信号 (BLUE>150)", f"{strong_signals} 只", help="BLUE > 150 的强势抄底信号")

    with col3:
        trend_opps = len(df[df['Strategy'].str.contains('Trend', na=False)]) if 'Strategy' in df.columns else 0
        st.metric("🚀 趋势突破", f"{trend_opps} 只", help="Strategy D: 趋势跟随")

    with col4:
        # 高流动性：成交额 > 10M
        if 'Turnover' in df.columns:
            high_liquidity = len(df[df['Turnover'] > 10])
            st.metric("💧 高流动性 (>$10M)", f"{high_liquidity} 只", help="日成交额 > 1000万美元")
        else:
            mood, color = get_market_mood(df)
            st.markdown(f"**市场情绪**")
            st.markdown(f"<h3 style='color: {color}; margin-top: -10px;'>{mood}</h3>", unsafe_allow_html=True)

    st.divider()

    # 3. 机会清单
    st.subheader("📋 机会清单 (Opportunity Matrix)")

    column_config = {
        "Ticker": st.column_config.TextColumn("代码", help="股票代码", width="small"),
        "Name": st.column_config.TextColumn("名称", width="medium"),
        "Mkt Cap": st.column_config.NumberColumn("市值 ($B)", format="%.2f", help="市值 (十亿美元)"),
        "Price": st.column_config.NumberColumn("现价", format="$%.2f"),
        "Turnover": st.column_config.NumberColumn("成交额 ($M)", format="%.1f", help="日成交额 (百万美元)"),
        "Day BLUE": st.column_config.ProgressColumn(
            "日 BLUE", format="%.0f", min_value=0, max_value=200,
            help="日线抄底信号强度 (0-200)"
        ),
        "Week BLUE": st.column_config.ProgressColumn(
            "周 BLUE", format="%.0f", min_value=0, max_value=200,
            help="周线抄底信号强度 (0-200)"
        ),
        "Month BLUE": st.column_config.ProgressColumn(
            "月 BLUE", format="%.0f", min_value=0, max_value=200,
            help="月线抄底信号强度 (0-200)"
        ),
        "ADX": st.column_config.NumberColumn("ADX", format="%.1f", help="趋势强度 (>25 趋势明确, >40 强趋势)"),
        "Strategy": st.column_config.TextColumn("策略标签", width="medium"),
        "Regime": st.column_config.TextColumn("波动属性", width="small"),
        "Cap_Category": st.column_config.TextColumn("市值规模", width="small"),
        "Stop Loss": st.column_config.NumberColumn("止损价", format="$%.2f", help="建议止损位"),
        "Shares Rec": st.column_config.NumberColumn("建议仓位", format="%d 股", help="基于$1000风险敞口的建议股数"),
        "Wave_Desc": st.column_config.TextColumn("波浪形态", width="medium", help="Elliott Wave"),
        "Chan_Desc": st.column_config.TextColumn("缠论形态", width="medium", help="Chan Theory"),
        "Profit_Ratio": st.column_config.NumberColumn("获利盘", format="%.0f%%", help="获利盘比例")
    }

    # 显示列顺序：核心指标在前，日/周/月 BLUE 放一起
    display_cols = ['Ticker', 'Name', 'Price', 'Turnover', 'Day BLUE', 'Week BLUE', 'Month BLUE', 'ADX', 'Strategy', 'Mkt Cap', 'Cap_Category', 'Wave_Desc', 'Chan_Desc', 'Stop Loss', 'Shares Rec', 'Regime']
    existing_cols = [c for c in display_cols if c in df.columns]

    # 默认按 Day BLUE 降序排列
    if 'Day BLUE' in df.columns:
        df = df.sort_values('Day BLUE', ascending=False)

    event = st.dataframe(
        df[existing_cols],
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )

    # 4. 深度透视
    selected_rows = event.selection.rows
    if selected_rows:
        selected_index = selected_rows[0]
        selected_row = df.iloc[selected_index]
        symbol = selected_row['Ticker']
        
        st.divider()
        st.subheader(f"🔍 深度透视: {symbol}")
        
        chart_col, info_col = st.columns([2, 1])
        
        with chart_col:
            # 周期切换选项
            period_options = {"📅 日线": "daily", "📆 周线": "weekly", "🗓️ 月线": "monthly"}
            selected_period_label = st.radio(
                "选择周期",
                options=list(period_options.keys()),
                horizontal=True,
                index=0  # 默认日线
            )
            selected_period = period_options[selected_period_label]
            
            with st.spinner(f"正在加载 {symbol} {selected_period_label} 图表..."):
                try:
                    # 5年数据以支持周线/月线分析
                    hist_data = fetch_data_from_polygon(symbol, days=3650)
                    if hist_data is not None and not hist_data.empty:
                        # 根据选择的周期重采样数据
                        if selected_period == 'weekly':
                            display_data = hist_data.resample('W-FRI').agg({
                                'Open': 'first', 'High': 'max', 'Low': 'min', 
                                'Close': 'last', 'Volume': 'sum'
                            }).dropna()
                            chart_title = f"{symbol} - 周线图"
                        elif selected_period == 'monthly':
                            display_data = hist_data.resample('ME').agg({
                                'Open': 'first', 'High': 'max', 'Low': 'min', 
                                'Close': 'last', 'Volume': 'sum'
                            }).dropna()
                            chart_title = f"{symbol} - 月线图"
                        else:
                            display_data = hist_data.tail(365)  # 日线只显示最近1年
                            chart_title = f"{symbol} - 日线图"
                        
                        # === 日期滑动条 - 用于动态筹码分布 ===
                        if len(display_data) > 10:
                            date_list = display_data.index.tolist()
                            
                            # 默认选择最后一天
                            default_idx = len(date_list) - 1
                            
                            selected_date_idx = st.slider(
                                "📅 拖动选择日期 (筹码分布会动态变化)",
                                min_value=10,  # 至少需要10根K线计算筹码
                                max_value=len(date_list) - 1,
                                value=default_idx,
                                format="",
                                key=f"date_slider_{symbol}_{selected_period}"
                            )
                            
                            selected_date = date_list[selected_date_idx]
                            st.caption(f"🎯 选中日期: **{selected_date.strftime('%Y-%m-%d')}** | 收盘价: **${display_data.loc[selected_date, 'Close']:.2f}**")
                            
                            # 只取选中日期之前的数据用于筹码计算
                            chart_data_for_vp = display_data.iloc[:selected_date_idx + 1].copy()
                        else:
                            chart_data_for_vp = display_data.copy()
                            selected_date = display_data.index[-1]
                        
                        # 创建图表，传入动态筹码数据
                        fig = create_candlestick_chart_dynamic(
                            display_data,  # 完整数据用于K线显示
                            chart_data_for_vp,  # 截止选中日期的数据用于筹码
                            symbol, chart_title,
                            period=selected_period, 
                            show_volume_profile=True,
                            stop_loss_price=selected_row.get('Stop Loss') if selected_period == 'daily' else None,
                            highlight_date=selected_date
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # === 显示筹码分析指标 ===
                        if hasattr(fig, '_chip_analysis'):
                            chip = fig._chip_analysis
                            
                            # 买点评估
                            st.markdown(f"### 📊 筹码分析 {chip.get('buy_signal_strength', '')}")
                            
                            # 核心指标卡片
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                profit_pct = chip.get('profit_ratio', 0) * 100
                                st.metric("🟢 获利盘", f"{profit_pct:.1f}%", 
                                         delta=f"{profit_pct - 50:.1f}%" if profit_pct != 50 else None,
                                         delta_color="normal")
                            with c2:
                                trapped_pct = chip.get('trapped_ratio', 0) * 100
                                st.metric("🔴 套牢盘", f"{trapped_pct:.1f}%",
                                         delta=f"{50 - trapped_pct:.1f}%" if trapped_pct != 50 else None,
                                         delta_color="inverse")
                            with c3:
                                conc = chip.get('concentration', 0) * 100
                                st.metric("📍 集中度", f"{conc:.1f}%", help="POC±10%区间筹码占比")
                            with c4:
                                avg_cost = chip.get('avg_cost', 0)
                                current = chip.get('current_close', 0)
                                cost_diff = (current - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
                                st.metric("💰 平均成本", f"${avg_cost:.2f}", 
                                         delta=f"{cost_diff:+.1f}%",
                                         delta_color="normal" if cost_diff > 0 else "inverse")
                            
                            # 支撑压力位
                            st.markdown("**关键价位**")
                            p1, p2, p3 = st.columns(3)
                            with p1:
                                support = chip.get('support_price')
                                if support:
                                    support_dist = (chip.get('current_close', 0) - support) / support * 100 if support > 0 else 0
                                    st.metric("⬇️ 支撑位", f"${support:.2f}", delta=f"距离 {support_dist:.1f}%")
                                else:
                                    st.metric("⬇️ 支撑位", "N/A")
                            with p2:
                                poc = chip.get('poc_price', 0)
                                poc_dist = chip.get('dist_to_poc_pct', 0)
                                st.metric("🎯 筹码峰(POC)", f"${poc:.2f}", delta=f"距离 {poc_dist:+.1f}%")
                            with p3:
                                resist = chip.get('resistance_price')
                                if resist:
                                    resist_dist = (resist - chip.get('current_close', 0)) / chip.get('current_close', 1) * 100
                                    st.metric("⬆️ 压力位", f"${resist:.2f}", delta=f"距离 {resist_dist:.1f}%", delta_color="inverse")
                                else:
                                    st.metric("⬆️ 压力位", "N/A")
                            
                            # 90%成本区间
                            cost_low = chip.get('cost_90_low', 0)
                            cost_high = chip.get('cost_90_high', 0)
                            st.caption(f"📏 90%成本区间: **${cost_low:.2f}** ~ **${cost_high:.2f}** (宽度: ${cost_high - cost_low:.2f})")
                            st.caption(f"📋 形态: **{chip.get('pattern_desc', 'N/A')}**")
                        
                        st.divider()
                        
                        # === 主力建仓/出货分析 ===
                        st.markdown("### 🏦 主力动向分析")
                        
                        # 选择对比天数
                        lookback_options = {
                            "5天": 5,
                            "10天": 10,
                            "20天": 20,
                            "30天": 30,
                            "60天": 60
                        }
                        selected_lookback = st.select_slider(
                            "对比周期",
                            options=list(lookback_options.keys()),
                            value="20天",
                            key=f"lookback_{symbol}"
                        )
                        lookback_days = lookback_options[selected_lookback]
                        
                        # 分析筹码流动
                        chip_flow = analyze_chip_flow(chart_data_for_vp, lookback_days=lookback_days)
                        
                        if chip_flow:
                            # 主力行为判断
                            st.markdown(f"## {chip_flow['action_emoji']} **{chip_flow['action']}**")
                            st.caption(chip_flow['action_desc'])
                            
                            # 详细指标
                            cf1, cf2, cf3 = st.columns(3)
                            with cf1:
                                st.metric(
                                    "低位筹码变化", 
                                    f"{chip_flow['low_chip_increase']:+.1f}%",
                                    help="当前价下方20%区间的筹码变化"
                                )
                            with cf2:
                                st.metric(
                                    "高位筹码流出", 
                                    f"{chip_flow['high_chip_decrease']:+.1f}%",
                                    help="当前价上方20%区间的筹码减少"
                                )
                            with cf3:
                                st.metric(
                                    "平均成本变化", 
                                    f"{chip_flow['cost_change_pct']:+.1f}%",
                                    delta=f"${chip_flow['past_avg_cost']:.2f} → ${chip_flow['current_avg_cost']:.2f}"
                                )
                            
                            cf4, cf5 = st.columns(2)
                            with cf4:
                                st.metric(
                                    "当前价附近筹码",
                                    f"{chip_flow['near_chip_change']:+.1f}%",
                                    help="±10%区间筹码变化"
                                )
                            with cf5:
                                st.metric(
                                    "集中度变化",
                                    f"{chip_flow['concentration_change']:+.1f}%",
                                    delta=f"{chip_flow['past_concentration']*100:.0f}% → {chip_flow['current_concentration']*100:.0f}%"
                                )
                            
                            # 筹码流动对比图
                            with st.expander("📊 查看筹码流动对比图", expanded=False):
                                # 对比图: 过去 vs 现在
                                st.markdown("#### 筹码分布对比")
                                flow_fig = create_chip_flow_chart(chip_flow, symbol)
                                if flow_fig:
                                    st.plotly_chart(flow_fig, use_container_width=True)
                                
                                st.markdown("#### 筹码增减变化")
                                change_fig = create_chip_change_chart(chip_flow)
                                if change_fig:
                                    st.plotly_chart(change_fig, use_container_width=True)
                                    
                                # 解读
                                st.info("""
                                **解读**: 
                                - 对比图: 灰色(过去) 在左，蓝色(现在) 在右
                                - 变化图: 🔴红色=筹码增加，🟢绿色=筹码减少
                                - **建仓**: 低位红色 + 高位绿色 | **出货**: 高位红色 + 低位绿色
                                """)
                        else:
                            st.warning("数据不足，无法分析筹码流动")
                        
                        st.divider()
                        
                        # 显示当前周期的 BLUE 值
                        if selected_period == 'daily':
                            st.info(f"📊 当前日线 BLUE: **{selected_row.get('Day BLUE', 0):.0f}**")
                        elif selected_period == 'weekly':
                            st.info(f"📊 当前周线 BLUE: **{selected_row.get('Week BLUE', 0):.0f}**")
                        else:
                            st.info(f"📊 当前月线 BLUE: **{selected_row.get('Month BLUE', 0):.0f}**")
                    else:
                        st.error("无法获取历史数据")
                except Exception as e:
                    st.error(f"图表加载失败: {e}")

        with info_col:
            # --- 0. 公司档案 (基本面) ---
            st.markdown("### 🏢 公司档案")
            name = selected_row.get('Name', symbol)
            industry = selected_row.get('Industry', 'Unknown')
            mkt_cap_str = selected_row.get('Mkt Cap', 'N/A')
            
            st.markdown(f"**{name}**")
            st.caption(f"行业: {industry}")
            st.metric("市值", mkt_cap_str)
            
            st.divider()

            st.markdown("### 📝 核心指标")
            
            # CSV 中的值 (扫描时)
            csv_day_blue = selected_row.get('Day BLUE', 0)
            csv_week_blue = selected_row.get('Week BLUE', 0)
            csv_month_blue = selected_row.get('Month BLUE', 0)
            csv_date = selected_row.get('Date', 'N/A')
            adx_val = selected_row.get('ADX', 0)
            turnover_val = selected_row.get('Turnover', 0)
            pr_val = selected_row.get('Profit_Ratio', 0.5)
            
            # 实时计算 BLUE (如果有 hist_data)
            realtime_day_blue = 0
            realtime_week_blue = 0
            realtime_month_blue = 0
            realtime_date = "N/A"
            
            try:
                if 'hist_data' in dir() and hist_data is not None and not hist_data.empty:
                    realtime_date = hist_data.index[-1].strftime('%Y-%m-%d')
                    
                    # 日线
                    rt_blue = calculate_blue_signal_series(
                        hist_data['Open'].values, hist_data['High'].values,
                        hist_data['Low'].values, hist_data['Close'].values
                    )
                    realtime_day_blue = rt_blue[-1] if len(rt_blue) > 0 else 0
                    
                    # 周线
                    df_w = hist_data.resample('W-FRI').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                    }).dropna()
                    if len(df_w) >= 10:
                        rt_week = calculate_blue_signal_series(
                            df_w['Open'].values, df_w['High'].values,
                            df_w['Low'].values, df_w['Close'].values
                        )
                        realtime_week_blue = rt_week[-1] if len(rt_week) > 0 else 0
                    
                    # 月线
                    df_m = hist_data.resample('ME').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                    }).dropna()
                    if len(df_m) >= 6:
                        rt_month = calculate_blue_signal_series(
                            df_m['Open'].values, df_m['High'].values,
                            df_m['Low'].values, df_m['Close'].values
                        )
                        realtime_month_blue = rt_month[-1] if len(rt_month) > 0 else 0
            except:
                pass
            
            # === BLUE 数据源选择 ===
            st.markdown("**🟦 BLUE 信号**")
            
            data_source = st.radio(
                "数据来源",
                options=[f"📅 实时 ({realtime_date})", f"📋 扫描时 ({csv_date})"],
                horizontal=True,
                key=f"blue_source_{symbol}",
                index=0
            )
            
            # 根据选择显示对应数据
            if "实时" in data_source:
                day_blue = realtime_day_blue
                week_blue = realtime_week_blue
                month_blue = realtime_month_blue
                show_date = realtime_date
            else:
                day_blue = csv_day_blue
                week_blue = csv_week_blue
                month_blue = csv_month_blue
                show_date = csv_date
            
            b1, b2, b3 = st.columns(3)
            with b1:
                color = "🟢" if day_blue > 100 else "⚪"
                st.metric(f"{color} 日线", f"{day_blue:.0f}")
            with b2:
                color = "🟢" if week_blue > 100 else "⚪"
                st.metric(f"{color} 周线", f"{week_blue:.0f}")
            with b3:
                color = "🟢" if month_blue > 100 else "⚪"
                st.metric(f"{color} 月线", f"{month_blue:.0f}")
            
            # 对比提示
            if realtime_date != "N/A" and csv_date != "N/A":
                day_diff = realtime_day_blue - csv_day_blue
                if abs(day_diff) > 30:
                    if day_diff > 0:
                        st.success(f"📈 日线 BLUE 上升: {csv_day_blue:.0f} → {realtime_day_blue:.0f} (+{day_diff:.0f})")
                    else:
                        st.warning(f"📉 日线 BLUE 下降: {csv_day_blue:.0f} → {realtime_day_blue:.0f} ({day_diff:.0f})")
            
            # 其他核心指标
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("📈 ADX", f"{adx_val:.1f}", help="趋势强度")
            with m2:
                st.metric("💧 成交额", f"${turnover_val:.1f}M", help="日成交额")
            with m3:
                st.metric("💰 获利盘", f"{pr_val*100:.0f}%", help="获利盘比例")
            st.divider()

            st.markdown("### 🧠 策略逻辑")
            strategy = selected_row.get('Strategy', 'N/A')
            regime = selected_row.get('Regime', 'N/A')
            thresh = selected_row.get('Adaptive_Thresh', 100)
            wave_phase = selected_row.get('Wave_Phase', 'N/A')
            wave_desc = selected_row.get('Wave_Desc', 'N/A')
            chan_signal = selected_row.get('Chan_Signal', 'N/A')
            chan_desc = selected_row.get('Chan_Desc', 'N/A')
            
            st.success(f"**触发策略**: {strategy}")
            
            col_w, col_c = st.columns(2)
            with col_w:
                st.info(f"**🌊 波浪**: {wave_desc} ({wave_phase})")
            with col_c:
                if "3rd Buy" in str(chan_signal):
                    st.success(f"**🧘 缠论**: {chan_desc}")
                elif "1st Buy" in str(chan_signal):
                    st.warning(f"**🧘 缠论**: {chan_desc}")
                else:
                    st.write(f"**🧘 缠论**: {chan_desc}")
            
            st.caption(f"入选理由分析：")
            st.markdown(f"""
            *   **市场属性**: `{regime}`
            *   **自适应阈值**: **{thresh}**
            *   **当前信号**: BLUE = **{day_blue:.1f}** ( > {thresh})
            """)
            st.divider()

            st.markdown("### 🛡️ 风控与仓位")
            sl_price = selected_row.get('Stop Loss')
            curr_price = selected_row.get('Price')
            shares = selected_row.get('Shares Rec')
            
            if pd.notna(sl_price) and pd.notna(curr_price):
                risk_pct = (curr_price - sl_price) / curr_price * 100
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("建议买入", f"{int(shares)} 股" if pd.notna(shares) else "N/A", help="基于 $1000 风险敞口")
                with col_b:
                     st.metric("止损价格", f"${sl_price:.2f}", f"-{risk_pct:.1f}%")
                st.caption(f"止损逻辑: 价格回撤至 {sl_price:.2f} (约 {risk_pct:.1f}%) 时离场。")
            
            st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。请结合大盘环境自主决策。")
    else:
        st.info("👈 请在上方表格中点击一行，查看该股票的详细图表和分析。")


def render_stock_lookup_page():
    """个股查询页面 - 输入任意股票代码，自动获取数据并生成详情"""
    st.header("🔍 个股查询")
    st.info("输入任意股票代码，系统将自动获取数据并生成完整的技术分析报告。")
    
    # 输入区域
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol_input = st.text_input("股票代码", value="", placeholder="例如: AAPL, NVDA, TSLA")
        symbol = symbol_input.upper().strip() if symbol_input else ""
        
        search_btn = st.button("🔍 查询", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("""
        **支持的股票类型:**
        - 美股 (NYSE, NASDAQ): AAPL, NVDA, TSLA, GOOGL...
        - ETF: SPY, QQQ, IWM...
        """)
    
    if search_btn and symbol:
        with st.spinner(f"正在获取 {symbol} 的数据，请稍候..."):
            try:
                # 获取历史数据 (10年)
                hist_data = fetch_data_from_polygon(symbol, days=3650)
                
                if hist_data is None or hist_data.empty:
                    st.error(f"❌ 无法获取 {symbol} 的数据，请检查股票代码是否正确。")
                    return
                
                st.success(f"✅ 成功获取 {symbol} 的 {len(hist_data)} 天历史数据")
                
                # 获取公司信息
                ticker_info = get_ticker_details(symbol)
                company_name = ticker_info.get('name', symbol) if ticker_info else symbol
                industry = ticker_info.get('sic_description', 'Unknown') if ticker_info else 'Unknown'
                market_cap = ticker_info.get('market_cap', 0) if ticker_info else 0
                
                # 计算各周期指标
                # 日线
                day_blue = calculate_blue_signal_series(
                    hist_data['Open'].values, hist_data['High'].values,
                    hist_data['Low'].values, hist_data['Close'].values
                )
                day_blue_val = day_blue[-1] if len(day_blue) > 0 else 0
                
                # 周线
                df_weekly = hist_data.resample('W-FRI').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
                week_blue_val = 0
                if len(df_weekly) >= 10:
                    week_blue = calculate_blue_signal_series(
                        df_weekly['Open'].values, df_weekly['High'].values,
                        df_weekly['Low'].values, df_weekly['Close'].values
                    )
                    week_blue_val = week_blue[-1] if len(week_blue) > 0 else 0
                
                # 月线
                df_monthly = hist_data.resample('ME').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
                month_blue_val = 0
                if len(df_monthly) >= 6:
                    month_blue = calculate_blue_signal_series(
                        df_monthly['Open'].values, df_monthly['High'].values,
                        df_monthly['Low'].values, df_monthly['Close'].values
                    )
                    month_blue_val = month_blue[-1] if len(month_blue) > 0 else 0
                
                # ADX
                adx_series = calculate_adx_series(
                    hist_data['High'].values, hist_data['Low'].values, hist_data['Close'].values
                )
                adx_val = adx_series[-1] if len(adx_series) > 0 else 0
                
                # 黑马/掘地信号
                heima, juedi = calculate_heima_signal_series(
                    hist_data['High'].values, hist_data['Low'].values,
                    hist_data['Close'].values, hist_data['Open'].values
                )
                has_heima = heima[-1] if len(heima) > 0 else False
                has_juedi = juedi[-1] if len(juedi) > 0 else False
                
                curr_price = hist_data['Close'].iloc[-1]
                turnover = (hist_data['Close'].iloc[-1] * hist_data['Volume'].iloc[-1]) / 1_000_000
                
                st.divider()
                
                # === 显示详情页 (复用扫描页的布局) ===
                st.subheader(f"🔍 {symbol} - {company_name}")
                
                # 顶部指标卡片
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                with m1:
                    st.metric("当前价格", f"${curr_price:.2f}")
                with m2:
                    st.metric("日 BLUE", f"{day_blue_val:.0f}", 
                             delta="信号" if day_blue_val > 100 else None)
                with m3:
                    st.metric("周 BLUE", f"{week_blue_val:.0f}",
                             delta="信号" if week_blue_val > 100 else None)
                with m4:
                    st.metric("月 BLUE", f"{month_blue_val:.0f}",
                             delta="信号" if month_blue_val > 100 else None)
                with m5:
                    st.metric("ADX", f"{adx_val:.1f}",
                             delta="强趋势" if adx_val > 25 else None)
                with m6:
                    signal_text = []
                    if has_heima:
                        signal_text.append("黑马")
                    if has_juedi:
                        signal_text.append("掘地")
                    st.metric("特殊信号", " + ".join(signal_text) if signal_text else "无")
                
                st.divider()
                
                # 图表区域
                chart_col, info_col = st.columns([2, 1])
                
                with chart_col:
                    # 周期切换
                    period_options = {"📅 日线": "daily", "📆 周线": "weekly", "🗓️ 月线": "monthly"}
                    selected_period_label = st.radio(
                        "选择周期",
                        options=list(period_options.keys()),
                        horizontal=True,
                        index=0,
                        key=f"lookup_period_{symbol}"
                    )
                    selected_period = period_options[selected_period_label]
                    
                    # 根据周期选择数据
                    if selected_period == 'weekly':
                        display_data = df_weekly
                        chart_title = f"{symbol} - 周线图"
                    elif selected_period == 'monthly':
                        display_data = df_monthly
                        chart_title = f"{symbol} - 月线图"
                    else:
                        display_data = hist_data.tail(365)
                        chart_title = f"{symbol} - 日线图"
                    
                    # 日期滑动条
                    if len(display_data) > 10:
                        date_list = display_data.index.tolist()
                        default_idx = len(date_list) - 1
                        
                        selected_date_idx = st.slider(
                            "📅 拖动选择日期 (筹码分布会动态变化)",
                            min_value=10,
                            max_value=len(date_list) - 1,
                            value=default_idx,
                            format="",
                            key=f"lookup_slider_{symbol}_{selected_period}"
                        )
                        
                        selected_date = date_list[selected_date_idx]
                        st.caption(f"🎯 选中日期: **{selected_date.strftime('%Y-%m-%d')}** | 收盘价: **${display_data.loc[selected_date, 'Close']:.2f}**")
                        
                        chart_data_for_vp = display_data.iloc[:selected_date_idx + 1].copy()
                    else:
                        chart_data_for_vp = display_data.copy()
                        selected_date = display_data.index[-1]
                    
                    # 创建图表
                    fig = create_candlestick_chart_dynamic(
                        display_data,
                        chart_data_for_vp,
                        symbol, chart_title,
                        period=selected_period,
                        show_volume_profile=True,
                        highlight_date=selected_date
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 筹码分析指标
                    if hasattr(fig, '_chip_analysis'):
                        chip = fig._chip_analysis
                        
                        st.markdown(f"### 📊 筹码分析 {chip.get('buy_signal_strength', '')}")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            profit_pct = chip.get('profit_ratio', 0) * 100
                            st.metric("🟢 获利盘", f"{profit_pct:.1f}%")
                        with c2:
                            trapped_pct = chip.get('trapped_ratio', 0) * 100
                            st.metric("🔴 套牢盘", f"{trapped_pct:.1f}%")
                        with c3:
                            conc = chip.get('concentration', 0) * 100
                            st.metric("📍 集中度", f"{conc:.1f}%")
                        with c4:
                            avg_cost = chip.get('avg_cost', 0)
                            st.metric("💰 平均成本", f"${avg_cost:.2f}")
                        
                        # 支撑压力位
                        st.markdown("**关键价位**")
                        p1, p2, p3 = st.columns(3)
                        with p1:
                            support = chip.get('support_price')
                            st.metric("⬇️ 支撑位", f"${support:.2f}" if support else "N/A")
                        with p2:
                            poc = chip.get('poc_price', 0)
                            st.metric("🎯 筹码峰(POC)", f"${poc:.2f}")
                        with p3:
                            resist = chip.get('resistance_price')
                            st.metric("⬆️ 压力位", f"${resist:.2f}" if resist else "N/A")
                    
                    st.divider()
                    
                    # 主力动向分析
                    st.markdown("### 🏦 主力动向分析")
                    
                    lookback_options = {"5天": 5, "10天": 10, "20天": 20, "30天": 30, "60天": 60}
                    selected_lookback = st.select_slider(
                        "对比周期",
                        options=list(lookback_options.keys()),
                        value="20天",
                        key=f"lookup_lookback_{symbol}"
                    )
                    lookback_days = lookback_options[selected_lookback]
                    
                    chip_flow = analyze_chip_flow(chart_data_for_vp, lookback_days=lookback_days)
                    
                    if chip_flow:
                        st.markdown(f"## {chip_flow['action_emoji']} **{chip_flow['action']}**")
                        st.caption(chip_flow['action_desc'])
                        
                        cf1, cf2, cf3 = st.columns(3)
                        with cf1:
                            st.metric("低位筹码变化", f"{chip_flow['low_chip_increase']:+.1f}%")
                        with cf2:
                            st.metric("高位筹码流出", f"{chip_flow['high_chip_decrease']:+.1f}%")
                        with cf3:
                            st.metric("平均成本变化", f"{chip_flow['cost_change_pct']:+.1f}%")
                        
                        with st.expander("📊 查看筹码流动对比图", expanded=False):
                            flow_fig = create_chip_flow_chart(chip_flow, symbol)
                            if flow_fig:
                                st.plotly_chart(flow_fig, use_container_width=True)
                            
                            change_fig = create_chip_change_chart(chip_flow)
                            if change_fig:
                                st.plotly_chart(change_fig, use_container_width=True)
                    else:
                        st.warning("数据不足，无法分析筹码流动")
                
                with info_col:
                    # 公司档案
                    st.markdown("### 🏢 公司档案")
                    st.markdown(f"**{company_name}**")
                    st.caption(f"行业: {industry}")
                    if market_cap:
                        st.metric("市值", format_large_number(market_cap))
                    
                    st.divider()
                    
                    # BLUE 信号详情
                    st.markdown("### 🟦 BLUE 信号")
                    
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        color = "🟢" if day_blue_val > 100 else "⚪"
                        st.metric(f"{color} 日线", f"{day_blue_val:.0f}")
                    with b2:
                        color = "🟢" if week_blue_val > 100 else "⚪"
                        st.metric(f"{color} 周线", f"{week_blue_val:.0f}")
                    with b3:
                        color = "🟢" if month_blue_val > 100 else "⚪"
                        st.metric(f"{color} 月线", f"{month_blue_val:.0f}")
                    
                    # 信号解读
                    signals = []
                    if day_blue_val > 100:
                        signals.append("日线抄底信号")
                    if week_blue_val > 100:
                        signals.append("周线抄底信号")
                    if month_blue_val > 100:
                        signals.append("月线抄底信号")
                    if has_heima:
                        signals.append("黑马信号")
                    if has_juedi:
                        signals.append("掘地信号")
                    
                    if signals:
                        st.success(f"**当前信号**: {', '.join(signals)}")
                    else:
                        st.info("当前无明显买入信号")
                    
                    st.divider()
                    
                    # 趋势强度
                    st.markdown("### 📈 趋势分析")
                    st.metric("ADX 趋势强度", f"{adx_val:.1f}")
                    
                    if adx_val > 40:
                        st.success("**极强趋势** - 顺势操作")
                    elif adx_val > 25:
                        st.info("**中等趋势** - 可考虑入场")
                    else:
                        st.warning("**弱趋势/震荡** - 谨慎操作")
                    
                    st.divider()
                    
                    # 成交额
                    st.markdown("### 💧 流动性")
                    st.metric("日成交额", f"${turnover:.2f}M")
                    
                    if turnover > 100:
                        st.success("流动性极佳")
                    elif turnover > 10:
                        st.info("流动性良好")
                    elif turnover > 1:
                        st.warning("流动性一般")
                    else:
                        st.error("流动性较差")
                
                st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。")
                
            except Exception as e:
                st.error(f"❌ 查询出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    elif search_btn and not symbol:
        st.warning("请输入股票代码")


def render_signal_tracker_page():
    """信号追踪页面 - 查看历史扫描信号的后续表现"""
    st.header("📈 信号追踪 (Signal Tracker)")
    st.info("查看历史扫描结果中股票的后续走势，验证信号有效性。")
    
    # 加载所有历史扫描结果
    files = glob.glob(os.path.join(current_dir, "enhanced_scan_results_*.csv"))
    if not files:
        st.warning("未找到历史扫描结果文件。")
        return
    
    # 按日期排序
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    
    # 提取文件名中的日期
    file_options = {}
    for f in files_sorted[:20]:  # 只显示最近20个
        basename = os.path.basename(f)
        # enhanced_scan_results_US_20260107_104820.csv
        try:
            parts = basename.replace('.csv', '').split('_')
            date_str = parts[3]  # 20260107
            time_str = parts[4]  # 104820
            display_name = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
            file_options[display_name] = f
        except:
            file_options[basename] = f
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_date = st.selectbox(
            "选择历史扫描日期",
            options=list(file_options.keys()),
            index=0
        )
        
        if selected_date:
            selected_file = file_options[selected_date]
            
            try:
                hist_df = pd.read_csv(selected_file)
                st.success(f"加载了 {len(hist_df)} 条信号记录")
                
                # 筛选条件
                st.markdown("### 筛选条件")
                
                min_blue = st.slider("最小 Day BLUE", 0, 200, 100)
                
                filter_options = st.multiselect(
                    "信号类型",
                    ["Strat_D_Trend", "Strat_C_Resonance", "Is_Heima"],
                    default=["Strat_D_Trend"]
                )
                
            except Exception as e:
                st.error(f"加载文件失败: {e}")
                return
    
    with col2:
        if selected_date and 'hist_df' in dir():
            # 应用筛选
            filtered_df = hist_df.copy()
            
            if 'Blue_Daily' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Blue_Daily'] >= min_blue]
            
            for opt in filter_options:
                if opt in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[opt] == True]
            
            st.markdown(f"### 筛选后: {len(filtered_df)} 条信号")
            
            if len(filtered_df) == 0:
                st.warning("没有符合条件的信号")
                return
            
            # 显示信号列表
            display_cols = ['Symbol', 'Price', 'Blue_Daily', 'Blue_Weekly', 'Date']
            existing_cols = [c for c in display_cols if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[existing_cols].head(50),
                use_container_width=True,
                hide_index=True
            )
    
    st.divider()
    
    # 信号追踪分析
    st.markdown("## 📊 信号后续表现分析")
    
    if 'filtered_df' in dir() and len(filtered_df) > 0:
        # 选择要追踪的股票
        symbols = filtered_df['Symbol'].tolist()[:20]  # 最多20个
        
        selected_symbols = st.multiselect(
            "选择要追踪的股票 (最多5个)",
            options=symbols,
            default=symbols[:min(3, len(symbols))],
            max_selections=5
        )
        
        # 追踪天数
        track_days = st.slider("追踪天数", 5, 60, 20)
        
        if st.button("🔍 分析后续走势", type="primary"):
            if not selected_symbols:
                st.warning("请选择至少一个股票")
                return
            
            # 获取信号日期
            try:
                signal_date_str = filtered_df[filtered_df['Symbol'] == selected_symbols[0]]['Date'].iloc[0]
                signal_date = pd.to_datetime(signal_date_str)
            except:
                signal_date = pd.Timestamp.now() - pd.Timedelta(days=30)
            
            st.markdown(f"**信号日期**: {signal_date.strftime('%Y-%m-%d')}")
            
            # 分析每个股票
            results = []
            
            progress_bar = st.progress(0)
            
            for i, symbol in enumerate(selected_symbols):
                progress_bar.progress((i + 1) / len(selected_symbols))
                
                with st.spinner(f"分析 {symbol}..."):
                    try:
                        # 获取数据
                        df = fetch_data_from_polygon(symbol, days=365)
                        
                        if df is None or df.empty:
                            continue
                        
                        # 找到信号日期
                        signal_row = filtered_df[filtered_df['Symbol'] == symbol].iloc[0]
                        signal_price = signal_row.get('Price', 0)
                        signal_blue = signal_row.get('Blue_Daily', 0)
                        
                        # 计算后续表现
                        # 找到信号日期之后的数据
                        df_after = df[df.index >= signal_date].head(track_days + 1)
                        
                        if len(df_after) < 2:
                            continue
                        
                        entry_price = df_after['Close'].iloc[0]
                        
                        # 计算各时间点收益
                        returns = {}
                        for d in [5, 10, 20, track_days]:
                            if d < len(df_after):
                                exit_price = df_after['Close'].iloc[d]
                                ret = (exit_price - entry_price) / entry_price * 100
                                returns[f'{d}D'] = ret
                        
                        # 最大回撤和最大涨幅
                        max_price = df_after['High'].max()
                        min_price = df_after['Low'].min()
                        max_gain = (max_price - entry_price) / entry_price * 100
                        max_dd = (min_price - entry_price) / entry_price * 100
                        
                        # 当前价格
                        current_price = df['Close'].iloc[-1]
                        current_ret = (current_price - entry_price) / entry_price * 100
                        
                        results.append({
                            'Symbol': symbol,
                            'Signal BLUE': signal_blue,
                            'Entry Price': entry_price,
                            'Current Price': current_price,
                            '5D Return': returns.get('5D', None),
                            '10D Return': returns.get('10D', None),
                            '20D Return': returns.get('20D', None),
                            'Max Gain': max_gain,
                            'Max DD': max_dd,
                            'Current Return': current_ret
                        })
                        
                    except Exception as e:
                        st.warning(f"{symbol} 分析失败: {e}")
            
            progress_bar.empty()
            
            if results:
                results_df = pd.DataFrame(results)
                
                # 统计摘要
                st.markdown("### 📋 表现统计")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_5d = results_df['5D Return'].mean() if '5D Return' in results_df else 0
                    st.metric("平均 5 日收益", f"{avg_5d:.2f}%",
                             delta="盈利" if avg_5d > 0 else "亏损",
                             delta_color="normal" if avg_5d > 0 else "inverse")
                
                with col2:
                    avg_10d = results_df['10D Return'].mean() if '10D Return' in results_df else 0
                    st.metric("平均 10 日收益", f"{avg_10d:.2f}%",
                             delta="盈利" if avg_10d > 0 else "亏损",
                             delta_color="normal" if avg_10d > 0 else "inverse")
                
                with col3:
                    avg_20d = results_df['20D Return'].mean() if '20D Return' in results_df else 0
                    st.metric("平均 20 日收益", f"{avg_20d:.2f}%",
                             delta="盈利" if avg_20d > 0 else "亏损",
                             delta_color="normal" if avg_20d > 0 else "inverse")
                
                with col4:
                    win_rate = (results_df['Current Return'] > 0).mean() * 100
                    st.metric("胜率", f"{win_rate:.0f}%",
                             delta="优秀" if win_rate > 60 else ("一般" if win_rate > 40 else "较差"))
                
                # 详细表格
                st.markdown("### 📊 详细数据")
                
                # 格式化显示
                styled_df = results_df.copy()
                for col in ['5D Return', '10D Return', '20D Return', 'Max Gain', 'Max DD', 'Current Return']:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                
                for col in ['Entry Price', 'Current Price']:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # 走势图
                st.markdown("### 📈 后续走势图")
                
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = go.Figure()
                
                for symbol in selected_symbols:
                    try:
                        df = fetch_data_from_polygon(symbol, days=365)
                        if df is None:
                            continue
                        
                        df_after = df[df.index >= signal_date].head(track_days + 1)
                        if len(df_after) < 2:
                            continue
                        
                        entry_price = df_after['Close'].iloc[0]
                        # 归一化为百分比变化
                        normalized = (df_after['Close'] / entry_price - 1) * 100
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(len(normalized))),
                            y=normalized,
                            mode='lines+markers',
                            name=symbol,
                            hovertemplate=f'{symbol}<br>Day %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                        ))
                    except:
                        pass
                
                # 添加零线
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="信号后续收益走势 (归一化)",
                    xaxis_title="交易日",
                    yaxis_title="收益率 (%)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 信号质量评估
                st.markdown("### 🎯 信号质量评估")
                
                avg_return = results_df['Current Return'].mean()
                avg_max_gain = results_df['Max Gain'].mean()
                avg_max_dd = results_df['Max DD'].mean()
                
                if avg_return > 10 and win_rate > 60:
                    st.success(f"""
                    **✅ 优质信号批次**
                    - 平均收益: {avg_return:.2f}%
                    - 胜率: {win_rate:.0f}%
                    - 平均最大涨幅: {avg_max_gain:.2f}%
                    - 平均最大回撤: {avg_max_dd:.2f}%
                    
                    该批次信号表现优秀，策略参数可以继续使用。
                    """)
                elif avg_return > 0 and win_rate > 40:
                    st.info(f"""
                    **🟡 一般信号批次**
                    - 平均收益: {avg_return:.2f}%
                    - 胜率: {win_rate:.0f}%
                    - 平均最大涨幅: {avg_max_gain:.2f}%
                    - 平均最大回撤: {avg_max_dd:.2f}%
                    
                    该批次信号表现一般，建议结合其他指标筛选。
                    """)
                else:
                    st.warning(f"""
                    **⚠️ 低质量信号批次**
                    - 平均收益: {avg_return:.2f}%
                    - 胜率: {win_rate:.0f}%
                    - 平均最大涨幅: {avg_max_gain:.2f}%
                    - 平均最大回撤: {avg_max_dd:.2f}%
                    
                    该批次信号表现不佳，建议调整策略参数或增加筛选条件。
                    """)
            else:
                st.warning("没有成功分析的股票")


def render_backtest_page():
    st.header("🧪 策略回测实验室 (Strategy Lab)")
    st.info("在这里您可以对单只股票进行历史回测，验证策略参数的有效性。")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbol_input = st.text_input("股票代码", value="NVDA", help="例如: NVDA, AAPL")
        symbol = symbol_input.upper().strip() if symbol_input else ""
    with col2:
        market = st.selectbox("市场", ["US", "CN"], index=0)
    with col3:
        initial_capital = st.number_input("初始资金", value=100000.0, step=10000.0)
    with col4:
        days = st.number_input("回测天数", value=1095, step=365, help="365天 = 1年")
        
    col5, col6 = st.columns(2)
    with col5:
        threshold = st.slider("BLUE 买入阈值", min_value=50.0, max_value=200.0, value=100.0, step=10.0)
    with col6:
        commission = st.number_input("佣金费率", value=0.001, format="%.4f")
        
    col7, col8 = st.columns(2)
    with col7:
        require_heima = st.checkbox("✅ 必须包含黑马/掘底信号", value=False, help="更严格：仅当同时出现黑马或掘底信号时才买入")
    with col8:
        require_week_blue = st.checkbox("✅ 必须包含周线BLUE共振", value=False, help="更严格：仅当周线BLUE同时也大于阈值时才买入")
        
    require_vp = st.checkbox("✅ 必须筹码形态良好", value=False, help="过滤掉获利盘极低且被筹码峰压制的假反弹")
    
    # 风控选项
    use_risk_mgmt = st.checkbox("🛡️ 启用专业风控 (ATR止损 + 动态仓位)", value=True, help="启用后，不再全仓买入。基于ATR计算仓位(单笔风险2%)，并使用移动止损。")
    
    # --- 智能推荐模块 ---
    if st.button("🚀 开始回测"):
        with st.spinner(f"正在回测 {symbol} ..."):
            try:
                # 初始化回测引擎
                backtester = SimpleBacktester(
                    symbol=symbol, 
                    market=market, 
                    initial_capital=initial_capital, 
                    days=days, 
                    commission_rate=commission,
                    blue_threshold=threshold,
                    require_heima=require_heima,
                    require_week_blue=require_week_blue,
                    require_vp_filter=require_vp,
                    use_risk_management=use_risk_mgmt
                )
                
                # 加载数据
                if not backtester.load_data():
                    st.error(f"❌ 数据加载失败: 无法获取 {symbol} 的数据。可能是网络问题或API限制，请稍后重试。")
                else:
                    # 运行回测
                    backtester.calculate_signals()
                    backtester.run_backtest()
                    
                    # 显示结果摘要
                    res = backtester.results
                    
                    st.success(f"✅ 回测完成！")
                    
                    # 显示自适应信息
                    if 'Adaptive Info' in res:
                        st.info(f"🤖 **自适应引擎已激活**: {res['Adaptive Info']}")
                    
                    # 关键指标卡片
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("总收益率", f"{res['Total Return']:.2%}", delta_color="normal")
                    m2.metric("年化收益率", f"{res['Annual Return']:.2%}")
                    m3.metric("最大回撤", f"{res['Max Drawdown']:.2%}", delta_color="inverse")
                    m4.metric("胜率", f"{res['Win Rate']:.2%}", f"{res['Total Trades']} 笔交易")
                    
                    # 资金曲线图
                    fig = backtester.plot_results(show=False)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # 交易详情表
                    if backtester.trades:
                        st.subheader("📋 交易记录 & 风控详情")
                        
                        trade_data = []
                        for t in backtester.trades:
                            trade_data.append({
                                "日期": t['date'].strftime('%Y-%m-%d'),
                                "类型": t['type'],
                                "价格": f"{t['price']:.2f}",
                                "数量": t['shares'],
                                "金额": f"{t['value']:.2f}",
                                "盈亏": f"{t.get('pnl', 0):.2f}" if 'pnl' in t else "-",
                                "交易理由": t.get('reason', '-'),
                                "止损价": f"{t.get('stop_loss', 0):.2f}" if t.get('stop_loss', 0) > 0 else "-"
                            })
                        
                        st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
                    else:
                        st.warning("在此期间未触发任何交易。")

                    # 被过滤的信号表
                    if hasattr(backtester, 'rejected_trades') and backtester.rejected_trades:
                        with st.expander("🚫 查看被过滤的信号 (诊断报告)", expanded=True):
                            st.caption("以下信号满足了基础 BLUE 阈值，但被您的高级过滤条件（周线/黑马/筹码分布）拒绝。")
                            
                            rejected_data = []
                            for r in backtester.rejected_trades:
                                rejected_data.append({
                                    "日期": r['date'].strftime('%Y-%m-%d'),
                                    "价格": f"{r['price']:.2f}",
                                    "Day BLUE": f"{r['blue']:.1f}",
                                    "Week BLUE": f"{r.get('week_blue', 0):.1f}",
                                    "拒绝原因 ❌": r['reason']
                                })
                            
                            st.dataframe(pd.DataFrame(rejected_data), use_container_width=True)
                        
            except Exception as e:
                st.error(f"回测出错: {str(e)}")

# --- 主导航 ---

st.sidebar.title("Coral Creek 🌊")
page = st.sidebar.radio("功能导航", ["📊 每日机会扫描", "🔍 个股查询", "📈 信号追踪", "🧪 策略回测实验"])

if page == "📊 每日机会扫描":
    render_scan_page()
elif page == "🔍 个股查询":
    render_stock_lookup_page()
elif page == "📈 信号追踪":
    render_signal_tracker_page()
elif page == "🧪 策略回测实验":
    render_backtest_page()
