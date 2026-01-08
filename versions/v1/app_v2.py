import streamlit as st
import pandas as pd
import glob
import os
import sys
import numpy as np
import plotly.graph_objects as go

# 添加当前目录到路径，以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from chart_utils import create_candlestick_chart
from data_fetcher import get_us_stock_data as fetch_data_from_polygon
from simple_backtest import SimpleBacktester

# 设置页面配置
st.set_page_config(
    page_title="Coral Creek V2.0 - 智能量化系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函数 ---

def load_latest_scan_results():
    """加载最新的扫描结果 CSV"""
    # 查找所有 enhanced_scan_results_*.csv 文件
    files = glob.glob(os.path.join(current_dir, "enhanced_scan_results_*.csv"))
    if not files:
        return None, None
    
    # 按修改时间排序，取最新的
    latest_file = max(files, key=os.path.getmtime)
    
    try:
        df = pd.read_csv(latest_file)
        
        # --- 数据标准化与列名映射 ---
        
        # 1. 映射关键列名
        col_map = {
            'Symbol': 'Ticker',
            'Blue_Daily': 'Day BLUE',
            'Blue_Weekly': 'Week BLUE',
            'Stop_Loss': 'Stop Loss',
            'Shares_Rec': 'Shares Rec',
            'VP_Rating': 'Vol Profile'
        }
        df.rename(columns=col_map, inplace=True)
        
        # 2. 合成 Strategy 列
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
        
        # 3. 合成 Score 列 (0-100)
        def calculate_score(row):
            score = 0
            # BLUE 分 (满分40)
            blue = row.get('Day BLUE', 0)
            score += min(blue / 200, 1.0) * 40
            # ADX 分 (满分30)
            adx = row.get('ADX', 0)
            score += min(adx / 60, 1.0) * 30
            # 筹码分 (满分30)
            pr = row.get('Profit_Ratio', 0.5)
            score += pr * 30
            return int(score)
            
        if 'Score' not in df.columns:
            df['Score'] = df.apply(calculate_score, axis=1)

        # 4. 类型转换
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        if 'Day BLUE' in df.columns:
            df['Day BLUE'] = pd.to_numeric(df['Day BLUE'], errors='coerce')
        
        if 'Week BLUE' in df.columns:
            df['Week BLUE'] = pd.to_numeric(df['Week BLUE'], errors='coerce')
            
        if 'Stop Loss' in df.columns:
            df['Stop Loss'] = pd.to_numeric(df['Stop Loss'], errors='coerce')
            
        if 'Score' in df.columns:
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        
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
    
    # 1. 加载数据
    df, filename = load_latest_scan_results()

    if df is None:
        st.warning("⚠️ 未找到扫描结果文件。请先运行 `enhanced_scan.py`。")
        st.info("💡 提示: 在终端运行 `python versions/v1/enhanced_scan.py` 生成最新数据。")
        return

    # 侧边栏：文件信息和全局过滤
    with st.sidebar:
        st.divider()
        st.header("📂 数据源")
        st.caption(f"当前文件: `{filename}`")
        
        file_time = os.path.getmtime(os.path.join(current_dir, filename))
        st.caption(f"生成时间: {pd.to_datetime(file_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("🔄 刷新数据"):
            st.rerun()
            
        st.subheader("🔍 快速筛选")
        
        # 策略筛选
        if 'Strategy' in df.columns:
            all_strategies = df['Strategy'].unique().tolist()
            selected_strategies = st.multiselect("策略类型", all_strategies, default=all_strategies)
            if selected_strategies:
                df = df[df['Strategy'].isin(selected_strategies)]
        
        # 评分筛选
        min_score = st.slider("最低评分 (Score)", 0, 100, 60)
        df = df[df['Score'] >= min_score]
        
        # 价格筛选
        if 'Price' in df.columns:
            max_price = st.number_input("最高价格 ($)", value=10000, step=100)
            df = df[df['Price'] <= max_price]

    # 2. 顶部仪表盘
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("今日机会总数", f"{len(df)} 只")

    with col2:
        super_stars = len(df[df['Score'] >= 90])
        st.metric("🌟 五星级机会", f"{super_stars} 只", help="评分 >= 90 的极品机会")

    with col3:
        trend_opps = len(df[df['Strategy'].str.contains('Trend', na=False)])
        st.metric("🚀 趋势突破", f"{trend_opps} 只", help="Strategy D: 趋势跟随")

    with col4:
        mood, color = get_market_mood(df)
        st.markdown(f"**市场情绪**")
        st.markdown(f"<h3 style='color: {color}; margin-top: -10px;'>{mood}</h3>", unsafe_allow_html=True)

    st.divider()

    # 3. 机会清单
    st.subheader("📋 机会清单 (Opportunity Matrix)")

    column_config = {
        "Ticker": st.column_config.TextColumn("代码", help="股票代码", width="small"),
        "Price": st.column_config.NumberColumn("现价", format="$%.2f"),
        "Day BLUE": st.column_config.NumberColumn("Day BLUE", format="%.1f"),
        "Week BLUE": st.column_config.NumberColumn("Week BLUE", format="%.1f"),
        "Strategy": st.column_config.TextColumn("策略标签", width="medium"),
        "Regime": st.column_config.TextColumn("波动属性", width="medium"),
        "Score": st.column_config.ProgressColumn(
            "综合评分", format="%d", min_value=0, max_value=100,
            help="基于信号强度、VP位置和波动率的综合打分"
        ),
        "Stop Loss": st.column_config.NumberColumn("止损价", format="$%.2f", help="建议止损位"),
        "Shares Rec": st.column_config.NumberColumn("建议仓位", format="%d 股", help="基于$1000风险敞口的建议股数"),
        "Risk/Trade": st.column_config.TextColumn("单笔风险", help="每笔交易的风险金额"),
        "Wave_Desc": st.column_config.TextColumn("波浪形态", width="medium", help="基于 ZigZag 识别的市场阶段")
    }

    display_cols = ['Ticker', 'Price', 'Strategy', 'Score', 'Wave_Desc', 'Day BLUE', 'Week BLUE', 'Stop Loss', 'Shares Rec', 'Regime', 'Vol Profile']
    existing_cols = [c for c in display_cols if c in df.columns]

    event = st.dataframe(
        df[existing_cols].style.background_gradient(subset=['Day BLUE'], cmap='Blues'),
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
            with st.spinner(f"正在加载 {symbol} 图表..."):
                try:
                    hist_data = fetch_data_from_polygon(symbol, days=365)
                    if hist_data is not None and not hist_data.empty:
                        fig = create_candlestick_chart(
                            hist_data, symbol, symbol,
                            period='daily', show_volume_profile=True,
                            stop_loss_price=selected_row.get('Stop Loss')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("无法获取历史数据")
                except Exception as e:
                    st.error(f"图表加载失败: {e}")

        with info_col:
            st.markdown("### 📝 评分雷达")
            score = selected_row.get('Score', 0)
            blue_val = selected_row.get('Day BLUE', 0)
            adx_val = selected_row.get('ADX', 0)
            pr_val = selected_row.get('Profit_Ratio', 0.5)
            
            blue_score = min(blue_val / 200, 1.0) * 40
            adx_score = min(adx_val / 60, 1.0) * 30
            chip_score = pr_val * 30
            
            st.metric("综合评分", f"{int(score)} 分")
            with st.expander("查看得分细则", expanded=True):
                st.markdown(f"""
                - **🟦 信号强度**: **{int(blue_score)}/40** (BLUE={blue_val:.1f})
                - **📈 趋势强度**: **{int(adx_score)}/30** (ADX={adx_val:.1f})
                - **💰 筹码结构**: **{int(chip_score)}/30** (获利盘 {pr_val*100:.0f}%)
                """)
            st.divider()

            st.markdown("### 🧠 策略逻辑")
            strategy = selected_row.get('Strategy', 'N/A')
            regime = selected_row.get('Regime', 'N/A')
            thresh = selected_row.get('Adaptive_Thresh', 100)
            wave_phase = selected_row.get('Wave_Phase', 'N/A')
            wave_desc = selected_row.get('Wave_Desc', 'N/A')
            
            st.success(f"**触发策略**: {strategy}")
            st.info(f"**🌊 波浪形态**: {wave_desc} ({wave_phase})")
            st.caption(f"入选理由分析：")
            st.markdown(f"""
            *   **市场属性**: `{regime}`
            *   **自适应阈值**: **{thresh}**
            *   **当前信号**: BLUE = **{blue_val:.1f}** ( > {thresh})
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
page = st.sidebar.radio("功能导航", ["📊 每日机会扫描", "🧪 策略回测实验"])

if page == "📊 每日机会扫描":
    render_scan_page()
elif page == "🧪 策略回测实验":
    render_backtest_page()
