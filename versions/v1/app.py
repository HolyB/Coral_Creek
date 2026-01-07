import streamlit as st
from database_manager import StockDatabase
import pandas as pd
import numpy as np
import os
import json
from data_fetcher import get_stock_data
from chart_utils import create_candlestick_chart
from simple_backtest import SimpleBacktester

st.set_page_config(page_title="股票信号扫描监控台", layout="wide", page_icon="📈")

st.title("📈 股票信号扫描监控台")

# 侧边栏导航
page = st.sidebar.radio("选择功能", ["📊 信号扫描", "❤️ 自选看板", "🔄 策略回测"])

# 初始化数据库连接
try:
    db = StockDatabase()
except Exception as e:
    st.error(f"无法连接数据库: {e}")
    st.stop()

# --------------------------
# 页面 1: 信号扫描 (原有功能)
# --------------------------
if page == "📊 信号扫描":
    # Sidebar Filters
    st.sidebar.header("数据过滤")
    
    dates = db.get_available_dates()
    if not dates:
        st.sidebar.warning("数据库中暂无数据。请先运行扫描脚本。")
        selected_date = None
    else:
        selected_date = st.sidebar.selectbox("选择扫描日期", dates, index=0)
    
    market_filter = st.sidebar.multiselect("市场", ["CN", "US"], default=["CN", "US"])
    
    # 收藏筛选
    st.sidebar.markdown("### ⭐ 收藏/自选")
    show_favorites_only = st.sidebar.checkbox("只看自选股", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.header("高级筛选")

    # 价格范围筛选
    st.sidebar.markdown("#### 💰 价格范围")
    price_filter_enabled = st.sidebar.checkbox("启用价格筛选", value=False)
    if price_filter_enabled:
        price_col1, price_col2 = st.sidebar.columns(2)
        with price_col1:
            min_price = st.number_input("最低价", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="min_price")
        with price_col2:
            max_price = st.number_input("最高价", min_value=0.0, value=1000.0, step=0.01, format="%.2f", key="max_price")
    else:
        min_price, max_price = None, None
    
    # 成交额范围筛选
    st.sidebar.markdown("#### 📊 成交额范围（万元）")
    turnover_filter_enabled = st.sidebar.checkbox("启用成交额筛选", value=False)
    if turnover_filter_enabled:
        turnover_col1, turnover_col2 = st.sidebar.columns(2)
        with turnover_col1:
            min_turnover = st.number_input("最小成交额", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="min_turnover")
        with turnover_col2:
            max_turnover = st.number_input("最大成交额", min_value=0.0, value=100000.0, step=100.0, format="%.2f", key="max_turnover")
    else:
        min_turnover, max_turnover = None, None
    
    # BLUE数值范围筛选
    st.sidebar.markdown("#### 🔵 BLUE数值范围")
    blue_filter_enabled = st.sidebar.checkbox("启用BLUE数值筛选", value=False)
    if blue_filter_enabled:
        blue_type = st.sidebar.radio("选择类型", ["日线BLUE", "周线BLUE", "两者都筛选"], horizontal=False, key="blue_type")
        blue_col1, blue_col2 = st.sidebar.columns(2)
        with blue_col1:
            min_blue = st.number_input("最小BLUE值", min_value=0.0, value=100.0, step=1.0, format="%.2f", key="min_blue")
        with blue_col2:
            max_blue = st.number_input("最大BLUE值", min_value=0.0, value=500.0, step=1.0, format="%.2f", key="max_blue")
    else:
        blue_type, min_blue, max_blue = None, None, None
    
    # 信号天数/周数筛选
    st.sidebar.markdown("#### 📈 信号强度")
    signal_strength_enabled = st.sidebar.checkbox("启用信号强度筛选", value=False)
    if signal_strength_enabled:
        day_blue_days = st.sidebar.number_input("日线BLUE最少天数", min_value=0, value=3, step=1, key="day_blue_days")
        week_blue_weeks = st.sidebar.number_input("周线BLUE最少周数", min_value=0, value=2, step=1, key="week_blue_weeks")
    else:
        day_blue_days, week_blue_weeks = None, None
    
    st.sidebar.markdown("---")
    st.sidebar.header("历史对比")
    compare_enabled = st.sidebar.checkbox("启用日期对比", value=False)
    if compare_enabled:
        dates = db.get_available_dates()
        if len(dates) >= 2:
            compare_date1 = st.sidebar.selectbox("对比日期1", dates, index=0, key="compare_date1")
            compare_date2 = st.sidebar.selectbox("对比日期2", dates, index=1 if len(dates) > 1 else 0, key="compare_date2")
        else:
            compare_date1 = None
            compare_date2 = None
            st.sidebar.warning("需要至少2个扫描日期才能对比")
    else:
        compare_date1 = None
        compare_date2 = None
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"数据库路径: {db.db_path}")

    # --------------------------
    # 页面 1: 信号扫描 (原有功能)
    # --------------------------
    if page == "📊 信号扫描":
        if selected_date:
            df = db.get_results_by_date(selected_date)
            
            if not df.empty:
                # 获取自选股数据
                favorites_df = db.get_all_favorites()
                favorite_symbols = set(favorites_df['symbol'].tolist()) if not favorites_df.empty else set()
        
                # Filter by market
            if market_filter:
                df = df[df['market'].isin(market_filter)]
            
            # Filter by Favorites
            if show_favorites_only:
                df = df[df['symbol'].isin(favorite_symbols)]
            
            # Summary Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("扫描到的股票总数", len(df))
            with col2:
                cn_count = len(df[df['market'] == 'CN'])
                st.metric("A股数量", cn_count)
            with col3:
                us_count = len(df[df['market'] == 'US'])
                st.metric("美股数量", us_count)
            with col4:
                day_blue_count = len(df[df.get('has_day_blue', pd.Series([False]*len(df))) == True]) if 'has_day_blue' in df.columns else 0
                st.metric("日线BLUE", day_blue_count)
            with col5:
                week_blue_count = len(df[df.get('has_week_blue', pd.Series([False]*len(df))) == True]) if 'has_week_blue' in df.columns else 0
                st.metric("周线BLUE", week_blue_count)
        
            # 统计分析
            st.markdown("### 📊 统计分析")
            stat_tab1, stat_tab2, stat_tab3 = st.tabs(["信号分布", "BLUE数值分布", "市场对比"])
        
            with stat_tab1:
                if 'has_day_blue' in df.columns and 'has_week_blue' in df.columns:
                    both_blue_count = len(df[(df['has_day_blue'] == True) & (df['has_week_blue'] == True)])
                    only_day_blue = len(df[(df['has_day_blue'] == True) & (df['has_week_blue'] == False)])
                    only_week_blue = len(df[(df['has_day_blue'] == False) & (df['has_week_blue'] == True)])
                    no_blue = len(df[(df['has_day_blue'] == False) & (df['has_week_blue'] == False)])
                
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("日线+周线都有", both_blue_count)
                    with stat_col2:
                        st.metric("仅日线BLUE", only_day_blue)
                    with stat_col3:
                        st.metric("仅周线BLUE", only_week_blue)
                    with stat_col4:
                        st.metric("无BLUE信号", no_blue)
        
            with stat_tab2:
                if 'blue_daily' in df.columns and 'blue_weekly' in df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**日线BLUE数值分布**")
                        day_blue_values = df[df['blue_daily'] > 0]['blue_daily']
                        if len(day_blue_values) > 0:
                            st.write(f"平均值: {day_blue_values.mean():.2f}")
                            st.write(f"中位数: {day_blue_values.median():.2f}")
                            st.write(f"最大值: {day_blue_values.max():.2f}")
                            st.write(f"最小值: {day_blue_values.min():.2f}")
                            st.write(f">150的数量: {(day_blue_values > 150).sum()}")
                            st.write(f">180的数量: {(day_blue_values > 180).sum()}")
                    with col2:
                        st.markdown("**周线BLUE数值分布**")
                        week_blue_values = df[df['blue_weekly'] > 0]['blue_weekly']
                        if len(week_blue_values) > 0:
                            st.write(f"平均值: {week_blue_values.mean():.2f}")
                            st.write(f"中位数: {week_blue_values.median():.2f}")
                            st.write(f"最大值: {week_blue_values.max():.2f}")
                            st.write(f"最小值: {week_blue_values.min():.2f}")
                            st.write(f">150的数量: {(week_blue_values > 150).sum()}")
                            st.write(f">180的数量: {(week_blue_values > 180).sum()}")
        
            with stat_tab3:
                if 'market' in df.columns:
                    market_stats = df.groupby('market').agg({
                        'symbol': 'count',
                        'has_day_blue': lambda x: (x == True).sum(),
                        'has_week_blue': lambda x: (x == True).sum(),
                        'price': ['mean', 'median'],
                        'turnover': 'mean'
                    }).round(2)
                    market_stats.columns = ['股票数', '日线BLUE', '周线BLUE', '平均价格', '价格中位数', '平均成交额']
                    st.dataframe(market_stats, use_container_width=True)
            
            st.markdown("### 📋 详细列表")
        
            # 添加搜索功能
            st.markdown("#### 🔍 搜索股票")
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input("输入股票代码或名称", key="search_input", placeholder="例如: AAPL 或 Apple")
            with search_col2:
                search_enabled = st.checkbox("启用搜索", value=False)
        
            # 应用搜索筛选
            if search_enabled and search_query:
                search_query_lower = search_query.lower().strip()
                if search_query_lower:
                    # 搜索股票代码和名称
                    mask = (
                        df['symbol'].astype(str).str.lower().str.contains(search_query_lower, na=False) |
                        df['name'].astype(str).str.lower().str.contains(search_query_lower, na=False)
                    )
                    df = df[mask]
                    if df.empty:
                        st.info(f"未找到包含 '{search_query}' 的股票")
        
            # 格式化显示 - 检查列是否存在
            available_cols = ['symbol', 'name', 'market', 'price', 'turnover', 'signals_summary', 'blue_daily', 'blue_weekly']
            display_cols = [col for col in available_cols if col in df.columns]
        
            if not display_cols:
                st.error("数据列不匹配，请检查数据库结构")
                st.write("可用列:", df.columns.tolist())
            else:
                display_df = df[display_cols].copy()
            
                # 重命名列以便阅读
                col_mapping = {
                    'symbol': '代码',
                    'name': '名称',
                    'market': '市场',
                    'price': '价格',
                    'turnover': '成交额(万)',
                    'signals_summary': '信号汇总',
                    'blue_daily': '日线Blue',
                    'blue_weekly': '周线Blue'
                }
                display_df.columns = [col_mapping.get(col, col) for col in display_df.columns]
            
                # 添加信号筛选
                st.markdown("#### 🔍 信号筛选")
                filter_both_blue = st.checkbox("仅显示日线和周线都有BLUE信号的股票", value=False)
            
                # 应用筛选（在原始df上筛选）
                filtered_df = df.copy()
            
                # 基础信号筛选
                if filter_both_blue:
                    # 只显示日线和周线都有信号的股票
                    if 'has_day_blue' in filtered_df.columns and 'has_week_blue' in filtered_df.columns:
                        # 兼容 0/1 和 True/False
                        filtered_df = filtered_df[
                            (filtered_df['has_day_blue'].astype(bool) == True) & 
                            (filtered_df['has_week_blue'].astype(bool) == True)
                        ]
            
                # 价格范围筛选
                if price_filter_enabled and min_price is not None and max_price is not None:
                    if 'price' in filtered_df.columns:
                        filtered_df = filtered_df[
                            (filtered_df['price'] >= min_price) & 
                            (filtered_df['price'] <= max_price)
                        ]
            
                # 成交额范围筛选
                if turnover_filter_enabled and min_turnover is not None and max_turnover is not None:
                    if 'turnover' in filtered_df.columns:
                        filtered_df = filtered_df[
                            (filtered_df['turnover'] >= min_turnover) & 
                            (filtered_df['turnover'] <= max_turnover)
                        ]
            
                # BLUE数值范围筛选
                if blue_filter_enabled and min_blue is not None and max_blue is not None:
                    if blue_type == "日线BLUE" and 'blue_daily' in filtered_df.columns:
                        filtered_df = filtered_df[
                            (filtered_df['blue_daily'] >= min_blue) & 
                            (filtered_df['blue_daily'] <= max_blue)
                        ]
                    elif blue_type == "周线BLUE" and 'blue_weekly' in filtered_df.columns:
                        filtered_df = filtered_df[
                            (filtered_df['blue_weekly'] >= min_blue) & 
                            (filtered_df['blue_weekly'] <= max_blue)
                        ]
                    elif blue_type == "两者都筛选":
                        if 'blue_daily' in filtered_df.columns and 'blue_weekly' in filtered_df.columns:
                            filtered_df = filtered_df[
                                ((filtered_df['blue_daily'] >= min_blue) & (filtered_df['blue_daily'] <= max_blue)) |
                                ((filtered_df['blue_weekly'] >= min_blue) & (filtered_df['blue_weekly'] <= max_blue))
                            ]
            
                # 信号强度筛选
                if signal_strength_enabled:
                    if day_blue_days is not None and 'blue_days' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['blue_days'] >= day_blue_days]
                    if week_blue_weeks is not None and 'blue_weeks' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['blue_weeks'] >= week_blue_weeks]
            
                # 重新构建display_df
                if not filtered_df.empty:
                    display_df = filtered_df[display_cols].copy()
                    display_df.columns = [col_mapping.get(col, col) for col in display_df.columns]
                
                    st.info(f"📊 筛选结果：共 {len(filtered_df)} 只股票（原始数据 {len(df)} 只）")
                else:
                    st.warning(f"⚠️ 筛选后没有数据！原始数据有 {len(df)} 只。请检查左侧筛选条件（如价格、成交额、BLUE值等）。")
                    # 显示当前的筛选条件状态，帮助用户排查
                    filters_info = []
                    if price_filter_enabled: filters_info.append(f"价格 ({min_price}-{max_price})")
                    if turnover_filter_enabled: filters_info.append(f"成交额 ({min_turnover}-{max_turnover})")
                    if blue_filter_enabled: filters_info.append(f"BLUE值 ({min_blue}-{max_blue})")
                    if signal_strength_enabled: filters_info.append(f"信号强度 (日>{day_blue_days}, 周>{week_blue_weeks})")
                    if search_enabled and search_query: filters_info.append(f"搜索 '{search_query}'")
                    if filter_both_blue: filters_info.append("仅显示双BLUE")
                    if show_favorites_only: filters_info.append("仅显示收藏")
                
                    if filters_info:
                        st.write("当前启用的筛选条件：", ", ".join(filters_info))
                
                    display_df = pd.DataFrame()
            
                # 排序功能
                if not display_df.empty:
                    st.markdown("#### 🔄 排序设置")
                    sort_col1, sort_col2 = st.columns(2)
                    with sort_col1:
                        sort_column = st.selectbox(
                            "排序字段",
                            options=["默认", "价格", "成交额(万)", "日线Blue", "周线Blue"],
                            index=0
                        )
                    with sort_col2:
                        sort_order = st.selectbox(
                            "排序方式",
                            options=["升序", "降序"],
                            index=1  # 默认降序
                        )
                
                    # 应用排序
                    if sort_column != "默认":
                        sort_col_map = {
                            "价格": "价格",
                            "成交额(万)": "成交额(万)",
                            "日线Blue": "日线Blue",
                            "周线Blue": "周线Blue"
                        }
                        actual_sort_col = sort_col_map.get(sort_column)
                        if actual_sort_col in display_df.columns:
                            ascending = (sort_order == "升序")
                            display_df = display_df.sort_values(by=actual_sort_col, ascending=ascending)
            
                # 显示表格
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "价格": st.column_config.NumberColumn(format="%.2f"),
                        "成交额(万)": st.column_config.NumberColumn(format="%.2f"),
                        "日线Blue": st.column_config.NumberColumn(format="%.2f"),
                        "周线Blue": st.column_config.NumberColumn(format="%.2f"),
                    }
                )

                # 显示信号日期详情
                st.markdown("---")
                st.markdown("#### 📅 信号日期详情")
            
                # 统计有日期信息的股票数量
                stocks_with_dates = 0
                for idx, row in filtered_df.iterrows():
                    has_dates = False
                    for col in ['day_blue_dates', 'week_blue_dates', 'heima_dates']:
                        if col in row and pd.notna(row[col]) and row[col]:
                            try:
                                dates = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                                if dates:
                                    has_dates = True
                                    break
                            except:
                                pass
                    if has_dates:
                        stocks_with_dates += 1
            
                if stocks_with_dates > 0:
                    st.info(f"💡 点击下方股票代码可展开查看每次信号出现的具体日期（共 {stocks_with_dates} 只股票有日期信息）")
                else:
                    st.warning("⚠️ 当前数据中没有信号日期信息。请重新运行扫描脚本以获取日期数据。")
            
                # 为每只股票创建可展开的详情
                stocks_displayed = 0
                for idx, row in filtered_df.iterrows():
                    symbol = row.get('symbol', 'N/A')
                    name = row.get('name', symbol)
                    market = row.get('market', 'N/A')
                
                    # 解析信号日期和数值（新格式：列表，每个元素是{"date": "2025-12-30", "value": 150.5}）
                    day_blue_data = []
                    week_blue_data = []
                    heima_dates = []
                
                    if 'day_blue_dates' in row and pd.notna(row['day_blue_dates']):
                        try:
                            data = json.loads(row['day_blue_dates']) if isinstance(row['day_blue_dates'], str) else row['day_blue_dates']
                            if isinstance(data, list):
                                # 新格式：日期-数值对列表
                                if len(data) > 0 and isinstance(data[0], dict) and 'date' in data[0]:
                                    day_blue_data = data
                                # 旧格式：只有日期列表（兼容旧数据）
                                elif len(data) > 0 and isinstance(data[0], str):
                                    day_blue_data = [{"date": d, "value": None} for d in data]
                                else:
                                    day_blue_data = []
                            else:
                                day_blue_data = []
                        except:
                            day_blue_data = []
                
                    if 'week_blue_dates' in row and pd.notna(row['week_blue_dates']):
                        try:
                            data = json.loads(row['week_blue_dates']) if isinstance(row['week_blue_dates'], str) else row['week_blue_dates']
                            if isinstance(data, list):
                                # 新格式：日期-数值对列表
                                if len(data) > 0 and isinstance(data[0], dict) and 'date' in data[0]:
                                    week_blue_data = data
                                # 旧格式：只有日期列表（兼容旧数据）
                                elif len(data) > 0 and isinstance(data[0], str):
                                    week_blue_data = [{"date": d, "value": None} for d in data]
                                else:
                                    week_blue_data = []
                            else:
                                week_blue_data = []
                        except:
                            week_blue_data = []
                
                    if 'heima_dates' in row and pd.notna(row['heima_dates']):
                        try:
                            heima_dates = json.loads(row['heima_dates']) if isinstance(row['heima_dates'], str) else row['heima_dates']
                            if not isinstance(heima_dates, list):
                                heima_dates = []
                        except:
                            heima_dates = []
                
                    # 只显示有信号的股票
                    has_any_signal = (row.get('has_day_blue', False) or 
                                     row.get('has_week_blue', False) or 
                                     row.get('has_heima', False))
                
                    # 如果有信号或者有日期信息，就显示
                    has_date_info = len(day_blue_data) > 0 or len(week_blue_data) > 0 or len(heima_dates) > 0
                
                    if has_any_signal or has_date_info:
                        stocks_displayed += 1
                        # 构建标题
                        title_parts = [f"📊 {symbol}"]
                        if name and name != symbol:
                            title_parts.append(f"({name})")
                        title_parts.append(f"- {market}")
                    
                        # 添加信号标识
                        signal_badges = []
                        if row.get('has_day_blue', False):
                            signal_badges.append("🔵日线")
                        if row.get('has_week_blue', False):
                            signal_badges.append("🔵周线")
                        if row.get('has_heima', False):
                            signal_badges.append("🐴黑马")
                    
                        if signal_badges:
                            title_parts.append(" ".join(signal_badges))
                    
                        title = " ".join(title_parts)
                    
                        # 检查是否收藏
                        is_favorite = symbol in favorite_symbols
                        fav_icon = "⭐" if is_favorite else ""
                        if is_favorite:
                            title = f"{fav_icon} {title}"

                        with st.expander(title, expanded=False):
                            # 添加图表按钮、周期选择和收藏按钮
                            chart_col1, chart_col2, chart_col3, fav_col = st.columns([1.5, 1, 1, 1])
                            with chart_col1:
                                show_chart = st.button(f"📈 查看 {symbol} 图表", key=f"chart_{symbol}_{idx}")
                            with chart_col2:
                                chart_period = st.selectbox(
                                    "信号周期",
                                    options=["daily", "weekly", "monthly"],
                                    format_func=lambda x: {"daily": "日线", "weekly": "周线", "monthly": "月线"}.get(x, x),
                                    index=0,
                                    key=f"period_{symbol}_{idx}",
                                    label_visibility="collapsed"
                                )
                            with chart_col3:
                                show_volume_profile = st.checkbox("筹码分布", value=True, key=f"vp_{symbol}_{idx}")
                        
                            # 如果开启了筹码分布，显示天数选择
                            profile_days = None
                            if show_volume_profile:
                                profile_days = st.slider(
                                    "筹码统计天数 (最近N天)", 
                                    min_value=10, 
                                    max_value=730, 
                                    value=180, 
                                    step=10, 
                                    key=f"vp_days_{symbol}_{idx}",
                                    help="调整筹码分布的统计范围。例如选择30天，则只统计最近30天的成交量分布。"
                                )
                        
                            with fav_col:
                                if is_favorite:
                                    if st.button("❌ 取消收藏", key=f"unfav_{symbol}_{idx}"):
                                        db.remove_favorite(symbol)
                                        st.rerun()
                                else:
                                    if st.button("⭐ 加入自选", key=f"fav_{symbol}_{idx}"):
                                        db.add_favorite(symbol)
                                        st.rerun()
                        
                            if show_chart:
                                with st.spinner(f"正在加载 {symbol} 的历史数据..."):
                                    try:
                                        # 根据周期决定获取多少数据
                                        days_map = {
                                            "daily": 730,  # 改为获取更多数据，以便支持长周期的筹码分布
                                            "weekly": 1095, # 3年
                                            "monthly": 1825 # 5年
                                        }
                                        days = days_map.get(chart_period, 730)
                                    
                                        # 获取历史数据
                                        hist_data = get_stock_data(symbol, market=market, days=days)
                                    
                                        if hist_data is not None and not hist_data.empty:
                                            # 创建图表
                                            fig = create_candlestick_chart(
                                                hist_data,
                                                symbol,
                                                name,
                                                period=chart_period,
                                                day_blue_dates=row.get('day_blue_dates'),
                                                week_blue_dates=row.get('week_blue_dates'),
                                                heima_dates=row.get('heima_dates'),
                                                show_volume_profile=show_volume_profile,
                                                profile_days=profile_days
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                            # 根据周期转换数据用于统计
                                            if chart_period == 'weekly':
                                                stat_df = hist_data.resample('W-MON').agg({
                                                    'Open': 'first',
                                                    'High': 'max',
                                                    'Low': 'min',
                                                    'Close': 'last'
                                                }).dropna()
                                                period_name = "周"
                                            elif chart_period == 'monthly':
                                                stat_df = hist_data.resample('ME').agg({
                                                    'Open': 'first',
                                                    'High': 'max',
                                                    'Low': 'min',
                                                    'Close': 'last'
                                                }).dropna()
                                                period_name = "月"
                                            else:
                                                stat_df = hist_data
                                                period_name = "天"
                                        
                                            # 显示数据统计
                                            st.markdown("#### 📊 数据统计")
                                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                            with stat_col1:
                                                st.metric(f"数据{period_name}数", len(stat_df))
                                            with stat_col2:
                                                st.metric("最新价格", f"{stat_df['Close'].iloc[-1]:.2f}")
                                            with stat_col3:
                                                price_change = ((stat_df['Close'].iloc[-1] - stat_df['Close'].iloc[0]) / stat_df['Close'].iloc[0]) * 100
                                                st.metric("期间涨跌幅", f"{price_change:.2f}%")
                                            with stat_col4:
                                                st.metric("最高价", f"{stat_df['High'].max():.2f}")
                                        else:
                                            st.warning(f"无法获取 {symbol} 的历史数据，请检查API配置或网络连接")
                                    except Exception as e:
                                        st.error(f"加载图表时出错: {str(e)}")
                        
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                        
                            with col1:
                                st.markdown("**🔵 日线BLUE信号**")
                                if day_blue_data:
                                    st.success(f"共 {len(day_blue_data)} 次")
                                    st.markdown("<div style='max-height: 200px; overflow-y: auto;'>", unsafe_allow_html=True)
                                    # 按日期排序
                                    sorted_data = sorted(day_blue_data, key=lambda x: x.get('date', ''))
                                    for item in sorted_data:
                                        date = item.get('date', 'N/A')
                                        value = item.get('value')
                                        if value is not None:
                                            st.markdown(f"  • `{date}`: **{value:.2f}**")
                                        else:
                                            st.markdown(f"  • `{date}`")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    if row.get('has_day_blue', False):
                                        st.info("有信号但无日期记录")
                                    else:
                                        st.write("无")
                        
                            with col2:
                                st.markdown("**🔵 周线BLUE信号**")
                                if week_blue_data:
                                    st.success(f"共 {len(week_blue_data)} 次")
                                    st.markdown("<div style='max-height: 200px; overflow-y: auto;'>", unsafe_allow_html=True)
                                    # 按日期排序
                                    sorted_data = sorted(week_blue_data, key=lambda x: x.get('date', ''))
                                    for item in sorted_data:
                                        date = item.get('date', 'N/A')
                                        value = item.get('value')
                                        if value is not None:
                                            st.markdown(f"  • `{date}`: **{value:.2f}**")
                                        else:
                                            st.markdown(f"  • `{date}`")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    if row.get('has_week_blue', False):
                                        st.info("有信号但无日期记录")
                                    else:
                                        st.write("无")
                        
                            with col3:
                                st.markdown("**🐴 黑马信号**")
                                if heima_dates:
                                    st.success(f"共 {len(heima_dates)} 次")
                                    st.markdown("<div style='max-height: 200px; overflow-y: auto;'>", unsafe_allow_html=True)
                                    for date in sorted(heima_dates):
                                        st.markdown(f"  • `{date}`")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    if row.get('has_heima', False):
                                        st.info("有信号但无日期记录")
                                    else:
                                        st.write("无")
            
                if stocks_displayed == 0:
                    st.info("当前筛选条件下没有显示信号日期的股票")
            
                # 添加下载按钮
                csv = display_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载CSV",
                    data=csv,
                    file_name=f"stock_signals_{selected_date}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"{selected_date} 当天没有数据")
    
        # 历史对比功能
        if compare_enabled and compare_date1 and compare_date2 and compare_date1 != compare_date2:
            st.markdown("---")
            st.markdown("### 📊 历史数据对比")
            st.info(f"对比日期: {compare_date1} vs {compare_date2}")
        
            try:
                comparison = db.compare_dates(compare_date1, compare_date2, market=market_filter[0] if len(market_filter) == 1 else None)
            
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("新出现股票", len(comparison['new']))
                with col2:
                    st.metric("消失股票", len(comparison['disappeared']))
                with col3:
                    st.metric("持续股票", len(comparison['persistent']))
            
                # 显示新出现的股票
                if comparison['new']:
                    with st.expander(f"🆕 新出现的股票 ({len(comparison['new'])}只)", expanded=False):
                        new_df = comparison['df2'][comparison['df2']['symbol'].isin(comparison['new'])]
                        if not new_df.empty:
                            display_cols = ['symbol', 'name', 'market', 'price', 'blue_daily', 'blue_weekly', 'signals_summary']
                            display_cols = [col for col in display_cols if col in new_df.columns]
                            new_display = new_df[display_cols].copy()
                            col_mapping = {
                                'symbol': '代码',
                                'name': '名称',
                                'market': '市场',
                                'price': '价格',
                                'blue_daily': '日线Blue',
                                'blue_weekly': '周线Blue',
                                'signals_summary': '信号汇总'
                            }
                            new_display.columns = [col_mapping.get(col, col) for col in new_display.columns]
                            st.dataframe(new_display, use_container_width=True)
            
                # 显示消失的股票
                if comparison['disappeared']:
                    with st.expander(f"❌ 消失的股票 ({len(comparison['disappeared'])}只)", expanded=False):
                        disappeared_df = comparison['df1'][comparison['df1']['symbol'].isin(comparison['disappeared'])]
                        if not disappeared_df.empty:
                            display_cols = ['symbol', 'name', 'market', 'price', 'blue_daily', 'blue_weekly', 'signals_summary']
                            display_cols = [col for col in display_cols if col in disappeared_df.columns]
                            disappeared_display = disappeared_df[display_cols].copy()
                            col_mapping = {
                                'symbol': '代码',
                                'name': '名称',
                                'market': '市场',
                                'price': '价格',
                                'blue_daily': '日线Blue',
                                'blue_weekly': '周线Blue',
                                'signals_summary': '信号汇总'
                            }
                            disappeared_display.columns = [col_mapping.get(col, col) for col in disappeared_display.columns]
                            st.dataframe(disappeared_display, use_container_width=True)
            
                # 显示持续股票的信号变化
                if comparison['persistent']:
                    with st.expander(f"🔄 持续股票信号变化 ({len(comparison['persistent'])}只)", expanded=False):
                        persistent_symbols = list(comparison['persistent'])[:50]  # 显示前50只
                    
                        changes = []
                        for symbol in persistent_symbols:
                            stock1_df = comparison['df1'][comparison['df1']['symbol'] == symbol]
                            stock2_df = comparison['df2'][comparison['df2']['symbol'] == symbol]
                        
                            if len(stock1_df) > 0 and len(stock2_df) > 0:
                                stock1 = stock1_df.iloc[0]
                                stock2 = stock2_df.iloc[0]
                            
                                price_change = stock2.get('price', 0) - stock1.get('price', 0)
                                price_change_pct = (price_change / stock1.get('price', 1)) * 100 if stock1.get('price', 0) > 0 else 0
                            
                                change_info = {
                                    'symbol': symbol,
                                    'name': stock2.get('name', symbol),
                                    'market': stock2.get('market', ''),
                                    '价格变化': f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                                    '日线BLUE变化': '是' if (stock2.get('has_day_blue', False) != stock1.get('has_day_blue', False)) else '否',
                                    '周线BLUE变化': '是' if (stock2.get('has_week_blue', False) != stock1.get('has_week_blue', False)) else '否',
                                    f'{compare_date1}价格': f"{stock1.get('price', 0):.2f}",
                                    f'{compare_date2}价格': f"{stock2.get('price', 0):.2f}"
                                }
                                changes.append(change_info)
                    
                        if changes:
                            changes_df = pd.DataFrame(changes)
                            st.dataframe(changes_df, use_container_width=True)
            except Exception as e:
                st.error(f"对比功能出错: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("请先运行扫描脚本生成数据。")
    
        st.markdown("""
        ### 如何开始:
        1. 运行 `python scan_cn_signals_blue_only.py` 扫描A股
        2. 运行 `python scan_us_signals.py` 扫描美股
        3. 刷新此页面查看结果
        """)
    
        # 显示数据库状态
        st.markdown("---")
        st.markdown("### 📊 数据库状态")
        try:
            dates = db.get_available_dates()
            if dates:
                st.success(f"数据库中有 {len(dates)} 个扫描日期的数据")
                st.write("最近扫描日期:", dates[:5])
            else:
                st.warning("数据库中暂无数据")
        except Exception as e:
            st.error(f"获取数据库状态失败: {e}")


# --------------------------
# 页面 2: 自选看板 (新功能)
# --------------------------
elif page == "❤️ 自选看板":
    st.header("❤️ 自选股行情看板")
    
    # 获取所有自选股
    favorites_df = db.get_all_favorites()
    
    if favorites_df.empty:
        st.info("您还没有添加任何自选股。请在“信号扫描”页面中点击 ⭐ 添加。")
    else:
        # 获取最新扫描日期以关联信号信息
        available_dates = db.get_available_dates()
        latest_date = available_dates[0] if available_dates else None
        
        # 准备显示数据
        # 我们需要从 database 获取自选股的基本信息 (symbol, note)
        # 并尝试从最新的扫描结果中获取补充信息 (name, market, 信号状态)
        
        display_data = favorites_df.copy()
        
        if latest_date:
            st.caption(f"信号状态基于最近扫描日期: {latest_date}")
            latest_scan_df = db.get_results_by_date(latest_date)
            # 合并扫描结果中的信息
            display_data = pd.merge(display_data, latest_scan_df, on='symbol', how='left', suffixes=('', '_scan'))
            # 合并后 name 列可能冲突，优先使用扫描结果的 name
            if 'name_scan' in display_data.columns:
                display_data['name'] = display_data['name_scan'].fillna(display_data['name'])
        else:
            st.warning("暂无扫描数据，无法显示信号状态。")
            display_data['market'] = '未知'
            display_data['price'] = 0
        
        # 统计信息
        st.markdown("### 📊 自选概览")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("自选股总数", len(display_data))
        with col2:
            if 'has_day_blue' in display_data.columns:
                has_signal = display_data[
                    (display_data['has_day_blue'] == 1) | 
                    (display_data['has_week_blue'] == 1) | 
                    (display_data['has_heima'] == 1)
                ]
                st.metric("今日有信号", len(has_signal))
            else:
                st.metric("今日有信号", 0)
        
        st.markdown("---")
        
        # 遍历显示每个自选股
        for idx, row in display_data.iterrows():
            symbol = row['symbol']
            name = row.get('name', symbol)
            if pd.isna(name): name = symbol
            
            # 尝试推断市场（如果缺失）
            market = row.get('market')
            if pd.isna(market) or market == '未知':
                if str(symbol)[0].isdigit():
                    market = 'CN'
                else:
                    market = 'US'
            
            # 价格信息
            price = row.get('price', 0)
            
            # 构建标题
            title = f"⭐ {symbol} ({name})"
            if price > 0:
                title += f" | 价格: {price:.2f}"
            
            # 信号标记
            signal_badges = []
            if row.get('has_day_blue') == 1: signal_badges.append("🔵日线")
            if row.get('has_week_blue') == 1: signal_badges.append("🔵周线")
            if row.get('has_heima') == 1: signal_badges.append("🐴黑马")
            
            if signal_badges:
                title += " " + " ".join(signal_badges)
            
            # 展开显示详情
            with st.expander(title, expanded=False):
                # 操作栏
                col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
                
                with col1:
                    show_chart = st.button(f"📈 查看图表", key=f"fav_chart_{symbol}_{idx}")
                
                with col2:
                    chart_period = st.selectbox(
                        "周期",
                        options=["daily", "weekly", "monthly"],
                        format_func=lambda x: {"daily": "日线", "weekly": "周线", "monthly": "月线"}.get(x, x),
                        index=0,
                        key=f"fav_period_{symbol}_{idx}",
                        label_visibility="collapsed"
                    )
                
                with col3:
                    show_volume_profile = st.checkbox("筹码分布", value=True, key=f"fav_vp_{symbol}_{idx}")
                
                # 筹码天数
                profile_days = None
                if show_volume_profile:
                    profile_days = st.slider(
                        "统计天数", 
                        min_value=10, 
                        max_value=730, 
                        value=180, 
                        step=10, 
                        key=f"fav_vp_days_{symbol}_{idx}"
                    )

                with col4:
                    if st.button("❌ 移除", key=f"fav_remove_{symbol}_{idx}"):
                        db.remove_favorite(symbol)
                        st.rerun()
                
                # 备注信息
                note = row.get('note')
                if note:
                    st.info(f"📝 备注: {note}")
                
                # 显示图表逻辑
                if show_chart:
                    with st.spinner(f"正在加载 {symbol} 的历史数据..."):
                        try:
                            # 获取数据
                            days_map = {"daily": 730, "weekly": 1095, "monthly": 1825}
                            days = days_map.get(chart_period, 730)
                            
                            hist_data = get_stock_data(symbol, market=market, days=days)
                            
                            if hist_data is not None and not hist_data.empty:
                                # 获取信号日期（如果有）
                                day_blue_dates = row.get('day_blue_dates') if pd.notna(row.get('day_blue_dates')) else None
                                week_blue_dates = row.get('week_blue_dates') if pd.notna(row.get('week_blue_dates')) else None
                                heima_dates = row.get('heima_dates') if pd.notna(row.get('heima_dates')) else None
                                
                                # 绘制图表
                                fig = create_candlestick_chart(
                                    hist_data,
                                    symbol,
                                    name,
                                    period=chart_period,
                                    day_blue_dates=day_blue_dates,
                                    week_blue_dates=week_blue_dates,
                                    heima_dates=heima_dates,
                                    show_volume_profile=show_volume_profile,
                                    profile_days=profile_days
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示最新价格统计
                                latest_close = hist_data['Close'].iloc[-1]
                                prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else latest_close
                                change_pct = (latest_close - prev_close) / prev_close * 100
                                
                                st.metric("最新收盘价", f"{latest_close:.2f}", f"{change_pct:.2f}%")
                                
                            else:
                                st.warning(f"无法获取 {symbol} 的历史数据")
                        except Exception as e:
                            st.error(f"加载图表失败: {e}")

# --------------------------
# 页面 3: 策略回测 (新功能)
# --------------------------
elif page == "🔄 策略回测":
    st.header("🔄 BLUE 策略回测 (v1.0)")
    st.info("策略逻辑: 日线 BLUE > 阈值 (买入) -> KDJ J > 100 (卖出)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbol_input = st.text_input("股票代码", value="NVDA", help="例如: NVDA, AAPL, 600519.SH")
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
    
    # --- 智能推荐模块 ---
    if st.button("🔍 分析波动率 & 推荐阈值"):
        with st.spinner(f"正在分析 {symbol} 的历史波动率..."):
            try:
                # 获取1年数据用于分析
                df_vol = get_stock_data(symbol, market, days=365)
                if df_vol is not None and not df_vol.empty:
                    # 计算日收益率
                    df_vol['returns'] = df_vol['Close'].pct_change()
                    # 计算年化波动率
                    volatility = df_vol['returns'].std() * np.sqrt(252)
                    
                    # 自适应逻辑
                    rec_threshold = 90 # 默认
                    stock_type = "中等波动 (正常)"
                    
                    if volatility > 0.45:
                        rec_threshold = 110
                        stock_type = "🔥 高波动 (成长/妖股)"
                    elif volatility < 0.20:
                        rec_threshold = 70
                        stock_type = "🛡️ 低波动 (防守/价值)"
                    elif volatility < 0.30:
                        rec_threshold = 80
                        stock_type = "⚖️ 中低波动 (稳健)"
                        
                    st.info(f"""
                    **分析结果**:
                    - 年化波动率: `{volatility:.2%}`
                    - 股票类型: **{stock_type}**
                    - 💡 **推荐 BLUE 阈值**: `{rec_threshold}` (请手动调整上方滑块)
                    """)
                else:
                    st.error("无法获取数据进行分析")
            except Exception as e:
                st.error(f"分析失败: {e}")
        
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
                    require_vp_filter=require_vp
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
                    
                    st.success("✅ 回测完成！")
                    
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
                        st.subheader("📋 交易记录 & 筹码分布")
                        
                        trade_data = []
                        for t in backtester.trades:
                            vp = t.get('vp_metrics', {})
                            trade_data.append({
                                "日期": t['date'].strftime('%Y-%m-%d'),
                                "类型": t['type'],
                                "价格": f"{t['price']:.2f}",
                                "数量": t['shares'],
                                "金额": f"{t['value']:.2f}",
                                "盈亏": f"{t.get('pnl', 0):.2f}" if 'pnl' in t else "-",
                                "筹码获利比": f"{vp.get('profit_ratio', 0):.2%}" if vp else "-",
                                "相对POC": vp.get('price_pos', '-') if vp else "-",
                                "筹码集中度": f"{vp.get('concentration', 0):.2f}" if vp else "-"
                            })
                        
                        st.dataframe(pd.DataFrame(trade_data), width="stretch")
                    else:
                        st.warning("在此期间未触发任何交易。")

                    # 被过滤的信号表 (New Feature)
                    if hasattr(backtester, 'rejected_trades') and backtester.rejected_trades:
                        with st.expander("🚫 查看被过滤的信号 (诊断报告)", expanded=True):
                            st.caption("以下信号满足了基础 BLUE 阈值，但被您的高级过滤条件（周线/黑马/筹码分布）拒绝。")
                            
                            rejected_data = []
                            for r in backtester.rejected_trades:
                                rejected_data.append({
                                    "日期": r['date'].strftime('%Y-%m-%d'),
                                    "价格": f"{r['price']:.2f}",
                                    "Day BLUE": f"{r['blue']:.1f}",
                                    "拒绝原因 ❌": r['reason']
                                })
                            
                            st.dataframe(pd.DataFrame(rejected_data), width="stretch")
                        
            except Exception as e:
                st.error(f"回测出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
