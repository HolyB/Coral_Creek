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

from chart_utils import create_candlestick_chart, create_candlestick_chart_dynamic, analyze_chip_flow, create_chip_flow_chart, create_chip_change_chart, quick_chip_analysis
from data_fetcher import get_us_stock_data as fetch_data_from_polygon, get_ticker_details
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_adx_series
from backtester import SimpleBacktester
from db.database import (
    query_scan_results, get_scanned_dates, get_db_stats, 
    get_stock_history, init_db, get_scan_job, get_stock_info_batch,
    get_first_scan_dates
)

# 设置页面配置
st.set_page_config(
    page_title="Coral Creek V2.1 - 智能量化系统",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 加载自定义 CSS ---
def load_custom_css():
    """加载自定义 CSS 样式"""
    css_path = os.path.join(current_dir, "static", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# 应用自定义样式
load_custom_css()


# --- 登录验证 ---

def check_password():
    """角色验证 - Admin 可管理持仓，Guest 只能查看"""
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None
    
    if st.session_state["user_role"] is None:
        st.markdown("## 🦅 Coral Creek V2.0")
        st.markdown("智能量化扫描系统")
        st.markdown("---")
        
        password = st.text_input("密码", type="password", key="password_input")
        
        if st.button("登录", type="primary"):
            # 获取密码配置
            try:
                admin_password = st.secrets.get("admin_password", "admin2026")
                guest_password = st.secrets.get("guest_password", "coral2026")
            except:
                admin_password = "admin2026"
                guest_password = "coral2026"
            
            if password == admin_password:
                st.session_state["user_role"] = "admin"
                st.success("✅ 欢迎，管理员！")
                st.rerun()
            elif password == guest_password:
                st.session_state["user_role"] = "guest"
                st.success("✅ 欢迎访客！")
                st.rerun()
            elif password:
                st.error("❌ 密码错误")
        
        st.markdown("---")
        st.caption("Admin: 完整功能 | Guest: 只读模式")
        st.stop()

def is_admin():
    """检查当前用户是否为管理员"""
    return st.session_state.get("user_role") == "admin"

check_password()

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
        
        # 从 stock_info 缓存补充缺失的名称和行业
        symbols_need_info = df[df['Name'].isna() | (df['Name'] == '')]['Ticker'].tolist()
        if symbols_need_info:
            stock_info_cache = get_stock_info_batch(symbols_need_info)
            for idx, row in df.iterrows():
                ticker = row['Ticker']
                if ticker in stock_info_cache and (pd.isna(row.get('Name')) or row.get('Name') == ''):
                    info = stock_info_cache[ticker]
                    df.at[idx, 'Name'] = info.get('name', '')
                    if pd.isna(row.get('Industry')) or row.get('Industry') == '':
                        df.at[idx, 'Industry'] = info.get('industry', '')
        
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


def render_market_pulse(market='US'):
    """
    Market Pulse Dashboard - 显示大盘指数状态
    US: SPY/QQQ/DIA/IWM + VIX
    CN: 上证/深证/创业板/沪深300
    """
    from data_fetcher import get_cn_index_data
    
    # 缓存键 (每10分钟刷新, 按市场区分)
    from datetime import datetime
    cache_time_key = datetime.now().strftime("%Y%m%d%H") + str(datetime.now().minute // 10)
    cache_key = f"market_pulse_{market}_{cache_time_key}"
    
    # 检查缓存
    if cache_key not in st.session_state:
        # 根据市场选择指数
        if market == 'CN':
            indices = {
                '000001.SH': {'name': '上证指数', 'emoji': '🔴'},
                '399001.SZ': {'name': '深证成指', 'emoji': '🟢'},
                '399006.SZ': {'name': '创业板指', 'emoji': '💡'},
                '000300.SH': {'name': '沪深300', 'emoji': '📊'},
            }
            data_fetcher = get_cn_index_data
            currency = '¥'
        else:
            indices = {
                'SPY': {'name': 'S&P 500', 'emoji': '📊'},
                'QQQ': {'name': 'Nasdaq 100', 'emoji': '💻'},
                'DIA': {'name': 'Dow 30', 'emoji': '🏭'},
                'IWM': {'name': 'Russell 2000', 'emoji': '🏢'},
            }
            data_fetcher = fetch_data_from_polygon
            currency = '$'
        
        index_data = {}
        index_data['_currency'] = currency
        index_data['_market'] = market
        
        for symbol, info in indices.items():
            try:
                # 获取日线数据
                df_daily = data_fetcher(symbol, days=100)
                
                if df_daily is not None and len(df_daily) >= 30:
                    # 计算日线 BLUE
                    blue_daily = calculate_blue_signal_series(
                        df_daily['Open'].values,
                        df_daily['High'].values,
                        df_daily['Low'].values,
                        df_daily['Close'].values
                    )
                    
                    # 计算周线 BLUE
                    df_weekly = df_daily.resample('W-MON').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    
                    blue_weekly = [0]
                    if len(df_weekly) >= 10:
                        blue_weekly = calculate_blue_signal_series(
                            df_weekly['Open'].values,
                            df_weekly['High'].values,
                            df_weekly['Low'].values,
                            df_weekly['Close'].values
                        )
                    
                    # 计算筹码形态
                    chip_result = quick_chip_analysis(df_daily)
                    chip_pattern = chip_result.get('label', '') if chip_result else ''
                    
                    # 最新价格和变化
                    latest_price = df_daily['Close'].iloc[-1]
                    prev_price = df_daily['Close'].iloc[-2] if len(df_daily) > 1 else latest_price
                    price_change = (latest_price - prev_price) / prev_price * 100
                    
                    index_data[symbol] = {
                        'name': info['name'],
                        'emoji': info['emoji'],
                        'price': latest_price,
                        'change': price_change,
                        'day_blue': blue_daily[-1] if len(blue_daily) > 0 else 0,
                        'week_blue': blue_weekly[-1] if len(blue_weekly) > 0 else 0,
                        'chip': chip_pattern
                    }
            except Exception as e:
                index_data[symbol] = {
                    'name': info['name'],
                    'emoji': info['emoji'],
                    'price': 0,
                    'change': 0,
                    'day_blue': 0,
                    'week_blue': 0,
                    'chip': '',
                    'error': str(e)
                }
        
        # VIX 数据 (仅美股, 使用 VIXY ETF 因为 VIX 直接指数无法获取)
        if market == 'US':
            try:
                vix_df = fetch_data_from_polygon('VIXY', days=30)
                if vix_df is not None and len(vix_df) > 0:
                    vix_price = vix_df['Close'].iloc[-1]
                    vix_prev = vix_df['Close'].iloc[-2] if len(vix_df) > 1 else vix_price
                    vix_change = vix_price - vix_prev
                    
                    # VIXY 的阈值需要调整 (ETF 价格不同于 VIX 指数)
                    if vix_price < 20:
                        vix_mood = "😌 极度贪婪"
                    elif vix_price < 25:
                        vix_mood = "🙂 平静"
                    elif vix_price < 30:
                        vix_mood = "😐 中性"
                    elif vix_price < 40:
                        vix_mood = "😟 焦虑"
                    else:
                        vix_mood = "😱 恐惧"
                        
                    index_data['VIX'] = {
                        'price': vix_price,
                        'change': vix_change,
                        'mood': vix_mood
                    }
                else:
                    index_data['VIX'] = {'price': 0, 'change': 0, 'mood': '数据不可用'}
            except:
                index_data['VIX'] = {'price': 0, 'change': 0, 'mood': '未知'}
        
        # 商品/加密资产数据 (仅美股: Gold, Silver, BTC)
        if market == 'US':
            alt_assets = {
                'GLD': {'name': '黄金', 'emoji': '🥇', 'format': '${:.2f}'},
                'SLV': {'name': '白银', 'emoji': '🥈', 'format': '${:.2f}'},
                'X:BTCUSD': {'name': 'BTC', 'emoji': '₿', 'format': '${:,.0f}'}
            }
            
            for symbol, info in alt_assets.items():
                try:
                    df = fetch_data_from_polygon(symbol, days=30)
                    if df is not None and len(df) > 0:
                        price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2] if len(df) > 1 else price
                        change = (price - prev_price) / prev_price * 100
                        
                        index_data[symbol] = {
                            'name': info['name'],
                            'emoji': info['emoji'],
                            'price': price,
                            'change': change,
                            'format': info['format']
                        }
                except:
                    index_data[symbol] = {
                        'name': info['name'],
                        'emoji': info['emoji'],
                        'price': 0,
                        'change': 0,
                        'format': info['format']
                    }
        
        # 计算市场情绪综合评分
        # 过滤掉私有键和VIX，只看主要指数
        main_indices = [k for k in index_data.keys() if not k.startswith('_') and k not in ['VIX', 'GLD', 'SLV', 'X:BTCUSD']]
        bullish_count = sum(1 for k in main_indices if index_data.get(k, {}).get('day_blue', 0) > 100)
        total_indices = len(main_indices)
        
        vix_ok = index_data.get('VIX', {}).get('price', 20) < 25 if market == 'US' else True
        
        if bullish_count >= 3 and vix_ok:
            market_sentiment = ("🟢 强势做多", "进攻型 60-80%", "#3fb950")
        elif bullish_count >= 2:
            market_sentiment = ("🟡 震荡偏多", "平衡型 40-60%", "#d29922")
        elif bullish_count >= 1:
            market_sentiment = ("🟠 分化观望", "防守型 20-40%", "#f85149")
        else:
            market_sentiment = ("🔴 弱势防守", "空仓或对冲", "#f85149")
        
        index_data['_sentiment'] = market_sentiment
        index_data['_bullish_count'] = bullish_count
        
        st.session_state[cache_key] = index_data
    else:
        index_data = st.session_state[cache_key]
    
    # === UI 渲染 ===
    with st.container():
        market = index_data.get('_market', 'US')
        currency = index_data.get('_currency', '$')
        
        market_title = "🇺🇸 US Market Pulse" if market == 'US' else "🇨🇳 A股大盘"
        st.markdown(f"### {market_title}")
        
        # 根据市场动态选择要显示的指数
        if market == 'CN':
            display_symbols = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH']
            col_count = 4
        else:
            display_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VIX']
            col_count = 5
        
        cols = st.columns(col_count)
        
        for i, (symbol, col) in enumerate(zip(display_symbols, cols)):
            with col:
                data = index_data.get(symbol, {})
                
                if symbol == 'VIX':
                    # VIX 特殊显示
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    mood = data.get('mood', '')
                    
                    delta_color = "inverse" if change < 0 else "normal"
                    st.metric(
                        label="VIX 恐惧指数",
                        value=f"{price:.1f}",
                        delta=f"{change:+.1f}",
                        delta_color=delta_color
                    )
                    st.caption(mood)
                else:
                    # 常规指数
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    day_blue = data.get('day_blue', 0)
                    week_blue = data.get('week_blue', 0)
                    chip = data.get('chip', '')
                    name = data.get('name', symbol)
                    emoji = data.get('emoji', '')
                    
                    # 趋势图标
                    if change > 0.5:
                        trend = "📈"
                    elif change < -0.5:
                        trend = "📉"
                    else:
                        trend = "➡️"
                    
                    # 显示标签：A股显示名称，美股显示代码
                    if market == 'CN':
                        display_label = f"{emoji} {name} {trend}"
                    else:
                        display_label = f"{symbol} {trend}"
                    
                    st.metric(
                        label=display_label,
                        value=f"{currency}{price:.2f}",
                        delta=f"{change:+.2f}%"
                    )
                    
                    # BLUE 信号 + 筹码
                    blue_text = f"D:{day_blue:.0f} W:{week_blue:.0f}"
                    if chip:
                        blue_text += f" {chip}"
                    
                    # 颜色编码
                    if day_blue > 100:
                        st.markdown(f"<span style='color:#3fb950;font-size:0.85rem;'>{blue_text}</span>", unsafe_allow_html=True)
                    elif day_blue > 50:
                        st.markdown(f"<span style='color:#d29922;font-size:0.85rem;'>{blue_text}</span>", unsafe_allow_html=True)
                    else:
                        st.caption(blue_text)
        
        # === 第二行: 商品/加密资产 (仅美股) ===
        if market == 'US':
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            alt_cols = st.columns(4)
            
            for i, (symbol, col) in enumerate(zip(['GLD', 'SLV', 'X:BTCUSD'], alt_cols[:3])):
                with col:
                    data = index_data.get(symbol, {})
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    name = data.get('name', symbol)
                    emoji = data.get('emoji', '')
                    fmt = data.get('format', '${:.2f}')
                    
                    # 趋势图标
                    if change > 0.5:
                        trend = "📈"
                    elif change < -0.5:
                        trend = "📉"
                    else:
                        trend = "➡️"
                    
                    # 格式化价格
                    try:
                        formatted_price = fmt.format(price)
                    except:
                        formatted_price = f"${price:.2f}"
                    
                    st.metric(
                        label=f"{emoji} {name} {trend}",
                        value=formatted_price,
                        delta=f"{change:+.2f}%"
                    )
        
        # 市场情绪总结
        sentiment = index_data.get('_sentiment', ('未知', '未知', 'gray'))
        bullish = index_data.get('_bullish_count', 0)
        
        st.markdown(f"""
        <div style="background: rgba(22, 27, 34, 0.8); border-radius: 8px; padding: 12px 16px; margin-top: 10px; border-left: 4px solid {sentiment[2]};">
            <span style="font-size: 1.1rem; font-weight: 600;">{sentiment[0]}</span>
            <span style="color: #8b949e; margin-left: 12px;">建议仓位: {sentiment[1]}</span>
            <span style="color: #8b949e; margin-left: 12px;">({bullish}/4 指数有日BLUE信号)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # === 指数详情展开 ===
        with st.expander("🔍 查看指数/资产详情 (筹码分布 & 资金流向)", expanded=False):
            # 可选指数列表
            all_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV', 'X:BTCUSD']
            symbol_labels = {
                'SPY': '📊 SPY (S&P 500)',
                'QQQ': '💻 QQQ (Nasdaq 100)',
                'DIA': '🏭 DIA (Dow 30)',
                'IWM': '🏢 IWM (Russell 2000)',
                'GLD': '🥇 GLD (黄金)',
                'SLV': '🥈 SLV (白银)',
                'X:BTCUSD': '₿ BTC (比特币)'
            }
            
            selected_index = st.selectbox(
                "选择要分析的指数/资产",
                options=all_symbols,
                format_func=lambda x: symbol_labels.get(x, x),
                key="market_pulse_index_detail"
            )
            
            if selected_index:
                with st.spinner(f"正在加载 {selected_index} 数据..."):
                    try:
                        # 获取数据
                        df_detail = fetch_data_from_polygon(selected_index, days=120)
                        
                        if df_detail is not None and len(df_detail) >= 30:
                            detail_cols = st.columns([2, 1])
                            
                            with detail_cols[0]:
                                # K线图 + BLUE 信号
                                st.markdown("##### 📈 K线图 & BLUE信号")
                                fig = create_candlestick_chart_dynamic(
                                    df_full=df_detail,
                                    df_for_vp=df_detail,
                                    symbol=selected_index,
                                    name=symbol_labels.get(selected_index, selected_index),
                                    period='daily',
                                    show_volume_profile=True
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("无法生成图表")
                            
                            with detail_cols[1]:
                                # 筹码分析摘要
                                st.markdown("##### 📊 筹码分析")
                                chip_result = quick_chip_analysis(df_detail)
                                
                                if chip_result:
                                    poc_pos = chip_result.get('poc_position', 50)
                                    bottom_ratio = chip_result.get('bottom_chip_ratio', 0) * 100
                                    max_chip = chip_result.get('max_chip_pct', 0)
                                    is_strong = chip_result.get('is_strong_bottom_peak', False)
                                    is_peak = chip_result.get('is_bottom_peak', False)
                                    
                                    # 显示指标
                                    st.metric("POC 位置", f"{poc_pos:.1f}%", help="成本峰值在价格区间的位置")
                                    st.metric("底部筹码", f"{bottom_ratio:.1f}%", help="底部30%价格区间的筹码占比")
                                    st.metric("单峰最大", f"{max_chip:.1f}%", help="最大筹码柱占比")
                                    
                                    if is_strong:
                                        st.success("🔥 强势顶格峰")
                                    elif is_peak:
                                        st.info("📍 底部密集")
                                    else:
                                        st.caption("普通形态")
                                else:
                                    st.warning("无法计算筹码分布")
                            
                            # 资金流向图表 (使用筹码流动对比)
                            st.markdown("##### 💰 筹码流动对比 (30天前 vs 现在)")
                            chip_flow_data = analyze_chip_flow(df_detail, lookback_days=30)
                            if chip_flow_data:
                                flow_fig = create_chip_flow_chart(chip_flow_data, selected_index)
                                if flow_fig:
                                    st.plotly_chart(flow_fig, use_container_width=True)
                            else:
                                st.info("数据不足，无法显示筹码流动")
                                
                        else:
                            st.warning(f"无法获取 {selected_index} 数据")
                            
                    except Exception as e:
                        st.error(f"加载失败: {e}")
        
        st.divider()


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
    st.header("🦅 每日机会扫描 (Opportunity Scanner)")
    
    # 侧边栏：数据源选择 (必须先执行，才能获得 market 值)
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
    
    # === Market Pulse Dashboard (顶部) - 传入选中的市场 ===
    render_market_pulse(market=selected_market)
    
    # 侧边栏：继续其他设置
    with st.sidebar:
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
            
            # 快捷按钮
            st.caption("快捷筛选:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("≥1M", key="t1m", help="成交额≥100万"):
                    st.session_state['turnover_filter'] = 1.0
            with col2:
                if st.button("≥5M", key="t5m", help="成交额≥500万"):
                    st.session_state['turnover_filter'] = 5.0
            with col3:
                if st.button("≥10M", key="t10m", help="成交额≥1000万"):
                    st.session_state['turnover_filter'] = 10.0
            
            # 获取筛选值
            default_val = st.session_state.get('turnover_filter', 0.5)
            
            min_turnover_val = st.slider(
                "最低日成交额 (百万)", 
                min_value=0.0, 
                max_value=min(max_turnover, 100.0),
                value=min(default_val, max_turnover),
                step=0.5,
                help="过滤成交额过低的股票，避免流动性风险。1M=100万"
            )
            st.session_state['turnover_filter'] = min_turnover_val
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
            
            # 筹码形态筛选 (需要先计算筹码)
            st.caption("💡 勾选下方「计算筹码形态」后可使用筹码筛选")
            chip_filter = st.selectbox(
                "🔥 筹码形态筛选",
                options=["全部", "仅强势顶格峰 🔥", "仅底部密集 📍", "有底部信号 (🔥+📍)"],
                index=0,
                help="需要先启用筹码形态计算"
            )
            # 存储到 session_state 供后续使用
            st.session_state['chip_filter'] = chip_filter
        
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
    
    # 底部顶格峰计算选项
    col_opt1, col_opt2 = st.columns([1, 3])
    with col_opt1:
        calc_chip = st.checkbox("🔥 计算筹码形态", value=False, help="计算底部顶格峰 (首次约 30-60 秒，后续使用缓存)")
    
    # 使用 session_state 缓存结果
    cache_key = f"chip_cache_{selected_date}_{selected_market}"
    
    if calc_chip:
        # 检查缓存
        if cache_key in st.session_state:
            cached_data = st.session_state[cache_key]
            # 验证缓存是否包含当前所有股票
            cached_tickers = set(cached_data.keys())
            current_tickers = set(df['Ticker'].tolist())
            if current_tickers.issubset(cached_tickers):
                # 使用缓存
                chip_labels = [cached_data.get(t, '') for t in df['Ticker'].tolist()]
                df['筹码形态'] = chip_labels
                strong_peaks = chip_labels.count('🔥')
                normal_peaks = chip_labels.count('📍')
                st.caption(f"⚡ 使用缓存 | 🔥 强势: {strong_peaks} | 📍 底部密集: {normal_peaks}")
            else:
                # 缓存不完整，需要重新计算
                st.session_state.pop(cache_key, None)
                st.rerun()
        else:
            # 并行计算
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            tickers = df['Ticker'].tolist()
            results = {}
            
            def calc_single(ticker):
                try:
                    stock_df = fetch_data_from_polygon(ticker, days=100)
                    if stock_df is not None and len(stock_df) >= 30:
                        result = quick_chip_analysis(stock_df)
                        return ticker, result.get('label', '') if result else ''
                    return ticker, ''
                except:
                    return ticker, ''
            
            progress_bar = st.progress(0, text="正在分析筹码分布...")
            
            # 使用线程池并行计算 (最多 10 个并发)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(calc_single, t): t for t in tickers}
                completed = 0
                for future in as_completed(futures):
                    ticker, label = future.result()
                    results[ticker] = label
                    completed += 1
                    progress_bar.progress(completed / len(tickers), 
                                          text=f"分析中 {completed}/{len(tickers)} ({ticker})")
            
            progress_bar.empty()
            
            # 保存到缓存
            st.session_state[cache_key] = results
            
            chip_labels = [results.get(t, '') for t in tickers]
            df['筹码形态'] = chip_labels
            
            strong_peaks = chip_labels.count('🔥')
            normal_peaks = chip_labels.count('📍')
            if strong_peaks > 0 or normal_peaks > 0:
                st.success(f"✅ 分析完成！🔥 强势顶格峰: {strong_peaks} 只 | 📍 底部密集: {normal_peaks} 只")
        
        # 应用筹码筛选器
        chip_filter = st.session_state.get('chip_filter', '全部')
        if '筹码形态' in df.columns and chip_filter != '全部':
            before_count = len(df)
            if chip_filter == "仅强势顶格峰 🔥":
                df = df[df['筹码形态'] == '🔥']
            elif chip_filter == "仅底部密集 📍":
                df = df[df['筹码形态'] == '📍']
            elif chip_filter == "有底部信号 (🔥+📍)":
                df = df[df['筹码形态'].isin(['🔥', '📍'])]
            st.info(f"📊 筹码筛选: {before_count} → {len(df)} 只")

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
        "Profit_Ratio": st.column_config.NumberColumn("获利盘", format="%.0f%%", help="获利盘比例"),
        "筹码形态": st.column_config.TextColumn("筹码", width="small", help="🔥=强势顶格峰 📍=底部密集"),
        "新发现": st.column_config.TextColumn("状态", width="small", help="🆕=今日新发现, 📅=之前出现过")
    }

    # === 新发现标记 ===
    # 查询每只股票首次出现在扫描结果中的日期
    if 'Ticker' in df.columns and len(df) > 0:
        tickers = df['Ticker'].tolist()
        first_dates = get_first_scan_dates(tickers, market=selected_market)
        
        def get_newness_label(ticker):
            first_date = first_dates.get(ticker)
            if not first_date:
                return "🆕新发现"  # 没有历史记录，是新发现
            
            # 比较首次日期和选择的日期
            if first_date == selected_date:
                return "🆕新发现"
            else:
                # 计算距今天数
                from datetime import datetime
                try:
                    first_dt = datetime.strptime(first_date, '%Y-%m-%d')
                    selected_dt = datetime.strptime(selected_date, '%Y-%m-%d')
                    days_diff = (selected_dt - first_dt).days
                    if days_diff <= 3:
                        return f"📅{days_diff}天前"
                    elif days_diff <= 7:
                        return f"📅{days_diff}天"
                    else:
                        return f"📅{days_diff}天"
                except:
                    return "📅老股"
        
        df['新发现'] = df['Ticker'].apply(get_newness_label)

    # 显示列顺序：核心指标在前，新发现标记靠前
    display_cols = ['新发现', 'Ticker', 'Name', 'Price', 'Turnover', 'Day BLUE', 'Week BLUE', 'Month BLUE', 'ADX', 'Strategy', '筹码形态', 'Mkt Cap', 'Cap_Category', 'Wave_Desc', 'Chan_Desc', 'Stop Loss', 'Shares Rec', 'Regime']
    existing_cols = [c for c in display_cols if c in df.columns]

    # === 按用户要求分4个标签页 ===
    # 预先计算各类别数据
    has_day = df['Day BLUE'] > 0 if 'Day BLUE' in df.columns else False
    has_week = df['Week BLUE'] > 0 if 'Week BLUE' in df.columns else False
    has_month = df['Month BLUE'] > 0 if 'Month BLUE' in df.columns else False
    
    # 1. 只日BLUE: Day > 0, Week = 0
    df_day_only = df[has_day & ~has_week].sort_values('Day BLUE', ascending=False) if 'Day BLUE' in df.columns else df.head(0)
    
    # 2. 日周/只周: (Day > 0 AND Week > 0) OR (Day = 0 AND Week > 0)
    df_day_week = df[(has_day & has_week) | (~has_day & has_week)].sort_values('Week BLUE', ascending=False) if 'Week BLUE' in df.columns else df.head(0)
    
    # 3. 日周月/只月: (Day > 0 AND Week > 0 AND Month > 0) OR (Month > 0)
    df_month = df[(has_day & has_week & has_month) | has_month].sort_values('Month BLUE', ascending=False) if 'Month BLUE' in df.columns else df.head(0)
    
    # 4. 特殊信号 (黑马/掘地) - 只要有黑马或掘地就显示，不管日周月
    heima_cache_key = f"heima_cache_{selected_date}_{selected_market}"
    if heima_cache_key in st.session_state:
        heima_data = st.session_state[heima_cache_key]
        df['黑马'] = df['Ticker'].map(lambda t: heima_data.get(t, {}).get('heima', False))
        df['掘地'] = df['Ticker'].map(lambda t: heima_data.get(t, {}).get('juedi', False))
        df_special = df[(df['黑马'] == True) | (df['掘地'] == True)].copy()
    else:
        df_special = df.head(0)
    
    # 计算各标签页数量
    count_day_only = len(df_day_only)
    count_day_week = len(df_day_week)
    count_month = len(df_month)
    count_special = len(df_special)
    
    # 创建标签页 (增加板块热度)
    tab_day_only, tab_day_week, tab_month, tab_special, tab_sector = st.tabs([
        f"📈 只日线 ({count_day_only})",
        f"📊 日+周线 ({count_day_week})",
        f"📅 含月线 ({count_month})",
        f"🐴⛏️ 特殊信号 ({count_special})",
        "🔥 板块热度"
    ])
    
    # 用于存储各标签页选择的行 (用于深度透视)
    selected_ticker = None
    selected_row_data = None
    
    with tab_day_only:
        st.caption("💡 只有日线信号，尚未形成周线共振，适合短线")
        if len(df_day_only) > 0:
            df_day_only = df_day_only.sort_values('Day BLUE', ascending=False)
            event1 = st.dataframe(
                df_day_only[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="df_day_only"
            )
            if event1 and hasattr(event1, 'selection') and event1.selection.rows:
                idx = event1.selection.rows[0]
                if idx < len(df_day_only):
                    selected_ticker = df_day_only.iloc[idx]['Ticker']
                    selected_row_data = df_day_only.iloc[idx]
        else:
            st.info("暂无只有日线信号的股票")
    
    with tab_day_week:
        st.caption("💡 日周双信号共振 或 周线独立信号，中期趋势确认")
        if len(df_day_week) > 0:
            df_day_week = df_day_week.sort_values('Week BLUE', ascending=False)
            event2 = st.dataframe(
                df_day_week[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="df_day_week"
            )
            if event2 and hasattr(event2, 'selection') and event2.selection.rows:
                idx = event2.selection.rows[0]
                if idx < len(df_day_week):
                    selected_ticker = df_day_week.iloc[idx]['Ticker']
                    selected_row_data = df_day_week.iloc[idx]
        else:
            st.info("暂无日周共振或周线信号的股票")
    
    with tab_month:
        st.caption("💡 日周月三重共振 或 月线信号，大级别底部机会")
        if len(df_month) > 0:
            df_month = df_month.sort_values('Month BLUE', ascending=False)
            event3 = st.dataframe(
                df_month[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="df_month"
            )
            if event3 and hasattr(event3, 'selection') and event3.selection.rows:
                idx = event3.selection.rows[0]
                if idx < len(df_month):
                    selected_ticker = df_month.iloc[idx]['Ticker']
                    selected_row_data = df_month.iloc[idx]
        else:
            st.info("暂无含月线信号的股票")
    
    with tab_special:
        st.caption("🐴 黑马 / ⛏️ 掘地 / 🔥 顶格峰：特殊形态信号")
        
        # === 扫描范围选择 ===
        scan_scope = st.radio(
            "扫描范围",
            options=["📋 当前信号股", "🌐 全量股票"],
            horizontal=True,
            help="当前信号股=只扫描已有BLUE信号的股票 | 全量股票=扫描市场所有股票",
            key="special_scan_scope"
        )
        
        # 根据选择确定扫描列表
        if scan_scope == "📋 当前信号股":
            scan_tickers = df['Ticker'].tolist()
            scope_label = "当前信号股"
        else:
            # 全量扫描 - 从数据库获取市场所有股票
            from db.database import get_stock_info_batch
            try:
                # 获取市场所有股票信息
                all_stocks = get_stock_info_batch(None)  # 获取所有
                if selected_market == 'CN':
                    scan_tickers = [s['symbol'] for s in all_stocks if s.get('market') == 'CN']
                else:
                    scan_tickers = [s['symbol'] for s in all_stocks if s.get('market') == 'US']
                scope_label = f"全量扫描 ({len(scan_tickers)} 只)"
            except Exception as e:
                scan_tickers = df['Ticker'].tolist()
                scope_label = f"当前信号股 (全量失败: {e})"
        
        st.caption(f"📊 扫描范围: {scope_label} | 共 {len(scan_tickers)} 只股票")
        
        # === 特殊信号缓存 ===
        special_cache_key = f"special_signals_{selected_date}_{selected_market}_{scan_scope}"
        
        if special_cache_key not in st.session_state:
            st.info("需要扫描特殊信号，点击下方按钮开始")
            
            if st.button("🔍 扫描黑马/掘地/顶格峰信号", key="scan_special", type="primary"):
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from indicator_utils import calculate_heima_signal_series
                from chart_utils import quick_chip_analysis
                
                results = {}
                
                def calc_special_signals(ticker):
                    """计算单只股票的特殊信号: 黑马、掘地、顶格峰"""
                    try:
                        stock_df = fetch_data_from_polygon(ticker, days=100)
                        if stock_df is None or len(stock_df) < 30:
                            return ticker, {'heima': False, 'juedi': False, 'bottom_peak': False}
                        
                        # 黑马/掘地信号
                        heima, juedi = calculate_heima_signal_series(
                            stock_df['High'].values,
                            stock_df['Low'].values,
                            stock_df['Close'].values,
                            stock_df['Open'].values
                        )
                        
                        # 顶格峰信号 (最近3天内出现)
                        bottom_peak = False
                        try:
                            chip = quick_chip_analysis(stock_df)
                            if chip and chip.get('is_strong_bottom_peak'):
                                bottom_peak = True
                            elif chip and chip.get('is_bottom_peak'):
                                bottom_peak = True
                        except:
                            pass
                        
                        return ticker, {
                            'heima': bool(heima[-1]) if len(heima) > 0 else False,
                            'juedi': bool(juedi[-1]) if len(juedi) > 0 else False,
                            'bottom_peak': bottom_peak
                        }
                    except:
                        return ticker, {'heima': False, 'juedi': False, 'bottom_peak': False}
                
                progress = st.progress(0, text="正在扫描特殊信号...")
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(calc_special_signals, t): t for t in scan_tickers}
                    completed = 0
                    for future in as_completed(futures):
                        ticker, signals = future.result()
                        results[ticker] = signals
                        completed += 1
                        progress.progress(completed / len(scan_tickers), text=f"扫描中 {completed}/{len(scan_tickers)}")
                
                progress.empty()
                st.session_state[special_cache_key] = results
                
                # 统计结果
                heima_count = sum(1 for r in results.values() if r['heima'])
                juedi_count = sum(1 for r in results.values() if r['juedi'])
                peak_count = sum(1 for r in results.values() if r['bottom_peak'])
                st.success(f"✅ 扫描完成！🐴 黑马: {heima_count} | ⛏️ 掘地: {juedi_count} | 🔥 顶格峰: {peak_count}")
                st.rerun()
        else:
            # 显示结果
            signal_data = st.session_state[special_cache_key]
            
            # 构建特殊信号数据框
            special_rows = []
            for ticker, signals in signal_data.items():
                if signals['heima'] or signals['juedi'] or signals['bottom_peak']:
                    signal_types = []
                    if signals['heima']:
                        signal_types.append('🐴黑马')
                    if signals['juedi']:
                        signal_types.append('⛏️掘地')
                    if signals['bottom_peak']:
                        signal_types.append('🔥顶格峰')
                    
                    # 尝试从 df 获取更多信息
                    ticker_info = df[df['Ticker'] == ticker]
                    if len(ticker_info) > 0:
                        row = ticker_info.iloc[0].to_dict()
                        row['信号类型'] = ' '.join(signal_types)
                        special_rows.append(row)
                    else:
                        # 只有ticker信息
                        special_rows.append({
                            'Ticker': ticker,
                            '信号类型': ' '.join(signal_types)
                        })
            
            if special_rows:
                df_special_result = pd.DataFrame(special_rows)
                
                # 统计显示
                st.markdown(f"**找到 {len(special_rows)} 只特殊信号股票**")
                
                display_with_signal = ['信号类型'] + existing_cols
                cols_to_show = [c for c in display_with_signal if c in df_special_result.columns]
                
                event4 = st.dataframe(
                    df_special_result[cols_to_show],
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    key="df_special"
                )
                if event4 and hasattr(event4, 'selection') and event4.selection.rows:
                    idx = event4.selection.rows[0]
                    if idx < len(df_special_result):
                        selected_ticker = df_special_result.iloc[idx]['Ticker']
                        selected_row_data = df_special_result.iloc[idx]
            else:
                st.info("暂无黑马、掘地或顶格峰信号的股票")
            
            # 清除缓存按钮
            if st.button("🔄 重新扫描", key="rescan_special"):
                del st.session_state[special_cache_key]
                st.rerun()

    # === 板块热度标签页 ===
    with tab_sector:
        st.caption("🔥 行业板块涨跌幅排名 - 追踪市场热点")
        
        from data_fetcher import get_sector_data, get_cn_sector_data_period, get_us_sector_data_period
        
        # 时间段选择
        period_options = {
            "📅 今日": "1d",
            "📆 本周": "1w", 
            "📊 本月": "1m",
            "📈 今年": "ytd"
        }
        selected_period_label = st.radio(
            "时间范围",
            options=list(period_options.keys()),
            horizontal=True,
            key="sector_period"
        )
        selected_period = period_options[selected_period_label]
        
        # 缓存板块数据 (按时间段)
        sector_cache_key = f"sector_data_{selected_market}_{selected_period}"
        
        col_refresh, col_info = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 刷新", key="refresh_sector"):
                # 清除所有时间段缓存
                for p in period_options.values():
                    key = f"sector_data_{selected_market}_{p}"
                    if key in st.session_state:
                        del st.session_state[key]
        
        if sector_cache_key not in st.session_state:
            with st.spinner(f"正在获取{selected_period_label}板块数据..."):
                try:
                    if selected_market == 'CN':
                        sector_df = get_cn_sector_data_period(period=selected_period)
                    else:
                        sector_df = get_us_sector_data_period(period=selected_period)
                    
                    if sector_df is not None:
                        st.session_state[sector_cache_key] = sector_df
                except Exception as e:
                    st.error(f"获取数据失败: {e}")
                    sector_df = None
        
        if sector_cache_key in st.session_state:
            sector_df = st.session_state[sector_cache_key]
            
            if sector_df is not None and len(sector_df) > 0:
                # 统计信息
                up_count = len(sector_df[sector_df['change_pct'] > 0])
                down_count = len(sector_df[sector_df['change_pct'] < 0])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("板块数量", len(sector_df))
                with col2:
                    st.metric("🔴 上涨", up_count)
                with col3:
                    st.metric("🟢 下跌", down_count)
                
                st.divider()
                
                # 分两列显示：涨幅榜和跌幅榜
                col_up, col_down = st.columns(2)
                
                with col_up:
                    st.markdown(f"### 📈 {selected_period_label} 涨幅榜 Top 15")
                    top_up = sector_df.head(15).copy()
                    top_up['change_pct'] = top_up['change_pct'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                    if 'amount' in top_up.columns:
                        top_up['amount'] = top_up['amount'].apply(lambda x: f"{x:.1f}亿" if pd.notna(x) else "N/A")
                    if 'stock_count' in top_up.columns:
                        display_cols_up = ['name', 'change_pct', 'amount', 'stock_count']
                    else:
                        display_cols_up = ['name', 'change_pct']
                    cols_to_show = [c for c in display_cols_up if c in top_up.columns]
                    st.dataframe(
                        top_up[cols_to_show],
                        column_config={
                            'name': '板块',
                            'change_pct': '涨跌幅',
                            'amount': '成交额',
                            'stock_count': '股票数'
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                with col_down:
                    st.markdown(f"### 📉 {selected_period_label} 跌幅榜 Top 15")
                    top_down = sector_df.tail(15).iloc[::-1].copy()
                    top_down['change_pct'] = top_down['change_pct'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                    if 'amount' in top_down.columns:
                        top_down['amount'] = top_down['amount'].apply(lambda x: f"{x:.1f}亿" if pd.notna(x) else "N/A")
                    if 'stock_count' in top_down.columns:
                        display_cols_down = ['name', 'change_pct', 'amount', 'stock_count']
                    else:
                        display_cols_down = ['name', 'change_pct']
                    cols_to_show = [c for c in display_cols_down if c in top_down.columns]
                    st.dataframe(
                        top_down[cols_to_show],
                        column_config={
                            'name': '板块',
                            'change_pct': '涨跌幅',
                            'amount': '成交额',
                            'stock_count': '股票数'
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info("暂无板块数据")
            
            # === 板块详情区域 ===
            st.divider()
            st.markdown("### 🔍 板块详情")
            
            # 板块选择下拉框
            sector_names = sector_df['name'].tolist()
            selected_sector = st.selectbox(
                "选择板块查看详情",
                options=sector_names,
                key="sector_detail_select"
            )
            
            if selected_sector:
                with st.expander(f"📊 {selected_sector} 详情", expanded=True):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("#### 🔥 板块热门股")
                        
                        # 获取该板块的热门股票
                        hot_stocks_key = f"hot_stocks_{selected_sector}_{selected_market}"
                        
                        if hot_stocks_key not in st.session_state:
                            with st.spinner("加载热门股..."):
                                try:
                                    if selected_market == 'CN':
                                        from data_fetcher import get_cn_sector_hot_stocks
                                        hot_df = get_cn_sector_hot_stocks(selected_sector)
                                    else:
                                        # 美股暂用 ETF 组成
                                        hot_df = None
                                    st.session_state[hot_stocks_key] = hot_df
                                except Exception as e:
                                    st.session_state[hot_stocks_key] = None
                        
                        hot_df = st.session_state.get(hot_stocks_key)
                        if hot_df is not None and len(hot_df) > 0:
                            st.dataframe(
                                hot_df[['name', 'pct_chg']].head(10),
                                column_config={
                                    'name': '股票',
                                    'pct_chg': '涨跌幅%'
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.info("暂无热门股数据")
                    
                    with detail_col2:
                        st.markdown("#### 📰 相关新闻")
                        
                        # 显示新闻搜索链接
                        if selected_market == 'CN':
                            search_term = f"{selected_sector}板块 股票 新闻"
                            baidu_url = f"https://www.baidu.com/s?wd={search_term}"
                            st.markdown(f"🔗 [百度搜索: {selected_sector}新闻]({baidu_url})")
                            
                            eastmoney_url = f"https://so.eastmoney.com/news/s?keyword={selected_sector}"
                            st.markdown(f"🔗 [东方财富: {selected_sector}]({eastmoney_url})")
                        else:
                            search_term = f"{selected_sector} sector stocks news"
                            google_url = f"https://www.google.com/search?q={search_term}&tbm=nws"
                            st.markdown(f"🔗 [Google News: {selected_sector}]({google_url})")
                            
                            yahoo_url = f"https://finance.yahoo.com/quote/{sector_df[sector_df['name']==selected_sector]['sector'].values[0] if len(sector_df[sector_df['name']==selected_sector]) > 0 else 'XLK'}"
                            st.markdown(f"🔗 [Yahoo Finance]({yahoo_url})")
                        
                        st.caption("💡 点击链接查看最新市场资讯")
        else:
            st.info("正在加载板块数据...")

    # 4. 深度透视 (所有标签页都支持选择)
    if selected_ticker is not None and selected_row_data is not None:
        symbol = selected_ticker
        selected_row = selected_row_data
        
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
                            
                            # 形态 + 底部顶格峰指标
                            pattern = chip.get('pattern_desc', 'N/A')
                            bottom_ratio = chip.get('bottom_chip_ratio', 0) * 100
                            
                            if chip.get('is_strong_bottom_peak'):
                                st.success(f"🔥 **底部顶格峰!** 底部筹码占比: {bottom_ratio:.1f}% | {pattern}")
                            elif chip.get('is_bottom_peak'):
                                st.info(f"📍 **底部密集** 底部筹码占比: {bottom_ratio:.1f}% | {pattern}")
                            else:
                                st.caption(f"📋 形态: **{pattern}** | 底部筹码: {bottom_ratio:.1f}%")
                        
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
                st.info(f"**🦅 波浪**: {wave_desc} ({wave_phase})")
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
    
    # 导入数据库函数
    from db.database import (
        get_signal_history, get_portfolio, add_to_watchlist, 
        add_trade, get_trades, update_watchlist_status, delete_from_watchlist
    )
    
    # Tab 结构
    tab1, tab2, tab3 = st.tabs(["📊 信号表现", "🔍 信号复盘", "💼 我的持仓"])
    
    # ==================== Tab 1: 信号表现 (原有功能) ====================
    with tab1:
        st.info("查看历史扫描结果中股票的后续走势，验证信号有效性。")
        render_signal_performance_tab()
    
    # ==================== Tab 2: 信号复盘 ====================
    with tab2:
        st.info("选择股票查看历史所有信号点及后续表现")
        render_signal_review_tab()
    
    # ==================== Tab 3: 我的持仓 ====================
    with tab3:
        st.info("添加并跟踪你的实际持仓，记录交易")
        render_portfolio_tab()


def render_signal_performance_tab():
    """信号表现 Tab (原有功能)"""
    from services.signal_tracker_service import batch_calculate_returns

    # 侧边栏设置
    with st.sidebar:
        st.subheader("📊 追踪设置")
        
        # 市场选择
        market = st.radio(
            "选择市场",
            ["🇺🇸 美股", "🇨🇳 A股"],
            horizontal=True,
            key="tracker_market"
        )
        market_code = "US" if "美股" in market else "CN"
        
        # 获取历史扫描日期
        dates = get_scanned_dates(market=market_code)
        
        if not dates:
            st.warning(f"暂无 {market} 的历史扫描数据")
            return
        
        # 日期选择
        selected_date = st.selectbox(
            "选择扫描日期",
            options=dates[:30],  # 最近30天
            index=0,
            help="选择要追踪的历史扫描日期"
        )
        
        # 追踪天数
        track_days = st.slider("追踪天数", 5, 30, 20)
        
        # 计算按钮
        calculate_btn = st.button("🔍 计算信号表现", type="primary", use_container_width=True)
    
    # 主区域
    if not calculate_btn:
        # 显示说明
        st.markdown("""
        ### 使用说明
        
        1. 在左侧选择 **市场** 和 **历史扫描日期**
        2. 点击 **"计算信号表现"** 按钮
        3. 系统将分析该日期扫描出的信号在后续的表现
        
        #### 指标说明
        - **胜率**: 信号后续上涨的比例
        - **平均收益**: 所有信号的平均收益率
        - **5D/10D/20D**: 信号后 5/10/20 个交易日的收益
        """)
        
        # 显示可用日期概览
        if dates:
            st.markdown("### 📅 可用历史日期")
            
            # 获取每个日期的信号数量
            date_info = []
            for d in dates[:10]:
                count = len(query_scan_results(scan_date=d, market=market_code, limit=1000))
                date_info.append({'日期': d, '信号数': count})
            
            if date_info:
                st.dataframe(pd.DataFrame(date_info), hide_index=True, use_container_width=True)
        return
    
    # 执行计算
    with st.spinner(f"正在计算 {selected_date} 的信号表现..."):
        # 获取该天的扫描结果
        scan_results = query_scan_results(scan_date=selected_date, market=market_code, limit=100)
        
        if not scan_results:
            st.error("该日期没有扫描结果")
            return
        
        st.success(f"找到 {len(scan_results)} 个信号，正在计算后续表现...")
        
        # 准备信号列表
        signals = [{
            'symbol': r['symbol'],
            'signal_date': selected_date,
            'day_blue': r.get('blue_daily', 0),
            'week_blue': r.get('blue_weekly', 0),
            'name': r.get('name', ''),
            'entry_price': r.get('price', 0)
        } for r in scan_results]
        
        # 批量计算收益
        progress_bar = st.progress(0, text="计算中...")
        returns = batch_calculate_returns(signals, market_code, max_workers=15)
        progress_bar.progress(100, text="计算完成!")
        
        if not returns:
            st.warning("无法获取足够的历史数据来计算收益")
            return
    
    # 转换为 DataFrame
    df = pd.DataFrame(returns)
    
    # 显示统计摘要
    st.markdown("---")
    st.markdown("### 📊 整体表现统计")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = len(returns)
        st.metric("分析信号数", f"{total}")
    
    with col2:
        if 'return_5d' in df.columns:
            valid_5d = df['return_5d'].dropna()
            avg_5d = valid_5d.mean() if len(valid_5d) > 0 else 0
            st.metric("平均 5D 收益", f"{avg_5d:+.2f}%",
                     delta="盈利" if avg_5d > 0 else "亏损",
                     delta_color="normal" if avg_5d > 0 else "inverse")
    
    with col3:
        if 'return_10d' in df.columns:
            valid_10d = df['return_10d'].dropna()
            avg_10d = valid_10d.mean() if len(valid_10d) > 0 else 0
            st.metric("平均 10D 收益", f"{avg_10d:+.2f}%",
                     delta="盈利" if avg_10d > 0 else "亏损",
                     delta_color="normal" if avg_10d > 0 else "inverse")
    
    with col4:
        if 'return_20d' in df.columns:
            valid_20d = df['return_20d'].dropna()
            avg_20d = valid_20d.mean() if len(valid_20d) > 0 else 0
            st.metric("平均 20D 收益", f"{avg_20d:+.2f}%",
                     delta="盈利" if avg_20d > 0 else "亏损",
                     delta_color="normal" if avg_20d > 0 else "inverse")
    
    with col5:
        if 'return_20d' in df.columns:
            valid = df['return_20d'].dropna()
            if len(valid) > 0:
                win_rate = len(valid[valid > 0]) / len(valid) * 100
                st.metric("20D 胜率", f"{win_rate:.0f}%",
                         delta="优秀" if win_rate > 60 else ("一般" if win_rate > 40 else "较差"))
    
    # 信号分类
    st.markdown("### 🎯 信号分类")
    
    if 'return_20d' in df.columns:
        df_valid = df.dropna(subset=['return_20d'])
        
        excellent = df_valid[df_valid['return_20d'] > 10]
        good = df_valid[(df_valid['return_20d'] > 0) & (df_valid['return_20d'] <= 10)]
        poor = df_valid[df_valid['return_20d'] <= 0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"### ✅ 优质信号: {len(excellent)}")
            st.caption("20D 收益 > 10%")
            if len(excellent) > 0:
                st.write(f"平均 BLUE: {excellent['day_blue'].mean():.0f}")
        
        with col2:
            st.info(f"### 🟡 一般信号: {len(good)}")
            st.caption("20D 收益 0-10%")
            if len(good) > 0:
                st.write(f"平均 BLUE: {good['day_blue'].mean():.0f}")
        
        with col3:
            st.warning(f"### ❌ 差信号: {len(poor)}")
            st.caption("20D 收益 < 0%")
            if len(poor) > 0:
                st.write(f"平均 BLUE: {poor['day_blue'].mean():.0f}")
    
    # 详细数据表格
    st.markdown("### 📋 详细数据")
    
    # 准备显示数据
    display_df = df[['symbol', 'name', 'day_blue', 'entry_price', 
                     'return_5d', 'return_10d', 'return_20d', 
                     'max_gain', 'max_drawdown', 'current_return']].copy()
    
    display_df.columns = ['代码', '名称', 'Day BLUE', '入场价', 
                          '5D收益', '10D收益', '20D收益', 
                          '最大涨幅', '最大回撤', '当前收益']
    
    # 格式化
    for col in ['5D收益', '10D收益', '20D收益', '最大涨幅', '最大回撤', '当前收益']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )
    
    display_df['入场价'] = display_df['入场价'].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) and x > 0 else "N/A"
    )
    
    # 排序选项
    sort_col = st.selectbox("排序方式", ['20D收益', '10D收益', '5D收益', 'Day BLUE'], key="sort_col")
    
    # 因为已经格式化为字符串，需要对原始数据排序
    sort_map = {'20D收益': 'return_20d', '10D收益': 'return_10d', '5D收益': 'return_5d', 'Day BLUE': 'day_blue'}
    if sort_map[sort_col] in df.columns:
        sort_idx = df[sort_map[sort_col]].sort_values(ascending=False).index
        display_df = display_df.loc[sort_idx]
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # 信号质量评估
    st.markdown("### 💡 信号质量评估")
    
    if 'return_20d' in df.columns:
        valid_20d = df['return_20d'].dropna()
        if len(valid_20d) > 0:
            avg_return = valid_20d.mean()
            win_rate = len(valid_20d[valid_20d > 0]) / len(valid_20d) * 100
            
            if avg_return > 5 and win_rate > 55:
                st.success(f"""
                **✅ 优质信号批次**
                
                - 平均 20D 收益: {avg_return:.2f}%
                - 胜率: {win_rate:.0f}%
                - 优质信号占比: {len(excellent)/len(df_valid)*100:.0f}%
                
                该批次信号表现优秀，策略参数有效！
                """)
            elif avg_return > 0 and win_rate > 40:
                st.info(f"""
                **🟡 一般信号批次**
                
                - 平均 20D 收益: {avg_return:.2f}%
                - 胜率: {win_rate:.0f}%
                
                该批次信号表现一般，建议结合其他指标筛选。
                """)
            else:
                st.warning(f"""
                **⚠️ 低质量信号批次**
                
                - 平均 20D 收益: {avg_return:.2f}%
                - 胜率: {win_rate:.0f}%
                
                该批次信号表现不佳，建议调整策略参数。
                """)


def render_signal_review_tab():
    """信号复盘 Tab - 查看个股历史信号"""
    from db.database import get_signal_history
    
    st.markdown("### 🔍 选择股票查看历史信号")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("输入股票代码", value="", placeholder="例如: NVDA, AAPL", key="review_symbol")
    
    with col2:
        market = st.radio("市场", ["US", "CN"], horizontal=True, key="review_market")
    
    if not symbol:
        st.info("请输入股票代码开始查看信号历史")
        return
    
    symbol = symbol.upper().strip()
    
    with st.spinner(f"正在加载 {symbol} 的历史信号..."):
        signals = get_signal_history(symbol, market=market, limit=100)
    
    if not signals:
        st.warning(f"未找到 {symbol} 的历史信号记录")
        return
    
    st.success(f"找到 {len(signals)} 条历史信号记录")
    
    # 转换为 DataFrame
    df = pd.DataFrame(signals)
    
    # 显示信号列表
    st.markdown("### 📋 历史信号列表")
    
    display_cols = ['scan_date', 'price', 'blue_daily', 'blue_weekly', 'wave_phase']
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()
    display_df.columns = ['信号日期', '当日价格', 'Day BLUE', 'Week BLUE', '波浪阶段'][:len(available_cols)]
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # 信号统计
    st.markdown("### 📊 信号统计")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("信号次数", len(signals))
    with col2:
        avg_blue = df['blue_daily'].mean() if 'blue_daily' in df.columns else 0
        st.metric("平均 Day BLUE", f"{avg_blue:.0f}" if avg_blue else "N/A")
    with col3:
        max_blue = df['blue_daily'].max() if 'blue_daily' in df.columns else 0
        st.metric("最高 Day BLUE", f"{max_blue:.0f}" if max_blue else "N/A")


def render_portfolio_tab():
    """我的持仓 Tab - 手动添加和跟踪持仓"""
    from db.database import (
        get_portfolio, add_to_watchlist, add_trade, 
        get_trades, update_watchlist_status, delete_from_watchlist
    )
    
    # 权限检查
    if not is_admin():
        st.warning("⚠️ 持仓管理需要管理员权限，您当前为访客模式（只读）")
        st.markdown("---")
    
    # 添加股票表单 (仅管理员可见)
    if is_admin():
        st.markdown("### ➕ 添加持仓")
        
        with st.expander("添加新持仓", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_symbol = st.text_input("股票代码", placeholder="NVDA", key="add_symbol")
            with col2:
                new_price = st.number_input("买入价格", min_value=0.01, value=100.0, key="add_price")
            with col3:
                new_shares = st.number_input("股数", min_value=1, value=100, key="add_shares")
            
            col4, col5 = st.columns(2)
            with col4:
                new_market = st.selectbox("市场", ["US", "CN"], key="add_market")
            with col5:
                new_date = st.date_input("买入日期", key="add_date")
            
            notes = st.text_input("备注", placeholder="可选", key="add_notes")
            
            if st.button("✅ 添加持仓", type="primary"):
                if new_symbol:
                    symbol = new_symbol.upper().strip()
                    entry_date = new_date.strftime('%Y-%m-%d')
                    
                    # 添加到持仓列表
                    add_to_watchlist(symbol, new_price, new_shares, entry_date, new_market, 'holding', notes)
                    
                    # 记录买入交易
                    add_trade(symbol, 'BUY', new_price, new_shares, entry_date, new_market, notes)
                    
                    st.success(f"✅ 已添加 {symbol} 到持仓")
                    st.rerun()
                else:
                    st.error("请输入股票代码")
    
    # 当前持仓
    st.markdown("### 💼 当前持仓")
    
    portfolio = get_portfolio(status='holding')
    
    if not portfolio:
        st.info("暂无持仓，点击上方添加")
    else:
        for item in portfolio:
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
            
            with col1:
                st.write(f"**{item['symbol']}**")
                st.caption(f"买入: ${item['entry_price']:.2f}")
            
            with col2:
                st.write(f"{item['shares']} 股")
                st.caption(f"日期: {item['entry_date']}")
            
            with col3:
                st.write(f"持仓中")
                st.caption(f"市场: {item['market']}")
            
            with col4:
                if item.get('notes'):
                    st.caption(item['notes'])
            
            with col5:
                if is_admin():
                    if st.button("卖出", key=f"sell_{item['id']}"):
                        st.session_state[f"show_sell_{item['id']}"] = True
            
            # 卖出对话框
            if is_admin() and st.session_state.get(f"show_sell_{item['id']}"):
                with st.container():
                    sell_price = st.number_input(
                        f"卖出价格 ({item['symbol']})", 
                        min_value=0.01, 
                        value=float(item['entry_price']),
                        key=f"sell_price_{item['id']}"
                    )
                    if st.button(f"确认卖出", key=f"confirm_sell_{item['id']}"):
                        add_trade(item['symbol'], 'SELL', sell_price, item['shares'], 
                                 datetime.now().strftime('%Y-%m-%d'), item['market'])
                        update_watchlist_status(item['symbol'], item['entry_date'], 'sold', item['market'])
                        st.success(f"✅ 已卖出 {item['symbol']}")
                        st.session_state[f"show_sell_{item['id']}"] = False
                        st.rerun()
            
            st.divider()
    
    # 交易历史
    st.markdown("### 📜 交易历史")
    
    trades = get_trades(limit=20)
    
    if trades:
        df = pd.DataFrame(trades)
        display_df = df[['symbol', 'trade_type', 'price', 'shares', 'trade_date', 'market']].copy()
        display_df.columns = ['代码', '类型', '价格', '股数', '日期', '市场']
        display_df['价格'] = display_df['价格'].apply(lambda x: f"${x:.2f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.info("暂无交易记录")


# --- 信号表现验证页面 (新增) ---

def render_signal_performance_page():
    """信号表现验证仪表盘 - 验证 BLUE 信号的历史有效性"""
    st.header("📊 信号表现验证 (Signal Performance)")
    st.info("验证 BLUE 信号的历史盈利能力，对比 SPY 基准表现")
    
    from services.backtest_service import run_signal_backtest, get_backtest_summary_table
    from datetime import datetime, timedelta
    
    # 侧边栏参数
    with st.sidebar:
        st.subheader("🎛️ 回测参数")
        
        # 市场选择
        market = st.radio("市场", ["US", "CN"], horizontal=True)
        
        # 日期范围
        col1, col2 = st.columns(2)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        with col1:
            start = st.date_input("开始日期", value=start_date)
        with col2:
            end = st.date_input("结束日期", value=end_date)
        
        # BLUE 阈值
        min_blue = st.slider("最低 BLUE 阈值", min_value=50, max_value=200, value=100, step=10)
        
        # 持仓周期
        forward_days = st.select_slider(
            "持仓周期 (天)",
            options=[5, 10, 20, 30],
            value=10,
            help="信号触发后持有多少天"
        )
        
        # 分析数量限制
        limit = st.number_input("最大分析数量", min_value=50, max_value=500, value=200, step=50)
        
        run_btn = st.button("🚀 开始验证", type="primary", use_container_width=True)
    
    # 使用说明
    if not run_btn:
        st.markdown("""
        ### 🎯 使用说明
        
        1. 在左侧设置 **回测参数**
        2. 点击 **开始验证** 按钮
        3. 查看 BLUE 信号的历史表现
        
        ---
        
        ### 📈 关键指标说明
        
        | 指标 | 说明 |
        |------|------|
        | **Win Rate** | 信号触发后盈利的比例 |
        | **Avg Return** | 平均每笔信号的收益率 |
        | **Sharpe Ratio** | 风险调整后收益 (>1 优秀) |
        | **Max Drawdown** | 最大回撤幅度 |
        | **Profit Factor** | 盈利/亏损比 (>1.5 优秀) |
        
        > ⚠️ **注意**: 需要信号日期后有足够的交易日数据才能计算收益
        """)
        return
    
    # 运行回测
    with st.spinner(f"正在分析 {market} 市场的 BLUE 信号..."):
        result = run_signal_backtest(
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
            market=market,
            min_blue=min_blue,
            forward_days=forward_days,
            limit=limit
        )
    
    metrics = result.get('metrics', {})
    spy_metrics = result.get('spy_metrics', {})
    signals = result.get('signals', [])
    params = result.get('params', {})
    
    # 顶部摘要卡片
    st.markdown("---")
    st.subheader("📊 表现概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        win_rate = metrics.get('win_rate', 0)
        st.metric(
            "胜率 (Win Rate)",
            f"{win_rate:.1f}%",
            delta=f"vs SPY {spy_metrics.get('win_rate', 0):.1f}%" if spy_metrics else None,
            delta_color="normal" if win_rate > spy_metrics.get('win_rate', 0) else "inverse"
        )
    
    with col2:
        avg_ret = metrics.get('avg_return', 0)
        st.metric(
            "平均收益",
            f"{avg_ret:.2f}%",
            delta=f"vs SPY {spy_metrics.get('avg_return', 0):.2f}%" if spy_metrics else None,
            delta_color="normal" if avg_ret > spy_metrics.get('avg_return', 0) else "inverse"
        )
    
    with col3:
        sharpe = metrics.get('sharpe', 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="优秀" if sharpe > 1 else ("良好" if sharpe > 0.5 else "待改进")
        )
    
    with col4:
        pf = metrics.get('profit_factor', 0)
        st.metric(
            "Profit Factor",
            f"{pf:.2f}",
            delta="优秀" if pf > 1.5 else ("一般" if pf > 1 else "亏损")
        )
    
    # 第二行指标
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("分析信号数", metrics.get('total_signals', 0))
    
    with col6:
        st.metric("盈利信号", metrics.get('winning_signals', 0))
    
    with col7:
        st.metric("亏损信号", metrics.get('losing_signals', 0))
    
    with col8:
        mdd = metrics.get('max_drawdown', 0)
        st.metric("最大回撤", f"{mdd:.2f}%", delta_color="inverse")
    
    # 对比表格
    st.markdown("---")
    st.subheader("📋 BLUE vs SPY 对比")
    
    summary_df = get_backtest_summary_table(result)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # 累积收益曲线图表
    st.markdown("---")
    st.subheader("📈 累积收益曲线 (Cumulative Returns)")
    
    from services.backtest_service import create_cumulative_returns_chart
    cumulative_chart = create_cumulative_returns_chart(result)
    st.plotly_chart(cumulative_chart, use_container_width=True)
    
    # 信号详情表
    if signals:
        st.markdown("---")
        st.subheader(f"📈 信号详情 ({len(signals)} 条)")
        
        # 转换为 DataFrame
        signals_df = pd.DataFrame(signals)
        
        # 根据 forward_days 动态确定列名
        ret_col = f'return_{forward_days}d'
        spy_ret_col = f'spy_return_{forward_days}d'
        
        # 格式化显示
        if ret_col in signals_df.columns:
            signals_df[f'{forward_days}d收益%'] = signals_df[ret_col].apply(
                lambda x: f"{x*100:.2f}%" if x is not None else "N/A"
            )
        if spy_ret_col in signals_df.columns:
            signals_df[f'SPY{forward_days}d%'] = signals_df[spy_ret_col].apply(
                lambda x: f"{x*100:.2f}%" if x is not None else "N/A"
            )
        if 'alpha' in signals_df.columns:
            signals_df['Alpha%'] = signals_df['alpha'].apply(
                lambda x: f"{x*100:.2f}%" if x is not None else "N/A"
            )
        
        display_cols = ['symbol', 'signal_date', 'blue_daily', 'price', 
                       f'{forward_days}d收益%', f'SPY{forward_days}d%', 'Alpha%']
        display_cols = [c for c in display_cols if c in signals_df.columns]
        
        # 重命名列
        rename_map = {
            'symbol': '代码',
            'signal_date': '信号日期',
            'blue_daily': 'Day BLUE',
            'price': '信号价格'
        }
        display_df = signals_df[display_cols].rename(columns=rename_map)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Day BLUE": st.column_config.ProgressColumn(
                    "Day BLUE",
                    format="%.0f",
                    min_value=0,
                    max_value=200
                )
            }
        )
    else:
        st.warning("未找到符合条件的信号。请调整日期范围或 BLUE 阈值。")
    
    # 参数摘要
    with st.expander("🔧 回测参数"):
        st.json(params)


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

# --- Baseline 对比页面 ---

def render_baseline_comparison_page():
    """Baseline 扫描对比页面"""
    st.header("🔄 Baseline 对比 (Scan Comparison)")
    st.info("对比 Baseline 扫描方法与当前扫描方法的结果差异")
    
    from db.database import query_baseline_results, compare_scan_results, get_scanned_dates
    
    # 侧边栏设置
    with st.sidebar:
        st.subheader("📊 对比设置")
        
        # 市场选择
        market = st.radio("选择市场", ["🇺🇸 US", "🇨🇳 CN"], horizontal=True, key="cmp_market")
        market_code = "US" if "US" in market else "CN"
        
        # 获取可用日期
        dates = get_scanned_dates(market=market_code)
        if not dates:
            st.warning("暂无扫描数据")
            return
        
        selected_date = st.selectbox("选择日期", dates[:30], key="cmp_date")
        
        compare_btn = st.button("🔍 开始对比", type="primary", use_container_width=True)
    
    if not compare_btn:
        st.markdown("""
        ### 使用说明
        
        1. 在左侧选择 **市场** 和 **日期**
        2. 点击 **开始对比** 按钮
        3. 查看两种扫描方法的结果差异
        
        #### Baseline vs 当前方法
        - **Baseline**: 原始的 BLUE 信号扫描算法
        - **当前方法**: 包含更多过滤条件的优化版本
        """)
        return
    
    with st.spinner("正在对比数据..."):
        comparison = compare_scan_results(selected_date, market_code)
        baseline_results = query_baseline_results(scan_date=selected_date, market=market_code, limit=200)
    
    # 显示统计摘要
    st.markdown("---")
    st.markdown("### 📊 对比统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Baseline 信号数", comparison['baseline_count'])
    with col2:
        st.metric("当前方法信号数", comparison['regular_count'])
    with col3:
        st.metric("共同发现", len(comparison['both']))
    with col4:
        overlap = 0
        if comparison['baseline_count'] > 0:
            overlap = len(comparison['both']) / comparison['baseline_count'] * 100
        st.metric("重叠率", f"{overlap:.0f}%")
    
    # 三列布局显示差异
    st.markdown("---")
    st.markdown("### 📋 详细对比")
    
    tab1, tab2, tab3 = st.tabs(["🟢 共同发现", "🔵 仅 Baseline", "🟠 仅当前方法"])
    
    with tab1:
        if comparison['both']:
            st.success(f"两种方法共同发现 {len(comparison['both'])} 只股票")
            st.write(", ".join(comparison['both'][:50]))
        else:
            st.info("没有共同发现的股票")
    
    with tab2:
        if comparison['baseline_only']:
            st.info(f"Baseline 独有 {len(comparison['baseline_only'])} 只股票（当前方法未发现）")
            st.write(", ".join(comparison['baseline_only'][:50]))
        else:
            st.success("Baseline 没有独有的发现")
    
    with tab3:
        if comparison['regular_only']:
            st.info(f"当前方法独有 {len(comparison['regular_only'])} 只股票（Baseline 未发现）")
            st.write(", ".join(comparison['regular_only'][:50]))
        else:
            st.success("当前方法没有独有的发现")
    
    # Baseline 详细结果
    if baseline_results:
        st.markdown("---")
        st.markdown("### 📈 Baseline 详细结果")
        
        df = pd.DataFrame(baseline_results)
        display_cols = ['symbol', 'company_name', 'price', 'latest_day_blue', 'latest_week_blue', 'scan_time']
        available_cols = [c for c in display_cols if c in df.columns]
        
        if available_cols:
            display_df = df[available_cols].copy()
            display_df.columns = ['代码', '名称', '价格', 'Day BLUE', 'Week BLUE', '扫描时段'][:len(available_cols)]
            st.dataframe(display_df, hide_index=True, use_container_width=True)


# --- ML Lab 页面 (新增) ---

def render_ml_lab_page():
    """机器学习实验室 - 统计ML、深度学习、LLM"""
    st.header("🤖 ML 实验室 (Machine Learning Lab)")
    
    # 检查依赖
    from ml.statistical_models import check_ml_dependencies, get_available_models
    deps = check_ml_dependencies()
    
    # 显示依赖状态
    with st.expander("📦 ML 依赖状态", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if deps['sklearn'] else "❌"
            st.write(f"{status} scikit-learn")
        with col2:
            status = "✅" if deps['xgboost'] else "❌"
            st.write(f"{status} XGBoost")
        with col3:
            status = "✅" if deps['lightgbm'] else "❌"
            st.write(f"{status} LightGBM")
        
        if not all(deps.values()):
            st.code("pip install scikit-learn xgboost lightgbm", language="bash")
    
    # 三个 Tab
    tab1, tab2, tab3 = st.tabs(["📊 统计ML", "🧠 深度学习", "💬 LLM智能"])
    
    with tab1:
        st.subheader("统计机器学习")
        st.info("使用 XGBoost, LightGBM, Random Forest 等模型预测信号成功率")
        
        available_models = get_available_models()
        if not available_models:
            st.error("未安装任何 ML 依赖。请运行: `pip install scikit-learn xgboost lightgbm`")
            return
        
        # 参数设置
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox("选择模型", available_models, help="XGBoost 通常表现最好")
        with col2:
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
        with col3:
            forward_days = st.selectbox("目标收益周期", [5, 10, 20], index=1, help="预测 N 天后的收益")
        
        # 数据范围
        st.markdown("#### 📅 训练数据范围")
        col4, col5, col6 = st.columns(3)
        with col4:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            train_start = st.date_input("开始日期", value=start_date)
        with col5:
            train_end = st.date_input("结束日期", value=end_date)
        with col6:
            min_blue = st.slider("最低 BLUE 阈值", 50, 150, 80, 10)
        
        # 训练按钮
        if st.button("🚀 开始训练", type="primary", use_container_width=True):
            with st.spinner("正在准备数据并训练模型..."):
                try:
                    # 1. 优先从缓存加载数据
                    from db.database import query_signal_performance, get_performance_stats
                    from ml.feature_engineering import prepare_training_data
                    from ml.statistical_models import SignalClassifier
                    
                    st.text("📊 正在从缓存加载历史信号数据...")
                    
                    # 尝试从缓存读取
                    cached_data = query_signal_performance(
                        start_date=train_start.strftime('%Y-%m-%d'),
                        end_date=train_end.strftime('%Y-%m-%d'),
                        market='US',
                        limit=1000
                    )
                    
                    if len(cached_data) >= 30:
                        st.text(f"✅ 从缓存加载了 {len(cached_data)} 条性能数据")
                        
                        # 转换为训练格式
                        ret_col = f'return_{forward_days}d'
                        valid_data = [d for d in cached_data if d.get(ret_col) is not None and d.get('blue_daily') is not None]
                        
                        import pandas as pd
                        X = pd.DataFrame([{
                            'blue_daily': d.get('blue_daily', 0),
                            'price': d.get('price', 0),
                        } for d in valid_data])
                        
                        y = pd.Series([1 if d[ret_col] > 0 else 0 for d in valid_data])
                        
                    else:
                        st.warning(f"⚠️ 缓存数据不足 ({len(cached_data)} 条)，尝试实时计算...")
                        
                        # 回退到实时计算
                        from services.backtest_service import run_signal_backtest
                        
                        result = run_signal_backtest(
                            start_date=train_start.strftime('%Y-%m-%d'),
                            end_date=train_end.strftime('%Y-%m-%d'),
                            market='US',
                            min_blue=min_blue,
                            forward_days=forward_days,
                            limit=500
                        )
                        
                        signals = result.get('signals', [])
                        if len(signals) < 30:
                            st.error(f"❌ 数据不足！仅找到 {len(signals)} 个信号")
                            st.info("💡 运行: `python scripts/compute_performance.py --limit 200` 预计算性能数据")
                            return
                        
                        X, y = prepare_training_data(signals, forward_days, 'binary')
                    
                    if X.empty or len(y) < 30:
                        st.error("❌ 特征准备失败，可能是收益数据不足")
                        return
                    
                    st.text(f"✅ 特征矩阵: {X.shape[0]} 样本, {X.shape[1]} 特征")
                    
                    # 3. 训练模型
                    st.text(f"🧠 正在训练 {model_type} 模型...")
                    classifier = SignalClassifier(model_type=model_type)
                    metrics = classifier.train(X, y, test_size=test_size)
                    
                    st.success("✅ 模型训练完成!")
                    
                    # 4. 显示结果
                    st.markdown("---")
                    st.subheader("📈 模型性能")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        acc = metrics.get('accuracy', 0) * 100
                        st.metric("准确率 (Accuracy)", f"{acc:.1f}%", 
                                 delta="好" if acc > 55 else "需改进")
                    with m2:
                        prec = metrics.get('precision', 0) * 100
                        st.metric("精确率 (Precision)", f"{prec:.1f}%")
                    with m3:
                        rec = metrics.get('recall', 0) * 100
                        st.metric("召回率 (Recall)", f"{rec:.1f}%")
                    with m4:
                        f1 = metrics.get('f1', 0) * 100
                        st.metric("F1 Score", f"{f1:.1f}%")
                    
                    # 5. 特征重要性
                    importance_df = classifier.get_feature_importance_df()
                    if not importance_df.empty:
                        st.markdown("---")
                        st.subheader("📊 特征重要性")
                        
                        import plotly.express as px
                        fig = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Feature Importance",
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 6. 模型解释
                    st.markdown("---")
                    st.subheader("💡 模型解读")
                    
                    if acc > 55:
                        st.success(f"""
                        **模型表现良好!** 准确率 {acc:.1f}% 高于随机猜测 (50%)。
                        
                        - 该模型可以作为信号筛选的辅助参考
                        - 高 BLUE 值的信号有更高的盈利概率
                        - 建议结合其他技术指标使用
                        """)
                    else:
                        st.warning(f"""
                        **模型准确率较低** ({acc:.1f}%)，可能原因：
                        
                        - 训练数据量不足 (当前: {len(signals)} 个信号)
                        - 特征与目标的相关性不强
                        - 市场噪音较大，难以预测
                        
                        💡 **建议**: 积累更多历史数据后重新训练
                        """)
                    
                except ImportError as e:
                    st.error(f"❌ 缺少依赖: {e}")
                    st.code("pip install scikit-learn xgboost lightgbm", language="bash")
                except Exception as e:
                    st.error(f"❌ 训练出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("深度学习 🧠")
        st.info("使用 LSTM/GRU 时间序列模型进行价格预测")
        
        from ml.deep_learning import check_torch_available
        
        if not check_torch_available():
            st.error("❌ PyTorch 未安装")
            st.code("pip install torch", language="bash")
            return
        
        st.success("✅ PyTorch 已安装")
        
        # 参数设置
        col1, col2, col3 = st.columns(3)
        with col1:
            dl_symbol = st.text_input("股票代码", value="AAPL", help="例如: AAPL, NVDA, TSLA")
        with col2:
            dl_model = st.selectbox("模型类型", ["LSTM", "GRU"], help="LSTM 更稳定, GRU 更快")
        with col3:
            dl_days = st.slider("训练数据天数", 50, 200, 100, 10)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            seq_length = st.slider("序列长度", 10, 50, 20, 5, help="回看多少天预测未来")
        with col5:
            dl_epochs = st.slider("训练轮数", 20, 200, 50, 10)
        with col6:
            hidden_size = st.selectbox("隐藏层大小", [32, 64, 128], index=1)
        
        if st.button("🚀 开始训练", type="primary", key="dl_train"):
            with st.spinner(f"正在训练 {dl_model} 模型..."):
                try:
                    from ml.deep_learning import train_price_predictor
                    
                    result = train_price_predictor(
                        symbol=dl_symbol.upper(),
                        days=dl_days,
                        seq_length=seq_length,
                        epochs=dl_epochs,
                        model_type=dl_model
                    )
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                        return
                    
                    st.success("✅ 训练完成!")
                    
                    # 显示指标
                    st.markdown("---")
                    st.subheader("📈 预测性能")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("MAE (平均绝对误差)", f"${result['mae']:.2f}")
                    with m2:
                        st.metric("RMSE (均方根误差)", f"${result['rmse']:.2f}")
                    with m3:
                        acc = result['direction_accuracy'] * 100
                        st.metric("方向准确率", f"{acc:.1f}%", 
                                 delta="好" if acc > 55 else "待改进")
                    with m4:
                        st.metric("验证损失", f"{result['val_loss']:.6f}")
                    
                    # 训练曲线
                    st.markdown("---")
                    st.subheader("📉 训练损失曲线")
                    
                    chart_data = result.get('chart_data', {})
                    if chart_data:
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=chart_data['epochs'],
                            y=chart_data['train_loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#58a6ff', width=2)
                        ))
                        if chart_data.get('val_loss'):
                            fig.add_trace(go.Scatter(
                                x=chart_data['epochs'],
                                y=chart_data['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='#f0883e', width=2, dash='dot')
                            ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=300,
                            xaxis_title="Epoch",
                            yaxis_title="Loss (MSE)",
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 预测 vs 实际
                    st.markdown("---")
                    st.subheader("🎯 预测 vs 实际 (最近10天)")
                    
                    pred_df = pd.DataFrame({
                        '实际价格': result.get('actuals', []),
                        '预测价格': result.get('predictions', [])
                    })
                    st.dataframe(pred_df.style.format("${:.2f}"), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 训练出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab3:
        st.subheader("LLM 智能分析 💬")
        st.info("使用大语言模型进行市场分析和自然语言查询")
        
        from ml.llm_intelligence import check_llm_available, LLMAnalyzer
        
        # 检查 API 状态
        llm_status = check_llm_available()
        
        col1, col2 = st.columns(2)
        with col1:
            status = "✅" if llm_status['openai'] else "❌"
            st.write(f"{status} OpenAI SDK")
        with col2:
            status = "✅" if llm_status['anthropic'] else "❌"
            st.write(f"{status} Anthropic SDK")
        
        # API Key 状态
        openai_key = os.environ.get('OPENAI_API_KEY', '')
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
        
        if not openai_key and not anthropic_key:
            st.warning("⚠️ 未配置 API Key。请设置 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY` 环境变量。")
            st.code("export OPENAI_API_KEY='your-api-key'", language="bash")
            
            # 允许临时输入
            with st.expander("🔑 临时输入 API Key"):
                temp_key = st.text_input("OpenAI API Key", type="password", key="temp_openai")
                if temp_key:
                    os.environ['OPENAI_API_KEY'] = temp_key
                    st.success("✅ API Key 已设置 (仅本次会话有效)")
                    st.rerun()
            return
        
        # 选择提供商
        provider = "openai" if openai_key else "anthropic"
        st.success(f"✅ 已配置 {provider.upper()} API")
        
        # 三个子功能
        llm_tab1, llm_tab2, llm_tab3 = st.tabs(["💬 AI 问答", "📊 情感分析", "📝 市场报告"])
        
        with llm_tab1:
            st.markdown("### 💬 AI 问答助手")
            st.caption("问我任何关于量化交易、技术指标的问题")
            
            # 聊天历史
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # 显示历史
            for msg in st.session_state.chat_history[-6:]:  # 最近 6 条
                with st.chat_message(msg['role']):
                    st.write(msg['content'])
            
            # 用户输入
            user_input = st.chat_input("输入你的问题...")
            
            if user_input:
                # 添加用户消息
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                # AI 回复
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        analyzer = LLMAnalyzer(provider)
                        response = analyzer.natural_query(user_input)
                        st.write(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with llm_tab2:
            st.markdown("### 📊 新闻情感分析")
            st.caption("分析财经新闻或社交媒体情感")
            
            sample_text = st.text_area(
                "输入文本",
                placeholder="粘贴新闻标题、推文或财经评论...",
                height=100
            )
            
            if st.button("🔍 分析情感", key="sentiment_btn"):
                if sample_text:
                    with st.spinner("分析中..."):
                        analyzer = LLMAnalyzer(provider)
                        result = analyzer.analyze_sentiment(sample_text)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            # 显示结果
                            sentiment = result.get('sentiment', 'neutral')
                            confidence = result.get('confidence', 0)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                emoji = "🟢" if sentiment == "bullish" else ("🔴" if sentiment == "bearish" else "⚪")
                                st.metric("情感", f"{emoji} {sentiment.upper()}")
                            with col2:
                                st.metric("置信度", f"{confidence:.0%}")
                            
                            st.markdown("**要点:**")
                            for point in result.get('key_points', []):
                                st.write(f"• {point}")
                            
                            st.markdown(f"**分析:** {result.get('reasoning', '')}")
                else:
                    st.warning("请输入文本")
        
        with llm_tab3:
            st.markdown("### 📝 AI 市场报告")
            st.caption("基于当日信号自动生成市场分析报告")
            
            if st.button("📄 生成报告", key="report_btn"):
                with st.spinner("正在生成报告..."):
                    # 获取今日信号
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    signals = query_scan_results(scan_date=today, market='US', limit=20)
                    
                    analyzer = LLMAnalyzer(provider)
                    report = analyzer.generate_market_report(signals)
                    
                    st.markdown(report)


# --- 主导航 ---

st.sidebar.title("Coral Creek 🦅")
page = st.sidebar.radio("功能导航", [
    "📊 每日机会扫描", 
    "🔍 个股查询", 
    "📈 信号追踪",
    "📉 信号验证",
    "🤖 ML实验室",  # 新增
    "🔄 Baseline对比", 
    "🧪 策略回测实验"
])

if page == "📊 每日机会扫描":
    render_scan_page()
elif page == "🔍 个股查询":
    render_stock_lookup_page()
elif page == "📈 信号追踪":
    render_signal_tracker_page()
elif page == "📉 信号验证":
    render_signal_performance_page()
elif page == "🤖 ML实验室":
    render_ml_lab_page()
elif page == "🔄 Baseline对比":
    render_baseline_comparison_page()
elif page == "🧪 策略回测实验":
    render_backtest_page()


