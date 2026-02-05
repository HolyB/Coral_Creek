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
from data_fetcher import get_us_stock_data as fetch_data_from_polygon, get_ticker_details, get_stock_data, get_cn_stock_data
from components.stock_detail import render_unified_stock_detail
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_adx_series
from backtester import SimpleBacktester
from db.database import (
    query_scan_results, get_scanned_dates, get_db_stats, 
    get_stock_history, init_db, get_scan_job, get_stock_info_batch,
    get_first_scan_dates
)

# 设置页面配置
st.set_page_config(
    page_title="Coral Creek V3.0 - 智能量化系统",
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

# --- 环境变量适配 ---
# 将 Streamlit Secrets 注入环境变量
def inject_secrets():
    """将 Streamlit Secrets 注入到环境变量"""
    try:
        if hasattr(st, "secrets"):
            # 遍历所有 secrets
            for key in st.secrets:
                value = st.secrets[key]
                # 只注入字符串值
                if isinstance(value, str):
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value
                        print(f"✅ Injected secret: {key}")
            
            # 特别检查 Supabase
            if 'SUPABASE_URL' in os.environ:
                print(f"✅ SUPABASE_URL: {os.environ['SUPABASE_URL'][:30]}...")
            else:
                print("⚠️ SUPABASE_URL not found in secrets")
    except Exception as e:
        print(f"⚠️ Secrets injection error: {e}")

inject_secrets()


# --- 后台调度器 (In-App Scheduler) ---
# 替代 GitHub Actions，直接在应用内运行监控
# 避免支付问题和数据同步问题

@st.cache_resource
def init_scheduler():
    """初始化并启动后台调度器 (单例模式)"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from scripts.intraday_monitor import monitor_portfolio
        import atexit
        
        # 创建调度器
        scheduler = BackgroundScheduler()
        
        # 防止重复添加
        if scheduler.get_job('intraday_monitor_job'):
            return scheduler
        
        # 定义任务
        def job_function():
            from datetime import datetime
            print(f"📱 盘中监控 - {datetime.now()}")
            try:
                # 运行美股扫描
                monitor_portfolio(market='US', run_once=True)
                # 运行A股扫描 (如果是在交易时段)
                monitor_portfolio(market='CN', run_once=True)
            except Exception as e:
                print(f"⚠️ [Scheduler] Job failed: {e}")
        
        # 添加任务 (每30分钟)
        scheduler.add_job(
            job_function,
            IntervalTrigger(minutes=30),
            id='intraday_monitor_job',
            replace_existing=True,
            name='Intraday Monitor (Every 30min)'
        )
        
        # 启动
        scheduler.start()
        print("✅ [Scheduler] Background scheduler started (Interval: 30min)")
        
        # 退出时关闭
        atexit.register(lambda: scheduler.shutdown())
        
        return scheduler
    except ImportError:
        print("⚠️ [Scheduler] APScheduler not installed. Skipping.")
        return None
    except Exception as e:
        print(f"⚠️ [Scheduler] Failed to start: {e}")
        return None

# 启动调度器
init_scheduler()

# --- 登录验证 ---

def check_password():
    """角色验证 - Admin 可管理持仓，Guest 只能查看"""
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None
    
    if st.session_state["user_role"] is None:
        st.markdown("## 🦅 Coral Creek V3.0")
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

# --- 侧边栏: 系统状态与测试 ---
with st.sidebar:
    st.markdown("---")
    st.caption("🔧 系统工具")
    if st.button("🔔 发送测试通知", help="点击此按钮测试 Telegram 连接"):
        from scripts.intraday_monitor import send_alert_telegram
        with st.spinner("正在发送测试消息..."):
            success = send_alert_telegram([{
                'type': 'test',
                'level': '🔔',
                'symbol': '从网站发出',
                'message': '这是一条测试消息',
                'footer': '如果您收到此消息，说明网站监控功能正常。'
            }])
            if success:
                st.toast("✅ 测试消息发送成功!", icon="✅")
            else:
                st.error("❌ 发送失败，请检查 Logs")
    
    # Supabase 调试
    if st.button("🔍 检查数据库", help="检查 Supabase 连接和数据"):
        st.write("**环境变量检查:**")
        supabase_url = os.environ.get('SUPABASE_URL', 'NOT SET')
        supabase_key = os.environ.get('SUPABASE_KEY', 'NOT SET')
        st.write(f"- SUPABASE_URL: `{supabase_url[:40] if supabase_url else 'None'}...`")
        st.write(f"- SUPABASE_KEY: `{'SET' if supabase_key and len(supabase_key) > 10 else 'NOT SET'}`")
        
        # 测试连接
        try:
            from db.supabase_db import get_supabase, is_supabase_available
            if is_supabase_available():
                supabase = get_supabase()
                result = supabase.table('scan_results').select('*').limit(5).execute()
                st.success(f"✅ Supabase 连接成功! 获取到 {len(result.data)} 条记录")
                if result.data:
                    # 检查 heima 列是否存在
                    cols = list(result.data[0].keys())
                    heima_cols = [c for c in cols if 'heima' in c.lower()]
                    st.write(f"**heima 相关列**: {heima_cols if heima_cols else '❌ 无'}")
                    st.json(result.data[0])
            else:
                st.error("❌ Supabase 不可用")
        except Exception as e:
            st.error(f"❌ 连接错误: {e}")
    
    # 修复 Supabase 表结构
    if st.button("🔧 修复黑马列", help="添加缺失的 heima_daily/weekly/monthly 列"):
        try:
            from db.supabase_db import get_supabase, is_supabase_available
            if is_supabase_available():
                supabase = get_supabase()
                
                # 检查是否需要添加列
                result = supabase.table('scan_results').select('*').limit(1).execute()
                if result.data:
                    existing_cols = set(result.data[0].keys())
                    needed_cols = ['heima_daily', 'heima_weekly', 'heima_monthly', 
                                   'juedi_daily', 'juedi_weekly', 'juedi_monthly']
                    missing_cols = [c for c in needed_cols if c not in existing_cols]
                    
                    if not missing_cols:
                        st.success("✅ 所有 heima 列已存在，无需修复")
                    else:
                        st.warning(f"缺失列: {missing_cols}")
                        st.info("""
请在 Supabase SQL Editor 中运行:
```sql
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS heima_daily BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS heima_weekly BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS heima_monthly BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS juedi_daily BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS juedi_weekly BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS juedi_monthly BOOLEAN DEFAULT FALSE;
```
                        """)
                else:
                    st.warning("表为空，请先运行扫描")
            else:
                st.error("❌ Supabase 不可用")
        except Exception as e:
            st.error(f"❌ 错误: {e}")

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
            'is_juedi': 'Is_Juedi',
            'heima_daily': 'Heima_Daily',
            'heima_weekly': 'Heima_Weekly',
            'heima_monthly': 'Heima_Monthly',
            'juedi_daily': 'Juedi_Daily',
            'juedi_weekly': 'Juedi_Weekly',
            'juedi_monthly': 'Juedi_Monthly',
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
        
        # 转换布尔字段 (SQLite=bytes, Supabase=bool/str)
        def robust_bool_convert(x):
            """健壮的布尔转换，处理所有可能的数据来源"""
            if x is None:
                return False
            if isinstance(x, bool):
                return x
            if isinstance(x, bytes):
                return x == b'\x01'
            if isinstance(x, (int, float)):
                return x == 1
            if isinstance(x, str):
                return x.lower() in ('true', '1', 't', 'yes')
            return False
        
        bool_cols = ['Is_Heima', 'Is_Juedi', 'Heima_Daily', 'Heima_Weekly', 'Heima_Monthly', 
                     'Juedi_Daily', 'Juedi_Weekly', 'Juedi_Monthly', 'Strat_D_Trend', 'Strat_C_Resonance']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].apply(robust_bool_convert)
        
        # 格式化市值
        if 'Mkt Cap Raw' in df.columns:
            df['Mkt Cap'] = pd.to_numeric(df['Mkt Cap Raw'], errors='coerce').fillna(0) / 1_000_000_000
        else:
            df['Mkt Cap'] = 0.0

        # [补丁] A股数据如果市值为0，尝试用 AkShare/yfinance 实时数据补全
        if market == 'CN' and (df['Mkt Cap'] == 0).mean() > 0.5:
            try:
                import streamlit as st
                
                # 缓存一下，避免每次rerun都拉取全市场
                cache_key = f"cn_mkt_cap_{datetime.now().strftime('%Y%m%d_%H')}"
                
                if cache_key in st.session_state:
                    mkt_map = st.session_state[cache_key]
                else:
                    mkt_map = {}
                    
                    # 方法1: 尝试 AkShare
                    try:
                        import akshare as ak
                        spot_df = ak.stock_zh_a_spot_em()
                        mkt_map = dict(zip(spot_df['代码'], spot_df['总市值']))
                    except Exception as e1:
                        print(f"AkShare failed: {e1}")
                        
                        # 方法2: 尝试 yfinance 批量获取 (只取前30个)
                        try:
                            import yfinance as yf
                            tickers = df['Ticker'].head(30).tolist()
                            yf_symbols = []
                            for t in tickers:
                                code = t.split('.')[0]
                                suffix = '.SS' if t.endswith('.SH') else '.SZ'
                                yf_symbols.append(code + suffix)
                            
                            objs = yf.Tickers(' '.join(yf_symbols))
                            for t, yf_t in zip(tickers, yf_symbols):
                                try:
                                    code = t.split('.')[0]
                                    mc = objs.tickers[yf_t].fast_info.get('marketCap', 0)
                                    if mc:
                                        mkt_map[code] = mc
                                except:
                                    pass
                        except Exception as e2:
                            print(f"yfinance CN failed: {e2}")
                    
                    st.session_state[cache_key] = mkt_map
                
                if mkt_map:
                    def fill_cn_cap(row):
                        if row['Mkt Cap'] > 0: 
                            return row['Mkt Cap']
                        code = row['Ticker'].split('.')[0]
                        cap = mkt_map.get(code, 0)
                        if cap and cap > 0:
                            return cap / 1_000_000_000
                        return 0
                    
                    df['Mkt Cap'] = df.apply(fill_cn_cap, axis=1)
                    
                    # 重新计算 Cap Category
                    def update_category(cap):
                        if cap >= 200: return 'Mega-Cap (超大盘)'
                        elif cap >= 10: return 'Large-Cap (大盘)'
                        elif cap >= 2: return 'Mid-Cap (中盘)'
                        elif cap >= 0.3: return 'Small-Cap (小盘)'
                        return 'Micro-Cap (微盘)'
                    
                    df['Cap_Category'] = df['Mkt Cap'].apply(update_category)
                
            except Exception as e:
                print(f"CN market cap fix failed: {e}")
        
        # [补丁] 美股数据如果市值为0，尝试用 yfinance 和 Polygon 补全
        if market == 'US' and (df['Mkt Cap'] == 0).mean() > 0.5:
            try:
                # 只修复前 30 个，避免加载太慢
                tickers_to_fix = df[df['Mkt Cap'] == 0]['Ticker'].tolist()[:30]
                
                if tickers_to_fix:
                    @st.cache_data(ttl=3600, show_spinner=False)
                    def fetch_us_caps_cached(tickers):
                        caps = {}
                        # 1. 尝试 Yahoo Finance
                        try:
                            import yfinance as yf
                            txt = " ".join(tickers)
                            objs = yf.Tickers(txt)
                            for t in tickers:
                                try:
                                    val = objs.tickers[t].fast_info.market_cap
                                    if val: caps[t] = val / 1_000_000_000
                                except: pass
                        except Exception as ye:
                             print(f"YF Error: {ye}")
                        
                        # 2. 尝试 Polygon (作为补充)
                        try:
                            from data_fetcher import get_ticker_details
                            import time
                            # 只对还没拿到的尝试，且限制数量防止超时
                            missing = [t for t in tickers if t not in caps][:10]
                            for t in missing:
                                try:
                                    det = get_ticker_details(t)
                                    if det and det.get('market_cap'):
                                        caps[t] = det.get('market_cap') / 1_000_000_000
                                    time.sleep(0.25) # 避免限流 (5 calls/min limit for free tier)
                                except: pass
                        except: pass
                        
                        return caps

                    caps_map = fetch_us_caps_cached(tickers_to_fix)
                    
                    def fill_us_cap(row):
                         if row['Mkt Cap'] > 0: return row['Mkt Cap']
                         return caps_map.get(row['Ticker'], 0)
                    
                    df['Mkt Cap'] = df.apply(fill_us_cap, axis=1)

                    def update_category_us(cap):
                        if cap == 0: return 'Unknown'
                        if cap >= 200: return 'Mega-Cap (超大盘)'
                        elif cap >= 10: return 'Large-Cap (大盘)'
                        elif cap >= 2: return 'Mid-Cap (中盘)'
                        elif cap >= 0.3: return 'Small-Cap (小盘)'
                        return 'Micro-Cap (微盘)'
                    df['Cap_Category'] = df['Mkt Cap'].apply(update_category_us)
            except Exception as e:
                print(f"US Cap fix failed: {e}")
        
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
        
        # === 北向资金 (仅 A股) ===
        if market == 'CN':
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            
            # 尝试从缓存获取北向资金数据
            north_cache_key = f"north_money_{cache_time_key}"
            
            if north_cache_key not in st.session_state:
                try:
                    from data_fetcher import get_north_money_today
                    north_data = get_north_money_today()
                    st.session_state[north_cache_key] = north_data
                except Exception as e:
                    st.session_state[north_cache_key] = {}
            else:
                north_data = st.session_state[north_cache_key]
            
            if north_data:
                north_cols = st.columns(4)
                
                with north_cols[0]:
                    north_val = north_data.get('north_money', 0)
                    color = "#3fb950" if north_val > 0 else "#f85149"
                    icon = "📈" if north_val > 0 else "📉"
                    st.metric(
                        label=f"🏦 北向资金 {icon}",
                        value=f"¥{abs(north_val):.2f}亿",
                        delta=f"{'净流入' if north_val > 0 else '净流出'}",
                        delta_color="normal" if north_val > 0 else "inverse"
                    )
                
                with north_cols[1]:
                    sh_val = north_data.get('sh_money', 0)
                    st.metric(
                        label="沪股通",
                        value=f"¥{abs(sh_val):.2f}亿",
                        delta=f"{'流入' if sh_val > 0 else '流出'}",
                        delta_color="normal" if sh_val > 0 else "inverse"
                    )
                
                with north_cols[2]:
                    sz_val = north_data.get('sz_money', 0)
                    st.metric(
                        label="深股通",
                        value=f"¥{abs(sz_val):.2f}亿",
                        delta=f"{'流入' if sz_val > 0 else '流出'}",
                        delta_color="normal" if sz_val > 0 else "inverse"
                    )
                
                with north_cols[3]:
                    st.caption(f"📅 {north_data.get('date', '--')}")
                    # 北向资金判断
                    if north_val > 50:
                        st.markdown("🟢 **大幅流入**")
                    elif north_val > 0:
                        st.markdown("🟡 **小幅流入**")
                    elif north_val > -50:
                        st.markdown("🟠 **小幅流出**")
                    else:
                        st.markdown("🔴 **大幅流出**")
        
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

def render_todays_picks_page():
    """🎯 每日工作台 - 20年交易员的每日工作流"""
    st.header("🎯 每日工作台")
    st.caption("开盘前准备 → 盘中执行 → 收盘复盘 | 一站式管理你的交易")
    
    # 导入模块
    try:
        from strategies.decision_system import get_strategy_manager
        from strategies.performance_tracker import get_all_strategy_performance
    except ImportError as e:
        st.error(f"策略模块导入失败: {e}")
        return
    
    from db.database import query_scan_results, get_scanned_dates, get_stock_info_batch
    from services.portfolio_service import get_portfolio_summary
    
    # 尝试导入工作流服务
    try:
        from services.daily_workflow import (
            get_workflow_service, get_today_tasks, 
            get_signal_pipeline, get_daily_summary
        )
        workflow_available = True
    except ImportError:
        workflow_available = False
    
    # 侧边栏: 设置
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ 工作台设置")
        
        market_choice = st.radio("市场", ["🇺🇸 美股", "🇨🇳 A股"], horizontal=True, key="picks_market")
        market = "US" if "美股" in market_choice else "CN"
        
        # 检测市场切换，清除之前选中的股票
        prev_market = st.session_state.get('_picks_prev_market', market)
        if prev_market != market:
            # 清除所有选中状态
            for key in ['action_selected_symbol', 'action_buy_symbol', 'discover_selected', 
                       'portfolio_selected', 'portfolio_sell', 'portfolio_add']:
                if key in st.session_state:
                    st.session_state[key] = None
            st.session_state['_picks_prev_market'] = market
        else:
            st.session_state['_picks_prev_market'] = market
        
        top_n = st.slider("每策略选股数", 3, 10, 5, key="picks_topn")
        
        show_performance = st.checkbox("显示策略历史表现", value=True, key="picks_perf")
        show_backtest = st.checkbox("显示回测追踪", value=False, key="picks_backtest")
    
    # ============================================
    # 📊 顶部: 行动摘要卡片
    # ============================================
    dates = get_scanned_dates(market=market)
    if not dates:
        st.warning(f"暂无 {market} 市场数据")
        return
    
    latest_date = dates[0]
    results = query_scan_results(scan_date=latest_date, market=market, limit=500)
    df = pd.DataFrame(results) if results else pd.DataFrame()
    
    # 获取持仓数据
    try:
        portfolio = get_portfolio_summary() or {}
        positions = portfolio.get('details', [])  # 从 summary 中获取持仓详情
    except:
        positions = []
        portfolio = {}
    
    # 计算行动项
    buy_opportunities = 0
    sell_signals = 0
    risk_alerts = 0
    
    if not df.empty:
        # 强买入信号: 日BLUE > 100 且 周BLUE > 50
        strong_buy = df[
            (df.get('blue_daily', pd.Series([0]*len(df))) > 100) & 
            (df.get('blue_weekly', pd.Series([0]*len(df))) > 50)
        ]
        buy_opportunities = len(strong_buy)
    
    # 检测持仓卖出信号
    position_alerts = []
    for pos in positions:
        symbol = pos.get('symbol', '')
        avg_cost = pos.get('avg_cost', 0)
        current_price = pos.get('current_price', 0)
        stop_loss = pos.get('stop_loss', avg_cost * 0.92)  # 默认8%止损
        
        if current_price > 0:
            pnl_pct = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
            
            # 检查止损
            if current_price < stop_loss:
                position_alerts.append({
                    'symbol': symbol,
                    'type': 'stop_loss',
                    'message': f'触及止损 ${current_price:.2f} < ${stop_loss:.2f}',
                    'action': '建议卖出',
                    'urgency': 'high'
                })
                sell_signals += 1
            
            # 检查大幅亏损
            elif pnl_pct < -10:
                position_alerts.append({
                    'symbol': symbol,
                    'type': 'loss',
                    'message': f'亏损 {pnl_pct:.1f}%',
                    'action': '检查止损',
                    'urgency': 'medium'
                })
                risk_alerts += 1
            
            # 检查是否有卖出信号 (BLUE 转弱)
            if not df.empty and symbol in df['symbol'].values:
                stock_data = df[df['symbol'] == symbol].iloc[0]
                day_blue = stock_data.get('blue_daily', 100)
                if day_blue < 30 and pnl_pct > 5:
                    position_alerts.append({
                        'symbol': symbol,
                        'type': 'signal_weak',
                        'message': f'BLUE信号转弱 ({day_blue:.0f}), 盈利 {pnl_pct:.1f}%',
                        'action': '考虑获利了结',
                        'urgency': 'low'
                    })
    
    # 行动摘要卡片
    st.markdown(f"### 📅 {latest_date} 行动摘要")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🟢 买入机会", 
            f"{buy_opportunities} 只",
            help="日BLUE>100 且 周BLUE>50 的强信号"
        )
    
    with col2:
        delta_color = "inverse" if sell_signals > 0 else "off"
        st.metric(
            "🔴 卖出信号", 
            f"{sell_signals} 只",
            delta="需要行动" if sell_signals > 0 else None,
            delta_color=delta_color
        )
    
    with col3:
        st.metric(
            "⚠️ 风险警告", 
            f"{risk_alerts} 只",
            delta="注意" if risk_alerts > 0 else None,
            delta_color="inverse" if risk_alerts > 0 else "off"
        )
    
    with col4:
        total_positions = len(positions)
        total_pnl = portfolio.get('total_pnl_pct', 0)
        st.metric(
            "💼 持仓", 
            f"{total_positions} 只",
            delta=f"{total_pnl:+.1f}%" if total_positions > 0 else None,
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )
    
    st.divider()
    
    # ============================================
    # 📋 核心工作区 (Tabs) - 重新设计的用户体验
    # ============================================
    work_tab1, work_tab2, work_tab3, work_tab4 = st.tabs([
        "⚡ 今日行动",
        "🔎 发现新股", 
        "🎯 策略精选",
        "💼 我的持仓"
    ])
    
    # === Tab 1: 今日行动 (重新设计 - 行动导向) ===
    with work_tab1:
        # 如果有紧急警报，用红色卡片突出显示
        if position_alerts:
            high_alerts = [a for a in position_alerts if a['urgency'] == 'high']
            if high_alerts:
                st.error(f"🚨 **紧急**: {len(high_alerts)}只股票需要立即处理!")
                
                for alert in high_alerts:
                    with st.container():
                        c1, c2, c3 = st.columns([2, 5, 2])
                        with c1:
                            st.markdown(f"### {alert['symbol']}")
                        with c2:
                            st.warning(f"⚠️ {alert['message']}")
                        with c3:
                            if st.button(f"🔴 {alert['action']}", key=f"urgent_{alert['symbol']}", type="primary"):
                                st.session_state[f"show_detail_{alert['symbol']}"] = True
                        
                        # 点击后显示详情
                        if st.session_state.get(f"show_detail_{alert['symbol']}"):
                            render_unified_stock_detail(
                                symbol=alert['symbol'],
                                market=market,
                                show_charts=True,
                                show_chips=False,
                                show_news=False,
                                key_prefix=f"urgent_{alert['symbol']}"
                            )
                
                st.divider()
        
        # 两列布局：左边买入机会，右边其他任务
        action_left, action_right = st.columns([1, 1])
        
        with action_left:
            st.markdown("### 🟢 今日买入机会")
            
            # 获取强势信号
            if not df.empty and 'blue_daily' in df.columns:
                strong = df[
                    (df['blue_daily'].fillna(0) > 100) & 
                    (df['blue_weekly'].fillna(0) > 50)
                ].head(5)
                
                if not strong.empty:
                    for idx, row in strong.iterrows():
                        symbol = row.get('symbol', '')
                        company_name = row.get('company_name', '')
                        blue_d = row.get('blue_daily', 0)
                        blue_w = row.get('blue_weekly', 0)
                        price = row.get('price', 0)
                        
                        # 价格符号和名称显示
                        price_sym = "¥" if market == "CN" else "$"
                        display_name = company_name if company_name else symbol
                        display_code = symbol.split('.')[0] if '.' in symbol else symbol
                        
                        # 卡片式展示
                        with st.container():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a472a22, #1a472a11); 
                                        border-left: 3px solid #00C853; padding: 12px; 
                                        border-radius: 8px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.1em; font-weight: bold;">{display_name}</span>
                                        <span style="font-size: 0.8em; color: #888; margin-left: 4px;">{display_code}</span>
                                    </div>
                                    <span style="color: #00C853;">{price_sym}{price:.2f}</span>
                                </div>
                                <div style="font-size: 0.9em; color: #888; margin-top: 4px;">
                                    日BLUE {blue_d:.0f} | 周BLUE {blue_w:.0f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 操作按钮
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("📊 查看详情", key=f"view_{symbol}", use_container_width=True):
                                    st.session_state['action_selected_symbol'] = symbol
                            with btn_col2:
                                if st.button("💰 模拟买入", key=f"buy_{symbol}", use_container_width=True):
                                    st.session_state['action_buy_symbol'] = symbol
                else:
                    st.info("今日暂无强势买入信号")
            else:
                st.info("正在加载数据...")
        
        with action_right:
            st.markdown("### 📋 其他待办")
            
            # 观察列表提醒
            try:
                from services.signal_tracker import get_watchlist
                watchlist = get_watchlist(market=market)
                
                if watchlist:
                    st.markdown(f"**👁️ {len(watchlist)}只股票在观察中**")
                    for w in watchlist[:3]:
                        symbol = w.get('symbol', '')
                        entry = w.get('entry_price', 0)
                        st.markdown(f"- `{symbol}` 等待入场 @ ${entry:.2f}")
                    
                    if len(watchlist) > 3:
                        st.caption(f"...还有 {len(watchlist) - 3} 只")
            except:
                pass
            
            st.divider()
            
            # 中等优先级警报
            medium_alerts = [a for a in position_alerts if a['urgency'] in ['medium', 'low']]
            if medium_alerts:
                st.markdown("**⚠️ 持仓提醒**")
                for alert in medium_alerts[:3]:
                    icon = '🟡' if alert['urgency'] == 'medium' else '🟢'
                    st.markdown(f"{icon} **{alert['symbol']}**: {alert['message']}")
        
        # 显示选中的股票详情
        if st.session_state.get('action_selected_symbol'):
            st.divider()
            symbol = st.session_state['action_selected_symbol']
            st.markdown(f"### 📊 {symbol} 详细分析")
            
            # 关闭按钮
            if st.button("❌ 关闭详情", key="close_action_detail"):
                st.session_state['action_selected_symbol'] = None
                st.rerun()
            
            render_unified_stock_detail(
                symbol=symbol,
                market=market,
                key_prefix=f"action_{symbol}"
            )
        
        # 处理模拟买入
        if st.session_state.get('action_buy_symbol'):
            symbol = st.session_state['action_buy_symbol']
            st.divider()
            st.markdown(f"### 💰 模拟买入 {symbol}")
            
            with st.form(f"buy_form_{symbol}"):
                price = df[df['symbol'] == symbol]['price'].iloc[0] if symbol in df['symbol'].values else 100
                shares = st.number_input("买入股数", min_value=1, value=max(1, int(1000 / price)))
                stop_loss = st.number_input("止损价", value=price * 0.92)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("✅ 确认买入", type="primary"):
                        try:
                            from services.portfolio_service import paper_buy
                            result = paper_buy(symbol, shares, price, market)
                            if result.get('success'):
                                st.success(f"🎉 买入成功! {symbol} x {shares}股")
                                st.balloons()
                                st.session_state['action_buy_symbol'] = None
                            else:
                                st.error(result.get('error', '买入失败'))
                        except Exception as e:
                            st.error(f"买入失败: {e}")
                with col2:
                    if st.form_submit_button("❌ 取消"):
                        st.session_state['action_buy_symbol'] = None
                        st.rerun()
    
    # === Tab 2: 发现新股 (重新设计 - 卡片式浏览) ===
    with work_tab2:
        st.markdown("### 🔎 发现新股")
        
        # 筛选器（横向排列）
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            signal_filter = st.selectbox(
                "信号类型", 
                ["🔥 全部强信号", "📊 日线BLUE>100", "📈 日周共振", "🚀 日周月共振", "🐴 黑马信号"],
                key="discover_filter"
            )
        with filter_col2:
            sort_by = st.selectbox(
                "排序方式",
                ["日BLUE↓", "周BLUE↓", "月BLUE↓", "价格↓", "ADX↓"],
                key="discover_sort"
            )
        with filter_col3:
            show_count = st.slider("显示数量", 5, 30, 12, key="discover_count")
        
        st.divider()
        
        # 根据筛选条件过滤
        if not df.empty:
            filtered_df = df.copy()
            
            if signal_filter == "📊 日线BLUE>100":
                filtered_df = filtered_df[filtered_df['blue_daily'].fillna(0) > 100]
            elif signal_filter == "📈 日周共振":
                filtered_df = filtered_df[
                    (filtered_df['blue_daily'].fillna(0) > 100) & 
                    (filtered_df['blue_weekly'].fillna(0) > 80)
                ]
            elif signal_filter == "🚀 日周月共振":
                filtered_df = filtered_df[
                    (filtered_df['blue_daily'].fillna(0) > 100) & 
                    (filtered_df['blue_weekly'].fillna(0) > 80) &
                    (filtered_df['blue_monthly'].fillna(0) > 60)
                ]
            elif signal_filter == "🐴 黑马信号":
                # 检查黑马列
                heima_cols = [c for c in filtered_df.columns if 'heima' in c.lower()]
                if heima_cols:
                    heima_mask = filtered_df[heima_cols].apply(
                        lambda x: x.isin([True, 1, b'\x01']).any(), axis=1
                    )
                    filtered_df = filtered_df[heima_mask]
            
            # 排序
            sort_map = {
                "日BLUE↓": ('blue_daily', False),
                "周BLUE↓": ('blue_weekly', False),
                "月BLUE↓": ('blue_monthly', False),
                "价格↓": ('price', False),
                "ADX↓": ('adx', False)
            }
            sort_col, sort_asc = sort_map.get(sort_by, ('blue_daily', False))
            if sort_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(sort_col, ascending=sort_asc)
            
            filtered_df = filtered_df.head(show_count)
            
            if filtered_df.empty:
                st.info("没有符合条件的股票")
            else:
                # 卡片式展示 (每行3个)
                st.markdown(f"**找到 {len(filtered_df)} 只股票** | 点击卡片查看详情")
                
                # 用session state记录选中的股票
                if 'discover_selected' not in st.session_state:
                    st.session_state['discover_selected'] = None
                
                # 使用columns展示卡片
                cols_per_row = 3
                for row_idx in range(0, len(filtered_df), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for col_idx, col in enumerate(cols):
                        data_idx = row_idx + col_idx
                        if data_idx >= len(filtered_df):
                            break
                        
                        row = filtered_df.iloc[data_idx]
                        symbol = row.get('symbol', 'N/A')
                        company_name = row.get('company_name', '')
                        price = row.get('price', 0)
                        blue_d = row.get('blue_daily', 0)
                        blue_w = row.get('blue_weekly', 0)
                        blue_m = row.get('blue_monthly', 0)
                        adx = row.get('adx', 0)
                        
                        # 价格符号
                        price_sym = "¥" if market == "CN" else "$"
                        
                        # 显示名称：有公司名则显示，否则只显示代码
                        display_name = f"{company_name}" if company_name else symbol
                        display_code = symbol.split('.')[0] if '.' in symbol else symbol  # 去掉 .SH/.SZ 后缀
                        
                        # 信号强度颜色
                        if blue_d > 100 and blue_w > 80:
                            card_color = "#00C853"  # 绿色 - 强信号
                            card_bg = "#1a472a"
                        elif blue_d > 100:
                            card_color = "#FFD600"  # 黄色 - 中等
                            card_bg = "#4a4a00"
                        else:
                            card_color = "#666"  # 灰色 - 弱
                            card_bg = "#333"
                        
                        with col:
                            # 卡片容器 - 显示名称和代码
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {card_bg}66, {card_bg}33); 
                                        border: 1px solid {card_color}44;
                                        border-radius: 12px; padding: 16px; margin-bottom: 12px;
                                        transition: transform 0.2s;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.2em; font-weight: bold; color: {card_color};">{display_name}</span>
                                        <span style="font-size: 0.85em; color: #888; margin-left: 6px;">{display_code}</span>
                                    </div>
                                    <span style="font-size: 1.1em;">{price_sym}{price:.2f}</span>
                                </div>
                                <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                                    <span style="background: #00C85333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        日B {blue_d:.0f}
                                    </span>
                                    <span style="background: #FFD60033; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        周B {blue_w:.0f}
                                    </span>
                                    <span style="background: #2196F333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        月B {blue_m:.0f}
                                    </span>
                                    <span style="background: #9C27B033; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">
                                        ADX {adx:.0f}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 操作按钮
                            if st.button(f"📊 {symbol} 详情", key=f"disc_{symbol}", use_container_width=True):
                                st.session_state['discover_selected'] = symbol
                
                # 显示选中股票的详情
                if st.session_state.get('discover_selected'):
                    st.divider()
                    symbol = st.session_state['discover_selected']
                    
                    # 关闭按钮
                    header_col1, header_col2 = st.columns([6, 1])
                    with header_col1:
                        st.markdown(f"### 📊 {symbol} 详细分析")
                    with header_col2:
                        if st.button("❌ 关闭", key="close_discover_detail"):
                            st.session_state['discover_selected'] = None
                            st.rerun()
                    
                    render_unified_stock_detail(
                        symbol=symbol,
                        market=market,
                        key_prefix=f"discover_{symbol}"
                    )
        else:
            st.info("正在加载数据...")
    
    # === Tab 3: 策略精选 (原有逻辑) ===
    with work_tab3:
        st.markdown("### 🎯 策略精选")
        st.caption("8大策略同时选股，多策略共识=高可信度")
        
        # 获取策略选股
        manager = get_strategy_manager()
        all_picks = manager.get_all_picks(df, top_n=top_n)
        consensus = manager.get_consensus_picks(df, min_votes=2)
        
        # 显示共识精选
        if consensus:
            st.markdown("#### 🏆 多策略共识 (被2个以上策略选中)")
            
            consensus_data = []
            for symbol, votes, avg_score in consensus[:10]:
                stock_row = df[df['symbol'] == symbol].iloc[0] if not df.empty and symbol in df['symbol'].values else {}
                
                blue_d = stock_row.get('blue_daily', 0) if hasattr(stock_row, 'get') else (stock_row['blue_daily'] if 'blue_daily' in getattr(stock_row, 'index', []) else 0)
                blue_w = stock_row.get('blue_weekly', 0) if hasattr(stock_row, 'get') else (stock_row['blue_weekly'] if 'blue_weekly' in getattr(stock_row, 'index', []) else 0)
                price = stock_row.get('price', 0) if hasattr(stock_row, 'get') else (stock_row['price'] if 'price' in getattr(stock_row, 'index', []) else 0)
                
                consensus_data.append({
                    '代码': symbol,
                    '⭐策略票数': votes,
                    '平均分': f"{avg_score:.0f}",
                    '日BLUE': f"{blue_d:.0f}",
                    '周BLUE': f"{blue_w:.0f}",
                    '价格': f"${price:.2f}" if price else '-',
                    '建议止损': f"${price*0.92:.2f}" if price else '-',
                    '建议目标': f"${price*1.15:.2f}" if price else '-'
                })
            
            consensus_df = pd.DataFrame(consensus_data)
            
            # 显示表格
            event = st.dataframe(
                consensus_df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
            
            # 处理行选择 - 显示股票详情
            selected_consensus_symbol = None
            if event and hasattr(event, 'selection') and event.selection.rows:
                idx = event.selection.rows[0]
                if idx < len(consensus_data):
                    selected_consensus_symbol = consensus_data[idx]['代码']
            
            # 加入观察列表按钮
            if consensus_data:
                sel_col1, sel_col2 = st.columns([3, 1])
                with sel_col1:
                    selected_symbol = st.selectbox(
                        "选择股票加入观察",
                        [c['代码'] for c in consensus_data],
                        key="consensus_select"
                    )
                with sel_col2:
                    if st.button("📋 加入观察", key="add_consensus_watch", type="primary"):
                        try:
                            from services.signal_tracker import add_to_watchlist
                            # 找到选中股票的数据
                            sel_data = next((c for c in consensus_data if c['代码'] == selected_symbol), None)
                            if sel_data:
                                price = float(sel_data['价格'].replace('$', '')) if sel_data['价格'] != '-' else 0
                                add_to_watchlist(
                                    symbol=selected_symbol,
                                    market=market,
                                    entry_price=price,
                                    target_price=price * 1.15,
                                    stop_loss=price * 0.92,
                                    signal_type='consensus',
                                    signal_score=float(sel_data['平均分']),
                                    notes=f"多策略共识 {sel_data['⭐策略票数']}票"
                                )
                                st.success(f"✅ {selected_symbol} 已加入观察列表")
                        except Exception as e:
                            st.error(f"添加失败: {e}")
            
            # 显示选中股票的详情
            if selected_consensus_symbol:
                st.divider()
                st.markdown(f"### 📊 {selected_consensus_symbol} 详细分析")
                render_unified_stock_detail(
                    symbol=selected_consensus_symbol,
                    market=market,
                    key_prefix=f"consensus_{selected_consensus_symbol}"
                )
        else:
            st.info("暂无共识股票，请检查扫描数据")
        
        st.divider()
        st.markdown("📊 更多策略详情请下滑查看...")
        # 详细的策略选股在下方继续显示
    
    # === Tab 4: 我的持仓 (重新设计 - 专注持仓管理) ===
    with work_tab4:
        st.markdown("### 💼 我的持仓")
        
        # 持仓概览
        total_value = portfolio.get('total_value', 0)
        total_pnl = portfolio.get('total_pnl_pct', 0)
        cash = portfolio.get('cash', 100000)
        
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.metric("总资产", f"${total_value + cash:,.0f}")
        with p_col2:
            st.metric("持仓市值", f"${total_value:,.0f}")
        with p_col3:
            delta_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("总盈亏", f"{total_pnl:+.1f}%", delta_color=delta_color)
        with p_col4:
            st.metric("可用现金", f"${cash:,.0f}")
        
        st.divider()
        
        if positions:
            st.markdown(f"**当前持有 {len(positions)} 只股票**")
            
            # 持仓列表（带详情展示）
            for pos in positions:
                symbol = pos.get('symbol', '')
                shares = pos.get('shares', 0)
                avg_cost = pos.get('avg_cost', 0)
                current_price = pos.get('current_price', avg_cost)
                
                pnl = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
                market_value = shares * current_price
                
                # 颜色
                pnl_color = "#00C853" if pnl >= 0 else "#FF1744"
                
                with st.container():
                    # 持仓卡片
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                border-left: 4px solid {pnl_color};
                                padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.4em; font-weight: bold;">{symbol}</span>
                                <span style="margin-left: 12px; color: #888;">{shares}股</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.2em; color: {pnl_color};">{pnl:+.1f}%</div>
                                <div style="color: #888; font-size: 0.9em;">${market_value:,.0f}</div>
                            </div>
                        </div>
                        <div style="margin-top: 8px; display: flex; gap: 16px; color: #888; font-size: 0.9em;">
                            <span>成本 ${avg_cost:.2f}</span>
                            <span>现价 ${current_price:.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 操作按钮
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        if st.button(f"📊 分析", key=f"pos_detail_{symbol}", use_container_width=True):
                            st.session_state['portfolio_selected'] = symbol
                    with btn_col2:
                        if st.button(f"➕ 加仓", key=f"pos_add_{symbol}", use_container_width=True):
                            st.session_state['portfolio_add'] = symbol
                    with btn_col3:
                        sell_label = "🔴 止损" if pnl < -5 else ("✅ 止盈" if pnl > 10 else "📤 卖出")
                        if st.button(sell_label, key=f"pos_sell_{symbol}", use_container_width=True):
                            st.session_state['portfolio_sell'] = symbol
            
            # 显示选中持仓的详情
            if st.session_state.get('portfolio_selected'):
                st.divider()
                symbol = st.session_state['portfolio_selected']
                
                header_col1, header_col2 = st.columns([6, 1])
                with header_col1:
                    st.markdown(f"### 📊 {symbol} 持仓分析")
                with header_col2:
                    if st.button("❌ 关闭", key="close_portfolio_detail"):
                        st.session_state['portfolio_selected'] = None
                        st.rerun()
                
                render_unified_stock_detail(
                    symbol=symbol,
                    market=market,
                    key_prefix=f"portfolio_{symbol}"
                )
            
            # 处理卖出
            if st.session_state.get('portfolio_sell'):
                symbol = st.session_state['portfolio_sell']
                pos = next((p for p in positions if p.get('symbol') == symbol), {})
                
                st.divider()
                st.markdown(f"### 📤 卖出 {symbol}")
                
                with st.form(f"sell_form_{symbol}"):
                    max_shares = pos.get('shares', 0)
                    sell_shares = st.number_input("卖出股数", min_value=1, max_value=max_shares, value=max_shares)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("✅ 确认卖出", type="primary"):
                            try:
                                from services.portfolio_service import paper_sell
                                price = pos.get('current_price', 0)
                                result = paper_sell(symbol, sell_shares, price)
                                if result.get('success'):
                                    pnl = result.get('realized_pnl', 0)
                                    st.success(f"🎉 卖出成功! 盈亏: ${pnl:+.2f}")
                                    st.session_state['portfolio_sell'] = None
                                    st.rerun()
                                else:
                                    st.error(result.get('error', '卖出失败'))
                            except Exception as e:
                                st.error(f"卖出失败: {e}")
                    with col2:
                        if st.form_submit_button("❌ 取消"):
                            st.session_state['portfolio_sell'] = None
                            st.rerun()
        else:
            st.info("📭 暂无持仓")
            st.markdown("前往「发现新股」或「策略精选」寻找买入机会！")


# Legacy code removed - all functionality is now in the 4 redesigned tabs above
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
    
    # === 调试: 检查数据加载后的 Heima 列 ===
    if df is not None and not df.empty and 'Heima_Daily' in df.columns:
        heima_true_count = df['Heima_Daily'].sum()
        heima_sample = df['Heima_Daily'].head(5).tolist()
        heima_types = [type(v).__name__ for v in heima_sample]
        print(f"[DEBUG] 加载后 Heima_Daily: True={heima_true_count}/{len(df)}, 样本={heima_sample}, 类型={heima_types}")

    if df is None or df.empty:
        st.warning("⚠️ 未找到扫描结果。")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **方式一**: 运行每日扫描\n```bash\ncd versions/v2\npython scripts/run_daily_scan.py\n```")
        with col2:
            st.info("💡 **方式二**: 批量回填历史数据\n```bash\ncd versions/v2\npython scripts/backfill.py --start 2025-12-01 --end 2026-01-07\n```")
        return
            
    # === 🏆 智能排序 & Alpha Picks ===
    # 在筛选之前先计算全量分数 (仅基础技术面分)
    try:
        from ml.ranking_system import get_ranking_system
        ranker = get_ranking_system()
        # 仅计算基础分，不自动加载耗时的大师/舆情数据
        df = ranker.calculate_integrated_score(df)
    except ImportError:
        pass
    except Exception as e:
        print(f"Ranking error: {e}")

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
        
        # === 5. 黑马信号筛选 ===
        st.subheader("🐴 黑马信号")
        
        # 从 session_state 获取当前筛选值
        heima_options = ["全部", "有日黑马", "有周黑马", "有月黑马", "有任意黑马"]
        current_heima = st.session_state.get('heima_filter', '全部')
        current_index = heima_options.index(current_heima) if current_heima in heima_options else 0
        
        heima_filter = st.radio(
            "黑马筛选",
            options=heima_options,
            index=current_index,
            horizontal=True,
            help="筛选出有黑马信号的股票",
            key="heima_filter_radio"
        )
        st.session_state['heima_filter'] = heima_filter
        
        # === 6. 高级筛选 (折叠) ===
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
        "新发现": st.column_config.TextColumn("状态", width="small", help="🆕=今日新发现, 📅=之前出现过"),
        "新闻": st.column_config.TextColumn("新闻", width="small", help="🟢利好/🔴利空 (利好数/利空数)")
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




    # === 新闻情绪分析 ===
    # 添加新闻情绪列 (按需加载)
    news_cache_key = f"news_sentiment_{selected_date}_{selected_market}"
    
    col_news1, col_news2 = st.columns([1, 4])
    with col_news1:
        analyze_news = st.button("📰 获取新闻情绪", help="分析前10只股票的新闻情绪")
    with col_news2:
        if news_cache_key in st.session_state:
            cached_count = len([v for v in st.session_state[news_cache_key].values() if v])
            st.caption(f"✅ 已缓存 {cached_count} 只股票的新闻情绪")
    
    if analyze_news and 'Ticker' in df.columns and len(df) > 0:
        try:
            from news import get_news_intelligence
            intel = get_news_intelligence(use_llm=False)
            
            # 只分析前10只 (避免太慢)
            tickers_to_analyze = df['Ticker'].tolist()[:10]
            news_results = {}
            
            progress = st.progress(0, text="正在分析新闻...")
            for i, ticker in enumerate(tickers_to_analyze):
                try:
                    events, impacts, digest = intel.analyze_symbol(ticker, market=selected_market)
                    
                    if digest.total_news_count > 0:
                        ratio = digest.sentiment_ratio()
                        if ratio > 0.3:
                            emoji = "🟢"
                        elif ratio < -0.3:
                            emoji = "🔴"
                        else:
                            emoji = "⚪"
                        
                        news_results[ticker] = f"{emoji}{digest.bullish_count}/{digest.bearish_count}"
                    else:
                        news_results[ticker] = "➖"
                except:
                    news_results[ticker] = "❓"
                
                progress.progress((i + 1) / len(tickers_to_analyze), 
                                 text=f"分析 {ticker} ({i+1}/{len(tickers_to_analyze)})")
            
            progress.empty()
            
            # 缓存结果
            st.session_state[news_cache_key] = news_results
            st.success(f"✅ 新闻分析完成！{len(news_results)} 只股票")
            st.rerun()
            
        except Exception as e:
            st.error(f"新闻分析失败: {e}")
    
    # 显示列顺序：核心指标在前，新发现标记靠前，新闻情绪列
    if news_cache_key in st.session_state and 'Ticker' in df.columns:
        news_data = st.session_state[news_cache_key]
        df['新闻'] = df['Ticker'].map(lambda t: news_data.get(t, '➖'))

    # === 大师策略深度分析 ===
    master_cache_key = f"master_analysis_{selected_date}_{selected_market}"
    master_details_key = f"{master_cache_key}_details"
    
    col_master1, col_master2 = st.columns([1, 4])
    with col_master1:
        analyze_master = st.button("🤖 大师深度分析", help="基于5位大师策略分析前20只股票 (需获取历史数据，较慢)")
    with col_master2:
        if master_cache_key in st.session_state:
            cached_master = len([v for v in st.session_state[master_cache_key].values() if v])
            st.caption(f"✅ 已生成 {cached_master} 份大师报告")

    if analyze_master and 'Ticker' in df.columns and len(df) > 0:
        try:
            from strategies.master_strategies import analyze_stock_for_master, get_master_summary_for_stock
            if selected_market == 'US':
                from data_fetcher import get_us_stock_data as get_data
            else:
                from data_fetcher import get_cn_stock_data as get_data
            
            # 先去重
            all_tickers = df['Ticker'].unique().tolist()
            # 分析前20只 (避免超时)
            tickers_to_analyze = all_tickers[:20]
            master_results = {}
            master_details = {} # 存储详细报告用于展示
            
            progress = st.progress(0, text="正在进行大师级推演...")
            
            for i, ticker in enumerate(tickers_to_analyze):
                try:
                    # 1. 获取近期历史数据 (用于计算均线、量比、九转)
                    hist_df = get_data(ticker, days=40)
                    
                    if hist_df is not None and not hist_df.empty:
                        # 准备参数
                        current_row = df[df['Ticker'] == ticker].iloc[0]
                        price = float(current_row.get('Price', 0))
                        
                        # 计算技术指标
                        sma5 = hist_df['Close'].rolling(5).mean().iloc[-1]
                        sma20 = hist_df['Close'].rolling(20).mean().iloc[-1]
                        
                        # 量比
                        vol = hist_df['Volume'].iloc[-1]
                        vol_ma5 = hist_df['Volume'].rolling(5).mean().iloc[-1]
                        vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
                        
                        # 九转计数 (简单计算)
                        close_prices = hist_df['Close'].values
                        td_count = 0
                        if len(close_prices) > 13:
                            # 简化的TD检测，实际应使用 SignalDetector
                            c = close_prices
                            if c[-1] < c[-5]: # 下跌
                                count = 0
                                for k in range(1, 10):
                                    if c[-k] < c[-k-4]: count -= 1
                                    else: break
                                td_count = count
                            elif c[-1] > c[-5]: # 上涨
                                count = 0
                                for k in range(1, 10):
                                    if c[-k] > c[-k-4]: count += 1
                                    else: break
                                td_count = count
                        
                        # 2. 调用大师分析
                        analyses = analyze_stock_for_master(
                            symbol=ticker,
                            blue_daily=float(current_row.get('Day BLUE', 0)),
                            blue_weekly=float(current_row.get('Week BLUE', 0)),
                            blue_monthly=float(current_row.get('Month BLUE', 0)),
                            adx=float(current_row.get('ADX', 0)),
                            vol_ratio=vol_ratio,
                            change_pct=float(hist_df['Close'].pct_change().iloc[-1] * 100),
                            price=price,
                            sma5=sma5,
                            sma20=sma20,
                            td_count=td_count,
                            is_heima=True if '黑马' in str(current_row.get('Strategy', '')) else False
                        )
                        
                        # 3. 汇总结果
                        summary = get_master_summary_for_stock(analyses)
                        
                        # 存入结果
                        master_results[ticker] = summary['overall_action']
                        master_details[ticker] = analyses
                        
                    else:
                        master_results[ticker] = "数据不足"
                        
                except Exception as e:
                    master_results[ticker] = "分析失败"
                    print(f"Error analyzing {ticker}: {e}")
                
                progress.progress((i + 1) / len(tickers_to_analyze), 
                                 text=f"大师正在分析 {ticker} ({i+1}/{len(tickers_to_analyze)})")
            
            progress.empty()
            
            # 缓存结果
            st.session_state[master_cache_key] = master_results
            st.session_state[master_details_key] = master_details
            st.success(f"✅ 大师分析完成！已生成 {len(master_results)} 份策略报告")
            st.rerun()
            
        except Exception as e:
            st.error(f"大师分析服务异常: {e}")
            import traceback
            st.code(traceback.format_exc())

    # 将大师建议合并到 DataFrame
    if master_cache_key in st.session_state and 'Ticker' in df.columns:
        master_data = st.session_state[master_cache_key]
        df['大师建议'] = df['Ticker'].map(lambda t: master_data.get(t, '➖'))

    # 更新列配置
    column_config.update({
        "大师建议": st.column_config.TextColumn("大师建议", width="medium", help="5位大师综合评级")
    })

    # === 添加黑马列 (修复版) ===
    # 检测黑马字段
    def get_col(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None
    
    def safe_bool_convert(series):
        """
        安全地将列转换为布尔值
        处理: 0/1, True/False, None, bytes (b'\x01'), strings ('True'/'False')
        """
        import numpy as np
        
        def to_bool(val):
            # 1. 处理 None 和 NaN
            if val is None:
                return False
            try:
                if pd.isna(val):
                    return False
            except (TypeError, ValueError):
                pass  # 某些类型不支持 pd.isna
            
            # 2. 处理布尔值 (包括 numpy bool)
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            
            # 3. 处理整数/浮点数
            if isinstance(val, (int, float, np.integer, np.floating)):
                return val == 1  # 只有 1 才是 True
            
            # 4. 处理字节 (SQLite BLOB)
            if isinstance(val, bytes):
                return val == b'\x01'
            
            # 5. 处理字符串 (Supabase JSON 可能返回字符串)
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 't')
            
            # 6. 未知类型，默认 False
            print(f"[DEBUG] safe_bool_convert: 未知类型 {type(val).__name__}: {val!r}")
            return False
        
        return series.apply(to_bool)
    
    heima_daily_col = get_col(df, ['Heima_Daily', 'heima_daily'])
    heima_weekly_col = get_col(df, ['Heima_Weekly', 'heima_weekly'])
    heima_monthly_col = get_col(df, ['Heima_Monthly', 'heima_monthly'])
    heima_any_col = get_col(df, ['Is_Heima', 'is_heima'])  # 兼容旧数据
    
    # 创建黑马布尔列 (用于过滤) - 使用安全转换
    # 日黑马: 优先使用 heima_daily, 回退到 is_heima
    if heima_daily_col:
        df['日黑马'] = safe_bool_convert(df[heima_daily_col])
    elif heima_any_col:
        df['日黑马'] = safe_bool_convert(df[heima_any_col])
    else:
        df['日黑马'] = False
    
    # 周黑马: 只使用 heima_weekly
    if heima_weekly_col:
        df['周黑马'] = safe_bool_convert(df[heima_weekly_col])
    else:
        df['周黑马'] = False
    
    # 月黑马: 只使用 heima_monthly
    if heima_monthly_col:
        df['月黑马'] = safe_bool_convert(df[heima_monthly_col])
    else:
        df['月黑马'] = False
    
    # 显示列 (🐴 图标)
    df['日🐴'] = df['日黑马'].apply(lambda x: '🐴' if x else '')
    df['周🐴'] = df['周黑马'].apply(lambda x: '🐴' if x else '')
    df['月🐴'] = df['月黑马'].apply(lambda x: '🐴' if x else '')
    
    # 更新列配置
    column_config.update({
        "日🐴": st.column_config.TextColumn("日🐴", width="small", help="日线黑马"),
        "周🐴": st.column_config.TextColumn("周🐴", width="small", help="周线黑马"),
        "月🐴": st.column_config.TextColumn("月🐴", width="small", help="月线黑马"),
    })
    
    # === 应用黑马筛选 ===
    heima_filter = st.session_state.get('heima_filter', '全部')
    before_heima_count = len(df)
    
    # 统计黑马数量 (调试用)
    day_heima_count = df['日黑马'].sum()
    week_heima_count = df['周黑马'].sum()
    month_heima_count = df['月黑马'].sum()
    
    # === 调试: 检查黑马数据类型和值 ===
    with st.expander("🔍 黑马调试信息", expanded=False):
        st.write(f"**数据来源**: {data_source}")
        st.write(f"**总记录数**: {len(df)}")
        st.write(f"**Heima_Daily 列存在**: {heima_daily_col}")
        
        if heima_daily_col:
            sample_values = df[heima_daily_col].head(10).tolist()
            sample_types = [type(v).__name__ for v in sample_values]
            unique_values = df[heima_daily_col].unique().tolist()[:10]  # 前10个唯一值
            st.write(f"**{heima_daily_col} 样本值**: {sample_values}")
            st.write(f"**样本类型**: {sample_types}")
            st.write(f"**唯一值 (前10)**: {unique_values}")
            st.write(f"**列 dtype**: {df[heima_daily_col].dtype}")
            
            # 统计各类型值的数量
            true_count = len(df[df[heima_daily_col] == True])
            false_count = len(df[df[heima_daily_col] == False])
            one_count = len(df[df[heima_daily_col] == 1])
            zero_count = len(df[df[heima_daily_col] == 0])
            none_count = df[heima_daily_col].isna().sum()
            st.write(f"**值统计**: True={true_count}, False={false_count}, 1={one_count}, 0={zero_count}, None/NaN={none_count}")
        else:
            st.warning(f"⚠️ Heima_Daily 列不存在！可用列: {list(df.columns)[:20]}...")
            # 检查 Is_Heima
            if 'Is_Heima' in df.columns:
                is_heima_true = df['Is_Heima'].sum()
                st.write(f"**Is_Heima True 数量**: {is_heima_true}/{len(df)}")
        
        st.write("---")
        st.write(f"**日黑马 样本值**: {df['日黑马'].head(10).tolist()}")
        st.write(f"**日黑马 dtype**: {df['日黑马'].dtype}")
        st.write(f"**日黑马 True 数量**: {day_heima_count}/{len(df)}")
        
        # 检查 🐴 列
        emoji_sample = df['日🐴'].head(10).tolist()
        emoji_non_empty = len([x for x in df['日🐴'].tolist() if x])
        st.write(f"**日🐴 样本值**: {emoji_sample}")
        st.write(f"**日🐴 非空数量**: {emoji_non_empty}/{len(df)}")
    
    if heima_filter == "有日黑马":
        df = df[df['日黑马'] == True]
    elif heima_filter == "有周黑马":
        df = df[df['周黑马'] == True]
    elif heima_filter == "有月黑马":
        df = df[df['月黑马'] == True]
    elif heima_filter == "有任意黑马":
        df = df[(df['日黑马'] == True) | (df['周黑马'] == True) | (df['月黑马'] == True)]
    
    # 显示筛选结果
    if heima_filter != "全部":
        st.info(f"🐴 黑马筛选 [{heima_filter}]: {before_heima_count} → {len(df)} 只")
    else:
        # 在"全部"模式下显示各类黑马统计
        st.caption(f"🐴 黑马统计: 日{day_heima_count} | 周{week_heima_count} | 月{month_heima_count}")

    # 显示列顺序
    display_cols = ['Rank_Score', '新发现', '日🐴', '周🐴', '月🐴', '新闻', '大师建议', 'Ticker', 'Name', 'Mkt Cap', 'Cap_Category', 'Price', 'Turnover', 'Day BLUE', 'Week BLUE', 'Month BLUE', 'ADX', 'Strategy', '筹码形态', 'Wave_Desc', 'Chan_Desc', 'Stop Loss', 'Shares Rec', 'Regime']
    existing_cols = [c for c in display_cols if c in df.columns]

    # === 按用户要求分4个标签页 ===
    # 预先计算各类别数据
    has_day = df['Day BLUE'] > 0 if 'Day BLUE' in df.columns else False
    has_week = df['Week BLUE'] > 0 if 'Week BLUE' in df.columns else False
    has_month = df['Month BLUE'] > 0 if 'Month BLUE' in df.columns else False
    
    # 1. 只日BLUE: Day > 0, Week = 0
    sort_col_day = 'Rank_Score' if 'Rank_Score' in df.columns else 'Day BLUE'
    df_day_only = df[has_day & ~has_week].sort_values(sort_col_day, ascending=False) if 'Day BLUE' in df.columns else df.head(0)
    
    # 2. 日周/只周: (Day > 0 AND Week > 0) OR (Day = 0 AND Week > 0)
    sort_col_week = 'Rank_Score' if 'Rank_Score' in df.columns else 'Week BLUE'
    df_day_week = df[(has_day & has_week) | (~has_day & has_week)].sort_values(sort_col_week, ascending=False) if 'Week BLUE' in df.columns else df.head(0)
    
    # 3. 日周月/只月: (Day > 0 AND Week > 0 AND Month > 0) OR (Month > 0)
    sort_col_month = 'Rank_Score' if 'Rank_Score' in df.columns else 'Month BLUE'
    df_month = df[(has_day & has_week & has_month) | has_month].sort_values(sort_col_month, ascending=False) if 'Month BLUE' in df.columns else df.head(0)
    
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
    
    # === 🐴 黑马快捷筛选 (在表格上方，更明显) ===
    st.markdown("---")
    heima_col1, heima_col2, heima_col3, heima_col4, heima_col5 = st.columns(5)
    with heima_col1:
        show_all = st.button("🔄 全部", key="heima_all", use_container_width=True)
        if show_all:
            st.session_state['heima_filter'] = '全部'
            st.rerun()
    with heima_col2:
        show_daily = st.button("🐴 日黑马", key="heima_d", use_container_width=True)
        if show_daily:
            st.session_state['heima_filter'] = '有日黑马'
            st.rerun()
    with heima_col3:
        show_weekly = st.button("🐴 周黑马", key="heima_w", use_container_width=True)
        if show_weekly:
            st.session_state['heima_filter'] = '有周黑马'
            st.rerun()
    with heima_col4:
        show_monthly = st.button("🐴 月黑马", key="heima_m", use_container_width=True)
        if show_monthly:
            st.session_state['heima_filter'] = '有月黑马'
            st.rerun()
    with heima_col5:
        show_any = st.button("🐴 任意黑马", key="heima_any", use_container_width=True)
        if show_any:
            st.session_state['heima_filter'] = '有任意黑马'
            st.rerun()
    
    # 显示当前黑马筛选状态
    current_filter = st.session_state.get('heima_filter', '全部')
    if current_filter != '全部':
        st.info(f"🐴 当前筛选: **{current_filter}** (共 {len(df)} 只)")
    
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
            df_day_only = df_day_only.sort_values(sort_col_day, ascending=False)
            event1 = st.dataframe(
                df_day_only[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
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
            df_day_week = df_day_week.sort_values(sort_col_week, ascending=False)
            event2 = st.dataframe(
                df_day_week[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
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
            df_month = df_month.sort_values(sort_col_month, ascending=False)
            event3 = st.dataframe(
                df_month[existing_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
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
            # 全量扫描 - 从 Polygon API 获取所有股票
            try:
                from data_fetcher import get_all_us_tickers, get_all_cn_tickers
                if selected_market == 'CN':
                    scan_tickers = get_all_cn_tickers()
                else:
                    scan_tickers = get_all_us_tickers()
                # 限制数量，避免太慢
                if len(scan_tickers) > 3000:
                    scan_tickers = scan_tickers[:3000]
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
        
        # 分析模式选择
        analysis_mode = st.radio(
            "分析模式",
            options=["📊 基础模式", "🔥 增强模式"],
            horizontal=True,
            key="sector_analysis_mode",
            help="增强模式显示量比、连涨天数、资金流向、综合热度"
        )
        
        if analysis_mode == "🔥 增强模式":
            # 增强模式：显示热度评分
            from data_fetcher import get_cn_sector_enhanced, get_us_sector_enhanced
            
            enhanced_key = f"sector_enhanced_{selected_market}"
            
            if st.button("🔄 刷新增强数据", key="refresh_enhanced"):
                if enhanced_key in st.session_state:
                    del st.session_state[enhanced_key]
            
            if enhanced_key not in st.session_state:
                with st.spinner("正在计算增强指标..."):
                    try:
                        if selected_market == 'CN':
                            enhanced_df = get_cn_sector_enhanced()
                        else:
                            enhanced_df = get_us_sector_enhanced()
                        st.session_state[enhanced_key] = enhanced_df
                    except Exception as e:
                        st.error(f"获取增强数据失败: {e}")
                        enhanced_df = None
            
            enhanced_df = st.session_state.get(enhanced_key)
            
            if enhanced_df is not None and len(enhanced_df) > 0:
                st.markdown("### 🔥 板块热度排行 (综合评分)")
                st.caption("评分 = 涨幅(30%) + 量比(25%) + 连涨(25%) + 资金流(20%)")
                
                # 格式化显示
                display_df = enhanced_df.copy()
                display_df['change_pct'] = display_df['change_pct'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.2f}x")
                display_df['consecutive_days'] = display_df['consecutive_days'].apply(lambda x: f"{x}天" if x > 0 else "-")
                if 'money_flow' in display_df.columns:
                    display_df['money_flow'] = display_df['money_flow'].apply(lambda x: f"+{x:.1f}亿" if x > 0 else f"{x:.1f}亿")
                display_df['heat_score'] = display_df['heat_score'].apply(lambda x: f"🔥{x:.0f}" if x >= 50 else f"{x:.0f}")
                
                display_cols = ['name', 'change_pct', 'volume_ratio', 'consecutive_days', 'heat_score']
                if 'money_flow' in display_df.columns:
                    display_cols.insert(4, 'money_flow')
                
                st.dataframe(
                    display_df[display_cols],
                    column_config={
                        'name': '板块',
                        'change_pct': '涨跌幅',
                        'volume_ratio': '量比',
                        'consecutive_days': '连涨',
                        'money_flow': '资金流',
                        'heat_score': '热度'
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # 可视化热度前10
                if len(enhanced_df) >= 5:
                    import plotly.express as px
                    top10 = enhanced_df.head(10)
                    fig = px.bar(
                        top10, x='name', y='heat_score',
                        title="🔥 热度 Top 10 板块",
                        color='heat_score',
                        color_continuous_scale='YlOrRd'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无增强数据")
        
        else:
            # 基础模式：原有逻辑
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
                                            from data_fetcher import get_us_sector_hot_stocks
                                            hot_df = get_us_sector_hot_stocks(selected_sector)
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


    # === 收集所有选中的股票 (用于批量分析) ===
    selected_tickers_set = set()
    
    # 辅助函数: 安全获取 event
    def collect_from_event(evt, source_df):
        if evt and hasattr(evt, 'selection') and evt.selection.rows:
            return [source_df.iloc[i]['Ticker'] for i in evt.selection.rows if i < len(source_df)]
        return []

    if 'event1' in locals(): selected_tickers_set.update(collect_from_event(event1, df_day_only))
    if 'event2' in locals(): selected_tickers_set.update(collect_from_event(event2, df_day_week))
    if 'event3' in locals(): selected_tickers_set.update(collect_from_event(event3, df_month))
    
    # === 🚀 批量深度分析工作台 ===
    if len(selected_tickers_set) > 0:
        st.divider()
        st.subheader(f"🚀 深度分析工作台 (已选 {len(selected_tickers_set)} 只)")
        
        selected_list = list(selected_tickers_set)
        
        # 批量分析按钮
        col_act, col_info = st.columns([1, 4])
        with col_act:
            do_batch_analyze = st.button("✨ 分析选中股票", type="primary", use_container_width=True)
            
        with col_info:
            st.caption(f"选中: {', '.join(selected_list[:10])} {'...' if len(selected_list)>10 else ''}")

        if do_batch_analyze:
            with st.status("正在进行全方位深度扫描...", expanded=True) as status:
                try:
                    from strategies.master_strategies import analyze_stock_for_master, get_master_summary_for_stock
                    if selected_market == 'US':
                        from data_fetcher import get_us_stock_data as get_data
                    else:
                        from data_fetcher import get_cn_stock_data as get_data

                    # 获取缓存
                    master_cache_key = f"master_analysis_{selected_date}_{selected_market}"
                    master_details_key = f"{master_cache_key}_details"
                    
                    if master_cache_key not in st.session_state: st.session_state[master_cache_key] = {}
                    if master_details_key not in st.session_state: st.session_state[master_details_key] = {}
                    
                    master_res = st.session_state[master_cache_key]
                    master_details = st.session_state[master_details_key]
                    
                    prog_bar = st.progress(0)
                    for i, ticker in enumerate(selected_list):
                        status.write(f"正在分析 {ticker}...")
                        try:
                            # 1. 获取近期历史数据
                            hist_df = get_data(ticker, days=40)
                            
                            if hist_df is not None and not hist_df.empty:
                                # 准备参数
                                current_row = df[df['Ticker'] == ticker].iloc[0]
                                price = float(current_row.get('Price', 0))
                                sma5 = hist_df['Close'].rolling(5).mean().iloc[-1]
                                sma20 = hist_df['Close'].rolling(20).mean().iloc[-1]
                                vol = hist_df['Volume'].iloc[-1]
                                vol_ma5 = hist_df['Volume'].rolling(5).mean().iloc[-1]
                                vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
                                
                                # 简易 TD
                                c = hist_df['Close'].values
                                td_count = 0
                                if len(c) > 13:
                                    if c[-1] > c[-5]: # 上涨
                                        count = 0
                                        for k in range(1, 10):
                                            if c[-k] > c[-k-4]: count += 1
                                            else: break
                                        td_count = count
                                
                                # 调用大师分析
                                analyses = analyze_stock_for_master(
                                    symbol=ticker,
                                    blue_daily=float(current_row.get('Day BLUE', 0)),
                                    blue_weekly=float(current_row.get('Week BLUE', 0)),
                                    blue_monthly=float(current_row.get('Month BLUE', 0)),
                                    adx=float(current_row.get('ADX', 0)),
                                    vol_ratio=vol_ratio,
                                    change_pct=float(hist_df['Close'].pct_change().iloc[-1] * 100),
                                    price=price,
                                    sma5=sma5,
                                    sma20=sma20,
                                    td_count=td_count,
                                    is_heima=True if '黑马' in str(current_row.get('Strategy', '')) else False
                                )
                                
                                # 汇总
                                summary = get_master_summary_for_stock(analyses)
                                master_res[ticker] = summary
                                master_details[ticker] = analyses
                                
                        except Exception as e:
                            print(f"Error analyzing {ticker}: {e}")
                        
                        prog_bar.progress((i + 1) / len(selected_list))
                    
                    # 更新缓存
                    st.session_state[master_cache_key] = master_res
                    st.session_state[master_details_key] = master_details
                    
                    # 重新计算 Rank
                    from ml.ranking_system import get_ranking_system
                    ranker = get_ranking_system()
                    # 这里不需要重新计算整个 df，只需要展示部分
                    
                    st.success("✅ 分析完成！请查看下方 Alpha Picks 报告")
                    st.session_state['show_batch_results'] = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"批量分析失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # === 展示批量分析结果 (Alpha Picks 风格) ===
        if st.session_state.get('show_batch_results', False):
            st.markdown("### 👑 选中股票优选报告 (Alpha Picks)")
            
            # 临时计算这些股票的 Rank Score
            try:
                from ml.ranking_system import get_ranking_system
                ranker = get_ranking_system()
                master_res = st.session_state.get(f"master_analysis_{selected_date}_{selected_market}", {})
                
                # 只对选中的股票计算
                subset_df = df[df['Ticker'].isin(selected_list)].copy()
                scored_df = ranker.calculate_integrated_score(subset_df, master_results=master_res)
                
                # 展示前 5 名
                top_picks = scored_df.head(5)
                
                cols = st.columns(len(top_picks))
                for i, (_, row) in enumerate(top_picks.iterrows()):
                    with cols[i]:
                        score = row['Rank_Score']
                        ticker = row['Ticker']
                        
                        tags = []
                        if score >= 80: tags.append("🔥 强推")
                        if master_res.get(ticker): tags.append("🤖 大师")
                        
                        with st.container(border=True):
                            name = row.get('Name', '')
                            # 如果名称缺失，尝试补充获取
                            if pd.isna(name) or str(name).strip() == '' or str(name) == 'nan':
                                try:
                                    from data_fetcher import get_cn_ticker_details, get_ticker_details
                                    if selected_market == 'CN':
                                        info_dict = get_cn_ticker_details(ticker)
                                    else:
                                        info_dict = get_ticker_details(ticker)
                                    
                                    if info_dict and info_dict.get('name'):
                                        name = info_dict.get('name')
                                except:
                                    name = ticker

                            st.metric(f"{ticker}", f"{score:.0f}分", str(name)[:6])
                            st.progress(score/100)
                            st.caption(" ".join(tags))
                            
                            if st.button(f"详情", key=f"btn_detail_{ticker}"):
                                selected_ticker = ticker
                                selected_row_data = row
                                st.rerun() # 触发下方深度透视
                                
            except Exception as e:
                st.error(f"结果展示出错: {e}")
            st.divider()

    # 4. 深度透视 (使用统一组件)
    if selected_ticker is not None and selected_row_data is not None:
        st.divider()
        
        # 使用统一的股票详情组件
        render_unified_stock_detail(
            symbol=selected_ticker,
            market=selected_market,
            key_prefix=f"scan_{selected_date}"
        )
        
        st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。请结合大盘环境自主决策。")
    else:
        st.info("👈 请在上方表格中点击一行，查看该股票的详细图表和分析。")

    # === 旧代码已被统一组件替代 (render_unified_stock_detail) ===
    # 原有功能包括: 全面智能诊断、大师分析、舆情分析、筹码分析等
    # 全部整合进 components/stock_detail.py，如需查看原实现请查看 git 历史
    
    # 删除旧代码占位符 - 开始删除标记
    _LEGACY_CODE_REMOVED = True  # 以下到 "删除标记结束" 之间的代码已删除
    # 旧代码 (670+行) 已删除，请查看 git 历史
    # 删除范围: 原 AI 诊断、大师分析、舆情分析、K线图表、筹码分析等
    # 替代方案: 全部功能已整合到 render_unified_stock_detail 组件
    
    # === 删除旧代码开始标记 ===
    if False:  # 永不执行 - 保留结构以便未来参考
        # 原有代码包括:
        # - AI 综合诊断 (LLMAnalyzer.generate_decision_dashboard)
        # - 大师量化视角 (master_strategies.analyze_stock_for_master)
        # - 社区舆情分析 (social_monitor.get_social_report)
        # - K线图表 (create_candlestick_chart_dynamic)
        # - 筹码分析 (analyze_chip_flow)
        # - BLUE 信号 (calculate_blue_signal_series)
        # 全部功能已迁移至 components/stock_detail.py
        pass
    # === 旧代码已删除 - 全部功能已迁移至 render_unified_stock_detail ===
    # 
    # 以下代码块 (原约670行) 已被删除:
    # - AI 诊断与决策仪表盘
    # - 大师量化分析 (蔡森/TD/萧明道/黑马/BLUE)
    # - 社区舆情监控
    # - K线图表 (日/周/月线)
    # - 筹码分布与主力动向分析
    # - 技术指标展示
    # - 风控与仓位建议
    #
    # 替代方案: 全部功能已整合到 components/stock_detail.py
    # 查看原实现请使用: git show HEAD~1:versions/v3/app.py
    #
    # === 保留大师分析详情查看器 (删除旧代码但保留此功能) ===

    # === 大师分析详情查看器 (全局) ===
    master_details_key = f"master_analysis_{selected_date}_{selected_market}_details"
    
    if master_details_key in st.session_state:
        st.divider()
        st.header("🔍 大师分析实验室 (Master's Lab)")
        
        details = st.session_state[master_details_key]
        analyzed_tickers = list(details.keys())
        
        if analyzed_tickers:
            col_sel, col_content = st.columns([1, 3])
            
            with col_sel:
                # 尝试获取股票名称
                def get_stock_label(tk):
                    name = ""
                    if 'Ticker' in df.columns and 'Name' in df.columns:
                        matches = df[df['Ticker'] == tk]
                        if not matches.empty:
                            name = matches['Name'].iloc[0]
                    return f"{tk} {name}"
                
                selected_ticker_for_detail = st.radio(
                    "已分析股票", 
                    analyzed_tickers,
                    format_func=get_stock_label
                )
            
            with col_content:
                if selected_ticker_for_detail:
                    analyses = details[selected_ticker_for_detail]
                    
                    # 1. 总体评价
                    from strategies.master_strategies import get_master_summary_for_stock
                    summary = get_master_summary_for_stock(analyses)
                    
                    st.success(f"### {summary['overall_action']}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🟢 看多票数", summary['buy_votes'])
                    c2.metric("🔴 看空票数", summary['sell_votes'])
                    c3.metric("🟡 观望/做T", summary['hold_votes'])
                    
                    if summary['best_opportunity']:
                        st.info(f"**最佳机会**: {summary['best_opportunity']}")
                    if summary['key_risk']:
                        st.warning(f"**主要风险**: {summary['key_risk']}")
                    
                    st.divider()
                    
                    # 2. 各大师详细观点
                    for key, analysis in analyses.items():
                        with st.expander(f"{analysis.icon} {analysis.master}: {analysis.action_emoji} {analysis.action}", expanded=True):
                            st.markdown(f"**判断逻辑**: {analysis.reason}")
                            st.markdown(f"**操作建议**: {analysis.operation}")
                            
                            if analysis.stop_loss:
                                st.markdown(f"🛑 **止损**: {analysis.stop_loss}")
                            if analysis.take_profit:
                                st.markdown(f"🎯 **目标**: {analysis.take_profit}")
                            
                            st.caption(f"信心指数: {'⭐' * analysis.confidence}")

    elif analyze_master: # 如果还没有详情但按钮被按了 (状态中)
        pass # 等待上面rerun
    else:
        st.divider()
        st.caption("ℹ️ 点击上方的 '🤖 大师深度分析' 按钮，可在此处查看 5 位大师对前 20 只股票的详细会诊报告。")



def render_stock_lookup_page():
    """个股查询页面 - 输入任意股票代码，使用统一组件生成详情"""
    st.header("🔍 个股查询")
    st.info("输入任意股票代码，系统将自动获取数据并生成完整的技术分析报告。")
    
    # 输入区域
    col1, col2, col3 = st.columns([1, 0.5, 2.5])
    with col1:
        symbol_input = st.text_input("股票代码", value="", placeholder="例如: AAPL, 600519")
        symbol = symbol_input.upper().strip() if symbol_input else ""
        
        search_btn = st.button("🔍 查询", type="primary", use_container_width=True)
    
    with col2:
        # 市场选择（自动检测）
        market_options = {"🇺🇸 美股": "US", "🇨🇳 A股": "CN"}
        # 自动检测：6位数字 = A股
        default_market = "🇨🇳 A股" if (symbol and symbol.isdigit() and len(symbol) == 6) else "🇺🇸 美股"
        lookup_market = st.radio("市场", options=list(market_options.keys()), index=0 if default_market == "🇺🇸 美股" else 1)
        selected_lookup_market = market_options[lookup_market]
    
    with col3:
        st.markdown("""
        **支持的股票类型:**
        - 美股 (NYSE, NASDAQ): AAPL, NVDA, TSLA, GOOGL...
        - A股 (沪深): 600519, 000001, 300750...
        - ETF: SPY, QQQ, 510300...
        """)
    
    if search_btn and symbol:
        # 使用统一股票详情组件
        st.divider()
        render_unified_stock_detail(
            symbol=symbol,
            market=selected_lookup_market,
            key_prefix=f"lookup_{symbol}"
        )
        st.warning("⚠️ **免责声明**: 以上仅为量化模型生成的参考信号，不构成投资建议。")
    
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
    
    # Tab 结构 - 新增"今日信号"
    tab0, tab1, tab2, tab3 = st.tabs(["🎯 今日信号", "📊 信号表现", "🔍 信号复盘", "💼 我的持仓"])
    
    # ==================== Tab 0: 今日买卖信号 (新增) ====================
    with tab0:
        st.info("🔔 每日买入/卖出信号推荐")
        render_todays_signals_tab()
    
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


def render_todays_signals_tab():
    """今日买卖信号 Tab"""
    # 侧边栏设置
    with st.sidebar:
        st.subheader("🎯 信号设置")
        
        market = st.radio(
            "选择市场",
            ["🇺🇸 美股", "🇨🇳 A股"],
            horizontal=True,
            key="signal_market"
        )
        market_code = "US" if "美股" in market else "CN"
        
        min_confidence = st.slider("最低信心度", 30, 90, 50, key="signal_conf")
        
        generate_btn = st.button("🔄 生成今日信号", type="primary", use_container_width=True)
    
    # 尝试导入信号系统
    try:
        from strategies.signal_system import get_signal_manager, SignalType
        manager = get_signal_manager()
    except Exception as e:
        st.error(f"信号系统加载失败: {e}")
        return
    
    # 生成信号
    if generate_btn:
        with st.spinner("正在生成交易信号..."):
            result = manager.generate_daily_signals(market=market_code)
            
            if 'error' in result:
                st.error(f"生成失败: {result['error']}")
            else:
                st.success(f"✅ 生成 {result.get('buy_signals', 0)} 个买入信号, {result.get('sell_signals', 0)} 个卖出信号")
                st.rerun()
    
    # 显示今日信号
    todays_signals = manager.get_todays_signals(market=market_code)
    
    if not todays_signals:
        st.warning("暂无今日信号，点击「生成今日信号」按钮")
        
        # 显示说明
        st.markdown("""
        ### 📋 信号类型说明
        
        | 信号 | 说明 | 操作建议 |
        |------|------|----------|
        | 🟢 **买入** | 满足买入条件 | 考虑建仓 |
        | 🔴 **卖出** | 获利回吐或趋势转弱 | 减仓或清仓 |
        | 🛑 **止损** | 跌破止损位 | 立即止损 |
        | 🎯 **止盈** | 达到目标价 | 落袋为安 |
        | 👀 **观察** | 待确认信号 | 继续观察 |
        
        ### 💡 信号强度
        
        - 🔥 **强烈**: 多条件共振，信心 > 70%
        - ⚡ **中等**: 主要条件满足，信心 50-70%
        - 💧 **弱**: 单一条件触发，信心 < 50%
        """)
        return
    
    # 过滤低信心度信号
    todays_signals = [s for s in todays_signals if s.get('confidence', 0) >= min_confidence]
    
    # 分类显示
    buy_signals = [s for s in todays_signals if s['signal_type'] == '买入']
    sell_signals = [s for s in todays_signals if s['signal_type'] in ['卖出', '止损', '止盈']]
    
    # 买入信号
    st.subheader("🟢 买入信号")
    if buy_signals:
        buy_df = pd.DataFrame([{
            '代码': s['symbol'],
            '强度': s['strength'],
            '价格': f"${s['price']:.2f}" if market_code == 'US' else f"¥{s['price']:.2f}",
            '目标': f"${s['target_price']:.2f}" if market_code == 'US' else f"¥{s['target_price']:.2f}",
            '止损': f"${s['stop_loss']:.2f}" if market_code == 'US' else f"¥{s['stop_loss']:.2f}",
            '策略': s['strategy'],
            '信心': f"{s['confidence']:.0f}%",
            '理由': s['reason']
        } for s in buy_signals])
        
        st.dataframe(buy_df, hide_index=True, use_container_width=True)
        
        # 可视化
        if len(buy_signals) > 0:
            st.markdown("#### 📊 信心度分布")
            chart_data = pd.DataFrame({
                '股票': [s['symbol'] for s in buy_signals[:10]],
                '信心度': [s['confidence'] for s in buy_signals[:10]]
            })
            st.bar_chart(chart_data.set_index('股票'), height=200)
    else:
        st.info("暂无买入信号")
    
    st.divider()
    
    # 卖出信号
    st.subheader("🔴 卖出/止损信号")
    if sell_signals:
        sell_df = pd.DataFrame([{
            '代码': s['symbol'],
            '类型': s['signal_type'],
            '强度': s['strength'],
            '价格': f"${s['price']:.2f}" if market_code == 'US' else f"¥{s['price']:.2f}",
            '策略': s['strategy'],
            '信心': f"{s['confidence']:.0f}%",
            '理由': s['reason']
        } for s in sell_signals])
        
        st.dataframe(sell_df, hide_index=True, use_container_width=True)
    else:
        st.info("暂无卖出信号")
    
    st.divider()
    
    # 历史信号统计
    st.subheader("📈 近7日信号统计")
    
    historical = manager.get_historical_signals(days=7, market=market_code)
    if historical:
        # 按日期统计
        date_counts = {}
        for s in historical:
            date = s.get('generated_at', 'Unknown')
            if date not in date_counts:
                date_counts[date] = {'买入': 0, '卖出': 0}
            if s['signal_type'] == '买入':
                date_counts[date]['买入'] += 1
            else:
                date_counts[date]['卖出'] += 1
        
        if date_counts:
            stats_df = pd.DataFrame([
                {'日期': date, '买入信号': counts['买入'], '卖出信号': counts['卖出']}
                for date, counts in date_counts.items()
            ])
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    else:
        st.info("暂无历史信号数据")


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
    """我的持仓 Tab - 实盘持仓 + 模拟交易"""
    from db.database import (
        get_portfolio, add_to_watchlist, add_trade, 
        get_trades, update_watchlist_status, delete_from_watchlist
    )
    from services.portfolio_service import (
        get_portfolio_summary, calculate_portfolio_pnl,
        get_paper_account, paper_buy, paper_sell, 
        get_paper_trades, reset_paper_account,
        get_paper_equity_curve, get_paper_monthly_returns, get_realized_pnl_history
    )
    
    # 选择模式
    mode = st.radio(
        "选择模式",
        ["💼 实盘持仓", "🎮 模拟交易"],
        horizontal=True,
        key="portfolio_mode"
    )
    
    st.divider()
    
    # ==================== 实盘持仓模式 ====================
    if mode == "💼 实盘持仓":
        # 权限检查
        if not is_admin():
            st.warning("⚠️ 持仓管理需要管理员权限，您当前为访客模式（只读）")
            st.markdown("---")
        
        # 获取持仓汇总
        with st.spinner("正在获取实时数据..."):
            summary = get_portfolio_summary()
        
        # 汇总统计卡片
        if summary['positions'] > 0:
            st.subheader("📊 持仓汇总")
            
            m1, m2, m3, m4 = st.columns(4)
            
            pnl_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
            
            m1.metric("总成本", f"${summary['total_cost']:,.2f}")
            m2.metric("总市值", f"${summary['total_market_value']:,.2f}")
            m3.metric("未实现盈亏", f"${summary['total_pnl']:+,.2f}", 
                     f"{summary['total_pnl_pct']:+.2f}%", delta_color=pnl_color)
            m4.metric("持仓数", f"{summary['positions']} 只",
                     f"🟢{summary['winners']} 🔴{summary['losers']}")
            
            st.divider()
        
        # 添加持仓表单 (仅管理员可见)
        if is_admin():
            with st.expander("➕ 添加新持仓", expanded=False):
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
                        
                        add_to_watchlist(symbol, new_price, new_shares, entry_date, new_market, 'holding', notes)
                        add_trade(symbol, 'BUY', new_price, new_shares, entry_date, new_market, notes)
                        
                        st.success(f"✅ 已添加 {symbol} 到持仓")
                        st.rerun()
                    else:
                        st.error("请输入股票代码")
        
        # 当前持仓列表 (带实时盈亏)
        st.subheader("💼 当前持仓")
        
        if summary.get('details'):
            for item in summary['details']:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{item['symbol']}**")
                        st.caption(f"成本: ${item['entry_price']:.2f} × {item['shares']}")
                    
                    with col2:
                        if item.get('current_price'):
                            st.markdown(f"现价: **${item['current_price']:.2f}**")
                            st.caption(f"市值: ${item['market_value']:,.2f}")
                        else:
                            st.markdown("现价: --")
                    
                    with col3:
                        if item.get('unrealized_pnl') is not None:
                            pnl = item['unrealized_pnl']
                            pnl_pct = item['unrealized_pnl_pct']
                            color = "green" if pnl >= 0 else "red"
                            st.markdown(f"盈亏: <span style='color:{color}'>${pnl:+,.2f}</span>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"<span style='color:{color}'>{pnl_pct:+.2f}%</span>", 
                                       unsafe_allow_html=True)
                        else:
                            st.markdown("盈亏: --")
                    
                    with col4:
                        st.caption(f"买入: {item['entry_date']}")
                        st.caption(f"市场: {item['market']}")
                    
                    with col5:
                        if is_admin():
                            if st.button("卖出", key=f"sell_{item['id']}"):
                                st.session_state[f"show_sell_{item['id']}"] = True
                    
                    # 卖出对话框
                    if is_admin() and st.session_state.get(f"show_sell_{item['id']}"):
                        sell_price = st.number_input(
                            f"卖出价格", 
                            min_value=0.01, 
                            value=float(item.get('current_price') or item['entry_price']),
                            key=f"sell_price_{item['id']}"
                        )
                        if st.button(f"确认卖出 {item['symbol']}", key=f"confirm_sell_{item['id']}"):
                            add_trade(item['symbol'], 'SELL', sell_price, item['shares'], 
                                     datetime.now().strftime('%Y-%m-%d'), item['market'])
                            update_watchlist_status(item['symbol'], item['entry_date'], 'sold', item['market'])
                            st.success(f"✅ 已卖出 {item['symbol']}")
                            st.session_state[f"show_sell_{item['id']}"] = False
                            st.rerun()
                
                st.divider()
        else:
            st.info("暂无持仓，点击上方添加")
        
        # 交易历史
        with st.expander("📜 交易历史", expanded=False):
            trades = get_trades(limit=20)
            if trades:
                df = pd.DataFrame(trades)
                display_df = df[['symbol', 'trade_type', 'price', 'shares', 'trade_date', 'market']].copy()
                display_df.columns = ['代码', '类型', '价格', '股数', '日期', '市场']
                display_df['类型'] = display_df['类型'].map({'BUY': '🟢买入', 'SELL': '🔴卖出'})
                display_df['价格'] = display_df['价格'].apply(lambda x: f"${x:.2f}")
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("暂无交易记录")
    
    # ==================== 模拟交易模式 ====================
    else:
        st.subheader("🎮 模拟交易账户")
        st.caption("使用虚拟资金测试交易策略，不用真金白银")
        
        # 获取模拟账户
        with st.spinner("加载模拟账户..."):
            account = get_paper_account()
        
        if not account:
            st.error("模拟账户加载失败")
            return
        
        # 账户汇总
        m1, m2, m3, m4 = st.columns(4)
        
        pnl_color = "normal" if account['total_pnl'] >= 0 else "inverse"
        
        m1.metric("初始资金", f"${account['initial_capital']:,.2f}")
        m2.metric("现金余额", f"${account['cash_balance']:,.2f}")
        m3.metric("持仓市值", f"${account['position_value']:,.2f}")
        m4.metric("总权益", f"${account['total_equity']:,.2f}",
                 f"{account['total_pnl_pct']:+.2f}%", delta_color=pnl_color)
        
        st.divider()
        
        # 交易面板
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            st.markdown("#### 🟢 买入")
            buy_symbol = st.text_input("股票代码", placeholder="AAPL", key="paper_buy_symbol")
            buy_shares = st.number_input("买入股数", min_value=1, value=10, key="paper_buy_shares")
            buy_price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0, key="paper_buy_price")
            buy_market = st.selectbox("市场", ["US", "CN"], key="paper_buy_market")
            
            if st.button("🛒 买入", type="primary", key="do_paper_buy"):
                if buy_symbol:
                    price = buy_price if buy_price > 0 else None
                    result = paper_buy(buy_symbol.upper(), buy_shares, price, buy_market)
                    
                    if result['success']:
                        st.success(f"✅ 买入成功! {result['symbol']} {result['shares']}股 @ ${result['price']:.2f}")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
                else:
                    st.error("请输入股票代码")
        
        with col_sell:
            st.markdown("#### 🔴 卖出")
            
            # 持仓下拉选择
            position_options = [f"{p['symbol']} ({p['shares']}股)" for p in account['positions']]
            if position_options:
                selected_pos = st.selectbox("选择持仓", position_options, key="paper_sell_select")
                sell_symbol = selected_pos.split(" ")[0] if selected_pos else ""
                
                # 找到选中的持仓
                selected_position = next((p for p in account['positions'] if p['symbol'] == sell_symbol), None)
                
                if selected_position:
                    max_shares = selected_position['shares']
                    sell_shares = st.number_input("卖出股数", min_value=1, max_value=max_shares, value=max_shares, key="paper_sell_shares")
                    sell_price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0, key="paper_sell_price")
                    
                    if st.button("💰 卖出", type="secondary", key="do_paper_sell"):
                        price = sell_price if sell_price > 0 else None
                        result = paper_sell(sell_symbol, sell_shares, price, selected_position['market'])
                        
                        if result['success']:
                            st.success(f"✅ 卖出成功! 盈亏: ${result['realized_pnl']:+.2f}")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")
            else:
                st.info("暂无持仓可卖出")
        
        st.divider()
        
        # 模拟持仓列表
        st.subheader("📋 模拟持仓")
        
        if account['positions']:
            pos_data = []
            for p in account['positions']:
                pos_data.append({
                    '代码': p['symbol'],
                    '股数': p['shares'],
                    '成本': f"${p['avg_cost']:.2f}",
                    '现价': f"${p['current_price']:.2f}" if p.get('current_price') else '--',
                    '市值': f"${p['market_value']:,.2f}" if p.get('market_value') else '--',
                    '盈亏': f"${p['unrealized_pnl']:+,.2f}" if p.get('unrealized_pnl') else '--',
                    '盈亏%': f"{p['unrealized_pnl_pct']:+.2f}%" if p.get('unrealized_pnl_pct') else '--'
                })
            
            st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)
        else:
            st.info("暂无模拟持仓")
        
        # 交易记录
        with st.expander("📜 模拟交易记录", expanded=False):
            paper_trades = get_paper_trades(limit=30)
            if paper_trades:
                trades_df = pd.DataFrame(paper_trades)
                display_cols = ['symbol', 'trade_type', 'price', 'shares', 'commission', 'trade_date', 'notes']
                available_cols = [c for c in display_cols if c in trades_df.columns]
                display_df = trades_df[available_cols].copy()
                display_df.columns = ['代码', '类型', '价格', '股数', '佣金', '日期', '备注'][:len(available_cols)]
                display_df['类型'] = display_df['类型'].map({'BUY': '🟢买入', 'SELL': '🔴卖出'})
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("暂无交易记录")
        
        # 权益曲线图
        st.subheader("📈 权益曲线")
        
        equity_curve = get_paper_equity_curve()
        
        if not equity_curve.empty and len(equity_curve) > 1:
            import plotly.graph_objects as go
            
            fig_equity = go.Figure()
            
            # 总权益曲线
            fig_equity.add_trace(go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['total_equity'],
                mode='lines+markers',
                name='总权益',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ))
            
            # 初始资金线
            initial = account['initial_capital']
            fig_equity.add_hline(y=initial, line_dash="dash", line_color="gray",
                                annotation_text=f"初始资金 ${initial:,.0f}")
            
            fig_equity.update_layout(
                title="账户权益变化",
                xaxis_title="日期",
                yaxis_title="权益 ($)",
                height=350,
                showlegend=True
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # 收益率曲线
            col_ret, col_heat = st.columns(2)
            
            with col_ret:
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Bar(
                    x=equity_curve['date'],
                    y=equity_curve['return_pct'],
                    marker_color=['#4CAF50' if r >= 0 else '#F44336' for r in equity_curve['return_pct']],
                    name='累计收益率'
                ))
                fig_ret.update_layout(
                    title="累计收益率 (%)",
                    height=250,
                    showlegend=False
                )
                st.plotly_chart(fig_ret, use_container_width=True)
            
            with col_heat:
                # 月度收益热力图
                monthly = get_paper_monthly_returns()
                if not monthly.empty:
                    import plotly.express as px
                    
                    # 创建透视表
                    pivot = monthly.pivot(index='year', columns='month', values='return_pct')
                    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
                    
                    fig_heat = px.imshow(
                        pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        aspect='auto',
                        title="月度收益热力图 (%)"
                    )
                    fig_heat.update_layout(height=250)
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("暂无足够数据生成热力图")
        else:
            st.info("开始交易后将显示权益曲线")
        
        # 已实现盈亏统计
        with st.expander("💰 已实现盈亏", expanded=False):
            realized = get_realized_pnl_history()
            if realized:
                total_realized = sum(r['realized_pnl'] for r in realized)
                wins = len([r for r in realized if r['realized_pnl'] > 0])
                losses = len([r for r in realized if r['realized_pnl'] <= 0])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("已实现盈亏", f"${total_realized:+,.2f}")
                c2.metric("盈利笔数", f"{wins} 笔")
                c3.metric("亏损笔数", f"{losses} 笔")
                
                # 明细表
                realized_df = pd.DataFrame(realized)
                realized_df['realized_pnl'] = realized_df['realized_pnl'].apply(lambda x: f"${x:+,.2f}")
                realized_df.columns = ['日期', '代码', '价格', '股数', '盈亏']
                st.dataframe(realized_df, hide_index=True, use_container_width=True)
            else:
                st.info("暂无已实现盈亏")
        
        # 重置账户
        st.divider()
        if st.button("🔄 重置模拟账户", help="清空所有模拟持仓和交易记录，重置为初始资金"):
            reset_paper_account()
            st.success("✅ 模拟账户已重置")
            st.rerun()


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


# ==================== AI 决策仪表盘 ====================

def render_ai_dashboard_page():
    """AI 决策仪表盘页面 - Gemini 分析"""
    st.header("🤖 AI 决策仪表盘")
    st.caption("基于 Gemini 大模型的智能股票分析，生成一句话结论和检查清单")
    
    from ml.llm_intelligence import generate_stock_decision, check_llm_available
    
    # 检查 LLM 可用性
    llm_status = check_llm_available()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="NVDA", key="ai_symbol").upper().strip()
        
    with col2:
        provider = st.selectbox("AI 模型", ["gemini", "openai"], index=0)
        if provider == "gemini" and not llm_status.get('gemini'):
            st.warning("Gemini 需要设置 GEMINI_API_KEY")
        elif provider == "openai" and not llm_status.get('openai'):
            st.warning("OpenAI 需要设置 OPENAI_API_KEY")
    
    if st.button("🔮 生成 AI 决策", key="gen_ai_decision"):
        with st.spinner(f"正在分析 {symbol}..."):
            try:
                # 获取股票数据
                from data_fetcher import get_us_stock_data
                from indicator_utils import calculate_blue_signal_series, MA
                
                df = get_us_stock_data(symbol, days=90)
                if df is None or len(df) < 30:
                    st.error("无法获取足够的股票数据")
                    return
                
                # 确保数据列存在
                df = df.reset_index(drop=True)
                
                # 计算均线
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA10'] = df['Close'].rolling(10).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                
                # 计算 BLUE 信号
                try:
                    blue_values = calculate_blue_signal_series(
                        df['Open'].values, df['High'].values, 
                        df['Low'].values, df['Close'].values
                    )
                    df['BLUE'] = blue_values
                except:
                    df['BLUE'] = 50  # 默认值
                
                latest = df.iloc[-1]
                price = float(latest['Close'])
                ma5 = float(latest['MA5']) if pd.notna(latest['MA5']) else price
                ma10 = float(latest['MA10']) if pd.notna(latest['MA10']) else price
                ma20 = float(latest['MA20']) if pd.notna(latest['MA20']) else price
                
                # 计算乖离率 (daily_stock_analysis 核心指标)
                bias_ma5 = (price - ma5) / ma5 * 100 if ma5 > 0 else 0
                
                # 判断均线排列
                ma_aligned = ma5 > ma10 > ma20  # 多头排列
                
                # 量比
                vol_ratio = float(latest['Volume']) / df['Volume'].rolling(5).mean().iloc[-1] if df['Volume'].rolling(5).mean().iloc[-1] > 0 else 1
                
                # 获取 BLUE 值
                blue_val = float(latest['BLUE']) if pd.notna(latest['BLUE']) and latest['BLUE'] != 0 else 50
                
                # 准备完整数据
                stock_data = {
                    'symbol': symbol,
                    'price': price,
                    'blue_daily': blue_val,
                    'blue_weekly': blue_val * 0.8,
                    'ma5': ma5,
                    'ma10': ma10,
                    'ma20': ma20,
                    'bias_ma5': bias_ma5,  # 乖离率
                    'ma_aligned': ma_aligned,  # 均线排列
                    'rsi': 50,
                    'volume_ratio': vol_ratio
                }
                
                # 显示技术数据预览
                with st.expander("📊 技术数据"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MA5", f"${ma5:.2f}")
                        st.metric("乖离率", f"{bias_ma5:+.2f}%", delta="危险" if bias_ma5 > 5 else None)
                    with col2:
                        st.metric("MA10", f"${ma10:.2f}")
                        st.metric("均线排列", "多头 ✅" if ma_aligned else "空头 ❌")
                    with col3:
                        st.metric("MA20", f"${ma20:.2f}")
                        st.metric("量比", f"{vol_ratio:.2f}x")
                
                # 生成决策
                from ml.llm_intelligence import LLMAnalyzer
                analyzer = LLMAnalyzer(provider=provider)
                result = analyzer.generate_decision_dashboard(stock_data)
                
                if 'error' in result:
                    st.error(f"分析失败: {result['error']}")
                    return
                
                # 显示结果
                st.success("✅ 分析完成")
                
                # 核心结论
                signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(result.get('signal', 'HOLD'), "🟡")
                st.markdown(f"### {signal_color} {result.get('verdict', '暂无结论')}")
                
                # 关键指标
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("信号", result.get('signal', 'N/A'))
                with col_b:
                    st.metric("置信度", f"{result.get('confidence', 0)}%")
                with col_c:
                    st.metric("入场价", f"${result.get('entry_price', 0):.2f}")
                with col_d:
                    st.metric("止损价", f"${result.get('stop_loss', 0):.2f}")
                
                # 目标价
                st.metric("🎯 目标价", f"${result.get('target_price', 0):.2f}")
                
                # 检查清单
                st.markdown("### ✅ 检查清单")
                checklist = result.get('checklist', [])
                for item in checklist:
                    status = item.get('status', '⚠️')
                    name = item.get('item', '')
                    detail = item.get('detail', '')
                    st.markdown(f"{status} **{name}**: {detail}")
                
                # 风险提示
                if result.get('risk_warning'):
                    st.warning(f"⚠️ {result.get('risk_warning')}")
                
            except Exception as e:
                st.error(f"分析出错: {str(e)}")


# ==================== 组合优化器 ====================

def render_portfolio_optimizer_page():
    """组合优化器页面 - Markowitz"""
    st.header("📐 组合优化器")
    st.caption("基于 Markowitz 均值-方差模型的资产配置优化")
    
    from research.portfolio_optimizer import optimize_portfolio_from_symbols
    
    # 输入股票
    symbols_input = st.text_input(
        "输入股票代码 (逗号分隔，最多10只)",
        value="AAPL, GOOGL, MSFT, NVDA, AMZN",
        key="portfolio_symbols"
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox("市场", ["US", "CN"], index=0, key="portfolio_market")
    with col2:
        days = st.number_input("历史天数", value=252, step=30, key="portfolio_days")
    
    if st.button("📊 优化组合", key="optimize_btn"):
        if len(symbols) < 2:
            st.error("至少需要2只股票")
            return
        
        with st.spinner("正在计算最优配置..."):
            try:
                result = optimize_portfolio_from_symbols(symbols, market=market, days=days)
                
                if 'error' in result:
                    st.error(f"优化失败: {result['error']}")
                    return
                
                st.success("✅ 优化完成")
                
                # 三种策略对比
                tab_sharpe, tab_vol, tab_parity = st.tabs(["📈 最大夏普", "🛡️ 最小波动", "⚖️ 风险平价"])
                
                with tab_sharpe:
                    sharpe = result.get('max_sharpe', {})
                    st.markdown("### 最大夏普比率组合")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("预期收益", f"{sharpe.get('expected_return', 0):.1f}%")
                    with col_b:
                        st.metric("波动率", f"{sharpe.get('volatility', 0):.1f}%")
                    with col_c:
                        st.metric("夏普比率", f"{sharpe.get('sharpe_ratio', 0):.2f}")
                    
                    st.markdown("**配置权重:**")
                    weights = sharpe.get('weights', {})
                    if weights:
                        import plotly.express as px
                        fig = px.pie(names=list(weights.keys()), values=list(weights.values()), 
                                     title="资产配置")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab_vol:
                    vol = result.get('min_vol', {})
                    st.markdown("### 最小波动率组合")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("预期收益", f"{vol.get('expected_return', 0):.1f}%")
                    with col_b:
                        st.metric("波动率", f"{vol.get('volatility', 0):.1f}%")
                    with col_c:
                        st.metric("夏普比率", f"{vol.get('sharpe_ratio', 0):.2f}")
                    
                    weights = vol.get('weights', {})
                    if weights:
                        st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: '权重'}))
                
                with tab_parity:
                    parity = result.get('risk_parity', {})
                    st.markdown("### 风险平价组合")
                    st.caption("每个资产对总风险的贡献相等")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("预期收益", f"{parity.get('expected_return', 0):.1f}%")
                    with col_b:
                        st.metric("波动率", f"{parity.get('volatility', 0):.1f}%")
                    with col_c:
                        st.metric("夏普比率", f"{parity.get('sharpe_ratio', 0):.2f}")
                    
                    weights = parity.get('weights', {})
                    if weights:
                        st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: '权重'}))
                
                # 相关性矩阵
                with st.expander("📊 相关性矩阵"):
                    corr = result.get('correlation', {})
                    if corr:
                        corr_df = pd.DataFrame(corr)
                        st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
                
            except Exception as e:
                st.error(f"优化出错: {str(e)}")


# ==================== 研究工具 ====================

def render_research_page():
    """研究工具页面 - 因子分析等"""
    st.header("🔬 研究工具")
    
    tab_factor, tab_ml, tab_charts = st.tabs(["📊 因子分析", "🤖 ML实验室", "📈 高级图表"])
    
    with tab_factor:
        st.subheader("📊 BLUE 因子 IC 分析")
        st.caption("分析 BLUE 信号对未来收益的预测能力")
        
        from research.factor_research import analyze_factors_from_scan
        
        col1, col2 = st.columns(2)
        with col1:
            market = st.selectbox("市场", ["US", "CN"], key="factor_market")
        
        if st.button("📈 分析 BLUE 因子", key="analyze_factor"):
            with st.spinner("正在分析..."):
                try:
                    result = analyze_factors_from_scan(market=market)
                    
                    if 'error' in result:
                        st.error(result['error'])
                        return
                    
                    st.success("✅ 分析完成")
                    
                    stats = result.get('stats', {})
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("平均 IC", f"{stats.get('mean_ic', 0):.4f}")
                    with col_b:
                        st.metric("IC_IR", f"{stats.get('ic_ir', 0):.4f}")
                    with col_c:
                        st.metric("IC 正向率", f"{stats.get('ic_positive_rate', 0):.1f}%")
                    with col_d:
                        st.metric("样本数", stats.get('n_periods', 0))
                    
                    # 解读
                    ic_ir = stats.get('ic_ir', 0)
                    if ic_ir > 0.5:
                        st.success("📈 BLUE 因子表现优秀 (IC_IR > 0.5)")
                    elif ic_ir > 0.3:
                        st.info("📊 BLUE 因子表现中等 (0.3 < IC_IR < 0.5)")
                    else:
                        st.warning("⚠️ BLUE 因子预测能力较弱 (IC_IR < 0.3)")
                    
                except Exception as e:
                    st.error(f"分析出错: {str(e)}")
    
    with tab_ml:
        # 保留原来的 ML 实验室内容
        render_ml_lab_page()
    
    with tab_charts:
        st.subheader("📈 高级图表工具")
        st.caption("专业级可视化分析工具")
        
        from advanced_charts import (
            create_multi_timeframe_heatmap,
            create_signal_radar_chart,
            create_drawdown_chart,
            create_volume_price_divergence_chart
        )
        from db.database import query_scan_results, get_scanned_dates
        
        chart_type = st.selectbox("选择图表类型", [
            "🔥 多周期共振热力图",
            "🎯 信号强度雷达图",
            "📉 回撤分析图",
            "📊 量价背离分析"
        ], key="chart_type_select")
        
        col1, col2 = st.columns(2)
        with col1:
            market = st.selectbox("市场", ["US", "CN"], key="adv_chart_market")
        
        if chart_type == "🔥 多周期共振热力图":
            if st.button("生成热力图", key="gen_heatmap"):
                with st.spinner("加载数据..."):
                    signals = query_scan_results(market=market, limit=30)
                    if signals:
                        data = {}
                        for s in signals:
                            symbol = s.get('symbol')
                            if symbol:
                                data[symbol] = {
                                    'day_blue': s.get('blue_daily', 0) or 0,
                                    'week_blue': s.get('blue_weekly', 0) or 0,
                                    'month_blue': s.get('blue_monthly', 0) or 0,
                                    'adx': s.get('adx', 0) or 0
                                }
                        fig = create_multi_timeframe_heatmap(data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无数据")
        
        elif chart_type == "🎯 信号强度雷达图":
            symbol = st.text_input("输入股票代码", value="AAPL", key="radar_symbol")
            if st.button("生成雷达图", key="gen_radar"):
                # 模拟获取信号数据
                signal_data = {
                    'blue_strength': np.random.randint(50, 100),
                    'trend_strength': np.random.randint(40, 90),
                    'volume_strength': np.random.randint(30, 80),
                    'chip_strength': np.random.randint(40, 85),
                    'momentum_strength': np.random.randint(45, 95)
                }
                fig = create_signal_radar_chart(signal_data)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("注: 数据为演示用途")
        
        elif chart_type == "📉 回撤分析图":
            if st.button("生成回撤图", key="gen_drawdown"):
                with st.spinner("计算..."):
                    from backtest.backtester import Backtester
                    signals = query_scan_results(market=market, limit=100)
                    if signals:
                        signals_df = pd.DataFrame(signals)
                        bt = Backtester()
                        result = bt.run_signal_backtest(signals_df, holding_days=10, market=market)
                        trades = result.get('trades', [])
                        if trades:
                            equity = [100000]
                            for t in trades:
                                equity.append(equity[-1] * (1 + t.get('pnl_pct', 0) / 100))
                            fig = create_drawdown_chart(equity)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无交易数据")
                    else:
                        st.warning("无信号数据")
        
        elif chart_type == "📊 量价背离分析":
            symbol = st.text_input("输入股票代码", value="AAPL", key="divergence_symbol")
            if st.button("分析量价背离", key="gen_divergence"):
                with st.spinner("加载数据..."):
                    from data_fetcher import get_us_stock_data, get_cn_stock_data
                    if market == "CN":
                        df = get_cn_stock_data(symbol, days=100)
                    else:
                        df = get_us_stock_data(symbol, days=100)
                    if df is not None and len(df) > 20:
                        fig = create_volume_price_divergence_chart(df, symbol)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无法获取数据")

# ==================== 回测实验室辅助函数 ====================

def render_parameter_lab():
    """参数实验室 - 批量回测验证不同参数组合"""
    import plotly.express as px
    from backtest.backtester import Backtester, backtest_blue_signals
    from db.database import query_scan_results, get_scanned_dates
    
    st.subheader("🔬 参数实验室")
    st.caption("基于历史扫描信号，批量验证不同参数组合的有效性")
    
    # --- 参数配置区 ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market = st.selectbox("选择市场", ["US", "CN"], index=0, key="param_lab_market")
        min_blue = st.slider("最低 BLUE 阈值", 50, 180, 100, step=10, key="param_lab_blue",
                            help="只测试 BLUE 值高于此阈值的信号")
    
    with col2:
        holding_days = st.slider("持有天数", 5, 30, 10, step=5, key="param_lab_days",
                                help="买入后固定持有的天数")
        signal_limit = st.slider("测试信号数量", 20, 200, 100, step=20, key="param_lab_limit",
                                help="最多测试多少个历史信号")
        backtest_mode = st.radio("回测模式", ["单信号回测", "组合回测"], horizontal=True, key="bt_mode",
                                help="组合回测模拟真实多仓操作")
    
    with col3:
        # 组合回测专用参数
        if backtest_mode == "组合回测":
            max_positions = st.slider("最大持仓数", 3, 15, 10, key="max_pos")
            position_pct = st.slider("单仓比例%", 5, 20, 10, key="pos_pct") / 100
        else:
            max_positions = 10
            position_pct = 0.1
        
        # 获取可用日期
        available_dates = get_scanned_dates(market=market)
        if available_dates:
            date_options = ["所有日期"] + available_dates[:30]
            selected_date = st.selectbox("指定日期 (可选)", date_options, key="param_lab_date")
        else:
            selected_date = "所有日期"
            st.warning("暂无扫描数据")
    
    # --- 运行回测 ---
    if st.button("🚀 开始批量回测", type="primary", key="run_param_lab"):
        with st.spinner("正在分析历史信号表现..."):
            try:
                # 获取历史信号
                scan_date = None if selected_date == "所有日期" else selected_date
                signals = query_scan_results(
                    scan_date=scan_date,
                    min_blue=min_blue,
                    market=market,
                    limit=signal_limit
                )
                
                if not signals:
                    st.warning("未找到符合条件的信号，请调整筛选条件")
                    return
                
                st.info(f"找到 **{len(signals)}** 个符合条件的信号，开始回测...")
                
                # 运行回测
                bt = Backtester()
                signals_df = pd.DataFrame(signals)
                
                if backtest_mode == "组合回测":
                    results = bt.run_portfolio_backtest(
                        signals_df, 
                        holding_days=holding_days, 
                        max_positions=max_positions,
                        position_size_pct=position_pct,
                        market=market
                    )
                else:
                    results = bt.run_signal_backtest(signals_df, holding_days=holding_days, market=market)
                
                # 获取基准对比
                benchmark = bt.compare_with_benchmark(
                    benchmark='SPY' if market == 'US' else '000001.SS',
                    period_days=30
                )
                
                # --- 显示结果 ---
                st.success("✅ 回测完成!")
                
                # 关键指标卡片
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("总交易数", results.get('total_trades', 0))
                m2.metric("胜率", f"{results.get('win_rate', 0):.1f}%", 
                         delta="好" if results.get('win_rate', 0) > 50 else "差")
                m3.metric("平均收益", f"{results.get('avg_return', 0):.2f}%")
                m4.metric("总收益", f"{results.get('total_return', 0):.2f}%")
                m5.metric("最大回撤", f"-{results.get('max_drawdown', 0):.2f}%", delta_color="inverse")
                
                # 增强指标
                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                with col_a:
                    st.metric("夏普比率", f"{results.get('sharpe_ratio', 0):.2f}")
                with col_b:
                    st.metric("Sortino", f"{results.get('sortino_ratio', 0):.2f}",
                             help="只惩罚下行波动")
                with col_c:
                    st.metric("Calmar", f"{results.get('calmar_ratio', 0):.2f}",
                             help="年化收益/最大回撤")
                with col_d:
                    st.metric("信息比率", f"{results.get('information_ratio', 0):.2f}",
                             help="超额收益稳定性")
                with col_e:
                    alpha = benchmark.get('alpha', 0)
                    st.metric("Alpha", f"{alpha:+.2f}%",
                             delta="跑赢大盘" if alpha > 0 else "跑输大盘")
                
                # --- 资金曲线图 ---
                if results.get('trades'):
                    st.subheader("📈 模拟资金曲线")
                    
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df['cumulative_return'] = (1 + trades_df['pnl_pct'] / 100).cumprod() * 100000
                    trades_df['trade_num'] = range(1, len(trades_df) + 1)
                    
                    fig = go.Figure()
                    
                    # 策略曲线
                    fig.add_trace(go.Scatter(
                        x=trades_df['trade_num'],
                        y=trades_df['cumulative_return'],
                        mode='lines+markers',
                        name='策略收益',
                        line=dict(color='#2196F3', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # 基准线 (初始资金)
                    fig.add_hline(y=100000, line_dash="dash", line_color="gray", 
                                 annotation_text="初始资金 $100,000")
                    
                    fig.update_layout(
                        title="累计收益曲线",
                        xaxis_title="交易序号",
                        yaxis_title="资金 ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- 收益分布图 ---
                    col_dist, col_monthly = st.columns(2)
                    
                    with col_dist:
                        st.subheader("📊 收益分布")
                        fig_dist = px.histogram(
                            trades_df, x='pnl_pct', nbins=20,
                            title="单笔收益分布",
                            labels={'pnl_pct': '收益率 (%)'},
                            color_discrete_sequence=['#4CAF50']
                        )
                        fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
                        fig_dist.update_layout(height=300)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col_monthly:
                        st.subheader("🗓️ 按月统计")
                        # 按月分组统计
                        trades_df['month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M').astype(str)
                        monthly_stats = trades_df.groupby('month').agg({
                            'pnl_pct': ['mean', 'sum', 'count']
                        }).round(2)
                        monthly_stats.columns = ['平均收益%', '总收益%', '交易数']
                        monthly_stats = monthly_stats.reset_index()
                        monthly_stats.columns = ['月份', '平均收益%', '总收益%', '交易数']
                        
                        st.dataframe(monthly_stats, use_container_width=True, hide_index=True)
                    
                    # --- 交易明细 ---
                    with st.expander("📋 查看交易明细", expanded=False):
                        display_df = trades_df[['symbol', 'entry_date', 'entry_price', 
                                               'exit_price', 'holding_days', 'pnl_pct', 'win']].copy()
                        display_df.columns = ['股票', '入场日期', '入场价', '出场价', '持有天数', '收益%', '盈利']
                        display_df['盈利'] = display_df['盈利'].map({True: '✅', False: '❌'})
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"回测出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # --- 参数对比实验 ---
    st.divider()
    st.subheader("⚗️ 参数对比实验")
    st.caption("对比不同 BLUE 阈值的回测效果")
    
    if st.button("🧪 运行对比实验", key="run_compare"):
        with st.spinner("正在对比不同参数..."):
            thresholds = [60, 80, 100, 120, 150]
            comparison_results = []
            
            progress_bar = st.progress(0)
            
            for i, threshold in enumerate(thresholds):
                try:
                    result = backtest_blue_signals(
                        min_blue=threshold,
                        holding_days=10,
                        market=market,
                        limit=50
                    )
                    
                    if 'error' not in result:
                        comparison_results.append({
                            'BLUE阈值': threshold,
                            '交易数': result.get('total_trades', 0),
                            '胜率%': result.get('win_rate', 0),
                            '平均收益%': result.get('avg_return', 0),
                            '总收益%': result.get('total_return', 0),
                            '最大回撤%': result.get('max_drawdown', 0),
                            '夏普比率': result.get('sharpe_ratio', 0)
                        })
                except Exception as e:
                    st.warning(f"阈值 {threshold} 回测失败: {e}")
                
                progress_bar.progress((i + 1) / len(thresholds))
            
            if comparison_results:
                compare_df = pd.DataFrame(comparison_results)
                
                # 显示对比表格
                st.dataframe(
                    compare_df.style.background_gradient(subset=['胜率%', '平均收益%'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 可视化对比
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    x=compare_df['BLUE阈值'].astype(str),
                    y=compare_df['胜率%'],
                    name='胜率%',
                    marker_color='#4CAF50'
                ))
                fig_compare.add_trace(go.Scatter(
                    x=compare_df['BLUE阈值'].astype(str),
                    y=compare_df['平均收益%'],
                    mode='lines+markers',
                    name='平均收益%',
                    yaxis='y2',
                    line=dict(color='#2196F3', width=3)
                ))
                
                fig_compare.update_layout(
                    title="不同 BLUE 阈值的回测效果对比",
                    xaxis_title="BLUE 阈值",
                    yaxis=dict(title="胜率 (%)", side='left'),
                    yaxis2=dict(title="平均收益 (%)", side='right', overlaying='y'),
                    height=400
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # 最佳参数建议
                best_row = compare_df.loc[compare_df['平均收益%'].idxmax()]
                st.success(f"📌 **最佳参数建议**: BLUE 阈值 = **{int(best_row['BLUE阈值'])}**，"
                          f"平均收益 {best_row['平均收益%']:.2f}%，胜率 {best_row['胜率%']:.1f}%")
    
    # --- Walk-Forward 验证 ---
    st.divider()
    st.subheader("🔄 Walk-Forward 验证")
    st.caption("滚动训练/测试窗口，验证策略稳健性，防止过拟合")
    
    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        train_days = st.slider("训练窗口 (天)", 30, 120, 60, step=15, key="wf_train")
    with wf_col2:
        test_days = st.slider("测试窗口 (天)", 10, 60, 20, step=10, key="wf_test")
    
    if st.button("🧪 运行 Walk-Forward 验证", key="run_wf"):
        with st.spinner("正在进行滚动验证..."):
            try:
                # 获取全部历史信号
                all_signals = query_scan_results(market=market, limit=500)
                
                if not all_signals or len(all_signals) < 50:
                    st.warning("历史数据不足，至少需要50条信号")
                else:
                    signals_df = pd.DataFrame(all_signals)
                    bt = Backtester()
                    wf_results = bt.walk_forward_backtest(
                        signals_df,
                        train_days=train_days,
                        test_days=test_days,
                        holding_days=10,
                        market=market
                    )
                    
                    if 'error' in wf_results:
                        st.warning(wf_results['error'])
                    else:
                        st.success(f"✅ 完成 **{wf_results['num_windows']}** 个滚动窗口验证!")
                        
                        # 汇总指标
                        wf_m1, wf_m2, wf_m3 = st.columns(3)
                        wf_m1.metric("平均胜率", f"{wf_results['avg_win_rate']:.1f}%")
                        wf_m2.metric("平均收益", f"{wf_results['avg_return']:.2f}%")
                        wf_m3.metric("平均夏普", f"{wf_results['avg_sharpe']:.2f}")
                        
                        # 窗口明细表
                        if wf_results.get('windows'):
                            windows_df = pd.DataFrame(wf_results['windows'])
                            display_cols = ['test_start', 'test_end', 'test_signals', 
                                          'test_win_rate', 'test_avg_return', 'test_sharpe']
                            windows_df = windows_df[display_cols]
                            windows_df.columns = ['测试开始', '测试结束', '信号数', '胜率%', '平均收益%', '夏普']
                            
                            st.dataframe(
                                windows_df.style.background_gradient(subset=['胜率%', '平均收益%'], cmap='RdYlGn'),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # 可视化各窗口表现
                            fig_wf = go.Figure()
                            fig_wf.add_trace(go.Bar(
                                x=[f"W{i+1}" for i in range(len(windows_df))],
                                y=windows_df['胜率%'],
                                name='胜率%',
                                marker_color='#4CAF50'
                            ))
                            fig_wf.add_trace(go.Scatter(
                                x=[f"W{i+1}" for i in range(len(windows_df))],
                                y=windows_df['平均收益%'],
                                mode='lines+markers',
                                name='平均收益%',
                                yaxis='y2',
                                line=dict(color='#2196F3', width=2)
                            ))
                            fig_wf.update_layout(
                                title="各窗口测试表现",
                                xaxis_title="窗口",
                                yaxis=dict(title="胜率%", side='left'),
                                yaxis2=dict(title="平均收益%", side='right', overlaying='y'),
                                height=350
                            )
                            st.plotly_chart(fig_wf, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Walk-Forward 验证出错: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # --- 蒙特卡洛模拟 ---
    st.divider()
    st.subheader("🎲 蒙特卡洛模拟")
    st.caption("通过随机抽样评估策略风险，计算盈利/破产概率")
    
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        num_sims = st.slider("模拟次数", 100, 2000, 500, step=100, key="mc_sims")
    with mc_col2:
        trades_per_sim = st.slider("每次模拟交易数", 20, 100, 50, step=10, key="mc_trades")
    with mc_col3:
        bankruptcy_pct = st.slider("破产阈值 (%)", 30, 70, 50, step=10, key="mc_bankrupt")
    
    if st.button("🎰 运行蒙特卡洛模拟", key="run_mc"):
        with st.spinner("正在进行蒙特卡洛模拟..."):
            try:
                from backtest.monte_carlo import monte_carlo_simulation, create_monte_carlo_charts
                
                # 获取历史交易数据
                all_signals = query_scan_results(market=market, limit=300)
                
                if not all_signals or len(all_signals) < 20:
                    st.warning("历史数据不足，至少需要20条信号")
                else:
                    # 先运行一次回测获取交易记录
                    signals_df = pd.DataFrame(all_signals)
                    bt = Backtester()
                    bt_result = bt.run_signal_backtest(signals_df, holding_days=10, market=market)
                    trades = bt_result.get('trades', [])
                    
                    if len(trades) < 10:
                        st.warning("有效交易数不足，无法进行模拟")
                    else:
                        # 运行蒙特卡洛
                        mc_result = monte_carlo_simulation(
                            trades,
                            num_simulations=num_sims,
                            trades_per_sim=trades_per_sim,
                            bankruptcy_threshold=bankruptcy_pct / 100
                        )
                        
                        if 'error' in mc_result:
                            st.warning(mc_result['error'])
                        else:
                            st.success(f"✅ 完成 **{num_sims}** 次模拟!")
                            
                            # 关键指标
                            mc_m1, mc_m2, mc_m3, mc_m4 = st.columns(4)
                            mc_m1.metric("盈利概率", f"{mc_result['profit_probability']:.1f}%",
                                        delta="好" if mc_result['profit_probability'] > 60 else "差")
                            mc_m2.metric("破产概率", f"{mc_result['bankruptcy_probability']:.1f}%",
                                        delta="低风险" if mc_result['bankruptcy_probability'] < 10 else "高风险",
                                        delta_color="inverse")
                            mc_m3.metric("平均收益", f"{mc_result['mean_return_pct']:.1f}%")
                            mc_m4.metric("平均最大回撤", f"-{mc_result['mean_max_drawdown']:.1f}%")
                            
                            # 置信区间
                            st.markdown(f"""
                            **90% 置信区间**: 终值在 **${mc_result['ci_5']:,.0f}** ~ **${mc_result['ci_95']:,.0f}** 之间
                            
                            (初始资金 $100,000)
                            """)
                            
                            # 图表
                            charts = create_monte_carlo_charts(mc_result)
                            
                            if 'distribution' in charts:
                                st.plotly_chart(charts['distribution'], use_container_width=True)
                            
                            if 'curves' in charts:
                                st.plotly_chart(charts['curves'], use_container_width=True)
                            
                            if 'gauges' in charts:
                                st.plotly_chart(charts['gauges'], use_container_width=True)
                            
            except Exception as e:
                st.error(f"蒙特卡洛模拟出错: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_picks_performance_tab():
    """📈 机会表现 - 追踪历史选股表现"""
    st.subheader("📈 每日机会历史表现")
    st.caption("追踪每日扫描出的机会后续表现，分析哪些特征与成功相关")
    
    try:
        from strategies.picks_tracker import (
            PicksPerformanceTracker, FeatureAnalyzer,
            record_todays_picks
        )
        
        tracker = PicksPerformanceTracker()
        analyzer = FeatureAnalyzer(tracker)
        
        # 操作区
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 更新收益数据", help="为缺少收益的记录计算前向收益"):
                with st.spinner("正在更新..."):
                    result = tracker.batch_update_returns(limit=50)
                    st.success(f"✅ 更新完成: {result['updated']}/{result['total']}")
        
        with col2:
            days = st.selectbox("分析周期", [30, 60, 90, 180], index=1)
        
        with col3:
            market = st.selectbox("市场", ["US", "CN", "全部"], index=0)
        
        # 表现汇总
        st.markdown("### 📊 表现汇总")
        
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        summary = tracker.get_performance_summary(
            start_date=start_date,
            end_date=end_date,
            market=market if market != "全部" else None
        )
        
        if summary.get('total_picks', 0) > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总机会数", summary.get('total_picks', 0))
            m2.metric("平均5日收益", f"{summary.get('avg_return_d5', 0)}%")
            m3.metric("5日胜率", f"{summary.get('win_rate_d5', 0)}%")
            m4.metric("平均最大涨幅", f"{summary.get('avg_max_gain', 'N/A')}%")
            
            # 最佳/最差选股
            col_best, col_worst = st.columns(2)
            with col_best:
                best = summary.get('best_pick')
                if best:
                    st.success(f"🏆 最佳: {best.get('symbol')} ({best.get('pick_date')}) +{best.get('return_d5')}%")
            with col_worst:
                worst = summary.get('worst_pick')
                if worst:
                    st.error(f"😢 最差: {worst.get('symbol')} ({worst.get('pick_date')}) {worst.get('return_d5')}%")
        else:
            st.info("📭 暂无足够的历史数据，请先记录每日机会")
        
        st.divider()
        
        # 特征分析
        st.markdown("### 🔬 特征重要性分析")
        
        importance = analyzer.feature_importance()
        
        if importance.get('n_samples', 0) > 20:
            # 相关性表
            corr = importance.get('correlations', {})
            if corr:
                corr_df = pd.DataFrame([
                    {'特征': k, '与5日收益相关性': v, 
                     '解读': '✅ 正相关' if v > 0.1 else ('❌ 负相关' if v < -0.1 else '➖ 弱相关')}
                    for k, v in corr.items()
                ])
                corr_df = corr_df.sort_values('与5日收益相关性', ascending=False)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            # 分类特征分析
            cat_analysis = importance.get('categorical_analysis', {})
            if cat_analysis:
                st.markdown("**分类特征影响:**")
                if 'heima_effect' in cat_analysis:
                    he = cat_analysis['heima_effect']
                    st.write(f"🐴 黑马信号: 有黑马 {he.get('heima_avg')}% vs 无黑马 {he.get('non_heima_avg')}% (提升 {he.get('lift')}%)")
                
                if 'new_discovery_effect' in cat_analysis:
                    ne = cat_analysis['new_discovery_effect']
                    st.write(f"🆕 新发现: 新 {ne.get('new_avg')}% vs 老 {ne.get('old_avg')}% (提升 {ne.get('lift')}%)")
        else:
            st.warning(f"样本不足 ({importance.get('n_samples', 0)} < 20)，无法进行特征分析")
        
        st.divider()
        
        # 策略有效性
        st.markdown("### 🎯 策略有效性排名")
        
        strategies = analyzer.strategy_effectiveness()
        
        if strategies:
            strategy_df = pd.DataFrame([
                {
                    '策略': name,
                    '选股数': stats['total_picks'],
                    '平均收益': f"{stats['avg_return_d5']}%",
                    '胜率': f"{stats['win_rate']}%",
                    'Sharpe-like': stats['sharpe_like'],
                    '最佳': f"{stats['best']}%",
                    '最差': f"{stats['worst']}%"
                }
                for name, stats in strategies.items()
            ])
            st.dataframe(strategy_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无策略表现数据")
            
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_strategy_optimizer_tab():
    """🎯 策略优化 - 自动寻找最优参数"""
    st.subheader("🎯 策略参数优化器")
    st.caption("通过历史数据自动寻找最优策略参数组合")
    
    try:
        from strategies.optimizer import (
            StrategyOptimizer, ContinuousOptimizer,
            StrategyConfig, optimize_strategies
        )
        
        optimizer = StrategyOptimizer()
        
        # 当前最优配置
        st.markdown("### 🏆 当前最优配置")
        
        best_config = optimizer.get_best_config()
        if best_config:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("策略名称", best_config.name[:15])
            col2.metric("BLUE日线阈值", best_config.blue_daily_min)
            col3.metric("BLUE周线阈值", best_config.blue_weekly_min)
            col4.metric("ADX阈值", best_config.adx_min)
            
            with st.expander("📋 完整配置"):
                st.json(best_config.to_dict())
        else:
            st.info("暂无保存的最优配置，请运行优化")
        
        st.divider()
        
        # 优化选项
        st.markdown("### 🔬 运行优化")
        
        opt_type = st.radio("优化方式", [
            "📊 比较预定义策略", 
            "🔍 网格搜索 (耗时较长)"
        ], horizontal=True)
        
        if st.button("🚀 开始优化", type="primary"):
            with st.spinner("正在优化策略参数..."):
                if "预定义" in opt_type:
                    results = optimizer.run_template_comparison()
                else:
                    results = optimizer.run_grid_search()
                
                if results:
                    st.success(f"✅ 优化完成！测试了 {len(results)} 种配置")
                    
                    # 显示结果表
                    results_df = pd.DataFrame([
                        {
                            '排名': r.rank,
                            '策略': r.config.name[:25],
                            '样本数': r.metrics.get('n_samples', 0),
                            '平均收益': f"{r.metrics.get('avg_return', 0)}%",
                            '胜率': f"{r.metrics.get('win_rate', 0)}%",
                            'Sharpe': r.metrics.get('sharpe_like', 0),
                            '综合得分': round(r.score, 1)
                        }
                        for r in results[:20]
                    ])
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # 保存最优
                    if st.button("💾 保存最优配置"):
                        if optimizer.save_best_config(results[0]):
                            st.success("✅ 已保存最优配置")
                        else:
                            st.error("保存失败")
                else:
                    st.warning("优化未产生有效结果，可能数据不足")
        
        st.divider()
        
        # 预定义策略模板
        st.markdown("### 📚 预定义策略模板")
        
        templates = StrategyOptimizer.STRATEGY_TEMPLATES
        template_df = pd.DataFrame([
            {
                '策略名称': name,
                'BLUE日线': cfg.blue_daily_min,
                'BLUE周线': cfg.blue_weekly_min,
                'ADX': cfg.adx_min,
                '黑马': '✅' if cfg.require_heima else '',
                '掘地': '✅' if cfg.require_juedi else '',
                '止损': f"{cfg.stop_loss_pct}%",
                '止盈': f"{cfg.take_profit_pct}%"
            }
            for name, cfg in templates.items()
        ])
        
        st.dataframe(template_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_historical_review():
    """历史复盘 - 查看某天信号的后续表现"""
    from services.signal_tracker_service import get_signal_performance_summary
    from db.database import get_scanned_dates
    
    st.subheader("📊 历史复盘")
    st.caption("选择一个历史扫描日期，查看当天信号的后续表现")
    
    col1, col2 = st.columns(2)
    
    with col1:
        market = st.selectbox("市场", ["US", "CN"], index=0, key="review_market")
    
    with col2:
        dates = get_scanned_dates(market=market)
        if not dates:
            st.warning("暂无扫描数据")
            return
        selected_date = st.selectbox("选择扫描日期", dates[:30], key="review_date")
    
    if st.button("📈 分析信号表现", type="primary", key="run_review"):
        with st.spinner(f"正在分析 {selected_date} 的信号表现..."):
            try:
                summary = get_signal_performance_summary(selected_date, market)
                
                if not summary:
                    st.warning("未找到该日期的信号数据")
                    return
                
                # 显示统计摘要
                st.success(f"✅ 分析完成！共 {summary.get('total_signals', 0)} 个信号")
                
                # 关键指标
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("5日胜率", f"{summary.get('win_rate_5d', 0):.1f}%")
                m2.metric("10日胜率", f"{summary.get('win_rate_10d', 0):.1f}%")
                m3.metric("20日胜率", f"{summary.get('win_rate_20d', 0):.1f}%")
                m4.metric("大赚 (>10%)", f"{summary.get('big_win_20d', 0)} 只")
                
                # 平均收益
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("5日平均", f"{summary.get('avg_5d', 0):+.2f}%")
                col_b.metric("10日平均", f"{summary.get('avg_10d', 0):+.2f}%")
                col_c.metric("20日平均", f"{summary.get('avg_20d', 0):+.2f}%")
                
                # 详细表格
                if summary.get('details'):
                    st.subheader("📋 信号明细")
                    
                    details_df = pd.DataFrame(summary['details'])
                    
                    # 选择显示的列
                    display_cols = ['symbol', 'name', 'entry_price', 'return_5d', 
                                   'return_10d', 'return_20d', 'max_gain', 'max_drawdown']
                    available_cols = [c for c in display_cols if c in details_df.columns]
                    
                    if available_cols:
                        display_df = details_df[available_cols].copy()
                        display_df.columns = ['股票', '名称', '入场价', '5日收益%', 
                                             '10日收益%', '20日收益%', '最大涨幅%', '最大回撤%'][:len(available_cols)]
                        
                        # 颜色编码
                        def color_returns(val):
                            if pd.isna(val):
                                return ''
                            try:
                                v = float(val)
                                if v > 0:
                                    return 'color: green'
                                elif v < 0:
                                    return 'color: red'
                            except:
                                pass
                            return ''
                        
                        st.dataframe(
                            display_df.style.applymap(color_returns, 
                                                     subset=[c for c in display_df.columns if '收益' in c or '涨幅' in c or '回撤' in c]),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # 收益分布图
                        if 'return_20d' in details_df.columns:
                            import plotly.express as px
                            fig = px.histogram(
                                details_df.dropna(subset=['return_20d']),
                                x='return_20d',
                                nbins=15,
                                title=f"{selected_date} 信号的 20 日收益分布",
                                labels={'return_20d': '20日收益率 (%)'}
                            )
                            fig.add_vline(x=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"分析出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_backtest_page():
    st.header("🧪 策略回测实验室 (Strategy Lab)")
    
    tab_param_lab, tab_single, tab_risk, tab_review, tab_picks, tab_optimizer = st.tabs([
        "🔬 参数实验室", 
        "📈 单股回测", 
        "🛡️ 风控计算器",
        "📊 历史复盘",
        "📈 机会表现",
        "🎯 策略优化"
    ])
    
    # === 参数实验室 Tab (新增) ===
    with tab_param_lab:
        render_parameter_lab()
    
    # === 机会表现 Tab (新增) ===
    with tab_picks:
        render_picks_performance_tab()
    
    # === 策略优化 Tab (新增) ===
    with tab_optimizer:
        render_strategy_optimizer_tab()

    
    # === 历史复盘 Tab (新增) ===
    with tab_review:
        render_historical_review()
    
    # === 风控计算器 Tab ===
    with tab_risk:
        st.subheader("🛡️ 仓位与风控计算器")
        st.caption("基于凯利公式和ATR计算最优仓位和止损")
        
        from backtest.risk_manager import RiskManager
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_capital = st.number_input("总资金", value=100000.0, step=10000.0)
            stock_price = st.number_input("股票价格", value=50.0, step=1.0)
        with col2:
            win_rate = st.slider("历史胜率%", 30, 80, 55) / 100
            avg_win = st.number_input("平均盈利%", value=8.0, step=1.0)
        with col3:
            avg_loss = st.number_input("平均亏损%", value=4.0, step=1.0)
            atr = st.number_input("ATR (可选)", value=2.0, step=0.5)
        
        if st.button("📊 计算仓位建议", key="calc_risk"):
            rm = RiskManager(total_capital=total_capital)
            
            # 计算建议
            rec = rm.recommend_position(
                symbol="INPUT",
                price=stock_price,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                atr=atr if atr > 0 else None
            )
            
            st.success("✅ 计算完成")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("建议股数", f"{rec['shares']} 股")
                st.metric("仓位比例", f"{rec['position_pct']:.1f}%")
            with col_b:
                st.metric("入场价", f"${rec['entry_price']:.2f}")
                st.metric("止损价", f"${rec['stop_loss']:.2f}")
            with col_c:
                st.metric("止盈价", f"${rec['take_profit']:.2f}")
                st.metric("风险回报比", f"1:{rec['risk_reward']}")
            
            # Kelly 公式解释
            with st.expander("📚 凯利公式说明"):
                kelly_raw = rm.calc_position_size_kelly(stock_price, win_rate, avg_win, avg_loss)
                st.markdown(f"""
                **凯利公式**: f* = W - (1-W)/R
                
                - 胜率 W = {win_rate*100:.0f}%
                - 盈亏比 R = {avg_win/avg_loss:.2f}
                - 原始Kelly仓位 = {kelly_raw.get('kelly_raw', 0):.1f}%
                - 调整后仓位 (1/4 Kelly) = {kelly_raw.get('kelly_adjusted', 0):.1f}%
                
                *使用分数凯利更保守,避免过度下注*
                """)
    
    # === 单股回测 Tab ===
    with tab_single:
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
    
    # 四个 Tab (新增 AutoML 和 Ensemble)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 统计ML", "🧠 深度学习", "💬 LLM智能", "🔧 AutoML/集成"])
    
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
    
    with tab4:
        st.subheader("🔧 AutoML & 模型集成")
        st.info("自动化模型选择和多模型融合")
        
        automl_tab1, automl_tab2 = st.tabs(["🤖 AutoML", "🔗 集成预测"])
        
        with automl_tab1:
            st.markdown("### 自动模型选择")
            st.caption("自动训练多种模型并选择最优")
            
            col1, col2 = st.columns(2)
            with col1:
                automl_market = st.selectbox("市场", ["US", "CN"], key="automl_market")
            with col2:
                cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="cv_folds")
            
            if st.button("🚀 运行 AutoML", key="run_automl"):
                with st.spinner("正在训练多个模型..."):
                    try:
                        from ml.ensemble import AutoML
                        from db.database import query_scan_results
                        from ml.feature_engineering import prepare_training_data
                        
                        # 获取数据
                        signals = query_scan_results(market=automl_market, limit=300)
                        if not signals or len(signals) < 50:
                            st.warning("数据不足")
                        else:
                            # 准备特征
                            X_list = []
                            y_list = []
                            for s in signals:
                                if s.get('blue_daily') is not None:
                                    X_list.append({
                                        'blue_daily': s.get('blue_daily', 0) or 0,
                                        'blue_weekly': s.get('blue_weekly', 0) or 0,
                                        'adx': s.get('adx', 0) or 0,
                                        'volatility': s.get('volatility', 0) or 0,
                                    })
                                    # 简化标签
                                    y_list.append(1 if (s.get('blue_daily', 0) or 0) > 100 else 0)
                            
                            X = pd.DataFrame(X_list).fillna(0)
                            y = np.array(y_list)
                            
                            if len(X) < 30:
                                st.warning("特征数据不足")
                            else:
                                automl = AutoML(market=automl_market)
                                result = automl.auto_train(X.values, y, cv_folds=cv_folds)
                                
                                if 'error' in result:
                                    st.error(result['error'])
                                else:
                                    st.success(f"✅ 最优模型: **{result['best_model_type']}** (CV Score: {result['best_cv_score']:.4f})")
                                    
                                    # 结果表格
                                    results_df = pd.DataFrame(result['all_results'])
                                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                                    
                                    # 保存到 session
                                    st.session_state['automl_instance'] = automl
                                    st.info("💡 可在「集成预测」Tab 使用这些模型创建集成")
                                    
                    except Exception as e:
                        st.error(f"AutoML 出错: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with automl_tab2:
            st.markdown("### 模型集成预测")
            st.caption("融合多个模型的预测结果")
            
            if 'automl_instance' in st.session_state:
                automl = st.session_state['automl_instance']
                
                # 创建集成
                if st.button("创建集成", key="create_ensemble"):
                    try:
                        ensemble = automl.create_ensemble()
                        st.session_state['ensemble'] = ensemble
                        st.success("✅ 集成已创建!")
                        
                        # 显示集成摘要
                        summary = ensemble.summary()
                        st.dataframe(summary, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"创建集成失败: {e}")
                
                if 'ensemble' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 使用集成预测")
                    
                    symbol = st.text_input("输入股票代码", value="AAPL", key="ensemble_symbol")
                    
                    if st.button("预测", key="ensemble_predict"):
                        st.info("正在预测... (演示)")
                        # 这里可以接入实际预测逻辑
                        prob = np.random.uniform(0.4, 0.8)
                        st.metric("盈利概率", f"{prob:.1%}")
            else:
                st.info("请先在「AutoML」Tab 训练模型")


# --- 博主推荐追踪页面 ---

def render_external_strategies_tab():
    """📊 外部策略 - TradingView 和社区策略"""
    st.subheader("📊 外部策略库")
    st.caption("TradingView 热门策略、社区策略、博主策略")
    
    try:
        from strategies.aggregator import StrategyAggregator, StrategySource, StrategyCategory
        from strategies.implementations import list_strategies
        
        aggregator = StrategyAggregator()
        
        # TradingView 热门策略
        st.markdown("### 📈 TradingView 热门策略")
        
        tv_strategies = aggregator.tv_scraper.get_popular_strategies()
        
        if tv_strategies:
            tv_df = pd.DataFrame([
                {
                    '策略名称': s.name,
                    '类别': s.category.value if isinstance(s.category, StrategyCategory) else s.category,
                    '入场规则': s.entry_rules[:50] + '...' if len(s.entry_rules) > 50 else s.entry_rules,
                    '出场规则': s.exit_rules[:50] + '...' if len(s.exit_rules) > 50 else s.exit_rules,
                    '声称胜率': f"{s.claimed_win_rate}%",
                    '主要指标': ', '.join(s.indicators[:3])
                }
                for s in tv_strategies
            ])
            
            st.dataframe(tv_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 可回测的策略
        st.markdown("### 🧪 可回测策略")
        st.caption("这些策略已实现完整逻辑，可直接回测")
        
        impl_strategies = list_strategies()
        
        impl_df = pd.DataFrame([
            {
                '策略ID': s['id'],
                '策略名称': s['name'],
                '描述': s['description'],
                '使用指标': ', '.join(s.get('indicators', []))
            }
            for s in impl_strategies
        ])
        
        st.dataframe(impl_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 博主列表
        st.markdown("### 👤 知名博主")
        
        authors = aggregator.get_all_authors()
        
        if authors:
            author_df = pd.DataFrame([
                {
                    '博主': a.name,
                    '平台': a.platform.value if isinstance(a.platform, StrategySource) else a.platform,
                    '专长': a.specialty,
                    '粉丝数': f"{a.followers:,}" if a.followers else 'N/A',
                    '简介': a.description[:30] + '...' if len(a.description) > 30 else a.description
                }
                for a in authors
            ])
            
            st.dataframe(author_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_article_crawler_tab():
    """🔍 文章爬取与策略分析 - 自动爬取量化博客文章"""
    st.subheader("🔍 量化博客文章爬取")
    st.caption("自动爬取中英文量化博客，分析其中的策略并回测验证")
    
    # 数据源列表
    st.markdown("### 📚 数据源")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🇺🇸 英文源**")
        en_sources = [
            ("Quantocracy", "量化聚合站", "https://quantocracy.com/"),
            ("Alpha Architect", "因子研究", "https://alphaarchitect.com/blog/"),
            ("Quantpedia", "策略库", "https://quantpedia.com/blog/"),
            ("SSRN Finance", "学术论文", "https://papers.ssrn.com/"),
            ("QuantStart", "量化教程", "https://www.quantstart.com/"),
        ]
        for name, cat, url in en_sources:
            st.markdown(f"• **{name}** - {cat}")
    
    with col2:
        st.markdown("**🇨🇳 中文源**")
        cn_sources = [
            ("雪球热帖", "社区热门", "https://xueqiu.com/"),
            ("聚宽社区", "量化策略", "https://www.joinquant.com/"),
            ("米筐研究", "量化策略", "https://www.ricequant.com/"),
            ("同花顺量化", "量化资讯", "https://quant.10jqka.com.cn/"),
        ]
        for name, cat, url in cn_sources:
            st.markdown(f"• **{name}** - {cat}")
    
    st.divider()
    
    # 爬取控制
    st.markdown("### 🚀 爬取文章")
    
    col_fetch1, col_fetch2 = st.columns(2)
    
    with col_fetch1:
        fetch_lang = st.radio("选择语言", ["全部", "英文", "中文"], horizontal=True)
    
    with col_fetch2:
        use_llm = st.checkbox("使用 LLM 分析策略", value=False, 
                              help="使用 GPT-4 提取更精准的策略，需要 OPENAI_API_KEY")
    
    if st.button("🔍 开始爬取", type="primary"):
        try:
            from services.blogger_tracker import (
                ArticleFetcher, StrategyExtractor, StrategyBacktester,
                BloggerTrackerDB
            )
            
            with st.spinner("正在爬取文章..."):
                fetcher = ArticleFetcher()
                results = fetcher.fetch_all(save=True)
                
                en_count = len(results.get('en', []))
                cn_count = len(results.get('cn', []))
                
                st.success(f"✅ 爬取完成! 英文: {en_count} 篇, 中文: {cn_count} 篇")
            
            # 分析策略
            if en_count + cn_count > 0:
                with st.spinner("正在分析策略..."):
                    db = BloggerTrackerDB()
                    extractor = StrategyExtractor()
                    
                    articles = db.get_recent_articles(days=1)
                    strategies_found = 0
                    
                    progress = st.progress(0)
                    for i, article in enumerate(articles):
                        if use_llm:
                            strategy = extractor.extract_strategy_with_llm(article)
                        else:
                            strategy = extractor.extract_strategy_rule_based(article)
                        
                        if strategy:
                            db.save_strategy(strategy)
                            strategies_found += 1
                        
                        progress.progress((i + 1) / len(articles))
                    
                    progress.empty()
                    st.success(f"✅ 分析完成! 提取了 {strategies_found} 个策略")
                    
                    # 回测
                    if strategies_found > 0:
                        with st.spinner("正在回测策略..."):
                            backtester = StrategyBacktester()
                            strategies_list = db.get_strategies_with_backtests()
                            
                            backtest_count = 0
                            for strategy in strategies_list:
                                if strategy.get('total_return') is None:
                                    result = backtester.backtest_extracted_strategy(strategy)
                                    if result:
                                        db.save_backtest(result)
                                        backtest_count += 1
                            
                            st.success(f"✅ 回测完成! 回测了 {backtest_count} 个策略")
        
        except ImportError as e:
            st.error(f"需要安装依赖: {e}")
            st.code("pip install beautifulsoup4 lxml")
        except Exception as e:
            st.error(f"爬取失败: {e}")
    
    st.divider()
    
    # 显示已爬取的文章
    st.markdown("### 📰 最新文章")
    
    try:
        from services.blogger_tracker import BloggerTrackerDB
        
        db = BloggerTrackerDB()
        articles = db.get_recent_articles(days=7)
        
        if articles:
            article_df = pd.DataFrame([
                {
                    '来源': a['source'],
                    '标题': a['title'][:50] + '...' if len(a['title']) > 50 else a['title'],
                    '作者': a['author'],
                    '类别': a['category'],
                    '语言': '🇨🇳' if a['language'] == 'cn' else '🇺🇸',
                    '日期': a['publish_date'],
                    '已分析': '✅' if a.get('analyzed') else '❌'
                }
                for a in articles[:30]
            ])
            
            st.dataframe(article_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无文章，请点击「开始爬取」")
    except Exception as e:
        st.warning(f"加载文章失败: {e}")
    
    st.divider()
    
    # 策略排行榜
    st.markdown("### 🏆 策略排行榜")
    st.caption("根据回测结果排序，展示最有效的策略")
    
    try:
        from services.blogger_tracker import BloggerTrackerDB
        
        db = BloggerTrackerDB()
        strategies = db.get_strategies_with_backtests()
        
        # 只显示有回测结果的
        strategies_with_bt = [s for s in strategies if s.get('total_return') is not None]
        
        if strategies_with_bt:
            # 按收益排序
            strategies_with_bt.sort(key=lambda x: x.get('sharpe_ratio', 0) or 0, reverse=True)
            
            strat_df = pd.DataFrame([
                {
                    '策略名称': s['strategy_name'][:40],
                    '类型': s['strategy_type'],
                    '来源文章': s.get('article_title', '')[:30] if s.get('article_title') else '-',
                    '总收益': f"{s['total_return']:.1f}%" if s.get('total_return') else '-',
                    'Sharpe': f"{s['sharpe_ratio']:.2f}" if s.get('sharpe_ratio') else '-',
                    '最大回撤': f"{s['max_drawdown']:.1f}%" if s.get('max_drawdown') else '-',
                    '胜率': f"{s['win_rate']:.0f}%" if s.get('win_rate') else '-',
                    '有效': '✅' if s.get('is_profitable') else '❌'
                }
                for s in strategies_with_bt[:20]
            ])
            
            st.dataframe(
                strat_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "策略名称": st.column_config.TextColumn("策略名称", width="large"),
                    "类型": st.column_config.TextColumn("类型", width="small"),
                    "来源文章": st.column_config.TextColumn("来源", width="medium"),
                    "总收益": st.column_config.TextColumn("收益", width="small"),
                    "Sharpe": st.column_config.TextColumn("Sharpe", width="small"),
                    "最大回撤": st.column_config.TextColumn("回撤", width="small"),
                    "胜率": st.column_config.TextColumn("胜率", width="small"),
                    "有效": st.column_config.TextColumn("有效", width="small"),
                }
            )
            
            # 统计
            profitable_count = sum(1 for s in strategies_with_bt if s.get('is_profitable'))
            st.info(f"📊 统计: {len(strategies_with_bt)} 个策略已回测, {profitable_count} 个盈利 ({profitable_count/len(strategies_with_bt)*100:.0f}%)")
        else:
            st.info("暂无回测结果，请先爬取并分析文章")
    except Exception as e:
        st.warning(f"加载策略失败: {e}")


def render_strategy_backtest_tab():
    """🧪 策略回测 - 回测外部策略"""
    st.subheader("🧪 外部策略回测")
    st.caption("选择策略和股票，验证策略有效性")
    
    try:
        from strategies.implementations import (
            list_strategies, backtest_external_strategy, get_strategy
        )
        
        # 策略选择
        strategies = list_strategies()
        strategy_options = {s['name']: s['id'] for s in strategies}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_name = st.selectbox("选择策略", list(strategy_options.keys()))
            selected_id = strategy_options.get(selected_name)
        
        with col2:
            symbol = st.text_input("股票代码", value="NVDA").upper().strip()
        
        with col3:
            days = st.selectbox("回测周期", [90, 180, 365, 730], index=2)
        
        # 显示策略详情
        strategy_info = next((s for s in strategies if s['id'] == selected_id), None)
        if strategy_info:
            st.info(f"**{strategy_info['name']}**: {strategy_info['description']}")
            st.caption(f"使用指标: {', '.join(strategy_info.get('indicators', []))}")
        
        # 运行回测
        if st.button("🚀 运行回测", type="primary"):
            with st.spinner(f"正在回测 {selected_name} on {symbol}..."):
                result = backtest_external_strategy(selected_id, symbol, days=days)
                
                if 'error' in result:
                    st.error(result['error'])
                elif result.get('total_signals', 0) == 0:
                    st.warning("该策略在此期间未产生任何信号")
                else:
                    st.success("✅ 回测完成!")
                    
                    # 显示结果
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("总信号数", result.get('total_signals', 0))
                    m2.metric("完成交易", result.get('completed_trades', 0))
                    m3.metric("胜率", f"{result.get('win_rate', 0)}%")
                    m4.metric("总收益", f"{result.get('total_return', 0)}%")
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("平均收益", f"{result.get('avg_return', 0)}%")
                    m6.metric("最大盈利", f"{result.get('max_gain', 0)}%")
                    m7.metric("最大亏损", f"{result.get('max_loss', 0)}%")
                    m8.metric("Sharpe", result.get('sharpe', 0))
        
        st.divider()
        
        # 批量对比
        st.markdown("### 📊 策略对比")
        st.caption("比较多个策略在同一股票上的表现")
        
        compare_symbol = st.text_input("对比股票", value="AAPL", key="compare_symbol").upper()
        
        if st.button("📊 对比所有策略"):
            with st.spinner("正在对比..."):
                results = []
                
                for s in strategies:
                    try:
                        r = backtest_external_strategy(s['id'], compare_symbol, days=365)
                        if 'error' not in r and r.get('completed_trades', 0) > 0:
                            results.append({
                                '策略': s['name'],
                                '信号数': r.get('total_signals', 0),
                                '交易数': r.get('completed_trades', 0),
                                '胜率': f"{r.get('win_rate', 0)}%",
                                '总收益': f"{r.get('total_return', 0)}%",
                                'Sharpe': r.get('sharpe', 0)
                            })
                    except:
                        pass
                
                if results:
                    compare_df = pd.DataFrame(results)
                    # 按总收益排序
                    compare_df['_sort'] = compare_df['总收益'].str.replace('%', '').astype(float)
                    compare_df = compare_df.sort_values('_sort', ascending=False).drop('_sort', axis=1)
                    
                    st.dataframe(compare_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("没有足够数据进行对比")
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_blogger_page():
    """📢 博主推荐追踪页面"""
    st.header("📢 博主推荐追踪")
    st.caption("追踪知名博主的股票推荐，计算收益表现")
    
    from db.database import (
        init_blogger_tables, get_all_bloggers, add_blogger, delete_blogger,
        get_recommendations, add_recommendation, delete_recommendation, get_blogger_stats
    )
    from services.blogger_service import get_recommendations_with_returns, get_blogger_performance
    
    # 确保表存在
    init_blogger_tables()
    
    tab_bloggers, tab_recs, tab_perf, tab_external, tab_backtest, tab_crawler = st.tabs([
        "👤 博主管理",
        "📝 推荐记录", 
        "🏆 业绩排行",
        "📊 外部策略",
        "🧪 策略回测",
        "🔍 文章爬取"
    ])
    
    # === Tab 4: 外部策略 ===
    with tab_external:
        render_external_strategies_tab()
    
    # === Tab 5: 策略回测 ===
    with tab_backtest:
        render_strategy_backtest_tab()
    
    # === Tab 6: 文章爬取与策略分析 ===
    with tab_crawler:
        render_article_crawler_tab()
    
    # === Tab 1: 博主管理 ===
    with tab_bloggers:
        st.subheader("博主列表")
        
        bloggers = get_all_bloggers()
        
        if bloggers:
            for b in bloggers:
                with st.expander(f"**{b['name']}** ({b.get('platform', 'N/A')})"):
                    st.write(f"专长: {b.get('specialty', 'N/A')}")
                    st.write(f"主页: {b.get('url', 'N/A')}")
                    if is_admin():
                        if st.button(f"🗑️ 删除", key=f"del_blogger_{b['id']}"):
                            delete_blogger(b['id'])
                            st.success("已删除")
                            st.rerun()
        else:
            st.info("暂无博主，请添加")
        
        st.divider()
        
        if is_admin():
            st.subheader("➕ 添加博主")
            with st.form("add_blogger_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_name = st.text_input("博主名称*", placeholder="如：唐朝")
                    new_platform = st.selectbox("平台", ["雪球", "微博", "抖音", "Twitter", "YouTube", "其他"])
                with col2:
                    new_specialty = st.selectbox("专长", ["A股", "美股", "港股", "混合"])
                    new_url = st.text_input("主页链接", placeholder="https://...")
                
                if st.form_submit_button("添加博主", type="primary"):
                    if new_name:
                        add_blogger(new_name, platform=new_platform, specialty=new_specialty, url=new_url)
                        st.success(f"✅ 已添加博主: {new_name}")
                        st.rerun()
                    else:
                        st.error("请输入博主名称")
    
    # === Tab 2: 推荐记录 ===
    with tab_recs:
        st.subheader("推荐记录")
        
        bloggers = get_all_bloggers()
        
        if not bloggers:
            st.warning("请先添加博主")
        else:
            # 筛选
            col1, col2 = st.columns(2)
            with col1:
                filter_blogger = st.selectbox(
                    "选择博主",
                    options=[None] + [b['id'] for b in bloggers],
                    format_func=lambda x: "全部" if x is None else next((b['name'] for b in bloggers if b['id'] == x), x)
                )
            with col2:
                filter_market = st.selectbox("市场", ["全部", "CN", "US"])
            
            # 获取并显示推荐
            recs = get_recommendations_with_returns(
                blogger_id=filter_blogger,
                market=None if filter_market == "全部" else filter_market,
                limit=50
            )
            
            if recs:
                rec_df = pd.DataFrame(recs)
                display_cols = ['blogger_name', 'ticker', 'rec_date', 'rec_type', 'rec_price', 'current_price', 'return_pct', 'days_held']
                display_cols = [c for c in display_cols if c in rec_df.columns]
                
                st.dataframe(
                    rec_df[display_cols],
                    column_config={
                        'blogger_name': '博主',
                        'ticker': '股票',
                        'rec_date': '推荐日期',
                        'rec_type': '类型',
                        'rec_price': '推荐价',
                        'current_price': '现价',
                        'return_pct': st.column_config.NumberColumn('收益%', format="%.2f%%"),
                        'days_held': '持有天数'
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("暂无推荐记录")
            
            st.divider()
            
            # 添加推荐
            if is_admin():
                st.subheader("➕ 添加推荐")
                with st.form("add_rec_form"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rec_blogger = st.selectbox(
                            "博主*",
                            options=[b['id'] for b in bloggers],
                            format_func=lambda x: next((b['name'] for b in bloggers if b['id'] == x), x)
                        )
                        rec_ticker = st.text_input("股票代码*", placeholder="如: 600519 或 AAPL")
                    with col2:
                        rec_market = st.selectbox("市场", ["CN", "US"])
                        rec_date = st.date_input("推荐日期", value=datetime.now())
                    with col3:
                        rec_type = st.selectbox("类型", ["BUY", "SELL", "HOLD"])
                        rec_price = st.number_input("推荐价格 (可选)", min_value=0.0, step=0.01)
                    
                    rec_notes = st.text_area("推荐理由", height=80)
                    
                    if st.form_submit_button("添加推荐", type="primary"):
                        if rec_ticker and rec_blogger:
                            add_recommendation(
                                blogger_id=rec_blogger,
                                ticker=rec_ticker,
                                market=rec_market,
                                rec_date=rec_date.strftime('%Y-%m-%d'),
                                rec_price=rec_price if rec_price > 0 else None,
                                rec_type=rec_type,
                                notes=rec_notes
                            )
                            st.success(f"✅ 已添加推荐: {rec_ticker}")
                            st.rerun()
                        else:
                            st.error("请填写必填项")
    
    # === Tab 3: 业绩排行 ===
    with tab_perf:
        st.subheader("🏆 博主业绩排行")
        
        if st.button("🔄 刷新统计"):
            st.cache_data.clear()
        
        perf = get_blogger_performance()
        
        if perf:
            perf_df = pd.DataFrame(perf)
            
            # 高亮显示
            st.dataframe(
                perf_df[['name', 'platform', 'rec_count', 'win_rate', 'avg_return', 'total_return']],
                column_config={
                    'name': '博主',
                    'platform': '平台',
                    'rec_count': '推荐数',
                    'win_rate': st.column_config.NumberColumn('胜率%', format="%.1f%%"),
                    'avg_return': st.column_config.NumberColumn('平均收益%', format="%.2f%%"),
                    'total_return': st.column_config.NumberColumn('累计收益%', format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # 胜率图表
            if len(perf_df) > 0 and perf_df['rec_count'].sum() > 0:
                import plotly.express as px
                fig = px.bar(
                    perf_df[perf_df['rec_count'] > 0],
                    x='name', y='avg_return',
                    title="博主平均收益率排名",
                    color='win_rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无数据，请先添加博主和推荐记录")


# ==================== V3 合并页面 ====================

def render_signal_center_page():
    """📈 信号中心 - 合并: 信号追踪 + 信号验证 + 健康监控"""
    st.header("📈 信号中心")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👁️ 观察追踪",
        "🩺 信号健康", 
        "📊 信号追踪", 
        "📉 信号验证", 
        "📧 历史追踪",
        "🔄 Baseline对比"
    ])
    
    with tab1:
        render_watchlist_tracking_tab()
    
    with tab2:
        render_signal_health_monitor()
    
    with tab3:
        render_signal_tracker_page()
    
    with tab4:
        render_signal_performance_page()
    
    with tab5:
        render_historical_tracking_tab()
    
    with tab6:
        render_baseline_comparison_page()


def render_watchlist_tracking_tab():
    """👁️ 观察列表追踪 - 持续跟踪已发现的机会股票"""
    import plotly.graph_objects as go
    
    st.subheader("👁️ 观察列表追踪")
    st.caption("持续关注已发现机会的股票，实时监控信号变化、卖出点、做T时机")
    
    # 侧边栏设置
    with st.sidebar:
        st.divider()
        st.subheader("👁️ 追踪设置")
        
        market_choice = st.radio(
            "市场", 
            ["🇺🇸 美股", "🇨🇳 A股"], 
            horizontal=True, 
            key="watchlist_market"
        )
        market = "US" if "美股" in market_choice else "CN"
    
    try:
        from services.signal_tracker import (
            get_watchlist, add_to_watchlist, remove_from_watchlist,
            get_signal_history, analyze_sell_signals, analyze_t_trade_opportunity,
            get_unread_alerts, mark_alert_read, get_tracking_summary, record_signal
        )
    except ImportError as e:
        st.error(f"追踪模块导入失败: {e}")
        return
    
    # 获取数据
    watchlist = get_watchlist(market=market)
    tracking_summary = get_tracking_summary(market=market)
    unread_alerts = get_unread_alerts(market=market)
    
    # === 顶部: 追踪概览 ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👁️ 观察中", f"{len(watchlist)} 只")
    
    with col2:
        buy_signals = tracking_summary.get('buy_signals', 0)
        st.metric("🟢 买入信号", f"{buy_signals} 条", 
                  delta="有机会" if buy_signals > 0 else None)
    
    with col3:
        sell_signals = tracking_summary.get('sell_signals', 0)
        st.metric("🔴 卖出信号", f"{sell_signals} 条",
                  delta="需关注" if sell_signals > 0 else None,
                  delta_color="inverse" if sell_signals > 0 else "off")
    
    with col4:
        st.metric("🔔 未读提醒", f"{len(unread_alerts)} 条")
    
    st.divider()
    
    # === 未读提醒 ===
    if unread_alerts:
        with st.expander(f"🔔 未读提醒 ({len(unread_alerts)} 条)", expanded=True):
            for alert in unread_alerts[:10]:
                urgency_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(alert['urgency'], '⚪')
                
                col1, col2, col3 = st.columns([2, 5, 1])
                with col1:
                    st.markdown(f"**{urgency_icon} {alert['symbol']}**")
                with col2:
                    st.markdown(f"{alert['message']}")
                    st.caption(f"{alert['alert_date']} | {alert['alert_type']}")
                with col3:
                    if st.button("✓", key=f"read_{alert['id']}"):
                        mark_alert_read(alert['id'])
                        st.rerun()
    
    # === 添加观察 ===
    with st.expander("➕ 添加股票到观察列表", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_symbol = st.text_input("股票代码", placeholder="NVDA / 600519.SH", key="add_symbol")
            entry_price = st.number_input("入场价", min_value=0.0, step=0.01, key="add_entry")
        
        with col2:
            target_price = st.number_input("目标价 (止盈)", min_value=0.0, step=0.01, key="add_target")
            stop_loss = st.number_input("止损价", min_value=0.0, step=0.01, key="add_stop")
        
        notes = st.text_input("备注", placeholder="买入理由...", key="add_notes")
        
        if st.button("➕ 添加", type="primary"):
            if new_symbol:
                add_to_watchlist(
                    symbol=new_symbol.upper(),
                    market=market,
                    entry_price=entry_price if entry_price > 0 else None,
                    target_price=target_price if target_price > 0 else None,
                    stop_loss=stop_loss if stop_loss > 0 else None,
                    notes=notes
                )
                st.success(f"✅ {new_symbol.upper()} 已添加")
                st.rerun()
            else:
                st.warning("请输入股票代码")
    
    # === 观察列表详情 ===
    if not watchlist:
        st.info("👆 观察列表为空，点击上方添加股票开始追踪")
        return
    
    st.markdown("### 📋 观察列表详情")
    
    # 获取最新扫描数据
    from db.database import query_scan_results, get_scanned_dates
    dates = get_scanned_dates(market=market)
    latest_date = dates[0] if dates else None
    latest_scan = {}
    
    if latest_date:
        scan_results = query_scan_results(scan_date=latest_date, market=market, limit=1000)
        for r in scan_results:
            latest_scan[r['symbol']] = r
    
    # 为每只股票创建追踪卡片
    for item in watchlist:
        symbol = item['symbol']
        entry_price = item.get('entry_price', 0) or 0
        target_price = item.get('target_price', 0) or (entry_price * 1.15 if entry_price else 0)
        stop_loss = item.get('stop_loss', 0) or (entry_price * 0.92 if entry_price else 0)
        added_date = item.get('added_date', '')
        notes = item.get('notes', '')
        
        # 获取最新数据
        scan_data = latest_scan.get(symbol, {})
        current_price = scan_data.get('price', entry_price) or entry_price
        blue_daily = scan_data.get('blue_daily', 0) or 0
        blue_weekly = scan_data.get('blue_weekly', 0) or 0
        heima = scan_data.get('heima', 0) or 0
        volume = scan_data.get('volume', 0) or 0
        
        # 计算盈亏
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        # 分析卖出信号
        sell_analysis = analyze_sell_signals(
            symbol, market, current_price, entry_price,
            target_price, stop_loss, blue_daily, blue_weekly
        )
        
        # 卡片样式
        urgency = sell_analysis['sell_urgency']
        border_color = {
            'critical': '#ff4444', 'high': '#ff8800', 
            'medium': '#ffcc00', 'low': '#44ff44', 'none': '#666666'
        }.get(urgency, '#666666')
        
        st.markdown(f"""
        <div style="border-left: 4px solid {border_color}; padding-left: 15px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        # 标题行
        col1, col2, col3 = st.columns([3, 5, 2])
        
        with col1:
            urgency_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'none': '⚪'}.get(urgency, '⚪')
            st.markdown(f"### {urgency_icon} {symbol}")
            st.caption(f"加入: {added_date}")
            if notes:
                st.caption(f"📝 {notes}")
        
        with col2:
            # 信号状态
            sub_cols = st.columns(4)
            price_symbol = "¥" if market == "CN" else "$"
            
            with sub_cols[0]:
                pnl_color = "green" if pnl_pct >= 0 else "red"
                st.markdown(f"**现价**")
                st.markdown(f"{price_symbol}{current_price:.2f}")
                st.markdown(f"<span style='color:{pnl_color}'>{pnl_pct:+.1f}%</span>", unsafe_allow_html=True)
            
            with sub_cols[1]:
                blue_color = "green" if blue_daily >= 100 else ("orange" if blue_daily >= 50 else "red")
                st.markdown(f"**日BLUE**")
                st.markdown(f"<span style='color:{blue_color}'>{blue_daily:.0f}</span>", unsafe_allow_html=True)
            
            with sub_cols[2]:
                st.markdown(f"**周BLUE**")
                st.markdown(f"{blue_weekly:.0f}")
            
            with sub_cols[3]:
                st.markdown(f"**黑马**")
                st.markdown("🐴" if heima else "-")
        
        with col3:
            # 交易计划
            st.markdown(f"🎯 {price_symbol}{target_price:.2f}")
            st.markdown(f"🛑 {price_symbol}{stop_loss:.2f}")
        
        # 卖出建议
        if sell_analysis['should_sell'] or sell_analysis['reasons']:
            with st.container():
                action_text = sell_analysis.get('recommended_action', 'hold')
                action_display = {
                    'sell_now': '🔴 建议立即卖出',
                    'take_profit': '🟢 已达止盈目标',
                    'consider_sell': '🟡 考虑卖出',
                    'consider_partial_sell': '🟡 考虑部分卖出',
                    'hold': '✅ 继续持有'
                }.get(action_text, '⚪ ' + action_text)
                
                st.markdown(f"**{action_display}**")
                
                for reason in sell_analysis['reasons']:
                    st.markdown(f"  • {reason}")
        
        # 操作按钮
        btn_cols = st.columns([1, 1, 1, 3])
        
        with btn_cols[0]:
            if st.button("📊 详情", key=f"detail_{symbol}"):
                st.session_state['stock_symbol'] = symbol
                st.info(f"请前往「个股查询」查看 {symbol} 详情")
        
        with btn_cols[1]:
            if st.button("💰 模拟买", key=f"sim_buy_{symbol}"):
                st.info("请前往「组合管理」执行模拟交易")
        
        with btn_cols[2]:
            if st.button("❌ 移除", key=f"del_{symbol}"):
                remove_from_watchlist(symbol, market)
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
    
    # === 信号历史图表 ===
    st.markdown("### 📈 信号历史对比")
    
    if len(watchlist) > 0:
        selected_symbol = st.selectbox(
            "选择股票查看历史",
            [w['symbol'] for w in watchlist],
            key="history_select"
        )
        
        if selected_symbol:
            history = get_signal_history(selected_symbol, market, days=30)
            
            if history:
                hist_df = pd.DataFrame(history)
                hist_df['record_date'] = pd.to_datetime(hist_df['record_date'])
                hist_df = hist_df.sort_values('record_date')
                
                # 创建图表
                fig = go.Figure()
                
                # 价格线
                if 'price' in hist_df.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_df['record_date'],
                        y=hist_df['price'],
                        name='价格',
                        line=dict(color='white', width=2),
                        yaxis='y'
                    ))
                
                # BLUE 指标
                if 'blue_daily' in hist_df.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_df['record_date'],
                        y=hist_df['blue_daily'],
                        name='日BLUE',
                        line=dict(color='#00ff88', width=1.5),
                        yaxis='y2'
                    ))
                
                if 'blue_weekly' in hist_df.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_df['record_date'],
                        y=hist_df['blue_weekly'],
                        name='周BLUE',
                        line=dict(color='#ffaa00', width=1.5),
                        yaxis='y2'
                    ))
                
                # 买入线 (BLUE=100)
                fig.add_hline(y=100, line_dash="dash", line_color="green", 
                              annotation_text="BLUE买入线", yref='y2')
                
                fig.update_layout(
                    title=f"{selected_symbol} 信号历史 (30日)",
                    xaxis_title="日期",
                    yaxis=dict(title="价格", side='left'),
                    yaxis2=dict(title="BLUE", overlaying='y', side='right'),
                    template='plotly_dark',
                    height=400,
                    legend=dict(x=0, y=1.1, orientation='h')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"暂无 {selected_symbol} 的历史数据")


def render_historical_tracking_tab():
    """📧 历史信号追踪 - 类似邮件报告的内容"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.subheader("📧 历史信号追踪报告")
    st.caption("追踪过去30天每个信号的后续表现")
    
    # 侧边栏设置
    with st.sidebar:
        st.subheader("📧 追踪设置")
        
        market_choice = st.radio(
            "市场", 
            ["🇺🇸 美股", "🇨🇳 A股"], 
            horizontal=True, 
            key="hist_track_market"
        )
        market = "US" if "美股" in market_choice else "CN"
        
        days = st.slider("追踪天数", 7, 60, 30, key="hist_track_days")
        min_blue = st.slider("最低 BLUE 阈值", 100, 180, 130, key="hist_track_blue")
        
        generate_btn = st.button("📊 生成报告", type="primary", use_container_width=True)
    
    if not generate_btn:
        st.info("👈 设置参数后点击「生成报告」查看历史信号表现")
        
        st.markdown("""
        ### 📋 报告内容
        
        - **信号统计**: 总数、胜率、平均收益
        - **各周期表现**: D+1, D+3, D+5, D+10 收益
        - **最佳/最差信号**: 表现最好和最差的股票
        - **详细列表**: 每个信号的完整表现
        
        ### 💡 说明
        
        - BLUE ≥ 130 被视为有效信号
        - 胜率 = 当前盈利的信号占比
        - 各周期收益 = 信号后第N个交易日的累计收益
        """)
        return
    
    # 获取数据
    from db.database import query_scan_results, get_scanned_dates
    from data_fetcher import get_stock_data
    
    with st.spinner("获取历史信号..."):
        dates = get_scanned_dates(market=market)
        if not dates:
            st.error("没有找到扫描数据")
            return
        
        all_signals = []
        for date in dates[:days]:
            results = query_scan_results(scan_date=date, market=market, limit=100)
            for r in results:
                blue = r.get('blue_daily', 0) or 0
                if blue >= min_blue:
                    all_signals.append({
                        'symbol': r['symbol'],
                        'signal_date': date,
                        'signal_price': r.get('price', 0),
                        'blue': blue,
                        'adx': r.get('adx', 0) or 0,
                        'is_heima': r.get('is_heima', False),
                        'is_juedi': r.get('is_juedi', False),
                        'company_name': r.get('company_name', '') or ''
                    })
        
        st.success(f"找到 {len(all_signals)} 个信号")
    
    if not all_signals:
        st.warning("没有找到符合条件的信号")
        return
    
    # 计算收益
    with st.spinner("计算信号收益 (可能需要几分钟)..."):
        results = []
        symbol_cache = {}
        progress = st.progress(0)
        
        for i, sig in enumerate(all_signals[:100]):  # 限制100个避免太慢
            symbol = sig['symbol']
            signal_price = sig['signal_price']
            
            if not signal_price or signal_price <= 0:
                continue
            
            # 获取价格
            if symbol not in symbol_cache:
                try:
                    df = get_stock_data(symbol, market=market, days=60)
                    symbol_cache[symbol] = df
                except:
                    symbol_cache[symbol] = None
            
            df = symbol_cache[symbol]
            if df is None or len(df) < 5:
                continue
            
            try:
                sig_dt = pd.to_datetime(sig['signal_date'])
                future_df = df[df.index > sig_dt]
                
                d1 = (future_df.iloc[0]['Close'] / signal_price - 1) * 100 if len(future_df) >= 1 else None
                d3 = (future_df.iloc[2]['Close'] / signal_price - 1) * 100 if len(future_df) >= 3 else None
                d5 = (future_df.iloc[4]['Close'] / signal_price - 1) * 100 if len(future_df) >= 5 else None
                d10 = (future_df.iloc[9]['Close'] / signal_price - 1) * 100 if len(future_df) >= 10 else None
                
                current_price = df.iloc[-1]['Close']
                current_return = (current_price / signal_price - 1) * 100
                
                results.append({
                    **sig,
                    'current_price': current_price,
                    'current_return': current_return,
                    'D1': d1, 'D3': d3, 'D5': d5, 'D10': d10,
                    'is_winner': current_return > 0
                })
            except:
                pass
            
            progress.progress((i + 1) / min(len(all_signals), 100))
        
        progress.empty()
    
    if not results:
        st.error("无法计算信号收益")
        return
    
    df = pd.DataFrame(results)
    
    # === 统计卡片 ===
    st.markdown("### 📊 整体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总信号数", len(df))
    
    with col2:
        winners = len(df[df['is_winner'] == True])
        win_rate = winners / len(df) * 100
        st.metric("胜率", f"{win_rate:.1f}%", delta="盈利多" if win_rate > 50 else "亏损多")
    
    with col3:
        avg_return = df['current_return'].mean()
        st.metric("平均收益", f"{avg_return:+.2f}%")
    
    with col4:
        st.metric("盈/亏", f"{winners}/{len(df) - winners}")
    
    st.divider()
    
    # === 各周期收益 ===
    st.markdown("### 📈 各周期平均收益")
    
    col1, col2, col3, col4 = st.columns(4)
    
    d1_avg = df['D1'].dropna().mean() if len(df['D1'].dropna()) > 0 else 0
    d3_avg = df['D3'].dropna().mean() if len(df['D3'].dropna()) > 0 else 0
    d5_avg = df['D5'].dropna().mean() if len(df['D5'].dropna()) > 0 else 0
    d10_avg = df['D10'].dropna().mean() if len(df['D10'].dropna()) > 0 else 0
    
    with col1:
        st.metric("D+1", f"{d1_avg:+.2f}%")
    with col2:
        st.metric("D+3", f"{d3_avg:+.2f}%")
    with col3:
        st.metric("D+5", f"{d5_avg:+.2f}%")
    with col4:
        st.metric("D+10", f"{d10_avg:+.2f}%")
    
    # 收益曲线图
    returns_data = pd.DataFrame({
        '周期': ['D+1', 'D+3', 'D+5', 'D+10'],
        '平均收益': [d1_avg, d3_avg, d5_avg, d10_avg]
    })
    
    fig = px.bar(returns_data, x='周期', y='平均收益', 
                 color='平均收益',
                 color_continuous_scale=['red', 'gray', 'green'],
                 title="各周期平均收益")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # === 最佳/最差 ===
    st.markdown("### 🏆 最佳 vs 最差")
    
    best = df.loc[df['current_return'].idxmax()]
    worst = df.loc[df['current_return'].idxmin()]
    price_sym = "$" if market == "US" else "¥"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🥇 **{best['symbol']}** {best.get('company_name', '')[:15]}")
        st.write(f"信号日期: {best['signal_date']}")
        st.write(f"信号价: {price_sym}{best['signal_price']:.2f}")
        st.write(f"当前价: {price_sym}{best['current_price']:.2f}")
        st.write(f"**收益: +{best['current_return']:.1f}%**")
    
    with col2:
        st.error(f"❌ **{worst['symbol']}** {worst.get('company_name', '')[:15]}")
        st.write(f"信号日期: {worst['signal_date']}")
        st.write(f"信号价: {price_sym}{worst['signal_price']:.2f}")
        st.write(f"当前价: {price_sym}{worst['current_price']:.2f}")
        st.write(f"**收益: {worst['current_return']:.1f}%**")
    
    st.divider()
    
    # === 详细列表 ===
    st.markdown("### 📋 信号详情")
    
    display_df = df[['signal_date', 'symbol', 'company_name', 'blue', 'signal_price', 
                     'D1', 'D3', 'D5', 'D10', 'current_return']].copy()
    display_df.columns = ['日期', '代码', '名称', 'BLUE', '信号价', 'D+1', 'D+3', 'D+5', 'D+10', '当前收益']
    
    # 格式化
    for col in ['D+1', 'D+3', 'D+5', 'D+10', '当前收益']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
    
    display_df['信号价'] = display_df['信号价'].apply(lambda x: f"{price_sym}{x:.2f}")
    display_df['名称'] = display_df['名称'].apply(lambda x: x[:12] if x else '')
    
    st.dataframe(display_df.sort_values('日期', ascending=False), 
                 hide_index=True, 
                 use_container_width=True,
                 height=400)


def render_signal_health_monitor():
    """🩺 信号健康度监控"""
    import plotly.graph_objects as go
    import plotly.express as px
    
    st.subheader("🩺 信号衰减监控")
    st.caption("实时追踪各类信号的胜率变化，及时发现信号失效")
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="health_market")
    with col2:
        min_blue = st.slider("BLUE 阈值", 50, 150, 100, key="health_blue")
    with col3:
        holding_days = st.selectbox("持有天数", [3, 5, 10, 20], index=1, key="health_days")
    
    # 获取健康度数据
    try:
        from services.signal_monitor import SignalMonitor, SignalType, HealthStatus
        
        with st.spinner("正在分析信号健康度..."):
            monitor = SignalMonitor(market=market, holding_days=holding_days)
            all_health = monitor.get_all_signals_health(min_blue=min_blue)
        
        # === 整体状态卡片 ===
        st.markdown("### 📊 整体状态")
        
        status_counts = {'healthy': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
        for health in all_health.values():
            status_counts[health.status.value] += 1
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("🟢 健康", status_counts['healthy'])
        with cols[1]:
            st.metric("🟡 关注", status_counts['warning'])
        with cols[2]:
            st.metric("🔴 衰减", status_counts['critical'])
        with cols[3]:
            st.metric("⚪ 未知", status_counts['unknown'])
        
        st.divider()
        
        # === 各信号详情 ===
        st.markdown("### 📋 各信号健康度")
        
        signal_names = {
            SignalType.DAILY_BLUE: "日 BLUE",
            SignalType.WEEKLY_BLUE: "周 BLUE",
            SignalType.MONTHLY_BLUE: "月 BLUE",
            SignalType.DAILY_WEEKLY: "日+周共振",
            SignalType.HEIMA: "黑马信号",
            SignalType.ALL_RESONANCE: "全共振"
        }
        
        status_icons = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.WARNING: "🟡",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.UNKNOWN: "⚪"
        }
        
        for signal_type, health in all_health.items():
            with st.expander(f"{status_icons[health.status]} {signal_names[signal_type]} - {health.status.value.upper()}", expanded=health.status != HealthStatus.UNKNOWN):
                
                if health.status == HealthStatus.UNKNOWN:
                    st.info("数据不足，无法评估")
                    continue
                
                # 胜率指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("7天胜率", f"{health.win_rate_7d:.0%}", 
                             delta=f"{(health.win_rate_7d - health.win_rate_90d)*100:+.0f}pp" if health.win_rate_90d > 0 else None)
                with col2:
                    st.metric("30天胜率", f"{health.win_rate_30d:.0%}",
                             delta=f"{(health.win_rate_30d - health.win_rate_90d)*100:+.0f}pp" if health.win_rate_90d > 0 else None)
                with col3:
                    st.metric("90天胜率", f"{health.win_rate_90d:.0%}")
                with col4:
                    decay_color = "normal" if health.decay_ratio >= 0.9 else "inverse"
                    st.metric("衰减比率", f"{health.decay_ratio:.0%}", delta_color=decay_color)
                
                # 收益指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("7天平均收益", f"{health.avg_return_7d:.1f}%")
                with col2:
                    st.metric("30天平均收益", f"{health.avg_return_30d:.1f}%")
                with col3:
                    st.metric("总平均收益", f"{health.avg_return_all:.1f}%")
                with col4:
                    st.metric("样本量(30天)", f"{health.sample_30d}")
                
                # 趋势图标
                trend_icons = {"improving": "📈 改善", "stable": "➡️ 稳定", "declining": "📉 下降"}
                st.caption(f"趋势: {trend_icons.get(health.trend, health.trend)}")
                
                # 建议
                if health.status == HealthStatus.CRITICAL:
                    st.error(f"💡 建议: {health.recommendation}")
                elif health.status == HealthStatus.WARNING:
                    st.warning(f"💡 建议: {health.recommendation}")
                else:
                    st.success(f"💡 建议: {health.recommendation}")
        
        st.divider()
        
        # === 胜率对比图 ===
        st.markdown("### 📈 胜率对比")
        
        # 准备数据
        chart_data = []
        for signal_type, health in all_health.items():
            if health.status != HealthStatus.UNKNOWN:
                chart_data.append({
                    '信号类型': signal_names[signal_type],
                    '7天': health.win_rate_7d * 100,
                    '30天': health.win_rate_30d * 100,
                    '90天': health.win_rate_90d * 100
                })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='7天', x=chart_df['信号类型'], y=chart_df['7天'], marker_color='#636EFA'))
            fig.add_trace(go.Bar(name='30天', x=chart_df['信号类型'], y=chart_df['30天'], marker_color='#EF553B'))
            fig.add_trace(go.Bar(name='90天', x=chart_df['信号类型'], y=chart_df['90天'], marker_color='#00CC96'))
            
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% 基准线")
            
            fig.update_layout(
                title="各信号胜率对比 (%)",
                barmode='group',
                yaxis_title="胜率 %",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # === 告警汇总 ===
        alerts = monitor.get_decay_alerts(min_blue)
        if alerts:
            st.markdown("### ⚠️ 告警")
            for alert in alerts:
                if alert.status == HealthStatus.CRITICAL:
                    st.error(f"🔴 **{signal_names[alert.signal_type]}**: {alert.recommendation}")
                else:
                    st.warning(f"🟡 **{signal_names[alert.signal_type]}**: {alert.recommendation}")
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_portfolio_management_page():
    """💼 组合管理 - 合并: 持仓管理 + 风控仪表盘 + 模拟交易"""
    st.header("💼 组合管理")
    
    tab1, tab2, tab3 = st.tabs(["🛡️ 风控仪表盘", "💰 持仓管理", "🎮 模拟交易"])
    
    with tab1:
        render_risk_dashboard()
    
    with tab2:
        render_portfolio_tab()
    
    with tab3:
        render_paper_trading_tab()


def render_risk_dashboard():
    """🛡️ 风控仪表盘 - 基于真实持仓数据"""
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, timedelta
    import numpy as np
    
    st.subheader("🛡️ 风险控制中心")
    
    # === 数据源选择 ===
    st.markdown("#### 📂 选择分析对象")
    
    source_options = {
        "🎮 模拟持仓": "paper",
        "💰 实盘持仓": "real",
        "📊 每日机会 (全部)": "daily_all",
        "🔵 仅日BLUE信号": "daily_blue",
        "🔷 日+周BLUE共振": "daily_weekly",
        "🔶 月BLUE信号": "monthly_blue",
        "🐴 黑马信号": "heima",
        "⭐ 全条件共振 (日+周+月+黑马)": "all_resonance"
    }
    
    data_source = st.selectbox(
        "数据来源",
        list(source_options.keys()),
        help="选择要分析的持仓/信号数据"
    )
    
    source_key = source_options[data_source]
    
    # === 信号筛选参数 ===
    if source_key not in ['paper', 'real']:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            market_filter = st.selectbox("市场", ["US", "CN"], key="risk_market")
        with filter_col2:
            days_back = st.slider("回看天数", 1, 30, 7, key="risk_days")
        with filter_col3:
            min_blue = st.slider("最低BLUE", 50, 150, 100, key="risk_blue")
    
    # === 获取数据 ===
    holdings = {}
    positions = []
    total_value = 0
    
    try:
        if source_key == "paper":
            # 模拟持仓
            from services.portfolio_service import get_paper_account
            account = get_paper_account()
            if account and account.get('positions'):
                positions = account['positions']
                total_value = account.get('total_equity', 0)
                
        elif source_key == "real":
            # 实盘持仓
            from db.database import get_portfolio
            from services.portfolio_service import get_current_price
            portfolio = get_portfolio()
            if portfolio:
                for p in portfolio:
                    price = get_current_price(p['symbol'], p.get('market', 'US'))
                    if price:
                        p['market_value'] = price * p['shares']
                        p['current_price'] = price
                    else:
                        p['market_value'] = p.get('cost_basis', 0) * p['shares']
                    total_value += p['market_value']
                positions = portfolio
                
        else:
            # 从扫描信号获取
            from db.database import query_scan_results
            from services.portfolio_service import get_current_price
            from datetime import date, timedelta
            
            # 获取最近 N 天的扫描结果
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            all_signals = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                try:
                    results = query_scan_results(date_str, market=market_filter, min_blue=min_blue)
                    if results:
                        for r in results:
                            r['scan_date'] = date_str
                        all_signals.extend(results)
                except:
                    pass
                current_date += timedelta(days=1)
            
            if all_signals:
                # 根据策略筛选
                filtered_signals = []
                
                for sig in all_signals:
                    # 字段名兼容 (数据库用小写，CSV用大写)
                    day_blue = sig.get('blue_daily', sig.get('Day_BLUE', 0)) or 0
                    week_blue = sig.get('blue_weekly', sig.get('Week_BLUE', 0)) or 0
                    month_blue = sig.get('blue_monthly', sig.get('Month_BLUE', 0)) or 0
                    heima = sig.get('is_heima', sig.get('Heima', False))
                    
                    if source_key == "daily_all":
                        # 所有信号
                        if day_blue >= min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "daily_blue":
                        # 仅日BLUE
                        if day_blue >= min_blue and week_blue < min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "daily_weekly":
                        # 日+周共振
                        if day_blue >= min_blue and week_blue >= min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "monthly_blue":
                        # 月BLUE
                        if month_blue >= min_blue:
                            filtered_signals.append(sig)
                            
                    elif source_key == "heima":
                        # 黑马信号
                        if heima:
                            filtered_signals.append(sig)
                            
                    elif source_key == "all_resonance":
                        # 全条件共振
                        if day_blue >= min_blue and week_blue >= min_blue and (month_blue >= min_blue or heima):
                            filtered_signals.append(sig)
                
                # 去重 (同一只股票只保留最新，按 BLUE 值排序)
                symbol_latest = {}
                for sig in filtered_signals:
                    sym = sig.get('symbol', sig.get('Symbol', ''))
                    if sym:
                        if sym not in symbol_latest or sig['scan_date'] > symbol_latest[sym]['scan_date']:
                            symbol_latest[sym] = sig
                
                # 按 blue_daily 排序，取 Top N
                MAX_POSITIONS = 20  # 限制最多分析 20 只
                sorted_symbols = sorted(
                    symbol_latest.items(),
                    key=lambda x: x[1].get('blue_daily', x[1].get('Day_BLUE', 0)) or 0,
                    reverse=True
                )[:MAX_POSITIONS]
                
                st.info(f"📊 筛选: {len(all_signals)} 条信号 → {len(filtered_signals)} 符合 → {len(symbol_latest)} 只股票 → Top {len(sorted_symbols)} (按BLUE排序)")
                
                # 转换为持仓格式 (等权重)
                if sorted_symbols:
                    equal_value = 100000 / len(sorted_symbols)  # 10万等分
                    
                    progress_bar = st.progress(0, text="正在获取价格数据...")
                    
                    for i, (sym, sig) in enumerate(sorted_symbols):
                        progress_bar.progress((i + 1) / len(sorted_symbols), text=f"获取 {sym} 价格...")
                        
                        # 先尝试用扫描时的价格
                        price = sig.get('price', sig.get('Close', None))
                        if not price:
                            price = get_current_price(sym, market_filter)
                        
                        if price and price > 0:
                            shares = int(equal_value / price)
                            market_value = shares * price
                            
                            positions.append({
                                'symbol': sym,
                                'shares': shares,
                                'avg_cost': price,
                                'current_price': price,
                                'market_value': market_value,
                                'market': market_filter,
                                'day_blue': sig.get('blue_daily', sig.get('Day_BLUE', 0)),
                                'week_blue': sig.get('blue_weekly', sig.get('Week_BLUE', 0)),
                                'unrealized_pnl_pct': 0
                            })
                            total_value += market_value
                    
                    progress_bar.empty()
                
    except Exception as e:
        st.warning(f"获取数据失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # 检查是否有持仓
    if not positions:
        st.info("📭 暂无持仓数据")
        st.markdown("""
        请先在以下位置添加持仓:
        - **模拟交易** Tab: 使用虚拟资金买入股票
        - **持仓管理** Tab: 手动添加实盘持仓
        """)
        
        # 显示仓位计算器作为替代
        st.divider()
        render_position_calculator()
        return
    
    # 计算持仓权重
    for pos in positions:
        symbol = pos.get('symbol', 'Unknown')
        market_value = pos.get('market_value', 0)
        if total_value > 0:
            holdings[symbol] = market_value / total_value
    
    symbols = list(holdings.keys())
    
    st.success(f"✅ 已加载 {len(positions)} 个持仓，总市值 ${total_value:,.0f}")
    
    # === 获取历史数据计算风险指标 ===
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_returns_from_scan_history(symbols_tuple, market, days_back=60):
        """从扫描历史数据计算收益率 (不调用外部 API)"""
        from db.database import get_connection
        from datetime import date, timedelta
        
        returns_dict = {}
        symbols_list = list(symbols_tuple)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # 获取每只股票的历史扫描价格
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        for sym in symbols_list:
            try:
                cursor.execute("""
                    SELECT scan_date, price 
                    FROM scan_results 
                    WHERE symbol = ? AND market = ? AND scan_date >= ? 
                    ORDER BY scan_date
                """, (sym, market, start_date.strftime('%Y-%m-%d')))
                
                rows = cursor.fetchall()
                if len(rows) >= 2:
                    prices = pd.Series(
                        {row['scan_date']: row['price'] for row in rows if row['price']}
                    )
                    if len(prices) >= 2:
                        returns = prices.pct_change().dropna()
                        if len(returns) >= 1:
                            returns_dict[sym] = returns
            except:
                continue
        
        conn.close()
        return returns_dict
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_returns_from_api(symbols_tuple, market, days=60):
        """从 API 获取收益率数据 (备选)"""
        from data_fetcher import get_us_stock_data, get_cn_stock_data
        import time
        
        returns_dict = {}
        symbols_list = list(symbols_tuple)
        
        for i, sym in enumerate(symbols_list[:10]):  # 最多取 10 只，避免 rate limit
            try:
                if market == 'CN' or sym.endswith('.SH') or sym.endswith('.SZ'):
                    df = get_cn_stock_data(sym, days=days)
                else:
                    df = get_us_stock_data(sym, days=days)
                
                if df is not None and len(df) > 10:
                    returns_dict[sym] = df['Close'].pct_change().dropna()
                
                if i < len(symbols_list) - 1:
                    time.sleep(0.2)
            except:
                continue
        
        return returns_dict
    
    # 获取风险数据
    current_market = market_filter if source_key not in ['paper', 'real'] else 'US'
    
    # 先尝试扫描历史
    with st.spinner("正在计算风险指标..."):
        returns_data = get_returns_from_scan_history(tuple(symbols), current_market, days_back=60)
    
    # 如果扫描历史不足，尝试 API (只取前 5 只)
    if len(returns_data) < 2:
        st.caption("📊 扫描历史稀疏，从 API 获取主要持仓数据...")
        returns_data = get_returns_from_api(tuple(symbols[:5]), current_market, days=60)
    
    st.caption(f"📈 获取到 {len(returns_data)} 只股票的历史数据")
    
    # === 第一行: 核心风险指标 ===
    st.markdown("### 📊 组合风险概览")
    
    # 计算组合收益率
    if returns_data and len(returns_data) > 0:
        # 对齐日期
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) > 20:
            # 计算组合加权收益
            weight_array = np.array([holdings.get(s, 0) for s in returns_df.columns])
            weight_array = weight_array / weight_array.sum()  # 归一化
            
            portfolio_returns = (returns_df * weight_array).sum(axis=1)
            
            # 计算风险指标
            var_95 = np.percentile(portfolio_returns, 5) * 100
            
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            
            excess_returns = portfolio_returns - 0.02/252  # 假设无风险利率 2%
            sharpe = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        else:
            var_95, max_dd, volatility, sharpe = -2.0, -5.0, 20.0, 1.0
            st.warning("历史数据不足，使用估算值")
    else:
        var_95, max_dd, volatility, sharpe = -2.0, -5.0, 20.0, 1.0
        st.warning("无法获取历史数据，使用估算值")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "VaR (95%, 1天)",
            f"{var_95:.2f}%",
            delta="正常" if var_95 > -5 else "警告",
            delta_color="normal" if var_95 > -5 else "inverse"
        )
        st.caption("单日最大损失估计")
    
    with col2:
        st.metric(
            "最大回撤",
            f"{max_dd:.1f}%",
            delta="可控" if max_dd > -15 else "需关注",
            delta_color="normal" if max_dd > -15 else "inverse"
        )
    
    with col3:
        st.metric(
            "年化波动率",
            f"{volatility:.1f}%",
            delta="中等" if volatility < 25 else "偏高",
            delta_color="normal" if volatility < 25 else "inverse"
        )
    
    with col4:
        st.metric(
            "Sharpe 比率",
            f"{sharpe:.2f}",
            delta="优秀" if sharpe > 1.5 else ("一般" if sharpe > 0.5 else "差"),
            delta_color="normal" if sharpe > 1.0 else "inverse"
        )
    
    st.divider()
    
    # === 第二行: 持仓集中度 + 持仓明细 ===
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📈 持仓集中度")
        
        # 饼图 - 使用真实持仓
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(holdings.keys()),
            values=[v * 100 for v in holdings.values()],
            hole=0.4,
            textinfo='label+percent',
            marker_colors=px.colors.qualitative.Set3
        )])
        fig_pie.update_layout(
            title=f"持仓分布 (共 {len(holdings)} 只)",
            height=300,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 集中度警告
        if holdings:
            max_symbol = max(holdings, key=holdings.get)
            max_weight = holdings[max_symbol]
            
            if max_weight > 0.25:
                st.error(f"🔴 单股集中度过高: {max_symbol} = {max_weight:.0%} (建议 < 25%)")
            elif max_weight > 0.20:
                st.warning(f"⚠️ 单股集中度偏高: {max_symbol} = {max_weight:.0%}")
            else:
                st.success(f"✅ 集中度正常: 最大持仓 {max_symbol} = {max_weight:.0%}")
    
    with col_right:
        st.markdown("### 📋 持仓明细")
        
        # 持仓表格
        pos_df = pd.DataFrame([{
            '代码': p.get('symbol'),
            '股数': p.get('shares'),
            '市值': f"${p.get('market_value', 0):,.0f}",
            '权重': f"{holdings.get(p.get('symbol'), 0):.1%}",
            '盈亏': f"{p.get('unrealized_pnl_pct', 0):.1f}%"
        } for p in positions])
        
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # === 第三行: 相关性矩阵 + 回撤曲线 ===
    col_corr, col_dd = st.columns(2)
    
    with col_corr:
        st.markdown("### 🔗 持仓相关性")
        
        if returns_data and len(returns_data) >= 2:
            # 使用 pairwise 相关性 (允许不同日期)
            returns_df = pd.DataFrame(returns_data)
            
            # 计算 pairwise 相关性 (使用重叠日期)
            corr_matrix = returns_df.corr(min_periods=2)  # 至少 2 个重叠点
            
            # 检查是否有有效的相关性
            valid_corr = corr_matrix.dropna(how='all').dropna(axis=1, how='all')
            
            if len(valid_corr) >= 2:
                fig_corr = px.imshow(
                    valid_corr.values,
                    x=valid_corr.columns.tolist(),
                    y=valid_corr.index.tolist(),
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    title=f"相关性矩阵 ({len(valid_corr)} 只股票)",
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # 高相关性警告
                high_corr_pairs = []
                cols = valid_corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        val = valid_corr.iloc[i, j]
                        if pd.notna(val) and val > 0.75:
                            high_corr_pairs.append((cols[i], cols[j], val))
                
                if high_corr_pairs:
                    st.warning(f"⚠️ 高相关性: {', '.join([f'{p[0]}-{p[1]}({p[2]:.2f})' for p in high_corr_pairs[:3]])}")
                else:
                    st.success("✅ 持仓分散度良好")
            else:
                st.info("📊 数据重叠不足，显示持仓列表")
                st.dataframe(pd.DataFrame({
                    '股票': list(returns_data.keys()),
                    '数据点': [len(v) for v in returns_data.values()]
                }), hide_index=True)
        else:
            st.info("需要至少 2 个持仓才能计算相关性")
    
    with col_dd:
        st.markdown("### 📉 个股收益分布")
        
        if returns_data and len(returns_data) > 0:
            # 计算每只股票的总收益和统计
            stock_stats = []
            for sym, rets in returns_data.items():
                if len(rets) > 0:
                    total_ret = (1 + rets).prod() - 1
                    avg_ret = rets.mean()
                    volatility = rets.std()
                    stock_stats.append({
                        'symbol': sym,
                        'total_return': total_ret * 100,
                        'avg_daily': avg_ret * 100,
                        'volatility': volatility * 100,
                        'days': len(rets)
                    })
            
            if stock_stats:
                stats_df = pd.DataFrame(stock_stats)
                
                # 收益分布柱状图
                fig_returns = go.Figure()
                colors = ['green' if r >= 0 else 'red' for r in stats_df['total_return']]
                fig_returns.add_trace(go.Bar(
                    x=stats_df['symbol'],
                    y=stats_df['total_return'],
                    marker_color=colors,
                    text=[f"{r:.1f}%" for r in stats_df['total_return']],
                    textposition='outside'
                ))
                fig_returns.add_hline(y=0, line_color="gray")
                fig_returns.update_layout(
                    title="各股票累计收益 (%)",
                    xaxis_title="股票",
                    yaxis_title="收益 %",
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # 统计摘要
                avg_return = stats_df['total_return'].mean()
                win_count = (stats_df['total_return'] > 0).sum()
                win_rate = win_count / len(stats_df) * 100
                
                st.caption(f"📊 平均收益: {avg_return:.1f}% | 胜率: {win_rate:.0f}% ({win_count}/{len(stats_df)})")
            else:
                st.info("数据不足")
        else:
            st.info("无历史数据")
    
    st.divider()
    
    # === 仓位计算器 ===
    render_position_calculator()


def render_position_calculator():
    """仓位计算器组件"""
    st.markdown("### 🧮 仓位计算器")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        st.markdown("#### 固定比例法")
        with st.form("position_calc"):
            total_capital = st.number_input("总资金 ($)", value=100000, step=10000)
            risk_per_trade = st.slider("每笔风险比例 (%)", 1, 5, 2) / 100
            entry_price = st.number_input("入场价格", value=150.0, step=1.0)
            stop_loss = st.number_input("止损价格", value=142.0, step=1.0)
            
            if st.form_submit_button("计算仓位"):
                risk_amount = total_capital * risk_per_trade
                risk_per_share = abs(entry_price - stop_loss)
                
                if risk_per_share > 0:
                    shares = int(risk_amount / risk_per_share)
                    position_value = shares * entry_price
                    position_pct = position_value / total_capital
                    
                    st.success(f"""
                    **建议仓位:**
                    - 股数: **{shares:,}** 股
                    - 仓位金额: **${position_value:,.0f}**
                    - 仓位比例: **{position_pct:.1%}**
                    - 最大亏损: **${risk_amount:,.0f}** ({risk_per_trade:.1%})
                    """)
                    
                    if position_pct > 0.20:
                        st.warning("⚠️ 仓位超过 20%，建议分批建仓")
                else:
                    st.error("止损价格不能等于入场价格")
    
    with calc_col2:
        st.markdown("#### 凯利公式")
        with st.form("kelly_calc"):
            win_rate = st.slider("胜率 (%)", 30, 80, 55) / 100
            avg_win = st.number_input("平均盈利 (%)", value=8.0, step=1.0) / 100
            avg_loss = st.number_input("平均亏损 (%)", value=4.0, step=1.0) / 100
            kelly_fraction = st.slider("凯利系数 (保守)", 0.25, 1.0, 0.5, step=0.25)
            
            if st.form_submit_button("计算最优仓位"):
                if avg_loss > 0:
                    # 凯利公式: f = (bp - q) / b
                    b = avg_win / avg_loss  # 赔率
                    p = win_rate
                    q = 1 - p
                    
                    full_kelly = (b * p - q) / b
                    adjusted_kelly = max(0, full_kelly * kelly_fraction)
                    
                    st.success(f"""
                    **凯利公式结果:**
                    - 赔率 (盈亏比): **{b:.2f}**
                    - 完整凯利: **{full_kelly:.1%}**
                    - {kelly_fraction:.0%} 凯利: **{adjusted_kelly:.1%}**
                    
                    建议仓位: **{min(adjusted_kelly, 0.20):.1%}** (上限 20%)
                    """)
                    
                    if full_kelly < 0:
                        st.error("❌ 期望值为负，不建议交易")
                else:
                    st.error("平均亏损必须大于 0")


def render_portfolio_tab():
    """💰 持仓管理 Tab"""
    st.subheader("💰 持仓管理")
    
    # 复用原有的 portfolio 渲染逻辑
    try:
        # 获取持仓数据
        from services.portfolio_service import (
            get_portfolio_summary, 
            get_current_price,
            get_paper_account
        )
        from db.database import get_portfolio, get_trades
        
        portfolio = get_portfolio()
        
        if portfolio:
            summary = get_portfolio_summary()
            
            # 显示汇总
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总成本", f"${summary['total_cost']:,.0f}")
            with col2:
                st.metric("市值", f"${summary['total_value']:,.0f}")
            with col3:
                pnl = summary['unrealized_pnl']
                st.metric("未实现盈亏", f"${pnl:,.0f}", 
                         delta=f"{summary['unrealized_pnl_pct']:.1f}%",
                         delta_color="normal" if pnl >= 0 else "inverse")
            with col4:
                st.metric("持仓数", f"{summary['position_count']}")
            
            # 持仓列表
            st.dataframe(
                pd.DataFrame(portfolio),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("暂无持仓记录")
            
    except Exception as e:
        st.warning(f"持仓数据加载失败: {e}")
        st.info("请先在数据库中添加持仓记录")


def render_paper_trading_tab():
    """🎮 模拟交易 Tab"""
    st.subheader("🎮 模拟交易")
    
    try:
        from services.portfolio_service import (
            get_paper_account,
            paper_buy,
            paper_sell,
            get_paper_trades,
            reset_paper_account,
            get_paper_equity_curve,
            get_paper_monthly_returns
        )
        
        # 获取账户信息
        account = get_paper_account()
        
        # 账户概览
        if account is None:
            st.warning("模拟账户未初始化")
            if st.button("初始化模拟账户"):
                from services.portfolio_service import init_paper_account
                init_paper_account()
                st.rerun()
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("初始资金", f"${account.get('initial_capital', 100000):,.0f}")
        with col2:
            st.metric("现金余额", f"${account.get('cash_balance', 0):,.0f}")
        with col3:
            st.metric("持仓市值", f"${account.get('position_value', 0):,.0f}")
        with col4:
            pnl = account.get('total_pnl', 0)
            initial = account.get('initial_capital', 100000)
            st.metric("总盈亏", f"${pnl:,.0f}",
                     delta=f"{pnl/initial*100:.1f}%" if initial > 0 else "0%",
                     delta_color="normal" if pnl >= 0 else "inverse")
        
        st.divider()
        
        # 交易面板
        trade_col1, trade_col2 = st.columns(2)
        
        with trade_col1:
            st.markdown("#### 🟢 买入")
            with st.form("paper_buy_form"):
                symbol = st.text_input("股票代码", placeholder="AAPL")
                shares = st.number_input("股数", min_value=1, value=10)
                price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0)
                market = st.selectbox("市场", ["US", "CN"])
                
                if st.form_submit_button("买入", type="primary"):
                    if symbol:
                        result = paper_buy(symbol.upper(), shares, price if price > 0 else None, market)
                        if result.get('success'):
                            st.success(f"✅ 买入成功: {shares} 股 {symbol}")
                            st.rerun()
                        else:
                            st.error(f"❌ 买入失败: {result.get('error')}")
        
        with trade_col2:
            st.markdown("#### 🔴 卖出")
            positions = account.get('positions', [])
            if positions:
                with st.form("paper_sell_form"):
                    pos_options = [f"{p['symbol']} ({p['shares']}股)" for p in positions]
                    selected = st.selectbox("选择持仓", pos_options)
                    sell_shares = st.number_input("卖出股数", min_value=1, value=1)
                    sell_price = st.number_input("价格 (0=市价)", min_value=0.0, value=0.0)
                    
                    if st.form_submit_button("卖出", type="primary"):
                        symbol = selected.split(" ")[0]
                        result = paper_sell(symbol, sell_shares, sell_price if sell_price > 0 else None)
                        if result.get('success'):
                            st.success(f"✅ 卖出成功: {sell_shares} 股 {symbol}")
                            st.rerun()
                        else:
                            st.error(f"❌ 卖出失败: {result.get('error')}")
            else:
                st.info("暂无持仓")
        
        # 权益曲线
        equity_curve = get_paper_equity_curve()
        if not equity_curve.empty and len(equity_curve) > 1:
            st.markdown("#### 📈 权益曲线")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['total_equity'],
                mode='lines+markers',
                name='总权益'
            ))
            initial_cap = account.get('initial_capital', 100000)
            fig.add_hline(y=initial_cap, line_dash="dash", 
                         annotation_text=f"初始资金 ${initial_cap:,.0f}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 重置按钮
        if st.button("🔄 重置模拟账户", type="secondary"):
            reset_paper_account()
            st.success("账户已重置")
            st.rerun()
            
    except Exception as e:
        st.warning(f"模拟交易模块加载失败: {e}")


def render_strategy_lab_page():
    """🧪 策略实验室 - 合并: 回测 + 研究工具"""
    st.header("🧪 策略实验室")
    
    tab1, tab2, tab3 = st.tabs(["📊 策略回测", "🔬 因子研究", "📐 组合优化"])
    
    with tab1:
        render_backtest_page()
    
    with tab2:
        render_research_page()
    
    with tab3:
        render_portfolio_optimizer_page()


def render_ai_center_page():
    """🤖 AI中心 - 重新设计: 智能选股 + 模型管理 + 博主追踪"""
    st.header("🤖 AI 选股中心")
    
    tab1, tab2, tab3 = st.tabs(["🎯 今日精选", "⚙️ 模型管理", "📢 博主追踪"])
    
    with tab1:
        render_ai_smart_picks()
    
    with tab2:
        render_ml_prediction_page()  # 保留原有模型管理
    
    with tab3:
        render_blogger_page()


def render_ai_smart_picks():
    """🎯 AI智能选股 - 核心推荐页面"""
    from pathlib import Path
    
    st.markdown("""
    <style>
    .pick-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 4px solid #00C853;
    }
    .pick-card.warning {
        border-left-color: #FFD600;
    }
    .star-rating {
        color: #FFD700;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 选项
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="ai_pick_market")
    with col2:
        horizon_options = {"短线 (1-5天)": "short", "中线 (10-30天)": "medium", "长线 (60天+)": "long"}
        horizon_label = st.selectbox("交易周期", list(horizon_options.keys()), key="ai_horizon")
        horizon = horizon_options[horizon_label]
    with col3:
        max_picks = st.selectbox("推荐数量", [3, 5, 8, 10], index=1, key="ai_max_picks")
    
    # 检查模型状态
    model_dir = Path(__file__).parent / "ml" / "saved_models" / f"v2_{market.lower()}"
    return_model_exists = (model_dir / "return_5d.joblib").exists()
    ranker_model_exists = (model_dir / f"ranker_{horizon}.joblib").exists()
    
    # 模型状态显示
    status_cols = st.columns(2)
    with status_cols[0]:
        if return_model_exists:
            st.success("✓ 收益预测模型已加载", icon="🎯")
        else:
            st.warning("⚠ 收益预测模型未训练", icon="⚠️")
    with status_cols[1]:
        if ranker_model_exists:
            st.success(f"✓ 排序模型 ({horizon}) 已加载", icon="🏆")
        else:
            st.info(f"💡 排序模型 ({horizon}) 未训练，使用规则引擎")
    
    st.divider()
    
    # 获取推荐
    if st.button("🔄 刷新推荐", type="primary", key="refresh_ai_picks"):
        st.session_state['ai_picks_loaded'] = False
    
    # 加载推荐
    with st.spinner("AI 分析中..."):
        try:
            from ml.smart_picker import get_todays_picks, SmartPicker
            from db.database import get_connection
            from db.stock_history import get_stock_history
            
            # 获取最新信号
            conn = get_connection()
            query = """
                SELECT DISTINCT symbol, scan_date, price, 
                       COALESCE(blue_daily, 0) as blue_daily,
                       COALESCE(blue_weekly, 0) as blue_weekly,
                       COALESCE(blue_monthly, 0) as blue_monthly,
                       COALESCE(is_heima, 0) as is_heima,
                       company_name
                FROM scan_results
                WHERE market = ?
                ORDER BY scan_date DESC
                LIMIT 100
            """
            signals_df = pd.read_sql_query(query, conn, params=(market,))
            conn.close()
            
            if signals_df.empty:
                st.warning("暂无信号数据，请先运行扫描")
                return
            
            latest_date = signals_df['scan_date'].iloc[0]
            today_signals = signals_df[signals_df['scan_date'] == latest_date]
            
            st.caption(f"📅 信号日期: {latest_date} | 共 {len(today_signals)} 只股票")
            
            # 获取价格历史
            price_history = {}
            progress = st.progress(0)
            symbols = today_signals['symbol'].unique()
            
            for i, symbol in enumerate(symbols):
                history = get_stock_history(symbol, market, days=100)
                if not history.empty:
                    price_history[symbol] = history
                progress.progress((i + 1) / len(symbols))
            progress.empty()
            
            # 智能选股 (使用排序模型)
            picker = SmartPicker(market=market, horizon=horizon)
            picks = picker.pick(today_signals, price_history, max_picks=max_picks)
            
            if not picks:
                st.info("今日没有高置信度的推荐")
                return
            
            # === 显示推荐 ===
            st.markdown(f"### 🎯 今日精选 ({len(picks)} 只)")
            
            # 汇总统计
            avg_score = sum(p.overall_score for p in picks) / len(picks)
            avg_rr = sum(p.risk_reward_ratio for p in picks) / len(picks)
            high_conf = sum(1 for p in picks if p.star_rating >= 4)
            
            sum_cols = st.columns(4)
            with sum_cols[0]:
                st.metric("平均评分", f"{avg_score:.0f}/100")
            with sum_cols[1]:
                st.metric("高置信度", f"{high_conf}/{len(picks)}")
            with sum_cols[2]:
                st.metric("平均风险收益比", f"1:{avg_rr:.1f}")
            with sum_cols[3]:
                avg_pred = sum(p.pred_return_5d for p in picks) / len(picks)
                st.metric("平均预测收益", f"{avg_pred:+.1f}%")
            
            st.divider()
            
            # 详细推荐卡片
            for i, pick in enumerate(picks):
                stars = "⭐" * pick.star_rating + "☆" * (5 - pick.star_rating)
                
                # 卡片颜色
                if pick.star_rating >= 4:
                    card_border = "#00C853"
                    card_bg = "#1a472a"
                elif pick.star_rating >= 3:
                    card_border = "#FFD600"
                    card_bg = "#4a4a00"
                else:
                    card_border = "#666"
                    card_bg = "#333"
                
                # 价格符号
                price_sym = "¥" if market == "CN" else "$"
                
                with st.container():
                    # 头部: 股票名称 + 评分
                    header_col1, header_col2 = st.columns([3, 1])
                    with header_col1:
                        display_name = pick.name if pick.name else pick.symbol
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 1.5em; font-weight: bold;">{display_name}</span>
                            <span style="color: #888; font-size: 0.9em;">{pick.symbol}</span>
                            <span class="star-rating">{stars}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with header_col2:
                        st.markdown(f"""
                        <div style="text-align: right;">
                            <span style="font-size: 1.3em; font-weight: bold;">{price_sym}{pick.price:.2f}</span>
                            <br>
                            <span style="font-size: 1.1em; color: {'#00C853' if pick.pred_return_5d > 0 else '#FF5252'};">
                                {pick.pred_return_5d:+.1f}% 预测
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 内容区
                    content_cols = st.columns([1, 1, 1])
                    
                    with content_cols[0]:
                        st.markdown("**📊 信号验证**")
                        for signal in pick.signals_confirmed[:4]:
                            st.markdown(f"<span style='color: #00C853;'>{signal}</span>", unsafe_allow_html=True)
                        for warning in pick.signals_warning[:2]:
                            st.markdown(f"<span style='color: #FFD600;'>{warning}</span>", unsafe_allow_html=True)
                    
                    with content_cols[1]:
                        st.markdown("**🎯 交易计划**")
                        st.markdown(f"""
                        - 止损: {price_sym}{pick.stop_loss_price:.2f} ({pick.stop_loss_pct:+.1f}%)
                        - 目标: {price_sym}{pick.target_price:.2f} (+{pick.target_pct:.1f}%)
                        - 风险收益比: **1:{pick.risk_reward_ratio:.1f}**
                        """)
                    
                    with content_cols[2]:
                        st.markdown("**💡 建议**")
                        # 获取当前周期的排名分
                        rank_score = pick.rank_score_short
                        if horizon == 'medium':
                            rank_score = pick.rank_score_medium
                        elif horizon == 'long':
                            rank_score = pick.rank_score_long
                        st.markdown(f"""
                        - 仓位: **{pick.suggested_position_pct:.0f}%**
                        - 上涨概率: **{pick.pred_direction_prob:.0%}**
                        - 排序得分: **{rank_score:.1f}**
                        - 综合评分: **{pick.overall_score:.0f}**/100
                        """)
                    
                    # 指标徽章
                    st.markdown(f"""
                    <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
                        <span style="background: #E91E6333; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">
                            🏆 排名分 {rank_score:.0f}
                        </span>
                        <span style="background: #00C85333; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            日B {pick.blue_daily:.0f}
                        </span>
                        <span style="background: #FFD60033; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            周B {pick.blue_weekly:.0f}
                        </span>
                        <span style="background: #2196F333; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            月B {pick.blue_monthly:.0f}
                        </span>
                        <span style="background: #9C27B033; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            RSI {pick.rsi:.0f}
                        </span>
                        <span style="background: #FF572233; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">
                            量比 {pick.volume_ratio:.1f}x
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 操作按钮
                    btn_cols = st.columns([1, 1, 1, 3])
                    with btn_cols[0]:
                        if st.button("📈 查看K线", key=f"ai_chart_{pick.symbol}"):
                            st.session_state[f'ai_detail_{pick.symbol}'] = True
                    with btn_cols[1]:
                        if st.button("💰 模拟买入", key=f"ai_buy_{pick.symbol}"):
                            st.session_state[f'ai_buy_form_{pick.symbol}'] = True
                    with btn_cols[2]:
                        if st.button("👁️ 加入观察", key=f"ai_watch_{pick.symbol}"):
                            try:
                                from services.signal_tracker import add_to_watchlist
                                add_to_watchlist(
                                    pick.symbol, market,
                                    entry_price=pick.price,
                                    target_price=pick.target_price,
                                    stop_loss=pick.stop_loss_price
                                )
                                st.success(f"已加入观察列表")
                            except Exception as e:
                                st.error(f"添加失败: {e}")
                    
                    # 详情展开
                    if st.session_state.get(f'ai_detail_{pick.symbol}'):
                        with st.expander("📊 详细分析", expanded=True):
                            from components.stock_detail import render_unified_stock_detail
                            render_unified_stock_detail(
                                symbol=pick.symbol,
                                market=market,
                                key_prefix=f"ai_detail_{pick.symbol}"
                            )
                    
                    # 买入表单
                    if st.session_state.get(f'ai_buy_form_{pick.symbol}'):
                        with st.expander("💰 模拟买入", expanded=True):
                            buy_col1, buy_col2 = st.columns(2)
                            with buy_col1:
                                buy_shares = st.number_input(
                                    "买入数量", 
                                    min_value=1, 
                                    value=100,
                                    key=f"ai_buy_shares_{pick.symbol}"
                                )
                            with buy_col2:
                                buy_price = st.number_input(
                                    "买入价格",
                                    value=pick.price,
                                    key=f"ai_buy_price_{pick.symbol}"
                                )
                            
                            total_cost = buy_shares * buy_price
                            st.info(f"总成本: {price_sym}{total_cost:,.2f}")
                            
                            if st.button("确认买入", key=f"ai_confirm_buy_{pick.symbol}", type="primary"):
                                try:
                                    from services.portfolio_service import paper_buy
                                    result = paper_buy(
                                        symbol=pick.symbol,
                                        market=market,
                                        shares=buy_shares,
                                        price=buy_price
                                    )
                                    if result.get('success'):
                                        st.success(f"✅ 成功买入 {buy_shares} 股 {pick.symbol}")
                                        st.session_state[f'ai_buy_form_{pick.symbol}'] = False
                                    else:
                                        st.error(result.get('error', '买入失败'))
                                except Exception as e:
                                    st.error(f"买入失败: {e}")
                    
                    st.divider()
            
            # === 风险提示 ===
            st.markdown("""
            ---
            ### ⚠️ 风险提示
            
            - 以上推荐基于 **技术分析 + ML模型**，仅供参考
            - **严格执行止损**，保护本金是第一位的
            - 建议单只股票仓位不超过 **15%**
            - 历史表现不代表未来收益
            """)
            
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def render_ml_prediction_page():
    """🎯 ML 模型预测页面 - 完整版"""
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path
    import json
    
    st.subheader("🎯 ML 智能选股")
    
    # 市场选择
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        market = st.selectbox("市场", ["US", "CN"], key="ml_market")
    with col2:
        horizon = st.selectbox("预测周期", ["5d", "1d", "10d", "30d"], key="ml_horizon")
    
    # 检查模型是否存在
    model_dir = Path(__file__).parent / "ml" / "saved_models" / f"v2_{market.lower()}"
    meta_path = model_dir / "return_predictor_meta.json"
    ranker_meta_path = model_dir / "ranker_meta.json"
    
    if not meta_path.exists():
        st.warning("⚠️ 模型未训练")
        st.info("""
        **训练步骤:**
        ```bash
        cd versions/v3
        python ml/pipeline.py --market US --days 60
        ```
        """)
        
        if st.button("🚀 开始训练", key="train_full"):
            with st.spinner("训练中... (约 30 秒)"):
                try:
                    from ml.pipeline import train_pipeline
                    result = train_pipeline(market=market, days_back=60)
                    if result and result.get('status') == 'success':
                        st.success("✅ 训练完成!")
                        st.rerun()
                    else:
                        st.error("训练失败")
                except Exception as e:
                    st.error(f"训练失败: {e}")
        return
    
    # 加载模型元数据
    with open(meta_path) as f:
        meta = json.load(f)
    
    # 加载排序模型元数据
    ranker_meta = {}
    if ranker_meta_path.exists():
        with open(ranker_meta_path) as f:
            ranker_meta = json.load(f)
    
    # ==================================
    # 📊 模型概览 - 详细指标
    # ==================================
    st.markdown("### 📊 模型概览")
    
    model_tab1, model_tab2, model_tab3, model_tab4, model_tab5 = st.tabs([
        "📈 收益预测模型", "🏆 排序模型", "🔧 特征重要性", "⚙️ 超参数调优", "🔗 模型对比"
    ])
    
    with model_tab1:
        st.markdown("**Return Predictor** - 预测 1/5/10/30 天收益率")
        
        # 所有周期指标对比表
        metrics_data = []
        for h, m in meta.get('metrics', {}).items():
            metrics_data.append({
                '周期': h,
                'R²': f"{m.get('r2', 0):.3f}",
                '方向准确率': f"{m.get('direction_accuracy', 0):.1%}",
                'RMSE': f"{m.get('rmse', 0):.2f}%",
                'MAE': f"{m.get('mae', 0):.2f}%",
                '训练样本': m.get('train_samples', 0),
                '测试样本': m.get('test_samples', 0)
            })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            # 缩短列名
            metrics_df.columns = ['周期', 'R²', '方向准确率', 'RMSE', 'MAE', '训练', '测试']
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # 方向准确率图
            fig_acc = go.Figure()
            horizons = [m['周期'] for m in metrics_data]
            accuracies = [float(m['方向准确率'].replace('%', '')) for m in metrics_data]
            
            fig_acc.add_trace(go.Bar(
                x=horizons, y=accuracies,
                marker_color=['#2ecc71' if a > 60 else '#f39c12' if a > 50 else '#e74c3c' for a in accuracies],
                text=[f"{a:.1f}%" for a in accuracies],
                textposition='outside'
            ))
            fig_acc.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="随机基准 50%")
            fig_acc.update_layout(
                title="各周期方向准确率",
                xaxis_title="预测周期", yaxis_title="准确率 (%)",
                height=300, yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # 模型解读
        horizon_meta = meta.get('metrics', {}).get(horizon, {})
        if horizon_meta:
            r2 = horizon_meta.get('r2', 0)
            dir_acc = horizon_meta.get('direction_accuracy', 0)
            
            st.markdown(f"""
            **当前选择: {horizon}**
            - R² = {r2:.3f}: {"优秀" if r2 > 0.5 else "良好" if r2 > 0.3 else "一般" if r2 > 0.1 else "较弱"} (解释了 {r2*100:.1f}% 的收益变化)
            - 方向准确率 = {dir_acc:.1%}: {"优秀" if dir_acc > 0.7 else "良好" if dir_acc > 0.6 else "一般" if dir_acc > 0.55 else "较弱"}
            """)
    
    with model_tab2:
        st.markdown("**Signal Ranker** - 排序最可能赚钱的股票 (短/中/长线)")
        
        if ranker_meta.get('metrics'):
            ranker_data = []
            horizon_labels = {'short': '短线 (1-5天)', 'medium': '中线 (10-30天)', 'long': '长线 (60+天)'}
            
            for h, m in ranker_meta.get('metrics', {}).items():
                ranker_data.append({
                    '周期': horizon_labels.get(h, h),
                    'NDCG@10': f"{m.get('ndcg@10', 0):.3f}",
                    'Top10平均收益': f"{m.get('top10_avg_return', 0):+.2f}%",
                    '训练样本': m.get('train_samples', 0),
                    '分组数': m.get('n_groups', 0)
                })
            
            ranker_df = pd.DataFrame(ranker_data)
            ranker_df.columns = ['周期', 'NDCG', 'Top10收益', '样本', '分组']
            st.dataframe(ranker_df, hide_index=True, use_container_width=True)
            
            st.markdown("""
            **指标说明:**
            - **NDCG@10**: 归一化折损累积增益，越接近 1 排序质量越好
            - **Top10平均收益**: 排名前 10 的股票平均实际收益
            """)
        else:
            st.info("排序模型未训练")
    
    with model_tab3:
        st.markdown("**特征重要性** - 哪些特征对预测最重要")
        
        try:
            import joblib
            model_path = model_dir / f"return_{horizon}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                feature_names = meta.get('feature_names', [])
                
                if hasattr(model, 'feature_importances_') and feature_names:
                    importance = dict(zip(feature_names, model.feature_importances_))
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Top 20 特征
                    top20 = sorted_imp[:20]
                    
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        y=[f[0] for f in top20][::-1],
                        x=[f[1] for f in top20][::-1],
                        orientation='h',
                        marker_color='steelblue'
                    ))
                    fig_imp.update_layout(
                        title=f"Top 20 重要特征 ({horizon})",
                        xaxis_title="重要性得分",
                        height=500,
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # 特征分类统计
                    categories = {
                        '均线特征': [f for f in feature_names if 'ma_' in f or 'ema_' in f],
                        '动量特征': [f for f in feature_names if 'momentum' in f or 'roc' in f or 'return' in f],
                        '波动率特征': [f for f in feature_names if 'volatility' in f or 'atr' in f],
                        'RSI特征': [f for f in feature_names if 'rsi' in f],
                        'MACD特征': [f for f in feature_names if 'macd' in f],
                        'KDJ特征': [f for f in feature_names if 'kdj' in f],
                        '布林带特征': [f for f in feature_names if 'bb_' in f],
                        '成交量特征': [f for f in feature_names if 'volume' in f or 'obv' in f],
                        'K线形态': [f for f in feature_names if 'body' in f or 'shadow' in f or 'doji' in f or 'hammer' in f],
                        'BLUE信号': [f for f in feature_names if 'blue' in f],
                    }
                    
                    cat_importance = []
                    for cat, feats in categories.items():
                        total_imp = sum(importance.get(f, 0) for f in feats)
                        cat_importance.append({'类别': cat, '总重要性': total_imp, '特征数': len(feats)})
                    
                    cat_df = pd.DataFrame(cat_importance).sort_values('总重要性', ascending=False)
                    cat_df['总重要性'] = cat_df['总重要性'].apply(lambda x: f"{x:.4f}")
                    cat_df.columns = ['类别', '重要性', '特征数']
                    
                    st.markdown("**特征类别重要性汇总:**")
                    st.dataframe(cat_df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(f"无法加载特征重要性: {e}")
    
    with model_tab4:
        st.markdown("**Hyperparameter Tuning** - GridSearch 找最优参数")
        
        # 检查是否有调优结果
        tuning_path = model_dir.parent.parent / 'tuning_results' / market.lower() / 'best_params.json'
        
        if tuning_path.exists():
            with open(tuning_path) as f:
                best_params = json.load(f)
            
            st.success("✅ 已有调优结果")
            
            # 显示最优参数
            for model_key, params in best_params.items():
                with st.expander(f"📊 {model_key}", expanded=True):
                    params_df = pd.DataFrame([
                        {'参数': k, '最优值': v} for k, v in params.items()
                    ])
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
            
            # 加载调优历史
            history_path = tuning_path.parent / 'tuning_history.json'
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                
                if history:
                    st.markdown("**调优效果对比:**")
                    history_df = pd.DataFrame(history)
                    history_df['提升'] = history_df['improvement'].apply(lambda x: f"{x:+.1f}%")
                    history_df['最优分数'] = history_df['best_score'].apply(lambda x: f"{x:.3f}")
                    history_df['默认分数'] = history_df['default_score'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(
                        history_df[['horizon', '默认分数', '最优分数', '提升']].rename(
                            columns={'horizon': '周期'}
                        ),
                        hide_index=True, use_container_width=True
                    )
        else:
            st.info("暂无调优结果")
        
        st.markdown("---")
        
        # 调优按钮
        col1, col2 = st.columns(2)
        with col1:
            fast_mode = st.checkbox("快速模式", value=True, help="使用较小的搜索空间")
        with col2:
            n_iter = st.slider("搜索次数", 10, 100, 30, help="RandomizedSearch 迭代次数")
        
        if st.button("🔧 开始调优", key="start_tuning", type="primary"):
            with st.spinner("调优中... (可能需要几分钟)"):
                try:
                    from ml.hyperparameter_tuning import run_tuning
                    results = run_tuning(market=market, fast=fast_mode)
                    
                    if results:
                        st.success("✅ 调优完成!")
                        st.rerun()
                    else:
                        st.error("调优失败")
                except Exception as e:
                    st.error(f"调优出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("""
        **说明:**
        - 快速模式: 约 2-5 分钟
        - 完整模式: 约 10-30 分钟
        - 调优使用 5 折交叉验证
        - 优化目标: 方向准确率
        """)
    
    with model_tab5:
        st.markdown("**Model Comparison** - 独立模型 vs 串联模型")
        
        st.markdown("""
        **两种架构:**
        
        | 模式 | 架构 | 特点 |
        |------|------|------|
        | 独立模型 | ReturnPredictor + SignalRanker 各自独立 | 简单，训练快 |
        | 串联模型 | ReturnPredictor → 预测特征 → SignalRanker | Ranker可学习"哪些预测更可信" |
        """)
        
        # 检查是否有对比结果
        comparison_path = model_dir / 'model_comparison.json'
        
        if comparison_path.exists():
            with open(comparison_path) as f:
                comparison = json.load(f)
            
            st.success("✅ 已有对比结果")
            
            # 显示对比表
            if 'comparison' in comparison:
                comp_df = pd.DataFrame(comparison['comparison'])
                comp_df['independent_ndcg'] = comp_df['independent_ndcg'].apply(lambda x: f"{x:.3f}")
                comp_df['ensemble_ndcg'] = comp_df['ensemble_ndcg'].apply(lambda x: f"{x:.3f}")
                comp_df['improvement'] = comp_df['improvement'].apply(lambda x: f"{x:+.1f}%")
                comp_df.columns = ['周期', '独立模型 NDCG', '串联模型 NDCG', '提升']
                
                st.markdown("**排序模型 NDCG@10 对比:**")
                st.dataframe(comp_df, hide_index=True, use_container_width=True)
                
                # 添加特征信息
                if 'ensemble' in comparison:
                    added = comparison['ensemble'].get('added_features', [])
                    if added:
                        st.markdown(f"**串联模型新增特征:** `{', '.join(added)}`")
        else:
            st.info("暂无对比结果")
        
        st.markdown("---")
        
        if st.button("🔗 运行模型对比", key="run_comparison", type="primary"):
            with st.spinner("训练并对比中... (约 1-2 分钟)"):
                try:
                    from ml.pipeline import MLPipeline
                    from ml.models.ensemble_predictor import compare_models
                    
                    # 准备数据
                    pipeline = MLPipeline(market=market)
                    X, returns_dict, drawdowns_dict, groups, feature_names, _ = pipeline.prepare_dataset()
                    
                    if X is not None and len(X) > 0:
                        # 运行对比
                        results = compare_models(X, returns_dict, drawdowns_dict, groups, feature_names)
                        
                        # 保存结果
                        with open(comparison_path, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        st.success("✅ 对比完成!")
                        st.rerun()
                    else:
                        st.error("无法准备数据")
                except Exception as e:
                    st.error(f"对比出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("""
        **串联模型新增特征:**
        - `pred_return_1d/5d/10d/30d`: 预测收益
        - `pred_return_mean`: 预测收益均值
        - `pred_return_std`: 预测不确定性
        - `pred_momentum`: 长短期预测差异
        - `pred_direction_consistency`: 方向一致性
        """)
    
    st.divider()
    
    # ==================================
    # 当前周期的核心指标卡片
    # ==================================
    horizon_meta = meta.get('metrics', {}).get(horizon, {})
    if horizon_meta:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r2 = horizon_meta.get('r2', 0)
            st.metric("R²", f"{r2:.3f}", help="决定系数，越高模型解释力越强")
        with col2:
            dir_acc = horizon_meta.get('direction_accuracy', 0)
            delta = f"+{(dir_acc-0.5)*100:.0f}%" if dir_acc > 0.5 else f"{(dir_acc-0.5)*100:.0f}%"
            st.metric("方向准确率", f"{dir_acc:.1%}", delta=delta, help="预测涨跌方向的准确率")
        with col3:
            st.metric("RMSE", f"{horizon_meta.get('rmse', 0):.2f}%", help="均方根误差，越低越好")
        with col4:
            st.metric("样本数", f"{horizon_meta.get('train_samples', 0):,}", help="训练样本数量")
    
    st.divider()
    
    # === 加载今日信号 ===
    from db.database import get_connection
    from db.stock_history import get_stock_history
    from ml.features.feature_calculator import FeatureCalculator
    
    conn = get_connection()
    
    # 获取最新信号
    query = """
        SELECT DISTINCT symbol, scan_date, price, 
               COALESCE(blue_daily, 0) as blue_daily,
               COALESCE(blue_weekly, 0) as blue_weekly,
               COALESCE(blue_monthly, 0) as blue_monthly,
               COALESCE(is_heima, 0) as is_heima
        FROM scan_results
        WHERE market = ?
        ORDER BY scan_date DESC
        LIMIT 200
    """
    signals_df = pd.read_sql_query(query, conn, params=(market,))
    conn.close()
    
    if signals_df.empty:
        st.info("暂无信号数据")
        return
    
    latest_date = signals_df['scan_date'].iloc[0]
    today_signals = signals_df[signals_df['scan_date'] == latest_date]
    
    st.markdown(f"### 📈 {latest_date} 信号预测 ({len(today_signals)} 只)")
    
    # 加载模型
    try:
        import joblib
        return_model = joblib.load(model_dir / f"return_{horizon}.joblib")
        feature_names = json.load(open(model_dir / "feature_names.json"))
        
        # 为每个信号计算特征并预测
        calc = FeatureCalculator()
        predictions = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, (_, signal) in enumerate(today_signals.iterrows()):
            symbol = signal['symbol']
            
            # 获取历史数据
            history = get_stock_history(symbol, market, days=100)
            
            if history.empty or len(history) < 60:
                continue
            
            # 计算特征
            blue_signals = {
                'blue_daily': signal['blue_daily'],
                'blue_weekly': signal['blue_weekly'],
                'blue_monthly': signal['blue_monthly'],
                'is_heima': signal['is_heima'],
                'is_juedi': 0
            }
            
            features = calc.get_latest_features(history, blue_signals)
            
            if not features:
                continue
            
            # 准备特征向量
            X = np.array([[features.get(f, 0) for f in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e6, 1e6)
            
            # 预测
            pred_return = return_model.predict(X)[0]
            
            predictions.append({
                'symbol': symbol,
                'price': signal['price'],
                'blue_daily': signal['blue_daily'],
                'blue_weekly': signal['blue_weekly'],
                'is_heima': signal['is_heima'],
                f'pred_{horizon}': pred_return,
                'direction': '📈' if pred_return > 0 else '📉'
            })
            
            progress.progress((i + 1) / len(today_signals))
            status.text(f"处理: {symbol} ({i+1}/{len(today_signals)})")
        
        progress.empty()
        status.empty()
        
        if not predictions:
            st.warning("无法计算预测 (缺少历史数据)")
            return
        
        # 结果 DataFrame
        result_df = pd.DataFrame(predictions)
        result_df = result_df.sort_values(f'pred_{horizon}', ascending=False)
        result_df['rank'] = range(1, len(result_df) + 1)
        
        # === 显示 Top 10 ===
        st.markdown("### 🏆 Top 10 推荐")
        
        top10 = result_df.head(10).copy()
        top10['heima'] = top10['is_heima'].apply(lambda x: '⭐' if x else '')
        
        # 直接用 dataframe，列名简短
        show_cols = {
            'rank': '#',
            'symbol': '代码', 
            f'pred_{horizon}': '预测%',
            'direction': '↑↓',
            'blue_daily': '日B',
            'blue_weekly': '周B', 
            'heima': '🐴',
            'price': '$'
        }
        show_df = top10[list(show_cols.keys())].rename(columns=show_cols)
        show_df['预测%'] = show_df['预测%'].apply(lambda x: f"{x:+.1f}")
        show_df['$'] = show_df['$'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(show_df, hide_index=True, use_container_width=True)
        
        # === 预测分布 ===
        st.markdown("### 📊 预测分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 收益分布图
            fig = px.histogram(
                result_df, 
                x=f'pred_{horizon}',
                nbins=20,
                title=f"{horizon} 预测收益分布",
                labels={f'pred_{horizon}': '预测收益 (%)', 'count': '数量'}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 统计
            positive = (result_df[f'pred_{horizon}'] > 0).sum()
            negative = (result_df[f'pred_{horizon}'] <= 0).sum()
            avg_return = result_df[f'pred_{horizon}'].mean()
            
            st.metric("📈 预测上涨", f"{positive} 只")
            st.metric("📉 预测下跌", f"{negative} 只")
            st.metric("平均预测收益", f"{avg_return:+.1f}%")
        
        # === Bottom 10 ===
        with st.expander("📉 Bottom 10 (预测下跌最多)", expanded=False):
            bottom10 = result_df.tail(10).copy()
            bottom10 = bottom10.iloc[::-1]
            bottom10['heima'] = bottom10['is_heima'].apply(lambda x: '⭐' if x else '')
            
            show_cols = {
                'rank': '#',
                'symbol': '代码', 
                f'pred_{horizon}': '预测%',
                'direction': '↑↓',
                'blue_daily': '日B',
                'blue_weekly': '周B', 
                'heima': '🐴',
                'price': '$'
            }
            show_df2 = bottom10[list(show_cols.keys())].rename(columns=show_cols)
            show_df2['预测%'] = show_df2['预测%'].apply(lambda x: f"{x:+.1f}")
            show_df2['$'] = show_df2['$'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(show_df2, hide_index=True, use_container_width=True)
        
    except Exception as e:
        st.error(f"预测失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # ==================================
    # 📦 数据管理
    # ==================================
    st.divider()
    
    with st.expander("📦 数据管理", expanded=False):
        st.markdown("**历史K线数据** - 用于训练ML模型")
        
        # 数据统计
        try:
            from db.stock_history import get_history_stats
            from db.database import get_connection
            
            stats = get_history_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("K线股票数", stats.get('total_symbols', 0))
            with col2:
                st.metric("K线记录数", f"{stats.get('total_records', 0):,}")
            with col3:
                # 获取信号股票数
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(DISTINCT symbol) FROM scan_results WHERE market = ?', (market,))
                signal_count = cursor.fetchone()[0]
                conn.close()
                coverage = stats.get('total_symbols', 0) / signal_count * 100 if signal_count > 0 else 0
                st.metric("数据覆盖率", f"{coverage:.1f}%")
            
            # 缺失数据提示
            missing = signal_count - stats.get('total_symbols', 0)
            if missing > 0:
                st.warning(f"⚠️ 有 {missing} 只信号股票缺少历史数据")
        except Exception as e:
            st.warning(f"获取统计失败: {e}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            max_fetch = st.slider("获取数量", 50, 1000, 200, 50, help="一次获取多少只股票")
        with col2:
            fetch_days = st.slider("历史天数", 90, 730, 365, 30, help="获取多少天历史")
        
        if st.button("📥 获取更多数据", key="fetch_more_data"):
            with st.spinner(f"获取中... (约 {max_fetch * 0.5 / 60:.1f} 分钟)"):
                try:
                    from ml.batch_fetch_data import run_fetch
                    result = run_fetch(
                        market=market,
                        max_symbols=max_fetch,
                        days=fetch_days,
                        delay=0.3
                    )
                    
                    if result['success'] > 0:
                        st.success(f"✅ 获取完成! 成功: {result['success']}, 失败: {result['failed']}")
                        st.rerun()
                    else:
                        st.info("没有新数据需要获取")
                except Exception as e:
                    st.error(f"获取失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.caption("💡 数据来源: Polygon API (优先) / yfinance (备用)")


# --- V3 主导航 (精简版 8 Tabs) ---

st.sidebar.title("Coral Creek V3 🦅")
st.sidebar.caption("ML量化交易系统")

page = st.sidebar.radio("功能导航", [
    "🎯 今日精选",       # 新增: 多策略选股仪表板
    "📊 每日扫描", 
    "🔍 个股查询", 
    "📰 新闻中心",      # 新增: 事件驱动新闻分析
    "📈 信号中心",      # 合并: 信号追踪 + 验证 + Baseline对比
    "💼 组合管理",      # 合并: 持仓 + 风控仪表盘 + 模拟交易
    "🧪 策略实验室",    # 合并: 回测 + 研究工具
    "🤖 AI中心"         # 合并: AI决策 + 博主追踪
])

if page == "🎯 今日精选":
    render_todays_picks_page()
elif page == "📊 每日扫描":
    render_scan_page()
elif page == "🔍 个股查询":
    render_stock_lookup_page()
elif page == "📰 新闻中心":
    try:
        from pages.news_center import render_news_center_page
        render_news_center_page()
    except Exception as e:
        st.error(f"新闻中心加载失败: {e}")
        st.info("请确保 news 模块正确安装")
elif page == "📈 信号中心":
    render_signal_center_page()
elif page == "💼 组合管理":
    render_portfolio_management_page()
elif page == "🧪 策略实验室":
    render_strategy_lab_page()
elif page == "🤖 AI中心":
    render_ai_center_page()

