"""
股票信号扫描器 - 支持信号类型和周期过滤

用法示例:
    # 只看日线和周线BLUE信号
    python scan_signals_multi_thread_claude_filtered.py --signal blue
    
    # 只看日线和周线LIRED信号（做空）
    python scan_signals_multi_thread_claude_filtered.py --signal red
    
    # 只看日线BLUE信号
    python scan_signals_multi_thread_claude_filtered.py --signal blue --period daily
    
    # 只看周线BLUE信号
    python scan_signals_multi_thread_claude_filtered.py --signal blue --period weekly
    
    # 只看日线信号（BLUE和LIRED都看）
    python scan_signals_multi_thread_claude_filtered.py --period daily
    
    # 只看周线信号（BLUE和LIRED都看）
    python scan_signals_multi_thread_claude_filtered.py --period weekly
    
    # 只看BLUE+黑马信号同时出现的股票
    python scan_signals_multi_thread_claude_filtered.py --signal blue --with-heima
    
    # 只看日线BLUE+黑马信号同时出现
    python scan_signals_multi_thread_claude_filtered.py --signal blue --period daily --with-heima
    
    # 默认：扫描所有信号
    python scan_signals_multi_thread_claude_filtered.py
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import time
import threading
import concurrent.futures
from tqdm import tqdm
import requests
import os
import json
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局变量存储公司信息
COMPANY_INFO = {}

# 信号类型枚举
class SignalType:
    ALL = 'all'       # 所有信号
    BLUE = 'blue'     # 只看BLUE（多头）
    RED = 'red'       # 只看LIRED（空头）

# 周期类型枚举
class PeriodType:
    ALL = 'all'       # 日线和周线都看
    DAILY = 'daily'   # 只看日线
    WEEKLY = 'weekly' # 只看周线

# 定义富途函数
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
    """移动平均"""
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def AVEDEV(series, periods):
    """平均绝对偏差"""
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values

def TROUGHBARS(n, m, k):
    """
    简化版TROUGHBARS - 返回第k个波谷到当前的周期数
    这里简化实现：检测最近m周期内的局部最低点
    """
    # 这个函数在实际使用中需要根据具体逻辑实现
    # 这里返回一个简化版本
    return 0

def ZIG(n, m):
    """
    简化版ZIG指标 - 之字转向
    n: 类型 (3=收盘价)
    m: 转向幅度百分比
    返回之字转向值
    """
    # 简化实现：使用收盘价的局部极值点
    return None

def calculate_heima_signal(high, low, close, open_price):
    """
    计算黑马信号
    
    黑马信号条件:
    1. VAR2 < -110 (CCI类超卖指标)
    2. VAR4 > 0 (ZIG底部反转信号)
    
    返回: heima_signal (布尔数组), juedi_signal (掘底买点布尔数组)
    """
    n = len(close)
    
    # VAR1 = (HIGH + LOW + CLOSE) / 3
    VAR1 = (high + low + close) / 3
    
    # VAR2 = (VAR1 - MA(VAR1, 14)) / (0.015 * AVEDEV(VAR1, 14))
    # 这是一个类似CCI的指标
    ma_var1 = MA(VAR1, 14)
    avedev_var1 = AVEDEV(VAR1, 14)
    # 避免除零
    avedev_var1 = np.where(avedev_var1 == 0, 0.0001, avedev_var1)
    VAR2 = (VAR1 - ma_var1) / (0.015 * avedev_var1)
    
    # VAR3 = IF(TROUGHBARS(3,16,1)=0 AND HIGH>LOW+0.04, 80, 0)
    # 简化: 检测当前是否为局部低点且有一定振幅
    # 使用滚动窗口检测局部最低点
    low_series = pd.Series(low)
    is_local_low = (low_series == low_series.rolling(window=16, min_periods=1, center=True).min())
    has_amplitude = (high - low) > 0.04
    VAR3 = np.where(is_local_low & has_amplitude, 80, 0)
    
    # VAR4 = IF(ZIG(3,22)>REF(ZIG(3,22),1) AND REF(ZIG(3,22),1)<=REF(ZIG(3,22),2) AND REF(ZIG(3,22),2)<=REF(ZIG(3,22),3), 50, 0)
    # 简化: 检测价格从下降转为上升的拐点
    # 使用收盘价的变化来近似ZIG
    close_series = pd.Series(close)
    
    # 计算22周期的局部最低点
    rolling_min = close_series.rolling(window=22, min_periods=1).min()
    rolling_max = close_series.rolling(window=22, min_periods=1).max()
    
    # 检测底部反转: 当前价格开始上涨，且之前连续下跌
    pct_change = close_series.pct_change()
    is_rising = pct_change > 0
    was_falling_1 = pct_change.shift(1) <= 0
    was_falling_2 = pct_change.shift(2) <= 0
    was_falling_3 = pct_change.shift(3) <= 0
    
    # 额外条件: 接近局部最低点
    near_low = (close_series - rolling_min) / (rolling_max - rolling_min + 0.0001) < 0.2
    
    VAR4 = np.where(is_rising & was_falling_1 & was_falling_2 & was_falling_3 & near_low, 50, 0)
    
    # 黑马信号: VAR2 < -110 AND VAR4 > 0
    heima_signal = (VAR2 < -110) & (VAR4 > 0)
    
    # 掘底买点: VAR2 < -110 AND VAR3 > 0
    juedi_signal = (VAR2 < -110) & (VAR3 > 0)
    
    return heima_signal, juedi_signal, VAR2, VAR3, VAR4


class SignalNotifier:
    """股票信号邮件通知类"""
    def __init__(self, symbol, result_data):
        self.symbol = symbol
        self.data = result_data
        self.sender_email = "stockprofile138@gmail.com"
        self.receiver_emails = ["stockprofile138@gmail.com"]
        self.email_password = "vselpmwrjacmgdib"
    
    @staticmethod
    def send_summary_email(results, signal_counts, signal_type=SignalType.ALL, period_type=PeriodType.ALL, only_dual_signals=False):
        """发送信号总结邮件"""
        if results.empty:
            print("没有检测到股票信号，不发送总结邮件")
            return False
        
        # 构建报告标题
        signal_desc = {
            SignalType.ALL: "全部信号",
            SignalType.BLUE: "BLUE信号(多头)",
            SignalType.RED: "LIRED信号(空头)"
        }.get(signal_type, "全部信号")
        
        period_desc = {
            PeriodType.ALL: "日线+周线",
            PeriodType.DAILY: "仅日线",
            PeriodType.WEEKLY: "仅周线"
        }.get(period_type, "日线+周线")
        
        if only_dual_signals:
            dual_signal_stocks = results[(results.get('has_day_week_blue', False) == True) | 
                                        (results.get('has_day_week_lired', False) == True)]
            if dual_signal_stocks.empty:
                print("没有检测到日周同时有信号的股票，不发送总结邮件")
                return False
            report_stocks = dual_signal_stocks
            report_title = f"日周同时出现信号 - {signal_desc} ({period_desc})"
        else:
            report_stocks = results
            report_title = f"股票信号扫描 - {signal_desc} ({period_desc})"
            
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com"]
        email_password = "vselpmwrjacmgdib"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"{report_title} ({current_time})"
        
        # 构建邮件正文
        body = f"{report_title} ({current_time})\n\n"
        
        body += "信号统计总结:\n"
        body += "-" * 40 + "\n"
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        body += "-" * 40 + "\n"
        body += f"共发现 {len(report_stocks)} 只股票有信号\n\n"
        
        body += f"股票列表：\n"
        body += "=" * 100 + "\n"
        body += f"{'代码':<8} | {'公司名称':<30} | {'价格':>8} | {'成交额(万)':>12} | {'信号':<40}\n"
        body += "-" * 100 + "\n"
        
        for _, row in report_stocks.iterrows():
            signals = []
            if row.get('has_day_week_blue', False):
                signals.append(f"日周BLUE同时(日:{row.get('latest_day_blue_value', 0):.2f},周:{row.get('latest_week_blue_value', 0):.2f})")
            if row.get('has_day_week_lired', False):
                signals.append(f"日周LIRED同时(日:{row.get('latest_day_lired_value', 0):.2f},周:{row.get('latest_week_lired_value', 0):.2f})")
            
            if not only_dual_signals:
                if row.get('blue_days', 0) >= 3 and not row.get('has_day_week_blue', False):
                    signals.append(f"日BLUE({row['blue_days']}天)")
                if row.get('blue_weeks', 0) >= 2 and not row.get('has_day_week_blue', False):
                    signals.append(f"周BLUE({row['blue_weeks']}周)")
                if row.get('lired_days', 0) >= 3 and not row.get('has_day_week_lired', False):
                    signals.append(f"日LIRED({row['lired_days']}天)")
                if row.get('lired_weeks', 0) >= 2 and not row.get('has_day_week_lired', False):
                    signals.append(f"周LIRED({row['lired_weeks']}周)")
            
            signals_str = ', '.join(signals)
            company_name = row.get('company_name', 'N/A')
            if len(str(company_name)) > 30:
                company_name = str(company_name)[:27] + "..."
            
            body += f"{row['symbol']:<8} | {company_name:<30} | {row['price']:8.2f} | {row['turnover']:12.2f} | {signals_str:<40}\n"
        
        body += "=" * 100 + "\n"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送{report_title}邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"{report_title}邮件发送成功，包含 {len(report_stocks)} 只股票")
            return True
        except Exception as e:
            print(f"{report_title}邮件发送失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False


def fetch_tickers_page(cursor=None):
    """单个线程获取一页股票数据"""
    key = "qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D"
    base_url = "https://api.polygon.io/v3/reference/tickers"
    
    params = {
        'market': 'stocks',
        'active': True,
        'sort': 'ticker',
        'order': 'asc',
        'limit': 1000,
        'apiKey': key
    }
    if cursor:
        params['cursor'] = cursor
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            tickers = [item['ticker'] for item in data.get('results', []) if item.get('market') == 'stocks' and item.get('active')]
            next_cursor = data.get('next_url', '').split('cursor=')[1] if 'next_url' in data else None
            return tickers, next_cursor
        else:
            with print_lock:
                print(f"请求失败: {response.status_code} - {response.text}")
            return [], None
    except Exception as e:
        with print_lock:
            print(f"请求出错: {e}")
        return [], None


def get_all_tickers():
    """使用多线程并发获取所有股票代码"""
    ticker_cache_file = 'tickers_cache.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ticker_cache_path = os.path.join(current_dir, ticker_cache_file)
    
    if os.path.exists(ticker_cache_path):
        with open(ticker_cache_path, 'r', encoding='utf-8') as f:
            tickers = json.load(f)
            print(f"从缓存加载股票列表，共 {len(tickers)} 只股票")
    else:
        tickers = set()
        initial_cursor = None
        cursors = [initial_cursor]
        
        first_page_tickers, next_cursor = fetch_tickers_page()
        tickers.update(first_page_tickers)
        while next_cursor:
            cursors.append(next_cursor)
            _, next_cursor = fetch_tickers_page(next_cursor)
        
        max_workers = min(50, len(cursors))
        with print_lock:
            print(f"开始并发获取股票列表，总页数: {len(cursors)}，使用 {max_workers} 个线程")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cursor = {executor.submit(fetch_tickers_page, cursor): cursor for cursor in cursors[1:]}
            for future in tqdm(as_completed(future_to_cursor), total=len(cursors)-1, desc="Fetching Tickers"):
                page_tickers, _ = future.result()
                with results_lock:
                    tickers.update(page_tickers)
        
        chinese_stocks = additional_chinese_stocks()
        tickers.update(chinese_stocks)
        
        tickers = list(tickers)
        with open(ticker_cache_path, 'w', encoding='utf-8') as f:
            json.dump(tickers, f, ensure_ascii=False, indent=2)
        print(f"股票列表已缓存到 {ticker_cache_path}，共 {len(tickers)} 只股票")
    
    return tickers


def additional_chinese_stocks():
    """完整中概股列表"""
    chinese_tickers = [
        'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI', 'BILI', 'BIDU', 'NTES', 'TME', 
        'EDU', 'TAL', 'HTHT', 'GDS', 'IQ', 'KC', 'ATHM', 'HUYA', 'VIPS', 'ZH', 
        'DADA', 'BGNE', 'ZLAB', 'YUMC', 'MNSO', 'API', 'TIGR', 'FUTU', 'UP', 
        'QFIN', 'LU', 'BEKE', 'TCOM', 'ZTO', 'BZUN', 'WB', 'MOMO', 'YY', 'SOHU', 
        'NOAH', 'LX', 'FINV', 'GOTU', 'HOLI', 'NIU', 'TUYA', 'WBAI', 'JKS', 
        'DQ', 'CSIQ', 'RENN', 'LEJU', 'EH', 'CANG', 'UXIN', 'KNDI', 'CAAS', 
        'XNET', 'SOGO', 'WIMI', 'YRD', 'XYF', 'HUIZ', 'QTT', 'CCNC', 'CMCM', 
        'LIZI', 'TOUR', 'CTK', 'NCTY', 'ZJYL', 'AMBO', 'REDU', 'COE', 'ONE', 
        'DLNG', 'FENG', 'GLG', 'GRCL', 'JZ', 'TEDU', 'LKCO', 'AIHS', 'DTSS', 
        'XIN', 'SINO', 'QH', 'SEED', 'WAFU', 'WEI', 'CNTF', 'JRJC', 'BEDU', 
        'MOHO', 'RYB', 'SFUN', 'YIN', 'CNET', 'CCM', 'CLPS', 'DOGZ', 'HGSH', 
        'HLG', 'HX', 'NIU', 'OGEN', 'QSG', 'RLYB', 'SG', 'TC', 'UTME', 'ZCMD', 
        'PETZ', 'PHCF', 'RAAS', 'RCON', 'SDH', 'SNDA', 'SXTC', 'THTI', 'UCAR', 
        'XRS', 'YI', 'YJ', 'ZKIN'
    ]
    return list(set(chinese_tickers))


def fetch_company_info(ticker):
    """获取单个股票的公司信息"""
    api_key = "qFVjhzvJAyc0zsYIegmpUclYLoWKMh7D"
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results'].get('name'):
                return ticker, data['results']['name']
        return ticker, f"{ticker} Stock"
    except Exception as e:
        with print_lock:
            print(f"获取 {ticker} 信息失败: {e}")
        return ticker, f"{ticker} Stock"


def get_company_info():
    """获取公司信息字典"""
    cache_file = 'company_info_cache.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, cache_file)
    
    tickers = get_all_tickers()
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            company_dict = json.load(f)
            if len(company_dict) >= len(tickers) * 0.9:
                print(f"从缓存加载公司信息: {len(company_dict)} 家公司")
                return company_dict
            else:
                print(f"缓存数据不完整，仅有 {len(company_dict)} 家公司，重新获取缺失部分")
    else:
        company_dict = {}

    missing_tickers = [t for t in tickers if t not in company_dict]
    if missing_tickers:
        print(f"\n需要获取 {len(missing_tickers)} 只股票的信息")
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_ticker = {executor.submit(fetch_company_info, ticker): ticker for ticker in missing_tickers}
            for i, future in enumerate(tqdm(as_completed(future_to_ticker), total=len(missing_tickers), desc="Fetching Company Info")):
                ticker, name = future.result()
                with results_lock:
                    company_dict[ticker] = name
                if (i + 1) % 100 == 0:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(company_dict, f, ensure_ascii=False, indent=2)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(company_dict, f, ensure_ascii=False, indent=2)
        print("公司信息已保存到缓存")
    
    return company_dict


def init_company_info():
    """初始化公司信息"""
    global COMPANY_INFO
    print("\n正在初始化公司信息...")
    COMPANY_INFO = get_company_info()
    print(f"初始化完成，共加载 {len(COMPANY_INFO)} 家公司信息")


def process_single_stock(symbol, thresholds=None, signal_type=SignalType.ALL, period_type=PeriodType.ALL, with_heima=False):
    """
    处理单个股票，支持信号类型和周期过滤
    
    Args:
        symbol: 股票代码
        thresholds: 阈值配置
        signal_type: 信号类型 (all/blue/red)
        period_type: 周期类型 (all/daily/weekly)
        with_heima: 是否要求同时有黑马信号
    """
    # 设置默认阈值
    default_thresholds = {
        'day_blue': 100,
        'day_lired': -100,
        'week_blue': 130,
        'week_lired': -130,
        'day_blue_count': 3,
        'week_blue_count': 2,
        'day_lired_count': 3,
        'week_lired_count': 2,
        'heima_lookback': 6  # 黑马信号回看周期
    }
    
    if thresholds:
        default_thresholds.update(thresholds)
    
    try:
        fetcher_daily = StockDataFetcher(symbol, source='polygon', interval='1d')
        data_daily = fetcher_daily.get_stock_data()
        
        if data_daily is None or data_daily.empty:
            return None
            
        # 计算成交额（万元）并添加过滤
        latest_turnover = data_daily['Volume'].iloc[-1] * data_daily['Close'].iloc[-1] / 10000
        if latest_turnover < 100:  # 过滤成交额小于100万的股票
            return None
        
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        OPEN_D, HIGH_D, LOW_D, CLOSE_D = data_daily['Open'].values, data_daily['High'].values, data_daily['Low'].values, data_daily['Close'].values
        OPEN_W, HIGH_W, LOW_W, CLOSE_W = data_weekly['Open'].values, data_weekly['High'].values, data_weekly['Low'].values, data_weekly['Close'].values
        
        # 日线计算 - BLUE/LIRED
        VAR1_D = REF((LOW_D + OPEN_D + CLOSE_D + HIGH_D) / 4, 1)
        VAR2_D = SMA(np.abs(LOW_D - VAR1_D), 13, 1) / SMA(np.maximum(LOW_D - VAR1_D, 0), 10, 1)
        VAR3_D = EMA(VAR2_D, 10)
        VAR4_D = LLV(LOW_D, 33)
        VAR5_D = EMA(IF(LOW_D <= VAR4_D, VAR3_D, 0), 3)
        VAR6_D = POW(np.abs(VAR5_D), 0.3) * np.sign(VAR5_D)
        
        VAR21_D = SMA(np.abs(HIGH_D - VAR1_D), 13, 1) / SMA(np.minimum(HIGH_D - VAR1_D, 0), 10, 1)
        VAR31_D = EMA(VAR21_D, 10)
        VAR41_D = HHV(HIGH_D, 33)
        VAR51_D = EMA(IF(HIGH_D >= VAR41_D, -VAR31_D, 0), 3)
        VAR61_D = POW(np.abs(VAR51_D), 0.3) * np.sign(VAR51_D)
        
        max_value_daily = np.nanmax(np.maximum(VAR6_D, np.abs(VAR61_D)))
        RADIO1_D = 200 / max_value_daily if max_value_daily > 0 else 1
        BLUE_D = IF(VAR5_D > REF(VAR5_D, 1), VAR6_D * RADIO1_D, 0)
        LIRED_D = IF(VAR51_D > REF(VAR51_D, 1), -VAR61_D * RADIO1_D, 0)
        
        # 日线计算 - 黑马信号
        heima_daily, juedi_daily, cci_daily, var3_daily, var4_daily = calculate_heima_signal(
            HIGH_D, LOW_D, CLOSE_D, OPEN_D
        )
        
        # 周线计算 - BLUE/LIRED
        VAR1_W = REF((LOW_W + OPEN_W + CLOSE_W + HIGH_W) / 4, 1)
        VAR2_W = SMA(np.abs(LOW_W - VAR1_W), 13, 1) / SMA(np.maximum(LOW_W - VAR1_W, 0), 10, 1)
        VAR3_W = EMA(VAR2_W, 10)
        VAR4_W = LLV(LOW_W, 33)
        VAR5_W = EMA(IF(LOW_W <= VAR4_W, VAR3_W, 0), 3)
        VAR6_W = POW(np.abs(VAR5_W), 0.3) * np.sign(VAR5_W)
        
        VAR21_W = SMA(np.abs(HIGH_W - VAR1_W), 13, 1) / SMA(np.minimum(HIGH_W - VAR1_W, 0), 10, 1)
        VAR31_W = EMA(VAR21_W, 10)
        VAR41_W = HHV(HIGH_W, 33)
        VAR51_W = EMA(IF(HIGH_W >= VAR41_W, -VAR31_W, 0), 3)
        VAR61_W = POW(np.abs(VAR51_W), 0.3) * np.sign(VAR51_W)
        
        max_value_weekly = np.nanmax(np.maximum(VAR6_W, np.abs(VAR61_W)))
        RADIO1_W = 200 / max_value_weekly if max_value_weekly > 0 else 1
        BLUE_W = IF(VAR5_W > REF(VAR5_W, 1), VAR6_W * RADIO1_W, 0)
        LIRED_W = IF(VAR51_W > REF(VAR51_W, 1), -VAR61_W * RADIO1_W, 0)
        
        # 周线计算 - 黑马信号
        heima_weekly, juedi_weekly, cci_weekly, var3_weekly, var4_weekly = calculate_heima_signal(
            HIGH_W, LOW_W, CLOSE_W, OPEN_W
        )
        
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'VAR5': VAR5_D, 'VAR6': VAR6_D, 'VAR51': VAR51_D,
            'BLUE': BLUE_D, 'LIRED': LIRED_D,
            'heima': heima_daily, 'juedi': juedi_daily, 'cci': cci_daily
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'VAR5': VAR5_W, 'VAR6': VAR6_W, 'VAR51': VAR51_W,
            'BLUE': BLUE_W, 'LIRED': LIRED_W,
            'heima': heima_weekly, 'juedi': juedi_weekly, 'cci': cci_weekly
        }, index=data_weekly.index)
        
        # 调整为最近6天和5周
        heima_lookback = default_thresholds['heima_lookback']
        recent_daily = df_daily.tail(heima_lookback)
        recent_weekly = df_weekly.tail(5)
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        
        # 根据信号类型和周期过滤
        day_blue_signals = []
        week_blue_signals = []
        day_lired_signals = []
        week_lired_signals = []
        
        # BLUE信号
        if signal_type in [SignalType.ALL, SignalType.BLUE]:
            if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                day_blue_signals = recent_daily[recent_daily['BLUE'] > default_thresholds['day_blue']]['BLUE'].tolist()
            if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                week_blue_signals = recent_weekly[recent_weekly['BLUE'] > default_thresholds['week_blue']]['BLUE'].tolist()
        
        # LIRED信号
        if signal_type in [SignalType.ALL, SignalType.RED]:
            if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                day_lired_signals = recent_daily[recent_daily['LIRED'] < default_thresholds['day_lired']]['LIRED'].tolist()
            if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                week_lired_signals = recent_weekly[recent_weekly['LIRED'] < default_thresholds['week_lired']]['LIRED'].tolist()
        
        day_blue_count = len(day_blue_signals)
        week_blue_count = len(week_blue_signals)
        day_lired_count = len(day_lired_signals)
        week_lired_count = len(week_lired_signals)
        
        # 黑马信号统计 - 最近N个周期内是否出现过
        day_heima_count = recent_daily['heima'].sum()  # 日线黑马信号出现次数
        week_heima_count = recent_weekly['heima'].sum()  # 周线黑马信号出现次数
        day_juedi_count = recent_daily['juedi'].sum()  # 日线掘底信号出现次数
        week_juedi_count = recent_weekly['juedi'].sum()  # 周线掘底信号出现次数
        
        has_day_heima = day_heima_count > 0
        has_week_heima = week_heima_count > 0
        has_day_juedi = day_juedi_count > 0
        has_week_juedi = week_juedi_count > 0
        
        # 存储最近一次满足条件的信号值
        latest_day_blue_value = day_blue_signals[-1] if day_blue_signals else 0
        latest_week_blue_value = week_blue_signals[-1] if week_blue_signals else 0
        latest_day_lired_value = day_lired_signals[-1] if day_lired_signals else 0
        latest_week_lired_value = week_lired_signals[-1] if week_lired_signals else 0
        
        # 根据过滤条件判断是否有信号
        has_blue_signal = False
        has_lired_signal = False
        
        if signal_type in [SignalType.ALL, SignalType.BLUE]:
            if period_type == PeriodType.DAILY:
                has_blue_signal = day_blue_count >= default_thresholds['day_blue_count']
            elif period_type == PeriodType.WEEKLY:
                has_blue_signal = week_blue_count >= default_thresholds['week_blue_count']
            else:  # ALL
                has_blue_signal = day_blue_count >= default_thresholds['day_blue_count'] or week_blue_count >= default_thresholds['week_blue_count']
        
        if signal_type in [SignalType.ALL, SignalType.RED]:
            if period_type == PeriodType.DAILY:
                has_lired_signal = day_lired_count >= default_thresholds['day_lired_count']
            elif period_type == PeriodType.WEEKLY:
                has_lired_signal = week_lired_count >= default_thresholds['week_lired_count']
            else:  # ALL
                has_lired_signal = day_lired_count >= default_thresholds['day_lired_count'] or week_lired_count >= default_thresholds['week_lired_count']
        
        # 如果要求同时有黑马信号，则需要额外检查
        if with_heima:
            has_heima = False
            if period_type == PeriodType.DAILY:
                has_heima = has_day_heima or has_day_juedi
            elif period_type == PeriodType.WEEKLY:
                has_heima = has_week_heima or has_week_juedi
            else:  # ALL
                has_heima = has_day_heima or has_week_heima or has_day_juedi or has_week_juedi
            
            # 只有同时有BLUE/LIRED信号和黑马信号才返回
            if not has_heima:
                return None
        
        if has_blue_signal or has_lired_signal:
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'] / 10000,
                'blue_daily': latest_daily['BLUE'],
                'max_blue_daily': recent_daily['BLUE'].max(),
                'blue_days': day_blue_count,
                'latest_day_blue_value': latest_day_blue_value,
                'blue_weekly': latest_weekly['BLUE'],
                'max_blue_weekly': recent_weekly['BLUE'].max(),
                'blue_weeks': week_blue_count, 
                'latest_week_blue_value': latest_week_blue_value,
                'lired_daily': latest_daily['LIRED'],
                'lired_days': day_lired_count,
                'latest_day_lired_value': latest_day_lired_value,
                'lired_weekly': latest_weekly['LIRED'],
                'lired_weeks': week_lired_count,
                'latest_week_lired_value': latest_week_lired_value,
                'has_day_week_blue': day_blue_count >= default_thresholds['day_blue_count'] and week_blue_count >= default_thresholds['week_blue_count'],
                'has_day_week_lired': day_lired_count >= default_thresholds['day_lired_count'] and week_lired_count >= default_thresholds['week_lired_count'],
                # 黑马信号相关
                'day_heima_count': int(day_heima_count),
                'week_heima_count': int(week_heima_count),
                'day_juedi_count': int(day_juedi_count),
                'week_juedi_count': int(week_juedi_count),
                'has_day_heima': has_day_heima,
                'has_week_heima': has_week_heima,
                'has_day_juedi': has_day_juedi,
                'has_week_juedi': has_week_juedi,
                'latest_cci_daily': float(latest_daily['cci']) if not np.isnan(latest_daily['cci']) else 0,
                'latest_cci_weekly': float(latest_weekly['cci']) if not np.isnan(latest_weekly['cci']) else 0
            }
            return result
        
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
            traceback.print_exc()
    return None


def scan_signals_parallel(max_workers=30, batch_size=100, cooldown=5, limit=None, thresholds=None, 
                          signal_type=SignalType.ALL, period_type=PeriodType.ALL, send_email=True, with_heima=False):
    """并行扫描股票信号"""
    tickers = get_all_tickers()
    
    if limit is not None:
        tickers = tickers[:limit]
        print(f"限制扫描前 {limit} 只股票，实际扫描 {len(tickers)} 只")
    else:
        print(f"扫描全部 {len(tickers)} 只股票")
    
    all_results = []
    batch_count = (len(tickers) + batch_size - 1) // batch_size
    
    with tqdm(total=batch_count, desc="Batch Progress") as batch_pbar:
        for i in range(0, len(tickers), batch_size):
            batch_start_time = time.time()
            batch_tickers = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            with print_lock:
                print(f"\nProcessing batch {batch_num}/{batch_count} ({len(batch_tickers)} stocks)")
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_stock, symbol, thresholds, signal_type, period_type, with_heima): symbol for symbol in batch_tickers}
                for future in tqdm(as_completed(futures), total=len(batch_tickers), desc="Stock Progress"):
                    result = future.result()
                    if result is not None:
                        with results_lock:
                            batch_results.append(result)
            
            if batch_results:
                all_results.extend(batch_results)
                with print_lock:
                    print(f"Batch {batch_num} found {len(batch_results)} stocks with signals")
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            with print_lock:
                print(f"Batch {batch_num} processing time: {batch_time:.2f} seconds")
            batch_pbar.update(1)
            
            if i + batch_size < len(tickers):
                with print_lock:
                    print(f"Cooldown for {cooldown} seconds before next batch...")
                time.sleep(cooldown)
    
    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return results_df


def main(limit=None, thresholds=None, send_email=True, only_dual_signals=False, 
         signal_type=SignalType.ALL, period_type=PeriodType.ALL, with_heima=False):
    """
    主函数
    
    Args:
        limit: 限制扫描的股票数量
        thresholds: 信号阈值配置
        send_email: 是否发送邮件通知
        only_dual_signals: 是否只发送日周同时有信号的股票
        signal_type: 信号类型 (all/blue/red)
        period_type: 周期类型 (all/daily/weekly)
        with_heima: 是否要求同时有黑马信号
    """
    # 设置默认阈值
    default_thresholds = {
        'day_blue': 100,
        'day_lired': -100,
        'week_blue': 130,
        'week_lired': -130,
        'day_blue_count': 3,
        'week_blue_count': 2,
        'day_lired_count': 3,
        'week_lired_count': 2
    }
    
    if thresholds:
        default_thresholds.update(thresholds)
    
    # 打印当前使用的配置
    signal_desc = {
        SignalType.ALL: "全部信号 (BLUE + LIRED)",
        SignalType.BLUE: "仅BLUE信号 (多头)",
        SignalType.RED: "仅LIRED信号 (空头)"
    }.get(signal_type, "全部信号")
    
    period_desc = {
        PeriodType.ALL: "日线 + 周线",
        PeriodType.DAILY: "仅日线",
        PeriodType.WEEKLY: "仅周线"
    }.get(period_type, "日线 + 周线")
    
    heima_desc = " + 黑马信号" if with_heima else ""
    
    print("\n" + "=" * 60)
    print(f"扫描模式: {signal_desc}{heima_desc}")
    print(f"周期范围: {period_desc}")
    if with_heima:
        print("黑马信号: 要求同时出现")
    print("=" * 60)
    
    print("\n当前信号阈值配置:")
    if signal_type in [SignalType.ALL, SignalType.BLUE]:
        if period_type in [PeriodType.ALL, PeriodType.DAILY]:
            print(f"日线BLUE阈值: {default_thresholds['day_blue']}, 所需天数: {default_thresholds['day_blue_count']}")
        if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
            print(f"周线BLUE阈值: {default_thresholds['week_blue']}, 所需周数: {default_thresholds['week_blue_count']}")
    if signal_type in [SignalType.ALL, SignalType.RED]:
        if period_type in [PeriodType.ALL, PeriodType.DAILY]:
            print(f"日线LIRED阈值: {default_thresholds['day_lired']}, 所需天数: {default_thresholds['day_lired_count']}")
        if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
            print(f"周线LIRED阈值: {default_thresholds['week_lired']}, 所需周数: {default_thresholds['week_lired_count']}")
    
    init_company_info()
    
    start_time = time.time()
    print("\n开始扫描股票...")
    
    results = scan_signals_parallel(
        max_workers=30, batch_size=500, cooldown=10, limit=limit, 
        thresholds=default_thresholds, signal_type=signal_type, 
        period_type=period_type, send_email=False, with_heima=with_heima
    )
    
    if not results.empty:
        results['company_name'] = results['symbol'].map(lambda x: COMPANY_INFO.get(x, 'N/A'))
        
        print(f"\n发现信号的股票 ({signal_desc}{heima_desc}, {period_desc})：")
        print("=" * 180)
        
        # 根据信号类型动态构建表头
        header_parts = [f"{'代码':<8}", f"{'公司名称':<40}", f"{'价格':>8}", f"{'成交额(万)':>12}"]
        
        if signal_type in [SignalType.ALL, SignalType.BLUE]:
            if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                header_parts.extend([f"{'日BLUE':>8}", f"{'日BLUE天数':>4}", f"{'最近日BLUE':>10}"])
            if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                header_parts.extend([f"{'周BLUE':>8}", f"{'周BLUE周数':>4}", f"{'最近周BLUE':>10}"])
        
        if signal_type in [SignalType.ALL, SignalType.RED]:
            if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                header_parts.extend([f"{'日LIRED':>8}", f"{'日LIRED数':>4}", f"{'最近日LIRED':>10}"])
            if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                header_parts.extend([f"{'周LIRED':>8}", f"{'周LIRED数':>4}", f"{'最近周LIRED':>10}"])
        
        header_parts.append(f"{'信号':<30}")
        print(" | ".join(header_parts))
        print("-" * 180)
        
        signal_counts = {}
        count = 0
        
        for _, row in results.iterrows():
            signals = []
            row_parts = [
                f"{row['symbol']:<8}",
                f"{row['company_name']:<40}",
                f"{row['price']:8.2f}",
                f"{row['turnover']:12.2f}"
            ]
            
            if signal_type in [SignalType.ALL, SignalType.BLUE]:
                if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                    row_parts.extend([
                        f"{row['blue_daily']:8.2f}",
                        f"{row['blue_days']:4d}",
                        f"{row['latest_day_blue_value']:10.2f}"
                    ])
                    if row['blue_days'] >= default_thresholds['day_blue_count']:
                        signals.append(f'日BLUE({row["blue_days"]}天)')
                        signal_counts['日BLUE'] = signal_counts.get('日BLUE', 0) + 1
                
                if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                    row_parts.extend([
                        f"{row['blue_weekly']:8.2f}",
                        f"{row['blue_weeks']:4d}",
                        f"{row['latest_week_blue_value']:10.2f}"
                    ])
                    if row['blue_weeks'] >= default_thresholds['week_blue_count']:
                        signals.append(f'周BLUE({row["blue_weeks"]}周)')
                        signal_counts['周BLUE'] = signal_counts.get('周BLUE', 0) + 1
            
            if signal_type in [SignalType.ALL, SignalType.RED]:
                if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                    row_parts.extend([
                        f"{row['lired_daily']:8.2f}",
                        f"{row['lired_days']:4d}",
                        f"{row['latest_day_lired_value']:10.2f}"
                    ])
                    if row['lired_days'] >= default_thresholds['day_lired_count']:
                        signals.append(f'日LIRED({row["lired_days"]}天)')
                        signal_counts['日LIRED'] = signal_counts.get('日LIRED', 0) + 1
                
                if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                    row_parts.extend([
                        f"{row['lired_weekly']:8.2f}",
                        f"{row['lired_weeks']:4d}",
                        f"{row['latest_week_lired_value']:10.2f}"
                    ])
                    if row['lired_weeks'] >= default_thresholds['week_lired_count']:
                        signals.append(f'周LIRED({row["lired_weeks"]}周)')
                        signal_counts['周LIRED'] = signal_counts.get('周LIRED', 0) + 1
            
            # 统计日周同时信号
            if row.get('has_day_week_blue', False):
                signal_counts['日周BLUE同时'] = signal_counts.get('日周BLUE同时', 0) + 1
            if row.get('has_day_week_lired', False):
                signal_counts['日周LIRED同时'] = signal_counts.get('日周LIRED同时', 0) + 1
            
            signals_str = ', '.join(signals)
            if signals_str:
                count += 1
                row_parts.append(f"{signals_str:<30}")
                print(" | ".join(row_parts))
        
        print("=" * 180)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, cnt in signal_counts.items():
            if cnt > 0:
                print(f"{signal:<15}: {cnt:>5} 只")
        print("-" * 40)
        print(f"共发现 {count} 只股票有信号")
        
        # 日周同时信号表格
        if period_type == PeriodType.ALL:
            dual_signal_stocks = results[
                (results.get('has_day_week_blue', False) == True) | 
                (results.get('has_day_week_lired', False) == True)
            ]
            
            if not dual_signal_stocks.empty:
                print("\n\n日线和周线同时出现信号的股票：")
                print("=" * 140)
                print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交额(万)':>12} | {'信号':<40}")
                print("-" * 140)
                
                for _, row in dual_signal_stocks.iterrows():
                    signals = []
                    if row.get('has_day_week_blue', False):
                        signals.append(f'日周BLUE(日:{row["latest_day_blue_value"]:.2f},周:{row["latest_week_blue_value"]:.2f})')
                    if row.get('has_day_week_lired', False):
                        signals.append(f'日周LIRED(日:{row["latest_day_lired_value"]:.2f},周:{row["latest_week_lired_value"]:.2f})')
                    
                    signals_str = ', '.join(signals)
                    print(f"{row['symbol']:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['turnover']:12.2f} | {signals_str:<40}")
                
                print("=" * 140)
                print(f"共发现 {len(dual_signal_stocks)} 只股票日线和周线同时出现信号")
        
        # 保存结果到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signal_suffix = signal_type if signal_type != SignalType.ALL else "all"
        period_suffix = period_type if period_type != PeriodType.ALL else "all"
        csv_filename = f"signals_{signal_suffix}_{period_suffix}_{timestamp}.csv"
        results.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {csv_filename}")
        
        # 发送邮件
        if send_email:
            try:
                print("\n准备发送信号总结邮件...")
                SignalNotifier.send_summary_email(
                    results, signal_counts, 
                    signal_type=signal_type, 
                    period_type=period_type,
                    only_dual_signals=only_dual_signals
                )
            except Exception as e:
                print(f"发送信号总结邮件失败: {e}")
                print(f"详细错误信息: {traceback.format_exc()}")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='股票信号扫描器 - 支持信号类型、周期过滤和黑马信号',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
    # 只看BLUE信号（多头）
    python scan_signals_multi_thread_claude_filtered.py --signal blue
    
    # 只看LIRED信号（空头）
    python scan_signals_multi_thread_claude_filtered.py --signal red
    
    # 只看日线BLUE信号
    python scan_signals_multi_thread_claude_filtered.py --signal blue --period daily
    
    # 只看周线BLUE信号
    python scan_signals_multi_thread_claude_filtered.py --signal blue --period weekly
    
    # 只看日线信号（BLUE和LIRED都看）
    python scan_signals_multi_thread_claude_filtered.py --period daily
    
    # 只看周线信号
    python scan_signals_multi_thread_claude_filtered.py --period weekly
    
    # 只看BLUE+黑马信号同时出现的股票
    python scan_signals_multi_thread_claude_filtered.py --signal blue --with-heima
    
    # 只看日线BLUE+黑马信号同时出现
    python scan_signals_multi_thread_claude_filtered.py --signal blue --period daily --with-heima
    
    # 限制扫描数量
    python scan_signals_multi_thread_claude_filtered.py --limit 1000
    
    # 不发送邮件
    python scan_signals_multi_thread_claude_filtered.py --no-email
        """
    )
    
    parser.add_argument(
        '--signal', '-s',
        choices=['all', 'blue', 'red'],
        default='all',
        help='信号类型: all=全部, blue=仅BLUE(多头), red=仅LIRED(空头)'
    )
    
    parser.add_argument(
        '--period', '-p',
        choices=['all', 'daily', 'weekly'],
        default='all',
        help='周期类型: all=日线+周线, daily=仅日线, weekly=仅周线'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=20000,
        help='限制扫描的股票数量 (默认: 20000)'
    )
    
    parser.add_argument(
        '--no-email',
        action='store_true',
        help='不发送邮件通知'
    )
    
    parser.add_argument(
        '--dual-only',
        action='store_true',
        help='只报告日周同时有信号的股票'
    )
    
    parser.add_argument(
        '--with-heima',
        action='store_true',
        help='要求同时出现黑马信号（★黑马信号或★掘底买点）'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(
        limit=args.limit,
        send_email=not args.no_email,
        only_dual_signals=args.dual_only,
        signal_type=args.signal,
        period_type=args.period,
        with_heima=args.with_heima
    )

