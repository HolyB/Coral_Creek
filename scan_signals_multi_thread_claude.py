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
from concurrent.futures import ThreadPoolExecutor, as_completed

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局变量存储公司信息
COMPANY_INFO = {}

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

class SignalNotifier:
    """股票信号邮件通知类"""
    def __init__(self, symbol, result_data):
        """
        初始化通知类
        
        Args:
            symbol (str): 股票代码
            result_data (dict): 股票信号数据
        """
        self.symbol = symbol
        self.data = result_data
        self.sender_email = "stockprofile138@gmail.com"
        self.receiver_emails = ["stockprofile138@gmail.com"]  # 可以修改为你的邮箱
        self.email_password = "vselpmwrjacmgdib"  # 邮箱应用专用密码
    
    def send_signal_email(self):
        """发送股票信号邮件"""
        subject = f"股票信号通知: {self.symbol} 出现交易信号"
        
        # 构建邮件正文
        body = f"股票代码: {self.symbol}\n"
        if 'company_name' in self.data:
            body += f"公司名称: {self.data['company_name']}\n"
        body += f"当前价格: {self.data['price']:.2f}\n"
        body += f"成交额(万): {self.data['turnover']:.2f}\n\n"
        
        # 添加信号信息
        body += "信号详情:\n"
        body += f"日线BLUE: {self.data['blue_daily']:.2f}, 最近信号值: {self.data['latest_day_blue_value']:.2f}, 出现天数: {self.data['blue_days']}\n"
        body += f"周线BLUE: {self.data['blue_weekly']:.2f}, 最近信号值: {self.data['latest_week_blue_value']:.2f}, 出现周数: {self.data['blue_weeks']}\n"
        body += f"日线LIRED: {self.data['lired_daily']:.2f}, 最近信号值: {self.data['latest_day_lired_value']:.2f}, 出现天数: {self.data['lired_days']}\n"
        body += f"周线LIRED: {self.data['lired_weekly']:.2f}, 最近信号值: {self.data['latest_week_lired_value']:.2f}, 出现周数: {self.data['lired_weeks']}\n\n"
        
        # 添加组合信号信息
        signals = []
        if self.data['blue_days'] >= 3:
            signals.append(f"日BLUE: {self.data['latest_day_blue_value']:.2f}")
        if self.data['blue_weeks'] >= 2:
            signals.append(f"周BLUE: {self.data['latest_week_blue_value']:.2f}")
        if self.data['lired_days'] >= 3:
            signals.append(f"日LIRED: {self.data['latest_day_lired_value']:.2f}")
        if self.data['lired_weeks'] >= 2:
            signals.append(f"周LIRED: {self.data['latest_week_lired_value']:.2f}")
            
        if self.data['has_day_week_blue']:
            body += "⭐ 强信号: 日线和周线BLUE同时满足条件\n"
        if self.data['has_day_week_lired']:
            body += "⭐ 强信号: 日线和周线LIRED同时满足条件\n"
            
        body += f"\n检测到的信号组合: {', '.join(signals)}\n"
        
        # 设置邮件
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送 {self.symbol} 的信号通知邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            
            # 调试信息
            print(f"SMTP连接成功，尝试登录...")
            
            server.login(self.sender_email, self.email_password)
            
            print(f"登录成功，正在发送邮件...")
            
            server.sendmail(self.sender_email, self.receiver_emails, msg.as_string())
            server.quit()
            print(f"股票 {self.symbol} 的信号通知邮件发送成功")
            return True
        except Exception as e:
            print(f"股票 {self.symbol} 的信号通知邮件发送失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    @staticmethod
    def send_batch_signal_email(stocks_data):
        """
        发送批量股票信号邮件
        
        Args:
            stocks_data (list): 包含多只股票信号数据的列表
        """
        if not stocks_data:
            print("没有检测到股票信号，不发送邮件")
            return False
            
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com"]  # 可以修改为你的邮箱
        email_password = "vselpmwrjacmgdib"  # 邮箱应用专用密码
        
        subject = f"股票信号批量通知: 检测到 {len(stocks_data)} 只股票信号"
        
        # 构建邮件正文
        body = f"检测到 {len(stocks_data)} 只股票出现交易信号:\n\n"
        
        # 添加每只股票的信号信息
        for i, stock in enumerate(stocks_data, 1):
            body += f"{i}. 股票代码: {stock['symbol']}\n"
            
            if 'company_name' in stock:
                body += f"   公司名称: {stock['company_name']}\n"
            
            body += f"   价格: {stock['price']:.2f}, 成交额(万): {stock['turnover']:.2f}\n"
            
            # 添加信号信息
            signals = []
            if stock['blue_days'] >= 3:
                signals.append(f"日BLUE: {stock['latest_day_blue_value']:.2f}")
            if stock['blue_weeks'] >= 2:
                signals.append(f"周BLUE: {stock['latest_week_blue_value']:.2f}")
            if stock['lired_days'] >= 3:
                signals.append(f"日LIRED: {stock['latest_day_lired_value']:.2f}")
            if stock['lired_weeks'] >= 2:
                signals.append(f"周LIRED: {stock['latest_week_lired_value']:.2f}")
                
            body += f"   信号: {', '.join(signals)}\n"
            
            # 添加组合信号
            if stock['has_day_week_blue']:
                body += f"   ⭐ 日线和周线BLUE同时满足条件\n"
            if stock['has_day_week_lired']:
                body += f"   ⭐ 日线和周线LIRED同时满足条件\n"
                
            body += "\n"
            
        # 添加信号统计总结
        body += "\n信号统计总结:\n"
        body += "-" * 40 + "\n"
        
        # 统计各类信号出现的次数
        signal_counts = {
            '日BLUE': len([s for s in stocks_data if s['blue_days'] >= 3]),
            '周BLUE': len([s for s in stocks_data if s['blue_weeks'] >= 2]),
            '日LIRED': len([s for s in stocks_data if s['lired_days'] >= 3]),
            '周LIRED': len([s for s in stocks_data if s['lired_weeks'] >= 2]),
            '日周BLUE同时': len([s for s in stocks_data if s['has_day_week_blue']]),
            '日周LIRED同时': len([s for s in stocks_data if s['has_day_week_lired']])
        }
        
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        
        body += "-" * 40 + "\n"
        body += f"共检测到 {len(stocks_data)} 只股票有信号\n"
        
        # 添加日线和周线同时出现信号的股票表格
        dual_signal_stocks = [s for s in stocks_data if s['has_day_week_blue'] or s['has_day_week_lired']]
        if dual_signal_stocks:
            body += "\n\n日线和周线同时出现信号的股票：\n"
            body += "-" * 60 + "\n"
            body += f"{'代码':<8} | {'公司名称':<20} | {'价格':>8} | {'信号':<20}\n"
            body += "-" * 60 + "\n"
            
            for stock in dual_signal_stocks:
                signals = []
                if stock['has_day_week_blue']:
                    signals.append(f"日周BLUE同时")
                if stock['has_day_week_lired']:
                    signals.append(f"日周LIRED同时")
                
                signals_str = ', '.join(signals)
                company_name = stock.get('company_name', 'N/A')
                if len(company_name) > 20:
                    company_name = company_name[:17] + "..."
                
                body += f"{stock['symbol']:<8} | {company_name:<20} | {stock['price']:8.2f} | {signals_str:<20}\n"
            
            body += "-" * 60 + "\n"
            body += f"共发现 {len(dual_signal_stocks)} 只股票日线和周线同时出现信号\n"
        
        # 设置邮件
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送批量股票信号通知邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            
            # 调试信息
            print(f"SMTP连接成功，尝试登录...")
            
            server.login(sender_email, email_password)
            
            print(f"登录成功，正在发送邮件...")
            
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"批量股票信号通知邮件发送成功，包含 {len(stocks_data)} 只股票")
            return True
        except Exception as e:
            print(f"批量股票信号通知邮件发送失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False

    @staticmethod
    def send_summary_email(results, signal_counts, only_dual_signals=False):
        """
        发送信号总结邮件
        
        Args:
            results (DataFrame): 包含股票信号数据的DataFrame
            signal_counts (dict): 各类信号的统计字典
            only_dual_signals (bool, optional): 是否只发送日周同时有信号的股票. 默认为 False
        """
        if results.empty:
            print("没有检测到股票信号，不发送总结邮件")
            return False
        
        # 如果只发送日周同时有信号的股票，先筛选数据
        if only_dual_signals:
            dual_signal_stocks = results[(results['has_day_week_blue'] == True) | (results['has_day_week_lired'] == True)]
            if dual_signal_stocks.empty:
                print("没有检测到日周同时有信号的股票，不发送总结邮件")
                return False
            report_stocks = dual_signal_stocks
            report_title = "日周同时出现信号的股票"
        else:
            report_stocks = results
            report_title = "股票信号扫描总结"
            
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com"]  # 可以修改为你的邮箱
        email_password = "vselpmwrjacmgdib"  # 邮箱应用专用密码
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"{report_title} ({current_time})"
        
        # 构建邮件正文
        body = f"{report_title} ({current_time})\n\n"
        
        # 添加信号统计摘要
        body += "信号统计总结:\n"
        body += "-" * 40 + "\n"
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        body += "-" * 40 + "\n"
        body += f"共发现 {len(report_stocks)} 只股票有信号\n\n"
        
        # 添加股票列表
        body += f"股票列表：\n"
        body += "=" * 100 + "\n"
        body += f"{'代码':<8} | {'公司名称':<30} | {'价格':>8} | {'成交额(万)':>12} | {'信号':<40}\n"
        body += "-" * 100 + "\n"
        
        for _, row in report_stocks.iterrows():
            signals = []
            if row['has_day_week_blue']:
                signals.append(f"日周BLUE同时(日:{row['latest_day_blue_value']:.2f},周:{row['latest_week_blue_value']:.2f})")
            if row['has_day_week_lired']:
                signals.append(f"日周LIRED同时(日:{row['latest_day_lired_value']:.2f},周:{row['latest_week_lired_value']:.2f})")
            
            if not only_dual_signals:
                # 如果不是只发送日周同时信号，添加其他信号
                if row['blue_days'] >= 3 and not row['has_day_week_blue']:
                    signals.append(f"日BLUE({row['blue_days']}天)")
                if row['blue_weeks'] >= 2 and not row['has_day_week_blue']:
                    signals.append(f"周BLUE({row['blue_weeks']}周)")
                if row['lired_days'] >= 3 and not row['has_day_week_lired']:
                    signals.append(f"日LIRED({row['lired_days']}天)")
                if row['lired_weeks'] >= 2 and not row['has_day_week_lired']:
                    signals.append(f"周LIRED({row['lired_weeks']}周)")
            
            signals_str = ', '.join(signals)
            company_name = row.get('company_name', 'N/A')
            if len(company_name) > 30:
                company_name = company_name[:27] + "..."
            
            body += f"{row['symbol']:<8} | {company_name:<30} | {row['price']:8.2f} | {row['turnover']:12.2f} | {signals_str:<40}\n"
        
        body += "=" * 100 + "\n"
        
        # 设置邮件
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
    """使用多线程并发获取所有股票代码，优化为付费账户"""
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
    """完整中概股列表（保持不变）"""
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
    """获取单个股票的公司信息（保持不变）"""
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
    """获取公司信息字典，仅使用Polygon API，优先使用缓存（保持不变）"""
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
    """初始化公司信息（保持不变）"""
    global COMPANY_INFO
    print("\n正在初始化公司信息...")
    COMPANY_INFO = get_company_info()
    print(f"初始化完成，共加载 {len(COMPANY_INFO)} 家公司信息")

def process_single_stock(symbol, thresholds=None):
    """处理单个股票，仅关注BLUE和LIRED信号，调整为最近6天和5周，支持可配置阈值"""
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
        
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'VAR5': VAR5_D, 'VAR6': VAR6_D, 'VAR51': VAR51_D,
            'BLUE': BLUE_D, 'LIRED': LIRED_D
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'VAR5': VAR5_W, 'VAR6': VAR6_W, 'VAR51': VAR51_W,
            'BLUE': BLUE_W, 'LIRED': LIRED_W
        }, index=data_weekly.index)
        
        if symbol in ['PLTR', 'TSLA']:
            recent_daily_20 = df_daily.tail(20)
            recent_weekly = df_weekly.tail(5)  # 调整为5周
            latest_daily = df_daily.iloc[-1]
            latest_weekly = df_weekly.iloc[-1]
            
            with print_lock:
                print(f"\n=== {symbol} 中间变量检查 ===")
                print(f"日线RADIO1: {RADIO1_D}")
                print(f"周线RADIO1: {RADIO1_W}")
                print(f"日线VAR5（最近20天）:\n{df_daily['VAR5'].tail(20).tolist()}")
                print(f"日线VAR6（最近20天）:\n{df_daily['VAR6'].tail(20).tolist()}")
                print(f"日线VAR51（最近20天）:\n{df_daily['VAR51'].tail(20).tolist()}")
                print(f"周线VAR5（最近5周）:\n{df_weekly['VAR5'].tail(5).tolist()}")
                print(f"周线VAR6（最近5周）:\n{df_weekly['VAR6'].tail(5).tolist()}")
                print(f"周线VAR51（最近5周）:\n{df_weekly['VAR51'].tail(5).tolist()}")
                
                print(f"\n=== 调试 {symbol} 的所有数值 ===")
                print(f"日线数据（最新一天）:\n{latest_daily.to_dict()}")
                print(f"周线数据（最新一周）:\n{latest_weekly.to_dict()}")
                print(f"最近20天日线BLUE:\n{recent_daily_20['BLUE'].tolist()}")
                print(f"最近5周周线BLUE:\n{recent_weekly['BLUE'].tolist()}")
                print(f"最近20天日线LIRED:\n{recent_daily_20['LIRED'].tolist()}")
                print(f"最近5周周线LIRED:\n{recent_weekly['LIRED'].tolist()}")
        
        # 调整为最近6天和5周
        recent_daily = df_daily.tail(6)
        recent_weekly = df_weekly.tail(5)
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        
        # 查找满足条件的具体数值（使用可配置阈值）
        day_blue_signals = recent_daily[recent_daily['BLUE'] > default_thresholds['day_blue']]['BLUE'].tolist()
        week_blue_signals = recent_weekly[recent_weekly['BLUE'] > default_thresholds['week_blue']]['BLUE'].tolist()
        day_lired_signals = recent_daily[recent_daily['LIRED'] < default_thresholds['day_lired']]['LIRED'].tolist()
        week_lired_signals = recent_weekly[recent_weekly['LIRED'] < default_thresholds['week_lired']]['LIRED'].tolist()
        
        day_blue_count = len(day_blue_signals)
        week_blue_count = len(week_blue_signals)
        day_lired_count = len(day_lired_signals)
        week_lired_count = len(week_lired_signals)
        
        # 存储最近一次满足条件的信号值
        latest_day_blue_value = day_blue_signals[-1] if day_blue_signals else 0
        latest_week_blue_value = week_blue_signals[-1] if week_blue_signals else 0
        latest_day_lired_value = day_lired_signals[-1] if day_lired_signals else 0
        latest_week_lired_value = week_lired_signals[-1] if week_lired_signals else 0
        
        has_blue_signal = day_blue_count >= default_thresholds['day_blue_count'] or week_blue_count >= default_thresholds['week_blue_count']
        has_lired_signal = day_lired_count >= default_thresholds['day_lired_count'] or week_lired_count >= default_thresholds['week_lired_count']
        
        if has_blue_signal or has_lired_signal:
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': latest_daily['Volume'] * latest_daily['Close'] / 10000,  # 单位：万
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
                # 添加组合信号标志
                'has_day_week_blue': day_blue_count >= default_thresholds['day_blue_count'] and week_blue_count >= default_thresholds['week_blue_count'],
                'has_day_week_lired': day_lired_count >= default_thresholds['day_lired_count'] and week_lired_count >= default_thresholds['week_lired_count']
            }
            return result
        
    except Exception as e:
        with print_lock:
            print(f"{symbol} 处理出错: {e}")
            import traceback
            traceback.print_exc()
    return None

def main(limit=None, thresholds=None, send_email=True, only_dual_signals=False):
    """主函数，支持限制扫描数量，可配置信号阈值，可选择是否发送邮件以及只发送日周同时有信号的股票
    
    Args:
        limit (int, optional): 限制扫描的股票数量. 默认为 None (扫描全部)
        thresholds (dict, optional): 信号阈值配置. 默认为 None (使用默认阈值)
            格式: {
                'day_blue': 100,     # 日线BLUE信号阈值
                'day_lired': -100,   # 日线LIRED信号阈值
                'week_blue': 130,    # 周线BLUE信号阈值
                'week_lired': -130,  # 周线LIRED信号阈值
                'day_blue_count': 3, # 日线BLUE信号所需天数
                'week_blue_count': 2, # 周线BLUE信号所需周数
                'day_lired_count': 3, # 日线LIRED信号所需天数
                'week_lired_count': 2  # 周线LIRED信号所需周数
            }
        send_email (bool, optional): 是否发送邮件通知. 默认为 True
        only_dual_signals (bool, optional): 是否只发送日周同时有信号的股票. 默认为 False
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
    
    # 使用传入的阈值更新默认阈值
    if thresholds:
        default_thresholds.update(thresholds)
    
    # 打印当前使用的阈值配置
    print("\n当前信号阈值配置:")
    print(f"日线BLUE阈值: {default_thresholds['day_blue']}, 所需天数: {default_thresholds['day_blue_count']}")
    print(f"日线LIRED阈值: {default_thresholds['day_lired']}, 所需天数: {default_thresholds['day_lired_count']}")
    print(f"周线BLUE阈值: {default_thresholds['week_blue']}, 所需周数: {default_thresholds['week_blue_count']}")
    print(f"周线LIRED阈值: {default_thresholds['week_lired']}, 所需周数: {default_thresholds['week_lired_count']}")
    
    init_company_info()
    
    start_time = time.time()
    print("\n开始扫描股票...")
    
    results = scan_signals_parallel(max_workers=30, batch_size=500, cooldown=10, limit=limit, thresholds=default_thresholds, send_email=False)
    
    if not results.empty:
        results['company_name'] = results['symbol'].map(lambda x: COMPANY_INFO.get(x, 'N/A'))
        
        print("\n发现信号的股票（仅BLUE和LIRED，成交额>100万）：")
        print("=" * 180)
        print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交额(万)':>12} | {'日BLUE':>8} | {'日BLUE天数':>4} | {'最近日BLUE':>10} | {'日LIRED':>8} | "
              f"{'日LIRED数':>4} | {'最近日LIRED':>10} | {'周BLUE':>8} | {'周BLUE周数':>4} | {'最近周BLUE':>10} | {'周LIRED':>8} | {'周LIRED数':>4} | "
              f"{'最近周LIRED':>10} | {'信号':<20}")
        print("-" * 180)
        
        signal_counts = {
            '日BLUE>150': 0,
            '周BLUE>150': 0,
            '日LIRED<-170': 0,
            '周LIRED<-170': 0,
            '日周BLUE同时': 0,
            '日周LIRED同时': 0
        }
        
        count = 0
        for _, row in results.iterrows():
            signals = []
            if row['blue_days'] >= default_thresholds['day_blue_count']:
                signals.append(f'日BLUE>{default_thresholds["day_blue"]}({row["blue_days"]}天,{row["latest_day_blue_value"]:.2f})')
                signal_counts['日BLUE>150'] += 1
            if row['blue_weeks'] >= default_thresholds['week_blue_count']:
                signals.append(f'周BLUE>{default_thresholds["week_blue"]}({row["blue_weeks"]}周,{row["latest_week_blue_value"]:.2f})')
                signal_counts['周BLUE>150'] += 1
            if row['lired_days'] >= default_thresholds['day_lired_count']:
                signals.append(f'日LIRED<{default_thresholds["day_lired"]}({row["lired_days"]}天,{row["latest_day_lired_value"]:.2f})')
                signal_counts['日LIRED<-170'] += 1
            if row['lired_weeks'] >= default_thresholds['week_lired_count']:
                signals.append(f'周LIRED<{default_thresholds["week_lired"]}({row["lired_weeks"]}周,{row["latest_week_lired_value"]:.2f})')
                signal_counts['周LIRED<-170'] += 1
            
            # 统计日周同时出现信号的情况
            if row['has_day_week_blue']:
                signal_counts['日周BLUE同时'] += 1
            if row['has_day_week_lired']:
                signal_counts['日周LIRED同时'] += 1
            
            signals_str = ', '.join(signals)
            if signals_str:
                count += 1
                print(f"{row['symbol']:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['turnover']:12.2f} | "
                      f"{row['blue_daily']:8.2f} | {row['blue_days']:4d} | {row['latest_day_blue_value']:10.2f} | "
                      f"{row['lired_daily']:8.2f} | {row['lired_days']:4d} | {row['latest_day_lired_value']:10.2f} | "
                      f"{row['blue_weekly']:8.2f} | {row['blue_weeks']:4d} | {row['latest_week_blue_value']:10.2f} | "
                      f"{row['lired_weekly']:8.2f} | {row['lired_weeks']:4d} | {row['latest_week_lired_value']:10.2f} | "
                      f"{signals_str:<20}")
        
        print("=" * 180)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, count in signal_counts.items():
            if count > 0:
                print(f"{signal:<15}: {count:>5} 只")
        print("-" * 40)
        print(f"共发现 {count} 只股票有信号")
        
        # 添加新的表格：日线和周线同时出现信号
        dual_signal_stocks = results[(results['has_day_week_blue'] == True) | (results['has_day_week_lired'] == True)]
        
        if not dual_signal_stocks.empty:
            print("\n\n日线和周线同时出现信号的股票（日周BLUE或日周LIRED同时）：")
            print("=" * 160)
            print(f"{'代码':<8} | {'公司名称':<40} | {'价格':>8} | {'成交额(万)':>12} | "
                  f"{'日BLUE':>8} | {'最近日BLUE':>10} | {'周BLUE':>8} | {'最近周BLUE':>10} | "
                  f"{'日LIRED':>8} | {'最近日LIRED':>10} | {'周LIRED':>8} | {'最近周LIRED':>10} | {'信号':<20}")
            print("-" * 160)
            
            for _, row in dual_signal_stocks.iterrows():
                signals = []
                if row['has_day_week_blue']:
                    signals.append(f'日周BLUE同时(日:{row["latest_day_blue_value"]:.2f},周:{row["latest_week_blue_value"]:.2f})')
                if row['has_day_week_lired']:
                    signals.append(f'日周LIRED同时(日:{row["latest_day_lired_value"]:.2f},周:{row["latest_week_lired_value"]:.2f})')
                
                signals_str = ', '.join(signals)
                
                print(f"{row['symbol']:<8} | {row['company_name']:<40} | {row['price']:8.2f} | {row['turnover']:12.2f} | "
                      f"{row['blue_daily']:8.2f} | {row['latest_day_blue_value']:10.2f} | "
                      f"{row['blue_weekly']:8.2f} | {row['latest_week_blue_value']:10.2f} | "
                      f"{row['lired_daily']:8.2f} | {row['latest_day_lired_value']:10.2f} | "
                      f"{row['lired_weekly']:8.2f} | {row['latest_week_lired_value']:10.2f} | "
                      f"{signals_str:<20}")
            
            print("=" * 160)
            print(f"共发现 {len(dual_signal_stocks)} 只股票日线和周线同时出现信号")
        else:
            print("\n\n未发现日线和周线同时出现信号的股票")
        
        # 发送信号总结邮件
        if send_email:
            try:
                print("\n准备发送信号总结邮件...")
                # 创建信号统计字典
                signal_counts = {
                    '日BLUE>150': len(results[results['blue_days'] >= default_thresholds['day_blue_count']]),
                    '周BLUE>150': len(results[results['blue_weeks'] >= default_thresholds['week_blue_count']]),
                    '日LIRED<-170': len(results[results['lired_days'] >= default_thresholds['day_lired_count']]),
                    '周LIRED<-170': len(results[results['lired_weeks'] >= default_thresholds['week_lired_count']]),
                    '日周BLUE同时': len(results[results['has_day_week_blue'] == True]),
                    '日周LIRED同时': len(results[results['has_day_week_lired'] == True])
                }
                
                # 发送总结邮件，可以选择是否只发送日周同时有信号的股票
                SignalNotifier.send_summary_email(results, signal_counts, only_dual_signals=only_dual_signals)
            except Exception as e:
                print(f"发送信号总结邮件失败: {e}")
                print(f"详细错误信息: {traceback.format_exc()}")

    
def scan_signals_parallel(max_workers=30, batch_size=100, cooldown=5, limit=None, thresholds=None, send_email=True):
    """并行扫描股票信号，使用批处理和进度条，可限制扫描数量，支持可配置阈值，可选择是否发送邮件
    
    Args:
        max_workers (int, optional): 最大工作线程数. 默认为 30
        batch_size (int, optional): 批处理大小. 默认为 100
        cooldown (int, optional): 批处理间冷却时间(秒). 默认为 5
        limit (int, optional): 限制扫描的股票数量. 默认为 None (扫描全部)
        thresholds (dict, optional): 信号阈值配置. 默认为 None (使用默认阈值)
        send_email (bool, optional): 是否发送邮件通知. 默认为 True
    """
    tickers = get_all_tickers()
    
    for debug_symbol in ['PLTR', 'TSLA']:
        if debug_symbol not in tickers:
            tickers.insert(0, debug_symbol)
    
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
                futures = {executor.submit(process_single_stock, symbol, thresholds): symbol for symbol in batch_tickers}
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
    
    # 如果启用邮件通知且有结果，发送批量信号邮件
    if send_email and not results_df.empty:
        try:
            print("\n准备发送邮件通知...")
            # 转换DataFrame为字典列表以便邮件发送
            stocks_data = results_df.to_dict('records')
            # 发送邮件通知
            SignalNotifier.send_batch_signal_email(stocks_data)
        except Exception as e:
            print(f"发送批量信号通知邮件失败: {e}")
    
    return results_df

if __name__ == "__main__":
    # 默认扫描1000只股票并发送邮件通知，只报告日周同时有信号的股票
    main(limit=20000, send_email=True, only_dual_signals=False)