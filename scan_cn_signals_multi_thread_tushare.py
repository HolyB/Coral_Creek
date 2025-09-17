import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import tushare as ts
import threading
import concurrent.futures
from tqdm import tqdm
import os
import traceback
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# tushare 配置 - 请替换为你的真实token
TUSHARE_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'  # 请替换为你的tushare token
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 创建全局线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

class SignalNotifier:
    """股票信号邮件通知类"""
    def __init__(self, symbol, result_data):
        self.symbol = symbol
        self.data = result_data
        self.sender_email = "stockprofile138@gmail.com"
        self.receiver_emails = ["stockprofile138@gmail.com"]
        self.email_password = "vselpmwrjacmgdib"
    
    def send_signal_email(self):
        subject = f"A股股票信号通知: {self.symbol} 出现交易信号"
        
        body = f"股票代码: {self.symbol}\n"
        if 'name' in self.data:
            body += f"公司名称: {self.data['name']}\n"
        body += f"当前价格: {self.data['price']:.2f}\n"
        body += f"成交额(万): {self.data['turnover']:.2f}\n\n"
        
        body += "信号详情:\n"
        if self.data.get('has_day_blue', False):
            body += f"⭐ 日线BLUE: {self.data['blue_daily']:.2f}, 最近信号值: {self.data['latest_day_blue_value']:.2f}, 天数: {self.data['blue_days']}\n"
        if self.data.get('has_week_blue', False):
            body += f"⭐ 周线BLUE: {self.data['blue_weekly']:.2f}, 最近信号值: {self.data['latest_week_blue_value']:.2f}, 周数: {self.data['blue_weeks']}\n"
        if self.data.get('has_day_lired', False):
            body += f"⭐ 日线LIRED: {self.data['lired_daily']:.2f}, 最近信号值: {self.data['latest_day_lired_value']:.2f}, 天数: {self.data['lired_days']}\n"
        if self.data.get('has_week_lired', False):
            body += f"⭐ 周线LIRED: {self.data['lired_weekly']:.2f}, 最近信号值: {self.data['latest_week_lired_value']:.2f}, 周数: {self.data['lired_weeks']}\n"
        
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送 {self.symbol} 的信号通知邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.email_password)
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
        if not stocks_data:
            print("没有检测到股票信号，不发送邮件")
            return False
            
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com"]
        email_password = "vselpmwrjacmgdib"
        
        subject = f"A股股票信号批量通知: 检测到 {len(stocks_data)} 只股票信号"
        
        body = f"检测到 {len(stocks_data)} 只股票出现交易信号:\n\n"
        
        for i, stock in enumerate(stocks_data, 1):
            body += f"{i}. 股票代码: {stock['symbol']}\n"
            if 'name' in stock:
                body += f"   公司名称: {stock['name']}\n"
            body += f"   价格: {stock['price']:.2f}, 成交额(万): {stock['turnover']:.2f}\n"
            if stock.get('has_day_blue'):
                body += f"   ⭐ 日BLUE: {stock['latest_day_blue_value']:.2f} ({stock['blue_days']}天)\n"
            if stock.get('has_week_blue'):
                body += f"   ⭐ 周BLUE: {stock['latest_week_blue_value']:.2f} ({stock['blue_weeks']}周)\n"
            if stock.get('has_day_lired'):
                body += f"   ⭐ 日LIRED: {stock['latest_day_lired_value']:.2f} ({stock['lired_days']}天)\n"
            if stock.get('has_week_lired'):
                body += f"   ⭐ 周LIRED: {stock['latest_week_lired_value']:.2f} ({stock['lired_weeks']}周)\n"
            body += "\n"
        
        body += "\n信号统计总结:\n"
        body += "-" * 40 + "\n"
        signal_counts = {
            '日BLUE': len([s for s in stocks_data if s.get('has_day_blue')]),
            '周BLUE': len([s for s in stocks_data if s.get('has_week_blue')]),
            '日LIRED': len([s for s in stocks_data if s.get('has_day_lired')]),
            '周LIRED': len([s for s in stocks_data if s.get('has_week_lired')])
        }
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        body += "-" * 40 + "\n"
        body += f"共检测到 {len(stocks_data)} 只股票有信号\n"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送批量股票信号通知邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"批量股票信号通知邮件发送成功，包含 {len(stocks_data)} 只股票")
            return True
        except Exception as e:
            print(f"批量股票信号通知邮件发送失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False

    @staticmethod
    def send_summary_email(results, signal_counts):
        if results.empty:
            print("没有检测到股票信号，不发送总结邮件")
            return False
        
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com"]
        email_password = "vselpmwrjacmgdib"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"A股日周信号扫描总结 ({current_time})"
        
        body = f"A股日周信号扫描总结 ({current_time})\n\n"
        
        body += "信号统计总结:\n"
        body += "-" * 40 + "\n"
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        body += "-" * 40 + "\n"
        body += f"共发现 {len(results)} 只股票有信号\n\n"
        
        body += "股票列表：\n"
        body += "=" * 100 + "\n"
        body += f"{'代码':<8} | {'公司名称':<20} | {'价格':>8} | {'成交额(万)':>12} | {'信号':<40}\n"
        body += "-" * 100 + "\n"
        
        for _, row in results.iterrows():
            signals = []
            if row['has_day_blue']:
                signals.append(f"日BLUE(日:{row['latest_day_blue_value']:.2f})")
            if row['has_week_blue']:
                signals.append(f"周BLUE(周:{row['latest_week_blue_value']:.2f})")
            if row['has_day_lired']:
                signals.append(f"日LIRED(日:{row['latest_day_lired_value']:.2f})")
            if row['has_week_lired']:
                signals.append(f"周LIRED(周:{row['latest_week_lired_value']:.2f})")
            signals_str = ', '.join(signals)
            company_name = row.get('name', 'N/A')
            if len(company_name) > 20:
                company_name = company_name[:17] + "..."
            body += f"{row['symbol']:<8} | {company_name:<20} | {row['price']:8.2f} | {row['turnover']:12.2f} | {signals_str:<40}\n"
        
        body += "=" * 100 + "\n"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送A股日周信号总结邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"邮件发送成功，包含 {len(results)} 只股票")
            return True
        except Exception as e:
            print(f"邮件发送失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False

def get_cn_tickers():
    """使用tushare获取A股股票列表"""
    try:
        # 获取A股股票基本信息
        stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        
        if stock_info is None or stock_info.empty:
            logging.error("从tushare获取股票列表失败")
            return pd.DataFrame()
        
        # 过滤A股（排除指数、债券等）
        stock_info = stock_info[stock_info['ts_code'].str.contains(r'\.(SH|SZ|BJ)$', regex=True)]
        
        # 重命名列以保持与原代码一致
        tickers = []
        for _, row in stock_info.iterrows():
            tickers.append({
                'code': row['ts_code'],  # tushare格式：600000.SH
                'name': row['name']
            })
        
        logging.info(f"从tushare获取到 {len(tickers)} 只A股")
        return pd.DataFrame(tickers)
        
    except Exception as e:
        logging.error(f"获取A股列表失败: {e}")
        return pd.DataFrame()

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

def get_stock_data_tushare(ts_code, period='D', start_date=None, end_date=None):
    """使用tushare获取股票历史数据"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 使用pro_bar获取数据，支持日线和周线
        if period == 'D':
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        elif period == 'W':
            df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            # 使用pro_bar作为备选方案
            df = pro.pro_bar(ts_code=ts_code, freq=period, start_date=start_date, end_date=end_date)
        
        if df is None or df.empty:
            return None
        
        # 重命名列以与原代码保持一致
        df = df.rename(columns={
            'trade_date': 'Date',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        })
        
        # 确保有Date列且格式正确
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df = df.set_index('Date').sort_index()
        
        # 确保数值列是数值型
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logging.error(f"获取股票数据失败 {ts_code}: {e}")
        return None

def process_single_stock(stock, thresholds=None, require_both_timeframes=True, signal_type='both', min_turnover=100):
    """处理单个股票，支持配置日周信号是否需同时满足及信号类型
    Args:
        ...
        min_turnover: 最小成交额（万元），默认100万
    """
    default_thresholds = {
        'day_blue': 150,
        'day_lired': -150,
        'week_blue': 150,
        'week_lired': -150,
        'day_blue_count': 3,
        'week_blue_count': 2,
        'day_lired_count': 3,
        'week_lired_count': 2
    }
    if thresholds:
        default_thresholds.update(thresholds)
    
    symbol = stock['code']  # tushare格式：600000.SH
    name = stock['name']
    
    try:
        logging.info(f"开始处理股票: {symbol} {name}")
        
        # 使用tushare获取日线数据
        data_daily = get_stock_data_tushare(symbol, period='D')
        if data_daily is None or data_daily.empty:
            with print_lock:
                logging.warning(f"获取日线数据出错 ({symbol} {name}): 数据为空")
            return None
        
        # 生成周线数据
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        # 计算日线指标
        OPEN_D, HIGH_D, LOW_D, CLOSE_D = data_daily['Open'].values, data_daily['High'].values, data_daily['Low'].values, data_daily['Close'].values
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
        
        # 计算周线指标
        OPEN_W, HIGH_W, LOW_W, CLOSE_W = data_weekly['Open'].values, data_weekly['High'].values, data_weekly['Low'].values, data_weekly['Close'].values
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
        
        # 创建DataFrame
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'BLUE': BLUE_D, 'LIRED': LIRED_D
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'BLUE': BLUE_W, 'LIRED': LIRED_W
        }, index=data_weekly.index)
        
        # 分析最近信号
        recent_daily = df_daily.tail(6)
        recent_weekly = df_weekly.tail(5)
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        
        day_blue_signals = recent_daily[recent_daily['BLUE'] > default_thresholds['day_blue']]['BLUE'].tolist()
        week_blue_signals = recent_weekly[recent_weekly['BLUE'] > default_thresholds['week_blue']]['BLUE'].tolist()
        day_lired_signals = recent_daily[recent_daily['LIRED'] < default_thresholds['day_lired']]['LIRED'].tolist()
        week_lired_signals = recent_weekly[recent_weekly['LIRED'] < default_thresholds['week_lired']]['LIRED'].tolist()
        
        day_blue_count = len(day_blue_signals)
        week_blue_count = len(week_blue_signals)
        day_lired_count = len(day_lired_signals)
        week_lired_count = len(week_lired_signals)
        
        latest_day_blue_value = day_blue_signals[-1] if day_blue_signals else 0
        latest_week_blue_value = week_blue_signals[-1] if week_blue_signals else 0
        latest_day_lired_value = day_lired_signals[-1] if day_lired_signals else 0
        latest_week_lired_value = week_lired_signals[-1] if week_lired_signals else 0
        
        # 判断信号是否满足条件
        has_day_blue = day_blue_count >= default_thresholds['day_blue_count']
        has_week_blue = week_blue_count >= default_thresholds['week_blue_count']
        has_day_lired = day_lired_count >= default_thresholds['day_lired_count']
        has_week_lired = week_lired_count >= default_thresholds['week_lired_count']
        
        logging.info(f"{symbol} 信号检查: BLUE日={day_blue_count}, 周={week_blue_count}; LIRED日={day_lired_count}, 周={week_lired_count}")
        
        # 根据 require_both_timeframes 判断是否需要日周同时满足
        if require_both_timeframes:
            has_day_week_blue = has_day_blue and has_week_blue
            has_day_week_lired = has_day_lired and has_week_lired
        else:
            has_day_week_blue = has_day_blue or has_week_blue
            has_day_week_lired = has_day_lired or has_week_lired
        
        # 根据 signal_type 筛选信号
        if signal_type == 'blue':
            signal_detected = has_day_week_blue
            has_day_week_lired = False  # 忽略 LIRED
        elif signal_type == 'lired':
            signal_detected = has_day_week_lired
            has_day_week_blue = False  # 忽略 BLUE
        else:  # 'both'
            signal_detected = has_day_week_blue or has_day_week_lired
        
        if signal_detected:
            turnover = latest_daily['Volume'] * latest_daily['Close'] / 10000  # 转换为万元
            
            # 添加成交额过滤条件
            if turnover < min_turnover:
                logging.info(f"{symbol} 成交额{turnover:.2f}万元 < {min_turnover}万元，忽略信号")
                return None
            
            result = {
                'symbol': symbol,
                'name': name,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': turnover,
                'blue_daily': latest_daily['BLUE'],
                'blue_days': day_blue_count,
                'latest_day_blue_value': latest_day_blue_value,
                'blue_weekly': latest_weekly['BLUE'],
                'blue_weeks': week_blue_count,
                'latest_week_blue_value': latest_week_blue_value,
                'lired_daily': latest_daily['LIRED'],
                'lired_days': day_lired_count,
                'latest_day_lired_value': latest_day_lired_value,
                'lired_weekly': latest_weekly['LIRED'],
                'lired_weeks': week_lired_count,
                'latest_week_lired_value': latest_week_lired_value,
                'has_day_blue': has_day_blue,
                'has_week_blue': has_week_blue,
                'has_day_lired': has_day_lired,
                'has_week_lired': has_week_lired,
                'has_day_week_blue': has_day_blue and has_week_blue,
                'has_day_week_lired': has_day_lired and has_week_lired
            }
            logging.info(f"{symbol} 发现信号，成交额: {turnover:.2f}万元")
            return result
        
        return None
    
    except Exception as e:
        with print_lock:
            logging.error(f"获取数据出错 ({symbol} {name}): {str(e)}")
            traceback.print_exc()
        return None

def _scan_batch(batch, max_workers=5, max_wait_time=300, thresholds=None, require_both_timeframes=True, signal_type='both', min_turnover=100):
    results = []
    problem_stocks = []
    completed_count = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def process_result(future, stock):
        nonlocal completed_count
        try:
            result = future.result(timeout=10)
            if result:
                with results_lock:
                    results.append(result)
        except concurrent.futures.TimeoutError:
            with print_lock:
                logging.warning(f"{stock['code']} {stock['name']} 处理超时")
            with results_lock:
                problem_stocks.append(stock)
        except Exception as e:
            with print_lock:
                logging.error(f"{stock['code']} {stock['name']} 处理失败: {e}")
            with results_lock:
                problem_stocks.append(stock)
        
        with results_lock:
            completed_count += 1
            pbar.update(1)
    
    with tqdm(total=len(batch), desc="批次扫描进度") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_stock, stock, thresholds, require_both_timeframes, signal_type, min_turnover): stock for _, stock in batch.iterrows()}
            for future, stock in futures.items():
                future.add_done_callback(lambda f: process_result(f, stock))
            
            start_time = time.time()
            while completed_count < len(batch):
                if time.time() - start_time > max_wait_time:
                    with print_lock:
                        remaining = len(batch) - completed_count
                        logging.warning(f"已等待 {max_wait_time} 秒，仍有 {remaining} 只股票未完成。继续处理下一批...")
                    pbar.update(len(batch) - completed_count)
                    for future, stock in futures.items():
                        if not future.done():
                            with results_lock:
                                problem_stocks.append(stock)
                    break
                time.sleep(1)
    
    if problem_stocks:
        problem_df = pd.DataFrame(problem_stocks)
        problem_df.to_csv(f'problem_stocks_{timestamp}.csv', index=False, encoding='utf-8-sig')
        with print_lock:
            logging.info(f"本批次有 {len(problem_stocks)} 只问题股票，已保存到 problem_stocks_{timestamp}.csv")
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def scan_in_batches(batch_size=500, cooldown=30, max_workers=5, start_batch=1, end_batch=None, max_wait_time=300, thresholds=None, send_email=True, require_both_timeframes=True, signal_type='both', min_turnover=100):
    logging.info("正在获取A股列表...")
    stock_list = get_cn_tickers()
    
    if stock_list.empty:
        logging.error("获取股票列表失败")
        return pd.DataFrame()
    
    total_stocks = len(stock_list)
    logging.info(f"共获取到 {total_stocks} 只股票")
    
    batch_count = (total_stocks + batch_size - 1) // batch_size
    if end_batch is None:
        end_batch = batch_count
    
    start_batch = max(1, min(start_batch, batch_count))
    end_batch = max(start_batch, min(end_batch, batch_count))
    
    logging.info(f"将扫描第 {start_batch} 到 {end_batch} 批次，共 {end_batch-start_batch+1} 个批次")
    
    all_results = []
    
    for batch_num in range(start_batch, end_batch + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_stocks)
        batch = stock_list.iloc[start_idx:end_idx].reset_index(drop=True)
        
        logging.info(f"开始扫描第 {batch_num}/{batch_count} 批次 ({len(batch)} 只股票)...")
        
        batch_start_time = time.time()
        results_df = _scan_batch(batch, max_workers=max_workers, max_wait_time=max_wait_time, thresholds=thresholds, require_both_timeframes=require_both_timeframes, signal_type=signal_type, min_turnover=min_turnover)
        batch_end_time = time.time()
        
        if not results_df.empty:
            all_results.append(results_df)
            logging.info(f"批次 {batch_num} 发现 {len(results_df)} 只有信号的股票")
        else:
            logging.info(f"批次 {batch_num} 未发现信号")
        
        batch_time = batch_end_time - batch_start_time
        logging.info(f"批次 {batch_num} 处理耗时: {batch_time:.2f} 秒")
        
        if all_results:
            interim_results = pd.concat(all_results, ignore_index=True)
            interim_results.to_csv(f'cn_signals_interim_{batch_num}.csv', index=False, encoding='utf-8-sig')
            logging.info(f"已将中间结果保存到 cn_signals_interim_{batch_num}.csv")
        
        if batch_num < end_batch:
            logging.info(f"批次 {batch_num} 完成，休息 {cooldown} 秒...")
            time.sleep(cooldown)
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        if send_email and not final_results.empty:
            try:
                logging.info("准备发送信号总结邮件...")
                signal_counts = {
                    '日BLUE': len(final_results[final_results['has_day_blue'] == True]),
                    '周BLUE': len(final_results[final_results['has_week_blue'] == True]),
                    '日LIRED': len(final_results[final_results['has_day_lired'] == True]),
                    '周LIRED': len(final_results[final_results['has_week_lired'] == True])
                }
                SignalNotifier.send_summary_email(final_results, signal_counts)
            except Exception as e:
                logging.error(f"发送信号总结邮件失败: {e}")
                logging.error(traceback.format_exc())
        
        return final_results
    else:
        return pd.DataFrame()

def main(batch_size=500, max_workers=30, start_batch=1, end_batch=None, thresholds=None, send_email=True, require_both_timeframes=True, signal_type='both', min_turnover=100):
    default_thresholds = {
        'day_blue': 100,
        'day_lired': -100,
        'week_blue': 100,
        'week_lired': -100,
        'day_blue_count': 3,
        'week_blue_count': 2,
        'day_lired_count': 3,
        'week_lired_count': 2
    }
    if thresholds:
        default_thresholds.update(thresholds)
    
    logging.info("\n当前信号阈值配置:")
    logging.info(f"日线BLUE阈值: {default_thresholds['day_blue']}, 所需天数: {default_thresholds['day_blue_count']}")
    logging.info(f"日线LIRED阈值: {default_thresholds['day_lired']}, 所需天数: {default_thresholds['day_lired_count']}")
    logging.info(f"周线BLUE阈值: {default_thresholds['week_blue']}, 所需周数: {default_thresholds['week_blue_count']}")
    logging.info(f"周线LIRED阈值: {default_thresholds['week_lired']}, 所需周数: {default_thresholds['week_lired_count']}")
    logging.info(f"日周信号要求: {'同时满足' if require_both_timeframes else '单独满足即可'}")
    logging.info(f"信号类型: {signal_type} ({'只看BLUE' if signal_type == 'blue' else '只看LIRED' if signal_type == 'lired' else 'BLUE和LIRED都看'})")
    logging.info(f"最小成交额要求: {min_turnover}万元")
    
    start_time = time.time()
    
    os.makedirs("stock_cache", exist_ok=True)
    
    # 获取股票列表
    stock_list = get_cn_tickers()
    if not stock_list.empty:
        print("\nA股股票示例（前5个）：")
        print(stock_list.head().to_string(index=False))
        print(f"总计A股数量: {len(stock_list)}")
    
    # 扫描股票
    results = scan_in_batches(
        batch_size=batch_size,
        cooldown=30,
        max_workers=max_workers,
        start_batch=start_batch,
        end_batch=end_batch,
        max_wait_time=300,
        thresholds=default_thresholds,
        send_email=send_email,
        require_both_timeframes=require_both_timeframes,
        signal_type=signal_type,
        min_turnover=min_turnover
    )
    
    if not results.empty:
        print("\n发现信号的股票：")
        print("=" * 160)
        print(f"{'代码':<12} | {'公司名称':<20} | {'价格':>8} | {'成交额(万)':>12} | "
              f"{'日BLUE':>8} | {'最近日BLUE':>10} | {'周BLUE':>8} | {'最近周BLUE':>10} | "
              f"{'日LIRED':>8} | {'最近日LIRED':>10} | {'周LIRED':>8} | {'最近周LIRED':>10} | {'信号':<20}")
        print("-" * 160)
        
        signal_counts = {
            '日BLUE': 0,
            '周BLUE': 0,
            '日LIRED': 0,
            '周LIRED': 0
        }
        
        for _, row in results.iterrows():
            signals = []
            if row['has_day_blue']:
                signals.append(f'日BLUE(日:{row["latest_day_blue_value"]:.2f})')
                signal_counts['日BLUE'] += 1
            if row['has_week_blue']:
                signals.append(f'周BLUE(周:{row["latest_week_blue_value"]:.2f})')
                signal_counts['周BLUE'] += 1
            if row['has_day_lired']:
                signals.append(f'日LIRED(日:{row["latest_day_lired_value"]:.2f})')
                signal_counts['日LIRED'] += 1
            if row['has_week_lired']:
                signals.append(f'周LIRED(周:{row["latest_week_lired_value"]:.2f})')
                signal_counts['周LIRED'] += 1
            
            signals_str = ', '.join(signals)
            print(f"{row['symbol']:<12} | {row['name']:<20} | {row['price']:8.2f} | {row['turnover']:12.2f} | "
                  f"{row['blue_daily']:8.2f} | {row['latest_day_blue_value']:10.2f} | "
                  f"{row['blue_weekly']:8.2f} | {row['latest_week_blue_value']:10.2f} | "
                  f"{row['lired_daily']:8.2f} | {row['latest_day_lired_value']:10.2f} | "
                  f"{row['lired_weekly']:8.2f} | {row['latest_week_lired_value']:10.2f} | "
                  f"{signals_str:<20}")
        
        print("=" * 160)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, count in signal_counts.items():
            if count > 0:
                print(f"{signal:<15}: {count:>5} 只")
        print("-" * 40)
        print(f"共发现 {len(results)} 只股票有信号")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cn_signals_tushare_{timestamp}.csv'
        results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    # 示例：只看 BLUE 信号，日周单独满足，成交额大于100万
    main(
        batch_size=500,
        max_workers=30,
        start_batch=1,
        end_batch=None,
        thresholds=None,
        send_email=True,
        require_both_timeframes=False,
        signal_type='blue',
        min_turnover=100  # 设置最小成交额为100万
    ) 