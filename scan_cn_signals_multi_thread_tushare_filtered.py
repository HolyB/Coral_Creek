"""
A股股票信号扫描器 - 支持信号类型、周期过滤和黑马信号

用法示例:
    # 只看BLUE信号（多头）
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue
    
    # 只看LIRED信号（空头）
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal red
    
    # 只看日线BLUE信号
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --period daily
    
    # 只看周线BLUE信号
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --period weekly
    
    # 只看BLUE+黑马信号同时出现的股票
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --with-heima
    
    # 只看日线BLUE+黑马信号同时出现
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --period daily --with-heima
    
    # 默认：扫描所有信号
    python scan_cn_signals_multi_thread_tushare_filtered.py
"""

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
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# tushare 配置
TUSHARE_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 创建全局线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

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
    """移动平均"""
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def AVEDEV(series, periods):
    """平均绝对偏差"""
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values


def calculate_heima_signal(high, low, close, open_price):
    """
    计算黑马信号
    
    黑马信号条件:
    1. VAR2 < -110 (CCI类超卖指标)
    2. VAR4 > 0 (ZIG底部反转信号)
    
    返回: heima_signal (布尔数组), juedi_signal (掘底买点布尔数组)
    """
    # VAR1 = (HIGH + LOW + CLOSE) / 3
    VAR1 = (high + low + close) / 3
    
    # VAR2 = (VAR1 - MA(VAR1, 14)) / (0.015 * AVEDEV(VAR1, 14))
    ma_var1 = MA(VAR1, 14)
    avedev_var1 = AVEDEV(VAR1, 14)
    avedev_var1 = np.where(avedev_var1 == 0, 0.0001, avedev_var1)
    VAR2 = (VAR1 - ma_var1) / (0.015 * avedev_var1)
    
    # VAR3: 检测局部低点且有一定振幅
    low_series = pd.Series(low)
    is_local_low = (low_series == low_series.rolling(window=16, min_periods=1, center=True).min())
    has_amplitude = (high - low) > 0.04
    VAR3 = np.where(is_local_low & has_amplitude, 80, 0)
    
    # VAR4: 检测价格从下降转为上升的拐点
    close_series = pd.Series(close)
    rolling_min = close_series.rolling(window=22, min_periods=1).min()
    rolling_max = close_series.rolling(window=22, min_periods=1).max()
    
    pct_change = close_series.pct_change()
    is_rising = pct_change > 0
    was_falling_1 = pct_change.shift(1) <= 0
    was_falling_2 = pct_change.shift(2) <= 0
    was_falling_3 = pct_change.shift(3) <= 0
    
    near_low = (close_series - rolling_min) / (rolling_max - rolling_min + 0.0001) < 0.2
    VAR4 = np.where(is_rising & was_falling_1 & was_falling_2 & was_falling_3 & near_low, 50, 0)
    
    # 黑马信号: VAR2 < -110 AND VAR4 > 0
    heima_signal = (VAR2 < -110) & (VAR4 > 0)
    
    # 掘底买点: VAR2 < -110 AND VAR3 > 0
    juedi_signal = (VAR2 < -110) & (VAR3 > 0)
    
    return heima_signal, juedi_signal, VAR2, VAR3, VAR4


# ==================== 邮件通知类 ====================

class SignalNotifier:
    """股票信号邮件通知类"""
    def __init__(self, symbol, result_data):
        self.symbol = symbol
        self.data = result_data
        self.sender_email = "stockprofile138@gmail.com"
        self.receiver_emails = ["stockprofile138@gmail.com"]
        self.email_password = "vselpmwrjacmgdib"
    
    @staticmethod
    def send_summary_email(results, signal_counts, signal_type=SignalType.ALL, period_type=PeriodType.ALL, with_heima=False):
        """发送信号总结邮件"""
        if results.empty:
            print("没有检测到股票信号，不发送总结邮件")
            return False
        
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
        
        heima_desc = " + 黑马信号" if with_heima else ""
        
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com"]
        email_password = "vselpmwrjacmgdib"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_title = f"A股信号扫描 - {signal_desc}{heima_desc} ({period_desc})"
        subject = f"{report_title} ({current_time})"
        
        body = f"{report_title} ({current_time})\n\n"
        
        body += "信号统计总结:\n"
        body += "-" * 40 + "\n"
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        body += "-" * 40 + "\n"
        body += f"共发现 {len(results)} 只股票有信号\n\n"
        
        body += "股票列表：\n"
        body += "=" * 100 + "\n"
        body += f"{'代码':<12} | {'公司名称':<20} | {'价格':>8} | {'成交额(万)':>12} | {'信号':<40}\n"
        body += "-" * 100 + "\n"
        
        for _, row in results.iterrows():
            signals = []
            if row.get('has_day_blue', False):
                signals.append(f"日BLUE({row.get('blue_days', 0)}天)")
            if row.get('has_week_blue', False):
                signals.append(f"周BLUE({row.get('blue_weeks', 0)}周)")
            if row.get('has_day_lired', False):
                signals.append(f"日LIRED({row.get('lired_days', 0)}天)")
            if row.get('has_week_lired', False):
                signals.append(f"周LIRED({row.get('lired_weeks', 0)}周)")
            if with_heima:
                if row.get('has_day_heima', False):
                    signals.append("日黑马")
                if row.get('has_week_heima', False):
                    signals.append("周黑马")
            
            signals_str = ', '.join(signals)
            company_name = row.get('name', 'N/A')
            if len(str(company_name)) > 20:
                company_name = str(company_name)[:17] + "..."
            
            body += f"{row['symbol']:<12} | {company_name:<20} | {row['price']:8.2f} | {row['turnover']:12.2f} | {signals_str:<40}\n"
        
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
            print(f"{report_title}邮件发送成功，包含 {len(results)} 只股票")
            return True
        except Exception as e:
            print(f"{report_title}邮件发送失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False


# ==================== 数据获取函数 ====================

def get_cn_tickers():
    """使用tushare获取A股股票列表"""
    try:
        stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        
        if stock_info is None or stock_info.empty:
            logging.error("从tushare获取股票列表失败")
            return pd.DataFrame()
        
        stock_info = stock_info[stock_info['ts_code'].str.contains(r'\.(SH|SZ|BJ)$', regex=True)]
        
        tickers = []
        for _, row in stock_info.iterrows():
            tickers.append({
                'code': row['ts_code'],
                'name': row['name']
            })
        
        logging.info(f"从tushare获取到 {len(tickers)} 只A股")
        return pd.DataFrame(tickers)
        
    except Exception as e:
        logging.error(f"获取A股列表失败: {e}")
        return pd.DataFrame()


def get_stock_data_tushare(ts_code, period='D', start_date=None, end_date=None):
    """使用tushare获取股票历史数据"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if period == 'D':
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        elif period == 'W':
            df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = pro.pro_bar(ts_code=ts_code, freq=period, start_date=start_date, end_date=end_date)
        
        if df is None or df.empty:
            return None
        
        df = df.rename(columns={
            'trade_date': 'Date',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        })
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df = df.set_index('Date').sort_index()
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logging.error(f"获取股票数据失败 {ts_code}: {e}")
        return None


# ==================== 核心处理函数 ====================

def process_single_stock(stock, thresholds=None, signal_type=SignalType.ALL, period_type=PeriodType.ALL, 
                         with_heima=False, min_turnover=100):
    """
    处理单个股票，支持信号类型、周期过滤和黑马信号
    
    Args:
        stock: 股票信息字典
        thresholds: 阈值配置
        signal_type: 信号类型 (all/blue/red)
        period_type: 周期类型 (all/daily/weekly)
        with_heima: 是否要求同时有黑马信号
        min_turnover: 最小成交额（万元）
    """
    default_thresholds = {
        'day_blue': 100,
        'day_lired': -100,
        'week_blue': 100,
        'week_lired': -100,
        'day_blue_count': 3,
        'week_blue_count': 2,
        'day_lired_count': 3,
        'week_lired_count': 2,
        'heima_lookback': 6
    }
    if thresholds:
        default_thresholds.update(thresholds)
    
    symbol = stock['code']
    name = stock['name']
    
    try:
        # 获取日线数据
        data_daily = get_stock_data_tushare(symbol, period='D')
        if data_daily is None or data_daily.empty:
            return None
        
        # 生成周线数据
        data_weekly = data_daily.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        if data_weekly.empty:
            return None
        
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
        heima_daily, juedi_daily, cci_daily, _, _ = calculate_heima_signal(HIGH_D, LOW_D, CLOSE_D, OPEN_D)
        
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
        heima_weekly, juedi_weekly, cci_weekly, _, _ = calculate_heima_signal(HIGH_W, LOW_W, CLOSE_W, OPEN_W)
        
        # 创建DataFrame
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'BLUE': BLUE_D, 'LIRED': LIRED_D,
            'heima': heima_daily, 'juedi': juedi_daily, 'cci': cci_daily
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'BLUE': BLUE_W, 'LIRED': LIRED_W,
            'heima': heima_weekly, 'juedi': juedi_weekly, 'cci': cci_weekly
        }, index=data_weekly.index)
        
        # 分析最近信号
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
        
        # 黑马信号统计
        day_heima_count = recent_daily['heima'].sum()
        week_heima_count = recent_weekly['heima'].sum()
        day_juedi_count = recent_daily['juedi'].sum()
        week_juedi_count = recent_weekly['juedi'].sum()
        
        has_day_heima = day_heima_count > 0 or day_juedi_count > 0
        has_week_heima = week_heima_count > 0 or week_juedi_count > 0
        
        latest_day_blue_value = day_blue_signals[-1] if day_blue_signals else 0
        latest_week_blue_value = week_blue_signals[-1] if week_blue_signals else 0
        latest_day_lired_value = day_lired_signals[-1] if day_lired_signals else 0
        latest_week_lired_value = week_lired_signals[-1] if week_lired_signals else 0
        
        # 判断信号
        has_day_blue = day_blue_count >= default_thresholds['day_blue_count']
        has_week_blue = week_blue_count >= default_thresholds['week_blue_count']
        has_day_lired = day_lired_count >= default_thresholds['day_lired_count']
        has_week_lired = week_lired_count >= default_thresholds['week_lired_count']
        
        # 根据过滤条件判断是否有信号
        has_blue_signal = False
        has_lired_signal = False
        
        if signal_type in [SignalType.ALL, SignalType.BLUE]:
            if period_type == PeriodType.DAILY:
                has_blue_signal = has_day_blue
            elif period_type == PeriodType.WEEKLY:
                has_blue_signal = has_week_blue
            else:
                has_blue_signal = has_day_blue or has_week_blue
        
        if signal_type in [SignalType.ALL, SignalType.RED]:
            if period_type == PeriodType.DAILY:
                has_lired_signal = has_day_lired
            elif period_type == PeriodType.WEEKLY:
                has_lired_signal = has_week_lired
            else:
                has_lired_signal = has_day_lired or has_week_lired
        
        # 如果要求同时有黑马信号
        if with_heima:
            has_heima = False
            if period_type == PeriodType.DAILY:
                has_heima = has_day_heima
            elif period_type == PeriodType.WEEKLY:
                has_heima = has_week_heima
            else:
                has_heima = has_day_heima or has_week_heima
            
            if not has_heima:
                return None
        
        if has_blue_signal or has_lired_signal:
            turnover = latest_daily['Volume'] * latest_daily['Close'] / 10000
            
            if turnover < min_turnover:
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
                'has_day_week_lired': has_day_lired and has_week_lired,
                # 黑马信号
                'day_heima_count': int(day_heima_count),
                'week_heima_count': int(week_heima_count),
                'day_juedi_count': int(day_juedi_count),
                'week_juedi_count': int(week_juedi_count),
                'has_day_heima': has_day_heima,
                'has_week_heima': has_week_heima,
                'latest_cci_daily': float(latest_daily['cci']) if not np.isnan(latest_daily['cci']) else 0,
                'latest_cci_weekly': float(latest_weekly['cci']) if not np.isnan(latest_weekly['cci']) else 0
            }
            return result
        
        return None
    
    except Exception as e:
        with print_lock:
            logging.error(f"处理股票出错 ({symbol} {name}): {str(e)}")
        return None


def _scan_batch(batch, max_workers=5, max_wait_time=300, thresholds=None, 
                signal_type=SignalType.ALL, period_type=PeriodType.ALL, 
                with_heima=False, min_turnover=100):
    """扫描一批股票"""
    results = []
    problem_stocks = []
    completed_count = 0
    
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
            futures = {
                executor.submit(process_single_stock, stock, thresholds, signal_type, period_type, with_heima, min_turnover): stock 
                for _, stock in batch.iterrows()
            }
            for future, stock in futures.items():
                future.add_done_callback(lambda f: process_result(f, stock))
            
            start_time = time.time()
            while completed_count < len(batch):
                if time.time() - start_time > max_wait_time:
                    with print_lock:
                        remaining = len(batch) - completed_count
                        logging.warning(f"已等待 {max_wait_time} 秒，仍有 {remaining} 只股票未完成")
                    pbar.update(len(batch) - completed_count)
                    for future, stock in futures.items():
                        if not future.done():
                            with results_lock:
                                problem_stocks.append(stock)
                    break
                time.sleep(1)
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def scan_in_batches(batch_size=500, cooldown=30, max_workers=5, start_batch=1, end_batch=None, 
                    max_wait_time=300, thresholds=None, send_email=True,
                    signal_type=SignalType.ALL, period_type=PeriodType.ALL, 
                    with_heima=False, min_turnover=100):
    """分批扫描股票"""
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
        results_df = _scan_batch(
            batch, max_workers=max_workers, max_wait_time=max_wait_time, 
            thresholds=thresholds, signal_type=signal_type, period_type=period_type,
            with_heima=with_heima, min_turnover=min_turnover
        )
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
            interim_results.to_csv(f'cn_signals_filtered_interim_{batch_num}.csv', index=False, encoding='utf-8-sig')
        
        if batch_num < end_batch:
            logging.info(f"批次 {batch_num} 完成，休息 {cooldown} 秒...")
            time.sleep(cooldown)
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        return final_results
    else:
        return pd.DataFrame()


def main(batch_size=500, max_workers=30, start_batch=1, end_batch=None, thresholds=None, 
         send_email=True, signal_type=SignalType.ALL, period_type=PeriodType.ALL, 
         with_heima=False, min_turnover=100):
    """主函数"""
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
    
    # 打印配置
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
    print(f"最小成交额: {min_turnover}万元")
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
    
    start_time = time.time()
    
    os.makedirs("stock_cache", exist_ok=True)
    
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
        signal_type=signal_type,
        period_type=period_type,
        with_heima=with_heima,
        min_turnover=min_turnover
    )
    
    if not results.empty:
        print(f"\n发现信号的股票 ({signal_desc}{heima_desc}, {period_desc})：")
        print("=" * 160)
        
        # 动态构建表头
        header_parts = [f"{'代码':<12}", f"{'公司名称':<15}", f"{'价格':>8}", f"{'成交额(万)':>12}"]
        
        if signal_type in [SignalType.ALL, SignalType.BLUE]:
            if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                header_parts.extend([f"{'日BLUE':>8}", f"{'日BLUE天':>4}"])
            if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                header_parts.extend([f"{'周BLUE':>8}", f"{'周BLUE周':>4}"])
        
        if signal_type in [SignalType.ALL, SignalType.RED]:
            if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                header_parts.extend([f"{'日LIRED':>8}", f"{'日LIRED天':>4}"])
            if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                header_parts.extend([f"{'周LIRED':>8}", f"{'周LIRED周':>4}"])
        
        if with_heima:
            header_parts.extend([f"{'日黑马':>4}", f"{'周黑马':>4}"])
        
        header_parts.append(f"{'信号':<30}")
        print(" | ".join(header_parts))
        print("-" * 160)
        
        signal_counts = {}
        
        for _, row in results.iterrows():
            signals = []
            row_parts = [
                f"{row['symbol']:<12}",
                f"{row['name']:<15}",
                f"{row['price']:8.2f}",
                f"{row['turnover']:12.2f}"
            ]
            
            if signal_type in [SignalType.ALL, SignalType.BLUE]:
                if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                    row_parts.extend([f"{row['blue_daily']:8.2f}", f"{row['blue_days']:4d}"])
                    if row['has_day_blue']:
                        signals.append(f'日BLUE({row["blue_days"]}天)')
                        signal_counts['日BLUE'] = signal_counts.get('日BLUE', 0) + 1
                if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                    row_parts.extend([f"{row['blue_weekly']:8.2f}", f"{row['blue_weeks']:4d}"])
                    if row['has_week_blue']:
                        signals.append(f'周BLUE({row["blue_weeks"]}周)')
                        signal_counts['周BLUE'] = signal_counts.get('周BLUE', 0) + 1
            
            if signal_type in [SignalType.ALL, SignalType.RED]:
                if period_type in [PeriodType.ALL, PeriodType.DAILY]:
                    row_parts.extend([f"{row['lired_daily']:8.2f}", f"{row['lired_days']:4d}"])
                    if row['has_day_lired']:
                        signals.append(f'日LIRED({row["lired_days"]}天)')
                        signal_counts['日LIRED'] = signal_counts.get('日LIRED', 0) + 1
                if period_type in [PeriodType.ALL, PeriodType.WEEKLY]:
                    row_parts.extend([f"{row['lired_weekly']:8.2f}", f"{row['lired_weeks']:4d}"])
                    if row['has_week_lired']:
                        signals.append(f'周LIRED({row["lired_weeks"]}周)')
                        signal_counts['周LIRED'] = signal_counts.get('周LIRED', 0) + 1
            
            if with_heima:
                row_parts.extend([
                    f"{row['day_heima_count']:4d}",
                    f"{row['week_heima_count']:4d}"
                ])
                if row['has_day_heima']:
                    signals.append('日黑马')
                    signal_counts['日黑马'] = signal_counts.get('日黑马', 0) + 1
                if row['has_week_heima']:
                    signals.append('周黑马')
                    signal_counts['周黑马'] = signal_counts.get('周黑马', 0) + 1
            
            signals_str = ', '.join(signals)
            row_parts.append(f"{signals_str:<30}")
            print(" | ".join(row_parts))
        
        print("=" * 160)
        print(f"\n信号统计总结:")
        print("-" * 40)
        for signal, count in signal_counts.items():
            if count > 0:
                print(f"{signal:<15}: {count:>5} 只")
        print("-" * 40)
        print(f"共发现 {len(results)} 只股票有信号")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signal_suffix = signal_type if signal_type != SignalType.ALL else "all"
        period_suffix = period_type if period_type != PeriodType.ALL else "all"
        heima_suffix = "_heima" if with_heima else ""
        filename = f'cn_signals_{signal_suffix}_{period_suffix}{heima_suffix}_{timestamp}.csv'
        results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
        
        # 发送邮件
        if send_email:
            try:
                print("\n准备发送信号总结邮件...")
                SignalNotifier.send_summary_email(results, signal_counts, signal_type, period_type, with_heima)
            except Exception as e:
                print(f"发送信号总结邮件失败: {e}")
    else:
        print("\n未发现任何信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='A股股票信号扫描器 - 支持信号类型、周期过滤和黑马信号',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
    # 只看BLUE信号（多头）
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue
    
    # 只看LIRED信号（空头）
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal red
    
    # 只看日线BLUE信号
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --period daily
    
    # 只看周线BLUE信号
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --period weekly
    
    # 只看BLUE+黑马信号同时出现的股票
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --with-heima
    
    # 只看日线BLUE+黑马信号同时出现
    python scan_cn_signals_multi_thread_tushare_filtered.py --signal blue --period daily --with-heima
    
    # 指定批次范围
    python scan_cn_signals_multi_thread_tushare_filtered.py --start-batch 1 --end-batch 3
    
    # 不发送邮件
    python scan_cn_signals_multi_thread_tushare_filtered.py --no-email
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
        '--with-heima',
        action='store_true',
        help='要求同时出现黑马信号（★黑马信号或★掘底买点）'
    )
    
    parser.add_argument(
        '--start-batch',
        type=int,
        default=1,
        help='起始批次 (默认: 1)'
    )
    
    parser.add_argument(
        '--end-batch',
        type=int,
        default=None,
        help='结束批次 (默认: 全部)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='每批次股票数量 (默认: 500)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=30,
        help='并发线程数 (默认: 30)'
    )
    
    parser.add_argument(
        '--min-turnover',
        type=int,
        default=100,
        help='最小成交额(万元) (默认: 100)'
    )
    
    parser.add_argument(
        '--no-email',
        action='store_true',
        help='不发送邮件通知'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(
        batch_size=args.batch_size,
        max_workers=args.workers,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
        send_email=not args.no_email,
        signal_type=args.signal,
        period_type=args.period,
        with_heima=args.with_heima,
        min_turnover=args.min_turnover
    )

