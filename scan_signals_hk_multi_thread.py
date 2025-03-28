import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import concurrent.futures
import threading
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tushare as ts
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局变量存储公司信息
COMPANY_INFO = {}

# 设置 Tushare Token
TS_TOKEN = 'gx03013e909f633ecb66722df66b360f070426613316ebf06ecd3482'
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 定义富途函数（与A股一致）
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
    def __init__(self, symbol, result_data):
        self.symbol = symbol
        self.data = result_data
        self.sender_email = "stockprofile138@gmail.com"
        self.receiver_emails = ["stockprofile138@gmail.com"]
        self.email_password = "vselpmwrjacmgdib"
    
    def send_signal_email(self):
        subject = f"港股信号通知: {self.symbol} 出现交易信号"
        body = f"股票代码: {self.symbol}\n"
        if 'company_name' in self.data:
            body += f"公司名称: {self.data['company_name']}\n"
        body += f"当前价格: {self.data['price']:.2f}\n"
        body += f"成交额: {self.data['turnover']:.2f}\n\n"
        body += "信号详情:\n"
        if self.data.get('has_day_blue', False):
            body += f"⭐ 日线BLUE满足条件: {self.data['blue_daily']:.2f}, 最近信号值: {self.data['latest_day_blue_value']:.2f}, 天数: {self.data['blue_days']}\n"
        if self.data.get('has_week_blue', False):
            body += f"⭐ 周线BLUE满足条件: {self.data['blue_weekly']:.2f}, 最近信号值: {self.data['latest_week_blue_value']:.2f}, 周数: {self.data['blue_weeks']}\n"
        if self.data.get('has_day_lired', False):
            body += f"⭐ 日线LIRED满足条件: {self.data['lired_daily']:.2f}, 最近信号值: {self.data['latest_day_lired_value']:.2f}, 天数: {self.data['lired_days']}\n"
        if self.data.get('has_week_lired', False):
            body += f"⭐ 周线LIRED满足条件: {self.data['lired_weekly']:.2f}, 最近信号值: {self.data['latest_week_lired_value']:.2f}, 周数: {self.data['lired_weeks']}\n"
        
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
        subject = f"港股日或周信号扫描总结 ({current_time})"
        
        body = f"港股日或周信号扫描总结 ({current_time})\n\n"
        
        body += "信号统计总结:\n"
        body += "-" * 40 + "\n"
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"{signal:<15}: {count:>5} 只\n"
        body += "-" * 40 + "\n"
        body += f"共发现 {len(results)} 只股票有日或周信号\n\n"
        
        body += "股票列表：\n"
        body += "=" * 100 + "\n"
        body += f"{'代码':<8} | {'公司名称':<20} | {'价格':>8} | {'成交额':>12} | {'信号':<40}\n"
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
            company_name = row.get('company_name', 'N/A')
            if len(company_name) > 20:
                company_name = company_name[:17] + "..."
            body += f"{row['symbol']:<8} | {company_name:<20} | {row['price']:8.2f} | {row['turnover']:12.2f} | {signals_str:<40}\n"
        
        body += "=" * 100 + "\n"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(receiver_emails)  # 修复：使用 receiver_emails 而非 self.receiver_emails
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            print(f"正在尝试发送港股日或周信号总结邮件...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"邮件发送成功，包含 {len(results)} 只股票")
            return True
        except Exception as e:
            print(f"邮件发送失败: {e}")
            return False

def get_hk_tradecal(start_date, end_date):
    """获取港股交易日历"""
    try:
        df = pro.hk_tradecal(start_date=start_date, end_date=end_date)
        trade_dates = df[df['is_open'] == 1]['cal_date'].tolist()
        print(f"获取到 {len(trade_dates)} 个交易日，从 {start_date} 到 {end_date}")
        return trade_dates
    except Exception as e:
        print(f"获取交易日历失败: {e}")
        return []

def get_hk_stocks():
    """使用 Tushare 获取港股列表"""
    cache_file = 'hk_stocks_cache.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, cache_file)
    
    if os.path.exists(cache_path):
        cache_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        today = datetime.now().date()
        cache_date = cache_mod_time.date()
        if cache_date == today:
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    tickers_df = pd.read_json(f, orient='records')
                print(f"从缓存加载 {len(tickers_df)} 只港股")
                print(f"缓存数据列名: {tickers_df.columns.tolist()}")
                if 'ts_code' not in tickers_df.columns:
                    print("缓存数据缺少 'ts_code' 列，将重新获取")
                    raise ValueError("缓存数据格式错误")
                return tickers_df
            except Exception as e:
                print(f"读取缓存失败或数据格式错误: {e}，重新获取港股列表")
    
    print("从 Tushare 获取港股列表...")
    try:
        tickers_df = pro.hk_basic(list_status='L')
        print(f"Tushare 返回数据列名: {tickers_df.columns.tolist()}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            tickers_df.to_json(f, orient='records', force_ascii=False)
        print(f"成功获取到 {len(tickers_df)} 只港股")
        return tickers_df
    except Exception as e:
        print(f"Tushare 接口调用失败: {e}")
        print("请确保账户积分达到 2000 以上以访问 hk_basic 接口，详情见: https://tushare.pro/document/1?doc_id=108")
        return None

def get_company_info():
    """从 Tushare 获取港股公司信息"""
    cache_file = 'hk_company_info_cache.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, cache_file)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                company_dict = json.load(f)
            print(f"从缓存加载港股公司信息: {len(company_dict)} 家公司")
            return company_dict
        except Exception as e:
            print(f"读取公司信息缓存失败: {e}，重新获取公司信息")
    
    print("从 Tushare 获取港股公司信息...")
    tickers_df = get_hk_stocks()
    if tickers_df is None or tickers_df.empty:
        print("无法获取港股列表，公司信息初始化失败")
        return {}
    
    try:
        company_dict = dict(zip(tickers_df['ts_code'].str.replace('.HK', ''), tickers_df['name']))
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(company_dict, f, ensure_ascii=False, indent=2)
        print(f"共获取到 {len(company_dict)} 家港股公司信息")
        return company_dict
    except Exception as e:
        print(f"生成公司信息字典失败: {e}")
        return {}

def init_company_info():
    """初始化公司信息"""
    global COMPANY_INFO
    print("\n正在初始化公司信息...")
    COMPANY_INFO = get_company_info()
    if COMPANY_INFO is None:
        COMPANY_INFO = {}
        print("公司信息初始化失败，返回空字典")
    print(f"初始化完成，共加载 {len(COMPANY_INFO)} 家公司信息")

def process_single_stock(symbol, trade_dates, thresholds=None, max_retries=3):
    """处理单个股票，添加重试机制以应对频率限制"""
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
    
    print(f"开始处理股票: {symbol}")
    for attempt in range(max_retries):
        try:
            if not symbol.endswith('.HK'):
                symbol = f"{symbol}.HK"
            
            # 使用 hk_daily 接口获取一年数据
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            end_date = datetime.now().strftime('%Y%m%d')
            data_daily = pro.hk_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            
            if data_daily is None or data_daily.empty or len(data_daily) < 20:
                print(f"{symbol} 数据不足或为空")
                return None
            
            data_daily = data_daily.rename(columns={
                'trade_date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'vol': 'Volume'
            }).set_index('Date')
            data_daily.index = pd.to_datetime(data_daily.index)
            
            print(f"{symbol} 日线数据长度: {len(data_daily)}")
            
            if len(data_daily) >= 5:
                recent_data = data_daily.iloc[-5:]
                min_price = recent_data['Low'].min()
                current_price = data_daily['Close'].iloc[-1]
                recent_change = (current_price / min_price - 1) * 100
                if recent_change > 5:
                    print(f"{symbol} 最近5天涨幅超过5%，跳过")
                    return None
            
            data_weekly = data_daily.resample('W-MON').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'mean'
            }).dropna()
            
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
            
            has_day_blue = day_blue_count >= default_thresholds['day_blue_count']
            has_week_blue = week_blue_count >= default_thresholds['week_blue_count']
            has_day_lired = day_lired_count >= default_thresholds['day_lired_count']
            has_week_lired = week_lired_count >= default_thresholds['week_lired_count']
            
            print(f"{symbol} 信号检查: 日BLUE={day_blue_count}, 周BLUE={week_blue_count}, 日LIRED={day_lired_count}, 周LIRED={week_lired_count}")
            
            if has_day_blue or has_week_blue or has_day_lired or has_week_lired:
                display_symbol = symbol.replace('.HK', '')
                result = {
                    'symbol': display_symbol,
                    'price': latest_daily['Close'],
                    'Volume': latest_daily['Volume'],
                    'turnover': latest_daily['Volume'] * latest_daily['Close'],
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
                    'has_week_lired': has_week_lired
                }
                print(f"{symbol} 发现日或周信号")
                return result
            else:
                print(f"{symbol} 未发现日或周信号")
            return None
        
        except Exception as e:
            error_msg = str(e)
            if "每分钟最多访问该接口400次" in error_msg:
                wait_time = 60 / (400 / max_workers) + 1  # 计算等待时间，确保不超过400次/分钟
                print(f"{symbol} 触发频率限制，第 {attempt + 1}/{max_retries} 次尝试，等待 {wait_time:.2f} 秒后重试")
                time.sleep(wait_time)
                continue
            else:
                with print_lock:
                    print(f"{symbol} 处理出错: {e}")
                return None
    print(f"{symbol} 达到最大重试次数，放弃处理")
    return None

def scan_signals_parallel(max_workers=10, batch_size=50, tickers=None, cooldown=10, thresholds=None):
    """并行扫描股票信号，优化频率控制"""
    print("正在获取港股列表...")
    if tickers is None:
        tickers_df = get_hk_stocks()
        if tickers_df is None:
            print("无法获取港股列表，扫描终止")
            return pd.DataFrame()
        print(f"tickers_df 列名: {tickers_df.columns.tolist()}")
        if 'ts_code' not in tickers_df.columns:
            raise KeyError("tickers_df 中缺少 'ts_code' 列，请检查数据源或缓存文件")
        tickers = tickers_df['ts_code'].tolist()
    print(f"共获取到 {len(tickers)} 只港股")
    
    # 获取交易日历
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')
    trade_dates = get_hk_tradecal(start_date, end_date)
    if not trade_dates:
        print("无法获取交易日历，使用默认日期范围")
        trade_dates = []
    
    all_results = []
    batch_count = (len(tickers) + batch_size - 1) // batch_size
    
    with tqdm(total=batch_count, desc="批次进度") as batch_pbar:
        for i in range(0, len(tickers), batch_size):
            batch_start_time = time.time()
            batch_tickers = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"\n处理第 {batch_num}/{batch_count} 批 ({len(batch_tickers)} 只股票)")
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_stock, symbol, trade_dates, thresholds): symbol for symbol in batch_tickers}
                for future in tqdm(as_completed(futures), total=len(batch_tickers), desc="股票处理"):
                    result = future.result()
                    if result is not None:
                        with results_lock:
                            batch_results.append(result)
            
            if batch_results:
                all_results.extend(batch_results)
                print(f"批次 {batch_num} 发现 {len(batch_results)} 只有信号的股票")
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            print(f"批次 {batch_num} 处理耗时: {batch_time:.2f} 秒")
            
            batch_pbar.update(1)
            
            if i + batch_size < len(tickers):
                print(f"休息 {cooldown} 秒后处理下一批...")
                time.sleep(cooldown)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def main(batch_size=50, max_workers=10, thresholds=None, send_email=True):
    """主函数"""
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
    
    logging.info("\n当前信号阈值配置:")
    logging.info(f"日线BLUE阈值: {default_thresholds['day_blue']}, 所需天数: {default_thresholds['day_blue_count']}")
    logging.info(f"日线LIRED阈值: {default_thresholds['day_lired']}, 所需天数: {default_thresholds['day_lired_count']}")
    logging.info(f"周线BLUE阈值: {default_thresholds['week_blue']}, 所需周数: {default_thresholds['week_blue_count']}")
    logging.info(f"周线LIRED阈值: {default_thresholds['week_lired']}, 所需周数: {default_thresholds['week_lired_count']}")
    
    init_company_info()
    start_time = time.time()
    print("\n开始扫描港股...")
    
    # 扫描所有港股
    results = scan_signals_parallel(max_workers=max_workers, batch_size=batch_size, cooldown=10, thresholds=default_thresholds)
    
    if not results.empty:
        results['company_name'] = results['symbol'].map(lambda x: COMPANY_INFO.get(x, 'N/A'))
        
        print("\n发现信号的股票（日或周信号）：")
        print("=" * 160)
        print(f"{'代码':<8} | {'公司名称':<20} | {'价格':>8} | {'成交额':>12} | "
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
            print(f"{row['symbol']:<8} | {row['company_name']:<20} | {row['price']:8.2f} | {row['turnover']:12.2f} | "
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
        print(f"共发现 {len(results)} 只股票有日或周信号")
        
        if send_email:
            try:
                logging.info("准备发送信号总结邮件...")
                SignalNotifier.send_summary_email(results, signal_counts)
            except Exception as e:
                logging.error(f"发送信号总结邮件失败: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'hk_signals_{timestamp}.csv'
        results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
    else:
        print("\n未发现任何日或周信号")
    
    end_time = time.time()
    print(f"\n扫描完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()