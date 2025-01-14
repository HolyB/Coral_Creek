import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm
import threading
import schedule
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import pytz

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.label_generation import LabelGenerator
from Stock_utils.notification import SignalNotifier
from stock_scanner import StockScanner
from Stock_utils.MyTT import MA, REF, MACD
from scan_signals_multi_thread_v2 import get_sp500_tickers

def process_stock_v2(ticker):
    """第二种信号检测逻辑"""
    try:
        # 获取数据
        fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
        df = fetcher.get_stock_data()
        
        if df is None or len(df) < 30:
            return None
            
        CLOSE = df['Close'].values
        HIGH = df['High'].values
        LOW = df['Low'].values
        
        # 计算涨停条件（近5天内有涨停）
        ZT = CLOSE > REF(CLOSE, 1) * 1.093  # 涨停判断
        recent_zt = ZT[-5:]  # 最近5天的涨停情况
        has_recent_zt = np.any(recent_zt)  # 是否有涨停
        
        # 计算均线
        MA5 = MA(CLOSE, 5)
        MA10 = MA(CLOSE, 10)
        
        # 判断5日均线是否向上
        ma5_trend = MA5[-1] > MA5[-2]
        
        # 计算MACD
        DIF, DEA, macd_hist = MACD(CLOSE)
        
        # 计算DIF角度
        dif_angle = np.arctan((DIF[-1] - DIF[-2])) * 180 / np.pi
        
        # 判断MACD金叉
        macd_golden_cross = (DIF[-1] > DEA[-1]) and (DIF[-2] <= DEA[-2])
        
        # 判断是否在零轴上方
        above_zero = DIF[-1] > 0
        
        # 获取当前价格和均线值
        price = CLOSE[-1]
        ma5_value = MA5[-1]
        ma10_value = MA10[-1]
        
        # 判断是否回踩到均线附近（允许3%的误差）
        near_ma5 = abs(price - ma5_value) / ma5_value < 0.03
        near_ma10 = abs(price - ma10_value) / ma10_value < 0.03
        
        # 检查条件
        conditions_met = (
            has_recent_zt and 
            ma5_trend and 
            macd_golden_cross and 
            above_zero and 
            (near_ma5 or near_ma10)
        )
        
        if conditions_met:
            return {
                'has_recent_zt': has_recent_zt,
                'ma5_trend': ma5_trend,
                'dif_angle': dif_angle,
                'macd_golden_cross': macd_golden_cross,
                'above_zero': above_zero,
                'near_ma5': near_ma5,
                'near_ma10': near_ma10,
                'price': price,
                'last_zt_days': np.where(recent_zt)[0][-1] + 1 if np.any(recent_zt) else None
            }
            
        return None
        
    except Exception as e:
        print(f"{ticker} V2信号处理失败: {e}")
        return None

class CombinedStockScanner:
    def __init__(self):
        """初始化多线程股票扫描器"""
        warnings.filterwarnings('ignore')
        
        # 创建线程锁
        self.print_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.detected_stocks = []
        
        # 使用 scan_signals_multi_thread_v2 中的函数获取 SP500 列表
        self.sp500_tickers = get_sp500_tickers()
        print(f"加载了 {len(self.sp500_tickers)} 只SP500股票")
        self.tickers = self.sp500_tickers

    def process_stock_combined(self, ticker):
        """处理单个股票（两种信号）"""
        try:
            # 获取数据
            fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
            data = fetcher.get_stock_data()
            
            if data is None or data.empty or len(data) <= 70:
                return None
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 首先检查第二种信号（scan_signals_multi_thread_v2）
            v2_result = process_stock_v2(ticker)
            if v2_result:
                # 如果有v2信号，再检查第一种信号
                analysis = StockAnalysis(data)
                result_df = analysis.calculate_all_strategies()
                label_generator = LabelGenerator(result_df)
                df_with_labels = label_generator.generate_labels()
                notifier = SignalNotifier(result_df, ticker)
                notifier.calculate_label_statistics()
                
                # 合并两种信号的结果
                combined_info = {
                    'ticker': ticker,
                    'has_v2_signal': True,
                    'price': v2_result['price'],
                    'last_zt_days': v2_result['last_zt_days'],
                    'ma5_trend': v2_result['ma5_trend'],
                    'dif_angle': v2_result['dif_angle'],
                    'near_ma5': v2_result['near_ma5'],
                    'near_ma10': v2_result['near_ma10']
                }
                
                # 如果也有第一种信号，添加相关信息
                if notifier.check_conditions():
                    combined_info.update({
                        'has_type1_signal': True,
                        'type1_signals': notifier.occurred_signals,
                        'tandi_price': notifier.tandi_price,
                        'price_increase': notifier.price_increase
                    })
                else:
                    combined_info.update({
                        'has_type1_signal': False,
                        'type1_signals': [],
                        'tandi_price': None,
                        'price_increase': None
                    })
                
                with self.results_lock:
                    self.detected_stocks.append(combined_info)
                    
                with self.print_lock:
                    print(f"\n发现信号 - {ticker}")
            
            return True
            
        except Exception as e:
            with self.print_lock:
                print(f"{ticker} 处理失败: {e}")
            return None

    def save_and_send_results(self):
        """保存结果到CSV并发送邮件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        if not self.detected_stocks:
            print("No signals detected.")
            return
            
        # 保存到CSV
        csv_filename = f"us_stock_signals_combined_{timestamp}.csv"
        df = pd.DataFrame(self.detected_stocks)
        
        # 重新排列列的顺序
        columns_order = [
            'ticker', 
            'price',
            'last_zt_days',
            'ma5_trend',
            'dif_angle',
            'near_ma5',
            'near_ma10',
            'has_type1_signal',
            'type1_signals',
            'tandi_price',
            'price_increase'
        ]
        
        df = df[columns_order]
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {csv_filename}")
        
        # 发送邮件
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com", "Lindaxiang7370@gmail.com"]
        subject = f"US Stock Signals Combined ({len(self.detected_stocks)} stocks)"
        
        body = f"发现 {len(self.detected_stocks)} 个满足条件的股票。\n\n"
        body += "详细信息请查看附件。\n"
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            msg['Bcc'] = ", ".join(receiver_emails)
            
            # 添加CSV附件
            with open(csv_filename, 'rb') as f:
                attachment = MIMEText(f.read(), 'base64', 'utf-8')
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=csv_filename)
                msg.attach(attachment)
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, "vselpmwrjacmgdib")
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print("Summary email sent successfully.")
        except Exception as e:
            print(f"Failed to send summary email: {e}")

    def scan_stocks(self, max_workers=20):
        """使用线程池扫描股票"""
        start_time = datetime.now()
        print(f"\n开始扫描 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 清空之前的结果
        self.detected_stocks = []
        
        # 创建进度条
        pbar = tqdm(total=len(self.tickers), desc="扫描进度")
        
        # 使用线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.process_stock_combined, ticker): ticker 
                for ticker in self.tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    future.result()
                except Exception as e:
                    with self.print_lock:
                        print(f"{ticker} 处理出错: {e}")
                finally:
                    pbar.update(1)
        
        pbar.close()
        
        # 保存结果并发送邮件
        self.save_and_send_results()
        
        # 打印统计信息
        print(f"\n扫描完成!")
        print(f"耗时: {datetime.now() - start_time}")
        print(f"发现信号数量: {len(self.detected_stocks)}")

def run_scan():
    """运行扫描"""
    scanner = CombinedStockScanner()
    scanner.scan_stocks()

def schedule_scans():
    """设置定时任务"""
    et = pytz.timezone('US/Eastern')
    
    # 盘前一小时（8:30 ET）
    schedule.every().day.at("08:30").do(run_scan)
    # 开盘（9:30 ET）
    schedule.every().day.at("09:30").do(run_scan)
    # 收盘（16:00 ET）
    schedule.every().day.at("16:00").do(run_scan)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # 直接运行一次
    run_scan()
    # 启动定时任务
    schedule_scans() 