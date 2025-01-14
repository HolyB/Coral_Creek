import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.label_generation import LabelGenerator
from Stock_utils.notification import SignalNotifier
from stock_scanner import StockScanner

class StockScannerMT:
    def __init__(self):
        """初始化多线程股票扫描器"""
        warnings.filterwarnings('ignore')
        
        # 使用 StockScanner 的方法获取 SP500 列表
        scanner = StockScanner()
        self.sp500_tickers = scanner.get_sp500_tickers()
        self.tickers = self.sp500_tickers
        
        # 创建线程锁
        self.print_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.detected_stocks = []

    def process_stock(self, ticker):
        """处理单个股票"""
        try:
            # 获取数据
            fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
            data = fetcher.get_stock_data()
            
            if data is None or data.empty or len(data) <= 70:
                return None
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 计算指标
            analysis = StockAnalysis(data)
            result_df = analysis.calculate_all_strategies()
            
            # 生成标签
            label_generator = LabelGenerator(result_df)
            df_with_labels = label_generator.generate_labels()
            
            # 检查信号
            notifier = SignalNotifier(result_df, ticker)
            notifier.calculate_label_statistics()
            
            if notifier.check_conditions():
                stock_info = {
                    'ticker': ticker,
                    'occurred_signals': notifier.occurred_signals,
                    'tandi_price': notifier.tandi_price,
                    'current_price': notifier.current_price,
                    'price_increase': notifier.price_increase,
                }
                
                with self.results_lock:
                    self.detected_stocks.append(stock_info)
                with self.print_lock:
                    print(f"\n发现信号 - {ticker}")
            
            return True
            
        except Exception as e:
            with self.print_lock:
                print(f"{ticker} 处理失败: {e}")
            return None

    def send_summary_email(self, detected_stocks):
        """发送汇总邮件"""
        if not detected_stocks:
            print("No stocks met the conditions. No summary email will be sent.")
            return

        # 保存到CSV文件
        csv_filename = f"us_stock_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df = pd.DataFrame(detected_stocks)
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {csv_filename}")

        # 构建邮件内容
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com", "Lindaxiang7370@gmail.com"]
        subject = f"US Stock Signals Summary ({len(detected_stocks)} stocks)"
        
        body = "以下是满足条件的股票和相关信号信息：\n\n"
        for stock in detected_stocks:
            body += (f"股票代码：{stock['ticker']}\n"
                    f"满足条件的信号：{', '.join(stock['occurred_signals'])}\n"
                    f"探底点价格：{stock['tandi_price']}\n"
                    f"当前价格：{stock['current_price']}\n"
                    f"当前价格比探底点价格上涨了：{stock['price_increase']:.2%}\n"
                    f"{'-' * 40}\n")

        # 发送邮件
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

    def scan_stocks(self, max_workers=10):
        """使用线程池扫描股票"""
        start_time = datetime.now()
        print(f"\n开始扫描 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建进度条
        pbar = tqdm(total=len(self.tickers), desc="扫描进度")
        
        # 使用线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_ticker = {
                executor.submit(self.process_stock, ticker): ticker 
                for ticker in self.tickers
            }
            
            # 处理完成的任务
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
        
        # 打印结果
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n扫描完成!")
        print(f"耗时: {duration}")
        print(f"发现信号数量: {len(self.detected_stocks)}")
        
        if self.detected_stocks:
            print("\n发现的信号:")
            print("=" * 120)
            print(f"{'代码':<8} | {'信号类型':<20} | {'探底价格':>10} | {'当前价格':>10} | {'涨幅':>8}")
            print("-" * 120)
            
            for stock in self.detected_stocks:
                print(f"{stock['ticker']:<8} | "
                      f"{', '.join(stock['occurred_signals']):<20} | "
                      f"{stock['tandi_price']:10.2f} | "
                      f"{stock['current_price']:10.2f} | "
                      f"{stock['price_increase']:8.2%}")
            
            print("=" * 120)
        
        # 在扫描完成后发送邮件
        if self.detected_stocks:
            self.send_summary_email(self.detected_stocks)
        
        return self.detected_stocks

def main():
    """主函数"""
    scanner = StockScannerMT()
    scanner.scan_stocks(max_workers=20)  # 可以调整线程数

if __name__ == "__main__":
    main() 