import pandas as pd
import numpy as np
import os
import warnings
import yaml
import sys
from datetime import datetime
import pytz
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from Stock_utils.stock_analysis import StockAnalysis
from Stock_utils.label_generation import LabelGenerator
from Stock_utils.stock_data_fetcher import StockDataFetcher
from Stock_utils.notification import SignalNotifier

class StockScanner:
    def __init__(self):
        """初始化股票扫描器"""
        warnings.filterwarnings('ignore')
        
        self.detected_stocks = []
        self.timezone = pytz.timezone('US/Eastern')
        
        # 获取当前文件所在目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 加载配置
        self._load_config()
        
        # 获取股票列表
        self.sp500_tickers = self.get_sp500_tickers()
        self.tickers = self.sp500_tickers

    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(self.current_dir, 'Stock_utils', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

    def get_sp500_tickers(self):
        """从本地CSV文件获取SP500股票列表"""
        try:
            sp500_file = os.path.join(self.current_dir, 'SP500.csv')
            sp500_df = pd.read_csv(sp500_file)
            tickers = sp500_df['Symbol'].tolist()
            print(f"成功加载 {len(tickers)} 只SP500股票")
            return tickers
        except Exception as e:
            print(f"读取SP500.csv失败: {e}")
            return []

    def send_summary_email(self, detected_stocks):
        """发送汇总邮件"""
        if not detected_stocks:
            print("No stocks met the conditions. No summary email will be sent.")
            return

        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com", "Lindaxiang7370@gmail.com"]
        subject = f"Summary of Detected Stocks ({len(detected_stocks)})"
        
        body = "以下是满足条件的股票和相关信号信息：\n\n"
        for stock_info in detected_stocks:
            body += self._format_stock_info(stock_info)

        self._send_email(sender_email, receiver_emails, subject, body)

    def _format_stock_info(self, stock_info):
        """格式化单个股票信息"""
        return (f"股票代码：{stock_info['ticker']}\n"
                f"满足条件的信号：{', '.join(stock_info['occurred_signals'])}\n"
                f"探底点价格：{stock_info['tandi_price']}\n"
                f"当前价格：{stock_info['current_price']}\n"
                f"当前价格比探底点价格上涨了：{stock_info['price_increase']:.2%}\n"
                f"{'-' * 40}\n")

    def _send_email(self, sender, receivers, subject, body):
        """发送邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            msg['Bcc'] = ", ".join(receivers)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender, "vselpmwrjacmgdib")
            server.sendmail(sender, receivers, msg.as_string())
            server.quit()
            print("Summary email sent successfully.")
        except Exception as e:
            print(f"Failed to send summary email: {e}")

    def _analyze_single_stock(self, ticker):
        """分析单个股票"""
        fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
        data = fetcher.get_stock_data()
        
        if data is None or data.empty or len(data) <= 70:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        analysis = StockAnalysis(data)
        result_df = analysis.calculate_all_strategies()
        
        label_generator = LabelGenerator(result_df)
        df_with_labels = label_generator.generate_labels()
        notifier = SignalNotifier(result_df, ticker)
        notifier.calculate_label_statistics()

        if notifier.check_conditions():
            return {
                'ticker': ticker,
                'occurred_signals': notifier.occurred_signals,
                'tandi_price': notifier.tandi_price,
                'current_price': notifier.current_price,
                'price_increase': notifier.price_increase,
            }
        return None

    def run_stock_analysis(self):
        """运行股票分析"""
        self.detected_stocks = []
        print(f"\n开始分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        total_stocks = len(self.tickers)
        processed = 0
        signals_found = 0

        for ticker in self.tickers:
            processed += 1
            try:
                stock_info = self._analyze_single_stock(ticker)
                if stock_info:
                    signals_found += 1
                    self.detected_stocks.append(stock_info)
                    print(f"\n发现信号 - {ticker}")

                if processed % 50 == 0:
                    print(f"进度: {processed}/{total_stocks} - 发现信号: {signals_found}")

            except Exception as e:
                continue

        print(f"\n分析完成 - 处理: {processed}/{total_stocks} - 发现信号: {signals_found}")
        if self.detected_stocks:
            self.send_summary_email(self.detected_stocks)

def test_single_stock(ticker='CELH'):
    """测试单个股票"""
    scanner = StockScanner()
    try:
        fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
        data = fetcher.get_stock_data()
        
        if data is None or data.empty:
            print("无法获取数据")
            return
            
        print(f"获取到 {len(data)} 天的数据")
        
        analysis = StockAnalysis(data)
        result_df = analysis.calculate_all_strategies()
        
        label_generator = LabelGenerator(result_df)
        df_with_labels = label_generator.generate_labels()
        
        notifier = SignalNotifier(result_df, ticker)
        notifier.calculate_label_statistics()
        
        if notifier.check_conditions():
            print("\n发现信号:")
            print(f"股票代码: {ticker}")
            print(f"信号类型: {notifier.occurred_signals}")
            print(f"探底价格: {notifier.tandi_price}")
            print(f"当前价格: {notifier.current_price}")
            print(f"价格增长: {notifier.price_increase:.2%}")
        else:
            print("未发现信号")
            
    except Exception as e:
        print(f"处理出错: {e}")

if __name__ == "__main__":
    test_single_stock('CELH')