import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np 
import os
# 在文件开头添加
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')  # 过滤所有警告
def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    过滤出指定日期范围的数据。

    :param df: 包含股票历史数据的 DataFrame
    :param start_date: 起始日期 (格式为 'YYYY-MM-DD')
    :param end_date: 结束日期 (格式为 'YYYY-MM-DD')
    :return: 指定日期范围内的数据的 DataFrame
    """
    return df.loc[(df.index >= start_date) & (df.index <= end_date)]


import yaml

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 修改配置文件路径
config_path = os.path.join(current_dir, 'Stock_utils', 'feature_config.yaml')

# 加载特征配置文件
with open(config_path, 'r', encoding='utf-8') as file:
    feature_config = yaml.safe_load(file)

categorical_features = feature_config['categorical_features']
numerical_features = feature_config['numerical_features']

# 查看加载的特征
print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)


import sys

# 添加 Stock_utils 目录到系统路径
stock_utils_path = os.path.join(current_dir, 'Stock_utils')
sys.path.append(stock_utils_path)

import importlib
# 移除缓存
if 'stock_analysis' in sys.modules:
    del sys.modules['stock_analysis']
if 'label_generation' in sys.modules:
    del sys.modules['label_generation']

if 'notification' in sys.modules:
    del sys.modules['notification']

if 'stock_data_fetcher' in sys.modules:
    del sys.modules['stock_data_fetcher']

if 'stock_ml_model' in sys.modules:
    del sys.modules['stock_ml_model']

if 'arima_forcast' in sys.modules:
    del sys.modules['arima_forcast']

if 'XGBoost' in sys.modules:
    del sys.modules['XGBoost']
# 重新导入模块
import stock_analysis
importlib.reload(stock_analysis)

import label_generation
importlib.reload(label_generation)

import notification
importlib.reload(notification)

import stock_data_fetcher
importlib.reload(stock_data_fetcher)

import stock_ml
importlib.reload(stock_ml)

import arima_forecast
importlib.reload(arima_forecast)

import XGBoost
importlib.reload(XGBoost)

import model_utils
importlib.reload(model_utils)

# Assuming ARIMAStockForecast is in a file named arima_forecast.py
from arima_forecast import ARIMAStockForecast


# 使用模块中的类
from stock_analysis import StockAnalysis
from label_generation import LabelGenerator
from stock_data_fetcher import StockDataFetcher
from notification import SignalNotifier
from stock_ml import StockMLModel

# 获取数据
fetcher = StockDataFetcher('CELH', source='polygon', interval='1d')
data = fetcher.get_stock_data()
print(f"Data for CELH:\n", data.head())

# 检查并展平多级列索引
if isinstance(data.columns, pd.MultiIndex):
    # 如果存在多级列索引，则将其展平
    data.columns = data.columns.get_level_values(0)
    print("Flattened columns:", data.columns)
else:
    print("Columns are already single-level:", data.columns)

print(f"Data for CELH after flattening columns:\n", data.head())

# 创建 StockAnalysis 类的实例并计算策略
analysis = StockAnalysis(data)
result_df = analysis.calculate_all_strategies()



import requests
from bs4 import BeautifulSoup
# 获取标普500的股票代码列表
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    sp500_df = pd.read_html(str(table))[0]
    sp500_tickers = sp500_df['Symbol'].tolist()
    # 一些股票代码中可能包含点号，需要替换为减号以符合Yahoo Finance的格式
    sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]
    return sp500_tickers

sp500_tickers = get_sp500_tickers()
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
def send_summary_email_for_all(detected_stocks):
    if not detected_stocks:
        print("No stocks met the conditions. No summary email will be sent.")
        return

    sender_email = "stockprofile138@gmail.com"
    receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com", "Lindaxiang7370@gmail.com"]#,"Olivialu13048@outlook.com"]  # 您的收件人列表
    subject = f"Summary of Detected Stocks ({len(detected_stocks)})"

    body = "以下是满足条件的股票和相关信号信息：\n\n"

    for stock_info in detected_stocks:
        ticker = stock_info['ticker']
        signals = stock_info['occurred_signals']
        tandi_price = stock_info['tandi_price']
        current_price = stock_info['current_price']
        price_increase = stock_info['price_increase']

        body += f"股票代码：{ticker}\n"
        body += f"满足条件的信号：{', '.join(signals)}\n"
        body += f"探底点价格：{tandi_price}\n"
        body += f"当前价格：{current_price}\n"
        body += f"当前价格比探底点价格上涨了：{price_increase:.2%}\n"
        body += "-" * 40 + "\n"

    # 设置邮件内容
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    msg['Bcc'] = ", ".join(receiver_emails)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # 使用 Gmail SMTP 服务器
        server.starttls()
        server.login(sender_email, "vselpmwrjacmgdib")  # 替换为您的邮箱密码或应用专用密码
        server.sendmail(sender_email, receiver_emails, msg.as_string())
        server.quit()
        print("Summary email sent successfully.")
    except Exception as e:
        print(f"Failed to send summary email: {e}")
import schedule
import time
from datetime import datetime
import pytz

# 使用模块中的类
from stock_analysis import StockAnalysis
from label_generation import LabelGenerator
from stock_data_fetcher import StockDataFetcher
from notification import SignalNotifier
from stock_ml import StockMLModel
import yaml

# 读取配置文件
config_path = os.path.join(current_dir, 'Stock_utils', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 从配置文件中获取tickers
config_tickers = config['tickers']

# 合并配置文件中的tickers和标普500的tickers，并去除重复项
tickers = list(set(config_tickers + sp500_tickers))
# 定义时间周期
intervals = ['1d']

# 初始化一个列表，用于收集满足条件的股票（可选）
detected_stocks = []

# 美股交易时段，时区为美东时间
timezone = pytz.timezone('US/Eastern')

def run_stock_analysis():
    """
    运行主逻辑，获取数据并发送信号通知
    """
    global detected_stocks
    detected_stocks = []
    print(f"\n开始分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_stocks = len(tickers)
    processed = 0
    signals_found = 0

    for ticker in tickers:
        processed += 1
        try:
            fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
            data = fetcher.get_stock_data()
            
            if data is None or data.empty or len(data) <= 70:
                continue

            # 检查并展平多级列索引
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # 分析数据
            analysis = StockAnalysis(data)
            result_df = analysis.calculate_all_strategies()
            
            label_generator = LabelGenerator(result_df)
            df_with_labels = label_generator.generate_labels()
            notifier = SignalNotifier(result_df, ticker)
            notifier.calculate_label_statistics()

            # 检查条件
            if notifier.check_conditions():
                signals_found += 1
                detected_stocks.append({
                    'ticker': ticker,
                    'occurred_signals': notifier.occurred_signals,
                    'tandi_price': notifier.tandi_price,
                    'current_price': notifier.current_price,
                    'price_increase': notifier.price_increase,
                })
                print(f"\n发现信号 - {ticker}")

            # 每处理50个股票打印一次进度
            if processed % 50 == 0:
                print(f"进度: {processed}/{total_stocks} - 发现信号: {signals_found}")

        except Exception as e:
            continue

    print(f"\n分析完成 - 处理: {processed}/{total_stocks} - 发现信号: {signals_found}")
    
    # 发送汇总邮件
    if detected_stocks:
        send_summary_email_for_all(detected_stocks)

# 定义任务调度
def schedule_tasks():
    # 开盘前1小时运行
    schedule.every().day.at("08:30").do(run_stock_analysis)
    # 开盘时运行
    schedule.every().day.at("09:30").do(run_stock_analysis)
    # 收盘前1小时运行
    schedule.every().day.at("15:00").do(run_stock_analysis)

# 启动调度并持续运行
def start_scheduler():
    # 立即运行一次任务
    run_stock_analysis()

    # 然后开始定时调度
    schedule_tasks()
    while True:
        schedule.run_pending()  # 检查是否有任务需要运行
        time.sleep(60)  # 每分钟检查一次

# 测试单个股票
def test_single_stock(ticker='CELH'):
    print(f"\n测试股票: {ticker}")
    
    try:
        # 获取数据
        fetcher = StockDataFetcher(ticker, source='polygon', interval='1d')
        data = fetcher.get_stock_data()
        
        if data is None or data.empty:
            print("无法获取数据")
            return
            
        print(f"获取到 {len(data)} 天的数据")
        
        # 分析数据
        analysis = StockAnalysis(data)
        result_df = analysis.calculate_all_strategies()
        
        # 生成标签
        label_generator = LabelGenerator(result_df)
        df_with_labels = label_generator.generate_labels()
        
        # 检查信号
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

# 运行测试
if __name__ == "__main__":
    test_single_stock('CELH')
