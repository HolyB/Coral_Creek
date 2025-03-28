import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import numpy as np
import time
import traceback

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

# 测试代码
if __name__ == "__main__":
    # 测试单个股票通知
    test_data = {
        'symbol': 'AAPL',
        'company_name': 'Apple Inc.',
        'price': 150.25,
        'turnover': 120.5,
        'blue_daily': 120.5,
        'blue_weekly': 140.2,
        'lired_daily': -90.3,
        'lired_weekly': -110.5,
        'blue_days': 4,
        'blue_weeks': 2,
        'lired_days': 3,
        'lired_weeks': 1,
        'latest_day_blue_value': 180.2,
        'latest_week_blue_value': 160.8,
        'latest_day_lired_value': -95.3,
        'latest_week_lired_value': -85.7,
        'has_day_week_blue': True,
        'has_day_week_lired': False
    }
    
    notifier = SignalNotifier('AAPL', test_data)
    notifier.send_signal_email()
    
    # 测试批量通知
    test_batch_data = [test_data]
    SignalNotifier.send_batch_signal_email(test_batch_data)