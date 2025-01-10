import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import schedule
import time
from datetime import datetime, timedelta
import pytz
import subprocess
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 邮件配置
SENDER_EMAIL = "stockprofile138@gmail.com"
EMAIL_PASSWORD = "vselpmwrjacmgdib"
RECEIVER_EMAILS = ["stockprofile138@gmail.com"]#, "xhemobile@gmail.com", "Lindaxiang7370@gmail.com"]

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('us_stock_scan.log'),
        logging.StreamHandler()
    ]
)

def send_email(subject, body):
    """发送邮件"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        msg['Bcc'] = ", ".join(RECEIVER_EMAILS)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        server.quit()
        logging.info("邮件发送成功")
    except Exception as e:
        logging.error(f"邮件发送失败: {e}")

def is_us_market_holiday():
    """检查是否为美股假期"""
    holidays_2024 = [
        '2024-01-01',  # 元旦
        '2024-01-15',  # 马丁·路德·金纪念日
        '2024-02-19',  # 总统日
        '2024-03-29',  # 耶稣受难日
        '2024-05-27',  # 阵亡将士纪念日
        '2024-06-19',  # 六月节
        '2024-07-04',  # 独立日
        '2024-09-02',  # 劳动节
        '2024-11-28',  # 感恩节
        '2024-12-25',  # 圣诞节
    ]
    
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    return today in holidays_2024

def run_scan():
    """运行扫描脚本"""
    try:
        now = datetime.now(pytz.timezone('US/Eastern'))
        scan_type = "盘前" if now.hour < 12 else "盘后"
        
        logging.info(f"开始运行{scan_type}股票扫描...")
        
        # 使用当前 Python 解释器路径
        python_executable = sys.executable
        
        # 获取脚本的完整路径
        script_path = os.path.join(current_dir, 'scan_signals_multi_thread_v2.py')
        
        # 运行扫描脚本并捕获输出
        process = subprocess.Popen(
            [python_executable, script_path],  # 使用完整路径
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy()  # 复制当前环境变量
        )
        
        # 实时获取输出
        stdout, stderr = process.communicate()
        
        # 检查返回码
        if process.returncode != 0:
            raise Exception(f"扫描脚本返回错误码 {process.returncode}\n错误信息:\n{stderr}")
        
        # 发送邮件通知
        subject = f"美股{scan_type}扫描完成 - {now.strftime('%Y-%m-%d %H:%M:%S')} ET"
        body = f"""
美股{scan_type}扫描已完成

时间: {now.strftime('%Y-%m-%d %H:%M:%S')} ET

扫描日志:
{stdout}

如有错误:
{stderr}

详细结果请查看邮件通知或日志文件。
"""
        send_email(subject, body)
        logging.info(f"{scan_type}扫描完成")
        
    except Exception as e:
        error_msg = f"{scan_type}扫描出错: {str(e)}"
        logging.error(error_msg)
        # 发送错误通知
        send_email(
            f"美股{scan_type}扫描出错 - {now.strftime('%Y-%m-%d %H:%M:%S')} ET",
            f"""
扫描过程中出现错误：

错误信息：
{str(e)}

请检查日志文件获取更多信息。
"""
        )

def should_run():
    """检查是否应该运行"""
    now = datetime.now(pytz.timezone('US/Eastern'))
    
    # 检查是否为周末
    if now.weekday() in [5, 6]:
        logging.info("今天是周末，不运行扫描")
        return False
    
    # 检查是否为假期
    if is_us_market_holiday():
        logging.info("今天是美股假期，不运行扫描")
        return False
    
    return True

def pre_market_scan():
    """盘前扫描"""
    if should_run():
        logging.info("执行盘前扫描")
        run_scan()

def post_market_scan():
    """盘后扫描"""
    if should_run():
        logging.info("执行盘后扫描")
        run_scan()

def main():
    """主函数"""
    logging.info("启动定时任务...")
    
    # 设置盘前扫描（美东时间09:00，比开盘早30分钟）
    schedule.every().day.at("09:00").do(pre_market_scan)
    
    # 设置盘后扫描（美东时间16:00，收盘后）
    schedule.every().day.at("16:00").do(post_market_scan)
    
    # 发送启动通知
    start_msg = """
美股扫描定时任务已启动

扫描时间:
- 盘前扫描: 美东时间 09:00 (开盘前30分钟)
- 盘后扫描: 美东时间 16:00 (收盘后)

扫描结果将通过邮件发送。
"""
    send_email("美股扫描定时任务已启动", start_msg)
    
    logging.info("定时任务已设置")
    logging.info("- 盘前扫描时间: 美东时间 09:00 (开盘前30分钟)")
    logging.info("- 盘后扫描时间: 美东时间 16:00 (收盘后)")
    
    # 立即执行一次扫描
    now = datetime.now(pytz.timezone('US/Eastern'))
    logging.info("执行启动时扫描...")
    run_scan()
    
    # 运行定时任务
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
        except Exception as e:
            error_msg = f"定时任务运行出错: {str(e)}"
            logging.error(error_msg)
            send_email("美股扫描定时任务出错", error_msg)
            time.sleep(300)  # 出错后等待5分钟再继续

if __name__ == "__main__":
    main() 