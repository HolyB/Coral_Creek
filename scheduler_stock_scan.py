import schedule
import time
import subprocess
import logging
from datetime import datetime, timedelta
import os
import sys
import threading
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_scanner_scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class StockScanScheduler:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cn_script = os.path.join(self.script_dir, 'scan_cn_signals_multi_thread_tushare.py')
        self.hk_script = os.path.join(self.script_dir, 'scan_hk_signals_multi_thread_tushare.py')
        
        # 检查脚本是否存在
        if not os.path.exists(self.cn_script):
            logging.error(f"A股扫描脚本不存在: {self.cn_script}")
        if not os.path.exists(self.hk_script):
            logging.error(f"港股扫描脚本不存在: {self.hk_script}")
    
    def run_cn_scan(self, timing=""):
        """运行A股扫描"""
        try:
            logging.info(f"开始运行A股扫描 - {timing}")
            
            # 使用subprocess运行A股扫描脚本
            result = subprocess.run([
                sys.executable, self.cn_script
            ], capture_output=True, text=True, encoding='utf-8', timeout=3600)  # 1小时超时
            
            if result.returncode == 0:
                logging.info(f"A股扫描完成 - {timing}")
                logging.info(f"A股扫描输出: {result.stdout}")
            else:
                logging.error(f"A股扫描失败 - {timing}")
                logging.error(f"A股扫描错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error(f"A股扫描超时 - {timing}")
        except Exception as e:
            logging.error(f"A股扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())
    
    def run_hk_scan(self, timing=""):
        """运行港股扫描"""
        try:
            logging.info(f"开始运行港股扫描 - {timing}")
            
            # 使用subprocess运行港股扫描脚本
            result = subprocess.run([
                sys.executable, self.hk_script
            ], capture_output=True, text=True, encoding='utf-8', timeout=3600)  # 1小时超时
            
            if result.returncode == 0:
                logging.info(f"港股扫描完成 - {timing}")
                logging.info(f"港股扫描输出: {result.stdout}")
            else:
                logging.error(f"港股扫描失败 - {timing}")
                logging.error(f"港股扫描错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error(f"港股扫描超时 - {timing}")
        except Exception as e:
            logging.error(f"港股扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())
    
    def run_both_scans(self, timing=""):
        """同时运行A股和港股扫描（并行）"""
        try:
            logging.info(f"开始并行运行A股和港股扫描 - {timing}")
            
            # 创建线程并行运行
            cn_thread = threading.Thread(target=self.run_cn_scan, args=(timing,))
            hk_thread = threading.Thread(target=self.run_hk_scan, args=(timing,))
            
            # 启动线程
            cn_thread.start()
            time.sleep(10)  # 错开10秒启动，避免API冲突
            hk_thread.start()
            
            # 等待线程完成
            cn_thread.join()
            hk_thread.join()
            
            logging.info(f"A股和港股扫描全部完成 - {timing}")
            
        except Exception as e:
            logging.error(f"并行扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())
    
    def run_sequential_scans(self, timing=""):
        """顺序运行A股和港股扫描"""
        try:
            logging.info(f"开始顺序运行A股和港股扫描 - {timing}")
            
            # 先运行A股扫描
            self.run_cn_scan(timing)
            
            # 等待5分钟后运行港股扫描
            logging.info("A股扫描完成，等待5分钟后开始港股扫描...")
            time.sleep(300)
            
            # 运行港股扫描
            self.run_hk_scan(timing)
            
            logging.info(f"A股和港股扫描全部完成 - {timing}")
            
        except Exception as e:
            logging.error(f"顺序扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())

def premarket_scan():
    """盘前扫描"""
    logging.info("=" * 50)
    logging.info("开始盘前扫描")
    logging.info("=" * 50)
    
    scheduler = StockScanScheduler()
    scheduler.run_sequential_scans("盘前扫描")

def postmarket_scan():
    """盘后扫描"""
    logging.info("=" * 50)
    logging.info("开始盘后扫描")
    logging.info("=" * 50)
    
    scheduler = StockScanScheduler()
    scheduler.run_sequential_scans("盘后扫描")

def weekend_scan():
    """周末扫描"""
    logging.info("=" * 50)
    logging.info("开始周末扫描")
    logging.info("=" * 50)
    
    scheduler = StockScanScheduler()
    scheduler.run_both_scans("周末扫描")

def setup_schedule():
    """设置定时任务"""
    # A股交易时间：9:30-11:30, 13:00-15:00
    
    # 盘前扫描 - 每个交易日 9:00
    schedule.every().monday.at("09:00").do(premarket_scan)
    schedule.every().tuesday.at("09:00").do(premarket_scan)
    schedule.every().wednesday.at("09:00").do(premarket_scan)
    schedule.every().thursday.at("09:00").do(premarket_scan)
    schedule.every().friday.at("09:00").do(premarket_scan)
    
    # 盘后扫描 - 每个交易日 15:30
    schedule.every().monday.at("15:30").do(postmarket_scan)
    schedule.every().tuesday.at("15:30").do(postmarket_scan)
    schedule.every().wednesday.at("15:30").do(postmarket_scan)
    schedule.every().thursday.at("15:30").do(postmarket_scan)
    schedule.every().friday.at("15:30").do(postmarket_scan)
    
    # 周末扫描 - 周六 10:00
    schedule.every().saturday.at("10:00").do(weekend_scan)
    
    logging.info("定时任务设置完成:")
    logging.info("- 盘前扫描: 周一至周五 9:00")
    logging.info("- 盘后扫描: 周一至周五 15:30")
    logging.info("- 周末扫描: 周六 10:00")

def main():
    """主函数"""
    logging.info("股票扫描定时任务启动")
    logging.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置定时任务
    setup_schedule()
    
    # 显示下次运行时间
    def show_next_runs():
        jobs = schedule.get_jobs()
        if jobs:
            logging.info("即将执行的任务:")
            for job in jobs[:5]:  # 显示前5个任务
                logging.info(f"- {job.next_run.strftime('%Y-%m-%d %H:%M:%S')}: {job.job_func.__name__}")
        else:
            logging.info("没有安排的任务")
    
    show_next_runs()
    
    # 运行定时任务
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
            
            # 每小时显示一次状态
            if datetime.now().minute == 0:
                logging.info(f"定时任务运行中... 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                show_next_runs()
                
    except KeyboardInterrupt:
        logging.info("定时任务被用户中断")
    except Exception as e:
        logging.error(f"定时任务运行异常: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 