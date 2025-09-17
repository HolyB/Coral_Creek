import schedule
import time
import subprocess
import logging
from datetime import datetime, timedelta
import os
import sys
import threading
import traceback
import pytz
from typing import List, Dict, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_stock_scanner_scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

class EnhancedStockScanScheduler:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cn_script = os.path.join(self.script_dir, 'scan_cn_signals_multi_thread_tushare.py')
        self.hk_script = os.path.join(self.script_dir, 'scan_hk_signals_multi_thread_tushare.py')
        
        # 检查脚本是否存在
        if not os.path.exists(self.cn_script):
            logging.error(f"A股扫描脚本不存在: {self.cn_script}")
        if not os.path.exists(self.hk_script):
            logging.error(f"港股扫描脚本不存在: {self.hk_script}")
            
        # A股交易时间段
        self.cn_trading_sessions = [
            {"start": "09:30", "end": "11:30", "name": "上午交易时段"},
            {"start": "13:00", "end": "15:00", "name": "下午交易时段"}
        ]
        
        # 港股交易时间段
        self.hk_trading_sessions = [
            {"start": "09:30", "end": "12:00", "name": "港股上午交易时段"},
            {"start": "13:00", "end": "16:00", "name": "港股下午交易时段"}
        ]
    
    def get_beijing_time(self) -> datetime:
        """获取北京时间"""
        return datetime.now(BEIJING_TZ)
    
    def is_trading_day(self, date: datetime = None) -> bool:
        """判断是否为交易日（周一到周五）"""
        if date is None:
            date = self.get_beijing_time()
        return date.weekday() < 5  # 0-4 代表周一到周五
    
    def is_in_trading_session(self, current_time: datetime = None, market: str = 'cn') -> Dict:
        """判断当前时间是否在交易时段内"""
        if current_time is None:
            current_time = self.get_beijing_time()
            
        current_time_str = current_time.strftime("%H:%M")
        sessions = self.cn_trading_sessions if market == 'cn' else self.hk_trading_sessions
        
        for session in sessions:
            if session["start"] <= current_time_str <= session["end"]:
                return {"in_session": True, "session_name": session["name"]}
                
        return {"in_session": False, "session_name": None}
    
    def run_cn_scan(self, timing: str = "", batch_size: int = 300, max_workers: int = 5):
        """运行A股扫描"""
        try:
            beijing_time = self.get_beijing_time()
            logging.info(f"开始运行A股扫描 - {timing} (北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 构建扫描参数
            cmd = [
                sys.executable, self.cn_script,
                "--batch_size", str(batch_size),
                "--max_workers", str(max_workers),
                "--send_email", "True"
            ]
            
            # 根据时间段调整扫描参数
            if "盘中" in timing:
                cmd.extend(["--signal_type", "both", "--min_turnover", "500"])
            elif "盘前" in timing:
                cmd.extend(["--signal_type", "bullish", "--min_turnover", "100"])
            elif "盘后" in timing:
                cmd.extend(["--signal_type", "both", "--min_turnover", "200"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=3600)
            
            if result.returncode == 0:
                logging.info(f"A股扫描完成 - {timing}")
                logging.info(f"A股扫描输出: {result.stdout[-500:]}")  # 只显示最后500字符
            else:
                logging.error(f"A股扫描失败 - {timing}")
                logging.error(f"A股扫描错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error(f"A股扫描超时 - {timing}")
        except Exception as e:
            logging.error(f"A股扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())
    
    def run_hk_scan(self, timing: str = "", batch_size: int = 200, max_workers: int = 8):
        """运行港股扫描"""
        try:
            beijing_time = self.get_beijing_time()
            logging.info(f"开始运行港股扫描 - {timing} (北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 构建扫描参数
            cmd = [
                sys.executable, self.hk_script,
                "--batch_size", str(batch_size),
                "--max_workers", str(max_workers),
                "--send_email", "True"
            ]
            
            # 根据时间段调整扫描参数
            if "盘中" in timing:
                cmd.extend(["--signal_type", "both", "--min_turnover", "1000"])
            elif "盘前" in timing:
                cmd.extend(["--signal_type", "bullish", "--min_turnover", "500"])
            elif "盘后" in timing:
                cmd.extend(["--signal_type", "both", "--min_turnover", "800"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=3600)
            
            if result.returncode == 0:
                logging.info(f"港股扫描完成 - {timing}")
                logging.info(f"港股扫描输出: {result.stdout[-500:]}")
            else:
                logging.error(f"港股扫描失败 - {timing}")
                logging.error(f"港股扫描错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error(f"港股扫描超时 - {timing}")
        except Exception as e:
            logging.error(f"港股扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())
    
    def run_parallel_scans(self, timing: str = ""):
        """并行运行A股和港股扫描"""
        try:
            beijing_time = self.get_beijing_time()
            logging.info(f"开始并行运行A股和港股扫描 - {timing} (北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 创建线程并行运行
            cn_thread = threading.Thread(target=self.run_cn_scan, args=(timing,))
            hk_thread = threading.Thread(target=self.run_hk_scan, args=(timing,))
            
            # 启动线程
            cn_thread.start()
            time.sleep(15)  # 错开15秒启动，避免API冲突
            hk_thread.start()
            
            # 等待线程完成
            cn_thread.join()
            hk_thread.join()
            
            logging.info(f"A股和港股扫描全部完成 - {timing}")
            
        except Exception as e:
            logging.error(f"并行扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())
    
    def run_sequential_scans(self, timing: str = ""):
        """顺序运行A股和港股扫描"""
        try:
            beijing_time = self.get_beijing_time()
            logging.info(f"开始顺序运行A股和港股扫描 - {timing} (北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 先运行A股扫描
            self.run_cn_scan(timing)
            
            # 等待3分钟后运行港股扫描
            logging.info("A股扫描完成，等待3分钟后开始港股扫描...")
            time.sleep(180)
            
            # 运行港股扫描
            self.run_hk_scan(timing)
            
            logging.info(f"A股和港股扫描全部完成 - {timing}")
            
        except Exception as e:
            logging.error(f"顺序扫描异常 - {timing}: {e}")
            logging.error(traceback.format_exc())

# 扫描任务函数
def premarket_scan_8_30():
    """盘前早期扫描 - 8:30"""
    beijing_time = datetime.now(BEIJING_TZ)
    if not beijing_time.weekday() < 5:  # 只在工作日执行
        logging.info("今天不是交易日，跳过盘前早期扫描")
        return
        
    logging.info("=" * 60)
    logging.info("开始盘前早期扫描 (8:30)")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_sequential_scans("盘前早期扫描")

def premarket_scan_9_00():
    """盘前扫描 - 9:00"""
    beijing_time = datetime.now(BEIJING_TZ)
    if not beijing_time.weekday() < 5:
        logging.info("今天不是交易日，跳过盘前扫描")
        return
        
    logging.info("=" * 60)
    logging.info("开始盘前扫描 (9:00)")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_parallel_scans("盘前扫描")

def intraday_scan_10_30():
    """盘中扫描 - 10:30"""
    beijing_time = datetime.now(BEIJING_TZ)
    if not beijing_time.weekday() < 5:
        logging.info("今天不是交易日，跳过盘中扫描")
        return
        
    logging.info("=" * 60)
    logging.info("开始盘中扫描 (10:30)")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_parallel_scans("盘中扫描")

def intraday_scan_14_00():
    """盘中扫描 - 14:00"""
    beijing_time = datetime.now(BEIJING_TZ)
    if not beijing_time.weekday() < 5:
        logging.info("今天不是交易日，跳过下午盘中扫描")
        return
        
    logging.info("=" * 60)
    logging.info("开始下午盘中扫描 (14:00)")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_parallel_scans("下午盘中扫描")

def postmarket_scan_15_30():
    """盘后扫描 - 15:30"""
    beijing_time = datetime.now(BEIJING_TZ)
    if not beijing_time.weekday() < 5:
        logging.info("今天不是交易日，跳过盘后扫描")
        return
        
    logging.info("=" * 60)
    logging.info("开始盘后扫描 (15:30)")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_sequential_scans("盘后扫描")

def postmarket_scan_16_30():
    """盘后深度扫描 - 16:30"""
    beijing_time = datetime.now(BEIJING_TZ)
    if not beijing_time.weekday() < 5:
        logging.info("今天不是交易日，跳过盘后深度扫描")
        return
        
    logging.info("=" * 60)
    logging.info("开始盘后深度扫描 (16:30)")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_parallel_scans("盘后深度扫描")

def weekend_scan():
    """周末扫描"""
    logging.info("=" * 60)
    logging.info("开始周末综合扫描")
    logging.info("=" * 60)
    
    scheduler = EnhancedStockScanScheduler()
    scheduler.run_parallel_scans("周末综合扫描")

def setup_enhanced_schedule():
    """设置增强版定时任务"""
    logging.info("设置增强版定时任务 (基于北京时间)...")
    
    # 盘前扫描
    schedule.every().monday.at("08:30").do(premarket_scan_8_30)
    schedule.every().tuesday.at("08:30").do(premarket_scan_8_30)
    schedule.every().wednesday.at("08:30").do(premarket_scan_8_30)
    schedule.every().thursday.at("08:30").do(premarket_scan_8_30)
    schedule.every().friday.at("08:30").do(premarket_scan_8_30)
    
    schedule.every().monday.at("09:00").do(premarket_scan_9_00)
    schedule.every().tuesday.at("09:00").do(premarket_scan_9_00)
    schedule.every().wednesday.at("09:00").do(premarket_scan_9_00)
    schedule.every().thursday.at("09:00").do(premarket_scan_9_00)
    schedule.every().friday.at("09:00").do(premarket_scan_9_00)
    
    # 盘中扫描
    schedule.every().monday.at("10:30").do(intraday_scan_10_30)
    schedule.every().tuesday.at("10:30").do(intraday_scan_10_30)
    schedule.every().wednesday.at("10:30").do(intraday_scan_10_30)
    schedule.every().thursday.at("10:30").do(intraday_scan_10_30)
    schedule.every().friday.at("10:30").do(intraday_scan_10_30)
    
    schedule.every().monday.at("14:00").do(intraday_scan_14_00)
    schedule.every().tuesday.at("14:00").do(intraday_scan_14_00)
    schedule.every().wednesday.at("14:00").do(intraday_scan_14_00)
    schedule.every().thursday.at("14:00").do(intraday_scan_14_00)
    schedule.every().friday.at("14:00").do(intraday_scan_14_00)
    
    # 盘后扫描
    schedule.every().monday.at("15:30").do(postmarket_scan_15_30)
    schedule.every().tuesday.at("15:30").do(postmarket_scan_15_30)
    schedule.every().wednesday.at("15:30").do(postmarket_scan_15_30)
    schedule.every().thursday.at("15:30").do(postmarket_scan_15_30)
    schedule.every().friday.at("15:30").do(postmarket_scan_15_30)
    
    schedule.every().monday.at("16:30").do(postmarket_scan_16_30)
    schedule.every().tuesday.at("16:30").do(postmarket_scan_16_30)
    schedule.every().wednesday.at("16:30").do(postmarket_scan_16_30)
    schedule.every().thursday.at("16:30").do(postmarket_scan_16_30)
    schedule.every().friday.at("16:30").do(postmarket_scan_16_30)
    
    # 周末扫描
    schedule.every().saturday.at("10:00").do(weekend_scan)
    schedule.every().sunday.at("20:00").do(weekend_scan)
    
    logging.info("增强版定时任务设置完成:")
    logging.info("工作日扫描时间表 (北京时间):")
    logging.info("- 盘前早期扫描: 8:30 (顺序执行，适合获取隔夜消息)")
    logging.info("- 盘前扫描: 9:00 (并行执行，开盘前最后扫描)")
    logging.info("- 盘中扫描: 10:30 (并行执行，上午交易时段中期)")
    logging.info("- 下午盘中扫描: 14:00 (并行执行，下午开盘后)")
    logging.info("- 盘后扫描: 15:30 (顺序执行，收盘后立即扫描)")
    logging.info("- 盘后深度扫描: 16:30 (并行执行，深度分析)")
    logging.info("周末扫描:")
    logging.info("- 周六: 10:00")
    logging.info("- 周日: 20:00 (为下周做准备)")

def show_next_runs(max_show: int = 10):
    """显示即将执行的任务"""
    jobs = schedule.get_jobs()
    if jobs:
        beijing_time = datetime.now(BEIJING_TZ)
        logging.info(f"当前北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logging.info(f"即将执行的 {min(len(jobs), max_show)} 个任务:")
        
        # 按时间排序
        sorted_jobs = sorted(jobs, key=lambda x: x.next_run)
        
        for i, job in enumerate(sorted_jobs[:max_show]):
            next_run_beijing = job.next_run.replace(tzinfo=pytz.timezone('Asia/Shanghai'))
            time_diff = next_run_beijing - beijing_time
            
            if time_diff.total_seconds() > 0:
                if time_diff.days > 0:
                    time_str = f"{time_diff.days}天后"
                elif time_diff.seconds > 3600:
                    hours = time_diff.seconds // 3600
                    time_str = f"{hours}小时后"
                else:
                    minutes = time_diff.seconds // 60
                    time_str = f"{minutes}分钟后"
            else:
                time_str = "即将执行"
                
            logging.info(f"  {i+1}. {next_run_beijing.strftime('%m-%d %H:%M')}: {job.job_func.__name__} ({time_str})")
    else:
        logging.info("没有安排的任务")

def main():
    """主函数"""
    beijing_time = datetime.now(BEIJING_TZ)
    logging.info("=" * 80)
    logging.info("增强版股票扫描定时任务启动")
    logging.info(f"启动时间 (北京时间): {beijing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logging.info("=" * 80)
    
    # 设置定时任务
    setup_enhanced_schedule()
    
    # 显示任务计划
    show_next_runs()
    
    # 运行定时任务
    try:
        last_status_time = datetime.now()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
            
            # 每小时显示一次状态
            current_time = datetime.now()
            if (current_time - last_status_time).total_seconds() >= 3600:  # 1小时
                beijing_time = datetime.now(BEIJING_TZ)
                logging.info("=" * 50)
                logging.info(f"定时任务运行中... 北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 显示交易状态
                scheduler = EnhancedStockScanScheduler()
                cn_status = scheduler.is_in_trading_session(beijing_time, 'cn')
                hk_status = scheduler.is_in_trading_session(beijing_time, 'hk')
                
                if cn_status["in_session"]:
                    logging.info(f"A股交易状态: {cn_status['session_name']}")
                else:
                    logging.info("A股交易状态: 非交易时段")
                    
                if hk_status["in_session"]:
                    logging.info(f"港股交易状态: {hk_status['session_name']}")
                else:
                    logging.info("港股交易状态: 非交易时段")
                
                show_next_runs(5)
                logging.info("=" * 50)
                last_status_time = current_time
                
    except KeyboardInterrupt:
        logging.info("定时任务被用户中断")
    except Exception as e:
        logging.error(f"定时任务运行异常: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 