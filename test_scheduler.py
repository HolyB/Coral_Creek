#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票扫描定时任务测试脚本
用于验证定时任务设置和功能
"""

import sys
import os
import json
from datetime import datetime, timedelta
import pytz
import logging

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from scheduler_stock_scan_enhanced import EnhancedStockScanScheduler, BEIJING_TZ
except ImportError as e:
    print(f"导入增强版调度器失败: {e}")
    print("请确保 scheduler_stock_scan_enhanced.py 文件存在")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SchedulerTester:
    def __init__(self):
        self.scheduler = EnhancedStockScanScheduler()
        self.config_file = "stock_scanner_config.json"
        
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logging.warning("配置文件不存在，使用默认设置")
                return {}
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            return {}
    
    def test_timezone(self):
        """测试时区设置"""
        print("\n" + "="*50)
        print("时区测试")
        print("="*50)
        
        beijing_time = self.scheduler.get_beijing_time()
        local_time = datetime.now()
        utc_time = datetime.utcnow()
        
        print(f"本地时间: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"UTC时间:  {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"北京时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # 验证北京时间是否正确
        expected_beijing = datetime.now(BEIJING_TZ)
        time_diff = abs((beijing_time - expected_beijing).total_seconds())
        
        if time_diff < 60:  # 允许1分钟误差
            print("✓ 北京时间设置正确")
        else:
            print("✗ 北京时间设置可能有误")
    
    def test_trading_day_check(self):
        """测试交易日判断"""
        print("\n" + "="*50)
        print("交易日测试")
        print("="*50)
        
        beijing_time = self.scheduler.get_beijing_time()
        
        # 测试本周每一天
        for i in range(7):
            test_date = beijing_time + timedelta(days=i-beijing_time.weekday())
            is_trading = self.scheduler.is_trading_day(test_date)
            weekday_name = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][test_date.weekday()]
            
            status = "交易日" if is_trading else "非交易日"
            symbol = "✓" if is_trading and test_date.weekday() < 5 else "○"
            
            print(f"{symbol} {test_date.strftime('%m-%d')} {weekday_name}: {status}")
    
    def test_trading_session_check(self):
        """测试交易时段判断"""
        print("\n" + "="*50)
        print("交易时段测试")
        print("="*50)
        
        # 测试关键时间点
        test_times = [
            "08:00", "08:30", "09:00", "09:30", "10:30", "11:30",
            "12:00", "13:00", "14:00", "15:00", "15:30", "16:00", "16:30"
        ]
        
        beijing_time = self.scheduler.get_beijing_time()
        
        print("A股交易时段测试:")
        for time_str in test_times:
            hour, minute = map(int, time_str.split(':'))
            test_time = beijing_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            cn_status = self.scheduler.is_in_trading_session(test_time, 'cn')
            status_str = cn_status['session_name'] if cn_status['in_session'] else "非交易时段"
            symbol = "📈" if cn_status['in_session'] else "○"
            
            print(f"  {symbol} {time_str}: {status_str}")
        
        print("\n港股交易时段测试:")
        for time_str in test_times:
            hour, minute = map(int, time_str.split(':'))
            test_time = beijing_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            hk_status = self.scheduler.is_in_trading_session(test_time, 'hk')
            status_str = hk_status['session_name'] if hk_status['in_session'] else "非交易时段"
            symbol = "📊" if hk_status['in_session'] else "○"
            
            print(f"  {symbol} {time_str}: {status_str}")
    
    def test_script_existence(self):
        """测试扫描脚本是否存在"""
        print("\n" + "="*50)
        print("扫描脚本检查")
        print("="*50)
        
        cn_script = self.scheduler.cn_script
        hk_script = self.scheduler.hk_script
        
        print(f"A股扫描脚本: {cn_script}")
        if os.path.exists(cn_script):
            print("  ✓ A股扫描脚本存在")
        else:
            print("  ✗ A股扫描脚本不存在")
        
        print(f"港股扫描脚本: {hk_script}")
        if os.path.exists(hk_script):
            print("  ✓ 港股扫描脚本存在")
        else:
            print("  ✗ 港股扫描脚本不存在")
    
    def test_config_file(self):
        """测试配置文件"""
        print("\n" + "="*50)
        print("配置文件测试")
        print("="*50)
        
        config = self.load_config()
        
        if config:
            print("✓ 配置文件加载成功")
            
            # 检查关键配置项
            required_sections = ['scheduler_settings', 'trading_sessions', 'scan_schedules', 'scan_parameters']
            
            for section in required_sections:
                if section in config:
                    print(f"  ✓ {section} 配置存在")
                else:
                    print(f"  ✗ {section} 配置缺失")
        else:
            print("✗ 配置文件加载失败或不存在")
    
    def test_schedule_preview(self):
        """预览定时任务安排"""
        print("\n" + "="*50)
        print("定时任务安排预览")
        print("="*50)
        
        beijing_time = self.scheduler.get_beijing_time()
        
        # 模拟未来一周的任务安排
        schedules = [
            {"time": "08:30", "name": "盘前早期扫描", "days": [0,1,2,3,4]},
            {"time": "09:00", "name": "盘前扫描", "days": [0,1,2,3,4]},
            {"time": "10:30", "name": "上午盘中扫描", "days": [0,1,2,3,4]},
            {"time": "14:00", "name": "下午盘中扫描", "days": [0,1,2,3,4]},
            {"time": "15:30", "name": "盘后扫描", "days": [0,1,2,3,4]},
            {"time": "16:30", "name": "盘后深度扫描", "days": [0,1,2,3,4]},
            {"time": "10:00", "name": "周六综合扫描", "days": [5]},
            {"time": "20:00", "name": "周日准备扫描", "days": [6]}
        ]
        
        for i in range(7):
            test_date = beijing_time + timedelta(days=i)
            weekday_name = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][test_date.weekday()]
            
            print(f"\n{test_date.strftime('%m-%d')} {weekday_name}:")
            
            day_schedules = []
            for schedule in schedules:
                if test_date.weekday() in schedule['days']:
                    day_schedules.append(f"  {schedule['time']} - {schedule['name']}")
            
            if day_schedules:
                for schedule_str in day_schedules:
                    print(schedule_str)
            else:
                print("  无扫描任务")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("股票扫描定时任务系统测试")
        print("="*80)
        
        try:
            self.test_timezone()
            self.test_trading_day_check()
            self.test_trading_session_check()
            self.test_script_existence()
            self.test_config_file()
            self.test_schedule_preview()
            
            print("\n" + "="*80)
            print("测试完成！")
            print("如果所有项目都显示 ✓，说明系统配置正确")
            print("如果有 ✗ 标记，请检查相应的配置或文件")
            print("="*80)
            
        except Exception as e:
            print(f"\n测试过程中出现异常: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    tester = SchedulerTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 