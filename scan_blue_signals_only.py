#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门扫描BLUE信号的A股扫描脚本
只关注BLUE信号，移除LIRED相关逻辑
基于scan_signals_multi_thread_claude.py简化而来
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import time
import threading
import concurrent.futures
from tqdm import tqdm
import requests
import os
import json
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入增强版股票列表
from enhanced_stock_list import get_enhanced_cn_stock_list

# 创建线程锁
print_lock = threading.Lock()
results_lock = threading.Lock()

# 全局变量存储公司信息
COMPANY_INFO = {}

# 定义富途函数
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

class BlueSignalNotifier:
    """BLUE信号邮件通知类"""
    def __init__(self, symbol, result_data):
        """
        初始化通知类
        
        Args:
            symbol (str): 股票代码
            result_data (dict): 股票BLUE信号数据
        """
        self.symbol = symbol
        self.data = result_data
        self.sender_email = "stockprofile138@gmail.com"
        self.receiver_emails = ["stockprofile138@gmail.com"]
        self.email_password = "vselpmwrjacmgdib"
    
    def send_signal_email(self):
        """发送BLUE信号邮件"""
        subject = f"BLUE信号通知: {self.symbol} 出现BLUE交易信号"
        
        # 构建邮件正文
        body = f"股票代码: {self.symbol}\n"
        if 'company_name' in self.data:
            body += f"公司名称: {self.data['company_name']}\n"
        body += f"当前价格: {self.data['price']:.2f}\n"
        body += f"成交额(万): {self.data['turnover']:.2f}\n\n"
        
        # 添加BLUE信号信息
        body += "BLUE信号详情:\n"
        body += f"日线BLUE: {self.data['blue_daily']:.2f}, 最近信号值: {self.data['latest_day_blue_value']:.2f}, 出现天数: {self.data['blue_days']}\n"
        body += f"周线BLUE: {self.data['blue_weekly']:.2f}, 最近信号值: {self.data['latest_week_blue_value']:.2f}, 出现周数: {self.data['blue_weeks']}\n\n"
        
        # 添加组合信号信息
        signals = []
        if self.data['blue_days'] >= 3:
            signals.append(f"日BLUE: {self.data['latest_day_blue_value']:.2f}")
        if self.data['blue_weeks'] >= 2:
            signals.append(f"周BLUE: {self.data['latest_week_blue_value']:.2f}")
            
        if self.data['has_day_week_blue']:
            body += "⭐ 强信号: 日线和周线BLUE同时满足条件\n"
            
        body += f"\n检测到的BLUE信号组合: {', '.join(signals)}\n"
        
        body += f"\n扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.receiver_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.email_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_emails, text)
            server.quit()
            
            print(f"✅ BLUE信号邮件已发送: {self.symbol}")
            return True
            
        except Exception as e:
            print(f"❌ 发送邮件失败: {e}")
            return False

    @staticmethod
    def send_summary_email(results, signal_counts):
        """发送BLUE信号汇总邮件"""
        if not results:
            return False
        
        subject = f"BLUE信号汇总报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # 构建邮件正文
        body = f"BLUE信号汇总报告\n"
        body += f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        body += f"共发现 {len(results)} 只股票出现BLUE信号\n\n"
        
        # 添加每只股票的详细信息
        body += "具体股票信息:\n"
        body += "-" * 60 + "\n"
        
        for stock in results:
            body += f"代码: {stock['symbol']}\n"
            if 'company_name' in stock:
                body += f"名称: {stock['company_name']}\n"
            body += f"价格: {stock['price']:.2f}, 成交额: {stock['turnover']:.2f}万\n"
            
            # 添加BLUE信号信息
            signals = []
            if stock['blue_days'] >= 3:
                signals.append(f"日BLUE: {stock['latest_day_blue_value']:.2f}")
            if stock['blue_weeks'] >= 2:
                signals.append(f"周BLUE: {stock['latest_week_blue_value']:.2f}")
                
            body += f"BLUE信号: {', '.join(signals)}\n"
            
            # 添加组合信号
            if stock['has_day_week_blue']:
                body += f"   ⭐ 日线和周线BLUE同时满足条件\n"
                
            body += "\n"
            
        # 添加统计信息
        body += f"\nBLUE信号统计:\n"
        
        # 统计各类BLUE信号出现的次数
        signal_counts = {
            '日BLUE': len([s for s in results if s['blue_days'] >= 3]),
            '周BLUE': len([s for s in results if s['blue_weeks'] >= 2]),
            '日周BLUE同时': len([s for s in results if s['has_day_week_blue']])
        }
        
        for signal, count in signal_counts.items():
            if count > 0:
                body += f"   {signal}: {count} 只股票\n"
        
        body += f"\n扫描完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 添加日线和周线同时出现BLUE信号的股票表格
        dual_signal_stocks = [s for s in results if s['has_day_week_blue']]
        if dual_signal_stocks:
            body += "\n\n日线和周线同时出现BLUE信号的股票：\n"
            body += "-" * 60 + "\n"
            body += f"{'代码':<8} | {'公司名称':<20} | {'价格':>8} | {'日BLUE':>8} | {'周BLUE':>8}\n"
            body += "-" * 60 + "\n"
            
            for stock in dual_signal_stocks:
                company_name = stock.get('company_name', 'N/A')[:18]
                body += f"{stock['symbol']:<8} | {company_name:<20} | {stock['price']:>8.2f} | {stock['latest_day_blue_value']:>8.1f} | {stock['latest_week_blue_value']:>8.1f}\n"
        
        try:
            sender_email = "stockprofile138@gmail.com"
            receiver_emails = ["stockprofile138@gmail.com"]
            email_password = "vselpmwrjacmgdib"
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(receiver_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, email_password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_emails, text)
            server.quit()
            
            print(f"✅ BLUE信号汇总邮件已发送")
            return True
            
        except Exception as e:
            print(f"❌ 发送汇总邮件失败: {e}")
            return False

def get_cn_tickers():
    """获取A股股票列表，包括北交所股票"""
    try:
        # 使用增强版股票列表获取方法
        stock_df = get_enhanced_cn_stock_list(force_refresh=False)
        
        if stock_df.empty:
            print("❌ 无法获取股票列表")
            return pd.DataFrame()
        
        # 转换为原脚本期望的格式
        tickers = []
        for _, row in stock_df.iterrows():
            tickers.append({
                'code': row['code'],
                'name': row['name']
            })
        
        print(f"✅ 获取到 {len(tickers)} 只A股")
        return pd.DataFrame(tickers)
        
    except Exception as e:
        print(f"❌ 获取A股列表失败: {e}")
        return pd.DataFrame()

def process_single_stock(symbol, thresholds=None):
    """处理单个股票，仅关注BLUE信号"""
    # 设置默认阈值
    default_thresholds = {
        'day_blue': 100,
        'week_blue': 130,
        'day_blue_count': 3,
        'week_blue_count': 2
    }
    
    if thresholds:
        default_thresholds.update(thresholds)
    
    try:
        from Stock_utils.stock_analysis import StockAnalysis
        from Stock_utils.stock_data_fetcher import StockDataFetcher
        
        # 获取股票数据
        fetcher = StockDataFetcher(symbol, source='akshare')
        data_daily = fetcher.get_stock_data()
        
        if data_daily is None or len(data_daily) < 100:
            return None
        
        # 转换为周线数据
        data_weekly = data_daily.resample('W', on='Date' if 'Date' in data_daily.columns else data_daily.index).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(data_weekly) < 20:
            return None
        
        # 计算BLUE信号（只保留BLUE相关计算）
        OPEN_D = data_daily['Open'].values
        HIGH_D = data_daily['High'].values
        LOW_D = data_daily['Low'].values
        CLOSE_D = data_daily['Close'].values
        
        OPEN_W = data_weekly['Open'].values
        HIGH_W = data_weekly['High'].values
        LOW_W = data_weekly['Low'].values
        CLOSE_W = data_weekly['Close'].values
        
        # 日线BLUE计算
        VAR1_D = REF((LOW_D + OPEN_D + CLOSE_D + HIGH_D) / 4, 1)
        VAR2_D = SMA(np.abs(LOW_D - VAR1_D), 13, 1) / SMA(np.maximum(LOW_D - VAR1_D, 0), 10, 1)
        VAR3_D = EMA(VAR2_D, 10)
        VAR4_D = LLV(LOW_D, 9)
        VAR5_D = HHV(VAR3_D, 30)
        VAR6_D = IF(LLV(LOW_D, 58) == VAR4_D, VAR3_D, 0)
        
        max_value_daily = np.nanmax(VAR6_D)
        RADIO1_D = 200 / max_value_daily if max_value_daily > 0 else 1
        BLUE_D = IF(VAR5_D > REF(VAR5_D, 1), VAR6_D * RADIO1_D, 0)
        
        # 周线BLUE计算
        VAR1_W = REF((LOW_W + OPEN_W + CLOSE_W + HIGH_W) / 4, 1)
        VAR2_W = SMA(np.abs(LOW_W - VAR1_W), 13, 1) / SMA(np.maximum(LOW_W - VAR1_W, 0), 10, 1)
        VAR3_W = EMA(VAR2_W, 10)
        VAR4_W = LLV(LOW_W, 9)
        VAR5_W = HHV(VAR3_W, 30)
        VAR6_W = IF(LLV(LOW_W, 58) == VAR4_W, VAR3_W, 0)
        
        max_value_weekly = np.nanmax(VAR6_W)
        RADIO1_W = 200 / max_value_weekly if max_value_weekly > 0 else 1
        BLUE_W = IF(VAR5_W > REF(VAR5_W, 1), VAR6_W * RADIO1_W, 0)
        
        df_daily = pd.DataFrame({
            'Open': OPEN_D, 'High': HIGH_D, 'Low': LOW_D, 'Close': CLOSE_D,
            'Volume': data_daily['Volume'].values,
            'BLUE': BLUE_D
        }, index=data_daily.index)
        
        df_weekly = pd.DataFrame({
            'Open': OPEN_W, 'High': HIGH_W, 'Low': LOW_W, 'Close': CLOSE_W,
            'Volume': data_weekly['Volume'].values,
            'BLUE': BLUE_W
        }, index=data_weekly.index)
        
        # 调整为最近6天和5周
        recent_daily = df_daily.tail(6)
        recent_weekly = df_weekly.tail(5)
        
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]
        
        # 查找满足BLUE条件的具体数值
        day_blue_signals = recent_daily[recent_daily['BLUE'] > default_thresholds['day_blue']]['BLUE'].tolist()
        week_blue_signals = recent_weekly[recent_weekly['BLUE'] > default_thresholds['week_blue']]['BLUE'].tolist()
        
        day_blue_count = len(day_blue_signals)
        week_blue_count = len(week_blue_signals)
        
        # 存储最近一次满足条件的信号值
        latest_day_blue_value = day_blue_signals[-1] if day_blue_signals else 0
        latest_week_blue_value = week_blue_signals[-1] if week_blue_signals else 0
        
        has_blue_signal = day_blue_count >= default_thresholds['day_blue_count'] or week_blue_count >= default_thresholds['week_blue_count']
        
        if has_blue_signal:
            # 计算成交额（万元）
            turnover = latest_daily['Volume'] * latest_daily['Close'] / 10000
            
            # 检查是否同时满足日线和周线BLUE条件
            has_day_week_blue = (day_blue_count >= default_thresholds['day_blue_count'] and 
                                week_blue_count >= default_thresholds['week_blue_count'])
            
            result = {
                'symbol': symbol,
                'price': latest_daily['Close'],
                'Volume': latest_daily['Volume'],
                'turnover': turnover,
                'blue_daily': latest_daily['BLUE'],
                'blue_weekly': latest_weekly['BLUE'],
                'blue_days': day_blue_count,
                'blue_weeks': week_blue_count,
                'latest_day_blue_value': latest_day_blue_value,
                'latest_week_blue_value': latest_week_blue_value,
                'has_day_week_blue': has_day_week_blue,
                'timestamp': datetime.now()
            }
            
            with print_lock:
                signal_desc = f"日BLUE:{day_blue_count}天" if day_blue_count >= default_thresholds['day_blue_count'] else ""
                if week_blue_count >= default_thresholds['week_blue_count']:
                    if signal_desc:
                        signal_desc += f",周BLUE:{week_blue_count}周"
                    else:
                        signal_desc = f"周BLUE:{week_blue_count}周"
                
                if has_day_week_blue:
                    signal_desc += " [日周同时]"
                
                print(f"✅ 发现BLUE信号: {symbol} - {signal_desc}, 价格: {latest_daily['Close']:.2f}, 成交额: {turnover:.0f}万")
            
            return result
        
        return None
        
    except Exception as e:
        with print_lock:
            print(f"⚠️ 处理 {symbol} 时出错: {e}")
        return None

def scan_blue_signals(tickers, max_workers=20, min_turnover=200, thresholds=None, send_email=False):
    """扫描BLUE信号"""
    
    print(f"🔍 开始扫描BLUE信号...")
    print(f"📊 股票数量: {len(tickers)}")
    print(f"⚙️ 参数: 线程数={max_workers}, 最小成交额={min_turnover}万")
    if thresholds:
        print(f"🎯 BLUE阈值: 日线>{thresholds.get('day_blue', 100)}, 周线>{thresholds.get('week_blue', 130)}")
    print("-" * 60)
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_symbol = {executor.submit(process_single_stock, ticker['code'], thresholds): ticker['code'] for ticker in tickers}
        
        # 处理结果
        for future in tqdm(as_completed(future_to_symbol), total=len(future_to_symbol), desc="扫描进度"):
            try:
                result = future.result(timeout=30)
                if result and result['turnover'] >= min_turnover:
                    with results_lock:
                        results.append(result)
                        
                        # 可选择发送单独邮件通知
                        if send_email:
                            try:
                                # 获取公司名称
                                company_name = COMPANY_INFO.get(result['symbol'], {}).get('name', '')
                                result['company_name'] = company_name
                                
                                notifier = BlueSignalNotifier(result['symbol'], result)
                                notifier.send_signal_email()
                            except Exception as e:
                                print(f"发送邮件失败: {e}")
                        
            except Exception as e:
                symbol = future_to_symbol[future]
                print(f"任务异常: {symbol} - {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n🎉 BLUE信号扫描完成!")
    print(f"⏱️ 耗时: {elapsed_time:.2f} 秒")
    print(f"🎯 发现 {len(results)} 只股票满足BLUE信号条件")
    
    if results:
        # 按成交额排序
        results.sort(key=lambda x: x['turnover'], reverse=True)
        
        # 保存结果到CSV
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'blue_signals_{timestamp}.csv'
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 结果已保存到: {filename}")
        
        # 显示前10个结果
        print(f"\n📋 前10个BLUE信号:")
        print("-" * 80)
        for i, result in enumerate(results[:10], 1):
            blue_desc = []
            if result['blue_days'] >= (thresholds or {}).get('day_blue_count', 3):
                blue_desc.append(f"日BLUE:{result['blue_days']}天({result['latest_day_blue_value']:.1f})")
            if result['blue_weeks'] >= (thresholds or {}).get('week_blue_count', 2):
                blue_desc.append(f"周BLUE:{result['blue_weeks']}周({result['latest_week_blue_value']:.1f})")
            
            blue_str = ", ".join(blue_desc)
            dual_flag = " ⭐" if result['has_day_week_blue'] else ""
            
            print(f"{i:2d}. {result['symbol']} - 价格:{result['price']:7.2f} 成交额:{result['turnover']:8.0f}万 {blue_str}{dual_flag}")
        
        # 统计信息
        day_blue_count = len([r for r in results if r['blue_days'] >= (thresholds or {}).get('day_blue_count', 3)])
        week_blue_count = len([r for r in results if r['blue_weeks'] >= (thresholds or {}).get('week_blue_count', 2)])
        dual_blue_count = len([r for r in results if r['has_day_week_blue']])
        
        print(f"\n📊 BLUE信号统计:")
        print(f"   日线BLUE信号: {day_blue_count} 只")
        print(f"   周线BLUE信号: {week_blue_count} 只")
        print(f"   日周同时BLUE: {dual_blue_count} 只")
        
        # 发送汇总邮件
        if send_email:
            try:
                BlueSignalNotifier.send_summary_email(results, {})
            except Exception as e:
                print(f"发送汇总邮件失败: {e}")
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股BLUE信号专用扫描器')
    parser.add_argument('--batch_size', type=int, default=0, help='批量处理数量 (0=全部)')
    parser.add_argument('--max_workers', type=int, default=20, help='最大线程数')
    parser.add_argument('--min_turnover', type=float, default=200, help='最小成交额(万元)')
    parser.add_argument('--day_blue', type=float, default=100, help='日线BLUE阈值')
    parser.add_argument('--week_blue', type=float, default=130, help='周线BLUE阈值')
    parser.add_argument('--day_blue_count', type=int, default=3, help='日线BLUE出现次数')
    parser.add_argument('--week_blue_count', type=int, default=2, help='周线BLUE出现次数')
    parser.add_argument('--send_email', action='store_true', help='发送邮件通知')
    parser.add_argument('--timing', type=str, default='', help='时机标识')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("🔵 A股BLUE信号专用扫描系统")
        print("=" * 80)
        print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.timing:
            print(f"扫描时机: {args.timing}")
        print("=" * 80)
        
        # 获取股票列表
        tickers_df = get_cn_tickers()
        if tickers_df.empty:
            print("❌ 无法获取股票列表")
            return
        
        tickers = tickers_df.to_dict('records')
        
        # 限制批量大小
        if args.batch_size > 0:
            tickers = tickers[:args.batch_size]
            print(f"🎯 本次扫描: {len(tickers)} 只股票")
        
        # 设置阈值
        thresholds = {
            'day_blue': args.day_blue,
            'week_blue': args.week_blue,
            'day_blue_count': args.day_blue_count,
            'week_blue_count': args.week_blue_count
        }
        
        # 开始扫描
        results = scan_blue_signals(
            tickers=tickers,
            max_workers=args.max_workers,
            min_turnover=args.min_turnover,
            thresholds=thresholds,
            send_email=args.send_email
        )
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断扫描")
    except Exception as e:
        print(f"❌ 扫描过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
