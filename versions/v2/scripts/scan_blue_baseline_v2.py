#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline BLUE Signal Scanner - US Market
使用数据库缓存股票信息，支持 Telegram + Email 通知

Usage:
    python scan_blue_baseline_v2.py
    python scan_blue_baseline_v2.py --limit 500 --no-email
    python scan_blue_baseline_v2.py --with-heima
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from tqdm import tqdm

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(parent_dir, '.env'))

# 导入 v2 模块
from data_fetcher import get_all_us_tickers, get_us_stock_data, get_ticker_details
from db.database import init_db, get_stock_info_batch, upsert_stock_info


# ==================== 技术指标函数 ====================

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

def MA(series, periods):
    return pd.Series(series).rolling(window=periods, min_periods=1).mean().values

def AVEDEV(series, periods):
    s = pd.Series(series)
    return s.rolling(window=periods, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values


def calculate_blue_signal(ohlcv_df):
    """计算 BLUE 信号"""
    if ohlcv_df is None or len(ohlcv_df) < 50:
        return None, None
    
    OPEN = ohlcv_df['Open'].values
    HIGH = ohlcv_df['High'].values
    LOW = ohlcv_df['Low'].values
    CLOSE = ohlcv_df['Close'].values
    
    VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
    VAR2 = SMA(np.abs(LOW - VAR1), 13, 1) / SMA(np.maximum(LOW - VAR1, 0), 10, 1)
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(LOW, 33)
    VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
    VAR6 = POW(np.abs(VAR5), 0.3) * np.sign(VAR5)
    
    VAR21 = SMA(np.abs(HIGH - VAR1), 13, 1) / SMA(np.minimum(HIGH - VAR1, 0), 10, 1)
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(HIGH, 33)
    VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)
    VAR61 = POW(np.abs(VAR51), 0.3) * np.sign(VAR51)
    
    max_value = np.nanmax(np.maximum(VAR6, np.abs(VAR61)))
    RADIO1 = 200 / max_value if max_value > 0 else 1
    BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
    
    return BLUE, CLOSE


def process_single_stock(symbol, thresholds):
    """处理单个股票 - 检查日线和周线同时有 BLUE 信号"""
    try:
        # 获取日线数据
        df = get_us_stock_data(symbol, days=365)
        if df is None or len(df) < 50:
            return None
        
        # 过滤成交额太低的股票
        latest_turnover = df['Volume'].iloc[-1] * df['Close'].iloc[-1] / 10000
        if latest_turnover < 100:  # 成交额 < 100万
            return None
        
        # 计算日线 BLUE 信号
        BLUE_D, CLOSE_D = calculate_blue_signal(df)
        if BLUE_D is None:
            return None
        
        # 生成周线数据
        df_weekly = df.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        }).dropna()
        
        if len(df_weekly) < 20:
            return None
        
        # 计算周线 BLUE 信号
        BLUE_W, CLOSE_W = calculate_blue_signal(df_weekly)
        if BLUE_W is None:
            return None
        
        # 检查最近几天/周的信号
        recent_daily = BLUE_D[-thresholds['day_lookback']:]
        recent_weekly = BLUE_W[-thresholds['week_lookback']:]
        
        day_blue_count = np.sum(recent_daily > thresholds['day_blue'])
        week_blue_count = np.sum(recent_weekly > thresholds['week_blue'])
        
        has_day_blue = day_blue_count >= thresholds['day_blue_count']
        has_week_blue = week_blue_count >= thresholds['week_blue_count']
        
        # 必须日线和周线同时有信号
        if has_day_blue and has_week_blue:
            return {
                'symbol': symbol,
                'price': float(CLOSE_D[-1]),
                'turnover': float(latest_turnover),
                'blue_daily': float(BLUE_D[-1]),
                'blue_days': int(day_blue_count),
                'max_day_blue': float(np.max(recent_daily)),
                'blue_weekly': float(BLUE_W[-1]),
                'blue_weeks': int(week_blue_count),
                'max_week_blue': float(np.max(recent_weekly)),
                'has_day_week_blue': True,
            }
        
        return None
        
    except Exception as e:
        return None



def scan_stocks(tickers, thresholds, max_workers=30):
    """批量扫描股票"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_stock, symbol, thresholds): symbol 
                   for symbol in tickers}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            result = future.result()
            if result:
                results.append(result)
    
    return results


def send_telegram_notification(results, market='US'):
    """发送 Telegram 通知"""
    import urllib.request
    import urllib.parse
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials not configured")
        return False
    
    date = datetime.now().strftime('%Y-%m-%d')
    total = len(results)
    
    lines = [
        '🔵 *Baseline BLUE Scan*',
        f'📅 日期: {date}',
        f'📊 市场: {market}',
        f'🎯 信号: {total} 个',
        '',
        '📈 *Top 10:*'
    ]
    
    # 按 BLUE 值排序
    sorted_results = sorted(results, key=lambda x: x['blue_daily'], reverse=True)[:10]
    
    for i, r in enumerate(sorted_results, 1):
        day_b = r.get('blue_daily', 0)
        week_b = r.get('blue_weekly', 0)
        lines.append(f"{i}. `{r['symbol']}` ${r['price']:.2f} D:{day_b:.0f} W:{week_b:.0f}")
    
    message = '\n'.join(lines)
    
    try:
        url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown',
        }).encode()
        
        urllib.request.urlopen(url, data, timeout=10)
        print("✅ Telegram notification sent")
        return True
        
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def send_email_notification(results, market='US'):
    """发送 Email 通知"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    smtp_sender = os.getenv('SMTP_SENDER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    receivers = os.getenv('EMAIL_RECEIVERS', smtp_sender)
    
    if not smtp_sender or not smtp_password:
        print("⚠️ Email credentials not configured")
        return False
    
    date = datetime.now().strftime('%Y-%m-%d')
    subject = f"🔵 Baseline BLUE Scan - {market} - {date} - {len(results)} signals"
    
    # 构建邮件正文
    body = f"Baseline BLUE Signal Scan Results\n"
    body += f"Date: {date}\n"
    body += f"Market: {market}\n"
    body += f"Total Signals: {len(results)}\n\n"
    body += "=" * 60 + "\n"
    body += f"{'Symbol':<8} | {'Price':>10} | {'Turnover':>12} | {'BLUE':>8} | {'Days':>4}\n"
    body += "-" * 60 + "\n"
    
    sorted_results = sorted(results, key=lambda x: x['blue_daily'], reverse=True)
    
    for r in sorted_results:
        body += f"{r['symbol']:<8} | ${r['price']:>9.2f} | {r['turnover']:>10.0f}万 | {r['blue_daily']:>7.1f} | {r['blue_days']:>4}\n"
    
    body += "=" * 60 + "\n"
    
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = smtp_sender
        msg['To'] = receivers
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(smtp_sender, smtp_password)
            server.sendmail(smtp_sender, receivers.split(','), msg.as_string())
        
        print("✅ Email notification sent")
        return True
        
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False


def enrich_with_company_info(results):
    """从数据库补充公司信息"""
    if not results:
        return results
    
    symbols = [r['symbol'] for r in results]
    stock_info = get_stock_info_batch(symbols)
    
    # 查找缺失的信息并获取
    missing = [s for s in symbols if s not in stock_info]
    if missing:
        print(f"Fetching info for {len(missing)} new stocks...")
        for symbol in missing[:50]:  # 限制 API 调用
            try:
                info = get_ticker_details(symbol)
                if info:
                    upsert_stock_info(symbol, info.get('name', symbol), 
                                     info.get('sic_description', ''), market='US')
                    stock_info[symbol] = {'name': info.get('name', symbol)}
            except:
                pass
    
    # 补充信息
    for r in results:
        info = stock_info.get(r['symbol'], {})
        r['name'] = info.get('name', '')[:20]
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Baseline BLUE Signal Scanner (US)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of stocks to scan')
    parser.add_argument('--no-email', action='store_true', help='Disable email notification')
    parser.add_argument('--no-telegram', action='store_true', help='Disable telegram notification')
    args = parser.parse_args()
    
    # 初始化数据库
    init_db()
    
    # 设置阈值 - 与原始 scan_blue_baseline.py 完全一致
    thresholds = {
        'day_blue': 100,        # 日线 BLUE > 100
        'week_blue': 130,       # 周线 BLUE > 130
        'day_blue_count': 3,    # 最近 6 天中至少 3 天有日线信号
        'week_blue_count': 2,   # 最近 5 周中至少 2 周有周线信号
        'day_lookback': 6,      # 日线回看 6 天
        'week_lookback': 5,     # 周线回看 5 周
    }
    
    print("\n" + "=" * 60)
    print("🔵 Baseline BLUE Signal Scanner (US Market)")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 获取股票列表
    print("\n📋 Fetching stock list...")
    tickers = get_all_us_tickers()
    print(f"Found {len(tickers)} stocks")
    
    if args.limit > 0:
        tickers = tickers[:args.limit]
        print(f"Limited to {len(tickers)} stocks")
    
    # 扫描股票
    print("\n🔍 Scanning stocks...")
    results = scan_stocks(tickers, thresholds)
    
    if results:
        # 补充公司信息
        print("\n📊 Enriching with company info...")
        results = enrich_with_company_info(results)
        
        # 按 BLUE 排序
        results = sorted(results, key=lambda x: x['blue_daily'], reverse=True)
        
        # 打印结果
        print(f"\n✅ Found {len(results)} stocks with BLUE signals:")
        print("=" * 70)
        print(f"{'Symbol':<8} | {'Name':<20} | {'Price':>8} | {'Turnover':>10} | {'BLUE':>6}")
        print("-" * 70)
        
        for r in results[:20]:
            print(f"{r['symbol']:<8} | {r.get('name', ''):<20} | ${r['price']:>7.2f} | {r['turnover']:>8.0f}万 | {r['blue_daily']:>5.0f}")
        
        if len(results) > 20:
            print(f"... and {len(results) - 20} more")
        print("=" * 70)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"baseline_scan_US_{timestamp}.csv"
        pd.DataFrame(results).to_csv(os.path.join(parent_dir, filename), index=False)
        print(f"\n💾 Results saved to {filename}")
        
        # 发送通知
        if not args.no_telegram:
            send_telegram_notification(results, 'US')
        if not args.no_email:
            send_email_notification(results, 'US')
    else:
        print("\n❌ No signals found")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
