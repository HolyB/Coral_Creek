#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
盘中实时监控 - 监控持仓股票价格变化，触发预警
"""
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime
import time

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(parent_dir, '.env'))
except ImportError:
    pass

import pandas as pd
import numpy as np
from db.database import get_portfolio
from data_fetcher import get_us_stock_data, get_cn_stock_data, get_stock_data
from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series

# ==================== 配置 ====================

ALERT_THRESHOLDS = {
    'stop_loss': -0.07,      # 止损线 -7%
    'take_profit': 0.15,     # 止盈线 +15%
    'daily_surge': 0.05,     # 日涨幅预警 +5%
    'daily_plunge': -0.05,   # 日跌幅预警 -5%
    'blue_breakout': 100,    # BLUE 突破 100
}


# ==================== 数据获取 ====================

def get_intraday_data(symbol: str, market: str = 'US', days: int = 65) -> dict:
    """
    获取股票数据并计算指标
    
    Args:
        days: 获取65天数据以足量计算指标
    """
    try:
        # 使用统一函数获取数据
        df = get_stock_data(symbol, market=market, days=days)
        
        if df is None or df.empty or len(df) < 30:
            return None
        
        # 获取最新价格信息
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_pct = (current_price - prev_close) / prev_close
        
        # 计算 BLUE 信号
        blue_series = calculate_blue_signal_series(
            df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        )
        
        current_blue = blue_series[-1]
        prev_blue = blue_series[-2] if len(blue_series) > 1 else 0
        
        return {
            'price': current_price,
            'prev_close': prev_close,
            'change_pct': change_pct,
            'blue': current_blue,
            'prev_blue': prev_blue,
            'market': market,
            'df': df  # 保留以备后续使用
        }
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None


# ==================== 预警逻辑 ====================

def check_alerts(stock: dict, data: dict) -> list:
    """
    检查股票是否触发预警 (价格 + 技术指标)
    """
    alerts = []
    
    symbol = stock['symbol']
    entry_price = float(stock.get('entry_price', 0))
    current_price = data['price']
    change_pct = data['change_pct']
    current_blue = data['blue']
    prev_blue = data['prev_blue']
    
    # --- 1. 价格预警 ---
    
    # 计算持仓盈亏
    if entry_price > 0:
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 止损预警
        if pnl_pct <= ALERT_THRESHOLDS['stop_loss']:
            alerts.append({
                'type': 'stop_loss',
                'level': '🚨',
                'symbol': symbol,
                'message': f"触发止损! 亏损 {pnl_pct*100:.1f}%",
                'footer': f"入场: ${entry_price:.2f} | 现价: ${current_price:.2f}"
            })
        
        # 止盈预警
        elif pnl_pct >= ALERT_THRESHOLDS['take_profit']:
            alerts.append({
                'type': 'take_profit',
                'level': '🎉',
                'symbol': symbol,
                'message': f"达到止盈! 盈利 +{pnl_pct*100:.1f}%",
                'footer': f"入场: ${entry_price:.2f} | 现价: ${current_price:.2f}"
            })
    
    # 日涨跌幅预警
    if change_pct >= ALERT_THRESHOLDS['daily_surge']:
        alerts.append({
            'type': 'daily_surge',
            'level': '🚀',
            'symbol': symbol,
            'message': f"今日大涨 +{change_pct*100:.1f}%",
            'footer': f"现价: ${current_price:.2f} | BLUE: {current_blue:.0f}"
        })
    elif change_pct <= ALERT_THRESHOLDS['daily_plunge']:
        alerts.append({
            'type': 'daily_plunge',
            'level': '📉',
            'symbol': symbol,
            'message': f"今日大跌 {change_pct*100:.1f}%",
            'footer': f"现价: ${current_price:.2f}"
        })
        
    # --- 2. 技术指标预警 (BLUE) ---
    
    # 场景 A: BLUE 突破 100 (强势爆发)
    if prev_blue < 100 and current_blue >= 100:
        alerts.append({
            'type': 'blue_breakout',
            'level': '🔥',
            'symbol': symbol,
            'message': f"BLUE 爆发! 突破 100 (现值 {current_blue:.0f})",
            'footer': "进入强势拉升区，重点关注"
        })
        
    # 场景 B: BLUE 趋势启动 (由负转正)
    elif prev_blue < 0 and current_blue >= 0:
        alerts.append({
            'type': 'blue_start',
            'level': '✅',
            'symbol': symbol,
            'message': f"趋势启动! BLUE 翻红 (现值 {current_blue:.0f})",
            'footer': "趋势可能反转向上"
        })
    
    # 场景 C: 高位死叉 (风险提示) - BLUE 从高位(>150)下跌
    elif prev_blue > 150 and current_blue < 150:
         alerts.append({
            'type': 'blue_drop',
            'level': '⚠️',
            'symbol': symbol,
            'message': f"高位回落! BLUE 跌破 150",
            'footer': "注意回调风险"
        })

    return alerts


# ==================== 通知发送 ====================

def send_alert_telegram(alerts: list) -> bool:
    """发送预警到 Telegram"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ Telegram 未配置")
        return False
    
    if not alerts:
        return True
    
    # 构建消息
    now = datetime.now().strftime('%H:%M')
    
    lines = [
        '━━━━━━━━━━━━━━━━━━',
        '🚨 *Coral Creek 实时监控*',
        f'⏰ {now}',
        '━━━━━━━━━━━━━━━━━━',
        ''
    ]
    
    # 按重要性排序 (止损/止盈/突破 最重要)
    priority = {'stop_loss': 0, 'take_profit': 1, 'blue_breakout': 2, 'blue_start': 3, 'daily_surge': 4, 'daily_plunge': 5, 'blue_drop': 6}
    alerts.sort(key=lambda x: priority.get(x['type'], 99))
    
    for alert in alerts:
        level = alert['level']
        symbol = alert['symbol']
        msg = alert['message']
        footer = alert.get('footer', '')
        
        lines.append(f'{level} `{symbol}` *{msg}*')
        if footer:
            lines.append(f'   _{footer}_')
        lines.append('')
    
    lines.extend([
        '[📱 打开监控面板](https://coralcreek.streamlit.app/)',
        '━━━━━━━━━━━━━━━━━━'
    ])
    
    message = '\n'.join(lines)
    
    try:
        url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': 'true'
        }).encode()
        
        urllib.request.urlopen(url, data, timeout=10)
        print("✅ 预警已发送到 Telegram")
        return True
        
    except Exception as e:
        print(f"❌ Telegram 发送失败: {e}")
        return False


# ==================== 主流程 ====================

def monitor_portfolio(market='US', **_kwargs):
    """监控持仓组合（兼容旧调度器传入 market 参数）"""
    print(f"\n{'='*50}")
    print(f"📱 盘中监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    # 获取持仓
    portfolio = get_portfolio(status='holding')
    
    if not portfolio:
        print("📋 当前无持仓，跳过监控")
        return
    
    print(f"📋 持仓数量: {len(portfolio)}")
    
    all_alerts = []
    
    for stock in portfolio:
        symbol = stock['symbol']
        market = stock.get('market', 'US')
        entry_price = stock.get('entry_price', 0)
        
        print(f"\n检查 {symbol} (入场价: ${entry_price:.2f})...")
        
        # 获取当前数据
        data = get_intraday_data(symbol, market, days=65)
        
        if not data:
            print(f"   ⚠️ 无法获取数据")
            continue
        
        current_price = data['price']
        change_pct = data['change_pct']
        current_blue = data['blue']
        
        print(f"   💰 现价: ${current_price:.2f} | 今日: {change_pct*100:+.1f}% | BLUE: {current_blue:.0f}")
        
        # 检查预警
        alerts = check_alerts(stock, data)
        
        if alerts:
            for alert in alerts:
                print(f"   {alert['level']} {alert['message']}")
            all_alerts.extend(alerts)
        else:
            print(f"   ✅ 正常")
        
        # 避免 API 限流
        time.sleep(0.5)
    
    # 发送预警
    if all_alerts:
        print(f"\n🚨 触发 {len(all_alerts)} 个预警")
        send_alert_telegram(all_alerts)
    else:
        print(f"\n✅ 所有持仓正常，无预警")


if __name__ == "__main__":
    monitor_portfolio()
