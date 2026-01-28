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

from db.database import get_portfolio
from data_fetcher import get_us_stock_data, get_cn_stock_data


# ==================== 配置 ====================

ALERT_THRESHOLDS = {
    'stop_loss': -0.07,      # 止损线 -7%
    'take_profit': 0.15,     # 止盈线 +15%
    'daily_surge': 0.05,     # 日涨幅预警 +5%
    'daily_plunge': -0.05,   # 日跌幅预警 -5%
}


# ==================== 数据获取 ====================

def get_current_price(symbol: str, market: str = 'US') -> dict:
    """
    获取股票当前价格和日涨跌幅
    
    Returns:
        {'price': 现价, 'change_pct': 涨跌幅%}
    """
    try:
        if market == 'US':
            df = get_us_stock_data(symbol, days=5)
        else:
            df = get_cn_stock_data(symbol, days=5)
        
        if df is None or df.empty:
            return None
        
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_pct = (current_price - prev_close) / prev_close
        
        return {
            'price': current_price,
            'prev_close': prev_close,
            'change_pct': change_pct
        }
    except Exception as e:
        print(f"获取 {symbol} 价格失败: {e}")
        return None


# ==================== 预警逻辑 ====================

def check_alerts(stock: dict, price_data: dict) -> list:
    """
    检查股票是否触发预警
    
    Args:
        stock: 持仓信息 {'symbol': 'AAPL', 'entry_price': 150, 'market': 'US'}
        price_data: 价格数据 {'price': 185, 'change_pct': 0.02}
    
    Returns:
        预警列表 [{'type': 'stop_loss', 'message': '...'}]
    """
    alerts = []
    
    symbol = stock['symbol']
    entry_price = float(stock.get('entry_price', 0))
    current_price = price_data['price']
    change_pct = price_data['change_pct']
    
    # 计算相对入场价涨跌幅
    if entry_price > 0:
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = 0
    
    # 1. 止损预警
    if pnl_pct <= ALERT_THRESHOLDS['stop_loss']:
        alerts.append({
            'type': 'stop_loss',
            'level': '🚨',
            'symbol': symbol,
            'message': f"触发止损! 亏损 {pnl_pct*100:.1f}%",
            'price': current_price,
            'entry_price': entry_price,
            'pnl_pct': pnl_pct
        })
    
    # 2. 止盈预警
    elif pnl_pct >= ALERT_THRESHOLDS['take_profit']:
        alerts.append({
            'type': 'take_profit',
            'level': '🎉',
            'symbol': symbol,
            'message': f"达到止盈! 盈利 +{pnl_pct*100:.1f}%",
            'price': current_price,
            'entry_price': entry_price,
            'pnl_pct': pnl_pct
        })
    
    # 3. 日内大涨预警
    if change_pct >= ALERT_THRESHOLDS['daily_surge']:
        alerts.append({
            'type': 'daily_surge',
            'level': '📈',
            'symbol': symbol,
            'message': f"今日大涨 +{change_pct*100:.1f}%",
            'price': current_price,
            'change_pct': change_pct
        })
    
    # 4. 日内大跌预警
    elif change_pct <= ALERT_THRESHOLDS['daily_plunge']:
        alerts.append({
            'type': 'daily_plunge',
            'level': '📉',
            'symbol': symbol,
            'message': f"今日大跌 {change_pct*100:.1f}%",
            'price': current_price,
            'change_pct': change_pct
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
        '🚨 *持仓预警* | Coral Creek',
        '━━━━━━━━━━━━━━━━━━',
        f'⏰ 时间: {now}',
        ''
    ]
    
    for alert in alerts:
        level = alert['level']
        symbol = alert['symbol']
        msg = alert['message']
        price = alert.get('price', 0)
        entry = alert.get('entry_price', 0)
        
        lines.append(f'{level} `{symbol}` *{msg}*')
        if entry > 0:
            lines.append(f'   💰 现价: ${price:.2f} | 入场: ${entry:.2f}')
        else:
            lines.append(f'   💰 现价: ${price:.2f}')
        lines.append('')
    
    # 建议
    stop_loss_alerts = [a for a in alerts if a['type'] == 'stop_loss']
    take_profit_alerts = [a for a in alerts if a['type'] == 'take_profit']
    
    if stop_loss_alerts:
        lines.append('💡 *建议:* 考虑止损离场')
    elif take_profit_alerts:
        lines.append('💡 *建议:* 考虑减仓锁定利润')
    
    lines.extend([
        '',
        '[📱 查看详情](https://coral-creek-park-way.onrender.com)',
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

def monitor_portfolio():
    """监控持仓组合"""
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
        
        # 获取当前价格
        price_data = get_current_price(symbol, market)
        
        if not price_data:
            print(f"   ⚠️ 无法获取价格")
            continue
        
        current_price = price_data['price']
        change_pct = price_data['change_pct']
        
        print(f"   💰 现价: ${current_price:.2f} | 今日: {change_pct*100:+.1f}%")
        
        # 检查预警
        alerts = check_alerts(stock, price_data)
        
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
