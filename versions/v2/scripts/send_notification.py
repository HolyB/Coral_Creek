#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一通知脚本 - 发送 Telegram 和 Email 通知
"""
import os
import sys
import json
import smtplib
import urllib.request
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 加载 .env 文件（本地开发用）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(parent_dir, '.env'))
except ImportError:
    pass  # dotenv not required in GitHub Actions



def load_scan_summary():
    """加载扫描摘要"""
    summary_files = ['scan_summary.json', 'scan_summary_cn.json']
    
    for filename in summary_files:
        filepath = os.path.join(parent_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    return None


def send_telegram(summary):
    """发送 Telegram 通知"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials not configured, skipping")
        return False
    
    date = summary.get('date', 'Unknown')
    market = summary.get('market', 'US')
    total = summary.get('total_signals', 0)
    top = summary.get('top_signals', [])[:10]
    
    market_name = "美股" if market == "US" else "A股"
    
    lines = [
        '🌊 *Coral Creek 每日扫描*',
        f'📅 日期: {date}',
        f'📊 市场: {market_name}',
        f'🎯 信号: {total} 个',
        '',
        '📈 *Top 10:*'
    ]
    
    for i, s in enumerate(top, 1):
        name = s.get('name', '')[:8] if s.get('name') else ''
        symbol = s.get('symbol', 'N/A')
        price = s.get('price', 0)
        day_blue = s.get('day_blue', 0)
        lines.append(f'{i}. `{symbol}` {name} ${price:.2f} BLUE:{day_blue:.0f}')
    
    lines.append('')
    lines.append('[查看详情](https://coral-creek-park-way.onrender.com)')
    
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
        print("✅ Telegram notification sent")
        return True
        
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def send_email(summary):
    """发送 Email 通知"""
    smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_sender = os.getenv('SMTP_SENDER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    receivers_str = os.getenv('EMAIL_RECEIVERS', '')
    
    if not smtp_sender or not smtp_password:
        print("⚠️ Email credentials not configured, skipping")
        return False
    
    receivers = [r.strip() for r in receivers_str.split(',') if r.strip()]
    if not receivers:
        print("⚠️ No email receivers configured, skipping")
        return False
    
    date = summary.get('date', 'Unknown')
    market = summary.get('market', 'US')
    total = summary.get('total_signals', 0)
    top = summary.get('top_signals', [])[:15]
    
    market_name = "美股 (US)" if market == 'US' else "A股 (CN)"
    
    # 生成表格行
    rows = ""
    for i, s in enumerate(top, 1):
        symbol = s.get('symbol', 'N/A')
        name = (s.get('name', '') or '')[:20]
        price = s.get('price', 0)
        day_blue = s.get('day_blue', 0)
        week_blue = s.get('week_blue', 0)
        
        color_day = '#4CAF50' if day_blue > 100 else '#666'
        color_week = '#2196F3' if week_blue > 100 else '#666'
        
        rows += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">{i}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{symbol}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">{name}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">${price:.2f}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; color: {color_day};">{day_blue:.1f}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; color: {color_week};">{week_blue:.1f}</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px;">
        <div style="max-width: 700px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 24px;">🌊 Coral Creek 每日扫描报告</h1>
                <p style="margin: 10px 0 0; opacity: 0.9;">{date}</p>
            </div>
            
            <div style="display: flex; justify-content: space-around; padding: 20px; background: #fafafa;">
                <div style="text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #333;">{total}</div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">信号数量</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: bold; color: #333;">{market_name}</div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">市场</div>
                </div>
            </div>
            
            <div style="padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #333;">📈 Top 信号列表</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f0f0f0;">
                            <th style="padding: 10px 8px; text-align: left; font-size: 12px; color: #666;">#</th>
                            <th style="padding: 10px 8px; text-align: left; font-size: 12px; color: #666;">代码</th>
                            <th style="padding: 10px 8px; text-align: left; font-size: 12px; color: #666;">名称</th>
                            <th style="padding: 10px 8px; text-align: right; font-size: 12px; color: #666;">价格</th>
                            <th style="padding: 10px 8px; text-align: right; font-size: 12px; color: #666;">Day BLUE</th>
                            <th style="padding: 10px 8px; text-align: right; font-size: 12px; color: #666;">Week BLUE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows if rows else '<tr><td colspan="6" style="padding: 20px; text-align: center; color: #999;">暂无信号</td></tr>'}
                    </tbody>
                </table>
            </div>
            
            <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
                <a href="https://coral-creek-park-way.onrender.com" style="color: #667eea;">查看完整报告</a>
                <br><br>
                Coral Creek V2.0 - 智能量化系统
            </div>
        </div>
    </body>
    </html>
    """
    
    subject = f"🌊 Coral Creek 扫描报告 - {date} ({market}) - {total} 个信号"
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_sender
        msg['To'] = ', '.join(receivers)
        
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_sender, smtp_password)
            server.sendmail(smtp_sender, receivers, msg.as_string())
        
        print(f"✅ Email sent to {', '.join(receivers)}")
        return True
        
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False


def main():
    print("📧 Loading scan summary...")
    summary = load_scan_summary()
    
    if not summary:
        print("⚠️ No scan summary found, skipping notifications")
        return
    
    date = summary.get('date', 'Unknown')
    market = summary.get('market', 'US')
    total = summary.get('total_signals', 0)
    
    print(f"📊 Date: {date}, Market: {market}, Signals: {total}")
    
    # 发送 Telegram
    send_telegram(summary)
    
    # 发送 Email
    send_email(summary)
    
    print("\n✅ Notification process completed")


if __name__ == "__main__":
    main()
