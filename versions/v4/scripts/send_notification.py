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
    """发送 Telegram 通知 - 增强版"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials not configured, skipping")
        return False
    
    date = summary.get('date', 'Unknown')
    market = summary.get('market', 'US')
    total = summary.get('total_signals', 0)
    top = summary.get('top_signals', [])[:10]
    
    market_name = "🇺🇸 美股" if market == "US" else "🇨🇳 A股"
    
    # 构建增强版消息
    lines = [
        '━━━━━━━━━━━━━━━━━━',
        '🦅 *Coral Creek 每日扫描*',
        '━━━━━━━━━━━━━━━━━━',
        f'📅 *日期:* `{date}`',
        f'📊 *市场:* {market_name}',
        f'🎯 *信号:* *{total}* 个',
        '',
        '━━ 📈 *Top 10 信号* ━━'
    ]
    
    # 按 BLUE 值分类
    strong_signals = [s for s in top if s.get('day_blue', 0) > 150]
    normal_signals = [s for s in top if 100 <= s.get('day_blue', 0) <= 150]
    weak_signals = [s for s in top if s.get('day_blue', 0) < 100]
    
    def format_signal(s, idx):
        name = s.get('name', '')[:8] if s.get('name') else ''
        name = name.replace('*', '').replace('_', '').replace('`', '').replace('[', '').replace(']', '')
        symbol = s.get('symbol', 'N/A')
        price = s.get('price', 0)
        day_blue = s.get('day_blue', 0)
        week_blue = s.get('week_blue', 0)
        chip = s.get('chip_pattern', '')
        
        # 信号强度指示
        if day_blue > 150:
            strength = '🔥'  # 强烈
        elif day_blue > 100:
            strength = '✅'  # 正常
        else:
            strength = '📍'  # 观望
        
        # 周线确认
        weekly_confirm = '⬆️' if week_blue > 80 else ''
        
        # 筹码形态
        chip_str = f' {chip}' if chip else ''
        
        return f'{strength} `{symbol}` {name} *${price:.2f}*{chip_str} D:{day_blue:.0f}{weekly_confirm}'
    
    for i, s in enumerate(top, 1):
        lines.append(format_signal(s, i))
    
    # === Top 3 详细分析 (参考 daily_stock_analysis 格式) ===
    if top[:3]:
        lines.append('')
        lines.append('━━ 📊 *Top 3 详情* ━━')
        
        for s in top[:3]:
            symbol = s.get('symbol', 'N/A')
            name = s.get('name', '')[:10].replace('*', '').replace('_', '')
            price = float(s.get('price', 0))
            day_blue = s.get('day_blue', 0)
            chip = s.get('chip_pattern', '')
            
            # 信号强度
            if day_blue > 150:
                signal_emoji = '🔥'
                signal_text = '强烈买入'
            elif day_blue > 100:
                signal_emoji = '✅'
                signal_text = '买入'
            else:
                signal_emoji = '📍'
                signal_text = '观望'
            
            # 计算狙击点位 (基于 price)
            entry_price = price * 0.98  # MA5 附近
            stop_loss = price * 0.93    # -7%
            target_price = price * 1.15 # +15%
            
            lines.append('')
            lines.append(f'{signal_emoji} *{signal_text}* | `{symbol}` {name}')
            lines.append(f'📌 BLUE={day_blue:.0f}{" " + chip if chip else ""}')
            lines.append(f'🎯买点:${entry_price:.2f} | 🛑止损:${stop_loss:.2f} | 🎊目标:${target_price:.2f}')
            
            # 简化的操作建议
            if day_blue > 100:
                lines.append(f'🆕 空仓: 当前价${price:.2f}可小仓介入')
            else:
                lines.append(f'🆕 空仓: 观望等待更佳买点')
    
    # 市场概览
    lines.append('')
    lines.append('━━ 📊 *信号概览* ━━')
    lines.append(f'🔥 强烈信号 (BLUE>150): *{len(strong_signals)}* 个')
    lines.append(f'✅ 标准信号 (100-150): *{len(normal_signals)}* 个')
    lines.append(f'📍 观望信号 (<100): *{len(weak_signals)}* 个')
    
    # 操作建议
    lines.append('')
    if len(strong_signals) >= 3:
        lines.append('💡 *建议:* 市场超卖，可择机低吸')
    elif len(strong_signals) >= 1:
        lines.append('💡 *建议:* 关注强势信号，等待确认')
    else:
        lines.append('💡 *建议:* 信号偏弱，继续观望')
    
    lines.append('')
    lines.append('[📱 查看详情](https://facaila.streamlit.app/)')
    lines.append('━━━━━━━━━━━━━━━━━━')
    
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


def send_telegram_detailed(summary):
    """
    发送第二份 Telegram 通知 - 详细版决策仪表盘
    
    参考 daily_stock_analysis 的格式，为 Top 5 个信号生成详细分析
    """
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        return False
    
    date = summary.get('date', 'Unknown')
    market = summary.get('market', 'US')
    top = summary.get('top_signals', [])[:5]
    
    if not top:
        return False
    
    market_name = "🇺🇸 美股" if market == "US" else "🇨🇳 A股"
    
    lines = [
        '━━━━━━━━━━━━━━━━━━',
        '🎯 *决策仪表盘* | 详细版',
        '━━━━━━━━━━━━━━━━━━',
        f'📅 {date} | {market_name}',
        '',
    ]
    
    for s in top:
        symbol = s.get('symbol', 'N/A')
        name = s.get('name', '')[:12].replace('*', '').replace('_', '').replace('`', '')
        price = float(s.get('price', 0))
        day_blue = s.get('day_blue', 0)
        week_blue = s.get('week_blue', 0)
        chip = s.get('chip_pattern', '')
        
        # 信号判定
        if day_blue > 150:
            signal_emoji = '🔥'
            signal_text = '强烈买入'
            advice_no_pos = f'可在${price*0.98:.2f}附近建仓'
            advice_has_pos = '继续持有，可适当加仓'
        elif day_blue > 100:
            signal_emoji = '✅'
            signal_text = '买入'
            advice_no_pos = f'可小仓介入，止损${price*0.93:.2f}'
            advice_has_pos = '持有观察'
        elif day_blue > 80:
            signal_emoji = '📍'
            signal_text = '观望'
            advice_no_pos = '等待更佳买点'
            advice_has_pos = '继续持有'
        else:
            signal_emoji = '⚪'
            signal_text = '弱势'
            advice_no_pos = '暂不介入'
            advice_has_pos = '考虑减仓'
        
        # 周线确认
        weekly_status = '✅周线共振' if week_blue > 80 else '⚠️周线未确认'
        
        # 狙击点位
        entry = price * 0.98
        stop_loss = price * 0.93
        target = price * 1.15
        
        lines.extend([
            f'━━ {signal_emoji} *{signal_text}* ━━',
            f'`{symbol}` {name}',
            f'💰 现价: *${price:.2f}* | BLUE: {day_blue:.0f}',
            f'📊 {weekly_status}{" " + chip if chip else ""}',
            '',
            '*📍 狙击点位*',
            f'🎯 买点: ${entry:.2f}',
            f'🛑 止损: ${stop_loss:.2f} (-7%)',
            f'🎊 目标: ${target:.2f} (+15%)',
            '',
            '*💼 操作建议*',
            f'🆕 空仓者: {advice_no_pos}',
            f'💰 持仓者: {advice_has_pos}',
            '',
        ])
    
    lines.extend([
        '━━━━━━━━━━━━━━━━━━',
        '⚠️ *本报告仅供参考*',
        '[📱 查看完整分析](https://facaila.streamlit.app/)',
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
        print("✅ Telegram detailed notification sent")
        return True
        
    except Exception as e:
        print(f"❌ Telegram detailed error: {e}")
        return False


def send_email(summary):
    """发送 Email 通知 - 使用更健壮的连接处理"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    smtp_host = os.getenv('SMTP_HOST') or 'smtp.gmail.com'
    smtp_port = int(os.getenv('SMTP_PORT') or 587)
    smtp_sender = os.getenv('SMTP_SENDER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    receivers_str = os.getenv('EMAIL_RECEIVERS', '')
    
    print(f"📧 SMTP Config: host={smtp_host}, port={smtp_port}, sender={smtp_sender}")
    
    if not smtp_sender or not smtp_password:
        print("⚠️ Email credentials not configured, skipping")
        return False
    
    receivers = [r.strip() for r in receivers_str.split(',') if r.strip()]
    if not receivers:
        print("⚠️ No email receivers configured, skipping")
        return False
    
    print(f"📧 Receivers: {receivers}")
    
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
        chip_pattern = s.get('chip_pattern', '')  # 🔥 or 📍
        
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
            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; font-size: 16px;">{chip_pattern}</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px;">
        <div style="max-width: 700px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 24px;">🦅 Coral Creek 每日扫描报告</h1>
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
                            <th style="padding: 10px 8px; text-align: center; font-size: 12px; color: #666;">筹码</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows if rows else '<tr><td colspan="6" style="padding: 20px; text-align: center; color: #999;">暂无信号</td></tr>'}
                    </tbody>
                </table>
            </div>
            
            <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
                <a href="https://facaila.streamlit.app/" style="color: #667eea;">查看完整报告</a>
                <br><br>
                Coral Creek V2.0 - 智能量化系统
            </div>
        </div>
    </body>
    </html>
    """
    
    subject = f"🦅 Coral Creek 扫描报告 - {date} ({market}) - {total} 个信号"
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_sender
        msg['To'] = ', '.join(receivers)
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        print(f"📧 Connecting to {smtp_host}:{smtp_port}...")
        
        # 使用 with 语句确保连接正确关闭
        with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
            print("📧 Connected, starting TLS...")
            server.starttls()
            print("📧 TLS started, logging in...")
            server.login(smtp_sender, smtp_password)
            print("📧 Logged in, sending email...")
            server.sendmail(smtp_sender, receivers, msg.as_string())
        
        print(f"✅ Email sent to {', '.join(receivers)}")
        return True
        
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {type(e).__name__}: {e}")
        return False
    except Exception as e:
        print(f"❌ Email error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
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
    
    # 发送 Telegram (简报)
    send_telegram(summary)
    
    # 发送 Telegram (详细版 - 决策仪表盘)
    import time
    time.sleep(1)  # 避免 rate limit
    send_telegram_detailed(summary)
    
    # 发送 Email
    send_email(summary)
    
    print("\n✅ Notification process completed")


if __name__ == "__main__":
    main()
