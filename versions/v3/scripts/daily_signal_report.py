#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日信号追踪报告
1. 获取历史信号记录
2. 计算每个信号的后续表现 (1天, 3天, 5天, 10天)
3. 生成 HTML 报告并发送邮件
"""
import os
import sys
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def get_historical_signals(days: int = 30, market: str = 'US') -> List[Dict]:
    """获取历史扫描信号"""
    from db.database import query_scan_results, get_scanned_dates
    
    dates = get_scanned_dates(market=market)
    if not dates:
        return []
    
    all_signals = []
    
    # 获取最近 N 天的扫描结果
    for date in dates[:days]:
        results = query_scan_results(scan_date=date, market=market, limit=100)
        for r in results:
            # 只保留强信号
            blue = r.get('blue_daily', 0) or 0
            if blue >= 130:  # BLUE >= 130 才算有效信号
                all_signals.append({
                    'symbol': r['symbol'],
                    'signal_date': date,
                    'signal_price': r.get('price', 0),
                    'blue': blue,
                    'adx': r.get('adx', 0) or 0,
                    'is_heima': r.get('is_heima', False),
                    'is_juedi': r.get('is_juedi', False),
                    'company_name': r.get('company_name', ''),
                    'industry': r.get('industry', '')
                })
    
    return all_signals


def calculate_signal_returns(signals: List[Dict], market: str = 'US') -> List[Dict]:
    """计算每个信号的后续收益"""
    if not signals:
        return []
    
    # 按股票分组获取价格
    from data_fetcher import get_stock_data
    
    results = []
    symbol_cache = {}  # 缓存已获取的价格数据
    
    for sig in signals:
        symbol = sig['symbol']
        signal_date = sig['signal_date']
        signal_price = sig['signal_price']
        
        if not signal_price or signal_price <= 0:
            continue
        
        # 获取价格数据 (使用缓存)
        if symbol not in symbol_cache:
            try:
                df = get_stock_data(symbol, market=market, days=60)
                if df is not None and not df.empty:
                    symbol_cache[symbol] = df
                else:
                    symbol_cache[symbol] = None
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                symbol_cache[symbol] = None
        
        df = symbol_cache[symbol]
        if df is None:
            continue
        
        # 转换日期
        try:
            sig_dt = pd.to_datetime(signal_date)
        except:
            continue
        
        # 计算 D+1, D+3, D+5, D+10 收益
        returns = {'D1': None, 'D3': None, 'D5': None, 'D10': None}
        
        # 获取信号日期后的数据
        future_df = df[df.index > sig_dt]
        
        if len(future_df) >= 1:
            returns['D1'] = (future_df.iloc[0]['Close'] / signal_price - 1) * 100
        if len(future_df) >= 3:
            returns['D3'] = (future_df.iloc[2]['Close'] / signal_price - 1) * 100
        if len(future_df) >= 5:
            returns['D5'] = (future_df.iloc[4]['Close'] / signal_price - 1) * 100
        if len(future_df) >= 10:
            returns['D10'] = (future_df.iloc[9]['Close'] / signal_price - 1) * 100
        
        # 获取当前价格
        current_price = df.iloc[-1]['Close'] if len(df) > 0 else 0
        current_return = (current_price / signal_price - 1) * 100 if signal_price > 0 else 0
        
        results.append({
            **sig,
            'current_price': current_price,
            'current_return': current_return,
            'D1': returns['D1'],
            'D3': returns['D3'],
            'D5': returns['D5'],
            'D10': returns['D10'],
            'is_winner': current_return > 0
        })
    
    return results


def analyze_signal_stats(results: List[Dict]) -> Dict:
    """分析信号统计"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    # 总体统计
    total_signals = len(df)
    winners = len(df[df['is_winner'] == True])
    win_rate = winners / total_signals * 100 if total_signals > 0 else 0
    
    avg_return = df['current_return'].mean()
    
    # 按信号类型统计
    high_blue = df[df['blue'] >= 160]
    medium_blue = df[(df['blue'] >= 130) & (df['blue'] < 160)]
    
    # D1, D3, D5, D10 平均收益
    d1_avg = df['D1'].dropna().mean() if len(df['D1'].dropna()) > 0 else 0
    d3_avg = df['D3'].dropna().mean() if len(df['D3'].dropna()) > 0 else 0
    d5_avg = df['D5'].dropna().mean() if len(df['D5'].dropna()) > 0 else 0
    d10_avg = df['D10'].dropna().mean() if len(df['D10'].dropna()) > 0 else 0
    
    # 最佳和最差
    best_signal = df.loc[df['current_return'].idxmax()] if len(df) > 0 else None
    worst_signal = df.loc[df['current_return'].idxmin()] if len(df) > 0 else None
    
    return {
        'total_signals': total_signals,
        'winners': winners,
        'losers': total_signals - winners,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'high_blue_count': len(high_blue),
        'high_blue_win_rate': len(high_blue[high_blue['is_winner']]) / len(high_blue) * 100 if len(high_blue) > 0 else 0,
        'd1_avg': d1_avg,
        'd3_avg': d3_avg,
        'd5_avg': d5_avg,
        'd10_avg': d10_avg,
        'best_signal': best_signal.to_dict() if best_signal is not None else None,
        'worst_signal': worst_signal.to_dict() if worst_signal is not None else None
    }


def generate_html_report(results: List[Dict], stats: Dict, market: str) -> str:
    """生成 HTML 邮件报告"""
    market_name = "🇺🇸 美股" if market == 'US' else "🇨🇳 A股"
    price_sym = "$" if market == 'US' else "¥"
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 按日期排序
    results_sorted = sorted(results, key=lambda x: x['signal_date'], reverse=True)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 10px 0 0; opacity: 0.9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .section {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .section h2 {{ margin: 0 0 15px; font-size: 18px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f8f9fa; padding: 12px 8px; text-align: left; font-weight: 600; color: #333; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .signal-date {{ color: #666; font-size: 12px; }}
        .return-badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
        .return-positive {{ background: #dcfce7; color: #166534; }}
        .return-negative {{ background: #fee2e2; color: #991b1b; }}
        .signal-tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-right: 3px; }}
        .tag-blue {{ background: #dbeafe; color: #1e40af; }}
        .tag-heima {{ background: #fef3c7; color: #92400e; }}
        .tag-juedi {{ background: #f3e8ff; color: #7c3aed; }}
        .footer {{ text-align: center; color: #666; font-size: 12px; padding: 20px; }}
        .chart-placeholder {{ background: #f0f0f0; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #999; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 {market_name} 信号追踪日报</h1>
        <p>📅 {today} | 过去30天信号表现分析</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats.get('total_signals', 0)}</div>
            <div class="stat-label">总信号数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'positive' if stats.get('win_rate', 0) >= 50 else 'negative'}">{stats.get('win_rate', 0):.1f}%</div>
            <div class="stat-label">胜率</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'positive' if stats.get('avg_return', 0) >= 0 else 'negative'}">{stats.get('avg_return', 0):+.1f}%</div>
            <div class="stat-label">平均收益</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get('winners', 0)}/{stats.get('losers', 0)}</div>
            <div class="stat-label">盈/亏</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 各周期平均收益</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value {'positive' if stats.get('d1_avg', 0) >= 0 else 'negative'}">{stats.get('d1_avg', 0):+.2f}%</div>
                <div class="stat-label">D+1</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if stats.get('d3_avg', 0) >= 0 else 'negative'}">{stats.get('d3_avg', 0):+.2f}%</div>
                <div class="stat-label">D+3</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if stats.get('d5_avg', 0) >= 0 else 'negative'}">{stats.get('d5_avg', 0):+.2f}%</div>
                <div class="stat-label">D+5</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if stats.get('d10_avg', 0) >= 0 else 'negative'}">{stats.get('d10_avg', 0):+.2f}%</div>
                <div class="stat-label">D+10</div>
            </div>
        </div>
    </div>
"""
    
    # 最佳和最差信号
    if stats.get('best_signal') and stats.get('worst_signal'):
        best = stats['best_signal']
        worst = stats['worst_signal']
        html += f"""
    <div class="section">
        <h2>🏆 最佳 vs 最差</h2>
        <table>
            <tr>
                <th>类型</th>
                <th>股票</th>
                <th>信号日期</th>
                <th>信号价</th>
                <th>当前价</th>
                <th>收益</th>
            </tr>
            <tr>
                <td>🥇 最佳</td>
                <td><strong>{best.get('symbol', '')}</strong> {best.get('company_name', '')[:15]}</td>
                <td class="signal-date">{best.get('signal_date', '')}</td>
                <td>{price_sym}{best.get('signal_price', 0):.2f}</td>
                <td>{price_sym}{best.get('current_price', 0):.2f}</td>
                <td><span class="return-badge return-positive">+{best.get('current_return', 0):.1f}%</span></td>
            </tr>
            <tr>
                <td>❌ 最差</td>
                <td><strong>{worst.get('symbol', '')}</strong> {worst.get('company_name', '')[:15]}</td>
                <td class="signal-date">{worst.get('signal_date', '')}</td>
                <td>{price_sym}{worst.get('signal_price', 0):.2f}</td>
                <td>{price_sym}{worst.get('current_price', 0):.2f}</td>
                <td><span class="return-badge return-negative">{worst.get('current_return', 0):.1f}%</span></td>
            </tr>
        </table>
    </div>
"""
    
    # 详细信号列表
    html += """
    <div class="section">
        <h2>📋 信号详情 (最近20条)</h2>
        <table>
            <tr>
                <th>日期</th>
                <th>股票</th>
                <th>信号</th>
                <th>信号价</th>
                <th>D+1</th>
                <th>D+3</th>
                <th>D+5</th>
                <th>D+10</th>
                <th>当前</th>
            </tr>
"""
    
    for r in results_sorted[:20]:
        signal_tags = []
        if r.get('blue', 0) >= 160:
            signal_tags.append('<span class="signal-tag tag-blue">🔵 高BLUE</span>')
        elif r.get('blue', 0) >= 130:
            signal_tags.append('<span class="signal-tag tag-blue">BLUE</span>')
        if r.get('is_heima'):
            signal_tags.append('<span class="signal-tag tag-heima">🐴 黑马</span>')
        if r.get('is_juedi'):
            signal_tags.append('<span class="signal-tag tag-juedi">⚡ 绝地</span>')
        
        def format_return(val):
            if val is None:
                return '<td>-</td>'
            cls = 'positive' if val >= 0 else 'negative'
            return f'<td class="{cls}">{val:+.1f}%</td>'
        
        current_cls = 'return-positive' if r.get('current_return', 0) >= 0 else 'return-negative'
        
        html += f"""
            <tr>
                <td class="signal-date">{r.get('signal_date', '')}</td>
                <td><strong>{r.get('symbol', '')}</strong><br><small>{r.get('company_name', '')[:12]}</small></td>
                <td>{''.join(signal_tags)}</td>
                <td>{price_sym}{r.get('signal_price', 0):.2f}</td>
                {format_return(r.get('D1'))}
                {format_return(r.get('D3'))}
                {format_return(r.get('D5'))}
                {format_return(r.get('D10'))}
                <td><span class="return-badge {current_cls}">{r.get('current_return', 0):+.1f}%</span></td>
            </tr>
"""
    
    html += """
        </table>
    </div>
    
    <div class="footer">
        <p>📊 Coral Creek 量化系统 | 每日自动生成</p>
        <p>⚠️ 仅供参考，不构成投资建议</p>
    </div>
</body>
</html>
"""
    
    return html


def send_email(html_content: str, subject: str, to_email: str):
    """发送邮件"""
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    from_email = os.getenv('FROM_EMAIL', smtp_user)
    
    if not all([smtp_user, smtp_password, to_email]):
        print("❌ 邮件配置不完整，请检查环境变量:")
        print("  - SMTP_USER, SMTP_PASSWORD, TO_EMAIL")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        # 纯文本版本
        text_part = MIMEText("请使用支持 HTML 的邮件客户端查看报告", 'plain', 'utf-8')
        # HTML 版本
        html_part = MIMEText(html_content, 'html', 'utf-8')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        print(f"✅ 邮件发送成功: {to_email}")
        return True
    
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False


def generate_telegram_message(stats: Dict, results: List[Dict], market: str) -> str:
    """生成 Telegram 消息 (精简版)"""
    import requests
    
    market_name = "🇺🇸 美股" if market == 'US' else "🇨🇳 A股"
    price_sym = "$" if market == 'US' else "¥"
    today = datetime.now().strftime('%Y-%m-%d')
    
    msg = f"📊 *{market_name} 信号追踪日报*\n"
    msg += f"📅 {today}\n"
    msg += "━" * 20 + "\n\n"
    
    # 统计摘要
    msg += "📈 *【统计摘要】*\n"
    msg += f"总信号数: {stats.get('total_signals', 0)}\n"
    
    win_rate = stats.get('win_rate', 0)
    win_emoji = "🟢" if win_rate >= 50 else "🔴"
    msg += f"胜率: {win_emoji} {win_rate:.1f}%\n"
    
    avg = stats.get('avg_return', 0)
    avg_emoji = "📈" if avg >= 0 else "📉"
    msg += f"平均收益: {avg_emoji} {avg:+.2f}%\n\n"
    
    # 各周期收益
    msg += "📊 *【各周期表现】*\n"
    d1 = stats.get('d1_avg', 0)
    d3 = stats.get('d3_avg', 0)
    d5 = stats.get('d5_avg', 0)
    d10 = stats.get('d10_avg', 0)
    
    msg += f"D+1: {d1:+.2f}% | D+3: {d3:+.2f}%\n"
    msg += f"D+5: {d5:+.2f}% | D+10: {d10:+.2f}%\n\n"
    
    # 最佳/最差
    if stats.get('best_signal') and stats.get('worst_signal'):
        best = stats['best_signal']
        worst = stats['worst_signal']
        
        msg += "🏆 *【最佳 vs 最差】*\n"
        msg += f"🥇 {best.get('symbol', '')} +{best.get('current_return', 0):.1f}%\n"
        msg += f"❌ {worst.get('symbol', '')} {worst.get('current_return', 0):.1f}%\n\n"
    
    # Top 5 信号
    if results:
        msg += "🔥 *【近期热门信号】*\n"
        sorted_results = sorted(results, key=lambda x: x.get('current_return', 0), reverse=True)
        for r in sorted_results[:5]:
            ret = r.get('current_return', 0)
            emoji = "🟢" if ret >= 0 else "🔴"
            msg += f"{emoji} {r['symbol']}: {ret:+.1f}%\n"
    
    msg += "\n━" * 20 + "\n"
    msg += f"🔗 [查看详情](https://facaila.streamlit.app/)\n"
    msg += "⚠️ 仅供参考，不构成投资建议"
    
    return msg


def send_telegram(message: str) -> bool:
    """发送消息到 Telegram + 企业微信（若已配置）"""
    from services.notification import NotificationManager
    nm = NotificationManager()
    tg_ok = nm.send_telegram(message) if nm.telegram_token else False
    wc_ok = nm.send_wecom(message, msg_type='markdown') if nm.wecom_webhook else False
    wx_ok = nm.send_wxpusher(title="Coral Creek 信号追踪日报", content=message) if nm.wxpusher_app_token else False
    bark_ok = nm.send_bark(title="Coral Creek 信号追踪日报", content=message) if nm.bark_url else False
    overall = bool(tg_ok or wc_ok or wx_ok or bark_ok)
    print(f"NOTIFY_STATUS|overall={overall}|telegram={tg_ok}|wecom={wc_ok}|wxpusher={wx_ok}|bark={bark_ok}")
    return overall


def generate_and_send_report(market: str = 'US', days: int = 30):
    """生成并发送报告"""
    print(f"\n📊 生成 {market} 市场信号追踪报告...")
    
    # 1. 获取历史信号
    print("📋 获取历史信号...")
    signals = get_historical_signals(days=days, market=market)
    print(f"   找到 {len(signals)} 个信号")
    
    if not signals:
        print("⚠️ 没有找到信号数据")
        return False
    
    # 2. 计算收益
    print("💹 计算信号收益...")
    results = calculate_signal_returns(signals, market=market)
    print(f"   计算完成 {len(results)} 个信号")
    
    if not results:
        print("⚠️ 无法计算收益")
        return False
    
    # 3. 统计分析
    print("📈 统计分析...")
    stats = analyze_signal_stats(results)
    
    # 4. 生成 HTML 报告
    print("🎨 生成 HTML 报告...")
    html = generate_html_report(results, stats, market)
    
    # 5. 发送邮件
    to_email = os.getenv('TO_EMAIL')
    if to_email:
        market_name = "美股" if market == 'US' else "A股"
        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"📊 {market_name}信号追踪日报 - {today}"
        
        print(f"📧 发送邮件到 {to_email}...")
        send_email(html, subject, to_email)
    else:
        print("⚠️ TO_EMAIL 未设置，跳过邮件发送")
        # 保存到本地
        output_file = f"signal_report_{market}_{datetime.now().strftime('%Y%m%d')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"💾 报告已保存到: {output_file}")
    
    # 6. 发送 Telegram
    print("📱 发送 Telegram 消息...")
    telegram_msg = generate_telegram_message(stats, results, market)
    send_telegram(telegram_msg)
    
    # 打印摘要
    print("\n" + "=" * 50)
    print(f"📊 {market} 信号追踪摘要")
    print("=" * 50)
    print(f"总信号数: {stats.get('total_signals', 0)}")
    print(f"胜率: {stats.get('win_rate', 0):.1f}%")
    print(f"平均收益: {stats.get('avg_return', 0):+.2f}%")
    print(f"D+1 平均: {stats.get('d1_avg', 0):+.2f}%")
    print(f"D+3 平均: {stats.get('d3_avg', 0):+.2f}%")
    print(f"D+5 平均: {stats.get('d5_avg', 0):+.2f}%")
    print(f"D+10 平均: {stats.get('d10_avg', 0):+.2f}%")
    print("=" * 50)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='每日信号追踪报告')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'],
                       help='市场 (US/CN)')
    parser.add_argument('--both', action='store_true', help='发送美股和A股报告')
    parser.add_argument('--days', type=int, default=30, help='追踪天数')
    
    args = parser.parse_args()
    
    print("🚀 信号追踪报告生成器")
    print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.both:
        generate_and_send_report('US', args.days)
        print("\n" + "-" * 50 + "\n")
        generate_and_send_report('CN', args.days)
    else:
        generate_and_send_report(args.market, args.days)
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
