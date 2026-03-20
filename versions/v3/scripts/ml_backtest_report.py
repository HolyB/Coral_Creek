#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGB_leaf+MMoE 2026 回测报告生成器
=================================
1. 生成精美 HTML 报告 (6-panel 图表 + 每日选股明细)
2. 发送邮件 (SMTP)
3. 推送 Telegram / WxPusher / Bark
4. 本地保存 HTML

PYTHONPATH=. python scripts/ml_backtest_report.py --market US
PYTHONPATH=. python scripts/ml_backtest_report.py --market CN
PYTHONPATH=. python scripts/ml_backtest_report.py --market BOTH
"""
import os, sys, argparse, smtplib, base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from pathlib import Path

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))

from dotenv import load_dotenv
load_dotenv(V3_DIR / '.env')


# =========================================================
# 1. HTML Report Generator
# =========================================================
def generate_html_report(market: str) -> str:
    """Generate a premium HTML email report for one market"""
    import json

    # ---- load data that was saved by backtest_full.py ----
    data_path = Path(f'/tmp/backtest_charts/{market}_report_data.json')
    chart_path = Path(f'/tmp/backtest_charts/{market}_backtest_2026.png')

    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
    else:
        data = {}

    # Embed chart as base64
    chart_b64 = ''
    if chart_path.exists():
        with open(chart_path, 'rb') as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    market_emoji = "🇺🇸" if market == 'US' else "🇨🇳"
    market_name = "美股" if market == 'US' else "A股"
    price_sym = "$" if market == 'US' else "¥"
    today = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Stats from data or defaults
    total_ret = data.get('total_ret', 0)
    sharpe = data.get('sharpe', 0)
    sortino = data.get('sortino', 0)
    max_dd = data.get('max_dd', 0)
    win_rate = data.get('win_rate', 0)
    trades = data.get('total_trades', 0)
    pf = data.get('profit_factor', 0)
    avg_win = data.get('avg_win', 0)
    avg_loss = data.get('avg_loss', 0)
    picks = data.get('daily_top1', [])
    tier_stats = data.get('tier_stats', [])
    monthly = data.get('monthly', [])

    # ---- Build HTML ----
    ret_color = '#22c55e' if total_ret >= 0 else '#ef4444'

    rows_html = ''
    for p in picks:
        def fmt(v):
            if v is None: return '<td style="color:#999">—</td>'
            c = '#22c55e' if v >= 0 else '#ef4444'
            return f'<td style="color:{c};font-weight:600">{v:+.1f}%</td>'
        mc = p.get('mcap', 0)
        if market == 'US':
            mcs = f"${mc/1e9:.1f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
        else:
            mcs = f"{mc/1e8:.0f}亿"
        marker = '✅' if (p.get('r20') or p.get('r_now') or 0) > 0 else '❌'
        rows_html += f"""<tr>
            <td style="color:#666;font-size:12px">{p.get('date','')}</td>
            <td><strong>{p.get('symbol','')}</strong><br><small style="color:#888">{p.get('name','')}</small></td>
            <td>{price_sym}{p.get('buy',0):.2f}</td>
            {fmt(p.get('r5'))}
            {fmt(p.get('r10'))}
            {fmt(p.get('r20'))}
            {fmt(p.get('r_now'))}
            <td style="font-size:11px">{p.get('tier','')}</td>
            <td style="font-size:11px">{mcs}</td>
            <td>{marker}</td>
        </tr>"""

    tier_rows = ''
    for t in tier_stats:
        act_c = '#22c55e' if t.get('actual', 0) >= 0 else '#ef4444'
        tier_rows += f"""<tr>
            <td>{t.get('tier','')}</td>
            <td style="text-align:center">{t.get('n',0)}</td>
            <td style="color:{act_c};font-weight:600;text-align:center">{t.get('actual',0):+.1f}%</td>
            <td style="text-align:center">{t.get('win_rate',0):.0f}%</td>
        </tr>"""

    monthly_rows = ''
    for m in monthly:
        pnl_c = '#22c55e' if m.get('total_pnl', 0) >= 0 else '#ef4444'
        monthly_rows += f"""<tr>
            <td>{m.get('month','')}</td>
            <td style="text-align:center">{m.get('trades',0)}</td>
            <td style="text-align:center">{m.get('win_rate',0):.0f}%</td>
            <td style="color:{pnl_c};font-weight:600;text-align:center">{m.get('avg_pnl',0):+.1f}%</td>
            <td style="color:{pnl_c};font-weight:700;text-align:center">{m.get('total_pnl',0):+.0f}%</td>
        </tr>"""

    chart_img = f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:12px;margin:15px 0" />' if chart_b64 else ''

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body {{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0;background:#0f172a;color:#e2e8f0}}
.container {{max-width:900px;margin:0 auto;padding:20px}}
.header {{background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#a855f7 100%);
  padding:40px 30px;border-radius:16px;margin-bottom:24px;text-align:center}}
.header h1 {{margin:0;font-size:28px;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.3)}}
.header p {{margin:8px 0 0;color:rgba(255,255,255,0.85);font-size:14px}}
.kpi-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.kpi {{background:#1e293b;border-radius:12px;padding:20px;text-align:center;
  border:1px solid #334155;transition: transform 0.2s}}
.kpi:hover {{transform:translateY(-2px);border-color:#6366f1}}
.kpi-value {{font-size:32px;font-weight:800;line-height:1.2}}
.kpi-label {{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:6px}}
.section {{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:24px;border:1px solid #334155}}
.section h2 {{margin:0 0 16px;font-size:18px;color:#e2e8f0;border-bottom:2px solid #6366f1;padding-bottom:8px}}
table {{width:100%;border-collapse:collapse;font-size:13px}}
th {{background:#0f172a;padding:10px 8px;text-align:left;font-weight:600;color:#94a3b8;
  font-size:11px;text-transform:uppercase;letter-spacing:0.5px}}
td {{padding:8px;border-bottom:1px solid #1e293b}}
tr:hover {{background:rgba(99,102,241,0.08)}}
.green {{color:#22c55e}} .red {{color:#ef4444}} .blue {{color:#818cf8}}
.footer {{text-align:center;color:#64748b;font-size:12px;padding:20px;margin-top:24px}}
.badge {{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:700;font-size:13px}}
.badge-green {{background:rgba(34,197,94,0.15);color:#22c55e}}
.badge-red {{background:rgba(239,68,68,0.15);color:#ef4444}}
@media(max-width:600px){{.kpi-grid{{grid-template-columns:repeat(2,1fr)}}
  table{{font-size:11px}}th,td{{padding:6px}}}}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>{market_emoji} Coral Creek Way Quantitative</h1>
  <p>📅 {today} | {market_name} 回测报告</p>
  <p style="margin-top:12px;font-size:20px">
    <span class="badge {'badge-green' if total_ret >= 0 else 'badge-red'}"
      style="font-size:22px;padding:6px 16px">{total_ret:+.1f}%</span>
  </p>
</div>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-value" style="color:{ret_color}">{total_ret:+.1f}%</div><div class="kpi-label">总收益</div></div>
  <div class="kpi"><div class="kpi-value blue">{sharpe:.2f}</div><div class="kpi-label">Sharpe</div></div>
  <div class="kpi"><div class="kpi-value {'green' if win_rate >= 60 else 'red'}">{win_rate:.0f}%</div><div class="kpi-label">胜率</div></div>
  <div class="kpi"><div class="kpi-value {'green' if max_dd > -10 else 'red'}">{max_dd:.1f}%</div><div class="kpi-label">最大回撤</div></div>
</div>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-value blue">{sortino:.2f}</div><div class="kpi-label">Sortino</div></div>
  <div class="kpi"><div class="kpi-value">{trades}</div><div class="kpi-label">总交易</div></div>
  <div class="kpi"><div class="kpi-value green">{pf:.1f}</div><div class="kpi-label">盈亏比</div></div>
  <div class="kpi"><div class="kpi-value green">+{avg_win:.0f}%</div><div class="kpi-label">平均盈利</div></div>
</div>

<div class="section">
  <h2>📈 分析图表</h2>
  {chart_img}
</div>

<div class="section">
  <h2>📅 月度表现</h2>
  <table><tr><th>月份</th><th style="text-align:center">交易</th><th style="text-align:center">胜率</th>
    <th style="text-align:center">平均PnL</th><th style="text-align:center">总PnL</th></tr>
  {monthly_rows}
  </table>
</div>

<div class="section">
  <h2>📊 市值分层</h2>
  <table><tr><th>Tier</th><th style="text-align:center">n</th>
    <th style="text-align:center">实际20d</th><th style="text-align:center">胜率</th></tr>
  {tier_rows}
  </table>
</div>

<!-- Per-tier pick tables -->
"""
    # Group picks by tier
    from collections import defaultdict
    tier_grouped = defaultdict(list)
    for p in picks:
        tier_grouped[p.get('tier', 'Unknown')].append(p)

    if market == 'US':
        tier_order = ['Mega (>$200B)', 'Large ($10-200B)', 'Mid ($2-10B)', 'Small ($300M-2B)', 'Micro ($50-300M)', 'Nano (<$50M)']
    else:
        tier_order = ['大盘 (>500亿)', '中盘 (100-500亿)', '小盘 (20-100亿)', '微盘 (<20亿)']
    all_tiers = [t for t in tier_order if t in tier_grouped] + [t for t in sorted(tier_grouped.keys()) if t not in tier_order]

    import numpy as np

    tier_colors = {
        'Mega (>$200B)': '#f59e0b', 'Large ($10-200B)': '#6366f1', 'Mid ($2-10B)': '#22c55e',
        'Small ($300M-2B)': '#818cf8', 'Micro ($50-300M)': '#94a3b8', 'Nano (<$50M)': '#64748b',
        '大盘 (>500亿)': '#f59e0b', '中盘 (100-500亿)': '#22c55e', '小盘 (20-100亿)': '#818cf8', '微盘 (<20亿)': '#94a3b8'
    }

    for tier in all_tiers:
        tier_p = tier_grouped[tier]
        r20_vals = [p.get('r20') for p in tier_p if p.get('r20') is not None]
        r_now_vals = [p.get('r_now') for p in tier_p if p.get('r_now') is not None]
        avg20 = np.mean(r20_vals) if r20_vals else 0
        wr20 = (np.array(r20_vals) > 0).mean() * 100 if r20_vals else 0
        avg_now = np.mean(r_now_vals) if r_now_vals else 0
        wr_now = (np.array(r_now_vals) > 0).mean() * 100 if r_now_vals else 0
        tc = tier_colors.get(tier, '#818cf8')
        a20c = '#22c55e' if avg20 >= 0 else '#ef4444'

        tier_rows_html = ''
        for p in tier_p:
            mc = p.get('mcap', 0)
            if market == 'US':
                mcs = f"${mc/1e9:.1f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
            else:
                mcs = f"{mc/1e8:.0f}亿"
            best_r = p.get('r20') if p.get('r20') is not None else p.get('r_now')
            marker = '✅' if best_r is not None and best_r > 0 else '❌' if best_r is not None else '⏳'
            tier_rows_html += f"""<tr>
                <td style="color:#94a3b8;font-size:12px">{p.get('date','')}</td>
                <td><strong>{p.get('symbol','')}</strong><br><small style="color:#64748b">{p.get('name','')}</small></td>
                <td>{price_sym}{p.get('buy',0):.2f}</td>
                {fmt(p.get('r5'))}
                {fmt(p.get('r10'))}
                {fmt(p.get('r20'))}
                {fmt(p.get('r_now'))}
                <td style="font-size:11px">{mcs}</td>
                <td>{marker}</td>
            </tr>"""

        html += f"""
<div class="section">
  <h2 style="border-color:{tc}">📊 {tier}
    <span style="float:right;font-size:13px;font-weight:400;color:#94a3b8">
      {len(tier_p)} picks |
      <span style="color:{a20c}">{avg20:+.1f}%</span> avg |
      WR {wr20:.0f}%
    </span>
  </h2>
  <div style="display:flex;gap:16px;margin-bottom:12px">
    <div style="flex:1;background:#0f172a;border-radius:8px;padding:12px;text-align:center">
      <div style="font-size:20px;font-weight:700;color:{a20c}">{avg20:+.1f}%</div>
      <div style="font-size:10px;color:#64748b;margin-top:4px">20D AVG RETURN</div>
    </div>
    <div style="flex:1;background:#0f172a;border-radius:8px;padding:12px;text-align:center">
      <div style="font-size:20px;font-weight:700;color:{'#22c55e' if wr20 >= 60 else '#ef4444'}">{wr20:.0f}%</div>
      <div style="font-size:10px;color:#64748b;margin-top:4px">WIN RATE</div>
    </div>
    <div style="flex:1;background:#0f172a;border-radius:8px;padding:12px;text-align:center">
      <div style="font-size:20px;font-weight:700;color:#818cf8">{len(tier_p)}</div>
      <div style="font-size:10px;color:#64748b;margin-top:4px">PICKS</div>
    </div>
    <div style="flex:1;background:#0f172a;border-radius:8px;padding:12px;text-align:center">
      <div style="font-size:20px;font-weight:700;color:{'#22c55e' if avg_now >= 0 else '#ef4444'}">{avg_now:+.1f}%</div>
      <div style="font-size:10px;color:#64748b;margin-top:4px">至今 AVG</div>
    </div>
  </div>
  <table>
  <tr><th>日期</th><th>股票</th><th>买入</th>
    <th>5d</th><th>10d</th><th>20d</th><th>至今</th>
    <th>MCap</th><th></th></tr>
  {tier_rows_html}
  </table>
</div>
"""

    html += """

<div class="footer">
  <p>🔗 <a href="https://facaila.streamlit.app/" style="color:#818cf8">在线查看</a></p>
  <p>⚠️ 仅供参考，不构成投资建议</p>
  <p style="font-size:9px;color:#475569">Model: XGBoost Leaf + MMoE (5d/10d/20d) Walk-Forward</p>
</div>

</div>
</body>
</html>"""
    return html


# =========================================================
# 2. Email Sender
# =========================================================
def send_email_report(html: str, market: str, chart_path: str = None):
    """Send HTML report via SMTP"""
    smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_SENDER') or os.getenv('SMTP_USER')
    smtp_pass = os.getenv('SMTP_PASSWORD')
    receivers = os.getenv('EMAIL_RECEIVERS', os.getenv('TO_EMAIL', ''))
    to_list = [e.strip() for e in receivers.split(',') if e.strip()]

    if not all([smtp_user, smtp_pass, to_list]):
        print("  ⚠️ Email not configured (SMTP_SENDER/SMTP_PASSWORD/EMAIL_RECEIVERS)")
        return False

    market_name = "美股" if market == 'US' else "A股"
    today = datetime.now().strftime('%Y-%m-%d')
    subject = f"Coral Creek Way | {market_name} 回测报告 — {today}"

    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = ', '.join(to_list)

    html_part = MIMEText(html, 'html', 'utf-8')
    msg.attach(html_part)

    # Attach chart as inline image
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, 'rb') as f:
            img = MIMEImage(f.read(), name=os.path.basename(chart_path))
            img.add_header('Content-ID', '<backtest_chart>')
            msg.attach(img)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"  ✅ Email sent to {', '.join(to_list)}")
        return True
    except Exception as e:
        print(f"  ❌ Email failed: {e}")
        return False


# =========================================================
# 3. Telegram / WxPusher / Bark Push
# =========================================================
def send_push_notification(market: str, data: dict):
    """Send summary via Telegram + all configured channels"""
    try:
        from services.notification import NotificationManager
        nm = NotificationManager()
    except ImportError:
        print("  ⚠️ NotificationManager not available")
        return False

    market_emoji = "🇺🇸" if market == 'US' else "🇨🇳"
    market_name = "美股" if market == 'US' else "A股"

    total_ret = data.get('total_ret', 0)
    sharpe = data.get('sharpe', 0)
    win_rate = data.get('win_rate', 0)
    max_dd = data.get('max_dd', 0)
    picks = data.get('daily_top1', [])

    # Build message
    lines = [
        f"📊 *{market_emoji} {market_name} Coral Creek Way 回测*",
        "",
        f"💰 总收益: *{total_ret:+.1f}%*",
        f"📈 Sharpe: *{sharpe:.2f}*",
        f"🎯 胜率: *{win_rate:.0f}%*",
        f"📉 最大回撤: *{max_dd:.1f}%*",
        "",
        "🔥 *近期 Top-1 选股:*",
    ]

    price_sym = "$" if market == 'US' else "¥"
    recent = picks[-10:] if picks else []
    for p in recent:
        r = p.get('r20') or p.get('r_now') or 0
        emoji = "🟢" if r > 0 else "🔴"
        lines.append(
            f"  {emoji} `{p.get('symbol','')}` {price_sym}{p.get('buy',0):.2f} → "
            f"{r:+.1f}%"
        )

    lines.append("")
    lines.append("⚠️ 仅供参考，不构成投资建议")
    msg = "\n".join(lines)

    results = nm.send_all(f"{market_name} Coral Creek Way 回测报告", msg)
    for ch, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {ch}")
    return any(results.values())


# =========================================================
# 4. Data Exporter (called from backtest_full.py)
# =========================================================
def save_report_data(market, stats, daily_top1, tier_stats, monthly):
    """Save JSON data for report generation"""
    import json
    data = {
        'market': market,
        'timestamp': datetime.now().isoformat(),
        'total_ret': stats.get('total_ret', 0),
        'sharpe': stats.get('sharpe', 0),
        'sortino': stats.get('sortino', 0),
        'max_dd': stats.get('max_dd', 0),
        'win_rate': stats.get('win_rate', 0),
        'total_trades': stats.get('total_trades', 0),
        'profit_factor': stats.get('profit_factor', 0),
        'avg_win': stats.get('avg_win', 0),
        'avg_loss': stats.get('avg_loss', 0),
        'daily_top1': daily_top1,
        'tier_stats': tier_stats,
        'monthly': monthly,
    }
    out = f'/tmp/backtest_charts/{market}_report_data.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  💾 Report data saved: {out}")
    return data


# =========================================================
# 5. Main
# =========================================================
def generate_and_send(market: str, send_email_flag: bool = True, send_push_flag: bool = True):
    """Generate and send report for one market"""
    import json
    print(f"\n{'='*60}")
    print(f"📊 {market} Coral Creek Way 回测报告")
    print(f"{'='*60}")

    data_path = f'/tmp/backtest_charts/{market}_report_data.json'
    chart_path = f'/tmp/backtest_charts/{market}_backtest_2026.png'

    if not os.path.exists(data_path):
        print(f"  ⚠️ No report data found at {data_path}")
        print(f"  Run: python /tmp/backtest_full.py {market} first")
        return False

    with open(data_path) as f:
        data = json.load(f)

    # 1. Generate HTML
    print("  🎨 Generating HTML report...")
    html = generate_html_report(market)
    
    # Save local copy
    html_path = f'/tmp/backtest_charts/{market}_backtest_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  💾 HTML saved: {html_path}")

    # 2. Send Email
    if send_email_flag:
        print("  📧 Sending email...")
        send_email_report(html, market, chart_path)

    # 3. Push notification
    if send_push_flag:
        print("  📱 Sending push notifications...")
        send_push_notification(market, data)

    print(f"  ✅ {market} report done!")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Backtest Report Generator')
    parser.add_argument('--market', default='BOTH', choices=['US', 'CN', 'BOTH'])
    parser.add_argument('--no-email', action='store_true', help='Skip email')
    parser.add_argument('--no-push', action='store_true', help='Skip push notifications')
    args = parser.parse_args()

    print(f"🚀 ML Backtest Report Generator")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    markets = ['US', 'CN'] if args.market == 'BOTH' else [args.market]
    for m in markets:
        generate_and_send(m, not args.no_email, not args.no_push)

    print("\n✅ All done!")
