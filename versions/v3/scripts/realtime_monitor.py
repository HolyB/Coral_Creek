#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控系统 - 每30分钟运行
1. 检查持仓状态 (是否触发止损/止盈)
2. 扫描实时机会 (盘中突破)
3. 发送 Telegram 通知
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# 时区
US_EASTERN = pytz.timezone('America/New_York')
CHINA_TZ = pytz.timezone('Asia/Shanghai')


def is_market_hours(market: str = 'US') -> Tuple[bool, str]:
    """检查是否在交易时间内"""
    now_utc = datetime.now(pytz.UTC)
    
    if market == 'US':
        now_local = now_utc.astimezone(US_EASTERN)
        # 美股: 9:30 - 16:00 ET, 周一至周五
        if now_local.weekday() >= 5:  # 周末
            return False, "周末休市"
        
        market_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_local.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_open <= now_local <= market_close:
            return True, f"交易中 {now_local.strftime('%H:%M')} ET"
        elif now_local < market_open:
            return False, f"盘前 (开盘 9:30 ET)"
        else:
            return False, f"盘后 (收盘 16:00 ET)"
    
    elif market == 'CN':
        now_local = now_utc.astimezone(CHINA_TZ)
        # A股: 9:30-11:30, 13:00-15:00, 周一至周五
        if now_local.weekday() >= 5:
            return False, "周末休市"
        
        morning_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
        morning_close = now_local.replace(hour=11, minute=30, second=0, microsecond=0)
        afternoon_open = now_local.replace(hour=13, minute=0, second=0, microsecond=0)
        afternoon_close = now_local.replace(hour=15, minute=0, second=0, microsecond=0)
        
        if morning_open <= now_local <= morning_close:
            return True, f"上午盘 {now_local.strftime('%H:%M')}"
        elif afternoon_open <= now_local <= afternoon_close:
            return True, f"下午盘 {now_local.strftime('%H:%M')}"
        elif now_local < morning_open:
            return False, "盘前"
        elif morning_close < now_local < afternoon_open:
            return False, "午休中"
        else:
            return False, "盘后"
    
    return False, "未知市场"


def get_realtime_price(symbol: str, market: str = 'US') -> Optional[float]:
    """获取实时价格"""
    try:
        if market == 'US':
            from polygon import RESTClient
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                return None
            
            client = RESTClient(api_key)
            # 获取最新报价
            quote = client.get_last_trade(symbol)
            if quote:
                return quote.price
        
        elif market == 'CN':
            import akshare as ak
            code = symbol.split('.')[0] if '.' in symbol else symbol
            df = ak.stock_zh_a_spot_em()
            if df is not None:
                row = df[df['代码'] == code]
                if not row.empty:
                    return float(row.iloc[0]['最新价'])
    except Exception as e:
        print(f"获取 {symbol} 价格失败: {e}")
    
    return None


def get_realtime_prices_batch(symbols: List[str], market: str = 'US') -> Dict[str, float]:
    """批量获取实时价格"""
    prices = {}
    
    try:
        if market == 'US':
            from polygon import RESTClient
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                return prices
            
            client = RESTClient(api_key)
            
            # 使用 snapshot API 批量获取
            snapshots = client.get_snapshot_all("stocks")
            if snapshots:
                symbol_set = set(symbols)
                for snap in snapshots:
                    if snap.ticker in symbol_set and snap.day:
                        prices[snap.ticker] = snap.day.close
        
        elif market == 'CN':
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            if df is not None:
                for symbol in symbols:
                    code = symbol.split('.')[0] if '.' in symbol else symbol
                    row = df[df['代码'] == code]
                    if not row.empty:
                        prices[symbol] = float(row.iloc[0]['最新价'])
    except Exception as e:
        print(f"批量获取价格失败: {e}")
    
    return prices


def check_positions(market: str = 'US') -> List[Dict]:
    """检查持仓状态，返回需要提醒的信号"""
    from strategies.signal_system import get_signal_manager
    
    manager = get_signal_manager()
    positions = manager.get_open_positions(market=market)
    
    if not positions:
        return []
    
    # 获取实时价格
    symbols = [p['symbol'] for p in positions]
    prices = get_realtime_prices_batch(symbols, market)
    
    alerts = []
    
    for pos in positions:
        symbol = pos['symbol']
        current_price = prices.get(symbol)
        
        if not current_price:
            continue
        
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        take_profit = pos['take_profit']
        
        pnl_pct = (current_price / entry_price - 1) * 100
        
        # 检查止损
        if current_price <= stop_loss:
            alerts.append({
                'symbol': symbol,
                'type': '🛑 止损提醒',
                'current_price': current_price,
                'trigger_price': stop_loss,
                'pnl_pct': pnl_pct,
                'message': f"已跌破止损位 {stop_loss:.2f}",
                'action': '建议立即止损'
            })
        
        # 检查止盈
        elif current_price >= take_profit:
            alerts.append({
                'symbol': symbol,
                'type': '🎯 止盈提醒',
                'current_price': current_price,
                'trigger_price': take_profit,
                'pnl_pct': pnl_pct,
                'message': f"已达目标价 {take_profit:.2f}",
                'action': '建议落袋为安'
            })
        
        # 接近止损 (距离止损 < 3%)
        elif (current_price - stop_loss) / stop_loss < 0.03:
            alerts.append({
                'symbol': symbol,
                'type': '⚠️ 接近止损',
                'current_price': current_price,
                'trigger_price': stop_loss,
                'pnl_pct': pnl_pct,
                'message': f"距离止损位仅 {(current_price/stop_loss-1)*100:.1f}%",
                'action': '密切关注'
            })
        
        # 接近止盈 (距离止盈 < 5%)
        elif (take_profit - current_price) / current_price < 0.05:
            alerts.append({
                'symbol': symbol,
                'type': '📈 接近目标',
                'current_price': current_price,
                'trigger_price': take_profit,
                'pnl_pct': pnl_pct,
                'message': f"距离目标价仅 {(take_profit/current_price-1)*100:.1f}%",
                'action': '考虑部分止盈'
            })
    
    return alerts


def scan_intraday_opportunities(market: str = 'US', top_n: int = 5) -> List[Dict]:
    """扫描盘中机会"""
    from db.database import query_scan_results, get_scanned_dates
    
    # 获取最近一次扫描结果
    dates = get_scanned_dates(market=market)
    if not dates:
        return []
    
    latest_date = dates[0]
    results = query_scan_results(scan_date=latest_date, market=market, limit=100)
    
    if not results:
        return []
    
    # 获取实时价格
    symbols = [r['symbol'] for r in results]
    prices = get_realtime_prices_batch(symbols, market)
    
    opportunities = []
    
    for r in results:
        symbol = r['symbol']
        current_price = prices.get(symbol)
        scan_price = r.get('price', 0)
        
        if not current_price or not scan_price:
            continue
        
        blue = r.get('blue_daily', 0) or 0
        adx = r.get('adx', 0) or 0
        
        # 价格变化
        price_change = (current_price - scan_price) / scan_price * 100
        
        # 机会评分
        score = 0
        reasons = []
        
        # 高 BLUE 信号
        if blue > 160:
            score += 30
            reasons.append(f"BLUE={blue:.0f}")
        elif blue > 130:
            score += 15
        
        # ADX 趋势确认
        if adx > 30:
            score += 20
            reasons.append(f"ADX={adx:.0f}")
        
        # 黑马/绝地信号
        if r.get('is_heima'):
            score += 25
            reasons.append("黑马")
        if r.get('is_juedi'):
            score += 20
            reasons.append("绝地")
        
        # 盘中上涨 (已经在涨)
        if 1 <= price_change <= 5:
            score += 15
            reasons.append(f"涨{price_change:.1f}%")
        
        # 盘中回调 (买入机会)
        if -3 <= price_change <= 0:
            score += 10
            reasons.append(f"回调{abs(price_change):.1f}%")
        
        if score >= 40:
            opportunities.append({
                'symbol': symbol,
                'name': r.get('company_name', ''),
                'score': score,
                'current_price': current_price,
                'scan_price': scan_price,
                'price_change': price_change,
                'blue': blue,
                'adx': adx,
                'reasons': reasons
            })
    
    # 排序并返回 top N
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities[:top_n]


def format_monitor_message(alerts: List[Dict], opportunities: List[Dict], market: str) -> str:
    """格式化监控消息"""
    market_name = "🇺🇸 美股" if market == 'US' else "🇨🇳 A股"
    price_sym = "$" if market == 'US' else "¥"
    now = datetime.now()
    
    msg = f"⏰ *{market_name} 实时监控*\n"
    msg += f"📅 {now.strftime('%Y-%m-%d %H:%M')}\n"
    msg += "━" * 25 + "\n\n"
    
    # 持仓提醒
    if alerts:
        msg += "🔔 *【持仓提醒】*\n\n"
        for a in alerts:
            msg += f"{a['type']} **{a['symbol']}**\n"
            msg += f"   现价: {price_sym}{a['current_price']:.2f}\n"
            msg += f"   盈亏: {a['pnl_pct']:+.1f}%\n"
            msg += f"   {a['message']}\n"
            msg += f"   👉 {a['action']}\n\n"
    else:
        msg += "✅ 持仓正常，无需操作\n\n"
    
    # 盘中机会
    if opportunities:
        msg += "💡 *【盘中机会】*\n\n"
        for o in opportunities[:3]:
            name = o['name'][:10] if o['name'] else ''
            msg += f"🔥 **{o['symbol']}** {name}\n"
            msg += f"   现价: {price_sym}{o['current_price']:.2f}"
            if o['price_change'] != 0:
                sign = "+" if o['price_change'] > 0 else ""
                msg += f" ({sign}{o['price_change']:.1f}%)"
            msg += "\n"
            msg += f"   评分: {o['score']}/100\n"
            msg += f"   理由: {', '.join(o['reasons'])}\n\n"
    
    msg += "━" * 25 + "\n"
    msg += "🔗 [详情](https://coralcreek.streamlit.app/)"
    
    return msg


def send_telegram_message(message: str) -> bool:
    """发送消息到 Telegram + 企业微信（若已配置）"""
    from services.notification import NotificationManager
    nm = NotificationManager()
    tg_ok = nm.send_telegram(message) if nm.telegram_token else False
    wc_ok = nm.send_wecom(message, msg_type='markdown') if nm.wecom_webhook else False
    wx_ok = nm.send_wxpusher(title="Coral Creek 盘中监控", content=message) if nm.wxpusher_app_token else False
    bark_ok = nm.send_bark(title="Coral Creek 盘中监控", content=message) if nm.bark_url else False
    overall = bool(tg_ok or wc_ok or wx_ok or bark_ok)
    print(f"NOTIFY_STATUS|overall={overall}|telegram={tg_ok}|wecom={wc_ok}|wxpusher={wx_ok}|bark={bark_ok}")
    return overall


def monitor_and_notify(market: str = 'US', notify: bool = True, force: bool = False):
    """主监控函数"""
    print(f"\n🔍 开始监控 {market} 市场...")
    
    # 检查交易时间
    is_open, status_msg = is_market_hours(market)
    print(f"📊 市场状态: {status_msg}")
    
    if not is_open and not force:
        print("⏸️ 非交易时间，跳过监控")
        return
    
    # 检查持仓
    print("📋 检查持仓状态...")
    alerts = check_positions(market)
    print(f"   发现 {len(alerts)} 个持仓提醒")
    
    # 扫描机会
    print("🔎 扫描盘中机会...")
    opportunities = scan_intraday_opportunities(market)
    print(f"   发现 {len(opportunities)} 个潜在机会")
    
    # 只有有内容时才发送
    if alerts or opportunities:
        message = format_monitor_message(alerts, opportunities, market)
        print("\n" + "=" * 40)
        print(message)
        print("=" * 40)
        
        if notify:
            send_telegram_message(message)
    else:
        print("ℹ️ 暂无需要提醒的内容")


def main():
    parser = argparse.ArgumentParser(description='实时交易监控')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'],
                       help='市场 (US/CN)')
    parser.add_argument('--both', action='store_true', help='监控美股和A股')
    parser.add_argument('--no-notify', action='store_true', help='不发送通知')
    parser.add_argument('--force', action='store_true', help='强制运行（忽略交易时间）')
    
    args = parser.parse_args()
    
    notify = not args.no_notify
    
    print("🚀 实时监控系统启动")
    print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.both:
        monitor_and_notify('US', notify, args.force)
        print("\n" + "-" * 40 + "\n")
        monitor_and_notify('CN', notify, args.force)
    else:
        monitor_and_notify(args.market, notify, args.force)
    
    print("\n✅ 监控完成!")


if __name__ == "__main__":
    main()
