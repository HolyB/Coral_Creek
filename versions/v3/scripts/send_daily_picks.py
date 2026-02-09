#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Telegram 今日精选推送
每日推送多策略共识股票
"""
import os
import sys
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def send_telegram_message(message: str, parse_mode: str = 'HTML') -> bool:
    """发送消息到 Telegram + 企业微信（若已配置）"""
    from services.notification import NotificationManager
    nm = NotificationManager()
    tg_ok = nm.send_telegram(message) if nm.telegram_token else False
    wc_ok = nm.send_wecom(message, msg_type='markdown') if nm.wecom_webhook else False
    wx_ok = nm.send_wxpusher(title="Coral Creek 今日精选", content=message) if nm.wxpusher_app_token else False
    bark_ok = nm.send_bark(title="Coral Creek 今日精选", content=message) if nm.bark_url else False
    print(f"notify result -> telegram={tg_ok}, wecom={wc_ok}, wxpusher={wx_ok}, bark={bark_ok}")
    return bool(tg_ok or wc_ok or wx_ok or bark_ok)


def generate_picks_message(market: str = 'US') -> str:
    """生成今日精选消息"""
    from strategies.decision_system import get_strategy_manager
    from db.database import query_scan_results, get_scanned_dates
    import pandas as pd
    
    # 获取最新数据
    try:
        dates = get_scanned_dates(market=market)
    except Exception as e:
        print(f"⚠️ get_scanned_dates failed ({market}): {e}")
        return None
    if not dates:
        return None
    
    latest_date = dates[0]
    try:
        results = query_scan_results(scan_date=latest_date, market=market, limit=500)
    except Exception as e:
        print(f"⚠️ query_scan_results failed ({market}, {latest_date}): {e}")
        return None
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # 获取策略选股
    manager = get_strategy_manager()
    consensus = manager.get_consensus_picks(df, min_votes=2)
    
    # 构建消息
    market_emoji = "🇺🇸" if market == "US" else "🇨🇳"
    market_name = "美股" if market == "US" else "A股"
    
    lines = [
        f"🎯 <b>今日精选 | {market_emoji} {market_name}</b>",
        f"📅 {latest_date}",
        "",
    ]
    
    # 共识股票
    if consensus:
        lines.append("🔥 <b>多策略共识 (重点关注)</b>")
        for symbol, votes, score in consensus[:5]:
            stars = "⭐" * votes
            lines.append(f"  • <code>{symbol}</code> {stars} ({score:.0f}分)")
        lines.append("")
    
    # 每个策略的 top pick
    all_picks = manager.get_all_picks(df, top_n=3)
    
    lines.append("📊 <b>各策略首选</b>")
    for strategy_key, picks in all_picks.items():
        if picks:
            strategy = manager.strategies[strategy_key]
            top = picks[0]
            lines.append(
                f"  {strategy.icon} {strategy.name}: "
                f"<code>{top.symbol}</code> ${top.entry_price:.2f} "
                f"(止损${top.stop_loss:.2f})"
            )
    
    lines.append("")
    lines.append("⚠️ 以上仅供参考，不构成投资建议")
    lines.append("🌐 <a href='https://coralcreek.streamlit.app/'>查看详情</a>")
    
    return "\n".join(lines)


def send_daily_picks(markets: list = None):
    """发送每日精选到 Telegram"""
    if markets is None:
        markets = ['US', 'CN']
    
    for market in markets:
        message = generate_picks_message(market)
        if message:
            send_telegram_message(message)
        else:
            print(f"⚠️ No picks for {market}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Send daily picks to Telegram')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN', 'ALL'])
    
    args = parser.parse_args()
    
    if args.market == 'ALL':
        send_daily_picks(['US', 'CN'])
    else:
        send_daily_picks([args.market])
