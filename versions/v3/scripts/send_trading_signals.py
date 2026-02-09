#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日买卖信号推送
通过 Telegram 发送每日买入/卖出信号
"""
import os
import sys
import argparse
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def send_telegram_message(message: str, bot_token: str = None, chat_id: str = None) -> bool:
    """发送消息到 Telegram + 企业微信（若已配置）"""
    from services.notification import NotificationManager
    nm = NotificationManager()
    tg_ok = nm.send_telegram(message) if nm.telegram_token else False
    wc_ok = nm.send_wecom(message, msg_type='markdown') if nm.wecom_webhook else False
    print(f"notify result -> telegram={tg_ok}, wecom={wc_ok}")
    return bool(tg_ok or wc_ok)


def generate_and_send_signals(market: str = 'US'):
    """生成并发送买卖信号"""
    from strategies.signal_system import get_signal_manager, format_signal_message
    
    manager = get_signal_manager()
    
    # 生成每日信号
    result = manager.generate_daily_signals(market=market)
    
    if 'error' in result:
        print(f"❌ 生成信号失败: {result['error']}")
        return False
    
    signals = result.get('signals', [])
    
    if not signals:
        print(f"ℹ️ {market} 市场今日无交易信号")
        return True
    
    # 分类信号
    buy_signals = [s for s in signals if s['signal_type'] == '买入']
    sell_signals = [s for s in signals if s['signal_type'] in ['卖出', '止损', '止盈']]
    
    # 构建消息
    market_name = "🇺🇸 美股" if market == 'US' else "🇨🇳 A股"
    date_str = result.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    message = f"📊 *{market_name} 每日交易信号*\n"
    message += f"📅 {date_str}\n"
    message += "━" * 25 + "\n\n"
    
    # 买入信号 (取前5个最强)
    if buy_signals:
        message += "🟢 *【买入信号】*\n\n"
        for sig in buy_signals[:5]:
            message += format_signal_message(sig, market)
            message += "\n"
    
    # 卖出信号
    if sell_signals:
        message += "🔴 *【卖出/止损信号】*\n\n"
        for sig in sell_signals[:5]:
            message += format_signal_message(sig, market)
            message += "\n"
    
    # 统计
    message += "━" * 25 + "\n"
    message += f"📈 买入信号: {len(buy_signals)}个\n"
    message += f"📉 卖出信号: {len(sell_signals)}个\n"
    message += f"\n🔗 [查看详情](https://coralcreek.streamlit.app/)\n"
    message += f"⚠️ 仅供参考，投资有风险"
    
    # 发送
    return send_telegram_message(message)


def main():
    parser = argparse.ArgumentParser(description='发送每日买卖信号')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'],
                       help='市场 (US/CN)')
    parser.add_argument('--both', action='store_true', help='发送美股和A股')
    
    args = parser.parse_args()
    
    print(f"🚀 开始生成交易信号...")
    
    if args.both:
        print("\n📊 生成美股信号...")
        generate_and_send_signals('US')
        print("\n📊 生成A股信号...")
        generate_and_send_signals('CN')
    else:
        generate_and_send_signals(args.market)
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
