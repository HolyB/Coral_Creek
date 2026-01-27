#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多渠道推送模块 - 企业微信、飞书、Telegram、邮件
"""
import os
import requests
import json
from typing import Dict, Optional


class NotificationManager:
    """多渠道通知管理器"""
    
    def __init__(self):
        self.wecom_webhook = os.environ.get('WECOM_WEBHOOK')
        self.feishu_webhook = os.environ.get('FEISHU_WEBHOOK')
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    def send_wecom(self, content: str, msg_type: str = 'markdown') -> bool:
        """
        发送企业微信消息
        
        Args:
            content: 消息内容 (markdown格式)
            msg_type: markdown 或 text
        
        Returns:
            是否发送成功
        """
        if not self.wecom_webhook:
            return False
        
        try:
            if msg_type == 'markdown':
                data = {
                    "msgtype": "markdown",
                    "markdown": {"content": content}
                }
            else:
                data = {
                    "msgtype": "text",
                    "text": {"content": content}
                }
            
            resp = requests.post(self.wecom_webhook, json=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"WeCom error: {e}")
            return False
    
    def send_feishu(self, title: str, content: str) -> bool:
        """
        发送飞书消息
        
        Args:
            title: 标题
            content: 内容
        
        Returns:
            是否发送成功
        """
        if not self.feishu_webhook:
            return False
        
        try:
            data = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": title},
                        "template": "blue"
                    },
                    "elements": [
                        {"tag": "markdown", "content": content}
                    ]
                }
            }
            
            resp = requests.post(self.feishu_webhook, json=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"Feishu error: {e}")
            return False
    
    def send_telegram(self, content: str) -> bool:
        """
        发送 Telegram 消息
        
        Args:
            content: 消息内容 (markdown格式)
        
        Returns:
            是否发送成功
        """
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": content,
                "parse_mode": "Markdown"
            }
            
            resp = requests.post(url, json=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def send_all(self, title: str, content: str) -> Dict[str, bool]:
        """
        发送到所有已配置的渠道
        
        Args:
            title: 标题
            content: 内容
        
        Returns:
            各渠道发送结果
        """
        results = {}
        
        if self.wecom_webhook:
            results['wecom'] = self.send_wecom(content)
        
        if self.feishu_webhook:
            results['feishu'] = self.send_feishu(title, content)
        
        if self.telegram_token:
            results['telegram'] = self.send_telegram(f"**{title}**\n\n{content}")
        
        return results
    
    def format_signal_message(self, signals: list, market: str = 'US') -> str:
        """
        格式化信号消息
        
        Args:
            signals: 信号列表
            market: 市场
        
        Returns:
            格式化后的 Markdown 消息
        """
        if not signals:
            return f"📊 **{market} 市场今日无信号**"
        
        msg = f"📊 **Coral Creek {market} 信号报告**\n\n"
        msg += f"共发现 **{len(signals)}** 个信号:\n\n"
        
        for i, s in enumerate(signals[:10], 1):
            symbol = s.get('symbol', 'N/A')
            blue = s.get('blue_daily', 0)
            price = s.get('price', 0)
            
            # 信号强度标记
            if blue > 150:
                emoji = "🔥"
            elif blue > 100:
                emoji = "✅"
            else:
                emoji = "📍"
            
            msg += f"{emoji} **{symbol}**: BLUE={blue:.0f}, ${price:.2f}\n"
        
        msg += "\n⚠️ *仅供参考，不构成投资建议*"
        return msg


def send_daily_report(signals: list, market: str = 'US') -> Dict[str, bool]:
    """发送每日报告到所有渠道"""
    nm = NotificationManager()
    content = nm.format_signal_message(signals, market)
    return nm.send_all(f"Coral Creek {market} 日报", content)


if __name__ == "__main__":
    print("Notification channels available:")
    nm = NotificationManager()
    print(f"  WeCom: {'✅' if nm.wecom_webhook else '❌'}")
    print(f"  Feishu: {'✅' if nm.feishu_webhook else '❌'}")
    print(f"  Telegram: {'✅' if nm.telegram_token else '❌'}")
