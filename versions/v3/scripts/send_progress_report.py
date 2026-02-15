#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推送项目进度与策略分析报告
"""
import os
import sys
from datetime import datetime

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(parent_dir, ".env"))
except Exception:
    pass

from services.notification import NotificationManager

def send_report():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report = f"""*📊 Coral Creek 项目进度与策略深度分析*
`{ts}`

*1. 历史数据回填 (Backfill)*
• 任务: 5年美股历史数据回填
• 进度: **29 / 208** 交易日 (约 14%)
• 状态: 🟢 正常运行中 (速度 ~12.5分/天)
• 预计完成: **37 小时** 后

*2. Social KOL Scan 修复*
• ✅ 搜索引擎修复 (ddgs替代)
• ✅ 智能Ticker识别 (过滤YOU/TRUE噪音，识别$ON)
• ✅ 中文代码支持 (6位数字+后缀)
• ✅ 推送升级 (显示具体Ticker名单)

*3. 🧠 策略第一性原则分析 (关键)*
经数据交叉验证，发现策略存在**严重逆势交易**问题：

🚩 **数据实证**:
• **Blue Breakout**: 70% 信号发在大盘弱势(SPY<MA20)时 → 假突破概率极高
• **黑马策略**: 63% 信号发在大盘弱势时，仅38%在上涨日触发 → 逆势接飞刀
• **绝地反击**: 表现最好，67%发在强势时，符合牛市回调逻辑

💡 **改进建议 (Actionable)**:
建议立即实施 **Market Regime Filter (市场红绿灯)**：
• **红灯 (SPY<MA20)**: 禁止普通突破，只做绝世妖股(Blue>250)或绝地反击。
• **绿灯 (SPY>MA20)**: 策略全开。
此举预计能过滤掉 **70%** 的低胜率逆势信号。

_请协作者评估是否立即实施 Market Regime Filter。_
"""
    
    nm = NotificationManager()
    results = {
        "telegram": nm.send_telegram(report) if nm.telegram_token else False,
        "wecom": nm.send_wecom(report, msg_type="markdown") if nm.wecom_webhook else False,
        "wxpusher": nm.send_wxpusher(title="Coral Creek 进度报告", content=report) if nm.wxpusher_app_token else False,
        "bark": nm.send_bark(title="Coral Creek 进度报告", content=report) if nm.bark_url else False,
    }
    
    print("推送结果:")
    print(f"Telegram: {results['telegram']}")
    print(f"WeCom: {results['wecom']}")
    print(f"WxPusher: {results['wxpusher']}")
    print(f"Bark: {results['bark']}")

if __name__ == "__main__":
    send_report()
