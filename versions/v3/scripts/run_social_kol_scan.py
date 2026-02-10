#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
社交大V喊单自动抓取 + 评估推送
"""
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(parent_dir, ".env"))
except Exception:
    pass

from services.blogger_service import (
    collect_social_kol_recommendations,
    get_blogger_performance,
)
from services.notification import NotificationManager


DEFAULT_KOLS = [
    "Twitter,Roaring Kitty,TheRoaringKitty,US",
    "Reddit,WallStreetBets,wallstreetbets,US",
    "雪球,雪球热榜,xueqiu,CN",
    "微博,财经博主样本,sinafinance,CN",
]


def _parse_kol_lines(text: str) -> List[Dict]:
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append({
            "platform": parts[0],
            "name": parts[1],
            "handle": parts[2],
            "market": parts[3] if len(parts) >= 4 else "",
        })
    return rows


def _format_report(ingest_ret: Dict, perf_rows: List[Dict], tag: str, horizon_days: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "*📡 社交大V喊单自动追踪*",
        f"`{ts}`",
        f"标签: `{tag}` | 评估周期: {horizon_days}天",
        "",
        (
            f"抓取结果: KOL {ingest_ret.get('processed_kols', 0)} | "
            f"新增博主 {ingest_ret.get('new_bloggers', 0)} | "
            f"新增推荐 {ingest_ret.get('added_recommendations', 0)} | "
            f"去重跳过 {ingest_ret.get('skipped_duplicates', 0)}"
        ),
        "",
        "Top 评估（按方向收益）:",
    ]

    if perf_rows:
        for r in perf_rows[:8]:
            lines.append(
                f"- {r.get('name', '-')}: 样本 {r.get('calculated_count', 0)} | "
                f"命中率 {float(r.get('win_rate', 0.0)):.1f}% | "
                f"方向收益 {float(r.get('avg_directional_return', 0.0)):+.2f}% | "
                f"平均收益 {float(r.get('avg_return', 0.0)):+.2f}%"
            )
    else:
        lines.append("- 暂无可评估样本")

    if ingest_ret.get("errors"):
        lines.append("")
        lines.append("错误摘要:")
        for e in ingest_ret.get("errors", [])[:3]:
            lines.append(f"- {e}")

    lines.append("")
    lines.append("_仅供研究，不构成投资建议_")
    return "\n".join(lines)


def _send_report(message: str) -> Dict[str, bool]:
    nm = NotificationManager()
    return {
        "telegram": nm.send_telegram(message) if nm.telegram_token else False,
        "wecom": nm.send_wecom(message, msg_type="markdown") if nm.wecom_webhook else False,
        "wxpusher": nm.send_wxpusher(title="Coral Creek 社交大V追踪", content=message) if nm.wxpusher_app_token else False,
        "bark": nm.send_bark(title="Coral Creek 社交大V追踪", content=message) if nm.bark_url else False,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect social KOL picks and send report")
    parser.add_argument("--tag", default=os.environ.get("SOCIAL_KOL_TAG", "AUTO_SOCIAL_KOL"))
    parser.add_argument("--horizon-days", type=int, default=int(os.environ.get("SOCIAL_KOL_HORIZON_DAYS", "10")))
    parser.add_argument("--max-per-kol", type=int, default=int(os.environ.get("SOCIAL_KOL_MAX_PER_KOL", "20")))
    parser.add_argument("--min-samples", type=int, default=int(os.environ.get("SOCIAL_KOL_MIN_SAMPLES", "3")))
    args = parser.parse_args()

    kol_text = os.environ.get("SOCIAL_KOL_LIST", "\n".join(DEFAULT_KOLS))
    kol_configs = _parse_kol_lines(kol_text)
    ingest_ret = collect_social_kol_recommendations(
        kol_configs=kol_configs,
        portfolio_tag=args.tag,
        max_results_per_kol=args.max_per_kol,
    )
    perf = get_blogger_performance(horizon_days=args.horizon_days, portfolio_tag=args.tag)
    perf = [x for x in perf if int(x.get("calculated_count", 0) or 0) >= int(args.min_samples)]
    perf.sort(key=lambda x: float(x.get("avg_directional_return", 0.0)), reverse=True)

    msg = _format_report(ingest_ret, perf, args.tag, args.horizon_days)
    print(msg)

    send_ret = _send_report(msg)
    overall = any(send_ret.values()) if send_ret else False
    print(
        "NOTIFY_STATUS|overall={}|telegram={}|wecom={}|wxpusher={}|bark={}".format(
            overall,
            send_ret.get("telegram", False),
            send_ret.get("wecom", False),
            send_ret.get("wxpusher", False),
            send_ret.get("bark", False),
        )
    )


if __name__ == "__main__":
    main()
