#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
默认策略组合自动执行
- 从最新扫描结果生成 5 个默认组合
- 按等金额买入到对应子账户
- 输出并推送（Telegram / 企业微信）组合绩效摘要
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(parent_dir, ".env"))
except Exception:
    pass

from db.database import get_scanned_dates, query_scan_results, get_connection
from services.portfolio_service import (
    create_paper_account,
    get_paper_account,
    get_paper_account_performance,
    paper_buy,
    reset_paper_account,
)
from services.notification import NotificationManager


def _pick_first(row: Dict, keys: List[str], default=None):
    for k in keys:
        if k in row and row.get(k) is not None:
            return row.get(k)
    return default


def _to_float(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "是")
    return False


def _extract_signal_fields(row: Dict) -> Dict:
    day_blue = _to_float(_pick_first(row, ["blue_daily", "Day_BLUE", "day_blue"], 0.0), 0.0)
    week_blue = _to_float(_pick_first(row, ["blue_weekly", "Week_BLUE", "week_blue"], 0.0), 0.0)
    month_blue = _to_float(_pick_first(row, ["blue_monthly", "Month_BLUE", "month_blue"], 0.0), 0.0)
    price = _to_float(_pick_first(row, ["price", "Close", "close"], 0.0), 0.0)

    is_heima = _to_bool(_pick_first(row, ["is_heima", "Is_Heima", "heima_daily", "Heima_Daily"], False))
    week_heima = _to_bool(_pick_first(row, ["heima_weekly", "Heima_Weekly"], False))
    month_heima = _to_bool(_pick_first(row, ["heima_monthly", "Heima_Monthly"], False))

    strategy_text = str(_pick_first(row, ["strategy", "Strategy"], "") or "")
    if ("黑马" in strategy_text) and not (is_heima or week_heima or month_heima):
        is_heima = True

    return {
        "symbol": str(_pick_first(row, ["symbol", "Symbol"], "") or "").upper().strip(),
        "price": price,
        "day_blue": day_blue,
        "week_blue": week_blue,
        "month_blue": month_blue,
        "is_heima": is_heima,
        "week_heima": week_heima,
        "month_heima": month_heima,
    }


def _strategy_defs():
    return [
        {"id": "daily_equal", "name": "每日组合等权", "account_tag": "daily_equal", "full_buy": False},
        {"id": "month_heima_all", "name": "月黑马全买", "account_tag": "month_heima_all", "full_buy": True},
        {"id": "week_heima_all", "name": "周黑马全买", "account_tag": "week_heima_all", "full_buy": True},
        {"id": "day_week_resonance", "name": "日周共振", "account_tag": "day_week_res", "full_buy": False},
        {"id": "core_resonance", "name": "核心共振", "account_tag": "core_res", "full_buy": False},
    ]


def _strategy_match(rule_id: str, f: Dict, min_blue: float) -> bool:
    d = f["day_blue"]
    w = f["week_blue"]
    m = f["month_blue"]
    h_any = f["is_heima"] or f["week_heima"] or f["month_heima"]
    if rule_id == "daily_equal":
        return d >= min_blue
    if rule_id == "month_heima_all":
        return f["month_heima"] or (m >= min_blue and h_any)
    if rule_id == "week_heima_all":
        return f["week_heima"] or (w >= min_blue and h_any)
    if rule_id == "day_week_resonance":
        return d >= min_blue and w >= min_blue
    if rule_id == "core_resonance":
        return d >= min_blue and w >= min_blue and (m >= min_blue or h_any)
    return False


def _strategy_score(f: Dict) -> float:
    base = f["day_blue"] * 0.45 + f["week_blue"] * 0.35 + f["month_blue"] * 0.20
    bonus = 0.0
    if f["is_heima"]:
        bonus += 25.0
    if f["week_heima"]:
        bonus += 20.0
    if f["month_heima"]:
        bonus += 20.0
    return base + bonus


def _already_bought_today(account_name: str, symbol: str, market: str, strategy_id: str) -> bool:
    trade_date = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM paper_trades
        WHERE account_name = ?
          AND symbol = ?
          AND market = ?
          AND trade_type = 'BUY'
          AND trade_date = ?
          AND notes LIKE ?
        LIMIT 1
        """,
        (account_name, symbol, market, trade_date, f"AUTO_BASKET:{strategy_id}%"),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def run_market(
    market: str,
    min_blue: float = 100,
    top_n: int = 20,
    full_cap: int = 120,
    deploy_pct: float = 90,
    seed_capital: float = 20000,
    reset_before_buy: bool = False,
    dry_run: bool = False,
) -> Dict:
    dates = get_scanned_dates(market=market)
    if not dates:
        return {"ok": False, "market": market, "error": "无扫描数据", "rows": []}

    latest_date = dates[0]
    rows = query_scan_results(scan_date=latest_date, market=market, limit=3000) or []
    defs = _strategy_defs()
    basket_results = {}

    for sd in defs:
        sym_map = {}
        for r in rows:
            f = _extract_signal_fields(r)
            sym = f["symbol"]
            if not sym or f["price"] <= 0:
                continue
            if not _strategy_match(sd["id"], f, min_blue):
                continue
            score = _strategy_score(f)
            prev = sym_map.get(sym)
            if (prev is None) or (score > prev["score"]):
                sym_map[sym] = {
                    "symbol": sym,
                    "price": f["price"],
                    "score": score,
                    "is_heima": f["is_heima"] or f["week_heima"] or f["month_heima"],
                }

        picks = sorted(sym_map.values(), key=lambda x: x["score"], reverse=True)
        picks = picks[:full_cap] if sd["full_buy"] else picks[:top_n]
        basket_results[sd["id"]] = picks

    exec_rows = []
    for sd in defs:
        picks = basket_results.get(sd["id"], [])
        account_name = f"auto_{market.lower()}_{sd['account_tag']}"
        create_ret = create_paper_account(account_name, float(seed_capital))
        if (not create_ret.get("success")) and ("已存在" not in str(create_ret.get("error", ""))):
            exec_rows.append({
                "strategy": sd["name"],
                "account": account_name,
                "candidates": len(picks),
                "success": 0,
                "skip": 0,
                "fail": len(picks),
                "error": create_ret.get("error", "创建账户失败"),
            })
            continue

        if reset_before_buy and not dry_run:
            reset_paper_account(account_name)

        acc = get_paper_account(account_name) or {}
        cash = float(acc.get("cash_balance", 0.0) or 0.0)
        deploy_cap = cash * (float(deploy_pct) / 100.0)
        per_budget = deploy_cap / max(len(picks), 1)

        success_cnt = 0
        skip_cnt = 0
        fail_cnt = 0
        first_err = ""

        for p in picks:
            sym = p["symbol"]
            px = float(p["price"])
            if px <= 0:
                skip_cnt += 1
                continue
            if _already_bought_today(account_name, sym, market, sd["id"]):
                skip_cnt += 1
                continue
            qty = int(per_budget / px)
            if qty < 1:
                skip_cnt += 1
                continue
            if dry_run:
                success_cnt += 1
                continue
            note = f"AUTO_BASKET:{sd['id']}|scan:{latest_date}|budget:{per_budget:.2f}"
            ret = paper_buy(sym, qty, px, market, account_name, notes=note)
            if ret.get("success"):
                success_cnt += 1
            else:
                fail_cnt += 1
                if not first_err:
                    first_err = ret.get("error", "下单失败")

        perf = get_paper_account_performance(account_name)
        acc_latest = get_paper_account(account_name) or {}
        pos = acc_latest.get("positions", []) or []
        open_winners = sum(1 for x in pos if float(x.get("unrealized_pnl", 0.0) or 0.0) > 0)
        open_win_rate = (open_winners / len(pos) * 100.0) if pos else 0.0

        exec_rows.append({
            "strategy": sd["name"],
            "account": account_name,
            "candidates": len(picks),
            "success": success_cnt,
            "skip": skip_cnt,
            "fail": fail_cnt,
            "total_return_pct": float(perf.get("total_return_pct", 0.0)),
            "closed_win_rate_pct": float(perf.get("win_rate_pct", 0.0)),
            "open_win_rate_pct": open_win_rate,
            "total_pnl": float(perf.get("total_pnl", 0.0)),
            "error": first_err,
        })

    return {
        "ok": True,
        "market": market,
        "scan_date": latest_date,
        "sample_count": len(rows),
        "rows": exec_rows,
    }


def _format_markdown_report(results: List[Dict], dry_run: bool) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "🧩 默认组合自动执行日报" + (" (DRY RUN)" if dry_run else "")
    lines = [f"*{title}*", f"`{ts}`", ""]
    for res in results:
        market = res.get("market")
        if not res.get("ok"):
            lines.append(f"❌ *{market}* 执行失败: {res.get('error', '-')}")
            lines.append("")
            continue
        lines.append(
            f"📊 *{market}* | 扫描日 `{res.get('scan_date')}` | 样本 {res.get('sample_count', 0)}"
        )
        lines.append("策略 | 候选 | 成功 | 跳过 | 失败 | 收益率 | 胜率(平仓) | 赢面(持仓)")
        for r in res.get("rows", []):
            lines.append(
                f"{r['strategy']} | {r['candidates']} | {r['success']} | {r['skip']} | {r['fail']} | "
                f"{r.get('total_return_pct', 0.0):+.2f}% | {r.get('closed_win_rate_pct', 0.0):.1f}% | "
                f"{r.get('open_win_rate_pct', 0.0):.1f}%"
            )
            if r.get("error"):
                lines.append(f"  ⚠️ {r['error']}")
        lines.append("")
    lines.append("_仅供研究，不构成投资建议_")
    return "\n".join(lines)


def _send_report(message: str) -> Dict[str, bool]:
    nm = NotificationManager()
    return {
        "telegram": nm.send_telegram(message) if nm.telegram_token else False,
        "wecom": nm.send_wecom(message, msg_type="markdown") if nm.wecom_webhook else False,
        "wxpusher": nm.send_wxpusher(title="Coral Creek 默认组合日报", content=message) if nm.wxpusher_app_token else False,
        "bark": nm.send_bark(title="Coral Creek 默认组合日报", content=message) if nm.bark_url else False,
    }


def main():
    parser = argparse.ArgumentParser(description="Run default strategy baskets and notify")
    parser.add_argument("--market", choices=["US", "CN", "ALL"], default="ALL")
    parser.add_argument("--min-blue", type=float, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--full-cap", type=int, default=120)
    parser.add_argument("--deploy-pct", type=float, default=90)
    parser.add_argument("--seed-capital", type=float, default=20000)
    parser.add_argument("--reset-before-buy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    markets = ["US", "CN"] if args.market == "ALL" else [args.market]
    results = []
    for m in markets:
        res = run_market(
            market=m,
            min_blue=args.min_blue,
            top_n=args.top_n,
            full_cap=args.full_cap,
            deploy_pct=args.deploy_pct,
            seed_capital=args.seed_capital,
            reset_before_buy=args.reset_before_buy,
            dry_run=args.dry_run,
        )
        results.append(res)

    msg = _format_markdown_report(results, args.dry_run)
    send_res = _send_report(msg)
    print(msg)
    overall = any(send_res.values()) if send_res else False
    print(
        "\nNOTIFY_STATUS|overall={}|telegram={}|wecom={}|wxpusher={}|bark={}".format(
            overall,
            send_res.get("telegram", False),
            send_res.get("wecom", False),
            send_res.get("wxpusher", False),
            send_res.get("bark", False),
        )
    )


if __name__ == "__main__":
    main()
