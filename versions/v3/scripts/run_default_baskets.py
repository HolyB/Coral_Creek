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
from services.candidate_tracking_service import get_candidate_tracking_rows
from services.meta_allocator_service import evaluate_strategy_baskets, allocate_meta_weights
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
    day_juedi = _to_bool(_pick_first(row, ["juedi_daily", "Juedi_Daily", "is_juedi", "Is_Juedi"], False))
    week_juedi = _to_bool(_pick_first(row, ["juedi_weekly", "Juedi_Weekly"], False))
    month_juedi = _to_bool(_pick_first(row, ["juedi_monthly", "Juedi_Monthly"], False))
    vp_rating = str(_pick_first(row, ["vp_rating", "VP_Rating"], "") or "")
    profit_ratio = _to_float(_pick_first(row, ["profit_ratio", "Profit_Ratio"], 0.0), 0.0)
    cap_category = str(_pick_first(row, ["cap_category", "Cap_Category"], "UNKNOWN") or "UNKNOWN")

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
        "day_juedi": day_juedi,
        "week_juedi": week_juedi,
        "month_juedi": month_juedi,
        "vp_rating": vp_rating,
        "profit_ratio": profit_ratio,
        "cap_category": cap_category,
    }


def _strategy_defs():
    defs = []
    pr = 1

    def _add_pair(prefix: str, title: str, th: int, strict_gt: bool, hj_scope: str, blue_scope: str):
        nonlocal pr, defs
        defs.append({
            "id": f"{prefix}_mid_large_{th}",
            "name": f"P{pr} 中大盘 {th}{title}",
            "account_tag": f"p{pr}_ml_{th}",
            "full_buy": False,
            "priority": pr,
            "cap_group": "mid_large",
            "blue_min": float(th),
            "strict_gt": bool(strict_gt),
            "hj_scope": hj_scope,     # day_week_month / day_week / day_only
            "blue_scope": blue_scope, # dwm / dw / d
        })
        pr += 1
        defs.append({
            "id": f"{prefix}_small_{th}",
            "name": f"P{pr} 小盘 {th}{title}",
            "account_tag": f"p{pr}_s_{th}",
            "full_buy": False,
            "priority": pr,
            "cap_group": "small",
            "blue_min": float(th),
            "strict_gt": bool(strict_gt),
            "hj_scope": hj_scope,
            "blue_scope": blue_scope,
        })
        pr += 1

    # P1-P8: 日周月 黑马/掘地 + 日周月Blue
    for th in (200, 150, 100, 50):
        _add_pair(
            prefix="p_dwm_hj_dwm_blue",
            title="三蓝",
            th=th,
            strict_gt=(th != 200),
            hj_scope="day_week_month",
            blue_scope="dwm",
        )

    # P9-P16: 日周 黑马/掘地 + 日周Blue
    for th in (200, 150, 100, 50):
        _add_pair(
            prefix="p_dw_hj_dw_blue",
            title="日周蓝",
            th=th,
            strict_gt=(th != 200),
            hj_scope="day_week",
            blue_scope="dw",
        )

    # P17-P24: 日 黑马/掘地 + 日Blue
    for th in (200, 150, 100, 50):
        _add_pair(
            prefix="p_d_hj_d_blue",
            title="日蓝",
            th=th,
            strict_gt=(th != 200),
            hj_scope="day_only",
            blue_scope="d",
        )

    return defs


def _strategy_key_map() -> Dict[str, str]:
    out = {}
    for d in _strategy_defs():
        sid = str(d.get("id") or "")
        if sid.startswith("p_dwm_hj_dwm_blue"):
            out[sid] = "blue_triple"
        elif sid.startswith("p_dw_hj_dw_blue"):
            out[sid] = "blue_day_week"
        elif sid.startswith("p_d_hj_d_blue"):
            out[sid] = "duokongwang"
        else:
            out[sid] = "defensive"
    return out


def _is_small_cap(cap_category: str) -> bool:
    txt = str(cap_category or "").lower()
    return ("small" in txt) or ("micro" in txt) or ("小盘" in txt) or ("微盘" in txt)


def _is_mid_large_cap(cap_category: str) -> bool:
    txt = str(cap_category or "").lower()
    return ("mid" in txt) or ("large" in txt) or ("mega" in txt) or ("中盘" in txt) or ("大盘" in txt) or ("巨头" in txt)


def _chip_good(f: Dict) -> bool:
    vp = str(f.get("vp_rating") or "").lower()
    pr = _to_float(f.get("profit_ratio"), 0.0)
    return (vp in ("good", "excellent")) or (pr >= 0.7)


def _strategy_match(rule_id: str, f: Dict, min_blue: float) -> bool:
    _ = min_blue  # 兼容旧参数
    cfg = None
    for x in _strategy_defs():
        if str(x.get("id")) == str(rule_id):
            cfg = x
            break
    if not cfg:
        return False

    d = _to_float(f.get("day_blue"), 0.0)
    w = _to_float(f.get("week_blue"), 0.0)
    m = _to_float(f.get("month_blue"), 0.0)
    threshold = _to_float(cfg.get("blue_min"), 100.0)
    strict_gt = bool(cfg.get("strict_gt", False))

    def _pass(v: float) -> bool:
        return v > threshold if strict_gt else v >= threshold

    day_hj = bool(f.get("is_heima")) or bool(f.get("day_juedi"))
    week_hj = bool(f.get("week_heima")) or bool(f.get("week_juedi"))
    month_hj = bool(f.get("month_heima")) or bool(f.get("month_juedi"))

    hj_scope = str(cfg.get("hj_scope") or "day_week_month")
    if hj_scope == "day_week_month":
        hj_ok = day_hj and week_hj and month_hj
    elif hj_scope == "day_week":
        hj_ok = day_hj and week_hj
    else:
        hj_ok = day_hj
    if not hj_ok:
        return False
    if not _chip_good(f):
        return False

    cap = str(f.get("cap_category") or "UNKNOWN")
    cap_group = str(cfg.get("cap_group") or "")
    if cap_group == "small" and not _is_small_cap(cap):
        return False
    if cap_group == "mid_large" and not _is_mid_large_cap(cap):
        return False

    blue_scope = str(cfg.get("blue_scope") or "dwm")
    if blue_scope == "dwm":
        return _pass(d) and _pass(w) and _pass(m)
    if blue_scope == "dw":
        return _pass(d) and _pass(w)
    return _pass(d)


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


def _build_segment_industry_weights(
    tracking_rows: List[Dict],
    strategy_id: str,
    min_blue: float,
) -> Dict[str, Dict[str, float]]:
    seg_stat: Dict[str, Dict[str, float]] = {}
    ind_stat: Dict[str, Dict[str, float]] = {}
    used = 0
    for tr in tracking_rows or []:
        tf = _extract_signal_fields(tr)
        if not _strategy_match(strategy_id, tf, min_blue=min_blue):
            continue
        used += 1
        pnl = _to_float(tr.get("pnl_pct"), 0.0)
        seg = str(tr.get("cap_category") or "UNKNOWN")
        ind = str(tr.get("industry") or "UNKNOWN")
        seg_bucket = seg_stat.setdefault(seg, {"n": 0.0, "w": 0.0, "p": 0.0})
        ind_bucket = ind_stat.setdefault(ind, {"n": 0.0, "w": 0.0, "p": 0.0})
        seg_bucket["n"] += 1
        ind_bucket["n"] += 1
        if pnl > 0:
            seg_bucket["w"] += 1
            ind_bucket["w"] += 1
        seg_bucket["p"] += pnl
        ind_bucket["p"] += pnl

    def _to_weight(stat: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        out = {}
        for k, v in stat.items():
            n = max(v.get("n", 0.0), 1.0)
            wr = v.get("w", 0.0) / n
            avg = v.get("p", 0.0) / n
            # 胜率主导，均收次之，做柔和限幅
            raw = 1.0 + 0.7 * (wr - 0.5) + 0.3 * (avg / 5.0)
            out[k] = max(0.70, min(1.30, raw))
        return out

    return {
        "sample": used,
        "segment": _to_weight(seg_stat),
        "industry": _to_weight(ind_stat),
    }


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


def _query_scan_results_local(scan_date: str, market: str, limit: int = 3000) -> List[Dict]:
    """
    组合执行优先读取本地 SQLite（保证与本地补齐后的字段一致）。
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM scan_results
        WHERE scan_date = ? AND market = ?
        ORDER BY blue_daily DESC
        LIMIT ?
        """,
        (scan_date, market, int(limit)),
    )
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return rows


def run_market(
    market: str,
    min_blue: float = 100,
    top_n: int = 20,
    full_cap: int = 120,
    deploy_pct: float = 90,
    seed_capital: float = 20000,
    track_days: int = 180,
    recent_scan_days: int = 20,
    reset_before_buy: bool = False,
    dry_run: bool = False,
) -> Dict:
    dates = get_scanned_dates(market=market)
    if not dates:
        return {"ok": False, "market": market, "error": "无扫描数据", "rows": []}

    latest_date = dates[0]
    use_dates = dates[: max(1, int(recent_scan_days))]
    pool_rows = []
    for d in use_dates:
        local_rows = _query_scan_results_local(scan_date=d, market=market, limit=3000)
        if local_rows:
            pool_rows.extend(local_rows)
        else:
            pool_rows.extend(query_scan_results(scan_date=d, market=market, limit=3000) or [])
    # 同一股票保留最近信号，扩大候选池但不重复买同票
    latest_by_symbol = {}
    for r in sorted(pool_rows, key=lambda x: str(x.get("scan_date") or ""), reverse=True):
        sym = str(_pick_first(r, ["symbol", "Symbol"], "") or "").upper().strip()
        if not sym:
            continue
        if sym not in latest_by_symbol:
            latest_by_symbol[sym] = r
    rows = list(latest_by_symbol.values())
    defs = _strategy_defs()
    basket_results = {}
    tracking_rows = []
    try:
        tracking_rows = get_candidate_tracking_rows(market=market, days_back=int(track_days)) or []
    except Exception:
        tracking_rows = []

    # 组合层权重（策略一级）+ 分层权重（市值/行业二级）
    strategy_weight_map: Dict[str, float] = {}
    if tracking_rows:
        try:
            perf_rows = evaluate_strategy_baskets(
                rows=tracking_rows,
                rule_name="fixed_10d",
                take_profit_pct=10.0,
                stop_loss_pct=6.0,
                max_hold_days=20,
                fee_bps=5.0,
                slippage_bps=5.0,
                min_samples=20,
                max_rows=1200,
            )
            alloc_rows = allocate_meta_weights(perf_rows, max_weight=0.45, min_weight=0.05)
            strategy_weight_map = {
                str(r.get("strategy_key")): _to_float(r.get("建议权重(%)"), 0.0) / 100.0
                for r in alloc_rows
            }
        except Exception:
            strategy_weight_map = {}

    segment_weight_map_by_rule: Dict[str, Dict[str, float]] = {}
    industry_weight_map_by_rule: Dict[str, Dict[str, float]] = {}
    strategy_weight_by_rule: Dict[str, float] = {}
    defs = sorted(defs, key=lambda x: int(x.get("priority", 999)))
    assigned_symbols = set()
    for sd in defs:
        w = _build_segment_industry_weights(
            tracking_rows=tracking_rows,
            strategy_id=sd["id"],
            min_blue=min_blue,
        )
        segment_weight_map_by_rule[sd["id"]] = w.get("segment", {})
        industry_weight_map_by_rule[sd["id"]] = w.get("industry", {})
        strategy_key = _strategy_key_map().get(sd["id"], "blue_day_week")
        strategy_weight_by_rule[sd["id"]] = strategy_weight_map.get(strategy_key, 0.20)

    for sd in defs:
        strategy_key = _strategy_key_map().get(sd["id"], "blue_day_week")
        strategy_w = strategy_weight_by_rule.get(sd["id"], 0.20)  # 没样本时给保守默认
        sym_map = {}
        for r in rows:
            f = _extract_signal_fields(r)
            sym = f["symbol"]
            if not sym or f["price"] <= 0:
                continue
            if sym in assigned_symbols:
                continue
            if not _strategy_match(sd["id"], f, min_blue):
                continue
            seg = str(r.get("cap_category") or "UNKNOWN")
            ind = str(r.get("industry") or "UNKNOWN")
            seg_w = segment_weight_map_by_rule.get(sd["id"], {}).get(seg, 1.0)
            ind_w = industry_weight_map_by_rule.get(sd["id"], {}).get(ind, 1.0)
            raw_score = _strategy_score(f)
            score = raw_score * strategy_w * seg_w * ind_w
            prev = sym_map.get(sym)
            if (prev is None) or (score > prev["score"]):
                sym_map[sym] = {
                    "symbol": sym,
                    "price": f["price"],
                    "score": score,
                    "raw_score": raw_score,
                    "strategy_weight": strategy_w,
                    "segment_weight": seg_w,
                    "industry_weight": ind_w,
                    "segment": seg,
                    "industry": ind,
                    "is_heima": f["is_heima"] or f["week_heima"] or f["month_heima"],
                }

        picks = sorted(sym_map.values(), key=lambda x: x["score"], reverse=True)
        picks = picks[:full_cap] if sd["full_buy"] else picks[:top_n]
        basket_results[sd["id"]] = picks
        for p in picks:
            assigned_symbols.add(str(p.get("symbol") or ""))

    exec_rows = []
    for sd in defs:
        picks = basket_results.get(sd["id"], [])
        strategy_key = _strategy_key_map().get(sd["id"], "blue_day_week")
        strategy_w = strategy_weight_by_rule.get(sd["id"], 0.20)
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

        matched_track_rows = []
        if tracking_rows:
            for tr in tracking_rows:
                tf = _extract_signal_fields(tr)
                if _strategy_match(sd["id"], tf, min_blue=min_blue):
                    matched_track_rows.append(tr)
        track_total = len(matched_track_rows)
        if track_total > 0:
            track_wins = sum(1 for r in matched_track_rows if float(r.get("pnl_pct") or 0) > 0)
            track_win_rate = track_wins / track_total * 100.0
            track_avg_pnl = sum(float(r.get("pnl_pct") or 0.0) for r in matched_track_rows) / track_total
            d2p_vals = sorted(
                int(r["first_positive_day"])
                for r in matched_track_rows
                if r.get("first_positive_day") is not None
            )
            if d2p_vals:
                mid = len(d2p_vals) // 2
                track_median_d2p = d2p_vals[mid] if len(d2p_vals) % 2 == 1 else int(round((d2p_vals[mid - 1] + d2p_vals[mid]) / 2))
            else:
                track_median_d2p = None
        else:
            track_win_rate = None
            track_avg_pnl = None
            track_median_d2p = None

        exec_rows.append({
            "strategy": sd["name"],
            "strategy_key": strategy_key,
            "strategy_weight_pct": round(strategy_w * 100.0, 1),
            "account": account_name,
            "candidates": len(picks),
            "success": success_cnt,
            "skip": skip_cnt,
            "fail": fail_cnt,
            "total_return_pct": float(perf.get("total_return_pct", 0.0)),
            "closed_win_rate_pct": float(perf.get("win_rate_pct", 0.0)),
            "open_win_rate_pct": open_win_rate,
            "track_total": track_total,
            "track_win_rate_pct": track_win_rate,
            "track_avg_pnl_pct": track_avg_pnl,
            "track_median_d2p": track_median_d2p,
            "total_pnl": float(perf.get("total_pnl", 0.0)),
            "top_pick": picks[0]["symbol"] if picks else "-",
            "top_pick_score": round(_to_float(picks[0].get("score"), 0.0), 2) if picks else 0.0,
            "top_pick_seg": picks[0].get("segment", "-") if picks else "-",
            "top_pick_ind": picks[0].get("industry", "-") if picks else "-",
            "error": first_err,
        })

    return {
        "ok": True,
        "market": market,
        "scan_date": latest_date,
        "scan_pool_days": int(recent_scan_days),
        "track_days": int(track_days),
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
            f"📊 *{market}* | 扫描日 `{res.get('scan_date')}` | 候选池 {res.get('scan_pool_days', 20)}天 | 样本 {res.get('sample_count', 0)} | 追踪窗 {res.get('track_days', 180)}天"
        )
        lines.append("策略 | 权重 | 候选 | 成功 | 跳过 | 失败 | 收益率 | 胜率(平仓) | 赢面(持仓) | 追踪样本 | 追踪胜率 | 追踪均收 | 转正中位天 | TOP")
        for r in res.get("rows", []):
            track_win_txt = f"{float(r.get('track_win_rate_pct')):.1f}%" if r.get("track_win_rate_pct") is not None else "-"
            track_avg_txt = f"{float(r.get('track_avg_pnl_pct')):+.2f}%" if r.get("track_avg_pnl_pct") is not None else "-"
            track_d2p_txt = f"{int(r.get('track_median_d2p'))}天" if r.get("track_median_d2p") is not None else "-"
            top_txt = f"{r.get('top_pick', '-')}/{r.get('top_pick_seg', '-')}"
            lines.append(
                f"{r['strategy']} | {float(r.get('strategy_weight_pct', 0.0)):.1f}% | "
                f"{r['candidates']} | {r['success']} | {r['skip']} | {r['fail']} | "
                f"{r.get('total_return_pct', 0.0):+.2f}% | {r.get('closed_win_rate_pct', 0.0):.1f}% | "
                f"{r.get('open_win_rate_pct', 0.0):.1f}% | "
                f"{int(r.get('track_total', 0) or 0)} | "
                f"{track_win_txt} | {track_avg_txt} | {track_d2p_txt} | {top_txt}"
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
    parser.add_argument("--track-days", type=int, default=180)
    parser.add_argument("--recent-scan-days", type=int, default=20)
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
            track_days=args.track_days,
            recent_scan_days=args.recent_scan_days,
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
