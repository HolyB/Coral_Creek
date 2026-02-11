#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一可交易评估 + 策略组合层分配器
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Sequence
import numpy as np

from services.candidate_tracking_service import evaluate_exit_rule


STRATEGY_TAG_DEFS = {
    "blue_day_week": {"name": "日周共振", "tags_any": ["DAY_BLUE", "WEEK_BLUE"]},
    "blue_triple": {"name": "三线共振", "tags_all": ["DAY_BLUE", "WEEK_BLUE", "MONTH_BLUE"]},
    "heima": {"name": "黑马策略", "tags_any": ["DAY_HEIMA", "WEEK_HEIMA", "MONTH_HEIMA"]},
    "chip": {"name": "筹码突破", "tags_any": ["CHIP_BREAKOUT", "CHIP_DENSE"]},
    "defensive": {"name": "防守均衡", "tags_any": ["DAY_BLUE", "CHIP_DENSE"]},
}


def _build_exit_rule_label(rule_name: str, take_profit_pct: float, stop_loss_pct: float, max_hold_days: int) -> str:
    if str(rule_name) == "tp_sl_time":
        return f"tp_sl_time(TP={float(take_profit_pct):.0f}%,SL={float(stop_loss_pct):.0f}%,Hold={int(max_hold_days)}d)"
    if str(rule_name) == "top_divergence_guard":
        return f"top_divergence_guard(TP={float(take_profit_pct):.0f}%,SL={float(stop_loss_pct):.0f}%,Hold={int(max_hold_days)}d)"
    if str(rule_name) == "kdj_dead_cross":
        return f"kdj_dead_cross(Hold={int(max_hold_days)}d)"
    return str(rule_name)


def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _match_strategy(row: Dict, cfg: Dict) -> bool:
    tags = set(row.get("signal_tags_list") or [])
    tags_all = set(cfg.get("tags_all") or [])
    tags_any = set(cfg.get("tags_any") or [])
    if tags_all and not tags_all.issubset(tags):
        return False
    if tags_any and tags.isdisjoint(tags_any):
        return False
    return bool(tags)


def _calc_trade_metrics_from_details(details: Sequence[Dict], avg_hold_days: float) -> Dict:
    if not details:
        return {
            "total_return_pct": 0.0,
            "ann_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "turnover_per_year": 0.0,
        }

    rets = np.array([_to_float(d.get("exit_return_pct")) / 100.0 for d in details], dtype=float)
    rets = rets[np.isfinite(rets)]
    # 物理边界保护：单笔收益率不应 <= -100%
    rets = np.clip(rets, -0.999, 5.0)
    if rets.size == 0:
        return {
            "total_return_pct": 0.0,
            "ann_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "turnover_per_year": 0.0,
        }
    curve = np.cumprod(1.0 + rets)
    total_ret = float(curve[-1] - 1.0)

    peaks = np.maximum.accumulate(curve)
    dd = (curve / np.where(peaks == 0, 1.0, peaks)) - 1.0
    max_dd = abs(float(dd.min())) if dd.size else 0.0

    hold = max(_to_float(avg_hold_days, 10.0), 1.0)
    trades_per_year = 252.0 / hold
    n = len(rets)

    # 原始几何年化（不截尾，诊断用）
    if n > 0:
        try:
            ann_raw = float(np.expm1(np.mean(np.log1p(rets)) * trades_per_year))
        except Exception:
            ann_raw = -1.0
    else:
        ann_raw = 0.0

    # 主口径：稳健几何年化（对单笔收益做截尾后再按log-return复利年化）
    rets_robust = np.clip(rets, -0.35, 0.35)
    try:
        ann = float(np.expm1(np.mean(np.log1p(rets_robust)) * trades_per_year)) if n > 0 else 0.0
    except Exception:
        ann = 0.0

    # 期望年化（算术均值法，仅作辅助对照）
    ann_expect = float(np.mean(rets) * trades_per_year) if n > 0 else 0.0

    ann_raw = float(np.clip(ann_raw, -0.99, 5.0))
    ann = float(np.clip(ann, -0.99, 5.0))
    ann_expect = float(np.clip(ann_expect, -0.99, 5.0))

    mu = float(np.mean(rets)) if n > 0 else 0.0
    sigma = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = (mu / sigma) * np.sqrt(trades_per_year) if sigma > 1e-10 else 0.0

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    gross_profit = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = abs(float(np.sum(losses))) if losses.size else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else (9.99 if gross_profit > 0 else 0.0)

    return {
        "total_return_pct": round(total_ret * 100.0, 2),
        "ann_return_pct": round(ann * 100.0, 2),
        "ann_return_raw_pct": round(ann_raw * 100.0, 2),
        "ann_return_expect_pct": round(ann_expect * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "profit_factor": round(profit_factor, 2),
        "turnover_per_year": round(trades_per_year, 2),
    }


def _calc_meta_score(row: Dict) -> float:
    """
    统一策略排序分：用于“按最优组合排序”与权重计算前的稳定排序。
    """
    sample = max(_to_float(row.get("sample"), 0.0), 1.0)
    sample_factor = min(sample / 120.0, 1.0)
    score = (
        0.45 * max(_to_float(row.get("sharpe"), 0.0), 0.0)
        + 0.30 * max(_to_float(row.get("ann_return_pct"), 0.0), 0.0) / 30.0
        + 0.15 * max(_to_float(row.get("net_win_rate_pct"), 0.0) - 50.0, 0.0) / 20.0
        + 0.10 * max(25.0 - _to_float(row.get("max_drawdown_pct"), 0.0), 0.0) / 25.0
    ) * sample_factor
    return round(float(max(score, 0.0)), 4)


def evaluate_strategy_unified(
    rows: Sequence[Dict],
    rule_name: str,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_hold_days: int,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    max_rows: int = 1200,
) -> Dict:
    gross = evaluate_exit_rule(
        rows=rows,
        rule_name=rule_name,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_hold_days=max_hold_days,
        max_rows=max_rows,
    )
    details = list(gross.get("details") or [])
    if not details:
        return {
            "sample": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "avg_hold_days": _to_float(gross.get("avg_exit_day"), 0.0),
            "net_avg_return_pct": 0.0,
            "net_win_rate_pct": 0.0,
            "total_return_pct": 0.0,
            "ann_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "turnover_per_year": 0.0,
        }

    round_trip_cost_pct = (fee_bps + slippage_bps) * 2.0 / 10000.0 * 100.0
    net_details = []
    for d in details:
        r = _to_float(d.get("exit_return_pct"))
        nd = dict(d)
        nd["exit_return_pct"] = max(r - round_trip_cost_pct, -99.9)
        net_details.append(nd)

    net_avg = float(np.mean([_to_float(d.get("exit_return_pct")) for d in net_details]))
    net_wins = sum(1 for d in net_details if _to_float(d.get("exit_return_pct")) > 0)
    sample = len(net_details)

    ext = _calc_trade_metrics_from_details(
        details=net_details,
        avg_hold_days=_to_float(gross.get("avg_exit_day"), 10.0),
    )
    return {
        "sample": sample,
        "win_rate_pct": _to_float(gross.get("win_rate_pct")),
        "avg_return_pct": _to_float(gross.get("avg_return_pct")),
        "avg_hold_days": _to_float(gross.get("avg_exit_day")),
        "net_avg_return_pct": round(net_avg, 2),
        "net_win_rate_pct": round(net_wins / sample * 100.0, 1) if sample else 0.0,
        **ext,
    }


def evaluate_strategy_baskets(
    rows: Sequence[Dict],
    rule_name: str,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_hold_days: int,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    min_samples: int = 15,
    max_rows: int = 1200,
) -> List[Dict]:
    out: List[Dict] = []
    data = list(rows or [])
    if not data:
        return out

    for key, cfg in STRATEGY_TAG_DEFS.items():
        picked = [r for r in data if _match_strategy(r, cfg)]
        metrics = evaluate_strategy_unified(
            rows=picked,
            rule_name=rule_name,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_hold_days=max_hold_days,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_rows=max_rows,
        )
        if int(metrics.get("sample", 0)) < int(min_samples):
            continue
        out.append(
            {
                "strategy_key": key,
                "策略": cfg.get("name", key),
                **metrics,
                "meta_score": _calc_meta_score(metrics),
                "exit_rule": rule_name,
                "exit_rule_desc": _build_exit_rule_label(rule_name, take_profit_pct, stop_loss_pct, max_hold_days),
            }
        )
    out.sort(key=lambda x: (x.get("meta_score", 0.0), x.get("sharpe", 0.0), x.get("ann_return_pct", 0.0)), reverse=True)
    return out


def evaluate_strategy_baskets_best_exit(
    rows: Sequence[Dict],
    exit_rule_candidates: Sequence[Dict],
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    min_samples: int = 15,
    max_rows: int = 1200,
) -> List[Dict]:
    """
    多规则评估后，对每个策略仅保留最优卖出规则。
    exit_rule_candidates 元素示例:
    {"rule_name":"fixed_10d","take_profit_pct":10,"stop_loss_pct":6,"max_hold_days":20}
    """
    if not rows:
        return []
    cands = list(exit_rule_candidates or [])
    if not cands:
        return []

    best_map: Dict[str, Dict] = {}
    for cand in cands:
        rule_name = str(cand.get("rule_name") or "fixed_10d")
        tp = _to_float(cand.get("take_profit_pct"), 10.0)
        sl = _to_float(cand.get("stop_loss_pct"), 6.0)
        hold = int(_to_float(cand.get("max_hold_days"), 20.0))
        rows_now = evaluate_strategy_baskets(
            rows=rows,
            rule_name=rule_name,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            max_hold_days=hold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_samples=min_samples,
            max_rows=max_rows,
        )
        for row in rows_now:
            k = str(row.get("strategy_key") or "")
            if not k:
                continue
            prev = best_map.get(k)
            if prev is None:
                best_map[k] = row
                continue
            prev_key = (
                _to_float(prev.get("meta_score"), 0.0),
                _to_float(prev.get("sharpe"), 0.0),
                _to_float(prev.get("ann_return_pct"), 0.0),
            )
            now_key = (
                _to_float(row.get("meta_score"), 0.0),
                _to_float(row.get("sharpe"), 0.0),
                _to_float(row.get("ann_return_pct"), 0.0),
            )
            if now_key > prev_key:
                best_map[k] = row

    out = list(best_map.values())
    out.sort(key=lambda x: (x.get("meta_score", 0.0), x.get("sharpe", 0.0), x.get("ann_return_pct", 0.0)), reverse=True)
    return out


def allocate_meta_weights(
    perf_rows: Sequence[Dict],
    max_weight: float = 0.45,
    min_weight: float = 0.05,
) -> List[Dict]:
    rows = list(perf_rows or [])
    if not rows:
        return []

    raw_scores = []
    for r in rows:
        sample = max(_to_float(r.get("sample"), 0.0), 1.0)
        sample_factor = min(sample / 120.0, 1.0)
        score = (
            0.40 * max(_to_float(r.get("sharpe"), 0.0), 0.0)
            + 0.25 * max(_to_float(r.get("ann_return_pct"), 0.0), 0.0) / 30.0
            + 0.20 * max(_to_float(r.get("net_win_rate_pct"), 0.0) - 50.0, 0.0) / 20.0
            # 回撤越小越好（正值口径）
            + 0.15 * max(20.0 - _to_float(r.get("max_drawdown_pct"), 0.0), 0.0) / 20.0
        ) * sample_factor
        raw_scores.append(max(score, 0.0))

    raw = np.array(raw_scores, dtype=float)
    if raw.sum() <= 1e-12:
        raw = np.ones_like(raw) / len(raw)
    else:
        raw = raw / raw.sum()

    clipped = np.clip(raw, min_weight, max_weight)
    clipped = clipped / clipped.sum()

    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "strategy_key": r.get("strategy_key"),
                "策略": r.get("策略"),
                "建议权重(%)": round(float(clipped[i] * 100.0), 1),
                "Sharpe": r.get("sharpe"),
                "年化(%)": r.get("ann_return_pct"),
                "回撤(%)": r.get("max_drawdown_pct"),
                "样本": r.get("sample"),
            }
        )
    out.sort(key=lambda x: x.get("建议权重(%)", 0.0), reverse=True)
    return out


def build_today_meta_plan(
    rows: Sequence[Dict],
    weight_rows: Sequence[Dict],
    top_n: int = 10,
    total_capital: float = 100000.0,
    include_history: bool = False,
    max_signal_age_days: int = 20,
) -> List[Dict]:
    if not rows or not weight_rows:
        return []

    latest_date = max(str(r.get("signal_date") or "") for r in rows if r.get("signal_date"))
    latest_dt = None
    try:
        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
    except Exception:
        latest_dt = None

    selected_rows = []
    for r in rows:
        sig_date = str(r.get("signal_date") or "")
        if not sig_date:
            continue
        if not include_history:
            if sig_date == latest_date:
                selected_rows.append(r)
            continue
        if latest_dt is None:
            selected_rows.append(r)
            continue
        try:
            sig_dt = datetime.strptime(sig_date, "%Y-%m-%d")
            age_days = (latest_dt - sig_dt).days
            if 0 <= age_days <= int(max_signal_age_days):
                selected_rows.append(r)
        except Exception:
            continue

    key2cfg = STRATEGY_TAG_DEFS
    picked: Dict[str, Dict] = {}
    w_map = {str(w.get("strategy_key")): _to_float(w.get("建议权重(%)")) for w in weight_rows}

    for r in selected_rows:
        sym = str(r.get("symbol") or "")
        if not sym:
            continue

        hit_strategies = []
        weight_sum = 0.0
        for key, cfg in key2cfg.items():
            if key not in w_map:
                continue
            if not _match_strategy(r, cfg):
                continue
            hit_strategies.append(cfg.get("name", key))
            weight_sum += _to_float(w_map.get(key))

        if not hit_strategies:
            continue

        day_blue = _to_float(r.get("blue_daily"))
        week_blue = _to_float(r.get("blue_weekly"))
        # 信号强度压缩到相对可解释区间，避免纯数值过大失真
        signal_strength = 0.7 * min(day_blue / 200.0, 1.5) + 0.3 * min(week_blue / 200.0, 1.5)
        raw_exec = max(weight_sum * signal_strength, 0.0)

        existing = picked.get(sym)
        if (existing is None) or (raw_exec > _to_float(existing.get("_raw_exec"))):
            signal_price = _to_float(r.get("signal_price"), 0.0)
            current_price = _to_float(r.get("current_price"), 0.0)
            if current_price <= 0:
                current_price = signal_price
            sig_date_txt = str(r.get("signal_date") or latest_date)
            age_days = int(_to_float(r.get("days_since_signal"), 0))
            if latest_dt is not None:
                try:
                    sig_dt = datetime.strptime(sig_date_txt, "%Y-%m-%d")
                    age_days = max((latest_dt - sig_dt).days, 0)
                except Exception:
                    pass
            px_change_pct = 0.0
            px_change_abs = 0.0
            if signal_price > 0 and current_price > 0:
                px_change_abs = current_price - signal_price
                px_change_pct = (px_change_abs / signal_price) * 100.0
            picked[sym] = {
                "信号日期": sig_date_txt,
                "symbol": sym,
                "命中策略数": len(hit_strategies),
                "命中策略": "、".join(sorted(set(hit_strategies))),
                "策略权重合计(%)": round(weight_sum, 1),
                "距信号天数": int(age_days),
                "信号价": round(signal_price, 3) if signal_price > 0 else None,
                "现价": round(current_price, 3) if current_price > 0 else None,
                "价格变化($)": round(px_change_abs, 3),
                "价格变化(%)": round(px_change_pct, 2),
                "blue_daily": round(day_blue, 1),
                "blue_weekly": round(week_blue, 1),
                "_raw_exec": raw_exec,
            }

    rows_out = list(picked.values())
    rows_out.sort(key=lambda x: x.get("_raw_exec", 0.0), reverse=True)
    if not rows_out:
        return []

    max_raw = max(_to_float(x.get("_raw_exec")) for x in rows_out)
    sum_raw = sum(_to_float(x.get("_raw_exec")) for x in rows_out)
    safe_capital = max(_to_float(total_capital, 100000.0), 0.0)

    for r in rows_out:
        raw = _to_float(r.get("_raw_exec"))
        score_100 = (raw / max_raw * 100.0) if max_raw > 0 else 0.0
        pos_pct = (raw / sum_raw * 100.0) if sum_raw > 0 else 0.0
        r["综合执行分(0-100)"] = round(score_100, 1)
        r["建议仓位(%)"] = round(pos_pct, 1)
        r["建议金额($)"] = round(safe_capital * pos_pct / 100.0, 2)
        r.pop("_raw_exec", None)

    return rows_out[: int(top_n)]
