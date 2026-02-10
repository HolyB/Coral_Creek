#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一可交易评估 + 策略组合层分配器
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

from services.candidate_tracking_service import evaluate_exit_rule


STRATEGY_TAG_DEFS = {
    "blue_day_week": {"name": "日周共振", "tags_any": ["DAY_BLUE", "WEEK_BLUE"]},
    "blue_triple": {"name": "三线共振", "tags_all": ["DAY_BLUE", "WEEK_BLUE", "MONTH_BLUE"]},
    "heima": {"name": "黑马策略", "tags_any": ["DAY_HEIMA", "WEEK_HEIMA", "MONTH_HEIMA"]},
    "chip": {"name": "筹码突破", "tags_any": ["CHIP_BREAKOUT", "CHIP_DENSE"]},
    "defensive": {"name": "防守均衡", "tags_any": ["DAY_BLUE", "CHIP_DENSE"]},
}


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
    if n > 0 and curve[-1] > 0:
        ann = float(curve[-1] ** (trades_per_year / n) - 1.0)
    else:
        ann = 0.0

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
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "profit_factor": round(profit_factor, 2),
        "turnover_per_year": round(trades_per_year, 2),
    }


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
            }
        )
    out.sort(key=lambda x: (x.get("sharpe", 0.0), x.get("ann_return_pct", 0.0)), reverse=True)
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
) -> List[Dict]:
    if not rows or not weight_rows:
        return []

    latest_date = max(str(r.get("signal_date") or "") for r in rows if r.get("signal_date"))
    latest_rows = [r for r in rows if str(r.get("signal_date") or "") == latest_date]

    key2cfg = STRATEGY_TAG_DEFS
    picked: Dict[Tuple[str, str], Dict] = {}
    w_map = {str(w.get("strategy_key")): _to_float(w.get("建议权重(%)")) for w in weight_rows}

    for r in latest_rows:
        for key, cfg in key2cfg.items():
            if key not in w_map:
                continue
            if not _match_strategy(r, cfg):
                continue
            sym = str(r.get("symbol") or "")
            if not sym:
                continue
            score = 0.7 * _to_float(r.get("blue_daily")) + 0.3 * _to_float(r.get("blue_weekly"))
            k = (sym, key)
            if (k not in picked) or (score > _to_float(picked[k].get("_score"))):
                picked[k] = {
                    "日期": latest_date,
                    "symbol": sym,
                    "策略": cfg.get("name", key),
                    "建议权重(%)": round(w_map.get(key, 0.0), 1),
                    "blue_daily": round(_to_float(r.get("blue_daily")), 1),
                    "blue_weekly": round(_to_float(r.get("blue_weekly")), 1),
                    "当前收益(%)": round(_to_float(r.get("pnl_pct")), 2),
                    "_score": score,
                }

    rows_out = list(picked.values())
    rows_out.sort(key=lambda x: (x.get("建议权重(%)", 0.0), x.get("_score", 0.0)), reverse=True)
    for r in rows_out:
        r.pop("_score", None)
    return rows_out[: int(top_n)]
