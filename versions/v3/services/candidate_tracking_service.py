#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
候选信号持续追踪服务
====================
记录信号触发当日快照，并持续更新当前表现，支持组合标签统计。
"""
from __future__ import annotations

import json
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

from db.database import get_db, init_db, get_scanned_dates, query_scan_results


CORE_TAGS = [
    "DAY_BLUE",
    "WEEK_BLUE",
    "MONTH_BLUE",
    "DAY_HEIMA",
    "WEEK_HEIMA",
    "MONTH_HEIMA",
    "CHIP_DENSE",
    "CHIP_BREAKOUT",
    "CHIP_OVERHANG",
]

DEFAULT_TAG_RULES = {
    "day_blue_min": 100.0,
    "week_blue_min": 80.0,
    "month_blue_min": 60.0,
    "chip_dense_profit_ratio_min": 0.7,
    "chip_breakout_profit_ratio_min": 0.9,
    "chip_overhang_profit_ratio_max": 0.3,
}


def _ensure_tracking_table() -> None:
    """自愈建表：兼容旧数据库未迁移场景。"""
    # 先触发全局初始化（包含其它依赖表）
    try:
        init_db()
    except Exception:
        pass

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                market VARCHAR(10) DEFAULT 'US',
                signal_date DATE NOT NULL,
                source VARCHAR(50) DEFAULT 'daily_scan',
                signal_price REAL NOT NULL,
                current_price REAL,
                pnl_pct REAL DEFAULT 0,
                days_since_signal INTEGER DEFAULT 0,
                first_positive_day INTEGER,
                max_up_pct REAL DEFAULT 0,
                max_drawdown_pct REAL DEFAULT 0,
                pnl_d1 REAL,
                pnl_d3 REAL,
                pnl_d5 REAL,
                pnl_d10 REAL,
                pnl_d20 REAL,
                cap_category VARCHAR(30),
                industry VARCHAR(200),
                signal_tags TEXT,
                blue_daily REAL,
                blue_weekly REAL,
                blue_monthly REAL,
                heima_daily BOOLEAN,
                heima_weekly BOOLEAN,
                heima_monthly BOOLEAN,
                vp_rating VARCHAR(20),
                profit_ratio REAL,
                status VARCHAR(20) DEFAULT 'tracking',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, market, signal_date)
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_date ON candidate_tracking(signal_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_market ON candidate_tracking(market)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_status ON candidate_tracking(status)")


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) == 1
    if isinstance(v, bytes):
        return v == b"\x01"
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "t")
    return False


def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _parse_date(text: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(text), "%Y-%m-%d")
    except Exception:
        return None


def derive_signal_tags(row: Dict, rules: Optional[Dict] = None) -> List[str]:
    """从扫描记录推导原子标签。"""
    cfg = dict(DEFAULT_TAG_RULES)
    if rules:
        cfg.update(rules)

    tags: List[str] = []

    day_blue = _to_float(row.get("blue_daily"))
    week_blue = _to_float(row.get("blue_weekly"))
    month_blue = _to_float(row.get("blue_monthly"))
    profit_ratio = _to_float(row.get("profit_ratio"))
    vp_rating = str(row.get("vp_rating", "") or "")

    if day_blue >= _to_float(cfg.get("day_blue_min"), 100.0):
        tags.append("DAY_BLUE")
    if week_blue >= _to_float(cfg.get("week_blue_min"), 80.0):
        tags.append("WEEK_BLUE")
    if month_blue >= _to_float(cfg.get("month_blue_min"), 60.0):
        tags.append("MONTH_BLUE")

    if _to_bool(row.get("heima_daily")):
        tags.append("DAY_HEIMA")
    if _to_bool(row.get("heima_weekly")):
        tags.append("WEEK_HEIMA")
    if _to_bool(row.get("heima_monthly")):
        tags.append("MONTH_HEIMA")

    if vp_rating in ("Good", "Excellent") or profit_ratio >= _to_float(cfg.get("chip_dense_profit_ratio_min"), 0.7):
        tags.append("CHIP_DENSE")
    if vp_rating == "Excellent" or profit_ratio >= _to_float(cfg.get("chip_breakout_profit_ratio_min"), 0.9):
        tags.append("CHIP_BREAKOUT")
    if vp_rating == "Poor" or (0 < profit_ratio <= _to_float(cfg.get("chip_overhang_profit_ratio_max"), 0.3)):
        tags.append("CHIP_OVERHANG")

    return sorted(set(tags))


def _upsert_snapshot(row: Dict, signal_date: str, market: str, source: str, rules: Optional[Dict] = None) -> None:
    symbol = (row.get("symbol") or "").strip()
    if not symbol:
        return

    tags = derive_signal_tags(row, rules=rules)
    if not tags:
        return

    snapshot_price = _to_float(row.get("price"), 0.0)
    if snapshot_price <= 0:
        return

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO candidate_tracking (
                symbol, market, signal_date, source,
                signal_price, current_price, pnl_pct, days_since_signal,
                first_positive_day, max_up_pct, max_drawdown_pct,
                cap_category, industry, signal_tags,
                blue_daily, blue_weekly, blue_monthly,
                heima_daily, heima_weekly, heima_monthly,
                vp_rating, profit_ratio, status,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, NULL, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracking', CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, market, signal_date) DO UPDATE SET
                source = excluded.source,
                signal_price = excluded.signal_price,
                cap_category = excluded.cap_category,
                industry = excluded.industry,
                signal_tags = excluded.signal_tags,
                blue_daily = excluded.blue_daily,
                blue_weekly = excluded.blue_weekly,
                blue_monthly = excluded.blue_monthly,
                heima_daily = excluded.heima_daily,
                heima_weekly = excluded.heima_weekly,
                heima_monthly = excluded.heima_monthly,
                vp_rating = excluded.vp_rating,
                profit_ratio = excluded.profit_ratio,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                symbol,
                market,
                signal_date,
                source,
                snapshot_price,
                snapshot_price,
                row.get("cap_category"),
                row.get("industry"),
                json.dumps(tags, ensure_ascii=False),
                _to_float(row.get("blue_daily")),
                _to_float(row.get("blue_weekly")),
                _to_float(row.get("blue_monthly")),
                int(_to_bool(row.get("heima_daily"))),
                int(_to_bool(row.get("heima_weekly"))),
                int(_to_bool(row.get("heima_monthly"))),
                row.get("vp_rating"),
                _to_float(row.get("profit_ratio")),
            ),
        )


def capture_daily_candidates(
    rows: Sequence[Dict],
    market: str,
    signal_date: str,
    source: str = "daily_scan",
    rules: Optional[Dict] = None,
) -> int:
    """批量写入当日候选追踪快照。"""
    _ensure_tracking_table()
    if not rows:
        return 0

    inserted = 0
    for row in rows:
        before = inserted
        _upsert_snapshot(row=row, signal_date=signal_date, market=market, source=source, rules=rules)
        inserted = before + 1
    return inserted


def backfill_candidates_from_scan_history(
    market: str,
    recent_days: int = 30,
    max_per_day: int = 300,
    rules: Optional[Dict] = None,
) -> int:
    """从历史扫描结果回填候选追踪，解决历史信号未入库问题。"""
    _ensure_tracking_table()

    dates = get_scanned_dates(market=market) or []
    if not dates:
        return 0

    target_dates = dates[: max(1, int(recent_days))]
    total = 0
    for d in target_dates:
        rows = query_scan_results(scan_date=d, market=market, limit=int(max_per_day)) or []
        if not rows:
            continue
        total += capture_daily_candidates(
            rows=rows,
            market=market,
            signal_date=d,
            source="history_backfill",
            rules=rules,
        )
    return total


def _get_price_series(symbol: str, market: str, signal_date: str) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT scan_date, price
            FROM scan_results
            WHERE symbol = ? AND market = ? AND scan_date >= ? AND price IS NOT NULL AND price > 0
            ORDER BY scan_date ASC
            """,
            (symbol, market, signal_date),
        )
        return [dict(x) for x in cursor.fetchall()]


def _horizon_return(prices: List[float], horizon: int) -> Optional[float]:
    if len(prices) <= horizon:
        return None
    base = prices[0]
    if not base:
        return None
    return (prices[horizon] / base - 1.0) * 100.0


def refresh_candidate_tracking(market: Optional[str] = None, max_rows: int = 2000) -> int:
    """刷新候选追踪当前表现。"""
    _ensure_tracking_table()
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT id, symbol, market, signal_date, signal_price
            FROM candidate_tracking
            WHERE status = 'tracking'
        """
        params: List = []
        if market:
            query += " AND market = ?"
            params.append(market)
        query += " ORDER BY signal_date DESC LIMIT ?"
        params.append(int(max_rows))
        cursor.execute(query, params)
        rows = [dict(x) for x in cursor.fetchall()]

    updated = 0
    for row in rows:
        symbol = row["symbol"]
        mk = row["market"]
        signal_date = row["signal_date"]
        signal_price = _to_float(row["signal_price"], 0.0)
        if signal_price <= 0:
            continue

        series = _get_price_series(symbol=symbol, market=mk, signal_date=signal_date)
        if not series:
            continue

        prices = [_to_float(x["price"], 0.0) for x in series if _to_float(x["price"], 0.0) > 0]
        if not prices:
            continue
        current_price = prices[-1]

        first_positive_day = None
        for idx, p in enumerate(prices):
            if p > signal_price:
                first_positive_day = idx
                break

        max_price = max(prices)
        min_price = min(prices)
        pnl_pct = (current_price / signal_price - 1.0) * 100.0
        max_up_pct = (max_price / signal_price - 1.0) * 100.0
        max_drawdown_pct = (min_price / signal_price - 1.0) * 100.0
        days_since_signal = max(len(prices) - 1, 0)

        pnl_d1 = _horizon_return(prices, 1)
        pnl_d3 = _horizon_return(prices, 3)
        pnl_d5 = _horizon_return(prices, 5)
        pnl_d10 = _horizon_return(prices, 10)
        pnl_d20 = _horizon_return(prices, 20)

        status = "tracking"
        if days_since_signal >= 20:
            status = "validated" if pnl_pct > 0 else "underperform"

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE candidate_tracking
                SET current_price = ?, pnl_pct = ?, days_since_signal = ?,
                    first_positive_day = COALESCE(first_positive_day, ?),
                    max_up_pct = ?, max_drawdown_pct = ?,
                    pnl_d1 = ?, pnl_d3 = ?, pnl_d5 = ?, pnl_d10 = ?, pnl_d20 = ?,
                    status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    current_price,
                    pnl_pct,
                    days_since_signal,
                    first_positive_day,
                    max_up_pct,
                    max_drawdown_pct,
                    pnl_d1,
                    pnl_d3,
                    pnl_d5,
                    pnl_d10,
                    pnl_d20,
                    status,
                    row["id"],
                ),
            )
        updated += 1
    return updated


def get_candidate_tracking_rows(market: Optional[str] = None, days_back: int = 180) -> List[Dict]:
    _ensure_tracking_table()
    query = """
        SELECT *
        FROM candidate_tracking
        WHERE 1=1
    """
    params: List = []
    if market:
        query += " AND market = ?"
        params.append(market)
    if days_back > 0:
        # 避免在不同 SQLite 环境中使用 date('now', ?) 参数导致兼容性错误
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=int(days_back))).strftime("%Y-%m-%d")
        query += " AND signal_date >= ?"
        params.append(cutoff)
    query += " ORDER BY signal_date DESC, pnl_pct DESC"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(x) for x in cursor.fetchall()]

    for row in rows:
        raw_tags = row.get("signal_tags")
        try:
            row["signal_tags_list"] = json.loads(raw_tags) if raw_tags else []
        except Exception:
            row["signal_tags_list"] = []
    return rows


def reclassify_tracking_tags(market: Optional[str] = None, rules: Optional[Dict] = None, max_rows: int = 5000) -> int:
    """按新规则重算候选追踪标签。"""
    _ensure_tracking_table()
    query = """
        SELECT id, blue_daily, blue_weekly, blue_monthly,
               heima_daily, heima_weekly, heima_monthly,
               vp_rating, profit_ratio
        FROM candidate_tracking
        WHERE 1=1
    """
    params: List = []
    if market:
        query += " AND market = ?"
        params.append(market)
    query += " ORDER BY signal_date DESC LIMIT ?"
    params.append(int(max_rows))

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(x) for x in cursor.fetchall()]

    updated = 0
    for row in rows:
        tags = derive_signal_tags(row, rules=rules)
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE candidate_tracking
                SET signal_tags = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(tags, ensure_ascii=False), row["id"]),
            )
        updated += 1
    return updated


def _iter_combo_keys(tags: Iterable[str], max_size: int = 4) -> Iterable[str]:
    uniq = sorted(set([t for t in tags if t in CORE_TAGS]))
    if not uniq:
        return []
    combos: List[str] = []
    for size in range(1, min(max_size, len(uniq)) + 1):
        for item in combinations(uniq, size):
            combos.append("+".join(item))
    return combos


def build_combo_stats(rows: Sequence[Dict], min_samples: int = 5) -> List[Dict]:
    agg: Dict[str, Dict] = {}

    for row in rows:
        tags = row.get("signal_tags_list") or []
        for key in _iter_combo_keys(tags, max_size=4):
            bucket = agg.setdefault(
                key,
                {
                    "combo": key,
                    "sample_count": 0,
                    "wins": 0,
                    "pnl_sum": 0.0,
                    "days_to_positive": [],
                    "d1_wins": 0,
                    "d1_count": 0,
                    "d3_wins": 0,
                    "d3_count": 0,
                    "d5_wins": 0,
                    "d5_count": 0,
                    "d10_wins": 0,
                    "d10_count": 0,
                    "d20_wins": 0,
                    "d20_count": 0,
                },
            )
            bucket["sample_count"] += 1
            pnl = _to_float(row.get("pnl_pct"), 0.0)
            bucket["pnl_sum"] += pnl
            if pnl > 0:
                bucket["wins"] += 1

            fp = row.get("first_positive_day")
            if fp is not None:
                try:
                    bucket["days_to_positive"].append(int(fp))
                except Exception:
                    pass

            for h in [1, 3, 5, 10, 20]:
                val = row.get(f"pnl_d{h}")
                if val is not None:
                    bucket[f"d{h}_count"] += 1
                    if _to_float(val, 0.0) > 0:
                        bucket[f"d{h}_wins"] += 1

    out = []
    for _, bucket in agg.items():
        sample = bucket["sample_count"]
        if sample < min_samples:
            continue
        d2p = sorted(bucket["days_to_positive"])
        median_d2p = d2p[len(d2p) // 2] if d2p else None
        out.append(
            {
                "组合": bucket["combo"],
                "样本数": sample,
                "当前胜率(%)": round(bucket["wins"] / sample * 100, 1),
                "当前平均收益(%)": round(bucket["pnl_sum"] / sample, 2),
                "D+1胜率(%)": round(bucket["d1_wins"] / bucket["d1_count"] * 100, 1) if bucket["d1_count"] else None,
                "D+3胜率(%)": round(bucket["d3_wins"] / bucket["d3_count"] * 100, 1) if bucket["d3_count"] else None,
                "D+5胜率(%)": round(bucket["d5_wins"] / bucket["d5_count"] * 100, 1) if bucket["d5_count"] else None,
                "D+10胜率(%)": round(bucket["d10_wins"] / bucket["d10_count"] * 100, 1) if bucket["d10_count"] else None,
                "D+20胜率(%)": round(bucket["d20_wins"] / bucket["d20_count"] * 100, 1) if bucket["d20_count"] else None,
                "首次转正中位天数": median_d2p,
            }
        )

    out.sort(key=lambda x: (x["当前胜率(%)"], x["当前平均收益(%)"]), reverse=True)
    return out


def build_segment_stats(rows: Sequence[Dict], by: str = "cap_category") -> List[Dict]:
    stats: Dict[str, Dict] = {}
    for row in rows:
        key = str(row.get(by) or "Unknown")
        bucket = stats.setdefault(key, {"segment": key, "sample": 0, "wins": 0, "pnl_sum": 0.0})
        bucket["sample"] += 1
        pnl = _to_float(row.get("pnl_pct"), 0.0)
        bucket["pnl_sum"] += pnl
        if pnl > 0:
            bucket["wins"] += 1

    out = []
    for _, b in stats.items():
        sample = b["sample"]
        out.append(
            {
                "分组": b["segment"],
                "样本数": sample,
                "胜率(%)": round(b["wins"] / sample * 100, 1) if sample else 0,
                "平均收益(%)": round(b["pnl_sum"] / sample, 2) if sample else 0,
            }
        )
    out.sort(key=lambda x: x["样本数"], reverse=True)
    return out
