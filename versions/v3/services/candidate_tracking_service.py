#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
候选信号持续追踪服务
====================
记录信号触发当日快照，并持续更新当前表现，支持组合标签统计。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np

from db.database import get_db, init_db, get_scanned_dates, query_scan_results
from db.stock_history import get_history_db_path


CORE_TAGS = [
    "DAY_BLUE",
    "WEEK_BLUE",
    "MONTH_BLUE",
    "DAY_HEIMA",
    "WEEK_HEIMA",
    "MONTH_HEIMA",
    "DAY_JUEDI",
    "WEEK_JUEDI",
    "MONTH_JUEDI",
    "CHIP_DENSE",
    "CHIP_BREAKOUT",
    "CHIP_OVERHANG",
    "DUOKONGWANG_BUY",
    "DUOKONGWANG_SELL",
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
                first_nonpositive_after_positive_day INTEGER,
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
                juedi_daily BOOLEAN,
                juedi_weekly BOOLEAN,
                juedi_monthly BOOLEAN,
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
        try:
            cursor.execute("SELECT first_nonpositive_after_positive_day FROM candidate_tracking LIMIT 1")
        except Exception:
            cursor.execute("ALTER TABLE candidate_tracking ADD COLUMN first_nonpositive_after_positive_day INTEGER")
        for col_name in ["juedi_daily", "juedi_weekly", "juedi_monthly"]:
            try:
                cursor.execute(f"ALTER TABLE candidate_tracking ADD COLUMN {col_name} BOOLEAN")
            except Exception:
                pass


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


def _pick_first_value(row: Dict, keys: Sequence[str]):
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _pick_bool(row: Dict, keys: Sequence[str]) -> bool:
    return _to_bool(_pick_first_value(row, keys))


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

    if _pick_bool(row, ["heima_daily", "Heima_Daily", "is_heima", "Is_Heima"]):
        tags.append("DAY_HEIMA")
    if _pick_bool(row, ["heima_weekly", "Heima_Weekly"]):
        tags.append("WEEK_HEIMA")
    if _pick_bool(row, ["heima_monthly", "Heima_Monthly"]):
        tags.append("MONTH_HEIMA")
    if _pick_bool(row, ["juedi_daily", "Juedi_Daily", "is_juedi", "Is_Juedi"]):
        tags.append("DAY_JUEDI")
    if _pick_bool(row, ["juedi_weekly", "Juedi_Weekly"]):
        tags.append("WEEK_JUEDI")
    if _pick_bool(row, ["juedi_monthly", "Juedi_Monthly"]):
        tags.append("MONTH_JUEDI")

    if vp_rating in ("Good", "Excellent") or profit_ratio >= _to_float(cfg.get("chip_dense_profit_ratio_min"), 0.7):
        tags.append("CHIP_DENSE")
    if vp_rating == "Excellent" or profit_ratio >= _to_float(cfg.get("chip_breakout_profit_ratio_min"), 0.9):
        tags.append("CHIP_BREAKOUT")
    if vp_rating == "Poor" or (0 < profit_ratio <= _to_float(cfg.get("chip_overhang_profit_ratio_max"), 0.3)):
        tags.append("CHIP_OVERHANG")

    if _pick_bool(row, ["duokongwang_buy", "Duokongwang_Buy"]):
        tags.append("DUOKONGWANG_BUY")
    if _pick_bool(row, ["duokongwang_sell", "Duokongwang_Sell"]):
        tags.append("DUOKONGWANG_SELL")

    return sorted(set(tags))


def _upsert_snapshot(row: Dict, signal_date: str, market: str, source: str, rules: Optional[Dict] = None) -> bool:
    symbol = (row.get("symbol") or "").strip()
    if not symbol:
        return False

    tags = derive_signal_tags(row, rules=rules)
    if not tags:
        return False

    snapshot_price = _to_float(_pick_first_value(row, ["price", "Price"]), 0.0)
    if snapshot_price <= 0:
        return False

    heima_daily = int(_pick_bool(row, ["heima_daily", "Heima_Daily", "is_heima", "Is_Heima"]))
    heima_weekly = int(_pick_bool(row, ["heima_weekly", "Heima_Weekly"]))
    heima_monthly = int(_pick_bool(row, ["heima_monthly", "Heima_Monthly"]))
    juedi_daily = int(_pick_bool(row, ["juedi_daily", "Juedi_Daily", "is_juedi", "Is_Juedi"]))
    juedi_weekly = int(_pick_bool(row, ["juedi_weekly", "Juedi_Weekly"]))
    juedi_monthly = int(_pick_bool(row, ["juedi_monthly", "Juedi_Monthly"]))

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
                juedi_daily, juedi_weekly, juedi_monthly,
                vp_rating, profit_ratio, status,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, NULL, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracking', CURRENT_TIMESTAMP)
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
                juedi_daily = excluded.juedi_daily,
                juedi_weekly = excluded.juedi_weekly,
                juedi_monthly = excluded.juedi_monthly,
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
                heima_daily,
                heima_weekly,
                heima_monthly,
                juedi_daily,
                juedi_weekly,
                juedi_monthly,
                row.get("vp_rating"),
                _to_float(row.get("profit_ratio")),
            ),
        )
        return cursor.rowcount > 0


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
        if _upsert_snapshot(row=row, signal_date=signal_date, market=market, source=source, rules=rules):
            inserted += 1
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
    # 1) 优先从全量历史库读取（覆盖非候选日，更接近真实连续K线）
    try:
        hist_conn = sqlite3.connect(get_history_db_path())
        hist_conn.row_factory = sqlite3.Row
        hcur = hist_conn.cursor()
        hcur.execute(
            """
            SELECT trade_date as scan_date, close as price, high as day_high, low as day_low, close as day_close
            FROM stock_history
            WHERE symbol = ? AND market = ? AND trade_date >= ?
              AND close IS NOT NULL AND close > 0
            ORDER BY trade_date ASC
            """,
            (symbol, market, signal_date),
        )
        hist_rows = [dict(x) for x in hcur.fetchall()]
        hist_conn.close()
        if hist_rows:
            return hist_rows
    except Exception:
        try:
            hist_conn.close()
        except Exception:
            pass

    # 2) 回退 scan_results（仅候选日快照）
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT scan_date, price, day_high, day_low, day_close
                FROM scan_results
                WHERE symbol = ? AND market = ? AND scan_date >= ? AND price IS NOT NULL AND price > 0
                ORDER BY scan_date ASC
                """,
                (symbol, market, signal_date),
            )
            rows = [dict(x) for x in cursor.fetchall()]
            if rows:
                return rows
        except Exception:
            pass
        # 兼容旧表结构：没有 day_high/day_low/day_close 时回退
        cursor.execute(
            """
            SELECT scan_date, price
            FROM scan_results
            WHERE symbol = ? AND market = ? AND scan_date >= ? AND price IS NOT NULL AND price > 0
            ORDER BY scan_date ASC
            """,
            (symbol, market, signal_date),
        )
        rows = [dict(x) for x in cursor.fetchall()]
        if rows:
            return rows

    # 3) 最后兜底：在线拉取从 signal_date 至今的日线（解决仅快照导致收益长期为0）
    try:
        from datetime import datetime as _dt
        from data_fetcher import get_stock_data

        try:
            sig_dt = _dt.strptime(str(signal_date), "%Y-%m-%d")
            days_needed = max(90, (_dt.now() - sig_dt).days + 30)
        except Exception:
            days_needed = 180
        days_needed = max(60, min(int(days_needed), 1200))

        df = get_stock_data(symbol, market=market, days=days_needed)
        if df is not None and len(df) > 0:
            out = []
            for idx, r in df.iterrows():
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                if d < str(signal_date):
                    continue
                close_px = _to_float(r.get("Close"), 0.0)
                if close_px <= 0:
                    continue
                out.append(
                    {
                        "scan_date": d,
                        "price": close_px,
                        "day_high": _to_float(r.get("High"), close_px),
                        "day_low": _to_float(r.get("Low"), close_px),
                        "day_close": close_px,
                    }
                )
            if out:
                return out
    except Exception:
        pass

    return []


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

        prices = []
        for x in series:
            px = _to_float(x.get("day_close"), 0.0)
            if px <= 0:
                px = _to_float(x.get("price"), 0.0)
            if px > 0:
                prices.append(px)
        if not prices:
            continue
        current_price = prices[-1]

        first_positive_day = None
        for idx, p in enumerate(prices):
            if p > signal_price:
                first_positive_day = idx
                break
        first_nonpositive_after_positive_day = None
        if first_positive_day is not None:
            for idx in range(first_positive_day + 1, len(prices)):
                if prices[idx] <= signal_price:
                    first_nonpositive_after_positive_day = idx
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
                    first_nonpositive_after_positive_day = COALESCE(first_nonpositive_after_positive_day, ?),
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
                    first_nonpositive_after_positive_day,
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
               juedi_daily, juedi_weekly, juedi_monthly,
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


def evaluate_exit_rule(
    rows: Sequence[Dict],
    rule_name: str = "fixed_10d",
    take_profit_pct: float = 10.0,
    stop_loss_pct: float = 6.0,
    max_hold_days: int = 20,
    max_rows: int = 20000,
) -> Dict:
    """
    规则平仓评估：
    - fixed_5d / fixed_10d / fixed_20d
    - tp_sl_time: 先触发止盈/止损，否则 max_hold_days 强平
    - kdj_dead_cross: KDJ(收盘价近似)出现死叉或过热回落即退出
    - top_divergence_guard: 价格创新高但动量不创新高（顶背离近似）时保护退出
    - duokongwang_sell: 多空王卖出（通道转弱 + KDJ/RSI过热回落 + 九转衰竭）
    """
    use_rows = list(rows or [])[: int(max_rows)]
    if not use_rows:
        return {
            "rule_name": rule_name,
            "sample": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "avg_exit_day": None,
            "avg_first_profit_day": None,
            "avg_first_nonprofit_day": None,
            "avg_profit_span_days": None,
        }

    exits = []

    def _prepare_ohlc(series_: List[Dict]) -> Dict[str, List[float]]:
        closes_: List[float] = []
        highs_: List[float] = []
        lows_: List[float] = []
        for item in series_:
            close_px = _to_float(item.get("day_close"), 0.0)
            if close_px <= 0:
                close_px = _to_float(item.get("price"), 0.0)
            if close_px <= 0:
                continue
            high_px = _to_float(item.get("day_high"), close_px)
            low_px = _to_float(item.get("day_low"), close_px)
            if high_px <= 0:
                high_px = close_px
            if low_px <= 0:
                low_px = close_px
            high_px = max(high_px, close_px)
            low_px = min(low_px, close_px)
            closes_.append(close_px)
            highs_.append(high_px)
            lows_.append(low_px)
        return {"close": closes_, "high": highs_, "low": lows_}

    def _calc_kdj_ohlc(highs_: List[float], lows_: List[float], closes_: List[float], period: int = 9) -> List[Dict]:
        # 标准 KDJ: RSV = (C-LLV(L,n)) / (HHV(H,n)-LLV(L,n)) * 100
        out_: List[Dict] = []
        k_prev, d_prev = 50.0, 50.0
        for i, c in enumerate(closes_):
            s = max(0, i - period + 1)
            lo = min(lows_[s : i + 1])
            hi = max(highs_[s : i + 1])
            if hi <= lo:
                rsv = 50.0
            else:
                rsv = (c - lo) / (hi - lo) * 100.0
            k = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * rsv
            d = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k
            j = 3.0 * k - 2.0 * d
            out_.append({"k": k, "d": d, "j": j})
            k_prev, d_prev = k, d
        return out_

    def _ema(values: List[float], period: int) -> List[float]:
        if not values:
            return []
        alpha = 2.0 / (period + 1.0)
        out = [float(values[0])]
        for v in values[1:]:
            out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
        return out

    def _sma_cn(values: List[float], n: int, m: int = 1) -> List[float]:
        # 通达信SMA: Y=(M*X+(N-M)*Y')/N
        if not values:
            return []
        out = [float(values[0])]
        for x in values[1:]:
            out.append((m * float(x) + (n - m) * out[-1]) / float(n))
        return out

    def _calc_duokongwang_sell_flags(highs_: List[float], lows_: List[float], closes_: List[float]) -> List[bool]:
        n = len(closes_)
        if n == 0:
            return []

        # 近似开盘（无开盘价存储时用前收代替）
        opens_ = [closes_[0]] + closes_[:-1]

        up = _ema(highs_, 13)
        dw = _ema(lows_, 13)

        # KDJ(14,3,3)
        rsv = []
        for i in range(n):
            s = max(0, i - 14 + 1)
            llv = min(lows_[s : i + 1])
            hhv = max(highs_[s : i + 1])
            if hhv <= llv:
                rsv.append(50.0)
            else:
                rsv.append((closes_[i] - llv) / (hhv - llv) * 100.0)
        k = _sma_cn(rsv, 3, 1)
        d = _sma_cn(k, 3, 1)
        j = [3.0 * k[i] - 2.0 * d[i] for i in range(n)]

        # RSI2(TGN1=9, 通达信口径)
        lc = [closes_[0]] + closes_[:-1]
        up_move = [max(closes_[i] - lc[i], 0.0) for i in range(n)]
        abs_move = [abs(closes_[i] - lc[i]) for i in range(n)]
        rsi_num = _sma_cn(up_move, 9, 1)
        rsi_den = _sma_cn(abs_move, 9, 1)
        rsi2 = [((rsi_num[i] / rsi_den[i]) * 100.0 if rsi_den[i] > 1e-12 else 50.0) for i in range(n)]

        # 九转上升计数: A1=C>REF(C,4)
        nt = [0] * n
        for i in range(n):
            a1 = i >= 4 and closes_[i] > closes_[i - 4]
            nt[i] = (nt[i - 1] + 1) if (a1 and i > 0) else (1 if a1 else 0)

        flags = [False] * n
        for i in range(1, n):
            cond = (closes_[i] > opens_[i] and (opens_[i] > up[i] or closes_[i] < dw[i])) or (
                closes_[i] < opens_[i] and (opens_[i] < dw[i] or closes_[i] > up[i])
            )
            cond2 = up[i] < up[i - 1] and dw[i] < dw[i - 1]  # 通道下行
            channel_weak = cond and cond2

            kdj_overheat_fade = (j[i - 1] >= 100.0 and j[i] < 95.0) or (j[i - 1] >= 90.0 and j[i] < j[i - 1] - 8.0)
            rsi_overbought_turn = rsi2[i - 1] >= 79.0 and rsi2[i] < 80.0
            nine_turn_exhaust = nt[i - 1] >= 9 and closes_[i] < closes_[i - 1]

            flags[i] = bool(channel_weak or kdj_overheat_fade or rsi_overbought_turn or nine_turn_exhaust)
        return flags

    for row in use_rows:
        signal_price = _to_float(row.get("signal_price"), 0.0)
        if signal_price <= 0:
            continue
        symbol = str(row.get("symbol") or "")
        market = str(row.get("market") or "US")
        signal_date = str(row.get("signal_date") or "")

        series = _get_price_series(symbol=symbol, market=market, signal_date=signal_date)
        ohlc = _prepare_ohlc(series)
        closes = ohlc["close"]
        highs = ohlc["high"]
        lows = ohlc["low"]
        if not closes:
            continue

        exit_day = None
        exit_ret = None

        if rule_name == "fixed_5d":
            day = min(5, len(closes) - 1)
            exit_day = day
            exit_ret = (closes[day] / signal_price - 1.0) * 100.0
        elif rule_name == "fixed_20d":
            day = min(20, len(closes) - 1)
            exit_day = day
            exit_ret = (closes[day] / signal_price - 1.0) * 100.0
        elif rule_name == "tp_sl_time":
            tp = signal_price * (1.0 + take_profit_pct / 100.0)
            sl = signal_price * (1.0 - stop_loss_pct / 100.0)
            limit_day = min(int(max_hold_days), len(closes) - 1)
            for i in range(1, limit_day + 1):
                h = highs[i]
                l = lows[i]
                c = closes[i]
                # 同日同时触发时，按更保守口径先记止损
                if l <= sl:
                    exit_day = i
                    exit_ret = (sl / signal_price - 1.0) * 100.0
                    break
                if h >= tp:
                    exit_day = i
                    exit_ret = (tp / signal_price - 1.0) * 100.0
                    break
                # 未触发时以当日收盘继续持有
                _ = c
            if exit_day is None:
                exit_day = limit_day
                exit_ret = (closes[limit_day] / signal_price - 1.0) * 100.0
        elif rule_name == "kdj_dead_cross":
            # 顶级交易员口径：过热区死叉优先离场，避免回撤吃掉利润（标准OHLC KDJ）
            kdj = _calc_kdj_ohlc(highs, lows, closes)
            limit_day = min(int(max_hold_days), len(closes) - 1)
            for i in range(2, limit_day + 1):
                k0, d0 = kdj[i - 1]["k"], kdj[i - 1]["d"]
                k1, d1 = kdj[i]["k"], kdj[i]["d"]
                j1 = kdj[i]["j"]
                # 死叉: 前一日K>=D，当日K<D；或 J>95 后快速回落
                dead_cross = (k0 >= d0 and k1 < d1)
                overheat_fade = (j1 > 95.0) or (i >= 2 and kdj[i - 1]["j"] > 95.0 and j1 < kdj[i - 1]["j"] - 8.0)
                if dead_cross and (k1 > 60.0 or d1 > 60.0):
                    exit_day = i
                    exit_ret = (closes[i] / signal_price - 1.0) * 100.0
                    break
                if overheat_fade:
                    exit_day = i
                    exit_ret = (closes[i] / signal_price - 1.0) * 100.0
                    break
            if exit_day is None:
                exit_day = limit_day
                exit_ret = (closes[limit_day] / signal_price - 1.0) * 100.0
        elif rule_name == "top_divergence_guard":
            # 顶背离近似：价格新高但J值不创新高，且出现回落时保护退出
            kdj = _calc_kdj_ohlc(highs, lows, closes)
            limit_day = min(int(max_hold_days), len(closes) - 1)
            peak_price = highs[0]
            peak_j = kdj[0]["j"]
            for i in range(1, limit_day + 1):
                h = highs[i]
                l = lows[i]
                p = closes[i]
                j = kdj[i]["j"]
                made_new_high = h > peak_price * 1.001
                if made_new_high:
                    # 新高但J没跟上，记为潜在背离；若随后单日转弱则离场
                    if j < peak_j - 5.0:
                        weak_today = i >= 1 and closes[i] < closes[i - 1]
                        if weak_today:
                            exit_day = i
                            exit_ret = (p / signal_price - 1.0) * 100.0
                            break
                    peak_price = h
                    peak_j = max(peak_j, j)
                # 防守底线：沿用统一止损/止盈
                tp = signal_price * (1.0 + take_profit_pct / 100.0)
                sl = signal_price * (1.0 - stop_loss_pct / 100.0)
                if l <= sl:
                    exit_day = i
                    exit_ret = (sl / signal_price - 1.0) * 100.0
                    break
                if h >= tp:
                    exit_day = i
                    exit_ret = (tp / signal_price - 1.0) * 100.0
                    break
            if exit_day is None:
                exit_day = limit_day
                exit_ret = (closes[limit_day] / signal_price - 1.0) * 100.0
        elif rule_name == "duokongwang_sell":
            # 多空王卖出组合：以“见弱就走”为主，避免利润回吐
            limit_day = min(int(max_hold_days), len(closes) - 1)
            flags = _calc_duokongwang_sell_flags(highs, lows, closes)
            for i in range(1, limit_day + 1):
                if i < len(flags) and flags[i]:
                    exit_day = i
                    exit_ret = (closes[i] / signal_price - 1.0) * 100.0
                    break
            if exit_day is None:
                exit_day = limit_day
                exit_ret = (closes[limit_day] / signal_price - 1.0) * 100.0
        else:
            day = min(10, len(closes) - 1)
            exit_day = day
            exit_ret = (closes[day] / signal_price - 1.0) * 100.0

        exits.append(
            {
                "symbol": symbol,
                "market": market,
                "signal_date": signal_date,
                "exit_day": int(exit_day),
                "exit_return_pct": float(exit_ret),
                "first_positive_day": row.get("first_positive_day"),
                "first_nonpositive_after_positive_day": row.get("first_nonpositive_after_positive_day"),
            }
        )

    if not exits:
        return {
            "rule_name": rule_name,
            "sample": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "avg_exit_day": None,
            "avg_first_profit_day": None,
            "avg_first_nonprofit_day": None,
            "avg_profit_span_days": None,
        }

    exit_returns = [x["exit_return_pct"] for x in exits]
    wins = sum(1 for x in exit_returns if x > 0)
    exit_days = [x["exit_day"] for x in exits]

    profit_days = []
    nonprofit_days = []
    span_days = []
    for x in exits:
        fp = x.get("first_positive_day")
        fn = x.get("first_nonpositive_after_positive_day")
        if fp is not None:
            try:
                fp_i = int(fp)
                profit_days.append(fp_i)
            except Exception:
                pass
        if fn is not None:
            try:
                fn_i = int(fn)
                nonprofit_days.append(fn_i)
                if fp is not None:
                    try:
                        fp_i2 = int(fp)
                        if fn_i >= fp_i2:
                            span_days.append(fn_i - fp_i2)
                    except Exception:
                        pass
            except Exception:
                pass

    return {
        "rule_name": rule_name,
        "sample": len(exits),
        "win_rate_pct": round(wins / len(exits) * 100.0, 1),
        "avg_return_pct": round(float(np.mean(exit_returns)), 2),
        "avg_exit_day": round(float(np.mean(exit_days)), 1) if exit_days else None,
        "avg_first_profit_day": round(float(np.mean(profit_days)), 1) if profit_days else None,
        "avg_first_nonprofit_day": round(float(np.mean(nonprofit_days)), 1) if nonprofit_days else None,
        "avg_profit_span_days": round(float(np.mean(span_days)), 1) if span_days else None,
        "details": exits,
    }
