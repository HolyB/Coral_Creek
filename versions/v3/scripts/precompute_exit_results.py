#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算退出规则结果 —— 每日定时运行一次，结果存入 SQLite。
前端 app.py 读取预计算结果即可秒出，无需实时计算。

用法:
  python scripts/precompute_exit_results.py --market US
  python scripts/precompute_exit_results.py --market US --rules fixed_5d,fixed_10d,fixed_20d,tp_sl_time
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.candidate_tracking_service import (
    evaluate_exit_rule,
    get_candidate_tracking_rows,
)
import services.candidate_tracking_service as _cts

# --------------- 数据库路径 ---------------
DB_DIR = os.path.join(ROOT, "db")
DB_PATH = os.path.join(DB_DIR, "coral_creek.db")


def _ensure_table(conn: sqlite3.Connection) -> None:
    """创建预计算结果表（如果不存在）"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precomputed_exit_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            take_profit_pct REAL NOT NULL DEFAULT 10.0,
            stop_loss_pct REAL NOT NULL DEFAULT 6.0,
            max_hold_days INTEGER NOT NULL DEFAULT 20,
            exit_day INTEGER,
            exit_return_pct REAL,
            first_positive_day INTEGER,
            first_nonpositive_after_positive_day INTEGER,
            computed_at TEXT NOT NULL,
            UNIQUE(symbol, market, signal_date, rule_name, take_profit_pct, stop_loss_pct, max_hold_days)
        )
    """)
    # 索引: 按 market + rule_name 快速查询
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_precomputed_market_rule
        ON precomputed_exit_results(market, rule_name)
    """)
    conn.commit()


def precompute_for_market(
    market: str,
    rules: list[str],
    tp: float = 10.0,
    sl: float = 6.0,
    hold: int = 20,
    batch_size: int = 5000,
) -> dict:
    """
    对指定 market 的全量 candidate_tracking 数据，
    按 batch_size 分批调用 evaluate_exit_rule，
    结果全量写入 precomputed_exit_results 表。
    """
    print(f"[{market}] 加载 candidate_tracking 数据...")
    t0 = time.time()
    rows = get_candidate_tracking_rows(market=market, days_back=0)
    print(f"[{market}] 共 {len(rows)} 条追踪记录，耗时 {time.time()-t0:.1f}s")

    if not rows:
        print(f"[{market}] 无数据，跳过。")
        return {"market": market, "total_rows": 0, "rules": {}}

    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)

    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = {}

    # 预计算模式：跳过 Polygon API 兜底，只用本地价格数据（覆盖率 97%+）
    _cts._skip_api_fallback = True

    for rule_name in rules:
        print(f"\n[{market}] 计算规则: {rule_name} ...")
        rule_t0 = time.time()
        total_details = 0
        inserted = 0

        # 分批处理，避免单次预取过多价格数据
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_label = f"batch {i//batch_size + 1} ({len(batch)} rows)"
            bt0 = time.time()

            try:
                res = evaluate_exit_rule(
                    rows=batch,
                    rule_name=rule_name,
                    take_profit_pct=tp,
                    stop_loss_pct=sl,
                    max_hold_days=hold,
                    max_rows=len(batch),
                )
            except Exception as e:
                print(f"  ⚠️ {batch_label} 计算失败: {e}")
                continue

            details = res.get("details") or []
            total_details += len(details)
            bt1 = time.time()
            print(f"  {batch_label}: {len(details)} 条结果, {bt1-bt0:.1f}s")

            # 写入数据库 (UPSERT)
            for d in details:
                try:
                    conn.execute(
                        """
                        INSERT INTO precomputed_exit_results
                            (symbol, market, signal_date, rule_name,
                             take_profit_pct, stop_loss_pct, max_hold_days,
                             exit_day, exit_return_pct,
                             first_positive_day, first_nonpositive_after_positive_day,
                             computed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, market, signal_date, rule_name,
                                    take_profit_pct, stop_loss_pct, max_hold_days)
                        DO UPDATE SET
                            exit_day = excluded.exit_day,
                            exit_return_pct = excluded.exit_return_pct,
                            first_positive_day = excluded.first_positive_day,
                            first_nonpositive_after_positive_day = excluded.first_nonpositive_after_positive_day,
                            computed_at = excluded.computed_at
                        """,
                        (
                            d.get("symbol", ""),
                            d.get("market", market),
                            d.get("signal_date", ""),
                            rule_name,
                            tp,
                            sl,
                            hold,
                            d.get("exit_day"),
                            d.get("exit_return_pct"),
                            d.get("first_positive_day"),
                            d.get("first_nonpositive_after_positive_day"),
                            now_str,
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    print(f"  ⚠️ INSERT 失败 {d.get('symbol')}: {e}")

            conn.commit()

        rule_elapsed = time.time() - rule_t0
        stats[rule_name] = {
            "details": total_details,
            "inserted": inserted,
            "elapsed_s": round(rule_elapsed, 1),
        }
        print(
            f"[{market}] 规则 {rule_name} 完成: "
            f"{total_details} 条计算, {inserted} 条写入, 耗时 {rule_elapsed:.1f}s"
        )

    _cts._skip_api_fallback = False
    conn.close()
    return {"market": market, "total_rows": len(rows), "rules": stats}


def get_precomputed_details(
    market: str,
    rule_name: str,
    tp: float = 10.0,
    sl: float = 6.0,
    hold: int = 20,
) -> list[dict]:
    """
    从预计算表读取结果，供 app.py 使用。
    返回格式和 evaluate_exit_rule 的 details 完全一致。
    """
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT symbol, market, signal_date,
                   exit_day, exit_return_pct,
                   first_positive_day, first_nonpositive_after_positive_day
            FROM precomputed_exit_results
            WHERE market = ? AND rule_name = ?
              AND take_profit_pct = ? AND stop_loss_pct = ? AND max_hold_days = ?
            """,
            (market, rule_name, tp, sl, hold),
        )
        rows = [dict(r) for r in cursor.fetchall()]
    except Exception:
        rows = []
    finally:
        conn.close()
    return rows


def get_precomputed_summary(
    market: str,
    rule_name: str,
    tp: float = 10.0,
    sl: float = 6.0,
    hold: int = 20,
) -> dict:
    """
    从预计算表读取汇总统计，供 app.py 显示规则胜率等摘要。
    返回格式和 evaluate_exit_rule 返回值一致（但不含 details 大列表）。
    """
    import numpy as np

    details = get_precomputed_details(market, rule_name, tp, sl, hold)
    if not details:
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

    exit_returns = [float(d.get("exit_return_pct") or 0) for d in details]
    wins = sum(1 for r in exit_returns if r > 0)
    exit_days = [int(d["exit_day"]) for d in details if d.get("exit_day") is not None]
    profit_days = [int(d["first_positive_day"]) for d in details if d.get("first_positive_day") is not None]
    nonprofit_days = [int(d["first_nonpositive_after_positive_day"]) for d in details if d.get("first_nonpositive_after_positive_day") is not None]
    span_days = []
    for d in details:
        fp = d.get("first_positive_day")
        fn = d.get("first_nonpositive_after_positive_day")
        if fp is not None and fn is not None:
            try:
                span = int(fn) - int(fp)
                if span >= 0:
                    span_days.append(span)
            except Exception:
                pass

    return {
        "rule_name": rule_name,
        "sample": len(details),
        "win_rate_pct": round(wins / len(details) * 100.0, 1),
        "avg_return_pct": round(float(np.mean(exit_returns)), 2),
        "avg_exit_day": round(float(np.mean(exit_days)), 1) if exit_days else None,
        "avg_first_profit_day": round(float(np.mean(profit_days)), 1) if profit_days else None,
        "avg_first_nonprofit_day": round(float(np.mean(nonprofit_days)), 1) if nonprofit_days else None,
        "avg_profit_span_days": round(float(np.mean(span_days)), 1) if span_days else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute exit rule results for all candidate tracking rows.")
    parser.add_argument("--market", type=str, default="US", choices=["US", "CN", "ALL"])
    parser.add_argument(
        "--rules",
        type=str,
        default="fixed_5d,fixed_10d,fixed_20d,tp_sl_time",
        help="逗号分隔的规则列表",
    )
    parser.add_argument("--tp", type=float, default=10.0, help="止盈百分比")
    parser.add_argument("--sl", type=float, default=6.0, help="止损百分比")
    parser.add_argument("--hold", type=int, default=20, help="最长持有天数")
    parser.add_argument("--batch-size", type=int, default=5000, help="每批处理行数")
    args = parser.parse_args()

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    markets = ["US", "CN"] if args.market == "ALL" else [args.market]

    all_stats = {}
    for m in markets:
        result = precompute_for_market(
            market=m,
            rules=rules,
            tp=args.tp,
            sl=args.sl,
            hold=args.hold,
            batch_size=args.batch_size,
        )
        all_stats[m] = result

    print("\n" + "=" * 60)
    print("📊 预计算统计汇总:")
    print(json.dumps(all_stats, indent=2, ensure_ascii=False))
