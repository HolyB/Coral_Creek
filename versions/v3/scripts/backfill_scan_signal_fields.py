#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回填 scan_results 里的历史信号字段（OHLC + 多空王买卖点）

用途:
1) 历史记录补齐 day_high/day_low/day_close
2) 历史记录补齐 duokongwang_buy/duokongwang_sell
"""

import argparse
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from db.database import DB_PATH
from db.stock_history import get_history_db_path
from indicator_utils import calculate_volume_profile_metrics


def _ema(vals: List[float], n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1.0)
    out = [float(vals[0])]
    for v in vals[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return np.array(out, dtype=float)


def _sma_cn(vals: List[float], n: int, m: int = 1) -> np.ndarray:
    out = [float(vals[0])]
    for x in vals[1:]:
        out.append((m * float(x) + (n - m) * out[-1]) / float(n))
    return np.array(out, dtype=float)


def _calc_duokongwang_flags(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, profile: str) -> Tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    if n < 30:
        return buy, sell

    opens_proxy = np.r_[closes[0], closes[:-1]]
    up = _ema(highs.tolist(), 13)
    dw = _ema(lows.tolist(), 13)

    rsv = np.zeros(n)
    for i in range(n):
        s = max(0, i - 13)
        llv = np.min(lows[s:i + 1])
        hhv = np.max(highs[s:i + 1])
        rsv[i] = 50.0 if hhv <= llv else (closes[i] - llv) / (hhv - llv) * 100.0
    k = _sma_cn(rsv.tolist(), 3, 1)
    d = _sma_cn(k.tolist(), 3, 1)
    j = 3.0 * k - 2.0 * d

    lc = np.r_[closes[0], closes[:-1]]
    up_move = np.maximum(closes - lc, 0.0)
    abs_move = np.abs(closes - lc)
    rsi_num = _sma_cn(up_move.tolist(), 9, 1)
    rsi_den = _sma_cn(abs_move.tolist(), 9, 1)
    rsi2 = np.where(rsi_den > 1e-12, rsi_num / rsi_den * 100.0, 50.0)

    nt = np.zeros(n, dtype=int)
    nt0 = np.zeros(n, dtype=int)
    for i in range(n):
        a1 = i >= 4 and closes[i] > closes[i - 4]
        b1 = i >= 4 and closes[i] < closes[i - 4]
        nt[i] = (nt[i - 1] + 1) if (a1 and i > 0) else (1 if a1 else 0)
        nt0[i] = (nt0[i - 1] + 1) if (b1 and i > 0) else (1 if b1 else 0)

    if profile == "aggressive":
        j_cross_level, j_oversold_prev = 20.0, 28.0
        rsi_prev_th, rsi_now_th, nine_min = 30.0, 26.0, 7
    elif profile == "conservative":
        j_cross_level, j_oversold_prev = 35.0, 18.0
        rsi_prev_th, rsi_now_th, nine_min = 20.0, 18.0, 9
    else:
        j_cross_level, j_oversold_prev = 30.0, 22.0
        rsi_prev_th, rsi_now_th, nine_min = 24.0, 20.0, 9

    for i in range(1, n):
        cond = (
            (closes[i] > opens_proxy[i] and (opens_proxy[i] > up[i] or closes[i] < dw[i]))
            or (closes[i] < opens_proxy[i] and (opens_proxy[i] < dw[i] or closes[i] > up[i]))
        )
        cond1 = up[i] > up[i - 1] and dw[i] > dw[i - 1]
        cond2 = up[i] < up[i - 1] and dw[i] < dw[i - 1]

        kdj_cross_up = (j[i - 1] <= j_cross_level and j[i] > j_cross_level)
        kdj_oversold_turn = (j[i - 1] < j_oversold_prev and j[i] > j[i - 1])
        rsi_oversold_turn = (rsi2[i - 1] <= rsi_prev_th and rsi2[i] > rsi_now_th)
        nine_down_exhaust = nt0[i] >= nine_min
        buy[i] = bool((cond and cond1 and (kdj_cross_up or rsi_oversold_turn)) or kdj_oversold_turn or rsi_oversold_turn or nine_down_exhaust)

        kdj_overheat_fade = (j[i - 1] >= 100.0 and j[i] < 95.0) or (j[i - 1] >= 90.0 and j[i] < j[i - 1] - 8.0)
        rsi_overbought_turn = (rsi2[i - 1] >= 79.0 and rsi2[i] < 80.0)
        nine_up_exhaust = nt[i] >= 9 and closes[i] < closes[i - 1]
        sell[i] = bool((cond and cond2) or kdj_overheat_fade or rsi_overbought_turn or nine_up_exhaust)
    return buy, sell


def run_backfill(market: str, profile: str = "balanced") -> None:
    main_conn = sqlite3.connect(DB_PATH)
    main_conn.row_factory = sqlite3.Row
    hist_conn = sqlite3.connect(get_history_db_path())
    hist_conn.row_factory = sqlite3.Row

    mc = main_conn.cursor()
    hc = hist_conn.cursor()

    # 先把OHLC补齐
    mc.execute("SELECT COUNT(*) FROM scan_results WHERE market=? AND (day_high IS NULL OR day_low IS NULL OR day_close IS NULL)", (market,))
    need_ohlc = int(mc.fetchone()[0] or 0)
    print(f"[{market}] OHLC 待补: {need_ohlc}")

    # 从历史库取OHLC映射，再更新主库
    hc.execute(
        """
        SELECT symbol, trade_date, high, low, close
        FROM stock_history
        WHERE market=?
        """,
        (market,),
    )
    ohlc_map = {(str(x["symbol"]), str(x["trade_date"])): (float(x["high"]), float(x["low"]), float(x["close"])) for x in hc.fetchall()}

    mc.execute(
        """
        SELECT symbol, scan_date
        FROM scan_results
        WHERE market = ?
          AND (day_high IS NULL OR day_low IS NULL OR day_close IS NULL)
        """,
        (market,),
    )
    rows = []
    for rr in mc.fetchall():
        key = (str(rr["symbol"]), str(rr["scan_date"]))
        if key in ohlc_map:
            h, l, c = ohlc_map[key]
            rows.append((h, l, c, market, key[0], key[1]))
    mc.executemany(
        """
        UPDATE scan_results
        SET day_high=?, day_low=?, day_close=?, updated_at=CURRENT_TIMESTAMP
        WHERE market=? AND symbol=? AND scan_date=?
        """,
        rows,
    )
    main_conn.commit()
    print(f"[{market}] OHLC 已补: {len(rows)}")

    # 回填多空王
    mc.execute("SELECT COUNT(*) FROM scan_results WHERE market=? AND (duokongwang_buy IS NULL OR duokongwang_sell IS NULL)", (market,))
    need_dkw = int(mc.fetchone()[0] or 0)
    print(f"[{market}] 多空王待补: {need_dkw}")

    hc.execute(
        """
        SELECT symbol, trade_date, high, low, close
        FROM stock_history
        WHERE market=?
        ORDER BY symbol, trade_date
        """,
        (market,),
    )
    by_symbol: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for r in hc.fetchall():
        by_symbol[str(r["symbol"])].append(r)

    updates = []
    for sym, bars in by_symbol.items():
        highs = np.array([float(x["high"]) for x in bars], dtype=float)
        lows = np.array([float(x["low"]) for x in bars], dtype=float)
        closes = np.array([float(x["close"]) for x in bars], dtype=float)
        buy, sell = _calc_duokongwang_flags(highs, lows, closes, profile=profile)
        for i, b in enumerate(bars):
            updates.append((int(bool(buy[i])), int(bool(sell[i])), market, sym, str(b["trade_date"])))

    mc.executemany(
        """
        UPDATE scan_results
        SET duokongwang_buy=?, duokongwang_sell=?, updated_at=CURRENT_TIMESTAMP
        WHERE market=? AND symbol=? AND scan_date=?
        """,
        updates,
    )
    main_conn.commit()
    print(f"[{market}] 多空王已回填: {len(updates)} (按历史K线映射更新)")

    main_conn.close()
    hist_conn.close()


def recalc_chip_latest_signals(market: str, recent_days: int = 90, max_bars: int = 240) -> None:
    main_conn = sqlite3.connect(DB_PATH)
    main_conn.row_factory = sqlite3.Row
    hist_conn = sqlite3.connect(get_history_db_path())
    hist_conn.row_factory = sqlite3.Row
    mc = main_conn.cursor()
    hc = hist_conn.cursor()

    mc.execute(
        """
        SELECT id, symbol, scan_date
        FROM scan_results
        WHERE market=?
          AND scan_date >= date('now', ?)
        ORDER BY symbol, scan_date DESC
        """,
        (market, f"-{int(recent_days)} day"),
    )
    rows = mc.fetchall()
    if not rows:
        print(f"[{market}] 无需重算筹码字段")
        main_conn.close()
        hist_conn.close()
        return

    # 同一symbol只重算最新一条（默认组合实际只取每个symbol最新信号）
    latest = {}
    for r in rows:
        sym = str(r["symbol"])
        if sym not in latest:
            latest[sym] = (int(r["id"]), str(r["scan_date"]))

    updated = 0
    for sym, (rid, d) in latest.items():
        hc.execute(
            """
            SELECT close, volume
            FROM stock_history
            WHERE market=? AND symbol=? AND trade_date<=?
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            (market, sym, d, int(max_bars)),
        )
        bars = hc.fetchall()
        if len(bars) < 20:
            continue
        closes = np.array([float(x["close"]) for x in bars[::-1]], dtype=float)
        vols = np.array([float(x["volume"]) for x in bars[::-1]], dtype=float)
        curr = float(closes[-1])
        try:
            vp = calculate_volume_profile_metrics(closes, vols, curr)
            pr = float(vp.get("profit_ratio", 0.0) or 0.0)
            rating = "Normal"
            if pr > 0.9:
                rating = "Excellent"
            elif pr >= 0.7:
                rating = "Good"
            elif pr < 0.3:
                rating = "Poor"
            mc.execute(
                """
                UPDATE scan_results
                SET profit_ratio=?, vp_rating=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (pr, rating, rid),
            )
            updated += 1
        except Exception:
            continue
    main_conn.commit()
    main_conn.close()
    hist_conn.close()
    print(f"[{market}] 筹码字段重算完成: {updated} (recent_days={recent_days})")


def main():
    parser = argparse.ArgumentParser(description="Backfill scan_results OHLC + duokongwang fields")
    parser.add_argument("--market", choices=["US", "CN", "ALL"], default="ALL")
    parser.add_argument("--profile", choices=["aggressive", "balanced", "conservative"], default="balanced")
    parser.add_argument("--recalc-chip", action="store_true", help="重算最近信号的筹码字段(vp_rating/profit_ratio)")
    parser.add_argument("--chip-recent-days", type=int, default=120)
    args = parser.parse_args()

    markets = ["US", "CN"] if args.market == "ALL" else [args.market]
    for m in markets:
        run_backfill(market=m, profile=args.profile)
        if args.recalc_chip:
            recalc_chip_latest_signals(market=m, recent_days=int(args.chip_recent_days))


if __name__ == "__main__":
    main()
