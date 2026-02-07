"""主题雷达服务：板块热点 + 龙头识别 + 社交热度辅助"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from data_fetcher import get_stock_data


DEFAULT_THEME_BASKETS: Dict[str, Dict[str, List[str]]] = {
    "US": {
        "AI算力龙头": ["NVDA", "MSFT", "META", "AVGO", "AMD", "TSM"],
        "比特币矿企": ["MSTR", "COIN", "IREN", "MARA", "RIOT", "CLSK"],
        "电力与公用事业": ["VST", "CEG", "NEE", "DUK", "SO", "TLN"],
        "半导体设备": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "ONTO"],
        "网络安全": ["CRWD", "PANW", "FTNT", "ZS", "S", "CYBR"],
        "云与SaaS": ["AMZN", "GOOGL", "ORCL", "SNOW", "DDOG", "NOW"],
    },
    "CN": {
        "算力与AI": ["300308.SZ", "300502.SZ", "002230.SZ", "603019.SH", "688256.SH"],
        "光伏储能": ["300274.SZ", "300750.SZ", "601012.SH", "002594.SZ", "688223.SH"],
        "电力设备": ["600406.SH", "600900.SH", "600795.SH", "600025.SH", "601985.SH"],
        "半导体": ["688981.SH", "600584.SH", "603501.SH", "002371.SZ", "300223.SZ"],
        "工业自动化": ["300124.SZ", "002747.SZ", "002508.SZ", "688017.SH", "603290.SH"],
    },
}


def get_theme_baskets(market: str = "US") -> Dict[str, List[str]]:
    """获取默认主题篮子"""
    mkt = (market or "US").upper()
    return DEFAULT_THEME_BASKETS.get(mkt, DEFAULT_THEME_BASKETS["US"])


def _safe_pct_return(close: pd.Series, lookback: int) -> Optional[float]:
    if close is None or len(close) <= lookback:
        return None
    prev = float(close.iloc[-(lookback + 1)])
    curr = float(close.iloc[-1])
    if prev <= 0:
        return None
    return (curr / prev - 1.0) * 100.0


def _build_scan_lookup(scan_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if scan_df is None or scan_df.empty or "symbol" not in scan_df.columns:
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in scan_df.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        lookup[symbol] = row.to_dict()
    return lookup


def _calc_leader_metrics(
    symbol: str,
    market: str,
    scan_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    hist = get_stock_data(symbol, market=market, days=140)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna()
    if len(close) < 25:
        return None

    volume = hist["Volume"].replace(0, np.nan) if "Volume" in hist.columns else pd.Series(dtype=float)
    vol5 = float(volume.tail(5).mean()) if len(volume) >= 5 else np.nan
    vol20 = float(volume.tail(20).mean()) if len(volume) >= 20 else np.nan
    vol_ratio = float(vol5 / vol20) if pd.notna(vol5) and pd.notna(vol20) and vol20 > 0 else np.nan

    ret20 = _safe_pct_return(close, 20)
    ret60 = _safe_pct_return(close, 60)

    base_row = scan_lookup.get(symbol.upper(), {})
    blue_daily = base_row.get("blue_daily", np.nan)
    blue_weekly = base_row.get("blue_weekly", np.nan)

    return {
        "symbol": symbol.upper(),
        "price": float(close.iloc[-1]),
        "ret20": float(ret20) if ret20 is not None else np.nan,
        "ret60": float(ret60) if ret60 is not None else np.nan,
        "vol_ratio": vol_ratio,
        "blue_daily": float(blue_daily) if pd.notna(blue_daily) else np.nan,
        "blue_weekly": float(blue_weekly) if pd.notna(blue_weekly) else np.nan,
    }


def _score_row(row: pd.Series) -> float:
    ret20 = row.get("ret20", np.nan)
    ret60 = row.get("ret60", np.nan)
    vol_ratio = row.get("vol_ratio", np.nan)
    blue_daily = row.get("blue_daily", np.nan)

    # 交易员视角：中期趋势 + 短期动量 + 量能确认 + 信号强度
    ret20 = 0.0 if pd.isna(ret20) else np.clip(ret20, -30, 50)
    ret60 = ret20 if pd.isna(ret60) else np.clip(ret60, -40, 80)
    vol_boost = 0.0 if pd.isna(vol_ratio) else np.clip((vol_ratio - 1.0) * 100, -30, 60)
    blue_norm = 0.0 if pd.isna(blue_daily) else np.clip(float(blue_daily), 0, 200) * 0.5

    return float(ret20 * 0.35 + ret60 * 0.35 + vol_boost * 0.2 + blue_norm * 0.1)


def build_theme_radar(
    market: str = "US",
    top_themes: int = 5,
    leaders_per_theme: int = 4,
    scan_df: Optional[pd.DataFrame] = None,
    include_social: bool = False,
) -> Dict[str, Any]:
    """生成主题雷达结果"""
    mkt = (market or "US").upper()
    baskets = get_theme_baskets(mkt)
    scan_lookup = _build_scan_lookup(scan_df)

    theme_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for theme_name, symbols in baskets.items():
        per_symbol: List[Dict[str, Any]] = []
        for sym in symbols:
            try:
                metrics = _calc_leader_metrics(sym, mkt, scan_lookup)
                if metrics:
                    metrics["leader_score"] = _score_row(pd.Series(metrics))
                    per_symbol.append(metrics)
            except Exception as exc:
                errors.append(f"{theme_name}:{sym}:{str(exc)[:80]}")

        if not per_symbol:
            continue

        theme_df = pd.DataFrame(per_symbol).sort_values("leader_score", ascending=False)
        leaders = theme_df.head(max(1, leaders_per_theme)).to_dict("records")

        pos_ratio = float((theme_df["ret20"].fillna(0) > 0).mean())
        heat_score = float(theme_df["leader_score"].head(3).mean() * 0.75 + pos_ratio * 100 * 0.25)

        row = {
            "theme": theme_name,
            "theme_score": round(heat_score, 2),
            "avg_ret20": round(float(theme_df["ret20"].fillna(0).mean()), 2),
            "avg_ret60": round(float(theme_df["ret60"].fillna(0).mean()), 2),
            "positive_ratio": round(pos_ratio * 100, 1),
            "leader_count": int(len(theme_df)),
            "leaders": leaders,
            "social": None,
        }
        theme_rows.append(row)

    if not theme_rows:
        return {
            "market": mkt,
            "themes": [],
            "errors": errors,
            "meta": {"top_themes": top_themes, "leaders_per_theme": leaders_per_theme},
        }

    theme_rows.sort(key=lambda x: x["theme_score"], reverse=True)
    selected = theme_rows[: max(1, top_themes)]

    if include_social:
        try:
            from services.social_monitor import get_social_service

            social_service = get_social_service()
            for theme in selected:
                leaders = theme.get("leaders", [])[:2]
                sentiment_scores: List[float] = []
                bull_count = 0
                bear_count = 0
                total_posts = 0
                for leader in leaders:
                    sym = leader.get("symbol")
                    if not sym:
                        continue
                    report = social_service.get_social_report(sym, market=mkt)
                    if not report:
                        continue
                    sentiment_scores.append(float(report.get("sentiment_score", 0)))
                    bull_count += int(report.get("bullish_count", 0))
                    bear_count += int(report.get("bearish_count", 0))
                    total_posts += int(report.get("total_posts", 0))

                if sentiment_scores:
                    theme["social"] = {
                        "avg_sentiment": round(float(np.mean(sentiment_scores)), 3),
                        "bullish_count": bull_count,
                        "bearish_count": bear_count,
                        "total_posts": total_posts,
                    }
        except Exception as exc:
            errors.append(f"social:{str(exc)[:120]}")

    return {
        "market": mkt,
        "themes": selected,
        "errors": errors,
        "meta": {"top_themes": top_themes, "leaders_per_theme": leaders_per_theme},
    }


def add_theme_leaders_to_watchlist(
    theme_data: Dict[str, Any],
    market: str = "US",
    top_n: int = 3,
) -> Dict[str, Any]:
    """将某个主题的龙头批量加入观察列表"""
    from services.signal_tracker import add_to_watchlist

    leaders = (theme_data or {}).get("leaders", [])[: max(1, top_n)]
    theme_name = (theme_data or {}).get("theme", "主题")

    added = 0
    failed: List[str] = []

    for leader in leaders:
        symbol = leader.get("symbol")
        if not symbol:
            continue

        price = float(leader.get("price", 0) or 0)
        ret20 = float(leader.get("ret20", 0) or 0)
        score = float(leader.get("leader_score", 0) or 0)

        ok = add_to_watchlist(
            symbol=symbol,
            market=market,
            entry_price=price if price > 0 else None,
            target_price=price * 1.15 if price > 0 else None,
            stop_loss=price * 0.92 if price > 0 else None,
            signal_type="theme_leader",
            signal_score=score,
            notes=f"{theme_name} 龙头 | 20日涨幅 {ret20:.1f}%",
        )

        if ok:
            added += 1
        else:
            failed.append(symbol)

    return {"added": added, "failed": failed, "theme": theme_name}
