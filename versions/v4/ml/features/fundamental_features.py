#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基本面特征构建器
================

目标:
1. 给 ML 训练补充估值/质量/成长等低频特征
2. 优先使用已有扫描字段，避免额外 API 压力
3. 可选启用外部 API (yfinance / polygon) 增强字段
"""

from __future__ import annotations

from typing import Dict, Optional
import math


def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_log1p(v: float) -> float:
    if v <= 0:
        return 0.0
    try:
        return float(math.log1p(v))
    except Exception:
        return 0.0


def _industry_hash(industry: str, bucket: int = 97) -> float:
    """
    行业类别转数值桶（避免高基数字符串直接入模）
    注意: 这是轻量编码，后续可替换为 target encoding。
    """
    if not industry:
        return 0.0
    return float(abs(hash(industry)) % bucket) / float(bucket)


def _extract_from_signal_row(signal_row: Optional[Dict]) -> Dict[str, float]:
    """从扫描结果行提取可直接用的基本面字段。"""
    if not signal_row:
        return {}

    market_cap = _to_float(signal_row.get("market_cap"), 0.0)
    cap_category = str(signal_row.get("cap_category") or "")
    industry = str(signal_row.get("industry") or "")

    cap_small = 1.0 if ("small" in cap_category.lower() or "小盘" in cap_category) else 0.0
    cap_mid = 1.0 if ("mid" in cap_category.lower() or "中盘" in cap_category) else 0.0
    cap_large = 1.0 if ("large" in cap_category.lower() or "大盘" in cap_category) else 0.0

    return {
        # 规模特征
        "fund_market_cap": market_cap,
        "fund_market_cap_log": _safe_log1p(market_cap),
        "fund_cap_small": cap_small,
        "fund_cap_mid": cap_mid,
        "fund_cap_large": cap_large,
        # 行业编码
        "fund_industry_hash": _industry_hash(industry),
        # 缺失标记
        "fund_has_market_cap": 1.0 if market_cap > 0 else 0.0,
    }


def _extract_from_external(
    symbol: str,
    market: str,
) -> Dict[str, float]:
    """
    可选外部基本面字段:
    - US: 优先 polygon ticker details + yfinance.info
    - CN/HK: 尝试 yfinance.info（可用则补充）
    """
    out: Dict[str, float] = {}

    # 1) Polygon ticker details: 至少拿到 market cap / shares
    try:
        from data_fetcher import get_ticker_details
        details = get_ticker_details(symbol) if market.upper() == "US" else None
        if details:
            mc = _to_float(details.get("market_cap"), 0.0)
            shares = _to_float(details.get("shares_outstanding"), 0.0)
            out.update(
                {
                    "fund_market_cap_ext": mc,
                    "fund_shares_outstanding": shares,
                    "fund_market_cap_ext_log": _safe_log1p(mc),
                }
            )
    except Exception:
        pass

    # 2) yfinance: 估值/质量/成长
    try:
        import yfinance as yf

        yf_symbol = symbol
        # A 股代码映射
        if market.upper() == "CN":
            s = symbol.upper()
            if s.endswith(".SH") or s.endswith(".SZ") or s.endswith(".BJ"):
                yf_symbol = s
            elif s.isdigit() and len(s) == 6:
                yf_symbol = f"{s}.SS" if s.startswith("6") else f"{s}.SZ"
        # 港股映射
        if market.upper() == "HK":
            s = symbol.upper().replace("HK", "").replace(".HK", "")
            yf_symbol = f"{s.zfill(4)}.HK"

        info = yf.Ticker(yf_symbol).info or {}

        # 估值
        out["fund_pe_ttm"] = _to_float(info.get("trailingPE"), 0.0)
        out["fund_pe_fwd"] = _to_float(info.get("forwardPE"), 0.0)
        out["fund_pb"] = _to_float(info.get("priceToBook"), 0.0)
        out["fund_ps"] = _to_float(info.get("priceToSalesTrailing12Months"), 0.0)
        out["fund_ev_ebitda"] = _to_float(info.get("enterpriseToEbitda"), 0.0)
        out["fund_dividend_yield"] = _to_float(info.get("dividendYield"), 0.0)
        out["fund_beta"] = _to_float(info.get("beta"), 0.0)

        # 质量
        out["fund_roe"] = _to_float(info.get("returnOnEquity"), 0.0)
        out["fund_roa"] = _to_float(info.get("returnOnAssets"), 0.0)
        out["fund_profit_margin"] = _to_float(info.get("profitMargins"), 0.0)
        out["fund_current_ratio"] = _to_float(info.get("currentRatio"), 0.0)
        out["fund_quick_ratio"] = _to_float(info.get("quickRatio"), 0.0)
        out["fund_debt_to_equity"] = _to_float(info.get("debtToEquity"), 0.0)

        # 成长
        out["fund_revenue_growth"] = _to_float(info.get("revenueGrowth"), 0.0)
        out["fund_earnings_growth"] = _to_float(info.get("earningsGrowth"), 0.0)

        # 目标价偏离（分析师预期）
        target_mean = _to_float(info.get("targetMeanPrice"), 0.0)
        current_price = _to_float(info.get("currentPrice"), 0.0)
        if target_mean > 0 and current_price > 0:
            out["fund_target_upside"] = (target_mean / current_price - 1.0) * 100.0
        else:
            out["fund_target_upside"] = 0.0
    except Exception:
        pass

    # 缺失标志：是否成功拿到外部估值字段
    out["fund_has_external"] = 1.0 if any(k.startswith("fund_pe") or k.startswith("fund_pb") for k in out) else 0.0
    return out


def build_fundamental_features(
    symbol: str,
    market: str,
    signal_row: Optional[Dict] = None,
    enable_external_api: bool = False,
    external_cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    构建单只股票的基本面特征。

    参数:
    - enable_external_api: 是否启用 yfinance/polygon 外部拉取（较慢）
    - external_cache: 外部拉取缓存，避免重复请求
    """
    feat = {}
    feat.update(_extract_from_signal_row(signal_row))

    if enable_external_api:
        cache = external_cache if external_cache is not None else {}
        key = f"{market}:{symbol}"
        if key not in cache:
            cache[key] = _extract_from_external(symbol=symbol, market=market)
        feat.update(cache.get(key, {}))
    else:
        # 关闭外部 API 时，保证字段存在（缺失用 0）
        for k in [
            "fund_pe_ttm", "fund_pe_fwd", "fund_pb", "fund_ps", "fund_ev_ebitda",
            "fund_dividend_yield", "fund_beta", "fund_roe", "fund_roa",
            "fund_profit_margin", "fund_current_ratio", "fund_quick_ratio",
            "fund_debt_to_equity", "fund_revenue_growth", "fund_earnings_growth",
            "fund_target_upside", "fund_has_external",
        ]:
            feat.setdefault(k, 0.0)

    return feat
