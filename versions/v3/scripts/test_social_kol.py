#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 social KOL scan 的关键环节"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== 1. DDGS 搜索测试 ===")
try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text("site:x.com/TheRoaringKitty (stock OR buy)", max_results=5) or [])
        print(f"  搜索结果: {len(results)} 条")
        for r in results:
            print(f"  [{r.get('href', '')[:60]}]")
            print(f"    title: {r.get('title', '')[:80]}")
            print(f"    body:  {r.get('body', '')[:80]}")
            print()
except Exception as e:
    print(f"  DDGS 搜索失败: {e}")

print("\n=== 2. ticker 识别测试 ===")
from services.blogger_service import _extract_tickers_from_text, _detect_market_from_ticker

test_texts = [
    "$NVDA is going to moon! Buy now!",
    "I'm bullish on $TSLA and $AAPL",
    "Roaring Kitty just posted about GameStop GME",
    "买入600519贵州茅台",
    "看好000001.SZ",
]
for text in test_texts:
    tickers = _extract_tickers_from_text(text)
    print(f"  text: {text[:60]}")
    print(f"  tickers: {tickers}")
    for tk in tickers:
        mkt, norm = _detect_market_from_ticker(tk)
        print(f"    {tk} -> market={mkt}, normalized={norm}")
    print()

print("=== 3. symbol universe 测试 ===")
from services.blogger_service import _build_symbol_universe
uni = _build_symbol_universe()
print(f"  US symbols: {len(uni.get('US', set()))}")
print(f"  CN symbols: {len(uni.get('CN', set()))}")
if uni.get("US"):
    sample = sorted(list(uni["US"]))[:10]
    print(f"  US sample: {sample}")

print("\n=== 4. 完整流程测试 (1 KOL) ===")
try:
    from services.blogger_service import collect_social_kol_recommendations
    from db.database import init_blogger_tables
    init_blogger_tables()
    
    test_kols = [{"platform": "Twitter", "name": "TestKOL", "handle": "TheRoaringKitty", "market": "US"}]
    result = collect_social_kol_recommendations(kol_configs=test_kols, portfolio_tag="TEST_KOL", max_results_per_kol=5)
    print(f"  processed_kols: {result.get('processed_kols')}")
    print(f"  new_bloggers: {result.get('new_bloggers')}")
    print(f"  added_recommendations: {result.get('added_recommendations')}")
    print(f"  skipped_duplicates: {result.get('skipped_duplicates')}")
    if result.get("errors"):
        print(f"  errors: {result['errors']}")
    if result.get("kol_stats"):
        for ks in result["kol_stats"]:
            print(f"  KOL: {ks.get('name')} | posts={ks.get('posts')} | tickers={ks.get('ticker_hits')} | added={ks.get('added')}")
except Exception as e:
    print(f"  完整流程失败: {e}")
    import traceback
    traceback.print_exc()
