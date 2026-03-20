#!/usr/bin/env python
"""
预计算 MMoE 排名分 — 每日扫描后运行一次
=========================================
读取最新 scan_results → 批量跑 MMoE → 存 JSON 缓存
页面加载时 RankingSystem 直接读缓存，毫秒级。

用法:
  python scripts/precompute_mmoe_scores.py --market US
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


CACHE_DIR = Path(__file__).parent.parent / 'ml' / 'saved_models' / 'mmoe_cache'


def precompute(market: str = 'US'):
    t0 = time.time()
    
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    from ml.smart_picker import SmartPicker
    
    init_db()
    
    dates = get_scanned_dates(market=market)
    if not dates:
        print("❌ 无扫描数据")
        return
    
    latest = dates[0]
    print(f"📅 预计算日期: {latest}, 市场: {market}")
    
    sigs = query_scan_results(scan_date=latest, market=market, limit=500)
    if not sigs:
        print("❌ 无信号")
        return
    
    print(f"📊 信号数: {len(sigs)}")
    
    picker = SmartPicker(market=market, horizon='short')
    has_mmoe = picker.mmoe_model is not None
    print(f"MMoE: {'✅' if has_mmoe else '❌ (XGBoost fallback)'}")
    
    results = {}
    success = 0
    
    for i, s in enumerate(sigs):
        sym = str(s.get('symbol', '')).strip().upper()
        price = float(s.get('price', 0) or 0)
        if not sym or price <= 0:
            continue
        
        try:
            h = get_stock_history(sym, market, days=300)
            if h is None or h.empty or len(h) < 60:
                continue
            
            if not isinstance(h.index, pd.DatetimeIndex):
                if 'Date' in h.columns:
                    h = h.set_index('Date')
                elif 'date' in h.columns:
                    h = h.set_index('date')
                h.index = pd.to_datetime(h.index)
            
            sig = pd.Series({
                'symbol': sym,
                'price': price,
                'blue_daily': float(s.get('blue_daily', 0) or 0),
                'blue_weekly': float(s.get('blue_weekly', 0) or 0),
                'blue_monthly': float(s.get('blue_monthly', 0) or 0),
                'is_heima': 1 if s.get('heima_daily') else 0,
            })
            
            # 临时去掉 symbol 以跳过新闻特征计算（调用 Gemini API 太慢）
            sig_for_predict = sig.copy()
            sig_for_predict['symbol'] = ''  # 跳过 _add_news_features
            
            pick = picker._analyze_stock(sig_for_predict, h, skip_prefilter=True)
            if pick:
                results[sym] = {
                    'dir_prob': round(float(np.clip(pick.pred_direction_prob, 0.01, 0.99)), 4),
                    'return_5d': round(float(np.clip(pick.pred_return_5d, -100, 200)), 2),
                    'return_20d': round(float(np.clip(getattr(pick, 'pred_return_20d', 0) or 0, -100, 500)), 2),
                    'max_dd': round(float(np.clip(getattr(pick, 'pred_max_dd', 0) or 0, -100, 0)), 2),
                    'overall_score': round(pick.overall_score, 1),
                    'star_rating': pick.star_rating,
                    'rank_short': round(pick.rank_score_short, 1),
                    'rank_medium': round(pick.rank_score_medium, 1),
                }
                success += 1
        except Exception:
            continue
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sigs)}... ({success} ok)")
    
    # 保存缓存
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {
        'date': latest,
        'market': market,
        'computed_at': datetime.now().isoformat(),
        'model': 'mmoe' if has_mmoe else 'xgboost',
        'total_signals': len(sigs),
        'computed': success,
        'scores': results,
    }
    
    # 始终保存按日期命名的文件（即使为空，用于记录）
    cache_file = CACHE_DIR / f'{market.lower()}_{latest}.json'
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    # 只有成功计算了分数才更新 latest（避免空结果覆盖好缓存）
    latest_file = CACHE_DIR / f'{market.lower()}_latest.json'
    if success > 0:
        with open(latest_file, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"\n✅ 预计算完成: {success}/{len(sigs)} 只")
        print(f"   缓存: {cache_file}")
    else:
        print(f"\n⚠️ 预计算 0 只成功，保留旧的 latest 缓存不覆盖")
        print(f"   日期文件: {cache_file}")
    
    print(f"   耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='US')
    args = parser.parse_args()
    precompute(market=args.market)
