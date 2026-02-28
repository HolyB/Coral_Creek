#!/usr/bin/env python
"""
批量预计算历史日期的 MMoE 缓存
================================
用法: python scripts/batch_precompute_mmoe.py --days 60
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, time
from pathlib import Path
from datetime import datetime

os.environ['GEMINI_API_KEY'] = ''  # disable Gemini
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

CACHE_DIR = Path(__file__).parent.parent / 'ml' / 'saved_models' / 'mmoe_cache'


def precompute_date(picker, date_str: str, market: str = 'US'):
    """预计算单个日期"""
    from db.database import query_scan_results
    from db.stock_history import get_stock_history
    
    # 检查是否已有缓存
    cache_file = CACHE_DIR / f'{market.lower()}_{date_str}.json'
    if cache_file.exists():
        with open(cache_file) as f:
            existing = json.load(f)
        if existing.get('computed', 0) > 0:
            print(f"  ⏭ {date_str}: 已有缓存 ({existing['computed']} 只)")
            return existing['computed']
    
    sigs = query_scan_results(scan_date=date_str, market=market, limit=500)
    if not sigs:
        print(f"  ⏭ {date_str}: 无信号")
        return 0
    
    results = {}
    success = 0
    
    for s in sigs:
        sym = str(s.get('symbol', '')).strip().upper()
        price = float(s.get('price', 0) or 0)
        if not sym or price <= 0:
            continue
        try:
            h = get_stock_history(sym, market, days=300)
            if h is None or h.empty or len(h) < 60:
                continue
            if not isinstance(h.index, pd.DatetimeIndex):
                if 'Date' in h.columns: h = h.set_index('Date')
                elif 'date' in h.columns: h = h.set_index('date')
                h.index = pd.to_datetime(h.index)
            
            # 截断到信号日（近似历史）
            h_before = h[h.index <= pd.to_datetime(date_str)]
            if len(h_before) < 60:
                h_before = h  # fallback 用全量
            
            sig = pd.Series({
                'symbol': sym, 'price': price,
                'blue_daily': float(s.get('blue_daily', 0) or 0),
                'blue_weekly': float(s.get('blue_weekly', 0) or 0),
                'blue_monthly': float(s.get('blue_monthly', 0) or 0),
                'is_heima': 1 if s.get('heima_daily') else 0,
            })
            pick = picker._analyze_stock(sig, h_before, skip_prefilter=True)
            if pick:
                results[sym] = {
                    'dir_prob': round(pick.pred_direction_prob, 4),
                    'return_5d': round(pick.pred_return_5d, 2),
                    'return_20d': round(getattr(pick, 'pred_return_20d', 0) or 0, 2),
                    'max_dd': round(getattr(pick, 'pred_max_dd', 0) or 0, 2),
                    'overall_score': round(pick.overall_score, 1),
                    'star_rating': pick.star_rating,
                    'rank_short': round(pick.rank_score_short, 1),
                    'rank_medium': round(pick.rank_score_medium, 1),
                }
                success += 1
        except:
            continue
    
    # 保存
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {
        'date': date_str, 'market': market,
        'computed_at': datetime.now().isoformat(),
        'model': 'mmoe' if picker.mmoe_model else 'xgboost',
        'total_signals': len(sigs), 'computed': success,
        'scores': results,
    }
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"  ✅ {date_str}: {success}/{len(sigs)} 只")
    return success


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=60, help='回溯天数')
    parser.add_argument('--market', default='US')
    args = parser.parse_args()
    
    from db.database import init_db, get_scanned_dates
    from ml.smart_picker import SmartPicker
    
    init_db()
    dates = get_scanned_dates(market=args.market)
    
    # 取最近 N 天
    target_dates = [d for d in dates if d >= (datetime.now() - pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')]
    print(f"📅 批量预计算: {len(target_dates)} 天 ({args.market})")
    
    # 加载模型一次
    picker = SmartPicker(market=args.market, horizon='short')
    print(f"MMoE: {'✅' if picker.mmoe_model else '❌'}")
    
    t0 = time.time()
    total = 0
    for i, d in enumerate(target_dates):
        n = precompute_date(picker, d, args.market)
        total += n
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(target_dates) - i - 1)
            print(f"  📊 {i+1}/{len(target_dates)} 天完成, 预计剩余 {remaining/60:.0f} 分钟")
    
    print(f"\n🏁 全部完成: {len(target_dates)} 天, {total} 只, {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
