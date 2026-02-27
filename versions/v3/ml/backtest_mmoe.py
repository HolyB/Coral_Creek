#!/usr/bin/env python
"""
MMoE vs XGBoost 回测对比
========================
用 candidate_tracking 的历史数据，对比两个模型选出的 Top5 真实收益。

用法: /Users/bertwang/miniconda3/bin/python ml/backtest_mmoe.py
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import defaultdict

def run_backtest():
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    from ml.smart_picker import SmartPicker
    
    init_db()
    
    # 获取最近的扫描日期
    dates = get_scanned_dates(market='US')
    # 用最近 30 个交易日
    test_dates = dates[:30]
    print(f"📅 回测: {len(test_dates)} 天 ({test_dates[-1]} ~ {test_dates[0]})")
    
    # 初始化 picker (MMoE 模式)
    picker = SmartPicker(market='US', horizon='short')
    has_mmoe = picker.mmoe_model is not None
    print(f"MMoE: {'✅' if has_mmoe else '❌'}")
    
    records = []
    
    for di, scan_date in enumerate(test_dates):
        signals = query_scan_results(scan_date=scan_date, market='US', limit=50)
        if not signals:
            continue
        
        picks = []
        for s in signals:
            sig = pd.Series(s)
            sym = sig.get('symbol', '')
            price = float(sig.get('price', 0))
            if not sym or price <= 0:
                continue
            
            h = get_stock_history(sym, 'US', days=300)
            if h is None or h.empty or len(h) < 60:
                continue
            
            # 确保有日期索引
            if not isinstance(h.index, pd.DatetimeIndex):
                if 'Date' in h.columns:
                    h = h.set_index('Date')
                elif 'date' in h.columns:
                    h = h.set_index('date')
                h.index = pd.to_datetime(h.index)
            
            # 截断到 scan_date 之前的数据 (避免未来信息)
            cutoff = pd.to_datetime(scan_date)
            h_before = h[h.index <= cutoff]
            if len(h_before) < 60:
                continue
            
            pick = picker._analyze_stock(sig, h_before, skip_prefilter=True)
            if pick:
                # 获取真实 5d 收益
                h_after = h[h.index > cutoff]
                if len(h_after) >= 5:
                    actual_5d = (float(h_after['Close'].iloc[4]) / price - 1) * 100
                else:
                    actual_5d = None
                
                # 获取真实 20d 收益
                if len(h_after) >= 20:
                    actual_20d = (float(h_after['Close'].iloc[19]) / price - 1) * 100
                else:
                    actual_20d = None
                
                picks.append({
                    'date': scan_date,
                    'symbol': sym,
                    'price': price,
                    'mmoe_score': pick.overall_score,
                    'mmoe_dir': pick.pred_direction_prob,
                    'mmoe_ret5d': pick.pred_return_5d,
                    'actual_5d': actual_5d,
                    'actual_20d': actual_20d,
                    'blue_daily': float(sig.get('blue_daily', 0)),
                })
        
        if picks:
            records.extend(picks)
        
        if (di + 1) % 5 == 0:
            print(f"  {di+1}/{len(test_dates)} dates, {len(records)} records...")
    
    if not records:
        print("❌ 无数据")
        return
    
    df = pd.DataFrame(records)
    df = df.dropna(subset=['actual_5d'])
    print(f"\n📊 总样本: {len(df)} (有5d真实收益)")
    
    # === 策略对比 ===
    results = {}
    
    # 策略1: MMoE Top5 (按 overall_score)
    top5_mmoe = []
    for date, group in df.groupby('date'):
        top = group.nlargest(5, 'mmoe_score')
        top5_mmoe.append(top)
    if top5_mmoe:
        t5m = pd.concat(top5_mmoe)
        results['MMoE Top5 (by score)'] = {
            'avg_5d': t5m['actual_5d'].mean(),
            'med_5d': t5m['actual_5d'].median(),
            'win_rate': (t5m['actual_5d'] > 0).mean() * 100,
            'avg_20d': t5m['actual_20d'].mean() if 'actual_20d' in t5m and t5m['actual_20d'].notna().any() else None,
            'n': len(t5m),
            'days': t5m['date'].nunique(),
        }
    
    # 策略2: MMoE Top5 (按 direction_prob)
    top5_dir = []
    for date, group in df.groupby('date'):
        top = group.nlargest(5, 'mmoe_dir')
        top5_dir.append(top)
    if top5_dir:
        t5d = pd.concat(top5_dir)
        results['MMoE Top5 (by dir_prob)'] = {
            'avg_5d': t5d['actual_5d'].mean(),
            'med_5d': t5d['actual_5d'].median(),
            'win_rate': (t5d['actual_5d'] > 0).mean() * 100,
            'avg_20d': t5d['actual_20d'].mean() if t5d['actual_20d'].notna().any() else None,
            'n': len(t5d),
            'days': t5d['date'].nunique(),
        }
    
    # 策略3: BLUE 基线 (按 blue_daily 排序)
    top5_blue = []
    for date, group in df.groupby('date'):
        top = group.nlargest(5, 'blue_daily')
        top5_blue.append(top)
    if top5_blue:
        t5b = pd.concat(top5_blue)
        results['BLUE Top5 (baseline)'] = {
            'avg_5d': t5b['actual_5d'].mean(),
            'med_5d': t5b['actual_5d'].median(),
            'win_rate': (t5b['actual_5d'] > 0).mean() * 100,
            'avg_20d': t5b['actual_20d'].mean() if t5b['actual_20d'].notna().any() else None,
            'n': len(t5b),
            'days': t5b['date'].nunique(),
        }
    
    # 策略4: 全部股票平均 (市场基准)
    results['All signals (market)'] = {
        'avg_5d': df['actual_5d'].mean(),
        'med_5d': df['actual_5d'].median(),
        'win_rate': (df['actual_5d'] > 0).mean() * 100,
        'avg_20d': df['actual_20d'].mean() if df['actual_20d'].notna().any() else None,
        'n': len(df),
        'days': df['date'].nunique(),
    }
    
    # === 打印结果 ===
    print(f"\n{'='*75}")
    print(f"📊 回测结果 ({test_dates[-1]} ~ {test_dates[0]})")
    print(f"{'='*75}")
    print(f"{'策略':<28s} {'5d均值':>8s} {'5d中位':>8s} {'胜率':>7s} {'20d均值':>8s} {'样本':>6s}")
    print(f"{'-'*75}")
    
    for name, r in sorted(results.items(), key=lambda x: -x[1]['avg_5d']):
        avg20 = f"{r['avg_20d']:+.2f}%" if r['avg_20d'] is not None else "N/A"
        print(f"{name:<28s} {r['avg_5d']:>+7.2f}% {r['med_5d']:>+7.2f}% {r['win_rate']:>6.1f}% {avg20:>8s} {r['n']:>6d}")
    
    # === 按日对比 ===
    print(f"\n{'='*75}")
    print(f"📅 逐日对比: MMoE Top5 vs BLUE Top5")
    print(f"{'='*75}")
    print(f"{'日期':<12s} {'MMoE 5d':>8s} {'BLUE 5d':>8s} {'差值':>8s} {'Winner':>8s}")
    print(f"{'-'*50}")
    
    mmoe_wins = 0
    total_days = 0
    
    for date in sorted(df['date'].unique()):
        dg = df[df['date'] == date]
        if len(dg) < 3:
            continue
        
        mmoe_top = dg.nlargest(5, 'mmoe_score')['actual_5d'].mean()
        blue_top = dg.nlargest(5, 'blue_daily')['actual_5d'].mean()
        diff = mmoe_top - blue_top
        winner = "MMoE ✅" if diff > 0 else "BLUE" if diff < 0 else "TIE"
        
        if diff > 0:
            mmoe_wins += 1
        total_days += 1
        
        print(f"{date:<12s} {mmoe_top:>+7.2f}% {blue_top:>+7.2f}% {diff:>+7.2f}% {winner:>8s}")
    
    if total_days > 0:
        print(f"\nMMoE 胜率: {mmoe_wins}/{total_days} = {mmoe_wins/total_days:.0%}")


if __name__ == '__main__':
    t0 = time.time()
    run_backtest()
    print(f"\n⏱ 总耗时: {time.time()-t0:.0f}s")
