#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 Supabase 中字段缺失的记录，重新计算并更新。
用于修复 GitHub Actions 写入时漏掉字段的问题。
"""
import os, sys, time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

os.environ.setdefault('SUPABASE_URL', 'https://worqpdsypymnzqjbidyz.supabase.co')
os.environ.setdefault('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndvcnFwZHN5cHltbnpxamJpZHl6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk4MTA5MjksImV4cCI6MjA4NTM4NjkyOX0.UzE54Q4QB1mQZqRp_jn4BWGFOtWN3GAscrmGpHpMG9U')

from db.supabase_db import get_supabase, _to_json_native
from services.scan_service import analyze_stock_for_date
from db.database import insert_scan_result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='Date to fix (YYYY-MM-DD)')
    parser.add_argument('--market', default='CN', help='Market (US/CN)')
    args = parser.parse_args()

    sb = get_supabase()
    if not sb:
        print("❌ Supabase not available")
        return

    # 查出该日期所有 regime=null 的记录
    r = sb.table('scan_results').select('symbol').eq('scan_date', args.date).eq('market', args.market).is_('regime', 'null').execute()
    symbols = [row['symbol'] for row in r.data]
    print(f"📋 {args.date} ({args.market}): {len(symbols)} records with missing fields")

    if not symbols:
        print("✅ All records already have data!")
        return

    updated = 0
    errors = 0
    for i, sym in enumerate(symbols):
        try:
            result = analyze_stock_for_date(sym, args.date, market=args.market)
            if result:
                # 构建更新记录
                update = _to_json_native({
                    'stop_loss': result.get('Stop_Loss'),
                    'shares_rec': result.get('Shares_Rec'),
                    'wave_phase': result.get('Wave_Phase'),
                    'wave_desc': result.get('Wave_Desc'),
                    'chan_signal': str(result.get('Chan_Signal', '')),
                    'chan_desc': result.get('Chan_Desc'),
                    'regime': result.get('Regime'),
                    'profit_ratio': result.get('Profit_Ratio'),
                    'vp_rating': result.get('VP_Rating'),
                    'risk_reward_score': result.get('Risk_Reward_Score'),
                    'strat_d_trend': result.get('Strat_D_Trend'),
                    'strat_c_resonance': result.get('Strat_C_Resonance'),
                    'legacy_signal': result.get('Legacy_Signal'),
                    'adaptive_thresh': result.get('Adaptive_Thresh'),
                })
                update = {k: v for k, v in update.items() if v is not None}
                
                sb.table('scan_results').update(update).eq(
                    'symbol', sym
                ).eq('scan_date', args.date).eq('market', args.market).execute()
                updated += 1
            else:
                pass  # 股票无信号，跳过
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️ {sym}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(symbols)}] updated={updated}, errors={errors}")

    print(f"\n✅ Done! Updated {updated}/{len(symbols)}, errors={errors}")


if __name__ == '__main__':
    main()
