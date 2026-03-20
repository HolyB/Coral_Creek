#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML 每日评估 — 生成预测 & 回填结果
==================================

用法:
  python ml/daily_eval.py                  # 记录今日预测 + 刷新历史结果
  python ml/daily_eval.py --report         # 只看准确率报告
  python ml/daily_eval.py --backfill 30    # 回填最近 30 天的历史预测
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def run_daily_predictions(market: str = 'US', date: str = None, picker_cache=None):
    """对指定日期的信号跑 SmartPicker，记录预测"""
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    from ml.smart_picker import SmartPicker
    from services.ml_prediction_tracker import log_predictions_batch

    init_db()

    if date is None:
        dates = get_scanned_dates(market=market)
        if not dates:
            print("❌ 无扫描数据")
            return 0
        date = dates[0]

    print(f"\n📊 {date} ({market})")

    signals = query_scan_results(scan_date=date, market=market, limit=1000)
    if not signals:
        print(f"   无信号")
        return 0

    signals_df = pd.DataFrame(signals)

    # 获取历史数据
    price_history = {}
    for sym in signals_df['symbol'].unique():
        h = get_stock_history(sym, market, days=250)
        if h is not None and not h.empty:
            price_history[sym] = h

    total_logged = 0
    for horizon in ['short', 'medium']:
        if picker_cache and horizon in picker_cache:
            picker = picker_cache[horizon]
        else:
            picker = SmartPicker(market=market, horizon=horizon)
            if picker_cache is not None:
                picker_cache[horizon] = picker

        picks = picker.pick(signals_df, price_history, max_picks=20)

        if picks:
            pick_dicts = [p.to_dict() for p in picks]
            logged = log_predictions_batch(
                pick_dicts, market, date,
                model_version="v2",
                source=f"smart_picker_{horizon}",
            )
            total_logged += logged
            print(f"   [{horizon}] {len(picks)} picks, logged {logged}")
            for i, p in enumerate(picks[:3]):
                star = "⭐" * p.star_rating
                print(f"     {i+1}. {p.symbol:6s} ${p.price:.2f}  "
                      f"score={p.overall_score:.0f}  "
                      f"pred_5d={p.pred_return_5d:+.1f}%  {star}")
        else:
            print(f"   [{horizon}] 无推荐")

    return total_logged


def run_backfill(market: str = 'US', days: int = 30):
    """回填最近 N 天的预测"""
    from db.database import init_db, get_scanned_dates
    init_db()

    dates = get_scanned_dates(market=market)
    target_dates = dates[:days]

    print(f"\n📥 回填最近 {len(target_dates)} 天的预测")
    total = 0
    picker_cache = {}  # 复用 picker 实例
    for d in target_dates:
        n = run_daily_predictions(market=market, date=d, picker_cache=picker_cache)
        total += n

    print(f"\n✅ 总计记录 {total} 条预测")
    return total


def refresh_results():
    """刷新实际结果"""
    from services.ml_prediction_tracker import refresh_prediction_results
    from db.database import init_db
    init_db()

    refreshed = refresh_prediction_results(days_back=90)
    print(f"\n🔄 刷新了 {refreshed} 条预测的实际结果")
    return refreshed


def show_report(market: str = 'US'):
    """显示准确率报告"""
    from services.ml_prediction_tracker import get_prediction_accuracy, get_model_performance_summary
    from db.database import init_db
    init_db()

    print(f"\n{'='*60}")
    print(f"📈 ML 模型准确率报告 ({market})")
    print(f"{'='*60}")

    report = get_prediction_accuracy(market=market, days_back=90)

    total = report.get('total_predictions', 0)
    validated = report.get('validated', 0)

    if total == 0:
        print("  ⚠️ 无预测记录。先运行: python ml/daily_eval.py --backfill 30")
        return

    print(f"\n  总预测数: {total}")
    print(f"  已验证: {validated}")

    dir_acc = report.get('direction_accuracy')
    if dir_acc is not None:
        print(f"  方向准确率: {dir_acc:.1%}")

    avg_pred = report.get('avg_predicted_return')
    avg_actual = report.get('avg_actual_return_5d')
    if avg_pred is not None:
        print(f"  平均预测收益: {avg_pred:+.2f}%")
    if avg_actual is not None:
        print(f"  平均实际收益 (5d): {avg_actual:+.2f}%")

    err = report.get('avg_return_error')
    if err is not None:
        print(f"  平均误差: {err:.2f}%")

    rank_corr = report.get('rank_correlation')
    if rank_corr is not None:
        print(f"  排序相关 (Spearman IC): {rank_corr:.4f}")

    # 按星级分析
    by_star = report.get('by_star', {})
    if by_star:
        print(f"\n  按星级:")
        for stars, data in sorted(by_star.items()):
            n = data.get('count', 0)
            acc = data.get('direction_accuracy')
            ret = data.get('avg_actual_return')
            star_str = "⭐" * int(stars)
            acc_str = f"{acc:.0%}" if acc is not None else "N/A"
            ret_str = f"{ret:+.2f}%" if ret is not None else "N/A"
            print(f"    {star_str} ({n}个): 方向={acc_str}, 实际收益={ret_str}")

    # 简洁摘要
    summary = get_model_performance_summary(market=market, days_back=90)
    if summary.get('status') == 'ok':
        print(f"\n  📊 摘要: {summary.get('summary', '')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML 每日评估')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--report', action='store_true', help='只看准确率报告')
    parser.add_argument('--backfill', type=int, default=0, help='回填 N 天历史预测')
    parser.add_argument('--date', type=str, default=None, help='指定日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.report:
        refresh_results()
        show_report(args.market)
    elif args.backfill > 0:
        run_backfill(args.market, args.backfill)
        refresh_results()
        show_report(args.market)
    else:
        # 默认: 记录今日 + 刷新历史 + 报告
        run_daily_predictions(args.market, args.date)
        refresh_results()
        show_report(args.market)
