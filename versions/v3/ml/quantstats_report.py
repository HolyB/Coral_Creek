#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QuantStats 回测报告生成器
========================

为所有策略和 ML 模型生成专业回测报告:
- 自动从 scan_results + 历史价格构建策略收益序列
- 对比基准 (SPY / 沪深300)
- 生成 HTML 报告 + 关键指标
- 支持日度/周度/月度报告
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import quantstats as qs
    QS_AVAILABLE = True
except ImportError:
    QS_AVAILABLE = False
    print("⚠️ pip install quantstats")

REPORT_DIR = Path(__file__).parent.parent / 'reports' / 'backtest'


def build_strategy_returns(market: str = 'US',
                           days_back: int = 180,
                           holding_days: int = 10,
                           min_blue: float = 50,
                           price_range: Tuple = (5, 500)) -> pd.Series:
    """
    从 scan_results 构建信号策略的日收益序列

    策略: 每天买入当天 BLUE >= min_blue 的信号，持有 holding_days 天后卖出
    等权分配到所有信号

    Returns:
        pd.Series: 日收益率序列 (index=DatetimeIndex)
    """
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history

    init_db()
    dates = get_scanned_dates(market=market)

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    dates = [d for d in dates if d >= cutoff]
    dates.sort()

    # 收集所有交易
    all_trades = []

    for scan_date in dates:
        results = query_scan_results(scan_date=scan_date, market=market, limit=500)
        for r in results:
            sym = r.get('symbol', '')
            blue = float(r.get('blue_daily', 0) or 0)
            price = float(r.get('price', 0) or 0)

            if not sym or blue < min_blue:
                continue
            if not (price_range[0] < price < price_range[1]):
                continue

            # 获取历史价格
            hist = get_stock_history(sym, market, days=holding_days + 30)
            if hist is None or hist.empty:
                continue

            # 标准化列名
            if 'Close' not in hist.columns:
                for c in hist.columns:
                    if c.lower() == 'close':
                        hist = hist.rename(columns={c: 'Close'})
                        break

            if 'Close' not in hist.columns:
                continue

            # 找入场日
            if not isinstance(hist.index, pd.DatetimeIndex):
                if 'Date' in hist.columns:
                    hist = hist.set_index('Date')
                elif 'date' in hist.columns:
                    hist = hist.set_index('date')
                hist.index = pd.to_datetime(hist.index)

            entry_date = pd.Timestamp(scan_date)
            mask = hist.index >= entry_date
            if mask.sum() < 2:
                continue

            future = hist.loc[mask].head(holding_days + 1)
            if len(future) < 2:
                continue

            entry_price = float(future.iloc[0]['Close'])
            if entry_price <= 0:
                continue

            # 每日收益
            for i in range(1, len(future)):
                day_return = float(future.iloc[i]['Close']) / float(future.iloc[i-1]['Close']) - 1
                trade_date = future.index[i]
                all_trades.append({
                    'date': trade_date,
                    'symbol': sym,
                    'return': day_return,
                })

    if not all_trades:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_trades)
    # 每日等权平均收益
    daily_returns = df.groupby('date')['return'].mean()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns = daily_returns.sort_index()

    return daily_returns


def build_mmoe_returns(market: str = 'US',
                       days_back: int = 90) -> pd.Series:
    """
    构建 MMoE 模型选股的日收益序列

    策略: MMoE score > 0.6 的信号，持有 5 天
    """
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history

    init_db()

    # 加载 MMoE 缓存
    cache_dir = Path(__file__).parent / 'saved_models' / 'mmoe_cache'
    dates = sorted(cache_dir.glob(f'{market.lower()}_*.json'), reverse=True)

    import json
    all_trades = []

    for cache_file in dates[:days_back]:
        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except:
            continue

        scan_date = cache_file.stem.split('_', 1)[1]

        for sym, data in cache.items():
            if not isinstance(data, dict):
                continue

            dir_prob = float(data.get('direction_prob', 0.5))
            pred_return = float(data.get('pred_return', 0))

            if dir_prob < 0.6 or pred_return < 0:
                continue

            # 获取 5 天后的实际收益
            hist = get_stock_history(sym, market, days=30)
            if hist is None or hist.empty:
                continue

            if 'Close' not in hist.columns:
                for c in hist.columns:
                    if c.lower() == 'close':
                        hist = hist.rename(columns={c: 'Close'})
                        break

            if not isinstance(hist.index, pd.DatetimeIndex):
                if 'Date' in hist.columns:
                    hist = hist.set_index('Date')
                elif 'date' in hist.columns:
                    hist = hist.set_index('date')
                hist.index = pd.to_datetime(hist.index)

            entry_date = pd.Timestamp(scan_date)
            mask = hist.index >= entry_date
            future = hist.loc[mask].head(6)
            if len(future) < 2:
                continue

            for i in range(1, len(future)):
                day_ret = float(future.iloc[i]['Close']) / float(future.iloc[i-1]['Close']) - 1
                all_trades.append({
                    'date': future.index[i],
                    'symbol': sym,
                    'return': day_ret,
                })

    if not all_trades:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_trades)
    daily_returns = df.groupby('date')['return'].mean()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    return daily_returns.sort_index()


def generate_report(returns: pd.Series,
                    strategy_name: str = 'Strategy',
                    benchmark: str = 'SPY',
                    market: str = 'US',
                    output_dir: Optional[str] = None) -> Dict:
    """
    用 QuantStats 生成回测报告

    Args:
        returns: 日收益率序列
        strategy_name: 策略名称
        benchmark: 基准 (SPY/QQQ/000300.SS)
        market: 市场
        output_dir: 输出目录

    Returns:
        关键指标字典
    """
    if not QS_AVAILABLE:
        return {'error': 'quantstats not installed'}

    if returns.empty or len(returns) < 5:
        return {'error': 'not enough data'}

    # 清理数据
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
    returns = returns.clip(-0.5, 0.5)  # 限制极端值

    if output_dir is None:
        output_dir = str(REPORT_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 基准
    if market == 'CN':
        benchmark = '000300.SS'  # 沪深300

    # 收集指标
    metrics = {}
    try:
        metrics['total_return'] = float(qs.stats.comp(returns) * 100)
        metrics['cagr'] = float(qs.stats.cagr(returns) * 100)
        metrics['sharpe'] = float(qs.stats.sharpe(returns))
        metrics['sortino'] = float(qs.stats.sortino(returns))
        metrics['max_drawdown'] = float(qs.stats.max_drawdown(returns) * 100)
        metrics['calmar'] = float(qs.stats.calmar(returns))
        metrics['volatility'] = float(qs.stats.volatility(returns) * 100)
        metrics['win_rate'] = float(qs.stats.win_rate(returns) * 100)
        metrics['avg_win'] = float(qs.stats.avg_win(returns) * 100)
        metrics['avg_loss'] = float(qs.stats.avg_loss(returns) * 100)
        metrics['profit_factor'] = float(qs.stats.profit_factor(returns))
        metrics['payoff_ratio'] = float(qs.stats.payoff_ratio(returns))
        metrics['best_day'] = float(returns.max() * 100)
        metrics['worst_day'] = float(returns.min() * 100)
        metrics['trading_days'] = len(returns)
    except Exception as e:
        metrics['error'] = str(e)

    # 生成 HTML 报告
    today = datetime.now().strftime('%Y%m%d')
    report_file = Path(output_dir) / f'{strategy_name}_{market}_{today}.html'

    try:
        qs.reports.html(
            returns,
            benchmark=benchmark,
            title=f'{strategy_name} - {market}',
            output=str(report_file),
        )
        metrics['report_path'] = str(report_file)
        print(f"  📊 HTML 报告: {report_file}")
    except Exception as e:
        print(f"  ⚠️ HTML 报告生成失败: {e}")
        # 用 basic 模式
        try:
            qs.reports.html(
                returns,
                title=f'{strategy_name} - {market}',
                output=str(report_file),
                benchmark=None,
            )
            metrics['report_path'] = str(report_file)
        except:
            pass

    # 保存指标 JSON
    import json
    metrics_file = Path(output_dir) / f'{strategy_name}_{market}_{today}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def generate_all_reports(market: str = 'US', days_back: int = 90):
    """
    生成所有策略的回测报告

    包含:
    1. BLUE 信号策略 (基础)
    2. MMoE 选股策略
    3. 对比报告
    """
    print(f"\n{'='*60}")
    print(f"📊 QuantStats 回测报告 — {market}")
    print(f"   周期: 近 {days_back} 天")
    print(f"{'='*60}")

    benchmark = 'SPY' if market == 'US' else '000300.SS'
    all_metrics = {}

    # 1. BLUE 信号策略
    print("\n1️⃣ BLUE 信号策略...")
    blue_returns = build_strategy_returns(
        market=market, days_back=days_back,
        holding_days=10, min_blue=50
    )
    if not blue_returns.empty:
        m = generate_report(blue_returns, 'BLUE_Signal', benchmark, market)
        all_metrics['BLUE_Signal'] = m
        print(f"   Return: {m.get('total_return',0):.1f}%  Sharpe: {m.get('sharpe',0):.2f}  MaxDD: {m.get('max_drawdown',0):.1f}%")
    else:
        print("   ⚠️ 无数据")

    # 2. BLUE 高质量 (>= 80)
    print("\n2️⃣ BLUE 高质量策略 (>=80)...")
    blue_hq = build_strategy_returns(
        market=market, days_back=days_back,
        holding_days=10, min_blue=80
    )
    if not blue_hq.empty:
        m = generate_report(blue_hq, 'BLUE_HQ', benchmark, market)
        all_metrics['BLUE_HQ'] = m
        print(f"   Return: {m.get('total_return',0):.1f}%  Sharpe: {m.get('sharpe',0):.2f}  MaxDD: {m.get('max_drawdown',0):.1f}%")

    # 3. MMoE 策略
    print("\n3️⃣ MMoE 选股策略...")
    mmoe_returns = build_mmoe_returns(market=market, days_back=days_back)
    if not mmoe_returns.empty:
        m = generate_report(mmoe_returns, 'MMoE_Picks', benchmark, market)
        all_metrics['MMoE_Picks'] = m
        print(f"   Return: {m.get('total_return',0):.1f}%  Sharpe: {m.get('sharpe',0):.2f}  MaxDD: {m.get('max_drawdown',0):.1f}%")
    else:
        print("   ⚠️ 无 MMoE 缓存数据")

    # 4. 对比表
    print(f"\n{'='*60}")
    print(f"📈 策略对比")
    print(f"{'='*60}")
    print(f"{'策略':<20} {'收益':>8} {'Sharpe':>8} {'最大回撤':>8} {'胜率':>8} {'交易天':>8}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m.get('total_return',0):>7.1f}% {m.get('sharpe',0):>8.2f} {m.get('max_drawdown',0):>7.1f}% {m.get('win_rate',0):>7.1f}% {m.get('trading_days',0):>8d}")

    # 保存汇总
    import json
    summary_file = REPORT_DIR / f'summary_{market}_{datetime.now().strftime("%Y%m%d")}.json'
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    return all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='QuantStats 回测报告')
    parser.add_argument('--market', default='US', choices=['US', 'CN'])
    parser.add_argument('--days', type=int, default=90)
    args = parser.parse_args()

    generate_all_reports(market=args.market, days_back=args.days)
