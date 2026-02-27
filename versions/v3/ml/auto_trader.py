#!/usr/bin/env python
"""
MMoE 自动交易 + 预测追踪 (多空双向)
====================================

每天收盘后运行:
1. 记录今日 MMoE 推荐 + 回填历史
2. 判断市场温度 → 决定做多/做空/观望
3. 检查持仓 → 达到止损/目标/期限的平仓
4. 多头市场: 买入高 dir_prob 股票
5. 空头市场: 做空低 dir_prob 股票

用法:
  python ml/auto_trader.py                    # 仅记录 + Alpaca 交易
  python ml/auto_trader.py --dry-run          # 仅打印，不下单
  python ml/auto_trader.py --record-only      # 仅记录，不交易
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def run(dry_run: bool = False, record_only: bool = False, max_picks: int = 3,
        max_position_pct: float = 0.10, hold_days: int = 5, market: str = 'US'):
    """
    主流程
    
    Args:
        dry_run: 打印计划但不下单
        record_only: 仅记录预测, 不交易
        max_picks: 最多同时持仓数 (新买入)
        max_position_pct: 单只股票占总权益比例上限
        hold_days: 最长持有天数
        market: 市场
    """
    t0 = time.time()
    today = date.today().isoformat()
    
    print(f"{'='*60}")
    print(f"🤖 MMoE Auto-Trader (多空双向)  {today}")
    print(f"   mode: {'DRY-RUN' if dry_run else 'RECORD-ONLY' if record_only else 'LIVE'}")
    print(f"{'='*60}")
    
    # === Step 1: 获取今日推荐 ===
    from ml.smart_picker import get_todays_picks, SmartPicker
    from services.picks_tracker import PicksTracker
    
    picks = get_todays_picks(market=market, max_picks=20)
    print(f"\n📊 今日推荐: {len(picks)} 只")
    
    # 记录到 tracker
    tracker = PicksTracker(market=market)
    for p in picks:
        tracker.record_pick(p, date=today)
    
    backfilled = tracker.backfill_returns(days_back=10)
    report = tracker.get_performance_report()
    
    print(f"   回填: {backfilled} 条")
    print(f"   历史胜率: {report.get('win_rate_5d', 'N/A')}%")
    print(f"   历史平均5d: {report.get('avg_return_5d', 'N/A')}%")
    
    # === 市场温度判断 ===
    dir_probs = [p.pred_direction_prob for p in picks if p.pred_direction_prob is not None]
    avg_dir = np.mean(dir_probs) if dir_probs else 0.5
    if avg_dir > 0.55:
        regime = 'BULL'
        regime_label = '🟢 多头市场'
    elif avg_dir > 0.45:
        regime = 'NEUTRAL'
        regime_label = '🟡 震荡市'
    elif avg_dir > 0.35:
        regime = 'BEAR_MILD'
        regime_label = '🟠 偏空'
    else:
        regime = 'BEAR'
        regime_label = '🔴 空头市场'
    
    print(f"\n🌡️ 市场温度: {regime_label} (平均 dir_prob={avg_dir:.1%})")
    
    if record_only:
        _print_picks(picks)
        print(f"\n⏱ 完成 ({time.time()-t0:.0f}s) [record-only]")
        return
    
    # === Step 2: 连接 Alpaca ===
    try:
        from execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader(paper=True)
        account = trader.get_account()
        print(f"\n💰 Alpaca 账户: ${account.equity:,.2f} (paper={account.is_paper})")
        print(f"   可用资金: ${account.cash:,.2f}")
    except Exception as e:
        print(f"\n❌ Alpaca 连接失败: {e}")
        _print_picks(picks)
        return
    
    # === Step 3: 检查持仓 → 平仓到期/止损/止盈的 ===
    positions = trader.get_positions()
    print(f"\n📦 当前持仓: {len(positions)} 只")
    
    sell_actions = []
    for pos in positions:
        sym = pos.symbol
        pnl_pct = pos.unrealized_plpc  # 已是百分比
        
        # 查询这只股票的推荐记录
        rec = _find_pick_record(tracker, sym)
        
        reason = None
        
        # 止损: 亏超过 -5%
        if pnl_pct < -5:
            reason = f"止损 ({pnl_pct:+.1f}%)"
        
        # 止盈: 赚超过 +8%
        elif pnl_pct > 8:
            reason = f"止盈 ({pnl_pct:+.1f}%)"
        
        # 到期: 持有超过 hold_days
        elif rec and rec.get('pick_date'):
            pick_date = datetime.strptime(rec['pick_date'], '%Y-%m-%d').date()
            held = (date.today() - pick_date).days
            if held >= hold_days:
                reason = f"到期 ({held}天, {pnl_pct:+.1f}%)"
        
        if reason:
            sell_actions.append({
                'symbol': sym,
                'qty': pos.qty,
                'pnl_pct': pnl_pct,
                'reason': reason,
            })
    
    # 执行卖出
    for sa in sell_actions:
        sym = sa['symbol']
        print(f"   🔴 SELL {sym}: {sa['reason']}")
        if not dry_run:
            try:
                trader.close_position(sym)
                print(f"      ✅ 已平仓")
            except Exception as e:
                print(f"      ❌ 平仓失败: {e}")
    
    # === Step 4: 根据市场温度决定多空 ===
    held_symbols = {p.symbol for p in positions} - {s['symbol'] for s in sell_actions}
    account = trader.get_account()  # 刷新
    budget_per = account.equity * max_position_pct
    available_slots = max_picks - len(held_symbols)
    
    to_buy = []
    to_short = []
    
    if regime in ('BULL', 'NEUTRAL'):
        # === 多头/震荡: 做多高 dir_prob ===
        buy_candidates = [p for p in picks
                         if p.symbol not in held_symbols
                         and p.pred_direction_prob >= 0.50
                         and p.overall_score >= 40]
        buy_candidates.sort(key=lambda x: x.pred_direction_prob, reverse=True)
        to_buy = buy_candidates[:max(available_slots, 0)]
    
    if regime in ('BEAR', 'BEAR_MILD'):
        # === 空头: 做空低 dir_prob ===
        short_candidates = [p for p in picks
                           if p.symbol not in held_symbols
                           and p.pred_direction_prob < 0.15
                           and p.price >= 5.0]  # 避免做空 penny stock
        short_candidates.sort(key=lambda x: x.pred_direction_prob)  # 最看跌的排前面
        to_short = short_candidates[:max(available_slots, 0)]
    
    # 执行做多
    if to_buy:
        print(f"\n   🟢 做多 {len(to_buy)} 只 (预算/股: ${budget_per:,.0f})")
        for p in to_buy:
            if p.price <= 0: continue
            qty = int(budget_per / p.price)
            if qty <= 0: continue
            print(f"   🟢 BUY {p.symbol}: ${p.price:.2f} x {qty} "
                  f"(dir={p.pred_direction_prob:.0%}, score={p.overall_score:.0f})")
            if not dry_run:
                try:
                    order = trader.buy_market(p.symbol, qty)
                    print(f"      ✅ 订单: {order.get('id', 'unknown')}")
                except Exception as e:
                    print(f"      ❌ 下单失败: {e}")
    
    # 执行做空
    if to_short:
        print(f"\n   🔴 做空 {len(to_short)} 只 (预算/股: ${budget_per:,.0f})")
        for p in to_short:
            if p.price <= 0: continue
            qty = int(budget_per / p.price)
            if qty <= 0: continue
            print(f"   🔴 SHORT {p.symbol}: ${p.price:.2f} x {qty} "
                  f"(dir={p.pred_direction_prob:.0%}, score={p.overall_score:.0f})")
            if not dry_run:
                try:
                    order = trader.sell_short(p.symbol, qty)
                    print(f"      ✅ 空单: {order.get('id', 'unknown')}")
                except Exception as e:
                    print(f"      ❌ 做空失败: {e}")
    
    if not to_buy and not to_short:
        print(f"\n   ℹ️ 无新交易 ({regime_label})")
    
    # === Step 5: 汇总 ===
    print(f"\n{'='*60}")
    print(f"📋 执行汇总:")
    print(f"   市场: {regime_label}")
    print(f"   平仓: {len(sell_actions)} 笔")
    print(f"   做多: {len(to_buy)} 笔")
    print(f"   做空: {len(to_short)} 笔")
    print(f"   持仓: {len(held_symbols)} 只")
    
    account = trader.get_account()
    print(f"   权益: ${account.equity:,.2f}")
    print(f"   现金: ${account.cash:,.2f}")
    print(f"⏱ 完成 ({time.time()-t0:.0f}s)")


def _print_picks(picks):
    """打印推荐列表"""
    print(f"\n{'Symbol':<8s} {'Price':>8s} {'Dir':>6s} {'Ret5d':>7s} {'Score':>6s} {'Stars':>5s}")
    print("-" * 45)
    for p in picks[:10]:
        print(f"{p.symbol:<8s} ${p.price:>6.2f} {p.pred_direction_prob:>5.0%} "
              f"{p.pred_return_5d:>+6.1f}% {p.overall_score:>5.0f} "
              f"{'⭐' * p.star_rating}")


def _find_pick_record(tracker, symbol: str) -> Optional[Dict]:
    """查找某只股票最近的推荐记录"""
    recent = tracker.get_recent_picks(days=30)
    for r in reversed(recent):
        rec = r if isinstance(r, dict) else r.__dict__ if hasattr(r, '__dict__') else {}
        if rec.get('symbol') == symbol:
            return rec
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MMoE Auto-Trader')
    parser.add_argument('--dry-run', action='store_true', help='打印计划不下单')
    parser.add_argument('--record-only', action='store_true', help='仅记录预测，不交易')
    parser.add_argument('--market', default='US', help='市场')
    parser.add_argument('--max-picks', type=int, default=3, help='最多同时持仓数')
    parser.add_argument('--hold-days', type=int, default=5, help='最长持有天数')
    parser.add_argument('--max-position', type=float, default=0.10, help='单只仓位上限')
    args = parser.parse_args()
    
    run(
        dry_run=args.dry_run,
        record_only=args.record_only,
        market=args.market,
        max_picks=args.max_picks,
        hold_days=args.hold_days,
        max_position_pct=args.max_position,
    )
