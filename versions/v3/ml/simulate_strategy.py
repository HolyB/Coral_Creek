#!/usr/bin/env python
"""
MMoE 策略全仿真回测
===================
模拟 auto_trader 的完整交易流程:
- 每天选 Top-3 买入 (方向概率排序)
- 止损 -5%, 止盈 +8%, 最长持有 5 天
- 跟踪每笔交易 + 组合净值曲线
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


def simulate(days_back: int = 60, initial_capital: float = 100000,
             max_positions: int = 3, position_pct: float = 0.10,
             stop_loss: float = -5, take_profit: float = 8, max_hold: int = 5,
             min_dir_prob: float = 0.20, min_score: float = 30):
    """全仿真回测"""
    t0 = time.time()
    
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    from ml.smart_picker import SmartPicker
    
    init_db()
    
    dates = get_scanned_dates(market='US')
    test_dates = sorted(dates[:days_back])
    print(f"📅 回测 {len(test_dates)} 天: {test_dates[0]} ~ {test_dates[-1]}")
    
    picker = SmartPicker(market='US', horizon='short')
    print(f"MMoE: {'✅' if picker.mmoe_model else '❌'}")
    
    # === 预加载所有信号 + 历史 ===
    print("📊 预加载信号数据...")
    daily_signals = {}
    all_symbols = set()
    for d in test_dates:
        sigs = query_scan_results(scan_date=d, market='US', limit=30)
        daily_signals[d] = sigs
        for s in sigs:
            sym = s.get('symbol', '')
            if sym:
                all_symbols.add(sym)
    
    print(f"   {len(all_symbols)} 只标的")
    
    # 预加载历史
    print("📈 预加载价格历史...")
    histories = {}
    for sym in all_symbols:
        h = get_stock_history(sym, 'US', days=400)
        if h is not None and not h.empty:
            if not isinstance(h.index, pd.DatetimeIndex):
                if 'Date' in h.columns:
                    h = h.set_index('Date')
                elif 'date' in h.columns:
                    h = h.set_index('date')
                h.index = pd.to_datetime(h.index)
            histories[sym] = h
    print(f"   {len(histories)} 只有历史数据")
    
    # === 仿真 ===
    cash = initial_capital
    positions = {}  # {symbol: {qty, entry_price, entry_date, pick}}
    trades = []     # 已完成的交易
    equity_curve = []
    daily_actions = []
    
    for di, trade_date in enumerate(test_dates):
        cutoff = pd.to_datetime(trade_date)
        
        # --- 检查持仓: 平仓 ---
        to_close = []
        for sym, pos in list(positions.items()):
            h = histories.get(sym)
            if h is None:
                continue
            
            # 找到 trade_date 当天的收盘价
            h_on = h[h.index <= cutoff]
            if h_on.empty:
                continue
            current_price = float(h_on['Close'].iloc[-1])
            pnl_pct = (current_price / pos['entry_price'] - 1) * 100
            held = (datetime.strptime(trade_date, '%Y-%m-%d') - 
                    datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
            
            reason = None
            if pnl_pct <= stop_loss:
                reason = f"止损({pnl_pct:+.1f}%)"
            elif pnl_pct >= take_profit:
                reason = f"止盈({pnl_pct:+.1f}%)"
            elif held >= max_hold:
                reason = f"到期({held}d,{pnl_pct:+.1f}%)"
            
            if reason:
                to_close.append((sym, current_price, pnl_pct, reason))
        
        for sym, exit_price, pnl_pct, reason in to_close:
            pos = positions.pop(sym)
            pnl = (exit_price - pos['entry_price']) * pos['qty']
            cash += exit_price * pos['qty']
            trades.append({
                'symbol': sym,
                'entry_date': pos['entry_date'],
                'exit_date': trade_date,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'qty': pos['qty'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'dir_prob': pos.get('dir_prob', 0),
            })
            daily_actions.append(f"  🔴 SELL {sym} {reason} pnl={pnl:+.0f}")
        
        # --- 选股: MMoE 评分 ---
        sigs = daily_signals.get(trade_date, [])
        picks = []
        for s in sigs:
            sig = pd.Series(s)
            sym = sig.get('symbol', '')
            price = float(sig.get('price', 0))
            if not sym or price <= 0 or sym in positions:
                continue
            h = histories.get(sym)
            if h is None:
                continue
            h_before = h[h.index <= cutoff]
            if len(h_before) < 60:
                continue
            pick = picker._analyze_stock(sig, h_before, skip_prefilter=True)
            if pick and pick.pred_direction_prob >= min_dir_prob and pick.overall_score >= min_score:
                picks.append(pick)
        
        # 按方向概率排序
        picks.sort(key=lambda x: x.pred_direction_prob, reverse=True)
        
        # --- 买入 ---
        slots = max_positions - len(positions)
        equity_now = cash + sum(
            float(histories.get(sym, pd.DataFrame()).loc[
                histories[sym].index <= cutoff, 'Close'].iloc[-1]) * pos['qty']
            for sym, pos in positions.items()
            if sym in histories and not histories[sym][histories[sym].index <= cutoff].empty
        )
        budget = equity_now * position_pct
        
        bought = 0
        for pick in picks[:max(slots, 0)]:
            price = pick.price
            qty = int(budget / price)
            if qty <= 0 or cash < price * qty:
                continue
            
            cost = price * qty
            cash -= cost
            positions[pick.symbol] = {
                'qty': qty,
                'entry_price': price,
                'entry_date': trade_date,
                'dir_prob': pick.pred_direction_prob,
            }
            bought += 1
            daily_actions.append(
                f"  🟢 BUY  {pick.symbol} ${price:.2f}x{qty} "
                f"dir={pick.pred_direction_prob:.0%} score={pick.overall_score:.0f}")
        
        # --- 计算当日净值 ---
        pos_value = 0
        for sym, pos in positions.items():
            h = histories.get(sym)
            if h is None:
                continue
            h_on = h[h.index <= cutoff]
            if h_on.empty:
                continue
            pos_value += float(h_on['Close'].iloc[-1]) * pos['qty']
        
        total_equity = cash + pos_value
        equity_curve.append({
            'date': trade_date,
            'equity': total_equity,
            'cash': cash,
            'positions': len(positions),
            'bought': bought,
            'sold': len(to_close),
        })
        
        if (di + 1) % 10 == 0:
            ret = (total_equity / initial_capital - 1) * 100
            print(f"  Day {di+1}: ${total_equity:,.0f} ({ret:+.1f}%) pos={len(positions)}")
    
    # === 强制平仓剩余持仓 ===
    final_date = test_dates[-1]
    cutoff = pd.to_datetime(final_date)
    for sym, pos in list(positions.items()):
        h = histories.get(sym)
        if h is None:
            continue
        h_on = h[h.index <= cutoff]
        if h_on.empty:
            continue
        exit_price = float(h_on['Close'].iloc[-1])
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
        pnl = (exit_price - pos['entry_price']) * pos['qty']
        cash += exit_price * pos['qty']
        trades.append({
            'symbol': sym, 'entry_date': pos['entry_date'],
            'exit_date': final_date, 'entry_price': pos['entry_price'],
            'exit_price': exit_price, 'qty': pos['qty'],
            'pnl': pnl, 'pnl_pct': pnl_pct, 'reason': '回测结束',
            'dir_prob': pos.get('dir_prob', 0),
        })
    
    # === 结果 ===
    df_eq = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trades)
    
    total_return = (cash / initial_capital - 1) * 100
    
    print(f"\n{'='*65}")
    print(f"📊 MMoE 策略回测结果 ({test_dates[0]} ~ {test_dates[-1]})")
    print(f"{'='*65}")
    print(f"初始资金:   ${initial_capital:,.0f}")
    print(f"最终资金:   ${cash:,.0f}")
    print(f"总收益:     {total_return:+.2f}%")
    
    if not df_eq.empty:
        peak = df_eq['equity'].cummax()
        dd = (df_eq['equity'] / peak - 1) * 100
        max_dd = dd.min()
        print(f"最大回撤:   {max_dd:.2f}%")
        print(f"年化收益:   {total_return / max(len(test_dates)/252, 0.01):+.1f}%")
    
    if not df_trades.empty:
        n = len(df_trades)
        wins = (df_trades['pnl'] > 0).sum()
        print(f"\n--- 交易统计 ---")
        print(f"总交易数:   {n}")
        print(f"胜率:       {wins/n:.1%} ({wins}/{n})")
        print(f"平均盈亏:   {df_trades['pnl_pct'].mean():+.2f}%")
        print(f"盈利均值:   {df_trades[df_trades['pnl']>0]['pnl_pct'].mean():+.2f}%" if wins > 0 else "")
        print(f"亏损均值:   {df_trades[df_trades['pnl']<=0]['pnl_pct'].mean():+.2f}%" if n-wins > 0 else "")
        print(f"最大单笔赢: {df_trades['pnl_pct'].max():+.2f}%")
        print(f"最大单笔亏: {df_trades['pnl_pct'].min():+.2f}%")
        print(f"总盈亏:     ${df_trades['pnl'].sum():+,.0f}")
        
        # 按平仓原因统计
        print(f"\n--- 按平仓原因 ---")
        for reason_prefix in ['止损', '止盈', '到期', '回测']:
            mask = df_trades['reason'].str.startswith(reason_prefix)
            if mask.any():
                sub = df_trades[mask]
                print(f"  {reason_prefix}: {len(sub)}笔, avg={sub['pnl_pct'].mean():+.1f}%, "
                      f"win={( sub['pnl']>0 ).mean():.0%}")
        
        # Top 10 交易
        print(f"\n--- Top 10 交易 ---")
        print(f"{'Symbol':<8s} {'Entry':>10s} {'Exit':>10s} {'PnL%':>7s} {'$PnL':>8s} {'Dir':>5s} {'Reason'}")
        print("-" * 65)
        for _, t in df_trades.nlargest(5, 'pnl_pct').iterrows():
            print(f"{t['symbol']:<8s} {t['entry_date']:>10s} {t['exit_date']:>10s} "
                  f"{t['pnl_pct']:>+6.1f}% ${t['pnl']:>+7.0f} {t['dir_prob']:>4.0%} {t['reason']}")
        print("...")
        for _, t in df_trades.nsmallest(5, 'pnl_pct').iterrows():
            print(f"{t['symbol']:<8s} {t['entry_date']:>10s} {t['exit_date']:>10s} "
                  f"{t['pnl_pct']:>+6.1f}% ${t['pnl']:>+7.0f} {t['dir_prob']:>4.0%} {t['reason']}")
    
    print(f"\n⏱ 耗时: {time.time()-t0:.0f}s")
    return df_eq, df_trades


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=60)
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--max-pos', type=int, default=3)
    parser.add_argument('--hold', type=int, default=5)
    parser.add_argument('--stop', type=float, default=-5)
    parser.add_argument('--profit', type=float, default=8)
    args = parser.parse_args()
    
    simulate(
        days_back=args.days,
        initial_capital=args.capital,
        max_positions=args.max_pos,
        max_hold=args.hold,
        stop_loss=args.stop,
        take_profit=args.profit,
    )
