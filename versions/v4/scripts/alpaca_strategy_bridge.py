#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpaca Strategy Bridge — 最优策略 → Alpaca Paper Trading
==========================================================
读取多策略模拟盘中表现最好的策略，将其持仓同步到 Alpaca。

流程:
  1. 从 paper_trading.db 获取最优策略的目标持仓
  2. 从 Alpaca 获取当前持仓
  3. 计算差异 (需要买入/卖出的)
  4. 执行交易

安全:
  - 默认 dry-run 模式（只打印不执行）
  - --execute 才真正下单
  - 使用 AlpacaTrader 的风控体系

用法:
    PYTHONPATH=. python scripts/alpaca_strategy_bridge.py --market US          # dry-run
    PYTHONPATH=. python scripts/alpaca_strategy_bridge.py --market US --execute # 实际执行
    PYTHONPATH=. python scripts/alpaca_strategy_bridge.py --market US --strategy top3_daily --execute
"""
import os, sys, json, sqlite3
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(parent_dir, '.env'))
except ImportError:
    pass

DB_DIR = os.path.join(parent_dir, 'db')
LOG_DB = os.path.join(DB_DIR, 'paper_trading.db')


def init_bridge_log():
    """Initialize bridge log table"""
    conn = sqlite3.connect(LOG_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS bridge_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            market TEXT,
            strategy_key TEXT,
            action TEXT,
            symbol TEXT,
            target_shares REAL,
            actual_shares REAL,
            price REAL,
            status TEXT,
            message TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_action(market, strategy_key, action, symbol, target_shares, actual_shares, price, status, message):
    """Log bridge action"""
    conn = sqlite3.connect(LOG_DB)
    conn.execute(
        '''INSERT INTO bridge_log (timestamp, market, strategy_key, action, symbol, target_shares, actual_shares, price, status, message)
        VALUES (?,?,?,?,?,?,?,?,?,?)''',
        (datetime.now().isoformat(), market, strategy_key, action, symbol,
         target_shares, actual_shares, price, status, message)
    )
    conn.commit()
    conn.close()


def run_bridge(market='US', strategy_key=None, execute=False):
    """
    Sync best strategy's holdings to Alpaca.

    Args:
        market: US only (Alpaca = US market)
        strategy_key: override auto-selected best strategy
        execute: if False, dry-run only
    """
    from scripts.paper_trading import get_strategy_target_holdings, STRATEGIES as PT_STRATEGIES

    init_bridge_log()

    # 1. Get target holdings
    if strategy_key is None:
        from scripts.paper_trading import get_best_strategy
        strategy_key = get_best_strategy(market)

    strat_name = PT_STRATEGIES.get(strategy_key, {}).get('name', strategy_key)
    print(f"\n🔗 Alpaca Strategy Bridge ({market})")
    print(f"   📋 Active strategy: {strat_name} ({strategy_key})")
    print(f"   {'🔴 DRY RUN' if not execute else '🟢 LIVE EXECUTION'}")
    print(f"{'='*60}")

    _, target_holdings = get_strategy_target_holdings(market, strategy_key)
    print(f"   🎯 Target holdings: {len(target_holdings)} positions")
    for sym, info in target_holdings.items():
        print(f"      {sym}: {info['shares']:.1f} shares @ ${info['buy_price']:.2f} (since {info['buy_date']})")

    # 2. Connect to Alpaca
    try:
        from execution.alpaca_trader import AlpacaTrader, ALPACA_SDK_AVAILABLE
        if not ALPACA_SDK_AVAILABLE:
            print("   ❌ alpaca-py not installed")
            return

        trader = AlpacaTrader(paper=True)
        account = trader.get_account()
        print(f"\n   💰 Alpaca Account: ${account.equity:,.0f} (cash: ${account.cash:,.0f})")
        print(f"   📦 Paper: {account.is_paper}")
    except Exception as e:
        print(f"   ❌ Alpaca connection failed: {e}")
        return

    # 3. Get current Alpaca positions
    current_positions = trader.get_positions()
    current_syms = {p.symbol: p for p in current_positions}
    print(f"   📦 Current Alpaca positions: {len(current_positions)}")

    # 4. Calculate diffs
    # Sells: in Alpaca but NOT in target
    to_sell = [sym for sym in current_syms if sym not in target_holdings]
    # Buys: in target but NOT in Alpaca
    to_buy = [sym for sym in target_holdings if sym not in current_syms]
    # Already held (no action)
    held = [sym for sym in target_holdings if sym in current_syms]

    print(f"\n   📊 Actions needed:")
    print(f"      🟢 Buy:  {len(to_buy)} ({', '.join(to_buy[:5])}{'...' if len(to_buy)>5 else ''})")
    print(f"      🔴 Sell: {len(to_sell)} ({', '.join(to_sell[:5])}{'...' if len(to_sell)>5 else ''})")
    print(f"      ⚪ Hold: {len(held)}")

    if not to_sell and not to_buy:
        print("\n   ✅ Portfolio already aligned, no trades needed")
        return

    # 5. Execute sells first (free up capital)
    for sym in to_sell:
        pos = current_syms[sym]
        print(f"\n   🔴 SELL {sym}: {pos.qty:.1f} shares @ ${pos.current_price:.2f}")
        if execute:
            try:
                order = trader.close_position(sym)
                log_action(market, strategy_key, 'sell', sym, 0, pos.qty, pos.current_price, 'executed', f"order_id={order['id']}")
                print(f"      ✅ Order submitted: {order['id']}")
            except Exception as e:
                log_action(market, strategy_key, 'sell', sym, 0, pos.qty, 0, 'failed', str(e))
                print(f"      ❌ Failed: {e}")
        else:
            log_action(market, strategy_key, 'sell', sym, 0, pos.qty, pos.current_price, 'dry_run', '')

    # 6. Execute buys
    # Calculate position size based on account equity
    if to_buy:
        account = trader.get_account()  # Refresh after sells
        pos_size = account.equity * PT_STRATEGIES[strategy_key].get('max_per_pos', 0.10)

    for sym in to_buy:
        target = target_holdings[sym]
        # Use Alpaca's actual price, not our recorded price
        try:
            price = trader.get_latest_price(sym)
            if price <= 0:
                price = target['buy_price']
        except:
            price = target['buy_price']

        qty = int(pos_size / price) if price > 0 else 0
        if qty <= 0:
            print(f"\n   🟢 BUY {sym}: SKIP (insufficient funds or price=0)")
            continue

        print(f"\n   🟢 BUY {sym}: {qty} shares @ ~${price:.2f} = ${qty*price:,.0f}")
        if execute:
            try:
                order = trader.buy_market(sym, qty)
                log_action(market, strategy_key, 'buy', sym, target['shares'], qty, price, 'executed', f"order_id={order['id']}")
                print(f"      ✅ Order submitted: {order['id']}")
            except Exception as e:
                log_action(market, strategy_key, 'buy', sym, target['shares'], qty, price, 'failed', str(e))
                print(f"      ❌ Failed: {e}")
        else:
            log_action(market, strategy_key, 'buy', sym, target['shares'], qty, price, 'dry_run', '')

    print(f"\n{'='*60}")
    print(f"{'✅ Execution complete' if execute else '📋 Dry run complete (use --execute to trade)'}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Alpaca Strategy Bridge')
    parser.add_argument('--market', default='US', choices=['US'])
    parser.add_argument('--strategy', default=None, help='Override strategy (default=auto-select best)')
    parser.add_argument('--execute', action='store_true', help='Actually execute trades (default=dry-run)')
    args = parser.parse_args()

    run_bridge(args.market, args.strategy, args.execute)
