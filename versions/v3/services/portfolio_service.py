#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持仓管理服务 - 实时盈亏计算、模拟交易
"""
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import (
    get_portfolio, add_to_watchlist, add_trade, 
    get_trades, update_watchlist_status, get_connection
)
from data_fetcher import get_stock_data


# ==================== 模拟账户配置 ====================

PAPER_ACCOUNT_INITIAL = 100000.0  # 模拟账户初始资金
PAPER_SUBACCOUNT_INITIAL = 20000.0  # 子账户默认初始资金
DEFAULT_MAX_SINGLE_POSITION_PCT = 0.30  # 子账户默认单票上限
DEFAULT_MAX_DRAWDOWN_PCT = 0.20  # 子账户默认最大回撤


def get_current_price(symbol: str, market: str = 'US') -> Optional[float]:
    """获取股票当前价格"""
    try:
        df = get_stock_data(symbol, market=market, days=5)
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
    except Exception as e:
        print(f"获取 {symbol} 价格失败: {e}")
    return None


def calculate_portfolio_pnl(portfolio: List[Dict]) -> List[Dict]:
    """
    计算持仓盈亏
    
    Args:
        portfolio: 持仓列表
    
    Returns:
        带有盈亏信息的持仓列表
    """
    enriched = []
    
    for item in portfolio:
        symbol = item['symbol']
        market = item.get('market', 'US')
        entry_price = float(item['entry_price'])
        shares = int(item['shares'])
        
        # 获取当前价格
        current_price = get_current_price(symbol, market)
        
        if current_price:
            # 计算盈亏
            cost = entry_price * shares
            market_value = current_price * shares
            unrealized_pnl = market_value - cost
            unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            
            item['current_price'] = current_price
            item['market_value'] = market_value
            item['cost'] = cost
            item['unrealized_pnl'] = unrealized_pnl
            item['unrealized_pnl_pct'] = unrealized_pnl_pct
        else:
            item['current_price'] = None
            item['market_value'] = None
            item['cost'] = entry_price * shares
            item['unrealized_pnl'] = None
            item['unrealized_pnl_pct'] = None
        
        enriched.append(item)
    
    return enriched


def get_portfolio_summary(market: str = None) -> Dict:
    """
    获取持仓汇总统计
    
    Returns:
        {
            'total_cost': 总成本,
            'total_market_value': 总市值,
            'total_pnl': 总盈亏,
            'total_pnl_pct': 总收益率,
            'positions': 持仓数,
            'winners': 盈利数,
            'losers': 亏损数
        }
    """
    portfolio = get_portfolio(status='holding', market=market)
    
    if not portfolio:
        return {
            'total_cost': 0,
            'total_market_value': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'positions': 0,
            'winners': 0,
            'losers': 0
        }
    
    # 计算盈亏
    enriched = calculate_portfolio_pnl(portfolio)
    
    total_cost = 0
    total_market_value = 0
    winners = 0
    losers = 0
    
    for item in enriched:
        cost = item.get('cost', 0) or 0
        mv = item.get('market_value', 0) or 0
        pnl = item.get('unrealized_pnl', 0) or 0
        
        total_cost += cost
        total_market_value += mv
        
        if pnl > 0:
            winners += 1
        elif pnl < 0:
            losers += 1
    
    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'total_cost': total_cost,
        'total_market_value': total_market_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'positions': len(enriched),
        'winners': winners,
        'losers': losers,
        'details': enriched
    }


# ==================== 模拟交易账户 ====================

def init_paper_account():
    """初始化模拟账户表"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_account (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name VARCHAR(50) DEFAULT 'default',
            initial_capital REAL DEFAULT 100000,
            cash_balance REAL DEFAULT 100000,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_account_name
        ON paper_account(account_name)
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name VARCHAR(50) DEFAULT 'default',
            symbol VARCHAR(20) NOT NULL,
            market VARCHAR(10) DEFAULT 'US',
            shares INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_name, symbol, market)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name VARCHAR(50) DEFAULT 'default',
            symbol VARCHAR(20) NOT NULL,
            market VARCHAR(10) DEFAULT 'US',
            trade_type VARCHAR(10) NOT NULL,
            price REAL NOT NULL,
            shares INTEGER NOT NULL,
            commission REAL DEFAULT 0,
            trade_date DATE NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_account_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name VARCHAR(50) NOT NULL,
            strategy_note TEXT DEFAULT '',
            max_single_position_pct REAL DEFAULT 0.30,
            max_drawdown_pct REAL DEFAULT 0.20,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_name)
        )
    """)
    
    # 检查是否已有默认账户
    cursor.execute("SELECT COUNT(*) FROM paper_account WHERE account_name = 'default'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO paper_account (account_name, initial_capital, cash_balance)
            VALUES ('default', ?, ?)
        """, (PAPER_ACCOUNT_INITIAL, PAPER_ACCOUNT_INITIAL))
    cursor.execute("SELECT COUNT(*) FROM paper_account_config WHERE account_name = 'default'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO paper_account_config (account_name, strategy_note, max_single_position_pct, max_drawdown_pct)
            VALUES ('default', '', ?, ?)
        """, (DEFAULT_MAX_SINGLE_POSITION_PCT, DEFAULT_MAX_DRAWDOWN_PCT))
    
    conn.commit()
    conn.close()


def list_paper_accounts() -> List[Dict]:
    """获取所有模拟子账户"""
    init_paper_account()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT account_name, initial_capital, cash_balance, created_at, updated_at
        FROM paper_account
        ORDER BY CASE WHEN account_name = 'default' THEN 0 ELSE 1 END, account_name
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def create_paper_account(account_name: str, initial_capital: float = PAPER_SUBACCOUNT_INITIAL) -> Dict:
    """创建策略子账户"""
    init_paper_account()
    account_name = (account_name or "").strip()
    if not account_name:
        return {'success': False, 'error': '账户名不能为空'}
    if len(account_name) > 50:
        return {'success': False, 'error': '账户名过长（最多50字符）'}
    if initial_capital <= 0:
        return {'success': False, 'error': '初始资金必须大于0'}

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM paper_account WHERE account_name = ?", (account_name,))
        if cursor.fetchone():
            return {'success': False, 'error': f'账户 {account_name} 已存在'}

        cursor.execute("""
            INSERT INTO paper_account (account_name, initial_capital, cash_balance)
            VALUES (?, ?, ?)
        """, (account_name, float(initial_capital), float(initial_capital)))
        cursor.execute("""
            INSERT INTO paper_account_config (account_name, strategy_note, max_single_position_pct, max_drawdown_pct)
            VALUES (?, '', ?, ?)
        """, (account_name, DEFAULT_MAX_SINGLE_POSITION_PCT, DEFAULT_MAX_DRAWDOWN_PCT))
        conn.commit()
        return {'success': True, 'account_name': account_name, 'initial_capital': float(initial_capital)}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def get_paper_account(account_name: str = 'default') -> Dict:
    """获取模拟账户信息"""
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 获取账户基本信息
    cursor.execute("""
        SELECT * FROM paper_account WHERE account_name = ?
    """, (account_name,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        seed_capital = PAPER_ACCOUNT_INITIAL if account_name == 'default' else PAPER_SUBACCOUNT_INITIAL
        created = create_paper_account(account_name, seed_capital)
        if not created.get('success'):
            return None

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM paper_account WHERE account_name = ?
        """, (account_name,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
    
    account = dict(row)
    
    # 获取持仓
    cursor.execute("""
        SELECT * FROM paper_positions WHERE account_name = ?
    """, (account_name,))
    positions = [dict(r) for r in cursor.fetchall()]

    # 获取账户风控配置
    cursor.execute("""
        SELECT strategy_note, max_single_position_pct, max_drawdown_pct
        FROM paper_account_config WHERE account_name = ?
    """, (account_name,))
    cfg_row = cursor.fetchone()
    if cfg_row:
        account_config = dict(cfg_row)
    else:
        account_config = {
            'strategy_note': '',
            'max_single_position_pct': DEFAULT_MAX_SINGLE_POSITION_PCT,
            'max_drawdown_pct': DEFAULT_MAX_DRAWDOWN_PCT
        }
    
    conn.close()
    
    # 计算持仓市值
    total_position_value = 0
    enriched_positions = []
    
    for pos in positions:
        current_price = get_current_price(pos['symbol'], pos['market'])
        if current_price:
            market_value = current_price * pos['shares']
            cost = pos['avg_cost'] * pos['shares']
            pnl = market_value - cost
            pnl_pct = (current_price - pos['avg_cost']) / pos['avg_cost'] * 100
            
            pos['current_price'] = current_price
            pos['market_value'] = market_value
            pos['cost'] = cost
            pos['unrealized_pnl'] = pnl
            pos['unrealized_pnl_pct'] = pnl_pct
            
            total_position_value += market_value
        else:
            pos['current_price'] = None
            pos['market_value'] = pos['avg_cost'] * pos['shares']
            pos['cost'] = pos['avg_cost'] * pos['shares']
            pos['unrealized_pnl'] = 0
            pos['unrealized_pnl_pct'] = 0
            total_position_value += pos['market_value']
        
        enriched_positions.append(pos)
    
    # 总资产 = 现金 + 持仓市值
    total_equity = account['cash_balance'] + total_position_value
    total_pnl = total_equity - account['initial_capital']
    total_pnl_pct = total_pnl / account['initial_capital'] * 100
    
    return {
        'account_name': account_name,
        'initial_capital': account['initial_capital'],
        'cash_balance': account['cash_balance'],
        'position_value': total_position_value,
        'total_equity': total_equity,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'positions': enriched_positions,
        'config': account_config
    }


def get_paper_account_config(account_name: str = 'default') -> Dict:
    """获取子账户配置"""
    init_paper_account()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT strategy_note, max_single_position_pct, max_drawdown_pct
        FROM paper_account_config WHERE account_name = ?
    """, (account_name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        'strategy_note': '',
        'max_single_position_pct': DEFAULT_MAX_SINGLE_POSITION_PCT,
        'max_drawdown_pct': DEFAULT_MAX_DRAWDOWN_PCT
    }


def update_paper_account_config(account_name: str,
                                strategy_note: str = '',
                                max_single_position_pct: float = DEFAULT_MAX_SINGLE_POSITION_PCT,
                                max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT) -> Dict:
    """更新子账户配置"""
    init_paper_account()
    max_single_position_pct = max(0.05, min(1.0, float(max_single_position_pct)))
    max_drawdown_pct = max(0.05, min(1.0, float(max_drawdown_pct)))
    strategy_note = (strategy_note or '').strip()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM paper_account WHERE account_name = ?", (account_name,))
        if not cursor.fetchone():
            created = create_paper_account(account_name, PAPER_SUBACCOUNT_INITIAL)
            if not created.get('success'):
                return {'success': False, 'error': f'账户不存在且创建失败: {account_name}'}

        cursor.execute("""
            INSERT INTO paper_account_config (account_name, strategy_note, max_single_position_pct, max_drawdown_pct, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(account_name) DO UPDATE SET
                strategy_note=excluded.strategy_note,
                max_single_position_pct=excluded.max_single_position_pct,
                max_drawdown_pct=excluded.max_drawdown_pct,
                updated_at=CURRENT_TIMESTAMP
        """, (account_name, strategy_note, max_single_position_pct, max_drawdown_pct))
        conn.commit()
        return {'success': True}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def paper_buy(symbol: str, shares: int, price: float = None, 
              market: str = 'US', account_name: str = 'default') -> Dict:
    """
    模拟买入
    
    Args:
        symbol: 股票代码
        shares: 买入股数
        price: 买入价格 (None 则使用当前价)
        market: 市场
        account_name: 账户名
    
    Returns:
        交易结果
    """
    init_paper_account()
    
    # 获取价格
    if price is None:
        price = get_current_price(symbol, market)
        if price is None:
            return {'success': False, 'error': f'无法获取 {symbol} 当前价格'}
    
    cost = price * shares
    commission = cost * 0.001  # 0.1% 佣金
    total_cost = cost + commission
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 检查账户与现金余额
        cursor.execute("SELECT initial_capital, cash_balance FROM paper_account WHERE account_name = ?",
                      (account_name,))
        row = cursor.fetchone()
        if not row:
            return {'success': False, 'error': '账户不存在'}
        
        initial_capital = float(row['initial_capital'])
        cash_balance = row['cash_balance']

        # 读取子账户配置（默认值兜底）
        cursor.execute("""
            SELECT strategy_note, max_single_position_pct, max_drawdown_pct
            FROM paper_account_config
            WHERE account_name = ?
        """, (account_name,))
        cfg = cursor.fetchone()
        max_single_position_pct = float(cfg['max_single_position_pct']) if cfg and cfg['max_single_position_pct'] else DEFAULT_MAX_SINGLE_POSITION_PCT
        max_drawdown_pct = float(cfg['max_drawdown_pct']) if cfg and cfg['max_drawdown_pct'] else DEFAULT_MAX_DRAWDOWN_PCT

        # 估算当前权益和仓位（用持仓成本估值，避免依赖外部行情）
        cursor.execute("""
            SELECT COALESCE(SUM(shares * avg_cost), 0) AS pos_value
            FROM paper_positions
            WHERE account_name = ?
        """, (account_name,))
        total_position_value = float(cursor.fetchone()['pos_value'] or 0.0)
        current_equity = float(cash_balance) + total_position_value

        # 风险限制1：单票仓位上限
        cursor.execute("""
            SELECT COALESCE(SUM(shares * avg_cost), 0) AS symbol_value
            FROM paper_positions
            WHERE account_name = ? AND symbol = ? AND market = ?
        """, (account_name, symbol, market))
        existing_symbol_value = float(cursor.fetchone()['symbol_value'] or 0.0)
        post_trade_symbol_value = existing_symbol_value + cost
        position_limit_value = current_equity * max_single_position_pct
        if position_limit_value > 0 and post_trade_symbol_value > position_limit_value:
            return {
                'success': False,
                'error': (
                    f'策略风控拦截: {symbol} 仓位将达 ${post_trade_symbol_value:,.2f}, '
                    f'超过单票上限 ${position_limit_value:,.2f} ({max_single_position_pct*100:.1f}%)'
                )
            }

        # 风险限制2：最大回撤上限（基于子账户权益曲线）
        equity_curve = get_paper_equity_curve(account_name)
        peak_equity = initial_capital
        if not equity_curve.empty and 'total_equity' in equity_curve.columns:
            peak_equity = max(float(equity_curve['total_equity'].max()), peak_equity)
        peak_equity = max(peak_equity, current_equity)
        drawdown_pct = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        if drawdown_pct >= max_drawdown_pct:
            return {
                'success': False,
                'error': (
                    f'策略风控拦截: 当前回撤 {drawdown_pct*100:.2f}% '
                    f'已超过阈值 {max_drawdown_pct*100:.2f}%，禁止开新仓'
                )
            }
        
        if cash_balance < total_cost:
            return {'success': False, 'error': f'现金不足: 需要 ${total_cost:.2f}, 余额 ${cash_balance:.2f}'}
        
        # 扣除现金
        new_balance = cash_balance - total_cost
        cursor.execute("""
            UPDATE paper_account SET cash_balance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE account_name = ?
        """, (new_balance, account_name))
        
        # 更新持仓 (计算平均成本)
        cursor.execute("""
            SELECT shares, avg_cost FROM paper_positions 
            WHERE account_name = ? AND symbol = ? AND market = ?
        """, (account_name, symbol, market))
        existing = cursor.fetchone()
        
        if existing:
            old_shares = existing['shares']
            old_cost = existing['avg_cost']
            new_shares = old_shares + shares
            new_avg_cost = (old_shares * old_cost + shares * price) / new_shares
            
            cursor.execute("""
                UPDATE paper_positions SET shares = ?, avg_cost = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_name = ? AND symbol = ? AND market = ?
            """, (new_shares, new_avg_cost, account_name, symbol, market))
        else:
            cursor.execute("""
                INSERT INTO paper_positions (account_name, symbol, market, shares, avg_cost)
                VALUES (?, ?, ?, ?, ?)
            """, (account_name, symbol, market, shares, price))
        
        # 记录交易
        cursor.execute("""
            INSERT INTO paper_trades (account_name, symbol, market, trade_type, price, shares, commission, trade_date)
            VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?)
        """, (account_name, symbol, market, price, shares, commission, datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        
        return {
            'success': True,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'cost': cost,
            'commission': commission,
            'new_balance': new_balance
        }
        
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def paper_sell(symbol: str, shares: int, price: float = None,
               market: str = 'US', account_name: str = 'default') -> Dict:
    """
    模拟卖出
    """
    init_paper_account()
    
    # 获取价格
    if price is None:
        price = get_current_price(symbol, market)
        if price is None:
            return {'success': False, 'error': f'无法获取 {symbol} 当前价格'}
    
    revenue = price * shares
    commission = revenue * 0.001  # 0.1% 佣金
    net_revenue = revenue - commission
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 检查持仓
        cursor.execute("""
            SELECT shares, avg_cost FROM paper_positions 
            WHERE account_name = ? AND symbol = ? AND market = ?
        """, (account_name, symbol, market))
        existing = cursor.fetchone()
        
        if not existing or existing['shares'] < shares:
            return {'success': False, 'error': f'持仓不足: 持有 {existing["shares"] if existing else 0} 股'}
        
        old_shares = existing['shares']
        avg_cost = existing['avg_cost']
        
        # 计算盈亏
        realized_pnl = (price - avg_cost) * shares - commission
        
        # 更新持仓
        new_shares = old_shares - shares
        if new_shares == 0:
            cursor.execute("""
                DELETE FROM paper_positions 
                WHERE account_name = ? AND symbol = ? AND market = ?
            """, (account_name, symbol, market))
        else:
            cursor.execute("""
                UPDATE paper_positions SET shares = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_name = ? AND symbol = ? AND market = ?
            """, (new_shares, account_name, symbol, market))
        
        # 增加现金
        cursor.execute("SELECT cash_balance FROM paper_account WHERE account_name = ?", 
                      (account_name,))
        cash_balance = cursor.fetchone()['cash_balance']
        new_balance = cash_balance + net_revenue
        
        cursor.execute("""
            UPDATE paper_account SET cash_balance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE account_name = ?
        """, (new_balance, account_name))
        
        # 记录交易
        cursor.execute("""
            INSERT INTO paper_trades (account_name, symbol, market, trade_type, price, shares, commission, trade_date, notes)
            VALUES (?, ?, ?, 'SELL', ?, ?, ?, ?, ?)
        """, (account_name, symbol, market, price, shares, commission, 
              datetime.now().strftime('%Y-%m-%d'), f'P&L: ${realized_pnl:.2f}'))
        
        conn.commit()
        
        return {
            'success': True,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'revenue': revenue,
            'commission': commission,
            'realized_pnl': realized_pnl,
            'new_balance': new_balance
        }
        
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def get_paper_trades(account_name: str = 'default', limit: int = 50) -> List[Dict]:
    """获取模拟交易记录"""
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM paper_trades 
        WHERE account_name = ?
        ORDER BY trade_date DESC, created_at DESC
        LIMIT ?
    """, (account_name, limit))
    
    trades = [dict(r) for r in cursor.fetchall()]
    conn.close()
    
    return trades


def get_paper_equity_curve(account_name: str = 'default') -> pd.DataFrame:
    """
    计算模拟账户的权益曲线 (按交易日)
    
    Returns:
        DataFrame with columns: date, cash, position_value, total_equity, return_pct
    """
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 获取初始资金
    cursor.execute("SELECT initial_capital, created_at FROM paper_account WHERE account_name = ?", 
                  (account_name,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return pd.DataFrame()
    
    initial = row['initial_capital']
    
    # 获取所有交易
    cursor.execute("""
        SELECT trade_date, trade_type, symbol, market, price, shares, commission
        FROM paper_trades 
        WHERE account_name = ?
        ORDER BY trade_date, created_at
    """, (account_name,))
    
    trades = cursor.fetchall()
    conn.close()
    
    if not trades:
        # 返回初始状态
        return pd.DataFrame([{
            'date': datetime.now().strftime('%Y-%m-%d'),
            'cash': initial,
            'position_value': 0,
            'total_equity': initial,
            'return_pct': 0
        }])
    
    # 计算每日权益
    equity_data = []
    cash = initial
    positions = {}  # {symbol: {'shares': x, 'avg_cost': y, 'market': m}}
    
    for trade in trades:
        date = trade['trade_date']
        symbol = trade['symbol']
        
        if trade['trade_type'] == 'BUY':
            cost = trade['price'] * trade['shares'] + trade['commission']
            cash -= cost
            
            # 更新持仓
            if symbol in positions:
                old = positions[symbol]
                new_shares = old['shares'] + trade['shares']
                new_avg = (old['shares'] * old['avg_cost'] + trade['shares'] * trade['price']) / new_shares
                positions[symbol] = {'shares': new_shares, 'avg_cost': new_avg, 'market': trade['market']}
            else:
                positions[symbol] = {'shares': trade['shares'], 'avg_cost': trade['price'], 'market': trade['market']}
                
        else:  # SELL
            revenue = trade['price'] * trade['shares'] - trade['commission']
            cash += revenue
            
            # 减少持仓
            if symbol in positions:
                positions[symbol]['shares'] -= trade['shares']
                if positions[symbol]['shares'] <= 0:
                    del positions[symbol]
        
        # 计算持仓市值 (使用交易价格作为估值)
        position_value = sum(p['shares'] * p['avg_cost'] for p in positions.values())
        total_equity = cash + position_value
        return_pct = (total_equity - initial) / initial * 100
        
        equity_data.append({
            'date': date,
            'cash': cash,
            'position_value': position_value,
            'total_equity': total_equity,
            'return_pct': return_pct
        })
    
    df = pd.DataFrame(equity_data)
    
    # 按日期去重，保留最后一条
    if not df.empty:
        df = df.groupby('date').last().reset_index()
    
    return df


def get_paper_monthly_returns(account_name: str = 'default') -> pd.DataFrame:
    """
    计算模拟账户的月度收益 (用于热力图)
    
    Returns:
        DataFrame with columns: year, month, return_pct
    """
    equity_curve = get_paper_equity_curve(account_name)
    
    if equity_curve.empty or len(equity_curve) < 2:
        return pd.DataFrame()
    
    # 转换日期
    equity_curve['date'] = pd.to_datetime(equity_curve['date'])
    equity_curve['year'] = equity_curve['date'].dt.year
    equity_curve['month'] = equity_curve['date'].dt.month
    
    # 计算每月的首末权益
    monthly = equity_curve.groupby(['year', 'month']).agg({
        'total_equity': ['first', 'last']
    }).reset_index()
    monthly.columns = ['year', 'month', 'start_equity', 'end_equity']
    
    # 计算月度收益率
    monthly['return_pct'] = (monthly['end_equity'] - monthly['start_equity']) / monthly['start_equity'] * 100
    
    return monthly[['year', 'month', 'return_pct']]


def get_multi_account_equity_curves(account_names: List[str] = None, normalize: bool = True) -> pd.DataFrame:
    """
    获取多个子账户的权益曲线（用于对比图表）
    
    Args:
        account_names: 账户名列表，None 表示所有账户
        normalize: 是否归一化到 100 起点（便于对比）
    
    Returns:
        DataFrame with columns: date, account_1, account_2, ...
        每列是该账户的权益值（或归一化后的值）
    """
    if account_names is None:
        accounts = list_paper_accounts()
        account_names = [a['account_name'] for a in accounts]
    
    if not account_names:
        return pd.DataFrame()
    
    # 收集所有账户的权益曲线
    all_curves = {}
    all_dates = set()
    
    for name in account_names:
        curve = get_paper_equity_curve(name)
        if not curve.empty:
            curve['date'] = pd.to_datetime(curve['date'])
            curve = curve.set_index('date')
            
            if normalize:
                # 归一化：初始值 = 100
                initial_equity = curve['total_equity'].iloc[0]
                if initial_equity > 0:
                    curve['normalized'] = curve['total_equity'] / initial_equity * 100
                else:
                    curve['normalized'] = 100
                all_curves[name] = curve['normalized']
            else:
                all_curves[name] = curve['total_equity']
            
            all_dates.update(curve.index.tolist())
    
    if not all_curves:
        return pd.DataFrame()
    
    # 创建日期索引
    date_index = pd.DatetimeIndex(sorted(all_dates))
    
    # 合并所有曲线
    result = pd.DataFrame(index=date_index)
    
    for name, series in all_curves.items():
        result[name] = series
    
    # 前向填充 + 后向填充缺失值
    result = result.ffill().bfill()
    
    # 重置索引
    result = result.reset_index()
    result = result.rename(columns={'index': 'date'})
    
    return result


def get_multi_account_performance_summary(account_names: List[str] = None) -> List[Dict]:
    """
    获取多个子账户的绩效摘要（用于对比表格）
    
    Returns:
        List of dicts with: account_name, total_return, win_rate, max_drawdown, trade_count
    """
    if account_names is None:
        accounts = list_paper_accounts()
        account_names = [a['account_name'] for a in accounts]
    
    results = []
    
    for name in account_names:
        perf = get_paper_account_performance(name)
        
        # 计算最大回撤
        curve = get_paper_equity_curve(name)
        max_dd = 0
        if not curve.empty and len(curve) > 1:
            equity = curve['total_equity'].values
            peak = equity[0]
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        
        results.append({
            'account_name': name,
            'total_return': perf.get('total_return_pct', 0),
            'win_rate': perf.get('win_rate', 0),
            'max_drawdown': max_dd,
            'trade_count': perf.get('total_trades', 0),
            'position_count': perf.get('total_positions', 0),
            'avg_position': perf.get('avg_position_size', 0)
        })
    
    # 按收益排序
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    return results


def get_realized_pnl_history(account_name: str = 'default') -> List[Dict]:

    """
    获取已实现盈亏历史 (每笔卖出的盈亏)
    """
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT trade_date, symbol, price, shares, commission, notes
        FROM paper_trades 
        WHERE account_name = ? AND trade_type = 'SELL'
        ORDER BY trade_date DESC
    """, (account_name,))
    
    sells = cursor.fetchall()
    conn.close()
    
    results = []
    for s in sells:
        # 从 notes 中解析 P&L
        notes = s['notes'] or ''
        pnl = 0
        if 'P&L:' in notes:
            try:
                pnl_str = notes.split('P&L:')[1].strip().replace('$', '').replace(',', '')
                pnl = float(pnl_str)
            except:
                pass
        
        results.append({
            'date': s['trade_date'],
            'symbol': s['symbol'],
            'price': s['price'],
            'shares': s['shares'],
            'realized_pnl': pnl
        })
    
    return results


def get_paper_account_performance(account_name: str = 'default') -> Dict:
    """获取单个子账户绩效指标"""
    account = get_paper_account(account_name)
    if not account:
        return {
            'account_name': account_name,
            'total_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate_pct': 0.0,
            'total_trades': 0,
            'closed_trades': 0,
            'profit_factor': 0.0,
            'total_pnl': 0.0
        }

    trades = get_paper_trades(account_name, limit=5000)
    total_trades = len(trades)
    closed_trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        if (t.get('trade_type') or '').upper() != 'SELL':
            continue
        closed_trades += 1
        notes = t.get('notes') or ''
        pnl = 0.0
        if 'P&L:' in notes:
            try:
                pnl_str = notes.split('P&L:')[1].strip().replace('$', '').replace(',', '')
                pnl = float(pnl_str)
            except Exception:
                pnl = 0.0
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)

    win_rate_pct = (wins / closed_trades * 100) if closed_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    equity_curve = get_paper_equity_curve(account_name)
    max_drawdown_pct = 0.0
    if not equity_curve.empty and 'total_equity' in equity_curve.columns:
        series = equity_curve['total_equity'].astype(float)
        peak = series.cummax()
        dd = (series - peak) / peak
        if len(dd) > 0:
            max_drawdown_pct = abs(float(dd.min())) * 100

    return {
        'account_name': account_name,
        'total_return_pct': float(account.get('total_pnl_pct', 0.0)),
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate_pct': win_rate_pct,
        'total_trades': total_trades,
        'closed_trades': closed_trades,
        'profit_factor': float(profit_factor),
        'total_pnl': float(account.get('total_pnl', 0.0)),
        'total_equity': float(account.get('total_equity', 0.0)),
        'initial_capital': float(account.get('initial_capital', 0.0)),
    }


def get_all_paper_accounts_performance() -> List[Dict]:
    """获取全部子账户绩效指标"""
    accounts = list_paper_accounts()
    results = []
    for a in accounts:
        name = a.get('account_name')
        if not name:
            continue
        perf = get_paper_account_performance(name)
        results.append(perf)
    return results


def reset_paper_account(account_name: str = 'default'):
    """重置模拟账户"""
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 删除持仓
    cursor.execute("DELETE FROM paper_positions WHERE account_name = ?", (account_name,))
    
    # 删除交易记录
    cursor.execute("DELETE FROM paper_trades WHERE account_name = ?", (account_name,))
    
    # 重置现金
    cursor.execute("""
        UPDATE paper_account SET cash_balance = initial_capital, updated_at = CURRENT_TIMESTAMP
        WHERE account_name = ?
    """, (account_name,))
    
    conn.commit()
    conn.close()


def delete_paper_account(account_name: str) -> Dict:
    """
    删除子账户及其所有数据
    
    Args:
        account_name: 账户名 (不能删除 'default')
    
    Returns:
        {'success': bool, 'error': str?}
    """
    if account_name == 'default':
        return {'success': False, 'error': '默认账户不能删除'}
    
    if not account_name:
        return {'success': False, 'error': '账户名为空'}
    
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 检查账户是否存在
        cursor.execute("SELECT 1 FROM paper_account WHERE account_name = ?", (account_name,))
        if not cursor.fetchone():
            return {'success': False, 'error': f'账户 {account_name} 不存在'}
        
        # 删除所有相关数据
        cursor.execute("DELETE FROM paper_positions WHERE account_name = ?", (account_name,))
        cursor.execute("DELETE FROM paper_trades WHERE account_name = ?", (account_name,))
        cursor.execute("DELETE FROM paper_account_config WHERE account_name = ?", (account_name,))
        cursor.execute("DELETE FROM paper_account WHERE account_name = ?", (account_name,))
        
        conn.commit()
        return {'success': True, 'deleted': account_name}
        
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def export_paper_account(account_name: str) -> Dict:
    """
    导出子账户数据为 JSON 格式
    
    Args:
        account_name: 账户名
    
    Returns:
        {
            'success': bool,
            'data': {账户数据},
            'error': str?
        }
    """
    init_paper_account()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 获取账户基本信息
        cursor.execute("""
            SELECT account_name, initial_capital, cash_balance, created_at, updated_at
            FROM paper_account WHERE account_name = ?
        """, (account_name,))
        account_row = cursor.fetchone()
        
        if not account_row:
            return {'success': False, 'error': f'账户 {account_name} 不存在'}
        
        account_data = dict(account_row)
        
        # 获取配置
        cursor.execute("""
            SELECT strategy_note, max_single_position_pct, max_drawdown_pct
            FROM paper_account_config WHERE account_name = ?
        """, (account_name,))
        config_row = cursor.fetchone()
        config_data = dict(config_row) if config_row else {}
        
        # 获取持仓
        cursor.execute("""
            SELECT symbol, market, shares, avg_cost, created_at, updated_at
            FROM paper_positions WHERE account_name = ?
        """, (account_name,))
        positions = [dict(r) for r in cursor.fetchall()]
        
        # 获取交易记录
        cursor.execute("""
            SELECT symbol, trade_type, price, shares, commission, market, notes, trade_date
            FROM paper_trades WHERE account_name = ?
            ORDER BY trade_date DESC LIMIT 500
        """, (account_name,))
        trades = [dict(r) for r in cursor.fetchall()]
        
        conn.close()
        
        export_data = {
            'version': '1.0',
            'export_time': datetime.now().isoformat(),
            'account': account_data,
            'config': config_data,
            'positions': positions,
            'trades': trades
        }
        
        return {'success': True, 'data': export_data}
        
    except Exception as e:
        conn.close()
        return {'success': False, 'error': str(e)}


def import_paper_account(data: Dict, new_account_name: str = None) -> Dict:
    """
    从 JSON 数据导入子账户
    
    Args:
        data: 导出的账户数据
        new_account_name: 新账户名 (可选，默认使用原名+_imported)
    
    Returns:
        {'success': bool, 'account_name': str?, 'error': str?}
    """
    if not data or 'account' not in data:
        return {'success': False, 'error': '无效的导入数据'}
    
    init_paper_account()
    
    original_name = data['account'].get('account_name', 'imported')
    account_name = new_account_name or f"{original_name}_imported"
    
    # 确保账户名唯一
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM paper_account WHERE account_name = ?", (account_name,))
    if cursor.fetchone():
        # 自动生成新名字
        import time
        account_name = f"{original_name}_{int(time.time()) % 10000}"
    
    try:
        # 创建账户
        initial_capital = float(data['account'].get('initial_capital', PAPER_SUBACCOUNT_INITIAL))
        cash_balance = float(data['account'].get('cash_balance', initial_capital))
        
        cursor.execute("""
            INSERT INTO paper_account (account_name, initial_capital, cash_balance)
            VALUES (?, ?, ?)
        """, (account_name, initial_capital, cash_balance))
        
        # 导入配置
        config = data.get('config', {})
        cursor.execute("""
            INSERT INTO paper_account_config (account_name, strategy_note, max_single_position_pct, max_drawdown_pct)
            VALUES (?, ?, ?, ?)
        """, (
            account_name,
            config.get('strategy_note', ''),
            float(config.get('max_single_position_pct', DEFAULT_MAX_SINGLE_POSITION_PCT)),
            float(config.get('max_drawdown_pct', DEFAULT_MAX_DRAWDOWN_PCT))
        ))
        
        # 导入持仓
        for pos in data.get('positions', []):
            cursor.execute("""
                INSERT INTO paper_positions (account_name, symbol, market, shares, avg_cost)
                VALUES (?, ?, ?, ?, ?)
            """, (
                account_name,
                pos.get('symbol'),
                pos.get('market', 'US'),
                int(pos.get('shares', 0)),
                float(pos.get('avg_cost', 0))
            ))
        
        # 导入交易记录 (最近100条)
        for trade in data.get('trades', [])[:100]:
            cursor.execute("""
                INSERT INTO paper_trades (account_name, symbol, trade_type, price, shares, commission, market, notes, trade_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_name,
                trade.get('symbol'),
                trade.get('trade_type'),
                float(trade.get('price', 0)),
                int(trade.get('shares', 0)),
                float(trade.get('commission', 0)),
                trade.get('market', 'US'),
                trade.get('notes', ''),
                trade.get('trade_date')
            ))
        
        conn.commit()
        
        positions_count = len(data.get('positions', []))
        trades_count = min(len(data.get('trades', [])), 100)
        
        return {
            'success': True,
            'account_name': account_name,
            'imported_positions': positions_count,
            'imported_trades': trades_count
        }
        
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


if __name__ == "__main__":
    # 测试
    print("测试模拟交易...")
    
    # 初始化
    init_paper_account()
    
    # 查看账户
    account = get_paper_account()
    print(f"账户: {account}")
    
    # 测试买入
    result = paper_buy('AAPL', 10, 150.0)
    print(f"买入结果: {result}")
    
    # 查看账户
    account = get_paper_account()
    print(f"买入后账户: {account}")

