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
    
    # 检查是否已有默认账户
    cursor.execute("SELECT COUNT(*) FROM paper_account WHERE account_name = 'default'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO paper_account (account_name, initial_capital, cash_balance)
            VALUES ('default', ?, ?)
        """, (PAPER_ACCOUNT_INITIAL, PAPER_ACCOUNT_INITIAL))
    
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
        'positions': enriched_positions
    }


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
        # 检查现金余额
        cursor.execute("SELECT cash_balance FROM paper_account WHERE account_name = ?", 
                      (account_name,))
        row = cursor.fetchone()
        if not row:
            return {'success': False, 'error': '账户不存在'}
        
        cash_balance = row['cash_balance']
        
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
