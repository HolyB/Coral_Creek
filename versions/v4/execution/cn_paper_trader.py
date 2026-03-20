#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CN Paper Trading - A股虚拟盘交易
================================
与 AlpacaTrader 接口完全一致，用 SQLite 跟踪持仓，stock_history.db 获取真实价格。

每个策略一个 account_id，互相隔离。
初始资金 ¥100,000。
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

V3 = Path(__file__).resolve().parent.parent


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str
    is_paper: bool = True


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str = 'long'


INITIAL_CASH = 100_000.0  # ¥100K per account


class CnPaperTrader:
    """
    A股虚拟盘交易客户端 — 与 AlpacaTrader 接口一致。

    每个 account_id 独立隔离（对应一个策略），数据存储在 cn_paper_trading.db。
    """

    def __init__(self, account_id: str = 'default', **kwargs):
        self.account_id = account_id
        self.db_path = V3 / 'db' / 'cn_paper_trading.db'
        self._init_db()

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                opened_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                status TEXT DEFAULT 'filled',
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pos_acct_sym 
                ON positions(account_id, symbol);
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                date TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_eq_acct_date
                ON equity_history(account_id, date);
        """)
        # Ensure account exists
        exists = conn.execute(
            "SELECT 1 FROM accounts WHERE account_id=?", (self.account_id,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO accounts (account_id, cash, created_at) VALUES (?, ?, ?)",
                (self.account_id, INITIAL_CASH, datetime.now().isoformat())
            )
        conn.commit()
        conn.close()

    def _conn(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path))

    def _get_price(self, symbol: str) -> float:
        """Get latest price from stock_history.db"""
        hdb = V3 / 'db' / 'stock_history.db'
        if not hdb.exists():
            return 0.0
        conn = sqlite3.connect(str(hdb))
        row = conn.execute(
            "SELECT close FROM stock_history WHERE symbol=? AND market='CN' ORDER BY trade_date DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0.0

    # === Account ===

    def get_account(self) -> AccountInfo:
        conn = self._conn()
        cash = conn.execute(
            "SELECT cash FROM accounts WHERE account_id=?", (self.account_id,)
        ).fetchone()[0]

        # Calculate positions value
        positions = self._get_positions_raw(conn)
        pos_value = 0.0
        for sym, qty, avg_price in positions:
            cur = self._get_price(sym)
            pos_value += qty * (cur if cur > 0 else avg_price)

        conn.close()
        equity = cash + pos_value
        return AccountInfo(
            equity=equity,
            cash=cash,
            buying_power=cash,
            portfolio_value=equity,
            status='ACTIVE',
            is_paper=True
        )

    def _get_positions_raw(self, conn):
        return conn.execute(
            "SELECT symbol, qty, avg_entry_price FROM positions WHERE account_id=? AND qty>0",
            (self.account_id,)
        ).fetchall()

    # === Positions ===

    def get_positions(self) -> List[Position]:
        conn = self._conn()
        rows = self._get_positions_raw(conn)
        conn.close()
        result = []
        for sym, qty, avg_price in rows:
            cur = self._get_price(sym)
            if cur <= 0:
                cur = avg_price
            mv = qty * cur
            pl = (cur - avg_price) * qty
            plpc = (cur / avg_price - 1) * 100 if avg_price > 0 else 0
            result.append(Position(
                symbol=sym, qty=qty, avg_entry_price=avg_price,
                current_price=cur, market_value=mv,
                unrealized_pl=pl, unrealized_plpc=plpc
            ))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        conn = self._conn()
        row = conn.execute(
            "SELECT qty, avg_entry_price FROM positions WHERE account_id=? AND symbol=? AND qty>0",
            (self.account_id, symbol)
        ).fetchone()
        conn.close()
        if not row:
            return None
        qty, avg_price = row
        cur = self._get_price(symbol)
        if cur <= 0:
            cur = avg_price
        mv = qty * cur
        pl = (cur - avg_price) * qty
        plpc = (cur / avg_price - 1) * 100 if avg_price > 0 else 0
        return Position(
            symbol=symbol, qty=qty, avg_entry_price=avg_price,
            current_price=cur, market_value=mv,
            unrealized_pl=pl, unrealized_plpc=plpc
        )

    # === Orders ===

    def buy_market(self, symbol: str, qty: float, **kwargs) -> Dict:
        price = self._get_price(symbol)
        if price <= 0:
            raise ValueError(f"Cannot get price for {symbol}")

        cost = price * qty
        conn = self._conn()
        cash = conn.execute(
            "SELECT cash FROM accounts WHERE account_id=?", (self.account_id,)
        ).fetchone()[0]

        if cost > cash:
            conn.close()
            raise ValueError(f"Insufficient cash: need ¥{cost:,.0f}, have ¥{cash:,.0f}")

        # Update cash
        conn.execute(
            "UPDATE accounts SET cash=cash-? WHERE account_id=?",
            (cost, self.account_id)
        )

        # Update or insert position
        existing = conn.execute(
            "SELECT qty, avg_entry_price FROM positions WHERE account_id=? AND symbol=?",
            (self.account_id, symbol)
        ).fetchone()

        if existing:
            old_qty, old_price = existing
            new_qty = old_qty + qty
            new_avg = (old_qty * old_price + qty * price) / new_qty
            conn.execute(
                "UPDATE positions SET qty=?, avg_entry_price=? WHERE account_id=? AND symbol=?",
                (new_qty, new_avg, self.account_id, symbol)
            )
        else:
            conn.execute(
                "INSERT INTO positions (account_id, symbol, qty, avg_entry_price, opened_at) VALUES (?, ?, ?, ?, ?)",
                (self.account_id, symbol, qty, price, datetime.now().isoformat())
            )

        # Record order
        order_id = self._record_order(conn, symbol, 'buy', qty, price)
        conn.commit()
        conn.close()
        return {'id': str(order_id), 'symbol': symbol, 'side': 'buy', 'qty': qty, 'status': 'filled'}

    def close_position(self, symbol: str) -> Dict:
        conn = self._conn()
        row = conn.execute(
            "SELECT qty, avg_entry_price FROM positions WHERE account_id=? AND symbol=? AND qty>0",
            (self.account_id, symbol)
        ).fetchone()
        if not row:
            conn.close()
            raise ValueError(f"No position for {symbol}")

        qty, avg_price = row
        price = self._get_price(symbol)
        if price <= 0:
            price = avg_price

        proceeds = price * qty
        conn.execute(
            "UPDATE accounts SET cash=cash+? WHERE account_id=?",
            (proceeds, self.account_id)
        )
        conn.execute(
            "UPDATE positions SET qty=0 WHERE account_id=? AND symbol=?",
            (self.account_id, symbol)
        )
        order_id = self._record_order(conn, symbol, 'sell', qty, price)
        conn.commit()
        conn.close()
        return {'id': str(order_id), 'symbol': symbol, 'side': 'sell', 'qty': qty, 'status': 'filled'}

    def sell_market(self, symbol: str, qty: float, **kwargs) -> Dict:
        return self.close_position(symbol)

    def get_latest_price(self, symbol: str) -> float:
        return self._get_price(symbol)

    def is_market_open(self) -> bool:
        """CN market: always return True (we trade on close prices anyway)"""
        return True

    def _record_order(self, conn, symbol, side, qty, price):
        cur = conn.execute(
            "INSERT INTO orders (account_id, symbol, side, qty, price, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (self.account_id, symbol, side, qty, price, datetime.now().isoformat())
        )
        return cur.lastrowid

    def snapshot_equity(self, date: str = None):
        """Record today's equity snapshot"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        account = self.get_account()
        pos_value = account.equity - account.cash
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO equity_history (account_id, date, equity, cash, positions_value) VALUES (?, ?, ?, ?, ?)",
            (self.account_id, date, account.equity, account.cash, pos_value)
        )
        conn.commit()
        conn.close()

    def get_equity_history(self) -> list:
        """Return [(date, equity, cash, positions_value), ...]"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT date, equity, cash, positions_value FROM equity_history WHERE account_id=? ORDER BY date",
            (self.account_id,)
        ).fetchall()
        conn.close()
        return rows
