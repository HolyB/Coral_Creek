#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
买卖点信号系统 (Signal System)
每日生成买入/卖出信号，追踪持仓状态
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class SignalType(Enum):
    """信号类型"""
    BUY = "买入"
    SELL = "卖出"
    HOLD = "持有"
    STOP_LOSS = "止损"
    TAKE_PROFIT = "止盈"
    WATCH = "观察"


class SignalStrength(Enum):
    """信号强度"""
    STRONG = "强烈"
    MEDIUM = "中等"
    WEAK = "弱"


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    price: float
    target_price: float = 0.0
    stop_loss: float = 0.0
    reason: str = ""
    strategy: str = ""
    confidence: float = 0.0  # 0-100
    generated_at: str = ""
    expires_at: str = ""
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'price': self.price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'reason': self.reason,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'generated_at': self.generated_at,
            'expires_at': self.expires_at
        }


@dataclass
class Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    strategy: str
    status: str = "open"  # open, closed, stopped
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


class SignalGenerator:
    """信号生成器"""
    
    def __init__(self):
        self.buy_conditions = []
        self.sell_conditions = []
        
    def generate_buy_signals(self, df: pd.DataFrame, market: str = 'US') -> List[TradingSignal]:
        """生成买入信号"""
        signals = []
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 确定列名
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Ticker'
        price_col = 'price' if 'price' in df.columns else 'Price'
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Day BLUE'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover'
        
        for _, row in df.iterrows():
            symbol = row.get(symbol_col)
            price = row.get(price_col, 0) or 0
            blue = row.get(blue_col, 0) or 0
            adx = row.get(adx_col, 0) or 0
            turnover = row.get(turnover_col, 0) or 0
            
            if not symbol or price <= 0:
                continue
            
            # === 买入信号条件 ===
            
            # 1. 强势BLUE突破信号
            if blue > 180:
                strength = SignalStrength.STRONG
                confidence = min(95, 60 + (blue - 180) / 2)
                stop_loss = price * 0.95
                target = price * 1.15
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    reason=f"BLUE={blue:.0f}强势突破",
                    strategy="BLUE突破",
                    confidence=confidence,
                    generated_at=today,
                    expires_at=tomorrow
                ))
            
            # 2. 趋势确认信号 (BLUE 150-180 + ADX > 25)
            elif 150 <= blue <= 180 and adx > 25:
                strength = SignalStrength.MEDIUM
                confidence = 50 + (blue - 150) / 2 + adx / 5
                stop_loss = price * 0.93
                target = price * 1.12
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    reason=f"BLUE={blue:.0f}趋势+ADX={adx:.0f}确认",
                    strategy="趋势确认",
                    confidence=confidence,
                    generated_at=today,
                    expires_at=tomorrow
                ))
            
            # 3. 黑马形态信号 (is_heima)
            is_heima = row.get('is_heima') or row.get('Is_Heima', False)
            if is_heima:
                strength = SignalStrength.MEDIUM
                confidence = 55
                stop_loss = price * 0.90
                target = price * 1.25
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    reason="黑马形态突破",
                    strategy="黑马突破",
                    confidence=confidence,
                    generated_at=today,
                    expires_at=tomorrow
                ))
            
            # 4. 绝地反击信号 (is_juedi)
            is_juedi = row.get('is_juedi') or row.get('Is_Juedi', False)
            if is_juedi:
                strength = SignalStrength.WEAK
                confidence = 45
                stop_loss = price * 0.88
                target = price * 1.20
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    reason="绝地反击信号",
                    strategy="绝地反击",
                    confidence=confidence,
                    generated_at=today,
                    expires_at=tomorrow
                ))
            
            # 5. 量价齐升信号 (高成交 + BLUE上升)
            if turnover > 50 and 120 <= blue <= 160:
                strength = SignalStrength.MEDIUM
                confidence = 55 + min(20, turnover / 10)
                stop_loss = price * 0.94
                target = price * 1.10
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    reason=f"量价齐升 成交={turnover:.0f}M",
                    strategy="量价突破",
                    confidence=confidence,
                    generated_at=today,
                    expires_at=tomorrow
                ))
        
        # 按信心度排序
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals
    
    def generate_sell_signals(self, positions: List[Position], current_prices: Dict[str, float]) -> List[TradingSignal]:
        """生成卖出信号 (基于持仓)"""
        signals = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for pos in positions:
            if pos.status != 'open':
                continue
            
            current_price = current_prices.get(pos.symbol, 0)
            if current_price <= 0:
                continue
            
            pnl_pct = (current_price / pos.entry_price - 1) * 100
            
            # 止损信号
            if current_price <= pos.stop_loss:
                signals.append(TradingSignal(
                    symbol=pos.symbol,
                    signal_type=SignalType.STOP_LOSS,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    target_price=0,
                    stop_loss=pos.stop_loss,
                    reason=f"触发止损 亏损{pnl_pct:.1f}%",
                    strategy=pos.strategy,
                    confidence=100,
                    generated_at=today,
                    expires_at=today
                ))
            
            # 止盈信号
            elif current_price >= pos.take_profit:
                signals.append(TradingSignal(
                    symbol=pos.symbol,
                    signal_type=SignalType.TAKE_PROFIT,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    target_price=pos.take_profit,
                    stop_loss=0,
                    reason=f"达到止盈 盈利{pnl_pct:.1f}%",
                    strategy=pos.strategy,
                    confidence=100,
                    generated_at=today,
                    expires_at=today
                ))
            
            # 持有超过10天考虑卖出
            elif pos.entry_date:
                try:
                    entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
                    days_held = (datetime.now() - entry_dt).days
                    if days_held > 10 and pnl_pct < 5:
                        signals.append(TradingSignal(
                            symbol=pos.symbol,
                            signal_type=SignalType.SELL,
                            strength=SignalStrength.WEAK,
                            price=current_price,
                            target_price=0,
                            stop_loss=0,
                            reason=f"持有{days_held}天 盈利不足{pnl_pct:.1f}%",
                            strategy=pos.strategy,
                            confidence=40,
                            generated_at=today,
                            expires_at=today
                        ))
                except:
                    pass
        
        return signals


class SignalManager:
    """信号管理器 - 存储和追踪信号"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(parent_dir, 'db', 'signals.db')
        self.generator = SignalGenerator()
        self._init_db()
    
    def _init_db(self):
        """初始化信号数据库"""
        import sqlite3
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength TEXT,
                    price REAL,
                    target_price REAL,
                    stop_loss REAL,
                    reason TEXT,
                    strategy TEXT,
                    confidence REAL,
                    generated_at TEXT,
                    expires_at TEXT,
                    status TEXT DEFAULT 'active',
                    market TEXT DEFAULT 'US',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_date TEXT,
                    entry_price REAL,
                    shares INTEGER,
                    stop_loss REAL,
                    take_profit REAL,
                    strategy TEXT,
                    status TEXT DEFAULT 'open',
                    exit_date TEXT,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    market TEXT DEFAULT 'US',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(generated_at)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)
            ''')
    
    def save_signals(self, signals: List[TradingSignal], market: str = 'US'):
        """保存信号到数据库"""
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            for sig in signals:
                conn.execute('''
                    INSERT INTO signals (symbol, signal_type, strength, price, target_price,
                                        stop_loss, reason, strategy, confidence, generated_at,
                                        expires_at, market)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sig.symbol, sig.signal_type.value, sig.strength.value,
                    sig.price, sig.target_price, sig.stop_loss, sig.reason,
                    sig.strategy, sig.confidence, sig.generated_at,
                    sig.expires_at, market
                ))
    
    def get_todays_signals(self, market: str = 'US', signal_type: str = None) -> List[dict]:
        """获取今日信号"""
        import sqlite3
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM signals WHERE generated_at = ? AND market = ?"
            params = [today, market]
            
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)
            
            query += " ORDER BY confidence DESC"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_historical_signals(self, days: int = 7, market: str = 'US') -> List[dict]:
        """获取历史信号"""
        import sqlite3
        
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM signals 
                WHERE generated_at >= ? AND market = ?
                ORDER BY generated_at DESC, confidence DESC
            ''', (cutoff, market))
            return [dict(row) for row in cursor.fetchall()]
    
    def add_position(self, symbol: str, entry_price: float, shares: int,
                    stop_loss: float, take_profit: float, strategy: str,
                    market: str = 'US') -> int:
        """添加持仓"""
        import sqlite3
        
        entry_date = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO positions (symbol, entry_date, entry_price, shares,
                                       stop_loss, take_profit, strategy, market)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, entry_date, entry_price, shares, stop_loss, take_profit, strategy, market))
            return cursor.lastrowid
    
    def close_position(self, position_id: int, exit_price: float, status: str = 'closed'):
        """关闭持仓"""
        import sqlite3
        
        exit_date = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # 获取持仓信息计算盈亏
            cursor = conn.execute('SELECT * FROM positions WHERE id = ?', (position_id,))
            row = cursor.fetchone()
            if row:
                entry_price = row[3]  # entry_price
                shares = row[4]  # shares
                pnl = (exit_price - entry_price) * shares
                pnl_pct = (exit_price / entry_price - 1) * 100
                
                conn.execute('''
                    UPDATE positions 
                    SET status = ?, exit_date = ?, exit_price = ?, pnl = ?, pnl_pct = ?
                    WHERE id = ?
                ''', (status, exit_date, exit_price, pnl, pnl_pct, position_id))
    
    def get_open_positions(self, market: str = 'US') -> List[dict]:
        """获取未平仓持仓"""
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM positions 
                WHERE status = 'open' AND market = ?
                ORDER BY entry_date DESC
            ''', (market,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_closed_positions(self, days: int = 30, market: str = 'US') -> List[dict]:
        """获取已平仓记录"""
        import sqlite3
        
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM positions 
                WHERE status != 'open' AND exit_date >= ? AND market = ?
                ORDER BY exit_date DESC
            ''', (cutoff, market))
            return [dict(row) for row in cursor.fetchall()]
    
    def generate_daily_signals(self, market: str = 'US') -> Dict:
        """生成每日信号 (主入口)"""
        from db.database import query_scan_results, get_scanned_dates
        
        # 获取最新扫描数据
        dates = get_scanned_dates(market=market)
        if not dates:
            return {'error': 'No scan data available', 'signals': []}
        
        latest_date = dates[0]
        results = query_scan_results(scan_date=latest_date, market=market, limit=500)
        
        if not results:
            return {'error': 'No scan results', 'signals': []}
        
        df = pd.DataFrame(results)
        
        # 生成买入信号
        buy_signals = self.generator.generate_buy_signals(df, market)
        
        # 获取持仓并生成卖出信号
        open_positions = self.get_open_positions(market)
        current_prices = {}
        for pos in open_positions:
            # 从扫描结果获取当前价格
            symbol = pos['symbol']
            matching = [r for r in results if r.get('symbol') == symbol]
            if matching:
                current_prices[symbol] = matching[0].get('price', 0)
        
        positions = [Position(**pos) for pos in open_positions if pos.get('entry_price')]
        sell_signals = self.generator.generate_sell_signals(positions, current_prices)
        
        # 保存信号
        all_signals = buy_signals + sell_signals
        self.save_signals(all_signals, market)
        
        return {
            'date': latest_date,
            'market': market,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'signals': [s.to_dict() for s in all_signals]
        }


def get_signal_manager() -> SignalManager:
    """获取信号管理器单例"""
    if not hasattr(get_signal_manager, '_instance'):
        get_signal_manager._instance = SignalManager()
    return get_signal_manager._instance


def format_signal_message(signal: dict, market: str = 'US') -> str:
    """格式化信号为消息"""
    sym = "¥" if market == 'CN' else "$"
    emoji = {
        '买入': '🟢',
        '卖出': '🔴',
        '止损': '🛑',
        '止盈': '🎯',
        '持有': '🟡',
        '观察': '👀'
    }.get(signal['signal_type'], '📊')
    
    strength_emoji = {
        '强烈': '🔥',
        '中等': '⚡',
        '弱': '💧'
    }.get(signal['strength'], '')
    
    msg = f"{emoji} **{signal['symbol']}** {signal['signal_type']} {strength_emoji}\n"
    msg += f"   价格: {sym}{signal['price']:.2f}\n"
    
    if signal.get('target_price'):
        msg += f"   目标: {sym}{signal['target_price']:.2f}\n"
    if signal.get('stop_loss'):
        msg += f"   止损: {sym}{signal['stop_loss']:.2f}\n"
    
    msg += f"   理由: {signal['reason']}\n"
    msg += f"   信心: {signal['confidence']:.0f}%\n"
    
    return msg
