#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日交易工作流服务
作为20年trader设计的每日工作流程

核心理念：
1. 开盘前：知道今天该关注什么、该买什么
2. 盘中：收到关键提醒，快速决策
3. 收盘后：复盘当日，准备明天
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json

# === 数据结构 ===

@dataclass
class DailyTask:
    """每日任务"""
    task_type: str  # 'buy_candidate', 'sell_alert', 'watch_update', 'review'
    priority: int  # 1=紧急, 2=重要, 3=一般
    symbol: str
    action: str  # 具体操作
    reason: str  # 原因
    price_target: float = 0
    stop_loss: float = 0
    created_at: str = ''
    status: str = 'pending'  # pending, done, skipped

@dataclass 
class SignalLifecycle:
    """信号生命周期"""
    symbol: str
    market: str
    # 发现阶段
    discovered_date: str
    discovered_price: float
    discovered_blue: float
    discovered_reason: str
    # 观察阶段
    watchlist_date: str = ''
    watchlist_entry_price: float = 0
    watchlist_target: float = 0
    watchlist_stop: float = 0
    # 买入阶段
    buy_date: str = ''
    buy_price: float = 0
    buy_shares: int = 0
    buy_reason: str = ''
    # 持有阶段
    current_price: float = 0
    current_pnl_pct: float = 0
    holding_days: int = 0
    max_gain: float = 0
    max_loss: float = 0
    # 卖出阶段
    sell_date: str = ''
    sell_price: float = 0
    sell_reason: str = ''
    final_return: float = 0
    # 状态
    stage: str = 'discovered'  # discovered, watching, holding, closed


class DailyWorkflowService:
    """每日工作流服务"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, '..', 'db', 'workflow.db')
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 每日任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                task_type TEXT,
                priority INTEGER,
                symbol TEXT,
                market TEXT DEFAULT 'US',
                action TEXT,
                reason TEXT,
                price_target REAL,
                stop_loss REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                completed_at TEXT
            )
        ''')
        
        # 信号生命周期表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                market TEXT,
                discovered_date TEXT,
                discovered_price REAL,
                discovered_blue REAL,
                discovered_reason TEXT,
                watchlist_date TEXT,
                watchlist_entry_price REAL,
                watchlist_target REAL,
                watchlist_stop REAL,
                buy_date TEXT,
                buy_price REAL,
                buy_shares INTEGER,
                buy_reason TEXT,
                current_price REAL,
                current_pnl_pct REAL,
                holding_days INTEGER,
                max_gain REAL,
                max_loss REAL,
                sell_date TEXT,
                sell_price REAL,
                sell_reason TEXT,
                final_return REAL,
                stage TEXT DEFAULT 'discovered',
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(symbol, market, discovered_date)
            )
        ''')
        
        # 每日复盘表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_review (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                market TEXT,
                new_signals INTEGER,
                buy_candidates INTEGER,
                actual_buys INTEGER,
                actual_sells INTEGER,
                total_pnl REAL,
                best_trade TEXT,
                worst_trade TEXT,
                lessons_learned TEXT,
                tomorrow_plan TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # === 每日任务管理 ===
    
    def generate_daily_tasks(self, date: str, market: str = 'US') -> List[DailyTask]:
        """
        生成每日任务清单
        
        核心逻辑：
        1. 从昨日新信号中筛选买入候选
        2. 检查观察列表是否有入场机会
        3. 检查持仓是否需要止损/止盈
        4. 检查是否有信号衰减需要关注
        """
        tasks = []
        today = date
        
        try:
            # 1. 获取昨日新发现的强信号 → 买入候选
            from db.database import query_scan_results
            
            yesterday = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            results = query_scan_results(scan_date=yesterday, market=market, limit=100)
            
            if results:
                for r in results:
                    blue_d = r.get('blue_daily', 0) or 0
                    blue_w = r.get('blue_weekly', 0) or 0
                    is_heima = r.get('is_heima', False)
                    symbol = r.get('symbol', '')
                    price = r.get('price', 0) or 0
                    
                    # 强信号：日BLUE>100 且 周BLUE>50
                    if blue_d > 100 and blue_w > 50:
                        tasks.append(DailyTask(
                            task_type='buy_candidate',
                            priority=1,
                            symbol=symbol,
                            action='考虑买入',
                            reason=f'日BLUE={blue_d:.0f} 周BLUE={blue_w:.0f} 多周期共振',
                            price_target=price * 1.15,
                            stop_loss=price * 0.92,
                            created_at=datetime.now().isoformat()
                        ))
                    # 黑马信号
                    elif is_heima and blue_d > 80:
                        tasks.append(DailyTask(
                            task_type='buy_candidate',
                            priority=2,
                            symbol=symbol,
                            action='关注黑马',
                            reason=f'🐴黑马信号 BLUE={blue_d:.0f}',
                            price_target=price * 1.20,
                            stop_loss=price * 0.90,
                            created_at=datetime.now().isoformat()
                        ))
            
            # 2. 检查观察列表
            from services.signal_tracker import get_watchlist, get_signal_history
            
            watchlist = get_watchlist(market=market)
            for item in watchlist:
                symbol = item['symbol']
                entry_price = item.get('entry_price', 0)
                target_price = item.get('target_price', 0)
                stop_loss = item.get('stop_loss', 0)
                
                # 获取最新价格
                history = get_signal_history(symbol, market, days=1)
                if history:
                    current_price = history[0].get('price', 0)
                    blue_d = history[0].get('blue_daily', 0)
                    
                    # 检查是否接近入场点
                    if entry_price > 0 and current_price > 0:
                        diff_pct = (current_price - entry_price) / entry_price * 100
                        
                        if -3 <= diff_pct <= 3:  # 在入场点±3%范围内
                            tasks.append(DailyTask(
                                task_type='watch_update',
                                priority=1,
                                symbol=symbol,
                                action='接近入场点',
                                reason=f'当前价${current_price:.2f} 入场点${entry_price:.2f} ({diff_pct:+.1f}%)',
                                price_target=target_price,
                                stop_loss=stop_loss,
                                created_at=datetime.now().isoformat()
                            ))
                        
                        # 检查信号是否衰减
                        if blue_d < 50 and entry_price < current_price:
                            tasks.append(DailyTask(
                                task_type='watch_update',
                                priority=2,
                                symbol=symbol,
                                action='信号衰减',
                                reason=f'BLUE降至{blue_d:.0f}，考虑移出观察',
                                created_at=datetime.now().isoformat()
                            ))
            
            # 3. 检查持仓
            from services.portfolio_service import get_portfolio_summary
            
            portfolio = get_portfolio_summary() or {}
            positions = portfolio.get('details', [])
            
            for pos in positions:
                symbol = pos.get('symbol', '')
                avg_cost = pos.get('avg_cost', 0)
                current_price = pos.get('current_price', 0)
                stop_loss = pos.get('stop_loss', avg_cost * 0.92)
                
                if current_price > 0 and avg_cost > 0:
                    pnl_pct = (current_price - avg_cost) / avg_cost * 100
                    
                    # 触及止损
                    if current_price < stop_loss:
                        tasks.append(DailyTask(
                            task_type='sell_alert',
                            priority=1,
                            symbol=symbol,
                            action='⚠️ 立即止损',
                            reason=f'当前${current_price:.2f} < 止损${stop_loss:.2f}',
                            created_at=datetime.now().isoformat()
                        ))
                    # 大幅盈利，考虑止盈
                    elif pnl_pct > 20:
                        tasks.append(DailyTask(
                            task_type='sell_alert',
                            priority=2,
                            symbol=symbol,
                            action='🎯 考虑止盈',
                            reason=f'盈利 {pnl_pct:.1f}%，可部分获利了结',
                            created_at=datetime.now().isoformat()
                        ))
                    # 大幅亏损警告
                    elif pnl_pct < -10:
                        tasks.append(DailyTask(
                            task_type='sell_alert',
                            priority=2,
                            symbol=symbol,
                            action='⚠️ 检查止损',
                            reason=f'亏损 {pnl_pct:.1f}%，确认止损位置',
                            created_at=datetime.now().isoformat()
                        ))
        
        except Exception as e:
            print(f"Generate tasks error: {e}")
        
        # 按优先级排序
        tasks.sort(key=lambda x: x.priority)
        
        # 保存任务
        self.save_tasks(date, market, tasks)
        
        return tasks
    
    def save_tasks(self, date: str, market: str, tasks: List[DailyTask]):
        """保存每日任务"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for task in tasks:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_tasks
                (date, task_type, priority, symbol, market, action, reason, 
                 price_target, stop_loss, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date, task.task_type, task.priority, task.symbol, market,
                task.action, task.reason, task.price_target, task.stop_loss,
                task.status, task.created_at
            ))
        
        conn.commit()
        conn.close()
    
    def get_tasks(self, date: str, market: str = 'US', status: str = None) -> List[Dict]:
        """获取每日任务"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute('''
                SELECT * FROM daily_tasks 
                WHERE date = ? AND market = ? AND status = ?
                ORDER BY priority, created_at
            ''', (date, market, status))
        else:
            cursor.execute('''
                SELECT * FROM daily_tasks 
                WHERE date = ? AND market = ?
                ORDER BY priority, created_at
            ''', (date, market))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def complete_task(self, task_id: int, status: str = 'done'):
        """完成任务"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE daily_tasks 
            SET status = ?, completed_at = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), task_id))
        
        conn.commit()
        conn.close()
    
    # === 信号生命周期管理 ===
    
    def create_signal_lifecycle(self, symbol: str, market: str, 
                                 discovered_date: str, discovered_price: float,
                                 discovered_blue: float, discovered_reason: str) -> int:
        """创建信号生命周期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO signal_lifecycle
            (symbol, market, discovered_date, discovered_price, discovered_blue,
             discovered_reason, stage, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'discovered', ?, ?)
        ''', (
            symbol, market, discovered_date, discovered_price, discovered_blue,
            discovered_reason, datetime.now().isoformat(), datetime.now().isoformat()
        ))
        
        conn.commit()
        lifecycle_id = cursor.lastrowid
        conn.close()
        return lifecycle_id
    
    def update_to_watching(self, symbol: str, market: str,
                           entry_price: float, target: float, stop: float):
        """更新为观察状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE signal_lifecycle
            SET stage = 'watching', 
                watchlist_date = ?,
                watchlist_entry_price = ?,
                watchlist_target = ?,
                watchlist_stop = ?,
                updated_at = ?
            WHERE symbol = ? AND market = ? AND stage = 'discovered'
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            entry_price, target, stop,
            datetime.now().isoformat(),
            symbol, market
        ))
        
        conn.commit()
        conn.close()
    
    def update_to_holding(self, symbol: str, market: str,
                          buy_price: float, shares: int, reason: str):
        """更新为持有状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE signal_lifecycle
            SET stage = 'holding',
                buy_date = ?,
                buy_price = ?,
                buy_shares = ?,
                buy_reason = ?,
                updated_at = ?
            WHERE symbol = ? AND market = ? AND stage IN ('discovered', 'watching')
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            buy_price, shares, reason,
            datetime.now().isoformat(),
            symbol, market
        ))
        
        conn.commit()
        conn.close()
    
    def update_to_closed(self, symbol: str, market: str,
                         sell_price: float, reason: str):
        """更新为已平仓状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取买入价计算最终收益
        cursor.execute('''
            SELECT buy_price FROM signal_lifecycle
            WHERE symbol = ? AND market = ? AND stage = 'holding'
        ''', (symbol, market))
        
        row = cursor.fetchone()
        buy_price = row[0] if row else 0
        final_return = (sell_price - buy_price) / buy_price * 100 if buy_price > 0 else 0
        
        cursor.execute('''
            UPDATE signal_lifecycle
            SET stage = 'closed',
                sell_date = ?,
                sell_price = ?,
                sell_reason = ?,
                final_return = ?,
                updated_at = ?
            WHERE symbol = ? AND market = ? AND stage = 'holding'
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            sell_price, reason, final_return,
            datetime.now().isoformat(),
            symbol, market
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_lifecycles(self, market: str = 'US') -> Dict[str, List[Dict]]:
        """获取活跃的信号生命周期"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        result = {
            'discovered': [],
            'watching': [],
            'holding': [],
            'closed': []
        }
        
        for stage in result.keys():
            if stage == 'closed':
                # 只取最近30天平仓的
                cursor.execute('''
                    SELECT * FROM signal_lifecycle
                    WHERE market = ? AND stage = ?
                    AND sell_date >= date('now', '-30 days')
                    ORDER BY sell_date DESC
                    LIMIT 20
                ''', (market, stage))
            else:
                cursor.execute('''
                    SELECT * FROM signal_lifecycle
                    WHERE market = ? AND stage = ?
                    ORDER BY updated_at DESC
                ''', (market, stage))
            
            result[stage] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return result
    
    # === 每日复盘 ===
    
    def generate_daily_summary(self, date: str, market: str = 'US') -> Dict:
        """生成每日总结"""
        summary = {
            'date': date,
            'market': market,
            'new_signals': 0,
            'buy_candidates': 0,
            'watching_count': 0,
            'holding_count': 0,
            'today_buys': 0,
            'today_sells': 0,
            'total_pnl': 0,
            'tasks_completed': 0,
            'tasks_pending': 0
        }
        
        try:
            # 获取当日新信号
            from db.database import query_scan_results
            results = query_scan_results(scan_date=date, market=market)
            summary['new_signals'] = len(results) if results else 0
            
            # 获取强信号数量
            if results:
                strong = [r for r in results 
                          if (r.get('blue_daily', 0) or 0) > 100 
                          and (r.get('blue_weekly', 0) or 0) > 50]
                summary['buy_candidates'] = len(strong)
            
            # 获取生命周期统计
            lifecycles = self.get_active_lifecycles(market)
            summary['watching_count'] = len(lifecycles['watching'])
            summary['holding_count'] = len(lifecycles['holding'])
            
            # 获取任务统计
            tasks = self.get_tasks(date, market)
            summary['tasks_completed'] = len([t for t in tasks if t['status'] == 'done'])
            summary['tasks_pending'] = len([t for t in tasks if t['status'] == 'pending'])
            
            # 获取当日持仓盈亏
            from services.portfolio_service import get_portfolio_summary
            portfolio = get_portfolio_summary() or {}
            summary['total_pnl'] = portfolio.get('total_pnl_pct', 0)
        
        except Exception as e:
            print(f"Summary error: {e}")
        
        return summary
    
    def save_daily_review(self, date: str, market: str, 
                          lessons: str = '', tomorrow_plan: str = ''):
        """保存每日复盘"""
        summary = self.generate_daily_summary(date, market)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_review
            (date, market, new_signals, buy_candidates, actual_buys, actual_sells,
             total_pnl, lessons_learned, tomorrow_plan, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date, market, summary['new_signals'], summary['buy_candidates'],
            summary.get('today_buys', 0), summary.get('today_sells', 0),
            summary['total_pnl'], lessons, tomorrow_plan,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()


# === 便捷函数 ===

_workflow_service = None

def get_workflow_service() -> DailyWorkflowService:
    global _workflow_service
    if _workflow_service is None:
        _workflow_service = DailyWorkflowService()
    return _workflow_service

def get_today_tasks(market: str = 'US') -> List[Dict]:
    """获取今日任务"""
    service = get_workflow_service()
    today = datetime.now().strftime('%Y-%m-%d')
    
    tasks = service.get_tasks(today, market)
    if not tasks:
        # 如果没有任务，生成新任务
        service.generate_daily_tasks(today, market)
        tasks = service.get_tasks(today, market)
    
    return tasks

def get_signal_pipeline(market: str = 'US') -> Dict[str, List[Dict]]:
    """获取信号流水线（各阶段的股票）"""
    service = get_workflow_service()
    return service.get_active_lifecycles(market)

def get_daily_summary(date: str = None, market: str = 'US') -> Dict:
    """获取每日总结"""
    service = get_workflow_service()
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    return service.generate_daily_summary(date, market)
