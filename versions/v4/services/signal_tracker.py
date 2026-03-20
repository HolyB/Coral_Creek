"""
信号追踪服务 - 持续跟踪已发现的机会股票

功能：
1. 观察列表管理 - 记录关注的股票
2. 信号变化监控 - 检测买入/卖出信号
3. 做T时机提醒 - 日内波动机会
4. 卖出点分析 - 止盈/止损/信号转弱
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import os

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'signal_tracker.db')


def init_db():
    """初始化信号追踪数据库"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 观察列表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT DEFAULT 'US',
            added_date TEXT NOT NULL,
            entry_price REAL,
            target_price REAL,
            stop_loss REAL,
            notes TEXT,
            status TEXT DEFAULT 'watching',  -- watching, bought, sold, expired
            signal_type TEXT,  -- blue_daily, consensus, heima, etc.
            signal_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, market)
        )
    ''')
    
    # 信号历史 (每日记录)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT DEFAULT 'US',
            record_date TEXT NOT NULL,
            price REAL,
            blue_daily REAL,
            blue_weekly REAL,
            blue_monthly REAL,
            heima INTEGER DEFAULT 0,
            juedi INTEGER DEFAULT 0,
            volume REAL,
            volume_ratio REAL,
            rsi REAL,
            signal_strength TEXT,  -- strong_buy, buy, hold, sell, strong_sell
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, market, record_date)
        )
    ''')
    
    # 交易机会记录
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT DEFAULT 'US',
            opportunity_date TEXT NOT NULL,
            opportunity_type TEXT,  -- t_trade, breakout, pullback, reversal
            entry_price REAL,
            target_price REAL,
            stop_loss REAL,
            risk_reward REAL,
            confidence TEXT,
            reason TEXT,
            status TEXT DEFAULT 'active',  -- active, executed, expired, missed
            result_pnl REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 提醒记录
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT DEFAULT 'US',
            alert_date TEXT NOT NULL,
            alert_type TEXT,  -- sell_signal, stop_loss, take_profit, signal_weak, t_trade
            message TEXT,
            urgency TEXT DEFAULT 'medium',  -- high, medium, low
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


# 初始化数据库
init_db()


# ============================================
# 观察列表管理
# ============================================

def add_to_watchlist(symbol: str, market: str = 'US', entry_price: float = None,
                     target_price: float = None, stop_loss: float = None,
                     signal_type: str = None, signal_score: float = None,
                     notes: str = None) -> bool:
    """添加股票到观察列表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO watchlist 
            (symbol, market, added_date, entry_price, target_price, stop_loss, 
             signal_type, signal_score, notes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'watching')
        ''', (symbol, market, datetime.now().strftime('%Y-%m-%d'),
              entry_price, target_price, stop_loss, signal_type, signal_score, notes))
        conn.commit()
        return True
    except Exception as e:
        print(f"添加观察列表失败: {e}")
        return False
    finally:
        conn.close()


def get_watchlist(market: str = None, status: str = 'watching') -> List[Dict]:
    """获取观察列表"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM watchlist WHERE status = ?"
    params = [status]
    
    if market:
        query += " AND market = ?"
        params.append(market)
    
    query += " ORDER BY added_date DESC"
    
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results


def update_watchlist_status(symbol: str, market: str, status: str) -> bool:
    """更新观察列表状态"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE watchlist SET status = ? WHERE symbol = ? AND market = ?
        ''', (status, symbol, market))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def remove_from_watchlist(symbol: str, market: str = 'US') -> bool:
    """从观察列表移除"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM watchlist WHERE symbol = ? AND market = ?', 
                       (symbol, market))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================
# 信号历史记录
# ============================================

def record_signal(symbol: str, market: str, record_date: str, 
                  price: float, blue_daily: float = None, blue_weekly: float = None,
                  blue_monthly: float = None, heima: int = 0, juedi: int = 0,
                  volume: float = None, volume_ratio: float = None, rsi: float = None) -> bool:
    """记录每日信号数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 计算信号强度
    signal_strength = calculate_signal_strength(blue_daily, blue_weekly, heima, juedi)
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO signal_history
            (symbol, market, record_date, price, blue_daily, blue_weekly, blue_monthly,
             heima, juedi, volume, volume_ratio, rsi, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, market, record_date, price, blue_daily, blue_weekly, blue_monthly,
              heima, juedi, volume, volume_ratio, rsi, signal_strength))
        conn.commit()
        return True
    except Exception as e:
        print(f"记录信号失败: {e}")
        return False
    finally:
        conn.close()


def calculate_signal_strength(blue_daily: float, blue_weekly: float, 
                              heima: int, juedi: int) -> str:
    """计算综合信号强度"""
    score = 0
    
    if blue_daily:
        if blue_daily > 150:
            score += 3
        elif blue_daily > 100:
            score += 2
        elif blue_daily > 50:
            score += 1
        elif blue_daily < 30:
            score -= 2
    
    if blue_weekly:
        if blue_weekly > 100:
            score += 2
        elif blue_weekly > 50:
            score += 1
        elif blue_weekly < 30:
            score -= 1
    
    if heima:
        score += 2
    
    if juedi:
        score += 1
    
    if score >= 5:
        return 'strong_buy'
    elif score >= 3:
        return 'buy'
    elif score >= 0:
        return 'hold'
    elif score >= -2:
        return 'sell'
    else:
        return 'strong_sell'


def get_signal_history(symbol: str, market: str = 'US', days: int = 30) -> List[Dict]:
    """获取信号历史"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM signal_history 
        WHERE symbol = ? AND market = ?
        ORDER BY record_date DESC
        LIMIT ?
    ''', (symbol, market, days))
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results


# ============================================
# 信号变化检测
# ============================================

def detect_signal_changes(symbol: str, market: str = 'US') -> List[Dict]:
    """检测信号变化，返回提醒列表"""
    alerts = []
    history = get_signal_history(symbol, market, days=5)
    
    if len(history) < 2:
        return alerts
    
    today = history[0]
    yesterday = history[1]
    
    today_date = today.get('record_date', datetime.now().strftime('%Y-%m-%d'))
    
    # 1. BLUE 信号转弱
    if today.get('blue_daily') and yesterday.get('blue_daily'):
        if yesterday['blue_daily'] > 100 and today['blue_daily'] < 50:
            alerts.append({
                'symbol': symbol,
                'market': market,
                'alert_type': 'signal_weak',
                'message': f"日BLUE大幅下降: {yesterday['blue_daily']:.0f} → {today['blue_daily']:.0f}",
                'urgency': 'high',
                'alert_date': today_date
            })
        elif yesterday['blue_daily'] > 80 and today['blue_daily'] < 60:
            alerts.append({
                'symbol': symbol,
                'market': market,
                'alert_type': 'signal_weak',
                'message': f"日BLUE转弱: {yesterday['blue_daily']:.0f} → {today['blue_daily']:.0f}",
                'urgency': 'medium',
                'alert_date': today_date
            })
    
    # 2. BLUE 信号转强 (买入机会)
    if today.get('blue_daily') and yesterday.get('blue_daily'):
        if yesterday['blue_daily'] < 50 and today['blue_daily'] > 100:
            alerts.append({
                'symbol': symbol,
                'market': market,
                'alert_type': 'buy_signal',
                'message': f"日BLUE突破: {yesterday['blue_daily']:.0f} → {today['blue_daily']:.0f}",
                'urgency': 'high',
                'alert_date': today_date
            })
    
    # 3. 新出黑马信号
    if today.get('heima') and not yesterday.get('heima'):
        alerts.append({
            'symbol': symbol,
            'market': market,
            'alert_type': 'heima_signal',
            'message': "新出黑马信号！",
            'urgency': 'high',
            'alert_date': today_date
        })
    
    # 4. 成交量异常
    if today.get('volume_ratio') and today['volume_ratio'] > 3:
        alerts.append({
            'symbol': symbol,
            'market': market,
            'alert_type': 'volume_spike',
            'message': f"成交量放大 {today['volume_ratio']:.1f}倍",
            'urgency': 'medium',
            'alert_date': today_date
        })
    
    # 5. RSI超买超卖
    if today.get('rsi'):
        if today['rsi'] > 80:
            alerts.append({
                'symbol': symbol,
                'market': market,
                'alert_type': 'overbought',
                'message': f"RSI超买 ({today['rsi']:.0f})，注意回调风险",
                'urgency': 'medium',
                'alert_date': today_date
            })
        elif today['rsi'] < 20:
            alerts.append({
                'symbol': symbol,
                'market': market,
                'alert_type': 'oversold',
                'message': f"RSI超卖 ({today['rsi']:.0f})，可能有反弹",
                'urgency': 'medium',
                'alert_date': today_date
            })
    
    return alerts


def save_alerts(alerts: List[Dict]) -> int:
    """保存提醒到数据库"""
    if not alerts:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    saved = 0
    for alert in alerts:
        try:
            cursor.execute('''
                INSERT INTO alerts (symbol, market, alert_date, alert_type, message, urgency)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (alert['symbol'], alert['market'], alert['alert_date'],
                  alert['alert_type'], alert['message'], alert['urgency']))
            saved += 1
        except:
            pass  # 忽略重复
    
    conn.commit()
    conn.close()
    return saved


def get_unread_alerts(market: str = None, limit: int = 50) -> List[Dict]:
    """获取未读提醒"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM alerts WHERE is_read = 0"
    params = []
    
    if market:
        query += " AND market = ?"
        params.append(market)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results


def mark_alert_read(alert_id: int) -> bool:
    """标记提醒已读"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('UPDATE alerts SET is_read = 1 WHERE id = ?', (alert_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================
# 做T时机分析
# ============================================

def analyze_t_trade_opportunity(symbol: str, market: str, 
                                 current_price: float, day_high: float, day_low: float,
                                 avg_cost: float = None, support: float = None, 
                                 resistance: float = None) -> Dict:
    """分析做T机会"""
    result = {
        'symbol': symbol,
        'has_opportunity': False,
        'opportunity_type': None,
        'action': None,
        'entry_price': None,
        'target_price': None,
        'stop_loss': None,
        'reason': None,
        'confidence': 'low'
    }
    
    # 计算日内波动
    day_range = (day_high - day_low) / day_low * 100 if day_low > 0 else 0
    
    # 日内波动 > 3% 才有做T价值
    if day_range < 3:
        result['reason'] = f"日内波动 {day_range:.1f}% 太小，不适合做T"
        return result
    
    result['has_opportunity'] = True
    
    # 确定当前位置
    mid_price = (day_high + day_low) / 2
    
    if current_price <= day_low * 1.01:  # 接近日内低点
        result['opportunity_type'] = 't_buy'
        result['action'] = '低吸'
        result['entry_price'] = current_price
        result['target_price'] = mid_price
        result['stop_loss'] = day_low * 0.98
        result['reason'] = f"接近日内低点 ${day_low:.2f}，可考虑低吸"
        result['confidence'] = 'medium'
        
    elif current_price >= day_high * 0.99:  # 接近日内高点
        result['opportunity_type'] = 't_sell'
        result['action'] = '高抛'
        result['entry_price'] = current_price
        result['target_price'] = mid_price
        result['reason'] = f"接近日内高点 ${day_high:.2f}，可考虑高抛"
        result['confidence'] = 'medium'
    
    # 如果有支撑/阻力位，增强判断
    if support and current_price <= support * 1.02:
        result['opportunity_type'] = 't_buy'
        result['action'] = '支撑位低吸'
        result['entry_price'] = support
        result['target_price'] = resistance if resistance else support * 1.05
        result['stop_loss'] = support * 0.97
        result['reason'] = f"接近支撑位 ${support:.2f}"
        result['confidence'] = 'high'
    
    if resistance and current_price >= resistance * 0.98:
        result['opportunity_type'] = 't_sell'
        result['action'] = '阻力位高抛'
        result['entry_price'] = resistance
        result['target_price'] = support if support else resistance * 0.95
        result['reason'] = f"接近阻力位 ${resistance:.2f}"
        result['confidence'] = 'high'
    
    return result


# ============================================
# 卖出点分析
# ============================================

def analyze_sell_signals(symbol: str, market: str, current_price: float,
                         avg_cost: float, target_price: float = None,
                         stop_loss: float = None, blue_daily: float = None,
                         blue_weekly: float = None, initial_blue_daily: float = None) -> Dict:
    """分析卖出信号"""
    result = {
        'symbol': symbol,
        'should_sell': False,
        'sell_urgency': 'none',  # none, low, medium, high, critical
        'reasons': [],
        'recommended_action': 'hold',
        'pnl_pct': 0
    }
    
    if avg_cost <= 0:
        return result
    
    # 计算盈亏
    pnl_pct = (current_price - avg_cost) / avg_cost * 100
    result['pnl_pct'] = pnl_pct
    
    # 默认止盈止损
    if not target_price:
        target_price = avg_cost * 1.15  # 15%止盈
    if not stop_loss:
        stop_loss = avg_cost * 0.92  # 8%止损
    
    # 1. 止损检测 (最高优先级)
    if current_price < stop_loss:
        result['should_sell'] = True
        result['sell_urgency'] = 'critical'
        result['reasons'].append(f"🔴 触及止损: ${current_price:.2f} < ${stop_loss:.2f}")
        result['recommended_action'] = 'sell_now'
        return result
    
    # 2. 止盈检测
    if current_price >= target_price:
        result['should_sell'] = True
        result['sell_urgency'] = 'medium'
        result['reasons'].append(f"🟢 达到止盈目标: ${current_price:.2f} >= ${target_price:.2f} (+{pnl_pct:.1f}%)")
        result['recommended_action'] = 'take_profit'
    
    # 3. BLUE信号检测
    if blue_daily is not None:
        if blue_daily < 30:
            result['should_sell'] = True
            result['sell_urgency'] = max(result['sell_urgency'], 'high') if result['sell_urgency'] != 'none' else 'high'
            result['reasons'].append(f"🔴 日BLUE严重转弱: {blue_daily:.0f}")
            result['recommended_action'] = 'sell_now' if pnl_pct > 0 else 'consider_sell'
            
        elif blue_daily < 50 and pnl_pct > 5:
            if result['sell_urgency'] == 'none':
                result['sell_urgency'] = 'medium'
            result['reasons'].append(f"🟡 日BLUE转弱: {blue_daily:.0f}，已盈利 {pnl_pct:.1f}%")
            result['recommended_action'] = 'consider_partial_sell'
        
        # 对比初始信号强度
        if initial_blue_daily and blue_daily < initial_blue_daily * 0.5:
            result['reasons'].append(f"📉 BLUE较买入时下降 {(1 - blue_daily/initial_blue_daily)*100:.0f}%")
    
    # 4. 周BLUE检测
    if blue_weekly is not None and blue_weekly < 30:
        result['sell_urgency'] = max(result['sell_urgency'], 'high') if result['sell_urgency'] != 'none' else 'high'
        result['reasons'].append(f"🔴 周BLUE转弱: {blue_weekly:.0f}")
    
    # 5. 大幅亏损警告
    if pnl_pct < -15:
        result['sell_urgency'] = 'high'
        result['reasons'].append(f"⚠️ 亏损较大: {pnl_pct:.1f}%，建议检查止损")
    
    # 判断是否应该卖出
    if result['sell_urgency'] in ['high', 'critical']:
        result['should_sell'] = True
    
    return result


# ============================================
# 综合追踪
# ============================================

def track_watchlist_signals(market: str = 'US') -> Dict:
    """追踪观察列表中所有股票的信号变化"""
    watchlist = get_watchlist(market=market, status='watching')
    
    if not watchlist:
        return {'total': 0, 'alerts': [], 'opportunities': []}
    
    all_alerts = []
    all_opportunities = []
    
    for item in watchlist:
        symbol = item['symbol']
        
        # 检测信号变化
        alerts = detect_signal_changes(symbol, market)
        all_alerts.extend(alerts)
    
    # 保存提醒
    save_alerts(all_alerts)
    
    return {
        'total': len(watchlist),
        'alerts': all_alerts,
        'opportunities': all_opportunities
    }


def get_tracking_summary(market: str = 'US') -> Dict:
    """获取追踪汇总"""
    watchlist = get_watchlist(market=market)
    unread_alerts = get_unread_alerts(market=market)
    
    # 统计各类信号
    buy_signals = [a for a in unread_alerts if a['alert_type'] in ['buy_signal', 'heima_signal']]
    sell_signals = [a for a in unread_alerts if a['alert_type'] in ['signal_weak', 'overbought']]
    
    return {
        'watchlist_count': len(watchlist),
        'unread_alerts': len(unread_alerts),
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'recent_alerts': unread_alerts[:10],
        'watchlist': watchlist
    }
