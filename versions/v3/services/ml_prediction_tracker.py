#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML 预测追踪器 — 验证闭环
========================

核心功能:
1. 记录 SmartPicker 每日推荐 (ML 预测值)
2. 与 candidate_tracking 的实际收益比较
3. 计算模型准确率、排名相关性
4. 生成预测 vs 实际的分析报告

数据库表:
- ml_predictions: 每日 ML 推荐记录
  - 对接 candidate_tracking 的 signal_date + symbol
  - 记录 ML 输出: pred_return_5d, direction_prob, rank_score, overall_score
"""
from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from db.database import get_db, init_db

logger = logging.getLogger(__name__)

_TABLE_READY = False


def _ensure_table() -> None:
    """创建 ml_predictions 表"""
    global _TABLE_READY
    if _TABLE_READY:
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'"
                )
                if cur.fetchone():
                    return
        except Exception:
            pass
        _TABLE_READY = False

    try:
        init_db()
    except Exception:
        pass

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                market VARCHAR(10) DEFAULT 'US',
                prediction_date DATE NOT NULL,
                
                -- ML 模型预测值
                pred_return_5d REAL,          -- 预测5日收益率
                pred_direction_prob REAL,     -- 预测上涨概率 (0-1)
                ml_confidence REAL,           -- ML 置信度
                rank_score_short REAL,        -- 短线排名分
                rank_score_medium REAL,       -- 中线排名分
                rank_score_long REAL,         -- 长线排名分
                overall_score REAL,           -- 综合评分 (0-100)
                star_rating INTEGER,          -- 星级 (1-5)
                is_trade_candidate BOOLEAN,   -- 是否为交易候选
                
                -- 预测时快照
                signal_price REAL,            -- 预测时价格
                blue_daily REAL,
                blue_weekly REAL,
                adx REAL,
                rsi REAL,
                volume_ratio REAL,
                signal_tags TEXT,             -- JSON: 确认的信号列表
                warning_tags TEXT,            -- JSON: 警告信号列表
                
                -- 实际结果 (后续由 refresh 填充)
                actual_return_d1 REAL,
                actual_return_d3 REAL,
                actual_return_d5 REAL,
                actual_return_d10 REAL,
                actual_return_d20 REAL,
                actual_max_up REAL,
                actual_max_down REAL,
                
                -- 预测准确度 (后续由 refresh 计算)
                direction_correct BOOLEAN,           -- 方向预测是否正确
                return_error_5d REAL,                -- 预测 vs 实际 5日收益的误差
                
                -- 元数据
                model_version VARCHAR(50),
                source VARCHAR(50) DEFAULT 'smart_picker',
                status VARCHAR(20) DEFAULT 'pending',  -- pending/validated/expired
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(symbol, market, prediction_date)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mlpred_date ON ml_predictions(prediction_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mlpred_status ON ml_predictions(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mlpred_symbol ON ml_predictions(symbol)"
        )
    _TABLE_READY = True


# =========================================================
# 写入接口
# =========================================================

def log_prediction(
    symbol: str,
    market: str,
    prediction_date: str,
    pick_dict: Dict,
    model_version: str = "v2",
    source: str = "smart_picker",
) -> bool:
    """记录单条 ML 预测
    
    Args:
        pick_dict: StockPick.to_dict() 的输出
    """
    _ensure_table()
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ml_predictions (
                    symbol, market, prediction_date,
                    pred_return_5d, pred_direction_prob, ml_confidence,
                    rank_score_short, rank_score_medium, rank_score_long,
                    overall_score, star_rating, is_trade_candidate,
                    signal_price, blue_daily, blue_weekly, adx, rsi, volume_ratio,
                    signal_tags, warning_tags,
                    model_version, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(symbol, market, prediction_date) DO UPDATE SET
                    pred_return_5d = excluded.pred_return_5d,
                    pred_direction_prob = excluded.pred_direction_prob,
                    ml_confidence = excluded.ml_confidence,
                    rank_score_short = excluded.rank_score_short,
                    rank_score_medium = excluded.rank_score_medium,
                    rank_score_long = excluded.rank_score_long,
                    overall_score = excluded.overall_score,
                    star_rating = excluded.star_rating,
                    is_trade_candidate = excluded.is_trade_candidate,
                    signal_tags = excluded.signal_tags,
                    warning_tags = excluded.warning_tags,
                    model_version = excluded.model_version,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                symbol, market, prediction_date,
                pick_dict.get('pred_return_5d', 0),
                pick_dict.get('pred_direction_prob', 0.5),
                pick_dict.get('ml_confidence', 0),
                pick_dict.get('rank_score_short', 0),
                pick_dict.get('rank_score_medium', 0),
                pick_dict.get('rank_score_long', 0),
                pick_dict.get('overall_score', 0),
                pick_dict.get('star_rating', 0),
                int(pick_dict.get('is_trade_candidate', False)),
                pick_dict.get('price', 0),
                pick_dict.get('blue_daily', 0),
                pick_dict.get('blue_weekly', 0),
                pick_dict.get('adx', 0),
                pick_dict.get('rsi', 0),
                pick_dict.get('volume_ratio', 0),
                json.dumps(pick_dict.get('signals_confirmed', []), ensure_ascii=False),
                json.dumps(pick_dict.get('signals_warning', []), ensure_ascii=False),
                model_version, source,
            ))
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"log_prediction error: {e}")
        return False


def log_predictions_batch(
    picks: List[Dict],
    market: str,
    prediction_date: str,
    model_version: str = "v2",
    source: str = "smart_picker",
) -> int:
    """批量记录 ML 预测"""
    _ensure_table()
    count = 0
    for pick in picks:
        symbol = pick.get('symbol', '')
        if not symbol:
            continue
        if log_prediction(symbol, market, prediction_date, pick, model_version, source):
            count += 1
    logger.info(f"Logged {count}/{len(picks)} ML predictions for {prediction_date}")
    return count


# =========================================================
# 刷新实际结果 (从 candidate_tracking 回填)
# =========================================================

def refresh_prediction_results(days_back: int = 30) -> int:
    """用 candidate_tracking 的实际收益回填 ml_predictions
    
    逻辑: 
    - 找到 status='pending' 且 prediction_date 在 days_back 内的记录
    - 匹配 candidate_tracking 的 (symbol, market, signal_date)
    - 填充 actual_return_d1/d3/d5/d10/d20 + 方向准确性
    """
    _ensure_table()
    
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 找到待验证的预测
        cursor.execute("""
            SELECT p.id, p.symbol, p.market, p.prediction_date,
                   p.pred_return_5d, p.pred_direction_prob, p.signal_price
            FROM ml_predictions p
            WHERE p.status = 'pending'
              AND p.prediction_date >= ?
            ORDER BY p.prediction_date DESC
        """, (cutoff,))
        pending = [dict(r) for r in cursor.fetchall()]
        
        if not pending:
            return 0
        
        updated = 0
        for pred in pending:
            # 从 candidate_tracking 获取实际结果
            cursor.execute("""
                SELECT pnl_d1, pnl_d3, pnl_d5, pnl_d10, pnl_d20,
                       max_up_pct, max_drawdown_pct, days_since_signal
                FROM candidate_tracking
                WHERE symbol = ? AND market = ? AND signal_date = ?
            """, (pred['symbol'], pred['market'], pred['prediction_date']))
            
            tracking = cursor.fetchone()
            if not tracking:
                continue
            
            tracking = dict(tracking)
            days_since = tracking.get('days_since_signal', 0) or 0
            
            # 至少要有 5 天数据才能验证
            if days_since < 5:
                continue
            
            pnl_d5 = tracking.get('pnl_d5')
            pred_return = pred.get('pred_return_5d', 0) or 0
            pred_dir_prob = pred.get('pred_direction_prob', 0.5) or 0.5
            
            # 计算方向准确性
            direction_correct = None
            return_error = None
            
            if pnl_d5 is not None:
                actual_up = pnl_d5 > 0
                predicted_up = pred_dir_prob > 0.5
                direction_correct = int(actual_up == predicted_up)
                return_error = pnl_d5 - pred_return
            
            # 确定状态
            status = 'pending'
            if days_since >= 20:
                status = 'validated'
            elif days_since >= 5 and pnl_d5 is not None:
                status = 'validated'
            
            cursor.execute("""
                UPDATE ml_predictions
                SET actual_return_d1 = ?,
                    actual_return_d3 = ?,
                    actual_return_d5 = ?,
                    actual_return_d10 = ?,
                    actual_return_d20 = ?,
                    actual_max_up = ?,
                    actual_max_down = ?,
                    direction_correct = ?,
                    return_error_5d = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                tracking.get('pnl_d1'),
                tracking.get('pnl_d3'),
                pnl_d5,
                tracking.get('pnl_d10'),
                tracking.get('pnl_d20'),
                tracking.get('max_up_pct'),
                tracking.get('max_drawdown_pct'),
                direction_correct,
                return_error,
                status,
                pred['id'],
            ))
            updated += 1
    
    logger.info(f"Refreshed {updated}/{len(pending)} prediction results")
    return updated


# =========================================================
# 分析报告
# =========================================================

def get_prediction_accuracy(
    market: Optional[str] = None,
    days_back: int = 90,
    min_star: Optional[int] = None,
) -> Dict:
    """计算 ML 预测准确率报告
    
    Returns:
        {
            'total_predictions': 总预测数,
            'validated': 已验证数,
            'direction_accuracy': 方向准确率,
            'avg_predicted_return': 平均预测收益,
            'avg_actual_return_5d': 平均实际5日收益,
            'avg_return_error': 平均误差,
            'by_star': {star: accuracy_dict},  # 按星级
            'by_date': [{date, count, accuracy}]  # 按日期
            'rank_correlation': Spearman 相关系数,
        }
    """
    _ensure_table()
    
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT *
            FROM ml_predictions
            WHERE prediction_date >= ?
              AND status = 'validated'
        """
        params = [cutoff]
        
        if market:
            query += " AND market = ?"
            params.append(market)
        if min_star:
            query += " AND star_rating >= ?"
            params.append(min_star)
        
        query += " ORDER BY prediction_date DESC"
        cursor.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
    
    if not rows:
        return {
            'total_predictions': 0,
            'validated': 0,
            'direction_accuracy': None,
            'avg_predicted_return': None,
            'avg_actual_return_5d': None,
            'avg_return_error': None,
            'by_star': {},
            'by_date': [],
            'rank_correlation': None,
        }
    
    # 基础统计
    total = len(rows)
    direction_correct = [r for r in rows if r.get('direction_correct') is not None]
    dir_acc = (
        sum(1 for r in direction_correct if r['direction_correct']) / len(direction_correct)
        if direction_correct else None
    )
    
    pred_returns = [r['pred_return_5d'] for r in rows if r.get('pred_return_5d') is not None]
    actual_returns = [r['actual_return_d5'] for r in rows if r.get('actual_return_d5') is not None]
    return_errors = [r['return_error_5d'] for r in rows if r.get('return_error_5d') is not None]
    
    avg_pred = np.mean(pred_returns) if pred_returns else None
    avg_actual = np.mean(actual_returns) if actual_returns else None
    avg_error = np.mean(return_errors) if return_errors else None
    
    # 按星级分组
    by_star = {}
    for star in range(1, 6):
        star_rows = [r for r in rows if r.get('star_rating') == star]
        if star_rows:
            star_dir = [r for r in star_rows if r.get('direction_correct') is not None]
            star_actual = [r['actual_return_d5'] for r in star_rows if r.get('actual_return_d5') is not None]
            by_star[star] = {
                'count': len(star_rows),
                'direction_accuracy': (
                    sum(1 for r in star_dir if r['direction_correct']) / len(star_dir)
                    if star_dir else None
                ),
                'avg_actual_return_5d': np.mean(star_actual) if star_actual else None,
                'avg_max_up': np.mean([
                    r['actual_max_up'] for r in star_rows
                    if r.get('actual_max_up') is not None
                ]) if any(r.get('actual_max_up') is not None for r in star_rows) else None,
            }
    
    # 按日期分组
    by_date = {}
    for r in rows:
        d = r.get('prediction_date', '')
        if d not in by_date:
            by_date[d] = {'date': d, 'count': 0, 'correct': 0, 'total_with_dir': 0,
                          'returns': []}
        by_date[d]['count'] += 1
        if r.get('direction_correct') is not None:
            by_date[d]['total_with_dir'] += 1
            if r['direction_correct']:
                by_date[d]['correct'] += 1
        if r.get('actual_return_d5') is not None:
            by_date[d]['returns'].append(r['actual_return_d5'])
    
    date_stats = []
    for d, stats in sorted(by_date.items(), reverse=True):
        date_stats.append({
            'date': d,
            'count': stats['count'],
            'accuracy': (
                stats['correct'] / stats['total_with_dir']
                if stats['total_with_dir'] > 0 else None
            ),
            'avg_return': np.mean(stats['returns']) if stats['returns'] else None,
        })
    
    # Spearman 排名相关
    rank_corr = None
    paired = [
        (r['overall_score'], r['actual_return_d5'])
        for r in rows
        if r.get('overall_score') is not None and r.get('actual_return_d5') is not None
    ]
    if len(paired) >= 10:
        try:
            from scipy import stats as sp_stats
            scores, returns = zip(*paired)
            corr, pval = sp_stats.spearmanr(scores, returns)
            rank_corr = {'correlation': round(corr, 4), 'p_value': round(pval, 4)}
        except ImportError:
            # 简易 Spearman (不依赖 scipy)
            pass
    
    return {
        'total_predictions': total,
        'validated': total,
        'direction_accuracy': round(dir_acc * 100, 1) if dir_acc is not None else None,
        'avg_predicted_return': round(avg_pred, 2) if avg_pred is not None else None,
        'avg_actual_return_5d': round(avg_actual, 2) if avg_actual is not None else None,
        'avg_return_error': round(avg_error, 2) if avg_error is not None else None,
        'by_star': by_star,
        'by_date': date_stats[:30],
        'rank_correlation': rank_corr,
    }


def get_top_predictions(
    market: Optional[str] = None,
    days_back: int = 7,
    status: str = 'pending',
) -> List[Dict]:
    """获取最近的 ML 预测记录，用于 UI 展示"""
    _ensure_table()
    
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT * FROM ml_predictions
            WHERE prediction_date >= ?
        """
        params = [cutoff]
        
        if market:
            query += " AND market = ?"
            params.append(market)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY prediction_date DESC, overall_score DESC LIMIT 100"
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def get_model_performance_summary(market: str = 'US', days_back: int = 90) -> Dict:
    """简洁的模型表现摘要，适合 sidebar 或 dashboard 展示"""
    report = get_prediction_accuracy(market=market, days_back=days_back)
    
    if not report['validated']:
        return {
            'status': 'no_data',
            'message': f'暂无 {days_back} 天内的已验证预测',
        }
    
    dir_acc = report.get('direction_accuracy')
    avg_actual = report.get('avg_actual_return_5d')
    
    # 评级
    if dir_acc is not None:
        if dir_acc >= 60:
            grade = 'A'
            emoji = '🟢'
        elif dir_acc >= 50:
            grade = 'B'
            emoji = '🟡'
        else:
            grade = 'C'
            emoji = '🔴'
    else:
        grade = '?'
        emoji = '⚪'
    
    return {
        'status': 'active',
        'emoji': emoji,
        'grade': grade,
        'validated_count': report['validated'],
        'direction_accuracy': dir_acc,
        'avg_actual_return_5d': avg_actual,
        'avg_return_error': report.get('avg_return_error'),
        'rank_correlation': report.get('rank_correlation'),
        'by_star': report.get('by_star', {}),
    }
