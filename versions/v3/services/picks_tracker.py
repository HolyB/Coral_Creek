#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Picks Performance Tracker
=========================

自动追踪 SmartPicker 推荐的股票表现。

功能:
1. 记录每日推荐 (symbol, date, entry_price, predicted_return, etc.)
2. N天后自动回填实际收益
3. 生成绩效报告 (胜率、平均收益、最大回撤)
4. 支持策略对比

使用:
```python
from services.picks_tracker import PicksTracker

tracker = PicksTracker(market='US')
tracker.record_pick(pick)  # 记录推荐
tracker.backfill_returns(days=5)  # 回填5天收益
report = tracker.get_performance_report()  # 获取报告
```
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass, asdict

# 添加路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))


@dataclass
class PickRecord:
    """推荐记录"""
    symbol: str
    market: str
    pick_date: str
    entry_price: float
    
    # 预测值
    pred_return_5d: float
    pred_direction_prob: float
    overall_score: float
    star_rating: int
    
    # 止损/目标
    stop_loss_price: float
    target_price: float
    
    # 实际表现 (后填)
    actual_return_1d: Optional[float] = None
    actual_return_3d: Optional[float] = None
    actual_return_5d: Optional[float] = None
    actual_return_10d: Optional[float] = None
    actual_high_5d: Optional[float] = None  # 5天内最高价
    actual_low_5d: Optional[float] = None   # 5天内最低价
    hit_target: Optional[bool] = None        # 是否触及目标
    hit_stop: Optional[bool] = None          # 是否触发止损
    
    # 元数据
    blue_daily: float = 0
    blue_weekly: float = 0
    signals_confirmed: str = ""
    backfilled: bool = False


class PicksTracker:
    """推荐绩效追踪器"""
    
    def __init__(self, market: str = 'US', db_path: Optional[str] = None):
        self.market = market
        
        # 使用数据库或 JSON 文件存储
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = current_dir / "db" / f"picks_{market.lower()}.json"
        
        self.picks: List[PickRecord] = []
        self._load()
    
    def _load(self):
        """加载历史记录"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.picks = [PickRecord(**r) for r in data]
            except Exception as e:
                print(f"加载记录失败: {e}")
                self.picks = []
    
    def _save(self):
        """保存记录"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump([asdict(p) for p in self.picks], f, indent=2, default=str)
        except Exception as e:
            print(f"保存记录失败: {e}")
    
    def record_pick(self, pick, date: Optional[str] = None) -> bool:
        """
        记录一个推荐
        
        Args:
            pick: StockPick 对象或字典
            date: 日期 (默认今天)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # 检查是否重复
        for existing in self.picks:
            if existing.symbol == pick.symbol and existing.pick_date == date:
                return False  # 已存在
        
        # 创建记录
        record = PickRecord(
            symbol=pick.symbol if hasattr(pick, 'symbol') else pick['symbol'],
            market=self.market,
            pick_date=date,
            entry_price=pick.price if hasattr(pick, 'price') else pick['price'],
            pred_return_5d=pick.pred_return_5d if hasattr(pick, 'pred_return_5d') else pick.get('pred_return_5d', 0),
            pred_direction_prob=pick.pred_direction_prob if hasattr(pick, 'pred_direction_prob') else pick.get('pred_direction_prob', 0.5),
            overall_score=pick.overall_score if hasattr(pick, 'overall_score') else pick.get('overall_score', 0),
            star_rating=pick.star_rating if hasattr(pick, 'star_rating') else pick.get('star_rating', 0),
            stop_loss_price=pick.stop_loss_price if hasattr(pick, 'stop_loss_price') else pick.get('stop_loss_price', 0),
            target_price=pick.target_price if hasattr(pick, 'target_price') else pick.get('target_price', 0),
            blue_daily=pick.blue_daily if hasattr(pick, 'blue_daily') else pick.get('blue_daily', 0),
            blue_weekly=pick.blue_weekly if hasattr(pick, 'blue_weekly') else pick.get('blue_weekly', 0),
            signals_confirmed=','.join(pick.signals_confirmed) if hasattr(pick, 'signals_confirmed') else ''
        )
        
        self.picks.append(record)
        self._save()
        return True
    
    def backfill_returns(self, days_back: int = 10) -> int:
        """
        回填实际收益
        
        Args:
            days_back: 回填多少天前的数据
            
        Returns:
            回填的记录数
        """
        try:
            from data_fetcher import get_stock_data
        except ImportError:
            print("无法导入 data_fetcher")
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        count = 0
        
        for record in self.picks:
            if record.backfilled:
                continue
            
            pick_date = datetime.strptime(record.pick_date, '%Y-%m-%d')
            
            # 只回填 N 天前的
            if pick_date > cutoff_date:
                continue
            
            # 获取价格数据
            try:
                df = get_stock_data(record.symbol, self.market, days=30)
                if df is None or df.empty:
                    continue
                
                # 找到推荐日期的索引
                df.index = df.index.date if hasattr(df.index, 'date') else df.index
                pick_dt = pick_date.date()
                
                if pick_dt not in df.index:
                    # 找最近的日期
                    dates = sorted(df.index)
                    closest = min(dates, key=lambda x: abs((x - pick_dt).days))
                    pick_idx = dates.index(closest)
                else:
                    pick_idx = list(df.index).index(pick_dt)
                
                entry = record.entry_price
                
                # 计算各周期收益
                if pick_idx + 1 < len(df):
                    record.actual_return_1d = (df['Close'].iloc[pick_idx + 1] / entry - 1) * 100
                if pick_idx + 3 < len(df):
                    record.actual_return_3d = (df['Close'].iloc[pick_idx + 3] / entry - 1) * 100
                if pick_idx + 5 < len(df):
                    record.actual_return_5d = (df['Close'].iloc[pick_idx + 5] / entry - 1) * 100
                    # 5天内高低点
                    high_5d = df['High'].iloc[pick_idx + 1:pick_idx + 6].max()
                    low_5d = df['Low'].iloc[pick_idx + 1:pick_idx + 6].min()
                    record.actual_high_5d = (high_5d / entry - 1) * 100
                    record.actual_low_5d = (low_5d / entry - 1) * 100
                    record.hit_target = high_5d >= record.target_price
                    record.hit_stop = low_5d <= record.stop_loss_price
                if pick_idx + 10 < len(df):
                    record.actual_return_10d = (df['Close'].iloc[pick_idx + 10] / entry - 1) * 100
                
                record.backfilled = True
                count += 1
                
            except Exception as e:
                print(f"回填 {record.symbol} 失败: {e}")
                continue
        
        self._save()
        return count
    
    def get_performance_report(self, 
                                min_date: Optional[str] = None,
                                max_date: Optional[str] = None) -> Dict:
        """
        获取绩效报告
        
        Returns:
            {
                'total_picks': int,
                'backfilled_picks': int,
                'win_rate_5d': float,      # 5天胜率
                'avg_return_5d': float,    # 5天平均收益
                'avg_return_10d': float,   # 10天平均收益
                'hit_target_rate': float,  # 触及目标率
                'hit_stop_rate': float,    # 触发止损率
                'avg_pred_return': float,  # 平均预测收益
                'prediction_accuracy': float,  # 预测准确度
                'best_pick': dict,
                'worst_pick': dict,
                'by_star_rating': dict,    # 按星级统计
            }
        """
        # 筛选
        picks = self.picks
        if min_date:
            picks = [p for p in picks if p.pick_date >= min_date]
        if max_date:
            picks = [p for p in picks if p.pick_date <= max_date]
        
        # 只统计已回填的
        filled = [p for p in picks if p.backfilled and p.actual_return_5d is not None]
        
        if not filled:
            return {
                'total_picks': len(picks),
                'backfilled_picks': 0,
                'win_rate_5d': 0,
                'avg_return_5d': 0,
                'message': '无已回填的记录'
            }
        
        # 基础统计
        returns_5d = [p.actual_return_5d for p in filled if p.actual_return_5d is not None]
        returns_10d = [p.actual_return_10d for p in filled if p.actual_return_10d is not None]
        
        win_rate_5d = sum(1 for r in returns_5d if r > 0) / len(returns_5d) if returns_5d else 0
        avg_return_5d = sum(returns_5d) / len(returns_5d) if returns_5d else 0
        avg_return_10d = sum(returns_10d) / len(returns_10d) if returns_10d else 0
        
        # 目标/止损触发率
        hit_targets = [p for p in filled if p.hit_target]
        hit_stops = [p for p in filled if p.hit_stop]
        
        # 预测准确度: 预测方向与实际方向一致的比例
        direction_correct = sum(
            1 for p in filled 
            if (p.pred_return_5d > 0 and p.actual_return_5d > 0) or 
               (p.pred_return_5d <= 0 and p.actual_return_5d <= 0)
        )
        prediction_accuracy = direction_correct / len(filled) if filled else 0
        
        # 找最佳/最差
        best_pick = max(filled, key=lambda p: p.actual_return_5d or 0)
        worst_pick = min(filled, key=lambda p: p.actual_return_5d or 0)
        
        # 按星级分组
        by_star = {}
        for star in range(1, 6):
            star_picks = [p for p in filled if p.star_rating == star]
            if star_picks:
                star_returns = [p.actual_return_5d for p in star_picks if p.actual_return_5d is not None]
                by_star[star] = {
                    'count': len(star_picks),
                    'avg_return': sum(star_returns) / len(star_returns) if star_returns else 0,
                    'win_rate': sum(1 for r in star_returns if r > 0) / len(star_returns) if star_returns else 0
                }
        
        return {
            'total_picks': len(picks),
            'backfilled_picks': len(filled),
            'win_rate_5d': round(win_rate_5d * 100, 1),
            'avg_return_5d': round(avg_return_5d, 2),
            'avg_return_10d': round(avg_return_10d, 2),
            'hit_target_rate': round(len(hit_targets) / len(filled) * 100, 1) if filled else 0,
            'hit_stop_rate': round(len(hit_stops) / len(filled) * 100, 1) if filled else 0,
            'avg_pred_return': round(sum(p.pred_return_5d for p in filled) / len(filled), 2) if filled else 0,
            'prediction_accuracy': round(prediction_accuracy * 100, 1),
            'best_pick': asdict(best_pick),
            'worst_pick': asdict(worst_pick),
            'by_star_rating': by_star
        }
    
    def get_recent_picks(self, days: int = 30) -> List[Dict]:
        """获取最近 N 天的推荐"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        recent = [p for p in self.picks if p.pick_date >= cutoff]
        return [asdict(p) for p in recent]
    
    def export_to_csv(self, path: str) -> bool:
        """导出到 CSV"""
        try:
            import pandas as pd
            df = pd.DataFrame([asdict(p) for p in self.picks])
            df.to_csv(path, index=False)
            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False


def daily_picks_job(market: str = 'US', max_picks: int = 5):
    """
    每日推荐任务
    
    1. 获取今日推荐
    2. 记录到 tracker
    3. 回填历史数据
    4. 生成报告
    """
    from ml.smart_picker import get_todays_picks
    
    # 获取推荐
    picks = get_todays_picks(market=market, max_picks=max_picks)
    
    # 记录
    tracker = PicksTracker(market=market)
    for pick in picks:
        tracker.record_pick(pick)
    
    # 回填 (10天前的)
    backfilled = tracker.backfill_returns(days_back=10)
    
    # 报告
    report = tracker.get_performance_report()
    
    print(f"\n=== {market} 推荐追踪报告 ===")
    print(f"今日推荐: {len(picks)} 只")
    print(f"回填: {backfilled} 条")
    print(f"总推荐: {report['total_picks']}")
    print(f"5天胜率: {report['win_rate_5d']}%")
    print(f"5天平均收益: {report['avg_return_5d']}%")
    print(f"预测准确度: {report['prediction_accuracy']}%")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Picks Tracker')
    parser.add_argument('--market', default='US', help='Market (US/CN)')
    parser.add_argument('--backfill', action='store_true', help='Only backfill')
    parser.add_argument('--report', action='store_true', help='Only show report')
    parser.add_argument('--export', type=str, help='Export to CSV')
    
    args = parser.parse_args()
    
    if args.report:
        tracker = PicksTracker(market=args.market)
        report = tracker.get_performance_report()
        print(json.dumps(report, indent=2, default=str))
    elif args.backfill:
        tracker = PicksTracker(market=args.market)
        count = tracker.backfill_returns()
        print(f"回填 {count} 条记录")
    elif args.export:
        tracker = PicksTracker(market=args.market)
        if tracker.export_to_csv(args.export):
            print(f"已导出到 {args.export}")
    else:
        daily_picks_job(market=args.market)
