"""
信号衰减监控系统
Signal Decay Monitor

功能:
- 追踪每个信号策略的历史胜率
- 计算滚动胜率 (7/30/90天)
- 检测信号衰减并告警
- 支持多种信号类型 (BLUE/黑马/共振)
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    DAILY_BLUE = "daily_blue"       # 日BLUE
    WEEKLY_BLUE = "weekly_blue"     # 周BLUE
    MONTHLY_BLUE = "monthly_blue"   # 月BLUE
    DAILY_WEEKLY = "daily_weekly"   # 日+周共振
    HEIMA = "heima"                 # 黑马
    ALL_RESONANCE = "all_resonance" # 全共振


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"         # 🟢 健康
    WARNING = "warning"         # 🟡 关注
    CRITICAL = "critical"       # 🔴 衰减
    UNKNOWN = "unknown"         # ⚪ 未知


@dataclass
class SignalHealth:
    """信号健康度报告"""
    signal_type: SignalType
    status: HealthStatus
    
    # 胜率统计
    win_rate_7d: float      # 近7天胜率
    win_rate_30d: float     # 近30天胜率
    win_rate_90d: float     # 近90天胜率
    win_rate_all: float     # 历史总胜率
    
    # 收益统计
    avg_return_7d: float
    avg_return_30d: float
    avg_return_all: float
    
    # 样本量
    sample_7d: int
    sample_30d: int
    sample_90d: int
    sample_all: int
    
    # 衰减指标
    decay_ratio: float      # 衰减比率 (近期/历史)
    trend: str              # "improving", "stable", "declining"
    
    # 建议
    recommendation: str


class SignalMonitor:
    """信号监控器"""
    
    def __init__(self, market: str = 'US', holding_days: int = 5):
        """
        Args:
            market: 市场 ('US' or 'CN')
            holding_days: 持有天数 (用于计算收益)
        """
        self.market = market
        self.holding_days = holding_days
    
    def get_signal_performance(self, 
                               signal_type: SignalType,
                               days_back: int = 90,
                               min_blue: int = 100) -> pd.DataFrame:
        """
        获取信号历史表现
        
        Returns:
            DataFrame with columns: symbol, signal_date, entry_price, exit_price, return_pct
        """
        from db.database import get_connection
        
        conn = get_connection()
        
        end_date = date.today() - timedelta(days=self.holding_days)  # 确保有足够时间计算收益
        start_date = end_date - timedelta(days=days_back)
        
        # 根据信号类型构建查询条件
        if signal_type == SignalType.DAILY_BLUE:
            condition = f"blue_daily >= {min_blue}"
        elif signal_type == SignalType.WEEKLY_BLUE:
            condition = f"blue_weekly >= {min_blue}"
        elif signal_type == SignalType.MONTHLY_BLUE:
            condition = f"blue_monthly >= {min_blue}"
        elif signal_type == SignalType.DAILY_WEEKLY:
            condition = f"blue_daily >= {min_blue} AND blue_weekly >= {min_blue}"
        elif signal_type == SignalType.HEIMA:
            condition = "is_heima = 1"
        elif signal_type == SignalType.ALL_RESONANCE:
            condition = f"blue_daily >= {min_blue} AND blue_weekly >= {min_blue} AND (blue_monthly >= {min_blue} OR is_heima = 1)"
        else:
            condition = f"blue_daily >= {min_blue}"
        
        # 查询信号
        query = f"""
            SELECT symbol, scan_date, price as entry_price
            FROM scan_results
            WHERE market = ? 
              AND scan_date >= ? 
              AND scan_date <= ?
              AND {condition}
            ORDER BY scan_date
        """
        
        df = pd.read_sql_query(query, conn, params=(self.market, 
                                                     start_date.strftime('%Y-%m-%d'),
                                                     end_date.strftime('%Y-%m-%d')))
        
        if df.empty:
            conn.close()
            return pd.DataFrame()
        
        # 获取每个信号的退出价格 (N天后)
        results = []
        for _, row in df.iterrows():
            symbol = row['symbol']
            signal_date = row['scan_date']
            entry_price = row['entry_price']
            
            if not entry_price or entry_price <= 0:
                continue
            
            # 查找 N 天后的价格
            exit_date = (pd.to_datetime(signal_date) + timedelta(days=self.holding_days)).strftime('%Y-%m-%d')
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT price FROM scan_results
                WHERE symbol = ? AND market = ? AND scan_date >= ?
                ORDER BY scan_date
                LIMIT 1
            """, (symbol, self.market, exit_date))
            
            exit_row = cursor.fetchone()
            
            if exit_row and exit_row['price']:
                exit_price = exit_row['price']
                return_pct = (exit_price - entry_price) / entry_price * 100
                
                results.append({
                    'symbol': symbol,
                    'signal_date': signal_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': return_pct,
                    'is_win': return_pct > 0
                })
        
        conn.close()
        return pd.DataFrame(results)
    
    def calculate_health(self, signal_type: SignalType, min_blue: int = 100) -> SignalHealth:
        """
        计算信号健康度
        """
        # 获取不同时间窗口的数据
        df_90d = self.get_signal_performance(signal_type, days_back=90, min_blue=min_blue)
        
        if df_90d.empty:
            return SignalHealth(
                signal_type=signal_type,
                status=HealthStatus.UNKNOWN,
                win_rate_7d=0, win_rate_30d=0, win_rate_90d=0, win_rate_all=0,
                avg_return_7d=0, avg_return_30d=0, avg_return_all=0,
                sample_7d=0, sample_30d=0, sample_90d=0, sample_all=0,
                decay_ratio=1.0,
                trend="unknown",
                recommendation="数据不足，无法评估"
            )
        
        # 按时间窗口筛选
        today = date.today()
        df_90d['signal_date'] = pd.to_datetime(df_90d['signal_date'])
        
        df_7d = df_90d[df_90d['signal_date'] >= (today - timedelta(days=7 + self.holding_days))]
        df_30d = df_90d[df_90d['signal_date'] >= (today - timedelta(days=30 + self.holding_days))]
        
        # 计算胜率
        win_rate_7d = df_7d['is_win'].mean() if len(df_7d) > 0 else 0
        win_rate_30d = df_30d['is_win'].mean() if len(df_30d) > 0 else 0
        win_rate_90d = df_90d['is_win'].mean() if len(df_90d) > 0 else 0
        
        # 计算平均收益
        avg_return_7d = df_7d['return_pct'].mean() if len(df_7d) > 0 else 0
        avg_return_30d = df_30d['return_pct'].mean() if len(df_30d) > 0 else 0
        avg_return_all = df_90d['return_pct'].mean() if len(df_90d) > 0 else 0
        
        # 计算衰减比率
        if win_rate_90d > 0:
            decay_ratio = win_rate_30d / win_rate_90d
        else:
            decay_ratio = 1.0
        
        # 判断趋势
        if win_rate_7d > win_rate_30d > win_rate_90d:
            trend = "improving"
        elif win_rate_7d < win_rate_30d < win_rate_90d:
            trend = "declining"
        else:
            trend = "stable"
        
        # 判断健康状态
        if decay_ratio >= 0.9 and win_rate_30d >= 0.5:
            status = HealthStatus.HEALTHY
            recommendation = "信号表现正常，可继续使用"
        elif decay_ratio >= 0.75 or win_rate_30d >= 0.45:
            status = HealthStatus.WARNING
            recommendation = "信号略有下降，建议减少仓位或提高筛选标准"
        else:
            status = HealthStatus.CRITICAL
            recommendation = "信号明显衰减，建议暂停使用或重新优化参数"
        
        return SignalHealth(
            signal_type=signal_type,
            status=status,
            win_rate_7d=win_rate_7d,
            win_rate_30d=win_rate_30d,
            win_rate_90d=win_rate_90d,
            win_rate_all=win_rate_90d,  # 用 90 天作为总体
            avg_return_7d=avg_return_7d,
            avg_return_30d=avg_return_30d,
            avg_return_all=avg_return_all,
            sample_7d=len(df_7d),
            sample_30d=len(df_30d),
            sample_90d=len(df_90d),
            sample_all=len(df_90d),
            decay_ratio=decay_ratio,
            trend=trend,
            recommendation=recommendation
        )
    
    def get_all_signals_health(self, min_blue: int = 100) -> Dict[SignalType, SignalHealth]:
        """获取所有信号类型的健康度"""
        results = {}
        for signal_type in SignalType:
            results[signal_type] = self.calculate_health(signal_type, min_blue)
        return results
    
    def get_decay_alerts(self, min_blue: int = 100) -> List[SignalHealth]:
        """获取需要告警的信号"""
        all_health = self.get_all_signals_health(min_blue)
        alerts = []
        
        for signal_type, health in all_health.items():
            if health.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alerts.append(health)
        
        return alerts


def check_signal_health(market: str = 'US', min_blue: int = 100) -> Dict:
    """
    快速检查信号健康度
    
    Returns:
        {
            'overall_status': 'healthy' | 'warning' | 'critical',
            'signals': {...},
            'alerts': [...]
        }
    """
    monitor = SignalMonitor(market=market)
    all_health = monitor.get_all_signals_health(min_blue)
    alerts = monitor.get_decay_alerts(min_blue)
    
    # 判断整体状态
    statuses = [h.status for h in all_health.values()]
    if HealthStatus.CRITICAL in statuses:
        overall = 'critical'
    elif HealthStatus.WARNING in statuses:
        overall = 'warning'
    else:
        overall = 'healthy'
    
    return {
        'overall_status': overall,
        'signals': {st.value: {
            'status': h.status.value,
            'win_rate_30d': h.win_rate_30d,
            'win_rate_90d': h.win_rate_90d,
            'decay_ratio': h.decay_ratio,
            'sample_30d': h.sample_30d,
            'recommendation': h.recommendation
        } for st, h in all_health.items()},
        'alerts': [{
            'signal_type': a.signal_type.value,
            'status': a.status.value,
            'message': a.recommendation
        } for a in alerts]
    }


# === 命令行测试 ===
if __name__ == "__main__":
    print("=== 信号健康度检查 ===\n")
    
    result = check_signal_health(market='US', min_blue=100)
    
    print(f"整体状态: {result['overall_status'].upper()}")
    print()
    
    print("各信号详情:")
    for sig_type, data in result['signals'].items():
        status_icon = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴', 'unknown': '⚪'}
        icon = status_icon.get(data['status'], '⚪')
        print(f"  {icon} {sig_type}: 胜率 {data['win_rate_30d']:.0%} (30天) / {data['win_rate_90d']:.0%} (90天) | 样本 {data['sample_30d']}")
    
    if result['alerts']:
        print("\n⚠️ 告警:")
        for alert in result['alerts']:
            print(f"  - {alert['signal_type']}: {alert['message']}")
