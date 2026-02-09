#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多策略决策系统
每个策略有独立的选股逻辑和历史表现追踪
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class StrategyPick:
    """策略选股结果"""
    symbol: str
    score: float  # 0-100 评分
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    confidence: str  # 高/中/低
    

@dataclass
class StrategyPerformance:
    """策略历史表现"""
    total_picks: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_return_5d: float = 0.0
    avg_return_10d: float = 0.0
    max_gain: float = 0.0
    max_loss: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.total_picks == 0:
            return 0.0
        return self.win_count / self.total_picks * 100


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, description: str, icon: str = "📊"):
        self.name = name
        self.description = description
        self.icon = icon
        self.performance = StrategyPerformance()
    
    @abstractmethod
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        """从信号中选股"""
        pass
    
    def calculate_stop_loss(self, price: float, volatility: float = 0.02) -> float:
        """计算止损价"""
        # 兼容异常波动率输入，避免出现负数止损价
        vol = float(volatility) if volatility is not None else 0.02
        if vol < 0:
            vol = 0.02
        # 实战约束：止损比例在 3%~35% 区间
        stop_pct = max(0.03, min(0.35, vol * 1.5))
        stop = price * (1 - stop_pct)
        # 保底不低于 0.01，避免低价股出现负值或 0
        return round(max(0.01, stop), 2)

    def calculate_take_profit(self, price: float, risk_reward: float = 2.0, stop_loss: float = None) -> float:
        """计算止盈价 (基于风险回报比)"""
        if stop_loss:
            risk = max(price - stop_loss, price * 0.03)  # 保底至少 3% 风险距离
            return round(price + risk * risk_reward, 2)
        return round(price * 1.08, 2)  # 默认8%止盈


class MomentumStrategy(BaseStrategy):
    """策略A: 动量突破策略
    选择 BLUE 值最高 + ADX 强势的股票
    """
    
    def __init__(self):
        super().__init__(
            name="动量突破",
            description="追踪强势动量，适合趋势行情",
            icon="🚀"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        # 筛选条件: BLUE > 80, ADX > 25, 成交额 > 5M
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 80]
        if adx_col in filtered.columns:
            filtered = filtered[filtered[adx_col] >= 20]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 3]
        
        if filtered.empty:
            return picks
        
        # 计算综合评分 (BLUE 权重 60%, ADX 权重 40%)
        if blue_col in filtered.columns and adx_col in filtered.columns:
            filtered['score'] = (
                filtered[blue_col].fillna(0) / 100 * 60 +
                filtered[adx_col].fillna(0) / 50 * 40
            ).clip(0, 100)
        else:
            filtered['score'] = filtered[blue_col].fillna(0) if blue_col in filtered.columns else 50
        
        # 排序并取 top_n
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.02) if vol_col in row else 0.02
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 2.5, stop_loss)
            
            blue_val = row.get(blue_col, 0)
            adx_val = row.get(adx_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BLUE={blue_val:.0f}, ADX={adx_val:.0f}",
                confidence="高" if row['score'] > 80 else "中" if row['score'] > 60 else "低"
            ))
        
        return picks


class ValueStrategy(BaseStrategy):
    """策略B: 价值洼地策略
    选择 BLUE 突破但价格相对低位的股票
    """
    
    def __init__(self):
        super().__init__(
            name="价值洼地",
            description="寻找被低估的突破机会",
            icon="💎"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        blue_weekly_col = 'blue_weekly' if 'blue_weekly' in df.columns else 'Blue_Weekly'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # 筛选: BLUE >= 70, 但波动率较低 (潜力股)
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 70]
        if vol_col in filtered.columns:
            filtered = filtered[filtered[vol_col] <= 0.3]  # 波动率不太高
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 2]
        
        if filtered.empty:
            return picks
        
        # 评分: BLUE + 周BLUE共振加分
        if blue_col in filtered.columns:
            filtered['score'] = filtered[blue_col].fillna(0) * 0.6
            if blue_weekly_col in filtered.columns:
                # 周线也是 BLUE 的加分
                filtered['score'] += (filtered[blue_weekly_col].fillna(0) > 50).astype(int) * 20
        else:
            filtered['score'] = 50
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.02) if vol_col in row else 0.02
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 2.0, stop_loss)
            
            blue_val = row.get(blue_col, 0)
            weekly_val = row.get(blue_weekly_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"日BLUE={blue_val:.0f}, 周={weekly_val:.0f}",
                confidence="高" if weekly_val > 50 else "中"
            ))
        
        return picks


class ConservativeStrategy(BaseStrategy):
    """策略C: 稳健策略
    低波动 + 高流动性 + BLUE 信号
    """
    
    def __init__(self):
        super().__init__(
            name="稳健保守",
            description="低波动高流动性，适合风险厌恶者",
            icon="🛡️"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # 筛选: BLUE >= 60, 高流动性, 低波动
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 60]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 10]  # 高成交额
        if vol_col in filtered.columns:
            filtered = filtered[filtered[vol_col] <= 0.25]  # 低波动
        
        if filtered.empty:
            return picks
        
        # 评分: 成交额越高越好, 波动率越低越好
        if turnover_col in filtered.columns and vol_col in filtered.columns:
            # 归一化
            turnover_norm = filtered[turnover_col] / filtered[turnover_col].max() * 50
            vol_norm = (1 - filtered[vol_col] / 0.5) * 30
            blue_norm = filtered[blue_col] / 100 * 20 if blue_col in filtered.columns else 10
            filtered['score'] = (turnover_norm + vol_norm + blue_norm).clip(0, 100)
        else:
            filtered['score'] = 50
        
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.015) if vol_col in row else 0.015
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 1.5, stop_loss)  # 保守目标
            
            turnover = row.get(turnover_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"成交额={turnover:.1f}M, 波动={vol:.1%}",
                confidence="高" if vol < 0.15 else "中"
            ))
        
        return picks


class AggressiveStrategy(BaseStrategy):
    """策略D: 激进策略
    高波动 + 高BLUE，追求高回报
    """
    
    def __init__(self):
        super().__init__(
            name="激进突破",
            description="高风险高回报，适合短线交易",
            icon="⚡"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # 筛选: 超高BLUE + 高ADX
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 90]
        if adx_col in filtered.columns:
            filtered = filtered[filtered[adx_col] >= 30]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 5]
        
        if filtered.empty:
            # 放宽条件
            filtered = df.copy()
            if blue_col in filtered.columns:
                filtered = filtered[filtered[blue_col] >= 85]
        
        if filtered.empty:
            return picks
        
        # 评分: BLUE 为主
        if blue_col in filtered.columns:
            filtered['score'] = filtered[blue_col].fillna(0)
            if adx_col in filtered.columns:
                filtered['score'] = filtered['score'] * 0.7 + filtered[adx_col].fillna(0) * 0.3
        else:
            filtered['score'] = 50
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.03) if vol_col in row else 0.03
            stop_loss = self.calculate_stop_loss(price, vol * 0.8)  # 更紧的止损
            take_profit = self.calculate_take_profit(price, 3.0, stop_loss)  # 更高目标
            
            blue_val = row.get(blue_col, 0)
            adx_val = row.get(adx_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BLUE={blue_val:.0f}, ADX={adx_val:.0f}",
                confidence="中"  # 激进策略默认中等置信
            ))
        
        return picks


class MultiTimeframeStrategy(BaseStrategy):
    """策略E: 多周期共振策略
    日线+周线同时 BLUE 的股票
    """
    
    def __init__(self):
        super().__init__(
            name="多周期共振",
            description="日线周线同向，趋势更可靠",
            icon="🔄"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        blue_weekly_col = 'blue_weekly' if 'blue_weekly' in df.columns else 'Blue_Weekly'
        blue_monthly_col = 'blue_monthly' if 'blue_monthly' in df.columns else 'Blue_Monthly'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        
        # 筛选: 日线和周线同时 BLUE >= 60
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 60]
        if blue_weekly_col in filtered.columns:
            filtered = filtered[filtered[blue_weekly_col] >= 50]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 2]
        
        if filtered.empty:
            return picks
        
        # 评分: 日线 + 周线 + 月线加分
        filtered['score'] = 0
        if blue_col in filtered.columns:
            filtered['score'] += filtered[blue_col].fillna(0) * 0.4
        if blue_weekly_col in filtered.columns:
            filtered['score'] += filtered[blue_weekly_col].fillna(0) * 0.4
        if blue_monthly_col in filtered.columns:
            filtered['score'] += (filtered[blue_monthly_col].fillna(0) > 50).astype(int) * 20
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.02)
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 2.5, stop_loss)
            
            d = row.get(blue_col, 0)
            w = row.get(blue_weekly_col, 0)
            m = row.get(blue_monthly_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"日={d:.0f} 周={w:.0f} 月={m:.0f}",
                confidence="高" if w > 60 and m > 50 else "中"
            ))
        
        return picks


class ReversalStrategy(BaseStrategy):
    """策略F: 超跌反弹策略
    寻找绝地反击信号
    """
    
    def __init__(self):
        super().__init__(
            name="超跌反弹",
            description="绝地反击，抄底机会",
            icon="🔃"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        is_juedi_col = 'is_juedi' if 'is_juedi' in df.columns else 'Is_Juedi'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        
        # 筛选: 绝地反击信号
        if is_juedi_col in filtered.columns:
            filtered = filtered[filtered[is_juedi_col] == True]
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 50]  # BLUE 也要起来
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 1]
        
        if filtered.empty:
            return picks
        
        # 评分: BLUE 越高越好
        if blue_col in filtered.columns:
            filtered['score'] = filtered[blue_col].fillna(0)
        else:
            filtered['score'] = 50
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.03)
            stop_loss = self.calculate_stop_loss(price, vol * 1.2)  # 更宽的止损
            take_profit = self.calculate_take_profit(price, 3.0, stop_loss)  # 高目标
            
            blue_val = row.get(blue_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"绝地反击 BLUE={blue_val:.0f}",
                confidence="中"  # 反弹策略风险较高
            ))
        
        return picks


class VolumeBreakoutStrategy(BaseStrategy):
    """策略G: 放量突破策略
    成交额突然放大 + BLUE 信号
    """
    
    def __init__(self):
        super().__init__(
            name="放量突破",
            description="量价齐升，主力入场",
            icon="📊"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        
        # 筛选: 高成交额 + BLUE 信号
        if turnover_col in filtered.columns:
            # 成交额排名前 20%
            threshold = filtered[turnover_col].quantile(0.8)
            filtered = filtered[filtered[turnover_col] >= threshold]
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 70]
        
        if filtered.empty:
            return picks
        
        # 评分: 成交额 + BLUE
        if turnover_col in filtered.columns and blue_col in filtered.columns:
            max_turnover = filtered[turnover_col].max()
            if max_turnover > 0:
                filtered['score'] = (
                    filtered[turnover_col] / max_turnover * 50 +
                    filtered[blue_col] / 100 * 50
                ).clip(0, 100)
            else:
                filtered['score'] = filtered[blue_col].fillna(0)
        else:
            filtered['score'] = 50
        
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.025)
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 2.0, stop_loss)
            
            turnover = row.get(turnover_col, 0)
            blue_val = row.get(blue_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"成交={turnover:.1f}M BLUE={blue_val:.0f}",
                confidence="高" if turnover > 50 else "中"
            ))
        
        return picks


class HeimaPatternStrategy(BaseStrategy):
    """策略H: 黑马形态策略
    识别黑马底部形态
    """
    
    def __init__(self):
        super().__init__(
            name="黑马形态",
            description="识别潜在黑马股",
            icon="🐴"
        )
    
    def select(self, df: pd.DataFrame, top_n: int = 5) -> List[StrategyPick]:
        picks = []
        
        if df.empty:
            return picks
        
        filtered = df.copy()
        
        blue_col = 'blue_daily' if 'blue_daily' in df.columns else 'Blue_Daily'
        is_heima_col = 'is_heima' if 'is_heima' in df.columns else 'Is_Heima'
        adx_col = 'adx' if 'adx' in df.columns else 'ADX'
        price_col = 'price' if 'price' in df.columns else 'Price'
        symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
        vol_col = 'volatility' if 'volatility' in df.columns else 'Volatility'
        turnover_col = 'turnover_m' if 'turnover_m' in df.columns else 'Turnover_M'
        
        # 筛选: 黑马信号
        if is_heima_col in filtered.columns:
            filtered = filtered[filtered[is_heima_col] == True]
        if blue_col in filtered.columns:
            filtered = filtered[filtered[blue_col] >= 60]
        if turnover_col in filtered.columns:
            filtered = filtered[filtered[turnover_col] >= 1]
        
        if filtered.empty:
            return picks
        
        # 评分: BLUE + ADX
        if blue_col in filtered.columns:
            filtered['score'] = filtered[blue_col].fillna(0) * 0.7
            if adx_col in filtered.columns:
                filtered['score'] += filtered[adx_col].fillna(0) * 0.3
        else:
            filtered['score'] = 50
        
        filtered['score'] = filtered['score'].clip(0, 100)
        filtered = filtered.nlargest(top_n, 'score')
        
        for _, row in filtered.iterrows():
            price = row.get(price_col, 0)
            vol = row.get(vol_col, 0.03)
            stop_loss = self.calculate_stop_loss(price, vol)
            take_profit = self.calculate_take_profit(price, 3.0, stop_loss)  # 黑马目标高
            
            blue_val = row.get(blue_col, 0)
            adx_val = row.get(adx_col, 0)
            
            picks.append(StrategyPick(
                symbol=row[symbol_col],
                score=round(row['score'], 1),
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"黑马 BLUE={blue_val:.0f} ADX={adx_val:.0f}",
                confidence="中"
            ))
        
        return picks


class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {
            'momentum': MomentumStrategy(),
            'value': ValueStrategy(),
            'conservative': ConservativeStrategy(),
            'aggressive': AggressiveStrategy(),
            'multi_timeframe': MultiTimeframeStrategy(),
            'reversal': ReversalStrategy(),
            'volume_breakout': VolumeBreakoutStrategy(),
            'heima': HeimaPatternStrategy(),
        }
    
    def get_all_picks(self, df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[StrategyPick]]:
        """获取所有策略的选股结果"""
        results = {}
        for name, strategy in self.strategies.items():
            results[name] = strategy.select(df, top_n)
        return results
    
    def get_consensus_picks(self, df: pd.DataFrame, min_votes: int = 2) -> List[Tuple[str, int, float]]:
        """获取多策略共识股票
        返回: [(symbol, 票数, 平均分)]
        """
        all_picks = self.get_all_picks(df)
        
        # 统计每个股票被几个策略选中
        symbol_votes = {}
        symbol_scores = {}
        
        for strategy_name, picks in all_picks.items():
            for pick in picks:
                if pick.symbol not in symbol_votes:
                    symbol_votes[pick.symbol] = 0
                    symbol_scores[pick.symbol] = []
                symbol_votes[pick.symbol] += 1
                symbol_scores[pick.symbol].append(pick.score)
        
        # 筛选被多个策略选中的
        consensus = []
        for symbol, votes in symbol_votes.items():
            if votes >= min_votes:
                avg_score = sum(symbol_scores[symbol]) / len(symbol_scores[symbol])
                consensus.append((symbol, votes, avg_score))
        
        # 按票数和分数排序
        consensus.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return consensus
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[Dict]:
        """列出所有策略"""
        return [
            {
                'key': key,
                'name': s.name,
                'description': s.description,
                'icon': s.icon,
                'win_rate': s.performance.win_rate
            }
            for key, s in self.strategies.items()
        ]


# 全局策略管理器
_strategy_manager = None

def get_strategy_manager() -> StrategyManager:
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
    return _strategy_manager
