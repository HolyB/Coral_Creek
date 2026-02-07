#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master Strategies
大师策略模块 - 融合蔡森、神奇九转、萧明道、黑马王子等大师的交易方法

功能:
1. 策略详细说明
2. 买卖点识别
3. 做T指导
4. 图形标注
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    T_BUY = "t_buy"      # 做T低吸
    T_SELL = "t_sell"    # 做T高抛
    HOLD = "hold"
    WATCH = "watch"


@dataclass
class TradingSignal:
    """交易信号"""
    date: str
    signal_type: SignalType
    price: float
    strength: int           # 1-5
    reason: str
    strategy: str
    action_desc: str = ""   # 具体操作描述


@dataclass
class MasterStrategy:
    """大师策略基类"""
    name: str
    master: str
    icon: str
    philosophy: str         # 核心理念
    description: str        # 策略描述
    buy_rules: List[str]    # 买入规则
    sell_rules: List[str]   # 卖出规则
    t_rules: List[str]      # 做T规则
    risk_rules: List[str]   # 风控规则
    
    def get_summary(self) -> Dict:
        return {
            'name': self.name,
            'master': self.master,
            'icon': self.icon,
            'philosophy': self.philosophy,
            'description': self.description
        }


# ==================================
# 大师策略定义
# ==================================

CAI_SEN_STRATEGY = MasterStrategy(
    name="蔡森量价突破",
    master="蔡森",
    icon="📊",
    philosophy="量价是市场的本质，量在价先",
    description="""
蔡森老师的核心理念是通过量价关系判断主力意图。
核心观点：
- 量是真的，价可以骗人
- 突破要放量，回踩要缩量
- 底部堆量是主力建仓信号
""",
    buy_rules=[
        "【黄金买点1】突破关键阻力位时放量超过5日均量1.5倍",
        "【黄金买点2】缩量回踩20日线，不破前低，出现放量阳线",
        "【黄金买点3】底部连续3天堆量，且量能逐日递增",
        "【黄金买点4】周线突破关键压力位，日线回踩周线支撑"
    ],
    sell_rules=[
        "【卖出信号1】放量滞涨：价格创新高但成交量萎缩",
        "【卖出信号2】跌破20日线且无法3天内收回",
        "【卖出信号3】高位出现天量阴线（见顶信号）",
        "【卖出信号4】MACD顶背离确认"
    ],
    t_rules=[
        "【做T低吸】阴线下跌到5日线支撑位低吸",
        "【做T高抛】阳线冲高回落到当日高点时减仓",
        "【每日T+0】早盘急跌3%低吸，午后反弹2%高抛"
    ],
    risk_rules=[
        "单笔止损不超过8%",
        "跌破20日均线强制止损",
        "仓位控制在30%以内",
        "大盘走弱时空仓等待"
    ]
)

TD_SEQUENTIAL = MasterStrategy(
    name="神奇九转",
    master="Tom DeMark",
    icon="🔢",
    philosophy="市场运行有时间周期规律",
    description="""
神奇九转（TD Sequential）由Tom DeMark发明，是一种基于时间周期的技术分析方法。
核心原理：
- 连续9根K线满足特定条件形成"买入准备"或"卖出准备"
- 市场在连续运动后有反转概率
- 结合支撑阻力效果更佳
""",
    buy_rules=[
        "【买入准备】连续9根K线收盘价都低于4根前的收盘价",
        "【确认信号】第9根K线出现后观察是否有止跌信号",
        "【最佳买点】第9根K线触及支撑位且出现下影线",
        "【加强信号】第8-9根K线成交量萎缩且RSI超卖"
    ],
    sell_rules=[
        "【卖出准备】连续9根K线收盘价都高于4根前的收盘价",
        "【确认信号】第9根K线出现后观察是否有滞涨",
        "【最佳卖点】第9根K线触及阻力位且出现上影线",
        "【加强信号】第8-9根K线量价背离"
    ],
    t_rules=[
        "【九转低吸】在第7-9根下跌K线时分批低吸",
        "【九转高抛】在第7-9根上涨K线时分批高抛",
        "【周期套利】结合日线和小时线九转差异做T"
    ],
    risk_rules=[
        "九转失败立即止损（第10根继续原方向）",
        "不要在九转未完成时提前进场",
        "配合大级别趋势使用效果更佳",
        "止损设在第9根K线极值外1-2%"
    ]
)

XIAO_MINGDAO = MasterStrategy(
    name="萧明道量价结构",
    master="萧明道",
    icon="📐",
    philosophy="量价结构决定一切",
    description="""
萧明道老师强调通过量价结构分析主力行为。
核心要点：
- 量价配合判断趋势健康度
- 结构完整性决定涨幅空间
- 主力成本线是关键支撑
""",
    buy_rules=[
        "【量价齐升】价涨量增，突破时量能放大至1.5倍以上",
        "【缩量回踩】回调时量能萎缩至均量50%以下",
        "【黄金坑】急跌后放量阳线吞噬前期阴线",
        "【平台突破】横盘整理后放量突破平台上沿"
    ],
    sell_rules=[
        "【量价背离】价格新高但量能萎缩",
        "【巨量滞涨】放出天量但价格横盘或微涨",
        "【破位确认】跌破关键支撑且3日不能收回",
        "【趋势反转】均线系统死叉确认"
    ],
    t_rules=[
        "【结构T】在上涨结构中箱体下沿低吸，上沿高抛",
        "【均线T】跌到5日线低吸，涨到10日线高抛",
        "【量能T】缩量阴线买，放量阳线卖"
    ],
    risk_rules=[
        "结构破坏立即止损",
        "单笔亏损控制在总资金2%以内",
        "永远不在下跌结构中抄底",
        "大盘不好时降低仓位"
    ]
)

HEIMA_PRINCE = MasterStrategy(
    name="黑马王子量学",
    master="黑马王子",
    icon="🐴",
    philosophy="量柱是主力留下的密码",
    description="""
黑马王子的"量柱擒涨停"理论，通过量柱分析主力意图。
核心概念：
- 倍量柱：涨停基因
- 高量柱：主力入场
- 缩量柱：洗盘信号
- 黄金柱：起爆点
""",
    buy_rules=[
        "【倍量起涨】今日量是昨日2倍以上，配合阳线突破",
        "【黄金柱】量柱缩到极小后突然放量，是起涨信号",
        "【百日低量】100日内最低量能后的第一根放量阳线",
        "【三缩二倍】连续3天缩量后出现倍量阳线"
    ],
    sell_rules=[
        "【天量顶】出现历史天量需警惕",
        "【量价背离顶】价格新高量柱萎缩",
        "【倍阴柱】阴线成交量是前日2倍以上",
        "【跌破支撑量柱】跌破关键量柱价位止损"
    ],
    t_rules=[
        "【量柱支撑T】回踩到重要量柱顶部支撑低吸",
        "【缩量回落T】极度缩量时低吸，放量时高抛",
        "【分时量T】分时量能萎缩时低吸，量能突增时高抛"
    ],
    risk_rules=[
        "跌破关键量柱强制止损",
        "天量后5日内不追高",
        "下跌途中的放量视为出货",
        "盘中异常放量需立即观察"
    ]
)

BLUE_INDICATOR = MasterStrategy(
    name="BLUE趋势共振",
    master="技术量化",
    icon="🔵",
    philosophy="多周期共振是趋势确认的关键",
    description="""
BLUE指标综合了多个技术因子，通过多周期共振判断趋势强度。
核心逻辑：
- 日线BLUE判断短期动能
- 周线BLUE判断中期趋势
- 月线BLUE判断长期方向
- 三线共振是最强信号
""",
    buy_rules=[
        "【强势信号】日线BLUE > 150，趋势启动",
        "【三线共振】日/周/月BLUE同时 > 80，强烈看多",
        "【黑马信号】日线BLUE从低点急速拉升超过50",
        "【超级强势】日线BLUE > 200，市场极度活跃"
    ],
    sell_rules=[
        "【动能衰减】日线BLUE从高点回落超过30%",
        "【趋势结束】周线BLUE跌破80",
        "【高位钝化】BLUE在高位横盘超过5日",
        "【死叉信号】短周期BLUE下穿长周期BLUE"
    ],
    t_rules=[
        "【BLUE回踩T】日BLUE回落至100-120区间低吸",
        "【冲高回落T】BLUE冲高回落时在高点减仓",
        "【日内T】盘中BLUE急跌时低吸，反弹时高抛"
    ],
    risk_rules=[
        "BLUE < 80 时不做多",
        "BLUE快速下跌时立即减仓",
        "只在BLUE上升趋势中做T",
        "尊重大周期BLUE方向"
    ]
)


# ==================================
# 策略管理器
# ==================================

MASTER_STRATEGIES = {
    'cai_sen': CAI_SEN_STRATEGY,
    'td_sequential': TD_SEQUENTIAL,
    'xiao_mingdao': XIAO_MINGDAO,
    'heima': HEIMA_PRINCE,
    'blue': BLUE_INDICATOR
}


def get_all_master_strategies() -> Dict[str, MasterStrategy]:
    """获取所有大师策略"""
    return MASTER_STRATEGIES


def get_strategy_guide(strategy_key: str) -> Optional[MasterStrategy]:
    """获取策略指南"""
    return MASTER_STRATEGIES.get(strategy_key)


# ==================================
# 信号识别
# ==================================

class SignalDetector:
    """信号检测器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self._prepare_indicators()
    
    def _prepare_indicators(self):
        """准备技术指标"""
        df = self.df
        
        # 基础指标
        df['sma5'] = df['Close'].rolling(5).mean()
        df['sma10'] = df['Close'].rolling(10).mean()
        df['sma20'] = df['Close'].rolling(20).mean()
        
        # 成交量
        df['vol_sma5'] = df['Volume'].rolling(5).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma5']
        
        # 涨跌幅
        df['change'] = df['Close'].pct_change() * 100
        
        # 振幅
        df['amplitude'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100
        
        self.df = df
    
    def detect_td_sequential(self) -> List[TradingSignal]:
        """检测神奇九转信号"""
        signals = []
        df = self.df.copy()
        
        # 计算买入准备 (收盘价 < 4天前收盘价的连续次数)
        df['buy_setup'] = (df['Close'] < df['Close'].shift(4)).astype(int)
        df['buy_count'] = 0
        
        count = 0
        for i in range(len(df)):
            if df['buy_setup'].iloc[i] == 1:
                count += 1
            else:
                count = 0
            df.iloc[i, df.columns.get_loc('buy_count')] = count
        
        # 计算卖出准备
        df['sell_setup'] = (df['Close'] > df['Close'].shift(4)).astype(int)
        df['sell_count'] = 0
        
        count = 0
        for i in range(len(df)):
            if df['sell_setup'].iloc[i] == 1:
                count += 1
            else:
                count = 0
            df.iloc[i, df.columns.get_loc('sell_count')] = count
        
        # 生成信号
        for i in range(len(df)):
            row = df.iloc[i]
            date = str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])
            
            if row['buy_count'] == 9:
                signals.append(TradingSignal(
                    date=date,
                    signal_type=SignalType.BUY,
                    price=row['Close'],
                    strength=4,
                    reason="神奇九转买入准备完成",
                    strategy="td_sequential",
                    action_desc="连续9天收盘价低于4天前，观察反转信号"
                ))
            
            if row['sell_count'] == 9:
                signals.append(TradingSignal(
                    date=date,
                    signal_type=SignalType.SELL,
                    price=row['Close'],
                    strength=4,
                    reason="神奇九转卖出准备完成",
                    strategy="td_sequential",
                    action_desc="连续9天收盘价高于4天前，观察见顶信号"
                ))
        
        return signals
    
    def detect_volume_signals(self) -> List[TradingSignal]:
        """检测量价信号 (蔡森/黑马王子)"""
        signals = []
        df = self.df
        
        for i in range(5, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            date = str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])
            
            vol_ratio = row['vol_ratio'] if not np.isnan(row['vol_ratio']) else 1
            
            # 倍量阳线 (黑马王子)
            if vol_ratio >= 2.0 and row['Close'] > row['Open'] and row['change'] > 2:
                signals.append(TradingSignal(
                    date=date,
                    signal_type=SignalType.BUY,
                    price=row['Close'],
                    strength=4,
                    reason=f"倍量阳线 ({vol_ratio:.1f}倍量)",
                    strategy="heima",
                    action_desc="量柱放大，主力入场信号，可适量跟进"
                ))
            
            # 缩量回踩支撑 (蔡森)
            if vol_ratio < 0.5 and row['Low'] <= row['sma20'] and row['Close'] > row['sma20']:
                signals.append(TradingSignal(
                    date=date,
                    signal_type=SignalType.T_BUY,
                    price=row['Close'],
                    strength=3,
                    reason="缩量回踩20日线支撑",
                    strategy="cai_sen",
                    action_desc="缩量回踩均线不破，可做T低吸"
                ))
            
            # 放量滞涨顶部 (蔡森)
            if vol_ratio > 1.5 and abs(row['change']) < 1 and row['Close'] > row['sma20'] * 1.1:
                signals.append(TradingSignal(
                    date=date,
                    signal_type=SignalType.T_SELL,
                    price=row['Close'],
                    strength=3,
                    reason="放量滞涨，可能见顶",
                    strategy="cai_sen",
                    action_desc="高位放量但涨幅有限，可做T高抛"
                ))
        
        return signals
    
    def detect_blue_signals(self, blue_daily: float = None, 
                             blue_weekly: float = None) -> List[TradingSignal]:
        """检测BLUE信号"""
        signals = []
        
        if blue_daily is None:
            return signals
        
        date = datetime.now().strftime('%Y-%m-%d')
        price = self.df['Close'].iloc[-1] if len(self.df) > 0 else 0
        
        # 强势信号
        if blue_daily > 180:
            signals.append(TradingSignal(
                date=date,
                signal_type=SignalType.BUY,
                price=price,
                strength=5,
                reason=f"BLUE日线强势 ({blue_daily:.0f})",
                strategy="blue",
                action_desc="趋势极强，可适当追高或等回踩"
            ))
        elif blue_daily > 150:
            signals.append(TradingSignal(
                date=date,
                signal_type=SignalType.BUY,
                price=price,
                strength=4,
                reason=f"BLUE日线突破 ({blue_daily:.0f})",
                strategy="blue",
                action_desc="趋势启动，可分批建仓"
            ))
        
        # 周线共振
        if blue_weekly and blue_weekly > 100 and blue_daily > 120:
            for s in signals:
                s.strength = min(5, s.strength + 1)
                s.reason += f" + 周线共振({blue_weekly:.0f})"
        
        return signals
    
    def get_all_signals(self, blue_daily: float = None, 
                         blue_weekly: float = None) -> List[TradingSignal]:
        """获取所有信号"""
        signals = []
        
        signals.extend(self.detect_td_sequential())
        signals.extend(self.detect_volume_signals())
        signals.extend(self.detect_blue_signals(blue_daily, blue_weekly))
        
        # 按日期排序
        signals.sort(key=lambda x: x.date, reverse=True)
        
        return signals


# ==================================
# 操作指南生成
# ==================================

def generate_trading_guide(symbol: str, df: pd.DataFrame,
                            blue_daily: float = None,
                            blue_weekly: float = None) -> Dict:
    """
    生成个股操作指南
    
    Returns:
        {
            'signals': List[TradingSignal],
            'recommendations': List[str],
            'risk_warnings': List[str],
            't_opportunities': List[str]
        }
    """
    detector = SignalDetector(df)
    signals = detector.get_all_signals(blue_daily, blue_weekly)
    
    # 生成建议
    recommendations = []
    risk_warnings = []
    t_opportunities = []
    
    # 基于BLUE的建议
    if blue_daily:
        if blue_daily > 180:
            recommendations.append("📈 BLUE强势，趋势明确，可持股待涨")
            t_opportunities.append("回踩5日线时可做T低吸")
        elif blue_daily > 150:
            recommendations.append("📊 BLUE启动，可分批建仓追踪")
            t_opportunities.append("冲高回落时可做T高抛，保留底仓")
        elif blue_daily > 100:
            recommendations.append("⚖️ BLUE中性偏强，观望为主")
        else:
            risk_warnings.append("⚠️ BLUE偏弱，不宜追高")
    
    # 基于量价的警示
    if len(df) > 5:
        recent_vol = df['Volume'].iloc[-5:].mean()
        prev_vol = df['Volume'].iloc[-10:-5].mean()
        if recent_vol > prev_vol * 1.5:
            if df['Close'].iloc[-1] > df['Close'].iloc[-5]:
                recommendations.append("📊 近期放量上涨，趋势健康")
            else:
                risk_warnings.append("⚠️ 放量下跌，注意风险")
    
    # 做T时机
    if len(df) > 0:
        current_price = df['Close'].iloc[-1]
        sma5 = df['Close'].rolling(5).mean().iloc[-1]
        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        
        if current_price < sma5 and current_price > sma20:
            t_opportunities.append(f"当前价格({current_price:.2f})回踩5日线，可考虑低吸")
        
        if current_price > sma5 * 1.03:
            t_opportunities.append(f"当前价格偏离5日线较远，可考虑高抛")
    
    return {
        'signals': signals,
        'recommendations': recommendations,
        'risk_warnings': risk_warnings,
        't_opportunities': t_opportunities
    }


def format_strategy_for_display(strategy: MasterStrategy) -> str:
    """格式化策略用于显示"""
    text = f"""
## {strategy.icon} {strategy.name}

**创始人**: {strategy.master}

**核心理念**: {strategy.philosophy}

{strategy.description}

---

### ✅ 买入规则
"""
    for rule in strategy.buy_rules:
        text += f"- {rule}\n"
    
    text += "\n### ❌ 卖出规则\n"
    for rule in strategy.sell_rules:
        text += f"- {rule}\n"
    
    text += "\n### 🔄 做T技巧\n"
    for rule in strategy.t_rules:
        text += f"- {rule}\n"
    
    text += "\n### ⚠️ 风控规则\n"
    for rule in strategy.risk_rules:
        text += f"- {rule}\n"
    
    return text


# ==================================
# 个股大师分析
# ==================================

@dataclass
class MasterAnalysis:
    """单个大师的分析结果"""
    master: str
    icon: str
    action: str              # 买入/卖出/做T/观望/持有
    action_emoji: str        # 🟢/🔴/🟡/⚪
    confidence: int          # 1-5
    reason: str              # 判断理由
    operation: str           # 具体操作说明
    stop_loss: str = ""      # 止损建议
    take_profit: str = ""    # 止盈建议


def analyze_stock_for_master(
    symbol: str,
    blue_daily: float = None,
    blue_weekly: float = None,
    blue_monthly: float = None,
    adx: float = None,
    vol_ratio: float = None,      # 今日量/5日均量
    change_pct: float = None,     # 今日涨跌幅
    price: float = None,
    sma5: float = None,
    sma20: float = None,
    is_heima: bool = False,
    td_count: int = 0,            # 神奇九转计数 (负数=下跌，正数=上涨)
    chip_pattern: str = ""
) -> Dict[str, MasterAnalysis]:
    """
    为单只股票生成各大师的操作建议
    
    Args:
        symbol: 股票代码
        blue_daily: 日线BLUE
        blue_weekly: 周线BLUE
        blue_monthly: 月线BLUE
        adx: ADX趋势强度
        vol_ratio: 量比
        change_pct: 涨跌幅
        price: 当前价格
        sma5: 5日均线
        sma20: 20日均线
        is_heima: 是否黑马信号
        td_count: TD九转计数
        chip_pattern: 筹码形态
    
    Returns:
        Dict[master_key, MasterAnalysis]
    """
    analyses = {}
    
    # === 蔡森量价分析 ===
    cai_sen = _analyze_cai_sen(
        vol_ratio=vol_ratio,
        change_pct=change_pct,
        price=price,
        sma5=sma5,
        sma20=sma20
    )
    analyses['cai_sen'] = cai_sen
    
    # === 神奇九转分析 ===
    td = _analyze_td_sequential(td_count=td_count)
    analyses['td_sequential'] = td
    
    # === 萧明道量价结构 ===
    xiao = _analyze_xiao_mingdao(
        vol_ratio=vol_ratio,
        change_pct=change_pct,
        price=price,
        sma5=sma5,
        sma20=sma20
    )
    analyses['xiao_mingdao'] = xiao
    
    # === 黑马王子量学 ===
    heima = _analyze_heima_prince(
        vol_ratio=vol_ratio,
        change_pct=change_pct,
        is_heima=is_heima
    )
    analyses['heima'] = heima
    
    # === BLUE趋势共振 ===
    blue = _analyze_blue_indicator(
        blue_daily=blue_daily,
        blue_weekly=blue_weekly,
        blue_monthly=blue_monthly,
        adx=adx
    )
    analyses['blue'] = blue
    
    return analyses


def _analyze_cai_sen(vol_ratio: float = None, change_pct: float = None,
                     price: float = None, sma5: float = None, sma20: float = None) -> MasterAnalysis:
    """蔡森量价分析"""
    vol_ratio = vol_ratio or 1.0
    change_pct = change_pct or 0
    
    # 判断当前状态
    if vol_ratio >= 1.5 and change_pct > 2:
        return MasterAnalysis(
            master="蔡森",
            icon="📊",
            action="买入",
            action_emoji="🟢",
            confidence=4,
            reason=f"放量突破 (量比{vol_ratio:.1f}倍)",
            operation="突破时放量，符合黄金买点1，可跟进",
            stop_loss="跌破20日线或下跌8%止损",
            take_profit="目标位：突破后涨幅15-20%"
        )
    elif vol_ratio < 0.6 and price and sma20 and abs(price - sma20) / sma20 < 0.02:
        return MasterAnalysis(
            master="蔡森",
            icon="📊",
            action="做T低吸",
            action_emoji="🟡",
            confidence=3,
            reason="缩量回踩20日线支撑",
            operation="符合黄金买点2，可在20日线附近低吸做T",
            stop_loss="跌破前低或20日线",
            take_profit="反弹至5日线上方高抛"
        )
    elif vol_ratio > 1.5 and abs(change_pct) < 1:
        return MasterAnalysis(
            master="蔡森",
            icon="📊",
            action="做T高抛",
            action_emoji="🟡",
            confidence=3,
            reason=f"放量滞涨 (量比{vol_ratio:.1f})",
            operation="高位放量但涨幅有限，可能见顶，建议做T高抛",
            stop_loss="无持仓则不操作",
            take_profit="减仓30-50%"
        )
    elif vol_ratio > 2 and change_pct < -3:
        return MasterAnalysis(
            master="蔡森",
            icon="📊",
            action="卖出/观望",
            action_emoji="🔴",
            confidence=4,
            reason=f"巨量阴线 (量比{vol_ratio:.1f}，跌{change_pct:.1f}%)",
            operation="高位巨量阴线是见顶信号，建议清仓或减仓",
            stop_loss="已破位，止损离场",
            take_profit=""
        )
    else:
        return MasterAnalysis(
            master="蔡森",
            icon="📊",
            action="观望",
            action_emoji="⚪",
            confidence=2,
            reason="量价关系不明确",
            operation="等待放量突破信号或缩量回踩机会"
        )


def _analyze_td_sequential(td_count: int = 0) -> MasterAnalysis:
    """神奇九转分析"""
    if td_count <= -7:
        return MasterAnalysis(
            master="Tom DeMark",
            icon="🔢",
            action="准备买入",
            action_emoji="🟢",
            confidence=4 if td_count <= -9 else 3,
            reason=f"九转下跌第{abs(td_count)}根",
            operation=f"连续{abs(td_count)}天收盘价低于4天前，" + 
                     ("九转完成，可分批低吸" if td_count <= -9 else "接近买点，准备资金"),
            stop_loss="九转失败(第10根继续跌)止损",
            take_profit="反弹至第5-6根K线高点"
        )
    elif td_count >= 7:
        return MasterAnalysis(
            master="Tom DeMark",
            icon="🔢",
            action="准备卖出",
            action_emoji="🔴",
            confidence=4 if td_count >= 9 else 3,
            reason=f"九转上涨第{td_count}根",
            operation=f"连续{td_count}天收盘价高于4天前，" +
                     ("九转完成，可分批高抛" if td_count >= 9 else "接近卖点，准备减仓"),
            stop_loss="设在第7根K线低点",
            take_profit="目标已达成，分批止盈"
        )
    elif 4 <= td_count <= 6:
        return MasterAnalysis(
            master="Tom DeMark",
            icon="🔢",
            action="持有/观察",
            action_emoji="🟡",
            confidence=2,
            reason=f"九转上涨第{td_count}根",
            operation="上涨中继，持股待涨，关注是否完成九转",
            take_profit="等待九转完成后分批止盈"
        )
    elif -6 <= td_count <= -4:
        return MasterAnalysis(
            master="Tom DeMark",
            icon="🔢",
            action="观望",
            action_emoji="🟡",
            confidence=2,
            reason=f"九转下跌第{abs(td_count)}根",
            operation="下跌中继，不要抄底，等待九转完成",
            stop_loss="已持仓考虑减仓"
        )
    else:
        return MasterAnalysis(
            master="Tom DeMark",
            icon="🔢",
            action="中性",
            action_emoji="⚪",
            confidence=1,
            reason="无明显九转信号",
            operation="数据不足或无连续趋势，继续观察"
        )


def _analyze_xiao_mingdao(vol_ratio: float = None, change_pct: float = None,
                          price: float = None, sma5: float = None, sma20: float = None) -> MasterAnalysis:
    """萧明道量价结构分析"""
    vol_ratio = vol_ratio or 1.0
    change_pct = change_pct or 0
    
    # 判断结构
    above_ma = price and sma20 and price > sma20
    near_ma5 = price and sma5 and abs(price - sma5) / sma5 < 0.02
    
    if vol_ratio >= 1.5 and change_pct > 2 and above_ma:
        return MasterAnalysis(
            master="萧明道",
            icon="📐",
            action="买入",
            action_emoji="🟢",
            confidence=4,
            reason="量价齐升，结构健康",
            operation="放量突破，上涨结构完整，可跟进做多",
            stop_loss="跌破关键支撑位(前低或20日线)",
            take_profit="根据结构目标位止盈"
        )
    elif vol_ratio < 0.5 and above_ma and change_pct < 0:
        return MasterAnalysis(
            master="萧明道",
            icon="📐",
            action="做T低吸",
            action_emoji="🟡",
            confidence=3,
            reason="缩量回调，洗盘形态",
            operation="缩量回踩，上涨结构完好，可在均线支撑低吸",
            stop_loss="结构破坏(跌破前低)止损",
            take_profit="反弹至结构高点附近"
        )
    elif vol_ratio > 2 and abs(change_pct) < 1 and above_ma:
        return MasterAnalysis(
            master="萧明道",
            icon="📐",
            action="做T高抛",
            action_emoji="🟡",
            confidence=3,
            reason="巨量滞涨，警惕",
            operation="放天量但价格不涨，主力可能出货，建议减仓",
            stop_loss="跌破当日低点",
            take_profit="当日高点附近减仓"
        )
    elif vol_ratio > 1.5 and change_pct < -3:
        return MasterAnalysis(
            master="萧明道",
            icon="📐",
            action="卖出",
            action_emoji="🔴",
            confidence=4,
            reason="破位确认",
            operation="放量下跌，结构可能破坏，建议离场",
            stop_loss="立即止损"
        )
    else:
        return MasterAnalysis(
            master="萧明道",
            icon="📐",
            action="观望",
            action_emoji="⚪",
            confidence=2,
            reason="结构不明确",
            operation="等待明确的量价结构信号"
        )


def _analyze_heima_prince(vol_ratio: float = None, change_pct: float = None,
                          is_heima: bool = False) -> MasterAnalysis:
    """黑马王子量学分析"""
    vol_ratio = vol_ratio or 1.0
    change_pct = change_pct or 0
    
    if vol_ratio >= 2.0 and change_pct > 3:
        return MasterAnalysis(
            master="黑马王子",
            icon="🐴",
            action="强烈买入",
            action_emoji="🟢",
            confidence=5,
            reason=f"倍量阳线! (量比{vol_ratio:.1f}倍，涨{change_pct:.1f}%)",
            operation="倍量阳线是涨停基因，可积极跟进，明日可能继续涨停",
            stop_loss=f"跌破今日低点或下跌5%",
            take_profit="持股待涨，涨停板附近减仓"
        )
    elif is_heima and change_pct > 0:
        return MasterAnalysis(
            master="黑马王子",
            icon="🐴",
            action="买入",
            action_emoji="🟢",
            confidence=4,
            reason="黑马信号确认",
            operation="出现黑马形态，主力建仓迹象明显，可跟进",
            stop_loss="跌破信号确认日低点",
            take_profit="目标涨幅15-30%"
        )
    elif vol_ratio < 0.3 and abs(change_pct) < 2:
        return MasterAnalysis(
            master="黑马王子",
            icon="🐴",
            action="关注",
            action_emoji="🟡",
            confidence=3,
            reason=f"极度缩量 (量比{vol_ratio:.1f})",
            operation="量柱萎缩至极小，可能是缩量洗盘，关注后续是否放量",
            stop_loss="跌破缩量区间低点",
            take_profit="等待倍量阳线出现"
        )
    elif vol_ratio > 2 and change_pct < -3:
        return MasterAnalysis(
            master="黑马王子",
            icon="🐴",
            action="卖出",
            action_emoji="🔴",
            confidence=5,
            reason=f"倍阴柱! (量比{vol_ratio:.1f}倍，跌{change_pct:.1f}%)",
            operation="倍量阴线是出货信号，立即清仓，不可恋战",
            stop_loss="已触发止损信号"
        )
    else:
        return MasterAnalysis(
            master="黑马王子",
            icon="🐴",
            action="观望",
            action_emoji="⚪",
            confidence=2,
            reason="量柱形态不明确",
            operation="等待明确的量柱信号(倍量/缩量)"
        )


def _analyze_blue_indicator(blue_daily: float = None, blue_weekly: float = None,
                            blue_monthly: float = None, adx: float = None) -> MasterAnalysis:
    """BLUE指标分析"""
    if blue_daily is None:
        return MasterAnalysis(
            master="BLUE",
            icon="🔵",
            action="无数据",
            action_emoji="⚪",
            confidence=0,
            reason="BLUE数据未获取",
            operation=""
        )
    
    # 三线共振判断
    triple_resonance = (
        blue_daily and blue_daily > 100 and
        blue_weekly and blue_weekly > 80 and
        blue_monthly and blue_monthly > 60
    )
    
    if blue_daily > 200:
        return MasterAnalysis(
            master="BLUE",
            icon="🔵",
            action="强烈买入",
            action_emoji="🟢",
            confidence=5,
            reason=f"BLUE超强势 ({blue_daily:.0f})" + (" + 三线共振" if triple_resonance else ""),
            operation="趋势极强，可适当追高或等回踩5日线低吸",
            stop_loss="BLUE跌破150或价格跌破5日线",
            take_profit="持股待涨，BLUE开始回落时减仓"
        )
    elif blue_daily > 150:
        return MasterAnalysis(
            master="BLUE",
            icon="🔵",
            action="买入",
            action_emoji="🟢",
            confidence=4,
            reason=f"BLUE强势 ({blue_daily:.0f})" + (" + 三线共振" if triple_resonance else ""),
            operation="趋势启动，可分批建仓，逢回调加仓",
            stop_loss="BLUE跌破100或价格跌破20日线",
            take_profit="目标位：前高或涨幅15%"
        )
    elif blue_daily > 100:
        return MasterAnalysis(
            master="BLUE",
            icon="🔵",
            action="做T/持有",
            action_emoji="🟡",
            confidence=3,
            reason=f"BLUE中性偏强 ({blue_daily:.0f})",
            operation="趋势尚可，可小仓位做T或持有底仓",
            stop_loss="BLUE跌破80",
            take_profit="等待BLUE突破150加仓"
        )
    elif blue_daily > 80:
        return MasterAnalysis(
            master="BLUE",
            icon="🔵",
            action="观望",
            action_emoji="⚪",
            confidence=2,
            reason=f"BLUE弱势 ({blue_daily:.0f})",
            operation="趋势偏弱，不宜追高，等待BLUE回升",
            stop_loss="已持仓考虑减仓"
        )
    else:
        return MasterAnalysis(
            master="BLUE",
            icon="🔵",
            action="回避",
            action_emoji="🔴",
            confidence=4,
            reason=f"BLUE很弱 ({blue_daily:.0f})",
            operation="趋势向下，不要抄底，等待BLUE企稳",
            stop_loss="清仓观望"
        )


def get_master_summary_for_stock(analyses: Dict[str, MasterAnalysis], profile: str = "medium") -> Dict:
    """
    汇总各大师的分析，给出综合建议
    
    使用加权共识机制:
    1. 每个大师的投票权重 = 信号置信度 / 5
    2. 买入/卖出权重求和
    3. 当存在冲突时，提供详细说明
    
    Returns:
        {
            'overall_action': str,       # 综合建议
            'overall_signal': str,       # 'BUY' / 'SELL' / 'HOLD' / 'CONFLICT'
            'consensus_score': float,    # 共识强度 0-100
            'buy_votes': int,
            'sell_votes': int,
            'hold_votes': int,
            'weighted_buy': float,       # 加权买入分
            'weighted_sell': float,      # 加权卖出分
            'best_opportunity': str,     # 最佳机会描述
            'key_risk': str,             # 主要风险
            'conflict_warning': str,     # 冲突警告 (如果有)
            'confidence_avg': float      # 平均置信度
        }
    """
    buy_votes = 0
    sell_votes = 0
    hold_votes = 0
    
    weighted_buy = 0.0
    weighted_sell = 0.0
    
    best_opportunity = ""
    best_confidence = 0
    key_risk = ""
    
    confidence_sum = 0
    
    # 策略组合层：按交易偏好给不同大师动态配权
    profile = (profile or "medium").lower()
    if profile == "short":
        default_weights = {
            'cai_sen': 1.15,       # 量价突破更重要
            'td_sequential': 1.00, # 短期拐点
            'xiao_mingdao': 0.85,
            'heima': 1.15,         # 爆发力
            'blue': 0.95
        }
    elif profile == "long":
        default_weights = {
            'cai_sen': 0.90,
            'td_sequential': 0.75,
            'xiao_mingdao': 1.20,  # 结构稳定
            'heima': 0.80,
            'blue': 1.30           # 趋势优先
        }
    else:
        # medium
        default_weights = {
            'cai_sen': 1.00,
            'td_sequential': 0.85,
            'xiao_mingdao': 1.05,
            'heima': 0.90,
            'blue': 1.20
        }

    # 根据 BLUE 强度做市场状态自适应，减少单一风格失效
    blue_analysis = analyses.get('blue')
    if blue_analysis:
        action_text = str(blue_analysis.action)
        if "强烈买入" in action_text:
            default_weights['blue'] = default_weights.get('blue', 1.0) * 1.15
            default_weights['cai_sen'] = default_weights.get('cai_sen', 1.0) * 1.08
        elif "回避" in action_text:
            default_weights['xiao_mingdao'] = default_weights.get('xiao_mingdao', 1.0) * 1.10
            default_weights['td_sequential'] = default_weights.get('td_sequential', 1.0) * 1.10
    
    buy_masters = []
    sell_masters = []
    
    for key, analysis in analyses.items():
        weight = default_weights.get(key, 1.0)
        confidence_score = analysis.confidence * weight
        confidence_sum += analysis.confidence
        
        if analysis.action in ['买入', '强烈买入', '做T低吸', '准备买入']:
            buy_votes += 1
            weighted_buy += confidence_score
            buy_masters.append(f"{analysis.icon}{analysis.master}")
            
            if analysis.confidence > best_confidence and '买入' in analysis.action:
                best_confidence = analysis.confidence
                best_opportunity = f"{analysis.icon}{analysis.master}: {analysis.reason}"
                
        elif analysis.action in ['卖出', '做T高抛', '准备卖出', '回避']:
            sell_votes += 1
            weighted_sell += confidence_score
            sell_masters.append(f"{analysis.icon}{analysis.master}")
            
            if analysis.confidence >= 4 and '卖' in analysis.action:
                key_risk = f"{analysis.icon}{analysis.master}: {analysis.reason}"
        else:
            hold_votes += 1
    
    # 计算共识强度
    total_votes = buy_votes + sell_votes + hold_votes
    confidence_avg = confidence_sum / total_votes if total_votes > 0 else 0
    
    # 共识分数 = |买入权重 - 卖出权重| / (买入权重 + 卖出权重 + 1) * 100
    consensus_score = abs(weighted_buy - weighted_sell) / (weighted_buy + weighted_sell + 0.01) * 100
    
    # 冲突检测
    conflict_warning = ""
    if buy_votes >= 2 and sell_votes >= 2:
        conflict_warning = f"⚠️ 大师分歧! 看多({', '.join(buy_masters)}) vs 看空({', '.join(sell_masters)})"
    elif buy_votes >= 1 and sell_votes >= 1:
        conflict_warning = f"⚠️ 信号冲突: {', '.join(buy_masters)} 看多 / {', '.join(sell_masters)} 看空"
    
    # 综合建议 (使用加权分数)
    net_score = weighted_buy - weighted_sell
    
    if net_score >= 4.0:
        overall = "🟢 强烈看多 - 多位大师一致看涨"
        overall_signal = "BUY"
    elif net_score >= 2.0:
        overall = "🟢 偏多 - 可适当参与"
        overall_signal = "BUY"
    elif net_score <= -4.0:
        overall = "🔴 强烈看空 - 建议回避或清仓"
        overall_signal = "SELL"
    elif net_score <= -2.0:
        overall = "🔴 偏空 - 建议减仓观望"
        overall_signal = "SELL"
    elif conflict_warning:
        overall = "⚠️ 信号冲突 - 等待明确方向"
        overall_signal = "CONFLICT"
    else:
        overall = "⚪ 信号不明确 - 建议观望"
        overall_signal = "HOLD"
    
    return {
        'overall_action': overall,
        'overall_signal': overall_signal,
        'profile': profile,
        'consensus_score': round(consensus_score, 1),
        'buy_votes': buy_votes,
        'sell_votes': sell_votes,
        'hold_votes': hold_votes,
        'weighted_buy': round(weighted_buy, 2),
        'weighted_sell': round(weighted_sell, 2),
        'best_opportunity': best_opportunity,
        'key_risk': key_risk,
        'conflict_warning': conflict_warning,
        'confidence_avg': round(confidence_avg, 1)
    }


def format_master_analysis_short(analyses: Dict[str, MasterAnalysis]) -> str:
    """生成简短的大师分析摘要 (用于表格显示)"""
    parts = []
    for key, analysis in analyses.items():
        parts.append(f"{analysis.action_emoji}")
    return "".join(parts)


def format_master_analysis_full(analyses: Dict[str, MasterAnalysis]) -> str:
    """生成完整的大师分析文本"""
    text = ""
    for key, analysis in analyses.items():
        text += f"\n**{analysis.icon} {analysis.master}**: {analysis.action_emoji} {analysis.action}\n"
        text += f"- 判断: {analysis.reason}\n"
        text += f"- 操作: {analysis.operation}\n"
        if analysis.stop_loss:
            text += f"- 止损: {analysis.stop_loss}\n"
        if analysis.take_profit:
            text += f"- 目标: {analysis.take_profit}\n"
    
    return text


if __name__ == "__main__":
    print("📚 Master Strategies Overview")
    print("=" * 50)
    
    for key, strategy in MASTER_STRATEGIES.items():
        print(f"\n{strategy.icon} {strategy.name} ({strategy.master})")
        print(f"   理念: {strategy.philosophy}")
        print(f"   买入规则: {len(strategy.buy_rules)}条")
        print(f"   卖出规则: {len(strategy.sell_rules)}条")
        print(f"   做T技巧: {len(strategy.t_rules)}条")
    
    # 测试个股分析
    print("\n" + "=" * 50)
    print("📊 测试个股分析 (NVDA)")
    
    analyses = analyze_stock_for_master(
        symbol="NVDA",
        blue_daily=165,
        blue_weekly=120,
        vol_ratio=1.8,
        change_pct=3.2,
        is_heima=True
    )
    
    for key, analysis in analyses.items():
        print(f"\n{analysis.icon} {analysis.master}: {analysis.action_emoji} {analysis.action}")
        print(f"   {analysis.reason}")
        print(f"   {analysis.operation}")
    
    summary = get_master_summary_for_stock(analyses)
    print(f"\n综合: {summary['overall_action']}")
    print(f"买入票数: {summary['buy_votes']}")
