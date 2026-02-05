"""
AI 智能选股器 (Smart Picker)
============================

核心设计理念:
1. 少而精 - 每天只推荐3-5只高置信度股票
2. 多重验证 - ML预测 + 技术信号 + 量价确认
3. 可执行 - 明确的入场/止损/目标价
4. 风控优先 - 先评估风险，再看收益

Pipeline:
    Stage 1: Pre-Filter (流动性、趋势、波动率)
    Stage 2: ML Score (收益预测 + 方向概率)
    Stage 3: Signal Validation (BLUE/MACD/成交量确认)
    Stage 4: Risk Assessment (止损位、仓位建议)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import json

try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@dataclass
class StockPick:
    """单个选股结果"""
    symbol: str
    name: str = ""
    price: float = 0.0
    
    # ML 预测
    pred_return_5d: float = 0.0
    pred_direction_prob: float = 0.5  # 上涨概率
    ml_confidence: float = 0.0
    
    # 信号验证
    signals_confirmed: List[str] = field(default_factory=list)
    signals_warning: List[str] = field(default_factory=list)
    signal_score: int = 0  # 0-5
    
    # 风险评估
    stop_loss_price: float = 0.0
    stop_loss_pct: float = 0.0
    target_price: float = 0.0
    target_pct: float = 0.0
    risk_reward_ratio: float = 0.0
    suggested_position_pct: float = 0.0
    
    # 综合评分
    overall_score: float = 0.0
    star_rating: int = 0  # 1-5 星
    
    # 元数据
    blue_daily: float = 0.0
    blue_weekly: float = 0.0
    blue_monthly: float = 0.0
    adx: float = 0.0
    rsi: float = 0.0
    volume_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': self.price,
            'pred_return_5d': self.pred_return_5d,
            'pred_direction_prob': self.pred_direction_prob,
            'ml_confidence': self.ml_confidence,
            'signals_confirmed': self.signals_confirmed,
            'signals_warning': self.signals_warning,
            'signal_score': self.signal_score,
            'stop_loss_price': self.stop_loss_price,
            'stop_loss_pct': self.stop_loss_pct,
            'target_price': self.target_price,
            'target_pct': self.target_pct,
            'risk_reward_ratio': self.risk_reward_ratio,
            'suggested_position_pct': self.suggested_position_pct,
            'overall_score': self.overall_score,
            'star_rating': self.star_rating,
            'blue_daily': self.blue_daily,
            'blue_weekly': self.blue_weekly,
            'blue_monthly': self.blue_monthly,
            'adx': self.adx,
            'rsi': self.rsi,
            'volume_ratio': self.volume_ratio,
        }


class SmartPicker:
    """智能选股器"""
    
    def __init__(self, market: str = 'US'):
        self.market = market
        self.model_dir = Path(__file__).parent / "saved_models" / f"v2_{market.lower()}"
        self.model = None
        self.feature_names = []
        self._load_model()
    
    def _load_model(self):
        """加载ML模型"""
        if not ML_AVAILABLE:
            return
        
        try:
            model_path = self.model_dir / "return_5d.joblib"
            if model_path.exists():
                self.model = joblib.load(model_path)
                
                feature_path = self.model_dir / "feature_names.json"
                if feature_path.exists():
                    with open(feature_path) as f:
                        self.feature_names = json.load(f)
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def pick(self, 
             signals_df: pd.DataFrame,
             price_history: Dict[str, pd.DataFrame],
             max_picks: int = 5) -> List[StockPick]:
        """
        智能选股
        
        Args:
            signals_df: 扫描信号 DataFrame (symbol, price, blue_daily, blue_weekly, etc.)
            price_history: 价格历史 {symbol: DataFrame}
            max_picks: 最大推荐数量
        
        Returns:
            排序后的选股结果
        """
        if signals_df.empty:
            return []
        
        picks = []
        
        for _, row in signals_df.iterrows():
            symbol = row.get('symbol', '')
            if not symbol:
                continue
            
            # 获取价格历史
            history = price_history.get(symbol, pd.DataFrame())
            
            # 创建选股对象
            pick = self._analyze_stock(row, history)
            if pick and pick.overall_score > 0:
                picks.append(pick)
        
        # 按综合评分排序
        picks.sort(key=lambda x: x.overall_score, reverse=True)
        
        # 返回 Top N
        return picks[:max_picks]
    
    def _analyze_stock(self, signal: pd.Series, history: pd.DataFrame) -> Optional[StockPick]:
        """分析单只股票"""
        symbol = signal.get('symbol', '')
        price = float(signal.get('price', 0))
        
        if price <= 0:
            return None
        
        pick = StockPick(
            symbol=symbol,
            name=signal.get('company_name', ''),
            price=price,
            blue_daily=float(signal.get('blue_daily', 0)),
            blue_weekly=float(signal.get('blue_weekly', 0)),
            blue_monthly=float(signal.get('blue_monthly', 0)),
        )
        
        # === Stage 1: Pre-Filter ===
        if not self._pass_prefilter(signal, history):
            return None
        
        # === Stage 2: ML Prediction ===
        ml_result = self._ml_predict(signal, history)
        pick.pred_return_5d = ml_result.get('pred_return', 0)
        pick.pred_direction_prob = ml_result.get('direction_prob', 0.5)
        pick.ml_confidence = ml_result.get('confidence', 0)
        
        # === Stage 3: Signal Validation ===
        validation = self._validate_signals(signal, history)
        pick.signals_confirmed = validation['confirmed']
        pick.signals_warning = validation['warnings']
        pick.signal_score = validation['score']
        pick.adx = validation.get('adx', 0)
        pick.rsi = validation.get('rsi', 50)
        pick.volume_ratio = validation.get('volume_ratio', 1.0)
        
        # === Stage 4: Risk Assessment ===
        risk = self._assess_risk(pick, history)
        pick.stop_loss_price = risk['stop_loss_price']
        pick.stop_loss_pct = risk['stop_loss_pct']
        pick.target_price = risk['target_price']
        pick.target_pct = risk['target_pct']
        pick.risk_reward_ratio = risk['risk_reward_ratio']
        pick.suggested_position_pct = risk['position_pct']
        
        # === 综合评分 ===
        pick.overall_score = self._calculate_overall_score(pick)
        pick.star_rating = self._get_star_rating(pick.overall_score)
        
        return pick
    
    def _pass_prefilter(self, signal: pd.Series, history: pd.DataFrame) -> bool:
        """预过滤"""
        # 1. 价格过滤 (排除仙股和高价股)
        price = float(signal.get('price', 0))
        if self.market == 'US':
            if price < 5 or price > 500:
                return False
        else:  # CN
            if price < 3 or price > 300:
                return False
        
        # 2. BLUE 信号过滤 (至少有日线信号)
        blue_d = float(signal.get('blue_daily', 0))
        if blue_d < 50:  # 太弱的信号不要
            return False
        
        # 3. 如果有历史数据，检查成交量
        if not history.empty and len(history) >= 20:
            vol_col = 'Volume' if 'Volume' in history.columns else 'volume'
            if vol_col in history.columns:
                recent_vol = history[vol_col].iloc[-5:].mean()
                avg_vol = history[vol_col].iloc[-20:].mean()
                # 成交量萎缩太厉害不要
                if recent_vol < avg_vol * 0.3:
                    return False
        
        return True
    
    def _ml_predict(self, signal: pd.Series, history: pd.DataFrame) -> Dict:
        """ML预测"""
        result = {
            'pred_return': 0.0,
            'direction_prob': 0.5,
            'confidence': 0.0
        }
        
        if self.model is None or history.empty or len(history) < 60:
            # 无模型或数据不足，使用规则估计
            blue_d = float(signal.get('blue_daily', 0))
            blue_w = float(signal.get('blue_weekly', 0))
            
            # 简单规则: BLUE越高，预期收益越好
            if blue_d >= 120 and blue_w >= 100:
                result['pred_return'] = 3.0
                result['direction_prob'] = 0.65
                result['confidence'] = 0.6
            elif blue_d >= 100:
                result['pred_return'] = 1.5
                result['direction_prob'] = 0.58
                result['confidence'] = 0.5
            else:
                result['pred_return'] = 0.5
                result['direction_prob'] = 0.52
                result['confidence'] = 0.3
            
            return result
        
        try:
            from ml.features.feature_calculator import FeatureCalculator
            
            calc = FeatureCalculator()
            blue_signals = {
                'blue_daily': signal.get('blue_daily', 0),
                'blue_weekly': signal.get('blue_weekly', 0),
                'blue_monthly': signal.get('blue_monthly', 0),
                'is_heima': signal.get('is_heima', 0),
                'is_juedi': 0
            }
            
            features = calc.get_latest_features(history, blue_signals)
            if not features:
                return result
            
            # 构建特征向量
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e6, 1e6)
            
            # 预测
            pred_return = float(self.model.predict(X)[0])
            
            # 方向概率 (基于预测值的sigmoid)
            direction_prob = 1 / (1 + np.exp(-pred_return / 2))
            
            # 置信度 (基于特征重要性和数据质量)
            # 简化: 如果预测值极端，置信度降低
            confidence = 0.7 if abs(pred_return) < 10 else 0.4
            
            result['pred_return'] = pred_return
            result['direction_prob'] = direction_prob
            result['confidence'] = confidence
            
        except Exception as e:
            pass
        
        return result
    
    def _validate_signals(self, signal: pd.Series, history: pd.DataFrame) -> Dict:
        """验证信号"""
        confirmed = []
        warnings = []
        score = 0
        
        blue_d = float(signal.get('blue_daily', 0))
        blue_w = float(signal.get('blue_weekly', 0))
        blue_m = float(signal.get('blue_monthly', 0))
        is_heima = bool(signal.get('is_heima', 0))
        
        adx = 0
        rsi = 50
        volume_ratio = 1.0
        
        # 1. BLUE 信号验证
        if blue_d >= 100:
            confirmed.append("✓ 日线BLUE强势")
            score += 1
        
        if blue_d >= 100 and blue_w >= 80:
            confirmed.append("✓ 日周BLUE共振")
            score += 1
        
        if blue_d >= 100 and blue_w >= 80 and blue_m >= 60:
            confirmed.append("✓ 日周月三线共振")
            score += 1
        
        if is_heima:
            confirmed.append("✓ 黑马信号")
            score += 1
        
        # 2. 技术指标验证 (需要历史数据)
        if not history.empty and len(history) >= 20:
            close_col = 'Close' if 'Close' in history.columns else 'close'
            vol_col = 'Volume' if 'Volume' in history.columns else 'volume'
            high_col = 'High' if 'High' in history.columns else 'high'
            low_col = 'Low' if 'Low' in history.columns else 'low'
            
            close = history[close_col].values
            
            # RSI
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi_values = 100 - (100 / (1 + rs))
            rsi = float(rsi_values.iloc[-1]) if not rsi_values.empty else 50
            
            if 30 < rsi < 70:
                confirmed.append("✓ RSI健康区间")
            elif rsi > 80:
                warnings.append("⚠ RSI超买")
            elif rsi < 20:
                confirmed.append("✓ RSI超卖反弹")
            
            # 成交量
            if vol_col in history.columns:
                vol = history[vol_col].values
                recent_vol = np.mean(vol[-5:])
                avg_vol = np.mean(vol[-20:])
                volume_ratio = recent_vol / (avg_vol + 1) if avg_vol > 0 else 1.0
                
                if volume_ratio > 1.5:
                    confirmed.append("✓ 放量突破")
                    score += 1
                elif volume_ratio < 0.5:
                    warnings.append("⚠ 成交萎缩")
            
            # ADX (趋势强度)
            if high_col in history.columns and low_col in history.columns:
                high = history[high_col].values
                low = history[low_col].values
                
                # 简化ADX计算
                tr = np.maximum(high[1:] - low[1:], 
                               np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1]))
                atr = pd.Series(tr).rolling(14).mean().iloc[-1]
                
                dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                                  np.maximum(high[1:] - high[:-1], 0), 0)
                dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                                   np.maximum(low[:-1] - low[1:], 0), 0)
                
                di_plus = pd.Series(dm_plus).rolling(14).mean() / (atr + 1e-10) * 100
                di_minus = pd.Series(dm_minus).rolling(14).mean() / (atr + 1e-10) * 100
                
                dx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
                adx = float(pd.Series(dx).rolling(14).mean().iloc[-1]) if len(dx) >= 14 else 25
                
                if adx > 25:
                    confirmed.append("✓ 趋势明确")
                    score += 1
            
            # MACD
            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            
            if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
                confirmed.append("✓ MACD金叉")
                score += 1
            elif macd.iloc[-1] > 0 and macd.iloc[-1] > macd.iloc[-2]:
                confirmed.append("✓ MACD向上")
        
        return {
            'confirmed': confirmed,
            'warnings': warnings,
            'score': min(score, 5),
            'adx': adx,
            'rsi': rsi,
            'volume_ratio': volume_ratio
        }
    
    def _assess_risk(self, pick: StockPick, history: pd.DataFrame) -> Dict:
        """风险评估"""
        price = pick.price
        
        # 默认止损止盈
        default_stop_pct = -5.0
        default_target_pct = 8.0
        
        if not history.empty and len(history) >= 20:
            close_col = 'Close' if 'Close' in history.columns else 'close'
            low_col = 'Low' if 'Low' in history.columns else 'low'
            high_col = 'High' if 'High' in history.columns else 'high'
            
            # 基于ATR的止损
            if high_col in history.columns and low_col in history.columns:
                close = history[close_col].values
                high = history[high_col].values
                low = history[low_col].values
                
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
                atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
                atr_pct = atr / price * 100
                
                # 止损 = 2倍ATR
                default_stop_pct = max(-2 * atr_pct, -8.0)  # 最大8%止损
                
                # 基于支撑位的止损
                recent_low = np.min(low[-20:])
                support_stop_pct = (recent_low - price) / price * 100
                
                # 取两者中更严格的
                if support_stop_pct > default_stop_pct:
                    default_stop_pct = support_stop_pct
            
            # 基于阻力位的目标价
            if high_col in history.columns:
                recent_high = np.max(history[high_col].values[-60:])
                resistance_target_pct = (recent_high - price) / price * 100
                
                if resistance_target_pct > 3:  # 至少3%空间
                    default_target_pct = min(resistance_target_pct, 15.0)  # 最大15%
        
        # 根据信号强度调整
        if pick.signal_score >= 4:
            default_target_pct *= 1.2  # 强信号提高目标
        
        stop_loss_price = price * (1 + default_stop_pct / 100)
        target_price = price * (1 + default_target_pct / 100)
        
        # 风险收益比
        risk = abs(default_stop_pct)
        reward = default_target_pct
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # 仓位建议 (基于Kelly公式简化版)
        # 凯利比例 = (bp - q) / b
        # 简化: 置信度越高、风险收益比越好，仓位越大
        win_prob = pick.pred_direction_prob
        if risk_reward_ratio >= 2 and win_prob >= 0.55:
            position_pct = 15.0  # 高置信度大仓位
        elif risk_reward_ratio >= 1.5 and win_prob >= 0.52:
            position_pct = 10.0
        else:
            position_pct = 5.0  # 保守仓位
        
        return {
            'stop_loss_price': round(stop_loss_price, 2),
            'stop_loss_pct': round(default_stop_pct, 1),
            'target_price': round(target_price, 2),
            'target_pct': round(default_target_pct, 1),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_pct': position_pct
        }
    
    def _calculate_overall_score(self, pick: StockPick) -> float:
        """计算综合评分 (0-100)"""
        score = 0.0
        
        # 1. ML预测分 (30%)
        # 预测收益越高越好，但要有置信度
        pred_score = min(pick.pred_return_5d * 5, 30)  # 6%收益=30分
        pred_score *= pick.ml_confidence  # 乘以置信度
        score += max(pred_score, 0)
        
        # 2. 信号验证分 (30%)
        signal_score = pick.signal_score * 6  # 5分=30
        score += signal_score
        
        # 3. 方向概率分 (20%)
        direction_score = (pick.pred_direction_prob - 0.5) * 200  # 0.6 = 20分
        score += max(direction_score, 0)
        
        # 4. 风险收益分 (20%)
        rr_score = min(pick.risk_reward_ratio * 5, 20)  # 4:1 = 20分
        score += rr_score
        
        return round(min(score, 100), 1)
    
    def _get_star_rating(self, score: float) -> int:
        """转换为星级 (1-5)"""
        if score >= 80:
            return 5
        elif score >= 65:
            return 4
        elif score >= 50:
            return 3
        elif score >= 35:
            return 2
        else:
            return 1
    
    def format_pick_summary(self, pick: StockPick) -> str:
        """格式化选股摘要"""
        stars = "⭐" * pick.star_rating
        
        # 理由
        reasons = pick.signals_confirmed[:3]  # 最多显示3个
        reasons_str = " | ".join(reasons) if reasons else "综合分析"
        
        return f"""
**{pick.symbol}** {pick.name} {stars} ({pick.overall_score:.0f}分)

📊 **预测**: {pick.pred_return_5d:+.1f}% (5天) | 上涨概率 {pick.pred_direction_prob:.0%}

📝 **理由**: {reasons_str}

🎯 **交易计划**:
- 入场: ${pick.price:.2f}
- 止损: ${pick.stop_loss_price:.2f} ({pick.stop_loss_pct:+.1f}%)
- 目标: ${pick.target_price:.2f} (+{pick.target_pct:.1f}%)
- 风险收益比: 1:{pick.risk_reward_ratio:.1f}
- 建议仓位: {pick.suggested_position_pct:.0f}%

⚠️ **注意**: {', '.join(pick.signals_warning) if pick.signals_warning else '无'}
"""


def get_todays_picks(market: str = 'US', max_picks: int = 5) -> List[StockPick]:
    """
    获取今日推荐
    
    便捷函数，直接从数据库获取信号并分析
    """
    import sys
    from pathlib import Path
    
    # 添加路径
    v3_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(v3_dir))
    
    from db.database import get_connection
    from db.stock_history import get_stock_history
    
    # 获取最新信号
    conn = get_connection()
    query = """
        SELECT DISTINCT symbol, scan_date, price, 
               COALESCE(blue_daily, 0) as blue_daily,
               COALESCE(blue_weekly, 0) as blue_weekly,
               COALESCE(blue_monthly, 0) as blue_monthly,
               COALESCE(is_heima, 0) as is_heima,
               company_name
        FROM scan_results
        WHERE market = ?
        ORDER BY scan_date DESC
        LIMIT 100
    """
    signals_df = pd.read_sql_query(query, conn, params=(market,))
    conn.close()
    
    if signals_df.empty:
        return []
    
    # 只取最新一天
    latest_date = signals_df['scan_date'].iloc[0]
    today_signals = signals_df[signals_df['scan_date'] == latest_date]
    
    # 获取价格历史
    price_history = {}
    for symbol in today_signals['symbol'].unique():
        history = get_stock_history(symbol, market, days=100)
        if not history.empty:
            price_history[symbol] = history
    
    # 智能选股
    picker = SmartPicker(market=market)
    return picker.pick(today_signals, price_history, max_picks=max_picks)


# === 测试 ===
if __name__ == "__main__":
    print("=== Smart Picker 测试 ===\n")
    
    # 模拟信号数据
    test_signals = pd.DataFrame([
        {'symbol': 'AAPL', 'price': 185.0, 'blue_daily': 125, 'blue_weekly': 110, 'blue_monthly': 90, 'is_heima': 1, 'company_name': 'Apple Inc'},
        {'symbol': 'MSFT', 'price': 420.0, 'blue_daily': 108, 'blue_weekly': 95, 'blue_monthly': 80, 'is_heima': 0, 'company_name': 'Microsoft'},
        {'symbol': 'NVDA', 'price': 880.0, 'blue_daily': 85, 'blue_weekly': 70, 'blue_monthly': 60, 'is_heima': 0, 'company_name': 'NVIDIA'},
        {'symbol': 'AMD', 'price': 155.0, 'blue_daily': 115, 'blue_weekly': 100, 'blue_monthly': 85, 'is_heima': 1, 'company_name': 'AMD'},
    ])
    
    picker = SmartPicker(market='US')
    
    # 模拟空历史 (测试规则模式)
    picks = picker.pick(test_signals, {}, max_picks=3)
    
    print(f"推荐 {len(picks)} 只股票:\n")
    for pick in picks:
        print(picker.format_pick_summary(pick))
        print("-" * 50)
