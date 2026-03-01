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
    
    # 排序模型分数
    rank_score_short: float = 0.0   # 短线排名分
    rank_score_medium: float = 0.0  # 中线排名分
    rank_score_long: float = 0.0    # 长线排名分
    
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
    is_trade_candidate: bool = False
    trade_block_reason: str = ""
    selected_rank_score: float = 0.0
    
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
            'rank_score_short': self.rank_score_short,
            'rank_score_medium': self.rank_score_medium,
            'rank_score_long': self.rank_score_long,
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
            'is_trade_candidate': self.is_trade_candidate,
            'trade_block_reason': self.trade_block_reason,
            'selected_rank_score': self.selected_rank_score,
            'blue_daily': self.blue_daily,
            'blue_weekly': self.blue_weekly,
            'blue_monthly': self.blue_monthly,
            'adx': self.adx,
            'rsi': self.rsi,
            'volume_ratio': self.volume_ratio,
        }


class SmartPicker:
    """智能选股器"""
    
    # 价格分层阈值 (与 pipeline.py 保持一致)
    PENNY_THRESHOLD = {'US': 5.0, 'CN': 3.0}
    
    def __init__(self, market: str = 'US', horizon: str = 'short'):
        """
        Args:
            market: 市场 ('US' or 'CN')
            horizon: 交易周期 ('short', 'medium', 'long')
        """
        self.market = market
        self.horizon = horizon
        self.model_dir = Path(__file__).parent / "saved_models" / f"v2_{market.lower()}"
        self.penny_model_dir = Path(__file__).parent / "saved_models" / f"v2_{market.lower()}_penny"
        self.ranker_meta = {}
        self.rank_weights = self._default_rank_weights()
        self.risk_profile = {}
        
        # Standard 模型
        self.return_model = None  # XGBoost fallback
        self.feature_names = []
        self.ranker_models = {}
        
        # MMoE 模型 (优先)
        self.mmoe_model = None
        self.penny_mmoe_model = None
        
        # Penny 模型 (低价股专用)
        self.penny_return_model = None
        self.penny_feature_names = []
        self.penny_ranker_models = {}
        
        # LightGBM Ensemble (Phase 3)
        self.lgb_model = None
        self.ensemble = None
        
        self._load_models()
        self._load_penny_models()
        self.rank_weights = self._compute_dynamic_rank_weights()
        self._load_risk_profile()

    def _load_risk_profile(self):
        """加载全局风控参数"""
        try:
            from risk.trading_profile import load_trading_profile
            self.risk_profile = load_trading_profile()
        except Exception:
            self.risk_profile = {}

    def _default_rank_weights(self) -> Dict[str, float]:
        """按当前交易偏好给出保守默认权重"""
        if self.horizon == 'medium':
            return {'short': 0.25, 'medium': 0.55, 'long': 0.20}
        if self.horizon == 'long':
            return {'short': 0.20, 'medium': 0.30, 'long': 0.50}
        return {'short': 0.55, 'medium': 0.30, 'long': 0.15}

    def _compute_dynamic_rank_weights(self) -> Dict[str, float]:
        """
        基于 ranker 历史表现动态配权 short/medium/long。
        使用 top10_avg_return + ndcg@10 + 样本规模三者融合。
        """
        metrics = (self.ranker_meta or {}).get('metrics', {})
        if not metrics:
            return self._default_rank_weights()

        raw = {}
        for horizon in ['short', 'medium', 'long']:
            m = metrics.get(horizon)
            if not isinstance(m, dict):
                raw[horizon] = -5.0
                continue

            top10 = float(m.get('top10_avg_return', 0.0) or 0.0)
            ndcg = float(m.get('ndcg@10', 0.0) or 0.0)
            sample_n = float(
                m.get('test_samples', 0)
                or m.get('train_samples', 0)
                or 0
            )

            # OOS收益为主，排序质量次之；样本不足则降权
            performance = (0.75 * top10) + (0.25 * ((ndcg - 0.5) * 10.0))
            sample_factor = min(max(sample_n / 300.0, 0.30), 1.00)
            raw[horizon] = performance * sample_factor

        values = np.array([raw[h] for h in ['short', 'medium', 'long']], dtype=float)
        values = np.clip(values, -8.0, 8.0)

        exp_vals = np.exp(values - np.max(values))
        if not np.isfinite(exp_vals).all() or exp_vals.sum() <= 0:
            return self._default_rank_weights()

        # 保底每个周期 10%，避免单周期过拟合独占
        min_w = 0.10
        soft = exp_vals / exp_vals.sum()
        alloc = min_w + (1 - 3 * min_w) * soft

        return {
            'short': float(alloc[0]),
            'medium': float(alloc[1]),
            'long': float(alloc[2]),
        }
    
    def _load_models(self):
        """加载所有ML模型 (MMoE 优先, XGBoost fallback)"""
        if not ML_AVAILABLE:
            return
        
        # 1. 尝试加载 MMoE 模型 (优先)
        mmoe_dir = self.model_dir.parent / f"v2_{self.market.lower()}_mmoe"
        if mmoe_dir.exists() and (mmoe_dir / 'mmoe_model.pt').exists():
            try:
                from ml.models.mmoe import MMoEPredictor
                self.mmoe_model = MMoEPredictor(device='cpu')
                self.mmoe_model.load(str(mmoe_dir))
                self.feature_names = self.mmoe_model.feature_names
                print(f"✓ MMoE 模型已加载 ({len(self.mmoe_model.task_defs)} tasks)")
            except Exception as e:
                print(f"MMoE 加载失败, 回退到 XGBoost: {e}")
                self.mmoe_model = None
        
        # 2. XGBoost fallback (或 MMoE 不可用)
        if self.mmoe_model is None:
            try:
                model_path = self.model_dir / "return_5d.joblib"
                if model_path.exists():
                    self.return_model = joblib.load(model_path)
                    feature_path = self.model_dir / "feature_names.json"
                    if feature_path.exists():
                        with open(feature_path) as f:
                            self.feature_names = json.load(f)
                    print(f"✓ XGBoost 收益预测模型已加载")
            except Exception as e:
                print(f"收益预测模型加载失败: {e}")
        
        # 3. 排序模型 (SignalRanker) — 仅 XGBoost 模式需要
        for horizon in ['short', 'medium', 'long']:
            try:
                ranker_path = self.model_dir / f"ranker_{horizon}.joblib"
                if ranker_path.exists():
                    self.ranker_models[horizon] = joblib.load(ranker_path)
                    print(f"✓ 排序模型 ({horizon}) 已加载")
            except Exception as e:
                print(f"排序模型 ({horizon}) 加载失败: {e}")
        
        # 加载排序模型元数据
        try:
            ranker_meta_path = self.model_dir / "ranker_meta.json"
            if ranker_meta_path.exists():
                with open(ranker_meta_path) as f:
                    self.ranker_meta = json.load(f)
        except:
            self.ranker_meta = {}
        
        # 4. LightGBM Ensemble (Phase 3)
        try:
            from ml.ensemble_predictor import LGBPredictor, EnsemblePredictor
            lgb_pred = LGBPredictor(market=self.market)
            if lgb_pred.load():
                self.lgb_model = lgb_pred
                self.ensemble = EnsemblePredictor()
                print(f"✓ LGB Ensemble 模型已加载 ({len(lgb_pred.models)} models)")
        except Exception:
            pass
    
    def _load_penny_models(self):
        """加载低价股专用模型"""
        if not ML_AVAILABLE or not self.penny_model_dir.exists():
            return
        
        # MMoE penny
        penny_mmoe_dir = self.model_dir.parent / f"v2_{self.market.lower()}_penny_mmoe"
        if penny_mmoe_dir.exists() and (penny_mmoe_dir / 'mmoe_model.pt').exists():
            try:
                from ml.models.mmoe import MMoEPredictor
                self.penny_mmoe_model = MMoEPredictor(device='cpu')
                self.penny_mmoe_model.load(str(penny_mmoe_dir))
                self.penny_feature_names = self.penny_mmoe_model.feature_names
                print(f"✓ Penny MMoE 模型已加载")
            except Exception as e:
                self.penny_mmoe_model = None
        
        # XGBoost penny fallback
        if self.penny_mmoe_model is None:
            try:
                model_path = self.penny_model_dir / "return_5d.joblib"
                if model_path.exists():
                    self.penny_return_model = joblib.load(model_path)
                    feat_path = self.penny_model_dir / "feature_names.json"
                    if feat_path.exists():
                        with open(feat_path) as f:
                            self.penny_feature_names = json.load(f)
                    print(f"✓ Penny XGBoost 模型已加载")
            except Exception as e:
                print(f"Penny 模型加载失败: {e}")
        
        for horizon in ['short', 'medium', 'long']:
            try:
                rp = self.penny_model_dir / f"ranker_{horizon}.joblib"
                if rp.exists():
                    self.penny_ranker_models[horizon] = joblib.load(rp)
            except:
                pass
    
    def _is_penny(self, price: float) -> bool:
        """判断是否为低价股"""
        threshold = self.PENNY_THRESHOLD.get(self.market, 5.0)
        return price < threshold
    
    def _get_models_for_price(self, price: float):
        """根据价格自动选择模型 (mmoe_model, return_model, feature_names, ranker_models)"""
        if self._is_penny(price):
            return (self.penny_mmoe_model, self.penny_return_model,
                    self.penny_feature_names or self.feature_names,
                    self.penny_ranker_models or self.ranker_models)
        return (self.mmoe_model, self.return_model,
                self.feature_names, self.ranker_models)
    
    @property
    def model(self):
        """兼容旧代码"""
        return self.return_model
    
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

        # 只返回可交易候选，避免“第一名却是低质量票”
        tradable_picks = [p for p in picks if p.is_trade_candidate]
        tradable_picks.sort(key=lambda x: x.overall_score, reverse=True)

        # 返回 Top N
        return tradable_picks[:max_picks]
    
    def _analyze_stock(self, signal: pd.Series, history: pd.DataFrame, skip_prefilter: bool = False) -> Optional[StockPick]:
        """分析单只股票
        
        Args:
            signal: 信号数据
            history: 历史价格数据
            skip_prefilter: 跳过预过滤 (用于个股详情页，用户已主动选择该股票)
        """
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
        if not skip_prefilter and not self._pass_prefilter(signal, history):
            return None
        
        # === Stage 2: ML Prediction (MMoE 多任务 / XGBoost) ===
        ml_result = self._ml_predict(signal, history)
        pick.pred_return_5d = ml_result.get('pred_return', 0)
        pick.pred_direction_prob = ml_result.get('direction_prob', 0.5)
        pick.ml_confidence = ml_result.get('confidence', 0)
        pick.pred_return_20d = ml_result.get('pred_return_20d', 0)
        pick.pred_max_dd = ml_result.get('pred_max_dd', 0)
        pick.pred_rank_score = ml_result.get('pred_rank', 0)
        
        # === Stage 2.5: Ranker Scoring (排序模型) ===
        rank_result = self._rank_score(signal, history)
        pick.rank_score_short = rank_result.get('short', 0)
        pick.rank_score_medium = rank_result.get('medium', 0)
        pick.rank_score_long = rank_result.get('long', 0)
        
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
        pick.selected_rank_score = self._get_horizon_rank_score(pick)
        pick.is_trade_candidate, pick.trade_block_reason = self._is_trade_candidate(pick)

        return pick

    def _get_horizon_rank_score(self, pick: StockPick) -> float:
        if self.horizon == 'medium':
            return float(pick.rank_score_medium or 0.0)
        if self.horizon == 'long':
            return float(pick.rank_score_long or 0.0)
        return float(pick.rank_score_short or 0.0)

    def _is_trade_candidate(self, pick: StockPick) -> Tuple[bool, str]:
        """
        交易硬门槛: 低质量信号直接拦截，不进入“今日精选”。
        可由风控配置覆盖。
        """
        cfg = self.risk_profile or {}
        min_overall = float(cfg.get("min_trade_overall_score", 50.0))
        min_prob = float(cfg.get("min_trade_direction_prob", 0.52))
        min_pred = float(cfg.get("min_trade_pred_return_pct", 0.0))
        min_rank = float(cfg.get("min_trade_rank_score", 10.0))
        min_rr = float(cfg.get("min_trade_rr", 1.2))

        blockers = []
        if float(pick.overall_score or 0.0) < min_overall:
            blockers.append(f"综合评分<{min_overall:.0f}")
        if float(pick.pred_direction_prob or 0.0) < min_prob:
            blockers.append(f"上涨概率<{min_prob:.0%}")
        if float(pick.pred_return_5d or 0.0) < min_pred:
            blockers.append(f"预测收益<{min_pred:.1f}%")
        if float(self._get_horizon_rank_score(pick) or 0.0) < min_rank:
            blockers.append(f"排序得分<{min_rank:.0f}")
        if float(pick.risk_reward_ratio or 0.0) < min_rr:
            blockers.append(f"风险收益比<1:{min_rr:.1f}")

        if blockers:
            return False, "；".join(blockers)
        return True, ""
    
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
        """ML预测 (MMoE 优先, XGBoost fallback)"""
        result = {
            'pred_return': 0.0,
            'pred_return_20d': 0.0,
            'direction_prob': 0.5,
            'confidence': 0.0,
            'pred_max_dd': 0.0,
            'pred_rank': 0.0,
        }
        
        price = float(signal.get('price', 0))
        mmoe, ret_model, feat_names, _ = self._get_models_for_price(price)
        
        has_model = mmoe is not None or ret_model is not None
        
        if not has_model or history.empty or len(history) < 60:
            # 无模型或数据不足，使用规则估计
            blue_d = float(signal.get('blue_daily', 0))
            blue_w = float(signal.get('blue_weekly', 0))
            
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
                'is_juedi': 0,
                'symbol': signal.get('symbol', ''),
                'market': self.market,
            }
            
            features = calc.get_latest_features(history, blue_signals)
            if not features:
                return result
            
            # 追加高级技术因子 (Phase 1: AdvancedFeatures)
            try:
                from ml.advanced_features import AdvancedFeatureEngineer
                adv_eng = AdvancedFeatureEngineer()
                
                hist_adv = history.copy()
                if not isinstance(hist_adv.index, pd.DatetimeIndex):
                    if 'Date' in hist_adv.columns:
                        hist_adv = hist_adv.set_index('Date')
                    elif 'date' in hist_adv.columns:
                        hist_adv = hist_adv.set_index('date')
                    hist_adv.index = pd.to_datetime(hist_adv.index)
                
                col_map = {}
                for c in hist_adv.columns:
                    cl = c.lower()
                    if cl == 'close': col_map[c] = 'Close'
                    elif cl == 'open': col_map[c] = 'Open'
                    elif cl == 'high': col_map[c] = 'High'
                    elif cl == 'low': col_map[c] = 'Low'
                    elif cl == 'volume': col_map[c] = 'Volume'
                if col_map:
                    hist_adv = hist_adv.rename(columns=col_map)
                
                if all(c in hist_adv.columns for c in ['Close', 'High', 'Low', 'Volume']):
                    adv_df = adv_eng.transform(hist_adv)
                    if len(adv_df) > 0:
                        adv_row = adv_df.iloc[-1]
                        skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                                    'open', 'high', 'low', 'close', 'volume'}
                        for col in adv_row.index:
                            if col not in skip_cols:
                                val = adv_row[col]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    features[f'adv_{col}'] = float(val)
            except Exception:
                pass
            
            # 追加 Alpha158 因子 (Phase 2)
            try:
                from ml.alpha_factors import Alpha158Factors
                a158 = Alpha158Factors()
                # 复用 hist_adv (已经标准化好了)
                a158_df = a158.compute(hist_adv)
                if len(a158_df) > 0:
                    a158_row = a158_df.iloc[-1]
                    skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                    for col in a158_row.index:
                        if col not in skip_cols:
                            val = a158_row[col]
                            if isinstance(val, (int, float, np.integer, np.floating)):
                                features[f'a158_{col}'] = float(val)
            except Exception:
                pass
            
            # 追加蔡森图表特征 (Phase 3)
            try:
                from ml.caisen_features import compute_caisen_features
                cs_df = compute_caisen_features(hist_adv)
                if cs_df is not None and len(cs_df) > 0:
                    cs_row = cs_df.iloc[-1]
                    for col in cs_row.index:
                        if col not in features:
                            val = cs_row[col]
                            if isinstance(val, (int, float, np.integer, np.floating)):
                                features[col] = float(val)
            except Exception:
                pass
            
            X = np.array([[features.get(f, 0) for f in feat_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e6, 1e6)
            
            if mmoe is not None:
                # === MMoE 多任务预测 ===
                preds = mmoe.predict(X)
                
                pred_5d = float(preds.get('return_5d', [0])[0])
                pred_20d = float(preds.get('return_20d', [0])[0])
                dir_prob = float(preds.get('direction', [0.5])[0])
                max_dd = float(preds.get('max_dd', [0])[0])
                rank_s = float(preds.get('rank_score', [0])[0])
                
                # === Phase 3: LGB Ensemble ===
                if self.lgb_model is not None and self.ensemble is not None:
                    try:
                        lgb_preds = self.lgb_model.predict(X)
                        lgb_dir = float(lgb_preds.get('direction', [0.5])[0])
                        lgb_r5 = float(lgb_preds.get('return_5d', [0])[0])
                        lgb_r20 = float(lgb_preds.get('return_20d', [0])[0])
                        
                        # ensemble 融合
                        dir_prob = self.ensemble.ensemble_direction(dir_prob, lgb_dir)
                        pred_5d = self.ensemble.ensemble_return(pred_5d, lgb_r5)
                        pred_20d = self.ensemble.ensemble_return(pred_20d, lgb_r20)
                        confidence = self.ensemble.ensemble_confidence(
                            float(preds.get('direction', [0.5])[0]), lgb_dir
                        )
                    except Exception:
                        confidence = min(abs(dir_prob - 0.5) * 2 + 0.3, 0.95)
                else:
                    confidence = min(abs(dir_prob - 0.5) * 2 + 0.3, 0.95)
                
                result['pred_return'] = pred_5d
                result['pred_return_20d'] = pred_20d
                result['direction_prob'] = np.clip(dir_prob, 0.01, 0.99)
                result['pred_max_dd'] = max_dd
                result['pred_rank'] = rank_s
                result['confidence'] = confidence
            else:
                # === XGBoost fallback ===
                pred_return = float(ret_model.predict(X)[0])
                direction_prob = 1 / (1 + np.exp(-pred_return / 2))
                confidence = 0.7 if abs(pred_return) < 10 else 0.4
                
                result['pred_return'] = pred_return
                result['direction_prob'] = direction_prob
                result['confidence'] = confidence
            
        except Exception as e:
            pass
        
        return result
    
    def _rank_score(self, signal: pd.Series, history: pd.DataFrame) -> Dict:
        """
        使用排序模型(Learning to Rank)计算排名分数
        
        排序模型综合考虑收益和风险，输出"赚钱概率"排名
        """
        result = {'short': 0.0, 'medium': 0.0, 'long': 0.0}
        
        price = float(signal.get('price', 0))
        _, _, feat_names, ranker_mods = self._get_models_for_price(price)
        
        if not ranker_mods or history.empty or len(history) < 60:
            # 无排序模型，使用规则估计
            blue_d = float(signal.get('blue_daily', 0))
            blue_w = float(signal.get('blue_weekly', 0))
            blue_m = float(signal.get('blue_monthly', 0))
            is_heima = bool(signal.get('is_heima', 0))
            
            # 规则: 信号强度 -> 排名分
            base_score = 0
            if blue_d >= 100: base_score += 30
            if blue_w >= 80: base_score += 20
            if blue_m >= 60: base_score += 10
            if is_heima: base_score += 20
            
            result['short'] = base_score
            result['medium'] = base_score * 0.9
            result['long'] = base_score * 0.8
            return result
        
        try:
            from ml.features.feature_calculator import FeatureCalculator
            
            calc = FeatureCalculator()
            blue_signals = {
                'blue_daily': signal.get('blue_daily', 0),
                'blue_weekly': signal.get('blue_weekly', 0),
                'blue_monthly': signal.get('blue_monthly', 0),
                'is_heima': signal.get('is_heima', 0),
                'is_juedi': 0,
                'symbol': signal.get('symbol', ''),
                'market': self.market,
            }
            
            features = calc.get_latest_features(history, blue_signals)
            if not features:
                return result
            
            # 追加高级技术因子 (Phase 1)
            try:
                from ml.advanced_features import AdvancedFeatureEngineer
                adv_eng = AdvancedFeatureEngineer()
                hist_adv = history.copy()
                if not isinstance(hist_adv.index, pd.DatetimeIndex):
                    if 'Date' in hist_adv.columns:
                        hist_adv = hist_adv.set_index('Date')
                    elif 'date' in hist_adv.columns:
                        hist_adv = hist_adv.set_index('date')
                    hist_adv.index = pd.to_datetime(hist_adv.index)
                col_map = {}
                for c in hist_adv.columns:
                    cl = c.lower()
                    if cl == 'close': col_map[c] = 'Close'
                    elif cl == 'open': col_map[c] = 'Open'
                    elif cl == 'high': col_map[c] = 'High'
                    elif cl == 'low': col_map[c] = 'Low'
                    elif cl == 'volume': col_map[c] = 'Volume'
                if col_map:
                    hist_adv = hist_adv.rename(columns=col_map)
                if all(c in hist_adv.columns for c in ['Close', 'High', 'Low', 'Volume']):
                    adv_df = adv_eng.transform(hist_adv)
                    if len(adv_df) > 0:
                        adv_row = adv_df.iloc[-1]
                        skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                        for col in adv_row.index:
                            if col not in skip_cols:
                                val = adv_row[col]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    features[f'adv_{col}'] = float(val)
            except Exception:
                pass
            
            # 追加 Alpha158 因子 (Phase 2)
            try:
                from ml.alpha_factors import Alpha158Factors
                a158 = Alpha158Factors()
                a158_df = a158.compute(hist_adv)
                if len(a158_df) > 0:
                    a158_row = a158_df.iloc[-1]
                    skip_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                    for col in a158_row.index:
                        if col not in skip_cols:
                            val = a158_row[col]
                            if isinstance(val, (int, float, np.integer, np.floating)):
                                features[f'a158_{col}'] = float(val)
            except Exception:
                pass
            
            # 构建特征向量
            X = np.array([[features.get(f, 0) for f in feat_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e6, 1e6)
            
            # 对每个周期使用排序模型
            for horizon in ['short', 'medium', 'long']:
                if horizon in ranker_mods:
                    try:
                        score = float(ranker_mods[horizon].predict(X)[0])
                        result[horizon] = score
                    except:
                        pass
            
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
        cfg = self.risk_profile or {}
        atr_mult = float(cfg.get("atr_stop_multiplier", 2.0))
        max_stop = float(cfg.get("max_stop_loss_pct", 8.0))
        target_cap = float(cfg.get("target_cap_pct", 15.0))
        strong_boost = float(cfg.get("strong_signal_target_boost", 1.2))
        rr_high = float(cfg.get("rr_high", 2.0))
        rr_mid = float(cfg.get("rr_mid", 1.5))
        prob_high = float(cfg.get("prob_high", 0.55))
        prob_mid = float(cfg.get("prob_mid", 0.52))
        pos_high = float(cfg.get("position_high_pct", 15.0))
        pos_mid = float(cfg.get("position_mid_pct", 10.0))
        pos_low = float(cfg.get("position_low_pct", 5.0))
        
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
                
                # 止损 = ATR倍数，且受最大止损约束
                default_stop_pct = max(-atr_mult * atr_pct, -max_stop)
                
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
                    default_target_pct = min(resistance_target_pct, target_cap)
        
        # 根据信号强度调整
        if pick.signal_score >= 4:
            default_target_pct *= strong_boost
        
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
        if risk_reward_ratio >= rr_high and win_prob >= prob_high:
            position_pct = pos_high
        elif risk_reward_ratio >= rr_mid and win_prob >= prob_mid:
            position_pct = pos_mid
        else:
            position_pct = pos_low
        
        return {
            'stop_loss_price': round(stop_loss_price, 2),
            'stop_loss_pct': round(default_stop_pct, 1),
            'target_price': round(target_price, 2),
            'target_pct': round(default_target_pct, 1),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_pct': position_pct
        }
    
    def _calculate_overall_score(self, pick: StockPick) -> float:
        """
        计算综合评分 (0-100)
        
        评分构成:
        - 排序模型分 (20%): Learning to Rank / MMoE rank
        - ML预测分 (20%): 收益预测 * 置信度
        - 信号验证分 (20%): BLUE/MACD/成交量等技术确认
        - 方向概率分 (25%): MMoE 方向预测概率
        - 风险收益分 (15%): 风险收益比 + 回撤预测
        """
        score = 0.0
        
        # 1. 排序模型分 (20%)
        # MMoE rank_score 优先，否则用 XGBoost ranker
        if hasattr(pick, 'pred_rank_score') and pick.pred_rank_score > 0:
            normalized_rank = min(pick.pred_rank_score, 1.0) * 20.0
        else:
            w = self.rank_weights
            rank_score = (
                w.get('short', 0.0) * pick.rank_score_short +
                w.get('medium', 0.0) * pick.rank_score_medium +
                w.get('long', 0.0) * pick.rank_score_long
            )
            normalized_rank = min(max(rank_score, 0.0), 100.0) / 100.0 * 20.0
        score += normalized_rank
        
        # 2. ML预测分 (20%)
        pred_score = min(pick.pred_return_5d * 3.33, 20)
        pred_score *= pick.ml_confidence
        score += max(pred_score, 0)
        
        # 3. 信号验证分 (20%)
        signal_score = pick.signal_score * 4  # 5分=20
        score += signal_score
        
        # 4. 方向概率分 (25%) — MMoE 的核心输出
        direction_score = (pick.pred_direction_prob - 0.5) * 250  # 0.6=25分
        score += max(min(direction_score, 25), 0)
        
        # 5. 风险收益分 (15%) — 加入回撤惩罚
        rr_score = min(pick.risk_reward_ratio * 3.75, 15)
        # MMoE 预测回撤越大越扣分
        if hasattr(pick, 'pred_max_dd') and pick.pred_max_dd < -5:
            dd_penalty = min(abs(pick.pred_max_dd) * 0.3, 5)
            rr_score = max(rr_score - dd_penalty, 0)
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
