
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class RankingSystem:
    """
    智能排序系统 (Hybrid Smart Ranker)
    整合 MMoE 预测、大师策略评分和舆情情绪评分，生成最终排序。
    """
    
    def __init__(self):
        # 权重配置 (可根据回测调整)
        self.weights = {
            'ml_technical': 0.5,   # 技术面 (MMoE 优先)
            'master_strategy': 0.3, # 大师策略共识
            'sentiment': 0.2       # 舆情情绪
        }
        self._picker = None
        self._picker_loaded = False
    
    def _get_picker(self):
        """惰性加载 SmartPicker (含 MMoE)"""
        if not self._picker_loaded:
            self._picker_loaded = True
            try:
                from ml.smart_picker import SmartPicker
                self._picker = SmartPicker(market='US', horizon='short')
                if self._picker.mmoe_model:
                    logger.info("RankingSystem: MMoE 模型已加载")
                else:
                    logger.info("RankingSystem: XGBoost fallback")
            except Exception as e:
                logger.warning(f"RankingSystem: SmartPicker 加载失败: {e}")
                self._picker = None
        return self._picker
    
    def _load_mmoe_cache(self, market: str = 'US', scan_date: str = None) -> Optional[Dict]:
        """尝试加载预计算的 MMoE 缓存（支持按日期加载）"""
        import json
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'saved_models', 'mmoe_cache')
        
        # 优先加载按日期的缓存
        candidates = []
        if scan_date:
            candidates.append(os.path.join(cache_dir, f'{market.lower()}_{scan_date}.json'))
        candidates.append(os.path.join(cache_dir, f'{market.lower()}_latest.json'))
        
        # fallback 路径
        if scan_date:
            candidates.append(os.path.join(os.getcwd(), 'ml', 'saved_models', 'mmoe_cache', f'{market.lower()}_{scan_date}.json'))
        candidates.append(os.path.join(os.getcwd(), 'ml', 'saved_models', 'mmoe_cache', f'{market.lower()}_latest.json'))
        
        cache_file = None
        for c in candidates:
            if os.path.exists(c):
                cache_file = c
                break
        
        if not cache_file:
            logger.warning(f"MMoE cache not found for {market} {scan_date or 'latest'}")
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            scores = cache.get('scores', {})
            if scores:
                logger.info(f"RankingSystem: 加载缓存 ({len(scores)} 只, date={cache.get('date', '?')}, file={os.path.basename(cache_file)})")
            return scores
        except Exception as e:
            logger.warning(f"RankingSystem: 缓存加载失败: {e}")
            return None
    
    def _batch_mmoe_predict(self, df: pd.DataFrame, scan_date: str = None) -> pd.DataFrame:
        """
        批量 MMoE 预测: 优先从缓存读取，否则实时计算
        
        返回 df 添加了:
          - mmoe_dir_prob: 方向概率
          - mmoe_return_5d: 5d 收益预测
          - mmoe_return_20d: 20d 收益预测
          - mmoe_max_dd: 最大回撤预测
        """
        # 需要的列映射
        ticker_col = 'Ticker' if 'Ticker' in df.columns else 'symbol'
        price_col = 'Price' if 'Price' in df.columns else 'price'
        
        if ticker_col not in df.columns or price_col not in df.columns:
            logger.warning(f"MMoE: missing columns. has={list(df.columns[:10])}, need={ticker_col},{price_col}")
            return df
        
        # === 优先从缓存读取 ===
        cache = self._load_mmoe_cache('US', scan_date=scan_date)
        if cache:
            # 调试: 显示 df 样本和 cache 样本
            sample_tickers = list(df[ticker_col].head(5))
            sample_cache = list(cache.keys())[:5]
            logger.info(f"MMoE: ticker_col={ticker_col}, df_sample={sample_tickers}, cache_sample={sample_cache}, cache_size={len(cache)}")
            
            df['mmoe_dir_prob'] = df[ticker_col].map(lambda s: cache.get(str(s).strip().upper(), {}).get('dir_prob', np.nan))
            df['mmoe_return_5d'] = df[ticker_col].map(lambda s: cache.get(str(s).strip().upper(), {}).get('return_5d', np.nan))
            df['mmoe_return_20d'] = df[ticker_col].map(lambda s: cache.get(str(s).strip().upper(), {}).get('return_20d', np.nan))
            df['mmoe_max_dd'] = df[ticker_col].map(lambda s: cache.get(str(s).strip().upper(), {}).get('max_dd', np.nan))
            df['mmoe_score'] = df[ticker_col].map(lambda s: cache.get(str(s).strip().upper(), {}).get('overall_score', np.nan))
            hit = df['mmoe_dir_prob'].notna().sum()
            logger.info(f"RankingSystem: 缓存命中 {hit}/{len(df)}")
            return df
        
        # === 没有缓存 → 实时计算 ===
        picker = self._get_picker()
        if picker is None:
            return df
        
        # 初始化新列
        df['mmoe_dir_prob'] = np.nan
        df['mmoe_return_5d'] = np.nan
        df['mmoe_return_20d'] = np.nan
        df['mmoe_max_dd'] = np.nan
        df['mmoe_score'] = np.nan
        
        try:
            from db.stock_history import get_stock_history
        except ImportError:
            return df
        
        success = 0
        for idx, row in df.iterrows():
            sym = str(row.get(ticker_col, '')).strip().upper()
            price = float(row.get(price_col, 0) or 0)
            if not sym or price <= 0:
                continue
            
            try:
                h = get_stock_history(sym, 'US', days=300)
                if h is None or h.empty or len(h) < 60:
                    continue
                
                if not isinstance(h.index, pd.DatetimeIndex):
                    if 'Date' in h.columns:
                        h = h.set_index('Date')
                    elif 'date' in h.columns:
                        h = h.set_index('date')
                    h.index = pd.to_datetime(h.index)
                
                # 构造信号
                sig = pd.Series({
                    'symbol': sym,
                    'price': price,
                    'blue_daily': float(row.get('Day BLUE', row.get('blue_daily', 0)) or 0),
                    'blue_weekly': float(row.get('Week BLUE', row.get('blue_weekly', 0)) or 0),
                    'blue_monthly': float(row.get('Month BLUE', row.get('blue_monthly', 0)) or 0),
                    'is_heima': 1 if row.get('黑马日') or row.get('heima_daily') else 0,
                })
                
                pick = picker._analyze_stock(sig, h, skip_prefilter=True)
                if pick:
                    df.at[idx, 'mmoe_dir_prob'] = pick.pred_direction_prob
                    df.at[idx, 'mmoe_return_5d'] = pick.pred_return_5d
                    df.at[idx, 'mmoe_return_20d'] = getattr(pick, 'pred_return_20d', 0) or 0
                    df.at[idx, 'mmoe_max_dd'] = getattr(pick, 'pred_max_dd', 0) or 0
                    df.at[idx, 'mmoe_score'] = pick.overall_score
                    success += 1
            except Exception as e:
                continue
        
        logger.info(f"RankingSystem: MMoE 预测完成 {success}/{len(df)}")
        return df
    
    def calculate_integrated_score(self, 
                                 df: pd.DataFrame, 
                                 master_results: Optional[Dict] = None,
                                 sentiment_results: Optional[Dict] = None,
                                 scan_date: str = None) -> pd.DataFrame:
        """
        计算综合排序分数
        
        Args:
            df: 股票基础数据 DataFrame
            master_results: 大师分析结果字典 {symbol: analysis_summary}
            sentiment_results: 舆情分析结果字典 {symbol: sentiment_report}
            
        Returns:
            df: 添加了 'Rank_Score' 和 MMoE 列的 DataFrame
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 0. 批量 MMoE 预测 (如果模型可用)
        df = self._batch_mmoe_predict(df, scan_date=scan_date)
        has_mmoe = df['mmoe_dir_prob'].notna().any() if 'mmoe_dir_prob' in df.columns else False
        
        # 1. 技术面评分
        if has_mmoe:
            # MMoE 方向概率 → 0~100 分
            mmoe_score = df['mmoe_dir_prob'].fillna(0.5) * 100
            heuristic_score = self._calculate_heuristic_tech_score(df)
            # 70% MMoE + 30% 启发式
            df['score_tech'] = mmoe_score * 0.7 + heuristic_score * 0.3
        elif 'Probability' in df.columns:
            df['score_tech'] = df['Probability'] * 100
        else:
            df['score_tech'] = self._calculate_heuristic_tech_score(df)
            
        # 2. 大师策略评分
        df['score_master'] = 50.0
        if master_results:
            ticker_col = 'Ticker' if 'Ticker' in df.columns else 'symbol'
            df['score_master'] = df[ticker_col].apply(
                lambda x: self._quantify_master_result(master_results.get(x))
            )
            
        # 3. 舆情评分
        df['score_sentiment'] = 50.0
        if sentiment_results:
            ticker_col = 'Ticker' if 'Ticker' in df.columns else 'symbol'
            df['score_sentiment'] = df[ticker_col].apply(
                lambda x: self._quantify_sentiment_result(sentiment_results.get(x))
            )
            
        # 4. 综合加权
        base_score = (
            df['score_tech'] * self.weights['ml_technical'] +
            df['score_master'] * self.weights['master_strategy'] +
            df['score_sentiment'] * self.weights['sentiment']
        )
        
        # Alpha Bonus
        bonus = pd.Series(0.0, index=df.index)
        all_good = (df['score_tech'] > 60) & (df['score_master'] > 55) & (df['score_sentiment'] > 50)
        bonus[all_good] += 10.0
        bonus[df['score_master'] > 80] += 5.0
        bonus[(df['score_sentiment'] > 80) & (df['score_tech'] > 50)] += 5.0
        
        # MMoE 额外奖惩
        if has_mmoe:
            # 方向概率 > 60%: 额外奖励
            bonus[df['mmoe_dir_prob'].fillna(0) > 0.6] += 8.0
            # 预测回撤 < -8%: 扣分
            bonus[df['mmoe_max_dd'].fillna(0) < -8] -= 5.0
        
        df['Rank_Score'] = (base_score + bonus).clip(0, 100)
        
        # 排序
        df = df.sort_values('Rank_Score', ascending=False)
        
        # 保存上下文
        self._save_ranking_context(df)
        
        return df
    
    def _save_ranking_context(self, df: pd.DataFrame):
        """保存当天的排序快照，用于构建 Learning to Rank 数据集"""
        import os
        import json
        from datetime import datetime
        
        try:
            # 只保存前 50 以及必要的特征列
            top_df = df.head(50).copy()
            
            # 特征列
            feature_cols = ['score_tech', 'score_master', 'score_sentiment', 
                           'Day BLUE', 'Week BLUE', 'ADX', 'Turnover', 'Profit_Ratio']
            cols_to_save = [c for c in feature_cols if c in top_df.columns]
            
            if not cols_to_save:
                return
                
            # 添加元数据
            data = {
                'timestamp': datetime.now().isoformat(),
                'items': []
            }
            
            for _, row in top_df.iterrows():
                item = {
                    'symbol': row.get('Ticker', row.get('symbol', 'Unknown')),
                    'rank_score': row.get('Rank_Score', 0),
                    'features': {col: float(row[col]) if pd.notnull(row[col]) else 0.0 for col in cols_to_save}
                }
                data['items'].append(item)
            
            # 保存到 logs 目录
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.agent', 'ranking_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            filename = f"rank_ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(log_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save ranking context: {e}")

    def _calculate_heuristic_tech_score(self, df: pd.DataFrame) -> pd.Series:
        """基于简单规则的技术面打分 (当 ML 模型不可用时)"""
        scores = pd.Series(50.0, index=df.index)
        
        # BLUE: 50以上加分, 100以上大加分
        if 'Day BLUE' in df.columns:
            scores += (df['Day BLUE'].fillna(0) / 200.0) * 30
            
        if 'Week BLUE' in df.columns:
             scores += (df['Week BLUE'].fillna(0) / 200.0) * 20
            
        if 'ADX' in df.columns:
            # ADX: 趋势越强分越高
            scores += (df['ADX'].fillna(0) / 100.0) * 15
            
        if 'Profit_Ratio' in df.columns:
            # 获利盘: 越高越好
            scores += (df['Profit_Ratio'].fillna(0) * 15)
            
        # 筹码集中度奖励
        if '筹码形态' in df.columns:
            scores[df['筹码形态'] == '🔥'] += 10
            scores[df['筹码形态'] == '📍'] += 5
            
        return scores.clip(0, 100)
    
    def _quantify_master_result(self, summary) -> float:
        """量化大师分析结果"""
        if not summary:
            return 50.0
        
        # 如果是简单的字符串
        if isinstance(summary, str):
            if "买入" in summary or "积极" in summary: return 80.0
            if "卖出" in summary or "回避" in summary: return 20.0
            return 50.0
            
        # 如果是字典
        if isinstance(summary, dict):
            buy = summary.get('buy_votes', 0)
            sell = summary.get('sell_votes', 0)
            hold = summary.get('hold_votes', 0)
            is_best = summary.get('best_opportunity', None)
            
            # 基础分
            score = 50 + (buy * 12) - (sell * 15) + (hold * 2)
            
            # 如果是 Best Opportunity，大加分
            if is_best:
                score += 10
                
            return float(np.clip(score, 0, 100))
            
        return 50.0

    def _quantify_sentiment_result(self, report) -> float:
        """量化舆情分析结果"""
        if not report:
            return 50.0
        
        if isinstance(report, str): return 50.0
            
        bull = report.get('bullish_count', 0)
        bear = report.get('bearish_count', 0)
        total = bull + bear + report.get('neutral_count', 0)
        
        if total == 0: return 50.0
            
        # (Bull - Bear) / Total -> [-1, 1]
        net = (bull - bear) / total
        
        # 映射到 [30, 90] 区间 (避免极端值)
        score = 60 + (net * 30)
        
        # 热度奖励 (讨论越多越重要)
        if total > 5: score += 5
        if total > 10: score += 5
        
        return float(np.clip(score, 0, 100))

# 🌟 Pairwise Ranker 模型 (XGBoost)
class PairwiseRanker:
    """
    使用 XGBRanker 进行成对排序学习 (Learning to Rank).
    
    目标: 学习 rank:pairwise (或 rank:ndcg)，使得模型能够预测
    在一组候选股票中，谁的未来收益更高。
    """
    def __init__(self):
        self.model = None
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            self.xgb = None
            logger.warning("XGBoost not installed. Pairwise ranking disabled.")

    def train(self, X_train, y_train, qid_train):
        """
        训练排序模型.
        qid (Query ID) 必须是指示每组数据的数组 (例如: [1, 1, 1, 2, 2, ...])
        XGBoost 需要数据按 qid 排序。
        """
        if not self.xgb: return
        
        self.model = self.xgb.XGBRanker(
            tree_method="hist",
            objective="rank:pairwise",
            learning_rate=0.1,
            n_estimators=100
        )
        self.model.fit(X_train, y_train, qid=qid_train)
        
    def predict(self, X):
        """预测排序分"""
        if self.model:
            return self.model.predict(X)
        return np.zeros(len(X))

    def save_model(self, path):
        if self.model:
            self.model.save_model(path)
            
    def load_model(self, path):
        if self.xgb and not self.model:
            self.model = self.xgb.XGBRanker()
        if self.model:
            self.model.load_model(path)


# 单例
_system = None
def get_ranking_system():
    global _system
    if _system is None:
        _system = RankingSystem()
    return _system
