
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class RankingSystem:
    """
    智能排序系统 (Hybrid Smart Ranker)
    整合 ML 技术评分、大师策略评分和舆情情绪评分，生成最终排序。
    """
    
    def __init__(self):
        # 权重配置 (可根据回测调整)
        self.weights = {
            'ml_technical': 0.4,   # 技术面 ML 模型
            'master_strategy': 0.4, # 大师策略共识
            'sentiment': 0.2       # 舆情情绪
        }
    
    def calculate_integrated_score(self, 
                                 df: pd.DataFrame, 
                                 master_results: Optional[Dict] = None,
                                 sentiment_results: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算综合排序分数
        
        Args:
            df: 股票基础数据 DataFrame
            master_results: 大师分析结果字典 {symbol: analysis_summary}
            sentiment_results: 舆情分析结果字典 {symbol: sentiment_report}
            
        Returns:
            df: 添加了 'Rank_Score' 和 'Score_Breakdown' 的 DataFrame
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 1. 技术面评分 (ML Technical Score)
        # 如果已有 ML 预测结果 (如 'Probability'), 直接使用
        # 否则使用基于 BLUE 和 趋势 的简单打分作为基线
        if 'Probability' in df.columns:
            df['score_tech'] = df['Probability'] * 100
        else:
            # 简易规则打分
            df['score_tech'] = self._calculate_heuristic_tech_score(df)
            
        # 2. 大师策略评分 (Master Strategy Score)
        df['score_master'] = 50.0 # 默认中性
        if master_results:
            df['score_master'] = df['Ticker'].apply(
                lambda x: self._quantify_master_result(master_results.get(x))
            )
            
        # 3. 舆情评分 (Sentiment Score)
        df['score_sentiment'] = 50.0 # 默认中性
        if sentiment_results:
            df['score_sentiment'] = df['Ticker'].apply(
                lambda x: self._quantify_sentiment_result(sentiment_results.get(x))
            )
            
        # 4. 综合加权 (Hybrid Scoring)
        # 基础分
        base_score = (
            df['score_tech'] * self.weights['ml_technical'] +
            df['score_master'] * self.weights['master_strategy'] +
            df['score_sentiment'] * self.weights['sentiment']
        )
        
        # 🌟 优中选优：Alpha Bonus (强强联合奖励)
        # 如果每一项都超过 60分，给予额外奖励
        bonus = pd.Series(0.0, index=df.index)
        all_good = (df['score_tech'] > 60) & (df['score_master'] > 55) & (df['score_sentiment'] > 50)
        bonus[all_good] += 10.0
        
        # 如果有大师强力推荐 (>80)，额外加分
        bonus[df['score_master'] > 80] += 5.0
        
        # 舆情极其火热 (>80)，且技术面不差 (>50)
        bonus[(df['score_sentiment'] > 80) & (df['score_tech'] > 50)] += 5.0
        
        df['Rank_Score'] = base_score + bonus
        
        # 归一化到 0-100
        df['Rank_Score'] = df['Rank_Score'].clip(0, 100)
        
        # 排序
        df = df.sort_values('Rank_Score', ascending=False)
        
        # 保存上下文用于未来 Pairwise 训练
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
