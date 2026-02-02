
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
            
        # 4. 综合加权
        df['Rank_Score'] = (
            df['score_tech'] * self.weights['ml_technical'] +
            df['score_master'] * self.weights['master_strategy'] +
            df['score_sentiment'] * self.weights['sentiment']
        )
        
        # 归一化到 0-99 (便于展示)
        # 稍微拉伸一下分布
        # df['Rank_Score'] = np.clip(df['Rank_Score'], 0, 99)
        
        return df.sort_values('Rank_Score', ascending=False)
    
    def _calculate_heuristic_tech_score(self, df: pd.DataFrame) -> pd.Series:
        """基于简单规则的技术面打分 (当 ML 模型不可用时)"""
        scores = pd.Series(50.0, index=df.index)
        
        if 'Day BLUE' in df.columns:
            # BLUE: 50以上加分, 100以上大加分
            scores += (df['Day BLUE'].fillna(0) / 200.0) * 30
            
        if 'ADX' in df.columns:
            # ADX: 趋势越强分越高
            scores += (df['ADX'].fillna(0) / 100.0) * 20
            
        if 'Profit_Ratio' in df.columns:
            # 获利盘: 越高越好
            scores += (df['Profit_Ratio'].fillna(0) * 20)
            
        return scores.clip(0, 100)
    
    def _quantify_master_result(self, summary) -> float:
        """量化大师分析结果"""
        if not summary:
            return 50.0
        
        # 如果是简单的字符串 (如 "数据不足")
        if isinstance(summary, str):
            if "买入" in summary or "积极" in summary: return 80.0
            if "卖出" in summary or "回避" in summary: return 20.0
            return 50.0
            
        # 如果是字典 (完整摘要)
        # {'buy_votes': 3, 'sell_votes': 0, ...}
        if isinstance(summary, dict):
            buy = summary.get('buy_votes', 0)
            sell = summary.get('sell_votes', 0)
            hold = summary.get('hold_votes', 0)
            
            # 简单模型: 基础50 + 买*10 - 卖*10 + 持*2
            # 满分 5票买 = 50 + 50 = 100
            # 最差 5票卖 = 50 - 50 = 0
            score = 50 + (buy * 10) - (sell * 10) + (hold * 2)
            return float(np.clip(score, 0, 100))
            
        return 50.0

    def _quantify_sentiment_result(self, report) -> float:
        """量化舆情分析结果"""
        if not report:
            return 50.0
            
        # 同样，如果是简单字符串
        if isinstance(report, str):
            return 50.0
            
        # 字典
        bull = report.get('bullish_count', 0)
        bear = report.get('bearish_count', 0)
        neutral = report.get('neutral_count', 0)
        total = bull + bear + neutral
        
        if total == 0:
            return 50.0
            
        # 情绪分: (Bull - Bear) / Total 映射到 0-100
        # 范围 -1 到 1 -> 0 到 100
        # -1 -> 0, 0 -> 50, 1 -> 100
        net_sentiment = (bull - bear) / total
        score = 50 + (net_sentiment * 50)
        
        # log 变换一下热度? 热度高加权? 暂时不
        return float(score)

# Pairwise Ranker 模型 (待实现 XGBRanker)
class PairwiseRanker:
    """使用 XGBRanker 进行成对排序学习"""
    pass

# 单例
_system = None
def get_ranking_system():
    global _system
    if _system is None:
        _system = RankingSystem()
    return _system
