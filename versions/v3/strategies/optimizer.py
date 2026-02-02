#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strategy Optimizer
策略参数优化器 - 自动寻找最优策略组合

功能:
1. 网格搜索最优参数
2. 策略组合测试
3. 自动持续优化
4. A/B 测试框架
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    
    # BLUE 参数
    blue_daily_min: float = 100.0
    blue_daily_max: float = 300.0
    blue_weekly_min: float = 0.0
    blue_monthly_min: float = 0.0
    
    # ADX 参数
    adx_min: float = 20.0
    adx_max: float = 100.0
    
    # 成交额过滤
    turnover_min: float = 1.0  # 百万
    
    # 特殊信号
    require_heima: bool = False
    require_juedi: bool = False
    require_new_discovery: bool = False
    
    # 筹码形态
    chip_patterns: List[str] = field(default_factory=list)  # ['🔥', '📍']
    
    # 仓位管理
    max_positions: int = 10
    position_size_pct: float = 10.0
    
    # 风控
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 20.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'blue_daily_min': self.blue_daily_min,
            'blue_daily_max': self.blue_daily_max,
            'blue_weekly_min': self.blue_weekly_min,
            'blue_monthly_min': self.blue_monthly_min,
            'adx_min': self.adx_min,
            'adx_max': self.adx_max,
            'turnover_min': self.turnover_min,
            'require_heima': self.require_heima,
            'require_juedi': self.require_juedi,
            'require_new_discovery': self.require_new_discovery,
            'chip_patterns': self.chip_patterns,
            'max_positions': self.max_positions,
            'position_size_pct': self.position_size_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    config: StrategyConfig
    metrics: Dict = field(default_factory=dict)
    rank: int = 0
    
    @property
    def score(self) -> float:
        """综合评分 (加权)"""
        return (
            self.metrics.get('avg_return', 0) * 0.4 +
            self.metrics.get('win_rate', 0) * 0.3 +
            self.metrics.get('sharpe_like', 0) * 10 * 0.2 +
            (100 - abs(self.metrics.get('max_drawdown', 0))) * 0.1
        )


class StrategyOptimizer:
    """策略优化器"""
    
    # 预定义策略模板
    STRATEGY_TEMPLATES = {
        'BLUE_强势': StrategyConfig(
            name='BLUE_强势',
            blue_daily_min=150,
            blue_weekly_min=100,
            adx_min=25
        ),
        'BLUE_保守': StrategyConfig(
            name='BLUE_保守',
            blue_daily_min=120,
            blue_weekly_min=80,
            blue_monthly_min=50,
            adx_min=20,
            stop_loss_pct=5
        ),
        '黑马猎手': StrategyConfig(
            name='黑马猎手',
            blue_daily_min=100,
            require_heima=True,
            adx_min=20
        ),
        '掘地反攻': StrategyConfig(
            name='掘地反攻',
            blue_daily_min=80,
            require_juedi=True,
            adx_min=15
        ),
        '三重共振': StrategyConfig(
            name='三重共振',
            blue_daily_min=100,
            blue_weekly_min=80,
            blue_monthly_min=60,
            adx_min=25
        ),
        '新股狩猎': StrategyConfig(
            name='新股狩猎',
            blue_daily_min=120,
            require_new_discovery=True,
            turnover_min=5
        ),
        '筹码精选': StrategyConfig(
            name='筹码精选',
            blue_daily_min=100,
            chip_patterns=['🔥', '📍'],
            adx_min=20
        ),
        '高ADX趋势': StrategyConfig(
            name='高ADX趋势',
            blue_daily_min=100,
            adx_min=40,
            adx_max=80
        )
    }
    
    def __init__(self):
        self.results: List[OptimizationResult] = []
        self._supabase = None
    
    def _get_supabase(self):
        if self._supabase is None:
            try:
                from supabase import create_client
                url = os.getenv('SUPABASE_URL')
                key = os.getenv('SUPABASE_KEY')
                if url and key:
                    self._supabase = create_client(url, key)
            except:
                pass
        return self._supabase
    
    def generate_parameter_grid(self, 
                                 blue_daily_range: List[float] = [100, 120, 150, 180],
                                 blue_weekly_range: List[float] = [0, 50, 80, 100],
                                 adx_range: List[float] = [15, 20, 25, 30, 40],
                                 stop_loss_range: List[float] = [5, 8, 10],
                                 take_profit_range: List[float] = [15, 20, 30]) -> List[StrategyConfig]:
        """生成参数网格"""
        configs = []
        
        for blue_d, blue_w, adx, sl, tp in itertools.product(
            blue_daily_range, blue_weekly_range, adx_range, stop_loss_range, take_profit_range
        ):
            config = StrategyConfig(
                name=f"grid_bd{blue_d}_bw{blue_w}_adx{adx}_sl{sl}_tp{tp}",
                blue_daily_min=blue_d,
                blue_weekly_min=blue_w,
                adx_min=adx,
                stop_loss_pct=sl,
                take_profit_pct=tp
            )
            configs.append(config)
        
        return configs
    
    def evaluate_strategy(self, config: StrategyConfig, 
                          historical_data: pd.DataFrame) -> OptimizationResult:
        """评估单个策略配置"""
        if historical_data.empty:
            return OptimizationResult(config=config, metrics={})
        
        # 应用过滤条件
        filtered = historical_data.copy()
        
        # BLUE 过滤
        if 'blue_daily' in filtered.columns:
            filtered = filtered[filtered['blue_daily'] >= config.blue_daily_min]
            if config.blue_daily_max < 300:
                filtered = filtered[filtered['blue_daily'] <= config.blue_daily_max]
        
        if config.blue_weekly_min > 0 and 'blue_weekly' in filtered.columns:
            filtered = filtered[filtered['blue_weekly'] >= config.blue_weekly_min]
        
        if config.blue_monthly_min > 0 and 'blue_monthly' in filtered.columns:
            filtered = filtered[filtered['blue_monthly'] >= config.blue_monthly_min]
        
        # ADX 过滤
        if 'adx' in filtered.columns:
            filtered = filtered[filtered['adx'] >= config.adx_min]
            if config.adx_max < 100:
                filtered = filtered[filtered['adx'] <= config.adx_max]
        
        # 成交额过滤
        if 'turnover' in filtered.columns:
            filtered = filtered[filtered['turnover'] >= config.turnover_min]
        
        # 特殊信号过滤
        if config.require_heima and 'is_heima' in filtered.columns:
            filtered = filtered[filtered['is_heima'] == True]
        
        if config.require_juedi and 'is_juedi' in filtered.columns:
            filtered = filtered[filtered['is_juedi'] == True]
        
        if config.require_new_discovery and 'is_new_discovery' in filtered.columns:
            filtered = filtered[filtered['is_new_discovery'] == True]
        
        # 筹码形态过滤
        if config.chip_patterns and 'chip_pattern' in filtered.columns:
            filtered = filtered[filtered['chip_pattern'].isin(config.chip_patterns)]
        
        # 计算指标
        if len(filtered) < 5:
            return OptimizationResult(config=config, metrics={'n_samples': len(filtered)})
        
        returns = filtered['return_d5'].dropna()
        
        if len(returns) < 5:
            return OptimizationResult(config=config, metrics={'n_samples': len(returns)})
        
        metrics = {
            'n_samples': len(returns),
            'avg_return': round(returns.mean(), 2),
            'win_rate': round((returns > 0).mean() * 100, 1),
            'std_return': round(returns.std(), 2),
            'sharpe_like': round(returns.mean() / returns.std(), 3) if returns.std() > 0 else 0,
            'max_gain': round(returns.max(), 2),
            'max_drawdown': round(returns.min(), 2),
            'median_return': round(returns.median(), 2)
        }
        
        return OptimizationResult(config=config, metrics=metrics)
    
    def run_grid_search(self, 
                        historical_data: pd.DataFrame = None,
                        n_workers: int = 4) -> List[OptimizationResult]:
        """运行网格搜索"""
        if historical_data is None:
            historical_data = self._load_historical_data()
        
        if historical_data.empty:
            print("No historical data available")
            return []
        
        # 生成参数网格
        configs = self.generate_parameter_grid()
        print(f"Testing {len(configs)} parameter combinations...")
        
        results = []
        
        # 并行评估
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self.evaluate_strategy, cfg, historical_data): cfg 
                      for cfg in configs}
            
            for future in as_completed(futures):
                result = future.result()
                if result.metrics.get('n_samples', 0) >= 10:
                    results.append(result)
        
        # 按得分排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 更新排名
        for i, r in enumerate(results):
            r.rank = i + 1
        
        self.results = results
        return results
    
    def run_template_comparison(self, 
                                 historical_data: pd.DataFrame = None) -> List[OptimizationResult]:
        """比较预定义策略模板"""
        if historical_data is None:
            historical_data = self._load_historical_data()
        
        results = []
        
        for name, config in self.STRATEGY_TEMPLATES.items():
            result = self.evaluate_strategy(config, historical_data)
            results.append(result)
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        
        return results
    
    def _load_historical_data(self) -> pd.DataFrame:
        """从 Supabase 加载历史数据"""
        supabase = self._get_supabase()
        if not supabase:
            return pd.DataFrame()
        
        try:
            response = supabase.table('daily_picks_performance').select('*').not_.is_(
                'return_d5', 'null'
            ).execute()
            
            return pd.DataFrame(response.data or [])
        except Exception as e:
            print(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def save_best_config(self, result: OptimizationResult) -> bool:
        """保存最优配置"""
        supabase = self._get_supabase()
        if not supabase:
            return False
        
        try:
            data = {
                'config_name': result.config.name,
                'config_json': json.dumps(result.config.to_dict()),
                'metrics_json': json.dumps(result.metrics),
                'score': result.score,
                'updated_at': datetime.now().isoformat()
            }
            
            supabase.table('strategy_configs').upsert(
                data, on_conflict='config_name'
            ).execute()
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def get_best_config(self) -> Optional[StrategyConfig]:
        """获取当前最优配置"""
        supabase = self._get_supabase()
        if not supabase:
            return None
        
        try:
            response = supabase.table('strategy_configs').select('*').order(
                'score', desc=True
            ).limit(1).execute()
            
            if response.data:
                config_dict = json.loads(response.data[0]['config_json'])
                return StrategyConfig(**config_dict)
        except:
            pass
        
        return None
    
    def generate_report(self, results: List[OptimizationResult] = None, 
                        top_n: int = 10) -> str:
        """生成优化报告"""
        if results is None:
            results = self.results
        
        if not results:
            return "No optimization results available."
        
        report = "# 🎯 策略优化报告\n\n"
        report += f"**测试配置数**: {len(results)}\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        report += "## 🏆 Top 10 策略配置\n\n"
        report += "| 排名 | 策略名称 | 样本数 | 平均收益 | 胜率 | Sharpe | 综合得分 |\n"
        report += "|------|----------|--------|----------|------|--------|----------|\n"
        
        for r in results[:top_n]:
            report += f"| {r.rank} | {r.config.name[:20]} | "
            report += f"{r.metrics.get('n_samples', 0)} | "
            report += f"{r.metrics.get('avg_return', 0)}% | "
            report += f"{r.metrics.get('win_rate', 0)}% | "
            report += f"{r.metrics.get('sharpe_like', 0)} | "
            report += f"{r.score:.1f} |\n"
        
        # 最优策略详情
        if results:
            best = results[0]
            report += f"\n## 🥇 最优策略详情: {best.config.name}\n\n"
            report += "**参数配置**:\n"
            report += f"- BLUE 日线阈值: {best.config.blue_daily_min}\n"
            report += f"- BLUE 周线阈值: {best.config.blue_weekly_min}\n"
            report += f"- ADX 范围: {best.config.adx_min} - {best.config.adx_max}\n"
            report += f"- 止损: {best.config.stop_loss_pct}%\n"
            report += f"- 止盈: {best.config.take_profit_pct}%\n"
            
            report += "\n**表现指标**:\n"
            for k, v in best.metrics.items():
                report += f"- {k}: {v}\n"
        
        return report


class ContinuousOptimizer:
    """持续优化器 - 自动定期优化"""
    
    def __init__(self, optimizer: StrategyOptimizer):
        self.optimizer = optimizer
        self.optimization_history: List[Dict] = []
    
    def run_daily_optimization(self) -> Dict:
        """每日优化任务"""
        print(f"🔄 Running daily optimization at {datetime.now()}")
        
        # 1. 比较预定义策略
        template_results = self.optimizer.run_template_comparison()
        
        # 2. 如果数据足够，运行网格搜索
        historical_data = self.optimizer._load_historical_data()
        grid_results = []
        
        if len(historical_data) >= 100:
            # 精简网格搜索
            grid_results = self.optimizer.run_grid_search(historical_data)
        
        # 3. 合并结果
        all_results = template_results + grid_results[:20]
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 4. 保存最优配置
        if all_results:
            self.optimizer.save_best_config(all_results[0])
        
        # 5. 记录历史
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'n_configs_tested': len(all_results),
            'best_strategy': all_results[0].config.name if all_results else None,
            'best_score': all_results[0].score if all_results else 0,
            'best_return': all_results[0].metrics.get('avg_return', 0) if all_results else 0
        }
        self.optimization_history.append(summary)
        
        return summary
    
    def get_recommended_config(self) -> StrategyConfig:
        """获取推荐配置"""
        # 优先使用已保存的最优配置
        best = self.optimizer.get_best_config()
        if best:
            return best
        
        # 否则使用默认模板
        return StrategyOptimizer.STRATEGY_TEMPLATES['BLUE_强势']


# 便捷函数
def optimize_strategies(n_top: int = 10) -> str:
    """运行策略优化并返回报告"""
    optimizer = StrategyOptimizer()
    results = optimizer.run_template_comparison()
    return optimizer.generate_report(results, n_top)


def get_optimal_strategy() -> StrategyConfig:
    """获取当前最优策略配置"""
    optimizer = StrategyOptimizer()
    best = optimizer.get_best_config()
    return best or StrategyOptimizer.STRATEGY_TEMPLATES['BLUE_强势']


if __name__ == "__main__":
    print("🎯 Strategy Optimizer Test")
    
    optimizer = StrategyOptimizer()
    
    # 测试预定义策略比较
    print("\n📊 Comparing predefined strategies...")
    results = optimizer.run_template_comparison()
    
    print("\n" + optimizer.generate_report(results))
