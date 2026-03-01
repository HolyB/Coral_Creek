#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL Trading Agent — 强化学习交易策略
====================================

使用 PPO 学习最优交易策略:
- State: MMoE/LGB 预测 + 技术因子 + 持仓状态
- Action: 仓位调整 (连续 [-1, 1], 负=做空)
- Reward: 风险调整收益 (Sharpe-like)

架构:
  observation = [MMoE_dir_prob, MMoE_return_5d, LGB_dir_prob, 
                 技术因子(20个精选), 持仓比例, 浮盈, 持仓天数]
  action = 目标仓位比例 [-1, 1]
  reward = 日收益率 - λ * |仓位变化| (鼓励稳定持仓)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


MODEL_DIR = Path(__file__).parent / 'saved_models' / 'rl_agent'


class TradingEnv(gym.Env):
    """
    股票交易强化学习环境
    
    单股票版本 — 每步决定一只股票的仓位
    
    State (观测空间):
        - 技术特征 (20 个精选): RSI, MACD, 布林带, 波动率, 动量等
        - ML 预测: direction_prob, pred_return_5d, pred_return_20d
        - 持仓状态: position_pct, unrealized_pnl, holding_days
        - 市场状态: recent_return_5d, volatility
        总计 ~28 维
    
    Action (动作空间):
        - 连续 [-1, 1]: 目标仓位比例
          -1 = 满仓做空, 0 = 空仓, 1 = 满仓做多
    
    Reward:
        - 日收益率 * 100 (放大)
        - 交易成本惩罚: -|仓位变化| * cost_penalty
        - 回撤惩罚: 浮亏超过阈值时额外惩罚
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 feature_data: np.ndarray,
                 feature_names: List[str],
                 initial_cash: float = 100000,
                 max_position: float = 1.0,
                 commission: float = 0.001,
                 cost_penalty: float = 0.5,
                 drawdown_penalty: float = 2.0,
                 max_steps: int = 200):
        """
        Args:
            price_data: DataFrame with 'Close' column
            feature_data: 特征矩阵 (T, N_features)
            feature_names: 特征名称
            initial_cash: 初始资金
            max_position: 最大仓位比例
            commission: 单边手续费率
            cost_penalty: 换手惩罚系数
            drawdown_penalty: 回撤惩罚系数
            max_steps: 最大步数
        """
        super().__init__()
        
        self.prices = price_data['Close'].values.astype(float)
        self.features = feature_data.astype(float)
        self.feature_names = feature_names
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.commission = commission
        self.cost_penalty = cost_penalty
        self.drawdown_penalty = drawdown_penalty
        self.max_steps = min(max_steps, len(self.prices) - 1)
        
        # 观测空间: 特征 + 持仓状态
        n_features = self.features.shape[1]
        self.n_obs = n_features + 5  # +5 for position state
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32
        )
        
        # 动作空间: 目标仓位 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 状态
        self.current_step = 0
        self.position = 0.0  # 仓位比例 [-1, 1]
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.peak_value = initial_cash
        self.entry_price = 0.0
        self.holding_days = 0
        self.trades = []
        self.equity_curve = []
    
    def _get_obs(self) -> np.ndarray:
        """构建观测"""
        feat = self.features[self.current_step].copy()
        
        # 持仓状态特征
        unrealized_pnl = 0.0
        if abs(self.position) > 0.01 and self.entry_price > 0:
            current_price = self.prices[self.current_step]
            unrealized_pnl = (current_price / self.entry_price - 1) * np.sign(self.position) * 100
        
        drawdown = (self.portfolio_value / self.peak_value - 1) * 100 if self.peak_value > 0 else 0
        
        state = np.concatenate([
            feat,
            [self.position,           # 当前仓位
             unrealized_pnl,          # 浮盈 (%)
             min(self.holding_days / 20.0, 1.0),  # 持仓天数 (归一化)
             drawdown,                # 组合回撤 (%)
             (self.portfolio_value / self.initial_cash - 1) * 100,  # 累计收益 (%)
            ]
        ])
        
        return state.astype(np.float32)
    
    def _calculate_reward(self, old_value: float, new_value: float, 
                         position_change: float) -> float:
        """计算奖励"""
        # 1. 日收益率 (主要奖励)
        daily_return = (new_value / old_value - 1) * 100 if old_value > 0 else 0
        
        # 2. 交易成本惩罚
        trade_cost = abs(position_change) * self.cost_penalty
        
        # 3. 回撤惩罚
        dd_penalty = 0
        if new_value < self.peak_value * 0.95:  # 回撤超过 5%
            dd = (1 - new_value / self.peak_value) * 100
            dd_penalty = dd * self.drawdown_penalty * 0.01
        
        # 4. 持仓过久惩罚
        hold_penalty = 0
        if self.holding_days > 30:
            hold_penalty = 0.01 * (self.holding_days - 30)
        
        reward = daily_return - trade_cost - dd_penalty - hold_penalty
        return float(reward)
    
    def step(self, action):
        """执行一步"""
        target_position = float(np.clip(action[0], -self.max_position, self.max_position))
        
        old_value = self.portfolio_value
        position_change = target_position - self.position
        
        # 执行交易
        current_price = self.prices[self.current_step]
        
        if abs(position_change) > 0.01:
            # 交易成本
            trade_value = abs(position_change) * self.portfolio_value
            cost = trade_value * self.commission
            self.cash -= cost
            
            if abs(target_position) > 0.01 and abs(self.position) < 0.01:
                self.entry_price = current_price
                self.holding_days = 0
            elif abs(target_position) < 0.01:
                self.entry_price = 0
                self.holding_days = 0
        
        self.position = target_position
        
        # 移动到下一步
        self.current_step += 1
        
        if self.current_step >= len(self.prices):
            return self._get_obs() if self.current_step < len(self.features) else np.zeros(self.n_obs, dtype=np.float32), 0, True, True, {}
        
        new_price = self.prices[self.current_step]
        
        # 更新组合价值
        if abs(self.position) > 0.01 and self.entry_price > 0:
            price_change = (new_price / current_price - 1)
            position_pnl = self.position * price_change * self.portfolio_value
            self.portfolio_value += position_pnl
            self.holding_days += 1
        
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.equity_curve.append(self.portfolio_value)
        
        # 奖励
        reward = self._calculate_reward(old_value, self.portfolio_value, position_change)
        
        # 记录交易
        if abs(position_change) > 0.05:
            self.trades.append({
                'step': self.current_step,
                'price': new_price,
                'position': self.position,
                'change': position_change,
                'portfolio_value': self.portfolio_value,
            })
        
        # 结束条件
        done = (self.current_step >= self.max_steps + self.start_step or 
                self.portfolio_value < self.initial_cash * 0.8)  # 亏损 20% 强制结束
        
        truncated = self.current_step >= self.max_steps + self.start_step
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'total_return': (self.portfolio_value / self.initial_cash - 1) * 100,
        }
        
        return self._get_obs(), reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机起始点 (避免总从头开始)
        max_start = max(0, len(self.prices) - self.max_steps - 10)
        self.start_step = np.random.randint(0, max(1, max_start))
        self.current_step = self.start_step
        
        self.position = 0.0
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.entry_price = 0.0
        self.holding_days = 0
        self.trades = []
        self.equity_curve = [self.initial_cash]
        
        return self._get_obs(), {}
    
    def get_stats(self) -> Dict:
        """获取回测统计"""
        if not self.equity_curve:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        
        total_return = (equity[-1] / equity[0] - 1) * 100
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        max_dd = np.max(1 - equity / np.maximum.accumulate(equity)) * 100
        
        return {
            'total_return_pct': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown_pct': float(max_dd),
            'n_trades': len(self.trades),
            'final_value': float(equity[-1]),
        }


def prepare_rl_data(symbol: str, market: str = 'US', days: int = 250) -> Tuple:
    """
    准备 RL 训练数据
    
    Returns:
        (price_df, feature_matrix, feature_names)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from db.stock_history import get_stock_history
    from ml.advanced_features import AdvancedFeatureEngineer
    from ml.alpha_factors import Alpha158Factors
    
    history = get_stock_history(symbol, market, days=days)
    if history is None or history.empty or len(history) < 60:
        return None, None, None
    
    # 标准化
    if not isinstance(history.index, pd.DatetimeIndex):
        if 'Date' in history.columns:
            history = history.set_index('Date')
        elif 'date' in history.columns:
            history = history.set_index('date')
        history.index = pd.to_datetime(history.index)
    
    col_map = {}
    for c in history.columns:
        cl = c.lower()
        if cl == 'close': col_map[c] = 'Close'
        elif cl == 'open': col_map[c] = 'Open'
        elif cl == 'high': col_map[c] = 'High'
        elif cl == 'low': col_map[c] = 'Low'
        elif cl == 'volume': col_map[c] = 'Volume'
    if col_map:
        history = history.rename(columns=col_map)
    
    # 计算特征
    adv = AdvancedFeatureEngineer()
    alpha = Alpha158Factors()
    
    df_adv = adv.transform(history)
    df_alpha = alpha.compute(history)
    
    # v9 优化特征集 (基于 permutation importance 分析)
    # 移除有害: atr_pct(-3.6%), kdj_j_9(-3.6%), trend_consistency(-0.6%), relative_volume_5(-0.4%)
    # 保留 top: obv_slope_10(+16%), mfi_14(+14%), momentum_acceleration(+12.5%)
    selected_features = [
        # 波动率 (保留有用的)
        'volatility_20', 'vol_ratio',
        # 动量 (top performers)
        'roc_5', 'roc_20', 'momentum_acceleration',
        # 趋势
        'ma_distance_20', 'adx_14',
        # 量价 (obv_slope_10 是最重要的!)
        'obv_slope_10',
        # Alpha158 (保留高 importance)
        'rsi_12', 'macd_dif_pct',
        'bband_pctb_20', 'vwap_bias',
        'mfi_14', 'cci_14', 'willr_14',
        'return_skew_20',
    ]
    
    # 合并特征
    combined = pd.DataFrame(index=history.index)
    for f in selected_features:
        if f in df_adv.columns:
            combined[f] = df_adv[f]
        elif f in df_alpha.columns:
            combined[f] = df_alpha[f]
    
    # 添加蔡森图表特征 (核心信号系统)
    try:
        from ml.caisen_features import compute_caisen_features
        cs = compute_caisen_features(history)
        # 选择最重要的蔡森特征加入 RL
        cs_selected = [
            'cs_blue_pct', 'cs_pink_norm', 'cs_fund_net_norm',
            'cs_cci_value', 'cs_signal_strength', 'cs_bull_bear_ratio',
            'cs_main_force', 'cs_heima', 'cs_juedi',
            'cs_regime_bottom', 'cs_regime_top',
            'cs_var50_slope',
        ]
        for col in cs_selected:
            if col in cs.columns:
                combined[col] = cs[col]
    except Exception:
        pass
    
    # 添加价格动量特征
    combined['return_1d'] = history['Close'].pct_change() * 100
    combined['return_5d'] = history['Close'].pct_change(5) * 100
    combined['return_20d'] = history['Close'].pct_change(20) * 100
    
    # 添加量价背离
    combined['volume_price_corr'] = (
        history['Close'].pct_change().rolling(10).corr(
            history['Volume'].pct_change()
        ) * 100
    )
    
    # 丢弃 NaN
    valid_mask = combined.notna().all(axis=1)
    combined = combined[valid_mask]
    price_df = history.loc[combined.index][['Close']].copy()
    
    # 标准化特征
    feature_matrix = combined.values.astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Z-score 标准化
    mean = np.mean(feature_matrix, axis=0)
    std = np.std(feature_matrix, axis=0) + 1e-8
    feature_matrix = (feature_matrix - mean) / std
    
    return price_df, feature_matrix, list(combined.columns)


def train_rl_agent(symbols: List[str] = None,
                   market: str = 'US',
                   total_timesteps: int = 100000,
                   algorithm: str = 'PPO') -> Dict:
    """
    训练 RL 交易智能体
    
    Args:
        symbols: 训练用的股票列表
        market: 市场
        total_timesteps: 总训练步数
        algorithm: 'PPO' 或 'A2C'
    
    Returns:
        训练结果
    """
    if not GYM_AVAILABLE or not SB3_AVAILABLE:
        return {'status': 'error', 'reason': 'Please install: pip install gymnasium stable-baselines3'}
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.database import init_db, get_scanned_dates, query_scan_results
    
    init_db()
    
    # 默认用最活跃的股票训练
    if symbols is None:
        dates = get_scanned_dates(market=market)
        from collections import Counter
        counts = Counter()
        for d in dates[:30]:
            results = query_scan_results(scan_date=d, market=market, limit=500)
            for r in results:
                sym = r.get('symbol', '')
                price = float(r.get('price', 0) or 0)
                if sym and 5 < price < 300:
                    counts[sym] += 1
        symbols = [s for s, _ in counts.most_common(20)]
    
    print(f"🤖 RL Agent 训练")
    print(f"   算法: {algorithm}")
    print(f"   训练股票: {len(symbols)} 只")
    print(f"   总步数: {total_timesteps:,}")
    
    # 准备多个环境 (不同股票)
    envs_data = []
    for sym in symbols:
        try:
            price_df, feat_matrix, feat_names = prepare_rl_data(sym, market)
            if price_df is not None and len(price_df) > 60:
                envs_data.append((sym, price_df, feat_matrix, feat_names))
        except Exception:
            continue
    
    if not envs_data:
        return {'status': 'error', 'reason': 'No valid training data'}
    
    print(f"   有效股票: {len(envs_data)} 只")
    
    # 创建环境工厂 (每次随机选一只股票)
    def make_env():
        def _init():
            idx = np.random.randint(len(envs_data))
            sym, price_df, feat_matrix, feat_names = envs_data[idx]
            env = TradingEnv(
                price_data=price_df,
                feature_data=feat_matrix,
                feature_names=feat_names,
                max_steps=min(len(price_df) - 1, 120),
            )
            return env
        return _init
    
    # 创建向量化环境
    vec_env = DummyVecEnv([make_env() for _ in range(4)])
    
    # 选择算法
    if algorithm == 'A2C':
        model = A2C(
            'MlpPolicy', vec_env,
            learning_rate=3e-4,
            n_steps=128,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
        )
    else:
        model = PPO(
            'MlpPolicy', vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
        )
    
    # 训练
    print(f"\n🏋️ 开始训练...")
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(MODEL_DIR / f'{algorithm.lower()}_trader')
    model.save(model_path)
    
    # 保存元数据
    meta = {
        'algorithm': algorithm,
        'total_timesteps': total_timesteps,
        'n_training_stocks': len(envs_data),
        'training_symbols': [d[0] for d in envs_data],
        'feature_names': envs_data[0][3] if envs_data else [],
        'n_features': len(envs_data[0][3]) if envs_data else 0,
    }
    with open(MODEL_DIR / 'rl_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n💾 模型已保存: {model_path}")
    
    # 评估
    print(f"\n📊 评估 (在训练股票上)...")
    eval_results = evaluate_rl_agent(model, envs_data[:5])
    
    meta['eval_results'] = eval_results
    with open(MODEL_DIR / 'rl_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    return {
        'status': 'success',
        'model_path': model_path,
        **meta,
        'eval': eval_results,
    }


def evaluate_rl_agent(model, envs_data: List, n_episodes: int = 3) -> Dict:
    """评估 RL 智能体"""
    all_stats = []
    
    for sym, price_df, feat_matrix, feat_names in envs_data:
        env = TradingEnv(
            price_data=price_df,
            feature_data=feat_matrix,
            feature_names=feat_names,
            max_steps=min(len(price_df) - 1, 120),
        )
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            stats = env.get_stats()
            stats['symbol'] = sym
            stats['episode'] = ep
            all_stats.append(stats)
    
    if not all_stats:
        return {}
    
    df = pd.DataFrame(all_stats)
    
    result = {
        'avg_return_pct': float(df['total_return_pct'].mean()),
        'avg_sharpe': float(df['sharpe_ratio'].mean()),
        'avg_max_dd_pct': float(df['max_drawdown_pct'].mean()),
        'avg_trades': float(df['n_trades'].mean()),
        'win_rate_pct': float((df['total_return_pct'] > 0).mean() * 100),
        'best_return': float(df['total_return_pct'].max()),
        'worst_return': float(df['total_return_pct'].min()),
        'episodes': len(df),
    }
    
    print(f"  平均收益: {result['avg_return_pct']:.1f}%")
    print(f"  Sharpe: {result['avg_sharpe']:.2f}")
    print(f"  最大回撤: {result['avg_max_dd_pct']:.1f}%")
    print(f"  胜率: {result['win_rate_pct']:.0f}%")
    print(f"  平均交易: {result['avg_trades']:.0f} 次")
    
    return result


# === 命令行入口 ===
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Trading Agent')
    parser.add_argument('--market', default='US')
    parser.add_argument('--algo', default='PPO', choices=['PPO', 'A2C'])
    parser.add_argument('--steps', type=int, default=50000, help='训练步数')
    parser.add_argument('--symbols', nargs='+', help='训练股票 (可选)')
    
    args = parser.parse_args()
    
    from db.database import init_db
    init_db()
    
    result = train_rl_agent(
        symbols=args.symbols,
        market=args.market,
        total_timesteps=args.steps,
        algorithm=args.algo,
    )
    
    print(f"\n{'='*50}")
    print(json.dumps(result, indent=2, default=str))
