#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略自动回测器 — Auto Strategy Backtester
===========================================
对爬取的策略自动回测:
  1. 从数据库取未回测的策略
  2. 解析策略规则 → 生成可执行的交易信号
  3. 在真实历史数据上回测
  4. 计算 Sharpe / Return / MaxDD / WinRate
  5. 标记优秀策略并通知
"""
import sys
import os
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================
# 内置策略库 — 可直接回测的经典策略
# =============================================================
BUILTIN_STRATEGIES = {
    # --- 均线类 ---
    "ma_cross_5_20": {
        "name": "MA5/MA20 金叉死叉",
        "source": "classic",
        "desc": "5日线上穿20日线买入，下穿卖出",
        "buy": lambda df: (df['sma5'] > df['sma20']) & (df['sma5'].shift(1) <= df['sma20'].shift(1)),
        "sell": lambda df: (df['sma5'] < df['sma20']) & (df['sma5'].shift(1) >= df['sma20'].shift(1)),
    },
    "ma_cross_10_60": {
        "name": "MA10/MA60 (中期趋势)",
        "source": "classic",
        "desc": "10日线上穿60日线买入",
        "buy": lambda df: (df['sma10'] > df['sma60']) & (df['sma10'].shift(1) <= df['sma60'].shift(1)),
        "sell": lambda df: (df['sma10'] < df['sma60']) & (df['sma10'].shift(1) >= df['sma60'].shift(1)),
    },
    "triple_ma": {
        "name": "三线共振 (MA5>MA20>MA60)",
        "source": "classic",
        "desc": "三条均线多头排列时买入，任一条死叉卖出",
        "buy": lambda df: (df['sma5'] > df['sma20']) & (df['sma20'] > df['sma60']) & 
                           ~((df['sma5'].shift(1) > df['sma20'].shift(1)) & (df['sma20'].shift(1) > df['sma60'].shift(1))),
        "sell": lambda df: (df['sma5'] < df['sma20']) & (df['sma5'].shift(1) >= df['sma20'].shift(1)),
    },
    
    # --- RSI 类 ---
    "rsi_oversold": {
        "name": "RSI 超卖反弹 (RSI<30买, >70卖)",
        "source": "classic",
        "desc": "RSI从超卖区回升时买入",
        "buy": lambda df: (df['rsi'] > 30) & (df['rsi'].shift(1) <= 30),
        "sell": lambda df: (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70),
    },
    
    # --- 量价类 ---
    "vol_breakout": {
        "name": "放量突破 (蔡森P01)",
        "source": "caisen",
        "desc": "突破20日高点且量比>1.5",
        "buy": lambda df: (df['Close'] > df['high_20']) & (df['vol_ratio'] >= 1.5),
        "sell": lambda df: (df['Close'] < df['sma20']) & (df['Close'].shift(1) >= df['sma20'].shift(1)),
    },
    "shrink_pullback": {
        "name": "缩量回踩 (蔡森P02)",
        "source": "caisen",
        "desc": "缩量回踩MA20不破",
        "buy": lambda df: (abs(df['Close'] - df['sma20']) / df['sma20'] <= 0.02) & 
                           (df['vol_ratio'] <= 0.75) & (df['Close'] >= df['Close'].shift(1)),
        "sell": lambda df: (df['Close'] < df['sma20'] * 0.95),
    },
    "vol_pile_bottom": {
        "name": "底部堆量 (蔡森P03)",
        "source": "caisen",
        "desc": "底部连续放量，主力吸筹",
        "buy": lambda df: (df['vol_sum3'] > df['vol_sum3_prev'] * 1.4) & (df['ret5'] < -3),
        "sell": lambda df: (df['ret1'] < -5) | (df['Close'] < df['sma20'] * 0.92),
    },
    "platform_breakout": {
        "name": "平台突破 (蔡森P04/萧明道X03)",
        "source": "caisen",
        "desc": "横盘收敛后放量突破",
        "buy": lambda df: (df['box_range'] < 0.12) & (df['Close'] > df['high_20']) & (df['vol_ratio'] > 1.3),
        "sell": lambda df: (df['Close'] < df['sma20']),
    },
    
    # --- MACD 类 ---
    "macd_golden": {
        "name": "MACD 金叉",
        "source": "classic",
        "desc": "MACD线上穿信号线",
        "buy": lambda df: (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0),
        "sell": lambda df: (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0),
    },
    
    # --- Bollinger Band ---
    "bband_squeeze": {
        "name": "布林带收窄突破",
        "source": "classic",
        "desc": "带宽收至低位后向上突破",
        "buy": lambda df: (df['Close'] > df['bb_upper']) & (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)),
        "sell": lambda df: (df['Close'] < df['bb_lower']),
    },
    
    # --- 复合策略 ---
    "blue_momentum": {
        "name": "BLUE + 动量",
        "source": "coral_creek",
        "desc": "BLUE信号>=100 且 RSI>50",
        "buy": lambda df: (df.get('cs_blue_value', pd.Series(0, index=df.index)) >= 100) & (df['rsi'] > 50),
        "sell": lambda df: (df.get('cs_blue_value', pd.Series(0, index=df.index)) == 0) | (df['rsi'] < 40),
    },
    "multi_signal": {
        "name": "多信号共振 (BLUE+黑马+量价)",
        "source": "coral_creek", 
        "desc": "BLUE>50 + pattern_score>2",
        "buy": lambda df: (df.get('cs_blue_pct', pd.Series(0, index=df.index)) > 0.25) & 
                           (df.get('cs_pattern_score', pd.Series(0, index=df.index)) >= 2),
        "sell": lambda df: (df.get('cs_pattern_score', pd.Series(0, index=df.index)) < -1) | 
                            (df.get('cs_blue_pct', pd.Series(0, index=df.index)) == 0),
    },
}


def prepare_features(hist: pd.DataFrame) -> pd.DataFrame:
    """为回测准备技术特征"""
    df = hist.copy()
    
    # 均线
    df['sma5'] = df['Close'].rolling(5).mean()
    df['sma10'] = df['Close'].rolling(10).mean()
    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma60'] = df['Close'].rolling(60).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd_hist'] = macd - signal
    
    # Bollinger Bands
    df['bb_mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)
    
    # 量比
    df['vol_sma20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_sma20'] + 1e-8)
    
    # 20日高低
    df['high_20'] = df['High'].rolling(20).max().shift(1)
    df['low_20'] = df['Low'].rolling(20).min().shift(1)
    
    # 收益
    df['ret1'] = df['Close'].pct_change() * 100
    df['ret5'] = df['Close'].pct_change(5) * 100
    
    # 量价聚合
    df['vol_sum3'] = df['Volume'].rolling(3).sum()
    df['vol_sum3_prev'] = df['Volume'].shift(3).rolling(3).sum()
    
    # 箱体
    df['box_range'] = (df['High'].rolling(20).max() - df['Low'].rolling(20).min()) / (df['Close'] + 1e-8)
    
    # 蔡森特征 (如果可用)
    try:
        from ml.caisen_features import compute_caisen_features
        cs = compute_caisen_features(hist)
        for col in cs.columns:
            if col not in df.columns:
                df[col] = cs[col]
    except Exception:
        pass
    
    return df.dropna(subset=['sma60'])


def backtest_strategy(
    strategy_name: str,
    buy_func,
    sell_func,
    hist: pd.DataFrame,
    initial_capital: float = 100000,
    commission: float = 0.001,
    stop_loss: float = 0.08,
) -> Dict:
    """
    对单个策略回测
    
    Returns:
        回测结果字典
    """
    df = prepare_features(hist)
    if len(df) < 60:
        return {'error': 'Not enough data'}
    
    try:
        buy_signals = buy_func(df).fillna(False)
        sell_signals = sell_func(df).fillna(False)
    except Exception as e:
        return {'error': f'Signal generation failed: {e}'}
    
    # 模拟交易
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    
    for i in range(1, len(df)):
        price = float(df['Close'].iloc[i])
        
        if position == 0 and bool(buy_signals.iloc[i]):
            # 买入
            shares = int(capital * 0.95 / price)  # 95% 仓位
            if shares > 0:
                cost = shares * price * (1 + commission)
                capital -= cost
                position = shares
                entry_price = price
        
        elif position > 0:
            # 检查卖出 / 止损
            pnl_pct = (price - entry_price) / entry_price
            
            should_sell = bool(sell_signals.iloc[i]) or pnl_pct <= -stop_loss
            
            if should_sell:
                proceeds = position * price * (1 - commission)
                capital += proceeds
                trades.append({
                    'entry': entry_price,
                    'exit': price,
                    'pnl_pct': pnl_pct * 100,
                    'days': i - (len(df) - len(trades) - 1),
                })
                position = 0
                entry_price = 0
        
        # 记录权益
        equity = capital + position * price
        equity_curve.append(equity)
    
    # 平掉剩余仓位
    if position > 0:
        final_price = float(df['Close'].iloc[-1])
        capital += position * final_price * (1 - commission)
        trades.append({
            'entry': entry_price,
            'exit': final_price,
            'pnl_pct': (final_price - entry_price) / entry_price * 100,
        })
        equity_curve[-1] = capital
    
    # 计算统计
    equity = np.array(equity_curve)
    daily_returns = np.diff(equity) / equity[:-1]
    
    total_return = (equity[-1] / equity[0] - 1) * 100
    n_days = len(daily_returns)
    ann_return = float(np.mean(daily_returns) * 252 * 100) if n_days > 0 else 0
    ann_vol = float(np.std(daily_returns) * np.sqrt(252) * 100) if n_days > 0 else 1
    sharpe = ann_return / (ann_vol + 1e-8)
    
    # MaxDD
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min() * 100)
    
    # Trades
    if trades:
        win_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(win_trades) / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in win_trades]) if win_trades else 0
        lose_trades = [t for t in trades if t['pnl_pct'] <= 0]
        avg_loss = np.mean([t['pnl_pct'] for t in lose_trades]) if lose_trades else 0
        profit_factor = abs(sum(t['pnl_pct'] for t in win_trades) / 
                           (sum(t['pnl_pct'] for t in lose_trades) + 1e-8))
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    n_buy = int(buy_signals.sum())
    
    return {
        'strategy': strategy_name,
        'total_return': round(total_return, 2),
        'annual_return': round(ann_return, 2),
        'annual_vol': round(ann_vol, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd, 2),
        'n_trades': len(trades),
        'n_buy_signals': n_buy,
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'n_days': n_days,
    }


def run_all_builtin_backtests(market: str = 'US', n_stocks: int = 30, days: int = 365) -> pd.DataFrame:
    """
    对所有内置策略进行大规模回测
    
    在多只股票上运行，取平均结果
    """
    from db.database import init_db, get_scanned_dates, query_scan_results
    from db.stock_history import get_stock_history
    from collections import Counter
    
    init_db()
    
    # 获取活跃股票
    dates = get_scanned_dates(market=market)
    sym_counts = Counter()
    for d in dates[:60]:
        for r in query_scan_results(scan_date=d, market=market, limit=500):
            sym = r.get('symbol', '')
            if sym:
                sym_counts[sym] += 1
    
    top_syms = [s for s, _ in sym_counts.most_common(n_stocks)]
    print(f"📊 Backtesting {len(top_syms)} stocks × {len(BUILTIN_STRATEGIES)} strategies")
    
    all_results = []
    
    for strat_key, strat_def in BUILTIN_STRATEGIES.items():
        strat_results = []
        
        for sym in top_syms:
            hist = get_stock_history(sym, market, days=days + 100)
            if hist is None or hist.empty or len(hist) < 100:
                continue
            
            # 标准化
            if not isinstance(hist.index, pd.DatetimeIndex):
                for c in ['Date', 'date']:
                    if c in hist.columns:
                        hist = hist.set_index(c)
                        break
                hist.index = pd.to_datetime(hist.index)
            
            for need in ['Open', 'High', 'Low', 'Close', 'Volume']:
                for c in hist.columns:
                    if c.lower() == need.lower() and c != need:
                        hist = hist.rename(columns={c: need})
            
            if 'Close' not in hist.columns:
                continue
            
            result = backtest_strategy(
                strat_key, strat_def['buy'], strat_def['sell'], hist
            )
            
            if 'error' not in result:
                strat_results.append(result)
        
        if strat_results:
            avg = {
                'strategy': strat_key,
                'name': strat_def['name'],
                'source': strat_def['source'],
                'n_stocks': len(strat_results),
                'avg_return': np.mean([r['total_return'] for r in strat_results]),
                'avg_sharpe': np.mean([r['sharpe'] for r in strat_results]),
                'avg_max_dd': np.mean([r['max_drawdown'] for r in strat_results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in strat_results]),
                'avg_trades': np.mean([r['n_trades'] for r in strat_results]),
                'avg_pf': np.mean([r['profit_factor'] for r in strat_results]),
                'median_return': np.median([r['total_return'] for r in strat_results]),
            }
            all_results.append(avg)
            
            print(f"  {strat_def['name']:<35} "
                  f"Ret={avg['avg_return']:+6.1f}%  "
                  f"Sharpe={avg['avg_sharpe']:+5.2f}  "
                  f"WR={avg['avg_win_rate']:5.1f}%  "
                  f"DD={avg['avg_max_dd']:6.1f}%  "
                  f"Trades={avg['avg_trades']:4.1f}")
    
    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.sort_values('avg_sharpe', ascending=False)
    
    return df


# =============================================================
# 策略信号 → ML 特征
# =============================================================
def generate_strategy_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    将所有内置策略的买/卖信号转为 ML 特征
    
    每个策略生成:
      - strat_{name}_buy: 买入信号 (0/1)
      - strat_{name}_sell: 卖出信号 (0/1)
    
    + 复合特征:
      - strat_total_bull: 同时发出买入的策略数
      - strat_total_bear: 同时发出卖出的策略数
      - strat_consensus: (bull - bear) / total
      - strat_diversity: 有信号的策略比例
    
    Returns:
        DataFrame with strategy features, same index as input
    """
    df = prepare_features(hist)
    features = pd.DataFrame(index=df.index)
    
    buy_cols = []
    sell_cols = []
    
    for key, strat in BUILTIN_STRATEGIES.items():
        try:
            buy_sig = strat['buy'](df).fillna(False).astype(float)
            sell_sig = strat['sell'](df).fillna(False).astype(float)
            
            buy_col = f'strat_{key}_buy'
            sell_col = f'strat_{key}_sell'
            features[buy_col] = buy_sig.values
            features[sell_col] = sell_sig.values
            buy_cols.append(buy_col)
            sell_cols.append(sell_col)
        except Exception:
            pass
    
    # 复合特征
    if buy_cols:
        features['strat_total_bull'] = features[buy_cols].sum(axis=1)
        features['strat_total_bear'] = features[sell_cols].sum(axis=1)
        n_strats = len(buy_cols)
        features['strat_consensus'] = (
            features['strat_total_bull'] - features['strat_total_bear']
        ) / n_strats
        features['strat_diversity'] = (
            (features[buy_cols].sum(axis=1) > 0).astype(float) +
            (features[sell_cols].sum(axis=1) > 0).astype(float)
        ) / (2 * n_strats)
    
    return features.fillna(0)


STRATEGY_FEATURE_NAMES = (
    [f'strat_{k}_buy' for k in BUILTIN_STRATEGIES] +
    [f'strat_{k}_sell' for k in BUILTIN_STRATEGIES] +
    ['strat_total_bull', 'strat_total_bear', 'strat_consensus', 'strat_diversity']
)


if __name__ == '__main__':
    print("🔬 Strategy Backtester\n")
    
    for market in ['US', 'CN']:
        print(f"\n{'='*70}")
        print(f"  {market} Market — {len(BUILTIN_STRATEGIES)} strategies")
        print(f"{'='*70}")
        results = run_all_builtin_backtests(market=market, n_stocks=20, days=300)
        
        if not results.empty:
            print(f"\n🏆 Top by Sharpe:")
            for _, row in results.head(5).iterrows():
                print(f"  {row['name']:<35} Sharpe={row['avg_sharpe']:+.2f}  Ret={row['avg_return']:+.1f}%")

