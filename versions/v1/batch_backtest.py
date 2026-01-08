import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime

# 导入核心回测类
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from simple_backtest import SimpleBacktester

def load_tickers(limit=1000):
    tickers = []
    cache_path = os.path.join(current_dir, 'tickers_cache.json')
    txt_path = os.path.join(current_dir, 'us_tickers.txt')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): tickers = data.get('US', [])
                elif isinstance(data, list): tickers = data
        except: pass
    if not tickers and os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'INTC', 'NFLX', 'PYPL', 'ADBE', 'CRM', 'ABNB', 'UBER']
    return tickers[:limit]

def run_single_backtest(symbol, strategy_mode, use_risk_management=True):
    try:
        # 根据策略模式配置参数
        require_heima = True
        require_week_blue = False
        require_vp = False
        
        # 策略 A: 严谨型 (双蓝+黑马+VP)
        if strategy_mode == 'A':
            require_heima = True
            require_week_blue = True
            require_vp = True
            
        # 策略 B: 平衡型 (日蓝+黑马+VP)
        elif strategy_mode == 'B':
            require_heima = True
            require_week_blue = False # 代码逻辑中会强制检查 has_heima_context
            require_vp = True
            
        # 策略 C: 宽松型 (BLUE OR Heima)
        elif strategy_mode == 'C':
            require_heima = True # 代码逻辑是 Loose Resonance
            require_week_blue = False
            require_vp = False
            
        # 策略 D: 激进型 (纯趋势)
        elif strategy_mode == 'D':
            require_heima = False # 只看 BLUE
            require_week_blue = False
            require_vp = False

        backtester = SimpleBacktester(
            symbol=symbol,
            market='US',
            initial_capital=100000,
            days=1095, 
            blue_threshold=100,
            strategy_mode=strategy_mode, # 传入策略模式
            require_heima=require_heima,
            require_week_blue=require_week_blue,
            require_vp_filter=require_vp,
            use_risk_management=use_risk_management
        )
        
        if not backtester.load_data():
            return None
            
        backtester.calculate_signals()
        backtester.run_backtest()
        
        res = backtester.results
        yearly = res.get('Yearly Returns', {})
        
        return {
            'Symbol': symbol,
            'Strategy': strategy_mode,
            'Return': res['Total Return'],
            'Max_Drawdown': res['Max Drawdown'],
            'Win_Rate': res['Win Rate'],
            'Trades': res['Total Trades'],
            '2023_Return': yearly.get(2023, 0),
            '2024_Return': yearly.get(2024, 0),
            '2025_Return': yearly.get(2025, 0)
        }
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description='Multi-Strategy Batch Backtest')
    parser.add_argument('--limit', type=int, default=50, help='Number of stocks')
    parser.add_argument('--workers', type=int, default=10, help='Workers')
    args = parser.parse_args()
    
    tickers = load_tickers(args.limit)
    print(f"📋 Loaded {len(tickers)} tickers.")
    
    # 定义要回测的策略集合
    strategies = {
        'A': '严谨型 (双蓝共振+黑马+VP)',
        'B': '平衡型 (日蓝+黑马+VP)',
        'C': '宽松型 (BLUE OR Heima)',
        'D': '激进型 (纯日线BLUE趋势)'
    }
    
    all_results = []
    summary_data = []
    
    print(f"🚀 Starting Multi-Strategy Tournament...")
    
    for strat_code, strat_desc in strategies.items():
        print(f"\n⚔️  Running Strategy {strat_code}: {strat_desc}")
        strat_results = []
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_symbol = {executor.submit(run_single_backtest, symbol, strat_code): symbol for symbol in tickers}
            for future in tqdm(as_completed(future_to_symbol), total=len(tickers), unit="stock"):
                res = future.result()
                if res:
                    strat_results.append(res)
                    all_results.append(res)
        
        # 计算该策略的平均表现
        if strat_results:
            df_strat = pd.DataFrame(strat_results)
            avg_ret = df_strat['Return'].mean()
            avg_dd = df_strat['Max_Drawdown'].mean()
            win_chance = len(df_strat[df_strat['Return'] > 0]) / len(df_strat)
            avg_trade_win = df_strat['Win_Rate'].mean()
            
            summary_data.append({
                'Strategy': strat_code,
                'Description': strat_desc,
                'Avg Return': avg_ret,
                'Max Drawdown': avg_dd,
                'Win Chance': win_chance,
                'Trade Win Rate': avg_trade_win,
                'Avg Trades': df_strat['Trades'].mean()
            })
            
    # === 最终对比报告 ===
    print("\n" + "="*80)
    print(f"🏆 STRATEGY TOURNAMENT RESULTS (N={len(tickers)} stocks)")
    print("="*80)
    
    summary_df = pd.DataFrame(summary_data)
    # 格式化显示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # 按收益率排序
    print(summary_df.sort_values('Avg Return', ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2%}"))
    
    print("-" * 80)
    best_strat = summary_df.loc[summary_df['Avg Return'].idxmax()]
    safest_strat = summary_df.loc[summary_df['Max Drawdown'].idxmax()] # Max Drawdown is negative, so max is closest to 0
    
    print(f"🌟 Best Yield Strategy : {best_strat['Strategy']} ({best_strat['Description']}) -> Return: {best_strat['Avg Return']:.2%}")
    print(f"🛡️ Safest Strategy     : {safest_strat['Strategy']} ({safest_strat['Description']}) -> Max DD: {safest_strat['Max Drawdown']:.2%}")
    
    # 保存明细
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(current_dir, f"multi_strat_results_{timestamp}.csv"), index=False)
    print(f"\n💾 Detailed results saved to: multi_strat_results_{timestamp}.csv")

if __name__ == "__main__":
    main()
