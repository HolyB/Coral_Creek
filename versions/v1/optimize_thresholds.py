import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from simple_backtest import SimpleBacktester
except ImportError:
    print("Error: Could not import SimpleBacktester.")
    sys.exit(1)

def optimize_thresholds():
    # 1. 定义测试样本
    test_stocks = [
        # 美股 - 科技巨头 (万亿市值)
        {'symbol': 'NVDA', 'market': 'US', 'type': 'Mega Tech'},
        {'symbol': 'AAPL', 'market': 'US', 'type': 'Mega Tech'},
        {'symbol': 'MSFT', 'market': 'US', 'type': 'Mega Tech'},
        
        # 美股 - 高波动/成长
        {'symbol': 'TSLA', 'market': 'US', 'type': 'High Volatility'},
        {'symbol': 'AMD',  'market': 'US', 'type': 'High Volatility'},
        {'symbol': 'COIN', 'market': 'US', 'type': 'High Volatility'},
        
        # 美股 - 传统价值/低波动
        {'symbol': 'KO',   'market': 'US', 'type': 'Defensive'},
        {'symbol': 'JNJ',  'market': 'US', 'type': 'Defensive'},
        
        # A股
        {'symbol': '600519.SH', 'market': 'CN', 'type': 'CN Blue Chip'}, # 茅台
        {'symbol': '300750.SZ', 'market': 'CN', 'type': 'CN Growth'},    # 宁德
    ]
    
    # 2. 定义参数网格
    thresholds = [50, 60, 70, 80, 90, 100, 110, 120, 130, 150]
    
    results = []
    
    print("开始参数优化测试 (Grid Search)...")
    print(f"测试股票数: {len(test_stocks)}, 阈值组合数: {len(thresholds)}")
    
    for stock in test_stocks:
        symbol = stock['symbol']
        market = stock['market']
        category = stock['type']
        
        print(f"\nProcessing {symbol} ({category})...")
        
        # 预加载数据，避免重复请求
        # 临时创建一个 backtester 只是为了加载数据
        loader = SimpleBacktester(symbol, market, days=1095) # 3年
        if not loader.load_data():
            print(f"Skipping {symbol} due to data load failure.")
            continue
            
        data_cache = loader.df
        
        best_return = -999
        best_threshold = 0
        best_stats = {}
        
        for th in tqdm(thresholds, desc=f"Testing {symbol}", leave=False):
            # 每次回测创建一个新实例，但注入缓存的数据
            bt = SimpleBacktester(symbol, market, initial_capital=100000, days=1095, blue_threshold=th)
            bt.df = data_cache.copy() # 注入数据
            
            bt.calculate_signals()
            bt.run_backtest()
            
            res = bt.results
            total_ret = res['Total Return']
            
            # 记录结果
            results.append({
                'symbol': symbol,
                'type': category,
                'threshold': th,
                'return': total_ret,
                'drawdown': res['Max Drawdown'],
                'trades': res['Total Trades'],
                'win_rate': res['Win Rate']
            })
            
            if total_ret > best_return:
                best_return = total_ret
                best_threshold = th
                best_stats = res
        
        print(f"  -> Best Threshold: {best_threshold} (Return: {best_return:.2%}, Trades: {best_stats['Total Trades']})")

    # 3. 分析结果
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("优化结果汇总 (按股票类型)")
    print("="*60)
    
    # 找出每只股票的最佳配置
    best_configs = df_res.loc[df_res.groupby('symbol')['return'].idxmax()].sort_values('type')
    
    print(best_configs[['symbol', 'type', 'threshold', 'return', 'drawdown', 'trades', 'win_rate']].to_string(index=False))
    
    print("\n" + "="*60)
    print("类型平均最佳阈值分析")
    print("="*60)
    avg_thresholds = best_configs.groupby('type')['threshold'].mean().sort_values()
    print(avg_thresholds)
    
    # 保存详细结果
    df_res.to_csv('threshold_optimization_results.csv', index=False)
    print("\n详细结果已保存至 'threshold_optimization_results.csv'")

if __name__ == "__main__":
    optimize_thresholds()
