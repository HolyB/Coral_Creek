import pandas as pd
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# 导入回测器
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from simple_backtest import SimpleBacktester

def test_single_threshold(symbol, threshold):
    try:
        # 关闭自适应风控以测试纯信号质量，或者开启以测试实战效果
        # 这里为了找“最佳信号阈值”，我们开启风控，因为这是最终使用场景
        bt = SimpleBacktester(
            symbol=symbol,
            market='US',
            initial_capital=100000,
            days=1095, # 3年
            blue_threshold=threshold,
            strategy_mode='D', # 激进趋势模式
            use_risk_management=True 
        )
        
        if not bt.load_data(): return None
        bt.calculate_signals()
        bt.run_backtest()
        
        res = bt.results
        return {
            'Symbol': symbol,
            'Threshold': threshold,
            'Return': res['Total Return'],
            'Drawdown': res['Max Drawdown'],
            'WinRate': res['Win Rate'],
            'Trades': res['Total Trades']
        }
    except Exception as e:
        return None

def main():
    targets = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    thresholds = [50, 60, 70, 80, 90, 100, 110, 120]
    
    print(f"🚀 Optimizing BLUE Thresholds for Mega Caps...")
    print(f"🎯 Targets: {', '.join(targets)}")
    print(f"🎚️  Range: {thresholds}")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for symbol in targets:
            for thresh in thresholds:
                futures.append(executor.submit(test_single_threshold, symbol, thresh))
                
        for f in futures:
            res = f.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    # 分析结果
    print("\n" + "="*60)
    print("🏆 BEST THRESHOLD PER STOCK")
    print("="*60)
    
    for symbol in targets:
        stock_df = df[df['Symbol'] == symbol]
        if stock_df.empty: continue
        
        # 找收益最高的
        best_ret = stock_df.loc[stock_df['Return'].idxmax()]
        # 找夏普最优 (简单用 Return / |DD|)
        stock_df['Score'] = stock_df['Return'] / stock_df['Drawdown'].abs()
        best_sharpe = stock_df.loc[stock_df['Score'].idxmax()]
        
        print(f"\n📌 {symbol}:")
        print(f"   Max Return: Thresh={best_ret['Threshold']} -> {best_ret['Return']:.2%} (DD: {best_ret['Drawdown']:.2%})")
        print(f"   Best Risk/Reward: Thresh={best_sharpe['Threshold']} -> {best_sharpe['Return']:.2%} (DD: {best_sharpe['Drawdown']:.2%})")
        
        # 打印局部详情
        # print(stock_df[['Threshold', 'Return', 'Drawdown']].to_string(index=False))

if __name__ == "__main__":
    main()



