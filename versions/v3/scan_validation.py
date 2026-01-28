import sys
import os
import pandas as pd
from scanner import analyze_stock

def main():
    # 测试列表: 包含之前的争议股票和典型代表
    targets = ['CSCO', 'NVDA', 'AAPL', 'TSLA', 'MCGAU', 'COE'] 
    
    print(f"🚀 Running Validation Scan on: {targets}")
    
    results = []
    for t in targets:
        print(f"Scanning {t}...")
        try:
            res = analyze_stock(t, market='US')
            if res:
                results.append(res)
            else:
                print(f"  -> No signal or insufficient data for {t}")
        except Exception as e:
            print(f"  -> Error: {e}")

    if results:
        df = pd.DataFrame(results)
        print("\n🏆 Validation Results:")
        cols = ['Symbol', 'Price', 'Blue_Daily', 'Adaptive_Thresh', 'Strat_D_Trend', 'Regime', 'Chan_Signal']
        # 确保列存在
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
    else:
        print("No results found.")

if __name__ == "__main__":
    main()

