#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试yfinance是否能正常工作"""
import yfinance as yf
import sys

print("测试yfinance连接...")
print("=" * 60)

# 测试单个股票
test_tickers = ['AAPL', 'MSFT', 'TSLA']

for ticker in test_tickers:
    print(f"\n测试 {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        print(f"  Ticker对象创建成功")
        
        # 尝试获取信息
        try:
            info = stock.info
            if info:
                print(f"  股票信息获取成功: {info.get('longName', 'N/A')}")
            else:
                print(f"  股票信息为空")
        except Exception as e:
            print(f"  获取股票信息失败: {e}")
        
        # 尝试获取历史数据
        try:
            df = stock.history(period="1mo")  # 先测试1个月
            if not df.empty:
                print(f"  历史数据获取成功: {len(df)}行")
                print(f"  最新价格: {df['Close'].iloc[-1]:.2f}")
            else:
                print(f"  历史数据为空")
        except Exception as e:
            print(f"  获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"  处理 {ticker} 时出错: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")



