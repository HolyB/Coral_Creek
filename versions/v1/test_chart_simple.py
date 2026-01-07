#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试图表功能和周期切换"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("图表功能测试")
print("=" * 60)

# 测试1: 检查模块导入
print("\n测试1: 检查模块导入")
try:
    from data_fetcher import get_stock_data
    from chart_utils import create_candlestick_chart, calculate_blue_signal
    import plotly.graph_objects as go
    print("[OK] 所有模块导入成功")
except Exception as e:
    print(f"[ERROR] 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 测试数据获取
print("\n测试2: 测试数据获取")
test_symbol = "AAPL"
test_market = "US"

try:
    print(f"正在获取 {test_symbol} 的历史数据...")
    hist_data = get_stock_data(test_symbol, market=test_market, days=90)
    
    if hist_data is not None and not hist_data.empty:
        print(f"[OK] 数据获取成功: {len(hist_data)} 条记录")
        print(f"   日期范围: {hist_data.index[0]} 到 {hist_data.index[-1]}")
        print(f"   最新价格: {hist_data['Close'].iloc[-1]:.2f}")
    else:
        print("[ERROR] 数据获取失败: 返回空数据")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] 数据获取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 测试BLUE信号计算
print("\n测试3: 测试BLUE信号计算")
try:
    OPEN = hist_data['Open'].values
    HIGH = hist_data['High'].values
    LOW = hist_data['Low'].values
    CLOSE = hist_data['Close'].values
    
    blue_signal = calculate_blue_signal(OPEN, HIGH, LOW, CLOSE)
    print(f"[OK] BLUE信号计算成功: {len(blue_signal)} 个值")
    print(f"   BLUE范围: {blue_signal.min():.2f} - {blue_signal.max():.2f}")
    print(f"   BLUE>100的数量: {(blue_signal > 100).sum()}")
except Exception as e:
    print(f"[ERROR] BLUE信号计算失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 测试图表创建（日线）
print("\n测试4: 测试图表创建（日线）")
try:
    test_day_dates = [
        {"date": hist_data.index[-5].strftime('%Y-%m-%d'), "value": 150.5},
        {"date": hist_data.index[-3].strftime('%Y-%m-%d'), "value": 165.2}
    ]
    
    fig = create_candlestick_chart(
        hist_data,
        test_symbol,
        "Apple Inc.",
        period='daily',
        day_blue_dates=test_day_dates,
        week_blue_dates=None,
        heima_dates=None
    )
    print("[OK] 日线图表创建成功")
    print(f"   图表类型: {type(fig)}")
    print(f"   数据点数量: {len(fig.data)}")
except Exception as e:
    print(f"[ERROR] 日线图表创建失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 测试图表创建（周线）
print("\n测试5: 测试图表创建（周线）")
try:
    weekly_data = hist_data.resample('W-MON').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    print(f"   周线数据: {len(weekly_data)} 周")
    
    if len(weekly_data) > 1:
        test_week_dates = [
            {"date": weekly_data.index[-2].strftime('%Y-%m-%d'), "value": 180.3}
        ]
        
        fig = create_candlestick_chart(
            hist_data,
            test_symbol,
            "Apple Inc.",
            period='weekly',
            day_blue_dates=None,
            week_blue_dates=test_week_dates,
            heima_dates=None
        )
        print("[OK] 周线图表创建成功")
    else:
        print("[SKIP] 周线数据不足，跳过测试")
except Exception as e:
    print(f"[ERROR] 周线图表创建失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 测试图表创建（月线）
print("\n测试6: 测试图表创建（月线）")
try:
    monthly_data = hist_data.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    print(f"   月线数据: {len(monthly_data)} 月")
    
    fig = create_candlestick_chart(
        hist_data,
        test_symbol,
        "Apple Inc.",
        period='monthly',
        day_blue_dates=None,
        week_blue_dates=None,
        heima_dates=None
    )
    print("[OK] 月线图表创建成功")
except Exception as e:
    print(f"[ERROR] 月线图表创建失败: {e}")
    import traceback
    traceback.print_exc()

# 测试7: 测试数据转换
print("\n测试7: 测试数据转换")
try:
    daily_count = len(hist_data)
    weekly_count = len(hist_data.resample('W-MON').agg({'Close': 'last'}).dropna())
    monthly_count = len(hist_data.resample('M').agg({'Close': 'last'}).dropna())
    
    print(f"[OK] 数据转换成功:")
    print(f"   日线: {daily_count} 条")
    print(f"   周线: {weekly_count} 条")
    print(f"   月线: {monthly_count} 条")
except Exception as e:
    print(f"[ERROR] 数据转换失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)


