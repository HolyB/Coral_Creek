#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略第一性原理验证：大盘环境对个股策略有效性的影响分析
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import get_db

def get_db_path():
    """获取数据库路径"""
    return os.path.join(parent_dir, 'db', 'coral_creek.db')

def load_signals():
    """从本地数据库加载所有已扫描的有效信号"""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"❌ 数据库不存在: {db_path}")
        return pd.DataFrame()
    
    print(f"📡 连接数据库: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # 提取我们关注的策略列
    query = """
    SELECT 
        symbol, scan_date, market, price, 
        blue_daily, is_heima, is_juedi, adx,
        day_close
    FROM scan_results 
    WHERE market = 'US' 
      AND (
          blue_daily >= 150 
          OR is_heima = 1 
          OR is_juedi = 1
      )
    ORDER BY scan_date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("⚠️ 没有加载到信号数据")
        return pd.DataFrame()
    
    df['scan_date'] = pd.to_datetime(df['scan_date'])
    print(f"✅ 加载信号记录: {len(df)} 条 | 日期范围: {df['scan_date'].min().date()} ~ {df['scan_date'].max().date()}")
    return df

def get_market_regime(start_date, end_date):
    """获取 SPY 大盘状态 (Regime) - 使用 polygon-api-client"""
    from polygon import RESTClient
    
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        # 尝试从 .env 读取
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('POLYGON_API_KEY')
        except:
            pass
            
    if not api_key:
        print("❌ 依然找不到 POLYGON_API_KEY")
        return pd.DataFrame()

    # 稍微多拉一点数据算均线
    start_dt = pd.to_datetime(start_date) - timedelta(days=300) # 200日均线需要很长历史
    end_dt = pd.to_datetime(end_date) + timedelta(days=10)
    
    s_str = start_dt.strftime('%Y-%m-%d')
    e_str = end_dt.strftime('%Y-%m-%d')
    print(f"📊 拉取 SPY 数据 (Polygon Client) {s_str} ~ {e_str}...")
    
    try:
        client = RESTClient(api_key)
        aggs = client.get_aggs("SPY", 1, "day", s_str, e_str, limit=50000)
        
        records = []
        for agg in aggs:
            # timestamp is ms
            dt = datetime.fromtimestamp(agg.timestamp / 1000)
            records.append({
                'Date': dt,
                'Close': float(agg.close),
                'Open': float(agg.open),
                'High': float(agg.high),
                'Low': float(agg.low),
                'Volume': float(agg.volume)
            })
            
        if not records:
             print("❌ Polygon 返回空数据")
             return pd.DataFrame()
             
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 计算大盘指标
        df['SPY_MA20'] = df['Close'].rolling(window=20).mean()
        df['SPY_MA50'] = df['Close'].rolling(window=50).mean()
        df['SPY_MA200'] = df['Close'].rolling(window=200).mean()
        
        # 定义环境 (Regime)
        # 1: 强势 (Bull) - 价格 > MA20
        # -1: 弱势 (Bear) - 价格 < MA20
        df['SPY_Regime_Short'] = np.where(df['Close'] > df['SPY_MA20'], 'Bull', 'Bear')
        
        # 长期趋势
        df['SPY_Regime_Long'] = np.where(df['Close'] > df['SPY_MA200'], 'Bull', 'Bear')
        
        # 当日涨跌
        df['SPY_Ret'] = df['Close'].pct_change()
        
        # 截取回我们需要的时间段
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        final_df = df.loc[mask].copy()
        
        # 日期对齐（忽略时分秒）
        final_df['Date'] = final_df['Date'].dt.normalize()
        
        return final_df
        
    except Exception as e:
        print(f"❌ 拉取 SPY 失败: {e}")
        return pd.DataFrame()

def calculate_forward_returns(signals_df, market_df):
    """计算每个信号的未来收益，并合并大盘状态"""
    print("🧮 计算信号的一致性分析...")
    
    # 关联大盘状态
    signals_df = signals_df.merge(market_df, left_on='scan_date', right_on='Date', how='inner')
    
    # 这里我们简化处理：不重新去拉每只个股的未来价格（太慢了）
    # 而是只统计**大盘环境分布**
    # 如果要精确验证收益，确实需要个股行情。
    # 我们可以尝试用 yfinance 批量拉取部分热门股的行情来验证
    
    # 统计策略在不同环境下的出现频率
    return signals_df

def analyze_strategies(df):
    """分组统计各策略在大盘不同状态下的分布"""
    strategies = {
        'Blue_Breakout': df['blue_daily'] > 180,
        'Blue_Trend': (df['blue_daily'] >= 150) & (df['blue_daily'] <= 180),
        'Heima': df['is_heima'] == 1,
        'Juedi': df['is_juedi'] == 1
    }
    
    results = []
    
    for name, mask in strategies.items():
        sub = df[mask]
        if sub.empty:
            continue
            
        total = len(sub)
        
        # 在 SPY > MA20 (短期强势) 时发出的信号数量
        bull_short = len(sub[sub['SPY_Regime_Short'] == 'Bull'])
        bear_short = len(sub[sub['SPY_Regime_Short'] == 'Bear'])
        
        # 在 SPY > MA200 (长期牛市) 时发出的信号数量
        bull_long = len(sub[sub['SPY_Regime_Long'] == 'Bull'])
        bear_long = len(sub[sub['SPY_Regime_Long'] == 'Bear'])
        
        # 在 SPY 当日上涨/下跌时的分布
        spy_up = len(sub[sub['SPY_Ret'] > 0])
        spy_down = len(sub[sub['SPY_Ret'] <= 0])
        
        results.append({
            'Strategy': name,
            'Total_Signals': total,
            'Bull_Environment% (MA20)': f"{bull_short / total * 100:.1f}%",
            'Bear_Environment% (MA20)': f"{bear_short / total * 100:.1f}%",
            'Long_Bull% (MA200)': f"{bull_long / total * 100:.1f}%",
            'SPY_Up_Day%': f"{spy_up / total * 100:.1f}%"
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📈 策略信号与大盘环境的相关性分析 (Signal-Regime Correlation)")
    print("="*80)
    print("这个表告诉你：你的策略发出的信号，有多少是顺大盘势的？")
    print("如果 Bear% 很高，说明策略经常在熊市/回调中试图'接飞刀'。\n")
    print(res_df.to_string(index=False))
    print("\n")
    
    return res_df

if __name__ == "__main__":
    signals = load_signals()
    if not signals.empty:
        start_date = signals['scan_date'].min()
        end_date = signals['scan_date'].max()
        
        market_data = get_market_regime(start_date, end_date)
        if not market_data.empty:
            analyzed = calculate_forward_returns(signals, market_data)
            analyze_strategies(analyzed)
        else:
            print("无法进行分析 (缺少大盘数据)")
    else:
        print("无法进行分析 (缺少信号数据)")
