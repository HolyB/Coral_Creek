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

# 导入工具模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from data_fetcher import get_stock_data, get_all_us_tickers
    from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_kdj_series, calculate_volume_profile_metrics, calculate_atr_series, calculate_adx_series, analyze_elliott_wave_proxy
except ImportError:
    # 兼容性导入
    from data_fetcher import get_stock_data, get_all_us_tickers
    from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_kdj_series, calculate_volume_profile_metrics, calculate_atr_series
    
    # 简单的 mock 函数，防止导入失败
    def analyze_elliott_wave_proxy(closes, highs, lows):
        return {'phase': 'N/A', 'desc': 'N/A'}
        
    # 本地定义 calculate_adx_series 如果导入失败
    def calculate_adx_series(high, low, close, period=14):
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        tr = calculate_atr_series(high, low, close, period=1)
        up_move = high_s - high_s.shift(1)
        down_move = pd.Series(low).shift(1) - pd.Series(low)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        tr_smooth = tr_smooth.replace(0, np.nan)
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.values

def analyze_stock(symbol, market='US', account_size=100000):
    """
    深度分析单只股票：计算信号、策略匹配、风控建议
    """
    try:
        # 1. 获取数据 (365天足够日线分析，周线需更多但为了速度折中)
        df = get_stock_data(symbol, market, days=730)
        if df is None or len(df) < 60:
            return None
            
        # 2. 计算指标
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        volumes = df['Volume'].values
        
        # 基础指标
        day_blue = calculate_blue_signal_series(opens, highs, lows, closes)
        heima, juedi = calculate_heima_signal_series(highs, lows, closes, opens)
        atr = calculate_atr_series(highs, lows, closes, period=14)
        adx = calculate_adx_series(highs, lows, closes)
        
        # 获取最新一天的值
        curr_blue = day_blue[-1]
        curr_heima = heima[-1]
        curr_juedi = juedi[-1]
        curr_atr = atr[-1]
        curr_adx = adx[-1]
        curr_price = closes[-1]
        curr_vol = volumes[-1] * curr_price # 成交额
        
        # 3. 深度体检 (提前计算以确定自适应阈值)
        log_ret = np.log(pd.Series(closes) / pd.Series(closes).shift(1))
        volatility = log_ret.tail(252).std() * np.sqrt(252) if len(log_ret) > 252 else 0.3 # 默认中等
        
        # 自适应阈值判定 V2.0 (基于 Grid Search 数据优化)
        adaptive_threshold = 100 # 默认标准
        regime_desc = "Standard"
        
        # 规则 1: 中低波动 (NVDA, TSLA, META 级别) -> 80 是黄金分割点
        if volatility < 0.35:
            adaptive_threshold = 80
            regime_desc = "Mid-Low Vol"
            
        # 规则 2: 极低波动 (AAPL, GOOGL, KO 级别) -> 必须放宽到 60
        if volatility < 0.20:
            adaptive_threshold = 60
            regime_desc = "Low Vol (稳健)"
            
        # 规则 3: 强趋势 (Trend Following) -> 只要在趋势中，80 即可确认
        if curr_adx > 25:
            adaptive_threshold = min(adaptive_threshold, 80)
            regime_desc += " | 强趋势"
            
        # 规则 4: 妖股 (High Vol) -> 必须强力突破才算
        if volatility > 0.60:
            adaptive_threshold = 110
            regime_desc = "High Vol (妖股)"
            
        # 4. 策略判定
        
        # [Strategy D] 激进趋势: 使用自适应阈值
        is_strat_d = (curr_blue > adaptive_threshold)
        
        # [Strategy C] 宽松共振: (BLUE OR Heima) + Context
        recent_blues = day_blue[-5:]
        recent_heimas = (heima[-5:] | juedi[-5:])
        
        # 共振中的 BLUE 也使用自适应阈值
        has_recent_blue = np.any(recent_blues > adaptive_threshold)
        has_recent_heima = np.any(recent_heimas)
        
        is_strat_c = False
        if (curr_blue > adaptive_threshold and has_recent_heima) or ((curr_heima or curr_juedi) and has_recent_blue):
            is_strat_c = True
            
        # [Legacy] 旧版信号: 严格 100
        is_legacy = (curr_blue > 100)
        
        # 如果没有任何策略命中，直接返回 None (节省空间)
        if not (is_strat_d or is_strat_c):
            return None
            
        # 风控参数计算
        stop_mult = 2.0
        risk_pct = 0.02
        
        if "High Vol" in regime_desc:
            stop_mult = 3.5
            risk_pct = 0.01
        elif "Low Vol" in regime_desc:
            stop_mult = 1.8
            risk_pct = 0.03
            
        if curr_adx > 30:
            stop_mult += 0.5 # 趋势中放宽止损
            
        # 周线确认 (Optional Context)
        df_weekly = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
        week_blue_val = 0
        if not df_weekly.empty:
            w_blue = calculate_blue_signal_series(
                df_weekly['Open'].values, df_weekly['High'].values, 
                df_weekly['Low'].values, df_weekly['Close'].values
            )
            week_blue_val = w_blue[-1]
            
        # Volume Profile (VP)
        vp_res = calculate_volume_profile_metrics(closes, volumes, curr_price)
        vp_rating = "Normal"
        if vp_res['profit_ratio'] > 0.9: vp_rating = "Excellent"
        elif vp_res['profit_ratio'] < 0.1: vp_rating = "Poor"
        elif vp_res['price_pos'] == 'Above': vp_rating = "Good"
        
        # 5. 波浪形态识别 (Elliott Wave Proxy)
        wave_res = analyze_elliott_wave_proxy(closes, highs, lows)
        
        # 6. 风控建议
        stop_loss_price = curr_price - (stop_mult * curr_atr)
        risk_amt = account_size * risk_pct
        shares = int(risk_amt / (stop_mult * curr_atr)) if curr_atr > 0 else 0
        
        return {
            'Symbol': symbol,
            'Price': curr_price,
            'Turnover_M': round(curr_vol / 1000000, 2),
            'Date': df.index[-1].strftime('%Y-%m-%d'),
            
            # 信号
            'Blue_Daily': round(curr_blue, 1),
            'Adaptive_Thresh': adaptive_threshold, # 显示当前用的阈值
            'Blue_Weekly': round(week_blue_val, 1),
            'Is_Heima': curr_heima or curr_juedi,
            
            # 策略标签
            'Strat_D_Trend': is_strat_d,
            'Strat_C_Resonance': is_strat_c,
            'Legacy_Signal': is_legacy,
            
            # 深度分析
            'Regime': regime_desc,
            'Volatility': round(volatility, 2),
            'ADX': round(curr_adx, 1),
            'VP_Rating': vp_rating,
            'Profit_Ratio': vp_res['profit_ratio'],
            'Wave_Phase': wave_res['phase'],
            'Wave_Desc': wave_res['desc'],
            
            # 执行建议
            'Stop_Loss': round(stop_loss_price, 2),
            'Shares_Rec': shares, # 基于10W账户
            'Risk_Reward_Score': round(curr_adx * (1-volatility), 2)
        }
        
    except Exception as e:
        # print(f"Error analyzing {symbol}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Enhanced Stock Scanner V2.0')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of tickers (0 for all)')
    parser.add_argument('--workers', type=int, default=20, help='Parallel workers')
    parser.add_argument('--market', type=str, default='US', help='Market (US/CN)')
    args = parser.parse_args()
    
    print(f"🚀 Starting Enhanced Scan (Market: {args.market})...")
    print(f"⚙️  Strategy: Adaptive Thresholds & Risk Management")
    
    # 1. 获取股票列表
    if args.market == 'US':
        tickers = get_all_us_tickers()
        if not tickers:
            print("Failed to fetch tickers.")
            return
    else:
        print("CN market not fully supported in auto-fetch yet.")
        return

    if args.limit > 0:
        tickers = tickers[:args.limit]
        
    print(f"📋 Scanning {len(tickers)} stocks...")
    
    results = []
    start_time = time.time()
    
    # 2. 并发扫描
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {executor.submit(analyze_stock, symbol, args.market): symbol for symbol in tickers}
        
        for future in tqdm(as_completed(future_to_symbol), total=len(tickers), unit="stock"):
            res = future.result()
            if res:
                results.append(res)
                
    elapsed = time.time() - start_time
    print(f"\n✅ Scan complete in {elapsed:.1f}s. Found {len(results)} candidates.")
    
    if not results:
        print("No signals found.")
        return
        
    # 3. 保存结果
    df = pd.DataFrame(results)
    
    # 排序：优先展示 Strat C (共振)，然后按 ADX 排序
    df.sort_values(by=['Strat_C_Resonance', 'ADX'], ascending=[False, False], inplace=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_scan_results_{args.market}_{timestamp}.csv"
    output_path = os.path.join(current_dir, filename)
    
    df.to_csv(output_path, index=False)
    print(f"💾 Results saved to: {filename}")
    
    # 4. 打印精华预览 (Top 10)
    print("\n🏆 Top 10 Candidates (Sorted by Resonance & Trend):")
    # 显示 Adaptive Thresh 以便验证
    cols = ['Symbol', 'Price', 'Blue_Daily', 'Adaptive_Thresh', 'Strat_D_Trend', 'Regime', 'Shares_Rec']
    print(df[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
