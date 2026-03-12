#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫描服务 - 支持指定日期扫描，结果写入数据库
"""
import os
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db.database import (
    insert_scan_result, bulk_insert_scan_results, 
    create_scan_job, update_scan_job, get_scan_job,
    init_db
)
from data_fetcher import get_us_stock_data, get_all_us_tickers, get_ticker_details, get_cn_stock_data, get_all_cn_tickers, get_cn_ticker_details
from indicator_utils import (
    calculate_blue_signal_series, calculate_heima_signal_series,
    calculate_adx_series, calculate_volume_profile_metrics,
    calculate_volatility, analyze_elliott_wave_proxy, analyze_chanlun_proxy
)


def analyze_stock_for_date(symbol, target_date, market='US'):
    """
    分析单只股票在指定日期的信号
    
    Args:
        symbol: 股票代码
        target_date: 目标日期 (str: 'YYYY-MM-DD' 或 datetime)
        market: 市场 ('US' 或 'CN')
    
    Returns:
        dict: 分析结果，如果无信号则返回 None
    """
    try:
        if isinstance(target_date, str):
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        else:
            target_dt = target_date
        
        # 获取历史数据 (目标日期之前的数据)
        # 需要足够长的历史数据来计算指标
        days_needed = 3650  # 10年
        
        if market == 'US':
            df = get_us_stock_data(symbol, days=days_needed)
        elif market == 'CN':
            df = get_cn_stock_data(symbol, days=days_needed)
        else:
            return None
        
        if df is None or len(df) < 60:
            return None
        
        # 截取到目标日期的数据
        df = df[df.index <= target_dt]
        
        if len(df) < 60:
            return None
        
        # 获取目标日期的数据 (最后一行)
        latest_date = df.index[-1]
        
        # 计算指标
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        volumes = df['Volume'].values
        
        curr_price = closes[-1]
        
        # 日线 BLUE
        day_blue = calculate_blue_signal_series(opens, highs, lows, closes)
        day_blue_val = day_blue[-1] if len(day_blue) > 0 else 0
        
        # 周线
        df_weekly = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        week_blue_val = 0
        if len(df_weekly) >= 10:
            week_blue = calculate_blue_signal_series(
                df_weekly['Open'].values, df_weekly['High'].values,
                df_weekly['Low'].values, df_weekly['Close'].values
            )
            week_blue_val = week_blue[-1] if len(week_blue) > 0 else 0
        
        # 月线
        df_monthly = df.resample('ME').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
        
        month_blue_val = 0
        if len(df_monthly) >= 6:
            month_blue = calculate_blue_signal_series(
                df_monthly['Open'].values, df_monthly['High'].values,
                df_monthly['Low'].values, df_monthly['Close'].values
            )
            month_blue_val = month_blue[-1] if len(month_blue) > 0 else 0
        
        # 黑马/掘地信号 - 日线
        heima_arr, juedi_arr = calculate_heima_signal_series(highs, lows, closes, opens)
        curr_heima = heima_arr[-1] if len(heima_arr) > 0 else False
        curr_juedi = juedi_arr[-1] if len(juedi_arr) > 0 else False
        
        # 黑马/掘地信号 - 周线
        heima_weekly = False
        juedi_weekly = False
        if len(df_weekly) >= 10:
            w_heima, w_juedi = calculate_heima_signal_series(
                df_weekly['High'].values, df_weekly['Low'].values,
                df_weekly['Close'].values, df_weekly['Open'].values
            )
            heima_weekly = w_heima[-1] if len(w_heima) > 0 else False
            juedi_weekly = w_juedi[-1] if len(w_juedi) > 0 else False
        
        # 黑马/掘地信号 - 月线
        heima_monthly = False
        juedi_monthly = False
        if len(df_monthly) >= 6:
            m_heima, m_juedi = calculate_heima_signal_series(
                df_monthly['High'].values, df_monthly['Low'].values,
                df_monthly['Close'].values, df_monthly['Open'].values
            )
            heima_monthly = m_heima[-1] if len(m_heima) > 0 else False
            juedi_monthly = m_juedi[-1] if len(m_juedi) > 0 else False
        
        # ADX
        adx_series = calculate_adx_series(highs, lows, closes)
        adx_val = adx_series[-1] if len(adx_series) > 0 else 0
        
        # 波动率
        volatility = calculate_volatility(closes)
        
        # 自适应阈值
        adaptive_threshold = 80
        if volatility > 0.6:
            adaptive_threshold = 110
        elif volatility > 0.4:
            adaptive_threshold = 100
        elif volatility < 0.2:
            adaptive_threshold = 70
        
        if adx_val > 30:
            adaptive_threshold = max(adaptive_threshold - 10, 60)
        
        # 策略判断
        is_strat_d = day_blue_val >= adaptive_threshold
        is_strat_c = (day_blue_val >= adaptive_threshold * 0.8) and (curr_heima or curr_juedi or week_blue_val > 100)
        legacy_signal = day_blue_val >= 100
        
        # 只保存有信号的股票
        has_signal = is_strat_d or is_strat_c or legacy_signal or week_blue_val > 100
        
        if not has_signal:
            return None
        
        # Volume Profile
        vp_res = calculate_volume_profile_metrics(closes, volumes, curr_price)
        vp_rating = "Normal"
        if vp_res['profit_ratio'] > 0.9:
            vp_rating = "Excellent"
        elif vp_res['profit_ratio'] > 0.7:
            vp_rating = "Good"
        elif vp_res['profit_ratio'] < 0.3:
            vp_rating = "Poor"
        
        # 波浪理论
        wave_result = analyze_elliott_wave_proxy(closes, highs, lows)
        wave_phase = wave_result.get('phase', 'Unknown') if isinstance(wave_result, dict) else str(wave_result)
        wave_desc = wave_result.get('desc', '') if isinstance(wave_result, dict) else ''
        
        # 缠论
        chan_result = analyze_chanlun_proxy(closes, highs, lows)
        chan_signal = chan_result.get('signal', 'N/A') if isinstance(chan_result, dict) else str(chan_result)
        chan_desc = chan_result.get('desc', '') if isinstance(chan_result, dict) else ''
        
        # Regime 分类
        regime = "Standard"
        if volatility > 0.6:
            regime = "High Vol (妖股)"
        elif volatility < 0.25:
            regime = "Mid-Low Vol"
        if adx_val > 30:
            regime += " | 强趋势"
        
        # 成交额
        turnover = (curr_price * volumes[-1]) / 1_000_000 if len(volumes) > 0 else 0
        
        # 止损和仓位
        atr = calculate_atr_for_stop(highs, lows, closes)
        stop_loss = round(curr_price - 2.5 * atr, 2) if atr else curr_price * 0.95
        risk_per_share = curr_price - stop_loss
        shares_rec = int(1000 / risk_per_share) if risk_per_share > 0 else 0
        
        # 风险回报评分
        risk_reward = (vp_res['profit_ratio'] * 100 + adx_val) / 2 if vp_res else adx_val
        
        return {
            'Symbol': symbol,
            'Date': latest_date.strftime('%Y-%m-%d'),
            'Price': round(curr_price, 2),
            'Turnover_M': round(turnover, 2),
            'Blue_Daily': round(day_blue_val, 1),
            'Blue_Weekly': round(week_blue_val, 1),
            'Blue_Monthly': round(month_blue_val, 1),
            'ADX': round(adx_val, 1),
            'Volatility': round(volatility, 2),
            'Is_Heima': curr_heima or curr_juedi,
            'Is_Juedi': curr_juedi,
            'Heima_Daily': bool(curr_heima),
            'Heima_Weekly': bool(heima_weekly),
            'Heima_Monthly': bool(heima_monthly),
            'Juedi_Daily': bool(curr_juedi),
            'Juedi_Weekly': bool(juedi_weekly),
            'Juedi_Monthly': bool(juedi_monthly),
            'Strat_D_Trend': is_strat_d,
            'Strat_C_Resonance': is_strat_c,
            'Legacy_Signal': legacy_signal,
            'Regime': regime,
            'Adaptive_Thresh': adaptive_threshold,
            'VP_Rating': vp_rating,
            'Profit_Ratio': round(vp_res['profit_ratio'], 4) if vp_res else 0,
            'Wave_Phase': wave_phase,
            'Wave_Desc': wave_desc,
            'Chan_Signal': str(chan_signal),
            'Chan_Desc': chan_desc,
            'Market_Cap': None,  # 稍后填充
            'Cap_Category': 'Unknown',
            'Company_Name': None,
            'Industry': None,
            'Stop_Loss': stop_loss,
            'Shares_Rec': shares_rec,
            'Risk_Reward_Score': round(risk_reward, 2),
            'Market': market
        }
        
    except Exception as e:
        return None


def calculate_atr_for_stop(highs, lows, closes, period=14):
    """计算 ATR 用于止损"""
    import numpy as np
    
    if len(closes) < period + 1:
        return None
    
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return None
    
    atr = np.mean(tr_list[-period:])
    return atr


def run_scan_for_date(target_date, market='US', max_workers=30, limit=0, save_to_db=True):
    """
    运行指定日期的扫描
    
    Args:
        target_date: 目标日期 ('YYYY-MM-DD')
        market: 市场
        max_workers: 并行线程数
        limit: 限制股票数量 (0=全部)
        save_to_db: 是否保存到数据库
    
    Returns:
        list: 扫描结果列表
    """
    print(f"\n🚀 Starting scan for date: {target_date} (Market: {market})")
    
    # 创建扫描任务
    if save_to_db:
        create_scan_job(target_date, market)
        update_scan_job(target_date, status='running', started_at=datetime.now().isoformat())
    
    # 获取股票列表
    if market == 'US':
        print("Fetching all active US tickers...")
        tickers = get_all_us_tickers()
        print(f"Fetched {len(tickers)} tickers.")
    elif market == 'CN':
        print("Fetching all active CN A-share tickers...")
        tickers = get_all_cn_tickers()
        print(f"Fetched {len(tickers)} tickers.")
    else:
        tickers = []
    
    if limit > 0:
        tickers = tickers[:limit]
    
    if save_to_db:
        update_scan_job(target_date, total_stocks=len(tickers))
    
    print(f"📋 Scanning {len(tickers)} stocks for {target_date}...")
    
    results = []
    scanned = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_stock_for_date, ticker, target_date, market): ticker 
            for ticker in tickers
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), unit="stock"):
            scanned += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                    # 实时写入数据库
                    if save_to_db:
                        insert_scan_result(result)
            except Exception as e:
                pass
            
            # 定期更新进度
            if scanned % 500 == 0 and save_to_db:
                update_scan_job(target_date, scanned_stocks=scanned, signals_found=len(results))
    
    # 获取公司信息 (只对有信号的股票)
    print(f"\n📇 Fetching company info for {len(results)} candidates...")
    for result in tqdm(results, desc="Fetching details"):
        try:
            # 根据市场选择不同的详情获取函数
            if market == 'CN':
                details = get_cn_ticker_details(result['Symbol'])
            else:
                details = get_ticker_details(result['Symbol'])
                
            if details:
                result['Company_Name'] = details.get('name')
                result['Industry'] = details.get('sic_description')
                result['Market_Cap'] = details.get('market_cap')
                
                # 市值分类
                mc = details.get('market_cap', 0) or 0
                if mc >= 200_000_000_000:
                    result['Cap_Category'] = 'Mega-Cap (超大盘)'
                elif mc >= 10_000_000_000:
                    result['Cap_Category'] = 'Large-Cap (大盘)'
                elif mc >= 2_000_000_000:
                    result['Cap_Category'] = 'Mid-Cap (中盘)'
                elif mc >= 300_000_000:
                    result['Cap_Category'] = 'Small-Cap (小盘)'
                elif mc > 0:
                    result['Cap_Category'] = 'Micro-Cap (微盘)'
                
                # 更新数据库
                if save_to_db:
                    insert_scan_result(result)
        except:
            pass
    
    # 完成扫描
    if save_to_db:
        update_scan_job(
            target_date, 
            status='done', 
            scanned_stocks=scanned, 
            signals_found=len(results),
            finished_at=datetime.now().isoformat()
        )
    
    print(f"\n✅ Scan complete. Found {len(results)} candidates.")
    
    return results


def backfill_dates(start_date, end_date, market='US', max_workers=30):
    """
    批量回填指定日期范围的数据
    
    Args:
        start_date: 开始日期 ('YYYY-MM-DD')
        end_date: 结束日期 ('YYYY-MM-DD')
        market: 市场
        max_workers: 并行线程数
    """
    from db.database import get_missing_dates
    
    missing = get_missing_dates(start_date, end_date)
    
    if not missing:
        print(f"✅ No missing dates between {start_date} and {end_date}")
        return
    
    print(f"📅 Found {len(missing)} missing dates to backfill:")
    for d in missing[:10]:
        print(f"   - {d}")
    if len(missing) > 10:
        print(f"   ... and {len(missing) - 10} more")
    
    for i, date in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] Backfilling {date}...")
        run_scan_for_date(date, market=market, max_workers=max_workers, save_to_db=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan stocks for a specific date')
    parser.add_argument('--date', type=str, default=None, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--market', type=str, default='US', help='Market (US/CN)')
    parser.add_argument('--workers', type=int, default=30, help='Number of workers')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of stocks')
    parser.add_argument('--backfill-start', type=str, help='Backfill start date')
    parser.add_argument('--backfill-end', type=str, help='Backfill end date')
    
    args = parser.parse_args()
    
    # 初始化数据库
    init_db()
    
    if args.backfill_start and args.backfill_end:
        # 批量回填模式
        backfill_dates(args.backfill_start, args.backfill_end, args.market, args.workers)
    else:
        # 单日扫描模式
        target_date = args.date or datetime.now().strftime('%Y-%m-%d')
        results = run_scan_for_date(target_date, args.market, args.workers, args.limit)
        
        # 打印 Top 10
        print("\n🏆 Top 10 Candidates:")
        sorted_results = sorted(results, key=lambda x: x['Blue_Daily'], reverse=True)[:10]
        for r in sorted_results:
            print(f"  {r['Symbol']:6} | Price: ${r['Price']:8.2f} | Day BLUE: {r['Blue_Daily']:5.1f} | Week BLUE: {r['Blue_Weekly']:5.1f}")




