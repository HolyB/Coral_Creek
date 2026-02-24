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
    calculate_volatility, analyze_elliott_wave_proxy, analyze_chanlun_proxy,
    calculate_phantom_indicator
)
from ml.data_cache import DataCache

# ML Components
try:
    import numpy as np
    import pandas as pd
    from ml.models.signal_ranker import SignalRanker, TradingHorizon
    from ml.feature_engineering import FeatureEngineer
    _ranker_model = None
    _feature_engineer = FeatureEngineer()
    ML_AVAILABLE = True
except Exception as e:
    print(f"⚠️ ML import failed: {e}")
    ML_AVAILABLE = False
    _ranker_model = None

# 初始化数据缓存
_data_cache = DataCache()


def _load_ranker_model():
    """按需加载排序模型"""
    return None  # 暂时禁用 ML Ranker 以防止 Backfill 崩溃
    
    global _ranker_model
    if not ML_AVAILABLE: return None
    if _ranker_model is not None: return _ranker_model
    
    try:
        model_path = os.path.join(parent_dir, "ml", "models", "trained", "signal_ranker_v1")
        ranker = SignalRanker()
        if ranker.load(str(model_path)):
            _ranker_model = ranker
        else:
            # print(f"⚠️ ML Ranker not found at {model_path}")
            pass
    except Exception as e:
        print(f"⚠️ Failed to load ML Ranker: {e}")
    
    return _ranker_model


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
            # 使用 DataCache 替代 get_us_stock_data
            df = _data_cache.get_stock_history(symbol, market='US', days=days_needed)
            
            # DataCache 返回全小写列名 (open, high, low, close, volume)
            # 需要兼容下面的大写逻辑，或者把下面改成小写
            # 为了最小改动，这里重命名为大写
            if df is not None:
                df = df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                })
                # Set index to date
                if 'date' in df.columns:
                    df = df.set_index('date')
                
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
        
        # 日线 LIRED + PINK (空头信号)
        lired_daily_val = 0.0
        pink_daily_val = 50.0
        try:
            phantom = calculate_phantom_indicator(opens, highs, lows, closes, volumes)
            lired_daily_val = float(phantom['lired'][-1]) if len(phantom['lired']) > 0 else 0.0
            pink_daily_val = float(phantom['pink'][-1]) if len(phantom['pink']) > 0 else 50.0
        except Exception:
            pass
        
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
        curr_heima = bool(heima_arr[-1]) if len(heima_arr) > 0 else False
        curr_juedi = bool(juedi_arr[-1]) if len(juedi_arr) > 0 else False
        heima_daily = curr_heima
        juedi_daily = curr_juedi
        
        # 黑马/掘地信号 - 周线
        heima_weekly = False
        juedi_weekly = False
        if len(df_weekly) >= 10:
            heima_w, juedi_w = calculate_heima_signal_series(
                df_weekly['High'].values, df_weekly['Low'].values,
                df_weekly['Close'].values, df_weekly['Open'].values
            )
            heima_weekly = heima_w[-1] if len(heima_w) > 0 else False
            juedi_weekly = juedi_w[-1] if len(juedi_w) > 0 else False
        
        # 黑马/掘地信号 - 月线
        heima_monthly = False
        juedi_monthly = False
        if len(df_monthly) >= 6:
            heima_m, juedi_m = calculate_heima_signal_series(
                df_monthly['High'].values, df_monthly['Low'].values,
                df_monthly['Close'].values, df_monthly['Open'].values
            )
            heima_monthly = heima_m[-1] if len(heima_m) > 0 else False
            juedi_monthly = juedi_m[-1] if len(juedi_m) > 0 else False
        
        # ADX
        adx_series = calculate_adx_series(highs, lows, closes)
        adx_val = float(adx_series[-1]) if len(adx_series) > 0 else 0.0
        
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

        # === 多空王买卖信号（执行版） ===
        def _ema(vals, n):
            alpha = 2.0 / (n + 1.0)
            out = [float(vals[0])]
            for v in vals[1:]:
                out.append(alpha * float(v) + (1 - alpha) * out[-1])
            return out

        def _sma_cn(vals, n, m=1):
            out = [float(vals[0])]
            for x in vals[1:]:
                out.append((m * float(x) + (n - m) * out[-1]) / float(n))
            return out

        profile = str(os.environ.get("DUOKONGWANG_PROFILE", "balanced")).lower()
        duokongwang_buy = False
        duokongwang_sell = False
        try:
            n = len(closes)
            if n >= 30:
                opens_proxy = [float(closes[0])] + [float(x) for x in closes[:-1]]
                up = _ema(highs, 13)
                dw = _ema(lows, 13)

                # KDJ(14,3,3)
                rsv = []
                for i in range(n):
                    s = max(0, i - 13)
                    llv = float(np.min(lows[s:i + 1]))
                    hhv = float(np.max(highs[s:i + 1]))
                    rsv.append(50.0 if hhv <= llv else (float(closes[i]) - llv) / (hhv - llv) * 100.0)
                k = _sma_cn(rsv, 3, 1)
                d = _sma_cn(k, 3, 1)
                j = [3.0 * k[i] - 2.0 * d[i] for i in range(n)]

                # RSI2(9)
                lc = [float(closes[0])] + [float(x) for x in closes[:-1]]
                up_move = [max(float(closes[i]) - lc[i], 0.0) for i in range(n)]
                abs_move = [abs(float(closes[i]) - lc[i]) for i in range(n)]
                rsi_num = _sma_cn(up_move, 9, 1)
                rsi_den = _sma_cn(abs_move, 9, 1)
                rsi2 = [(rsi_num[i] / rsi_den[i] * 100.0) if rsi_den[i] > 1e-12 else 50.0 for i in range(n)]

                # 九转计数（简化）
                nt = [0] * n
                nt0 = [0] * n
                for i in range(n):
                    a1 = i >= 4 and float(closes[i]) > float(closes[i - 4])
                    b1 = i >= 4 and float(closes[i]) < float(closes[i - 4])
                    nt[i] = (nt[i - 1] + 1) if (a1 and i > 0) else (1 if a1 else 0)
                    nt0[i] = (nt0[i - 1] + 1) if (b1 and i > 0) else (1 if b1 else 0)

                i = n - 1
                cond = (
                    (float(closes[i]) > opens_proxy[i] and (opens_proxy[i] > up[i] or float(closes[i]) < dw[i]))
                    or (float(closes[i]) < opens_proxy[i] and (opens_proxy[i] < dw[i] or float(closes[i]) > up[i]))
                )
                cond1 = bool(i >= 1 and up[i] > up[i - 1] and dw[i] > dw[i - 1])
                cond2 = bool(i >= 1 and up[i] < up[i - 1] and dw[i] < dw[i - 1])

                if profile == "aggressive":
                    j_cross_level, j_oversold_prev = 20.0, 28.0
                    rsi_prev_th, rsi_now_th, nine_min = 30.0, 26.0, 7
                elif profile == "conservative":
                    j_cross_level, j_oversold_prev = 35.0, 18.0
                    rsi_prev_th, rsi_now_th, nine_min = 20.0, 18.0, 9
                else:  # balanced
                    j_cross_level, j_oversold_prev = 30.0, 22.0
                    rsi_prev_th, rsi_now_th, nine_min = 24.0, 20.0, 9

                kdj_cross_up = bool(i >= 1 and j[i - 1] <= j_cross_level and j[i] > j_cross_level)
                kdj_oversold_turn = bool(i >= 1 and j[i - 1] < j_oversold_prev and j[i] > j[i - 1])
                rsi_oversold_turn = bool(i >= 1 and rsi2[i - 1] <= rsi_prev_th and rsi2[i] > rsi_now_th)
                nine_down_exhaust = bool(nt0[i] >= nine_min)
                duokongwang_buy = bool(
                    (cond and cond1 and (kdj_cross_up or rsi_oversold_turn))
                    or kdj_oversold_turn
                    or rsi_oversold_turn
                    or nine_down_exhaust
                )

                kdj_overheat_fade = bool(i >= 1 and ((j[i - 1] >= 100.0 and j[i] < 95.0) or (j[i - 1] >= 90.0 and j[i] < j[i - 1] - 8.0)))
                rsi_overbought_turn = bool(i >= 1 and rsi2[i - 1] >= 79.0 and rsi2[i] < 80.0)
                nine_up_exhaust = bool(nt[i] >= 9 and i >= 1 and float(closes[i]) < float(closes[i - 1]))
                duokongwang_sell = bool((cond and cond2) or kdj_overheat_fade or rsi_overbought_turn or nine_up_exhaust)
        except Exception:
            duokongwang_buy = False
            duokongwang_sell = False
        
        # 只保存有信号的股票（含空头信号）
        has_signal = (is_strat_d or is_strat_c or legacy_signal or week_blue_val > 100 
                      or duokongwang_buy or duokongwang_sell 
                      or lired_daily_val > 0 or pink_daily_val > 90)
        
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
        wave_phase, wave_desc = analyze_elliott_wave_proxy(closes, highs, lows)
        
        # 缠论
        chan_signal, chan_desc = analyze_chanlun_proxy(closes, highs, lows)
        
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
        
        # ML Scoring
        ml_rank_score = None
        try:
            ranker = _load_ranker_model()
            if ranker and _feature_engineer:
                # Rename back to lowercase for FE
                ml_df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Transform (extract features)
                ml_feats = _feature_engineer.transform(ml_df)
                
                if not ml_feats.empty:
                    # Predict on latest data
                    # Ensure features match model requirements
                    valid_feats = [col for col in ranker.feature_names if col in ml_feats.columns]
                    if len(valid_feats) == len(ranker.feature_names):
                        X_pred = ml_feats.iloc[[-1]][ranker.feature_names].values
                        # Handle Inf/Nan
                        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        score = ranker.rank(X_pred, TradingHorizon.SHORT)[0]
                        ml_rank_score = float(score)
        except Exception:
            pass
        
        return {
            'Symbol': symbol,
            'Date': latest_date.strftime('%Y-%m-%d'),
            'ML_Rank_Score': ml_rank_score,
            'Price': round(curr_price, 2),
            'Turnover_M': round(turnover, 2),
            'Blue_Daily': round(day_blue_val, 1),
            'Blue_Weekly': round(week_blue_val, 1),
            'Blue_Monthly': round(month_blue_val, 1),
            'Day_High': round(float(highs[-1]), 4) if len(highs) else round(curr_price, 4),
            'Day_Low': round(float(lows[-1]), 4) if len(lows) else round(curr_price, 4),
            'Day_Close': round(float(closes[-1]), 4) if len(closes) else round(curr_price, 4),
            'ADX': round(adx_val, 1),
            'Volatility': round(volatility, 2),
            'Is_Heima': heima_daily or heima_weekly or heima_monthly,  # 任何周期有黑马
            'Is_Juedi': juedi_daily or juedi_weekly or juedi_monthly,  # 任何周期有掘地
            'Heima_Daily': heima_daily,
            'Heima_Weekly': heima_weekly,
            'Heima_Monthly': heima_monthly,
            'Juedi_Daily': juedi_daily,
            'Juedi_Weekly': juedi_weekly,
            'Juedi_Monthly': juedi_monthly,
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
            'Duokongwang_Buy': bool(duokongwang_buy),
            'Duokongwang_Sell': bool(duokongwang_sell),
            'Lired_Daily': round(lired_daily_val, 2),
            'Pink_Daily': round(pink_daily_val, 2),
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
    
    # 获取公司信息 (只对有信号的股票，且仅在最近日期执行，避免 Backfill 变慢)
    # 注意：get_ticker_details 返回的是当前快照，对历史回测会有 Lookahead Bias，且速度极慢。
    try:
        if isinstance(target_date, str):
            t_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            t_date_obj = target_date
        days_diff = (datetime.now().date() - t_date_obj).days
    except:
        days_diff = 0

    if days_diff < 5:
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
    else:
        print(f"\n⏩ Skipping company details fetch for historical date {target_date} (Speed optimization)")
    
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


